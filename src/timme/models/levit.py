"""LeViT for timme — encoder/classifier split.

Design:
  - Levit IS the encoder. Stem (strided convs) + LevitStages stay.
    No pooling, no classifier — those live in the head.
  - LeViT has two forward modes:
      use_conv=False (default): tokens are NLC after the stem flattens
      use_conv=True: features stay NCHW throughout
    The encoder traits reflect whichever mode the cfg picks.
  - LeViT's head is unusual: BatchNorm1d + Linear (timm.models.levit.NormLinear),
    applied to the pooled representation. For distilled LeViT there are two
    such heads whose outputs are averaged at inference.
  - This file defines a local LevitNormLinearHead that:
      pools over the spatial / token dimension
      applies BN1d + dropout + Linear
    ...rather than adding a LeViT-specific one-off to heads.py.

Old checkpoints remap via LEVIT_WEIGHT_LAYOUT (distilled) /
LEVIT_NON_DISTILLED_WEIGHT_LAYOUT (non-distilled).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import get_act_layer, to_2tuple, to_ntuple, trunc_normal_
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint, checkpoint_seq
from timm.models.levit import LevitStage, Stem8, Stem16

from ..arch import (
    ArchTraits,
    DistilledImageClassifier,
    ImageClassifier,
    ImageEncoder,
    ImageHead,
    WeightLayout,
    remap_state_dict,
)
from ..arch import ConfigMixin


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class LevitCfg(ConfigMixin):
    """Architecture config for LeViT encoder.

    img_size IS architecturally meaningful for LeViT — Attention modules
    precompute attention_bias_idxs from the resolution. Both img_size and
    in_chans are deploy fields.
    """

    img_size: Union[int, Tuple[int, int]] = 224
    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'img_size', 'in_chans'})

    # Core architecture — one entry per stage (tuples)
    embed_dim: Tuple[int, ...] = (192,)
    key_dim: int = 64
    depth: Tuple[int, ...] = (12,)
    num_heads: Union[int, Tuple[int, ...]] = (3,)
    attn_ratio: Union[float, Tuple[float, ...]] = 2.0
    mlp_ratio: Union[float, Tuple[float, ...]] = 2.0

    # Stem + downsample
    stem_type: str = 's16'  # 's8' or 's16'
    down_op: str = 'subsample'  # 'subsample' or '' — inter-stage downsampling op

    # Activations + layout mode
    act_layer: str = 'hard_swish'
    attn_act_layer: Optional[str] = None  # None -> same as act_layer
    use_conv: bool = False  # True = NCHW throughout (conv mode), False = NLC

    # Regularization (encoder-local only, NOT head dropout)
    drop_path_rate: float = 0.0

    def __post_init__(self):
        if isinstance(self.embed_dim, list):
            self.embed_dim = tuple(self.embed_dim)
        if isinstance(self.depth, list):
            self.depth = tuple(self.depth)
        if isinstance(self.num_heads, list):
            self.num_heads = tuple(self.num_heads)
        if isinstance(self.attn_ratio, list):
            self.attn_ratio = tuple(self.attn_ratio)
        if isinstance(self.mlp_ratio, list):
            self.mlp_ratio = tuple(self.mlp_ratio)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)


class Levit(ImageEncoder):
    """LeViT encoder.

    stem (Stem8/Stem16) -> LevitStages. No pool, no classifier.
    Output: (B, N, C) (NLC) if use_conv=False, else (B, C, H, W) (NCHW).
    """

    def __init__(
            self,
            cfg: Optional[LevitCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        if cfg is None:
            from dataclasses import fields as _fields

            cfg = LevitCfg(**{k: v for k, v in kwargs.items() if k in {f.name for f in _fields(LevitCfg)}})
        else:
            cfg = cfg.overlay(**kwargs)

        dd = {'device': device, 'dtype': dtype}
        act_layer = get_act_layer(cfg.act_layer)
        attn_act_layer = get_act_layer(cfg.attn_act_layer or cfg.act_layer)

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.use_conv = cfg.use_conv
        self.grad_checkpointing = False

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depth) == num_stages
        num_heads = to_ntuple(num_stages)(cfg.num_heads)
        attn_ratio = to_ntuple(num_stages)(cfg.attn_ratio)
        mlp_ratio = to_ntuple(num_stages)(cfg.mlp_ratio)

        # --- Stem ---
        assert cfg.stem_type in ('s16', 's8')
        stem_cls = Stem16 if cfg.stem_type == 's16' else Stem8
        self.stem = stem_cls(cfg.in_chans, cfg.embed_dim[0], act_layer=act_layer, **dd)
        stride = self.stem.stride
        resolution = tuple([i // p for i, p in zip(to_2tuple(cfg.img_size), to_2tuple(stride))])

        # --- LeViT stages ---
        in_dim = cfg.embed_dim[0]
        stages = []
        feature_info = []
        for i in range(num_stages):
            stage_stride = 2 if i > 0 else 1
            stages.append(
                LevitStage(
                    in_dim,
                    cfg.embed_dim[i],
                    cfg.key_dim,
                    depth=cfg.depth[i],
                    num_heads=num_heads[i],
                    attn_ratio=attn_ratio[i],
                    mlp_ratio=mlp_ratio[i],
                    act_layer=act_layer,
                    attn_act_layer=attn_act_layer,
                    resolution=resolution,
                    use_conv=cfg.use_conv,
                    downsample=cfg.down_op if stage_stride == 2 else '',
                    drop_path=cfg.drop_path_rate,
                    **dd,
                )
            )
            stride *= stage_stride
            resolution = tuple([(r - 1) // stage_stride + 1 for r in resolution])
            feature_info.append(
                dict(num_chs=cfg.embed_dim[i], reduction=stride, module=f'stages.{i}'),
            )
            in_dim = cfg.embed_dim[i]
        self.stages = nn.Sequential(*stages)
        self.feature_info = FeatureInfo(feature_info, out_indices=out_indices or (num_stages - 1,))

        # Final spatial resolution (for reshape NLC <-> NCHW in intermediates)
        self._feat_size = resolution

        # Traits
        output_dim = cfg.embed_dim[-1]
        self.traits = ArchTraits(
            output_fmt='NCHW' if cfg.use_conv else 'NLC',
            output_dim=output_dim,
            default_pool_type='avg',
        )

        self.init_weights(needs_reset=False)

    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        self.apply(partial(self._init_weights, needs_reset=needs_reset))

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if needs_reset and hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    @property
    def feat_size(self) -> Tuple[int, int]:
        return self._feat_size

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=[(r'^stages\.(\d+)', None)],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates: List[torch.Tensor] = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        x = self.stem(x)
        B, C, H, W = x.shape
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)

        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        for feat_idx, stage in enumerate(stages):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(stage, x)
            else:
                x = stage(x)
            if feat_idx in take_indices:
                if self.use_conv:
                    intermediates.append(x)
                else:
                    intermediates.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2))
            # Stages at index > 0 do 2x downsampling
            H = (H + 2 - 1) // 2
            W = (W + 2 - 1) // 2

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]
        return take_indices


# ======================================================================
# Head — LeViT-specific BatchNorm1d + Linear
# ======================================================================


class LevitNormLinearHead(ImageHead):
    """LeViT head: avg-pool (over spatial or token dim) -> BN1d -> Linear.

    Accepts both NLC (use_conv=False) and NCHW (use_conv=True) encoder
    outputs. Pre-logits = the pooled representation before BN.

    Equivalent to timm.models.levit.NormLinear applied after avg pool.
    """

    accepted_fmts = ('NLC', 'NCHW')

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            drop_rate: float = 0.0,
            std: float = 0.02,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes

        self.bn = nn.BatchNorm1d(in_features, **dd)
        self.drop = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features, num_classes, **dd) if num_classes > 0 else nn.Identity()
        if num_classes > 0:
            trunc_normal_(self.linear.weight, std=std)
            if self.linear.bias is not None:
                nn.init.constant_(self.linear.bias, 0)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._in_features

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        # LeViT only supports avg pool — pool_type is ignored.
        self.linear = nn.Linear(self._in_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.linear

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> (B, C)   or   NLC -> (B, C)
        if x.ndim == 4:
            return x.mean(dim=(-2, -1))
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self._pool(x)
        if pre_logits:
            return x
        x = self.bn(x)
        x = self.drop(x)
        return self.linear(x)


# ======================================================================
# Weight layouts for old timm1 checkpoints
# ======================================================================

# Non-distilled LeViT doesn't exist in the official timm registry (all
# pretrained LeViT weights are distilled), but we support it for completeness.
LEVIT_NON_DISTILLED_WEIGHT_LAYOUT = WeightLayout(
    encoder=('stem', 'stages'),
    # timm1 already nests head.bn.* / head.linear.* under 'head.' — identity.
    head=(('head', 'head'),),
)

# Distilled LeViT: head.{bn,linear} and head_dist.{bn,linear} are siblings.
LEVIT_WEIGHT_LAYOUT = WeightLayout(
    encoder=('stem', 'stages'),
    head=(('head', 'head'), ('head_dist', 'head_dist')),
)


# ======================================================================
# Variant configs
# ======================================================================

LEVIT_CFGS = {
    'levit_128s': LevitCfg(
        embed_dim=(128, 256, 384),
        key_dim=16,
        num_heads=(4, 6, 8),
        depth=(2, 3, 4),
    ),
    'levit_128': LevitCfg(
        embed_dim=(128, 256, 384),
        key_dim=16,
        num_heads=(4, 8, 12),
        depth=(4, 4, 4),
    ),
    'levit_192': LevitCfg(
        embed_dim=(192, 288, 384),
        key_dim=32,
        num_heads=(3, 5, 6),
        depth=(4, 4, 4),
    ),
    'levit_256': LevitCfg(
        embed_dim=(256, 384, 512),
        key_dim=32,
        num_heads=(4, 6, 8),
        depth=(4, 4, 4),
    ),
    'levit_384': LevitCfg(
        embed_dim=(384, 512, 768),
        key_dim=32,
        num_heads=(6, 9, 12),
        depth=(4, 4, 4),
    ),
    # Conv-mode variant: same arch, NCHW layout
    'levit_conv_192': LevitCfg(
        embed_dim=(192, 288, 384),
        key_dim=32,
        num_heads=(3, 5, 6),
        depth=(4, 4, 4),
        use_conv=True,
    ),
}


# ======================================================================
# Head factories
# ======================================================================


def create_levit_head(
        encoder: Levit,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
) -> LevitNormLinearHead:
    return LevitNormLinearHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        drop_rate=drop_rate,
        device=device,
        dtype=dtype,
    )


def create_levit_distilled_heads(
        encoder: Levit,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
) -> Tuple[LevitNormLinearHead, LevitNormLinearHead]:
    head = create_levit_head(encoder, num_classes, drop_rate, device, dtype)
    # head_dist doesn't apply dropout in the original
    head_dist = LevitNormLinearHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        drop_rate=0.0,
        device=device,
        dtype=dtype,
    )
    return head, head_dist


# ======================================================================
# Builders + registry
# ======================================================================


def build_levit_encoder(
        cfg: LevitCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> Levit:
    return Levit(cfg=cfg, out_indices=out_indices, **kwargs)


def build_levit_distilled_classifier(
        cfg: LevitCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        distilled_training: bool = False,
        **kwargs,
) -> DistilledImageClassifier:
    encoder = Levit(cfg=cfg, **kwargs)
    head, head_dist = create_levit_distilled_heads(
        encoder,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return DistilledImageClassifier(
        encoder,
        head,
        head_dist,
        distilled_training=distilled_training,
    )


# Filter for LeViT: discard non-persistent attention_bias_idxs (mirrors timm1).
def _levit_filter_fn(state_dict, model):
    return {k: v for k, v in state_dict.items() if 'attention_bias_idxs' not in k}


# Build the variant dict matching timm's name set, including the `_conv_*`
# variants which share the same arch as the linear-mode ones but flip use_conv.
_LEVIT_VARIANTS: Dict[str, LevitCfg] = {}
for _name, _cfg in LEVIT_CFGS.items():
    _LEVIT_VARIANTS[_name] = _cfg
# Conv-mode variants: same arch as their linear counterpart with use_conv=True
for _base in ('levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384'):
    _LEVIT_VARIANTS[_base.replace('levit_', 'levit_conv_')] = LEVIT_CFGS[_base].overlay(use_conv=True)

from ._factory import register_family  # noqa: E402

register_family(
    family_name='levit',
    variants=_LEVIT_VARIANTS,
    build_classifier=build_levit_distilled_classifier,
    build_encoder=build_levit_encoder,
    weight_layout=LEVIT_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_levit_filter_fn,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   # Distilled LeViT (default — matches pretrained weights)
#   model = create_model('levit_128', pretrained=True)
#   model.distilled_training = True
#   cls, dist = model(images)                     # training
#   model.eval()
#   logits = model(images)                        # inference (averaged)
#
#   # Encoder only — tokens (or spatial features with use_conv=True)
#   encoder = Levit(cfg=LEVIT_CFGS['levit_128'])
#   tokens = encoder(images)                      # (B, N, 384)
#
#   # Conv mode
#   encoder = Levit(cfg=LEVIT_CFGS['levit_conv_192'])
#   features = encoder(images)                    # (B, 384, H, W)
