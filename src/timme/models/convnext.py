"""ConvNeXt for timme — encoder/classifier split.

Design:
  - ConvNeXt IS the encoder. stem + 4 stages + optional norm_pre stay.
  - The head variability is captured by a head factory:
      head_norm_first=True  -> norm lives in encoder (norm_pre), head is SpatialLinearHead.
      head_norm_first=False -> head is SpatialNormMlpHead (norm + optional hidden MLP).
  - Old checkpoints remap via CONVNEXT_WEIGHT_LAYOUT. The only wrinkle is that
    the old head's ``head.norm`` can be semantically equivalent to the new
    encoder's ``norm_pre`` in the head_norm_first=True case — but that variant
    doesn't have a head.norm anyway (ClassifierHead has no norm), so the split
    is clean in both configurations.

This is the reference example for SpatialNormMlpHead usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import (
    LayerType,
    get_act_layer,
    get_norm_layer,
    make_divisible,
    to_ntuple,
)
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models.convnext import (
    ConvNeXtStage,
    _init_weights,
    calculate_drop_path_rates,
    named_apply,
)

from ..arch import ArchTraits, ImageEncoder, ImageClassifier, WeightLayout, remap_state_dict
from ..arch import ConfigMixin
from ..heads import SpatialLinearHead, SpatialNormMlpHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class ConvNeXtCfg(ConfigMixin):
    """Architecture config for ConvNeXt / ConvNeXt-V2 encoder.

    No img_size — ConvNeXt is fully convolutional.
    in_chans stays because it determines the stem conv input channels.
    head_norm_first decides whether the pre-head norm lives in the encoder
    (SpatialLinearHead) or in the head (SpatialNormMlpHead).
    """

    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'in_chans'})

    depths: Tuple[int, ...] = (3, 3, 9, 3)
    dims: Tuple[int, ...] = (96, 192, 384, 768)
    kernel_sizes: Union[int, Tuple[int, ...]] = 7
    ls_init_value: Optional[float] = 1e-6
    stem_type: str = 'patch'  # 'patch', 'overlap', 'overlap_tiered', 'overlap_act'
    patch_size: int = 4
    conv_mlp: bool = False
    conv_bias: bool = True
    use_grn: bool = False
    act_layer: str = 'gelu'
    norm_layer: Optional[str] = None  # None -> layernorm2d / layernorm
    norm_eps: Optional[float] = None
    output_stride: int = 32
    drop_path_rate: float = 0.0

    # Head layout toggle (affects where norm_pre lives)
    head_norm_first: bool = False

    def __post_init__(self):
        if isinstance(self.depths, list):
            self.depths = tuple(self.depths)
        if isinstance(self.dims, list):
            self.dims = tuple(self.dims)
        if isinstance(self.kernel_sizes, list):
            self.kernel_sizes = tuple(self.kernel_sizes)


def _get_norm_layers(
        norm_layer: Optional[Union[str, Callable]],
        conv_mlp: bool,
        norm_eps: Optional[float],
):
    """Resolve (norm_layer, norm_layer_cl) for ConvNeXt — a 2D norm and a
    channels-last / linear-friendly norm. Mirrors timm.models.convnext."""
    from functools import partial

    norm_layer = norm_layer or 'layernorm2d'
    norm_layer = get_norm_layer(norm_layer)
    norm_layer_cl = get_norm_layer('layernorm') if not conv_mlp else norm_layer
    if norm_eps is not None:
        norm_layer = partial(norm_layer, eps=norm_eps)
        norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
    return norm_layer, norm_layer_cl


class ConvNeXt(ImageEncoder):
    """ConvNeXt / ConvNeXt-V2 encoder.

    stem -> 4 stages -> optional norm_pre.
    No pool, no classifier. Output: (B, dims[-1], H, W) in NCHW.
    """

    def __init__(
            self,
            cfg: Optional[ConvNeXtCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        if cfg is None:
            from dataclasses import fields as _fields

            cfg = ConvNeXtCfg(**{k: v for k, v in kwargs.items() if k in {f.name for f in _fields(ConvNeXtCfg)}})
        else:
            cfg = cfg.overlay(**kwargs)

        dd = {'device': device, 'dtype': dtype}
        assert cfg.output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(cfg.kernel_sizes)
        norm_layer, norm_layer_cl = _get_norm_layers(cfg.norm_layer, cfg.conv_mlp, cfg.norm_eps)
        act_layer = get_act_layer(cfg.act_layer)

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.grad_checkpointing = False

        # --- Stem ---
        assert cfg.stem_type in ('patch', 'overlap', 'overlap_tiered', 'overlap_act')
        if cfg.stem_type == 'patch':
            self.stem = nn.Sequential(
                nn.Conv2d(
                    cfg.in_chans,
                    cfg.dims[0],
                    kernel_size=cfg.patch_size,
                    stride=cfg.patch_size,
                    bias=cfg.conv_bias,
                    **dd,
                ),
                norm_layer(cfg.dims[0], **dd),
            )
            stem_stride = cfg.patch_size
        else:
            mid_chs = make_divisible(cfg.dims[0] // 2) if 'tiered' in cfg.stem_type else cfg.dims[0]
            self.stem = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv2d(
                            cfg.in_chans,
                            mid_chs,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=cfg.conv_bias,
                            **dd,
                        ),
                        act_layer() if 'act' in cfg.stem_type else None,
                        nn.Conv2d(
                            mid_chs,
                            cfg.dims[0],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=cfg.conv_bias,
                            **dd,
                        ),
                        norm_layer(cfg.dims[0], **dd),
                    ],
                )
            )
            stem_stride = 4

        # --- 4 Stages ---
        dp_rates = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depths, stagewise=True)
        stages = []
        feature_info = []
        prev_chs = cfg.dims[0]
        curr_stride = stem_stride
        dilation = 1
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= cfg.output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = cfg.dims[i]
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=cfg.depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=cfg.ls_init_value,
                    conv_mlp=cfg.conv_mlp,
                    conv_bias=cfg.conv_bias,
                    use_grn=cfg.use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                    **dd,
                )
            )
            prev_chs = out_chs
            feature_info.append(dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)
        self.feature_info = FeatureInfo(feature_info, out_indices=out_indices or (3,))

        # --- Optional pre-head norm (owned by encoder iff head_norm_first=True) ---
        # In the old monolithic ConvNeXt, head_norm_first moves the norm out of the head
        # into self.norm_pre. The encoder version stores it here regardless so that
        # _forward_features produces a fully ready representation for either head.
        output_dim = prev_chs
        self._head_norm_first = cfg.head_norm_first
        if cfg.head_norm_first:
            self.norm_pre = norm_layer(output_dim, **dd)
        else:
            self.norm_pre = nn.Identity()

        # Traits
        self.traits = ArchTraits(
            output_fmt='NCHW',
            output_dim=output_dim,
            default_pool_type='avg',
        )

        named_apply(_init_weights, self)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)'
            if coarse
            else [
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        for s in self.stages:
            s.grad_checkpointing = enable

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
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
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        x = self.stem(x)

        last_idx = len(self.stages) - 1
        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        feat_idx = -1
        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(self.norm_pre(x) if (norm and feat_idx == last_idx) else x)

        if intermediates_only:
            return intermediates

        if feat_idx == last_idx:
            x = self.norm_pre(x)
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]
        if prune_norm:
            self.norm_pre = nn.Identity()
        return take_indices


# ======================================================================
# Weight layout for old timm1 checkpoints
# ======================================================================

CONVNEXT_WEIGHT_LAYOUT = WeightLayout(
    encoder=('stem', 'stages', 'norm_pre'),
    # timm1 already nests head.norm.* / head.fc.* under 'head.' — identity map.
    head=(('head', 'head'),),
)


# ======================================================================
# Variant configs
# ======================================================================

CONVNEXT_CFGS = {
    'convnext_tiny': ConvNeXtCfg(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)),
    'convnext_small': ConvNeXtCfg(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768)),
    'convnext_base': ConvNeXtCfg(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)),
    'convnext_large': ConvNeXtCfg(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536)),
    # ConvNeXt-V2 uses GRN
    'convnextv2_base': ConvNeXtCfg(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        use_grn=True,
        ls_init_value=None,
    ),
    # head_norm_first variant (norm moves from head into encoder's norm_pre)
    'convnext_tiny_hnf': ConvNeXtCfg(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        head_norm_first=True,
    ),
}


# ======================================================================
# Head factory
# ======================================================================


def create_convnext_head(
        encoder: ConvNeXt,
        num_classes: int = 1000,
        global_pool: str = 'avg',
        drop_rate: float = 0.0,
        hidden_size: Optional[int] = None,
        act_layer: Union[str, Callable] = 'gelu',
        device=None,
        dtype=None,
) -> Union[SpatialLinearHead, SpatialNormMlpHead]:
    """Create the appropriate head for ConvNeXt.

    head_norm_first=True (encoder has norm_pre):
        Head is SpatialLinearHead (no norm needed, hidden_size disallowed).
    head_norm_first=False (default, FB pretrained):
        Head is SpatialNormMlpHead (norm lives in the head).
    """
    dd = dict(device=device, dtype=dtype)
    cfg = encoder.cfg
    embed_dim = encoder.output_dim

    if cfg.head_norm_first:
        assert hidden_size is None, 'head_norm_first=True does not use hidden_size'
        return SpatialLinearHead(
            in_features=embed_dim,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )

    # Default: norm-mlp head. Use same norm layer family as the encoder blocks.
    norm_layer, _ = _get_norm_layers(cfg.norm_layer, cfg.conv_mlp, cfg.norm_eps)
    return SpatialNormMlpHead(
        in_features=embed_dim,
        num_classes=num_classes,
        pool_type=global_pool,
        drop_rate=drop_rate,
        norm_layer=norm_layer,
        hidden_size=hidden_size,
        act_layer=act_layer,
        **dd,
    )


# ======================================================================
# Factory (sketch — registry + checkpoint loading live in _factory.py)
# ======================================================================


def build_convnext_encoder(
        cfg: ConvNeXtCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> ConvNeXt:
    """Build a ConvNeXt encoder. Kwargs overlaid onto cfg; unknowns dropped."""
    return ConvNeXt(cfg=cfg, out_indices=out_indices, **kwargs)


def build_convnext_classifier(
        cfg: ConvNeXtCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        hidden_size: Optional[int] = None,
        **kwargs,
) -> ImageClassifier:
    """Build a ConvNeXt classifier (encoder + SpatialLinearHead or SpatialNormMlpHead)."""
    cfg = cfg.overlay(**kwargs)
    encoder = ConvNeXt(cfg=cfg, **kwargs)
    head = create_convnext_head(
        encoder,
        num_classes=num_classes,
        global_pool=global_pool or 'avg',
        drop_rate=drop_rate,
        hidden_size=hidden_size,
    )
    return ImageClassifier(encoder, head)


# timm1 ConvNeXt has a checkpoint filter that normalizes FB/OpenCLIP/etc. key
# shapes before load; the split-remap runs after.
from timm.models.convnext import checkpoint_filter_fn as _timm1_convnext_filter_fn  # noqa: E402


def _convnext_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_convnext_filter_fn(state_dict, target)


from ._factory import register_family  # noqa: E402

# The timm names users type include architecture-level sizes (not just base).
_CONVNEXT_VARIANTS = {
    'convnext_tiny': CONVNEXT_CFGS['convnext_tiny'],
    'convnext_small': CONVNEXT_CFGS['convnext_small'],
    'convnext_base': CONVNEXT_CFGS['convnext_base'],
    'convnext_large': CONVNEXT_CFGS['convnext_large'],
    'convnextv2_base': CONVNEXT_CFGS['convnextv2_base'],
}

register_family(
    family_name='convnext',
    variants=_CONVNEXT_VARIANTS,
    build_classifier=build_convnext_classifier,
    build_encoder=build_convnext_encoder,
    weight_layout=CONVNEXT_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_convnext_filter_fn,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   # Encoder only — forward() IS the features
#   encoder = create_encoder('convnext_base', pretrained=True)
#   features = encoder(images)                  # (B, 1024, H, W)
#
#   # With intermediate features (replaces features_only=True)
#   encoder = create_encoder('convnext_base', pretrained=True, out_indices=(0, 1, 2, 3))
#   feature_list = encoder(images)              # [stage0, stage1, stage2, stage3]
#
#   # Full classifier with default norm-mlp head
#   model = create_model('convnext_base', pretrained=True)
#   logits = model(images)
#
#   # Head variant with MLP projection
#   model = create_model('convnext_base', hidden_size=4096)
#
#   # Classifier split API: forward_features -> encoder(), forward_head -> head()
#   features = model.forward_features(images)   # encoder(images)
#   embed = model.forward_head(features, pre_logits=True)  # head(features)
