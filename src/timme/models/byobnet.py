"""ByobNet (Bring-Your-Own-Blocks) for timme — encoder/classifier split.

Design:
  - ByobNet IS the encoder. stem + stages + final_conv stay.
  - The head selection (cfg.head_type) moves to a head factory:
      ''/'classifier' -> SpatialLinearHead
      'mlp'           -> SpatialNormMlpHead (with head_hidden_size)
      'attn_abs'      -> SpatialAttentionHead(AttentionPool2d)
      'attn_rot'      -> SpatialAttentionHead(RotAttentionPool2d)
  - Old checkpoints remap via BYOBNET_WEIGHT_LAYOUT.

Notes:
  - The existing `ByoModelCfg` / `ByoBlockCfg` dataclasses in _config.py are
    the authoritative config format. ByobNet's constructor takes a cfg and
    overlays deploy kwargs (img_size, in_chans) + any arch overrides.
  - The attention-pool heads need a spatial feat_size — it's resolved at head
    creation time from cfg.img_size and the encoder's total reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import partial
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import (
    AttentionPool2d,
    RotAttentionPool2d,
    to_2tuple,
)
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models.byobnet import (
    _init_weights,
    create_byob_stages,
    create_byob_stem,
    get_layer_fns,
    named_apply,
    reduce_feat_size,
)

from ..arch import ArchTraits, ImageEncoder, ImageClassifier, WeightLayout, remap_state_dict
from ..arch import ConfigMixin
from ..heads import (
    SpatialAttentionHead,
    SpatialLinearHead,
    SpatialNormMlpHead,
)


# ======================================================================
# Architecture configs
# ======================================================================


@dataclass
class ByoBlockCfg(ConfigMixin):
    """Block config — one per stage."""

    type: str = 'bottle'
    d: int = 1
    c: int = 64
    s: int = 2
    gs: Optional[int] = None
    br: float = 1.0
    attn_layer: Optional[str] = None
    attn_kwargs: Optional[Dict[str, Any]] = None
    self_attn_layer: Optional[str] = None
    self_attn_kwargs: Optional[Dict[str, Any]] = None
    block_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ByoModelCfg(ConfigMixin):
    """Nested config for ByobNet family.

    img_size is a deploy field — only required when head_type is 'attn_abs' or
    'attn_rot' (those heads need a spatial feat_size). For classifier / mlp
    heads it is ignored.
    in_chans is always architectural.
    """

    in_chans: int = 3
    img_size: Optional[Union[int, Tuple[int, int]]] = None

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'in_chans', 'img_size'})

    # Per-stage block configs
    blocks: Tuple[ByoBlockCfg, ...] = ()

    # Stem + stages
    downsample: str = 'conv1x1'
    stem_type: str = '3x3'
    stem_pool: Optional[str] = 'maxpool'
    stem_chs: Union[int, Tuple[int, ...]] = 32
    width_factor: float = 1.0
    num_features: int = 0

    # Head
    head_type: str = 'classifier'  # '', 'classifier', 'mlp', 'attn_abs', 'attn_rot'
    head_hidden_size: Optional[int] = None

    # Layer types (all strings)
    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm'
    aa_layer: str = ''
    attn_layer: Optional[str] = None
    self_attn_layer: Optional[str] = None
    attn_kwargs: Optional[Dict[str, Any]] = None
    self_attn_kwargs: Optional[Dict[str, Any]] = None
    block_kwargs: Optional[Dict[str, Any]] = None

    # Misc
    fixed_input_size: bool = False
    zero_init_last: bool = True

    def __post_init__(self):
        if isinstance(self.blocks, list):
            self.blocks = tuple(
                ByoBlockCfg(**b) if isinstance(b, dict) else b
                for b in self.blocks
            )
        if isinstance(self.stem_chs, list):
            self.stem_chs = tuple(self.stem_chs)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ByoModelCfg':
        """Override to reconstruct nested ByoBlockCfg instances."""
        d = dict(d)
        if 'blocks' in d:
            d['blocks'] = tuple(
                ByoBlockCfg.from_dict(b) if isinstance(b, dict) else b
                for b in d['blocks']
            )
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class ByobNet(ImageEncoder):
    """Bring-your-own-blocks encoder.

    stem -> stages -> final_conv. Output: (B, num_features, H, W).
    No pool, no classifier.
    """

    def __init__(
            self,
            cfg: ByoModelCfg,
            out_indices: Optional[Tuple[int, ...]] = None,
            output_stride: int = 32,
            drop_block_rate: float = 0.0,
            drop_block_size: int = 3,
            drop_path_rate: float = 0.0,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)
        dd = {'device': device, 'dtype': dtype}

        cfg = cfg.overlay(**kwargs)
        if cfg.fixed_input_size:
            assert cfg.img_size is not None, 'img_size is required for fixed input size model'

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.grad_checkpointing = False

        stem_layers = get_layer_fns(cfg, allow_aa=False)
        stage_layers = get_layer_fns(cfg)
        feat_size = to_2tuple(cfg.img_size) if cfg.img_size is not None else None

        feature_info: List[Dict[str, Any]] = []
        # Stem width
        if isinstance(cfg.stem_chs, (list, tuple)):
            stem_chs = [int(round(c * cfg.width_factor)) for c in cfg.stem_chs]
        else:
            stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))

        self.stem, stem_feat = create_byob_stem(
            in_chs=cfg.in_chans,
            out_chs=stem_chs,
            stem_type=cfg.stem_type,
            pool_type=cfg.stem_pool,
            layers=stem_layers,
            **dd,
        )
        feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]['reduction'])

        self.stages, stage_feat, feat_size = create_byob_stages(
            cfg,
            drop_path_rate,
            output_stride,
            stem_feat[-1],
            drop_block_rate=drop_block_rate,
            drop_block_size=drop_block_size,
            layers=stage_layers,
            feat_size=feat_size,
            **dd,
        )
        feature_info.extend(stage_feat[:-1])
        reduction = stage_feat[-1]['reduction']

        prev_chs = stage_feat[-1]['num_chs']
        if cfg.num_features:
            num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = stage_layers.conv_norm_act(prev_chs, num_features, 1, **dd)
        else:
            num_features = prev_chs
            self.final_conv = nn.Identity()
        feature_info.append(
            dict(
                num_chs=num_features,
                reduction=reduction,
                module='final_conv',
                stage=len(self.stages),
            )
        )
        self.stage_ends = [f['stage'] for f in feature_info]
        self.feature_info = FeatureInfo(feature_info, out_indices=out_indices or (len(self.stage_ends) - 1,))

        # Remember the spatial feat size after all reductions — needed by
        # attention-pool heads to build positional embeddings.
        self._feat_size = feat_size

        # Traits
        self.traits = ArchTraits(
            output_fmt='NCHW',
            output_dim=num_features,
            default_pool_type='avg' if cfg.head_type in ('', 'classifier', 'mlp') else 'token',
        )

        named_apply(partial(_init_weights, zero_init_last=cfg.zero_init_last), self)

    @property
    def feat_size(self) -> Optional[Tuple[int, int]]:
        """Spatial size of the encoder output, if resolvable from cfg.img_size."""
        return self._feat_size

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^stem',
            blocks=[
                (r'^stages\.(\d+)' if coarse else r'^stages\.(\d+)\.(\d+)', None),
                (r'^final_conv', (99999,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            from timm.models._manipulate import checkpoint_seq

            x = checkpoint_seq(self.stages, x, flatten=True)
        else:
            x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            exclude_final_conv: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates: List[torch.Tensor] = []
        # Stem is index 0, stages are 1..N, final_conv is N+1 (when present)
        n_feats = 1 + len(self.stages) + (0 if isinstance(self.final_conv, nn.Identity) else 1)
        take_indices, max_index = feature_take_indices(n_feats, indices)

        feat_idx = 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for stage in stages:
            feat_idx += 1
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if not isinstance(self.final_conv, nn.Identity):
            feat_idx += 1
            x = self.final_conv(x)
            if not exclude_final_conv and feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        n_feats = 1 + len(self.stages) + (0 if isinstance(self.final_conv, nn.Identity) else 1)
        take_indices, max_index = feature_take_indices(n_feats, indices)
        # Keep stages up to max_index (index 0 is stem, stages start at 1)
        stage_keep = max(max_index - 1, 0)
        if stage_keep < len(self.stages):
            self.stages = nn.Sequential(*list(self.stages)[:stage_keep])
            self.final_conv = nn.Identity()
        return take_indices


# ======================================================================
# Weight layout for old timm1 checkpoints
# ======================================================================

BYOBNET_WEIGHT_LAYOUT = WeightLayout(
    encoder=('stem', 'stages', 'final_conv'),
    head=(('head', 'head'),),  # timm1 already nests head.fc.* etc. — identity
)


# ======================================================================
# Head factory
# ======================================================================


def create_byobnet_head(
        encoder: ByobNet,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
) -> Union[SpatialLinearHead, SpatialNormMlpHead, SpatialAttentionHead]:
    """Create the appropriate head for a ByobNet encoder.

    Head type is read from the encoder's cfg.head_type. Picking the default
    global_pool mirrors the old ByobNet constructor logic.
    """
    dd = dict(device=device, dtype=dtype)
    cfg = encoder.cfg
    num_features = encoder.output_dim
    head_type = cfg.head_type or 'classifier'

    if head_type == 'mlp':
        pool_type = global_pool or 'avg'
        return SpatialNormMlpHead(
            in_features=num_features,
            num_classes=num_classes,
            pool_type=pool_type,
            drop_rate=drop_rate,
            norm_layer=cfg.norm_layer,
            hidden_size=cfg.head_hidden_size,
            act_layer=cfg.act_layer,
            **dd,
        )

    if head_type in ('attn_abs', 'attn_rot'):
        pool_type = global_pool or 'token'
        assert pool_type in ('', 'token'), f"{head_type} requires global_pool in ('', 'token')"
        if encoder.feat_size is None:
            raise ValueError(
                f"{head_type} head requires a spatial feat_size; set cfg.img_size on the encoder so it can be derived."
            )
        embed_dim = cfg.head_hidden_size or num_features
        if head_type == 'attn_abs':
            attn_pool = AttentionPool2d(
                num_features,
                embed_dim=cfg.head_hidden_size,
                out_features=None,  # head owns the final fc
                feat_size=encoder.feat_size,
                pool_type=pool_type,
                qkv_separate=True,
                **dd,
            )
        else:
            attn_pool = RotAttentionPool2d(
                num_features,
                embed_dim=cfg.head_hidden_size,
                out_features=None,
                ref_feat_size=encoder.feat_size,
                pool_type=pool_type,
                qkv_separate=True,
                **dd,
            )
        return SpatialAttentionHead(
            in_features=num_features,
            num_classes=num_classes,
            attn_pool=attn_pool,
            pool_out_features=embed_dim,
            drop_rate=drop_rate,
            **dd,
        )

    # Default classifier head
    pool_type = global_pool or 'avg'
    assert cfg.head_hidden_size is None, 'classifier head does not support head_hidden_size'
    return SpatialLinearHead(
        in_features=num_features,
        num_classes=num_classes,
        pool_type=pool_type,
        drop_rate=drop_rate,
        **dd,
    )


# ======================================================================
# Variant configs — translated from timm1's model_cfgs
# ======================================================================

# ByobNet has so many variants (~80+) and complex per-stage block specs that
# hand-copying isn't practical; we translate timm1's model_cfgs at import time
# into our ByoModelCfg / ByoBlockCfg shapes. The dataclass fields overlap
# almost perfectly; only deploy fields (img_size, in_chans) are timme-only.
from timm.models.byobnet import (  # noqa: E402
    model_cfgs as _timm1_byob_cfgs,
    checkpoint_filter_fn as _timm1_byobnet_filter_fn,
)


def _translate_byob_cfg(src) -> Optional[ByoModelCfg]:
    # mobileone et al. use Tuple[List[ByoBlockCfg], ...] for stages with
    # heterogeneous blocks. timme's ByoModelCfg currently only supports
    # the flat Tuple[ByoBlockCfg, ...] shape — skip variants that need
    # the nested form for now.
    if any(isinstance(b, (list, tuple)) for b in src.blocks):
        return None
    blocks = tuple(
        ByoBlockCfg(
            type=b.type,
            d=b.d,
            c=b.c,
            s=b.s,
            gs=b.gs,
            br=b.br,
            attn_layer=b.attn_layer,
            attn_kwargs=b.attn_kwargs,
            self_attn_layer=b.self_attn_layer,
            self_attn_kwargs=b.self_attn_kwargs,
            block_kwargs=b.block_kwargs,
        )
        for b in src.blocks
    )
    return ByoModelCfg(
        blocks=blocks,
        downsample=src.downsample,
        stem_type=src.stem_type,
        stem_pool=src.stem_pool,
        stem_chs=src.stem_chs,
        width_factor=src.width_factor,
        num_features=src.num_features,
        head_type=src.head_type,
        head_hidden_size=src.head_hidden_size,
        act_layer=src.act_layer,
        norm_layer=src.norm_layer,
        aa_layer=src.aa_layer,
        attn_layer=src.attn_layer,
        attn_kwargs=src.attn_kwargs,
        self_attn_layer=src.self_attn_layer,
        self_attn_kwargs=src.self_attn_kwargs,
        block_kwargs=src.block_kwargs,
        fixed_input_size=src.fixed_input_size,
        zero_init_last=src.zero_init_last,
    )


_BYOBNET_VARIANTS = {
    name: tcfg
    for name, src_cfg in _timm1_byob_cfgs.items()
    for tcfg in [_translate_byob_cfg(src_cfg)]
    if tcfg is not None
}


def _byobnet_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_byobnet_filter_fn(state_dict, target)


# ======================================================================
# Builders + registry
# ======================================================================


def build_byobnet_encoder(
        cfg: ByoModelCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> ByobNet:
    return ByobNet(cfg=cfg, out_indices=out_indices, **kwargs)


def build_byobnet_classifier(
        cfg: ByoModelCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    encoder = ByobNet(cfg=cfg, **kwargs)
    head = create_byobnet_head(
        encoder,
        num_classes=num_classes,
        global_pool=global_pool,
        drop_rate=drop_rate,
    )
    return ImageClassifier(encoder, head)


from ._factory import register_family  # noqa: E402

register_family(
    family_name='byobnet',
    variants=_BYOBNET_VARIANTS,
    build_classifier=build_byobnet_classifier,
    build_encoder=build_byobnet_encoder,
    weight_layout=BYOBNET_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_byobnet_filter_fn,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   from timm.models.byobnet import model_cfgs  # real variant configs
#
#   # Encoder only (classifier head variant)
#   encoder = ByobNet(cfg=model_cfgs['gernet_m'])
#   features = encoder(images)                  # (B, 2560, H, W)
#
#   # CLIP-style attention-pool encoder — img_size required for feat_size
#   cfg = replace(model_cfgs['resnet50_clip'], img_size=224)
#   encoder = ByobNet(cfg=cfg)
#   model = ImageClassifier(encoder, create_byobnet_head(encoder, num_classes=1024))
#
#   # Intermediate features (stem=0, stages=1..N, final_conv=N+1)
#   encoder = ByobNet(cfg=cfg, out_indices=(1, 2, 3, 4))
#   feats = encoder(images)
