"""MobileNetV3 for timme — encoder/classifier split.

Design:
  - MobileNetV3 IS the encoder. conv_stem + bn1 + blocks stay.
  - The "efficient head" (pool -> conv_head -> norm/act -> flatten -> classifier)
    becomes a SpatialPipelineHead with a conv_pipeline module.
  - This is the hard case because pool happens BEFORE the conv head, and the
    conv head is tightly coupled to the pool output shape.

Key insight: SpatialPipelineHead runs pool first, then the conv pipeline.
The MobileNetV3 "efficient head" conv pipeline is:
    nn.Sequential(conv_head, norm_head, act2)
which runs on the already-pooled 1x1 spatial output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.layers import (
    Linear,
    LayerType,
    PadType,
    create_conv2d,
    get_norm_act_layer,
)
from timm.models._efficientnet_blocks import SqueezeExcite
from timm.models._efficientnet_builder import (
    BlockArgs,
    EfficientNetBuilder,
    decode_arch_def,
    efficientnet_init_weights,
    round_channels,
)
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint_seq

from ..arch import ArchTraits, ImageEncoder, ImageClassifier, WeightLayout, remap_state_dict
from ..arch import ConfigMixin
from ..heads import SpatialEfficientHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class MobileNetV3Cfg(ConfigMixin):
    """Architecture config for MobileNetV3 encoder.

    Uses the existing arch-def string encoding for block definitions.
    This is already effectively a serializable format — the strings encode
    block type, kernel, stride, expansion, channels, SE ratio, etc.

    No img_size — MobileNetV3 is fully convolutional.
    in_chans stays because it determines the stem conv input channels.
    """

    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'in_chans'})

    arch_def: Tuple[Tuple[str, ...], ...] = ()  # block definition strings
    stem_size: int = 16
    fix_stem: bool = False
    round_chs_fn: str = 'round_channels'  # string ref to channel rounding function
    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm2d'
    se_from_exp: bool = True
    drop_path_rate: float = 0.0
    layer_scale_init_value: Optional[float] = None
    # Head params that affect encoder output (efficient head conv)
    num_features: int = 1280
    head_bias: bool = True
    head_norm: bool = False

    def __post_init__(self):
        if isinstance(self.arch_def, list):
            self.arch_def = tuple(tuple(s) for s in self.arch_def)


class MobileNetV3(ImageEncoder):
    """MobileNetV3 encoder.

    conv_stem + bn1 + all IR/ER/DS blocks.
    No pooling, no conv_head, no classifier.
    Output: (B, C, H, W) where C = last block output channels.
    """

    def __init__(
            self,
            block_args: BlockArgs,
            in_chans: int = 3,
            stem_size: int = 16,
            fix_stem: bool = False,
            pad_type: str = '',
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            se_from_exp: bool = True,
            round_chs_fn: Callable = round_channels,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = None,
            # Encoder-specific
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
    ):
        super().__init__(out_indices=out_indices)
        dd = {'device': device, 'dtype': dtype}
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.in_chans = in_chans
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type, **dd)
        self.bn1 = norm_act_layer(stem_size, inplace=True, **dd)

        # Blocks
        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            **dd,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices=out_indices or (4,))
        self.stage_ends = [f['stage'] for f in builder.features]
        output_dim = builder.in_chs

        # --- NO pool, NO conv_head, NO classifier ---
        # Those belong to the head now.

        self.traits = ArchTraits(
            output_fmt='NCHW',
            output_dim=output_dim,
            default_pool_type='avg',
        )

        efficientnet_init_weights(self)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    # ------------------------------------------------------------------
    # Forward — encoder's forward() IS the features
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Final-only fast path — (B, C, H, W)."""
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    # forward() inherited from ImageEncoder:
    #   - no out_indices  -> calls _forward_features(x)
    #   - with out_indices -> calls forward_intermediates(x, indices=..., intermediates_only=True)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            extra_blocks: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        if stop_early:
            assert intermediates_only
        intermediates = []
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            take_indices = [self.stage_ends[i] for i in take_indices]
            max_index = self.stage_ends[max_index]

        feat_idx = 0
        x = self.conv_stem(x)
        x = self.bn1(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index]
        for feat_idx, blk in enumerate(blocks, start=1):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(blk, x)
            else:
                x = blk(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            extra_blocks: bool = False,
    ) -> List[int]:
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            max_index = self.stage_ends[max_index]
        self.blocks = self.blocks[:max_index]
        return take_indices


# ======================================================================
# Weight layout for old timm1 checkpoints
# ======================================================================

MNV3_WEIGHT_LAYOUT = WeightLayout(
    encoder=('conv_stem', 'bn1', 'blocks'),
    head=(
        'conv_head',
        'norm_head',
        ('classifier', 'head.fc'),  # rename Linear at 'classifier' into head.fc
    ),
)


# ======================================================================
# Variant configs
# ======================================================================

MOBILENETV3_CFGS = {
    # Matches timm's _gen_mobilenet_v3 for the 100% width variant.
    # Note: final 'cn_*' stage produces the encoder's channel expansion (160 -> 960),
    # so the encoder output_dim is 960; the SpatialEfficientHead conv_head is 960 -> 1280.
    'mobilenetv3_large_100': MobileNetV3Cfg(
        arch_def=(
            ('ds_r1_k3_s1_e1_c16_nre',),
            ('ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'),
            ('ir_r3_k5_s2_e3_c40_se0.25_nre',),
            ('ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'),
            ('ir_r2_k3_s1_e6_c112_se0.25',),
            ('ir_r3_k5_s2_e6_c160_se0.25',),
            ('cn_r1_k1_s1_c960',),
        ),
        stem_size=16,
        num_features=1280,
        act_layer='hard_swish',
    ),
    'mobilenetv3_small_100': MobileNetV3Cfg(
        arch_def=(
            ('ds_r1_k3_s2_e1_c16_se0.25_nre',),
            ('ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'),
            ('ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'),
            ('ir_r2_k5_s1_e3_c48_se0.25',),
            ('ir_r3_k5_s2_e6_c96_se0.25',),
            ('cn_r1_k1_s1_c576',),
        ),
        stem_size=16,
        num_features=1024,
        act_layer='hard_swish',
    ),
}


# ======================================================================
# Head factory for MobileNetV3
# ======================================================================


def create_mnv3_head(
        encoder: MobileNetV3,
        num_classes: int = 1000,
        global_pool: str = 'avg',
        drop_rate: float = 0.0,
        # Efficient head settings
        num_features: int = 1280,
        head_norm: bool = False,
        act_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        device=None,
        dtype=None,
) -> SpatialEfficientHead:
    """Create the MobileNetV3 efficient head.

    The "efficient head" pattern: pool -> conv1x1 -> [norm] -> act -> flatten -> drop -> fc.
    """
    return SpatialEfficientHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        num_features=num_features,
        pool_type=global_pool,
        drop_rate=drop_rate,
        head_norm=head_norm,
        act_layer=act_layer or 'relu',
        norm_layer=norm_layer or 'batchnorm2d',
        device=device,
        dtype=dtype,
    )


# ======================================================================
# Builders + registry
# ======================================================================


def _mnv3_kwargs(cfg: MobileNetV3Cfg) -> Dict[str, Any]:
    """Translate a MobileNetV3Cfg into MobileNetV3 constructor kwargs.

    Mirrors timm's _gen_mobilenet_v3: hard_sigmoid SE gate, ReLU force-act
    inside SE, channel-rounding inside SE rd computation.
    """
    from functools import partial as _partial

    arch_def = [list(stage) for stage in cfg.arch_def]  # decode_arch_def expects nested lists
    se_layer = _partial(
        SqueezeExcite,
        gate_layer='hard_sigmoid',
        force_act_layer=nn.ReLU,
        rd_round_fn=round_channels,
    )
    return dict(
        block_args=decode_arch_def(arch_def),
        in_chans=cfg.in_chans,
        stem_size=cfg.stem_size,
        fix_stem=cfg.fix_stem,
        round_chs_fn=round_channels,
        act_layer=cfg.act_layer,
        norm_layer=cfg.norm_layer,
        se_layer=se_layer,
        se_from_exp=cfg.se_from_exp,
        drop_path_rate=cfg.drop_path_rate,
        layer_scale_init_value=cfg.layer_scale_init_value,
    )


def build_mnv3_encoder(
        cfg: MobileNetV3Cfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> MobileNetV3:
    cfg = cfg.overlay(**kwargs)
    return MobileNetV3(out_indices=out_indices, **_mnv3_kwargs(cfg))


def build_mnv3_classifier(
        cfg: MobileNetV3Cfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    encoder = build_mnv3_encoder(cfg, **kwargs)
    head = create_mnv3_head(
        encoder,
        num_classes=num_classes,
        global_pool=global_pool or 'avg',
        drop_rate=drop_rate,
        num_features=cfg.num_features,
        head_norm=cfg.head_norm,
        act_layer=cfg.act_layer,
        norm_layer=cfg.norm_layer,
    )
    return ImageClassifier(encoder, head)


from ._factory import register_family  # noqa: E402

register_family(
    family_name='mnv3',
    variants=MOBILENETV3_CFGS,
    build_classifier=build_mnv3_classifier,
    build_encoder=build_mnv3_encoder,
    weight_layout=MNV3_WEIGHT_LAYOUT,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   # Just the encoder — forward() IS the features
#   encoder = create_encoder('mobilenetv3_large_100', pretrained=True)
#   features = encoder(images)                  # (B, 960, 7, 7)
#
#   # Encoder with intermediate features (replaces features_only=True)
#   encoder = create_encoder('mobilenetv3_large_100', pretrained=True, out_indices=(0, 1, 2, 3, 4))
#   feature_list = encoder(images)              # [stem, stage1, stage2, stage3, stage4]
#
#   # Full classifier with efficient head
#   model = create_model('mobilenetv3_large_100', pretrained=True)
#   logits = model(images)                      # (B, 1000)
#
#   # Classifier split API: forward_features -> encoder(), forward_head -> head()
#   features = model.forward_features(images)   # encoder(images) -> (B, 960, 7, 7)
#   embed = model.forward_head(features, pre_logits=True)  # head(features) -> (B, 1280)
