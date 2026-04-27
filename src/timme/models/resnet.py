"""ResNet for timme — encoder/classifier split.

Design:
  - ResNet IS the encoder. It's the current ResNet class with pool/fc/drop_rate
    removed. All architecture params stay, all classification params go.
  - The old ResNet classifier becomes ImageClassifier(ResNet(...), SpatialLinearHead(...))
  - Old checkpoints are remapped via RESNET_WEIGHT_LAYOUT.

This file shows the pattern that ~30 CNN families would follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.layers import get_act_layer, get_norm_layer, create_aa, LayerType
from timm.layers.helpers import to_ntuple
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint_seq
from timm.models.resnet import BasicBlock, Bottleneck, make_blocks

from ..arch import ArchTraits, ImageEncoder, ImageClassifier, WeightLayout, remap_state_dict
from ..arch import ConfigMixin
from ..heads import SpatialLinearHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class ResNetCfg(ConfigMixin):
    """Architecture config for ResNet encoder.

    No img_size — ResNet is fully convolutional, any input size works.
    img_size is training recipe metadata living in pretrained_cfg, not here.
    in_chans stays because it determines the stem conv input channels.
    """

    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'in_chans'})

    block: str = 'Bottleneck'  # 'BasicBlock' or 'Bottleneck'
    layers: Tuple[int, ...] = (3, 4, 6, 3)
    channels: Tuple[int, ...] = (64, 128, 256, 512)
    stem_width: int = 64
    stem_type: str = ''
    replace_stem_pool: bool = False
    output_stride: int = 32
    cardinality: int = 1
    base_width: int = 64
    block_reduce_first: int = 1
    down_kernel_size: int = 1
    avg_down: bool = False
    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm2d'
    aa_layer: str = ''
    drop_path_rate: float = 0.0
    drop_block_rate: float = 0.0
    zero_init_last: bool = True

    def __post_init__(self):
        if isinstance(self.layers, list):
            self.layers = tuple(self.layers)
        if isinstance(self.channels, list):
            self.channels = tuple(self.channels)


class ResNet(ImageEncoder):
    """ResNet / ResNeXt / SE-ResNeXt / SENet encoder.

    This is the old ResNet with pool + fc stripped out.
    All params that controlled forward_features() stay.
    num_classes, global_pool, drop_rate are gone — those live in the head.
    """

    def __init__(
            self,
            block: Union[Type[BasicBlock], Type[Bottleneck]],
            layers: Tuple[int, ...],
            in_chans: int = 3,
            output_stride: int = 32,
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_path_rate: float = 0.0,
            drop_block_rate: float = 0.0,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            # Encoder-specific
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
    ):
        super().__init__(out_indices=out_indices)
        dd = {'device': device, 'dtype': dtype}
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.in_chans = in_chans
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # --- Stem (identical to old ResNet) ---
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False, **dd),
                norm_layer(stem_chs[0], **dd),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False, **dd),
                norm_layer(stem_chs[1], **dd),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False, **dd),
            ])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False, **dd)
        self.bn1 = norm_layer(inplanes, **dd)
        self.act1 = act_layer(inplace=True)
        self.feature_info = FeatureInfo(
            [dict(num_chs=inplanes, reduction=2, module='act1')],
            out_indices=out_indices or (4,),
        )

        # --- Stem pool ---
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False, **dd),
                        create_aa(aa_layer, channels=inplanes, stride=2, **dd) if aa_layer is not None else None,
                        norm_layer(inplanes, **dd),
                        act_layer(inplace=True),
                    ],
                )
            )
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2, **dd),
                    ])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Residual stages ---
        block_fns = to_ntuple(len(channels))(block)
        stage_modules, stage_feature_info = make_blocks(
            block_fns,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
            **dd,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info = FeatureInfo(
            [dict(num_chs=inplanes, reduction=2, module='act1')] + stage_feature_info,
            out_indices=out_indices or (4,),
        )

        # --- NO pool, NO fc, NO drop_rate ---
        # Those belong to the head now.

        # Set traits
        output_dim = channels[-1] * block_fns[-1].expansion
        self.traits = ArchTraits(
            output_fmt='NCHW',
            output_dim=output_dim,
            default_pool_type='avg',
        )

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, str]:
        return dict(
            stem=r'^conv1|bn1|maxpool',
            blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    # ------------------------------------------------------------------
    # Forward — encoder's forward() IS the features
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Final-only fast path — (B, C, H, W)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
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
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        feat_idx = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.maxpool(x)

        layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        take_indices, max_index = feature_take_indices(5, indices)
        layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
        for n in layer_names[max_index:]:
            setattr(self, n, nn.Identity())
        return take_indices


# ======================================================================
# Weight layout for old timm1 checkpoints
# ======================================================================

RESNET_WEIGHT_LAYOUT = WeightLayout(
    encoder=('conv1', 'bn1', 'act1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'),
    head=('fc',),
)


# ======================================================================
# Variant configs (between modeling code and entry-point fns, like timm)
# ======================================================================

RESNET_CFGS = {
    'resnet18': ResNetCfg(block='BasicBlock', layers=(2, 2, 2, 2)),
    'resnet34': ResNetCfg(block='BasicBlock', layers=(3, 4, 6, 3)),
    'resnet50': ResNetCfg(block='Bottleneck', layers=(3, 4, 6, 3)),
    'resnet101': ResNetCfg(block='Bottleneck', layers=(3, 4, 23, 3)),
    'resnet152': ResNetCfg(block='Bottleneck', layers=(3, 8, 36, 3)),
    'resnet50d': ResNetCfg(
        block='Bottleneck',
        layers=(3, 4, 6, 3),
        stem_type='deep',
        avg_down=True,
        stem_width=32,
    ),
    'seresnet50': ResNetCfg(
        block='Bottleneck',
        layers=(3, 4, 6, 3),
        # block_args would handle SE — extension point
    ),
}


# ======================================================================
# Builders + registry
# ======================================================================

_RESNET_BLOCKS = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}


def _resnet_kwargs(cfg: ResNetCfg) -> Dict[str, Any]:
    """Unpack a ResNetCfg into the ResNet constructor's kwargs."""
    return dict(
        block=_RESNET_BLOCKS[cfg.block],
        layers=cfg.layers,
        in_chans=cfg.in_chans,
        channels=cfg.channels,
        stem_width=cfg.stem_width,
        stem_type=cfg.stem_type,
        replace_stem_pool=cfg.replace_stem_pool,
        output_stride=cfg.output_stride,
        cardinality=cfg.cardinality,
        base_width=cfg.base_width,
        block_reduce_first=cfg.block_reduce_first,
        down_kernel_size=cfg.down_kernel_size,
        avg_down=cfg.avg_down,
        act_layer=cfg.act_layer,
        norm_layer=cfg.norm_layer,
        aa_layer=cfg.aa_layer or None,
        drop_path_rate=cfg.drop_path_rate,
        drop_block_rate=cfg.drop_block_rate,
        zero_init_last=cfg.zero_init_last,
    )


def build_resnet_encoder(
        cfg: ResNetCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> ResNet:
    """Build a ResNet encoder. Unknown kwargs (e.g. img_size) are dropped via overlay."""
    cfg = cfg.overlay(**kwargs)
    return ResNet(out_indices=out_indices, **_resnet_kwargs(cfg))


def build_resnet_classifier(
        cfg: ResNetCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    """Build a ResNet classifier (encoder + SpatialLinearHead)."""
    encoder = build_resnet_encoder(cfg, **kwargs)
    head = SpatialLinearHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        pool_type=global_pool or 'avg',
        drop_rate=drop_rate,
    )
    return ImageClassifier(encoder, head)


# Register with the factory
from ._factory import register_family  # noqa: E402  (late import avoids circularity)

register_family(
    family_name='resnet',
    variants=RESNET_CFGS,
    build_classifier=build_resnet_classifier,
    build_encoder=build_resnet_encoder,
    weight_layout=RESNET_WEIGHT_LAYOUT,
    checkpoint_filter_fn=None,  # ResNet checkpoints are stable
)


# ======================================================================
# Example usage (what the user sees)
# ======================================================================
#
#   # Just the encoder — forward() IS the features
#   encoder = create_encoder('resnet50', pretrained=True)
#   features = encoder(images)                  # (B, 2048, 7, 7)
#
#   # Encoder with intermediate features (replaces features_only=True)
#   encoder = create_encoder('resnet50', pretrained=True, out_indices=(0, 1, 2, 3, 4))
#   feature_list = encoder(images)              # [stem, layer1, layer2, layer3, layer4]
#
#   # Full classifier
#   model = create_model('resnet50', pretrained=True, num_classes=1000)
#   logits = model(images)                      # (B, 1000)
#
#   # Classifier split API: forward_features delegates to encoder.forward(),
#   # forward_head delegates to head.forward()
#   features = model.forward_features(images)   # (B, 2048, 7, 7)  — encoder(images)
#   embeds = model.forward_head(features, pre_logits=True)  # (B, 2048) — head(features)
#
#   # Runtime intermediate feature gathering (research path, always available)
#   intermediates = encoder.forward_intermediates(images, indices=[2, 4])
