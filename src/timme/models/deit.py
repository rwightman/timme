"""DeiT for timme — encoder/classifier split.

Design:
  - Non-distilled DeiT is just a VisionTransformer variant — see DEIT_CFGS.
    No new encoder class needed; use VisionTransformer + the variant config.
  - Distilled DeiT gets its own encoder (VisionTransformerDistilled) that
    subclasses VisionTransformer and adds a second prefix token (dist_token).
    The pos_embed shape grows to (patches + 2, embed_dim).
  - The distilled classifier pairs two TokenSelectHead instances via
    DistilledImageClassifier:
        head      = TokenSelectHead(token_index=0)  # cls
        head_dist = TokenSelectHead(token_index=1)  # dist

Old checkpoints for distilled DeiT remap via DEIT_DISTILLED_WEIGHT_LAYOUT.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import resample_abs_pos_embed

from ..arch import (
    ArchTraits,
    DistilledImageClassifier,
    ImageClassifier,
    WeightLayout,
    remap_state_dict,
)
from ..heads import TokenLinearHead, TokenSelectHead
from .vision_transformer import VIT_WEIGHT_LAYOUT, VisionTransformer, VisionTransformerCfg


class VisionTransformerDistilled(VisionTransformer):
    """ViT + distillation token.

    Two prefix tokens: [cls_token, dist_token, ...patches].
    Output: (B, num_patches + 2, embed_dim) in NLC.
    """

    def __init__(
            self,
            cfg: Optional[VisionTransformerCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        # Skip parent weight init — we'll run it once after dist_token is added
        parent_weight_init = kwargs.pop('weight_init', '')
        super().__init__(
            cfg=cfg,
            out_indices=out_indices,
            device=device,
            dtype=dtype,
            weight_init='skip',
            **kwargs,
        )
        dd = {'device': device, 'dtype': dtype}

        # Override: two prefix tokens (cls + dist), no reg tokens
        assert self.cfg.class_token, 'distilled ViT requires class_token=True'
        assert self.cfg.reg_tokens == 0, 'distilled ViT does not support reg_tokens'

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.empty(1, 1, self.embed_dim, **dd))

        # Resize pos_embed to include dist slot
        embed_len = self.patch_embed.num_patches
        if not self.cfg.no_embed_class:
            embed_len += self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.empty(1, embed_len, self.embed_dim, **dd))

        # Traits: two prefix tokens now
        self.traits = ArchTraits(
            output_fmt='NLC',
            output_dim=self.embed_dim,
            num_prefix_tokens=2,
            has_cls_token=True,
            pool_include_prefix=self.cfg.pool_include_prefix,
            default_pool_type='token',
            supports_variable_input=self.cfg.dynamic_img_size,
        )

        self.weight_init_mode = 'reset' if parent_weight_init == 'skip' else parent_weight_init
        if parent_weight_init != 'skip':
            self.init_weights(needs_reset=False)

    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        super().init_weights(mode=mode, needs_reset=needs_reset)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^cls_token|dist_token|pos_embed|patch_embed',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = [
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1),
        ]
        if self.no_embed_class:
            x = x + pos_embed
            x = torch.cat(to_cat + [x], dim=1)
        else:
            x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)


# ======================================================================
# DeiT variant configs (flat VisionTransformerCfg values, no new dataclass)
# ======================================================================

DEIT_CFGS = {
    # Non-distilled DeiT / DeiT-3
    'deit_tiny_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
    ),
    'deit_small_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
    ),
    'deit_base_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ),
    # DeiT-3 uses init_values (layer scale) + no_embed_class
    'deit3_small_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_values=1e-6,
        no_embed_class=True,
    ),
    'deit3_base_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-6,
        no_embed_class=True,
    ),
    # Distilled variants — same arch as non-distilled, distinction is head/wrapper
    'deit_tiny_distilled_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
    ),
    'deit_small_distilled_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
    ),
    'deit_base_distilled_patch16_224': VisionTransformerCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ),
}


# ======================================================================
# Weight layout for distilled DeiT
# ======================================================================

# Distilled DeiT: the two classifiers are nn.Linear at the top level in timm1
# (`head.weight`, `head_dist.weight`) — in timme each is wrapped in a
# TokenSelectHead whose Linear sits at `.head`, so we need to push both old
# keys one level deeper into their respective head / head_dist namespaces.
DEIT_DISTILLED_WEIGHT_LAYOUT = WeightLayout(
    encoder=(
        'patch_embed',
        'cls_token',
        'dist_token',
        'pos_embed',
        'pos_drop',
        'patch_drop',
        'norm_pre',
        'blocks',
        'norm',
    ),
    head=(
        ('head', 'head.fc'),  # head.weight      -> head.fc.weight
        ('head_dist', 'head_dist.fc'),  # head_dist.weight -> head_dist.fc.weight
    ),
)


# ======================================================================
# Head factory
# ======================================================================


def create_deit_head(
        encoder: VisionTransformer,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
) -> TokenLinearHead:
    """Non-distilled DeiT uses the standard ViT head (token pool + Linear)."""
    dd = dict(device=device, dtype=dtype)
    return TokenLinearHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        pool_type='token',
        num_prefix_tokens=encoder.traits.num_prefix_tokens,
        drop_rate=drop_rate,
        **dd,
    )


def create_deit_distilled_heads(
        encoder: VisionTransformerDistilled,
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
) -> Tuple[TokenSelectHead, TokenSelectHead]:
    """Return (head_cls, head_dist) — one TokenSelectHead per prefix token."""
    dd = dict(device=device, dtype=dtype)
    head_cls = TokenSelectHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        token_index=0,
        drop_rate=drop_rate,
        **dd,
    )
    head_dist = TokenSelectHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        token_index=1,
        drop_rate=drop_rate,
        **dd,
    )
    return head_cls, head_dist


# ======================================================================
# Factory (sketch — registry + checkpoint loading live in _factory.py)
# ======================================================================

# ======================================================================
# Builders + registry
# ======================================================================


def build_deit_encoder(
        cfg: VisionTransformerCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> VisionTransformer:
    return VisionTransformer(cfg=cfg, out_indices=out_indices, **kwargs)


def build_deit_classifier(
        cfg: VisionTransformerCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    encoder = VisionTransformer(cfg=cfg, **kwargs)
    head = create_deit_head(encoder, num_classes=num_classes, drop_rate=drop_rate)
    return ImageClassifier(encoder, head)


def build_deit_distilled_encoder(
        cfg: VisionTransformerCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> VisionTransformerDistilled:
    return VisionTransformerDistilled(cfg=cfg, out_indices=out_indices, **kwargs)


def build_deit_distilled_classifier(
        cfg: VisionTransformerCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        distilled_training: bool = False,
        **kwargs,
) -> DistilledImageClassifier:
    encoder = VisionTransformerDistilled(cfg=cfg, **kwargs)
    head, head_dist = create_deit_distilled_heads(
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


# DeiT shares ViT's checkpoint_filter_fn (handles pos_embed resize, layer-scale
# rename via adapt_layer_scale=True for DeiT3).
from timm.models.vision_transformer import checkpoint_filter_fn as _timm1_vit_filter_fn  # noqa: E402
from functools import partial as _partial  # noqa: E402


def _deit_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_vit_filter_fn(state_dict, target, adapt_layer_scale=True)


# Split DEIT_CFGS into distilled vs non-distilled by name. Both register to
# the same family but differ in builder + weight layout.
_DEIT_VARIANTS = {k: v for k, v in DEIT_CFGS.items() if 'distilled' not in k}
_DEIT_DISTILLED_VARIANTS = {k: v for k, v in DEIT_CFGS.items() if 'distilled' in k}

from ._factory import register_family  # noqa: E402

register_family(
    family_name='deit',
    variants=_DEIT_VARIANTS,
    build_classifier=build_deit_classifier,
    build_encoder=build_deit_encoder,
    weight_layout=VIT_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_deit_filter_fn,
)
register_family(
    family_name='deit_distilled',
    variants=_DEIT_DISTILLED_VARIANTS,
    build_classifier=build_deit_distilled_classifier,
    build_encoder=build_deit_distilled_encoder,
    weight_layout=DEIT_DISTILLED_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_deit_filter_fn,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   # Non-distilled DeiT — just ViT with a DeiT config
#   model = create_model('deit_base_patch16_224', pretrained=True)
#
#   # Distilled DeiT — produces pair (cls, dist) during distilled training,
#   # averaged at inference
#   model = create_model('deit_base_distilled_patch16_224', pretrained=True)
#   model.distilled_training = True
#   logits_cls, logits_dist = model(images)       # training
#   model.eval()
#   logits = model(images)                        # inference, averaged
