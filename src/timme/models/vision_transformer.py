"""VisionTransformer for timme — encoder/classifier split.

Design:
  - VisionTransformer IS the encoder. Everything through self.norm stays.
    attn_pool, fc_norm, head_drop, head are removed — they become the head.
  - The head type depends on global_pool:
      'token'/'avg'/'max'/''  -> TokenLinearHead (with optional fc_norm)
      'map'                   -> TokenAttentionHead (AttentionPoolLatent)
      'prr'                   -> TokenAttentionHead (AttentionPoolPrr)
  - Old checkpoints remap via VIT_WEIGHT_LAYOUT.

This is the most complex example because of the head variability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Final, FrozenSet, List, Literal, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from timm.layers import (
    LayerType,
    Mlp,
    PatchEmbed,
    PatchDropout,
    AttentionPoolLatent,
    AttentionPoolPrr,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
)
from timm.layers.norm import LayerNorm
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint, checkpoint_seq
from timm.models.vision_transformer import (
    Attention,
    Block,
    calculate_drop_path_rates,
    get_init_weights_vit,
    named_apply,
)

from ..arch import ArchTraits, ImageEncoder, ImageClassifier, DistilledImageClassifier, WeightLayout, remap_state_dict
from ..arch import ConfigMixin
from ..heads import TokenLinearHead, TokenAttentionHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class VisionTransformerCfg(ConfigMixin):
    """Architecture config for VisionTransformer encoder.

    img_size is included because it's architecturally meaningful for ViT —
    it determines pos_embed shape and num_patches (unless dynamic_img_size=True).
    in_chans is always architectural (determines patch_embed input channels).
    Both can be overridden at deploy time via overlay().
    """

    # Architecturally meaningful but deploy-overridable
    img_size: Union[int, Tuple[int, int]] = 224
    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'img_size', 'in_chans'})

    # Core architecture
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Attention
    qkv_bias: bool = True
    qk_norm: bool = False
    scale_attn_norm: bool = False
    scale_mlp_norm: bool = False
    proj_bias: bool = True

    # Tokens and position embedding
    class_token: bool = True
    reg_tokens: int = 0
    no_embed_class: bool = False
    pos_embed: str = 'learn'
    pool_include_prefix: bool = False

    # Regularization (encoder-local drops only, NOT classifier drop_rate)
    pos_drop_rate: float = 0.0
    patch_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # Norms, activations, layer types — all strings for serialization
    norm_layer: str = 'layernorm'
    act_layer: str = 'gelu'
    block_fn: str = 'Block'
    mlp_layer: str = 'Mlp'
    attn_layer: str = 'Attention'
    embed_layer: str = 'PatchEmbed'

    # Architecture knobs
    pre_norm: bool = False
    final_norm: bool = True
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False
    init_values: Optional[float] = None
    weight_init: str = ''
    fix_init: bool = False

    def __post_init__(self):
        # Coerce lists back to tuples (JSON round-trip)
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)


# Layer name -> class resolution for config string references
_LAYER_REGISTRY = {
    'Block': Block,
    'Mlp': Mlp,
    'Attention': Attention,
    'PatchEmbed': PatchEmbed,
}


def _resolve_layer(name: str, default=None):
    """Resolve a string layer name to a class. Falls through to timm layer factories."""
    if name in _LAYER_REGISTRY:
        return _LAYER_REGISTRY[name]
    # For norm/act layers, delegate to timm's existing string resolvers
    resolved = get_norm_layer(name) or get_act_layer(name)
    return resolved or default


class VisionTransformer(ImageEncoder):
    """Vision Transformer encoder.

    Everything through the final norm. No pooling, no classifier.
    Output: (B, num_patches + num_prefix_tokens, embed_dim) in NLC format.

    Can be constructed from:
      - A VisionTransformerCfg dataclass (preferred, serializable)
      - Direct kwargs (backward compat)
    """

    dynamic_img_size: Final[bool]

    def __init__(
            self,
            cfg: Optional[VisionTransformerCfg] = None,
            # Runtime params (never in config)
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            # Direct kwargs — overlaid onto config (deploy overrides + backward compat)
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        # If no config provided, build one from kwargs
        if cfg is None:
            cfg = VisionTransformerCfg(**{
                k: v
                for k, v in kwargs.items()
                if k in {f.name for f in __import__('dataclasses').fields(VisionTransformerCfg)}
            })
        else:
            # Overlay any kwargs (img_size, in_chans, or arch overrides)
            cfg = cfg.overlay(**kwargs)

        # Read deploy fields from the config
        img_size = cfg.img_size
        in_chans = cfg.in_chans

        dd = {'device': device, 'dtype': dtype}
        norm_layer = get_norm_layer(cfg.norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(kwargs.get('embed_norm_layer'))  # not in config yet
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        block_fn = _resolve_layer(cfg.block_fn, Block)
        mlp_layer = _resolve_layer(cfg.mlp_layer, Mlp)
        attn_layer_cls = _resolve_layer(cfg.attn_layer, Attention)
        embed_layer = _resolve_layer(cfg.embed_layer, PatchEmbed)

        self.cfg = cfg
        self.in_chans = in_chans
        self.embed_dim = cfg.embed_dim
        self.num_prefix_tokens = 1 if cfg.class_token else 0
        self.num_prefix_tokens += cfg.reg_tokens
        self.num_reg_tokens = cfg.reg_tokens
        self.has_class_token = cfg.class_token
        self.no_embed_class = cfg.no_embed_class
        self.pool_include_prefix = cfg.pool_include_prefix
        self.dynamic_img_size = cfg.dynamic_img_size
        self.grad_checkpointing = False

        embed_dim = cfg.embed_dim

        # Patch embedding
        embed_args = {}
        if cfg.dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=cfg.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not cfg.pre_norm,
            dynamic_img_pad=cfg.dynamic_img_pad,
            **embed_args,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else cfg.patch_size

        # Tokens and position embedding
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, **dd)) if cfg.class_token else None
        self.reg_token = nn.Parameter(torch.empty(1, cfg.reg_tokens, embed_dim, **dd)) if cfg.reg_tokens else None
        embed_len = num_patches if cfg.no_embed_class else num_patches + self.num_prefix_tokens
        if not cfg.pos_embed or cfg.pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.empty(1, embed_len, embed_dim, **dd))
        self.pos_drop = nn.Dropout(p=cfg.pos_drop_rate)
        self.patch_drop = (
            PatchDropout(cfg.patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
            if cfg.patch_drop_rate > 0
            else nn.Identity()
        )
        self.norm_pre = norm_layer(embed_dim, **dd) if cfg.pre_norm else nn.Identity()

        # Transformer blocks
        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                scale_attn_norm=cfg.scale_attn_norm,
                scale_mlp_norm=cfg.scale_mlp_norm,
                proj_bias=cfg.proj_bias,
                init_values=cfg.init_values,
                proj_drop=cfg.proj_drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                attn_layer=attn_layer_cls,
                depth=i,
                **dd,
            )
            for i in range(cfg.depth)
        ])
        self.feature_info = FeatureInfo(
            [dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(cfg.depth)],
            out_indices=out_indices or (cfg.depth - 1,),
        )

        # Final norm (part of encoder, NOT the head's fc_norm)
        self.norm = norm_layer(embed_dim, **dd) if cfg.final_norm else nn.Identity()

        # --- NO attn_pool, NO fc_norm, NO head_drop, NO head ---
        # Those belong to the head now.

        # Traits
        self.traits = ArchTraits(
            output_fmt='NLC',
            output_dim=embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            has_cls_token=cfg.class_token,
            pool_include_prefix=cfg.pool_include_prefix,
            default_pool_type='token' if cfg.class_token else 'avg',
            supports_variable_input=cfg.dynamic_img_size,
        )

        self.weight_init_mode = 'reset' if cfg.weight_init == 'skip' else cfg.weight_init
        self.fix_init = cfg.fix_init
        if cfg.weight_init != 'skip':
            self.init_weights(needs_reset=False)

    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        mode = mode or self.weight_init_mode
        head_bias = -math.log(1000) if 'nlhb' in mode else 0.0
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias, needs_reset=needs_reset), self)
        if self.fix_init:
            with torch.no_grad():
                for layer_id, layer in enumerate(self.blocks):
                    scale = math.sqrt(2.0 * (layer_id + 1))
                    layer.attn.proj.weight.div_(scale)
                    layer.mlp.fc2.weight.div_(scale)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, 'set_grad_checkpointing'):
            self.patch_embed.set_grad_checkpointing(enable)

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    )
                )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.pos_embed is None:
            return torch.cat(to_cat + [x.view(x.shape[0], -1, x.shape[-1])], dim=1)

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

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    # ------------------------------------------------------------------
    # Forward — encoder's forward() IS the features
    # ------------------------------------------------------------------

    def _forward_features(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        """Final-only fast path — (B, N, embed_dim) in NLC format."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None or is_causal:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    # forward() inherited from ImageEncoder:
    #   - no out_indices  -> calls _forward_features(x, **kwargs)
    #   - with out_indices -> calls forward_intermediates(x, indices=..., intermediates_only=True, **kwargs)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if attn_mask is not None or is_causal:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal)
            elif self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
            if i in take_indices:
                intermediates.append(self.norm(x) if norm else x)

        # Split prefix tokens from spatial tokens
        if self.num_prefix_tokens:
            prefix_tokens = [y[:, :self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]
        if prune_norm:
            self.norm = nn.Identity()
        return take_indices


# ======================================================================
# Weight layout for old timm1 checkpoints
# ======================================================================

VIT_WEIGHT_LAYOUT = WeightLayout(
    encoder=(
        'patch_embed',
        'cls_token',
        'reg_token',
        'pos_embed',
        'pos_drop',
        'patch_drop',
        'norm_pre',
        'blocks',
        'norm',
    ),
    head=(
        'attn_pool',
        'fc_norm',
        'head_drop',
        ('head', 'head.fc'),  # timm1 Linear at 'head' -> TokenLinearHead.fc
    ),
)


# ======================================================================
# Variant configs — keyed by the timm names users will call create_model with
# (resolution baked into the name; img_size on the cfg matches).
# ======================================================================

VIT_CFGS = {
    'vit_tiny_patch16_224': VisionTransformerCfg(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
    ),
    'vit_small_patch16_224': VisionTransformerCfg(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
    ),
    'vit_base_patch16_224': VisionTransformerCfg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
    ),
    'vit_large_patch16_224': VisionTransformerCfg(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
    ),
    'vit_huge_patch14_224': VisionTransformerCfg(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
    ),
    # 384 input
    'vit_base_patch16_384': VisionTransformerCfg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, img_size=384,
    ),
    'vit_large_patch16_384': VisionTransformerCfg(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, img_size=384,
    ),
    # CLIP-style ViT with pre_norm
    'vit_base_patch16_clip_224': VisionTransformerCfg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        pre_norm=True, no_embed_class=True,
    ),
    # DINOv2 register tokens
    'vit_base_patch14_reg4_dinov2': VisionTransformerCfg(
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        reg_tokens=4, no_embed_class=True, class_token=True,
    ),
}


# ======================================================================
# Head factory: select head type based on global_pool
# ======================================================================


def create_vit_head(
        encoder: VisionTransformer,
        num_classes: int = 1000,
        global_pool: str = 'token',
        drop_rate: float = 0.0,
        fc_norm: Optional[bool] = None,
        # For attention pooling
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        device=None,
        dtype=None,
) -> Union[TokenLinearHead, TokenAttentionHead]:
    """Create the appropriate head for a ViT encoder based on pool type."""
    dd = dict(device=device, dtype=dtype)
    _norm_layer = get_norm_layer(norm_layer) or LayerNorm
    embed_dim = encoder.output_dim

    # Determine whether fc_norm is needed
    use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm

    if global_pool == 'map':
        attn_pool = AttentionPoolLatent(
            embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=_norm_layer,
            act_layer=get_act_layer(act_layer) or nn.GELU,
            **dd,
        )
        return TokenAttentionHead(
            in_features=embed_dim,
            num_classes=num_classes,
            attn_pool=attn_pool,
            num_prefix_tokens=encoder.traits.num_prefix_tokens,
            pool_include_prefix=False,
            norm_layer=_norm_layer if use_fc_norm else None,
            drop_rate=drop_rate,
            **dd,
        )

    elif global_pool == 'prr':
        attn_pool = AttentionPoolPrr(
            embed_dim,
            num_heads=num_heads,
            pool_type='token' if encoder.traits.has_cls_token else 'avg',
            norm_layer=_norm_layer,
            **dd,
        )
        return TokenAttentionHead(
            in_features=embed_dim,
            num_classes=num_classes,
            attn_pool=attn_pool,
            num_prefix_tokens=encoder.traits.num_prefix_tokens,
            pool_include_prefix=True,
            norm_layer=_norm_layer if use_fc_norm else None,
            drop_rate=drop_rate,
            **dd,
        )

    else:
        # token / avg / max / avgmax / ''
        return TokenLinearHead(
            in_features=embed_dim,
            num_classes=num_classes,
            pool_type=global_pool,
            num_prefix_tokens=encoder.traits.num_prefix_tokens,
            pool_include_prefix=encoder.traits.pool_include_prefix,
            norm_layer=_norm_layer if use_fc_norm else None,
            drop_rate=drop_rate,
            **dd,
        )


# ======================================================================
# Builders + registry
# ======================================================================


def build_vit_encoder(
        cfg: VisionTransformerCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> VisionTransformer:
    """Build a ViT encoder. Kwargs (img_size, in_chans, arch overrides) are overlaid onto cfg."""
    return VisionTransformer(cfg=cfg, out_indices=out_indices, **kwargs)


def build_vit_classifier(
        cfg: VisionTransformerCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        fc_norm: Optional[bool] = None,
        **kwargs,
) -> ImageClassifier:
    """Build a ViT classifier (encoder + selected token head)."""
    cfg = cfg.overlay(**kwargs)
    encoder = VisionTransformer(cfg=cfg, **kwargs)
    pool_type = global_pool or ('token' if cfg.class_token else 'avg')
    head = create_vit_head(
        encoder,
        num_classes=num_classes,
        global_pool=pool_type,
        drop_rate=drop_rate,
        fc_norm=fc_norm,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        norm_layer=cfg.norm_layer,
        act_layer=cfg.act_layer,
    )
    return ImageClassifier(encoder, head)


# timm's checkpoint_filter_fn inspects the model to handle pos_embed resize /
# patch_embed reshape / CLIP conversion. We pass the encoder to it so those
# attribute lookups resolve (self.patch_embed, self.pos_embed, ...).
from timm.models.vision_transformer import checkpoint_filter_fn as _timm1_vit_filter_fn  # noqa: E402


def _vit_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_vit_filter_fn(state_dict, target, adapt_layer_scale=True)


from ._factory import register_family  # noqa: E402

register_family(
    family_name='vit',
    variants=VIT_CFGS,
    build_classifier=build_vit_classifier,
    build_encoder=build_vit_encoder,
    weight_layout=VIT_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_vit_filter_fn,
)


# ======================================================================
# Example usage
# ======================================================================
#
#   # === Config-based construction (preferred) ===
#
#   cfg = VIT_CFGS['vit_base_patch16']
#   encoder = VisionTransformer(cfg=cfg)           # uses cfg.img_size=224
#   encoder = VisionTransformer(cfg=cfg, img_size=384)  # deploy-time override
#
#   # Customize a variant via overlay
#   my_cfg = VIT_CFGS['vit_base_patch16'].overlay(depth=6, drop_path_rate=0.1)
#   encoder = VisionTransformer(cfg=my_cfg)
#
#   # Serialize / deserialize — config is self-contained (includes img_size, in_chans)
#   json_str = cfg.to_json()   # '{"img_size": 224, "in_chans": 3, "patch_size": 16, ...}'
#   cfg2 = VisionTransformerCfg.from_json(json_str)
#
#   # Architecture-only dict (excludes deploy fields)
#   arch_dict = cfg.to_arch_dict()  # no img_size/in_chans
#
#   # === Factory API ===
#
#   # Encoder only — forward() IS the features
#   encoder = create_encoder('vit_base_patch16_224', pretrained=True)
#   tokens = encoder(images)                  # (B, 197, 768) — NLC
#
#   # Encoder with intermediate features
#   encoder = create_encoder('vit_base_patch16_224', pretrained=True, out_indices=(3, 6, 9, 11))
#   features = encoder(images)                # [block3_out, block6_out, block9_out, block11_out]
#
#   # Classifier with token pooling
#   model = create_model('vit_base_patch16_224', pretrained=True, global_pool='token')
#   logits = model(images)                    # (B, 1000)
#
#   # Classifier split API: forward_features -> encoder(), forward_head -> head()
#   tokens = model.forward_features(images)   # encoder(images) -> (B, 197, 768)
#   embed = model.forward_head(tokens, pre_logits=True)  # head(tokens) -> (B, 768)
