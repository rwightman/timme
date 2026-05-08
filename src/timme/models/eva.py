"""EVA / EVA02 / RoPE-ViT for timme — encoder/head split.

Mirrors timm.models.eva. timme owns the Eva encoder class (init / forward
/ forward_intermediates). Building blocks (EvaBlock, EvaBlockPostNorm,
PatchEmbed, AttentionPoolLatent, RoPE helpers) are still imported from timm.

Refs:
- EVA — https://arxiv.org/abs/2211.07636
- EVA-02 — https://arxiv.org/abs/2303.11331
- RoPE-ViT — https://arxiv.org/abs/2403.13298
- DINOv3 — https://arxiv.org/abs/2508.10104
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import (
    AttentionPoolLatent,
    LayerNorm,
    PatchDropoutWithIndices,
    PatchEmbed,
    apply_keep_indices_nlc,
    calculate_drop_path_rates,
    create_rope_embed,
    get_norm_layer,
    resample_abs_pos_embed,
    to_2tuple,
    trunc_normal_,
)
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint
from timm.models.eva import EvaBlock, EvaBlockPostNorm
from timm.models.eva import checkpoint_filter_fn as _timm1_eva_filter_fn

from ..arch import (
    ArchTraits,
    ConfigMixin,
    ImageClassifier,
    ImageEncoder,
    WeightLayout,
)
from ..heads import TokenAttentionHead, TokenLinearHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class EvaCfg(ConfigMixin):
    """Architecture config for the Eva encoder family.

    Mirrors timm.models.eva.Eva.__init__. Head-side fields (num_classes,
    global_pool, drop_rate, head_init_scale) live on the head, not here.
    """

    img_size: Union[int, Tuple[int, int]] = 224
    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'img_size', 'in_chans'})

    # Architecture
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Attention / MLP variants
    qkv_bias: bool = True
    qkv_fused: bool = True
    swiglu_mlp: bool = False
    swiglu_align_to: int = 0
    scale_mlp: bool = False
    scale_attn_inner: bool = False
    attn_type: str = 'eva'

    # Regularization (encoder-local)
    pos_drop_rate: float = 0.0
    patch_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # Init / norms
    norm_layer: str = 'layernorm'
    norm_eps: Optional[float] = None
    init_values: Optional[float] = None

    # Tokens
    class_token: bool = True
    num_reg_tokens: int = 0
    no_embed_class: bool = False

    # Position embeddings
    use_abs_pos_emb: bool = True
    use_rot_pos_emb: bool = False
    rope_type: Optional[str] = 'cat'  # 'cat' | 'mixed' | 'dinov3' | etc.
    rope_grid_offset: float = 0.0
    rope_grid_indexing: str = 'ij'
    rope_temperature: float = 10000.0
    rope_rotate_half: bool = False
    ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None

    # Norm placement
    use_post_norm: bool = False
    use_pre_transformer_norm: bool = False
    use_post_transformer_norm: Optional[bool] = None
    use_fc_norm: Optional[bool] = None

    # Head selection (informs head builder; head modules themselves live on the head)
    global_pool: str = 'avg'
    attn_pool_num_heads: Optional[int] = None
    attn_pool_mlp_ratio: Optional[float] = None

    # Dynamic input
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False

    def __post_init__(self):
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)
        if isinstance(self.ref_feat_shape, list):
            self.ref_feat_shape = tuple(self.ref_feat_shape)


def _resolve_norm_layer(cfg: EvaCfg) -> Callable:
    norm_layer = get_norm_layer(cfg.norm_layer) or LayerNorm
    if cfg.norm_eps is not None:
        norm_layer = partial(norm_layer, eps=cfg.norm_eps)
    return norm_layer


# ======================================================================
# Encoder
# ======================================================================


class Eva(ImageEncoder):
    """EVA / EVA02 / RoPE-ViT encoder.

    Owns: patch_embed, cls/reg tokens, abs pos_embed, rope module,
    pos_drop, patch_drop, norm_pre, blocks, norm.
    Head-side modules (attn_pool, fc_norm, head_drop, head) live in the head.

    The fc_norm / post-transformer-norm placement decision is mirrored from
    timm's Eva so checkpoints align: when fc_norm is enabled, the encoder's
    final norm is Identity and the head owns the post-pool norm.
    """

    def __init__(
            self,
            cfg: Optional[EvaCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        if cfg is None:
            from dataclasses import fields as _fields

            cfg = EvaCfg(**{k: v for k, v in kwargs.items() if k in {f.name for f in _fields(EvaCfg)}})
        else:
            cfg = cfg.overlay(**kwargs)

        dd = {'device': device, 'dtype': dtype}
        assert cfg.global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')

        norm_layer = _resolve_norm_layer(cfg)

        # Resolve norm / pool placement (mirrors timm.models.eva.Eva)
        activate_pre_norm = cfg.use_pre_transformer_norm
        if cfg.use_fc_norm is not None:
            activate_fc_norm = cfg.use_fc_norm
        else:
            activate_fc_norm = cfg.global_pool == 'avg'
        if cfg.use_post_transformer_norm is not None:
            activate_post_norm = cfg.use_post_transformer_norm
        else:
            activate_post_norm = not activate_fc_norm

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.embed_dim = cfg.embed_dim
        self.num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.num_reg_tokens
        self.no_embed_class = cfg.no_embed_class
        self.dynamic_img_size = cfg.dynamic_img_size
        self.grad_checkpointing = False
        self._head_uses_fc_norm = activate_fc_norm  # consumed by head builder

        # Patch embedding
        embed_args: Dict[str, Any] = {}
        if cfg.dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            dynamic_img_pad=cfg.dynamic_img_pad,
            bias=not cfg.use_pre_transformer_norm,
            **embed_args,
            **dd,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else cfg.patch_size

        # Tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim, **dd)) if cfg.class_token else None
        self.reg_token = (
            nn.Parameter(torch.empty(1, cfg.num_reg_tokens, cfg.embed_dim, **dd)) if cfg.num_reg_tokens else None
        )
        self.cls_embed = cfg.class_token and self.reg_token is None

        # Abs pos embed
        num_pos_tokens = num_patches if cfg.no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = (
            nn.Parameter(torch.empty(1, num_pos_tokens, cfg.embed_dim, **dd)) if cfg.use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=cfg.pos_drop_rate)
        if cfg.patch_drop_rate > 0:
            self.patch_drop = PatchDropoutWithIndices(
                cfg.patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = None

        # ROPE
        self.rope_mixed = False
        if cfg.use_rot_pos_emb:
            ref_feat_shape = to_2tuple(cfg.ref_feat_shape) if cfg.ref_feat_shape is not None else None
            rope_kwargs: Dict[str, Any] = dict(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                feat_shape=None if cfg.dynamic_img_size else self.patch_embed.grid_size,
                temperature=cfg.rope_temperature,
                grid_indexing=cfg.rope_grid_indexing,
                **dd,
            )
            if cfg.rope_type == 'mixed':
                rope_kwargs.update(dict(depth=cfg.depth))
                self.rope_mixed = True
            elif cfg.rope_type == 'cat':
                rope_kwargs.update(
                    dict(
                        in_pixels=False,
                        grid_offset=cfg.rope_grid_offset,
                        ref_feat_shape=ref_feat_shape,
                    )
                )
            self.rope = create_rope_embed(rope_type=cfg.rope_type, **rope_kwargs)
        else:
            self.rope = None

        # Norms / blocks
        self.norm_pre = norm_layer(cfg.embed_dim, **dd) if activate_pre_norm else nn.Identity()

        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)
        block_fn = EvaBlockPostNorm if cfg.use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                qkv_bias=cfg.qkv_bias,
                qkv_fused=cfg.qkv_fused,
                mlp_ratio=cfg.mlp_ratio,
                swiglu_mlp=cfg.swiglu_mlp,
                swiglu_align_to=cfg.swiglu_align_to,
                scale_mlp=cfg.scale_mlp,
                scale_attn_inner=cfg.scale_attn_inner,
                attn_type=cfg.attn_type,
                rotate_half=cfg.rope_rotate_half,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=cfg.proj_drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=cfg.init_values,
                **dd,
            )
            for i in range(cfg.depth)
        ])
        self.feature_info = FeatureInfo(
            [dict(module=f'blocks.{i}', num_chs=cfg.embed_dim, reduction=r) for i in range(cfg.depth)],
            out_indices=out_indices or (cfg.depth - 1,),
        )

        self.norm = norm_layer(cfg.embed_dim, **dd) if activate_post_norm else nn.Identity()

        # Traits
        self.traits = ArchTraits(
            output_fmt='NLC',
            output_dim=cfg.embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            has_cls_token=cfg.class_token,
            default_pool_type=cfg.global_pool or 'avg',
            supports_variable_input=cfg.dynamic_img_size,
        )

        self.init_weights(needs_reset=False)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        """Initialize weights — abs pos / cls / reg tokens + per-layer fix init."""
        self.apply(partial(self._init_weights, needs_reset=needs_reset))
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=0.02)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        """Per-layer-id rescaling to attention proj + MLP fc2."""
        with torch.no_grad():
            for layer_id, layer in enumerate(self.blocks):
                scale = math.sqrt(2.0 * (layer_id + 1))
                layer.attn.proj.weight.div_(scale)
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'fc2'):
                    layer.mlp.fc2.weight.div_(scale)

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif needs_reset and hasattr(m, 'reset_parameters') and m is not self:
            m.reset_parameters()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        nwd = {'pos_embed', 'cls_token'}
        if self.rope is not None and hasattr(self.rope, 'no_weight_decay'):
            nwd.update(f'rope.{p}' for p in self.rope.no_weight_decay())
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )

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
        if self.rope is not None:
            self.rope.update_feat_shape(self.patch_embed.grid_size)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                prev_grid_size = self.patch_embed.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W),
                    old_size=prev_grid_size,
                    num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            if pos_embed is not None:
                x = x + pos_embed

        x = self.pos_drop(x)

        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                if self.rope_mixed:
                    rot_pos_embed = rot_pos_embed.transpose(0, 1)
                else:
                    rot_pos_embed = rot_pos_embed.unsqueeze(1)

        return x, rot_pos_embed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_mixed and rot_pos_embed is not None:
            for i, blk in enumerate(self.blocks):
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rot_pos_embed[i], attn_mask=attn_mask, is_causal=is_causal)
                else:
                    x = blk(x, rope=rot_pos_embed[i], attn_mask=attn_mask, is_causal=is_causal)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
                else:
                    x = blk(x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)

        return self.norm(x)

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
        assert output_fmt in ('NCHW', 'NLC'), 'Output format for EVA-ViT features must be NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates: List[torch.Tensor] = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_mixed and rot_pos_embed is not None:
            for i, blk in enumerate(blocks):
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rot_pos_embed[i], attn_mask=attn_mask, is_causal=is_causal)
                else:
                    x = blk(x, rope=rot_pos_embed[i], attn_mask=attn_mask, is_causal=is_causal)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for i, blk in enumerate(blocks):
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
                else:
                    x = blk(x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)

        # Split prefix vs spatial tokens
        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
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
        return self.norm(x), intermediates

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

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self._out_indices is not None:
            return self.forward_intermediates(
                x,
                indices=list(self._out_indices),
                intermediates_only=True,
                attn_mask=attn_mask,
                is_causal=is_causal,
                **kwargs,
            )
        return self._forward_features(x, attn_mask=attn_mask, is_causal=is_causal)


# ======================================================================
# Weight layout
# ======================================================================

EVA_WEIGHT_LAYOUT = WeightLayout(
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
        'rope',
    ),
    head=(
        'attn_pool',
        'fc_norm',
        'head_drop',
        ('head', 'head.fc'),
    ),
)


# ======================================================================
# Variant configs
# ======================================================================

EVA_CFGS: Dict[str, EvaCfg] = {
    # EVA giant — abs pos embed, avg pool, no swiglu
    'eva_giant_patch14_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=6144 / 1408,
    ),
    'eva_giant_patch14_336': EvaCfg(
        img_size=336,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=6144 / 1408,
    ),
    'eva_giant_patch14_560': EvaCfg(
        img_size=560,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=6144 / 1408,
    ),
    # EVA02 — RoPE + SwiGLU + scale_mlp
    'eva02_tiny_patch14_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_small_patch14_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_base_patch14_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_large_patch14_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_tiny_patch14_336': EvaCfg(
        img_size=336,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_small_patch14_336': EvaCfg(
        img_size=336,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_base_patch14_448': EvaCfg(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    'eva02_large_patch14_448': EvaCfg(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    # CLIP variants — pool defaults to 'token', no fc_norm
    'eva_giant_patch14_clip_224': EvaCfg(
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=6144 / 1408,
        global_pool='token',
    ),
    # DINOv3 ViT models — RoPE DINOv3 + storage/register tokens, no abs pos embed.
    'vit_small_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_small_patch16_dinov3_qkvb': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=True,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_small_plus_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=False,
        init_values=1.0e-5,
        swiglu_mlp=True,
        swiglu_align_to=8,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_small_plus_patch16_dinov3_qkvb': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=384,
        depth=12,
        num_heads=6,
        qkv_bias=True,
        init_values=1.0e-5,
        swiglu_mlp=True,
        swiglu_align_to=8,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_base_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_base_patch16_dinov3_qkvb': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_large_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=False,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_large_patch16_dinov3_qkvb': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=True,
        init_values=1.0e-5,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_huge_plus_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        qkv_bias=False,
        init_values=1.0e-5,
        swiglu_mlp=True,
        swiglu_align_to=8,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_huge_plus_patch16_dinov3_qkvb': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        qkv_bias=True,
        init_values=1.0e-5,
        swiglu_mlp=True,
        swiglu_align_to=8,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
    'vit_7b_patch16_dinov3': EvaCfg(
        img_size=256,
        patch_size=16,
        dynamic_img_size=True,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        qkv_bias=False,
        mlp_ratio=2,
        init_values=1.0e-5,
        swiglu_mlp=True,
        swiglu_align_to=64,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type='dinov3',
        rope_temperature=100,
        rope_rotate_half=True,
        num_reg_tokens=4,
        use_fc_norm=False,
        norm_eps=1.0e-5,
    ),
}


# ======================================================================
# Head factory
# ======================================================================


def create_eva_head(
        encoder: Eva,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
):
    cfg = encoder.cfg
    pool_type = global_pool or cfg.global_pool or 'avg'
    embed_dim = encoder.output_dim
    norm_layer = _resolve_norm_layer(cfg)

    if pool_type == 'map':
        attn_pool = AttentionPoolLatent(
            embed_dim,
            num_heads=cfg.attn_pool_num_heads or cfg.num_heads,
            mlp_ratio=cfg.attn_pool_mlp_ratio or cfg.mlp_ratio,
            norm_layer=norm_layer,
            act_layer=nn.GELU,
            device=device,
            dtype=dtype,
        )
        return TokenAttentionHead(
            in_features=embed_dim,
            num_classes=num_classes,
            attn_pool=attn_pool,
            num_prefix_tokens=encoder.traits.num_prefix_tokens,
            pool_include_prefix=False,
            norm_layer=None,
            drop_rate=drop_rate,
            device=device,
            dtype=dtype,
        )

    use_fc_norm = encoder._head_uses_fc_norm
    return TokenLinearHead(
        in_features=embed_dim,
        num_classes=num_classes,
        pool_type=pool_type,
        num_prefix_tokens=encoder.traits.num_prefix_tokens,
        norm_layer=norm_layer if use_fc_norm else None,
        drop_rate=drop_rate,
        device=device,
        dtype=dtype,
    )


# ======================================================================
# Builders + registry
# ======================================================================


def build_eva_encoder(
        cfg: EvaCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> Eva:
    return Eva(cfg=cfg, out_indices=out_indices, **kwargs)


def build_eva_classifier(
        cfg: EvaCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    cfg = cfg.overlay(**kwargs)
    encoder = Eva(cfg=cfg, **kwargs)
    head = create_eva_head(
        encoder,
        num_classes=num_classes,
        global_pool=global_pool,
        drop_rate=drop_rate,
    )
    return ImageClassifier(encoder, head)


def _eva_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_eva_filter_fn(state_dict, target)


from ._factory import register_family  # noqa: E402

register_family(
    family_name='eva',
    variants=EVA_CFGS,
    build_classifier=build_eva_classifier,
    build_encoder=build_eva_encoder,
    weight_layout=EVA_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_eva_filter_fn,
)
