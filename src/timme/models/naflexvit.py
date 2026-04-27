"""NaFlexVit for timme — encoder/head split.

Mirrors timm.models.naflexvit. timme owns the NaFlexVit encoder class
(all init / forward / forward_intermediates / NaFlex grid handling).
Building blocks (NaFlexEmbeds, EvaBlock, RotaryEmbedding*, attention pool,
NaFlex helpers) are still imported from timm for now.

Original paper / refs:
- NaViT — https://arxiv.org/abs/2307.06304
- NaFlex (SigLip-2) — https://arxiv.org/abs/2502.14786
- FlexiViT — https://arxiv.org/abs/2212.08013
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import (
    AttentionPoolLatent,
    LayerNorm,
    Mlp,
    PatchDropoutWithIndices,
    _assert,
    apply_keep_indices_nlc,
    calculate_drop_path_rates,
    disable_compiler,
    get_act_layer,
    get_norm_layer,
)
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint, named_apply
from timm.models.eva import EvaBlock
from timm.models.naflexvit import (
    NaFlexEmbeds,
    NaFlexRopeIterator,
    calculate_naflex_grid_sizes,
    create_attention_mask,
    get_block_fn as _timm1_get_block_fn,
    get_init_weights_vit,
    global_pool_naflex,
)
from timm.models.naflexvit import checkpoint_filter_fn as _timm1_naflexvit_filter_fn
from timm.models.vision_transformer import Block

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
class NaFlexVitCfg(ConfigMixin):
    """Architecture config for NaFlexVit.

    Mirrors timm.models.naflexvit.NaFlexVitCfg, plus deploy-time fields
    (img_size, in_chans) that are constructor args in the timm version.

    img_size is optional — NaFlexVit can run with dynamic shapes when
    ``patch_coord`` / ``patch_valid`` are passed. The img_size value here
    is the default for non-NaFlex paths and informs the embed module's
    grid_size resolution.
    """

    img_size: Optional[Union[int, Tuple[int, int]]] = None
    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'img_size', 'in_chans'})

    # Architecture
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    scale_mlp_norm: bool = False

    # Attention
    qkv_bias: bool = True
    qk_norm: bool = False
    proj_bias: bool = True
    attn_drop_rate: float = 0.0
    scale_attn_inner_norm: bool = False

    # Regularization (encoder-local; head dropout lives on the head)
    init_values: Optional[float] = None
    pos_drop_rate: float = 0.0
    patch_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # Prefix tokens
    class_token: bool = False
    reg_tokens: int = 0

    # Position embedding
    pos_embed: str = 'learned'  # '' | 'none' | 'learned' | 'factorized'
    pos_embed_grid_size: Optional[Tuple[int, int]] = (16, 16)
    pos_embed_interp_mode: str = 'bicubic'
    pos_embed_ar_preserving: bool = False
    pos_embed_use_grid_sample: bool = False

    # ROPE
    rope_type: str = ''  # '' | 'none' | 'axial' | 'mixed' | 'dinov3'
    rope_temperature: float = 10000.0
    rope_ref_feat_shape: Optional[Tuple[int, int]] = None
    rope_grid_offset: float = 0.0
    rope_grid_indexing: str = 'ij'
    rope_rotate_half: bool = False

    # Image processing
    dynamic_img_pad: bool = False

    # Architecture knobs
    pre_norm: bool = False
    final_norm: bool = True
    fc_norm: Optional[bool] = None  # None -> auto (True iff global_pool == 'avg')

    # Pool / head selection (kept on the encoder cfg for self-description;
    # the head factory reads this to pick the right ImageHead)
    global_pool: str = 'map'
    pool_include_prefix: bool = False
    attn_pool_num_heads: Optional[int] = None
    attn_pool_mlp_ratio: Optional[float] = None

    # Init
    weight_init: str = ''
    fix_init: bool = True

    # Embedding
    embed_proj_type: str = 'linear'
    input_norm_layer: Optional[str] = None
    embed_norm_layer: Optional[str] = None

    # Layer types
    norm_layer: Optional[str] = None
    act_layer: Optional[str] = None
    block_fn: Optional[str] = None
    mlp_layer: Optional[str] = None
    attn_layer: Optional[str] = None

    # EVA-style block support
    attn_type: str = 'standard'
    swiglu_mlp: bool = False
    qkv_fused: bool = True

    # Variable patch size
    enable_patch_interpolator: bool = False

    def __post_init__(self):
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)
        if isinstance(self.pos_embed_grid_size, list):
            self.pos_embed_grid_size = tuple(self.pos_embed_grid_size)
        if isinstance(self.rope_ref_feat_shape, list):
            self.rope_ref_feat_shape = tuple(self.rope_ref_feat_shape)


# ======================================================================
# Encoder
# ======================================================================


class NaFlexVit(ImageEncoder):
    """NaFlexVit encoder — encoder/head split.

    Owns the embedding stack (NaFlexEmbeds), optional ROPE module,
    transformer blocks, optional pre/post norms. The pool path
    (attn_pool / fc_norm / head_drop / head) lives in the head.
    """

    def __init__(
            self,
            cfg: Optional[NaFlexVitCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        if cfg is None:
            from dataclasses import fields as _fields

            cfg = NaFlexVitCfg(**{k: v for k, v in kwargs.items() if k in {f.name for f in _fields(NaFlexVitCfg)}})
        else:
            cfg = cfg.overlay(**kwargs)

        dd = {'device': device, 'dtype': dtype}

        assert cfg.global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert cfg.class_token or cfg.global_pool != 'token'
        assert cfg.pos_embed in ('', 'none', 'learned', 'factorized')

        norm_layer = get_norm_layer(cfg.norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(cfg.embed_norm_layer)
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        block_fn = _timm1_get_block_fn(cfg)
        mlp_layer = cfg.mlp_layer or Mlp

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.embed_dim = cfg.embed_dim
        self.num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.reg_tokens
        self.num_reg_tokens = cfg.reg_tokens
        self.has_class_token = cfg.class_token
        self.pool_include_prefix = cfg.pool_include_prefix
        self.grad_checkpointing = False

        # NaFlexEmbeds — patch + abs pos + class/reg tokens, dynamic-shape aware
        self.embeds = NaFlexEmbeds(
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            proj_type=cfg.embed_proj_type,
            proj_bias=not cfg.pre_norm,
            class_token=cfg.class_token,
            reg_tokens=cfg.reg_tokens,
            default_img_size=cfg.img_size,
            dynamic_img_pad=cfg.dynamic_img_pad,
            pos_embed=cfg.pos_embed,
            pos_embed_grid_size=cfg.pos_embed_grid_size,
            pos_embed_interp_mode=cfg.pos_embed_interp_mode,
            pos_embed_ar_preserving=cfg.pos_embed_ar_preserving,
            pos_embed_use_grid_sample=cfg.pos_embed_use_grid_sample,
            proj_norm_layer=embed_norm_layer,
            pos_drop_rate=cfg.pos_drop_rate,
            enable_patch_interpolator=cfg.enable_patch_interpolator,
            **dd,
        )
        self.norm_pre = norm_layer(cfg.embed_dim, **dd) if cfg.pre_norm else nn.Identity()

        # ROPE — optional, with three flavors
        self.rope: Optional[nn.Module] = None
        self.rope_is_mixed = False
        if cfg.rope_type and cfg.rope_type != 'none':
            from timm.layers.pos_embed_sincos import (
                RotaryEmbeddingCat,
                RotaryEmbeddingDinoV3,
                RotaryEmbeddingMixed,
            )

            if cfg.rope_type == 'mixed':
                self.rope = RotaryEmbeddingMixed(
                    cfg.embed_dim,
                    depth=cfg.depth,
                    num_heads=cfg.num_heads,
                    temperature=cfg.rope_temperature,
                    feat_shape=None,
                    grid_indexing=cfg.rope_grid_indexing,
                    **dd,
                )
                self.rope_is_mixed = True
            elif cfg.rope_type == 'axial':
                self.rope = RotaryEmbeddingCat(
                    cfg.embed_dim // cfg.num_heads,
                    temperature=cfg.rope_temperature,
                    in_pixels=False,
                    feat_shape=None,
                    ref_feat_shape=cfg.rope_ref_feat_shape,
                    grid_offset=cfg.rope_grid_offset,
                    grid_indexing=cfg.rope_grid_indexing,
                    **dd,
                )
            elif cfg.rope_type == 'dinov3':
                self.rope = RotaryEmbeddingDinoV3(
                    cfg.embed_dim // cfg.num_heads,
                    temperature=cfg.rope_temperature,
                    feat_shape=None,
                    grid_indexing=cfg.rope_grid_indexing,
                    rotate_half=cfg.rope_rotate_half,
                    **dd,
                )
            else:
                raise ValueError(f"Unknown rope_type: {cfg.rope_type}")

        # Patch dropout (with index tracking for ROPE coordination)
        if cfg.patch_drop_rate > 0:
            self.patch_drop = PatchDropoutWithIndices(
                cfg.patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = None

        # Transformer blocks
        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                proj_bias=cfg.proj_bias,
                init_values=cfg.init_values,
                proj_drop=cfg.proj_drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                depth=i,
                **dd,
            )
            for i in range(cfg.depth)
        ])
        patch_reduction = self.embeds.feat_ratio(as_scalar=True)
        self.feature_info = FeatureInfo(
            [dict(module=f'blocks.{i}', num_chs=cfg.embed_dim, reduction=patch_reduction) for i in range(cfg.depth)],
            out_indices=out_indices or (cfg.depth - 1,),
        )

        # Final encoder norm.
        # timm1 places the post-block norm in either `self.norm` (when fc_norm
        # is False / unused) OR in `self.fc_norm` (head-side, when avg pool
        # uses post-pool norm). Mirror that convention so checkpoints match.
        fc_norm = cfg.fc_norm
        if fc_norm is None:
            fc_norm = cfg.global_pool == 'avg'
        self._head_uses_fc_norm = fc_norm  # consumed by the head builder
        self.norm = norm_layer(cfg.embed_dim, **dd) if (cfg.final_norm and not fc_norm) else nn.Identity()

        self.traits = ArchTraits(
            output_fmt='NLC',
            output_dim=cfg.embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            has_cls_token=cfg.class_token,
            pool_include_prefix=cfg.pool_include_prefix,
            default_pool_type=cfg.global_pool or 'avg',
            supports_variable_input=True,
        )

        self.weight_init_mode = cfg.weight_init
        self.fix_init = cfg.fix_init
        self.init_weights(cfg.weight_init, needs_reset=False)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def fix_init_weight(self) -> None:
        """Apply per-layer-id rescaling to attention proj + MLP fc2."""

        def rescale(param: torch.Tensor, _layer_id: int) -> None:
            with torch.no_grad():
                param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer, 'attn'):
                rescale(layer.attn.proj.weight, layer_id + 1)
            if hasattr(layer, 'mlp'):
                rescale(layer.mlp.fc2.weight, layer_id + 1)
            if hasattr(layer, 'attn_out_proj'):
                rescale(layer.attn_out_proj.weight, layer_id + 1)
            if hasattr(layer, 'mlp_out_proj'):
                rescale(layer.mlp_out_proj.weight, layer_id + 1)

    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        """Run the per-mode weight init from timm1's ViT init helpers."""
        mode = mode or self.weight_init_mode
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        # No num_classes here (head-side); head_bias only matters for nlhb on the head.
        head_bias = 0.0
        named_apply(get_init_weights_vit(mode, head_bias, needs_reset=needs_reset), self)
        if self.fix_init:
            self.fix_init_weight()

    @torch.jit.ignore
    def no_weight_decay(self):
        skip = {'embeds.pos_embed', 'embeds.cls_token', 'embeds.reg_token'}
        if self.rope is not None and hasattr(self.rope, 'no_weight_decay'):
            skip.update(self.rope.no_weight_decay())
        return skip

    @torch.jit.ignore
    def get_patch_size(self) -> Tuple[int, int]:
        return self.embeds.patch_size

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^embeds',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.embeds, 'patch_embed') and hasattr(self.embeds.patch_embed, 'set_grad_checkpointing'):
            self.embeds.patch_embed.set_grad_checkpointing(enable)

    # ------------------------------------------------------------------
    # NaFlex helpers
    # ------------------------------------------------------------------

    @disable_compiler
    def _generate_rope_naflex(
            self,
            x: torch.Tensor,
            patch_coord: torch.Tensor,
    ) -> Union[torch.Tensor, Any]:
        """ROPE embeddings for NaFlex batches with variable grid sizes."""
        naflex_grid_sizes = calculate_naflex_grid_sizes(patch_coord)

        size_to_indices: Dict[Tuple[int, int], List[int]] = {}
        unique_sizes: List[Tuple[int, int]] = []
        for bi, grid_size in enumerate(naflex_grid_sizes):
            if grid_size not in size_to_indices:
                size_to_indices[grid_size] = []
                unique_sizes.append(grid_size)
            size_to_indices[grid_size].append(bi)

        B, N, C = x.shape
        seq_len = N - self.num_prefix_tokens

        if self.rope_is_mixed:
            return NaFlexRopeIterator(
                self.rope,
                size_to_indices,
                unique_sizes,
                B,
                seq_len,
                x.dtype,
                x.device,
            )

        rope_embeds = torch.zeros(B, seq_len, self.rope.dim * 2, dtype=x.dtype, device=x.device)
        if hasattr(self.rope, 'get_batch_embeds'):
            unique_embeds = self.rope.get_batch_embeds(unique_sizes)
            for grid_size, embed, batch_indices in zip(unique_sizes, unique_embeds, size_to_indices.values()):
                h, w = grid_size
                actual_len = h * w
                for bi in batch_indices:
                    rope_embeds[bi, :actual_len] = embed[:actual_len]
        else:
            for grid_size, batch_indices in size_to_indices.items():
                rope_embed = self.rope.get_embed(shape=grid_size)
                h, w = grid_size
                actual_len = h * w
                rope_embeds[batch_indices, :actual_len] = rope_embed[:actual_len]
        return rope_embeds.unsqueeze(1)

    def _forward_embeds(
            self,
            x: torch.Tensor,
            patch_coord: Optional[torch.Tensor],
            patch_valid: Optional[torch.Tensor],
            attn_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Patch + abs pos + ROPE + patch-dropout + attention-mask preparation."""
        naflex_mode = patch_coord is not None

        x, grid_size = self.embeds(
            x,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
        )

        rope_embeds = None
        if self.rope is not None:
            if patch_coord is not None:
                rope_embeds = self._generate_rope_naflex(x, patch_coord)
            elif grid_size is not None:
                rope_embeds = self.rope.get_embed(shape=grid_size)
            else:
                assert False, 'Expected one of patch_coord or grid_size to be valid'

        keep_indices: Optional[torch.Tensor] = None
        if self.training and self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if patch_valid is not None:
                patch_valid = patch_valid.gather(1, keep_indices)
            if rope_embeds is not None and not self.rope_is_mixed:
                rope_embeds = apply_keep_indices_nlc(
                    x,
                    rope_embeds,
                    keep_indices,
                    pos_embed_has_batch=naflex_mode,
                )
                if not naflex_mode:
                    rope_embeds = rope_embeds.unsqueeze(1)

        if attn_mask is None:
            attn_mask = create_attention_mask(
                patch_valid,
                num_prefix_tokens=self.num_prefix_tokens,
                dtype=x.dtype,
            )

        x = self.norm_pre(x)
        return {
            'patches': x,
            'patch_valid': patch_valid,
            'rope_embeds': rope_embeds,
            'attn_mask': attn_mask,
            'keep_indices': keep_indices,
            'naflex_mode': naflex_mode,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(
            self,
            x: torch.Tensor,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        naflex_mode = patch_coord is not None

        embeds = self._forward_embeds(
            x,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds['patches']
        rope_embeds = embeds.get('rope_embeds', None)
        keep_indices = embeds.get('keep_indices', None)
        attn_mask = embeds.get('attn_mask', None)

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            for _i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embed, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
        elif rope_embeds is not None:
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embeds, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, attn_mask=attn_mask)
                else:
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        if naflex_mode:
            return {
                'patches': x,
                'patch_valid': embeds.get('patch_valid', None),
            }
        return x

    def forward_intermediates(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Intermediate-feature extraction with optional NaFlex inputs."""
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates: List[torch.Tensor] = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        if isinstance(x, Dict):
            patch_coord = x['patch_coord']
            patch_valid = x['patch_valid']
            patches = x['patches']
            assert False, 'WIP, patch dict input not yet supported on intermediates path'
        else:
            patches = x
            height, width = x.shape[-2:]
            H, W = self.embeds.dynamic_feat_size((height, width))

        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )
        x = embeds['patches']
        rope_embeds = embeds.get('rope_embeds', None)
        keep_indices = embeds.get('keep_indices', None)
        attn_mask = embeds.get('attn_mask', None)

        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=embeds.get('naflex_mode', False),
                    )
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embed, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for i, blk in enumerate(blocks):
                if rope_embeds is not None:
                    if do_checkpointing:
                        x = checkpoint(blk, x, rope=rope_embeds, attn_mask=attn_mask)
                    else:
                        x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
                else:
                    if do_checkpointing:
                        x = checkpoint(blk, x, attn_mask=attn_mask)
                    else:
                        x = blk(x, attn_mask=attn_mask)
                if i in take_indices:
                    intermediates.append(self.norm(x) if norm else x)

        if self.num_prefix_tokens:
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            intermediates = [y.reshape(y.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if output_dict:
            result_dict: Dict[str, Any] = {'image_intermediates': intermediates}
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict['image_intermediates_prefix'] = prefix_tokens
            if not intermediates_only:
                result_dict['image_features'] = self.norm(x)
            return result_dict

        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates
        return self.norm(x), intermediates

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """Encoder forward — returns final features (or intermediates if out_indices set).

        Supports NaFlex inputs: ``x`` may be a dict with ``patches`` /
        ``patch_coord`` / ``patch_valid`` keys (from a NaFlex collator), or a
        plain image tensor with the NaFlex args supplied as kwargs.
        """
        # Unwrap NaFlex dict input
        input_is_dict = isinstance(x, Dict)
        if input_is_dict:
            patches = x['patches']
            patch_valid = x.get('patch_valid', patch_valid)
            patch_coord = x.get('patch_coord', patch_coord)
            attn_mask = x.get('attn_mask', attn_mask)
            x = patches

        if self._out_indices is not None:
            return self.forward_intermediates(
                x,
                indices=list(self._out_indices),
                intermediates_only=True,
                patch_coord=patch_coord,
                patch_valid=patch_valid,
                attn_mask=attn_mask,
                **kwargs,
            )
        return self._forward_features(
            x,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
        )


# ======================================================================
# Weight layout for upstream timm1 NaFlexVit checkpoints
# ======================================================================

# After timm1's checkpoint_filter_fn runs, upstream keys land at:
#   embeds.*       (patch_embed, pos_embed, cls_token, reg_token)
#   norm_pre.*     (when pre_norm=True; SigLIP variants don't set it)
#   blocks.*
#   norm.*         (when not fc_norm)
#   rope.*         (when rope_type set)
#   attn_pool.*    (when global_pool == 'map')
#   fc_norm.*      (when fc_norm True)
#   head.*         (Linear, num_classes>0; SigLIP feature-only sets num_classes=0)
NAFLEXVIT_WEIGHT_LAYOUT = WeightLayout(
    encoder=(
        'embeds',
        'norm_pre',
        'rope',
        'patch_drop',
        'blocks',
        'norm',
    ),
    head=(
        'attn_pool',
        'fc_norm',
        ('head', 'head.fc'),  # timm1 Linear at 'head' -> TokenLinearHead/TokenAttentionHead .fc
    ),
)


# ======================================================================
# Variant configs
# ======================================================================

NAFLEXVIT_CFGS: Dict[str, NaFlexVitCfg] = {
    'naflexvit_base_patch16_gap': NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    ),
    'naflexvit_base_patch16_par_gap': NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        pos_embed_ar_preserving=True,
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    ),
    'naflexvit_base_patch16_parfac_gap': NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        pos_embed_ar_preserving=True,
        pos_embed='factorized',
        global_pool='avg',
        reg_tokens=4,
        fc_norm=True,
    ),
    'naflexvit_base_patch16_map': NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        global_pool='map',
        reg_tokens=1,
    ),
    'naflexvit_so150m2_patch16_reg1_gap': NaFlexVitCfg(
        patch_size=16,
        embed_dim=832,
        depth=21,
        num_heads=13,
        mlp_ratio=34 / 13,
        init_values=1e-5,
        qkv_bias=False,
        reg_tokens=1,
        global_pool='avg',
        fc_norm=True,
    ),
    'naflexvit_so150m2_patch16_reg1_map': NaFlexVitCfg(
        patch_size=16,
        embed_dim=832,
        depth=21,
        num_heads=13,
        mlp_ratio=34 / 13,
        init_values=1e-5,
        qkv_bias=False,
        reg_tokens=1,
        global_pool='map',
    ),
    'naflexvit_base_patch16_siglip': NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        act_layer='gelu_tanh',
        global_pool='map',
    ),
    'naflexvit_so400m_patch16_siglip': NaFlexVitCfg(
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        act_layer='gelu_tanh',
        global_pool='map',
    ),
}


# ======================================================================
# Head factory
# ======================================================================


def create_naflexvit_head(
        encoder: NaFlexVit,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
):
    """Pick TokenLinearHead (avg/token pool) or TokenAttentionHead (map pool)
    matching the encoder's cfg.global_pool."""
    cfg = encoder.cfg
    pool_type = global_pool or cfg.global_pool or 'avg'
    embed_dim = encoder.output_dim
    norm_layer = get_norm_layer(cfg.norm_layer) or LayerNorm
    act_layer = get_act_layer(cfg.act_layer) or nn.GELU

    if pool_type == 'map':
        attn_pool = AttentionPoolLatent(
            embed_dim,
            num_heads=cfg.attn_pool_num_heads or cfg.num_heads,
            mlp_ratio=cfg.attn_pool_mlp_ratio or cfg.mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            device=device,
            dtype=dtype,
        )
        return TokenAttentionHead(
            in_features=embed_dim,
            num_classes=num_classes,
            attn_pool=attn_pool,
            num_prefix_tokens=encoder.traits.num_prefix_tokens,
            pool_include_prefix=encoder.pool_include_prefix,
            norm_layer=None,  # 'map' variants don't apply fc_norm
            drop_rate=drop_rate,
            device=device,
            dtype=dtype,
        )

    # Avg / token pool — use TokenLinearHead with optional fc_norm
    use_fc_norm = encoder._head_uses_fc_norm
    return TokenLinearHead(
        in_features=embed_dim,
        num_classes=num_classes,
        pool_type=pool_type,
        num_prefix_tokens=encoder.traits.num_prefix_tokens,
        pool_include_prefix=encoder.pool_include_prefix,
        norm_layer=norm_layer if use_fc_norm else None,
        drop_rate=drop_rate,
        device=device,
        dtype=dtype,
    )


# ======================================================================
# Builders + registry
# ======================================================================


def build_naflexvit_encoder(
        cfg: NaFlexVitCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> NaFlexVit:
    return NaFlexVit(cfg=cfg, out_indices=out_indices, **kwargs)


def build_naflexvit_classifier(
        cfg: NaFlexVitCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    cfg = cfg.overlay(**kwargs)
    encoder = NaFlexVit(cfg=cfg, **kwargs)
    head = create_naflexvit_head(
        encoder,
        num_classes=num_classes,
        global_pool=global_pool,
        drop_rate=drop_rate,
    )
    return ImageClassifier(encoder, head)


def _naflexvit_filter_fn(state_dict, model):
    """Wrap timm1's checkpoint_filter_fn so its model-introspection lookups
    (model.embeds.* shape checks) resolve against our encoder."""
    target = getattr(model, 'encoder', model)
    return _timm1_naflexvit_filter_fn(state_dict, target)


from ._factory import register_family  # noqa: E402

register_family(
    family_name='naflexvit',
    variants=NAFLEXVIT_CFGS,
    build_classifier=build_naflexvit_classifier,
    build_encoder=build_naflexvit_encoder,
    weight_layout=NAFLEXVIT_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_naflexvit_filter_fn,
)
