"""Gemma4 Vision Transformer for timme — encoder/head split.

Mirrors timm.models.gemma4_vit. timme owns the Gemma4Vit encoder class (init /
forward / forward_intermediates). Building blocks (Gemma4PatchEmbed, Gemma4Block,
Gemma4RotaryEmbedding2D, Gemma4VisionPooler) are imported from timm.

Gemma4's vision tower is closely related to NaFlexVit: it consumes NaFlex-style
pre-patchified inputs (``patches`` / ``patch_coord`` / ``patch_valid``) as well as
raw ``(B, C, H, W)`` images, and shares the ``batch_patchify`` helper. It differs
from a standard ViT in: a Linear patch projection with a 2D position-embedding
table, 2D RoPE, gated MLP, QKV RMS-norm, 4-norm sandwich blocks, and a spatial
``k×k`` soft-token pooler.

There are no classifier-trained weights — the natural fit is an encoder. The
soft pool is a native vision-tower *representation* path (spatial k×k pool + √D
scale + optional standardization), not a classifier head pool, so it lives on
the encoder cfg as ``output_pool`` (distinct from the head-side ``global_pool``
that ``create_model`` passes to the classifier head). The default
``output_pool='soft'`` reproduces the native VLM behaviour (bit-perfect with HF
``Gemma4VisionModel`` on matching weights).

Refs:
- Gemma 4 — https://ai.google.dev/gemma/docs/core/model_card_4
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from timm.layers import RmsNorm, get_act_layer, to_2tuple
from timm.models._features import FeatureInfo, feature_take_indices
from timm.models._manipulate import checkpoint, named_apply
from timm.models.gemma4_vit import (
    Gemma4Block,
    Gemma4PatchEmbed,
    Gemma4RotaryEmbedding2D,
    Gemma4VisionPooler,
    get_init_weights_gemma4_vit,
)
from timm.models.gemma4_vit import checkpoint_filter_fn_encoder as _timm1_gemma4_filter_fn

from ..arch import (
    ArchTraits,
    ConfigMixin,
    ImageClassifier,
    ImageEncoder,
    WeightLayout,
)
from ..heads import TokenLinearHead


# ======================================================================
# Architecture config
# ======================================================================


@dataclass
class Gemma4VitCfg(ConfigMixin):
    """Architecture config for the Gemma4Vit encoder family.

    Mirrors timm.models.gemma4_vit.Gemma4VitEncoder.__init__. ``img_size`` is a
    deploy-time default only — the encoder is shape-flexible (NaFlex coords or
    raw images of any conformant size), so it isn't baked into the architecture.
    """

    img_size: Union[int, Tuple[int, int]] = 768
    in_chans: int = 3

    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset({'img_size', 'in_chans'})

    # Architecture
    patch_size: Union[int, Tuple[int, int]] = 16
    embed_dim: int = 768
    depth: int = 16
    num_heads: int = 12
    head_dim: int = 64
    num_kv_heads: Optional[int] = None
    intermediate_size: int = 3072

    # Norm / RoPE
    norm_eps: float = 1e-6
    rope_theta: float = 100.0

    # Patch embedding / pooling
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    standardize: bool = False
    use_clipped_linears: bool = False

    # Native representation pool (NOT a classifier head pool):
    # 'soft' (native VLM k×k soft tokens), 'avg', or 'none'/''.
    output_pool: str = 'soft'

    # Regularization (encoder-local)
    proj_drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # Layer types / init
    act_layer: Optional[str] = None  # None -> GELU(tanh), Gemma4's default
    weight_init: str = ''

    def __post_init__(self):
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if isinstance(self.img_size, list):
            self.img_size = tuple(self.img_size)


def _resolve_act_layer(cfg: Gemma4VitCfg) -> Callable:
    if cfg.act_layer:
        return get_act_layer(cfg.act_layer)
    return partial(nn.GELU, approximate='tanh')


# ======================================================================
# Encoder
# ======================================================================


class Gemma4Vit(ImageEncoder):
    """Gemma4 vision encoder.

    Owns: patch_embed (Linear proj + 2D pos table), rotary_emb (2D RoPE),
    blocks (4-norm sandwich, gated MLP, QKV RMS-norm), pooler (spatial k×k
    soft-token pool), and optional std_bias/std_scale standardization buffers.

    There is no classifier head and no final transformer norm (the sandwich
    blocks norm internally). The output of ``forward()`` depends on ``output_pool``:

    - ``'soft'`` (default): ``(B, num_soft_tokens, embed_dim)`` — spatial k×k
      pool + √D scale (+ standardization for the 31B variant).
    - ``'avg'``: ``(B, embed_dim)`` — masked mean over patch tokens.
    - ``'none'`` / ``''``: ``(B, N, embed_dim)`` — raw patch tokens.
    """

    def __init__(
            self,
            cfg: Optional[Gemma4VitCfg] = None,
            out_indices: Optional[Tuple[int, ...]] = None,
            device=None,
            dtype=None,
            **kwargs,
    ) -> None:
        super().__init__(out_indices=out_indices)

        if cfg is None:
            from dataclasses import fields as _fields

            cfg = Gemma4VitCfg(**{k: v for k, v in kwargs.items() if k in {f.name for f in _fields(Gemma4VitCfg)}})
        else:
            cfg = cfg.overlay(**kwargs)

        dd = {'device': device, 'dtype': dtype}
        assert cfg.output_pool in ('soft', 'avg', 'none', ''), \
            f"output_pool must be one of 'soft', 'avg', 'none' (or ''); got {cfg.output_pool!r}"

        self.cfg = cfg
        self.in_chans = cfg.in_chans
        self.embed_dim = cfg.embed_dim
        self.output_pool = cfg.output_pool
        self.num_prefix_tokens = 0
        self.patch_size = to_2tuple(cfg.patch_size)
        self.pooling_kernel_size = cfg.pooling_kernel_size
        self.use_clipped_linears = cfg.use_clipped_linears
        self.grad_checkpointing = False

        act_layer = _resolve_act_layer(cfg)

        # Linear patch embedding + 2D position-embedding table.
        self.patch_embed = Gemma4PatchEmbed(
            patch_size=self.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            position_embedding_size=cfg.position_embedding_size,
            **dd,
        )

        # 2D RoPE.
        self.rotary_emb = Gemma4RotaryEmbedding2D(
            head_dim=cfg.head_dim,
            rope_theta=cfg.rope_theta,
            **dd,
        )

        # Transformer blocks.
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.depth)]
        self.blocks = nn.ModuleList([
            Gemma4Block(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                num_kv_heads=cfg.num_kv_heads,
                intermediate_size=cfg.intermediate_size,
                norm_eps=cfg.norm_eps,
                attn_drop=cfg.attn_drop_rate,
                proj_drop=cfg.proj_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                use_clipped_linears=cfg.use_clipped_linears,
                **dd,
            )
            for i in range(cfg.depth)
        ])

        # Spatial soft-token pooler (no learnable params).
        self.pooler = Gemma4VisionPooler(
            hidden_size=cfg.embed_dim,
            pooling_kernel_size=cfg.pooling_kernel_size,
        )

        # Optional standardization buffers (31B / 570m variant).
        if cfg.standardize:
            self.register_buffer('std_bias', torch.empty(cfg.embed_dim, **dd))
            self.register_buffer('std_scale', torch.empty(cfg.embed_dim, **dd))
        else:
            self.std_bias = None
            self.std_scale = None

        _red = max(self.patch_size)
        self.feature_info = FeatureInfo(
            [dict(module=f'blocks.{i}', num_chs=cfg.embed_dim, reduction=_red) for i in range(cfg.depth)],
            out_indices=out_indices or (cfg.depth - 1,),
        )

        self.traits = ArchTraits(
            output_fmt='NLC',
            output_dim=cfg.embed_dim,
            num_prefix_tokens=0,
            has_cls_token=False,
            # 'soft' is a Gemma4-native spatial pool, not a standard token pool;
            # expose 'avg' as the classifier-friendly default for head builders.
            default_pool_type='avg',
            supports_variable_input=True,
        )

        self.weight_init_mode = 'reset' if cfg.weight_init == 'skip' else cfg.weight_init
        if cfg.weight_init != 'skip':
            self.init_weights(needs_reset=False)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def init_weights(self, mode: str = '', needs_reset: bool = True) -> None:
        """Trunc-normal-TF Linear init + no-op standardization buffers."""
        mode = mode or self.weight_init_mode
        assert mode in ('', 'reset')
        if self.std_bias is not None:
            nn.init.zeros_(self.std_bias)
        if self.std_scale is not None:
            nn.init.ones_(self.std_scale)
        named_apply(get_init_weights_gemma4_vit(mode, needs_reset=needs_reset), self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'patch_embed.position_embedding_table'}

    @torch.jit.ignore
    def get_patch_size(self) -> Tuple[int, int]:
        return self.patch_size

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return dict(
            stem=r'^patch_embed|^rotary_emb',
            blocks=[(r'^blocks\.(\d+)', None), (r'^pooler|^std_', (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def set_clamp_enabled(self, enabled: bool = True) -> None:
        """Toggle the Gemma4 clippable-linear clamp ops (E4B checkpoints ship
        finite clamp buffers that can stall fine-tuning gradients)."""
        from timm.models.gemma4_vit import Gemma4ClippableLinear

        for mod in self.modules():
            if isinstance(mod, Gemma4ClippableLinear):
                mod.use_clipped = enabled

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _assert_raw_img_conformant(self, x: torch.Tensor) -> None:
        """Raw-image H/W must divide ``patch_size * pooling_kernel_size`` under
        the soft-token pooler so the pool-cell grid is integral. Pre-patchified
        / NaFlex inputs are assumed conformant already."""
        if x.ndim != 4 or self.output_pool != 'soft':
            return
        H, W = x.shape[-2:]
        ph, pw = self.patch_size
        k = self.pooling_kernel_size
        cell_h, cell_w = ph * k, pw * k
        if H % cell_h != 0 or W % cell_w != 0:
            raise ValueError(
                f"Image size ({H}, {W}) must be divisible by "
                f"(patch_size * pooling_kernel_size) = ({cell_h}, {cell_w}) when output_pool='soft'. "
                f"Resize to multiples of ({cell_h}, {cell_w}), or use output_pool='avg'/'none'."
            )

    def _encode(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            padding_positions: torch.Tensor,
            block_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
            max_block_index: Optional[int] = None,
    ) -> torch.Tensor:
        """RoPE + transformer-block pipeline over already-embedded tokens."""
        B, N = x.shape[:2]
        rope_cos, rope_sin = self.rotary_emb(x, position_ids)

        attn_mask: Optional[torch.Tensor] = None
        if padding_positions.any():
            # Column-only additive mask broadcast over heads: (B, 1, 1, N).
            attn_mask = torch.zeros(B, 1, 1, N, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(padding_positions[:, None, None, :], float('-inf'))

        blocks = self.blocks if max_block_index is None else self.blocks[:max_block_index + 1]

        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        for i, blk in enumerate(blocks):
            if do_checkpointing:
                x = checkpoint(blk, x, rope_cos, rope_sin, attn_mask)
            else:
                x = blk(x, rope_cos, rope_sin, attn_mask=attn_mask)
            if block_callback is not None:
                block_callback(i, x)

        return x

    def _apply_pool(
            self,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.output_pool == 'soft':
            x, _ = self.pooler(x, position_ids, padding_positions)
            # Standardization is applied post-pool for 'soft' (HF-native ordering).
            if self.std_bias is not None:
                x = (x - self.std_bias) * self.std_scale
        elif self.output_pool == 'avg':
            if padding_positions.any():
                x = x.masked_fill(padding_positions.unsqueeze(-1), 0.0)
                x = x.sum(dim=1) / (~padding_positions).sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(dim=1)
        # 'none' / '': raw (B, N, D) patch tokens unchanged.
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(
            self,
            x: torch.Tensor,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._assert_raw_img_conformant(x)
        x, position_ids, padding_positions = self.patch_embed(x, patch_coord, patch_valid)
        x = self._encode(x, position_ids, padding_positions)
        return self._apply_pool(x, position_ids, padding_positions)

    def forward_intermediates(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Block-level intermediate features (pre-pool patch tokens).

        ``norm`` is accepted for API symmetry but is a no-op — Gemma4 has no
        final transformer norm (the sandwich blocks norm internally).
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        if isinstance(x, dict):
            patch_coord = x.get('patch_coord', patch_coord)
            patch_valid = x.get('patch_valid', patch_valid)
            x = x['patches']
        raw_input_ndim = x.ndim
        self._assert_raw_img_conformant(x)
        x, position_ids, padding_positions = self.patch_embed(x, patch_coord, patch_valid)

        intermediates: List[torch.Tensor] = []

        def _cb(i: int, y: torch.Tensor) -> None:
            if i in take_indices:
                intermediates.append(y)

        max_block_index = max_index if (stop_early and not torch.jit.is_scripting()) else None
        x = self._encode(
            x,
            position_ids,
            padding_positions,
            block_callback=_cb,
            max_block_index=max_block_index,
        )

        if reshape:
            if raw_input_ndim != 4:
                raise ValueError("output_fmt='NCHW' requires a raw image (B, C, H, W) input.")
            B = position_ids.shape[0]
            pW = int(position_ids[..., 0].max().item()) + 1
            pH = int(position_ids[..., 1].max().item()) + 1
            intermediates = [y.reshape(B, pH, pW, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]
        return take_indices

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
            patch_coord: Optional[torch.Tensor] = None,
            patch_valid: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encoder forward — final features (or intermediates if out_indices set).

        Accepts NaFlex inputs: ``x`` may be a dict with ``patches`` /
        ``patch_coord`` / ``patch_valid`` keys, or a raw image tensor with the
        NaFlex args supplied as kwargs.
        """
        if isinstance(x, dict):
            patch_coord = x.get('patch_coord', patch_coord)
            patch_valid = x.get('patch_valid', patch_valid)
            x = x['patches']

        if self._out_indices is not None:
            return self.forward_intermediates(
                x,
                indices=list(self._out_indices),
                intermediates_only=True,
                patch_coord=patch_coord,
                patch_valid=patch_valid,
                **kwargs,
            )
        return self._forward_features(x, patch_coord=patch_coord, patch_valid=patch_valid)


# ======================================================================
# Weight layout
# ======================================================================

# timm's checkpoint_filter_fn_encoder lands all keys in bare-encoder form
# (patch_embed.*, blocks.*, pooler.*, std_bias/std_scale). There is no
# classifier head in any released checkpoint, so the head side is empty.
GEMMA4_WEIGHT_LAYOUT = WeightLayout(
    encoder=(
        'patch_embed',
        'rotary_emb',
        'blocks',
        'pooler',
        'std_bias',
        'std_scale',
    ),
    head=(),
)


# ======================================================================
# Variant configs
# ======================================================================

# timme is encoder-first, so the variants drop timm's '_enc' suffix (which timm
# uses only to disambiguate from its classifier wrappers). create_encoder pulls
# the same weights — timm's classifier repo nests the identical encoder under
# 'encoder.'; _gemma4_filter_fn strips that, the same way other timme encoders
# drop the head from their timm classifier checkpoints.
GEMMA4_CFGS: Dict[str, Gemma4VitCfg] = {
    # ~167M (E2B/E4B vision tower). E4B checkpoint ships clamp buffers.
    'gemma4_vit_167m': Gemma4VitCfg(
        embed_dim=768,
        depth=16,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        standardize=False,
        use_clipped_linears=True,
        output_pool='soft',
    ),
    # ~570M (26B/31B vision tower). Applies std_bias/std_scale post-pool.
    'gemma4_vit_570m': Gemma4VitCfg(
        embed_dim=1152,
        depth=27,
        num_heads=16,
        head_dim=72,
        intermediate_size=4304,
        standardize=True,
        output_pool='soft',
    ),
}


# ======================================================================
# Head factory + builders + registry
# ======================================================================


def build_gemma4_encoder(
        cfg: Gemma4VitCfg,
        out_indices: Optional[Tuple[int, ...]] = None,
        **kwargs,
) -> Gemma4Vit:
    return Gemma4Vit(cfg=cfg, out_indices=out_indices, **kwargs)


def build_gemma4_classifier(
        cfg: Gemma4VitCfg,
        num_classes: int = 1000,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        **kwargs,
) -> ImageClassifier:
    """Build an avg-pool linear-probe classifier on raw patch tokens.

    No classifier-trained Gemma4 weights exist; this mirrors timm's
    Gemma4VitClassifier default (native soft-pool disabled, masked-mean over
    patch tokens, param-less RmsNorm, linear head) so the model can be
    fine-tuned. ``global_pool`` here is the *head* pool; the encoder's native
    ``output_pool`` is forced to ``'none'`` so the head owns all pooling.
    """
    kwargs.pop('output_pool', None)  # head owns pooling on the classifier path
    cfg = cfg.overlay(output_pool='none', **kwargs)
    encoder = Gemma4Vit(cfg=cfg, **kwargs)
    pool_type = global_pool or 'avg'
    # Param-less RmsNorm matches timm's classifier norm (affine=False) and keeps
    # the head free of params that pretrained encoder checkpoints don't carry.
    norm_layer = partial(RmsNorm, eps=cfg.norm_eps, affine=False)
    head = TokenLinearHead(
        in_features=encoder.output_dim,
        num_classes=num_classes,
        pool_type=pool_type,
        num_prefix_tokens=0,
        norm_layer=norm_layer if pool_type in ('avg', 'avgmax', 'max') else None,
        drop_rate=drop_rate,
    )
    return ImageClassifier(encoder, head)


def _gemma4_filter_fn(state_dict, model):
    target = getattr(model, 'encoder', model)
    return _timm1_gemma4_filter_fn(state_dict, target)


from ._factory import register_family  # noqa: E402

# timme uses the clean (encoder-first) arch names; the weights live under timm's
# '_enc' tags. Remap the pretrained_cfg lookup so create_encoder/create_model
# resolve the right timm repo from the clean name.
GEMMA4_PRETRAINED_ALIASES = {
    'gemma4_vit_167m': 'gemma4_vit_167m_enc',
    'gemma4_vit_570m': 'gemma4_vit_570m_enc',
}

register_family(
    family_name='gemma4_vit',
    variants=GEMMA4_CFGS,
    build_classifier=build_gemma4_classifier,
    build_encoder=build_gemma4_encoder,
    weight_layout=GEMMA4_WEIGHT_LAYOUT,
    checkpoint_filter_fn=_gemma4_filter_fn,
    pretrained_aliases=GEMMA4_PRETRAINED_ALIASES,
)
