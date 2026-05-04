"""NEPA (Next Embedding Prediction Architecture) training task.

NEPA is a self-supervised learning method for Vision Transformers that trains
the model to predict input embeddings from output embeddings, either at shifted
positions (predict next token) or same positions.

This implementation trains bare timme ViT/EVA-style encoders without requiring
architecture modifications.

Optionally includes pixel-space prediction for hybrid embedding + pixel loss.
"""

import logging
from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, GluMlp, SwiGLU, LayerScale, DropPath
from timm.data import patchify_image
from timm.models._manipulate import checkpoint, checkpoint_seq
from timm.utils import unwrap_model
from .task import TrainingTask
from .eval_task import FeatureEvalModel, SSLEvalTask

_logger = logging.getLogger(__name__)


# Map string names to MLP layer classes
MLP_LAYER_MAP = {
    'mlp': Mlp,
    'glu': GluMlp,
    'swiglu': SwiGLU,
}


def _resolve_encoder(model: nn.Module) -> nn.Module:
    """Return the encoder inside a timme classifier, DDP, or EMA wrapper."""
    model = unwrap_model(model)
    return unwrap_model(getattr(model, 'encoder', model))


def _is_supported_encoder(encoder: nn.Module) -> bool:
    return all(hasattr(encoder, name) for name in ('patch_embed', '_pos_embed', '_forward_features'))


def prediction_loss(
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        shift: bool = True,
        num_prefix_tokens: int = 0,
) -> torch.Tensor:
    """NEPA prediction loss - negative cosine similarity between embeddings.

    Computes the similarity between input embeddings (target) and output
    embeddings (prediction), optionally with a position shift for next-token
    prediction.

    Args:
        h_in: Input embeddings [B, N, D] (target, will be detached)
        h_out: Output embeddings [B, N, D] (prediction)
        shift: If True, compare h_out[:, :-1] with h_in[:, 1:] (predict next).
               If False, compare at same positions.
        num_prefix_tokens: Number of prefix tokens (CLS, register, etc.) to exclude
                          from the loss computation. These tokens have no spatial
                          relationship to adjacent patch tokens.

    Returns:
        Scalar loss (negative cosine similarity, lower is better)
    """
    # Detach target to prevent gradient flow
    h_in = h_in.detach()

    # Exclude prefix tokens (CLS, register, etc.) - they have no spatial relationship
    # to adjacent patch tokens, so shifted prediction doesn't make sense for them
    if num_prefix_tokens > 0:
        h_in = h_in[:, num_prefix_tokens:, :]
        h_out = h_out[:, num_prefix_tokens:, :]

    if shift:
        # Predict next position: output[t] predicts input[t+1]
        prediction = h_out[:, :-1, :]  # Predictions
        target = h_in[:, 1:, :]  # Targets (shifted)
    else:
        # Same position prediction
        prediction = h_out
        target = h_in

    # L2 normalize
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)

    # Negative cosine similarity (mean over all positions and batches)
    loss = -(prediction * target).sum(dim=-1).mean()

    return loss


def pixel_prediction_loss(
        pixel_pred: torch.Tensor,
        images: torch.Tensor,
        patch_size: int,
        num_prefix_tokens: int = 1,
        norm_target: bool = True,
        loss_type: str = 'mse',
) -> torch.Tensor:
    """Compute pixel prediction loss for next-patch prediction.

    Args:
        pixel_pred: Predicted pixels [B, N_patches-1, patch_pixels]
        images: Original images [B, C, H, W] (may be normalized)
        patch_size: Patch size for patchifying
        num_prefix_tokens: Number of prefix tokens (CLS, etc.) to skip
        norm_target: Normalize target patches per-patch (helps with brightness variation)
        loss_type: 'mse' or 'l1'

    Returns:
        Scalar loss
    """
    B = images.shape[0]

    # Patchify images to get target patches
    # patchify_image works on single images, so we loop (or could batch)
    patches_list = []
    for i in range(B):
        patches, _, _ = patchify_image(
            images[i],
            (patch_size, patch_size),
            pad=False,
            include_info=True,
            flatten_patches=True,
        )
        patches_list.append(patches)
    target = torch.stack(patches_list, dim=0)  # [B, N_patches, patch_pixels]

    # Shift to get next-patch targets
    # pixel_pred[t] predicts patch[t+1], so target is patches[1:]
    target = target[:, 1:, :]  # [B, N_patches-1, patch_pixels]

    if norm_target:
        # Per-patch normalization (zero mean, unit variance)
        target = (target - target.mean(dim=-1, keepdim=True)) / (target.std(dim=-1, keepdim=True) + 1e-6)

    if loss_type == 'mse':
        loss = F.mse_loss(pixel_pred, target)
    elif loss_type == 'l1':
        loss = F.l1_loss(pixel_pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


class ResidualMlpBlock(nn.Module):
    """Residual MLP block with LayerScale and DropPath.

    Similar to a transformer block but MLP-only: pre-norm, MLP, residual
    connection, with optional LayerScale and stochastic depth.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (MLP expansion)
        out_features: Output dimension (default: same as in_features)
        mlp_layer: MLP class (Mlp, SwiGLU, GluMlp)
        norm_layer: Normalization layer class
        act_layer: Activation layer (for non-gated MLPs)
        drop: Dropout rate in MLP
        drop_path: Stochastic depth rate
        init_values: LayerScale initial value (None to disable)
        bias: Whether to use bias in MLP
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: Optional[int] = None,
            mlp_layer: Type[nn.Module] = Mlp,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.0,
            drop_path: float = 0.0,
            init_values: Optional[float] = None,
            bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features

        self.norm = norm_layer(in_features)

        # Build MLP - handle different signatures for gated vs standard
        mlp_kwargs = dict(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
            drop=drop,
        )
        # Only add act_layer for non-gated MLPs (gated have built-in activation)
        if mlp_layer not in (SwiGLU, GluMlp):
            mlp_kwargs['act_layer'] = act_layer
        self.mlp = mlp_layer(**mlp_kwargs)

        self.ls = LayerScale(out_features, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Projection if dimensions change
        self.proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x)
        x = shortcut + self.drop_path(self.ls(self.mlp(self.norm(x))))
        return x


class PixelDecoder(nn.Module):
    """Decoder that predicts patch pixels from embeddings.

    Takes output embeddings and predicts raw pixel values for the next patch.
    Uses a stack of residual MLP blocks followed by a linear projection
    to patch pixel space.

    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden dimension in MLP blocks
        patch_size: Patch size (pixels)
        in_chans: Number of image channels
        depth: Number of residual MLP blocks
        mlp_layer: MLP variant ('mlp', 'swiglu', 'glu') or class
        mlp_ratio: Hidden dim multiplier for MLP (alternative to hidden_dim)
        norm_layer: Normalization layer class
        act_layer: Activation layer (for non-gated MLPs)
        drop: Dropout rate
        drop_path_rate: Stochastic depth rate (linearly increases)
        init_values: LayerScale initial value (None to disable)
        bias: Whether to use bias in MLPs
        norm_pred: Normalize predictions per-patch
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: Optional[int] = None,
            patch_size: int = 16,
            in_chans: int = 3,
            depth: int = 2,
            mlp_layer: Union[str, Type[nn.Module]] = 'mlp',
            mlp_ratio: float = 4.0,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.0,
            drop_path_rate: float = 0.0,
            init_values: Optional[float] = None,
            bias: bool = True,
            norm_pred: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.output_dim = patch_size * patch_size * in_chans
        self.norm_pred = norm_pred

        # Resolve hidden dimension
        hidden_dim = hidden_dim or int(embed_dim * mlp_ratio)

        # Resolve MLP layer type
        if isinstance(mlp_layer, str):
            mlp_layer = MLP_LAYER_MAP[mlp_layer]

        # Build residual MLP stack with linearly increasing drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks = []
        for i in range(depth):
            blocks.append(
                ResidualMlpBlock(
                    in_features=embed_dim if i == 0 else hidden_dim,
                    hidden_features=hidden_dim,
                    out_features=hidden_dim,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop=drop,
                    drop_path=dpr[i],
                    init_values=init_values,
                    bias=bias,
                )
            )
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()

        # Handle case where depth=0 (direct projection)
        final_dim = hidden_dim if depth > 0 else embed_dim

        # Final norm and projection to pixel space
        self.norm = norm_layer(final_dim)
        self.head = nn.Linear(final_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        # Small init for output projection (like MAE decoder)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings [B, N, embed_dim]

        Returns:
            Predicted pixels [B, N, patch_size * patch_size * in_chans]
        """
        x = self.blocks(x)
        x = self.head(self.norm(x))

        if self.norm_pred:
            # Normalize predictions per-patch
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

        return x


def create_causal_mask(
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_prefix_tokens: int = 0,
        prefix_bidirectional: bool = False,
) -> torch.Tensor:
    """Create a causal attention mask for next-token prediction.

    Args:
        seq_len: Sequence length (num_prefix_tokens + num_patches)
        device: Device to create mask on
        dtype: Data type for mask (should match model dtype)
        num_prefix_tokens: Number of prefix tokens (CLS, register, etc.)
        prefix_bidirectional: If True, prefix tokens can attend to each other bidirectionally.
            If False, prefix tokens are also causally masked.

    Returns:
        Causal mask [seq_len, seq_len] where:
        - If prefix_bidirectional=True: prefix tokens attend to all prefix tokens,
          patch tokens are causal (can attend to all prefix + previous patches)
        - If prefix_bidirectional=False: fully causal (position i attends to positions <= i)
        Mask values are 0 (can attend) or -inf (cannot attend).
    """
    # Start with upper triangular mask (True = cannot attend)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    # If prefix_bidirectional, allow prefix tokens to attend to each other
    if prefix_bidirectional and num_prefix_tokens > 1:
        # Unmask the prefix-to-prefix region (top-left block)
        mask[:num_prefix_tokens, :num_prefix_tokens] = False

    # Convert to float mask: 0 for can attend, -inf for cannot attend
    return mask.float().masked_fill(mask, float('-inf')).to(dtype)


class NEPATrainableModule(nn.Module):
    """Trainable module wrapper for NEPA training with timme ViT/EVA encoders.

    Wraps a timm Vision Transformer to provide access to both input embeddings
    (after patch embedding + position embedding) and output embeddings (after
    transformer blocks) needed for NEPA's prediction loss.

    Optionally includes a pixel decoder for hybrid embedding + pixel prediction.

    Uses the model's forward_intermediates() method with return_input_embeddings
    for clean integration without replicating forward logic.

    Compatible with:
        - vision_transformer.py (VisionTransformer, VisionTransformerDistilled, etc.)
        - eva.py (Eva, etc.)
        - Other ViTs with forward_intermediates() supporting return_input_embeddings

    Args:
        model: A timme ViT/EVA-style encoder
        pixel_decoder: Optional PixelDecoder for pixel prediction
        causal: If True, use causal masking to prevent attending to future tokens.
                Required for next-token prediction (shift=True in NEPATask).
        prefix_bidirectional: If True (default), prefix tokens (CLS, registers) can attend
                to each other bidirectionally while patches remain causal. If False,
                fully causal masking is applied to all tokens.

    Attributes:
        model: The wrapped ViT model
        pixel_decoder: Optional pixel prediction decoder
        causal: Whether causal masking is enabled
        prefix_bidirectional: Whether prefix tokens have bidirectional attention
    """

    def __init__(
            self,
            model: nn.Module,
            pixel_decoder: Optional[PixelDecoder] = None,
            causal: bool = False,
            prefix_bidirectional: bool = False,
    ):
        super().__init__()
        encoder = _resolve_encoder(model)
        if not _is_supported_encoder(encoder):
            raise ValueError(
                f"Model {model.__class__.__name__} must expose a timme ViT/EVA-style encoder with "
                "'patch_embed', '_pos_embed', and '_forward_features'."
            )
        self.model = model
        self.encoder = encoder
        self.pixel_decoder = pixel_decoder
        self.causal = causal
        self.prefix_bidirectional = prefix_bidirectional

    def _forward_embeddings(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder = self.encoder
        x = encoder.patch_embed(x)
        pos_result = encoder._pos_embed(x)
        rot_pos_embed = None
        if isinstance(pos_result, tuple):
            x, rot_pos_embed = pos_result
        else:
            x = pos_result

        if not isinstance(pos_result, tuple) and hasattr(encoder, 'patch_drop') and encoder.patch_drop is not None:
            x = encoder.patch_drop(x)
        x = encoder.norm_pre(x)
        input_embeddings = x

        do_checkpointing = getattr(encoder, 'grad_checkpointing', False) and not torch.jit.is_scripting()
        if rot_pos_embed is not None or hasattr(encoder, 'rope_mixed'):
            if getattr(encoder, 'rope_mixed', False) and rot_pos_embed is not None:
                for i, blk in enumerate(encoder.blocks):
                    rope = rot_pos_embed[i]
                    if do_checkpointing:
                        x = checkpoint(blk, x, rope=rope, attn_mask=attn_mask, is_causal=is_causal)
                    else:
                        x = blk(x, rope=rope, attn_mask=attn_mask, is_causal=is_causal)
            else:
                for blk in encoder.blocks:
                    if do_checkpointing:
                        x = checkpoint(blk, x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
                    else:
                        x = blk(x, rope=rot_pos_embed, attn_mask=attn_mask, is_causal=is_causal)
        elif attn_mask is not None or is_causal:
            for blk in encoder.blocks:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal)
        elif do_checkpointing:
            x = checkpoint_seq(encoder.blocks, x)
        else:
            x = encoder.blocks(x)

        return input_embeddings, encoder.norm(x)

    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning embeddings and optional pixel predictions.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary containing:
                - 'output_embeddings': After transformer blocks + norm [B, N, D]
                - 'input_embeddings': After pos embed, before blocks [B, N, D]
                - 'pixel_pred': Predicted pixels [B, N_patches-1, patch_pixels] (if decoder enabled)
        """
        # Determine masking strategy:
        # - If not causal, no masking needed
        # - If causal and prefix_bidirectional=False, use fast is_causal flag
        # - If causal and prefix_bidirectional=True, need custom mask for bidirectional prefix
        attn_mask = None
        is_causal = False

        if self.causal:
            if not self.prefix_bidirectional:
                # Use fast is_causal path (fully causal, no special prefix handling)
                is_causal = True
            else:
                # Need custom mask for bidirectional prefix attention
                patch_embed = self.encoder.patch_embed
                H, W = patch_embed.dynamic_feat_size((x.shape[2], x.shape[3]))
                num_patches = H * W
                num_prefix = getattr(self.encoder, 'num_prefix_tokens', 1)
                seq_len = num_prefix + num_patches
                attn_mask = create_causal_mask(
                    seq_len,
                    x.device,
                    x.dtype,
                    num_prefix_tokens=num_prefix,
                    prefix_bidirectional=self.prefix_bidirectional,
                )

        input_embeddings, output_embeddings = self._forward_embeddings(
            x,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        output = {
            'output_embeddings': output_embeddings,
            'input_embeddings': input_embeddings,
        }

        if self.pixel_decoder is not None:
            # Predict pixels for next-patch prediction
            # Exclude prefix tokens (CLS, etc.) and last token (no next target)
            num_prefix = getattr(self.encoder, 'num_prefix_tokens', 1)
            patch_embeds = output_embeddings[:, num_prefix:-1, :]  # [B, N_patches-1, D]
            output['pixel_pred'] = self.pixel_decoder(patch_embeds)

        return output


class NEPATask(TrainingTask):
    """NEPA (Next Embedding Prediction Architecture) training task.

    Self-supervised task that trains a ViT to predict input embeddings from
    output embeddings. The prediction target is the input embedding at either
    the next position (shift=True) or same position (shift=False).

    Optionally includes pixel-space prediction for hybrid loss.

    This task wraps bare timme ViT/EVA encoders without requiring architecture changes.

    Args:
        model: A timme ViT/EVA-style ImageEncoder.
        shift: If True, predict next position (h_out[t] -> h_in[t+1]).
               If False, predict same position (default: True)
        pixel_decoder_cfg: Config dict for PixelDecoder (None to disable pixel loss)
        pixel_loss_weight: Weight for pixel prediction loss (default: 1.0)
        pixel_loss_type: Pixel loss type ('mse' or 'l1')
        norm_pixel_target: Normalize pixel targets per-patch
        prefix_bidirectional: If True (default), prefix tokens (CLS, registers) can attend
            to each other bidirectionally while patches remain causal. If False, fully
            causal masking is applied to all tokens including prefix.
        device: Device for task components
        dtype: Data type for task components
        verbose: Whether to log task configuration

    Example:
        >>> # Basic NEPA (embedding prediction only)
        >>> model = timme.create_encoder('vit_base_patch16_224', pretrained=False)
        >>> task = NEPATask(model, shift=True)
        >>>
        >>> # With pixel prediction
        >>> task = NEPATask(
        ...     model,
        ...     shift=True,
        ...     pixel_decoder_cfg=dict(
        ...         hidden_dim=1024,
        ...         depth=2,
        ...         mlp_layer='swiglu',
        ...         drop_path_rate=0.1,
        ...     ),
        ...     pixel_loss_weight=0.5,
        ... )
        >>>
        >>> # Forward pass
        >>> x = torch.randn(32, 3, 224, 224)
        >>> output = task(x)
        >>> loss = output['loss']  # Combined loss
        >>> embed_loss = output['embed_loss']
        >>> pixel_loss = output.get('pixel_loss')  # None if decoder disabled
    """

    def __init__(
            self,
            model: nn.Module,
            shift: bool = True,
            pixel_decoder_cfg: Optional[Dict] = None,
            pixel_loss_weight: float = 1.0,
            pixel_loss_type: str = 'mse',
            norm_pixel_target: bool = True,
            prefix_bidirectional: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        encoder = _resolve_encoder(model)
        if not _is_supported_encoder(encoder):
            raise TypeError(
                f"NEPATask requires a timme ViT/EVA-style encoder, got {model.__class__.__name__}."
            )

        self.shift = shift
        self.pixel_loss_weight = pixel_loss_weight
        self.pixel_loss_type = pixel_loss_type
        self.norm_pixel_target = norm_pixel_target
        self.prefix_bidirectional = prefix_bidirectional

        # Store num_prefix_tokens for excluding CLS/register tokens from prediction loss
        self._num_prefix_tokens = getattr(encoder, 'num_prefix_tokens', 1)

        # Build pixel decoder if config provided
        pixel_decoder = None
        if pixel_decoder_cfg is not None:
            pixel_decoder = PixelDecoder(
                embed_dim=encoder.embed_dim,
                patch_size=encoder.patch_embed.patch_size[0],
                in_chans=getattr(encoder, 'in_chans', 3),
                **pixel_decoder_cfg,
            )
            self._patch_size = encoder.patch_embed.patch_size[0]

        # Use causal masking when shift=True (next-token prediction)
        # to prevent attending to future tokens
        self.trainable_module = NEPATrainableModule(
            model=model,
            pixel_decoder=pixel_decoder,
            causal=shift,
            prefix_bidirectional=prefix_bidirectional,
        )

        if self.verbose:
            pixel_info = f", pixel_decoder={pixel_decoder_cfg is not None}"
            if pixel_decoder_cfg:
                pixel_info += f", pixel_weight={pixel_loss_weight}"
            _logger.info(
                f"NEPATask: shift={shift}, causal={shift}, prefix_bidirectional={prefix_bidirectional}, "
                f"num_prefix_tokens={self._num_prefix_tokens}{pixel_info}, model={encoder.__class__.__name__}"
            )

    @property
    def pixel_decoder(self) -> Optional[PixelDecoder]:
        """Access the pixel decoder (if present)."""
        return unwrap_model(self.trainable_module).pixel_decoder

    def get_eval_task(self, use_ema: bool = True) -> SSLEvalTask:
        """Get evaluation task for feature extraction.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            SSLEvalTask configured for NEPA (last token pooling for causal models)
        """
        if use_ema and self.has_ema():
            module = self.get_eval_model(ema=True)
        else:
            module = getattr(self, 'eval_feature_model', None)
            if module is None:
                module = self.get_trainable_module(use_ema=False)
        return SSLEvalTask(module, pool='last')

    def compile(
            self,
            backend: str = 'inductor',
            mode: Optional[str] = None,
            **compile_kwargs,
    ) -> nn.Module:
        """Compile the DDP/train wrapper and keep a compiled eval encoder path."""
        trainable = self.get_trainable_module()
        self._maybe_disable_ddp_dynamo_optimizer(trainable)
        raw_trainable = unwrap_model(trainable)
        self.eval_feature_model = torch.compile(
            FeatureEvalModel(raw_trainable.model),
            backend=backend,
            mode=mode,
            **compile_kwargs,
        )
        self.trainable_module = torch.compile(
            trainable,
            backend=backend,
            mode=mode,
            **compile_kwargs,
        )
        return self.trainable_module

    def forward(
            self,
            input: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with NEPA prediction loss.

        Args:
            input: Input images [B, C, H, W]
            target: Ignored (self-supervised task)

        Returns:
            Dictionary containing:
                - 'loss': Combined loss (for optimization)
                - 'embed_loss': Embedding prediction loss
                - 'pixel_loss': Pixel prediction loss (if decoder enabled)
                - 'output': Output embeddings [B, N, D]
                - 'input_embeddings': Input embeddings [B, N, D]
        """
        result = self.trainable_module(input)

        output_embeddings = result['output_embeddings']
        input_embeddings = result['input_embeddings']

        # Embedding prediction loss (excluding prefix tokens which have no spatial relationship)
        embed_loss = prediction_loss(
            input_embeddings,
            output_embeddings,
            shift=self.shift,
            num_prefix_tokens=self._num_prefix_tokens,
        )

        # Pixel prediction loss (if decoder enabled)
        if 'pixel_pred' in result:
            pixel_loss = pixel_prediction_loss(
                result['pixel_pred'],
                input,
                patch_size=self._patch_size,
                num_prefix_tokens=self._num_prefix_tokens,
                norm_target=self.norm_pixel_target,
                loss_type=self.pixel_loss_type,
            )
            total_loss = embed_loss + self.pixel_loss_weight * pixel_loss
        else:
            pixel_loss = None
            total_loss = embed_loss

        output = {
            'loss': total_loss,
            'embed_loss': embed_loss,
            'output': output_embeddings,
            'input_embeddings': input_embeddings,
        }
        if pixel_loss is not None:
            output['pixel_loss'] = pixel_loss

        return output
