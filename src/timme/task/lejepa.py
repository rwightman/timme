"""LeJEPA (Lean Joint-Embedding Predictive Architecture) training task.

LeJEPA is a self-supervised learning method that uses:
- SIGReg (Sketched Isotropic Gaussian Regularization) to constrain embeddings
- Invariance loss across multiple augmented views
- Single hyperparameter (lambda) for loss weighting

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning
    Without the Heuristics", arXiv:2511.08544, 2025.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from timm.utils import unwrap_model

from .task import TrainingTask
from .eval_task import FeatureEvalModel, SSLEvalTask

_logger = logging.getLogger(__name__)


def _forward_features(model: nn.Module, input: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'forward_features'):
        return model.forward_features(input)
    return model(input)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization loss.

    Statistical test that constrains embeddings to follow an isotropic Gaussian
    distribution using random slicing and the Epps-Pulley characteristic function test.

    The loss measures deviation from Gaussianity by comparing the empirical
    characteristic function to the theoretical Gaussian characteristic function
    along random 1D projections.

    Args:
        num_knots: Number of quadrature points for numerical integration (default: 17)
        num_slices: Number of random 1D projections for slicing (default: 256)
        t_max: Maximum integration bound (default: 3.0)

    Example:
        >>> sigreg = SIGReg(num_slices=256)
        >>> projections = torch.randn(4, 32, 128)  # [V, B, proj_dim]
        >>> loss = sigreg(projections)
    """

    def __init__(
            self,
            num_knots: int = 17,
            num_slices: int = 256,
            t_max: float = 3.0,
    ):
        super().__init__()
        self.num_slices = num_slices

        # Quadrature weights for trapezoidal integration on [0, t_max]
        # We use symmetry of ECF to integrate on [0, t_max] and double
        t = torch.linspace(0, t_max, num_knots, dtype=torch.float32)
        dt = t_max / (num_knots - 1)
        weights = torch.full((num_knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Trapezoidal rule endpoints

        # Gaussian characteristic function: exp(-t^2 / 2)
        phi_gaussian = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi_gaussian", phi_gaussian)
        self.register_buffer("weights", weights * phi_gaussian)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings [V, B, proj_dim] or [B, proj_dim]
                  where V is number of views, B is batch size

        Returns:
            Scalar loss value
        """
        # Handle both [V, B, D] and [B, D] inputs
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        # Random projection directions (normalized)
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project onto random directions: [V, B, num_slices]
        x_proj = proj @ A

        # Compute empirical characteristic function at quadrature points
        # x_t: [V, B, num_slices, num_knots]
        x_t = x_proj.unsqueeze(-1) * self.t

        # ECF components: E[cos(tx)] and E[sin(tx)]
        # Average over batch dimension (dim=-3 in [V, B, num_slices, num_knots])
        cos_mean = x_t.cos().mean(dim=-3)  # [V, num_slices, num_knots]
        sin_mean = x_t.sin().mean(dim=-3)  # [V, num_slices, num_knots]

        # Squared error from Gaussian ECF (which has sin component = 0)
        err = (cos_mean - self.phi_gaussian).square() + sin_mean.square()

        # Weighted integration and scale by batch size
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class LeJEPATrainableModule(nn.Module):
    """Trainable module for LeJEPA containing model and projector.

    Wraps the encoder model and adds a projector MLP. All trainable forward
    operations happen inside forward() for proper DDP/FSDP wrapping.

    Args:
        model: Backbone encoder model. Prefer a bare timme ImageEncoder.
        proj_dim: Output dimension of projector (default: 128)
        proj_hidden: Hidden dimension of projector MLP (default: 2048)
        proj_layers: Number of hidden layers in projector (default: 2)
    """

    def __init__(
            self,
            model: nn.Module,
            proj_dim: int = 128,
            proj_hidden: int = 2048,
            proj_layers: int = 2,
    ):
        super().__init__()
        self.model = model  # Core encoder model

        # Get encoder output dimension
        num_features = getattr(model, 'num_features', None)
        if num_features is None:
            num_features = getattr(model, 'output_dim', None)
        if num_features is None and hasattr(model, 'encoder'):
            num_features = getattr(model.encoder, 'output_dim', None)
        if num_features is None:
            raise ValueError(
                f"Model {model.__class__.__name__} must have 'num_features' or 'output_dim' attribute. "
                "timme encoders expose output_dim; timm classifiers usually expose num_features."
            )

        # Build projector MLP: Linear -> BN -> GELU -> ... -> Linear
        layers = []
        in_dim = num_features
        for i in range(proj_layers):
            layers.extend([
                nn.Linear(in_dim, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.GELU(),
            ])
            in_dim = proj_hidden
        layers.append(nn.Linear(proj_hidden, proj_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and projector.

        Args:
            x: Multi-view input images [B, V, C, H, W] where V is number of views

        Returns:
            Tuple of:
                - embeddings: Encoder outputs [B*V, num_features] (for optional probe)
                - projections: Projector outputs [V, B, proj_dim] (for loss computation)
        """
        if x.dim() != 5:
            raise ValueError(
                f"LeJEPA expects multi-view input [B, V, C, H, W] but got shape {x.shape}. "
                f"Make sure you're using create_multiview_train_loader() or a multi-view dataset."
            )

        B, V = x.shape[:2]

        # Flatten views into batch dimension
        x_flat = x.flatten(0, 1)  # [B*V, C, H, W]

        # Encode. timme encoders return features directly from forward().
        embeddings = _forward_features(self.model, x_flat)  # [B*V, ...features]

        # Pool if needed (ViT returns [B, N, D], CNNs return [B, D] or [B, D, H, W])
        if embeddings.dim() == 3:
            # ViT-style: use the classifier's encoder traits to choose pooling.
            encoder = getattr(self.model, 'encoder', self.model)
            traits = getattr(encoder, 'traits', None)
            pool_type = getattr(traits, 'default_pool_type', None)
            num_prefix = getattr(traits, 'num_prefix_tokens', 0)
            include_prefix = getattr(traits, 'pool_include_prefix', False)
            if pool_type == 'avg':
                if num_prefix and not include_prefix:
                    embeddings = embeddings[:, num_prefix:]
                embeddings = embeddings.mean(dim=1)  # [B*V, D]
            else:
                embeddings = embeddings[:, 0]  # CLS token [B*V, D]
        elif embeddings.dim() == 4:
            # CNN-style: global average pool
            embeddings = embeddings.mean(dim=(2, 3))  # [B*V, D]

        # Project
        projections = self.projector(embeddings)  # [B*V, proj_dim]

        # Reshape projections for loss: [V, B, proj_dim]
        projections = projections.reshape(B, V, -1).transpose(0, 1)

        return embeddings, projections


class LeJEPATask(TrainingTask):
    """LeJEPA self-supervised training task.

    Combines SIGReg loss (ensures Gaussian embedding distribution) with
    invariance loss (views of same image should have similar embeddings).

    Loss = lambda * SIGReg + (1 - lambda) * Invariance

    Args:
        model: Encoder model. Prefer a bare timme ImageEncoder with output_dim.
        proj_dim: Projector output dimension (default: 128)
        proj_hidden: Projector hidden dimension (default: 2048)
        proj_layers: Number of projector hidden layers (default: 2)
        lamb: Loss weighting hyperparameter (default: 0.02)
            - Higher lambda = more weight on SIGReg (Gaussianity)
            - Lower lambda = more weight on invariance
        num_slices: Number of random projections for SIGReg (default: 256)
        num_knots: Quadrature points for SIGReg integration (default: 17)
        device: Device for task components
        dtype: Data type for task components
        verbose: Whether to log task configuration

    Example:
        >>> # With a timme encoder
        >>> model = timme.create_encoder('vit_small_patch16_224', pretrained=False)
        >>> task = LeJEPATask(model, proj_dim=128, lamb=0.02)
        >>>
        >>> # Forward pass with multi-view input
        >>> x = torch.randn(32, 4, 3, 224, 224)  # [B, V, C, H, W]
        >>> output = task(x)
        >>> loss = output['loss']
    """

    def __init__(
            self,
            model: nn.Module,
            proj_dim: int = 128,
            proj_hidden: int = 2048,
            proj_layers: int = 2,
            lamb: float = 0.02,
            num_slices: int = 256,
            num_knots: int = 17,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
    ):
        super().__init__(device=device, dtype=dtype, verbose=verbose)

        self.trainable_module = LeJEPATrainableModule(
            model=model,
            proj_dim=proj_dim,
            proj_hidden=proj_hidden,
            proj_layers=proj_layers,
        )
        self.sigreg = SIGReg(num_knots=num_knots, num_slices=num_slices)
        self.lamb = lamb

        # Move to device/dtype (encoder already on device, but projector and sigreg need moving)
        if device is not None or dtype is not None:
            self.trainable_module.projector.to(device=device, dtype=dtype)
            self.sigreg.to(device=device)

        if self.verbose:
            _logger.info(
                f"LeJEPATask: proj_dim={proj_dim}, proj_hidden={proj_hidden}, "
                f"proj_layers={proj_layers}, lambda={lamb}, num_slices={num_slices}"
            )

    def get_eval_task(self, use_ema: bool = True) -> SSLEvalTask:
        """Get evaluation task for feature extraction.

        Args:
            use_ema: If True and EMA exists, use EMA weights for evaluation

        Returns:
            SSLEvalTask configured for LeJEPA (avg pooling)
        """
        if use_ema and self.has_ema():
            module = self.get_eval_model(ema=True)
        else:
            module = getattr(self, 'eval_feature_model', None)
            if module is None:
                module = self.get_trainable_module(use_ema=False)
        return SSLEvalTask(module, pool='avg')

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
        """Forward pass with LeJEPA loss computation.

        Args:
            input: Multi-view input images [B, V, C, H, W]
            target: Ignored (self-supervised task)

        Returns:
            Dictionary containing:
                - 'loss': Combined LeJEPA loss (for optimization)
                - 'output': Encoder embeddings [B*V, num_features] (for metrics/probing)
                - 'sigreg_loss': SIGReg component (for logging)
                - 'inv_loss': Invariance component (for logging)
        """
        embeddings, projections = self.trainable_module(input)

        # SIGReg loss - constrain to isotropic Gaussian
        sigreg_loss = self.sigreg(projections)

        # Invariance loss - views should have similar projections
        # proj_mean: [1, B, proj_dim], projections: [V, B, proj_dim]
        proj_mean = projections.mean(dim=0, keepdim=True)
        inv_loss = (proj_mean - projections).square().mean()

        # Combined loss with single hyperparameter
        total_loss = sigreg_loss * self.lamb + inv_loss * (1.0 - self.lamb)

        return {
            'loss': total_loss,
            'output': embeddings,
            'sigreg_loss': sigreg_loss,
            'inv_loss': inv_loss,
        }
