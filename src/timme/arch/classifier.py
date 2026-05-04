"""ImageClassifier and DistilledImageClassifier — composition of encoder + head.

These classes are duck-typed against the head: they don't import ImageHead
from heads/, they only require the head expose ``accepted_fmts`` and a
``forward(x, pre_logits=False)`` callable. That keeps arch/ free of any
upward / sideways dependency on heads/.

forward_features() = encoder.forward()  (encoder IS features)
forward_head()     = head.forward()
forward()          = features -> head
"""

from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn as nn

from .encoder import ImageEncoder


class ImageClassifier(nn.Module):
    """Standard image classification model: encoder + head."""

    def __init__(self, encoder: ImageEncoder, head: nn.Module):
        super().__init__()
        # Validate format compatibility via traits + duck-typed accepted_fmts.
        accepted = getattr(head, 'accepted_fmts', ())
        assert encoder.traits.output_fmt in accepted, (
            f"Encoder output format '{encoder.traits.output_fmt}' not accepted by "
            f"{type(head).__name__} (accepts {accepted})"
        )
        self.encoder = encoder
        self.head = head

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    @property
    def num_features(self) -> int:
        """Encoder output dimension."""
        return self.encoder.output_dim

    @property
    def in_chans(self) -> int:
        """Input channel count expected by the encoder."""
        return getattr(self.encoder, 'in_chans', 3)

    @property
    def head_hidden_size(self) -> int:
        """Dimension of head's pre-logits representation."""
        return self.head.pre_logits_dim

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.head.reset(num_classes, pool_type=global_pool)

    def get_classifier(self) -> nn.Module:
        return self.head.get_classifier()

    @torch.jit.ignore
    def get_patch_size(self):
        if hasattr(self.encoder, 'get_patch_size'):
            return self.encoder.get_patch_size()
        return None

    # ------------------------------------------------------------------
    # The split API: maps directly to sub-modules
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encoder features. Delegates to encoder.forward()."""
        return self.encoder(x, **kwargs)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Head output. Delegates to head.forward()."""
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.encoder(x, **kwargs)
        return self.head(x)

    # ------------------------------------------------------------------
    # Delegated intermediate features
    # ------------------------------------------------------------------

    def forward_intermediates(self, x: torch.Tensor, **kwargs):
        """Pass-through to encoder.forward_intermediates."""
        return self.encoder.forward_intermediates(x, **kwargs)

    def prune_intermediate_layers(self, indices=1, prune_norm=False):
        return self.encoder.prune_intermediate_layers(indices, prune_norm=prune_norm)

    # ------------------------------------------------------------------
    # Delegated utilities
    # ------------------------------------------------------------------

    @property
    def feature_info(self):
        return self.encoder.feature_info

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        # Prefix encoder matcher keys so they resolve under self.encoder.*
        raw = self.encoder.group_matcher(coarse)
        return {k: _prefix_pattern('encoder.', v) for k, v in raw.items()}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        nwd = set()
        if hasattr(self.encoder, 'no_weight_decay'):
            nwd.update('encoder.' + k for k in self.encoder.no_weight_decay())
        return nwd


class DistilledImageClassifier(ImageClassifier):
    """Classification model with a distillation head.

    Two heads (head + head_dist) each consume the full encoder output. Each
    head is responsible for its own token selection / pooling — for DeiT
    that's index-0 (cls) vs index-1 (dist); for LeViT both heads avg-pool
    the same spatial tokens. The distilled classifier just composes them.
    """

    def __init__(
            self,
            encoder: ImageEncoder,
            head: nn.Module,
            head_dist: nn.Module,
            distilled_training: bool = True,
    ):
        super().__init__(encoder, head)
        accepted = getattr(head_dist, 'accepted_fmts', ())
        assert encoder.traits.output_fmt in accepted, (
            f"Encoder output format '{encoder.traits.output_fmt}' not accepted by "
            f"{type(head_dist).__name__} (accepts {accepted})"
        )
        self.head_dist = head_dist
        self.distilled_training = distilled_training

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.head.reset(num_classes, pool_type=global_pool)
        self.head_dist.reset(num_classes, pool_type=global_pool)

    def set_distilled_training(self, enable: bool = True) -> None:
        self.distilled_training = enable

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            x_cls = self.head(x, pre_logits=True)
            x_dist = self.head_dist(x, pre_logits=True)
            return (x_cls + x_dist) / 2
        x_cls = self.head(x)
        x_dist = self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            return x_cls, x_dist
        return (x_cls + x_dist) / 2

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.encoder(x, **kwargs)
        return self.forward_head(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefix_pattern(prefix: str, pattern):
    """Add a prefix to group_matcher regex patterns."""
    if isinstance(pattern, str):
        return prefix + pattern
    if isinstance(pattern, (list, tuple)):
        return [(prefix + p if isinstance(p, str) else (prefix + p[0], p[1])) for p in pattern]
    return pattern
