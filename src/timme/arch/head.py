"""ImageHead ABC.

A head owns everything in the old forward_head() path.
Its forward() IS the head computation.
pre_logits=True always means "right before the final classifier linear."

This base class is also a duck-typed contract: any nn.Module that exposes
``accepted_fmts``, ``num_classes``, ``in_features``, ``pre_logits_dim``,
``reset(num_classes, pool_type)``, ``get_classifier()`` and a
``forward(x, pre_logits=False)`` will work as a head, regardless of whether
it inherits from ImageHead. ImageClassifier validates only via
``head.accepted_fmts``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ImageHead(nn.Module):
    """Base class for classification / task heads.

    A head owns everything in the old forward_head() path.
    ``forward(x, pre_logits=False)`` IS the head computation;
    ``pre_logits=True`` always means "right before the final classifier linear."
    """

    # Head declares what encoder output formats it can consume.
    # ImageClassifier validates this at construction time.
    accepted_fmts: Tuple[str, ...] = ()

    def __init__(self):
        super().__init__()

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @property
    def in_features(self) -> int:
        """Input feature dimension (encoder output_dim)."""
        raise NotImplementedError

    @property
    def pre_logits_dim(self) -> int:
        """Dimension of representation right before the classifier linear."""
        raise NotImplementedError

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        """Rebuild head for new num_classes and/or pool type."""
        raise NotImplementedError

    def get_classifier(self) -> nn.Module:
        """Return the final classifier linear/conv module."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        raise NotImplementedError
