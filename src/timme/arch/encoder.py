"""ImageEncoder ABC.

The encoder's forward() IS the features — there is no separate forward_features().
Subclasses implement _forward_features() and forward_intermediates(); the base
class dispatches between them based on whether out_indices was set at construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from timm.models._features import FeatureInfo

from .traits import ArchTraits


# ---------------------------------------------------------------------------
# Encoder base
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """Base class for all image encoders.

    The encoder's forward() IS the features — there is no separate
    forward_features(). The encoder is a self-contained module whose
    forward() returns its representation.

    Output contract (fixed per instance, safe for tracing):
      - No out_indices:    forward(x) -> Tensor       (final encoder features)
      - With out_indices:  forward(x) -> List[Tensor] (intermediate features)

    Runtime-flexible path (always available, not the default forward):
      - forward_intermediates(x, indices=..., ...) for arbitrary feature taps
    """

    traits: ArchTraits
    feature_info: FeatureInfo

    def __init__(self, out_indices: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self._out_indices = out_indices

    @property
    def out_indices(self) -> Optional[Tuple[int, ...]]:
        return self._out_indices

    @property
    def output_dim(self) -> int:
        return self.traits.output_dim

    @property
    def output_fmt(self) -> str:
        return self.traits.output_fmt

    # ------------------------------------------------------------------
    # Primary forward — subclasses MUST override this
    # ------------------------------------------------------------------

    def forward(
            self,
            x: torch.Tensor,
            **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Encoder forward. Output type is fixed per instance.

        Without out_indices: returns final encoder features (Tensor).
        With out_indices: returns intermediate features (List[Tensor]).

        Subclasses implement _forward_features() and forward_intermediates()
        (or override forward() directly if the two paths share enough logic).
        """
        if self._out_indices is not None:
            return self.forward_intermediates(
                x,
                indices=list(self._out_indices),
                intermediates_only=True,
                **kwargs,
            )
        return self._forward_features(x, **kwargs)

    # ------------------------------------------------------------------
    # Internal implementations — subclasses override these
    # ------------------------------------------------------------------

    def _forward_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Final-features-only fast path. Subclasses must implement."""
        raise NotImplementedError

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            **kwargs,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Runtime-flexible intermediate feature gathering.

        Always available regardless of how the encoder was constructed.
        This is the research/inspection API.

        Args:
            indices: Which feature taps to return.
                None -> all, int -> last N, list -> specific indices.
                Negative indices count from end.
            norm: Apply encoder's final norm to intermediates.
            stop_early: Stop forward pass after last requested index.
            output_fmt: 'NCHW' or 'NLC' for intermediate reshape.
            intermediates_only: If True, return List[Tensor]. If False,
                return (final_features, List[Tensor]).
        """
        raise NotImplementedError

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
    ) -> List[int]:
        """Prune layers not needed for specified intermediates."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Standard encoder utilities (subclasses override as needed)
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return {}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        pass

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return set()
