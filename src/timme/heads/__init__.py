"""Canonical classification heads.

9 concrete heads cover all ~89 model families:

Spatial (NCHW encoder output — CNNs):
  SpatialLinearHead       pool -> drop -> fc
  SpatialNormMlpHead      pool -> norm -> flatten -> [mlp] -> drop -> fc
  SpatialEfficientHead    pool -> conv1x1 -> [norm] -> act -> flatten -> drop -> fc
  SpatialMlpHead          pool(flatten) -> fc1 -> act -> norm -> drop -> fc2
  SpatialAttentionHead    attn_pool -> drop -> fc

Token (NLC encoder output — transformers):
  TokenLinearHead         token/avg/max pool -> [norm] -> drop -> fc
  TokenNormMlpHead        pool -> norm -> [mlp] -> drop -> fc
  TokenSelectHead         select index -> [norm] -> drop -> fc  (for distilled models)
  TokenAttentionHead      attn_pool -> [norm] -> drop -> fc

pre_logits=True always means "the representation immediately before the
final classifier linear" — useful for embeddings, retrieval, kNN probes.
"""

from .spatial import (
    SpatialLinearHead,
    SpatialNormMlpHead,
    SpatialEfficientHead,
    SpatialMlpHead,
    SpatialAttentionHead,
)
from .token import (
    TokenLinearHead,
    TokenNormMlpHead,
    TokenSelectHead,
    TokenAttentionHead,
)

__all__ = [
    'SpatialLinearHead',
    'SpatialNormMlpHead',
    'SpatialEfficientHead',
    'SpatialMlpHead',
    'SpatialAttentionHead',
    'TokenLinearHead',
    'TokenNormMlpHead',
    'TokenSelectHead',
    'TokenAttentionHead',
]
