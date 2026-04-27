"""ArchTraits — declarative per-architecture properties.

Set at construction by the encoder, used by heads (and ImageClassifier)
for compatibility checks instead of runtime ``isinstance`` introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ArchTraits:
    """Declarative properties of an encoder architecture.

    Defaults are set per model family. Specific arch configs and pretrained
    configs can override individual fields. Heads use these to validate
    compatibility at construction time, not at runtime.
    """

    output_fmt: Literal['NCHW', 'NLC'] = 'NCHW'
    output_dim: int = 0
    num_prefix_tokens: int = 0
    has_cls_token: bool = False
    pool_include_prefix: bool = False
    default_pool_type: str = 'avg'
    supports_variable_input: bool = False
