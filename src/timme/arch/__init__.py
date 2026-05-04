"""Architecture primitives — package vocabulary, no upward deps.

Both ``timme.models`` and ``timme.heads`` reach into this subpackage for
the abstractions every family builds on.
"""

from .traits import ArchTraits
from .encoder import ImageEncoder
from .head import ImageHead
from .classifier import ImageClassifier, DistilledImageClassifier
from .weights import WeightLayout, clean_state_dict, remap_state_dict
from .config import ConfigMixin, HeadCfg, ModelSpec

__all__ = [
    'ArchTraits',
    'ImageEncoder',
    'ImageHead',
    'ImageClassifier',
    'DistilledImageClassifier',
    'WeightLayout',
    'clean_state_dict',
    'remap_state_dict',
    'ConfigMixin',
    'HeadCfg',
    'ModelSpec',
]
