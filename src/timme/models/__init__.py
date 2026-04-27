"""Model families.

Each family module follows the same shape:
  cfg dataclass (top of file) -> encoder class + weight layout
    -> variant CFGS dict -> entry-point builders -> register_family(...)

Importing this package triggers ``register_family(...)`` for every family,
populating the factory's registry with all variants. After this import, any
registered name can be requested via ``timme.create_model(name)`` or
``timme.create_encoder(name)``.
"""

from ._factory import (
    create_encoder,
    create_model,
    register_family,
    list_models,
    is_registered,
)

# Trigger registration for every family at import time.
# Each module's bottom calls register_family(...).
from .resnet import ResNetCfg, RESNET_CFGS
from .vision_transformer import VisionTransformerCfg, VIT_CFGS
from .convnext import ConvNeXtCfg, CONVNEXT_CFGS
from .mobilenetv3 import MobileNetV3Cfg, MOBILENETV3_CFGS
from .byobnet import ByoBlockCfg, ByoModelCfg
from .levit import LevitCfg, LEVIT_CFGS
from .naflexvit import NaFlexVitCfg, NAFLEXVIT_CFGS
from .eva import EvaCfg, EVA_CFGS
from . import deit  # noqa: F401  (registers; no separate cfg dataclass — reuses VisionTransformerCfg)

__all__ = [
    'create_encoder',
    'create_model',
    'register_family',
    'list_models',
    'is_registered',
    'ResNetCfg',
    'RESNET_CFGS',
    'VisionTransformerCfg',
    'VIT_CFGS',
    'ConvNeXtCfg',
    'CONVNEXT_CFGS',
    'MobileNetV3Cfg',
    'MOBILENETV3_CFGS',
    'ByoBlockCfg',
    'ByoModelCfg',
    'LevitCfg',
    'LEVIT_CFGS',
    'NaFlexVitCfg',
    'NAFLEXVIT_CFGS',
    'EvaCfg',
    'EVA_CFGS',
]
