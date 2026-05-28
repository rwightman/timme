"""timme — timm, evolved.

Encoder/head architecture split for image models, layered on top of timm.

Public API:
  create_encoder(name, pretrained=False, out_indices=None, **kwargs) -> ImageEncoder
  create_model(name, pretrained=False, num_classes=None, ...)        -> ImageClassifier

Layout:
  arch/   — ImageEncoder, ImageHead, ImageClassifier, WeightLayout, ConfigMixin
  models/ — per-family encoders + factory + registry
  heads/  — 9 canonical classification heads
  layers/ — shared layer primitives (currently empty; layers are reused from timm)
"""

# Architecture vocabulary
from .arch import (
    ArchTraits,
    ImageEncoder,
    ImageHead,
    ImageClassifier,
    DistilledImageClassifier,
    WeightLayout,
    remap_state_dict,
    ConfigMixin,
    HeadCfg,
    ModelSpec,
)

# Heads
from .heads import (
    SpatialLinearHead,
    SpatialNormMlpHead,
    SpatialEfficientHead,
    SpatialMlpHead,
    SpatialAttentionHead,
    TokenLinearHead,
    TokenNormMlpHead,
    TokenSelectHead,
    TokenAttentionHead,
)

# Models — importing triggers registration of every family
from .models import (
    create_encoder,
    create_model,
    register_family,
    list_models,
    is_registered,
    ResNetCfg,
    RESNET_CFGS,
    VisionTransformerCfg,
    VIT_CFGS,
    ConvNeXtCfg,
    CONVNEXT_CFGS,
    MobileNetV3Cfg,
    MOBILENETV3_CFGS,
    ByoBlockCfg,
    ByoModelCfg,
    LevitCfg,
    LEVIT_CFGS,
    NaFlexVitCfg,
    NAFLEXVIT_CFGS,
    EvaCfg,
    EVA_CFGS,
)

__version__ = '0.0.4'
