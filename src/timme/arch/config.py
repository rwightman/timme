"""Base config infrastructure for timme.

This module owns the shared types every family config builds on:
  - ConfigMixin: serialization + overlay + deploy-field semantics
  - HeadCfg:     lightweight head-side config
  - ModelSpec:   encoder_cfg + head_cfg envelope for full model serialization

Per-family architecture configs (VisionTransformerCfg, ResNetCfg, etc.) and
their per-variant CFGS dicts live in their respective modeling files —
that keeps each family self-contained and easier to scale out.

Field categories (by convention, formalized via _deploy_fields):
  - Architecture fields = define the encoder structure (fixed per variant)
    patch_size, embed_dim, depth, num_heads, mlp_ratio, block types, ...
  - Deploy fields = architecturally meaningful but overridable at construction
    img_size (only for models where it matters, e.g. ViT pos_embed),
    in_chans (always — determines stem input channels).
    Stored in config for self-contained serialization. overlay() overrides them.
    For fully convolutional models (ResNet, MobileNetV3), img_size is NOT in
    the config — it's just training metadata in pretrained_cfg.
  - Runtime fields = never in the config
    out_indices, device, dtype
  - Head fields = not part of encoder config at all
    num_classes, global_pool, drop_rate
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, ClassVar, Dict, FrozenSet, Optional, Type, TypeVar

T = TypeVar('T')


# ---------------------------------------------------------------------------
# Config base mixin
# ---------------------------------------------------------------------------


@dataclass
class ConfigMixin:
    """Mixin providing serialization for config dataclasses.

    All config dataclasses should inherit from this to get consistent
    to_dict/from_dict/to_json/from_json support.

    Subclasses may define a ``_deploy_fields`` ClassVar to declare which
    fields are deployment-time defaults (e.g. img_size, in_chans). These
    fields are always included in serialization (the config is self-contained)
    but can be overridden at construction via overlay() or factory kwargs.
    """

    # Subclasses override to declare deploy-time fields.
    # Empty by default — all fields are treated as architecture fields.
    _deploy_fields: ClassVar[FrozenSet[str]] = frozenset()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict. Tuples preserved as lists (JSON compat)."""
        return asdict(self)

    def to_arch_dict(self) -> Dict[str, Any]:
        """Convert to dict with only architecture fields (excluding deploy fields)."""
        return {k: v for k, v in self.to_dict().items() if k not in self._deploy_fields}

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """Reconstruct from a dict. Handles list->tuple coercion.

        For nested configs, subclasses should override this to reconstruct
        nested dataclass fields.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls: Type[T], s: str) -> T:
        return cls.from_dict(json.loads(s))

    def overlay(self: T, **kwargs) -> T:
        """Return a copy with specified fields replaced.

        Only updates fields that exist in the config. Unknown kwargs are
        silently ignored (useful for passing mixed encoder/head/runtime kwargs).
        """
        known = {f.name for f in fields(self)}
        valid = {k: v for k, v in kwargs.items() if k in known}
        return replace(self, **valid) if valid else self


# ---------------------------------------------------------------------------
# Head config (lightweight, separate from encoder arch)
# ---------------------------------------------------------------------------


@dataclass
class HeadCfg(ConfigMixin):
    """Configuration for the classification head.

    Paired with an encoder config to create a full model spec.
    Not every field applies to every head type — unused fields are ignored.
    """

    head_type: str = 'auto'  # 'auto', 'spatial_linear', 'spatial_norm_mlp', 'spatial_efficient', 'spatial_mlp',
    #                          'spatial_attention', 'token_linear', 'token_norm_mlp', 'token_attention'
    num_classes: int = 1000
    pool_type: str = ''  # '' means use encoder's default_pool_type
    drop_rate: float = 0.0

    # For NormMlp / pipeline heads
    norm_layer: Optional[str] = None
    hidden_size: Optional[int] = None
    act_layer: Optional[str] = None

    # For attention pool heads
    fc_norm: Optional[bool] = None  # None = auto-detect based on pool_type


# ---------------------------------------------------------------------------
# Full model spec (encoder config + head config + pretrained reference)
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec(ConfigMixin):
    """Complete model specification — enough to reconstruct any model.

    This is what gets serialized to represent "what model is this?"
    Deploy params (img_size, in_chans) live inside encoder_cfg now —
    the encoder config is self-contained.
    """

    family: str = ''  # 'vision_transformer', 'resnet', 'mobilenetv3', ...
    encoder_cfg: Dict[str, Any] = field(default_factory=dict)  # serialized encoder config (includes deploy fields)
    head_cfg: Dict[str, Any] = field(default_factory=dict)  # serialized head config
