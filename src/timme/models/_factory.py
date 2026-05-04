"""Model factory for timme.

Public API:
  create_encoder(name, pretrained=False, out_indices=None, **kwargs) -> ImageEncoder
  create_model(name, pretrained=False, num_classes=1000, ...)         -> ImageClassifier

Internally this piggybacks on timm1 infrastructure:
  - timm.models._registry.get_pretrained_cfg(name)  — metadata (URL/hub/mean/std/input_size)
  - timm.models._builder.load_pretrained(...)       — URL/hub download, in_chans adapt,
                                                      classifier strip, load_state_dict

Families register themselves at import time via `register_family(...)`. Each entry
exposes a `build_classifier` and `build_encoder`, a `WeightLayout`, and an optional
`checkpoint_filter_fn` carried over from timm1 (pos_embed resize, key renames, etc.).

Weight loading chain:
  1. Fetch state_dict from URL / HF hub / file (timm handles this).
  2. Unwrap container keys ('state_dict', 'model', ...).
  3. Apply the family's timm1 checkpoint_filter_fn (still in timm1 namespace).
  4. Apply timme remap_state_dict to split into 'encoder.*' / 'head.*' keys.
  5. timm.load_pretrained adapts first_conv for in_chans and strips classifier
     for num_classes mismatch (we rewrite those keys to the timme namespace).
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.models._builder import load_pretrained
from timm.models._registry import get_pretrained_cfg, split_model_name_tag

from ..arch import ImageEncoder, ImageClassifier, WeightLayout, clean_state_dict, remap_state_dict


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class _FamilyEntry:
    build_classifier: Callable[..., ImageClassifier]
    build_encoder: Callable[..., ImageEncoder]
    weight_layout: WeightLayout
    checkpoint_filter_fn: Optional[Callable] = None


_FAMILIES: Dict[str, _FamilyEntry] = {}
_VARIANTS: Dict[str, Tuple[str, Any]] = {}  # variant arch name -> (family_name, cfg)


def register_family(
        family_name: str,
        variants: Dict[str, Any],
        build_classifier: Callable[..., ImageClassifier],
        build_encoder: Callable[..., ImageEncoder],
        weight_layout: WeightLayout,
        checkpoint_filter_fn: Optional[Callable] = None,
) -> None:
    """Register a model family and all its variants.

    Args:
        family_name: short family identifier (e.g. 'resnet').
        variants: dict of {arch_name: cfg} — every arch name a user can request.
        build_classifier: callable(cfg, **kwargs) -> ImageClassifier.
        build_encoder: callable(cfg, out_indices, **kwargs) -> ImageEncoder.
        weight_layout: WeightLayout describing timm1 key split.
        checkpoint_filter_fn: optional timm1 per-family filter fn.
            Signature either (state_dict, model) or (state_dict,).
    """
    entry = _FamilyEntry(build_classifier, build_encoder, weight_layout, checkpoint_filter_fn)
    _FAMILIES[family_name] = entry
    for arch_name, cfg in variants.items():
        _VARIANTS[arch_name] = (family_name, cfg)


def list_models(
        filter: Union[str, List[str], Tuple[str, ...]] = '',
        pretrained: bool = False,
        exclude_filters: Union[str, List[str], Tuple[str, ...]] = '',
) -> List[str]:
    """List registered timme arch names.

    Args mirror the commonly used subset of ``timm.list_models`` so reference
    validation scripts can use this registry directly.
    """
    names = sorted(_VARIANTS.keys())
    if filter:
        filters = (filter,) if isinstance(filter, str) else tuple(filter)
        names = [n for n in names if any(fnmatch(n, f) for f in filters)]
    if exclude_filters:
        excludes = (exclude_filters,) if isinstance(exclude_filters, str) else tuple(exclude_filters)
        names = [n for n in names if not any(fnmatch(n, f) for f in excludes)]
    if pretrained:
        names = [n for n in names if get_pretrained_cfg(n, allow_unregistered=True) is not None]
    return names


def is_registered(name: str) -> bool:
    arch, _ = split_model_name_tag(name)
    return arch in _VARIANTS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(name: str) -> Tuple[_FamilyEntry, Any, Dict[str, Any]]:
    """Return (family_entry, cfg, pretrained_cfg_dict) for a model name.

    `name` may include a pretrained tag (e.g. 'resnet50.a1_in1k') — we split it
    to find the arch for the timme registry, and keep the full name for
    pretrained_cfg lookup via timm.
    """
    arch, _tag = split_model_name_tag(name)
    if arch not in _VARIANTS:
        raise RuntimeError(
            f"Model '{name}' not registered in timme (arch '{arch}' unknown). Available: {list_models()[:10]}..."
        )
    family_name, cfg = _VARIANTS[arch]
    entry = _FAMILIES[family_name]
    pcfg_obj = get_pretrained_cfg(name, allow_unregistered=True)
    pcfg = pcfg_obj.to_dict() if pcfg_obj is not None else {}
    return entry, cfg, pcfg


def _merge_pretrained_cfg(
        pcfg: Dict[str, Any],
        pretrained_cfg: Optional[Union[str, Dict[str, Any], Any]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply explicit pretrained cfg inputs using timm-compatible semantics."""
    if pretrained_cfg is not None:
        if isinstance(pretrained_cfg, str):
            pcfg_obj = get_pretrained_cfg(pretrained_cfg, allow_unregistered=True)
            pcfg = pcfg_obj.to_dict() if pcfg_obj is not None else {}
        elif hasattr(pretrained_cfg, 'to_dict'):
            pcfg = pretrained_cfg.to_dict()
        elif isinstance(pretrained_cfg, dict):
            pcfg = dict(pretrained_cfg)
        else:
            raise TypeError(f"Unsupported pretrained_cfg type: {type(pretrained_cfg).__name__}")
    else:
        pcfg = dict(pcfg)
    if pretrained_cfg_overlay:
        pcfg.update(pretrained_cfg_overlay)
    return pcfg


def _remap_single_key(key: str, layout: WeightLayout) -> str:
    """Compute the post-remap name for a single state_dict key."""
    for old_prefix, new_prefix in layout.renames:
        if key == old_prefix or key.startswith(old_prefix + '.'):
            key = new_prefix + key[len(old_prefix) :]
            break
    root = key.split('.', 1)[0]
    for old_root, target_prefix in (*layout.encoder, *layout.head):
        if root == old_root:
            return target_prefix + key[len(root) :]
    return key


def _adjust_pretrained_cfg(
        pcfg: Dict[str, Any],
        layout: WeightLayout,
        target: str = 'classifier',
) -> Dict[str, Any]:
    """Rewrite first_conv / classifier keys to match the timme namespace.

    For target='encoder', strip the leading 'encoder.' prefix so the
    keys land on a bare-encoder load (since the encoder's own state_dict
    has no namespace prefix).
    """
    out = dict(pcfg)

    def _rewrite(name: str) -> str:
        new = _remap_single_key(name, layout)
        if target == 'encoder' and new.startswith('encoder.'):
            return new[len('encoder.') :]
        if target == 'head':
            for ns in ('head.', 'head_dist.'):
                if new.startswith(ns):
                    return new[len(ns) :]
        return new

    for field in ('first_conv', 'classifier'):
        v = out.get(field)
        if v is None:
            continue
        if isinstance(v, str):
            out[field] = _rewrite(v)
        elif isinstance(v, (list, tuple)):
            out[field] = type(v)(_rewrite(x) for x in v)
    return out


def _make_filter_fn(entry: _FamilyEntry, target: str = 'classifier') -> Callable:
    """Build a filter_fn for timm.load_pretrained that:
    1. unwraps container state dicts
    2. applies the family's timm1 checkpoint_filter_fn
    3. applies timme remap_state_dict to split encoder/head namespaces
    """

    def _filter(state_dict, model):
        if isinstance(state_dict, dict):
            for k in ('state_dict', 'model', 'model_state'):
                if k in state_dict and isinstance(state_dict[k], dict):
                    state_dict = state_dict[k]
                    break
            state_dict = clean_state_dict(state_dict)
        if entry.checkpoint_filter_fn is not None:
            try:
                state_dict = entry.checkpoint_filter_fn(state_dict, model)
            except TypeError:
                state_dict = entry.checkpoint_filter_fn(state_dict)
        return remap_state_dict(state_dict, entry.weight_layout, target=target)

    return _filter


def _move_to_requested_device_dtype(model: nn.Module, kwargs: Dict[str, Any]) -> None:
    to_kwargs = {}
    if kwargs.get('device') is not None:
        to_kwargs['device'] = kwargs['device']
    if kwargs.get('dtype') is not None:
        to_kwargs['dtype'] = kwargs['dtype']
    if to_kwargs:
        model.to(**to_kwargs)


def _drop_none_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _apply_local_checkpoint(
        model: nn.Module,
        entry: _FamilyEntry,
        path: str,
        target: str = 'classifier',
) -> None:
    """Load a local checkpoint through the same filter chain as pretrained."""
    sd = torch.load(path, map_location='cpu')
    filter_fn = _make_filter_fn(entry, target=target)
    sd = filter_fn(sd, model)
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f'[timme] missing keys on local checkpoint load: {result.missing_keys[:5]}...')


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_encoder(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any]]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
        out_indices: Optional[Union[int, Tuple[int, ...]]] = None,
        checkpoint_path: Optional[str] = None,
        cache_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
        **kwargs,
) -> ImageEncoder:
    """Create an image encoder (no classification head).

    With out_indices set, forward() returns intermediate features — this
    replaces the old `features_only=True` wrapper.
    """
    if isinstance(out_indices, int):
        out_indices = tuple(range(-out_indices, 0))
    elif out_indices is not None:
        out_indices = tuple(out_indices)

    entry, cfg, pcfg = _resolve(model_name)
    pcfg = _merge_pretrained_cfg(pcfg, pretrained_cfg, pretrained_cfg_overlay)
    kwargs = _drop_none_kwargs(kwargs)

    # Source img_size from pretrained_cfg if user didn't supply one.
    if 'img_size' not in kwargs and pcfg.get('input_size'):
        kwargs['img_size'] = pcfg['input_size'][-2:]

    encoder = entry.build_encoder(cfg=cfg, out_indices=out_indices, **kwargs)
    _move_to_requested_device_dtype(encoder, kwargs)
    encoder.pretrained_cfg = pcfg
    encoder.default_cfg = pcfg

    if pretrained and pcfg:
        adjusted_pcfg = _adjust_pretrained_cfg(pcfg, entry.weight_layout, target='encoder')
        filter_fn = _make_filter_fn(entry, target='encoder')
        # For encoder-only, strip classifier key — there's no head to load.
        adjusted_pcfg = dict(adjusted_pcfg, classifier=None)
        load_pretrained(
            encoder,
            pretrained_cfg=adjusted_pcfg,
            num_classes=pcfg.get('num_classes', 1000),
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=filter_fn,
            strict=False,
            cache_dir=cache_dir,
        )
    elif checkpoint_path:
        _apply_local_checkpoint(encoder, entry, checkpoint_path, target='encoder')

    return encoder


def create_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any]]] = None,
        pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        global_pool: Optional[str] = None,
        drop_rate: float = 0.0,
        checkpoint_path: Optional[str] = None,
        cache_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
        # Compat
        features_only: bool = False,
        out_indices: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs,
) -> Union[ImageClassifier, ImageEncoder]:
    """Create a classification model (encoder + head).

    Backward-compatible with timm1's create_model. features_only=True delegates
    to create_encoder(), matching the old FeatureListNet behavior.
    """
    if features_only:
        if out_indices is None:
            out_indices = (0, 1, 2, 3, 4)
        return create_encoder(
            model_name,
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            out_indices=out_indices,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            **kwargs,
        )

    entry, cfg, pcfg = _resolve(model_name)
    pcfg = _merge_pretrained_cfg(pcfg, pretrained_cfg, pretrained_cfg_overlay)
    kwargs = _drop_none_kwargs(kwargs)

    if 'img_size' not in kwargs and pcfg.get('input_size'):
        kwargs['img_size'] = pcfg['input_size'][-2:]
    if num_classes is None:
        num_classes = pcfg.get('num_classes', 1000)

    head_kwargs: Dict[str, Any] = dict(num_classes=num_classes, drop_rate=drop_rate)
    if global_pool is not None:
        head_kwargs['global_pool'] = global_pool

    model = entry.build_classifier(cfg=cfg, **head_kwargs, **kwargs)
    _move_to_requested_device_dtype(model, kwargs)
    adjusted_pcfg = _adjust_pretrained_cfg(pcfg, entry.weight_layout)
    model.pretrained_cfg = adjusted_pcfg
    model.default_cfg = adjusted_pcfg

    if pretrained and pcfg:
        filter_fn = _make_filter_fn(entry)
        load_pretrained(
            model,
            pretrained_cfg=adjusted_pcfg,
            num_classes=num_classes,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=filter_fn,
            strict=True,
            cache_dir=cache_dir,
        )
    elif checkpoint_path:
        _apply_local_checkpoint(model, entry, checkpoint_path)

    return model
