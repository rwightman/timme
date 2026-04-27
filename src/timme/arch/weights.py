"""WeightLayout + remap_state_dict — old monolithic state_dict -> encoder/head split.

A WeightLayout describes how the keys of a timm1-style flat state_dict map
into the timm-evolved ``encoder.*`` / ``head.*`` namespaces. Each entry in
``encoder`` / ``head`` is one of:

  * a plain string — shorthand for "this root stays under the implied
    namespace with the same name". An entry ``'conv1'`` in ``encoder`` is
    the same as ``('conv1', 'encoder.conv1')``; an entry ``'fc'`` in
    ``head`` is the same as ``('fc', 'head.fc')``.

  * an explicit ``(old_root, new_full_prefix)`` pair — full control over
    where the root lands. Use this for identity mapping when the upstream
    keys already nest under the right namespace (``('head', 'head')``),
    cross-name renames (``('classifier', 'head.fc')``), or distilled
    sibling namespaces (``('head_dist', 'head_dist.fc')``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import torch


@dataclass
class WeightLayout:
    """Describes how old monolithic model keys map to encoder/head split."""

    encoder: Tuple[Union[str, Tuple[str, str]], ...]
    head: Tuple[Union[str, Tuple[str, str]], ...]
    renames: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        self.encoder = tuple(_normalize_layout_entry(e, 'encoder') for e in self.encoder)
        self.head = tuple(_normalize_layout_entry(e, 'head') for e in self.head)


def _normalize_layout_entry(entry, namespace: str) -> Tuple[str, str]:
    """Expand a shorthand string entry to an explicit (old, new) pair."""
    if isinstance(entry, str):
        return (entry, f'{namespace}.{entry}')
    old_root, new_prefix = entry
    return (old_root, new_prefix)


def remap_state_dict(
        state_dict: Dict[str, torch.Tensor],
        layout: WeightLayout,
        target: Literal['encoder', 'head', 'classifier'] = 'classifier',
) -> Dict[str, torch.Tensor]:
    """Remap a flat upstream state_dict into encoder/head namespaced keys.

    Args:
        state_dict: Flat upstream state dict (keys like 'conv1.weight', 'fc.weight').
        layout: Describes which prefixes belong to encoder vs head.
        target:
            'encoder' — return only encoder keys
            'head' — return only head keys
            'classifier' — return all keys, prefixed with encoder.* / head.*
    """
    out = {}
    for key, value in state_dict.items():
        new_key = key

        # Apply explicit pre-split renames first (intra-key mutations).
        for old_prefix, new_prefix in layout.renames:
            if key == old_prefix or key.startswith(old_prefix + '.'):
                new_key = new_prefix + key[len(old_prefix):]
                break

        root = new_key.split('.', 1)[0]

        # Look up the root against encoder / head entries.
        matched_prefix: Optional[str] = None
        matched_side: Optional[str] = None
        for old_root, target_prefix in layout.encoder:
            if root == old_root:
                matched_prefix, matched_side = target_prefix, 'encoder'
                break
        if matched_prefix is None:
            for old_root, target_prefix in layout.head:
                if root == old_root:
                    matched_prefix, matched_side = target_prefix, 'head'
                    break
        if matched_prefix is None:
            continue  # unrecognized key

        remapped = matched_prefix + new_key[len(root):]

        if target == 'classifier':
            out[remapped] = value
        elif target == 'encoder' and matched_side == 'encoder':
            # Strip the leading 'encoder.' so the key lands on a bare encoder.
            out[_strip_namespace(remapped, 'encoder')] = value
        elif target == 'head' and matched_side == 'head':
            # Strip leading 'head.' / 'head_dist.' for bare-head loading.
            # First segment is the namespace ('head' or 'head_dist').
            ns = remapped.split('.', 1)[0]
            out[_strip_namespace(remapped, ns)] = value

    return out


def _strip_namespace(key: str, namespace: str) -> str:
    """Return key with a leading ``{namespace}.`` removed, or unchanged
    if it doesn't start with that prefix (defensive — shouldn't happen
    given how remap routes targets to namespaces)."""
    prefix = namespace + '.'
    if key.startswith(prefix):
        return key[len(prefix):]
    if key == namespace:
        return ''
    return key
