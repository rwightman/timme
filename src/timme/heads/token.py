"""Token heads (NLC encoder output — ViTs, transformers, sequence models).

  TokenLinearHead    token/avg/max pool -> [norm] -> drop -> fc
  TokenNormMlpHead   pool -> norm -> [mlp] -> drop -> fc
  TokenSelectHead    select index -> [norm] -> drop -> fc   (for distilled models)
  TokenAttentionHead attn_pool -> [norm] -> drop -> fc
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from timm.layers import global_pool_nlc
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm import get_norm_layer

from ..arch.head import ImageHead


class TokenLinearHead(ImageHead):
    """Token/avg/max pool -> norm -> dropout -> linear.

    Covers: ViT, EVA, Hiera, BEiT, Swin, DaViT, MLP-Mixer, PVTv2,
    XCiT, CaiT, MetaFormer, Sequencer, TNT, Twins, etc.

    pool_type='token' takes cls_token (index 0).
    pool_type='avg'/'max'/'avgmax' reduces over spatial tokens.
    """

    accepted_fmts = ('NLC',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'token',
            num_prefix_tokens: int = 1,
            pool_include_prefix: bool = False,
            norm_layer: Optional[Union[str, Callable]] = None,
            drop_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self.pool_type = pool_type
        self.num_prefix_tokens = num_prefix_tokens
        self.pool_include_prefix = pool_include_prefix

        if norm_layer is not None:
            self.fc_norm = get_norm_layer(norm_layer)(in_features, **dd)
        else:
            self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._in_features

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return global_pool_nlc(
            x,
            pool_type=self.pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            self.pool_type = pool_type
        self.fc = nn.Linear(self._in_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self._pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        if pre_logits:
            return x
        return self.fc(x)


class TokenNormMlpHead(ImageHead):
    """Pool -> Norm -> [Mlp] -> Dropout -> Linear, for NLC inputs.

    The token-sequence parallel to SpatialNormMlpHead. Pools over
    the sequence dimension, then norm -> optional MLP -> classify.

    Covers: Hiera, HieraDet SAM2, MambaOut.

    Equivalent to timm.layers.ClNormMlpClassifierHead.
    """

    accepted_fmts = ('NLC',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            num_prefix_tokens: int = 0,
            pool_include_prefix: bool = False,
            drop_rate: float = 0.0,
            norm_layer: Union[str, Callable] = 'layernorm',
            hidden_size: Optional[int] = None,
            act_layer: Union[str, Callable] = 'gelu',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self._hidden_size = hidden_size
        self.pool_type = pool_type
        self.num_prefix_tokens = num_prefix_tokens
        self.pool_include_prefix = pool_include_prefix

        self.norm = get_norm_layer(norm_layer)(in_features, **dd)

        feat_dim = in_features
        if hidden_size:
            _act = get_act_layer(act_layer)
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(feat_dim, hidden_size, **dd)),
                    ('act', _act()),
                ])
            )
            feat_dim = hidden_size
        else:
            self.pre_logits = nn.Identity()

        self._pre_logits_dim = feat_dim
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(feat_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._pre_logits_dim

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return global_pool_nlc(
            x,
            pool_type=self.pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            self.pool_type = pool_type
        self.fc = nn.Linear(self._pre_logits_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self._pool(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)


class TokenSelectHead(ImageHead):
    """Select one token by index -> [norm] -> dropout -> Linear.

    For models where the encoder produces prefix tokens that should be
    consumed individually — e.g. DeiT distilled has [cls, dist, ...] and
    pairs two of these heads (token_index=0 and token_index=1) via
    DistilledImageClassifier.

    Identical to TokenLinearHead except it selects a single token position
    rather than pool-reducing over the sequence.
    """

    accepted_fmts = ('NLC',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            token_index: int = 0,
            norm_layer: Optional[Union[str, Callable]] = None,
            drop_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self.token_index = token_index

        if norm_layer is not None:
            self.fc_norm = get_norm_layer(norm_layer)(in_features, **dd)
        else:
            self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._in_features

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        # pool_type ignored — token index is structural
        self.fc = nn.Linear(self._in_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x[:, self.token_index]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        if pre_logits:
            return x
        return self.fc(x)


class TokenAttentionHead(ImageHead):
    """Attention pool -> norm -> dropout -> linear.

    Covers: ViT with global_pool='map' (AttentionPoolLatent)
            or global_pool='prr' (AttentionPoolPrr), EVA attention pooling.

    The attention pool module is passed in since variants have different
    constructor signatures (latent query vs pooled-token query, etc.).
    """

    accepted_fmts = ('NLC',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            attn_pool: nn.Module,
            num_prefix_tokens: int = 1,
            pool_include_prefix: bool = False,
            norm_layer: Optional[Union[str, Callable]] = None,
            drop_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self.num_prefix_tokens = num_prefix_tokens
        self.pool_include_prefix = pool_include_prefix

        self.attn_pool = attn_pool
        if norm_layer is not None:
            self.fc_norm = get_norm_layer(norm_layer)(in_features, **dd)
        else:
            self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._in_features

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        # Attention pool is structural — pool_type ignored
        self.fc = nn.Linear(self._in_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if not self.pool_include_prefix:
            x = x[:, self.num_prefix_tokens:]
        x = self.attn_pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        if pre_logits:
            return x
        return self.fc(x)
