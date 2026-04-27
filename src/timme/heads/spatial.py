"""Spatial heads (NCHW encoder output — CNNs, ConvNets).

  SpatialLinearHead       pool -> drop -> fc
  SpatialNormMlpHead      pool -> norm -> flatten -> [mlp] -> drop -> fc
  SpatialEfficientHead    pool -> conv1x1 -> [norm] -> act -> flatten -> drop -> fc
  SpatialMlpHead          pool(flatten) -> fc1 -> act -> norm -> drop -> fc2
  SpatialAttentionHead    attn_pool -> drop -> fc

pre_logits=True always means "the representation immediately before the
final classifier linear" — useful for embeddings, retrieval, kNN probes.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from timm.layers import SelectAdaptivePool2d
from timm.layers.create_act import get_act_layer
from timm.layers.create_norm import get_norm_layer

from ..arch.head import ImageHead


class SpatialLinearHead(ImageHead):
    """Pool -> Dropout -> Linear.

    Covers: ResNet, EfficientNet, RegNet, VGG, DenseNet, SENet,
    CSPNet, DPN, Res2Net, ResNeSt, SKNet, etc.

    Equivalent to timm.layers.ClassifierHead with default settings.
    """

    accepted_fmts = ('NCHW',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.0,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self._use_conv = use_conv

        flatten_in_pool = not use_conv
        if not pool_type:
            flatten_in_pool = False
        self.global_pool = SelectAdaptivePool2d(
            pool_type=pool_type,
            flatten=flatten_in_pool,
        )
        self._num_pooled = in_features * self.global_pool.feat_mult()
        self.drop = nn.Dropout(drop_rate)
        self.fc = self._make_fc(self._num_pooled, num_classes, **dd)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def _make_fc(self, in_feat, num_classes, **dd):
        if num_classes <= 0:
            return nn.Identity()
        if self._use_conv:
            return nn.Conv2d(in_feat, num_classes, 1, bias=True, **dd)
        return nn.Linear(in_feat, num_classes, bias=True, **dd)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._num_pooled

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            flatten_in_pool = not self._use_conv and bool(pool_type)
            self.global_pool = SelectAdaptivePool2d(
                pool_type=pool_type,
                flatten=flatten_in_pool,
            )
            self._num_pooled = self._in_features * self.global_pool.feat_mult()
            self.flatten = nn.Flatten(1) if self._use_conv and pool_type else nn.Identity()
        self.fc = self._make_fc(self._num_pooled, num_classes)
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class SpatialNormMlpHead(ImageHead):
    """Pool -> Norm -> Flatten -> [Mlp] -> Dropout -> Linear.

    Covers: ConvNeXt, ConvFormer, MetaFormer, CAFormer, PoolFormer,
    HorNet, FocalNet, DaViT (CNN path), InternImage, etc.

    Equivalent to timm.layers.NormMlpClassifierHead.

    When hidden_size is set, an FC+act MLP projects to a different
    dimension before the classifier — this is the pre_logits representation.
    When hidden_size is None, norm output is the pre_logits representation.
    """

    accepted_fmts = ('NCHW',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.0,
            norm_layer: Union[str, Callable] = 'layernorm2d',
            hidden_size: Optional[int] = None,
            act_layer: Union[str, Callable] = 'tanh',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self._hidden_size = hidden_size

        # Pool (no flatten — norm operates on NCHW before flatten)
        self.use_conv = not pool_type
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        feat_dim = in_features * self.global_pool.feat_mult()

        # Norm (2D, e.g. LayerNorm2d, before flatten)
        self.norm = get_norm_layer(norm_layer)(feat_dim, **dd)

        # Flatten after norm
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()

        # Optional pre-logits MLP
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if hidden_size:
            _act = get_act_layer(act_layer)
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', linear_layer(feat_dim, hidden_size, **dd)),
                    ('act', _act()),
                ])
            )
            feat_dim = hidden_size
        else:
            self.pre_logits = nn.Identity()

        self._pre_logits_dim = feat_dim
        self.drop = nn.Dropout(drop_rate)
        self.fc = linear_layer(feat_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._pre_logits_dim

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
            self.use_conv = not pool_type
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear
        if self._hidden_size:
            # Handle conv<->linear swap for pre_logits MLP
            cur_fc = self.pre_logits.fc
            if (isinstance(cur_fc, nn.Conv2d) and not self.use_conv) or (
                isinstance(cur_fc, nn.Linear) and self.use_conv
            ):
                with torch.no_grad():
                    new_fc = linear_layer(self._in_features, self._hidden_size)
                    new_fc.weight.copy_(cur_fc.weight.reshape(new_fc.weight.shape))
                    new_fc.bias.copy_(cur_fc.bias)
                    self.pre_logits.fc = new_fc
        self.fc = linear_layer(self._pre_logits_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)


class SpatialEfficientHead(ImageHead):
    """Pool -> Conv1x1 -> [Norm] -> Act -> Flatten -> Dropout -> Linear.

    The "efficient head" pattern from MobileNetV3: global pool first,
    then a 1x1 conv expands channels (e.g. 960 -> 1280), then classify.
    Pooling before the conv reduces compute vs doing the conv at full spatial.

    Covers: MobileNetV3, MobileNetV4, TinyNet, FBNet, LCNet, HGNet, HGNetV2.

    Variants:
      - head_norm=False (MNV3 default): conv(bias=True) -> act
      - head_norm=True  (MNV4 style):   conv(bias=False) -> norm_act
    """

    accepted_fmts = ('NCHW',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            num_features: int = 1280,
            pool_type: str = 'avg',
            drop_rate: float = 0.0,
            head_norm: bool = False,
            act_layer: Union[str, Callable] = 'relu',
            norm_layer: Union[str, Callable] = 'batchnorm2d',
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self._num_features = num_features
        self._head_norm = head_norm

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # 1. Pool (no flatten — conv operates on NCHW)
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        num_pooled = in_features * self.global_pool.feat_mult()

        # 2. Channel expansion conv + optional norm + act
        if head_norm:
            # MNV4 style: conv(no bias) -> norm -> act
            self.conv_head = nn.Conv2d(num_pooled, num_features, 1, bias=False, **dd)
            self.norm_head = norm_layer(num_features, **dd)
            self.act = act_layer(inplace=True)
        else:
            # MNV3 style: conv(bias) -> act (no norm)
            self.conv_head = nn.Conv2d(num_pooled, num_features, 1, bias=True, **dd)
            self.norm_head = nn.Identity()
            self.act = act_layer(inplace=True)

        # 3. Flatten -> drop -> fc
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(num_features, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._num_features

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
            self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.fc = nn.Linear(self._num_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)


class SpatialMlpHead(ImageHead):
    """Pool(flatten) -> FC1 -> Act -> Norm -> Dropout -> FC2.

    An MLP head that expands to a hidden dim before classifying.
    Unlike SpatialNormMlpHead, the expansion is a full linear layer
    (not a pre-logits projection) and norm comes after activation.

    Covers: InceptionNeXt, and other architectures with MLP-style heads.
    """

    accepted_fmts = ('NCHW',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.0,
            mlp_ratio: float = 3.0,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Union[str, Callable] = 'layernorm',
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        hidden_features = int(mlp_ratio * in_features)
        self._hidden_features = hidden_features

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Pool with flatten (FC operates on flat vectors)
        assert pool_type, 'SpatialMlpHead requires pooling'
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        num_pooled = in_features * self.global_pool.feat_mult()

        # MLP: fc1 -> act -> norm -> drop
        self.fc1 = nn.Linear(num_pooled, hidden_features, bias=bias, **dd)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features, **dd)
        self.drop = nn.Dropout(drop_rate)

        # Classifier
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._hidden_features

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            assert pool_type, 'SpatialMlpHead requires pooling'
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.fc2 = nn.Linear(self._hidden_features, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc2

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc2(x)


class SpatialAttentionHead(ImageHead):
    """AttentionPool2d -> drop -> fc.

    Covers: BYOB models with AttentionPool2d / RotAttentionPool2d.
    The attention pool module is passed in — this head doesn't construct it,
    since there are several variants with different constructor signatures.
    """

    accepted_fmts = ('NCHW',)

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            attn_pool: nn.Module,
            pool_out_features: Optional[int] = None,
            drop_rate: float = 0.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        dd = dict(device=device, dtype=dtype)
        self._in_features = in_features
        self._num_classes = num_classes
        self._pre_logits_dim = pool_out_features or in_features

        self.attn_pool = attn_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self._pre_logits_dim, num_classes, **dd) if num_classes > 0 else nn.Identity()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def pre_logits_dim(self) -> int:
        return self._pre_logits_dim

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        # pool_type ignored — attention pool is structural, can't swap at runtime
        self.fc = nn.Linear(self._pre_logits_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._num_classes = num_classes

    def get_classifier(self) -> nn.Module:
        return self.fc

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.attn_pool(x)
        x = self.drop(x)
        if pre_logits:
            return x
        return self.fc(x)
