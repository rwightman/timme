# Architecture

```
src/timme/
├── __init__.py                # public API: create_model, create_encoder, ABCs, heads, configs
├── arch/                      # package vocabulary — depended on by models/ and heads/
│   ├── __init__.py
│   ├── encoder.py             # ImageEncoder
│   ├── head.py                # ImageHead (also a duck-typed contract)
│   ├── classifier.py          # ImageClassifier, DistilledImageClassifier
│   ├── traits.py              # ArchTraits
│   ├── weights.py             # WeightLayout, remap_state_dict
│   └── config.py              # ConfigMixin, HeadCfg, ModelSpec
├── models/                    # per-family encoders + factory + registry
│   ├── __init__.py            # imports each family (triggers registration), exports configs
│   ├── _factory.py            # _FamilyEntry, register_family, create_model, create_encoder
│   ├── resnet.py              # ResNetCfg + ResNet + RESNET_CFGS + builders
│   ├── vision_transformer.py  # VisionTransformerCfg + VisionTransformer + VIT_CFGS + builders
│   ├── convnext.py            # ConvNeXtCfg + ConvNeXt + CONVNEXT_CFGS + builders
│   ├── mobilenetv3.py         # MobileNetV3Cfg + MobileNetV3 + MOBILENETV3_CFGS + builders
│   ├── byobnet.py             # ByoModelCfg/ByoBlockCfg + ByobNet + variants + builders
│   ├── deit.py                # VisionTransformerDistilled + DEIT_CFGS + builders
│   ├── levit.py               # LevitCfg + Levit + LEVIT_CFGS + builders
│   ├── naflexvit.py           # NaFlexVitCfg + NaFlexVit + NAFLEXVIT_CFGS + builders
│   └── eva.py                 # EvaCfg + Eva + EVA_CFGS + builders (EVA / EVA02 / RoPE-ViT)
├── heads/                     # 9 canonical classification heads
│   ├── __init__.py
│   ├── spatial.py             # 5 NCHW heads
│   └── token.py               # 4 NLC heads
└── layers/                    # shared layer primitives (currently empty; reuses timm.layers)
    └── __init__.py
```

Dependency direction is unidirectional: `models/` and `heads/` reach up into `arch/` for the package vocabulary; nothing reaches sideways. `arch/classifier.py` interacts with the head via duck-typing (`accepted_fmts` attr) so it doesn't have to import from `heads/`.

Each modeling file follows a consistent shape (mirrors timm's `class -> default_cfgs -> @register_model fns` pattern):

```
imports
<FamilyName>Cfg dataclass            # at the top
class <FamilyName>(ImageEncoder)     # the encoder
<FAMILY>_WEIGHT_LAYOUT               # state-dict remap layout
<FAMILY>_CFGS                        # variant dict (between modeling code and entry fns)
[head factory + filter fn]
build_<family>_encoder(cfg, ...)     # entry-point builder
build_<family>_classifier(cfg, ...)  # entry-point builder
register_family(...)                 # registration call
```

## Core abstractions

**`ImageEncoder`** — base class. `forward()` IS the features; no separate `forward_features()`. Two output contracts (fixed per instance, safe for tracing):

- without `out_indices`: `forward(x) -> Tensor` (final features)
- with `out_indices`: `forward(x) -> List[Tensor]` (intermediate features at given indices, replaces timm's `features_only=True` wrapper)

`forward_intermediates(x, indices=..., ...)` is always available regardless of construction — that's the runtime-flexible research path.

**`ImageHead`** — base class / duck-typed contract. `forward(x, pre_logits=False)` IS the head. `pre_logits=True` always means "right before the final classifier linear" — the embedding useful for retrieval, kNN probes, CLIP-style use.

Each head declares `accepted_fmts: Tuple[str, ...]` (e.g. `('NCHW',)` or `('NLC',)`); `ImageClassifier` validates encoder/head format compatibility at construction.

**`ImageClassifier`** — composition: `encoder + head`. `forward_features() = encoder(x)`, `forward_head() = head(x)`, `forward() = head(encoder(x))`. `state_dict` keys are `encoder.*` and `head.*` — sibling, not nested.

**`DistilledImageClassifier`** — `encoder + head + head_dist`. Each of the two heads consumes the full encoder output and handles its own token selection / pooling. Averaged at inference; pair returned during distilled training.

**`ArchTraits`** — declarative per-arch properties (`output_fmt`, `output_dim`, `num_prefix_tokens`, `default_pool_type`, etc.) set at construction. Heads use these for compatibility checks instead of runtime `isinstance`.

**`WeightLayout`** — describes how a flat upstream timm state_dict maps into `encoder.*` / `head.*` namespaces. `encoder` and `head` fields are tuples of either:

- a plain string `'conv1'` — shorthand for `('conv1', 'encoder.conv1')` (identity under the implied namespace)
- an explicit `(old_root, new_full_prefix)` pair for renames or non-identity targets

Identity is the common case; renames cover things like `('classifier', 'head.fc')` (MobileNetV3) or `('head', 'head.fc')` (ViT, where the timm1 `head` Linear lives one level deeper inside `TokenLinearHead`).

## Heads

| Head                 | Flow                                                | Covers                                                         |
|----------------------|-----------------------------------------------------|----------------------------------------------------------------|
| `SpatialLinearHead`  | pool → drop → fc                                    | ResNet, RegNet, VGG, DenseNet, SENet, ~30 families             |
| `SpatialNormMlpHead` | pool → norm → flatten → [mlp] → drop → fc           | ConvNeXt, MetaFormer, CAFormer, HorNet, FocalNet, ~10 families |
| `SpatialEfficientHead` | pool → conv1x1 → [norm] → act → flatten → drop → fc | MobileNetV3/V4, HGNet, LCNet, TinyNet, ~5 families           |
| `SpatialMlpHead`     | pool(flatten) → fc1 → act → norm → drop → fc2       | InceptionNeXt, ~4 families                                     |
| `SpatialAttentionHead` | attn_pool → drop → fc                             | BYOB attn-pool models (e.g. resnet50_clip), CoAtNet            |
| `TokenLinearHead`    | token/avg/max pool → [norm] → drop → fc             | ViT, EVA, Hiera, BEiT, Swin, ~20 families                      |
| `TokenNormMlpHead`   | pool → norm → [mlp] → drop → fc                     | Hiera, HieraDet SAM2, MambaOut                                 |
| `TokenSelectHead`    | select index → [norm] → drop → fc                   | Distilled models (DeiT cls vs dist)                            |
| `TokenAttentionHead` | attn_pool → [norm] → drop → fc                      | ViT map/prr, EVA attention pool                                |

## Factory + weight loading

`timme.create_model(name)` resolves the arch name against the registry, builds the model from its config dataclass, and (if `pretrained=True`) loads weights via `timm.models._builder.load_pretrained`. The filter chain is:

1. Fetch state_dict from URL / HF hub / file (timm handles this).
2. Unwrap container dicts (`'state_dict'`, `'model'`, ...).
3. Apply the family's upstream `checkpoint_filter_fn` (pos_embed resize, key renames — still in upstream namespace).
4. Apply timme's `remap_state_dict` to split into the requested target's namespace.
5. timm's `load_pretrained` adapts `first_conv` for `in_chans` and strips `classifier` weights for `num_classes` mismatch — timme rewrites those keys in the `pretrained_cfg` to point at the right namespace.

`remap_state_dict` honors the load target:

- `target='classifier'` (used by `create_model`) — keys land at `encoder.X` / `head.X` to match the composed `ImageClassifier(encoder, head)` state dict.
- `target='encoder'` (used by `create_encoder`) — keys land at bare `X`, with the leading `encoder.` stripped so they load directly into a bare `ImageEncoder`.
- `target='head'` — same treatment for `head.` / `head_dist.` (used internally for head-only loads).

`_adjust_pretrained_cfg` mirrors the same target-aware rewrite on the `first_conv` / `classifier` keys so timm's `load_pretrained` adaptations land on keys that actually exist in the filtered state dict.
