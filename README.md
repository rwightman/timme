# timme — timm, evolved

`timme` is an experimental refactor of [timm](https://github.com/huggingface/pytorch-image-models) that splits every image model into a reusable **encoder** and a separate **head**. It's a thin layer on top of timm — for now it reuses timm's blocks, layers, hub integration, and pretrained-weight infrastructure, while exposing a cleaner API for feature extraction, head swapping, and weight remapping.

> **Status:** alpha / proof-of-concept. 9 model families wired (~109 variants). The shape of the public API is settling but not stable. Not production-ready.

## Why

timm models are wonderful but each one is a monolithic `nn.Module` with `forward_features` + `forward_head` baked together. That makes a few things awkward:

- using a backbone for downstream tasks (detection, segmentation, dense prediction) requires hand-rolling head removal,
- swapping a head (token-pool vs avg-pool vs attention-pool, distillation, MAP) requires reaching into model internals,
- intermediate features (`features_only=True`) go through a separate `FeatureListNet` wrapper rather than the model itself.

`timme` says: every model is `ImageClassifier(encoder, head)` where `encoder` IS the features (no `forward_features` indirection) and `head` is a swappable, well-typed module. Pretrained weights load via a small per-family `WeightLayout` that splits old monolithic state dicts into `encoder.*` / `head.*`.

## Install

```bash
pip install timme         # not yet on PyPI; for now:
pip install -e .          # from a clone
```

Runtime dependency: `timm>=1.0`, `torch>=2.0`.

## Usage

Drop-in replacement for `timm.create_model`:

```python
import timme

# Pretrained classifier — same names as timm
model = timme.create_model('resnet50.a1_in1k', pretrained=True)
model.eval()

# Logits match timm bit-for-bit
import torch, timm
x = torch.randn(1, 3, 224, 224)
y2 = model(x)
y1 = timm.create_model('resnet50.a1_in1k', pretrained=True).eval()(x)
assert torch.equal(y1, y2)
```

Encoder-only (replaces `features_only=True`):

```python
encoder = timme.create_encoder('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
features = encoder(x)                                 # (B, 197, 768) — NLC
encoder = timme.create_encoder('resnet50', pretrained=True, out_indices=(0, 1, 2, 3, 4))
stages = encoder(x)                                   # list of stage tensors
```

Swap heads or change `num_classes` / `in_chans`:

```python
model = timme.create_model('resnet50.a1_in1k', pretrained=True, num_classes=10)
model = timme.create_model('resnet50.a1_in1k', pretrained=True, in_chans=1)
```

## Training And Eval Apps

Native task-style apps live under `timme.apps` and are installed as console scripts.
The intended naming is `train_{task}.py` and `eval_{task}.py`; today that means
classification training/eval, SSL training, and k-NN representation eval.

```bash
# console scripts
timme-train-cls --data.path /data/imagenet --model.model resnet50
timme-train-ssl --data.path /data/imagenet --model.model vit_tiny_patch16_224 --ssl.ssl_method nepa
timme-eval-cls --data-dir /data/imagenet/validation --model resnet50 --pretrained
timme-eval-knn --data-dir /data/imagenet --model vit_tiny_patch16_224 --checkpoint /path/to/last.pth.tar

# equivalent module entry points, useful from a source checkout
python -m timme.apps.train_cls --data.path /data/imagenet --model.model resnet50
python -m timme.apps.eval_cls --data-dir /data/imagenet/validation --model resnet50 --pretrained
```

Classification training:

```bash
torchrun --nproc-per-node=8 -m timme.apps.train_cls \
  --data.path /data/imagenet \
  --model.model resnet50 \
  --model.pretrained true \
  --loader.batch_size 128 \
  --scheduler.epochs 100 \
  --device.amp true
```

Override the train defaults when needed:

```bash
timme-train-cls \
  --data.path /data/imagenet \
  --model.model convnext_tiny \
  --optimizer.lr 5e-4 \
  --optimizer.weight_decay 0.05 \
  --model.torchcompile inductor
```

Self-supervised training is encoder-native: `timme-train-ssl` builds a bare
`timme.create_encoder(...)` via `create_train_model(..., target='encoder')`.
The SSL tasks train representations directly, without a classifier wrapper/head.

```bash
# NEPA
timme-train-ssl \
  --data.path /data/imagenet \
  --model.model vit_tiny_patch16_224 \
  --ssl.ssl_method nepa \
  --scheduler.epochs 100 \
  --device.amp true

# LeJEPA multi-view training
timme-train-ssl \
  --data.path /data/imagenet \
  --model.model vit_small_patch16_224 \
  --ssl.ssl_method lejepa \
  --ssl.num_views 2 \
  --ssl.lejepa_lamb 0.02
```

Evaluate checkpoints:

```bash
timme-eval-cls \
  --data-dir /data/imagenet/validation \
  --model resnet50 \
  --checkpoint output/train/model_best.pth.tar

timme-eval-knn \
  --data-dir /data/imagenet \
  --model vit_tiny_patch16_224 \
  --checkpoint output/ssl/model_best.pth.tar \
  --k 1 5 20 100
```

`timme-eval-knn` also uses `timme.create_encoder(...)` and normalizes task,
DDP, and `torch.compile` checkpoint wrapper keys before loading into the bare
encoder. For DDP plus `torch.compile`, the apps apply distributed wrapping before
task compilation; when gradient checkpointing is enabled, timme disables Dynamo's
DDP optimizer guard path that is known to fail in recent PyTorch versions.

Config files are YAML dataclass trees with the same dotted keys as the CLI:

```yaml
model:
  model: vit_tiny_patch16_224
data:
  path: /data/imagenet
loader:
  batch_size: 256
optimizer:
  opt: adamw
  lr: 3e-4
  weight_decay: 0.01
scheduler:
  epochs: 100
  warmup_epochs: 5
ssl:
  ssl_method: nepa
```

```bash
timme-train-ssl -c configs/nepa_vit.yaml --data.path /data/imagenet
```

## What's implemented

Current model coverage is 109 variants across 9 families, all exact-matching
timm pretrained weights:

| family       | example variants                                                        |
|--------------|-------------------------------------------------------------------------|
| ResNet       | `resnet18`, `resnet50`, `resnet101`, `resnet50d`, `seresnet50`          |
| ViT          | `vit_tiny/small/base/large_patch16_224/384`, CLIP/DINOv2 variants       |
| ConvNeXt     | `convnext_tiny/small/base/large`, `convnextv2_base`                     |
| MobileNetV3  | `mobilenetv3_large_100`, `mobilenetv3_small_100`                        |
| ByobNet      | ~70 variants — gernet, resnet51q, regnetz, eca_resnet, etc.             |
| DeiT         | `deit_*`, `deit3_*`, distilled `deit_*_distilled_*`                     |
| LeViT        | `levit_128/192/256/384`, conv-mode variants                             |
| NaFlexViT    | `naflexvit_base/so150m2/so400m_patch16_*` (gap, par_gap, map, siglip)   |
| EVA / EVA02  | `eva_giant_patch14_*`, `eva02_tiny/small/base/large_patch14_*`, CLIP    |

9 canonical heads cover the head-side variability (5 spatial for CNNs, 4 token for transformers). See [ARCHITECTURE.md](ARCHITECTURE.md).

Native apps currently include:

- `timme.apps.train_cls` / `timme-train-cls`: classification, distillation, DDP, AMP, EMA, mixup/cutmix, update-step scheduling.
- `timme.apps.eval_cls` / `timme-eval-cls`: classifier validation.
- `timme.apps.train_ssl` / `timme-train-ssl`: encoder-native NEPA and LeJEPA training.
- `timme.apps.eval_knn` / `timme-eval-knn`: encoder-native k-NN representation eval.
- `timme.apps.sweep` / `timme-sweep`: lightweight config sweep runner.

## What's not implemented yet

- The remaining ~80 timm families. Each one needs the same wiring: encoder class + `WeightLayout` + builder + `register_family(...)`.
- Many advanced application scripts from timm/OpenCLIP are not ported yet. Classification and SSL task apps are the active native paths.
- Standalone hub story. timme reuses `timm.models._registry` for pretrained_cfg metadata and `timm.models._builder.load_pretrained` for downloads, so it depends on timm's hub integration today.
- Standalone layer primitives. `timme.layers` is a placeholder; blocks/norms/activations/attention pools are imported from `timm.layers`.

## Roadmap

The plan is for timme to grow into a fully standalone replacement for timm. Near-term priorities:

1. **More families** — work through the rest of timm's model zoo, prioritizing actively-developed ones.
2. **Vendor / fork shared layers** as needed — the runtime dep on timm is fine for v0 but limits independent evolution.
3. **More native apps** — extend the task app base toward additional workflows.
4. **Hub integration** — read pretrained metadata from timme's own registry instead of borrowing timm's.

## License

Apache-2.0. Built on timm (also Apache-2.0). See `LICENSE`.
