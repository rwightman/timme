from collections import OrderedDict
from contextlib import suppress

import pytest
import torch

import timme
from timme.apps.eval_knn import unwrap_encoder_state_dict
from timme.arch import ImageClassifier, ImageEncoder, clean_state_dict
from timme.engine.config import DeviceConfig, ModelConfig
from timme.engine.device import DeviceEnv
from timme.engine.model import create_train_model
from timme.task import LeJEPATask, NEPATask


def _cpu_env() -> DeviceEnv:
    return DeviceEnv(
        device=torch.device('cpu'),
        world_size=1,
        rank=0,
        local_rank=0,
        distributed=False,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
    )


def test_create_train_model_can_target_classifier_or_encoder():
    device_env = _cpu_env()

    classifier, _ = create_train_model(
        ModelConfig(model='resnet18', num_classes=7),
        DeviceConfig(device='cpu'),
        device_env,
        target='classifier',
    )
    assert isinstance(classifier, ImageClassifier)
    assert classifier.num_classes == 7

    encoder, _ = create_train_model(
        ModelConfig(model='resnet18'),
        DeviceConfig(device='cpu'),
        device_env,
        target='encoder',
    )
    assert isinstance(encoder, ImageEncoder)
    assert not hasattr(encoder, 'head')


def test_lejepa_uses_bare_timme_encoder_for_train_and_eval():
    encoder = timme.create_encoder('vit_tiny_patch16_224', img_size=32)
    task = LeJEPATask(
        encoder,
        proj_dim=8,
        proj_hidden=16,
        proj_layers=1,
        num_slices=4,
        num_knots=5,
        verbose=False,
    )

    assert isinstance(task.trainable_module.model, ImageEncoder)

    output = task(torch.randn(2, 2, 3, 32, 32))
    assert output['loss'].ndim == 0
    assert output['output'].shape[0] == 4

    eval_task = task.get_eval_task(use_ema=False)
    eval_output = eval_task(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    assert eval_output['features'].shape[0] == 2


def test_nepa_uses_bare_timme_encoder_for_train_and_eval():
    encoder = timme.create_encoder('vit_tiny_patch16_224', img_size=32)
    task = NEPATask(encoder, shift=False, verbose=False)

    assert isinstance(task.trainable_module.model, ImageEncoder)
    assert task.trainable_module.encoder is encoder

    output = task(torch.randn(2, 3, 32, 32))
    assert output['loss'].ndim == 0
    assert output['output'].dim() == 3

    eval_task = task.get_eval_task(use_ema=False)
    eval_output = eval_task(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    assert eval_output['features'].shape[0] == 2


def test_ssl_compile_keeps_encoder_eval_path():
    if not hasattr(torch, 'compile'):
        pytest.skip('torch.compile is not available')

    encoder = timme.create_encoder('vit_tiny_patch16_224', img_size=32)
    task = LeJEPATask(
        encoder,
        proj_dim=8,
        proj_hidden=16,
        proj_layers=1,
        num_slices=4,
        num_knots=5,
        verbose=False,
    )

    task.compile(backend='eager')
    eval_task = task.get_eval_task(use_ema=False)
    eval_output = eval_task(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    assert eval_output['features'].shape[0] == 2


def test_clean_state_dict_strips_compile_and_ddp_wrappers():
    state_dict = OrderedDict([
        ('module._orig_mod.blocks.0._orig_mod.norm.weight', torch.ones(1)),
    ])
    normalized = clean_state_dict(state_dict)
    assert list(normalized) == ['blocks.0.norm.weight']


def test_eval_knn_checkpoint_keys_normalize_to_bare_encoder():
    state_dict = OrderedDict([
        ('model.patch_embed.proj.weight', torch.ones(1)),
        ('projector.0.weight', torch.zeros(1)),
    ])
    normalized = unwrap_encoder_state_dict(state_dict)
    assert list(normalized) == ['patch_embed.proj.weight']

    state_dict = OrderedDict([
        ('module._orig_mod.model.encoder.blocks.0._orig_mod.norm.weight', torch.ones(1)),
        ('module._orig_mod.model.head.fc.weight', torch.zeros(1)),
    ])
    normalized = unwrap_encoder_state_dict(state_dict)
    assert list(normalized) == ['blocks.0.norm.weight']

    state_dict = OrderedDict([
        ('model.encoder.patch_embed.proj.weight', torch.ones(1)),
        ('model.head.fc.weight', torch.zeros(1)),
    ])
    normalized = unwrap_encoder_state_dict(state_dict)
    assert list(normalized) == ['patch_embed.proj.weight']
