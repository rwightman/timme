from contextlib import suppress

import torch

from timme.engine.config import SchedulerConfig, TrainConfig
from timme.engine.device import DeviceEnv
from timme.engine.optim import create_train_scheduler


def test_train_optimizer_defaults_are_adamw_fixed_lr():
    cfg = TrainConfig()
    assert cfg.optimizer.opt == 'adamw'
    assert cfg.optimizer.lr == 3e-4
    assert cfg.optimizer.weight_decay == 0.01


def test_scheduler_defaults_are_update_based_with_prefix_warmup():
    cfg = SchedulerConfig(epochs=2, warmup_epochs=1)
    assert cfg.warmup_lr == 0.0
    assert not hasattr(cfg, 'sched_on_updates')
    assert not hasattr(cfg, 'warmup_prefix')

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    device_env = DeviceEnv(
        device=torch.device('cpu'),
        world_size=1,
        rank=0,
        local_rank=0,
        distributed=False,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
    )
    scheduler, num_epochs = create_train_scheduler(
        optimizer,
        cfg,
        updates_per_epoch=10,
        device_env=device_env,
    )

    assert scheduler.warmup_prefix is True
    assert scheduler.warmup_t == 10
    assert scheduler.t_initial == 20
    assert num_epochs == 3
