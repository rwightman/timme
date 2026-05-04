from timme.engine.config import TrainConfig


def test_train_optimizer_defaults_are_adamw_fixed_lr():
    cfg = TrainConfig()
    assert cfg.optimizer.opt == 'adamw'
    assert cfg.optimizer.lr == 3e-4
    assert cfg.optimizer.weight_decay == 0.01
