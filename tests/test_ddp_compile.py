import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from timme.task import ClassificationTask


class CheckpointBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class TinyCheckpointModel(nn.Module):
    def __init__(self, dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.grad_checkpointing = False
        self.block = CheckpointBlock(dim)
        self.head = nn.Linear(dim, num_classes)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x):
        if self.grad_checkpointing and self.training:
            x = checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        return self.head(x)


def _ddp_compile_worker(rank, world_size, init_method, use_grad_checkpointing, error_queue):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        torch.manual_seed(123 + rank)

        model = TinyCheckpointModel()
        model.set_grad_checkpointing(use_grad_checkpointing)
        task = ClassificationTask(model, nn.CrossEntropyLoss(), verbose=False)
        optimizer = torch.optim.SGD(task.get_trainable_module().parameters(), lr=0.01)

        task.prepare_distributed(device_ids=None)
        assert isinstance(task.trainable_module, DDP)

        import torch._dynamo as dynamo

        has_optimize_ddp = hasattr(dynamo.config, "optimize_ddp")
        if has_optimize_ddp:
            dynamo.config.optimize_ddp = True

        task.compile(backend="eager")
        assert hasattr(task.trainable_module, "_orig_mod")
        assert isinstance(task.trainable_module._orig_mod, DDP)
        if has_optimize_ddp:
            assert dynamo.config.optimize_ddp is (not use_grad_checkpointing)

        # First accumulation step should use DDP no_sync through the compiled wrapper.
        input = torch.randn(4, 8)
        target = torch.randint(0, 3, (4,))
        with task.no_sync():
            result = task(input, target)
            (result["loss"] / 2).backward()

        input = torch.randn(4, 8)
        target = torch.randint(0, 3, (4,))
        result = task(input, target)
        (result["loss"] / 2).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        eval_task = task.get_eval_task(use_ema=False)
        assert hasattr(eval_task.trainable_module, "_orig_mod")
        eval_result = eval_task(torch.randn(2, 8), torch.randint(0, 3, (2,)))
        assert "acc1" in eval_result

        dist.barrier()
    except Exception as exc:
        error_queue.put(repr(exc))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("use_grad_checkpointing", [False, True])
def test_ddp_wrapped_trainable_compile_supports_eval_and_no_sync(use_grad_checkpointing):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    world_size = 2
    ctx = mp.get_context("spawn")
    error_queue = ctx.SimpleQueue()

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "ddp_init")
        init_method = f"file://{init_file}"
        mp.start_processes(
            _ddp_compile_worker,
            args=(world_size, init_method, use_grad_checkpointing, error_queue),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    assert not errors
