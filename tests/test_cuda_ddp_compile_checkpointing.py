import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from timme.task import ClassificationTask


class CudaCheckpointBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class CudaCheckpointModel(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, num_classes: int = 11):
        super().__init__()
        self.grad_checkpointing = False
        self.stem = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([CudaCheckpointBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(self.norm(x))


def _dynamo_counters():
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    return {key: dict(value) for key, value in counters.items() if "ddp" in key.lower() or "graph" in key.lower()}


def _cuda_ddp_compile_worker(
        rank,
        world_size,
        init_method,
        optimize_ddp,
        backend,
        mode,
        bucket_cap_mb,
        result_queue,
):
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=180),
        )

        import torch._dynamo as dynamo

        dynamo.reset()
        dynamo.config.suppress_errors = False
        dynamo.config.optimize_ddp = optimize_ddp

        torch.manual_seed(1000 + rank)
        model = CudaCheckpointModel().to(device)
        model.set_grad_checkpointing(True)
        task = ClassificationTask(
            model,
            nn.CrossEntropyLoss().to(device),
            device=device,
            verbose=False,
        )
        optimizer = torch.optim.AdamW(task.get_trainable_module().parameters(), lr=1e-3)

        task.prepare_distributed(
            device_ids=[rank],
            output_device=rank,
            bucket_cap_mb=bucket_cap_mb,
        )
        assert isinstance(task.trainable_module, DDP)

        # This file is a raw PyTorch behavior probe. Bypass timme's production
        # guard so optimize_ddp=True actually exercises Dynamo's DDPOptimizer.
        task._maybe_disable_ddp_dynamo_optimizer = lambda module=None: False
        task.compile(backend=backend, mode=mode)
        assert hasattr(task.trainable_module, "_orig_mod")
        assert isinstance(task.trainable_module._orig_mod, DDP)

        for step in range(3):
            # Exercise gradient accumulation + DDP no_sync on the compiled wrapper.
            for micro_step in range(2):
                x = torch.randn(16, 128, device=device)
                target = torch.randint(0, 11, (16,), device=device)
                if micro_step == 0:
                    with task.no_sync():
                        loss = task(x, target)["loss"] / 2
                        loss.backward()
                else:
                    loss = task(x, target)["loss"] / 2
                    loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)

        eval_task = task.get_eval_task(use_ema=False)
        x = torch.randn(8, 128, device=device)
        target = torch.randint(0, 11, (8,), device=device)
        result = eval_task(x, target)
        assert "acc1" in result
        torch.cuda.synchronize(device)

        if rank == 0:
            result_queue.put({
                "status": "passed",
                "optimize_ddp": optimize_ddp,
                "backend": backend,
                "mode": mode,
                "bucket_cap_mb": bucket_cap_mb,
                "torch": torch.__version__,
                "counters": _dynamo_counters(),
            })
    except Exception as exc:
        result_queue.put({
            "status": "failed",
            "rank": rank,
            "optimize_ddp": optimize_ddp,
            "backend": backend,
            "mode": mode,
            "bucket_cap_mb": bucket_cap_mb,
            "torch": torch.__version__,
            "error": repr(exc),
            "counters": _dynamo_counters(),
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "optimize_ddp",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "PyTorch 2.9 and 2.11 DDPOptimizer can fail with "
                    "torch.compile + DDP + non-reentrant grad checkpointing."
                ),
            ),
        ),
        False,
    ],
)
def test_cuda_ddp_compile_grad_checkpointing_optimize_ddp(optimize_ddp):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("requires NCCL distributed backend")
    if not hasattr(torch._dynamo.config, "optimize_ddp"):
        pytest.skip("torch._dynamo.config.optimize_ddp is not available")

    backend = os.environ.get("TIMME_DDP_COMPILE_BACKEND", "inductor")
    mode = os.environ.get("TIMME_DDP_COMPILE_MODE") or None
    bucket_cap_mb = float(os.environ.get("TIMME_DDP_BUCKET_CAP_MB", "0.001"))

    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, f"cuda_ddp_compile_init_{int(optimize_ddp)}")
        init_method = f"file://{init_file}"
        mp.start_processes(
            _cuda_ddp_compile_worker,
            args=(
                world_size,
                init_method,
                optimize_ddp,
                backend,
                mode,
                bucket_cap_mb,
                result_queue,
            ),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    failures = [result for result in results if result["status"] == "failed"]
    assert not failures, failures
    assert any(
        result["status"] == "passed" and result["optimize_ddp"] == optimize_ddp
        for result in results
    ), results
