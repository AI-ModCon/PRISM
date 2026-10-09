"""Regression test for defect A: DDP does not forward `.config` to the
wrapped module, so `getattr(model, "config", None)` (the validation-loader
dispatch in trainer_native.py) silently returned None on every rank of a
real multi-rank DDP job, and the time-series validation loader was never
built.

Real 2-rank gloo spawn (not a mock) — this is the exact mechanism that was
proven manually during review. `import torch` is slow on this filesystem
(measured 43s-228s depending on load), so this file is marked integration
and excluded from the default fast selector.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeseries]


def _worker(rank, world_size, result_queue):
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method="tcp://127.0.0.1:29511",
    )
    try:
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.config = "real-config-object"

        model = _Model()
        ddp_model = DDP(model)

        # Defect A: bare `getattr(model, "config", None)` on the DDP wrapper.
        wrapped_config = getattr(ddp_model, "config", None)
        # Fix: unwrap via `.module` first, matching trainer_native.py's
        # `unwrapped_model = model.module if hasattr(model, "module") else model`.
        unwrapped_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
        unwrapped_config = getattr(unwrapped_model, "config", None)

        result_queue.put((rank, wrapped_config, unwrapped_config))
    finally:
        dist.destroy_process_group()


def test_ddp_does_not_forward_config_attribute():
    """Proves the root cause: `model.config` is None on a real DDP-wrapped
    module on every rank, while `model.module.config` is intact."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    world_size = 2
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, result_queue))
        p.start()
        procs.append(p)

    results = {}
    for _ in range(world_size):
        rank, wrapped_config, unwrapped_config = result_queue.get(timeout=60)
        results[rank] = (wrapped_config, unwrapped_config)

    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker process failed with exitcode {p.exitcode}"

    for rank in range(world_size):
        wrapped_config, unwrapped_config = results[rank]
        assert wrapped_config is None, (
            f"rank {rank}: expected DDP to NOT forward .config (that's the "
            f"bug this test documents), got {wrapped_config!r}"
        )
        assert unwrapped_config == "real-config-object", (
            f"rank {rank}: unwrapping via .module must recover the real config"
        )
