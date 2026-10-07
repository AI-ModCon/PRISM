"""Regression test for defect B: the rank-symmetric validation-loader
readiness gate in trainer_native.py.

Before the fix, a per-rank try/except around validation-loader construction
could leave `val_loader` built on some ranks and None on others (e.g. one
rank's shard read hit a transient failure). The held-out-validation eval
block later enters a `dist.all_reduce()` on every rank where `val_loader is
not None` — if even one rank skips that collective while others enter it,
the job deadlocks.

The fix all-reduces a single readiness flag once, right after the per-rank
try/except, so every rank agrees on whether validation is enabled before
any of them reach the eval-forward's own collective.

Real 2-rank gloo spawn (not a mock) — proves the actual collective doesn't
hang and produces the correct decision. `import torch` is slow on this
filesystem, so this file is marked integration.

The flag tensor is an **int64** counter SUM-reduced and compared against
world_size. The dtype is load-bearing, not stylistic.

Every earlier version of this gate (int32 MIN, fp32 MIN, fp32 SUM, fp32 SUM
plus an explicit barrier) produced false negatives on real Aurora XPU runs:
every rank logged its validation loader as ready, yet the gate still
disabled validation on fully healthy jobs (8788586 1N, 8788736 2N, 8789128
2N, 8789134 2N, 8817030 1N).

Root-caused 2026-09-10 on a held debug allocation by instrumenting the live
call site with a dtype x element-count probe matrix on 12 ranks: oneCCL's
ring allreduce silently drops the LAST rank when the buffer is a single
fp32 element. fp32 numel=1 gave 11.0 on ranks 0..10 and 1.0 on rank 11;
fp32 numel>=2, int64, fp64 and bf16 all reduced correctly to 12. Every
failed mitigation had kept a numel=1 fp32/int32 payload, which is why
changing the op or adding a barrier never helped.

gloo on CPU cannot reproduce an oneCCL defect, so these spawn tests only
prove the gate's decision logic. `test_gate_uses_int64_not_fp32_scalar`
below is the actual regression guard for the dtype.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeseries]


def _worker(rank, world_size, rank0_has_loader, rank1_has_loader, result_queue):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method="tcp://127.0.0.1:29512",
    )
    try:
        # Simulate the per-rank outcome of the try/except val-loader init
        # block: some ranks may fail to build a loader while others succeed.
        has_loader = rank0_has_loader if rank == 0 else rank1_has_loader
        val_loader = object() if has_loader else None

        # Mirrors trainer_native.py's rank-symmetric readiness gate.
        device = torch.device("cpu")
        _val_ready = torch.tensor(
            [1 if val_loader is not None else 0], dtype=torch.int64, device=device
        )
        dist.all_reduce(_val_ready, op=dist.ReduceOp.SUM)
        _all_ranks_ready = int(_val_ready.item()) >= world_size
        if not _all_ranks_ready and val_loader is not None:
            val_loader = None

        result_queue.put((rank, val_loader is not None))
    finally:
        dist.destroy_process_group()


def _run_gate(rank0_has_loader: bool, rank1_has_loader: bool):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    world_size = 2
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(rank, world_size, rank0_has_loader, rank1_has_loader, result_queue),
        )
        p.start()
        procs.append(p)

    results = {}
    for _ in range(world_size):
        rank, has_loader_after_gate = result_queue.get(timeout=60)
        results[rank] = has_loader_after_gate

    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker process failed with exitcode {p.exitcode}"

    return results


def test_both_ranks_ready_keeps_validation_enabled():
    results = _run_gate(rank0_has_loader=True, rank1_has_loader=True)
    assert results == {0: True, 1: True}


def test_one_rank_missing_loader_disables_validation_on_both_ranks():
    """The exact defect B scenario: rank 1's val-loader init failed (e.g. a
    transient shard read error) while rank 0's succeeded. Without the gate,
    rank 0 would later enter the eval-forward's all_reduce while rank 1
    never does -> deadlock. With the gate, both ranks agree validation is
    OFF for this run."""
    results = _run_gate(rank0_has_loader=True, rank1_has_loader=False)
    assert results == {0: False, 1: False}


def test_neither_rank_ready_stays_disabled():
    results = _run_gate(rank0_has_loader=False, rank1_has_loader=False)
    assert results == {0: False, 1: False}


def test_gate_uses_int64_not_fp32_scalar():
    """Source-level guard for the oneCCL numel=1 fp32 defect.

    gloo on CPU reduces a single fp32 element correctly, so the spawn tests
    above pass either way — they cannot catch a regression to fp32 here.
    Only a real XPU run can, and that is exactly how this shipped broken
    five times. So assert on the source instead: the readiness gate must
    build an int64 tensor.

    If you are here because this test failed after you changed the gate:
    do not just update the assertion. Re-read the module docstring — a
    numel=1 fp32 all_reduce silently drops the last rank on Aurora, and the
    failure mode is silent (validation turns itself off, training looks
    perfectly healthy, exit code 0).
    """
    import pathlib
    import re

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "training"
        / "trainer_native.py"
    ).read_text()

    gate = re.search(
        r"_val_ready = torch\.tensor\((.*?)\)\s*\n\s*dist\.all_reduce\(\s*_val_ready",
        src,
        re.DOTALL,
    )
    assert gate is not None, "readiness gate all_reduce not found in trainer_native.py"
    body = gate.group(1)
    assert "torch.int64" in body, (
        "readiness gate must reduce an int64 counter; found:\n" + body
    )
    assert "float32" not in body, (
        "readiness gate reverted to fp32 — oneCCL drops the last rank on a "
        "numel=1 fp32 buffer (see module docstring):\n" + body
    )


def test_cross_rank_val_averaging_uses_fp64_pair():
    """The eval-loss reduction hit the same oneCCL defect (it segfaulted 4/4
    runs and was disabled entirely). It now reduces a 2-element fp64
    [loss_sum, batch_count] buffer, which is both >1 element and not fp32.
    """
    import pathlib
    import re

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "training"
        / "trainer_native.py"
    ).read_text()

    red = re.search(
        r"_val_stats = torch\.tensor\((.*?)\)\s*\n\s*dist\.all_reduce\(\s*_val_stats",
        src,
        re.DOTALL,
    )
    assert red is not None, "cross-rank validation reduction not found"
    body = red.group(1)
    assert "torch.float64" in body, "val-loss reduction must be fp64:\n" + body
    assert "val_batch_count" in body, (
        "loss_sum and batch_count must go in ONE buffer so the reduction is "
        "numel=2 and the mean stays sample-weighted:\n" + body
    )
