"""Regression tests for WebDataset sharding when num_shards < world_size.

Splitting a shard list `shards[rank::world_size]` only produces a disjoint
cover when there are at least as many shards as ranks. Below that, some ranks
get an empty list. The previous behaviour handed every starved rank
`shards[:1]` — the SAME shard — so those ranks trained on identical samples
while the rest of the dataset went unseen, with only a log warning.

The fix keeps the full shard list on starved configurations and strides at the
sample level instead (`keep i where i % world_size == rank`), mirroring the
`_StridedTSValidation` approach already used for the 3-shard SciTS validation
split and the HF-stream fallback in `test_shard_fewfiles_fallback.py`.

This is what lets the 27-shard SciTS dataset train on 12 ranks/node without
capping `ngpus` down to the shard count.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multi_webdataset import _SampleStride  # noqa: E402

pytestmark = pytest.mark.unit


def test_sample_stride_partitions_disjointly():
    """Across all ranks, every sample is claimed exactly once."""
    world_size, n_samples = 12, 240
    claimed = {}
    for rank in range(world_size):
        pred = _SampleStride(rank, world_size)
        for i in range(n_samples):
            if pred(f"sample-{i}"):
                assert i not in claimed, (
                    f"sample {i} claimed by rank {claimed.get(i)} and {rank}"
                )
                claimed[i] = rank

    assert len(claimed) == n_samples, (
        f"{n_samples - len(claimed)} samples went unclaimed by any rank"
    )
    # Even split: 240 / 12 = 20 each.
    for rank in range(world_size):
        assert sum(r == rank for r in claimed.values()) == 20


def test_sample_stride_matches_expected_indices():
    pred = _SampleStride(offset=5, stride=12)
    kept = [i for i in range(120) if pred(object())]
    assert kept == list(range(5, 120, 12))
    assert all(i % 12 == 5 for i in kept)


def test_sample_stride_is_picklable():
    """DataLoader with multiprocessing_context='spawn' pickles the pipeline."""
    import pickle

    pred = _SampleStride(3, 12)
    for _ in range(7):
        pred(object())

    revived = pickle.loads(pickle.dumps(pred))
    assert isinstance(revived, _SampleStride)
    assert (revived.offset, revived.stride) == (3, 12)


def test_sample_stride_counter_is_per_instance():
    """Two ranks' predicates must not share a counter."""
    a, b = _SampleStride(0, 3), _SampleStride(0, 3)
    for _ in range(3):
        a(object())
    # `b` is untouched, so its first sample is still index 0 -> kept.
    assert b(object()) is True


@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
def test_stride_stays_disjoint_across_dataloader_workers(tmp_path, num_workers):
    """num_workers > 1 must not break the per-rank disjointness.

    webdataset splits the shard list across workers, and each worker gets its
    own ``_SampleStride`` with its own counter. This pins that the combination
    still yields a disjoint cover rather than duplicating or dropping samples.
    """
    wds = pytest.importorskip("webdataset")
    import tarfile

    import torch
    from src.data.multi_webdataset import _identity

    # Fewer shards than ranks is the regime under test: 3 shards, 6 ranks.
    n_shards, world_size, per_shard = 3, 6, 60
    shard_paths = []
    for s in range(n_shards):
        p = tmp_path / f"shard-{s:04d}.tar"
        with tarfile.open(p, "w") as tf:
            for j in range(per_shard):
                payload = b"x"
                info = tarfile.TarInfo(name=f"s{s:04d}_{j:04d}.txt")
                info.size = len(payload)
                tf.addfile(info, __import__("io").BytesIO(payload))
        shard_paths.append(str(p))

    owners: dict[str, list[int]] = {}
    for rank in range(world_size):
        ds = (
            wds.WebDataset(
                shard_paths,
                shardshuffle=False,
                empty_check=False,
                nodesplitter=_identity,
                resampled=False,
                detshuffle=False,
                seed=42,
            )
            .select(_SampleStride(rank, world_size))
            .to_tuple("__key__", handler=wds.warn_and_continue)
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers
        )
        for (key,) in loader:
            owners.setdefault(key, []).append(rank)

    cross = {k: v for k, v in owners.items() if len(set(v)) > 1}
    within = {k: v for k, v in owners.items() if len(v) != len(set(v))}
    assert not cross, f"sample claimed by multiple ranks: {list(cross.items())[:3]}"
    assert not within, f"sample claimed twice by one rank: {list(within.items())[:3]}"
    assert len(owners) == n_shards * per_shard, (
        f"expected full coverage of {n_shards * per_shard} samples, got {len(owners)}"
    )


def test_every_rank_gets_shards_when_shards_fewer_than_ranks():
    """The starved-rank path must not hand out a duplicate single shard.

    Reproduces the SciTS shape: 3 validation shards across 12 ranks.
    """
    shards = [f"scits-{i:06d}.tar" for i in range(3)]
    world_size = 12

    # Old behaviour: shards[rank::world_size] then shards[:1] on starvation.
    old = []
    for rank in range(world_size):
        got = shards[rank::world_size]
        old.append(got if got else shards[:1])
    starved = [r for r in range(world_size) if not shards[r::world_size]]
    assert starved, "precondition: some ranks must be starved at 3 shards/12 ranks"
    # Every starved rank got the identical shard — the bug.
    assert all(old[r] == [shards[0]] for r in starved)

    # New behaviour: all ranks read every shard, disjointness comes from the
    # sample-level stride, so no rank is handed a duplicate subset.
    preds = [_SampleStride(r, world_size) for r in range(world_size)]
    n_samples = 120
    owners = [
        [r for r in range(world_size) if preds[r](i)] for i in range(n_samples)
    ]
    assert all(len(o) == 1 for o in owners), "each sample must have exactly one owner"
