"""Regression test for the n_shards < world_size sharding fallback.

HF datasets IterableDataset.shard() raises IndexError in
datasets/utils/sharding.py:_merge_gen_kwargs when the JSON loader has
fewer files than ranks. ts_qa (1 jsonl) and ts_instruction (3 jsonl)
both hit this with the standard 12-rank Aurora DDP config — every
per-modality sweep cell that uses local JSONL crashes at iter time.

The fix in src/data/multimodal.py wraps .shard() with a fallback to
.filter(with_indices=True) when n_shards < world_size. This test
exercises the fallback path against a real HF IterableDataset.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_shard_native_raises_on_few_files(tmp_path):
    """Pin the upstream bug: HF .shard() raises when n_shards < world_size."""
    pytest.importorskip("datasets")
    from datasets import load_dataset

    f = tmp_path / "single.jsonl"
    f.write_text("\n".join(json.dumps({"x": i}) for i in range(50)) + "\n")
    ds = load_dataset("json", data_files=[str(f)], split="train", streaming=True)
    assert ds.n_shards == 1

    with pytest.raises(IndexError):
        sharded = ds.shard(num_shards=12, index=5)
        next(iter(sharded))


def test_filter_fallback_yields_strided_samples(tmp_path):
    """The fallback path must yield approximately 1/world_size of samples."""
    pytest.importorskip("datasets")
    from datasets import load_dataset

    f = tmp_path / "single.jsonl"
    # 120 samples / world_size=12 = 10 per rank
    f.write_text("\n".join(json.dumps({"x": i}) for i in range(120)) + "\n")
    ds = load_dataset("json", data_files=[str(f)], split="train", streaming=True)

    rank, world_size = 5, 12
    sharded = ds.filter(
        lambda _ex, idx, _r=rank, _w=world_size: (idx % _w) == _r,
        with_indices=True,
    )
    seen = [s["x"] for s in sharded]
    # Every yielded sample's index should have idx % 12 == 5
    assert all(x % 12 == 5 for x in seen), f"Strided sampling broken: {seen}"
    # ~10 samples (one per stride hit)
    assert len(seen) == 10, f"Expected 10 strided samples, got {len(seen)}"


def test_filter_fallback_no_cross_rank_overlap(tmp_path):
    """Two different ranks must yield disjoint sample indices."""
    pytest.importorskip("datasets")
    from datasets import load_dataset

    f = tmp_path / "single.jsonl"
    f.write_text("\n".join(json.dumps({"x": i}) for i in range(60)) + "\n")
    world_size = 6
    seen_per_rank = []
    for rank in range(world_size):
        ds = load_dataset("json", data_files=[str(f)], split="train", streaming=True)
        sharded = ds.filter(
            lambda _ex, idx, _r=rank, _w=world_size: (idx % _w) == _r,
            with_indices=True,
        )
        seen_per_rank.append({s["x"] for s in sharded})

    # Sets must be disjoint
    for i in range(world_size):
        for j in range(i + 1, world_size):
            overlap = seen_per_rank[i] & seen_per_rank[j]
            assert not overlap, f"Rank {i} and {j} both got: {overlap}"
    # Union should cover everything
    union = set().union(*seen_per_rank)
    assert union == set(range(60)), f"Missed samples: {set(range(60)) - union}"
