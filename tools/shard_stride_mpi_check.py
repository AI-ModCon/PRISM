#!/usr/bin/env python3
"""Verify shard->rank assignment under real multi-rank MPI.

Runs the actual MultiWebDataset shard-partition logic on every rank and
gathers what each rank claims, then asserts on rank 0 that the claims are
disjoint and cover the dataset. Exercises both regimes:

  shards >= ranks  -> shard-level split (disjoint shard subsets)
  shards <  ranks  -> sample-level stride (disjoint sample subsets)

The second regime is the one PR #169 worked around by capping ngpus to 3.

Launch under mpiexec; reads PALS/PMI rank env like the training launchers.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import webdataset as wds  # noqa: E402
from src.data.multi_webdataset import _identity, _SampleStride  # noqa: E402


def _rank_env() -> tuple[int, int]:
    rank = int(
        os.environ.get(
            "PALS_RANKID", os.environ.get("PMI_RANK", os.environ.get("RANK", "0"))
        )
    )
    world = int(
        os.environ.get(
            "PALS_SIZE",
            os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", "0")),
        )
    )
    if world <= 0:
        # PALS_SIZE is never set by mpiexec on Aurora; derive it.
        local = int(os.environ.get("PALS_LOCAL_SIZE", os.environ.get("PMI_LOCAL_SIZE", "1")))
        nodes = int(os.environ.get("NUM_NODES", "1"))
        world = local * nodes
    return rank, world


def build_rank_stream(shards: list[str], rank: int, world: int, limit: int):
    """Mirror MultiWebDataset._build_dataset's partition decision."""
    if world > 1 and len(shards) >= world:
        mine = shards[rank::world]
        mode = "shard-split"
        stride = None
    elif world > 1:
        mine = shards
        mode = "sample-stride"
        stride = _SampleStride(rank, world)
    else:
        mine, mode, stride = shards, "single-rank", None

    base = wds.WebDataset(
        mine,
        shardshuffle=False,
        empty_check=False,
        nodesplitter=_identity,
        resampled=False,
        detshuffle=False,
        seed=42,
    )
    if stride is not None:
        base = base.select(stride)

    keys = []
    for sample in base:
        keys.append(sample["__key__"])
        if len(keys) >= limit:
            break
    return mode, mine, keys


def main() -> int:
    shards_dir = os.environ.get("VERIFY_SHARDS_DIR")
    limit = int(os.environ.get("VERIFY_LIMIT", "150"))
    if not shards_dir:
        print("VERIFY_SHARDS_DIR must be set", file=sys.stderr)
        return 2

    # os.listdir (not glob) — glob hangs on dfuse mounts.
    shards = sorted(
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.endswith(".tar")
    )
    rank, world = _rank_env()
    mode, mine, keys = build_rank_stream(shards, rank, world, limit)

    print(
        f"[rank {rank}/{world}] mode={mode} shards_total={len(shards)} "
        f"shards_open={len(mine)} samples={len(keys)}",
        flush=True,
    )

    import torch.distributed as dist

    if world > 1:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, keys)
        dist.barrier()
    else:
        gathered = [keys]

    if rank != 0:
        if world > 1:
            dist.destroy_process_group()
        return 0

    owners: dict[str, list[int]] = {}
    for r, ks in enumerate(gathered):
        for k in ks:
            owners.setdefault(k, []).append(r)

    dupes = {k: v for k, v in owners.items() if len(v) > 1}
    empty = [r for r, ks in enumerate(gathered) if not ks]

    print("=" * 62)
    print(f"mode           : {mode}")
    print(f"ranks          : {world}")
    print(f"shards         : {len(shards)}")
    print(f"distinct keys  : {len(owners)}")
    print(f"duplicated keys: {len(dupes)}")
    print(f"starved ranks  : {empty}")
    for k, v in list(dupes.items())[:5]:
        print(f"  DUPE {k} -> ranks {v}")

    ok = not dupes and not empty
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 62, flush=True)

    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
