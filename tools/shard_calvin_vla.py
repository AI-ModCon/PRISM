#!/usr/bin/env python3
"""Shard CALVIN VLA samples into WebDataset format.

Reads the LeRobot-format CALVIN dataset (parquet per episode + jsonl meta)
that `src/data/calvin_vla.py` consumes, and writes WebDataset shards that
`ModalityAwareWebDatasetWrapper(modalities=["vla"])` can load.

Per-sample keys produced:
    <basename>.head.jpg          observation.images.top  PNG → JPEG
    <basename>.wrist.jpg         observation.images.wrist PNG → JPEG
    <basename>.pose.npy          observation.state (float32 rank-1)
    <basename>.action.npy        next-step action      (float32 rank-1)
    <basename>.instruction.txt   raw NL instruction (UTF-8, no tokenization)
    <basename>.meta.json         {episode, step, task_index}

Sharding policy (Markov-safe):
    - One shard contains *whole episodes only*; we never split an episode
      across shards. This lets the loader set `shardshuffle=False` and
      `shuffle=0` and still keep (obs_t, action_{t+1}) pairs intact.
    - Within an episode, samples are written in frame order so the loader's
      default sequential read preserves Markov order.

Login-node tool. No GPU, no MPI, no Aurora dependencies.

Example:
    python tools/shard_calvin_vla.py \\
        --root /flare/ModCon/sww/vla_training/calvin_dataset \\
        --split validation --max-episodes 10 \\
        --out /tmp/calvin_test
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("shard_calvin_vla")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# CALVIN reader — mirrors src/data/calvin_vla.py without depending on torch
# or the tokenizer. Same `max_chunk <= 20` guard, same parquet columns.
# ---------------------------------------------------------------------------

# Same cap as CalvinVLADataset (src/data/calvin_vla.py:36-37,156). Episode
# index → chunk is `ep // 1000`; we read at most 21 chunks (0..20) to stay
# inside the validated data subset.
MAX_CHUNK = 20


def _episode_path(root: Path, episode_idx: int) -> Path:
    chunk = episode_idx // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"


def _load_meta(root: Path, split: str) -> tuple[range, dict[int, str], dict[int, str], dict[int, int]]:
    info = json.loads((root / "meta" / "info.json").read_text())
    splits = info["splits"]
    if split not in splits:
        raise SystemExit(
            f"Unknown split '{split}'. Available: {sorted(splits.keys())}"
        )
    start_s, end_s = splits[split].split(":")
    split_range = range(int(start_s), int(end_s))

    task_by_index: dict[int, str] = {}
    with open(root / "meta" / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            task_by_index[int(row["task_index"])] = row["task"]

    episode_task_fallback: dict[int, str] = {}
    episode_length: dict[int, int] = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            ep_idx = int(row["episode_index"])
            episode_length[ep_idx] = int(row["length"])
            tasks = row.get("tasks", [])
            if tasks:
                episode_task_fallback[ep_idx] = tasks[0]

    return split_range, task_by_index, episode_task_fallback, episode_length


def _iter_calvin_episodes(
    root: Path,
    split: str,
    max_episodes: int | None,
) -> Iterator[tuple[int, pd.DataFrame, str]]:
    """Yield (episode_idx, dataframe, instruction_text) per usable episode."""
    import pandas as pd

    split_range, task_by_index, episode_task_fallback, episode_length = _load_meta(
        root, split
    )

    yielded = 0
    for episode_idx in split_range:
        if episode_idx // 1000 > MAX_CHUNK:
            continue
        path = _episode_path(root, episode_idx)
        if not path.exists():
            continue
        length = int(episode_length.get(episode_idx, 0))
        if length < 2:
            continue
        df = pd.read_parquet(
            path,
            columns=[
                "action",
                "observation.state",
                "observation.images.top",
                "observation.images.wrist",
                "task_index",
                "frame_index",
                "episode_index",
            ],
        )
        if len(df) < 2:
            continue
        task_idx = int(df["task_index"].iloc[0])
        text = task_by_index.get(task_idx) or episode_task_fallback.get(episode_idx)
        if not text:
            # Same failure mode as CalvinVLADataset.__getitem__: better to skip
            # than to ship a sample we know will crash at training time.
            logger.warning(
                f"episode={episode_idx} missing instruction (task_index={task_idx}); skipping"
            )
            continue
        yield episode_idx, df, text
        yielded += 1
        if max_episodes is not None and yielded >= max_episodes:
            return


def _re_encode_jpeg(image_entry: dict, quality: int) -> bytes:
    """CALVIN parquet stores PNG bytes; transcode to JPEG to shrink shards."""
    from PIL import Image

    if not isinstance(image_entry, dict) or "bytes" not in image_entry:
        raise RuntimeError("Expected image entry with embedded bytes")
    img = Image.open(io.BytesIO(image_entry["bytes"])).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _iter_calvin_samples(
    root: Path,
    split: str,
    max_episodes: int | None,
    jpeg_quality: int,
) -> Iterator[tuple[str, dict, int]]:
    """Yield (basename, sample_dict, episode_idx) in episode-then-frame order."""
    import numpy as np

    for episode_idx, df, text in _iter_calvin_episodes(root, split, max_episodes):
        for step_idx in range(len(df) - 1):
            curr = df.iloc[step_idx]
            nxt = df.iloc[step_idx + 1]

            pose = np.asarray(curr["observation.state"], dtype="float32").reshape(-1)
            action = np.asarray(nxt["action"], dtype="float32").reshape(-1)
            if pose.ndim != 1 or action.ndim != 1:
                # Same invariant as CalvinVLADataset.__getitem__; surface early.
                raise RuntimeError(
                    f"episode={episode_idx} step={step_idx}: pose/action must be rank-1"
                )

            pose_buf = io.BytesIO()
            np.save(pose_buf, pose, allow_pickle=False)
            action_buf = io.BytesIO()
            np.save(action_buf, action, allow_pickle=False)

            head_bytes = _re_encode_jpeg(curr["observation.images.top"], jpeg_quality)
            wrist_bytes = _re_encode_jpeg(curr["observation.images.wrist"], jpeg_quality)

            meta = {
                "episode": int(episode_idx),
                "step": int(step_idx),
                "task_index": int(curr["task_index"]),
            }

            basename = f"ep{episode_idx:06d}_s{step_idx:04d}"
            sample = {
                "__key__": basename,
                "head.jpg": head_bytes,
                "wrist.jpg": wrist_bytes,
                "pose.npy": pose_buf.getvalue(),
                "action.npy": action_buf.getvalue(),
                "instruction.txt": text,
                "meta.json": json.dumps(meta),
            }
            yield basename, sample, episode_idx


# ---------------------------------------------------------------------------
# Shard writer — emits one shard per episode-group, never splitting episodes.
# ---------------------------------------------------------------------------


def write_shards(
    samples: Iterator[tuple[str, dict, int]],
    out_dir: Path,
    shard_maxbytes: int,
) -> dict:
    """Write WebDataset shards where each shard contains whole episodes only.

    We bypass `wds.ShardWriter`'s implicit episode-splitting by managing
    `tarfile` directly: we open a new shard, write all samples of the
    current episode, and only roll over to a new shard at episode
    boundaries once the current shard has exceeded `shard_maxbytes`.

    Output layout matches what `MultiWebDataset` expects:
        <out_dir>/manifest.json
        <out_dir>/shards/calvin-NNNNNN.tar
    """
    import tarfile

    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    sample_count = 0
    current_tar = None
    current_path = None
    current_size = 0
    current_episode = None
    shard_files: list[str] = []

    def _open_new_shard() -> tuple[tarfile.TarFile, Path]:
        nonlocal shard_index
        path = shards_dir / f"calvin-{shard_index:06d}.tar"
        shard_index += 1
        return tarfile.open(path, "w"), path

    def _add_file(tar: tarfile.TarFile, key: str, payload: bytes) -> int:
        info = tarfile.TarInfo(name=key)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        return len(payload) + 512  # rough overhead per tar record

    try:
        for basename, sample, episode_idx in samples:
            if current_tar is None:
                current_tar, current_path = _open_new_shard()
                current_episode = episode_idx
                current_size = 0
            elif (
                episode_idx != current_episode and current_size >= shard_maxbytes
            ):
                current_tar.close()
                shard_files.append(current_path.name)
                logger.info(
                    f"closed {current_path.name} ({current_size / 1e6:.1f} MB)"
                )
                current_tar, current_path = _open_new_shard()
                current_episode = episode_idx
                current_size = 0
            elif episode_idx != current_episode:
                current_episode = episode_idx

            for key, value in sample.items():
                if key == "__key__":
                    continue
                payload = value.encode("utf-8") if isinstance(value, str) else value
                current_size += _add_file(
                    current_tar, f"{basename}.{key}", payload
                )

            sample_count += 1
            if sample_count % 1000 == 0:
                logger.info(f"  wrote {sample_count} samples …")
    finally:
        if current_tar is not None:
            current_tar.close()
            shard_files.append(current_path.name)
            logger.info(
                f"closed {current_path.name} ({current_size / 1e6:.1f} MB)"
            )

    # Manifest format matches `MultiWebDataset._get_shards_for_dataset` fast path:
    # `shards` is a list of {name: ...} dicts so the loader can construct paths
    # without globbing.
    manifest = {
        "dataset": "calvin-vla",
        "modality": "vla",
        "total_samples": sample_count,
        "num_train_shards": len(shard_files),
        "num_shards": len(shard_files),
        "shards": [
            {"name": name, "samples": -1, "size_bytes": 0, "start_idx": -1}
            for name in shard_files
        ],
        # Episode-aligned sharding ⇒ loader MUST disable shuffle to keep
        # (obs_t, action_{t+1}) pairs intact.
        "shardshuffle": False,
        "sample_shuffle": 0,
        "sample_keys": [
            "head.jpg",
            "wrist.jpg",
            "pose.npy",
            "action.npy",
            "instruction.txt",
            "meta.json",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"wrote {sample_count} samples → {len(shard_files)} shards in {out_dir}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--root",
        default="/flare/ModCon/sww/vla_training/calvin_dataset",
        help="LeRobot-format CALVIN root (contains meta/ and data/).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name as defined in meta/info.json (train, validation, ...)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for shards + shards.json",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap on number of episodes to shard (for smoke runs).",
    )
    parser.add_argument(
        "--shard-mb", type=int, default=384,
        help="Approx shard size in MB; shards roll over only at episode boundaries.",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=92,
        help="JPEG quality for the top/wrist cameras (default 92).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: --root does not exist: {root}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    samples = _iter_calvin_samples(
        root, args.split, args.max_episodes, args.jpeg_quality
    )
    write_shards(samples, out_dir, args.shard_mb * 1024 * 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
