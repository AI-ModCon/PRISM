#!/usr/bin/env python3
"""Validate WebDataset shards for data quality issues.

Checks for:
- Missing text files
- Empty text files
- Missing image files
- Corrupted images
- Optional per-key probes (pose, action) that np.load-decode

`--check` takes a comma-separated list of keys whose *probe* runs on every
sample (e.g. `--check pose,action,image,text`). For `pose`/`action` the
file must exist and numpy-load as a 1-D float array — exactly what
`ModalityAwareWebDatasetWrapper`'s decoders attempt at training time.
Catches shards where the sharder forgot `allow_pickle=False` or wrote the
wrong dtype.

Exit status is non-zero if any required key (text/image, plus anything in
`--check`) is missing or fails its probe — the sharder uses this as a gate.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from collections import defaultdict

try:
    from PIL import Image  # noqa: F401  # availability probe

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available, skipping image validation")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Extensions we treat as "image" when looking for the image payload.
_IMAGE_EXTS = ("jpg", "jpeg", "png", "gif", "webp")

# Multi-segment extensions we need to recognise so `head.jpg` and
# `pose.npy` parse as (key, ext) = ("000001", "head.jpg") instead of
# ("000001.head", "jpg"). Order matters: longest match wins.
_MULTI_EXTS = (
    "instruction.txt",
    "head.jpg",
    "wrist.jpg",
    "head.png",
    "wrist.png",
    "pose.npy",
    "action.npy",
    "ts.npy",
    "graph.pt",
    "mask.json",
    "meta.json",
)


def _split_key_ext(name: str) -> tuple[str, str]:
    """Split a tarinfo name into (sample_key, ext).

    Handles `000001.head.jpg` → ("000001", "head.jpg") for VLA shards while
    preserving the legacy `000001.jpg` → ("000001", "jpg") behaviour.
    """
    for multi in _MULTI_EXTS:
        suffix = "." + multi
        if name.endswith(suffix):
            return name[: -len(suffix)], multi
    if "." in name:
        key, ext = name.rsplit(".", 1)
        return key, ext
    return name, ""


def _probe_numpy_1d(content: bytes, label: str) -> str | None:
    """Return None on success, else an error string."""
    if not HAS_NUMPY:
        return f"{label}: numpy unavailable"
    try:
        arr = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        return f"{label}: np.load failed ({exc!r})"
    if arr.ndim != 1:
        return f"{label}: expected rank-1, got shape {tuple(arr.shape)}"
    if arr.dtype.kind not in ("f", "i", "u"):
        return f"{label}: unexpected dtype {arr.dtype}"
    return None


def validate_shard(
    shard_path: str,
    check_images: bool = False,
    extra_checks: tuple[str, ...] = (),
) -> dict:
    """Validate a single WebDataset shard.

    Returns dict with validation results.

    `extra_checks` is a tuple of extension names (e.g. "pose.npy",
    "action.npy") that must exist on every sample and pass a per-key probe.
    """
    results: dict = {
        "shard": os.path.basename(shard_path),
        "total_samples": 0,
        "missing_text": [],
        "empty_text": [],
        "missing_image": [],
        "corrupted_image": [],
        "valid_samples": 0,
        "extra_missing": defaultdict(list),
        "extra_failed": defaultdict(list),
    }

    # Group files by sample key
    samples: dict[str, dict] = defaultdict(dict)

    try:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                key, ext = _split_key_ext(member.name)
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()
                samples[key][ext] = {
                    "size": len(content),
                    "content": content,
                }
    except Exception as e:  # noqa: BLE001
        results["error"] = str(e)
        return results

    results["total_samples"] = len(samples)

    # If the caller passes --check explicitly, only enforce keys they asked
    # for (so a pose/action-only validation doesn't fail on a `text` shard's
    # lack of `jpg`). Default behaviour (no --check) keeps the original
    # text+image gate.
    if extra_checks:
        text_required = "text" in extra_checks
        image_required = "image" in extra_checks
    else:
        text_required = True
        image_required = True

    for key, files in samples.items():
        has_text = False
        has_image = False
        text_empty = False

        # Text: stored as `.txt` historically; `instruction.txt` for VLA.
        text_content = None
        if "txt" in files:
            has_text = True
            text_content = files["txt"]["content"]
        elif "instruction.txt" in files:
            has_text = True
            text_content = files["instruction.txt"]["content"]

        if has_text:
            if not text_content or not text_content.strip():
                text_empty = True
                results["empty_text"].append(key)
        elif text_required:
            results["missing_text"].append(key)

        # Image: any of the standard extensions, OR the VLA head/wrist pair.
        found_image = None
        for ext in _IMAGE_EXTS:
            if ext in files:
                found_image = ext
                break
        if found_image is None:
            for ext in ("head.jpg", "wrist.jpg", "head.png", "wrist.png"):
                if ext in files:
                    found_image = ext
                    break
        has_image = found_image is not None

        if not has_image and image_required:
            results["missing_image"].append(key)
        elif check_images and HAS_PIL and found_image:
            try:
                Image.open(io.BytesIO(files[found_image]["content"])).verify()
            except Exception:  # noqa: BLE001
                results["corrupted_image"].append(key)

        # Extra per-key probes (pose.npy, action.npy, …).
        for ext in extra_checks:
            if ext in ("text", "image"):
                continue
            entry = files.get(ext)
            if entry is None:
                results["extra_missing"][ext].append(key)
                continue
            if ext.endswith(".npy"):
                err = _probe_numpy_1d(entry["content"], ext)
                if err:
                    results["extra_failed"][ext].append((key, err))

        # A sample is "valid" if every requested gate passed.
        ok = True
        if text_required and (not has_text or text_empty):
            ok = False
        if image_required and not has_image:
            ok = False
        for ext in extra_checks:
            if ext in ("text", "image"):
                continue
            if key in results["extra_missing"].get(ext, []):
                ok = False
            if any(k == key for k, _ in results["extra_failed"].get(ext, [])):
                ok = False
        if ok:
            results["valid_samples"] += 1

    return results


def validate_dataset(
    dataset_path: str,
    max_shards: int | None = None,
    check_images: bool = False,
    extra_checks: tuple[str, ...] = (),
):
    """Validate all shards in a dataset directory."""

    if os.path.isdir(os.path.join(dataset_path, "shards")):
        shards_dir = os.path.join(dataset_path, "shards")
    else:
        shards_dir = dataset_path

    shard_files = sorted(f for f in os.listdir(shards_dir) if f.endswith(".tar"))
    if max_shards:
        shard_files = shard_files[:max_shards]

    print(f"Validating {len(shard_files)} shards from {shards_dir}")
    if extra_checks:
        print(f"Extra checks: {list(extra_checks)}")
    print("=" * 60)

    total_stats: dict = {
        "total_samples": 0,
        "valid_samples": 0,
        "missing_text": 0,
        "empty_text": 0,
        "missing_image": 0,
        "extra_missing": defaultdict(int),
        "extra_failed": defaultdict(int),
        "problem_shards": [],
    }

    for i, shard_file in enumerate(shard_files):
        shard_path = os.path.join(shards_dir, shard_file)
        results = validate_shard(shard_path, check_images, extra_checks)

        if "error" in results:
            print(f"[{i + 1}/{len(shard_files)}] {shard_file}: ERROR {results['error']}")
            total_stats["problem_shards"].append(
                {"shard": shard_file, "problems": [f"open_error={results['error']}"]}
            )
            continue

        total_stats["total_samples"] += results["total_samples"]
        total_stats["valid_samples"] += results["valid_samples"]
        total_stats["missing_text"] += len(results["missing_text"])
        total_stats["empty_text"] += len(results["empty_text"])
        total_stats["missing_image"] += len(results["missing_image"])
        for ext, keys in results["extra_missing"].items():
            total_stats["extra_missing"][ext] += len(keys)
        for ext, items in results["extra_failed"].items():
            total_stats["extra_failed"][ext] += len(items)

        problems = []
        if results["missing_text"]:
            problems.append(f"missing_text={len(results['missing_text'])}")
        if results["empty_text"]:
            problems.append(f"empty_text={len(results['empty_text'])}")
        if results["missing_image"]:
            problems.append(f"missing_image={len(results['missing_image'])}")
        for ext, keys in results["extra_missing"].items():
            if keys:
                problems.append(f"missing_{ext}={len(keys)}")
        for ext, items in results["extra_failed"].items():
            if items:
                problems.append(f"failed_{ext}={len(items)}")

        if problems:
            total_stats["problem_shards"].append(
                {"shard": shard_file, "problems": problems}
            )
            print(
                f"[{i + 1}/{len(shard_files)}] {shard_file}: "
                f"{results['total_samples']} samples, PROBLEMS: {', '.join(problems)}"
            )
        elif (i + 1) % 50 == 0:
            print(
                f"[{i + 1}/{len(shard_files)}] {shard_file}: "
                f"{results['total_samples']} samples - OK"
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:  {total_stats['total_samples']}")
    print(f"Valid samples:  {total_stats['valid_samples']}")
    print(f"Missing text:   {total_stats['missing_text']}")
    print(f"Empty text:     {total_stats['empty_text']}")
    print(f"Missing image:  {total_stats['missing_image']}")
    for ext, count in total_stats["extra_missing"].items():
        print(f"Missing {ext}: {count}")
    for ext, count in total_stats["extra_failed"].items():
        print(f"Failed  {ext}: {count}")
    print(f"Problem shards: {len(total_stats['problem_shards'])}")

    return total_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate WebDataset shards")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument(
        "--max-shards", type=int, default=None, help="Max shards to check"
    )
    parser.add_argument(
        "--check-images", action="store_true", help="Validate image files (PIL.verify)"
    )
    parser.add_argument(
        "--check",
        default="",
        help=(
            "Comma-separated list of per-sample keys to require and probe. "
            "Recognised: text, image, pose, action, plus arbitrary `<name>.npy` "
            "(probed as numpy float). Example: --check pose,action,image,text"
        ),
    )
    args = parser.parse_args(argv)

    raw_checks = [c.strip() for c in args.check.split(",") if c.strip()]
    expanded: list[str] = []
    for c in raw_checks:
        if c == "pose":
            expanded.append("pose.npy")
        elif c == "action":
            expanded.append("action.npy")
        else:
            expanded.append(c)
    extra_checks = tuple(expanded)

    stats = validate_dataset(
        args.dataset_path, args.max_shards, args.check_images, extra_checks
    )

    failed = (
        stats["missing_text"] > 0
        or stats["empty_text"] > 0
        or stats["missing_image"] > 0
        or any(v > 0 for v in stats["extra_missing"].values())
        or any(v > 0 for v in stats["extra_failed"].values())
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
