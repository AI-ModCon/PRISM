"""Smoke test for SciTS WebDataset shards.

This test is intentionally lightweight and opt-in by path convention.
It loads the first shard from a local SciTS conversion output, prints a
short summary, and validates that the tarball and its payloads are not
corrupted.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]

def _find_shard_dir() -> Path:
    candidates = [
        os.environ.get("SCITS_SHARD_DIR"),
        os.environ.get("SCITS_WEBDATASET_DIR"),
        "/tmp/smoke_scits/shards",
        "/tmp/smoke_scits",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and path.suffix == ".tar":
            return path.parent
        if path.is_dir():
            return path

    pytest.skip("No SciTS shard directory found; set SCITS_SHARD_DIR to run this smoke test")


def _first_shard_path(shard_dir: Path) -> Path:
    tar_candidates = sorted(shard_dir.glob("*.tar"))
    if tar_candidates:
        return tar_candidates[0]

    shards_subdir = shard_dir / "shards"
    tar_candidates = sorted(shards_subdir.glob("*.tar"))
    if tar_candidates:
        return tar_candidates[0]

    pytest.skip(f"No .tar shards found under {shard_dir}")


def _sample_key_and_suffix(member_name: str) -> tuple[str, str] | None:
    if member_name.endswith(".ts.npy"):
        return member_name[: -len(".ts.npy")], "ts"
    if member_name.endswith(".text"):
        return member_name[: -len(".text")], "text"
    if member_name.endswith(".meta.json"):
        return member_name[: -len(".meta.json")], "meta"
    return None


def test_scits_shard_smoke_load_and_validate():
    shard_dir = _find_shard_dir()
    shard_path = _first_shard_path(shard_dir)

    assert tarfile.is_tarfile(shard_path), f"Not a valid tar archive: {shard_path}"

    sample_groups: dict[str, set[str]] = {}
    sample_meta: dict[str, dict] = {}
    sample_count = 0

    with tarfile.open(shard_path, "r") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        assert members, f"Shard has no files: {shard_path}"

        for it, member in enumerate(members):
            parsed = _sample_key_and_suffix(member.name)
            assert parsed is not None, f"Unexpected shard member name: {member.name}"
            prefix, suffix = parsed
            sample_groups.setdefault(prefix, set()).add(suffix)

            extracted = tar.extractfile(member)
            assert extracted is not None, f"Could not extract {member.name} from {shard_path}"
            payload = extracted.read()
            assert payload, f"Empty payload for {member.name} in {shard_path}"

            if it <= 10:
                print(f"Loaded {member.name} (size={len(payload)} bytes)", flush=True)

            if suffix == "ts":
                arr = np.load(io.BytesIO(payload), allow_pickle=False)
                assert arr.size > 0, f"Empty array in {member.name}"
                if it <= 10:
                    print(arr, arr.shape, arr.dtype, flush=True)
            elif suffix == "meta":
                sample_meta[prefix] = json.loads(payload.decode("utf-8"))
                if it <=10:
                    print(sample_meta[prefix], flush=True)
            elif suffix == "text":
                assert payload.decode("utf-8").strip(), f"Empty text in {member.name}"
                if it <=10:
                    print(payload.decode("utf-8"), flush=True)
            
            
        
        for prefix, suffixes in sample_groups.items():
            assert {"ts", "text", "meta"}.issubset(suffixes), (
                f"Incomplete sample {prefix} in {shard_path}: {sorted(suffixes)}"
            )
            #print(f"Validated sample {prefix} with suffixes {sorted(suffixes)}", flush=True)
            sample_count += 1

    assert sample_meta, f"No metadata entries were loaded from {shard_path}"
    first_meta = next(iter(sample_meta.values()), None)
    print(
        "SciTS shard smoke: "
        f"shard={shard_path.name} samples={sample_count} metadata={len(sample_meta)} "
        f"first_meta_keys={sorted(first_meta) if first_meta else []}",
        flush=True,
    )
    assert sample_count > 0
