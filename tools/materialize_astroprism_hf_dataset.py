#!/usr/bin/env python3
"""Materialize the published AstroPRISM Hugging Face dataset for training.

The training configs under ``src/conf/data/astroprism_*.yaml`` read a local
tree of manifests (``artifact_type: astroprism_raw_images`` and friends) whose
``outputs`` entries point at the ``.npy`` arrays and candidate JSONL beside
them. The Hugging Face package ships the same payloads under a different
envelope: its root ``MANIFEST.json`` is an upload index
(``artifact_type: astroprism_huggingface_dataset``) and ``metadata/
candidates.jsonl`` is the candidate file itself rather than a manifest naming
one. Nothing in PRISM resolves the ``hf://`` paths the original run used.

This script bridges the two. It downloads (or reuses) the dataset snapshot and
writes the manifest envelope the loaders expect, leaving the large arrays in
the Hugging Face cache and referencing them by absolute path, so nothing is
copied twice.

    export HF_TOKEN=...          # the dataset is gated
    python tools/materialize_astroprism_hf_dataset.py --output-dir _local/astroprism

Then train with the shipped configs, which read that tree::

    python tools/launch_aurora.py --file experiments/astroprism_stage1a.yaml ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_FILES = [
    "MANIFEST.json",
    "metadata/candidates.jsonl",
    "assets/raw/images/MANIFEST.json",
    "assets/raw/spectra/MANIFEST.json",
]
ARRAY_PATTERNS = ["assets/raw/images/*.npy", "assets/raw/spectra/*.npy"]

REPO_ID = "Sand33p/AstroPRISM"


class MaterializeError(RuntimeError):
    """Raised when the snapshot does not match what the loaders require."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializeError(f"cannot read {path}: {exc}") from exc


def _absolutize(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Rewrite every ``outputs[*].path`` to an absolute path in the snapshot.

    The loaders resolve a relative output path against the manifest's own
    directory. Writing the manifest elsewhere would break that, so the paths
    are pinned to where the arrays actually live.
    """
    out = json.loads(json.dumps(manifest))  # deep copy
    outputs = out.get("outputs")
    if not isinstance(outputs, dict):
        raise MaterializeError(f"{manifest_path} has no outputs mapping")
    for name, entry in outputs.items():
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        resolved = (manifest_path.parent / entry["path"]).resolve()
        if not resolved.is_file():
            raise MaterializeError(
                f"{manifest_path.name} names {name} at {resolved}, which does not exist. "
                "Re-run without --no-download, or pass --snapshot-dir for a complete copy."
            )
        entry["path"] = str(resolved)
    return out


def _raw_candidate_rows(snapshot: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the per-candidate list the raw-image loader cross-checks.

    The HF manifest records band order once, as a mapping, plus two parallel
    ``.npy`` arrays holding the per-row source and AION band labels. The loader
    wants one object per candidate carrying both, so read the arrays and pair
    them with the roster rows, preserving index order.
    """
    import numpy as np

    images = snapshot / "assets/raw/images"
    try:
        source_bands = np.load(images / "image_band.npy", allow_pickle=False)
        aion_bands = np.load(images / "aion_image_band.npy", allow_pickle=False)
    except OSError as exc:
        raise MaterializeError(f"cannot read per-candidate band arrays: {exc}") from exc
    if len(source_bands) != len(records) or len(aion_bands) != len(records):
        raise MaterializeError(
            f"band arrays hold {len(source_bands)}/{len(aion_bands)} rows but the roster "
            f"has {len(records)}"
        )

    def _decode(row: Any) -> list[str]:
        return [b.decode() if isinstance(b, bytes) else str(b) for b in row]

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "candidate_index": index,
                "paired_targetid": record.get("paired_targetid"),
                "caption_object_id": record.get("caption_object_id"),
                "bands": {
                    "source": _decode(source_bands[index]),
                    "aion": _decode(aion_bands[index]),
                },
            }
        )
    return rows


def _candidate_manifest(candidates: Path, *, verify: bool) -> dict[str, Any]:
    """Build the manifest wrapper the Stage 1A/1B loaders expect."""
    rows = sum(1 for line in candidates.open(encoding="utf-8") if line.strip())
    if rows == 0:
        raise MaterializeError(f"{candidates} is empty")
    entry: dict[str, Any] = {"path": str(candidates.resolve()), "row_count": rows}
    if verify:
        entry["sha256"] = _sha256(candidates)
    return {
        # One of the two artifact types the loaders accept; see
        # src/data/astroprism_stage1a_dataset.py::_validate_manifest_types.
        "artifact_type": "astroprism_aion_feature_parity_reference",
        "schema_version": 1,
        "source": {"repo_id": REPO_ID, "repo_type": "dataset"},
        "outputs": {"candidates": entry, "captions.jsonl": entry},
    }


def _fetch_snapshot(revision: str | None, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=SNAPSHOT_FILES + ARRAY_PATTERNS,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--output-dir",
        default="_local/astroprism",
        help="where to write the manifest tree (default: %(default)s, what the configs read)",
    )
    ap.add_argument("--snapshot-dir", help="use an existing snapshot instead of downloading")
    ap.add_argument("--revision", help="dataset revision to pin")
    ap.add_argument("--no-download", action="store_true", help="fail rather than fetch")
    ap.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="skip the candidate SHA-256, which the loaders check when present",
    )
    args = ap.parse_args(argv)

    if args.snapshot_dir:
        snap = Path(args.snapshot_dir).expanduser().resolve()
        if not snap.is_dir():
            print(f"error: --snapshot-dir {snap} is not a directory", file=sys.stderr)
            return 1
    elif args.no_download:
        print("error: --no-download requires --snapshot-dir", file=sys.stderr)
        return 1
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print(
                "error: the dataset is gated; set HF_TOKEN to a token whose account "
                "has been granted access at https://huggingface.co/datasets/" + REPO_ID,
                file=sys.stderr,
            )
            return 1
        snap = _fetch_snapshot(args.revision, token)

    out = Path(args.output_dir).expanduser().resolve()
    verify = not args.no_verify_hashes

    try:
        images = _absolutize(_read_json(snap / "assets/raw/images/MANIFEST.json"),
                             snap / "assets/raw/images/MANIFEST.json")
        spectra = _absolutize(_read_json(snap / "assets/raw/spectra/MANIFEST.json"),
                              snap / "assets/raw/spectra/MANIFEST.json")
        candidates_file = snap / "metadata/candidates.jsonl"
        candidates = _candidate_manifest(candidates_file, verify=verify)
        records = [
            json.loads(line)
            for line in candidates_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The loader cross-checks the raw manifest's candidate list against the
        # roster row for row, so both must be present and in the same order.
        images["candidates"] = _raw_candidate_rows(snap, records)
        spectra["candidates"] = images["candidates"]
    except MaterializeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The loaders key on these exact artifact_type strings.
    images["artifact_type"] = "astroprism_raw_images"
    spectra["artifact_type"] = "astroprism_raw_spectra"

    written = {
        "raw_images/manifest.json": images,
        "raw_spectra/manifest.json": spectra,
        "stage1a_accepted/manifest.json": candidates,
        "stage1b_spectrum_caption_accepted/manifest.json": candidates,
    }
    for rel, payload in written.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {dest}")

    print(f"\nsnapshot: {snap}")
    print(f"candidates: {candidates['outputs']['candidates']['row_count']} rows")
    print(
        "\nThe arrays stay in the Hugging Face cache; these manifests point at them by\n"
        "absolute path. Move or clear that cache and regenerate this tree."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
