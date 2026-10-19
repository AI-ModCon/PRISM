"""The HF package and the training configs use different manifest envelopes.

These cover the bridge between them, with a synthetic snapshot: no network, no
gated repo, no 113 GB of arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tools import materialize_astroprism_hf_dataset as mat

BANDS_SOURCE = [b"des-g", b"des-r", b"des-i", b"des-z"]
BANDS_AION = [b"DES-G", b"DES-R", b"DES-I", b"DES-Z"]


def _snapshot(tmp_path: Path, rows: int = 3) -> Path:
    """Build a miniature stand-in for the published dataset tree."""
    snap = tmp_path / "snapshot"
    images = snap / "assets" / "raw" / "images"
    spectra = snap / "assets" / "raw" / "spectra"
    meta = snap / "metadata"
    for d in (images, spectra, meta):
        d.mkdir(parents=True)

    np.save(images / "image_array.npy", np.zeros((rows, 4, 8, 8), dtype=np.float32))
    np.save(images / "image_band.npy", np.array([BANDS_SOURCE] * rows, dtype="S5"))
    np.save(images / "aion_image_band.npy", np.array([BANDS_AION] * rows, dtype="S5"))
    np.save(spectra / "spectrum_flux.npy", np.zeros((rows, 16), dtype=np.float32))

    (images / "MANIFEST.json").write_text(json.dumps({
        "artifact_type": "astroprism_portable_raw_images",
        "row_count": rows,
        "outputs": {"image_array": {"path": "image_array.npy", "shape": [rows, 4, 8, 8]}},
    }))
    (spectra / "MANIFEST.json").write_text(json.dumps({
        "artifact_type": "astroprism_portable_raw_spectra",
        "row_count": rows,
        "outputs": {"spectrum_flux": {"path": "spectrum_flux.npy", "shape": [rows, 16]}},
    }))
    # Spectrum captions live in their own Parquet config, not in
    # candidates.jsonl -- the published dataset splits them that way.
    import pyarrow as pa
    import pyarrow.parquet as pq

    spec_cfg = snap / "data" / "spectrum_caption"
    spec_cfg.mkdir(parents=True)
    for split, idxs in (("train", list(range(rows))), ("validation", []), ("test", [])):
        pq.write_table(
            pa.table({
                "candidate_index": idxs,
                "paired_targetid": [1000 + i for i in idxs],
                "caption_object_id": [f"obj-{i}" for i in idxs],
                "source_group_id": [f"grp-{i}" for i in idxs],
                "split": [split] * len(idxs),
                "caption": [f"a spectrum {i}" for i in idxs],
                "caption_status": ["accepted"] * len(idxs),
                "review_status": ["accepted"] * len(idxs),
                "training_allowed": [split == "train"] * len(idxs),
                "dataset_ready": [True] * len(idxs),
            }),
            spec_cfg / f"{split}.parquet",
        )

    with open(meta / "candidates.jsonl", "w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(json.dumps({
                "candidate_index": i,
                "paired_targetid": 1000 + i,
                "caption_object_id": f"obj-{i}",
                "split": "train",
                "acceptance_status": "accepted",
                "caption": {"original": "a galaxy", "status": "released_reused"},
            }) + "\n")
    return snap


def _run(snap: Path, out: Path, *extra: str) -> int:
    return mat.main(["--snapshot-dir", str(snap), "--no-download",
                     "--output-dir", str(out), *extra])


def test_writes_the_four_manifests_the_configs_read(tmp_path: Path) -> None:
    out = tmp_path / "local"
    assert _run(_snapshot(tmp_path), out) == 0
    for rel in (
        "raw_images/manifest.json",
        "raw_spectra/manifest.json",
        "stage1a_accepted/manifest.json",
        "stage1b_spectrum_caption_accepted/manifest.json",
    ):
        assert (out / rel).is_file(), rel


def test_artifact_types_match_what_the_loaders_demand(tmp_path: Path) -> None:
    """The loaders reject any other value outright."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path), out)
    images = json.loads((out / "raw_images/manifest.json").read_text())
    spectra = json.loads((out / "raw_spectra/manifest.json").read_text())
    cands = json.loads((out / "stage1a_accepted/manifest.json").read_text())
    assert images["artifact_type"] == "astroprism_raw_images"
    assert spectra["artifact_type"] == "astroprism_raw_spectra"
    assert cands["artifact_type"] in {
        "astroprism_stage1a_candidates",
        "astroprism_aion_feature_parity_reference",
    }


def test_output_paths_are_absolute_so_the_tree_can_live_anywhere(tmp_path: Path) -> None:
    """Loaders resolve relative paths against the manifest's own directory."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path), out)
    images = json.loads((out / "raw_images/manifest.json").read_text())
    path = Path(images["outputs"]["image_array"]["path"])
    assert path.is_absolute() and path.is_file()


def test_candidate_rows_carry_index_and_band_order(tmp_path: Path) -> None:
    """The loader cross-checks these row for row against the roster."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path, rows=3), out)
    images = json.loads((out / "raw_images/manifest.json").read_text())
    rows = images["candidates"]
    assert [r["candidate_index"] for r in rows] == [0, 1, 2]
    assert rows[0]["bands"]["source"] == ["des-g", "des-r", "des-i", "des-z"]
    assert rows[0]["bands"]["aion"] == ["DES-G", "DES-R", "DES-I", "DES-Z"]
    assert rows[1]["caption_object_id"] == "obj-1"


def test_candidate_row_count_and_hash_are_recorded(tmp_path: Path) -> None:
    out = tmp_path / "local"
    _run(_snapshot(tmp_path, rows=3), out)
    cands = json.loads((out / "stage1a_accepted/manifest.json").read_text())
    entry = cands["outputs"]["candidates"]
    assert entry["row_count"] == 3
    assert len(entry["sha256"]) == 64


def test_hash_is_skipped_on_request(tmp_path: Path) -> None:
    out = tmp_path / "local"
    _run(_snapshot(tmp_path), out, "--no-verify-hashes")
    cands = json.loads((out / "stage1a_accepted/manifest.json").read_text())
    assert "sha256" not in cands["outputs"]["candidates"]


def test_missing_array_fails_loudly(tmp_path: Path) -> None:
    """A partial snapshot must not yield a manifest pointing at nothing."""
    snap = _snapshot(tmp_path)
    (snap / "assets/raw/images/image_array.npy").unlink()
    assert _run(snap, tmp_path / "local") == 1


def test_band_array_length_mismatch_fails(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, rows=3)
    np.save(snap / "assets/raw/images/image_band.npy",
            np.array([BANDS_SOURCE] * 2, dtype="S5"))
    assert _run(snap, tmp_path / "local") == 1


def test_no_download_without_a_snapshot_is_rejected(tmp_path: Path) -> None:
    assert mat.main(["--no-download", "--output-dir", str(tmp_path / "local")]) == 1


def test_missing_token_is_reported_rather_than_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert mat.main(["--output-dir", str(tmp_path / "local")]) == 1


def test_spectrum_roster_comes_from_the_parquet_not_candidates_jsonl(
    tmp_path: Path,
) -> None:
    """The published candidates.jsonl carries image captions only."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path, rows=3), out)
    rows = [
        json.loads(line)
        for line in (out / "stage1b_spectrum_caption_accepted/captions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [r["candidate_index"] for r in rows] == [0, 1, 2]
    assert rows[0]["spectrum_caption"]["original"] == "a spectrum 0"
    assert rows[0]["spectrum_caption"]["status"] == "accepted"


def test_split_roles_and_gradient_policy_follow_the_dataset_card(
    tmp_path: Path,
) -> None:
    """Only train may take gradients; validation and test are inspection-only."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path, rows=2), out)
    rows = [
        json.loads(line)
        for line in (out / "stage1b_spectrum_caption_accepted/captions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    policy = rows[0]["training_policy"]
    assert policy["role"] == "alignment_training"
    assert policy["gradient_training_allowed"] is True
    assert policy["eligible_for_evaluation"] is False


def test_both_manifests_declare_the_same_roster(tmp_path: Path) -> None:
    """The loader compares their SHA-256 to prove one roster describes both."""
    out = tmp_path / "local"
    _run(_snapshot(tmp_path), out)
    raw = json.loads((out / "raw_spectra/manifest.json").read_text())
    caps = json.loads((out / "stage1b_spectrum_caption_accepted/manifest.json").read_text())
    assert raw["roster"]["sha256"] == caps["roster"]["sha256"]
    assert raw["roster"]["row_count"] == caps["roster"]["row_count"]


def test_missing_spectrum_parquet_fails_with_a_usable_message(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    (snap / "data/spectrum_caption/train.parquet").unlink()
    assert _run(snap, tmp_path / "local") == 1


def test_training_allowed_disagreeing_with_the_split_is_rejected(
    tmp_path: Path,
) -> None:
    """A row marked trainable outside train must not quietly become trainable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    snap = _snapshot(tmp_path, rows=2)
    bad = snap / "data/spectrum_caption/test.parquet"
    pq.write_table(
        pa.table({
            "candidate_index": [0],
            "paired_targetid": [1000],
            "caption_object_id": ["obj-0"],
            "source_group_id": ["grp-0"],
            "split": ["test"],
            "caption": ["a spectrum"],
            "caption_status": ["accepted"],
            "review_status": ["accepted"],
            "training_allowed": [True],  # wrong for a test-split row
            "dataset_ready": [True],
        }),
        bad,
    )
    assert _run(snap, tmp_path / "local") == 1
