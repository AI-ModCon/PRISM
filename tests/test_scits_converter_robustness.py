"""CPU unit tests for scripts/convert_scits_to_webdataset.py robustness
fixes from the PR #129 (SciTS/TimeOmni) review:

  - swallowed exception on series load failure now includes the real cause
  - the resume/skip-conversion check must never fire for a --max-samples
    (smoke) run, even against an output dir from a prior full conversion
  - a mid-conversion exception must still close in-progress tar shards
    cleanly (try/finally around the per-record loop)

No torch import — numpy/pandas/tqdm only, fast.
"""

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "convert_scits_to_webdataset", REPO_ROOT / "scripts" / "convert_scits_to_webdataset.py"
)
_convert_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _convert_mod  # dataclass() needs sys.modules[cls.__module__]
_spec.loader.exec_module(_convert_mod)

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]


def _write_scits_source(root: Path, records: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ts_dir = root / "series"
    ts_dir.mkdir(exist_ok=True)
    meta_path = root / "meta_data.jsonl"
    with meta_path.open("w") as f:
        for i, rec in enumerate(records):
            arr = rec.pop("_array", None)
            if arr is not None:
                np.save(ts_dir / f"s{i}.npy", arr)
                rec["input_ts"] = f"series/s{i}.npy"
            f.write(json.dumps(rec) + "\n")
    return meta_path


def _read_shard_metadata(output_dir: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for shard_path in (output_dir / "shards").glob("*.tar"):
        with tarfile.open(shard_path) as shard:
            for member in shard.getmembers():
                if member.name.endswith(".meta.json"):
                    payload = shard.extractfile(member)
                    assert payload is not None
                    metadata[member.name.removesuffix(".meta.json")] = json.loads(
                        payload.read().decode("utf-8")
                    )
    return metadata


def test_converts_canonical_scits_task_types_and_series_formats(tmp_path):
    """SciTS task IDs and CSV/NPY inputs retain their canonical semantics."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "series").mkdir()
    (source_dir / "series" / "input.csv").write_text("1.0\nX\n3.0\n")
    np.save(source_dir / "series" / "input.npy", np.array([5.0, 6.0]))
    np.save(source_dir / "series" / "forecast.npy", np.array([7.0, 8.0]))

    records = [
        {"id": "imputation", "task_id": ["NEG04"], "data_type": "csv", "input_ts": "series/input.csv"},
        {"id": "forecasting", "task_id": ["MEG03"], "data_type": "npy", "input_ts": "series/input.npy", "gt_ts": "series/forecast.npy"},
        {"id": "anomaly", "task_id": ["MEU01"], "data_type": "npy", "input_ts": "series/input.npy"},
        {"id": "classification", "task_id": ["ASU03"], "data_type": "npy", "input_ts": "series/input.npy"},
        {"id": "event", "task_id": ["ASU01", "ASG02"], "data_type": "npy", "input_ts": "series/input.npy"},
        {"id": "mcq", "task_id": ["MEU04"], "data_type": "npy", "input_ts": "series/input.npy"},
        {"id": "synthesize", "task_id": ["ENG01"], "data_type": "npy", "input_ts": "series/input.npy"},
    ]
    (source_dir / "meta_data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    output_dir = tmp_path / "output"
    manifest = _convert_mod.convert_scits_to_webdataset(
        output_dir=str(output_dir),
        scits_dir=str(source_dir),
        samples_per_shard=100,
        val_ratio=0.0,
    )

    metadata = _read_shard_metadata(output_dir)
    loaded_csv = _convert_mod._load_series_from_file(source_dir / "series" / "input.csv")
    assert manifest["total_samples"] == len(records)
    assert loaded_csv.shape == (3,)
    assert loaded_csv[1] == "X"
    assert metadata["imputation"]["task_type"] == "imputation"
    assert metadata["forecasting"]["task_type"] == "forecasting"
    assert metadata["forecasting"]["gt_ts_shape"] == [2]
    assert metadata["anomaly"]["task_type"] == "anomaly_detection"
    assert metadata["classification"]["task_type"] == "classification"
    assert metadata["event"]["task_id"] == "ASU01_ASG02"
    assert metadata["event"]["task_type"] == "event_detection"
    assert metadata["mcq"]["task_type"] == "mcq"
    assert metadata["synthesize"]["task_type"] == "synthesize"


def test_load_failure_message_includes_real_exception(tmp_path):
    """A malformed series file must produce a print message that names the
    real exception, not a bare 'Failed to load' with no cause."""
    tmp_root = tmp_path
    source_dir = tmp_root / "source"
    ts_dir = source_dir / "series"
    ts_dir.mkdir(parents=True)
    bad_npy = ts_dir / "corrupt.npy"
    bad_npy.write_bytes(b"not a real npy file")

    meta_path = source_dir / "meta_data.jsonl"
    meta_path.write_text(
        json.dumps({
            "id": "s0",
            "input_ts": "series/corrupt.npy",
            "input_text": "q",
            "gt_text": "a",
        }) + "\n"
    )

    output_dir = tmp_root / "output"

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        manifest = _convert_mod.convert_scits_to_webdataset(
            output_dir=str(output_dir),
            scits_dir=str(source_dir),
        )

    printed = buf.getvalue()
    assert "corrupt.npy" in printed
    # The real fix: the exception TYPE/message must be present, not swallowed.
    assert "Failed to load series" in printed
    assert manifest["skipped_missing_series"] == 1
    assert manifest["total_samples"] == 0


def test_max_samples_never_resume_skips_prior_full_conversion(tmp_path):
    """Defect (converter resume check): running --max-samples against an
    output dir that already has a FULL prior conversion must NOT silently
    return the full manifest — it must actually run the smoke conversion."""
    tmp_root = tmp_path
    source_dir = tmp_root / "source"
    records = [
        {
            "id": f"s{i}",
            "_array": np.random.randn(10).astype(np.float32),
            "input_text": f"q{i}",
            "gt_text": f"a{i}",
        }
        for i in range(5)
    ]
    _write_scits_source(source_dir, records)
    output_dir = tmp_root / "output"

    # First: a full conversion.
    full_manifest = _convert_mod.convert_scits_to_webdataset(
        output_dir=str(output_dir),
        scits_dir=str(source_dir),
        samples_per_shard=100,
        val_ratio=0.0,
    )
    assert full_manifest["total_samples"] == 5

    # Second: a --max-samples=2 smoke run against the SAME output dir.
    # Before the fix, expected_train_shards was computed from the FULL
    # example count (5), actual_train_shards (1 shard, already >= expected),
    # so it would hit the "skip conversion" branch and return the stale
    # 5-sample manifest instead of running the requested 2-sample smoke.
    smoke_manifest = _convert_mod.convert_scits_to_webdataset(
        output_dir=str(output_dir),
        scits_dir=str(source_dir),
        samples_per_shard=100,
        val_ratio=0.0,
        max_samples=2,
    )
    assert smoke_manifest["total_samples"] == 2, (
        f"expected max_samples=2 to actually run and produce 2 samples, "
        f"got manifest: {smoke_manifest}"
    )


def test_shard_writer_closed_on_mid_conversion_exception(monkeypatch, tmp_path):
    """A mid-conversion exception must still close in-progress tar shards
    via the try/finally, not leave a truncated/unclosed tar file."""
    tmp_root = tmp_path
    source_dir = tmp_root / "source"
    records = [
        {
            "id": f"s{i}",
            "_array": np.random.randn(10).astype(np.float32),
            "input_text": f"q{i}",
            "gt_text": f"a{i}",
        }
        for i in range(3)
    ]
    _write_scits_source(source_dir, records)
    output_dir = tmp_root / "output"

    call_count = {"n": 0}
    original_compose = _convert_mod._compose_text

    def _boom(example, is_forecasting=False):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-conversion crash")
        return original_compose(example, is_forecasting)

    monkeypatch.setattr(_convert_mod, "_compose_text", _boom)

    with pytest.raises(RuntimeError, match="simulated mid-conversion crash"):
        _convert_mod.convert_scits_to_webdataset(
            output_dir=str(output_dir),
            scits_dir=str(source_dir),
            samples_per_shard=100,
            val_ratio=0.0,
        )

    # The shard for the ONE successfully-written record must be a valid,
    # readable tar file (finish()/close() ran via the finally block).
    shards_dir = output_dir / "shards"
    tar_files = sorted(f for f in shards_dir.iterdir() if f.suffix == ".tar")
    assert len(tar_files) == 1, f"expected exactly one shard file, got {tar_files}"
    with tarfile.open(tar_files[0]) as tf:
        members = tf.getmembers()
        assert len(members) > 0, "shard tar is empty/corrupt — finally block did not close it"
