"""Tests for src/utils/perf_log.py and tools/perf_aggregate.py."""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from src.utils.perf_log import log_perf_record, model_modalities


def test_log_perf_record_writes_jsonl(tmp_path: Path):
    log_perf_record(tmp_path, {"step": 10, "samples_per_sec": 5.4})
    log_perf_record(tmp_path, {"step": 20, "samples_per_sec": 6.1})
    path = tmp_path / "perf.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["step"] == 10
    assert rec0["samples_per_sec"] == 5.4
    assert "wall_time" in rec0


def test_log_perf_record_with_none_output_dir_is_noop():
    log_perf_record(None, {"step": 1})


def test_log_perf_record_creates_missing_directory(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "out"
    log_perf_record(nested, {"step": 1})
    assert (nested / "perf.jsonl").exists()


def test_aggregator_emits_csv_from_two_runs(tmp_path: Path):
    run_a = tmp_path / "runA"
    run_b = tmp_path / "runB"
    log_perf_record(run_a, {"step": 10, "samples_per_sec": 5.0, "site": "x"})
    log_perf_record(run_a, {"step": 20, "samples_per_sec": 5.5, "site": "x"})
    log_perf_record(run_b, {"step": 10, "samples_per_sec": 8.0, "site": "x"})

    repo_root = Path(__file__).resolve().parent.parent
    out = subprocess.check_output(
        [sys.executable, str(repo_root / "tools" / "perf_aggregate.py"), str(tmp_path)],
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 3
    assert {r["run_dir"] for r in rows} == {"runA", "runB"}
    assert {r["samples_per_sec"] for r in rows} == {"5.0", "5.5", "8.0"}


def test_aggregator_filters(tmp_path: Path):
    run = tmp_path / "run"
    log_perf_record(run, {"step": 1, "site": "trainer_native_per_50"})
    log_perf_record(run, {"step": 2, "site": "trainer_native_per_log"})
    log_perf_record(run, {"step": 3, "site": "trainer_native_per_50"})

    repo_root = Path(__file__).resolve().parent.parent
    out = subprocess.check_output(
        [
            sys.executable,
            str(repo_root / "tools" / "perf_aggregate.py"),
            str(tmp_path),
            "--filter",
            "site=trainer_native_per_50",
        ],
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 2
    assert all(r["site"] == "trainer_native_per_50" for r in rows)


def test_aggregator_handles_no_files(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "tools" / "perf_aggregate.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # No CSV body since there are no records.


def test_aggregator_missing_root_returns_error(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "perf_aggregate.py"),
            str(tmp_path / "does-not-exist"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_aggregator_invalid_filter_format(tmp_path: Path):
    log_perf_record(tmp_path, {"step": 1})
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "perf_aggregate.py"),
            str(tmp_path),
            "--filter",
            "nope_no_equals",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@pytest.fixture(autouse=True)
def _clear_cache():
    """perf_log caches output_dir → path. Reset between tests."""
    from src.utils import perf_log

    perf_log._LOG_FILENAMES.clear()
    yield
    perf_log._LOG_FILENAMES.clear()


# --- Regression: DictConfig/ListConfig must not crash log_perf_record ---


def test_log_perf_record_handles_omegaconf_listconfig(tmp_path: Path):
    """OmegaConf ListConfig in a record (e.g. from `cfg.model.modalities`)
    must not crash json.dumps. Regression for the bug that killed graph
    smoke runs at step 10: model_modalities returned ListConfig and
    log_perf_record raised `TypeError: Object of type DictConfig is not
    JSON serializable`, which propagated and killed rank 0."""
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create({"modalities": ["text", "image", "graph"]})
    log_perf_record(tmp_path, {"step": 10, "modalities": cfg.modalities})
    line = json.loads((tmp_path / "perf.jsonl").read_text().strip())
    # default=str coerces ListConfig to its repr; the point is no crash.
    assert "modalities" in line


def test_log_perf_record_silently_skips_unserializable_objects(tmp_path: Path):
    """Final backstop: an object whose default=str fallback still can't
    encode should result in a dropped record + a warning, never an
    exception bubbling into the training step."""

    class _Boom:
        def __str__(self):
            raise RuntimeError("nope")

    log_perf_record(tmp_path, {"step": 1, "bad": _Boom()})
    # File may exist (mkdir succeeded) but be empty (record dropped).
    path = tmp_path / "perf.jsonl"
    if path.exists():
        assert path.read_text() == ""


def test_model_modalities_returns_plain_list_of_strings():
    """Regression: must return list[str], not omegaconf.ListConfig or
    list[Modality], so the result is JSON-serializable."""
    omegaconf = pytest.importorskip("omegaconf")

    class _FakeCfg:
        def __init__(self):
            self.modalities = omegaconf.OmegaConf.create(["text", "image"])

    class _FakeModel:
        config = _FakeCfg()

    out = model_modalities(_FakeModel())
    assert isinstance(out, list)
    for m in out:
        assert isinstance(m, str)
    # Round-trip through json to be doubly sure.
    json.dumps(out)


def test_model_modalities_unwraps_ddp_module():
    """Regression: must unwrap `.module` to find the inner config."""

    class _Cfg:
        modalities = ["text", "image"]

    class _Inner:
        config = _Cfg()

    class _DDPWrapper:
        module = _Inner()

    out = model_modalities(_DDPWrapper())
    assert out == ["text", "image"]


def test_model_modalities_returns_none_when_no_config():
    class _Bare:
        pass

    assert model_modalities(_Bare()) is None


def test_log_perf_record_round_trips_isoflop_keys(tmp_path: Path):
    """The IsoFLOP per-step extensions must serialize cleanly.

    Mirrors the dict shape `trainer_zone_a.train()` constructs after the
    PR-1 instrumentation patches: sequence percentiles, flop counters,
    projector knobs.
    """
    log_perf_record(
        tmp_path,
        {
            "site": "trainer_zone_a",
            "step": 50,
            "samples_per_sec": 12.3,
            "seq_p50": 200.0,
            "seq_p95": 410.0,
            "seq_p99": 480.0,
            "seq_max": 512.0,
            "padding_ratio": 0.42,
            "flops_per_step": 1.5e15,
            "cumulative_flops": 7.5e16,
            "projector_hidden_mult": 2,
            "projector_num_layers": 4,
        },
    )
    line = (tmp_path / "perf.jsonl").read_text().strip().splitlines()
    assert len(line) == 1
    rec = json.loads(line[0])
    for key in (
        "seq_p50",
        "seq_p95",
        "seq_p99",
        "seq_max",
        "padding_ratio",
        "flops_per_step",
        "cumulative_flops",
        "projector_hidden_mult",
        "projector_num_layers",
    ):
        assert key in rec, f"missing {key}"
    assert rec["projector_hidden_mult"] == 2
    assert rec["projector_num_layers"] == 4
    assert rec["flops_per_step"] == 1.5e15


def test_log_perf_record_handles_null_flops_for_uncalibrated(tmp_path: Path):
    """Uncalibrated runs emit `flops_per_step=None`; collector treats as missing."""
    log_perf_record(
        tmp_path,
        {
            "site": "trainer_zone_a",
            "step": 50,
            "flops_per_step": None,
            "cumulative_flops": None,
        },
    )
    line = (tmp_path / "perf.jsonl").read_text().strip().splitlines()
    rec = json.loads(line[0])
    assert rec["flops_per_step"] is None
    assert rec["cumulative_flops"] is None
