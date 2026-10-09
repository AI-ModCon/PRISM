"""Tests for tools/isoflop_collect.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT = REPO_ROOT / "tools" / "isoflop_collect.py"

# Derive header from the canonical column tuple so a schema add doesn't
# require fixing brittle positional fixtures here.
sys.path.insert(0, str(REPO_ROOT))
from tools.isoflop_collect import CSV_COLUMNS  # noqa: E402

CSV_HEADER = ",".join(CSV_COLUMNS)


def _write_perf_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_csv_planned(path: Path, run_id: str, family: str = "text_image") -> None:
    """Pre-seed CSV with a planned row, like isoflop_launch would have left it."""
    row = {c: "" for c in CSV_COLUMNS}
    row.update({
        "run_id": run_id, "phase": "M3", "family": family,
        "regime": "projector_only", "backbone": "OLMO3-1B",
        "projector_variant": "BASE",
        "budget_flops": "1.000000e+18", "seed": "0",
        "nodes": "1", "batch_size": "8",
        "max_seq_length": "2048", "max_steps": "100",
        "status": "planned",
    })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerow(row)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(COLLECT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )


def test_help_runs():
    r = _run("--help")
    assert r.returncode == 0


def test_full_round_trip(tmp_path: Path):
    """Plan-rows + perf.jsonl → CSV rows flipped to done with metrics."""
    run_id = "ISO-text_image-OLMO3-1B-BASE-1e18-s0"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "2026-05-27" / "12-00-00" / "perf.jsonl"
    records = [
        {"event": "startup_param_count", "total": 1_500_000_000, "active": 1_500_000_000,
         "train": 5_000_000, "projector_hidden_mult": 1, "projector_num_layers": 2},
    ]
    # 8 throughput rows; warmup=5 means rows 6-8 average.
    for i in range(8):
        records.append({
            "step": (i + 1) * 50, "samples_per_sec": 10.0 + i * 0.1,
            "flops_per_step": 2.0e15, "cumulative_flops": (i + 1) * 1.0e17,
            "seq_p50": 200, "seq_p95": 400, "padding_ratio": 0.3,
        })
    records.append({"event": "eval", "step": 400, "family": "image", "loss": 2.5})
    _write_perf_jsonl(perf, records)
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id)
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["status"] == "done"
    assert row["n_total_params"] == "1500000000"
    assert row["loss_main"] == row["loss_caption"]
    assert float(row["loss_caption"]) == 2.5
    assert float(row["samples_per_sec"]) > 10.0


def test_idempotent_skips_done(tmp_path: Path):
    """Re-running on a done row without --force should leave mtime alone."""
    run_id = "ISO-X"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id)
    # First collect: planned → done
    r1 = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r1.returncode == 0
    assert "updated=1" in r1.stderr
    # Second collect: should skip
    r2 = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r2.returncode == 0
    assert "skipped(already done)=1" in r2.stderr


def test_force_recollects_done(tmp_path: Path):
    run_id = "ISO-Y"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id)
    _run("--outputs", str(outputs), "--csv", str(csv_path))
    r = _run("--outputs", str(outputs), "--csv", str(csv_path), "--force")
    assert r.returncode == 0
    assert "updated=1" in r.stderr


def test_warmup_drop_skips_first_5(tmp_path: Path):
    """Throughput average must skip the first 5 samples."""
    run_id = "ISO-Z"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    # 5 warmup rows at 1.0; 3 post-warmup rows at 100.0 → avg should be 100.0
    records = [{"step": i, "samples_per_sec": 1.0} for i in range(5)]
    records += [{"step": 5 + i, "samples_per_sec": 100.0} for i in range(3)]
    _write_perf_jsonl(perf, records)
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id)
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r.returncode == 0
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert float(row["samples_per_sec"]) == 100.0


def test_loss_source_native_proxy_flows_to_csv(tmp_path: Path):
    """trainer_native stamps `loss_source: "train_running_mean"` in its
    eval records; collector must surface it into experiments.csv. Without
    this, a held-out-eval cell and a training-loss-proxy cell would be
    silently compared in the same parabola (see PR feedback on PR #98).
    """
    run_id = "ISO-LOSS-SOURCE-NATIVE"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 50, "samples_per_sec": 10.0},
        {"event": "eval", "step": 50, "family": "image", "loss": 3.5,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["loss_source"] == "train_running_mean"
    assert float(row["loss_caption"]) == 3.5


def test_loss_source_held_out_eval_flows_to_csv(tmp_path: Path):
    """Counterpart: trainer_zone_a stamps `loss_source: "held_out_eval"`."""
    run_id = "ISO-LOSS-SOURCE-HELDOUT"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 50, "samples_per_sec": 10.0},
        {"event": "eval", "step": 50, "family": "image", "loss": 2.0,
         "loss_source": "held_out_eval"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["loss_source"] == "held_out_eval"


def test_loss_source_in_csv_columns_contract():
    """Collector's CSV_COLUMNS must include loss_source — it's part of
    the schema contract launchers / fitters depend on.
    """
    from tools.isoflop_collect import CSV_COLUMNS as COLLECT_COLS
    from tools.isoflop_launch import CSV_COLUMNS as LAUNCH_COLS
    assert "loss_source" in COLLECT_COLS, "collect schema must declare loss_source"
    assert "loss_source" in LAUNCH_COLS, "launch schema must mirror collect"
    # Both schemas must agree (otherwise a launch-created CSV will warn
    # about schema drift when collect reads it).
    assert set(COLLECT_COLS) == set(LAUNCH_COLS), (
        f"CSV schema drift: only_collect={set(COLLECT_COLS) - set(LAUNCH_COLS)}, "
        f"only_launch={set(LAUNCH_COLS) - set(COLLECT_COLS)}"
    )


def test_loss_main_is_mean_over_window(tmp_path: Path):
    """Default --eval-window=5 should average the last 5 eval rows per family.

    Without this, unconverged cells (Stage A round 1 BASE@3e17 had stdev=1.17
    over its last 10 evals) report a noisy moving-target as loss_main.
    """
    run_id = "ISO-WINDOWED"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    records = [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
    ]
    # 10 eval rows, last 5 = [3, 4, 5, 6, 7] → mean = 5.0
    # Single-last-value would be 7.0 (a misleadingly high "noisy" sample).
    losses = [10.0, 9.0, 8.0, 7.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for i, loss in enumerate(losses):
        records.append({"event": "eval", "step": (i + 1) * 5, "family": "image",
                        "loss": loss, "loss_source": "train_running_mean"})
    _write_perf_jsonl(perf, records)
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    # Mean of [3, 4, 5, 6, 7] = 5.0 (not 7.0 = last value)
    assert float(row["loss_main"]) == pytest.approx(5.0), (
        f"loss_main should be mean of last 5 evals (5.0), got {row['loss_main']!r}"
    )
    # Stability = sample stdev over [3, 4, 5, 6, 7] ≈ 1.5811
    assert float(row["loss_stability"]) == pytest.approx(1.5811, rel=1e-3), (
        f"loss_stability should be sample stdev (~1.58), got {row['loss_stability']!r}"
    )


def test_eval_window_one_restores_old_behavior(tmp_path: Path):
    """--eval-window=1 should reproduce the pre-fix single-last-value behavior
    so users with prior CSVs can re-collect with consistent semantics.
    """
    run_id = "ISO-WIN-1"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    records = [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
        {"event": "eval", "step": 10, "family": "image", "loss": 3.0,
         "loss_source": "train_running_mean"},
        {"event": "eval", "step": 20, "family": "image", "loss": 7.0,
         "loss_source": "train_running_mean"},
    ]
    _write_perf_jsonl(perf, records)
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path), "--eval-window", "1")
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    # With window=1, loss_main = last value = 7.0
    assert float(row["loss_main"]) == pytest.approx(7.0)
    # stability over a single value is 0.0 (sample stdev needs n>=2)
    assert float(row["loss_stability"]) == pytest.approx(0.0)


def test_loss_window_per_family_independent(tmp_path: Path):
    """For runs with multiple eval families (text_image + text_ts in the same
    run, e.g. multi-modality cells), the window should slide PER family, not
    over the overall interleaved sequence.
    """
    run_id = "ISO-MULTIFAM"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    records = [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
    ]
    # Interleave 5 evals per family (image, ts). Image losses = [1..5],
    # TS losses = [10..50]. Window=3 should mean:
    # image_mean = mean([3,4,5]) = 4.0, ts_mean = mean([30,40,50]) = 40.0
    for i, (img_loss, ts_loss) in enumerate(
        zip([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0], strict=True)
    ):
        step = (i + 1) * 10
        records.append({"event": "eval", "step": step, "family": "image", "loss": img_loss,
                        "loss_source": "train_running_mean"})
        records.append({"event": "eval", "step": step, "family": "time_series", "loss": ts_loss,
                        "loss_source": "train_running_mean"})
    _write_perf_jsonl(perf, records)
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path), "--eval-window", "3")
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    # loss_main for text_image family → image column = mean of last 3 image = 4.0
    assert float(row["loss_caption"]) == pytest.approx(4.0), (
        f"image loss should be mean([3,4,5])=4.0, got {row['loss_caption']!r}"
    )
    assert float(row["loss_ts_qa"]) == pytest.approx(40.0), (
        f"ts loss should be mean([30,40,50])=40.0, got {row['loss_ts_qa']!r}"
    )


def test_loss_stability_in_csv_columns_contract():
    """loss_stability must appear in both collector and launch CSV_COLUMNS."""
    from tools.isoflop_collect import CSV_COLUMNS as COLLECT_COLS
    from tools.isoflop_launch import CSV_COLUMNS as LAUNCH_COLS
    assert "loss_stability" in COLLECT_COLS
    assert "loss_stability" in LAUNCH_COLS


@pytest.mark.parametrize("bad_window", ["0", "-1", "-5"])
def test_eval_window_rejects_non_positive(tmp_path: Path, bad_window: str):
    """--eval-window 0 used to silently average ALL evals (slice [-0:] = full
    sequence); negative values silently dropped a forward slice. Both produce
    garbage loss_main with no error. Argparse must reject them.
    """
    run_id = "ISO-BAD-WIN"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 10, "samples_per_sec": 5.0},
        {"event": "eval", "step": 10, "family": "image", "loss": 3.0},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--eval-window", bad_window)
    assert r.returncode != 0, (
        f"--eval-window {bad_window} should be rejected, "
        f"but exited 0 with stderr={r.stderr!r}"
    )
    assert "eval-window" in r.stderr.lower()


def test_loss_window_fewer_evals_than_window(tmp_path: Path):
    """A short smoke (max_steps=10, eval_every=5) produces only 2 evals.
    Window=5 must clamp to the available count without erroring, and stdev
    is the sample stdev over the 2 values.
    """
    run_id = "ISO-SHORT"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1_000_000, "active": 1_000_000, "train": 1000},
        {"step": 5, "samples_per_sec": 5.0},
        {"event": "eval", "step": 5, "family": "image", "loss": 4.0,
         "loss_source": "train_running_mean"},
        {"event": "eval", "step": 10, "family": "image", "loss": 6.0,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path))  # default window=5
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    # mean of [4.0, 6.0] = 5.0; sample stdev = sqrt(((4-5)^2 + (6-5)^2)/1) = sqrt(2)
    assert float(row["loss_main"]) == pytest.approx(5.0)
    assert float(row["loss_stability"]) == pytest.approx(2 ** 0.5, rel=1e-6)


# --- --since / --sweep-id filter tests (PR follow-up) -----------------------
#
# These guard the contamination-loop fix: an old Smoke 4 perf.jsonl in
# outputs/<cell>/2026-05-27/... was overwriting freshly-trained cell loss
# on every collect. The --since (mtime cutoff) and --sweep-id (startup
# event field) filters give the user two independent levers to scope a
# collect to just the run they care about.

import os
import time


def test_since_excludes_old_perf_jsonl(tmp_path: Path):
    """A perf.jsonl with mtime BEFORE --since should be skipped silently."""
    run_id = "ISO-OLD"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1, "active": 1, "train": 1},
        {"event": "eval", "step": 1, "family": "image", "loss": 0.5,
         "loss_source": "train_running_mean"},
    ])
    # Set mtime to ~1 year ago
    old = time.time() - 365 * 86400
    os.utime(perf, (old, old))
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--since", "2026-01-01")
    assert r.returncode == 0, r.stderr
    # Row stays planned (collect skipped it)
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["status"] == "planned"
    assert row["loss_main"] == ""
    assert "filtered(--since)=1" in r.stderr


def test_since_includes_new_perf_jsonl(tmp_path: Path):
    """A perf.jsonl with mtime AFTER --since is processed normally."""
    run_id = "ISO-NEW"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1, "active": 1, "train": 1},
        {"event": "eval", "step": 1, "family": "image", "loss": 0.5,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--since", "2020-01-01")
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["status"] == "done"
    assert float(row["loss_main"]) == 0.5


def test_since_relative_now_form(tmp_path: Path):
    """`--since now-7d` parses without error."""
    run_id = "ISO-REL"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1, "active": 1, "train": 1},
        {"event": "eval", "step": 1, "family": "image", "loss": 0.5,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--since", "now-7d")
    assert r.returncode == 0, r.stderr


def test_since_rejects_garbage(tmp_path: Path):
    """`--since notadate` should hard-fail at parse time, not silently
    include/exclude everything."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, "ISO-X")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--since", "not-a-date")
    assert r.returncode != 0
    assert "not a recognized timestamp format" in r.stderr


def test_sweep_id_filter_excludes_mismatched(tmp_path: Path):
    """A perf.jsonl whose startup record has a different sweep_id is
    skipped (returns 0, row stays planned, summary counter increments)."""
    run_id = "ISO-SWEEP"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1, "active": 1, "train": 1,
         "sweep_id": "OLD-SWEEP"},
        {"event": "eval", "step": 1, "family": "image", "loss": 0.5,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--sweep-id", "NEW-SWEEP")
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["status"] == "planned"
    assert "filtered(--sweep-id)=1" in r.stderr


def test_sweep_id_filter_includes_matching(tmp_path: Path):
    """Matching sweep_id passes the filter."""
    run_id = "ISO-SWEEP-OK"
    outputs = tmp_path / "outputs"
    perf = outputs / run_id / "perf.jsonl"
    _write_perf_jsonl(perf, [
        {"event": "startup_param_count", "total": 1, "active": 1, "train": 1,
         "sweep_id": "WANTED-SWEEP"},
        {"event": "eval", "step": 1, "family": "image", "loss": 0.5,
         "loss_source": "train_running_mean"},
    ])
    csv_path = tmp_path / "exp.csv"
    _write_csv_planned(csv_path, run_id, family="text_image")
    r = _run("--outputs", str(outputs), "--csv", str(csv_path),
             "--sweep-id", "WANTED-SWEEP")
    assert r.returncode == 0, r.stderr
    with open(csv_path) as f:
        row = list(csv.DictReader(f))[0]
    assert row["status"] == "done"
