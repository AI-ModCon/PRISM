#!/usr/bin/env python3
"""Collect IsoFLOP run outputs into experiments.csv.

For each run directory under `outputs/<run_id>/...`:
  1. Locate `perf.jsonl` (via `tools/perf_aggregate.find_perf_files`).
  2. Segregate records by event:
       - `startup_param_count` → fills `n_total_params, n_active_params,
         n_trainable_params, projector_hidden_mult, projector_num_layers`.
       - throughput rows (no `event` key) → average `samples_per_sec,
         tokens_per_sec, seq_p50/95/99, padding_ratio, flops_per_step`
         over a post-warmup window (skip first 5).
       - `event=eval` rows → take *last* `loss` per family as
         `loss_<family>` columns.
  3. Look up the CSV row by `run_id`. If `status==done` and not `--force`,
     skip. Else write columns + flip status to `done`.

Idempotent. Single-owner: PR-2 owns CSV writes; PR-1 only emits perf.jsonl.

Usage:
    python tools/isoflop_collect.py
    python tools/isoflop_collect.py --outputs outputs/ --csv /path/to/experiments.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.perf_aggregate import find_perf_files  # noqa: E402

# Mirror the schema pinned in isoflop_launch.py so a column drift surfaces
# loudly in both places.
CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "phase", "family", "regime", "backbone", "projector_variant",
    "projector_hidden_mult", "projector_num_layers",
    "budget_flops", "seed", "nodes", "batch_size", "grad_accum",
    "max_seq_length", "max_steps", "n_total_params", "n_active_params",
    "n_trainable_params", "flops_per_step", "cumulative_flops",
    "d_compute_tokens", "d_label_tokens", "d_modality_tokens",
    "samples_per_sec", "tokens_per_sec", "seq_p50", "seq_p95", "seq_p99",
    "padding_ratio", "loss_main", "loss_caption",
    "loss_pointing", "loss_count", "loss_ts_qa", "loss_graph_qa",
    "loss_source",  # held_out_eval (trainer_zone_a) vs train_running_mean (trainer_native)
    "loss_stability",  # sample stdev of loss over the last `window` evals — large
                       # values flag unconverged cells whose loss_main is noisy
                       # (Stage A round 1 BASE@3e17 had stdev=1.17 over 10 evals)
    "downstream_metric", "status", "job_id", "launched_at",
    "completed_at", "wandb_url", "notes",
)


# How many leading throughput rows to drop before averaging — matches
# tools/perf_aggregate.py convention so collector aggregates stay
# comparable across the two tools.
THROUGHPUT_WARMUP = 5


def _read_perf_records(path: Path) -> list[dict[str, Any]]:
    """Parse one perf.jsonl into a list of dicts. Bad lines warn + skip."""
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: {path}:{line_no} skipped (not JSON): {e}", file=sys.stderr)
    return out


def _parse_run_id(path: Path, outputs_root: Path) -> str | None:
    """Derive run_id from `outputs/<run_id>/...` path layout.

    Returns None if the perf.jsonl isn't nested under `outputs_root`.
    """
    try:
        rel = path.relative_to(outputs_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]


def _aggregate_throughput(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average post-warmup throughput records.

    A throughput row is any record without an `event` field (the trainer
    emits `event=startup_param_count` and `event=eval`; everything else
    is a per-window throughput sample).
    """
    samples = [r for r in records if "event" not in r]
    # Drop warmup. Trainer flushes once per ~50 training steps, so 5 ==
    # ~250 steps of warmup — generous enough that BS-warmup distortions
    # don't reach the fit.
    samples = samples[THROUGHPUT_WARMUP:] if len(samples) > THROUGHPUT_WARMUP else samples
    if not samples:
        return {}

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in samples if isinstance(r.get(key), (int, float))]
        return mean(vals) if vals else None

    out: dict[str, Any] = {
        "samples_per_sec": _avg("samples_per_sec"),
        "tokens_per_sec": _avg("tokens_per_sec"),
        "seq_p50": _avg("seq_p50"),
        "seq_p95": _avg("seq_p95"),
        "seq_p99": _avg("seq_p99"),
        "padding_ratio": _avg("padding_ratio"),
        "flops_per_step": _avg("flops_per_step"),
    }
    # cumulative_flops: take the LAST seen value across the full run
    # (not the post-warmup window) so we record the actual final budget.
    all_thr = [r for r in records if "event" not in r]
    cum_vals = [r["cumulative_flops"] for r in all_thr
                if isinstance(r.get("cumulative_flops"), (int, float))]
    if cum_vals:
        out["cumulative_flops"] = cum_vals[-1]
    return out


def _last_eval_losses(
    records: list[dict[str, Any]],
    window: int = 5,
) -> tuple[dict[str, float], dict[str, float], str | None]:
    """Per-family eval loss aggregated over the last `window` eval points
    → `loss_<family>` mapping + per-family stability + loss_source.

    The trainer emits one `event=eval` record per active family per eval
    interval. The previous implementation took the SINGLE last value; for
    cells that hadn't converged that's a moving target (Stage A round 1
    BASE@3e17 had stdev=1.17 over its last 10 evals — final-value-only
    sampled a wildly bouncing curve). Taking the mean of the last `window`
    evals stabilizes the fit-input loss without delaying or biasing it.

    `window` is the eval-row count (not step count); for a cell with
    eval_every=5 and max_steps=51, that's the last 25 steps of training.

    The optional `loss_source` field disambiguates two semantically-different
    sources:
      - `held_out_eval` — trainer_zone_a's run_evaluation against a real
        evaluator suite
      - `train_running_mean` — trainer_native's training-loss proxy
        (added in PR #98)
    Two cells with different loss_source values compared in the same
    parabola would silently mislead the fit, so the collector surfaces it
    into `experiments.csv:loss_source` for audit.

    Returns `(by_family_mean, by_family_stdev, loss_source)`:
      - by_family_mean: arithmetic mean of last `window` eval values per family
      - by_family_stdev: sample stdev over the same window (0.0 if window=1).
        Surfaces in CSV as `loss_stability` — large stdev (>~0.1 relative)
        signals an unconverged cell whose `loss_main` is noisy.
      - loss_source: last seen value across all eval records (assumes a
        single trainer emits all eval records for a run, which is true
        today). None if no eval record carried a loss_source.
    """
    # Collect per-family eval values in order so the window is over the
    # most recent `window` rows for THAT family (not the overall last K
    # rows, which would mix families for runs with multiple eval families).
    per_family_seq: dict[str, list[float]] = {}
    loss_source: str | None = None
    for r in records:
        if r.get("event") != "eval":
            continue
        fam = r.get("family")
        loss = r.get("loss")
        if not fam or not isinstance(loss, (int, float)):
            continue
        per_family_seq.setdefault(str(fam), []).append(float(loss))
        src = r.get("loss_source")
        if isinstance(src, str) and src:
            loss_source = src

    by_family_mean: dict[str, float] = {}
    by_family_stdev: dict[str, float] = {}
    for fam, seq in per_family_seq.items():
        tail = seq[-window:] if window > 0 else seq
        if not tail:
            continue
        by_family_mean[fam] = sum(tail) / len(tail)
        if len(tail) >= 2:
            m = by_family_mean[fam]
            var = sum((x - m) ** 2 for x in tail) / (len(tail) - 1)
            by_family_stdev[fam] = var ** 0.5
        else:
            by_family_stdev[fam] = 0.0
    return by_family_mean, by_family_stdev, loss_source


def _startup_params(records: list[dict[str, Any]]) -> dict[str, Any]:
    for r in records:
        if r.get("event") == "startup_param_count":
            return {
                "n_total_params": r.get("total"),
                "n_active_params": r.get("active"),
                "n_trainable_params": r.get("train"),
                "projector_hidden_mult": r.get("projector_hidden_mult"),
                "projector_num_layers": r.get("projector_num_layers"),
            }
    return {}


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("run_id")
            if rid:
                rows[rid] = row
    return rows


def _write_csv(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for rid in sorted(rows):
            row = {col: rows[rid].get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row)


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6e}" if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3) else f"{v:.6f}"
    return str(v)


# Map both cell family (text_image) and modality family (image) to the
# main-loss column name. Mirrors the column set in the experiments.csv
# header (loss_caption / loss_ts_qa / loss_graph_qa).
#
# Asymmetry note: the cell-level `family` (column in experiments.csv) is
# e.g. `text_image`, but the trainer emits `event=eval` records keyed off
# `model.config.modalities` so the per-record `family` is the modality
# name (`text`, `image`, ...). The mapping below accepts both forms.
_FAMILY_LOSS_COL: dict[str, str] = {
    # Cell families
    "text_image": "loss_caption",
    "text_ts": "loss_ts_qa",
    "text_graph": "loss_graph_qa",
    # Modality names (as the trainer's eval records emit them)
    "image": "loss_caption",
    "time_series": "loss_ts_qa",
    "graph": "loss_graph_qa",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", default="outputs/", help="Root of run outputs")
    p.add_argument(
        "--csv",
        default=None,
        help="experiments.csv path. Defaults to <scaling-study-dir>/experiments.csv",
    )
    p.add_argument(
        "--scaling-study-dir",
        default="/flare/ModCon/ngetty/BaseMM_PRISM/scaling-study/",
    )
    p.add_argument("--force", action="store_true",
                   help="Re-collect rows whose status is already 'done'")
    p.add_argument(
        "--eval-window", type=int, default=5,
        help="Number of trailing eval rows (per family) to average for "
             "loss_main. Default 5: with eval_every_n_steps=5 and max_steps=51 "
             "that's the last 25 steps; with eval_every=50 and max_steps=508 "
             "that's the last 250 steps. The previous implementation used the "
             "single last value (--eval-window=1), which sampled a moving "
             "target for unconverged cells. Set 1 to restore old behavior. "
             "Must be >= 1; 0 or negative values are rejected to avoid "
             "silently averaging the wrong slice.",
    )
    p.add_argument(
        "--since",
        default=None,
        help="Only collect perf.jsonl files modified at or after this time. "
             "Accepts an ISO-8601 timestamp (e.g. '2026-05-29' or "
             "'2026-05-29T13:00:00'), 'now-Nd' / 'now-Nh' relative form, or "
             "a numeric Unix epoch. Stops the stale-perf.jsonl contamination "
             "loop where an old Smoke 4 perf.jsonl in outputs/<cell>/ "
             "overwrites a newer run's loss data on collect. Resolution is "
             "filesystem mtime — the file is included iff mtime >= cutoff.",
    )
    p.add_argument(
        "--sweep-id",
        default=None,
        help="Only collect perf.jsonl files whose startup_param_count event "
             "carries this sweep_id. Complements --since when multiple sweeps "
             "share an outputs/ tree (Stage A round 1 vs. round 2). Skips "
             "files whose startup record has a different sweep_id; files "
             "with no startup record are skipped silently. Compose with "
             "--since for a tight filter.",
    )
    args = p.parse_args(argv)
    if args.eval_window < 1:
        p.error(
            f"--eval-window must be >= 1 (got {args.eval_window}). "
            "Use 1 to restore single-last-value semantics; larger to smooth "
            "noisy unconverged cells."
        )
    # Resolve --since into an epoch float here so the validation error is
    # near the user's argument, not buried in the per-file loop.
    args.since_epoch = _resolve_since(args.since) if args.since else None
    return args


def _resolve_since(s: str) -> float:
    """Parse `--since` into a Unix epoch float.

    Accepts:
      - Numeric (interpreted as Unix epoch seconds)
      - ISO-8601 (e.g. '2026-05-29', '2026-05-29T13:00:00')
      - 'now-7d' / 'now-12h' relative
    Raises SystemExit on unparseable input — better to fail loudly than
    silently include or exclude every file.
    """
    s = s.strip()
    if not s:
        raise SystemExit("ERROR: --since cannot be empty")
    # Numeric epoch
    try:
        return float(s)
    except ValueError:
        pass
    # Relative form: now-Nd / now-Nh
    if s.startswith("now-") and (s.endswith("d") or s.endswith("h")):
        try:
            n = float(s[4:-1])
        except ValueError as err:
            raise SystemExit(f"ERROR: --since {s!r} has non-numeric offset") from err
        secs = n * (86400 if s.endswith("d") else 3600)
        return _dt.datetime.now().timestamp() - secs
    # ISO-8601 — try common shapes
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise SystemExit(
        f"ERROR: --since {s!r}: not a recognized timestamp format. "
        "Accepts: ISO-8601 ('2026-05-29' or '2026-05-29T13:00:00'), "
        "relative ('now-7d', 'now-12h'), or numeric epoch."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs_root = Path(args.outputs).resolve()
    csv_path = Path(args.csv) if args.csv else (
        Path(args.scaling_study_dir) / "experiments.csv"
    )
    if not outputs_root.exists():
        print(f"ERROR: --outputs {outputs_root} not found", file=sys.stderr)
        return 2

    rows = _load_csv(csv_path)
    perf_files = find_perf_files(outputs_root)
    if not perf_files:
        print(f"# isoflop_collect: no perf.jsonl under {outputs_root}", file=sys.stderr)
        return 0

    n_updated = 0
    n_skipped = 0
    n_filtered_since = 0
    n_filtered_sweep = 0
    for path in perf_files:
        # --since filter: drop perf.jsonl files older than the cutoff before
        # we even try to parse them. Cheap pre-filter that closes the stale-
        # data contamination loop where an old Smoke 4 perf.jsonl in
        # outputs/<cell>/2026-05-27/... gets re-collected and overwrites a
        # newer cell's loss data.
        if args.since_epoch is not None:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < args.since_epoch:
                n_filtered_since += 1
                continue
        run_id = _parse_run_id(path, outputs_root)
        if not run_id:
            print(f"# skip {path}: cannot derive run_id", file=sys.stderr)
            continue
        existing = rows.get(run_id, {})
        status = (existing.get("status") or "").strip()
        if status == "done" and not args.force:
            n_skipped += 1
            continue
        records = _read_perf_records(path)
        if not records:
            print(f"# skip {run_id}: empty perf.jsonl", file=sys.stderr)
            continue
        # --sweep-id filter: complement to --since for cases where multiple
        # sweeps share an outputs/ tree. Reads the startup_param_count event's
        # sweep_id (stamped by trainer_native / trainer_zone_a since PR #100).
        if args.sweep_id is not None:
            startup_sweep = None
            for r in records:
                if r.get("event") == "startup_param_count":
                    startup_sweep = r.get("sweep_id")
                    break
            if startup_sweep != args.sweep_id:
                n_filtered_sweep += 1
                continue

        startup = _startup_params(records)
        throughput = _aggregate_throughput(records)
        losses, loss_stability_by_family, loss_source = _last_eval_losses(
            records, window=args.eval_window,
        )
        family = (existing.get("family") or "").strip()
        main_loss_col = _FAMILY_LOSS_COL.get(family)

        # Build the merged row. Use existing values as the base so manual
        # CSV edits (notes, downstream_metric) survive collection.
        merged = dict(existing) if existing else {"run_id": run_id}
        for k, v in startup.items():
            if k in CSV_COLUMNS:
                merged[k] = _fmt(v)
        for k, v in throughput.items():
            if k in CSV_COLUMNS:
                merged[k] = _fmt(v)
        # loss_main = the family-specific loss when we know it (the fit
        # keys off loss_main). Always also write the per-family column.
        for fam, loss in losses.items():
            col = _FAMILY_LOSS_COL.get(fam)
            if col:
                merged[col] = _fmt(loss)
            if col == main_loss_col:
                merged["loss_main"] = _fmt(loss)
                # Surface the per-family stdev as loss_stability for the
                # row's main family — large values flag unconverged cells.
                if fam in loss_stability_by_family:
                    merged["loss_stability"] = _fmt(loss_stability_by_family[fam])
        # PR feedback on PR #98: surface loss_source so the fit can tell
        # held-out eval (trainer_zone_a) from training-loss proxy
        # (trainer_native). Without this, two cells with very different
        # loss semantics would be silently compared in the same parabola.
        if loss_source:
            merged["loss_source"] = loss_source

        merged["status"] = "done"
        merged["completed_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        rows[run_id] = merged
        n_updated += 1

    _write_csv(csv_path, rows)
    summary = (
        f"# isoflop_collect: updated={n_updated} "
        f"skipped(already done)={n_skipped}"
    )
    if args.since_epoch is not None:
        summary += f" filtered(--since)={n_filtered_since}"
    if args.sweep_id is not None:
        summary += f" filtered(--sweep-id)={n_filtered_sweep}"
    summary += f" csv={csv_path}"
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
