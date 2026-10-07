#!/usr/bin/env python3
"""Glob `perf.jsonl` across PRISM `outputs/` and emit a tidy CSV.

Usage:
  python tools/perf_aggregate.py outputs/                       # → stdout
  python tools/perf_aggregate.py outputs/ -o scaling.csv        # → file
  python tools/perf_aggregate.py outputs/ --filter site=trainer_native_per_50

The output CSV has one row per JSON line; columns are the union of all
keys observed across all files, with a `run_dir` column carrying the
relative path of the perf.jsonl's parent (so multiple runs can be
distinguished after aggregation).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def find_perf_files(root: Path) -> list[Path]:
    """Return every `perf.jsonl` under `root` (recursive)."""
    if root.is_file() and root.name == "perf.jsonl":
        return [root]
    return sorted(root.rglob("perf.jsonl"))


def iter_records(paths: Iterable[Path], root: Path) -> Iterable[dict[str, Any]]:
    for path in paths:
        run_dir = str(path.parent.relative_to(root)) if path.parent != root else "."
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"WARN: {path}:{line_no} skipped (not JSON): {e}",
                        file=sys.stderr,
                    )
                    continue
                rec["run_dir"] = run_dir
                yield rec


def parse_filters(filter_args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in filter_args:
        if "=" not in f:
            raise SystemExit(f"--filter expects key=value, got {f!r}")
        k, v = f.split("=", 1)
        out[k] = v
    return out


def apply_filters(record: dict[str, Any], filters: dict[str, str]) -> bool:
    for k, v in filters.items():
        if str(record.get(k)) != v:
            return False
    return True


def write_csv(records: list[dict[str, Any]], out) -> int:
    if not records:
        return 0
    columns: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k)
                columns.append(k)
    # Bubble run_dir to column 0 so each row's source is the first thing the reader sees.
    if "run_dir" in seen:
        columns.remove("run_dir")
        columns.insert(0, "run_dir")
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow(r)
    return len(records)


def _per_modality_aggregate(
    records: list[dict[str, Any]],
    sweep_id: str | None,
    warmup: int = 10,
) -> list[dict[str, Any]]:
    """Per-preset rollup for the Phase 2 sweep.

    Filters to throughput records (those with samples_per_sec), groups by
    preset, drops the first `warmup` records per (run_dir, preset) so warmup
    noise doesn't poison the mean. Pulls dataloader_modalities /
    model_modalities from the matching `event=startup_modality_check`
    record so each row self-validates.

    Output columns per preset:
        preset, n_records, samples_per_sec_{mean,median,std},
        tokens_per_sec_{mean,median,std}, tokens_per_batch_mean,
        batch_modality_counts (most-frequent value),
        dataloader_modalities, model_modalities, mismatch
    """
    if sweep_id is not None:
        records = [r for r in records if r.get("sweep_id") == sweep_id]

    # Startup check map: (run_dir, preset) -> startup record. If a run
    # restarts with strict mode disabled we'd see multiple startup records
    # for the same key; this keeps the most recent one (last write wins).
    startup_by_key = {}
    for r in records:
        if r.get("event") == "startup_modality_check":
            key = (r.get("run_dir", ""), r.get("preset"))
            startup_by_key[key] = r

    # Group throughput records by preset, with warmup drop per run_dir.
    by_preset: dict[str, list[dict[str, Any]]] = {}
    perf_by_run: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("samples_per_sec") is None:
            continue
        key = (r.get("run_dir", ""), r.get("preset"))
        perf_by_run.setdefault(key, []).append(r)

    for (_run_dir, preset), recs in perf_by_run.items():
        recs_sorted = sorted(recs, key=lambda x: x.get("step", 0))
        post_warmup = recs_sorted[warmup:] if len(recs_sorted) > warmup else recs_sorted
        by_preset.setdefault(str(preset), []).extend(post_warmup)

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        if not vals:
            return (float("nan"),) * 3
        mean = statistics.fmean(vals)
        med = statistics.median(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, med, std

    rows = []
    for preset, recs in sorted(by_preset.items()):
        sps = [float(r["samples_per_sec"]) for r in recs if r.get("samples_per_sec") is not None]
        tps = [float(r["tokens_per_sec"]) for r in recs if r.get("tokens_per_sec") is not None]
        tpb = [float(r["tokens_per_batch"]) for r in recs if r.get("tokens_per_batch") is not None]

        sps_mean, sps_med, sps_std = _stats(sps)
        tps_mean, tps_med, tps_std = _stats(tps)
        tpb_mean, _, _ = _stats(tpb)

        # Pull the startup-check record for any run_dir under this preset.
        startup = None
        for r in recs:
            key = (r.get("run_dir", ""), preset)
            if key in startup_by_key:
                startup = startup_by_key[key]
                break

        # Most frequent batch_modality_counts seen across the post-warmup window.
        counts_seen = [
            json.dumps(r.get("batch_modality_counts") or {}, sort_keys=True)
            for r in recs
            if r.get("batch_modality_counts") is not None
        ]
        bmc_str = Counter(counts_seen).most_common(1)[0][0] if counts_seen else ""

        rows.append(
            {
                "preset": preset,
                "n_records": len(recs),
                "samples_per_sec_mean": sps_mean,
                "samples_per_sec_median": sps_med,
                "samples_per_sec_std": sps_std,
                "tokens_per_sec_mean": tps_mean,
                "tokens_per_sec_median": tps_med,
                "tokens_per_sec_std": tps_std,
                "tokens_per_batch_mean": tpb_mean,
                "batch_modality_counts": bmc_str,
                "dataloader_modalities": json.dumps(
                    startup.get("dataloader_modalities") if startup else []
                ),
                "model_modalities": json.dumps(
                    startup.get("model_modalities") if startup else []
                ),
                "mismatch": (startup.get("mismatch") if startup else None),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("root", type=Path, help="Directory to scan (e.g. outputs/)")
    parser.add_argument("-o", "--output", type=Path, help="CSV output path (default: stdout)")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="key=value filter (repeatable). e.g. --filter site=trainer_native_per_50",
    )
    parser.add_argument(
        "--per-modality",
        action="store_true",
        help="Emit a per-preset rollup (Phase 2 sweep). Use with --sweep-id.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Restrict records to this sweep_id (recommended with --per-modality).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Drop the first N throughput records per run when --per-modality is set (default: 10).",
    )
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"ERROR: {args.root} does not exist", file=sys.stderr)
        return 1

    paths = find_perf_files(args.root)
    if not paths:
        print(f"WARN: no perf.jsonl files found under {args.root}", file=sys.stderr)
        return 0

    filters = parse_filters(args.filter)
    records = [r for r in iter_records(paths, args.root) if apply_filters(r, filters)]

    if args.per_modality:
        rows = _per_modality_aggregate(records, args.sweep_id, warmup=args.warmup)
        if args.output:
            with open(args.output, "w", newline="") as f:
                n = write_csv(rows, f)
            print(
                f"wrote {n} per-preset rows from {len(paths)} files → {args.output}",
                file=sys.stderr,
            )
        else:
            n = write_csv(rows, sys.stdout)
            print(f"# {n} per-preset rows from {len(paths)} files", file=sys.stderr)
        return 0

    if args.output:
        with open(args.output, "w", newline="") as f:
            n = write_csv(records, f)
        print(f"wrote {n} rows from {len(paths)} files → {args.output}", file=sys.stderr)
    else:
        n = write_csv(records, sys.stdout)
        print(f"# {n} rows from {len(paths)} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
