#!/usr/bin/env python3
"""Plot audited connector-pilot loss logs with consistent fixed probe cohorts.

Usage:
  python tools/plot_prism_image_connector.py --run-dir RUN --output-dir RUN/plots
  python tools/plot_prism_image_connector.py --run-dir FIRST --run-dir RESUME \
      --output-dir COMBINED_PLOTS

The stochastic optimizer loss has its own panel. Train and validation probe
curves share a scale and retain the initial probe IDs/seeds throughout. A larger
final validation cohort is shown separately, never spliced into a probe curve.
PNG, PDF, and a JSON provenance/series artifact are written without model loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path):
    if not path.is_file():
        return []
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Incomplete or invalid JSON log at {path}:{number}") from error
    return result


def _finite(value, context):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {context}")
    return float(value)


def _step(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Optimizer step must be a nonnegative integer")
    return value


def _average(values):
    return math.fsum(values) / len(values)


def _probe_rows(detail, split, step):
    rows = detail.get("examples", [])
    if not rows or detail.get("count") != len(rows):
        raise ValueError(f"Probe count mismatch for {split} at step {step}")
    ids = set()
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError(f"Duplicate or absent probe ID for {split} at step {step}")
        ids.add(row["id"])
        if not isinstance(row.get("seed"), int) or not row.get("shuffled_prompt_id"):
            raise ValueError("Fixed probe needs its seed and wrong-caption ID")
        correct = _finite(row.get("loss"), "probe loss")
        wrong = _finite(row.get("shuffled_loss"), "wrong-caption loss")
        gap = _finite(row.get("shuffled_minus_correct"), "caption gap")
        if not math.isclose(wrong - correct, gap, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("Recorded caption gap disagrees with its paired losses")
    for key, field in (
        ("mean_loss", "loss"),
        ("mean_shuffled_loss", "shuffled_loss"),
        ("mean_shuffled_minus_correct", "shuffled_minus_correct"),
    ):
        if not math.isclose(
            _finite(detail.get(key), key),
            _average([row[field] for row in rows]),
            rel_tol=1e-7,
            abs_tol=1e-10,
        ):
            raise ValueError(f"Recorded probe aggregate {key} disagrees with examples")
    return rows


def _point(step, rows):
    return {
        "step": step,
        "count": len(rows),
        "mean_loss": _average([row["loss"] for row in rows]),
        "mean_shuffled_loss": _average([row["shuffled_loss"] for row in rows]),
        "mean_shuffled_minus_correct": _average([row["shuffled_minus_correct"] for row in rows]),
        "optimized_count": sum(row.get("optimized_before_evaluation") is True for row in rows),
    }


def collect_series(run_dirs):
    """Load a single run or its explicit resume chain, without reading images."""
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    runs, sources, step_rows, evaluation_rows = [], {}, {}, {}
    protocol = None
    for directory in run_dirs:
        root = Path(directory).resolve(strict=True)
        report_path = root / "report.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("evidence_kind")
            not in {
                "real_checkpoint_connector_webdataset_pilot",
                "real_checkpoint_connector_diffusion_webdataset_pilot",
                "fixture_only",
            }
            or report.get("qualification") != "unqualified"
        ):
            raise ValueError("Expected an explicitly unqualified WebDataset pilot report")
        if not report.get("resume_protocol"):
            raise ValueError("Pilot report is missing its run/data/runtime protocol")
        if protocol is None:
            protocol = report["resume_protocol"]
        elif protocol != report["resume_protocol"]:
            raise ValueError("Run protocols differ; do not merge separate experiments")
        runs.append(
            {
                "path": str(root),
                "status": report.get("status"),
                "completed_steps": report.get("completed_steps"),
                "evidence_kind": report["evidence_kind"],
            }
        )
        for name in ("report.json", "steps.jsonl", "evaluations.jsonl"):
            path = root / name
            if path.is_file():
                sources[str(path)] = _sha256(path)
        for row in _jsonl(root / "steps.jsonl"):
            step = _step(row.get("step"))
            if not step:
                raise ValueError("Optimizer loss cannot be recorded before the first update")
            _finite(row.get("loss"), "optimizer loss")
            seen = row.get("examples_seen")
            if not isinstance(seen, int) or seen < 1:
                raise ValueError("Optimizer record lacks actual examples_seen")
            if step in step_rows and step_rows[step] != row:
                raise ValueError(f"Conflicting optimizer records at step {step}")
            step_rows[step] = row
        evaluations = _jsonl(root / "evaluations.jsonl")
        if report.get("initial_evaluation"):
            evaluations.append(report["initial_evaluation"])
        for evaluation in evaluations:
            step = _step(evaluation.get("step"))
            if set(evaluation.get("splits", {})) != {"train", "validation"}:
                raise ValueError("Evaluation must record both train and validation splits")
            for split, detail in evaluation["splits"].items():
                _probe_rows(detail, split, step)
            if evaluation.get("full_validation") is True and evaluation["splits"]["validation"][
                "count"
            ] != report.get("validation_count"):
                raise ValueError("Full-validation count disagrees with the dataset size")
            if step in evaluation_rows and evaluation_rows[step] != evaluation:
                raise ValueError(f"Conflicting evaluation records at step {step}")
            evaluation_rows[step] = evaluation
    if not step_rows or not evaluation_rows:
        raise ValueError("No optimizer steps or probe evaluations are available yet")
    steps = [step_rows[key] for key in sorted(step_rows)]
    evaluations = [evaluation_rows[key] for key in sorted(evaluation_rows)]
    for previous, current in zip(steps, steps[1:], strict=False):
        if current["examples_seen"] <= previous["examples_seen"]:
            raise ValueError("Cumulative examples_seen must increase with optimizer step")
    first = evaluations[0]
    cohorts = {
        split: [
            {"id": row["id"], "seed": row["seed"], "shuffled_prompt_id": row["shuffled_prompt_id"]}
            for row in first["splits"][split]["examples"]
        ]
        for split in ("train", "validation")
    }
    fixed = {"train": [], "validation": []}
    full_validation, alternate_validation, missing_cohorts = [], [], []
    for evaluation in evaluations:
        step = evaluation["step"]
        for split in ("train", "validation"):
            rows = evaluation["splits"][split]["examples"]
            by_id = {row["id"]: row for row in rows}
            cohort = cohorts[split]
            available = []
            for original in cohort:
                row = by_id.get(original["id"])
                if row is not None:
                    if any(row[key] != original[key] for key in ("seed", "shuffled_prompt_id")):
                        raise ValueError(
                            f"Fixed probe noise/control changed for {split}/{row['id']}"
                        )
                    available.append(row)
            if len(available) == len(cohort):
                fixed[split].append(_point(step, available))
            else:
                missing_cohorts.append(
                    {
                        "step": step,
                        "split": split,
                        "expected": len(cohort),
                        "present": len(available),
                    }
                )
            if split == "validation":
                if evaluation.get("full_validation") is True:
                    full_validation.append(_point(step, rows))
                elif evaluation.get("final") and set(by_id) != {row["id"] for row in cohort}:
                    alternate_validation.append(_point(step, rows))
    final_report = json.loads((Path(run_dirs[-1]) / "report.json").read_text())
    return {
        "schema_version": 1,
        "training_scope": (
            "connector_and_diffusion"
            if final_report["evidence_kind"]
            == "real_checkpoint_connector_diffusion_webdataset_pilot"
            else "connector"
        ),
        "qualification": "unqualified",
        "runs": runs,
        "source_sha256": sources,
        "plotter_sha256": _sha256(__file__),
        "data_fingerprint": final_report.get("data_fingerprint"),
        "train_count": final_report.get("train_count"),
        "selected_train_count": final_report.get("selected_train_count"),
        "validation_count": final_report.get("validation_count"),
        "cohorts": cohorts,
        "fixed_probes": fixed,
        "full_validation": full_validation,
        "alternate_final_validation": alternate_validation,
        "missing_fixed_cohorts": missing_cohorts,
        "optimizer": [
            {"step": row["step"], "loss": row["loss"], "examples_seen": row["examples_seen"]}
            for row in steps
        ],
        "coverage": {
            "first_logged_step": steps[0]["step"],
            "last_logged_step": steps[-1]["step"],
            "logged_optimizer_steps": len(steps),
            "cumulative_examples_seen": steps[-1]["examples_seen"],
            "observed_unique_train_ids": len(
                {
                    identifier
                    for row in steps
                    for batch in row.get("microbatches", [])
                    for identifier in batch.get("ids", [])
                }
            ),
        },
        "interpretation": {
            "optimization": "Stochastic timestep/noise losses are plotted separately from fixed-cohort probes with RNG reset.",
            "fixed_cohort": "Each curve retains initial IDs, seeds and wrong-caption IDs; the same subset is recomputed from larger final evaluations.",
            "full_validation": "Separate markers use every final validation example; the fixed probe curve retains its original cohort.",
            "gap": "Wrong-caption loss minus matched-caption loss with RNG reset. Actual paired noise and timesteps are unverified by these logs; positive values nominally favor the matched caption. This is not a visual-quality benchmark.",
        },
    }


def _rolling_segments(points, window):
    """Do not smooth across missing optimizer steps in a partial resume history."""
    x, y, pending, previous = [], [], [], None
    for row in points:
        if previous is not None and row["step"] != previous + 1:
            x.append(None)
            y.append(None)
            pending = []
        pending.append(row["loss"])
        pending = pending[-window:]
        x.append(row["step"])
        y.append(_average(pending))
        previous = row["step"]
    return x, y


def _figure_labels(series):
    """Keep exposure counts distinct from unique IDs, including partial histories."""
    joint = series.get("training_scope") == "connector_and_diffusion"
    title = ("Connector + diffusion" if joint else "Connector") + " training diagnostics"
    fixture = any(run["evidence_kind"] == "fixture_only" for run in series["runs"])
    status = "Fixture only" if fixture else "Experimental"
    if series["runs"][-1]["status"] != "completed":
        status += f" ({series['runs'][-1]['status']})"
    coverage = series["coverage"]
    step_label = (
        f"{status} | Optimizer steps {coverage['first_logged_step']:,}–{coverage['last_logged_step']:,} "
        f"({coverage['logged_optimizer_steps']:,} logged)"
    )
    exposure_label = (
        f"Cumulative training exposures: {coverage['cumulative_examples_seen']:,} | "
        f"Unique IDs in shown logs: {coverage['observed_unique_train_ids']:,}"
    )
    pool = []
    if series.get("selected_train_count") is not None:
        pool.append(f"Selected training pool: {series['selected_train_count']:,}")
    if series.get("train_count") is not None:
        pool.append(f"Full training pool: {series['train_count']:,}")
    return title, step_label, exposure_label, " | ".join(pool)


def plot_series(series, output_dir, *, rolling_window=20, dpi=180):
    if rolling_window < 1 or dpi < 72:
        raise ValueError("rolling-window must be positive and dpi at least 72")
    # Isolate the font cache for compute nodes and sandboxed local rendering.
    with tempfile.TemporaryDirectory(prefix="prism-plot-cache-") as cache:
        old_cache = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
        os.environ.update(MPLCONFIGDIR=cache, XDG_CACHE_HOME=cache)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator

            output = Path(output_dir).resolve()
            output.mkdir(parents=True, exist_ok=True)
            colors = {"train": "#2166AC", "validation": "#D95F02", "full": "#7B3294"}
            with plt.rc_context(
                {
                    "font.size": 10,
                    "axes.titlesize": 11,
                    "axes.labelsize": 10,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "pdf.fonttype": 42,
                    "savefig.facecolor": "white",
                }
            ):
                figure, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True)
                figure.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.16, hspace=0.30)
                joint = series.get("training_scope") == "connector_and_diffusion"
                title, step_label, exposure_label, pool_label = _figure_labels(series)
                figure.suptitle(title, x=0.10, y=0.975, ha="left", fontsize=16, fontweight="bold")
                coverage = series["coverage"]
                for y, label in ((0.942, step_label), (0.923, exposure_label), (0.904, pool_label)):
                    figure.text(0.10, y, label, ha="left", fontsize=9)
                figure.text(
                    0.10,
                    0.880,
                    (
                        "Connector and full diffusion transformer are trained. PRISM, VAE and native conditioner stay frozen."
                        if joint
                        else "Only the connector is trained. Loss is flow-matching MSE; sampling quality is not established."
                    ),
                    ha="left",
                    fontsize=9,
                    color="#444444",
                )

                optimizer = series["optimizer"]
                axes[0].plot(
                    [row["step"] for row in optimizer],
                    [row["loss"] for row in optimizer],
                    ".",
                    markersize=2.5,
                    color="#6B7280",
                    alpha=0.45,
                    label="Optimizer-step mean",
                )
                x, y = _rolling_segments(optimizer, rolling_window)
                axes[0].plot(
                    x,
                    y,
                    color="#222222",
                    linewidth=1.6,
                    label=f"Trailing mean, up to {rolling_window} steps",
                )
                axes[0].set_title(
                    "A  Optimization loss — stochastic timestep and noise", loc="left"
                )
                axes[0].set_ylabel("Flow MSE")
                axes[0].legend(loc="best", frameon=False, fontsize=8)

                for split in ("train", "validation"):
                    cohort_size = len(series["cohorts"][split])
                    points = series["fixed_probes"][split]
                    label = f"Fixed {split} cohort (n={cohort_size})"
                    for axis, key in (
                        (axes[1], "mean_loss"),
                        (axes[2], "mean_shuffled_minus_correct"),
                    ):
                        axis.plot(
                            [row["step"] for row in points],
                            [row[key] for row in points],
                            "o-",
                            markersize=4,
                            color=colors[split],
                            linewidth=1.4,
                            label=label,
                        )
                for rows, label, marker in (
                    (series["full_validation"], "Full validation", "D"),
                    (series["alternate_final_validation"], "Final limited validation", "s"),
                ):
                    for count in sorted({row["count"] for row in rows}):
                        selected = [row for row in rows if row["count"] == count]
                        for axis, key in (
                            (axes[1], "mean_loss"),
                            (axes[2], "mean_shuffled_minus_correct"),
                        ):
                            axis.scatter(
                                [row["step"] for row in selected],
                                [row[key] for row in selected],
                                s=64,
                                facecolors="none",
                                edgecolors=colors["full"],
                                marker=marker,
                                linewidths=1.5,
                                zorder=5,
                                label=f"{label} (n={count})",
                            )
                axes[1].set_title("B  Fixed cohorts — matched captions, RNG reset", loc="left")
                axes[1].set_ylabel("Mean flow MSE")
                axes[1].legend(loc="best", frameon=False, fontsize=8)
                axes[2].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
                axes[2].set_title(
                    "C  RNG-reset caption control — paired inputs unverified",
                    loc="left",
                )
                axes[2].set_ylabel("Wrong − matched MSE")
                axes[2].set_xlabel("Optimizer step")
                axes[2].legend(loc="best", frameon=False, fontsize=8)
                for axis in axes:
                    axis.grid(axis="y", alpha=0.2)
                    axis.margins(x=0.04)
                for axis in axes[:2]:
                    # Nearly flat cohorts otherwise leave a final scatter marker
                    # clipped against the upper boundary after anchoring at zero.
                    axis.set_ylim(bottom=0, top=max(axis.get_ylim()[1], axis.dataLim.y1 * 1.08))
                axes[2].set_xlim(left=-max(0.05, coverage["last_logged_step"] * 0.02))
                axes[2].xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=2))
                final_train = series["fixed_probes"]["train"][-1]
                footnote = (
                    f"At the last train probe, {final_train['optimized_count']}/{final_train['count']} examples had reached the optimizer. "
                    "Validation examples never enter optimization.\n"
                    "Full-validation markers show the complete split. Fixed curves retain their original IDs/seeds.\n"
                    "Actual paired noise/timesteps are unverified. Caption gaps do not establish image quality.\n"
                    "Panels A and B use separate y-axis scales."
                )
                if series["missing_fixed_cohorts"]:
                    footnote += "\nSome final smoke evaluations omit the full fixed cohort; those points are not joined to its curve."
                figure.text(
                    0.10, 0.045, footnote, va="bottom", fontsize=8, color="#444444", linespacing=1.5
                )
                paths = {}
                for extension in ("png", "pdf"):
                    stem = "connector-diffusion" if joint else "connector"
                    path = output / f"{stem}-losses.{extension}"
                    figure.savefig(path, dpi=dpi)
                    paths[extension] = str(path)
                plt.close(figure)
            stem = (
                "connector-diffusion"
                if series.get("training_scope") == "connector_and_diffusion"
                else "connector"
            )
            series_path = output / f"{stem}-loss-series.json"
            series_path.write_text(
                json.dumps(series, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            paths["series"] = str(series_path)
            return paths
        finally:
            for key, value in old_cache.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat in chronological order to include previous resumed logs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    paths = plot_series(
        collect_series(args.run_dir),
        args.output_dir,
        rolling_window=args.rolling_window,
        dpi=args.dpi,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
