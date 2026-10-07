#!/usr/bin/env python3
"""Render completed caption-feature alignment logs without loading models.

Usage: python plot_feature_alignment.py --run-dir RUN [--output-dir RUN/plots]
The report and JSONL logs must agree on finite metrics, fixed cohorts and exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

METRICS = ("content_mse", "content_cosine", "template_mse", "template_cosine")
KINDS = {
    "real_checkpoint_connector_native_feature_alignment",
    "fixture_only_connector_native_feature_alignment",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric(value, name):
    require(
        isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
        f"Nonfinite or nonnumeric {name}",
    )
    if name.endswith("_mse"):
        require(value >= 0, f"Negative {name}")
    if name.endswith("_cosine"):
        require(-1.000001 <= value <= 1.000001, f"Invalid cosine {name}")
    return float(value)


def collect(run_dir):
    root = Path(run_dir).resolve(strict=True)
    sources = {name: root / name for name in ("report.json", "evaluations.jsonl", "steps.jsonl")}
    report = json.loads(sources["report.json"].read_text())
    require(report.get("evidence_kind") in KINDS, "Expected a distinct feature-alignment report")
    require(report.get("status") == "completed", "Run must be completed")
    require(report.get("qualification") == "unqualified", "Unexpected qualification")
    require(
        report.get("training_scope") == ["decoders.image.connector"], "Scope is not connector only"
    )
    objective = report.get("objective", {})
    require(
        objective.get("name") == "native_caption_rmsnorm_content_mse"
        and objective.get("flow_loss_used") is False
        and objective.get("image_quality_established") is False
        and objective.get("template_positions")
        == "excluded from optimization; reported separately",
        "Unsupported objective or unsupported quality claim",
    )
    require(report.get("frozen_state_unchanged") is True, "Frozen-state audit did not pass")
    require(report.get("connector_state_changed") is True, "Connector update was not established")
    selection = report["selection"]
    require(set(selection) == {"train", "validation"}, "Expected train and held-out cohorts")
    cohorts = {split: [row["id"] for row in selection[split]] for split in selection}
    for split, ids in cohorts.items():
        require(ids and len(set(ids)) == len(ids), f"Empty or duplicate {split} cohort")
    require(not set(cohorts["train"]) & set(cohorts["validation"]), "Cohorts overlap")
    evaluations = read_jsonl(sources["evaluations.jsonl"])
    require(evaluations == report.get("evaluations"), "Report and evaluation log disagree")
    completed = report["completed_steps"]
    require(type(completed) is int and completed > 0, "Invalid completed step")
    require(
        completed == report["settings"]["steps"], "Run did not reach its requested terminal step"
    )
    eval_steps = [row["step"] for row in evaluations]
    require(
        eval_steps
        and eval_steps[0] == 0
        and eval_steps[-1] == completed
        and all(type(step) is int for step in eval_steps)
        and all(a < b for a, b in zip(eval_steps, eval_steps[1:], strict=False)),
        "Evaluation steps are not a complete increasing fixed-cohort sequence",
    )
    curves = {split: [] for split in cohorts}
    for evaluation in evaluations:
        require(set(evaluation["splits"]) == set(cohorts), "Evaluation split mismatch")
        for split, detail in evaluation["splits"].items():
            rows = detail["examples"]
            require(
                detail["count"] == len(cohorts[split])
                and [row["id"] for row in rows] == cohorts[split],
                f"Fixed {split} cohort changed at step {evaluation['step']}",
            )
            point = {"step": evaluation["step"], "count": len(rows)}
            for name in METRICS:
                mean = math.fsum(metric(row[name], name) for row in rows) / len(rows)
                recorded = metric(detail[name], name)
                require(
                    math.isclose(mean, recorded, rel_tol=1e-7, abs_tol=1e-10),
                    f"Recorded {split}/{name} mean disagrees with examples",
                )
                point[name] = recorded
            curves[split].append(point)
    steps = read_jsonl(sources["steps.jsonl"])
    require(
        [row["step"] for row in steps] == list(range(1, completed + 1)), "Optimizer steps missing"
    )
    counts = Counter({identifier: 0 for identifier in cohorts["train"]})
    seen = 0
    for row in steps:
        require(len(row["ids"]) == report["settings"]["batch_size"], "Optimizer batch size changed")
        require(set(row["ids"]) <= set(counts), "Optimizer consumed an ID outside training cohort")
        counts.update(row["ids"])
        seen += len(row["ids"])
        require(row["examples_seen"] == seen, "Optimizer exposure counter disagrees with IDs")
        for name in METRICS:
            metric(row[name], name)
    exposure = {
        "examples_seen": seen,
        "unique_examples_seen": sum(count > 0 for count in counts.values()),
        "per_id_counts": dict(counts),
        "selected_equivalent_epochs": seen / len(counts),
    }
    require(exposure == report.get("sample_exposure"), "Report/log sample exposure differs")
    summary = {}
    for split, points in curves.items():
        first, last = points[0], points[-1]
        summary[split] = {
            "count": len(cohorts[split]),
            "initial": first,
            "final": last,
            "content_mse_reduction_percent": 100 * (1 - last["content_mse"] / first["content_mse"])
            if first["content_mse"]
            else None,
        }
    return {
        "schema_version": 1,
        "evidence_kind": report["evidence_kind"],
        "qualification": "unqualified",
        "status": report["status"],
        "completed_steps": completed,
        "run_directory": str(root),
        "source_sha256": {str(path): sha256(path) for path in sources.values()},
        "plotter_sha256": sha256(__file__),
        "training_runner_sha256": report["runner_sha256"],
        "data_fingerprint": report["data_fingerprint"],
        "objective": objective,
        "caption_normalization": report["caption_normalization"],
        "cohorts": cohorts,
        "fixed_cohort_series": curves,
        "optimizer": steps,
        "exposure": exposure,
        "summary": summary,
        "interpretation": {
            "content": "Only caption-content MSE after the frozen DiT RMSNorm is optimized.",
            "template": "Template positions are excluded from the training loss; metrics are diagnostic.",
            "quality": "Feature matching is not evidence of image-generation quality.",
            "weighting": "Fixed-cohort metrics average captions equally, irrespective of token count.",
        },
    }


def render(series, output_dir, *, dpi=180):
    require(type(dpi) is int and dpi >= 72, "DPI must be an integer of at least 72")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prism-feature-plot-") as cache:
        previous = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
        os.environ.update(MPLCONFIGDIR=cache, XDG_CACHE_HOME=cache)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator

            with plt.rc_context(
                {
                    "font.size": 10,
                    "pdf.fonttype": 42,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "savefig.facecolor": "white",
                }
            ):
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.7), sharex=True)
                fig.subplots_adjust(
                    left=0.09, right=0.97, top=0.84, bottom=0.18, hspace=0.36, wspace=0.25
                )
                fixture = series["evidence_kind"].startswith("fixture_only")
                fig.suptitle(
                    "PRISM connector feature alignment" + (" — fixture" if fixture else ""),
                    x=0.09,
                    y=0.968,
                    ha="left",
                    fontsize=18,
                    fontweight="bold",
                )
                train_count = len(series["cohorts"]["train"])
                val_count = len(series["cohorts"]["validation"])
                exposure = series["exposure"]
                fig.text(
                    0.09,
                    0.923,
                    f"{series['completed_steps']:,} optimizer steps | {train_count} train / {val_count} held-out captions | "
                    f"{exposure['examples_seen']:,} exposures ({exposure['selected_equivalent_epochs']:g} passes over the training subset)",
                    fontsize=10,
                )
                fig.text(
                    0.09,
                    0.895,
                    "Only the connector is trained. PRISM, the native teacher, and the original generator remain frozen.",
                    fontsize=9,
                    color="#444444",
                )
                titles = (
                    "A  Caption-content MSE — training objective",
                    "B  Caption-content cosine similarity",
                    "C  Template MSE — excluded from loss",
                    "D  Template cosine — excluded from loss",
                )
                colors = {"train": "#2166AC", "validation": "#D95F02"}
                for axis, name, title in zip(axes.flat, METRICS, titles, strict=True):
                    for split, points in series["fixed_cohort_series"].items():
                        axis.plot(
                            [point["step"] for point in points],
                            [point[name] for point in points],
                            "o-",
                            color=colors[split],
                            markersize=4,
                            linewidth=1.7,
                            label=f"Train (n={train_count})"
                            if split == "train"
                            else f"Held-out (n={val_count})",
                        )
                    axis.set_title(title, loc="left", fontsize=10.5, pad=10)
                    axis.set_ylabel(
                        "Normalized feature MSE"
                        if name.endswith("_mse")
                        else "Mean cosine similarity"
                    )
                    axis.grid(axis="y", alpha=0.2)
                    axis.margins(x=0.04)
                    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                    axis.set_ylim(
                        (-0.03, 1.03)
                        if name.endswith("_cosine")
                        else (
                            0,
                            1.2
                            * max(
                                point[name]
                                for points in series["fixed_cohort_series"].values()
                                for point in points
                            ),
                        )
                    )
                    if name.startswith("template"):
                        axis.set_facecolor("#F7F7F7")
                    else:
                        axis.legend(
                            frameon=False,
                            fontsize=8,
                            loc="lower right" if name.endswith("_cosine") else "upper right",
                        )
                for axis in axes[1]:
                    axis.set_xlabel("Optimizer step")
                fig.text(
                    0.09,
                    0.065,
                    "Features are compared after the frozen DiT caption RMSNorm; fixed-cohort means weight each caption equally.\n"
                    "Template positions are excluded from optimization; their feature similarity is diagnostic only.\n"
                    "Feature matching is not evidence of image-generation quality; paired flow controls and sampled images are separate checks.",
                    fontsize=8.5,
                    color="#444444",
                    linespacing=1.55,
                )
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for text in [*fig.texts, *(axis._left_title for axis in axes.flat)]:
                    bounds = text.get_window_extent(renderer)
                    require(
                        bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width, "Plot text exceeds canvas"
                    )
                paths = {}
                for extension in ("png", "pdf"):
                    path = output / f"feature-alignment.{extension}"
                    fig.savefig(path, dpi=dpi)
                    paths[extension] = str(path)
                plt.close(fig)
            series_path = output / "feature-alignment-series.json"
            series_path.write_text(
                json.dumps(series, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            paths["series"] = str(series_path)
            return paths
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    series = collect(args.run_dir)
    paths = render(series, args.output_dir or args.run_dir / "plots", dpi=args.dpi)
    print(json.dumps({"artifacts": paths, "summary": series["summary"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
