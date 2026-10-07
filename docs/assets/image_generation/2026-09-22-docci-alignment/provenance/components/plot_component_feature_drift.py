#!/usr/bin/env python3
"""Reproduce the component feature-drift chart from a collected report only.

Requires Matplotlib and NumPy. Uses matched captions once per connector; wrong
captions and duplicate DiT phases are excluded. No model or image is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def aggregate(report):
    if (
        report.get("status") != "completed"
        or report.get("evidence_kind") != "real_checkpoint_joint_component_diagnostic"
        or report.get("training_performed") is not False
        or report.get("joint_checkpoint", {}).get("step") != 6
    ):
        raise ValueError("Expected a completed frozen component diagnostic report")
    rows = report["feature_drift"]
    feature = {}
    for split in ("train", "validation"):
        selected = {row["id"] for row in report["selection"][split]}
        if len(selected) != 2:
            raise ValueError("Chart requires exactly two matched captions per split")
        feature[split] = {}
        for connector in ("aligned", "joint"):
            phases = {}
            for dit in ("original", "joint"):
                chosen = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["connector"] == connector
                    and row["dit"] == dit
                    and row["caption_kind"] == "matched"
                ]
                if len(chosen) != 2 or {row["id"] for row in chosen} != selected:
                    raise ValueError("Missing or duplicated matched-caption feature rows")
                if any(
                    row["normalization"] != "captured_original_dit_rmsnorm"
                    or row["caption_id"] != row["id"]
                    for row in chosen
                ):
                    raise ValueError("Feature rows use a different normalization or caption")
                phases[dit] = {row["id"]: row["partitions"] for row in chosen}
            if phases["original"] != phases["joint"]:
                raise ValueError("Original-norm connector metrics changed across DiT phases")
            feature[split][connector] = {}
            for region in ("prefix", "content", "suffix"):
                values = [row[region]["mse"] for row in phases["original"].values()]
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError("Log-scale chart requires finite positive feature MSE")
                feature[split][connector][region] = statistics.mean(values)
        if any(
            feature[split]["joint"][region] <= feature[split]["aligned"][region]
            for region in ("prefix", "content", "suffix")
        ):
            raise ValueError("This chart's drift title requires all three regions to worsen")
    return feature


def render(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = Path(run).resolve(strict=True)
    report_path = run / "report.json"
    feature = aggregate(json.loads(report_path.read_text()))
    regions = ["prefix", "content", "suffix"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), dpi=150, sharey=True)
    for ax, (split, title) in zip(
        axes, [("train", "Two training captions"), ("validation", "Two held-out captions")]
    ):
        x = np.arange(3)
        for dx, connector, label, color in (
            (-0.18, "aligned", "Aligned connector", "#187ca6"),
            (0.18, "joint", "After six joint updates", "#b35806"),
        ):
            values = [feature[split][connector][region] for region in regions]
            bars = ax.bar(x + dx, values, 0.33, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.12,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_yscale("log")
        ax.set_ylim(0.0003, 0.14)
        ax.set_xticks(x, regions)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", which="major", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MSE to native features (log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.09),
        fontsize=9,
        frameon=False,
        ncol=2,
    )
    fig.suptitle(
        "Connector feature alignment drifts after six joint updates", fontsize=14, y=0.98
    )
    fig.text(
        0.5,
        0.87,
        "All measurements use the same original DiT RMSNorm; equal weight per caption.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.035,
        "Prefix, content and suffix all worsen. Four matched captions total; "
        "feature MSE does not measure image quality.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.09, right=0.98, wspace=0.12)
    output = run / "feature-drift.png"
    fig.savefig(output)
    plt.close(fig)
    return {
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "matched_caption_mse": feature,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Collected joint-components run directory")
    print(json.dumps(render(parser.parse_args().run), indent=2))
