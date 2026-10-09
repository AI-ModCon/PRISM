#!/usr/bin/env python3
"""Render a selected saved overfit step, including explicitly interim snapshots.

Usage: python render_overfit.py RUN_ARTIFACT_DIRECTORY --step 128

Requires the chosen trained step for every gallery case, verifies every recorded
sample/target SHA and per-case starting-noise identity, and never invokes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-overfit-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-overfit-cache"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_image(run, name, expected):
    from PIL import Image, ImageOps

    path = run / name
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or changed image: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def guidance(sample, report):
    value = sample.get("text_guidance_scale", report["settings"].get("text_guidance_scale"))
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Actual text guidance scale is not recorded")
    return float(value)


def collect_cases(run, report, targets, step):
    samples = report.get("samples", [])
    if not samples:
        raise ValueError("The report has no saved gallery cases")
    lookup, cases, latents = {}, {"train": [], "validation": []}, {}
    for sample in samples:
        split, identifier = sample["split"], sample["id"]
        if split not in cases or identifier not in targets or targets[identifier]["split"] != split:
            raise ValueError("Sample ID/split differs from target metadata")
        if sample.get("sampling_target_free") is not True:
            raise ValueError("Sample lacks target-free provenance")
        if sample["prompt"] != targets[identifier]["prompt"]:
            raise ValueError("Sample caption differs from target metadata")
        if identifier not in cases[split]:
            cases[split].append(identifier)
        key = (identifier, sample["stage"], sample["step"])
        if key in lookup:
            raise ValueError("Duplicate case/stage/step in report")
        lookup[key] = sample
        latents.setdefault(identifier, set()).add(sample["initial_latent_sha256"])
        local_image(run, Path(sample["path"]).name, sample["sha256"])
        guidance(sample, report)
    if any(len(values) != 1 for values in latents.values()):
        raise ValueError("Saved stages do not use identical initial latents within each case")
    missing = []
    for identifiers in cases.values():
        for identifier in identifiers:
            for stage, stage_step in (("warm-start", 0), ("trained", step)):
                if (identifier, stage, stage_step) not in lookup:
                    missing.append({"id": identifier, "stage": stage, "step": stage_step})
            local_image(run, "targets/" + identifier + ".jpg", targets[identifier]["image_sha256"])
    if missing:
        raise ValueError(f"Requested step must include every saved gallery case: {missing}")
    return cases, lookup


def render(run, step, output_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    run = Path(run).resolve(strict=True)
    output_dir = Path(output_dir).resolve() if output_dir else run
    output_dir.mkdir(parents=True, exist_ok=True)
    report_raw = (run / "report.json").read_bytes()
    report = json.loads(report_raw)
    if step < 1 or step > report.get("completed_steps", 0):
        raise ValueError("Requested gallery step exceeds recorded optimizer progress")
    if report.get("status") == "completed" and report.get("frozen_state_unchanged") is not True:
        raise ValueError("Completed report does not establish unchanged frozen weights")
    targets = {
        row["id"]: row for row in json.loads((run / "gallery-targets.json").read_text())["records"]
    }
    cases, lookup = collect_cases(run, report, targets, step)
    final = (
        report.get("status") == "completed"
        and step == report["completed_steps"] == report["settings"]["steps"]
    )
    status = "FINAL saved gallery" if final else "INTERIM gallery · not the final result"
    if report.get("evidence_kind") == "fixture_only":
        status = "FIXTURE · layout validation"
    warm = report.get("connector_warm_start") or {}
    joint_init = report.get("joint_stage_initialization") or report["settings"].get(
        "init_joint_checkpoint"
    )
    warm_label = (
        "Stage-start connector + DiT"
        if joint_init
        else "Feature-aligned connector / original DiT"
        if report.get("alignment_initialization")
        else (
            f"Connector step {warm['step']} / original DiT"
            if "step" in warm
            else "Warm-start connector / original DiT"
        )
    )
    native_label = (
        "Native conditioner / stage-start DiT" if joint_init else "Native pretrained OmniGen2"
    )
    result = {
        "report_status": report.get("status"),
        "report_phase": report.get("phase"),
        "report_completed_steps": report.get("completed_steps"),
        "selected_gallery_step": step,
        "gallery_is_final": final,
        "all_image_hashes_verified": True,
        "all_case_initial_latents_identical": True,
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "outputs": {},
    }
    for split, identifiers in cases.items():
        if not identifiers:
            continue
        columns = [
            ("target", 0, "DOCCI target · loss only"),
            ("warm-start", 0, warm_label),
            ("trained", step, f"Connector + DiT at step {step}"),
        ]
        native_available = any(
            (identifier, "native-pretrained", 0) in lookup for identifier in identifiers
        )
        if native_available:
            columns.append(("native-pretrained", 0, native_label))
        height = 4.4 * len(identifiers) + 2.2
        width = 3.5 * len(columns)
        fig = plt.figure(figsize=(width, height), facecolor="white")
        grid = fig.add_gridspec(
            len(identifiers) * 2,
            len(columns),
            height_ratios=[1.0, 0.33] * len(identifiers),
            left=0.035,
            right=0.985,
            top=1 - 1.45 / height,
            bottom=1.02 / height,
            hspace=0.26,
            wspace=0.08,
        )
        title = "Training examples" if split == "train" else "Validation examples"
        fig.suptitle(
            f"DOCCI / PRISM Qwen3-1.7B — {title}",
            x=0.035,
            y=1 - 0.2 / height,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        fig.text(
            0.035,
            1 - 0.67 / height,
            status + f" · report at step {report['completed_steps']} ({report['status']})",
            fontsize=10.5,
            color="#7a4100" if not final else "#333333",
        )
        fig.text(
            0.035,
            1 - 1.02 / height,
            f"{report['settings']['sampling_steps']} denoising steps · identical starting noise per case · caption-only generation",
            fontsize=10,
            color="#333333",
        )
        shown_samples = []
        for row_index, identifier in enumerate(identifiers):
            for column, (stage, stage_step, label) in enumerate(columns):
                axis = fig.add_subplot(grid[2 * row_index, column])
                axis.set_axis_off()
                if stage == "target":
                    image = local_image(
                        run, "targets/" + identifier + ".jpg", targets[identifier]["image_sha256"]
                    )
                    title = label
                else:
                    sample = lookup.get((identifier, stage, stage_step))
                    if sample is None:
                        axis.set_title(label, fontsize=10, pad=8)
                        axis.text(
                            0.5,
                            0.5,
                            "Not sampled for this case",
                            ha="center",
                            va="center",
                            color="#777777",
                            transform=axis.transAxes,
                        )
                        continue
                    image = local_image(run, Path(sample["path"]).name, sample["sha256"])
                    cfg = guidance(sample, report)
                    title = label + f"\nCFG {cfg:g}" + (" · no guidance" if cfg <= 1 else "")
                    shown_samples.append(
                        {
                            "id": identifier,
                            "stage": stage,
                            "step": stage_step,
                            "sha256": sample["sha256"],
                            "text_guidance_scale": cfg,
                            "prompt_format": sample.get(
                                "prompt_format", report["settings"].get("prompt_format", "raw")
                            ),
                            "negative_conditioning": sample.get(
                                "negative_conditioning",
                                report["settings"].get("negative_conditioning", "native"),
                            ),
                        }
                    )
                axis.imshow(
                    image.resize(
                        (report["settings"]["width"], report["settings"]["height"]),
                        Image.Resampling.BICUBIC,
                    )
                )
                axis.set_title(title, fontsize=10, pad=8)
            text_axis = fig.add_subplot(grid[2 * row_index + 1, :])
            text_axis.set_axis_off()
            sample = lookup[(identifier, "trained", step)]
            if split == "validation":
                note = "held out from these optimizer updates"
            elif sample.get("optimized_before_sampling"):
                note = "optimized training example"
            else:
                note = "training-pool example; exposure not established by sample record"
            text_axis.text(
                0,
                1,
                identifier + " · " + note,
                va="top",
                fontsize=10.5,
                fontweight="bold",
                transform=text_axis.transAxes,
            )
            caption = targets[identifier]["prompt"]
            limit = 310 if len(columns) == 4 else 250
            excerpt = caption[:limit].rsplit(" ", 1)[0] + "…" if len(caption) > limit else caption
            text_axis.text(
                0,
                0.58,
                textwrap.fill("Caption excerpt: " + excerpt, 125 if len(columns) == 4 else 93),
                va="top",
                fontsize=9.2,
                transform=text_axis.transAxes,
                linespacing=1.2,
            )
        prompt_format = report["settings"].get("prompt_format", "raw")
        negative = report["settings"].get("negative_conditioning", "native")
        fig.text(
            0.035,
            0.67 / height,
            f"PRISM prompt format: {prompt_format}. Negative conditioner: {negative} (unused when CFG ≤ 1).",
            fontsize=9,
            color="#444444",
        )
        fig.text(
            0.035,
            0.27 / height,
            "Targets: Google DOCCI (CC BY 4.0). Qualitative generation diagnostic; no reconstruction-accuracy or benchmark claim.",
            fontsize=8.3,
            color="#444444",
        )
        path = output_dir / f"{split}-comparison-step-{step:06d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        result["outputs"][split] = {
            "path": str(path),
            "sha256": sha256(path),
            "cases": identifiers,
            "samples": shown_samples,
        }
    (output_dir / f"overfit-render-step-{step:06d}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.step, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
