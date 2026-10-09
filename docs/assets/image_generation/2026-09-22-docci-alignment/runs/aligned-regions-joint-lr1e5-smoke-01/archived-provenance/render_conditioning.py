#!/usr/bin/env python3
"""Render verified diagnostic artifacts; never invokes or edits model outputs.

Usage: python render_conditioning.py RUN_ARTIFACT_DIRECTORY

Produces separate raw/chat five-column comparisons. Missing interim samples are
labeled, never substituted. Targets are shown only for visual reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-conditioning-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-conditioning-cache"))


def verified_image(path, expected_sha):
    from PIL import Image, ImageOps

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Image SHA256 mismatch: {path}")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def render(run, output_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run).resolve(strict=True)
    output_dir = Path(output_dir).resolve() if output_dir else run
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    if (
        "conditioning_diagnostic" not in report.get("evidence_kind", "")
        and report.get("evidence_kind") != "fixture_only"
    ):
        raise ValueError("Conditioning renderer requires a conditioning diagnostic report")
    gallery = json.loads((run / "gallery-targets.json").read_text())
    targets = {row["id"]: row for row in gallery["records"]}
    cases = sorted(
        case for case in gallery.get("case_targets", {}) if case.startswith("validation-")
    )[:2]
    if not cases:
        raise ValueError("No collected validation gallery cases yet")
    lookup = {}
    for sample in report["samples"]:
        key = (sample["case_id"], sample["route"])
        if key in lookup:
            raise ValueError("Duplicate case/route in conditioning gallery")
        lookup[key] = sample
    for case in cases:
        values = {
            sample["initial_latent_sha256"]
            for (case_id, _), sample in lookup.items()
            if case_id == case
        }
        if len(values) != 1:
            raise ValueError("Gallery variants do not share identical starting noise")
    adapted = bool(report.get("joint_checkpoint"))
    native_route = "native_adapted_diffusion" if adapted else "native_pretrained"
    native_label = (
        "Native conditioner / adapted DiT\nCFG 5"
        if adapted
        else "Native pretrained OmniGen2\nCFG 5"
    )
    modes = report.get("settings", {}).get("prism_formats", ["raw", "chat"])
    outputs = {}
    for mode in modes:
        if mode not in ("raw", "chat"):
            raise ValueError("Unknown PRISM prompt format")
        last_label = "PRISM positive + PRISM negative\nCFG 5"
        columns = [
            ("target", "DOCCI target\nvisual reference only"),
            (native_route, native_label),
            (
                f"prism_{mode}_native_negative_cfg5",
                "PRISM positive + native negative\nCFG 5 · current route",
            ),
            (f"prism_{mode}_cfg1", "PRISM positive\nCFG 1 · no guidance"),
            (f"prism_{mode}_prism_negative_cfg5", last_label),
        ]
        height = 4.4 * len(cases) + 2.0
        fig = plt.figure(figsize=(17.0, height), facecolor="white")
        grid = fig.add_gridspec(
            len(cases) * 2,
            5,
            height_ratios=[1.0, 0.32] * len(cases),
            left=0.025,
            right=0.985,
            top=1 - 1.25 / height,
            bottom=1.04 / height,
            hspace=0.24,
            wspace=0.08,
        )
        status = "completed" if report["status"] == "completed" else "INTERIM · " + report["status"]
        if report.get("evidence_kind") == "fixture_only":
            status = "FIXTURE · layout validation"
        fig.suptitle(
            f"DOCCI conditioning audit — {mode} PRISM prompts",
            x=0.025,
            y=1 - 0.2 / height,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        settings = report["settings"]
        detail = f"{settings['sampling_steps']} diffusion steps · {settings['width']} × {settings['height']} · identical starting noise within each case · {status}"
        fig.text(0.025, 1 - 0.73 / height, detail, fontsize=10.5, color="#333333")
        for row_index, case in enumerate(cases):
            target = targets[gallery["case_targets"][case]["id"]]
            for column, (route, title) in enumerate(columns):
                axis = fig.add_subplot(grid[2 * row_index, column])
                axis.set_axis_off()
                axis.set_title(title, fontsize=10.5, pad=8, color="#222222")
                if route == "target":
                    image = verified_image(
                        run / "targets" / (target["id"] + ".jpg"), target["image_sha256"]
                    )
                else:
                    sample = lookup.get((case, route))
                    if sample is None:
                        axis.set_facecolor("#f1f3f4")
                        axis.text(
                            0.5,
                            0.5,
                            "Not collected yet"
                            if report["status"] != "completed"
                            else "Not sampled",
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="#666666",
                            transform=axis.transAxes,
                        )
                        continue
                    if sample.get("target_free") is not True:
                        raise ValueError("Sample lacks target-free generation provenance")
                    image = verified_image(run / Path(sample["path"]).name, sample["sha256"])
                # Match the training target's square resize for a fair spatial display.
                axis.imshow(image.resize((settings["width"], settings["height"])))
            caption_axis = fig.add_subplot(grid[2 * row_index + 1, :])
            caption_axis.set_axis_off()
            caption_axis.text(
                0,
                1,
                target["id"] + " · held out from optimizer updates",
                va="top",
                fontsize=11,
                fontweight="bold",
                transform=caption_axis.transAxes,
            )
            caption = target["prompt"]
            excerpt = caption[:330].rsplit(" ", 1)[0] + "…" if len(caption) > 330 else caption
            caption_axis.text(
                0,
                0.57,
                textwrap.fill("Caption excerpt: " + excerpt, 155),
                va="top",
                fontsize=9.2,
                transform=caption_axis.transAxes,
                linespacing=1.25,
            )
        anchor = report.get("negative_condition_statistics", {}).get(mode, {}).get("empty_anchor")
        if mode == "raw" and anchor == "eos":
            note = "PRISM raw negative = one EOS token. This anchor was not trained as unconditional conditioning."
        elif mode == "chat":
            note = "PRISM chat negative = a formatted empty user message. This route was not trained as unconditional conditioning."
        else:
            note = "PRISM negative conditioning is an experimental anchor; this diagnostic does not establish unconditional training."
        fig.text(0.025, 0.64 / height, note, fontsize=10, color="#7a4100")
        fig.text(
            0.025,
            0.28 / height,
            "Targets: Google DOCCI (CC BY 4.0). Generated images use captions alone. Qualitative diagnostic; no accuracy or benchmark claim.",
            fontsize=9,
            color="#444444",
        )
        path = output_dir / f"conditioning-{mode}-comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs[mode] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {
        "report_status": report["status"],
        "cases": cases,
        "native_route": native_route,
        "outputs": outputs,
        "rendering": "PIL display resizing and Matplotlib layout; original generated files are unchanged.",
    }
    (output_dir / "conditioning-render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
