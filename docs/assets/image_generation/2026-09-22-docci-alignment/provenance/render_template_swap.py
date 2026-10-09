#!/usr/bin/env python3
"""Audit and render completed teacher-assisted template-swap diagnostics.

Usage: python render_template_swap.py RUN_ARTIFACT_DIRECTORY [--audit-only]

No model is imported. Recorded tensor-partition proofs, file hashes, case/route
coverage and initial-noise replay are checked before any output is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-template-swap-mpl"))
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-template-swap-cache")
)

ROUTES = (
    "native_pretrained",
    "prism_aligned",
    "prism_native_prefix",
    "prism_native_suffix",
    "prism_native_template",
)
SOURCES = {
    "native_pretrained": {"prefix": "native", "content": "native", "suffix": "native"},
    "prism_aligned": {"prefix": "aligned", "content": "aligned", "suffix": "aligned"},
    "prism_native_prefix": {"prefix": "native", "content": "aligned", "suffix": "aligned"},
    "prism_native_suffix": {"prefix": "aligned", "content": "aligned", "suffix": "native"},
    "prism_native_template": {"prefix": "native", "content": "aligned", "suffix": "native"},
}


def require(condition, message):
    if not condition:
        raise ValueError("Template-swap artifact: " + message)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_in_run(run, name):
    path = (run / name).resolve()
    require(run in path.parents and path.is_file(), "missing or unsafe image path")
    return path


def audit(run):
    run = Path(run).resolve(strict=True)
    report_raw = (run / "report.json").read_bytes()
    report = json.loads(report_raw)
    require(
        report.get("status") == "completed",
        "completed report required; partial galleries are refused",
    )
    require(
        report.get("evidence_kind") in {"real_checkpoint_conditioning_diagnostic", "fixture_only"},
        "conditioning diagnostic evidence required",
    )
    require(report.get("training_performed") is False, "this must be a frozen diagnostic")
    settings, protocol = report["settings"], report.get("template_swap_protocol", {})
    require(
        all(
            type(settings.get(key)) is int and settings[key] > 0
            for key in ("width", "height", "sampling_steps", "sample_count")
        ),
        "invalid image or sampling settings",
    )
    require(
        settings.get("template_swap_controls") is True, "template swap controls were not requested"
    )
    require(settings.get("prism_formats") == ["chat"], "aligned chat conditioning required")
    require(
        not report.get("joint_checkpoint") and not settings.get("joint_checkpoint"),
        "adapted joint DiT is not this experiment",
    )
    require(protocol.get("routes") == list(ROUTES), "template route contract differs")
    require(protocol.get("text_guidance_scale") == 1.0, "all template routes must use CFG 1")
    require(
        protocol.get("swap_stage") == "unnormalized_before_dit_rmsnorm",
        "swap location must precede DiT RMSNorm",
    )
    require(
        protocol.get("teacher_assisted") is True
        and protocol.get("teacher_cache_persisted") is False,
        "teacher assistance/cache policy missing",
    )
    lineage = report.get("alignment_checkpoint", {})
    require(
        lineage.get("restore_policy") == "connector_weights_only_from_native_feature_alignment",
        "strict alignment restoration missing",
    )
    require(
        lineage.get("prompt_format") == "chat"
        and lineage.get("exact_token_and_content_audit_revalidated") is True,
        "alignment chat/token audit missing",
    )
    require(
        lineage.get("frozen_state_unchanged") is True
        and lineage.get("reference_checkpoint_sha256")
        == report.get("generator", {}).get("manifest_sha256"),
        "original pretrained generator identity not established",
    )
    require(
        report.get("native_pretrained_state_unchanged") is True,
        "native baseline changed frozen weights",
    )
    require(
        report.get("frozen_state_unchanged") is True
        and bool(report.get("frozen_hashes_before"))
        and report.get("frozen_hashes_before") == report.get("frozen_hashes_after"),
        "diagnostic changed a frozen tensor",
    )
    selection = report["selection"]
    require(
        set(selection) == {"train", "validation"} and all(selection.values()),
        "both training and heldout cohorts are required",
    )
    require(
        all(
            len({row["id"] for row in rows}) == len(rows)
            and len({row["index"] for row in rows}) == len(rows)
            for rows in selection.values()
        ),
        "duplicate selected ID or index",
    )
    require(
        not {row["id"] for row in selection["train"]}
        & {row["id"] for row in selection["validation"]},
        "training and heldout IDs overlap",
    )
    expected_stats = {
        (split, row["id"], kind)
        for split, rows in selection.items()
        for row in rows
        for kind in ("matched", "wrong")
    }
    statistics = report.get("template_swap_statistics", [])
    actual_stats = [(row["split"], row["id"], row["caption_kind"]) for row in statistics]
    require(
        len(actual_stats) == len(set(actual_stats)) and set(actual_stats) == expected_stats,
        "matched/wrong template audits do not cover every selected target",
    )
    selected_indexes = {
        (split, row["id"]): row["index"] for split, rows in selection.items() for row in rows
    }
    stats_lookup = {(row["split"], row["id"], row["caption_kind"]): row for row in statistics}
    for row in statistics:
        require(
            row["index"] == selected_indexes[(row["split"], row["id"])],
            "template audit index differs from selection",
        )
        require(
            (row["caption_id"] == row["id"]) == (row["caption_kind"] == "matched"),
            "matched/wrong caption identity is ambiguous",
        )
        require(
            all(
                row.get(key) is True
                for key in (
                    "actual_native_forward_inputs_verified",
                    "exact_input_ids_match",
                    "exact_input_masks_match",
                    "exact_formatted_input_match",
                )
            ),
            "actual native input/token identity audit missing",
        )
        require(row.get("target_pixels_read") is False, "template features must be caption-only")
        require(
            digest(row.get("input_ids_sha256")) and digest(row.get("input_mask_sha256")),
            "token/mask digests missing",
        )
        counts, spans = row["partition_token_counts"], row["partition_spans"]
        require(
            set(counts) == set(spans) == {"prefix", "content", "suffix"},
            "template partition names differ",
        )
        require(
            all(type(counts[key]) is int and counts[key] > 0 for key in counts),
            "all three token partitions must be nonempty",
        )
        start, end = row["content_span"]
        require(
            spans
            == {
                "prefix": [0, start],
                "content": [start, end],
                "suffix": [end, sum(counts.values())],
            },
            "partition spans do not cover exactly prefix/content/suffix",
        )
        require(
            all(spans[key][1] - spans[key][0] == counts[key] for key in counts),
            "partition counts disagree with spans",
        )
        routes = row["routes"]
        require(set(routes) == set(ROUTES), "per-caption route audit incomplete")
        for route in ROUTES:
            detail = routes[route]
            require(
                detail.get("region_sources") == SOURCES[route],
                "hybrid substitutes an unexpected region",
            )
            require(
                digest(detail.get("embeds_sha256")) and digest(detail.get("attention_mask_sha256")),
                "route embedding/mask digest missing",
            )
            require(
                detail["attention_mask_sha256"]
                == routes["native_pretrained"]["attention_mask_sha256"],
                "a hybrid changed its attention mask",
            )
            for region, source in SOURCES[route].items():
                reference = routes["native_pretrained" if source == "native" else "prism_aligned"]
                region_hash = detail["partitions"][region].get("unnormalized_sha256")
                require(
                    digest(region_hash)
                    and region_hash == reference["partitions"][region].get("unnormalized_sha256"),
                    "hybrid region differs from recorded source partition",
                )
    timesteps = settings.get("flow_timesteps", [])
    require(
        bool(timesteps)
        and all(
            type(value) in (float, int) and math.isfinite(value) and 0 < value < 1
            for value in timesteps
        ),
        "invalid requested flow timesteps",
    )
    flow = report.get("flow_controls", [])
    expected_flow = {
        (split, row["id"], repeat)
        for split, rows in selection.items()
        for row in rows
        for repeat in range(len(timesteps))
    }
    actual_flow = [(row["split"], row["id"], row["repeat"]) for row in flow]
    require(
        len(actual_flow) == len(set(actual_flow)) and set(actual_flow) == expected_flow,
        "paired flow controls do not cover every selected target and timestep",
    )
    for row in flow:
        require(
            row["index"] == selected_indexes[(row["split"], row["id"])],
            "flow index differs from selection",
        )
        require(
            row.get("requested_timestep") == timesteps[row["repeat"]],
            "flow repeat differs from requested timestep",
        )
        require(
            row.get("actual_conditioning_verified") is True,
            "actual DiT conditioning boundary was not verified",
        )
        actual = row.get("actual_inputs", {})
        require(
            digest(actual.get("noisy_latent_sha256")) and digest(actual.get("timestep_sha256")),
            "actual shared noisy input/timestep digests missing",
        )
        require(
            isinstance(actual.get("timestep"), list)
            and len(actual["timestep"]) == 1
            and type(actual["timestep"][0]) in (float, int)
            and math.isfinite(actual["timestep"][0]),
            "actual flow timestep missing",
        )
        require(set(row["routes"]) == set(ROUTES), "paired flow route coverage differs")
        for route, values in row["routes"].items():
            for kind in ("matched", "wrong"):
                require(
                    type(values.get(kind)) in (float, int)
                    and math.isfinite(values[kind])
                    and values[kind] >= 0,
                    "invalid flow loss",
                )
                caption = stats_lookup[(row["split"], row["id"], kind)]
                require(
                    kind != "wrong" or row["wrong_id"] == caption["caption_id"],
                    "flow wrong caption differs from token audit",
                )
                expected = {
                    key: caption["routes"][route][key]
                    for key in ("embeds_sha256", "attention_mask_sha256")
                }
                require(
                    values.get(kind + "_condition") == expected,
                    "actual flow conditioning differs from recorded template route",
                )
            require(
                math.isclose(
                    values.get("wrong_minus_matched", math.nan),
                    values["wrong"] - values["matched"],
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ),
                "flow caption gap does not match losses",
            )
    # Sample routes are separate from flow controls; original CFG 5 may also be saved.
    expected_cases = {f"validation-{i:02d}" for i in range(settings["sample_count"])}
    require(
        expected_cases and len(expected_cases) <= len(selection["validation"]),
        "invalid or empty gallery cohort",
    )
    lookup, initial_latents, seeds = {}, {}, {}
    for sample in report.get("samples", []):
        key = (sample["case_id"], sample["route"])
        require(key not in lookup, "duplicate generation case/route")
        lookup[key] = sample
        path = file_in_run(run, Path(sample["path"]).name)
        require(sha256(path) == sample["sha256"], "generated image digest differs")
        require(sample.get("target_free") is True, "generation must not receive target pixels")
        require(digest(sample.get("initial_latent_sha256")), "starting-noise digest missing")
        initial_latents.setdefault(sample["case_id"], set()).add(sample["initial_latent_sha256"])
        seeds.setdefault(sample["case_id"], set()).add(sample["seed"])
    require(
        all(len(values) == 1 for values in initial_latents.values())
        and all(len(values) == 1 for values in seeds.values()),
        "generation routes do not share starting noise and seed",
    )
    for case in expected_cases:
        for route in ROUTES:
            require(
                (case, route + "_cfg1") in lookup, "completed gallery is missing a requested route"
            )
            sample = lookup[(case, route + "_cfg1")]
            require(
                sample.get("text_guidance_scale") == 1.0
                and sample.get("sampling_steps") == settings["sampling_steps"],
                "template sample guidance or denoising steps differ",
            )
            require(
                sample.get("template_swap") is True
                and sample.get("actual_condition_verified") is True,
                "actual generated template conditioning was not verified",
            )
            selected = selection["validation"][int(case.rsplit("-", 1)[1])]
            detail = stats_lookup[("validation", selected["id"], "matched")]["routes"][route]
            require(
                sample.get("condition")
                == {key: detail[key] for key in ("embeds_sha256", "attention_mask_sha256")},
                "generation conditioning differs from recorded template route",
            )
    gallery = json.loads((run / "gallery-targets.json").read_text())
    targets = {row["id"]: row for row in gallery["records"]}
    cases = []
    for case in sorted(expected_cases):
        position = int(case.rsplit("-", 1)[1])
        selected = selection["validation"][position]
        entry = gallery["case_targets"][case]
        require(
            entry == {"split": "validation", "id": selected["id"]},
            "gallery target differs from selected validation case",
        )
        target = targets[selected["id"]]
        require(target["split"] == "validation", "gallery target is not heldout")
        require(
            sha256(file_in_run(run, "targets/" + target["id"] + ".jpg")) == target["image_sha256"],
            "target image digest differs",
        )
        require(
            all(lookup[(case, route + "_cfg1")]["prompt"] == target["prompt"] for route in ROUTES),
            "route prompts differ from selected caption",
        )
        cases.append(
            {
                "case_id": case,
                "id": target["id"],
                "initial_latent_sha256": next(iter(initial_latents[case])),
            }
        )
    result = {
        "schema_version": 1,
        "artifact_audit_passed": True,
        "real_checkpoint_evidence": report["evidence_kind"]
        == "real_checkpoint_conditioning_diagnostic",
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "cases": cases,
        "routes": list(ROUTES),
        "text_guidance_scale": 1.0,
        "swap_stage": protocol["swap_stage"],
        "teacher_assisted_hybrids": list(ROUTES[2:]),
        "template_audits_checked": len(statistics),
        "saved_image_hashes_checked": len(lookup),
        "paired_flow_controls_checked": len(flow),
        "unique_target_caption_pairs": sum(len(rows) for rows in selection.values()),
        "timesteps_per_target": len(timesteps),
        "conditioning_boundary_hashes_verified": True,
        "frozen_tensor_hashes_compared": len(report["frozen_hashes_before"]),
        "alignment_checkpoint_sha256": lineage.get("sha256"),
        "original_generator_manifest_sha256": lineage["reference_checkpoint_sha256"],
        "scope": "Artifact validation of recorded runtime proofs. Does not re-run teacher tokenization or model activations; teacher-assisted hybrids are not deployable PRISM-only generation or an accuracy benchmark.",
    }
    return result, report, lookup, targets


def render(run, *, audit_only=False):
    run = Path(run).resolve(strict=True)
    result, report, lookup, targets = audit(run)
    if not audit_only:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps

        cases = result["cases"]
        height = 4.6 * len(cases) + 2.7
        fig = plt.figure(figsize=(21, height), facecolor="white")
        grid = fig.add_gridspec(
            len(cases) * 2,
            6,
            height_ratios=[1, 0.34] * len(cases),
            left=0.025,
            right=0.985,
            top=1 - 1.95 / height,
            bottom=1.1 / height,
            hspace=0.26,
            wspace=0.09,
        )
        fixture = " · FIXTURE" if not result["real_checkpoint_evidence"] else ""
        fig.suptitle(
            "DOCCI / PRISM — template-swap diagnosis" + fixture,
            x=0.025,
            y=1 - 0.2 / height,
            ha="left",
            fontsize=19,
            fontweight="bold",
        )
        fig.text(
            0.025,
            1 - 0.72 / height,
            f"Frozen original DiT · CFG 1 for every route · {report['settings']['sampling_steps']} denoising steps · identical starting noise per case",
            fontsize=11,
        )
        fig.text(
            0.025,
            1 - 1.08 / height,
            "The three rightmost columns use native-teacher template features, substituted before DiT RMSNorm.",
            fontsize=11,
            color="#7a4100",
        )
        labels = [
            "DOCCI target\nvisual reference only",
            "Native teacher\nCFG 1",
            "Aligned PRISM\nCFG 1",
            "Native prefix + PRISM rest\nteacher-assisted · CFG 1",
            "Native suffix + PRISM rest\nteacher-assisted · CFG 1",
            "Native prefix and suffix\nPRISM caption · teacher-assisted",
        ]
        for i, case in enumerate(cases):
            target = targets[case["id"]]
            paths = [run / "targets" / (case["id"] + ".jpg")] + [
                run / Path(lookup[(case["case_id"], route + "_cfg1")]["path"]).name
                for route in ROUTES
            ]
            for column, (path, label) in enumerate(zip(paths, labels, strict=True)):
                ax = fig.add_subplot(grid[2 * i, column])
                ax.set_axis_off()
                with Image.open(path) as im:
                    pixels = (
                        ImageOps.exif_transpose(im)
                        .convert("RGB")
                        .resize(
                            (report["settings"]["width"], report["settings"]["height"]),
                            Image.Resampling.BICUBIC,
                        )
                    )
                ax.imshow(pixels)
                ax.set_title(label, fontsize=10.5, pad=8)
            ax = fig.add_subplot(grid[2 * i + 1, :])
            ax.set_axis_off()
            ax.text(
                0,
                1,
                case["id"] + " · held out from optimizer updates",
                va="top",
                fontsize=11,
                fontweight="bold",
                transform=ax.transAxes,
            )
            caption = target["prompt"]
            excerpt = caption[:390].rsplit(" ", 1)[0] + "…" if len(caption) > 390 else caption
            ax.text(
                0,
                0.58,
                textwrap.fill("Caption excerpt: " + excerpt, 190),
                va="top",
                fontsize=9.5,
                linespacing=1.2,
                transform=ax.transAxes,
            )
        fig.text(
            0.025,
            0.67 / height,
            "Teacher-assisted swaps test the representation interface. They do not establish PRISM-only generation quality.",
            fontsize=10,
            color="#7a4100",
        )
        fig.text(
            0.025,
            0.28 / height,
            "Targets: Google DOCCI (CC BY 4.0). Generated images receive captions alone. Qualitative diagnostic; no accuracy claim.",
            fontsize=9,
            color="#444444",
        )
        path = run / "template-swap-comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        result["render"] = {"path": str(path), "sha256": sha256(path)}
    (run / "template-swap-artifact-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args.run, audit_only=args.audit_only), indent=2))


if __name__ == "__main__":
    main()
