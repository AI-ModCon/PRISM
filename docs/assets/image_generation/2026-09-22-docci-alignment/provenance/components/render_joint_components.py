#!/usr/bin/env python3
"""Strict metadata-only audit and gallery for a frozen DiT/connector factorial.

Recorded execution evidence is audited; no model is imported or rerun. A passed
integrity audit is not an image-quality or full-data-training acceptance gate.
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
from pathlib import Path, PurePosixPath

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prism-components-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prism-components-cache")
)

PHASES = (
    (
        "original_aligned",
        "original",
        "aligned",
        ("native_original", "aligned_original"),
    ),
    ("original_joint", "original", "joint", ("joint_original",)),
    ("joint_aligned", "joint", "aligned", ("native_joint", "aligned_joint")),
    ("joint_joint", "joint", "joint", ("joint_joint",)),
)
ROUTES = tuple(route for _, _, _, routes in PHASES for route in routes)
CONNECTOR_PREFIX = "decoders.image.connector."
DIT_PREFIX = "decoders.image.backend.transformer."
RUNNER = "tools/diagnose_prism_joint_components.py"


def require(condition, message):
    if not condition:
        raise ValueError("Component artifact: " + message)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hashes(values):
    return (
        isinstance(values, dict)
        and bool(values)
        and all(isinstance(k, str) and digest(v) for k, v in values.items())
    )


def finite(value, *, nonnegative=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (not nonnegative or value >= 0)
    )


def audit_payload(report_raw, members, prefix):
    report = json.loads(report_raw)
    require(report.get("status") == "completed", "completed report required")
    require(report.get("training_performed") is False, "diagnostic must not train")
    require(
        report.get("quality_benchmark") is False,
        "diagnostic cannot claim a quality benchmark",
    )
    evidence = json.loads(members[prefix + "collection-evidence.json"])
    require(
        evidence.get("report_sha256") == sha(report_raw),
        "collected report digest changed",
    )
    require(
        evidence.get("checkpoint_bytes_included") is False, "weights must remain remote"
    )
    require(
        evidence.get("recorded_source_files_unchanged") is True,
        "executed source identity missing",
    )
    sources = dict(report.get("source_sha256", {}))
    require(digest(report.get("runner_sha256")), "runner digest missing")
    require(
        sources.get(RUNNER, report["runner_sha256"]) == report["runner_sha256"],
        "runner source digest disagrees",
    )
    sources[RUNNER] = report["runner_sha256"]
    require(hashes(sources), "source digests missing")
    checks = evidence.get("source_identity_checks", {})
    require(
        set(checks) == set(sources) and all(v is True for v in checks.values()),
        "source coverage differs",
    )
    for name, expected in sources.items():
        require(
            sha(members["prism/" + name]) == expected,
            "archived executed source digest differs",
        )
    collectors = evidence.get("collector_sha256", {})
    require(
        set(collectors)
        == {"pack_joint_components.py", "render_joint_components.py", "pack_run.py"},
        "collector source coverage differs",
    )
    for name, expected in collectors.items():
        require(
            digest(expected)
            and sha(members[prefix + "collector-snapshots/" + name]) == expected,
            "collector snapshot differs",
        )
    checkpoint_checks = evidence.get("checkpoint_checks", {})
    require(
        set(checkpoint_checks) == {"region", "joint"}, "both checkpoint checks required"
    )
    source_reports = {}
    for label, key in (
        ("region", "alignment_checkpoint"),
        ("joint", "joint_checkpoint"),
    ):
        lineage = report.get(key, {})
        check = checkpoint_checks[label]
        require(
            digest(lineage.get("sha256")) and digest(lineage.get("report_sha256")),
            "strict checkpoint lineage digests missing",
        )
        require(
            check.get("independently_verified") is True
            and check.get("actual_sha256")
            == check.get("expected_sha256")
            == lineage["sha256"],
            "checkpoint was not independently verified",
        )
        require(
            check.get("checkpoint") == lineage.get("checkpoint"),
            "checkpoint path differs",
        )
        raw = members[prefix + label + "-source-report.json"]
        require(
            sha(raw) == lineage["report_sha256"] == check.get("report_sha256"),
            "source checkpoint report differs",
        )
        source = json.loads(raw)
        require(
            source.get("status") == "completed"
            and source.get("completed_steps") == lineage.get("step"),
            "source checkpoint is not completed terminal step",
        )
        matched = [
            v
            for v in source.get("checkpoints", [])
            if v.get("sha256") == lineage["sha256"]
            and v.get("step") == lineage.get("step")
        ]
        require(
            len(matched) == 1,
            "terminal checkpoint is not uniquely bound to source report",
        )
        require(
            source.get("frozen_state_unchanged") is True
            and hashes(source.get("frozen_hashes_before"))
            and source.get("frozen_hashes_before") == source.get("frozen_hashes_after"),
            "source frozen audit failed",
        )
        source_reports[label] = source
    _audit_contract(report, source_reports)
    samples, targets, cases = _audit_gallery(report, members, prefix)
    return {
        "schema_version": 1,
        "artifact_audit_passed": True,
        "final_acceptance_evaluated": True,
        "report_sha256": sha(report_raw),
        "real_checkpoint_evidence": report["evidence_kind"].startswith(
            "real_checkpoint_"
        ),
        "routes": list(ROUTES),
        "flow_rows_checked": len(report["flow_controls"]),
        "images_checked": len(samples),
        "cases": cases,
        "checkpoint_sha256": {
            k: v["actual_sha256"] for k, v in checkpoint_checks.items()
        },
        "scope": "Audits recorded factorial controls and artifact integrity; does not establish semantic image quality or authorize scaling.",
    }


def condition_hashes(value):
    pair = {key: value.get(key) for key in ("embeds_sha256", "attention_mask_sha256")}
    require(all(digest(v) for v in pair.values()), "runtime condition digests missing")
    return pair


def expected_state(report, dit, connector):
    values = dict(report["invariant_frozen_hashes_before"])
    for prefix, key in (
        (DIT_PREFIX, "dit_" + dit),
        (CONNECTOR_PREFIX, "connector_" + connector),
    ):
        values.update(
            {prefix + k: v for k, v in report["state_identities"][key].items()}
        )
    return values


def _audit_contract(report, source_reports):
    require(
        report.get("schema_version") == 1
        and report.get("evidence_kind")
        in {"real_checkpoint_joint_component_diagnostic", "fixture_only"},
        "unsupported diagnostic kind/schema",
    )
    require(
        report.get("qualification") == "unqualified"
        and report.get("completed_steps") == 0,
        "diagnostic qualification or training step count differs",
    )
    require(
        report.get("cache_saved") is False and report.get("checkpoints_saved") is False,
        "diagnostic must not persist features or model checkpoints",
    )
    settings, protocol = report["settings"], report.get("component_protocol", {})
    fixed = {
        "expected_connector_step": 500,
        "expected_parent_tensors": 526,
        "train_probe_count": 2,
        "validation_probe_count": 2,
        "sample_count": 2,
        "sampling_steps": 50,
        "height": 256,
        "width": 256,
        "max_text_length": 1024,
        "seed": 42,
        "flow_timesteps": [0.1, 0.5, 0.9],
        "prism_formats": ["chat"],
        "dtype": "bfloat16",
        "attention_backend": "math",
        "deterministic": True,
    }
    require(
        all(settings.get(k) == v for k, v in fixed.items()),
        "fixed reviewed execution settings differ",
    )
    expected_phases = [
        {"name": name, "dit": dit, "connector": connector, "routes": list(routes)}
        for name, dit, connector, routes in PHASES
    ]
    require(
        protocol.get("routes") == list(ROUTES)
        and protocol.get("phases") == expected_phases,
        "explicit factorial route/phase protocol differs",
    )
    require(
        protocol.get("intentional_runtime_swaps") is True
        and protocol.get("runtime_dit_snapshot_dtype") == "bfloat16"
        and protocol.get("connector_snapshot_dtype") == "float32"
        and protocol.get("optimizer_created") is False
        and protocol.get("teacher_cache_persisted") is False
        and protocol.get("final_state") == "original_dit_aligned_connector"
        and protocol.get("feature_normalization") == "captured_original_dit_rmsnorm",
        "runtime swap, precision, frozen or normalization contract differs",
    )
    alignment, joint = report["alignment_checkpoint"], report["joint_checkpoint"]
    region_source, joint_source = source_reports["region"], source_reports["joint"]
    require(
        alignment.get("schema_version") == 2
        and alignment.get("evidence_kind")
        == "real_checkpoint_connector_native_region_feature_alignment"
        and alignment.get("restore_policy")
        == "connector_weights_only_from_native_feature_alignment"
        and alignment.get("prompt_format") == "chat"
        and alignment.get("region_partition_audit_revalidated") is True
        and alignment.get("exact_token_and_content_audit_revalidated") is True
        and alignment.get("frozen_state_unchanged") is True
        and all(
            alignment.get(k) is False
            for k in (
                "optimizer_state_restored",
                "rng_state_restored",
                "sampler_state_restored",
            )
        ),
        "strict schema-2 connector admission is missing",
    )
    require(
        alignment.get("step") == 1000
        and region_source.get("schema_version") == 2
        and region_source.get("evidence_kind") == alignment["evidence_kind"],
        "expected terminal region checkpoint required",
    )
    require(
        joint.get("policy") == "completed_joint_checkpoint_fp32_masters_fresh_stage"
        and joint.get("step") == 6
        and joint.get("returned_optimizer_masters") is False
        and all(
            joint.get(k) is True
            for k in ("fresh_optimizer", "fresh_rng", "fresh_sampler")
        ),
        "strict completed joint-six weights-only admission is missing",
    )
    require(
        joint_source.get("evidence_kind")
        == "real_checkpoint_connector_diffusion_webdataset_pilot"
        and joint_source.get("settings", {}).get("steps") == 6
        and joint_source.get("settings", {}).get("prompt_format") == "chat",
        "joint source contract differs",
    )
    for key, value in (
        ("alignment_checkpoint", alignment),
        ("joint_checkpoint", joint),
    ):
        require(
            settings.get(key) == value["checkpoint"]
            and settings.get(key + "_sha256") == value["sha256"],
            "settings do not bind restored checkpoint",
        )
    required_lineage = (
        "sha256",
        "report_sha256",
        "checkpoint",
        "selection",
        "parent",
        "index_sha256",
        "data_fingerprint",
        "reference_checkpoint_sha256",
        "evidence_kind",
        "schema_version",
    )
    require(
        all(
            joint_source.get("alignment_initialization", {}).get(k) == alignment.get(k)
            for k in required_lineage
        ),
        "joint checkpoint did not initialize from this region checkpoint",
    )
    require(
        digest(report.get("data_fingerprint")) and hashes(report.get("index_sha256")),
        "data identities missing",
    )
    for source in (region_source, joint_source):
        require(
            source.get("data_fingerprint") == report["data_fingerprint"]
            and source.get("parent") == report.get("parent")
            and source.get("generator", {}).get("manifest_sha256")
            == report.get("generator", {}).get("manifest_sha256"),
            "source parent/data/generator identities differ",
        )
    require(
        region_source.get("index_sha256") == report["index_sha256"]
        and {
            split: joint_source.get(split + "_index_sha256")
            for split in ("train", "validation")
        }
        == report["index_sha256"],
        "source index identities differ",
    )
    require(
        alignment.get("selection") == region_source.get("selection")
        and joint_source.get("train_selection") == alignment["selection"]["train"],
        "source caption cohorts differ",
    )
    selection = report["selection"]
    require(
        set(selection) == {"train", "validation"}
        and all(len(v) == 2 for v in selection.values()),
        "2/2 cohort required",
    )
    require(
        selection["train"] == alignment["selection"]["train"][:2]
        and selection["validation"] == alignment["selection"]["validation"][:2],
        "diagnostic cohort differs from admitted source",
    )
    selected = {
        (split, row["id"]): row["index"]
        for split, rows in selection.items()
        for row in rows
    }
    require(
        len(selected) == 4
        and len({row["id"] for rows in selection.values() for row in rows}) == 4,
        "duplicate or overlapping diagnostic IDs",
    )
    require(
        all(type(i) is int and i >= 0 for i in selected.values()),
        "invalid selected index",
    )
    identities = report.get("state_identities", {})
    require(
        set(identities)
        == {"dit_original", "dit_joint", "connector_aligned", "connector_joint"}
        and all(hashes(v) for v in identities.values()),
        "component state identity coverage missing",
    )
    require(
        set(identities["dit_original"]) == set(identities["dit_joint"])
        and set(identities["connector_aligned"]) == set(identities["connector_joint"]),
        "factor snapshot scopes differ",
    )
    require(
        identities["connector_aligned"] == region_source.get("connector_hashes_after"),
        "aligned connector differs from region terminal hashes",
    )
    original_dit = {
        k[len(DIT_PREFIX) :]: v
        for k, v in region_source["frozen_hashes_after"].items()
        if k.startswith(DIT_PREFIX)
    }
    require(
        identities["dit_original"] == original_dit,
        "original runtime DiT differs from original frozen generator",
    )
    require(
        hashes(report.get("original_caption_norm_sha256"))
        and report["original_caption_norm_sha256"]
        == alignment.get("caption_normalization", {}).get("state_sha256"),
        "captured original RMSNorm identity differs",
    )
    invariant = report.get("invariant_frozen_hashes_before")
    require(
        hashes(invariant)
        and invariant
        == report.get("invariant_frozen_hashes_after")
        == joint_source["frozen_hashes_after"]
        and report.get("invariant_frozen_state_unchanged") is True,
        "PRISM/VAE/native invariant state changed",
    )
    require(
        not any(k.startswith((DIT_PREFIX, CONNECTOR_PREFIX)) for k in invariant),
        "intentional factors leaked into invariant scope",
    )
    require(
        report.get("final_state_restored") is True
        and report.get("baseline_hashes")
        == report.get("final_hashes")
        == expected_state(report, "original", "aligned"),
        "final baseline restoration differs",
    )
    audits = report.get("phase_audits", [])
    require(
        len(audits) == 4 and [a.get("phase") for a in audits] == [p[0] for p in PHASES],
        "phase audit coverage/order differs",
    )
    negative = None
    for audit, (phase, dit, connector, routes) in zip(audits, PHASES):
        require(
            audit.get("dit") == dit
            and audit.get("connector") == connector
            and audit.get("routes") == list(routes),
            "phase factor labels differ",
        )
        require(
            all(
                audit.get(k) is True
                for k in (
                    "all_frozen_eval",
                    "frozen_state_unchanged",
                    "invariant_frozen_state_unchanged",
                    "original_norm_unchanged",
                )
            ),
            "frozen phase audit failed",
        )
        require(
            audit.get("hashes_before")
            == audit.get("hashes_after")
            == expected_state(report, dit, connector),
            "phase executed state differs from selected factors",
        )
        pair = condition_hashes(audit.get("native_negative", {}))
        require(
            negative is None or negative == pair,
            "native negative condition changed across phases",
        )
        negative = pair
    features = report.get("feature_drift", [])
    keys = [(f["phase"], f["split"], f["id"], f["caption_kind"]) for f in features]
    expected = {
        (p[0], split, identifier, kind)
        for p in PHASES
        for split, identifier in selected
        for kind in ("matched", "wrong")
    }
    require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "feature drift audit coverage differs",
    )
    features_by_key = dict(zip(keys, features))
    shared_audits, shared_metrics = {}, {}
    for feature in features:
        phase = next(p for p in PHASES if p[0] == feature["phase"])
        require(
            feature.get("connector") == phase[2]
            and feature.get("dit") == phase[1]
            and feature.get("normalization") == "captured_original_dit_rmsnorm",
            "feature factor/normalization label differs",
        )
        require(
            (feature["caption_id"] == feature["id"])
            == (feature["caption_kind"] == "matched"),
            "wrong caption is not distinct",
        )
        audit = feature.get("audit", {})
        require(
            all(
                audit.get(k) is True
                for k in (
                    "exact_formatted_input_match",
                    "exact_input_ids_match",
                    "exact_input_masks_match",
                    "actual_native_forward_inputs_verified",
                )
            )
            and audit.get("target_pixels_read") is False,
            "actual token or target-free feature proof missing",
        )
        require(
            all(
                digest(audit.get(k))
                for k in (
                    "prompt_sha256",
                    "formatted_prompt_sha256",
                    "input_ids_sha256",
                    "input_mask_sha256",
                    "prism_hidden_sha256",
                    "native_features_sha256",
                    "teacher_normalized_sha256",
                )
            ),
            "feature input/state digests missing",
        )
        token_ids = audit.get("input_token_ids", [])
        require(
            len(token_ids) == 1
            and token_ids[0]
            and all(type(v) is int and v >= 0 for v in token_ids[0]),
            "invalid full prompt token IDs",
        )
        span = audit.get("content_span", [])
        require(
            len(span) == 2
            and all(type(v) is int for v in span)
            and 0 < span[0] < span[1] < len(token_ids[0]),
            "invalid caption partition spans",
        )
        parts = feature.get("partitions", {})
        counts = {
            "prefix": span[0],
            "content": span[1] - span[0],
            "suffix": len(token_ids[0]) - span[1],
        }
        require(
            set(parts) == set(counts)
            and all(
                p.get("tokens") == counts[k]
                and finite(p.get("mse"), nonnegative=True)
                and finite(p.get("cosine"))
                and abs(p["cosine"]) <= 1.00001
                for k, p in parts.items()
            ),
            "invalid original-norm region feature metrics",
        )
        key = (feature["split"], feature["caption_id"])
        require(
            key not in shared_audits or shared_audits[key] == audit,
            "frozen caption/native features changed across factor phases",
        )
        shared_audits[key] = audit
        key = (feature["connector"], *key)
        require(
            key not in shared_metrics or shared_metrics[key] == parts,
            "original-norm connector feature metrics changed across DiTs",
        )
        shared_metrics[key] = parts
    statistics = report.get("condition_statistics", [])
    stat_keys = [(s["phase"], s["split"], s["id"]) for s in statistics]
    require(
        len(stat_keys) == len(set(stat_keys))
        and set(stat_keys)
        == {
            (p[0], split, identifier) for p in PHASES for split, identifier in selected
        },
        "condition statistics coverage differs",
    )
    conditions = {}
    cross_dit = {}
    for stat in statistics:
        phase = next(p for p in PHASES if p[0] == stat["phase"])
        require(
            set(stat["routes"]) == set(phase[3]),
            "condition phase route coverage differs",
        )
        for route, values in stat["routes"].items():
            require(
                set(values) == {"matched", "wrong"},
                "matched/wrong condition coverage missing",
            )
            for kind, value in values.items():
                pair = condition_hashes(value)
                conditions[(stat["split"], stat["id"], route, kind)] = pair
                family = "native" if route.startswith("native_") else phase[2]
                key = (stat["split"], stat["id"], family, kind)
                require(
                    key not in cross_dit or cross_dit[key] == pair,
                    "pre-RMSNorm conditioning changed solely with DiT",
                )
                cross_dit[key] = pair
    flow = report.get("flow_controls", [])
    flow_keys = [(r["split"], r["id"], r["repeat"]) for r in flow]
    require(
        len(flow_keys) == len(set(flow_keys))
        and set(flow_keys)
        == {
            (split, identifier, repeat)
            for split, identifier in selected
            for repeat in range(3)
        },
        "paired flow case/time coverage differs",
    )
    for row in flow:
        require(
            row.get("index") == selected[(row["split"], row["id"])]
            and row.get("phases") == [p[0] for p in PHASES]
            and row.get("actual_conditioning_verified") is True
            and set(row["routes"]) == set(ROUTES),
            "cross-phase flow proof or route coverage missing",
        )
        require(
            row.get("requested_timestep") == settings["flow_timesteps"][row["repeat"]]
            and row.get("seed")
            == 42
            + 100000
            + (0 if row["split"] == "train" else 50000)
            + row["index"] * 8
            + row["repeat"],
            "flow requested seed/time differs",
        )
        actual = row.get("actual_inputs", {})
        require(
            digest(actual.get("noisy_latent_sha256"))
            and digest(actual.get("timestep_sha256"))
            and isinstance(actual.get("timestep"), list)
            and len(actual["timestep"]) == 1
            and finite(actual["timestep"][0]),
            "actual noisy input/timestep hashes missing",
        )
        for route, values in row["routes"].items():
            require(
                all(
                    finite(values.get(k), nonnegative=True)
                    for k in ("matched", "wrong", "prediction_change_mse")
                )
                and finite(values.get("wrong_minus_matched"))
                and math.isclose(
                    values["wrong_minus_matched"],
                    values["wrong"] - values["matched"],
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ),
                "invalid loss or caption gap",
            )
            for kind in ("matched", "wrong"):
                require(
                    values.get(kind + "_condition")
                    == conditions[(row["split"], row["id"], route, kind)],
                    "actual flow condition differs from intended factor route",
                )
            phase = next(p[0] for p in PHASES if route in p[3])
            require(
                row["wrong_id"]
                == features_by_key[(phase, row["split"], row["id"], "wrong")][
                    "caption_id"
                ],
                "wrong caption differs across recorded controls",
            )


def _audit_gallery(report, members, prefix):
    lookup, noise, schedule = {}, {}, set()
    phases = {route: phase for phase, _, _, routes in PHASES for route in routes}
    stats = {
        (s["phase"], s["split"], s["id"]): s for s in report["condition_statistics"]
    }
    phase_audits = {a["phase"]: a for a in report["phase_audits"]}
    expected = {(f"validation-{i:02d}", route) for i in range(2) for route in ROUTES}
    for sample in report.get("samples", []):
        key = (sample["case_id"], sample["route"])
        require(
            key not in lookup and key in expected,
            "unexpected or duplicate image route/case",
        )
        lookup[key] = sample
        basename = Path(sample["path"]).name
        require(
            sha(members[prefix + basename]) == sample["sha256"],
            "generated image digest differs",
        )
        require(
            sample.get("target_free") is True
            and sample.get("quality_claim") is False
            and sample.get("actual_condition_verified") is True
            and sample.get("negative_conditioner") == "original_frozen_native"
            and sample.get("text_guidance_scale") == 5.0
            and sample.get("sampling_steps") == 50,
            "generation conditioning/guidance contract differs",
        )
        position = int(sample["case_id"].rsplit("-", 1)[1])
        require(sample.get("seed") == 200042 + position, "sample seed differs")
        selected = report["selection"]["validation"][position]
        phase = phases[sample["route"]]
        positive = condition_hashes(
            stats[(phase, "validation", selected["id"])]["routes"][sample["route"]][
                "matched"
            ]
        )
        negative = condition_hashes(phase_audits[phase]["native_negative"])
        trace = sample.get("trace_sha256", {})
        require(hashes(trace), "sampling trace hashes missing")
        for label, expected_pair, branch in (
            ("positive", positive, 0),
            ("negative", negative, 1),
        ):
            require(
                trace.get("condition." + label)
                == trace.get(f"condition.branch{branch}")
                == expected_pair["embeds_sha256"]
                and trace.get("mask." + label)
                == trace.get(f"mask.branch{branch}")
                == expected_pair["attention_mask_sha256"],
                "sampling CFG branch differs from audited condition",
            )
        require(
            {k for k in trace if k.startswith("condition.branch")}
            == {"condition.branch0", "condition.branch1"},
            "unexpected CFG branch coverage",
        )
        require(
            all(
                digest(trace.get(k))
                for k in (
                    "latents.initial",
                    "latents.final",
                    "prediction.step0",
                    "schedule.timesteps",
                )
            ),
            "initial/final latent, prediction, or schedule digest missing",
        )
        require(
            sample.get("initial_latent_sha256") == trace["latents.initial"],
            "sample initial latent identity differs",
        )
        noise.setdefault(sample["case_id"], set()).add(trace["latents.initial"])
        schedule.add(trace["schedule.timesteps"])
    require(
        set(lookup) == expected
        and all(len(v) == 1 for v in noise.values())
        and len(schedule) == 1,
        "complete gallery route coverage, shared noise or shared schedule missing",
    )
    gallery = json.loads(members[prefix + "gallery-targets.json"])
    targets = {t["id"]: t for t in gallery["records"]}
    require(
        len(targets) == len(gallery["records"]) == 2, "target image coverage differs"
    )
    cases = []
    for i, selected in enumerate(report["selection"]["validation"]):
        case = f"validation-{i:02d}"
        require(
            gallery["case_targets"].get(case)
            == {"split": "validation", "id": selected["id"]},
            "target case identity differs",
        )
        target = targets[selected["id"]]
        require(
            target["split"] == "validation"
            and sha(members[prefix + "targets/" + selected["id"] + ".jpg"])
            == target["image_sha256"],
            "target image hash or split differs",
        )
        require(
            all(
                lookup[(case, route)]["prompt"] == target["prompt"] for route in ROUTES
            ),
            "image caption differs from target",
        )
        cases.append(
            {
                "case_id": case,
                "id": selected["id"],
                "initial_latent_sha256": next(iter(noise[case])),
            }
        )
    return lookup, targets, cases


def read_members(run):
    run = Path(run).resolve(strict=True)
    raw = (run / "report.json").read_bytes()
    report = json.loads(raw)
    members = {}
    prefix = "runs/" + run.name + "/"
    for path in run.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink artifact refused")
        if path.is_file():
            require(path.stat().st_size <= 64 * 1024 * 1024, "oversize artifact")
            members[prefix + path.relative_to(run).as_posix()] = path.read_bytes()
    # Root relocates archive source members into one immutable per-run directory.
    asset_root = run.parent.parent
    roots = (
        asset_root / "provenance/source-snapshots" / run.name,
        asset_root / "prism",
    )
    for relative in set(report.get("source_sha256", {})) | {RUNNER}:
        value = PurePosixPath(relative)
        require(
            not value.is_absolute()
            and ".." not in value.parts
            and "\\" not in relative,
            "unsafe source path",
        )
        found = [root / relative for root in roots if (root / relative).is_file()]
        require(bool(found), "missing executed source snapshot: " + relative)
        path = found[0]
        require(
            not path.is_symlink() and path.stat().st_size <= 64 * 1024 * 1024,
            "unsafe source snapshot",
        )
        members["prism/" + relative] = path.read_bytes()
    return raw, members, prefix


def render(run, *, audit_only=False):
    run = Path(run).resolve(strict=True)
    raw, members, prefix = read_members(run)
    result = audit_payload(raw, members, prefix)
    report = json.loads(raw)
    if not audit_only:
        _render_gallery(run, report, members, prefix, result)
    (run / "component-artifact-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _render_gallery(run, report, members, prefix, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    gallery = json.loads(members[prefix + "gallery-targets.json"])
    targets = {r["id"]: r for r in gallery["records"]}
    lookup = {(r["case_id"], r["route"]): r for r in report["samples"]}
    fig = plt.figure(figsize=(24.5, 12.5), facecolor="white")
    grid = fig.add_gridspec(
        4,
        7,
        height_ratios=[1, 0.42, 1, 0.42],
        left=0.022,
        right=0.987,
        top=0.83,
        bottom=0.1,
        hspace=0.17,
        wspace=0.055,
    )
    fixture_label = (
        " · METADATA FIXTURE" if not result["real_checkpoint_evidence"] else ""
    )
    fig.suptitle(
        "PRISM / DOCCI — connector × diffusion transformer" + fixture_label,
        x=0.025,
        y=0.965,
        ha="left",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.922,
        "Frozen factorial diagnostic · CFG 5 with the same native negative · 50 sampling steps · replayed initial noise",
        fontsize=12,
    )
    fig.text(
        0.025,
        0.888,
        "Two held-out captions. Matched flow loss and feature alignment do not establish subject fidelity.",
        fontsize=11,
        color="#445",
    )
    for row, case in enumerate(result["cases"]):
        target = targets[case["id"]]
        for col, route in enumerate(("target", *ROUTES)):
            ax = fig.add_subplot(grid[row * 2, col])
            if route == "target":
                raw = members[prefix + "targets/" + case["id"] + ".jpg"]
                title = "Target image"
            else:
                sample = lookup[(case["case_id"], route)]
                raw = members[prefix + Path(sample["path"]).name]
                title = route.replace("_", " ")
            ax.imshow(Image.open(io.BytesIO(raw)))
            ax.set_axis_off()
            ax.set_title(textwrap.fill(title, 22), fontsize=10, pad=9)
        caption_ax = fig.add_subplot(grid[row * 2 + 1, :])
        caption_ax.set_axis_off()
        caption_ax.text(
            0,
            0.98,
            case["id"] + " · " + textwrap.fill(target["prompt"], 190),
            va="top",
            fontsize=9.5,
            linespacing=1.3,
        )
    footer = (
        "Synthetic fixture images; no model was executed."
        if not result["real_checkpoint_evidence"]
        else "Targets: Google DOCCI, CC BY 4.0. Generated samples are experimental; inspect requested objects, counts, placement and color."
    )
    fig.text(0.025, 0.04, footer, fontsize=10, color="#445")
    path = run / "component-comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    result["gallery"] = str(path)
    result["gallery_sha256"] = sha(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args.run, audit_only=args.audit_only), indent=2))


if __name__ == "__main__":
    main()
