"""Metadata fixtures only: these tests do not load or execute any model."""

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import pack_joint_components as packer
import render_joint_components as renderer

BASE = Path(__file__).resolve().parent


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture():
    from PIL import Image, ImageDraw

    prefix = "runs/fixture/"
    connector = {
        "0.weight": h("aligned0"),
        "0.bias": h("aligned1"),
        "1.weight": h("aligned2"),
        "1.bias": h("aligned3"),
    }
    dit = {
        "weight": h("original"),
        "buffer": h("originalbuffer"),
        "time_caption_embed.caption_embedder.0.weight": h("norm"),
    }
    invariant = {
        "encoder.weight": h("prism"),
        "decoders.image.backend.vae.weight": h("vae"),
        "decoders.image.backend.mllm.weight": h("native"),
    }
    parent, generator = {"fixture": True}, {"manifest_sha256": h("generator")}
    selection = {
        "train": [{"index": i, "id": f"train{i}"} for i in range(32)],
        "validation": [{"index": i, "id": f"val{i}"} for i in range(8)],
    }
    index_hash = {"train": h("trainindex"), "validation": h("valindex")}
    alignment = {
        "checkpoint": "/fixture/region/step.pt",
        "sha256": h("regioncheckpoint"),
        "report_sha256": "",
        "step": 1000,
        "schema_version": 2,
        "evidence_kind": "real_checkpoint_connector_native_region_feature_alignment",
        "restore_policy": "connector_weights_only_from_native_feature_alignment",
        "prompt_format": "chat",
        "region_partition_audit_revalidated": True,
        "exact_token_and_content_audit_revalidated": True,
        "frozen_state_unchanged": True,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "sampler_state_restored": False,
        "selection": selection,
        "parent": parent,
        "index_sha256": index_hash,
        "data_fingerprint": h("data"),
        "reference_checkpoint_sha256": generator["manifest_sha256"],
        "caption_normalization": {"state_sha256": {"weight": h("norm")}},
    }
    region_source = {
        "schema_version": 2,
        "evidence_kind": alignment["evidence_kind"],
        "status": "completed",
        "completed_steps": 1000,
        "checkpoints": [{"step": 1000, "sha256": alignment["sha256"]}],
        "frozen_state_unchanged": True,
        "frozen_hashes_before": {
            **invariant,
            **{renderer.DIT_PREFIX + k: v for k, v in dit.items()},
        },
        "connector_hashes_after": connector,
        "data_fingerprint": h("data"),
        "parent": parent,
        "generator": generator,
        "index_sha256": index_hash,
        "selection": selection,
    }
    region_source["frozen_hashes_after"] = copy.deepcopy(
        region_source["frozen_hashes_before"]
    )
    alignment["report_sha256"] = renderer.sha(raw(region_source))
    joint_source = {
        "schema_version": 1,
        "evidence_kind": "real_checkpoint_connector_diffusion_webdataset_pilot",
        "status": "completed",
        "completed_steps": 6,
        "settings": {"steps": 6, "prompt_format": "chat"},
        "checkpoints": [{"step": 6, "sha256": h("jointcheckpoint")}],
        "frozen_state_unchanged": True,
        "frozen_hashes_before": invariant,
        "frozen_hashes_after": invariant,
        "data_fingerprint": h("data"),
        "parent": parent,
        "generator": generator,
        "train_index_sha256": index_hash["train"],
        "validation_index_sha256": index_hash["validation"],
        "train_selection": selection["train"],
        "alignment_initialization": copy.deepcopy(alignment),
    }
    joint = {
        "checkpoint": "/fixture/joint/step.pt",
        "sha256": h("jointcheckpoint"),
        "report_sha256": renderer.sha(raw(joint_source)),
        "step": 6,
        "policy": "completed_joint_checkpoint_fp32_masters_fresh_stage",
        "returned_optimizer_masters": False,
        "fresh_optimizer": True,
        "fresh_rng": True,
        "fresh_sampler": True,
    }
    settings = {
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
    for key, lineage in [
        ("alignment_checkpoint", alignment),
        ("joint_checkpoint", joint),
    ]:
        settings[key], settings[key + "_sha256"] = (
            lineage["checkpoint"],
            lineage["sha256"],
        )
    source_raw = b"# fixture source, no model execution\n"
    report = {
        "schema_version": 1,
        "status": "completed",
        "evidence_kind": "fixture_only",
        "qualification": "unqualified",
        "training_performed": False,
        "quality_benchmark": False,
        "completed_steps": 0,
        "cache_saved": False,
        "checkpoints_saved": False,
        "settings": settings,
        "source_sha256": {
            "tools/prism_image_conditioning.py": renderer.sha(source_raw)
        },
        "runner_sha256": renderer.sha(source_raw),
        "data_fingerprint": h("data"),
        "index_sha256": index_hash,
        "parent": parent,
        "generator": generator,
        "alignment_checkpoint": alignment,
        "joint_checkpoint": joint,
        "component_protocol": {
            "routes": list(renderer.ROUTES),
            "phases": [
                {"name": p, "dit": d, "connector": c, "routes": list(r)}
                for p, d, c, r in renderer.PHASES
            ],
            "intentional_runtime_swaps": True,
            "runtime_dit_snapshot_dtype": "bfloat16",
            "connector_snapshot_dtype": "float32",
            "optimizer_created": False,
            "teacher_cache_persisted": False,
            "final_state": "original_dit_aligned_connector",
            "feature_normalization": "captured_original_dit_rmsnorm",
        },
        "selection": {split: rows[:2] for split, rows in selection.items()},
        "state_identities": {
            "connector_aligned": connector,
            "connector_joint": {k: h("joint" + k) for k in connector},
            "dit_original": dit,
            "dit_joint": {k: h("jointdit" + k) for k in dit},
        },
        "original_caption_norm_sha256": {"weight": h("norm")},
        "invariant_frozen_hashes_before": invariant,
        "invariant_frozen_hashes_after": invariant,
        "invariant_frozen_state_unchanged": True,
        "final_state_restored": True,
        "phase_audits": [],
        "condition_statistics": [],
        "feature_drift": [],
        "flow_controls": [],
        "samples": [],
    }
    report["baseline_hashes"] = renderer.expected_state(report, "original", "aligned")
    report["final_hashes"] = copy.deepcopy(report["baseline_hashes"])
    negative = {
        "embeds_sha256": h("negative"),
        "attention_mask_sha256": h("negative_mask"),
    }
    conditions = {}
    members = {
        prefix + "region-source-report.json": raw(region_source),
        prefix + "joint-source-report.json": raw(joint_source),
        "prism/tools/prism_image_conditioning.py": source_raw,
        "prism/" + renderer.RUNNER: source_raw,
    }
    for phase, dit_kind, connector_kind, routes in renderer.PHASES:
        state = renderer.expected_state(report, dit_kind, connector_kind)
        report["phase_audits"].append(
            {
                "phase": phase,
                "dit": dit_kind,
                "connector": connector_kind,
                "routes": list(routes),
                "hashes_before": state,
                "hashes_after": state,
                "all_frozen_eval": True,
                "frozen_state_unchanged": True,
                "invariant_frozen_state_unchanged": True,
                "original_norm_unchanged": True,
                "native_negative": negative,
            }
        )
        for split, rows in report["selection"].items():
            for row in rows:
                statistic = {
                    "phase": phase,
                    "split": split,
                    "id": row["id"],
                    "routes": {route: {} for route in routes},
                }
                for kind in ("matched", "wrong"):
                    caption_id = (
                        row["id"]
                        if kind == "matched"
                        else (
                            f"train{row['index'] + 1}"
                            if split == "train"
                            else f"val{row['index'] + 1}"
                        )
                    )
                    audit = {
                        k: h(caption_id + k)
                        for k in (
                            "prompt_sha256",
                            "formatted_prompt_sha256",
                            "input_ids_sha256",
                            "input_mask_sha256",
                            "prism_hidden_sha256",
                            "native_features_sha256",
                            "teacher_normalized_sha256",
                        )
                    }
                    audit.update(
                        input_token_ids=[[1, 2, 3, 4, 5, 6]],
                        content_span=[2, 5],
                        exact_formatted_input_match=True,
                        exact_input_ids_match=True,
                        exact_input_masks_match=True,
                        actual_native_forward_inputs_verified=True,
                        target_pixels_read=False,
                    )
                    parts = {
                        k: {
                            "mse": 0.02 if connector_kind == "aligned" else 0.025,
                            "cosine": 0.9,
                            "tokens": n,
                        }
                        for k, n in [("prefix", 2), ("content", 3), ("suffix", 1)]
                    }
                    report["feature_drift"].append(
                        {
                            "phase": phase,
                            "dit": dit_kind,
                            "connector": connector_kind,
                            "split": split,
                            "id": row["id"],
                            "caption_id": caption_id,
                            "caption_kind": kind,
                            "normalization": "captured_original_dit_rmsnorm",
                            "audit": audit,
                            "partitions": parts,
                        }
                    )
                    for route in routes:
                        family = (
                            "native" if route.startswith("native_") else connector_kind
                        )
                        pair = {
                            "embeds_sha256": h(caption_id + family),
                            "attention_mask_sha256": h(family + "mask"),
                        }
                        statistic["routes"][route][kind] = pair
                        conditions[(split, row["id"], route, kind)] = pair
                report["condition_statistics"].append(statistic)
    for split, rows in report["selection"].items():
        for row in rows:
            for repeat, t in enumerate(settings["flow_timesteps"]):
                values = {
                    route: {
                        "matched": 0.5,
                        "wrong": 0.55,
                        "wrong_minus_matched": 0.05,
                        "prediction_change_mse": 0.03,
                        "matched_condition": conditions[
                            (split, row["id"], route, "matched")
                        ],
                        "wrong_condition": conditions[
                            (split, row["id"], route, "wrong")
                        ],
                    }
                    for route in renderer.ROUTES
                }
                report["flow_controls"].append(
                    {
                        "split": split,
                        "id": row["id"],
                        "index": row["index"],
                        "repeat": repeat,
                        "wrong_id": (
                            f"train{row['index'] + 1}"
                            if split == "train"
                            else f"val{row['index'] + 1}"
                        ),
                        "seed": 100042
                        + (0 if split == "train" else 50000)
                        + row["index"] * 8
                        + repeat,
                        "requested_timestep": t,
                        "phases": [p[0] for p in renderer.PHASES],
                        "actual_conditioning_verified": True,
                        "actual_inputs": {
                            "noisy_latent_sha256": h(row["id"] + str(t)),
                            "timestep_sha256": h(t),
                            "timestep": [t],
                        },
                        "routes": values,
                    }
                )
    targets, cases = [], {}
    for i, row in enumerate(report["selection"]["validation"]):
        case = f"validation-{i:02d}"
        prompt = "A fixture photograph of a red object beside a blue object, used only for metadata and rendering checks."
        cases[case] = {"split": "validation", "id": row["id"]}
        target_image = Image.new("RGB", (256, 256), "#e8e8e8")
        draw = ImageDraw.Draw(target_image)
        draw.rectangle((32, 50, 110, 140), fill="#b84445")
        draw.ellipse((125, 60, 220, 155), fill="#427baf")
        b = io.BytesIO()
        target_image.save(b, format="JPEG")
        target_raw = b.getvalue()
        members[prefix + "targets/" + row["id"] + ".jpg"] = target_raw
        targets.append(
            {
                "id": row["id"],
                "split": "validation",
                "prompt": prompt,
                "image_sha256": renderer.sha(target_raw),
            }
        )
        for route in renderer.ROUTES:
            positive = conditions[("validation", row["id"], route, "matched")]
            trace = {
                "condition.positive": positive["embeds_sha256"],
                "condition.branch0": positive["embeds_sha256"],
                "mask.positive": positive["attention_mask_sha256"],
                "mask.branch0": positive["attention_mask_sha256"],
                "condition.negative": negative["embeds_sha256"],
                "condition.branch1": negative["embeds_sha256"],
                "mask.negative": negative["attention_mask_sha256"],
                "mask.branch1": negative["attention_mask_sha256"],
                "latents.initial": h(case),
                "latents.final": h(case + route),
                "prediction.step0": h(route + "prediction"),
                "schedule.timesteps": h("schedule"),
            }
            filename = f"sample-{case}-{route}.png"
            b = io.BytesIO()
            target_image.save(b, format="PNG")
            payload = b.getvalue()
            members[prefix + filename] = payload
            report["samples"].append(
                {
                    "case_id": case,
                    "route": route,
                    "path": "/remote/" + filename,
                    "sha256": renderer.sha(payload),
                    "seed": 200042 + i,
                    "initial_latent_sha256": h(case),
                    "prompt": prompt,
                    "text_guidance_scale": 5.0,
                    "sampling_steps": 50,
                    "target_free": True,
                    "quality_claim": False,
                    "actual_condition_verified": True,
                    "negative_conditioner": "original_frozen_native",
                    "trace_sha256": trace,
                }
            )
    members[prefix + "gallery-targets.json"] = raw(
        {"records": targets, "case_targets": cases}
    )
    collector_hashes = {}
    for name in (
        "pack_joint_components.py",
        "render_joint_components.py",
        "pack_run.py",
    ):
        payload = (BASE / name).read_bytes()
        members[prefix + "collector-snapshots/" + name] = payload
        collector_hashes[name] = renderer.sha(payload)
    evidence = {
        "report_sha256": renderer.sha(raw(report)),
        "checkpoint_bytes_included": False,
        "recorded_source_files_unchanged": True,
        "source_identity_checks": {
            k: True for k in [*report["source_sha256"], renderer.RUNNER]
        },
        "collector_sha256": collector_hashes,
        "checkpoint_checks": {},
    }
    for name, lineage in [("region", alignment), ("joint", joint)]:
        evidence["checkpoint_checks"][name] = {
            "independently_verified": True,
            "actual_sha256": lineage["sha256"],
            "expected_sha256": lineage["sha256"],
            "checkpoint": lineage["checkpoint"],
            "report_sha256": lineage["report_sha256"],
        }
    members[prefix + "report.json"] = raw(report)
    members[prefix + "collection-evidence.json"] = raw(evidence)
    return report, members, prefix


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.report, self.members, self.prefix = fixture()

    def audit(self):
        report_raw = raw(self.report)
        evidence = json.loads(self.members[self.prefix + "collection-evidence.json"])
        evidence["report_sha256"] = renderer.sha(report_raw)
        self.members[self.prefix + "collection-evidence.json"] = raw(evidence)
        return renderer.audit_payload(report_raw, self.members, self.prefix)

    def test_complete_fixture(self):
        result = self.audit()
        self.assertTrue(result["artifact_audit_passed"])
        self.assertFalse(result["real_checkpoint_evidence"])
        self.assertEqual(result["images_checked"], 12)

    def test_negative_report_cases(self):
        mutations = {
            "missing_phase": lambda r: r["phase_audits"].pop(),
            "missing_route": lambda r: r["flow_controls"][0]["routes"].pop(
                "joint_joint"
            ),
            "missing_noise": lambda r: r["flow_controls"][0]["actual_inputs"].pop(
                "noisy_latent_sha256"
            ),
            "wrong_flow_condition": lambda r: r["flow_controls"][0]["routes"][
                "joint_joint"
            ]["matched_condition"].update(embeds_sha256=h("corrupt")),
            "changed_invariant": lambda r: r["invariant_frozen_hashes_after"].update(
                new=h("new")
            ),
            "wrong_final": lambda r: r["final_hashes"].update(new=h("new")),
            "wrong_phase_dit": lambda r: r["phase_audits"][0].update(dit="joint"),
            "untrained_negative": lambda r: r["samples"][0].update(
                negative_conditioner="prism"
            ),
            "wrong_guidance": lambda r: r["samples"][0].update(text_guidance_scale=1),
            "wrong_seed": lambda r: r["samples"][0].update(seed=999),
            "wrong_schedule": lambda r: r["samples"][0]["trace_sha256"].update(
                {"schedule.timesteps": h("different")}
            ),
            "wrong_noise": lambda r: r["samples"][0]["trace_sha256"].update(
                {"latents.initial": h("different")}
            ),
            "missing_final_latent": lambda r: r["samples"][0]["trace_sha256"].pop(
                "latents.final"
            ),
            "cfg_extra_branch": lambda r: r["samples"][0]["trace_sha256"].update(
                {"condition.branch2": h("extra")}
            ),
            "wrong_cfg_positive": lambda r: r["samples"][0]["trace_sha256"].update(
                {"condition.branch0": h("wrong")}
            ),
            "feature_tokens_changed": lambda r: r["feature_drift"][-1]["audit"].update(
                prism_hidden_sha256=h("changed")
            ),
            "feature_normalization_changed": lambda r: r["feature_drift"][0].update(
                normalization="adapted"
            ),
            "feature_metric_infinite": lambda r: r["feature_drift"][0]["partitions"][
                "prefix"
            ].update(mse=float("inf")),
            "feature_partition_wrong": lambda r: r["feature_drift"][0]["partitions"][
                "prefix"
            ].update(tokens=9),
            "missing_checkpoint_restore": lambda r: r["alignment_checkpoint"].update(
                region_partition_audit_revalidated=False
            ),
            "nonterminal_joint": lambda r: r["joint_checkpoint"].update(step=5),
            "loss_gap_wrong": lambda r: r["flow_controls"][0]["routes"][
                "joint_joint"
            ].update(wrong_minus_matched=0.9),
            "incomplete_gallery": lambda r: r["samples"].pop(),
            "duplicate_gallery": lambda r: r["samples"].__setitem__(
                1, copy.deepcopy(r["samples"][0])
            ),
            "training_claim": lambda r: r.update(training_performed=True),
            "partial": lambda r: r.update(status="running"),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                self.report, self.members, self.prefix = fixture()
                change(self.report)
                with self.assertRaises((ValueError, KeyError)):
                    self.audit()

    def test_negative_external_evidence(self):
        mutations = {
            "checkpoint_not_verified": lambda m, p: m.__setitem__(
                p + "collection-evidence.json",
                raw(
                    {
                        **json.loads(m[p + "collection-evidence.json"]),
                        "checkpoint_checks": {},
                    }
                ),
            ),
            "source_bytes_changed": lambda m, p: m.__setitem__(
                "prism/" + renderer.RUNNER, b"changed"
            ),
            "source_report_changed": lambda m, p: m.__setitem__(
                p + "joint-source-report.json", b"{}"
            ),
            "collector_changed": lambda m, p: m.__setitem__(
                p + "collector-snapshots/pack_joint_components.py", b"changed"
            ),
            "image_changed": lambda m, p: m.__setitem__(
                p + Path(self.report["samples"][0]["path"]).name, b"changed"
            ),
            "target_changed": lambda m, p: m.__setitem__(
                p + "targets/val0.jpg", b"changed"
            ),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                self.report, self.members, self.prefix = fixture()
                change(self.members, self.prefix)
                with self.assertRaises((ValueError, KeyError)):
                    self.audit()

    def test_collector_archive_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            run = root / "runs/fixture"
            run.mkdir(parents=True)
            source = root / "source"
            source.mkdir()
            for name, payload in self.members.items():
                if name.startswith("prism/"):
                    path = source / name[len("prism/") :]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                elif name.startswith(self.prefix) and name.endswith(".png"):
                    (run / Path(name).name).write_bytes(payload)
            web = root / "webdataset"
            web.mkdir()
            gallery = json.loads(self.members[self.prefix + "gallery-targets.json"])
            with tarfile.open(web / "validation.tar", "w") as tar:
                for row in gallery["records"]:
                    payload = self.members[
                        self.prefix + "targets/" + row["id"] + ".jpg"
                    ]
                    info = tarfile.TarInfo(row["id"] + ".jpg")
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
            rows_by_split = {
                split: [
                    {"id": row["id"], "split": split, "prompt": "unused"}
                    for row in self.report["alignment_checkpoint"]["selection"][split]
                ]
                for split in ("train", "validation")
            }
            with tarfile.open(web / "validation.tar") as tar:
                for i, row in enumerate(gallery["records"]):
                    info = tar.getmember(row["id"] + ".jpg")
                    rows_by_split["validation"][i] = {
                        **row,
                        "member": info.name,
                        "header_offset": info.offset,
                        "data_offset": info.offset_data,
                        "size": info.size,
                        "shard": "validation.tar",
                    }
            for split, rows in rows_by_split.items():
                path = web / (split + ".jsonl")
                path.write_bytes(b"".join(raw(row) + b"\n" for row in rows))
                self.report["settings"][split + "_index"] = str(path)
                self.report["index_sha256"][split] = renderer.sha(path.read_bytes())
            alignment = self.report["alignment_checkpoint"]
            joint = self.report["joint_checkpoint"]
            alignment["index_sha256"] = copy.deepcopy(self.report["index_sha256"])
            region_source = json.loads(
                self.members[self.prefix + "region-source-report.json"]
            )
            joint_source = json.loads(
                self.members[self.prefix + "joint-source-report.json"]
            )
            region_source["index_sha256"] = copy.deepcopy(self.report["index_sha256"])
            for label, lineage, source_report in [
                ("region", alignment, region_source),
                ("joint", joint, joint_source),
            ]:
                directory = root / label
                directory.mkdir()
                cp = directory / "fixture.pt"
                cp.write_bytes((label + " fixture bytes only").encode())
                lineage["checkpoint"] = str(cp)
                lineage["sha256"] = renderer.sha(cp.read_bytes())
                source_report["checkpoints"][0]["sha256"] = lineage["sha256"]
                if label == "joint":
                    source_report["alignment_initialization"] = copy.deepcopy(alignment)
                    source_report["train_index_sha256"] = self.report["index_sha256"][
                        "train"
                    ]
                    source_report["validation_index_sha256"] = self.report[
                        "index_sha256"
                    ]["validation"]
                source_path = directory / "report.json"
                source_path.write_bytes(raw(source_report))
                lineage["report_sha256"] = renderer.sha(source_path.read_bytes())
                setting = (
                    "alignment_checkpoint" if label == "region" else "joint_checkpoint"
                )
                self.report["settings"][setting] = str(cp)
                self.report["settings"][setting + "_sha256"] = lineage["sha256"]
            (run / "report.json").write_bytes(raw(self.report))
            result = packer.pack(
                root, "fixture", source_root=source, verify_checkpoints=True
            )
            self.assertTrue(result["final_acceptance_passed"])
            with tarfile.open(result["archive"]) as archive:
                names = archive.getnames()
                self.assertFalse(any(name.endswith(".pt") for name in names))
                self.assertTrue(all(item.isfile() for item in archive.getmembers()))
                # Safe regular-file extraction of this trusted test-generated archive.
                extracted = root / "extracted"
                extracted.mkdir()
                for item in archive.getmembers():
                    self.assertNotIn("..", Path(item.name).parts)
                    target = extracted / item.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(item).read())
            audit = renderer.render(extracted / "runs/fixture", audit_only=True)
            self.assertTrue(audit["artifact_audit_passed"])
            (root / "joint/fixture.pt").write_bytes(b"corrupt checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint digest"):
                packer.pack(
                    root, "fixture", source_root=source, verify_checkpoints=True
                )

    def test_render_and_relocated_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, payload in self.members.items():
                if name.startswith("prism/"):
                    name = (
                        "provenance/source-snapshots/fixture/" + name[len("prism/") :]
                    )
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            result = renderer.render(root / "runs/fixture")
            self.assertTrue(Path(result["gallery"]).is_file())
            self.assertEqual(
                result["gallery_sha256"],
                renderer.sha(Path(result["gallery"]).read_bytes()),
            )
            # Keep one explicit fixture image for visual QA; no generation-quality claim.
            (BASE / "fixture-gallery.png").write_bytes(
                Path(result["gallery"]).read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
