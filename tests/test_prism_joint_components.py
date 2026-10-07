"""Offline evidence-protocol tests, not proof of accelerator or image quality."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tools import diagnose_prism_joint_components as diagnostic
from tools.train_image_decoder import _tensor_hash
from torch import nn


class Component(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0).reshape(2, 2))
        self.register_buffer("ephemeral", torch.tensor([4.0]), persistent=False)


def test_snapshot_roundtrip_includes_nonpersistent_buffer_and_detaches_storage():
    module = Component()
    hashes = diagnostic.state_hashes(module)
    saved = diagnostic.capture_state(module, connector=True)
    assert set(saved) == {"weight", "ephemeral"}
    assert all(value.device.type == "cpu" and not value.requires_grad for value in saved.values())
    assert saved["weight"].data_ptr() != module.weight.data_ptr()
    with torch.no_grad():
        module.weight.add_(2)
        module.ephemeral.add_(3)
    diagnostic.restore_state(module, saved, hashes)
    assert diagnostic.state_hashes(module) == hashes


@pytest.mark.parametrize(
    "damage", ["missing", "buffer_missing", "dtype", "shape", "digest", "nan", "requires_grad"]
)
def test_snapshot_rejects_all_damage_before_any_copy(damage):
    module = Component()
    saved = diagnostic.capture_state(module)
    hashes = diagnostic.state_hashes(module)
    with torch.no_grad():
        module.weight.add_(10)
    before = diagnostic.state_hashes(module)
    if damage == "missing":
        del saved["weight"]
    elif damage == "buffer_missing":
        del saved["ephemeral"]
    elif damage == "dtype":
        saved["weight"] = saved["weight"].bfloat16()
    elif damage == "shape":
        saved["weight"] = saved["weight"].reshape(-1)
    elif damage == "digest":
        saved["weight"][0, 0] += 1
    elif damage == "nan":
        saved["weight"][0, 0] = float("nan")
    else:
        saved["weight"].requires_grad_(True)
    with pytest.raises(ValueError):
        diagnostic.restore_state(module, saved, hashes)
    assert diagnostic.state_hashes(module) == before


def test_connector_capture_requires_real_fp32_not_bf16_upcast():
    with pytest.raises(ValueError, match="FP32"):
        diagnostic.capture_state(Component().bfloat16(), connector=True)


def lineage_fixture():
    lineage = {
        "sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "checkpoint": "/region.pt",
        "selection": {"train": [{"index": 0, "id": "t0"}], "validation": []},
        "parent": {"strict": True},
        "index_sha256": {"train": "c" * 64, "validation": "d" * 64},
        "data_fingerprint": "data",
        "reference_checkpoint_sha256": "e" * 64,
        "evidence_kind": "real_checkpoint_connector_native_region_feature_alignment",
        "schema_version": 2,
    }
    report = {
        "alignment_initialization": copy.deepcopy(lineage),
        "status": "completed",
        "completed_steps": 6,
        "frozen_state_unchanged": True,
        "data_fingerprint": "data",
        "train_index_sha256": "c" * 64,
        "validation_index_sha256": "d" * 64,
        "settings": {"prompt_format": "chat", "steps": 6},
        "train_selection": lineage["selection"]["train"],
    }
    return lineage, report


def test_joint_lineage_accepts_actual_runner_field_names():
    lineage, report = lineage_fixture()
    diagnostic.validate_joint_lineage(
        report, lineage, data_fingerprint="data", index_sha256=lineage["index_sha256"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "region_kind",
        "schema",
        "index",
        "prompt",
        "steps",
        "selection",
        "frozen",
        "status",
    ],
)
def test_joint_lineage_rejects_unrelated_or_incomplete_joint_run(damage):
    lineage, report = lineage_fixture()
    if damage == "digest":
        report["alignment_initialization"]["sha256"] = "f" * 64
    elif damage == "region_kind":
        report["alignment_initialization"]["evidence_kind"] = "feature_alignment_v1"
    elif damage == "schema":
        report["alignment_initialization"]["schema_version"] = 1
    elif damage == "index":
        report["train_index_sha256"] = "f" * 64
    elif damage == "prompt":
        report["settings"]["prompt_format"] = "raw"
    elif damage == "steps":
        report["settings"]["steps"] = 5
    elif damage == "selection":
        report["train_selection"] = []
    elif damage == "frozen":
        report["frozen_state_unchanged"] = False
    else:
        report["status"] = "running"
    with pytest.raises(ValueError):
        diagnostic.validate_joint_lineage(
            report, lineage, data_fingerprint="data", index_sha256=lineage["index_sha256"]
        )


def flow_result(route):
    return {
        "split": "train",
        "index": 2,
        "id": "t2",
        "wrong_id": "t3",
        "repeat": 0,
        "seed": 42,
        "requested_timestep": 0.1,
        "actual_inputs": {"noise": "a", "time": "b"},
        "actual_conditioning_verified": True,
        "routes": {route: {"matched": 1, "wrong": 2}},
    }


@pytest.mark.parametrize("field", ["actual_inputs", "seed", "id", "wrong_id", "requested_timestep"])
def test_cross_phase_inputs_must_match_actual_bytes_and_caption_identity(field):
    rows = {}
    diagnostic.merge_flow(rows, flow_result("native_original"), phase="first")
    changed = flow_result("joint_original")
    changed[field] = "different"
    with pytest.raises(RuntimeError, match="actual paired"):
        diagnostic.merge_flow(rows, changed, phase="second")
    assert list(rows.values())[0]["routes"] == flow_result("native_original")["routes"]


def test_merge_rejects_duplicate_routes_or_missing_actual_condition_audit():
    rows = {}
    first = flow_result("native_original")
    diagnostic.merge_flow(rows, first, phase="first")
    with pytest.raises(RuntimeError, match="Duplicate"):
        diagnostic.merge_flow(rows, first, phase="second")
    broken = flow_result("joint_original")
    broken["actual_conditioning_verified"] = False
    with pytest.raises(RuntimeError, match="actual conditioning"):
        diagnostic.merge_flow(rows, broken, phase="second")


def condition(value=1):
    return {
        "embeds": torch.full((1, 3, 2), float(value), dtype=torch.bfloat16),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def trace_fixture(positive, negative):
    result = {}
    for label, value, branch in (("positive", positive, 0), ("negative", negative, 1)):
        result[f"condition.{label}"] = value["embeds"].clone()
        result[f"condition.branch{branch}"] = value["embeds"].clone()
        result[f"mask.{label}"] = value["attention_mask"].clone()
        result[f"mask.branch{branch}"] = value["attention_mask"].clone()
    for key in ("latents.initial", "latents.final", "prediction.step0", "schedule.timesteps"):
        result[key] = torch.ones(1, dtype=torch.bfloat16)
    return result


def test_cfg5_trace_preserves_distinct_positive_negative_and_final_latent_hash():
    positive, negative = condition(1), condition(2)
    trace = trace_fixture(positive, negative)
    trace["latents.final"] *= 3
    hashes = diagnostic.verify_cfg5_trace(trace, positive, negative)
    assert hashes["condition.branch0"] != hashes["condition.branch1"]
    assert hashes["latents.initial"] != hashes["latents.final"]


@pytest.mark.parametrize(
    "damage",
    [
        "positive",
        "negative",
        "branch0",
        "branch1",
        "mask",
        "extra_branch",
        "no_final",
        "no_schedule",
        "nan",
    ],
)
def test_cfg5_rejects_trace_corruption(damage):
    positive, negative = condition(1), condition(2)
    trace = trace_fixture(positive, negative)
    if damage in ("positive", "negative", "branch0", "branch1"):
        trace["condition." + damage].add_(1)
    elif damage == "mask":
        trace["mask.branch1"][0, 0] = False
    elif damage == "extra_branch":
        trace["condition.branch2"] = positive["embeds"]
    elif damage == "no_final":
        del trace["latents.final"]
    elif damage == "no_schedule":
        del trace["schedule.timesteps"]
    else:
        trace["latents.final"][0] = float("nan")
    with pytest.raises(RuntimeError):
        diagnostic.verify_cfg5_trace(trace, positive, negative)


class SamplingBackend:
    def __init__(self, positive, negative, corrupt_noise=False):
        self.positive, self.negative, self.corrupt_noise = positive, negative, corrupt_noise
        self.last_trace = {}

    def generate_conditioned(self, embeds, mask, **kwargs):
        assert kwargs["text_guidance_scale"] == 5 and kwargs["negative_prompt"] == ""
        assert "negative_prompt_embeds" not in kwargs
        self.last_trace = trace_fixture(self.positive, self.negative)
        if "latents" in kwargs:
            self.last_trace["latents.initial"] = kwargs["latents"].clone()
        if self.corrupt_noise:
            self.last_trace["latents.initial"].add_(1)
        return [Image.new("RGB", (8, 8), color="red")]

    def generate_reference(self, native, **kwargs):
        return self.generate_conditioned(None, None, **kwargs)


def test_sampling_writes_only_image_with_condition_and_noise_proof(tmp_path):
    positive, negative = condition(1), condition(2)
    args = SimpleNamespace(
        height=256,
        width=256,
        sampling_steps=50,
        max_text_length=1024,
        device="cpu",
        dtype="bfloat16",
        output_dir=tmp_path,
    )
    sample, latent = diagnostic.sample_route(
        SamplingBackend(positive, negative),
        "dog",
        positive,
        negative,
        route="aligned_original",
        native=False,
        args=args,
        case_id="validation-00",
        seed=200042,
        saved_latent=None,
    )
    assert sample["actual_condition_verified"] and sample["target_free"]
    assert list(tmp_path.iterdir()) == [Path(sample["path"])]
    with pytest.raises(RuntimeError, match="initial noise"):
        diagnostic.sample_route(
            SamplingBackend(positive, negative, corrupt_noise=True),
            "dog",
            positive,
            negative,
            route="joint_original",
            native=False,
            args=args,
            case_id="validation-00",
            seed=200042,
            saved_latent=latent,
        )


def command_fixture(tmp_path):
    result = []
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "train-index",
        "validation-index",
        "repeatability-report",
        "connector-checkpoint",
        "alignment-checkpoint",
        "joint-checkpoint",
        "output-dir",
    ):
        result += ["--" + name, str(tmp_path / name)]
    for name in (
        "connector-checkpoint-sha256",
        "alignment-checkpoint-sha256",
        "joint-checkpoint-sha256",
    ):
        result += ["--" + name, "a" * 64]
    return result + ["--deterministic"]


def test_cli_accepts_both_checkpoint_types_and_enforces_pbs(tmp_path, monkeypatch):
    argv = command_fixture(tmp_path)
    args = diagnostic._parser().parse_args(argv)
    diagnostic.validate_budget(args)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="PBS"):
        diagnostic.main(argv)


@pytest.mark.parametrize(
    "change",
    ["seed", "sample_count", "sampling_steps", "train_probe_count", "alignment_checkpoint_sha256"],
)
def test_cli_rejects_unreviewed_protocol_drift(tmp_path, change):
    args = diagnostic._parser().parse_args(command_fixture(tmp_path))
    setattr(args, change, "x" if change.endswith("sha256") else 1)
    with pytest.raises(ValueError):
        diagnostic.validate_budget(args)


def test_feature_pair_casts_actual_connected_features_and_uses_original_norm(monkeypatch):
    import tools.align_prism_image_conditioning as align
    import tools.prism_image_conditioning as conditioning

    ids = torch.tensor([[9, 2, 3, 8]])
    native = {
        "formatted_prompt": "prefix caption suffix",
        "input_ids": ids,
        "input_attention_mask": torch.ones_like(ids),
        "attention_mask": torch.ones_like(ids),
        "embeds": torch.ones(1, 4, 2, dtype=torch.bfloat16),
        "format": "native",
        "empty_anchor": None,
    }
    prism = copy.deepcopy(native) | {
        "embeds": torch.ones(1, 4, 2, dtype=torch.float32) * 2,
        "hidden_states": torch.ones(1, 4, 2, dtype=torch.bfloat16),
        "format": "chat",
    }
    monkeypatch.setattr(conditioning, "encode_prism_prompt", lambda *a, **k: prism)
    monkeypatch.setattr(align, "encode_observed_native", lambda *a, **k: native)

    def tokenizer(*a, **k):
        return {"input_ids": torch.tensor([[2, 3]]), "attention_mask": torch.ones(1, 2)}

    norm = nn.RMSNorm(2).bfloat16().requires_grad_(False)
    before = diagnostic.state_hashes(norm)
    positive, teacher, metrics = diagnostic.feature_pair(
        None,
        tokenizer,
        None,
        "caption",
        norm,
        args=SimpleNamespace(device="cpu", max_text_length=1024),
    )
    assert positive["embeds"].dtype == torch.bfloat16
    assert metrics["audit"]["content_span"] == [1, 3]
    assert metrics["normalization"] == "captured_original_dit_rmsnorm"
    assert set(metrics["partitions"]) == {"prefix", "content", "suffix"}
    assert diagnostic.state_hashes(norm) == before
    assert diagnostic.runtime_condition_metadata(positive)["embeds_sha256"] == _tensor_hash(
        positive["embeds"]
    )


def test_actual_collected_smoke_report_admission_when_available():
    path = Path(
        "/private/tmp/prism-connector-20260921/docs/assets/image_generation/2026-09-22-docci-alignment/runs/aligned-regions-joint-smoke-01/report.json"
    )
    if not path.exists():
        pytest.skip("Real metadata artifact is not present in this test checkout")
    report = json.loads(path.read_text())
    lineage = report["alignment_initialization"]
    diagnostic.validate_joint_lineage(
        report,
        lineage,
        data_fingerprint=report["data_fingerprint"],
        index_sha256=lineage["index_sha256"],
    )


@pytest.mark.parametrize("fail_in_phase", [False, True])
def test_integrated_four_phase_protocol_restores_original_runtime_on_success_or_error(
    tmp_path, monkeypatch, fail_in_phase
):
    import src.data.image_generation_webdataset as data_module
    import src.decoders.loading as loading
    import tools.align_prism_image_conditioning as align
    import tools.prism_image_alignment_checkpoint as alignment_restore
    import tools.train_prism_image_connector as connector_runner
    import tools.train_prism_image_diffusion as joint_runner

    class Transformer(Component):
        def __init__(self):
            super().__init__()
            self.time_caption_embed = nn.Module()
            self.time_caption_embed.caption_embedder = nn.Sequential(nn.RMSNorm(2))

    def encoded(value):
        result = condition(value)
        return result | {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_attention_mask": result["attention_mask"],
            "formatted_prompt": "caption",
            "format": "native",
            "empty_anchor": None,
        }

    class Backend(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()
            self.vae = Component()
            self.mllm = Component()
            self.last_trace = {}

        def ensure_loaded(self):
            pass

        def checkpoint_manifest(self):
            return {"manifest_sha256": "generator"}

        def provenance(self):
            return {"backend": "offline_fixture"}

        def configure_training(self, *, train_diffusion, gradient_checkpointing):
            self.train_diffusion = train_diffusion
            self.gradient_checkpointing = gradient_checkpointing
            self.transformer.requires_grad_(train_diffusion)

        def _sample(self, positive, kwargs):
            self.last_trace = trace_fixture(positive, encoded(0))
            if "latents" in kwargs:
                self.last_trace["latents.initial"] = kwargs["latents"].clone()
            return [Image.new("RGB", (8, 8), color="blue")]

        def generate_reference(self, context, **kwargs):
            return self._sample(encoded(3), kwargs)

        def generate_conditioned(self, embeds, mask, **kwargs):
            return self._sample({"embeds": embeds, "attention_mask": mask}, kwargs)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.connector = Component()
            self.backend = Backend()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Component()
            self.decoders = nn.ModuleDict({"image": Decoder()})

    class Dataset:
        def __init__(self, index, *, target_size, split):
            self.index = index
            self.data_fingerprint = "data"
            self.records = [
                SimpleNamespace(
                    id=f"{split}{i}",
                    prompt=f"caption {i}",
                    task="t2i",
                    split=split,
                    source_ids=[],
                    source_paths=[],
                )
                for i in range(32 if split == "train" else 8)
            ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return {"target_image": torch.zeros(3, 8, 8)}

    argv = command_fixture(tmp_path)
    args = diagnostic._parser().parse_args(argv)
    args.device = "cpu"
    args.output_dir = tmp_path / "output"
    for key, value in vars(args).items():
        if isinstance(value, Path) and key != "output_dir":
            value.write_text(key)
    parent = {
        "fixture": True,
        "restoration": {
            "strict_parent": True,
            "loaded_key_count": 526,
            "missing_parent_keys": [],
            "unexpected_keys": [],
        },
    }
    indices = {
        split: loading.file_sha256(getattr(args, split + "_index"))
        for split in ("train", "validation")
    }
    lineage, joint_report = lineage_fixture()
    lineage.update(
        parent=parent,
        index_sha256=indices,
        selection={
            split: [{"id": f"{split}{i}", "index": i} for i in range(32 if split == "train" else 8)]
            for split in ("train", "validation")
        },
    )
    joint_report.update(
        alignment_initialization=copy.deepcopy(lineage),
        train_index_sha256=indices["train"],
        validation_index_sha256=indices["validation"],
        train_selection=lineage["selection"]["train"],
    )
    (tmp_path / "report.json").write_text(json.dumps(joint_report))
    model = Model()
    initial_dit, aligned_connector, events = {}, {}, []
    monkeypatch.setattr(data_module, "ImageGenerationWebDataset", Dataset)
    monkeypatch.setattr(
        loading,
        "load_image_training_bundle",
        lambda *a: {"model": model, "tokenizer": object(), "provenance": parent},
    )
    monkeypatch.setattr(
        connector_runner, "validate_repeatability_report", lambda *a, **k: {"fixture": True}
    )
    monkeypatch.setattr(joint_runner, "restore_warm_connector", lambda *a, **k: {"fixture": True})

    def restore_region(actual, *a, **k):
        assert not events
        events.append("strict_region_original")
        with torch.no_grad():
            actual.decoders["image"].connector.weight.add_(1)
        initial_dit.update(diagnostic.state_hashes(actual.decoders["image"].backend.transformer))
        aligned_connector.update(diagnostic.state_hashes(actual.decoders["image"].connector))
        return copy.deepcopy(lineage)

    def configure(actual):
        actual.train().requires_grad_(True)
        return {"fixture": True}

    def restore_joint(actual, *a, **k):
        assert events == ["strict_region_original"]
        assert k["restore_masters"] is False
        assert diagnostic.state_hashes(actual.decoders["image"].backend.transformer) == initial_dit
        events.append("strict_joint")
        with torch.no_grad():
            actual.decoders["image"].connector.weight.add_(1)
            actual.decoders["image"].backend.transformer.weight.add_(1)
            actual.decoders["image"].backend.transformer.ephemeral.add_(1)
        return {}, {"fixture": True}

    monkeypatch.setattr(alignment_restore, "restore_alignment_connector", restore_region)
    monkeypatch.setattr(joint_runner, "configure_joint_scope", configure)
    monkeypatch.setattr(joint_runner, "restore_joint_stage", restore_joint)
    monkeypatch.setattr(align, "encode_observed_native", lambda *a, **k: encoded(0))

    def pair(actual, tokenizer, backend, prompt, original_norm, *, args):
        assert not backend.train_diffusion and not backend.gradient_checkpointing
        assert all(not p.requires_grad for p in actual.parameters())
        assert all(not m.training for m in actual.modules())
        positive = encoded(float(actual.decoders["image"].connector.weight[0, 0]))
        return (
            positive,
            encoded(3),
            {
                "normalization": "captured_original_dit_rmsnorm",
                "partitions": {
                    region: {"mse": 1.0, "cosine": 0.5, "tokens": 1}
                    for region in ("prefix", "content", "suffix")
                },
                "audit": {"prompt": prompt},
            },
        )

    def paired(backend, target, conditions, *, seed, timestep, device, verify_conditioning):
        assert verify_conditioning
        if fail_in_phase and "joint_original" in conditions:
            raise RuntimeError("injected phase failure")
        route_values = {
            route: {
                "matched": 1.0,
                "wrong": 2.0,
                "wrong_minus_matched": 1.0,
                "prediction_change_mse": 1.0,
                **{
                    kind + "_condition": {
                        "embeds_sha256": _tensor_hash(value["embeds"]),
                        "attention_mask_sha256": _tensor_hash(value["attention_mask"]),
                    }
                    for kind, value in pair.items()
                },
            }
            for route, pair in conditions.items()
        }
        return {
            "seed": seed,
            "requested_timestep": timestep,
            "actual_inputs": {"seed": seed, "time": timestep},
            "actual_conditioning_verified": True,
            "routes": route_values,
        }

    monkeypatch.setattr(diagnostic, "feature_pair", pair)
    monkeypatch.setattr(diagnostic, "paired_flow_losses", paired)
    if fail_in_phase:
        with pytest.raises(RuntimeError, match="injected phase failure"):
            diagnostic._run(args)
        report = json.loads((args.output_dir / "report.json").read_text())
        assert report["status"] == "failed"
    else:
        report = diagnostic._run(args)
        assert report["status"] == "completed"
        assert len(report["flow_controls"]) == 12
        assert all(set(row["routes"]) == set(diagnostic.ROUTES) for row in report["flow_controls"])
        assert len(report["samples"]) == 12 and len(report["feature_drift"]) == 32
        assert len(report["phase_audits"]) == 4
        assert all(audit["frozen_state_unchanged"] for audit in report["phase_audits"])
    assert report["final_state_restored"] and report["invariant_frozen_state_unchanged"]
    assert diagnostic.state_hashes(model.decoders["image"].backend.transformer) == initial_dit
    assert diagnostic.state_hashes(model.decoders["image"].connector) == aligned_connector
    assert len(list(args.output_dir.glob("*.pt"))) == 0
