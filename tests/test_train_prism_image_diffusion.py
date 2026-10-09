"""Offline dense-training contracts; no accelerator or generation capability claim."""

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from applications.vision_language.prepare_docci_webdataset import convert_docci
from PIL import Image
from safetensors.torch import save_file
from src.data.image_generation_webdataset import ImageGenerationWebDataset
from src.decoders.image import ImageDecoder
from src.decoders.loading import file_sha256
from src.decoders.omnigen2_backend import OmniGen2Backend
from tools.train_image_decoder import frozen_state_hashes
from tools.train_prism_image_connector import PRECHECK_SOURCES, ROOT
from tools.train_prism_image_diffusion import (
    CONNECTOR_PREFIX,
    DIFFUSION_PREFIX,
    MODULES,
    _parser,
    configure_joint_scope,
    evaluation_mode,
    load_original_masters,
    main,
    restore_joint_stage,
    restore_warm_connector,
    selected_training_indices,
    validate_budget,
)
from torch import nn


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": False,
        }
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
        return "CHAT:" + messages[0]["content"] + ":" + messages[1]["content"]

    def __call__(self, prompts, **kwargs):
        ids = torch.tensor([[sum(map(ord, prompt)) % 7 + 1, 2, 3] for prompt in prompts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.active = nn.Linear(3, 3)
        self.reference_only = nn.Linear(3, 3, bias=False)
        self.register_buffer("training_calls", torch.tensor(0))
        self.gradient_checkpointing = False
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.linspace(0.012345, 0.123456, parameter.numel()).reshape(parameter.shape)
                )

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(self, value):
        if self.training:
            self.training_calls += 1
        return self.active(value)


class TinyBackend(OmniGen2Backend):
    def __init__(self, root):
        super().__init__(model_id=str(root), conditioning_dim=3)
        self.transformer = TinyTransformer()
        self.vae = nn.Linear(3, 3, bias=False)
        self.mllm = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.vae.weight.copy_(torch.eye(3))
            self.mllm.weight.copy_(torch.eye(3))
        self._pipeline = SimpleNamespace(transformer=self.transformer, vae=self.vae, mllm=self.mllm)
        self._apply_training_policy()
        self.last_trace = {}
        self.native_calls = 0
        self.poison = False

    def checkpoint_manifest(self):
        path = Path(self.model_id) / "transformer/model.safetensors"
        return {
            "manifest_sha256": "b" * 64,
            "model_id": self.model_id,
            "files": {"transformer/model.safetensors": file_sha256(path)},
        }

    def provenance(self):
        return {
            "kernel_policy": {"policy": "fixture"},
            "training_scope": "connector_and_diffusion"
            if self.train_diffusion
            else "connector_only",
            "gradient_checkpointing": self._gradient_checkpointing,
        }

    def training_step(self, embeds, attention_mask, targets, native_context=None, **kwargs):
        assert not self.vae.training and not self.mllm.training
        assert not self.vae.weight.requires_grad and not self.mllm.weight.requires_grad
        output = self.transformer(embeds.mean(1))
        loss = (output - targets.mean((-2, -1)) + torch.randn_like(output) * 0.001).square().mean()
        if self.poison and torch.is_grad_enabled():
            loss *= float("nan")
        return output, loss

    def _sample(self, options):
        assert not self.transformer.training and not torch.is_grad_enabled()
        latent = options.get("latents")
        if latent is None:
            latent = torch.randn(1, 3, 2, 2, generator=options["generator"])
        self.last_trace = {"latents.initial": latent.detach().clone()}
        return [Image.new("RGB", (32, 32), "blue")]

    def generate_conditioned(self, embeds, mask, native_context=None, **options):
        return self._sample(options)

    def generate_reference(self, native_context, **options):
        self.native_calls += 1
        assert int(self.transformer.training_calls) == 0
        return self._sample(options)


class TinyParent(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(3))
        self.decoders = nn.ModuleDict({"image": ImageDecoder(3, backend=TinyBackend(root))})

    def forward_outputs(self, inputs, *, targets, requested_outputs, native_context, output_specs):
        assert not self.training and not self.backbone.training and not self.dropout.training
        hidden = self.backbone(inputs["text"].float()).unsqueeze(1)
        _, loss = self.decoders["image"](
            hidden,
            targets=targets["image"],
            native_context=native_context["image"],
            output_spec=output_specs["image"],
        )
        return SimpleNamespace(losses={"image": loss})

    def predict(self, *, inputs, requested_outputs, native_context, decoder_kwargs):
        hidden = self.backbone(inputs["text"].float()).unsqueeze(1)
        images = self.decoders["image"].generate(
            hidden, native_context=native_context["image"], **decoder_kwargs["image"]
        )
        return SimpleNamespace(predictions={"image": images})


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import src.decoders.loading as loading

    descriptions, rows = tmp_path / "descriptions.jsonl", []
    archive = tmp_path / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for index, split in enumerate(["train"] * 3 + ["qual_dev"] * 2 + ["test", "qual_test"]):
            identifier = f"{split}_{index:05d}"
            rows.append(
                {
                    "example_id": identifier,
                    "image_file": identifier + ".jpg",
                    "split": split,
                    "description": f"A square with index {index}.",
                }
            )
            buffer = io.BytesIO()
            Image.new("RGB", (12, 8), (20 + index * 25, 50, 170)).save(buffer, format="JPEG")
            header = tarfile.TarInfo("images/" + identifier + ".jpg")
            header.size = len(buffer.getvalue())
            handle.addfile(header, io.BytesIO(buffer.getvalue()))
    descriptions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = tmp_path / "data"
    convert_docci(descriptions, archive, data, expected_counts=None)
    generator_root = tmp_path / "generator"
    (generator_root / "transformer").mkdir(parents=True)
    torch.manual_seed(42)
    prototype = TinyParent(generator_root)
    save_file(
        prototype.decoders["image"].backend.transformer.state_dict(),
        generator_root / "transformer/model.safetensors",
    )
    parent = {
        "parent_checkpoint_sha256": "a" * 64,
        "fixture": True,
        "restoration": {
            "strict_parent": True,
            "loaded_key_count": 2,
            "missing_parent_keys": [],
            "unexpected_keys": [],
        },
    }
    fingerprint = ImageGenerationWebDataset(data / "train.jsonl").data_fingerprint
    warm_root = tmp_path / "warm"
    warm_root.mkdir()
    warm = warm_root / "connector-pilot-step-000500.pt"
    frozen = frozen_state_hashes(prototype, (MODULES[0],))
    torch.save(
        {
            "schema_version": 1,
            "evidence_kind": "fixture_only",
            "qualification": "unqualified",
            "step": 500,
            "connector_modules": [MODULES[0]],
            "connector_state_dict": {
                name: value
                for name, value in prototype.state_dict().items()
                if name.startswith(CONNECTOR_PREFIX)
            },
            "protocol": {
                "parent": parent,
                "reference_checkpoint_sha256": "b" * 64,
                "data_fingerprint": fingerprint,
            },
            "frozen_hashes": frozen,
            "optimizer_state_dict": {"must_not_restore": True},
        },
        warm,
    )
    (warm_root / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_kind": "fixture_only",
                "completed_steps": 500,
                "frozen_state_unchanged": True,
                "frozen_hashes_before": frozen,
                "frozen_hashes_after": frozen,
                "checkpoints": [{"step": 500, "sha256": file_sha256(warm)}],
            }
        )
    )
    precheck = tmp_path / "repeatability.json"
    precheck.write_text(
        json.dumps(
            {
                "status": "completed",
                "mode": "repeatability_diagnostic",
                "evidence_kind": "real_checkpoint_diagnostic",
                "fixture": False,
                "comparisons": {
                    key: {"passed": True}
                    for key in ("native_repeat", "adapter_vs_native", "native_after_adapter")
                },
                "source_hashes": {name: file_sha256(ROOT / name) for name in PRECHECK_SOURCES},
                "identity": {
                    "dtype": "float32",
                    "device_type": "cpu",
                    "device_name": "cpu",
                    "torch_version": str(torch.__version__),
                    "numerical_policy": {
                        "deterministic_algorithms": True,
                        "attention_backend": "math",
                    },
                    "kernel_policy": {"policy": "fixture"},
                    "checkpoint_manifest_sha256": "b" * 64,
                },
            }
        )
    )
    assets = {
        "train-index": data / "train.jsonl",
        "validation-index": data / "validation.jsonl",
        "connector-checkpoint": warm,
        "repeatability-report": precheck,
    }
    for name in ("model-config", "checkpoint", "tokenizer", "source-processor"):
        assets[name] = tmp_path / name
        assets[name].write_text("fixture")
    models, poison = [], [False]

    def load(*args):
        model = TinyParent(generator_root)
        model.decoders["image"].backend.poison = poison[0]
        models.append(model)
        return {
            "model": model,
            "tokenizer": TinyTokenizer(),
            "source_transform": None,
            "provenance": parent,
        }

    monkeypatch.setattr(loading, "load_image_training_bundle", load)
    args = [value for name, path in assets.items() for value in ("--" + name, str(path))]
    args += [
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--height",
        "32",
        "--width",
        "32",
        "--expected-parent-tensors",
        "2",
        "--deterministic",
        "--probe-count",
        "1",
        "--final-validation-count",
        "1",
        "--sample-count",
        "0",
        "--sampling-steps",
        "2",
        "--learning-rate",
        "0.01",
        "--diffusion-learning-rate",
        "0.01",
    ]
    return SimpleNamespace(
        root=tmp_path,
        args=args,
        prototype=prototype,
        parent=parent,
        fingerprint=fingerprint,
        warm=warm,
        generator_root=generator_root,
        models=models,
        poison=poison,
    )


def run(experiment, name, steps, *extra):
    output = experiment.root / name
    result = main(experiment.args + ["--output-dir", str(output), "--steps", str(steps), *extra])
    return result, json.loads((output / "report.json").read_text())


def test_scope_and_evaluation_modes_keep_frozen_root_eval(experiment):
    model = experiment.prototype
    groups = configure_joint_scope(model)
    assert set(groups) == {"connector", "diffusion"}
    assert not model.training and not model.dropout.training
    assert model.decoders["image"].connector.training
    assert model.decoders["image"].backend.transformer.training
    with evaluation_mode(model):
        assert not any(module.training for module in model.modules())
    assert model.decoders["image"].backend.transformer.training and not model.training
    assert all(value.requires_grad for group in groups.values() for value in group.values())


def test_full_training_updates_both_groups_keeps_parent_frozen_and_saves_terminal_only(experiment):
    result, report = run(experiment, "joint", 2)
    assert result == 0 and report["status"] == "completed"
    assert report["evidence_kind"] == "fixture_only" and report["qualification"] == "unqualified"
    assert report["frozen_state_unchanged"] and all(report["trainable_groups_changed"].values())
    assert report["generator_runtime"]["gradient_checkpointing"]
    assert report["pretrained_generator_runtime"]["training_scope"] == "connector_only"
    assert len(report["checkpoints"]) == 1 and report["checkpoints"][0]["step"] == 2
    saved = torch.load(report["checkpoints"][0]["path"], weights_only=True)
    assert saved["master_state_includes_full_diffusion"]
    assert any(
        name.startswith(DIFFUSION_PREFIX) for name in saved["optimizer_state_dict"]["masters"]
    )
    assert saved["diffusion_buffer_state"]["training_calls"].item() == 2
    assert report["last_optimizer_diagnostics"]["groups"]["diffusion"][
        "missing_gradient_names"
    ] == [DIFFUSION_PREFIX + "reference_only.weight"]
    assert not report["samples"]
    assert all(
        value.dtype == torch.float32 for value in saved["optimizer_state_dict"]["masters"].values()
    )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def test_joint_resume_matches_full_optimizer_parameters_rng_and_sampler(experiment):
    _, full = run(experiment, "full", 4)
    _, first = run(experiment, "first", 2)
    _, resumed = run(experiment, "resume", 4, "--resume", first["checkpoints"][-1]["path"])
    expected = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    actual = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    for key in (
        "connector_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "sampler_state",
        "diffusion_buffer_state",
    ):
        _equal(expected[key], actual[key])
    assert full["evaluations"][-1] == resumed["evaluations"][-1]


def test_native_baseline_is_only_before_training_and_latents_replay(experiment):
    _, report = run(experiment, "samples", 2, "--sample-count", "1", "--native-baseline")
    assert experiment.models[-1].decoders["image"].backend.native_calls == 1
    assert all(row["step"] == 0 for row in report["samples"] if row["stage"] == "native-pretrained")
    for identifier in {row["id"] for row in report["samples"]}:
        selected = [row for row in report["samples"] if row["id"] == identifier]
        assert len({row["initial_latent_sha256"] for row in selected}) == 1


def test_bad_loss_aborts_with_failed_partial_report(experiment):
    experiment.poison[0] = True
    with pytest.raises(RuntimeError, match="finite scalar"):
        run(experiment, "bad", 2)
    report = json.loads((experiment.root / "bad/report.json").read_text())
    assert (
        report["status"] == "failed"
        and report["completed_steps"] == 0
        and report["frozen_state_unchanged"]
    )
    assert not report["checkpoints"]


@pytest.mark.parametrize("change", ["parent", "data", "step", "hash", "report"])
def test_warm_start_rejects_mismatched_or_unaudited_connector(experiment, change):
    kwargs = {
        "parent": experiment.parent,
        "reference_sha256": "b" * 64,
        "data_fingerprint": experiment.fingerprint,
        "frozen_hashes": frozen_state_hashes(experiment.prototype, MODULES),
        "expected_step": 500,
        "fixture": True,
    }
    if change == "parent":
        kwargs["parent"] = {"different": True}
    elif change == "data":
        kwargs["data_fingerprint"] = "different"
    elif change == "step":
        kwargs["expected_step"] = 499
    elif change == "hash":
        kwargs["expected_sha256"] = "f" * 64
    else:
        report = json.loads((experiment.warm.parent / "report.json").read_text())
        report["frozen_state_unchanged"] = False
        (experiment.warm.parent / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="[Ww]arm|checkpoint"):
        restore_warm_connector(experiment.prototype, experiment.warm, **kwargs)


def test_original_master_loader_preserves_f32_not_bf16_round_trip(experiment):
    groups = configure_joint_scope(experiment.prototype)
    experiment.prototype.decoders["image"].backend.transformer.bfloat16()
    manifest = experiment.prototype.decoders["image"].backend.checkpoint_manifest()
    masters, audit = load_original_masters(
        experiment.generator_root / "transformer", groups["diffusion"], manifest
    )
    assert audit["policy"] == "exact_original_fp32_safetensors"
    assert any(not torch.equal(value, value.bfloat16().float()) for value in masters.values())
    assert all(
        torch.equal(value.bfloat16(), groups["diffusion"][name]) for name, value in masters.items()
    )
    manifest["files"]["transformer/model.safetensors"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        load_original_masters(
            experiment.generator_root / "transformer", groups["diffusion"], manifest
        )


def stage_args(experiment, checkpoint):
    args = list(experiment.args)
    position = args.index("--connector-checkpoint")
    args[position : position + 2] = ["--init-joint-checkpoint", checkpoint]
    return SimpleNamespace(root=experiment.root, args=args)


def test_subset_repeats_selected_ids_and_keeps_probes_controls_and_samples_inside(experiment):
    _, report = run(
        experiment,
        "subset",
        5,
        "--train-subset-size",
        "2",
        "--probe-count",
        "3",
        "--sample-count",
        "2",
    )
    selected = {row["id"] for row in report["train_selection"]}
    assert len(selected) == 2 and report["train_count"] == 3
    assert report["selected_train_count"] == 2
    full = ImageGenerationWebDataset(experiment.root / "data/train.jsonl")
    expected = selected_training_indices(3, 2, 42)
    assert report["train_selection"] == [{"index": i, "id": full.records[i].id} for i in expected]
    rows = [
        json.loads(line)
        for line in (experiment.root / "subset/steps.jsonl").read_text().splitlines()
    ]
    consumed = [
        identifier for row in rows for batch in row["microbatches"] for identifier in batch["ids"]
    ]
    assert set(consumed) == selected and len(consumed) == 5
    assert report["sample_exposure"]["per_id_counts"] == {
        key: consumed.count(key) for key in selected
    }
    assert report["sample_exposure"]["unique_examples_seen"] == 2
    assert report["sample_exposure"]["selected_equivalent_epochs"] == 2.5
    assert report["sample_exposure"]["full_pool_equivalent_epochs"] == 5 / 3
    for evaluation in report["evaluations"]:
        for row in evaluation["splits"]["train"]["examples"]:
            assert row["id"] in selected and row["shuffled_prompt_id"] in selected
            assert row["id"] != row["shuffled_prompt_id"]
    assert all(row["id"] in selected for row in report["samples"] if row["split"] == "train")
    saved = torch.load(report["checkpoints"][-1]["path"], weights_only=True)
    assert saved["sample_exposure"] == report["sample_exposure"]
    assert report["data_fingerprint"] == experiment.fingerprint


@pytest.mark.parametrize("size", [-1, 1, 4])
def test_invalid_subset_size_rejected(experiment, size):
    with pytest.raises(ValueError, match="subset"):
        run(experiment, "invalid", 1, "--train-subset-size", str(size))


def test_subset_exact_resume_keeps_ids_and_exposure(experiment):
    _, full = run(experiment, "subset-full", 5, "--train-subset-size", "2")
    _, first = run(experiment, "subset-first", 2, "--train-subset-size", "2")
    _, resumed = run(
        experiment,
        "subset-resume",
        5,
        "--train-subset-size",
        "2",
        "--resume",
        first["checkpoints"][-1]["path"],
    )
    expected = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    actual = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    for key in ("optimizer_state_dict", "sampler_state", "sample_exposure", "rng_state"):
        _equal(expected[key], actual[key])
    with pytest.raises(ValueError, match="protocol"):
        run(experiment, "subset-wrong-resume", 5, "--resume", first["checkpoints"][-1]["path"])


def test_joint_stage_uses_completed_weights_but_fresh_optimizer_sampler_and_data_selection(
    experiment,
):
    _, source = run(experiment, "source", 3, "--train-subset-size", "2")
    stage = stage_args(experiment, source["checkpoints"][-1]["path"])
    _, report = run(stage, "stage", 1)
    assert report["master_initialization"]["fresh_optimizer"]
    assert report["master_initialization"]["fresh_sampler"]
    assert report["master_initialization"]["fresh_rng"]
    assert report["master_initialization"]["step"] == 3
    assert report["connector_warm_start"] is None
    assert report["trainable_audit_before"] == source["trainable_audit_after"]
    assert report["selected_train_count"] == 3
    assert report["sample_exposure"]["examples_seen"] == 1
    saved = torch.load(report["checkpoints"][-1]["path"], weights_only=True)
    assert saved["step"] == 1
    assert all(
        float(value["step"]) == 1
        for value in saved["optimizer_state_dict"]["optimizer"]["state"].values()
    )
    assert saved["diffusion_buffer_state"]["training_calls"].item() == 4
    assert report["frozen_state_unchanged"]
    _, full = run(stage, "stage-full", 2)
    _, resumed = run(stage, "stage-resume", 2, "--resume", report["checkpoints"][-1]["path"])
    expected = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    actual = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    for key in ("optimizer_state_dict", "sampler_state", "sample_exposure", "rng_state"):
        _equal(expected[key], actual[key])
    with pytest.raises(ValueError, match="Native-pretrained baseline"):
        run(stage, "stage-native", 1, "--native-baseline")


@pytest.mark.parametrize(
    "change",
    ["parent", "generator", "frozen", "report", "masters", "connector", "buffers", "nonfinite"],
)
def test_joint_stage_rejects_identity_or_incomplete_state(experiment, change):
    _, source = run(experiment, "joint-source", 2)
    checkpoint = Path(source["checkpoints"][-1]["path"])
    saved = torch.load(checkpoint, weights_only=True)
    kwargs = {
        "parent": experiment.parent,
        "reference_sha256": "b" * 64,
        "frozen_hashes": frozen_state_hashes(experiment.prototype, MODULES),
        "fixture": True,
        "named_groups": configure_joint_scope(experiment.prototype),
    }
    if change in {"parent", "generator", "frozen"}:
        key = {"parent": "parent", "generator": "reference_sha256", "frozen": "frozen_hashes"}[
            change
        ]
        kwargs[key] = "different"
    elif change == "report":
        source["status"] = "failed"
        (checkpoint.parent / "report.json").write_text(json.dumps(source))
    else:
        if change == "masters":
            saved["optimizer_state_dict"]["masters"].pop(
                next(iter(saved["optimizer_state_dict"]["masters"]))
            )
        elif change == "connector":
            saved["connector_state_dict"] = {}
        elif change == "buffers":
            saved["diffusion_buffer_state"] = {}
        else:
            saved["optimizer_state_dict"]["masters"][
                next(iter(saved["optimizer_state_dict"]["masters"]))
            ].fill_(float("nan"))
        torch.save(saved, checkpoint)
        source["checkpoints"][-1]["sha256"] = file_sha256(checkpoint)
        (checkpoint.parent / "report.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Joint initialization"):
        restore_joint_stage(experiment.prototype, checkpoint, **kwargs)


def test_joint_stage_readonly_weight_restore_avoids_returning_optimizer_masters(experiment):
    _, source = run(experiment, "source-readonly", 2)
    groups = configure_joint_scope(experiment.prototype)
    masters, audit = restore_joint_stage(
        experiment.prototype,
        source["checkpoints"][-1]["path"],
        named_groups=groups,
        parent=experiment.parent,
        reference_sha256="b" * 64,
        frozen_hashes=frozen_state_hashes(experiment.prototype, MODULES),
        fixture=True,
        restore_masters=False,
    )
    assert masters == {} and not audit["returned_optimizer_masters"]
    assert audit["restored_master_tensors"] > 0
    saved = torch.load(source["checkpoints"][-1]["path"], weights_only=True)
    for values in groups.values():
        for name, value in values.items():
            assert torch.equal(value, saved["optimizer_state_dict"]["masters"][name])


def test_full_data_training_budget_is_large_but_bounded(experiment):
    args = _parser().parse_args(
        experiment.args + ["--output-dir", str(experiment.root / "budget"), "--steps", "50000"]
    )
    validate_budget(args)
    args.steps = 50001
    with pytest.raises(ValueError, match="steps"):
        validate_budget(args)
    assert selected_training_indices(3, 0, 42) == [0, 1, 2]


def test_chat_format_and_prism_negative_are_explicit_sampling_options(experiment, monkeypatch):
    import tools.prism_image_conditioning as conditioning

    formatted, negatives, sample_options = [], [], []
    original_format = conditioning.format_items
    original_sample = TinyBackend.generate_conditioned

    def format_items(items, tokenizer, mode="raw"):
        result = original_format(items, tokenizer, mode)
        formatted.extend(result)
        return result

    def negative(model, tokenizer, prompt, **kwargs):
        negatives.append((prompt, kwargs))
        return {
            "embeds": torch.ones(1, 1, 3),
            "attention_mask": torch.ones(1, 1),
            "empty_anchor": None,
        }

    def sample(backend, embeds, mask, native_context=None, **kwargs):
        sample_options.append(kwargs)
        return original_sample(backend, embeds, mask, native_context=native_context, **kwargs)

    monkeypatch.setattr(conditioning, "format_items", format_items)
    monkeypatch.setattr(conditioning, "encode_prism_prompt", negative)
    monkeypatch.setattr(TinyBackend, "generate_conditioned", sample)
    _, report = run(
        experiment,
        "chat-negative",
        1,
        "--prompt-format",
        "chat",
        "--negative-conditioning",
        "prism",
        "--sample-count",
        "1",
    )
    assert formatted and all(item["prompt"].startswith("CHAT:") for item in formatted)
    assert len(negatives) == 4 and all(
        row[0] == "" and row[1]["mode"] == "chat" for row in negatives
    )
    assert all(
        "negative_prompt_embeds" in row and "negative_prompt_attention_mask" in row
        for row in sample_options
    )
    assert all(
        row["prompt_format"] == "chat" and row["negative_conditioning"] == "prism"
        for row in report["samples"]
    )
    saved = torch.load(report["checkpoints"][-1]["path"], weights_only=True)
    assert saved["protocol"]["settings"]["prompt_format"] == "chat"
    assert saved["protocol"]["settings"]["negative_conditioning"] == "prism"


# Reuse the actual alignment-writer fixture, then add a tiny flow objective to
# its frozen models. This exercises the real strict alignment adapter rather
# than replacing it with a successful stub.
import test_align_prism_image_conditioning as alignment_fixtures

alignment_experiment = alignment_fixtures.experiment


@pytest.fixture
def aligned_training_experiment(alignment_experiment, monkeypatch):
    import src.data.image_generation_webdataset as data_module
    import src.training.cpu_master_adamw as master_module
    import tools.train_prism_image_diffusion as runner
    from tools import align_prism_image_conditioning as alignment
    from tools.train_prism_image_connector import target_free_item

    original_dataset = data_module.ImageGenerationWebDataset
    prototype, _ = alignment_fixtures.model_and_tokenizer()
    directory = alignment_experiment.root / "original-transformer"
    directory.mkdir()
    save_file(
        prototype.decoders["image"].backend.transformer.state_dict(),
        directory / "model.safetensors",
    )
    original_diffusion = {
        name: value.clone()
        for name, value in prototype.decoders["image"].backend.transformer.state_dict().items()
    }
    manifest = {
        "manifest_sha256": "b" * 64,
        "files": {"transformer/model.safetensors": file_sha256(directory / "model.safetensors")},
    }
    monkeypatch.setattr(alignment_fixtures.Backend, "checkpoint_manifest", lambda self: manifest)
    alignment._run(alignment_experiment.args)
    alignment_report_path = alignment_experiment.root / "output/report.json"
    alignment_report = json.loads(alignment_report_path.read_text())
    alignment_checkpoint = Path(alignment_report["checkpoints"][-1]["path"])
    aligned_state = torch.load(alignment_checkpoint, weights_only=True)["connector_state_dict"]
    native_calls, master_initializations = [], []

    class FlowDataset(original_dataset):
        def __init__(self, index, *, target_size, split):
            super().__init__(index, split=split)
            self.target_size = target_size
            for row in self.records:
                row.group_ids = (row.id,)

        def __getitem__(self, index):
            item = target_free_item(self.records[index])
            item["target_image"] = torch.full((3, *self.target_size), 0.1 + index * 0.2)
            return item

    def training_step(backend, embeds, attention_mask, targets, native_context=None, **kwargs):
        values = backend.transformer.time_caption_embed.caption_embedder(embeds.mean(1))
        loss = (values - targets.mean((1, 2, 3))[:, None]).square().mean()
        return values, loss

    def forward_outputs(model, inputs, *, targets, requested_outputs, native_context, output_specs):
        condition = model._output_condition(inputs)[0]
        _, loss = model.decoders["image"].forward_condition(condition, targets=targets["image"])
        return SimpleNamespace(losses={"image": loss})

    def sample(backend, options):
        assert not backend.transformer.training and not torch.is_grad_enabled()
        latent = options.get("latents")
        if latent is None:
            latent = torch.randn(1, 4, 2, 2, generator=options["generator"])
        backend.last_trace = {"latents.initial": latent.detach().clone()}
        return [Image.new("RGB", (32, 32), "blue")]

    def generate_conditioned(backend, embeds, mask, native_context=None, **options):
        return sample(backend, options)

    def generate_reference(backend, native_context, **options):
        assert all(
            torch.equal(value, backend.transformer.state_dict()[name])
            for name, value in original_diffusion.items()
        )
        native_calls.append("original_diffusion")
        return sample(backend, options)

    def predict(model, *, inputs, requested_outputs, native_context, decoder_kwargs):
        condition = model._output_condition(inputs)[0]
        images = model.decoders["image"].generate_condition(condition, **decoder_kwargs["image"])
        return SimpleNamespace(predictions={"image": images})

    original_optimizer_init = master_module.CPUMasterAdamW.__init__

    def optimizer_init(optimizer, **kwargs):
        master_initializations.append(
            {name: value.detach().clone() for name, value in kwargs["master_values"].items()}
        )
        original_optimizer_init(optimizer, **kwargs)
        assert not optimizer.optimizer.state  # New stage does not inherit alignment Adam moments.

    monkeypatch.setattr(data_module, "ImageGenerationWebDataset", FlowDataset)
    monkeypatch.setattr(
        alignment_fixtures.Transformer,
        "enable_gradient_checkpointing",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        alignment_fixtures.Transformer,
        "disable_gradient_checkpointing",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(alignment_fixtures.Backend, "training_step", training_step)
    monkeypatch.setattr(alignment_fixtures.Backend, "generate_conditioned", generate_conditioned)
    monkeypatch.setattr(alignment_fixtures.Backend, "generate_reference", generate_reference)
    monkeypatch.setattr(
        alignment_fixtures.Parent, "forward_outputs", forward_outputs, raising=False
    )
    monkeypatch.setattr(alignment_fixtures.Parent, "predict", predict, raising=False)
    monkeypatch.setattr(master_module.CPUMasterAdamW, "__init__", optimizer_init)
    monkeypatch.setattr(
        runner, "validate_repeatability_report", lambda *args, **kwargs: {"fixture": True}
    )
    args = []
    for name in (
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "train_index",
        "validation_index",
        "repeatability_report",
        "connector_checkpoint",
    ):
        args += ["--" + name.replace("_", "-"), str(getattr(alignment_experiment.args, name))]
    args += [
        "--init-alignment-checkpoint",
        str(alignment_checkpoint),
        "--prompt-format",
        "chat",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--height",
        "32",
        "--width",
        "32",
        "--expected-parent-tensors",
        "2",
        "--deterministic",
        "--probe-count",
        "1",
        "--final-validation-count",
        "1",
        "--sample-count",
        "0",
        "--sampling-steps",
        "2",
        "--learning-rate",
        "0.01",
        "--diffusion-learning-rate",
        "0.01",
        "--transformer-checkpoint-dir",
        str(directory),
    ]
    return SimpleNamespace(
        root=alignment_experiment.root,
        args=args,
        checkpoint=alignment_checkpoint,
        report_path=alignment_report_path,
        aligned_state=aligned_state,
        original_diffusion=original_diffusion,
        native_calls=native_calls,
        master_initializations=master_initializations,
    )


def test_alignment_initialization_preserves_full_fp32_masters_and_original_native_baseline(
    aligned_training_experiment,
):
    experiment = aligned_training_experiment
    _, report = run(experiment, "aligned-stage", 1, "--sample-count", "1", "--native-baseline")
    assert experiment.native_calls == ["original_diffusion"]
    initial = experiment.master_initializations[0]
    for name, value in experiment.aligned_state.items():
        assert torch.equal(initial[name], value)
        assert initial[name].dtype == torch.float32
    assert any(
        not torch.equal(value, value.bfloat16().float())
        for value in experiment.aligned_state.values()
    )
    for name, value in experiment.original_diffusion.items():
        assert torch.equal(initial[DIFFUSION_PREFIX + name], value)
    lineage = report["alignment_initialization"]
    assert lineage["evidence_kind"] == "fixture_only_connector_native_feature_alignment"
    assert lineage["fresh_optimizer"] and lineage["fresh_sampler"] and lineage["fresh_rng"]
    assert lineage["stage_initialization"] == "aligned_connector_original_diffusion"
    assert report["connector_warm_start"]["step"] == 500
    saved = torch.load(report["checkpoints"][-1]["path"], weights_only=True)
    assert saved["protocol"]["alignment_initialization"] == lineage
    assert saved["protocol"]["settings"]["prompt_format"] == "chat"
    assert saved["step"] == 1 and saved["sampler_state"]["examples_seen"] == 1
    assert all(
        float(state["step"]) == 1
        for state in saved["optimizer_state_dict"]["optimizer"]["state"].values()
    )
    assert report["frozen_state_unchanged"] and all(report["trainable_groups_changed"].values())


def test_alignment_initialized_stage_exact_resume_binds_original_source(
    aligned_training_experiment,
):
    experiment = aligned_training_experiment
    _, full = run(experiment, "aligned-full", 3)
    _, first = run(experiment, "aligned-first", 1)
    _, resumed = run(experiment, "aligned-resume", 3, "--resume", first["checkpoints"][-1]["path"])
    expected = torch.load(full["checkpoints"][-1]["path"], weights_only=True)
    actual = torch.load(resumed["checkpoints"][-1]["path"], weights_only=True)
    for key in ("optimizer_state_dict", "sampler_state", "sample_exposure", "rng_state"):
        _equal(expected[key], actual[key])


@pytest.mark.parametrize("change", ["data", "source", "digest"])
def test_alignment_initialization_rejects_changed_data_or_core_source(
    aligned_training_experiment, change
):
    experiment = aligned_training_experiment
    saved = torch.load(experiment.checkpoint, weights_only=True)
    report = json.loads(experiment.report_path.read_text())
    extra = []
    if change == "data":
        saved["protocol"]["data_fingerprint"] = report["data_fingerprint"] = "wrong-data"
    elif change == "source":
        saved["protocol"]["source_sha256"]["src/model.py"] = "f" * 64
        report["source_sha256"] = saved["protocol"]["source_sha256"]
    else:
        extra = ["--init-alignment-checkpoint-sha256", "f" * 64]
    torch.save(saved, experiment.checkpoint)
    report["checkpoints"][-1]["sha256"] = file_sha256(experiment.checkpoint)
    experiment.report_path.write_text(json.dumps(report))
    with pytest.raises(
        ValueError, match="data conversion identity|conditioning source|requested checkpoint digest"
    ):
        run(experiment, "aligned-invalid", 1, *extra)
    assert experiment.master_initializations == []
    failed = json.loads((experiment.root / "aligned-invalid/report.json").read_text())
    assert (
        failed["status"] == "failed"
        and failed["completed_steps"] == 0
        and failed["frozen_state_unchanged"]
    )


def test_alignment_initialization_requires_chat_and_excludes_joint_source(experiment):
    args = _parser().parse_args(
        experiment.args
        + ["--output-dir", str(experiment.root / "guard"), "--init-alignment-checkpoint", "/unused"]
    )
    with pytest.raises(ValueError, match="prompt-format chat"):
        validate_budget(args)
    args.prompt_format = "chat"
    args.init_joint_checkpoint = Path("/joint")
    with pytest.raises(ValueError, match="incompatible"):
        validate_budget(args)
    args.init_joint_checkpoint = None
    args.expected_connector_step = 499
    with pytest.raises(ValueError, match="500-step"):
        validate_budget(args)
