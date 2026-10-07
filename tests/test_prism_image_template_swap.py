"""Offline evidence contracts for teacher-assisted swaps, not image-quality tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.loading import file_sha256
from tools import diagnose_prism_image_conditioning as diagnostic
from tools import prism_image_template_swap as swaps
from tools.align_prism_image_conditioning import caption_norm, encode_observed_native
from tools.prism_image_conditioning import encode_prism_prompt
from tools.train_image_decoder import _tensor_hash, frozen_state_hashes

import test_align_prism_image_conditioning as fixtures

experiment = fixtures.experiment


def encoded():
    model, tokenizer = fixtures.model_and_tokenizer()
    backend = model.decoders["image"].backend
    prism = encode_prism_prompt(model, tokenizer, "red blue", device="cpu", mode="chat")
    native = encode_observed_native(backend, "red blue", max_text_length=32)
    return model, tokenizer, prism, native


def test_swaps_preserve_content_and_copy_only_exact_unnormalized_partitions():
    model, _, prism, native = encoded()
    originals = prism["embeds"].clone(), native["embeds"].clone()
    routes, audit = swaps.compose_conditions(
        prism, native, torch.tensor([[4, 5]]), caption_norm(model.decoders["image"].backend)
    )
    assert list(routes) == list(swaps.ROUTE_SOURCES)
    assert audit["partition_spans"] == {"prefix": [0, 2], "content": [2, 4], "suffix": [4, 5]}
    assert audit["partition_token_counts"] == {"prefix": 2, "content": 2, "suffix": 1}
    assert audit["swap_stage"] == "unnormalized_before_dit_rmsnorm"
    for route, value in routes.items():
        for region, source in swaps.ROUTE_SOURCES[route].items():
            start, end = audit["partition_spans"][region]
            reference = native if source == "native" else prism
            assert torch.equal(value["embeds"][:, start:end], reference["embeds"][:, start:end])
            reference_route = "native_pretrained" if source == "native" else "prism_aligned"
            assert (
                audit["routes"][route]["partitions"][region]["unnormalized_sha256"]
                == (audit["routes"][reference_route]["partitions"][region]["unnormalized_sha256"])
            )
        assert torch.equal(value["attention_mask"], native["attention_mask"])
    assert torch.equal(prism["embeds"], originals[0])
    assert torch.equal(native["embeds"], originals[1])
    routes["prism_native_template"]["embeds"].zero_()
    assert torch.equal(routes["prism_aligned"]["embeds"], originals[0])
    assert torch.equal(native["embeds"], originals[1])
    # Actual non-unit RMSNorm weights are used for statistics, not a unit-vector surrogate.
    norm = caption_norm(model.decoders["image"].backend)
    mse = (norm(originals[0])[:, 2:4] - norm(originals[1])[:, 2:4]).square().mean()
    assert audit["routes"]["prism_aligned"]["partitions"]["content"][
        "normalized_mse_to_native"
    ] == pytest.approx(float(mse))


@pytest.mark.parametrize(
    "change", ["shape", "nan", "ids", "mask", "hole", "no_suffix", "ambiguous"]
)
def test_invalid_shapes_masks_and_caption_spans_fail_closed(change):
    model, _, prism, native = encoded()
    raw = torch.tensor([[4, 5]])
    if change == "shape":
        prism["embeds"] = prism["embeds"][..., :3]
    elif change == "nan":
        prism["embeds"][0, 0, 0] = float("nan")
    elif change == "ids":
        native["input_ids"][0, 0] += 1
    elif change == "mask":
        native["attention_mask"][0, 0] = 0
    elif change == "hole":
        for values in (prism, native):
            values["input_attention_mask"] = torch.tensor([[1, 0, 1, 1, 1]])
            values["attention_mask"] = values["input_attention_mask"].clone()
    elif change == "no_suffix":
        raw = torch.tensor([[4, 5, 3]])
    else:
        for values in (prism, native):
            values["input_ids"] = torch.tensor([[1, 4, 5, 4, 5]])
    with pytest.raises(ValueError):
        swaps.compose_conditions(prism, native, raw, caption_norm(model.decoders["image"].backend))


def test_actual_teacher_inputs_verified_and_wrong_caption_has_own_span():
    model, tokenizer = fixtures.model_and_tokenizer()
    backend = model.decoders["image"].backend
    before = frozen_state_hashes(model, ())
    audits = []
    for caption in ("red blue", "green yellow round"):
        _, audit = swaps.encode_template_swap_conditions(
            model, tokenizer, backend, caption, device="cpu", max_text_length=32
        )
        audits.append(audit)
        assert audit["actual_native_forward_inputs_verified"]
        assert not audit["target_pixels_read"]
    assert audits[0]["content_span"] == [2, 4] and audits[1]["content_span"] == [2, 5]
    assert frozen_state_hashes(model, ()) == before
    backend.actual_token_mismatch = True
    with pytest.raises(RuntimeError, match="Actual native teacher"):
        swaps.encode_template_swap_conditions(
            model, tokenizer, backend, "red blue", device="cpu", max_text_length=32
        )
    assert not backend.mllm._forward_pre_hooks


def install_inference_fixtures(monkeypatch):
    def forward(self, *, hidden_states, timestep, text_hidden_states, text_attention_mask):
        features = self.time_caption_embed.caption_embedder[0](text_hidden_states)
        return (
            hidden_states
            + timestep[:, None, None, None] * features[text_attention_mask.bool()].mean()
        )

    def training_step(self, embeds, mask, target, *, generator, timesteps):
        noise = torch.randn(target.shape, generator=generator)
        if getattr(self, "break_noise", False):
            noise = noise + float(embeds.mean())
        actual = embeds + 1 if getattr(self, "break_condition", False) else embeds
        prediction = self.transformer(
            hidden_states=noise,
            timestep=timesteps,
            text_hidden_states=actual,
            text_attention_mask=mask,
        )
        return prediction, (prediction - target).square().mean()

    def generate_conditioned(self, embeds, mask, **options):
        if options["text_guidance_scale"] == 1:
            assert not any(key.startswith("negative") for key in options)
        latent = options.get("latents")
        if latent is None:
            latent = torch.randn(1, 4, 2, 2, generator=options["generator"])
        if getattr(self, "break_noise", False):
            latent = latent + 1
        self.last_trace = {
            "latents.initial": latent.clone(),
            "condition.positive": embeds.detach().cpu().clone(),
            "mask.positive": mask.cpu().clone(),
            "condition.branch0": embeds.detach().cpu().clone(),
            "mask.branch0": mask.cpu().clone(),
        }
        if getattr(self, "break_condition", False):
            self.last_trace["condition.branch0"] += 1
        if getattr(self, "break_cfg", False):
            self.last_trace["condition.branch1"] = embeds.clone()
        self.calls = getattr(self, "calls", []) + [options]
        return [Image.new("RGB", (32, 32), "blue")]

    def generate_reference(self, native_context, **options):
        embeds, mask, _, _ = self.encode_prompt(prompt=[native_context["prompt"]])
        return generate_conditioned(self, embeds, mask, **options)

    monkeypatch.setattr(fixtures.Transformer, "forward", forward)
    monkeypatch.setattr(fixtures.Backend, "training_step", training_step)
    monkeypatch.setattr(fixtures.Backend, "generate_conditioned", generate_conditioned)
    monkeypatch.setattr(fixtures.Backend, "generate_reference", generate_reference)


def pairs(model, tokenizer):
    backend = model.decoders["image"].backend
    result = {}
    for kind, caption in (("matched", "red blue"), ("wrong", "green yellow round")):
        values, _ = swaps.encode_template_swap_conditions(
            model, tokenizer, backend, caption, device="cpu", max_text_length=32
        )
        for route, value in values.items():
            result.setdefault(route, {})[kind] = value
    return result


def test_flow_verifies_actual_noise_time_and_condition_per_route_and_caption(monkeypatch):
    install_inference_fixtures(monkeypatch)
    model, tokenizer = fixtures.model_and_tokenizer()
    backend = model.decoders["image"].backend
    conditions = pairs(model, tokenizer)
    result = diagnostic.paired_flow_losses(
        backend,
        torch.zeros(1, 3, 2, 2),
        conditions,
        seed=12,
        timestep=0.5,
        device="cpu",
        verify_conditioning=True,
    )
    assert result["actual_conditioning_verified"] and list(result["routes"]) == list(
        swaps.ROUTE_SOURCES
    )
    for route, value in result["routes"].items():
        for kind in ("matched", "wrong"):
            assert value[kind + "_condition"]["embeds_sha256"] == _tensor_hash(
                conditions[route][kind]["embeds"]
            )
    assert not backend.transformer._forward_pre_hooks
    for flag, message in (
        ("break_condition", "different template-swap"),
        ("break_noise", "changed actual noisy"),
    ):
        setattr(backend, flag, True)
        with pytest.raises(RuntimeError, match=message):
            diagnostic.paired_flow_losses(
                backend,
                torch.zeros(1, 3, 2, 2),
                conditions,
                seed=12,
                timestep=0.5,
                device="cpu",
                verify_conditioning=True,
            )
        setattr(backend, flag, False)
        assert not backend.transformer._forward_pre_hooks


def test_sampling_five_cfg1_routes_replays_noise_and_checks_actual_states(monkeypatch, tmp_path):
    install_inference_fixtures(monkeypatch)
    model, tokenizer = fixtures.model_and_tokenizer()
    backend = model.decoders["image"].backend
    positive = {route: pair["matched"] for route, pair in pairs(model, tokenizer).items()}
    args = SimpleNamespace(
        height=32, width=32, sampling_steps=2, max_text_length=32, device="cpu", dtype="float32"
    )
    kwargs = dict(
        seed=12,
        args=args,
        output_dir=tmp_path,
        case_id="validation-00",
        saved_latent=torch.randn(1, 4, 2, 2),
    )
    rows = swaps.sample_template_swaps(backend, "red blue", positive, **kwargs)
    assert [row["route"] for row in rows] == [route + "_cfg1" for route in swaps.ROUTE_SOURCES]
    assert len({row["initial_latent_sha256"] for row in rows}) == 1
    assert all(
        row["actual_condition_verified"]
        and row["template_swap"]
        and row["text_guidance_scale"] == 1
        for row in rows
    )
    for flag, message in (
        ("break_condition", "changed the intended"),
        ("break_noise", "changed initial noise"),
        ("break_cfg", "another conditioning branch"),
    ):
        setattr(backend, flag, True)
        with pytest.raises(RuntimeError, match=message):
            swaps.sample_template_swaps(backend, "red blue", positive, **kwargs)
        setattr(backend, flag, False)


def cli_args():
    argv = []
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "train-index",
        "validation-index",
        "repeatability-report",
        "connector-checkpoint",
        "output-dir",
    ):
        argv += ["--" + name, "/unused"]
    return argv + ["--deterministic"]


def test_invalid_cli_combinations_fail_before_execution():
    base = cli_args() + ["--template-swap-controls"]
    for extra in (
        [],
        ["--prism-formats", "chat"],
        ["--alignment-checkpoint", "/aligned"],
        ["--alignment-checkpoint", "/aligned", "--prism-formats", "raw"],
    ):
        with pytest.raises(ValueError, match="Template-swap controls require"):
            diagnostic.validate_budget(diagnostic._parser().parse_args(base + extra))
    diagnostic.validate_budget(
        diagnostic._parser().parse_args(
            base + ["--alignment-checkpoint", "/aligned", "--prism-formats", "chat"]
        )
    )
    assert diagnostic._parser().parse_args(cli_args()).template_swap_controls is False


def test_completed_fixture_uses_real_strict_alignment_restore_and_emits_no_cache(
    experiment, monkeypatch
):
    import src.data.image_generation_webdataset as data_module
    from tools import align_prism_image_conditioning as alignment

    # Produce a real fixture artifact through the existing training/provenance path.
    alignment._run(experiment.args)
    training_report = json.loads((experiment.root / "output/report.json").read_text())
    checkpoint = Path(training_report["checkpoints"][-1]["path"])
    metadata_dataset = data_module.ImageGenerationWebDataset
    reads = []

    class TargetDataset(metadata_dataset):
        def __init__(self, index, *, split, target_size):
            super().__init__(index, split=split)
            self.split = split

        def __getitem__(self, index):
            reads.append((self.split, index))
            return {"target_image": torch.zeros(3, 32, 32)}

    monkeypatch.setattr(data_module, "ImageGenerationWebDataset", TargetDataset)
    monkeypatch.setattr(
        diagnostic, "validate_repeatability_report", lambda *a, **k: {"fixture": True}
    )
    install_inference_fixtures(monkeypatch)
    args = diagnostic._parser().parse_args(cli_args())
    for key in (
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "train_index",
        "validation_index",
        "repeatability_report",
        "connector_checkpoint",
    ):
        setattr(args, key, getattr(experiment.args, key))
    args.output_dir = experiment.root / "diagnostic"
    args.alignment_checkpoint = checkpoint
    args.alignment_checkpoint_sha256 = file_sha256(checkpoint)
    args.template_swap_controls, args.prism_formats = True, ["chat"]
    args.train_probe_count = args.validation_probe_count = 2
    args.sample_count, args.sampling_steps = 1, 2
    args.flow_timesteps = [0.1, 0.9]
    args.expected_parent_tensors = 2
    args.device, args.dtype = "cpu", "float32"
    args.height = args.width = 32
    report = diagnostic._run(args)
    assert report["status"] == "completed" and report["evidence_kind"] == "fixture_only"
    assert report["frozen_state_unchanged"] and report["native_pretrained_state_unchanged"]
    assert report["alignment_checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert len(report["template_swap_statistics"]) == 8
    assert len(report["flow_controls"]) == 8
    assert len(report["samples"]) == 6
    assert (
        report["samples"][0]["route"] == "native_pretrained"
        and report["samples"][0]["text_guidance_scale"] == 5
    )
    assert all(row["actual_conditioning_verified"] for row in report["flow_controls"])
    assert (
        len(reads) == 4
    )  # Only target reads for four flow cases, none during encoding/generation.
    assert report["negative_condition_statistics"] == {}
    assert report["source_sha256"]["tools/prism_image_template_swap.py"] == file_sha256(
        Path(swaps.__file__)
    )
    assert not report["template_swap_protocol"]["teacher_cache_persisted"]
    assert {path.suffix for path in args.output_dir.iterdir()} == {".json", ".png"}
    # A digest mismatch still fails through the original strict restore before hybrid inference.
    args.output_dir = experiment.root / "rejected"
    args.alignment_checkpoint_sha256 = "f" * 64
    with pytest.raises(ValueError, match="SHA256|digest|sha256"):
        diagnostic._run(args)
    rejected = json.loads((args.output_dir / "report.json").read_text())
    assert rejected["status"] == "failed" and rejected["template_swap_statistics"] == []
