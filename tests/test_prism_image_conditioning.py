"""Offline controls for diagnostic validity, not real-model quality evidence."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.image import ImageDecoder
from src.decoders.types import DecoderCondition
from tools.diagnose_prism_image_conditioning import (
    condition_metadata,
    encode_native_prompt,
    main,
    paired_flow_losses,
    sample_variants,
    select_wrong_caption,
)
from tools.prism_image_conditioning import (
    IMAGE_SYSTEM_PROMPT,
    encode_prism_prompt,
    feature_statistics,
    format_items,
    format_prompt,
    preserved_model_state,
    summarize_caption_controls,
    tokenize_prompt,
)
from torch import nn


class Tokenizer:
    eos_token_id = 3

    def __call__(self, prompts, **kwargs):
        assert kwargs.get("truncation") is False
        assert kwargs["return_tensors"] == "pt"
        rows = [[ord(c) % 10 for c in prompt] for prompt in prompts]
        ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0] == {"role": "system", "content": IMAGE_SYSTEM_PROMPT}
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": False,
        }
        return "SYSTEM:" + messages[0]["content"] + " USER:" + messages[1]["content"]


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Embedding(10, 4)
        self.decoders = nn.ModuleDict(
            {"image": ImageDecoder(4, backend=nn.Linear(4, 4), conditioning_dim=4)}
        )
        self.seen_inputs = None

    def _output_condition(self, inputs):
        self.seen_inputs = inputs
        condition = DecoderCondition(
            hidden_states=self.backbone(inputs["text"]),
            attention_mask=inputs["text_attention_mask"],
        )
        return condition, None, None, None


def test_format_preserves_raw_targets_and_uses_native_role_structure():
    tokenizer = Tokenizer()
    target = torch.zeros(3, 2, 2)
    rows = [{"prompt": "a dog", "target_image": target}]
    raw = format_items(rows, tokenizer)
    chat = format_items(rows, tokenizer, "chat")
    assert raw == rows and raw[0] is not rows[0]
    assert chat[0]["prompt"].endswith("USER:a dog")
    assert chat[0]["target_image"] is target
    assert rows[0]["prompt"] == "a dog"
    with pytest.raises(ValueError, match="raw or chat"):
        format_prompt("x", tokenizer, "other")


def test_empty_raw_anchor_is_explicit_and_chat_does_not_use_eos():
    raw = tokenize_prompt(Tokenizer(), "", device="cpu")
    assert raw["input_ids"].tolist() == [[3]]
    assert raw["input_attention_mask"].tolist() == [[1]]
    assert raw["empty_anchor"] == "eos"
    chat = tokenize_prompt(Tokenizer(), "", device="cpu", mode="chat")
    assert chat["empty_anchor"] is None and chat["input_ids"].shape[-1] > 1


def test_encoding_is_target_free_and_uses_real_connector():
    parent = Parent()
    result = encode_prism_prompt(parent, Tokenizer(), "ab", device="cpu")
    assert set(parent.seen_inputs) == {"text", "text_attention_mask"}
    assert result["embeds"].shape == (1, 2, 4)
    assert torch.equal(result["attention_mask"], torch.ones(1, 2, dtype=torch.bool))
    assert condition_metadata(result)["prism_hidden_states"]["valid_token_counts"] == [2]


def test_tokenization_never_silently_truncates():
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_prompt(Tokenizer(), "abcdef", device="cpu", max_text_length=2)


def test_statistics_ignore_padding_and_report_nonfinite_valid_values():
    value = torch.tensor([[[3.0, 4.0], [float("nan"), 0.0]]])
    stats = feature_statistics(value, torch.tensor([[1, 0]]))
    assert stats["finite"] and stats["token_norm_mean"] == 5
    assert stats["rms"] == pytest.approx((12.5) ** 0.5)
    assert stats["token_norm_quantiles"]["0.95"] == 5
    bad = feature_statistics(value, torch.tensor([[1, 1]]))
    assert not bad["finite"] and bad["nonfinite_values"] == 1
    assert "rms" not in bad
    with pytest.raises(ValueError, match="at least one"):
        feature_statistics(value, torch.zeros(1, 2))


def test_frozen_eval_context_restores_rng_flags_and_modes_after_error():
    parent = Parent().train()
    parent.backbone.eval()
    modes = [module.training for module in parent.modules()]
    flags = [parameter.requires_grad for parameter in parent.parameters()]
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="failure"):
        with preserved_model_state(parent):
            assert all(not p.requires_grad for p in parent.parameters())
            assert all(not m.training for m in parent.modules())
            torch.randn(5)
            raise RuntimeError("failure")
    assert [module.training for module in parent.modules()] == modes
    assert [parameter.requires_grad for parameter in parent.parameters()] == flags
    assert torch.equal(torch.get_rng_state(), rng)


class Transformer(nn.Module):
    def forward(self, *, hidden_states, timestep, text_hidden_states):
        return hidden_states + text_hidden_states.mean() * timestep[:, None, None, None]


class Backend:
    def __init__(self):
        self.transformer = Transformer()
        self.last_trace = {}
        self.calls = []
        self.changed_noise = False

    def training_step(self, embeds, mask, target, *, generator, timesteps):
        noisy = torch.randn(target.shape, generator=generator)
        if self.changed_noise:
            noisy += len(self.calls)
        prediction = self.transformer(
            hidden_states=noisy, timestep=timesteps, text_hidden_states=embeds
        )
        self.calls.append("flow")
        return prediction, (prediction - target).square().mean()

    def _sample(self, options):
        latent = options.get("latents")
        if latent is None:
            latent = torch.randn(1, 4, 2, 2, generator=options["generator"])
        if self.changed_noise and self.calls:
            latent = latent + 1
        self.last_trace = {"latents.initial": latent.clone()}
        self.calls.append(options)
        return [Image.new("RGB", (32, 32), "blue")]

    def generate_reference(self, native_context, **options):
        assert set(native_context) == {"prompt"}
        return self._sample(dict(options, reference=True))

    def generate_conditioned(self, embeds, mask, **options):
        return self._sample(options)


def conditions():
    def value(v):
        return {"embeds": torch.full((1, 2, 4), v), "attention_mask": torch.ones(1, 2)}

    return {
        "native_pretrained": {"matched": value(0.1), "wrong": value(0.8)},
        "prism_raw": {"matched": value(0.2), "wrong": value(0.9)},
    }


def test_flow_controls_fix_actual_noise_and_time_and_remove_hook():
    backend = Backend()
    rng = torch.get_rng_state().clone()
    result = paired_flow_losses(
        backend, torch.zeros(1, 3, 2, 2), conditions(), seed=12, timestep=0.4, device="cpu"
    )
    assert len(backend.calls) == 4
    assert result["actual_inputs"]["timestep"] == pytest.approx([0.4])
    assert result["routes"]["prism_raw"]["prediction_change_mse"] > 0
    assert not backend.transformer._forward_pre_hooks
    assert torch.equal(torch.get_rng_state(), rng)


def test_flow_controls_reject_changed_noisy_input_and_remove_hook():
    backend = Backend()
    backend.changed_noise = True
    with pytest.raises(RuntimeError, match="changed actual noisy"):
        paired_flow_losses(
            backend, torch.zeros(1, 3, 2, 2), conditions(), seed=12, timestep=0.4, device="cpu"
        )
    assert not backend.transformer._forward_pre_hooks


def test_summary_preserves_gap_direction_and_split():
    rows = [
        {"split": "train", "routes": {"native": {"matched": 1.0, "wrong": 2.0}}},
        {"split": "train", "routes": {"native": {"matched": 3.0, "wrong": 2.0}}},
        {"split": "validation", "routes": {"native": {"matched": 2.0, "wrong": 3.0}}},
    ]
    result = summarize_caption_controls(rows)
    assert result["train"]["native"]["wrong_minus_matched"] == 0
    assert result["train"]["native"]["matched_wins"] == 1
    assert result["validation"]["native"]["wrong_minus_matched"] == 1


def sample_args():
    return SimpleNamespace(
        height=32, width=32, sampling_steps=2, max_text_length=1024, device="cpu", dtype="float32"
    )


def test_cfg_gallery_replays_latents_and_uses_explicit_negative_route(tmp_path):
    backend = Backend()
    pair = conditions()["prism_raw"]
    rows, latent = sample_variants(
        backend,
        "caption",
        {"raw": pair["matched"]},
        {"raw": pair["wrong"]},
        seed=12,
        args=sample_args(),
        output_dir=tmp_path,
        case_id="validation-00",
    )
    assert len(rows) == 4
    assert len({row["initial_latent_sha256"] for row in rows}) == 1
    assert backend.calls[0]["reference"]
    assert "negative_prompt_embeds" not in backend.calls[1]
    assert backend.calls[2]["text_guidance_scale"] == 1
    assert backend.calls[3]["negative_prompt_embeds"] is pair["wrong"]["embeds"]
    assert latent.shape == (1, 4, 2, 2)
    assert all(row["target_free"] for row in rows)


def test_cfg_gallery_rejects_broken_latent_replay(tmp_path):
    backend = Backend()
    backend.changed_noise = True
    pair = conditions()["prism_raw"]
    with pytest.raises(RuntimeError, match="changed initial noise"):
        sample_variants(
            backend,
            "caption",
            {"raw": pair["matched"]},
            {"raw": pair["wrong"]},
            seed=12,
            args=sample_args(),
            output_dir=tmp_path,
            case_id="validation-00",
        )


def test_optional_native_cfg1_uses_native_path_and_identical_noise(tmp_path):
    backend = Backend()
    args = sample_args()
    args.native_cfg1_control = True
    pair = conditions()["prism_raw"]
    rows, latent = sample_variants(
        backend,
        "caption",
        {"raw": pair["matched"]},
        {"raw": pair["wrong"]},
        seed=12,
        args=args,
        output_dir=tmp_path,
        case_id="validation-00",
    )
    assert [row["route"] for row in rows[:2]] == ["native_pretrained", "native_pretrained_cfg1"]
    assert backend.calls[1]["reference"]
    assert backend.calls[1]["text_guidance_scale"] == 1
    assert backend.calls[1]["image_guidance_scale"] == 1
    assert len(rows) == 5 and len({row["initial_latent_sha256"] for row in rows}) == 1
    # Subsequent restored-checkpoint sampling must not duplicate native baselines.
    rows, _ = sample_variants(
        backend,
        "caption",
        {"raw": pair["matched"]},
        {"raw": pair["wrong"]},
        seed=12,
        args=args,
        output_dir=tmp_path,
        case_id="validation-01",
        saved_latent=latent,
        skip_native=True,
    )
    assert len(rows) == 3 and all(row["route"].startswith("prism_") for row in rows)


def test_native_encoding_checks_actual_token_mask_and_no_truncation():
    pipe = SimpleNamespace(
        _apply_chat_template=lambda text: "N:" + text,
        processor=SimpleNamespace(tokenizer=Tokenizer()),
        transformer=SimpleNamespace(device="cpu"),
    )
    pipe.encode_prompt = lambda **kwargs: (
        torch.zeros(1, 3, 4),
        torch.ones(1, 3, dtype=torch.long),
        None,
        None,
    )
    backend = SimpleNamespace(_pipeline=pipe)
    result = encode_native_prompt(backend, "x")
    assert result["input_ids"].shape == (1, 3)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_native_prompt(backend, "abcdef", max_text_length=2)
    pipe.encode_prompt = lambda **kwargs: (
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
        None,
        None,
    )
    with pytest.raises(RuntimeError, match="changed token mask"):
        encode_native_prompt(backend, "x")


def test_real_cli_refuses_login_or_local_execution(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
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
    with pytest.raises(RuntimeError, match="PBS compute allocation"):
        main(argv + ["--deterministic", "--attention-backend", "math"])
    with pytest.raises(ValueError, match="already include native CFG1"):
        main(argv + ["--deterministic", "--native-cfg1-control", "--template-swap-controls"])


def test_wrong_caption_stays_in_the_selected_training_pool():
    rows = [SimpleNamespace(prompt=value) for value in ("a", "b", "c", "a", "d")]
    assert select_wrong_caption(rows, 0, [0, 3, 4]) == 4
    assert select_wrong_caption(rows, 4, [0, 3, 4]) == 0
    with pytest.raises(ValueError, match="no distinct"):
        select_wrong_caption(rows, 0, [0, 3])
    with pytest.raises(ValueError, match="outside"):
        select_wrong_caption(rows, 1, [0, 3, 4])


@pytest.mark.parametrize("aligned", [False, True])
def test_fixture_runner_writes_complete_controls_and_preserves_frozen_state(
    tmp_path, monkeypatch, aligned
):
    import src.data.image_generation_webdataset as data_module
    import src.decoders.loading as loading
    import tools.diagnose_prism_image_conditioning as diagnostic
    import tools.train_prism_image_diffusion as training

    class FixtureDataset:
        def __init__(self, index, *, target_size, split):
            self.index = index
            self.data_fingerprint = "fixture-data"
            self.records = [
                SimpleNamespace(
                    id=f"{split}-{i}",
                    prompt=f"image {i}",
                    task="t2i",
                    source_ids=(),
                    source_paths=(),
                    split=split,
                )
                for i in range(3)
            ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return {"target_image": torch.full((3, 32, 32), index / 3)}

    class FixtureBackend(nn.Module, Backend):
        def __init__(self):
            nn.Module.__init__(self)
            Backend.__init__(self)
            self.conditioning_dim = 4
            self.transformer.device = "cpu"
            self._pipeline = SimpleNamespace(
                _apply_chat_template=lambda text: "N:" + text,
                processor=SimpleNamespace(tokenizer=Tokenizer()),
                transformer=self.transformer,
            )

            def native_encode(**kwargs):
                text = "N:" + kwargs["prompt"][0]
                ids = Tokenizer()([text], truncation=False, return_tensors="pt")
                features = ids["input_ids"].float().unsqueeze(-1).expand(-1, -1, 4) / 10
                return features, ids["attention_mask"], None, None

            self._pipeline.encode_prompt = native_encode

        def ensure_loaded(self):
            return self

        def checkpoint_manifest(self):
            return {"manifest_sha256": "fixture-generator"}

        def provenance(self):
            return {"kernel_policy": "fixture"}

    class FixtureParent(Parent):
        def __init__(self):
            super().__init__()
            self.decoders["image"] = ImageDecoder(4, backend=FixtureBackend(), conditioning_dim=4)

    parent = FixtureParent()
    events = []
    original_samples = diagnostic.sample_variants

    def samples(*args, **kwargs):
        if kwargs.get("only_native"):
            events.append("native_pretrained_baseline")
        return original_samples(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "sample_variants", samples)
    monkeypatch.setattr(data_module, "ImageGenerationWebDataset", FixtureDataset)
    monkeypatch.setattr(
        loading,
        "load_image_training_bundle",
        lambda *args: {
            "model": parent,
            "tokenizer": Tokenizer(),
            "provenance": {
                "fixture": True,
                "restoration": {
                    "strict_parent": True,
                    "loaded_key_count": 2,
                    "missing_parent_keys": [],
                    "unexpected_keys": [],
                },
            },
        },
    )
    monkeypatch.setattr(
        training,
        "restore_warm_connector",
        lambda *args, **kwargs: events.append("warm500") or {"fixture": True},
    )
    monkeypatch.setattr(
        diagnostic, "validate_repeatability_report", lambda *args, **kwargs: {"fixture": True}
    )
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
    ):
        path = tmp_path / name
        path.write_text("fixture")
        argv += ["--" + name, str(path)]
    argv += [
        "--output-dir",
        str(tmp_path / "output"),
        "--train-probe-count",
        "2",
        "--validation-probe-count",
        "1",
        "--sample-count",
        "1",
        "--prism-formats",
        "raw",
        "--flow-timesteps",
        ".2",
        ".8",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--expected-parent-tensors",
        "2",
        "--sampling-steps",
        "2",
        "--height",
        "32",
        "--width",
        "32",
    ]
    if aligned:
        import tools.prism_image_alignment_checkpoint as adapter

        source = tmp_path / "alignment"
        source.mkdir()
        checkpoint = source / "alignment.pt"
        checkpoint.write_text("fixture")
        selection = [{"index": 2, "id": "train-2"}, {"index": 0, "id": "train-0"}]
        (source / "report.json").write_text(
            json.dumps({"data_fingerprint": "fixture-data", "selection": {"train": selection}})
        )

        def restore(model, path, **kwargs):
            assert events == ["warm500", "native_pretrained_baseline"]
            assert not any(parameter.requires_grad for parameter in model.parameters())
            assert path == checkpoint and kwargs["fixture"] is True
            assert set(kwargs["records"]) == {"train", "validation"}
            assert kwargs["expected_sha256"] == "f" * 64
            events.append("alignment_weights")
            model.decoders["image"].connector[1].bias.add_(0.125)
            return {
                "selection": {"train": selection},
                "evidence_kind": "fixture_only_connector_native_feature_alignment",
                "prompt_format": "chat",
            }

        monkeypatch.setattr(adapter, "restore_alignment_connector", restore)
        argv += [
            "--alignment-checkpoint",
            str(checkpoint),
            "--alignment-checkpoint-sha256",
            "f" * 64,
        ]
    report = diagnostic._run(diagnostic._parser().parse_args(argv))
    assert report["status"] == "completed" and report["evidence_kind"] == "fixture_only"
    assert report["training_performed"] is False
    assert report["frozen_state_unchanged"] and report["native_pretrained_state_unchanged"]
    assert len(report["flow_controls"]) == 6
    assert len(report["samples"]) == 4
    assert report["negative_condition_statistics"]["raw"]["empty_anchor"] == "eos"
    assert report["summary"]["train"]["native_pretrained"]["paired_evaluations"] == 4

    if aligned:
        assert events == ["warm500", "native_pretrained_baseline", "alignment_weights"]
        assert report["alignment_training_prompt_format"] == "chat"
        assert (
            report["alignment_checkpoint"]["evidence_kind"]
            == "fixture_only_connector_native_feature_alignment"
        )
        assert report["selection"]["train"] == selection
        train_controls = [row for row in report["flow_controls"] if row["split"] == "train"]
        assert all({row["id"], row["wrong_id"]} == {"train-0", "train-2"} for row in train_controls)
        assert all("native_pretrained" in row["routes"] for row in report["flow_controls"])
        assert not any("adapted_diffusion" in row["route"] for row in report["samples"])


def test_diagnostic_adaptation_sources_are_mutually_exclusive():
    import tools.diagnose_prism_image_conditioning as diagnostic

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
    with pytest.raises(SystemExit):
        diagnostic._parser().parse_args(
            argv + ["--joint-checkpoint", "/joint", "--alignment-checkpoint", "/aligned"]
        )
    args = diagnostic._parser().parse_args(argv + ["--alignment-checkpoint-sha256", "f" * 64])
    with pytest.raises(ValueError, match="requires --alignment-checkpoint"):
        diagnostic.validate_budget(args)
