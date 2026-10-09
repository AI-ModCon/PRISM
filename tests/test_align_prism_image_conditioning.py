"""Offline alignment contracts; no assertion of real generation quality."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.decoders.image import ImageDecoder
from src.decoders.loading import file_sha256
from src.decoders.omnigen2_backend import OmniGen2Backend
from src.decoders.types import DecoderCondition
from tools import align_prism_image_conditioning as alignment
from tools.prism_image_conditioning import format_prompt
from tools.train_image_decoder import frozen_state_hashes
from torch import nn


class Tokenizer:
    vocabulary = {"red": 4, "blue": 5, "green": 6, "yellow": 7, "round": 8, "square": 9}

    def apply_chat_template(self, messages, **kwargs):
        return (
            "<system>"
            + messages[0]["content"]
            + "</system><user>"
            + messages[1]["content"]
            + "</user>"
        )

    def __call__(self, prompts, **kwargs):
        rows = []
        for prompt in prompts:
            wrapped = prompt.startswith("<system>")
            caption = prompt.split("<user>")[1].split("</user>")[0] if wrapped else prompt
            ids = [self.vocabulary[word] for word in caption.split()]
            rows.append([1, 2] + ids + [3] if wrapped else ids)
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            ),
        }


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_caption_embed = nn.Module()
        norm = nn.RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))
        self.time_caption_embed.caption_embedder = nn.Sequential(norm, nn.Linear(4, 4))

    @property
    def device(self):
        return next(self.parameters()).device


class Backend(OmniGen2Backend):
    def __init__(self, tokenizer):
        super().__init__(conditioning_dim=4)
        self.transformer = Transformer()
        self.mllm = Teacher()
        self.vae = nn.Linear(4, 4)
        self._pipeline = SimpleNamespace(
            transformer=self.transformer,
            mllm=self.mllm,
            processor=SimpleNamespace(tokenizer=tokenizer),
            _apply_chat_template=lambda text: format_prompt(text, tokenizer, "chat"),
            encode_prompt=self.encode_prompt,
        )
        self.actual_token_mismatch = False
        self.ids_as_keyword = False
        self._apply_training_policy()

    def encode_prompt(self, *, prompt, **kwargs):
        tokens = self._pipeline.processor.tokenizer(
            [self._pipeline._apply_chat_template(text) for text in prompt]
        )
        ids = tokens["input_ids"].clone()
        if self.actual_token_mismatch:
            ids[:, 0] += 1
        hidden = (
            self.mllm(input_ids=ids, attention_mask=tokens["attention_mask"])
            if self.ids_as_keyword
            else self.mllm(ids, attention_mask=tokens["attention_mask"])
        )
        return hidden, tokens["attention_mask"], None, None

    def checkpoint_manifest(self):
        return {"manifest_sha256": "b" * 64, "files": {"fixture": "c" * 64}}

    def provenance(self):
        return {"fixture": True, "training_scope": "connector_only"}


class Parent(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.decoders = nn.ModuleDict({"image": ImageDecoder(4, backend=Backend(tokenizer))})

    def _output_condition(self, inputs):
        assert set(inputs) == {"text", "text_attention_mask"}
        condition = DecoderCondition(self.embedding(inputs["text"]), inputs["text_attention_mask"])
        return condition, None, None, None


def model_and_tokenizer():
    torch.manual_seed(7)
    tokenizer = Tokenizer()
    model = Parent(tokenizer)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def record(identifier, caption="red blue", split="train"):
    return SimpleNamespace(
        id=identifier, prompt=caption, split=split, task="t2i", source_ids=(), source_paths=()
    )


def test_content_span_requires_unique_exact_valid_raw_sequence():
    ids = torch.tensor([[1, 2, 4, 5, 3]])
    mask = torch.ones_like(ids)
    content, span = alignment.caption_content_mask(ids, mask, torch.tensor([[4, 5]]))
    assert span == [2, 4]
    assert content.tolist() == [[False, False, True, True, False]]
    for raw in (torch.tensor([[8]]), torch.tensor([[1, 2, 4, 5, 3]])):
        with pytest.raises(ValueError):
            alignment.caption_content_mask(ids, mask, raw)
    with pytest.raises(ValueError, match="ambiguous"):
        alignment.caption_content_mask(
            torch.tensor([[1, 4, 4, 3]]), torch.ones(1, 4), torch.tensor([[4]])
        )
    with pytest.raises(ValueError, match="missing"):
        alignment.caption_content_mask(ids, torch.tensor([[1, 1, 0, 1, 1]]), torch.tensor([[4, 5]]))


@pytest.mark.parametrize(
    "change", ["formatted_prompt", "input_ids", "input_attention_mask", "attention_mask"]
)
def test_feature_pair_rejects_any_text_or_token_or_mask_mismatch(change):
    features = torch.ones(1, 5, 4)
    prism = {
        "formatted_prompt": "same",
        "input_ids": torch.tensor([[1, 2, 4, 5, 3]]),
        "input_attention_mask": torch.ones(1, 5),
        "attention_mask": torch.ones(1, 5),
        "hidden_states": features,
    }
    native = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in prism.items()
    }
    native["embeds"] = features
    if change == "formatted_prompt":
        native[change] = "different"
    else:
        native[change][0, 0] += 1
    with pytest.raises(ValueError, match="differ"):
        alignment.validate_pair(prism, native, torch.tensor([[4, 5]]))


def test_native_teacher_actual_inputs_are_checked_and_hook_removed():
    model, _ = model_and_tokenizer()
    backend = model.decoders["image"].backend
    backend.actual_token_mismatch = True
    with pytest.raises(RuntimeError, match="Actual native teacher"):
        alignment.encode_observed_native(backend, "red blue", max_text_length=32)
    assert not backend.mllm._forward_pre_hooks


def test_cache_is_detached_caption_only_and_records_exact_alignment():
    model, tokenizer = model_and_tokenizer()
    cache, audit = alignment.build_feature_cache(
        model,
        tokenizer,
        model.decoders["image"].backend,
        [record("train-1")],
        device="cpu",
        max_text_length=32,
    )
    assert all(
        not cache[0][key].requires_grad and cache[0][key].device.type == "cpu"
        for key in ("hidden", "teacher")
    )
    assert audit[0]["actual_native_forward_inputs_verified"]
    assert audit[0]["content_span"] == [2, 4] and audit[0]["template_tokens"] == 3
    assert not audit[0]["target_pixels_read"]
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="max_text_length"):
        alignment.build_feature_cache(
            model,
            tokenizer,
            model.decoders["image"].backend,
            [record("train-1")],
            device="cpu",
            max_text_length=3,
        )


def test_actual_rmsnorm_weights_and_content_only_objective_define_gradients():
    connector = nn.Linear(4, 4, bias=False)
    norm = nn.RMSNorm(4, eps=1e-6).requires_grad_(False)
    with torch.no_grad():
        connector.weight.copy_(torch.eye(4))
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]])
    target = norm(hidden).detach().clone()
    target[:, 1] = -target[:, 1]  # Deliberately wrong template must not affect content gradients.
    batch = {
        "hidden": hidden,
        "teacher": target,
        "content": torch.tensor([[True, False]]),
        "template": torch.tensor([[False, True]]),
        "conditioning_dtype": torch.float32,
    }
    losses = alignment.alignment_losses(connector, norm, batch)
    assert losses["content_mse"].item() == 0 and losses["template_mse"].item() > 0
    losses["content_mse"].backward()
    assert torch.equal(connector.weight.grad, torch.zeros_like(connector.weight))
    assert norm.weight.grad is None
    batch["teacher"] = torch.zeros_like(target)
    losses = alignment.alignment_losses(connector, norm, batch)
    assert torch.allclose(losses["content_mse"], norm(hidden)[:, 0].square().mean())
    assert not torch.allclose(
        losses["content_mse"], nn.functional.normalize(hidden[:, 0], dim=-1).square().mean()
    )


def test_cached_alignment_updates_only_connector_and_leaves_all_frozen_weights_unchanged():
    model, tokenizer = model_and_tokenizer()
    backend, connector = model.decoders["image"].backend, model.decoders["image"].connector
    cache, _ = alignment.build_feature_cache(
        model,
        tokenizer,
        backend,
        [record("one"), record("two", "green yellow")],
        device="cpu",
        max_text_length=32,
    )
    before = frozen_state_hashes(model, alignment.MODULES)
    connector.requires_grad_(True)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=0.02, weight_decay=0.0)
    norm = alignment.caption_norm(backend)
    batch = alignment.cached_batch(cache, "cpu")
    initial = float(alignment.alignment_losses(connector, norm, batch)["content_mse"].detach())
    for _ in range(80):
        optimizer.zero_grad()
        alignment.alignment_losses(connector, norm, batch)["content_mse"].backward()
        assert all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith(alignment.PREFIX)
        )
        optimizer.step()
    final = float(alignment.alignment_losses(connector, norm, batch)["content_mse"].detach())
    assert final < initial * 0.2
    assert frozen_state_hashes(model, alignment.MODULES) == before


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import src.data.image_generation_webdataset as data_module
    import src.decoders.loading as loading

    model, tokenizer = model_and_tokenizer()
    parent = {
        "fixture": True,
        "parent_checkpoint_sha256": "a" * 64,
        "restoration": {
            "strict_parent": True,
            "loaded_key_count": 2,
            "missing_parent_keys": [],
            "unexpected_keys": [],
        },
    }
    models = []

    def load(*args):
        instance, tokens = model_and_tokenizer()
        models.append(instance)
        return {"model": instance, "tokenizer": tokens, "provenance": parent}

    monkeypatch.setattr(loading, "load_image_training_bundle", load)
    monkeypatch.setattr(
        alignment, "validate_repeatability_report", lambda *args, **kwargs: {"fixture": True}
    )
    rows = {
        "train": [
            record("train-1"),
            record("train-2", "green yellow"),
            record("train-3", "round square"),
        ],
        "validation": [
            record("val-1", "red square", "validation"),
            record("val-2", "blue round", "validation"),
        ],
    }

    class MetadataOnlyDataset:
        def __init__(self, index, *, split):
            self.index, self.records = Path(index), rows[split]
            self.data_fingerprint, self.validation_report = "data-fingerprint", {"fixture": True}

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            raise AssertionError("Alignment must never load target images")

    monkeypatch.setattr(data_module, "ImageGenerationWebDataset", MetadataOnlyDataset)
    assets = {}
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "train-index",
        "validation-index",
        "repeatability-report",
    ):
        assets[name] = tmp_path / name
        assets[name].write_text("fixture")
    warm_root = tmp_path / "warm"
    warm_root.mkdir()
    checkpoint = warm_root / "connector.pt"
    frozen = frozen_state_hashes(model, alignment.MODULES)
    torch.save(
        {
            "schema_version": 1,
            "evidence_kind": "fixture_only",
            "qualification": "unqualified",
            "step": 500,
            "connector_modules": list(alignment.MODULES),
            "connector_state_dict": {
                name: value
                for name, value in model.state_dict().items()
                if name.startswith(alignment.PREFIX)
            },
            "protocol": {
                "parent": parent,
                "reference_checkpoint_sha256": "b" * 64,
                "data_fingerprint": "data-fingerprint",
            },
            "frozen_hashes": frozen,
        },
        checkpoint,
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
                "checkpoints": [{"sha256": file_sha256(checkpoint), "step": 500}],
            }
        )
    )
    assets["connector-checkpoint"] = checkpoint
    argv = [value for name, path in assets.items() for value in ("--" + name, str(path))]
    argv += [
        "--output-dir",
        str(tmp_path / "output"),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--expected-parent-tensors",
        "2",
        "--train-subset-size",
        "2",
        "--validation-count",
        "2",
        "--steps",
        "8",
        "--eval-every",
        "4",
        "--learning-rate",
        "0.01",
        "--deterministic",
    ]
    return SimpleNamespace(
        args=alignment._parser().parse_args(argv), argv=argv, models=models, root=tmp_path
    )


def test_complete_fixture_provenance_excludes_cache_and_preserves_kind(experiment):
    alignment.validate_budget(experiment.args)
    assert alignment._run(experiment.args) == 0
    report = json.loads((experiment.root / "output/report.json").read_text())
    assert report["evidence_kind"] == alignment.FIXTURE_KIND
    assert report["schema_version"] == 1
    assert (
        not {"objective", "init_alignment_checkpoint", "init_alignment_checkpoint_sha256"}
        & report["settings"].keys()
    )
    assert report["frozen_state_unchanged"] and report["connector_state_changed"]
    assert report["objective"]["template_positions"].startswith("excluded")
    assert report["sample_exposure"]["examples_seen"] == 32
    assert report["sample_exposure"]["unique_examples_seen"] == 2
    assert (
        not report["cache_saved"]
        and not report["target_pixels_read"]
        and not report["negative_anchor_trained"]
    )
    assert not report["flow_training_performed"] and not report["quality_benchmark"]
    assert (
        len(report["cache_audit"]["train"]) == 2 and len(report["cache_audit"]["validation"]) == 2
    )
    assert (
        report["evaluations"][-1]["splits"]["train"]["content_mse"]
        < report["evaluations"][0]["splits"]["train"]["content_mse"]
    )
    saved = torch.load(report["checkpoints"][-1]["path"], weights_only=True)
    assert saved["evidence_kind"] == alignment.FIXTURE_KIND
    assert saved["protocol"]["objective"] == alignment.OBJECTIVE
    assert saved["protocol"]["selection"] == report["selection"]
    assert saved["frozen_hashes"] == report["frozen_hashes_after"]
    assert "cache" not in saved and "hidden" not in saved and "teacher" not in saved
    assert all(tensor.dtype == torch.float32 for tensor in saved["connector_state_dict"].values())
    assert all(
        parameter.grad is None
        for name, parameter in experiment.models[-1].named_parameters()
        if not name.startswith(alignment.PREFIX)
    )
    assert not list((experiment.root / "output").glob("*cache*"))
    assert set(report["evaluations"][0]["splits"]["train"]["examples"][0]) == {
        "id",
        "content_mse",
        "content_cosine",
        "template_mse",
        "template_cosine",
    }


def test_cli_requires_pbs_xpu_and_bounded_steps(experiment, monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="PBS"):
        alignment.main(experiment.argv)
    experiment.args.steps = 5001
    with pytest.raises(ValueError, match="steps"):
        alignment.validate_budget(experiment.args)


@pytest.mark.parametrize("keyword_ids", [False, True])
def test_native_teacher_accepts_upstream_positional_ids_or_unambiguous_keyword_ids(keyword_ids):
    model, _ = model_and_tokenizer()
    backend = model.decoders["image"].backend
    backend.ids_as_keyword = keyword_ids
    native = alignment.encode_observed_native(backend, "red blue", max_text_length=32)
    assert native["input_ids"].tolist() == [[1, 2, 4, 5, 3]]
    assert not backend.mllm._forward_pre_hooks


def test_region_masks_cover_valid_tokens_and_exclude_padding():
    full = torch.tensor([[1, 1, 1, 1, 1, 0]])
    content = torch.tensor([[False, False, True, True, False, False]])
    masks = alignment.caption_region_masks(full, content, [2, 4])
    assert {key: value.sum().item() for key, value in masks.items()} == {
        "prefix": 2,
        "content": 2,
        "suffix": 1,
    }
    assert all(not value[0, -1] for value in masks.values())
    for bad_mask, bad_content, span in (
        (torch.tensor([[1, 0, 1, 1, 1, 0]]), content, [2, 4]),
        (full, ~content, [2, 4]),
        (full, content, [0, 4]),
        (full, content, [2, 5]),
        (torch.tensor([[1, 1, 1, 1, 0, 0]]), content, [2, 4]),
    ):
        with pytest.raises(ValueError):
            alignment.caption_region_masks(bad_mask, bad_content, span)


def test_equal_region_loss_uses_equal_captions_not_token_counts_and_real_norm():
    torch.manual_seed(81)
    norm = nn.RMSNorm(3, eps=1e-6).requires_grad_(False)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 0.5]))
    hidden = torch.randn(2, 7, 3, requires_grad=True)
    target = norm(torch.randn(2, 7, 3)).detach()
    prefix = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    content = torch.tensor([[0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    suffix = torch.tensor([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    batch = {
        "hidden": hidden,
        "teacher": target,
        "content": content,
        "prefix": prefix,
        "suffix": suffix,
        "template": prefix | suffix,
        "conditioning_dtype": torch.float32,
    }
    losses = alignment.alignment_losses(nn.Identity(), norm, batch, objective="regions")
    token_mse = (norm(hidden) - target).square().mean(-1)
    manual = torch.stack(
        [
            torch.stack([token_mse[i][mask[i]].mean() for mask in (prefix, content, suffix)]).mean()
            for i in range(2)
        ]
    ).mean()
    assert torch.allclose(losses["region_balanced_mse"], manual)
    global_token_mean = token_mse[prefix | content | suffix].mean()
    assert not torch.isclose(manual, global_token_mean)
    losses["region_balanced_mse"].backward()
    assert (hidden.grad[prefix | content | suffix].abs().sum(-1) > 0).all()
    assert not hidden.grad[~(prefix | content | suffix)].any()
    assert norm.weight.grad is None
    hidden.grad = None
    legacy = alignment.alignment_losses(nn.Identity(), norm, batch)
    legacy["content_mse"].backward()
    assert not hidden.grad[prefix | suffix].any()
    assert set(legacy) == {"content_mse", "content_cosine", "template_mse", "template_cosine"}
    bad = dict(batch, prefix=content)
    with pytest.raises(ValueError, match="exactly once"):
        alignment.alignment_losses(nn.Identity(), norm, bad, objective="regions")


def test_region_cache_contains_verified_boundaries_without_images_or_changed_v1_fields():
    model, tokenizer = model_and_tokenizer()
    backend = model.decoders["image"].backend
    rows = [record("train-1"), record("train-2", "green yellow round")]
    legacy, old_audits = alignment.build_feature_cache(
        model, tokenizer, backend, rows, device="cpu", max_text_length=32
    )
    regions, audits = alignment.build_feature_cache(
        model, tokenizer, backend, rows, device="cpu", max_text_length=32, objective="regions"
    )
    for old, new in zip(old_audits, audits, strict=True):
        assert all(new[key] == value for key, value in old.items())
        assert new["actual_native_forward_inputs_verified"] and not new["target_pixels_read"]
        assert set(new["region_spans"]) == {"prefix", "content", "suffix"}
    batch = alignment.cached_batch(regions, "cpu")
    assert batch["prefix"].sum(1).tolist() == [2, 2]
    assert batch["content"].sum(1).tolist() == [2, 3]
    assert batch["suffix"].sum(1).tolist() == [1, 1]
    with pytest.raises(ValueError, match="cannot mix"):
        alignment.cached_batch([legacy[0], regions[1]], "cpu")


@pytest.fixture
def region_experiment(experiment):
    assert alignment._run(experiment.args) == 0
    source_report = json.loads((experiment.root / "output/report.json").read_text())
    checkpoint = Path(source_report["checkpoints"][-1]["path"])
    args = copy.copy(experiment.args)
    args.objective = "regions"
    args.output_dir = experiment.root / "regions"
    args.init_alignment_checkpoint = checkpoint
    args.init_alignment_checkpoint_sha256 = file_sha256(checkpoint)
    experiment.region_args = args
    experiment.source_report = source_report
    experiment.source_checkpoint = checkpoint
    return experiment


def test_region_stage_has_distinct_evidence_strict_v1_initialization_and_fresh_state(
    region_experiment,
):
    experiment = region_experiment
    args = experiment.region_args
    assert alignment._run(args) == 0
    report = json.loads((args.output_dir / "report.json").read_text())
    source = experiment.source_report
    assert (
        report["schema_version"] == 2 and report["evidence_kind"] == alignment.REGIONS_FIXTURE_KIND
    )
    assert report["objective"] == alignment.REGIONS_OBJECTIVE
    assert report["optimized_loss_key"] == "region_balanced_mse"
    assert report["settings"]["objective"] == "regions"
    assert report["alignment_initialization"]["sha256"] == file_sha256(experiment.source_checkpoint)
    assert report["selection"] == source["selection"]
    assert report["connector_hashes_before"] == source["connector_hashes_after"]
    assert report["warm_connector_hashes_before_initializer"] == source["connector_hashes_before"]
    assert report["sample_exposure"]["examples_seen"] == args.steps * args.batch_size
    assert report["sample_exposure"]["unique_examples_seen"] == args.train_subset_size
    assert (
        report["frozen_state_unchanged"]
        and not report["cache_saved"]
        and not report["target_pixels_read"]
    )
    for flag in ("optimizer_state_restored", "sampler_state_restored", "rng_state_restored"):
        assert report["alignment_initialization"][flag] is False
    assert report["stage_initialization"] == {
        "optimizer": "fresh_adamw",
        "sampler": "fresh_seeded_order",
        "rng": "fresh_seed",
        "source_optimizer_restored": False,
        "source_sampler_restored": False,
        "source_rng_restored": False,
    }
    for evaluation in report["evaluations"]:
        for split in evaluation["splits"].values():
            for entry in split["examples"]:
                assert entry["region_balanced_mse"] == pytest.approx(
                    sum(entry[r + "_mse"] for r in ("prefix", "content", "suffix")) / 3
                )
    assert (
        report["evaluations"][-1]["splits"]["train"]["region_balanced_mse"]
        < report["evaluations"][0]["splits"]["train"]["region_balanced_mse"]
    )
    checkpoint = Path(report["checkpoints"][-1]["path"])
    assert checkpoint.name.startswith("connector-region-feature-alignment")
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["schema_version"] == 2 and saved["evidence_kind"] == alignment.REGIONS_FIXTURE_KIND
    assert saved["protocol"]["alignment_initialization"] == report["alignment_initialization"]
    assert (
        saved["protocol"]["warm_connector_hashes_before_initializer"]
        == report["warm_connector_hashes_before_initializer"]
    )
    assert saved["protocol"]["stage_initialization"] == report["stage_initialization"]
    assert all(
        int(state["step"]) == args.steps
        for state in saved["optimizer_state_dict"]["state"].values()
    )
    assert saved["sampler_state"]["examples_seen"] == args.steps * args.batch_size
    assert not any(key in saved for key in ("cache", "hidden", "teacher"))
    assert len(report["cache_audit"]["train"]) == args.train_subset_size
    assert len(report["cache_audit"]["validation"]) == args.validation_count
    assert not list(args.output_dir.glob("*cache*"))


@pytest.mark.parametrize("change", ["sha", "selection", "kind"])
def test_region_initialization_rejects_wrong_digest_cohort_or_kind_before_cache(
    region_experiment, change
):
    experiment = region_experiment
    args = experiment.region_args
    if change == "sha":
        args.init_alignment_checkpoint_sha256 = "f" * 64
    elif change == "selection":
        args.train_subset_size = 3
    else:
        source = dict(experiment.source_report, evidence_kind=alignment.REGIONS_FIXTURE_KIND)
        (experiment.source_checkpoint.parent / "report.json").write_text(json.dumps(source))
    with pytest.raises(ValueError):
        alignment._run(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed" and not report.get("cache_audit")
    connector = experiment.models[-1].decoders["image"].connector
    from tools.train_image_decoder import _tensor_hash

    assert {
        name: _tensor_hash(value) for name, value in connector.state_dict().items()
    } == experiment.source_report["connector_hashes_before"]


def test_region_cli_requires_initializer_and_legacy_refuses_initializers(experiment):
    args = copy.copy(experiment.args)
    args.objective = "regions"
    with pytest.raises(ValueError, match="init-alignment-checkpoint"):
        alignment.validate_budget(args)
    args.objective = "content"
    args.init_alignment_checkpoint = Path("/unused")
    with pytest.raises(ValueError, match="does not accept"):
        alignment.validate_budget(args)


def test_region_stage_rejects_frozen_feature_drift_before_any_update(
    region_experiment, monkeypatch
):
    original = alignment.build_feature_cache

    def changed_audit(*args, **kwargs):
        cache, audit = original(*args, **kwargs)
        if kwargs.get("objective") == "regions":
            audit[0]["native_features_sha256"] = "0" * 64
        return cache, audit

    monkeypatch.setattr(alignment, "build_feature_cache", changed_audit)
    args = region_experiment.region_args
    with pytest.raises(ValueError, match="frozen feature audit differs"):
        alignment._run(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed" and report["completed_steps"] == 0
    assert report["checkpoints"] == [] and report["evaluations"] == []
    assert report["frozen_state_unchanged"]
    from tools.train_image_decoder import _tensor_hash

    connector = region_experiment.models[-1].decoders["image"].connector
    assert {
        name: _tensor_hash(value) for name, value in connector.state_dict().items()
    } == region_experiment.source_report["connector_hashes_after"]
