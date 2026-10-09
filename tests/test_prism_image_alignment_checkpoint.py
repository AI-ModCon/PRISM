"""Strict admission of feature-alignment artifacts, without optimizer restoration."""

import copy
import json
from pathlib import Path

import pytest
import torch
from src.decoders.loading import file_sha256
from tools import align_prism_image_conditioning as alignment
from tools.prism_image_alignment_checkpoint import restore_alignment_connector
from tools.train_image_decoder import frozen_state_hashes

import test_align_prism_image_conditioning as alignment_fixtures
from test_align_prism_image_conditioning import model_and_tokenizer

experiment = alignment_fixtures.experiment


@pytest.fixture
def artifact(experiment):
    import src.data.image_generation_webdataset as data_module

    alignment._run(experiment.args)
    report_path = experiment.root / "output/report.json"
    report = json.loads(report_path.read_text())
    checkpoint = Path(report["checkpoints"][-1]["path"])
    model, tokenizer = model_and_tokenizer()
    kwargs = {
        "parent": report["parent"],
        "generator_manifest": model.decoders["image"].backend.checkpoint_manifest(),
        "data_fingerprint": report["data_fingerprint"],
        "index_sha256": report["index_sha256"],
        "records": {
            split: data_module.ImageGenerationWebDataset(
                getattr(experiment.args, split + "_index"), split=split
            ).records
            for split in ("train", "validation")
        },
        "tokenizer": tokenizer,
        "fixture": True,
    }
    return model, checkpoint, report_path, report, kwargs


def rewrite(checkpoint, report_path, report, saved):
    torch.save(saved, checkpoint)
    report["checkpoints"][-1]["sha256"] = file_sha256(checkpoint)
    report_path.write_text(json.dumps(report))


def test_restore_weights_only_preserves_frozen_rng_modes_and_flags(artifact):
    model, checkpoint, _, report, kwargs = artifact
    model.decoders["image"].connector.train()
    model.decoders["image"].connector.requires_grad_(True)
    frozen = frozen_state_hashes(model, alignment.MODULES)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    lineage = restore_alignment_connector(model, checkpoint, **kwargs)
    assert lineage["evidence_kind"] == alignment.FIXTURE_KIND
    assert lineage["restore_policy"] == "connector_weights_only_from_native_feature_alignment"
    assert (
        not lineage["optimizer_state_restored"]
        and not lineage["rng_state_restored"]
        and not lineage["sampler_state_restored"]
    )
    assert (
        lineage["exact_token_and_content_audit_revalidated"]
        and lineage["selection"] == report["selection"]
    )
    assert lineage["prompt_format"] == "chat" and not lineage["negative_anchor_trained"]
    assert frozen_state_hashes(model, alignment.MODULES) == frozen
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert torch.equal(rng, torch.get_rng_state())
    saved = torch.load(checkpoint, weights_only=True)
    for name, value in model.decoders["image"].connector.state_dict().items():
        assert torch.equal(value, saved["connector_state_dict"][alignment.PREFIX + name])


@pytest.mark.parametrize(
    "change",
    [
        "flow_kind",
        "unfinished",
        "nonterminal",
        "digest",
        "parent",
        "generator",
        "frozen",
        "missing",
        "nonfinite",
        "dtype",
        "norm",
        "token_hash",
        "content_span",
        "actual_token_audit",
        "heldout",
        "caption",
        "data",
        "index",
        "source",
        "exposure",
        "flow_claim",
    ],
)
def test_rejects_invalid_alignment_without_partial_mutation(artifact, change):
    model, checkpoint, report_path, report, kwargs = artifact
    saved = torch.load(checkpoint, weights_only=True)
    if change == "flow_kind":
        saved["evidence_kind"] = report["evidence_kind"] = "fixture_only"
    elif change == "unfinished":
        report["status"] = "running"
    elif change == "nonterminal":
        saved["step"] -= 1
    elif change == "digest":
        kwargs["expected_sha256"] = "f" * 64
    elif change == "parent":
        kwargs["parent"] = dict(kwargs["parent"], parent_checkpoint_sha256="f" * 64)
    elif change == "generator":
        kwargs["generator_manifest"] = dict(kwargs["generator_manifest"], manifest_sha256="f" * 64)
    elif change == "frozen":
        with torch.no_grad():
            model.embedding.weight.add_(0.125)
    elif change == "missing":
        saved["connector_state_dict"].pop(next(iter(saved["connector_state_dict"])))
    elif change == "nonfinite":
        saved["connector_state_dict"][next(iter(saved["connector_state_dict"]))].fill_(float("nan"))
    elif change == "dtype":
        key = next(iter(saved["connector_state_dict"]))
        saved["connector_state_dict"][key] = saved["connector_state_dict"][key].bfloat16()
    elif change == "norm":
        saved["protocol"]["caption_normalization"]["eps"] = 0.2
        report["caption_normalization"] = saved["protocol"]["caption_normalization"]
    elif change in {"token_hash", "content_span", "actual_token_audit"}:
        key, value = {
            "token_hash": ("input_ids_sha256", "f" * 64),
            "content_span": ("content_span", [0, 2]),
            "actual_token_audit": ("actual_native_forward_inputs_verified", False),
        }[change]
        saved["protocol"]["cache_audit"]["train"][0][key] = value
        report["cache_audit"] = saved["protocol"]["cache_audit"]
    elif change == "heldout":
        kwargs["records"]["validation"][0].id = "another-heldout-ID"
    elif change == "caption":
        kwargs["records"]["train"][report["selection"]["train"][0]["index"]].prompt = "red round"
    elif change == "data":
        kwargs["data_fingerprint"] = "another-conversion"
    elif change == "index":
        kwargs["index_sha256"] = dict(kwargs["index_sha256"], validation="f" * 64)
    elif change == "source":
        saved["protocol"]["source_sha256"]["src/model.py"] = "f" * 64
        report["source_sha256"] = saved["protocol"]["source_sha256"]
    elif change == "exposure":
        saved["sample_exposure"]["val-1"] = 1
    elif change == "flow_claim":
        report["flow_training_performed"] = True
    rewrite(checkpoint, report_path, report, saved)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        restore_alignment_connector(model, checkpoint, **kwargs)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_fixture_cannot_be_admitted_as_real_alignment(artifact):
    model, checkpoint, _, _, kwargs = artifact
    kwargs["fixture"] = False
    with pytest.raises(ValueError, match="fixture"):
        restore_alignment_connector(model, checkpoint, **kwargs)


def test_original_diffusion_changes_are_rejected_even_when_manifest_is_same(artifact):
    model, checkpoint, _, _, kwargs = artifact
    with torch.no_grad():
        model.decoders["image"].backend.transformer.time_caption_embed.caption_embedder[
            1
        ].weight.add_(0.1)
    with pytest.raises(ValueError, match="frozen"):
        restore_alignment_connector(model, checkpoint, **kwargs)


def test_pending_gradients_or_trainable_teacher_are_rejected(artifact):
    model, checkpoint, _, _, kwargs = artifact
    model.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        restore_alignment_connector(model, checkpoint, **kwargs)
    model.embedding.weight.requires_grad_(False)
    next(model.decoders["image"].connector.parameters()).grad = torch.ones_like(
        next(model.decoders["image"].connector.parameters())
    )
    with pytest.raises(ValueError, match="gradients"):
        restore_alignment_connector(model, checkpoint, **kwargs)


@pytest.mark.parametrize("change", ["lineage", "current_weights"])
def test_audited_500_step_starting_connector_is_mandatory(artifact, change):
    model, checkpoint, report_path, report, kwargs = artifact
    if change == "lineage":
        saved = torch.load(checkpoint, weights_only=True)
        saved["protocol"]["connector_warm_start"]["step"] = 499
        report["connector_warm_start"] = saved["protocol"]["connector_warm_start"]
        rewrite(checkpoint, report_path, report, saved)
    else:
        with torch.no_grad():
            model.decoders["image"].connector[1].bias.add_(0.1)
    with pytest.raises(ValueError, match="500-step"):
        restore_alignment_connector(model, checkpoint, **kwargs)


def test_current_native_tokenizer_mismatch_is_rejected(artifact):
    model, checkpoint, _, _, kwargs = artifact
    pipeline = model.decoders["image"].backend._pipeline
    original = pipeline.processor.tokenizer

    def different(*args, **values):
        tokens = original(*args, **values)
        tokens["input_ids"][0, 0] += 1
        return tokens

    pipeline.processor.tokenizer = different
    with pytest.raises(ValueError, match="current PRISM/native token IDs"):
        restore_alignment_connector(model, checkpoint, **kwargs)


@pytest.fixture
def region_artifact(artifact, experiment):
    model, source, _, source_report, kwargs = artifact
    args = copy.deepcopy(experiment.args)
    args.objective = "regions"
    args.init_alignment_checkpoint = source
    args.init_alignment_checkpoint_sha256 = file_sha256(source)
    args.output_dir = experiment.root / "regions"
    alignment._run(args)
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text())
    checkpoint = Path(report["checkpoints"][-1]["path"])
    return model, checkpoint, report_path, report, kwargs, source, source_report


def test_region_restore_validates_v1_chain_before_one_final_copy(region_artifact, monkeypatch):
    model, checkpoint, _, report, kwargs, source, source_report = region_artifact
    connector = model.decoders["image"].connector
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    frozen = frozen_state_hashes(model, alignment.MODULES)
    rng = torch.get_rng_state().clone()
    copies = []
    original_load = connector.load_state_dict

    def observe(state, *args, **options):
        copies.append({name: value.clone() for name, value in state.items()})
        return original_load(state, *args, **options)

    monkeypatch.setattr(connector, "load_state_dict", observe)
    lineage = restore_alignment_connector(model, checkpoint, **kwargs)
    assert len(copies) == 1
    saved = torch.load(checkpoint, weights_only=True)
    assert all(
        torch.equal(value, saved["connector_state_dict"][alignment.PREFIX + name])
        for name, value in copies[0].items()
    )
    assert lineage["evidence_kind"] == alignment.REGIONS_FIXTURE_KIND
    assert lineage["schema_version"] == 2
    assert lineage["objective"] == alignment.REGIONS_OBJECTIVE
    assert lineage["alignment_initialization"]["evidence_kind"] == alignment.FIXTURE_KIND
    assert lineage["alignment_initialization"]["sha256"] == file_sha256(source)
    assert lineage["region_partition_audit_revalidated"]
    assert lineage["selection"] == source_report["selection"] == report["selection"]
    assert report["connector_hashes_before"] == source_report["connector_hashes_after"]
    assert (
        report["warm_connector_hashes_before_initializer"]
        == source_report["connector_hashes_before"]
    )
    assert frozen_state_hashes(model, alignment.MODULES) == frozen
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "change",
    [
        "v1_masquerade",
        "flow_masquerade",
        "objective_weight",
        "objective_formula",
        "initializer_digest",
        "initializer_report_digest",
        "initializer_kind",
        "initializer_path",
        "initializer_exposure",
        "initializer_frozen",
        "initializer_weights",
        "nested_v2",
        "warm_hash",
        "start_hash",
        "cohort",
        "region_span",
        "region_count",
        "frozen_feature",
        "balanced_loss",
        "caption_reduction",
        "nonfinite_metric",
        "optimizer_progress",
        "optimizer_moments",
        "fresh_stage",
        "optimized_loss_key",
        "data",
    ],
)
def test_region_lineage_and_objective_rejected_before_any_copy(
    region_artifact, monkeypatch, change
):
    model, checkpoint, report_path, report, kwargs, source, source_report = region_artifact
    saved = torch.load(checkpoint, weights_only=True)
    protocol = saved["protocol"]
    if change == "v1_masquerade":
        saved["schema_version"] = report["schema_version"] = 1
        saved["evidence_kind"] = report["evidence_kind"] = alignment.FIXTURE_KIND
        report["checkpoints"][-1]["evidence_kind"] = alignment.FIXTURE_KIND
    elif change == "flow_masquerade":
        saved["evidence_kind"] = report["evidence_kind"] = "fixture_only"
    elif change in {"objective_weight", "objective_formula"}:
        if change == "objective_weight":
            protocol["objective"]["region_weighting"]["prefix"] = 0.5
        else:
            protocol["objective"]["formula"] = "mean_all_tokens"
        report["objective"] = protocol["objective"]
    elif change in {"initializer_digest", "initializer_report_digest", "initializer_kind"}:
        key, value = {
            "initializer_digest": ("sha256", "f" * 64),
            "initializer_report_digest": ("report_sha256", "f" * 64),
            "initializer_kind": ("evidence_kind", alignment.REGIONS_FIXTURE_KIND),
        }[change]
        protocol["alignment_initialization"][key] = value
        report["alignment_initialization"] = protocol["alignment_initialization"]
    elif change == "initializer_path":
        protocol["settings"]["init_alignment_checkpoint"] = str(checkpoint)
        report["settings"] = protocol["settings"]
    elif change.startswith("initializer_") or change == "nested_v2":
        original = torch.load(source, weights_only=True)
        if change == "initializer_exposure":
            original["sample_exposure"]["foreign-id"] = 1
        elif change == "initializer_frozen":
            source_report["frozen_state_unchanged"] = False
        elif change == "initializer_weights":
            original["connector_state_dict"][next(iter(original["connector_state_dict"]))].add_(0.1)
        else:
            original["schema_version"] = source_report["schema_version"] = 2
        rewrite(source, source.parent / "report.json", source_report, original)
        # Admit the changed source digest to reach the recursive semantic guards.
        protocol["alignment_initialization"]["sha256"] = file_sha256(source)
        protocol["alignment_initialization"]["report_sha256"] = file_sha256(
            source.parent / "report.json"
        )
        report["alignment_initialization"] = protocol["alignment_initialization"]
    elif change == "warm_hash":
        protocol["warm_connector_hashes_before_initializer"] = {"wrong": "f" * 64}
        report["warm_connector_hashes_before_initializer"] = protocol[
            "warm_connector_hashes_before_initializer"
        ]
    elif change == "start_hash":
        report["connector_hashes_before"] = report["warm_connector_hashes_before_initializer"]
    elif change == "cohort":
        protocol["selection"]["validation"][0]["id"] = "other-heldout"
        report["selection"] = protocol["selection"]
    elif change in {"region_span", "region_count", "frozen_feature"}:
        audit = protocol["cache_audit"]["train"][0]
        if change == "region_span":
            audit["region_spans"]["prefix"] = [0, 1]
        elif change == "region_count":
            audit["region_token_counts"]["suffix"] += 1
        else:
            audit["prism_hidden_sha256"] = "f" * 64
        report["cache_audit"] = protocol["cache_audit"]
    elif change in {"balanced_loss", "caption_reduction", "nonfinite_metric"}:
        metrics = report["evaluations"][-1]["splits"]["validation"]
        if change == "balanced_loss":
            metrics["examples"][0]["region_balanced_mse"] += 0.1
        elif change == "caption_reduction":
            metrics["prefix_mse"] += 0.3
            metrics["region_balanced_mse"] += 0.1
        else:
            metrics["examples"][0]["suffix_cosine"] = float("nan")
    elif change.startswith("optimizer_"):
        values = next(iter(saved["optimizer_state_dict"]["state"].values()))
        if change == "optimizer_progress":
            values["step"] += source_report["completed_steps"]
        else:
            values["exp_avg"].fill_(float("nan"))
    elif change == "fresh_stage":
        protocol["stage_initialization"]["source_optimizer_restored"] = True
        report["stage_initialization"] = protocol["stage_initialization"]
    elif change == "optimized_loss_key":
        protocol["optimized_loss_key"] = report["optimized_loss_key"] = "content_mse"
    else:
        kwargs["data_fingerprint"] = "another-conversion"
    rewrite(checkpoint, report_path, report, saved)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    def refuse_copy(*args, **options):
        pytest.fail("No source or final tensor copy is permitted before all validations pass")

    monkeypatch.setattr(model.decoders["image"].connector, "load_state_dict", refuse_copy)
    with pytest.raises(ValueError):
        restore_alignment_connector(model, checkpoint, **kwargs)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
