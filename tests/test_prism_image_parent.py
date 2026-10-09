"""Parent diagnostic fixtures: no pretrained checkpoint or generator is loaded."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from src.decoders.image import ImageDecoder
from src.decoders.types import DecoderCondition

SCRIPT = Path(__file__).resolve().parents[1] / "tools/validate_prism_image_parent.py"
SPEC = importlib.util.spec_from_file_location("parent_diagnostic_fixture", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FixtureTokenizer:
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        assert kwargs["truncation"] is False
        ids = torch.tensor([[sum(map(ord, word)) % 13 + 1 for word in prompts[0].split()]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens, **kwargs):
        return "blue"


class FixtureBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *, inputs_embeds, attention_mask, position_ids):
        return inputs_embeds


class FixtureBackend(torch.nn.Module):
    conditioning_dim = 6
    _pipeline = None

    def generate_conditioned(self, *args, **kwargs):
        raise AssertionError("diagnostic must not load or sample the generator")


class FixtureModel(torch.nn.Module):
    def __init__(self, *, mutate=False, text_regression=False):
        super().__init__()
        self.backbone = FixtureBackbone()
        self.projector = torch.nn.Linear(3, 4)
        self.decoders = torch.nn.ModuleDict({"image": ImageDecoder(4, backend=FixtureBackend())})
        self.register_buffer("state_marker", torch.tensor(0.0))
        self.mutate = mutate
        self.text_regression = text_regression

    def predict(self, inputs, requested_outputs, native_context=None, decoder_kwargs=None):
        if requested_outputs == ["text"]:
            ids = [[4, 5]] if self.text_regression else [[5, 6]]
            return SimpleNamespace(predictions={"text": torch.tensor(ids)}, loss=None)
        if self.mutate:
            self.state_marker.add_(1)
        valid = inputs["text_attention_mask"].bool()
        hidden = self.backbone.embeddings(inputs["text"][valid]).unsqueeze(0)
        if "image" in inputs:
            image = inputs["image"][0, inputs["image_mask"][0]].mean(dim=(0, 2, 3))[None]
            hidden = hidden + self.projector(image)[:, None]
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        hidden = self.backbone(
            inputs_embeds=hidden, attention_mask=mask, position_ids=mask.long().cumsum(-1) - 1
        )
        condition = DecoderCondition(
            hidden,
            mask,
            native_context=native_context["image"],
            provenance={"fixture": True},
        )
        image = self.decoders["image"].generate_condition(condition)
        return SimpleNamespace(predictions={"image": image}, loss=None)

    def generate(self, **kwargs):
        return torch.tensor([[5, 6]])


def fixture_setup(tmp_path, **model_options):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(source)
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "fixture",
                "prompt": "What color?",
                "source_images": [source.name],
                "comparison_prompt": "What shape?",
                "expected_answers": ["blue"],
            }
        )
    )
    cases = diagnostic.load_cases(path)
    bundle = {
        "model": FixtureModel(**model_options),
        "tokenizer": FixtureTokenizer(),
        "source_transform": lambda image: (
            torch.tensor(list(image.getdata())[0]).float()[:, None, None].expand(3, 4, 4) / 255
        ),
        "provenance": {"evidence_kind": "fixture_only"},
    }
    args = argparse.Namespace(
        output_dir=tmp_path / "run", device="cpu", dtype="float32", max_new_tokens=4
    )
    return path, cases, bundle, args


def test_real_route_and_connector_boundary_without_generator(tmp_path):
    _, cases, bundle, args = fixture_setup(tmp_path)
    decoder = bundle["model"].decoders["image"]
    assert "generate_condition" not in decoder.__dict__
    report = diagnostic.run_parent_diagnostic(bundle, cases, args, fixture=True)
    assert "generate_condition" not in decoder.__dict__
    assert report["status"] == "completed"
    assert report["evidence_kind"] == "fixture_only"
    assert report["alignment_gate"] == "not_evaluated"
    assert report["p1_acceptance"] == "unproven"
    assert report["generator_loaded"] is False
    assert report["frozen_parent_unchanged"]
    row = report["cases"][0]
    assert row["hidden_shape"] == [1, 2, 4]
    assert row["connector_shape"] == [1, 2, 6]
    assert row["text"]["exact_route_match"]
    assert len(row["same_input_repeats"]) == 2
    assert all(check["all_valid_exact_match"] for check in row["same_input_repeats"])
    for name in ("text_padding_comparison", "masked_source_padding_comparison"):
        assert all(x["exact_match"] for x in row[name]["compiled_inputs"].values())
    assert row["text_padding_comparison"]["last_valid_exact_match"]
    assert row["masked_source_padding_comparison"]["all_valid_exact_match"]
    assert not row["encoder_zero_pixels_sensitivity"]["last_valid_exact_match"]
    assert row["target_fields_in_inputs"] is False
    trace = diagnostic.validation.read_trace(args.output_dir / "traces/fixture/baseline")
    assert trace["provenance"]["evidence_kind"] == "fixture_only"
    assert "compiled.inputs_embeds" in trace["tensors"]
    assert "compiled.attention_mask" in trace["tensors"]
    assert "compiled.position_ids" in trace["tensors"]
    assert not bundle["model"].backbone._forward_pre_hooks


def test_text_route_difference_is_reported_without_inventing_acceptance(tmp_path):
    _, cases, bundle, args = fixture_setup(tmp_path, text_regression=True)
    report = diagnostic.run_parent_diagnostic(bundle, cases, args, fixture=True)
    assert report["status"] == "completed"
    assert not report["cases"][0]["text"]["exact_route_match"]
    assert report["p1_acceptance"] == "unproven"


def test_frozen_state_mutation_fails_diagnostic(tmp_path):
    _, cases, bundle, args = fixture_setup(tmp_path, mutate=True)
    report = diagnostic.run_parent_diagnostic(bundle, cases, args, fixture=True)
    assert report["status"] == "failed"
    assert not report["frozen_parent_unchanged"]
    assert "state_marker" in report["changed_frozen_keys"]


def test_unverified_parent_and_target_fields_rejected(tmp_path):
    path, cases, bundle, args = fixture_setup(tmp_path)
    with pytest.raises(ValueError, match="strict complete-parent"):
        diagnostic.run_parent_diagnostic(bundle, cases, args)
    row = json.loads(path.read_text())
    row["target_image"] = "never-open-this.png"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="targets?.*forbidden"):
        diagnostic.load_cases(path)


def test_case_bound_and_token_bound_are_enforced(tmp_path):
    path, cases, bundle, args = fixture_setup(tmp_path)
    args.max_new_tokens = 33
    with pytest.raises(ValueError, match="32 generated tokens"):
        diagnostic.run_parent_diagnostic(bundle, cases, args, fixture=True)
    row = json.loads(path.read_text())
    path.write_text("\n".join(json.dumps({**row, "id": f"case-{n}"}) for n in range(9)))
    with pytest.raises(ValueError, match="one to eight"):
        diagnostic.load_cases(path)


def test_restoration_report_cannot_omit_trained_parent_tensors():
    provenance = {
        "restoration": {
            "strict_parent": True,
            "missing_parent_keys": [],
            "unexpected_keys": [],
            "loaded_key_count": 3,
            "new_connector_keys": ["decoders.image.connector.0.weight"],
        }
    }
    diagnostic._restoration_check(provenance, fixture=False)
    provenance["restoration"]["missing_parent_keys"] = ["projectors.image.weight"]
    with pytest.raises(ValueError, match="not restored exactly"):
        diagnostic._restoration_check(provenance, fixture=False)


def test_same_input_drift_is_distinct_from_compiled_padding_drift(tmp_path):
    _, cases, bundle, args = fixture_setup(tmp_path)
    counter = [0]

    def drift(module, arguments, output):
        counter[0] += 1
        return output + counter[0] * 0.01

    hook = bundle["model"].backbone.register_forward_hook(drift)
    try:
        report = diagnostic.run_parent_diagnostic(bundle, cases, args, fixture=True)
    finally:
        hook.remove()
    row = report["cases"][0]
    assert report["status"] == "completed"
    for comparison in [*row["same_input_repeats"], row["text_padding_comparison"]]:
        assert not comparison["all_valid_exact_match"]
        assert all(item["exact_match"] for item in comparison["compiled_inputs"].values())
    assert report["p1_acceptance"] == "unproven"
