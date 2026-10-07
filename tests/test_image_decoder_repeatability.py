"""Diagnostic orchestration fixtures; no pretrained model or accelerator runs."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "tools/diagnose_image_decoder_repeatability.py"
SPEC = importlib.util.spec_from_file_location("repeatability_fixture", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FixtureBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("placement", torch.empty(0))
        self.loads = 0

    def ensure_loaded(self):
        self.loads += 1
        self.eval()

    def provenance(self):
        return {"kernel_policy": {"policy": "fixture_only"}}

    def checkpoint_manifest(self):
        return {"manifest_sha256": "f" * 64, "evidence_kind": "fixture_only"}


def setup_fixture(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    trace = {
        "tensors": {
            name: np.zeros((1,), dtype=np.float32)
            for name in diagnostic.validation.FULL_PIPELINE_BOUNDARIES
        },
        "discrete": {"fixture": True},
        "provenance": {"real_checkpoint": False, "evidence_kind": "fixture_only"},
    }
    diagnostic.validation.archive_trace(original / "comparisons" / "fixture" / "42", trace)
    (original / "manifest.json").write_text(
        json.dumps({"real_checkpoint": False, "mode": "reference"})
    )
    cases = tmp_path / "case.jsonl"
    cases.write_text(
        json.dumps(
            {
                "case_id": "fixture",
                "task": "text_to_image",
                "prompt": "fixture",
                "reference_paths": [],
                "group": "fixture",
                "split": "smoke",
                "seeds": [42],
                "width": 16,
                "height": 16,
            }
        )
    )
    case = diagnostic.validation.load_cases(cases, smoke=True)[0]
    args = argparse.Namespace(
        output_dir=tmp_path / "diagnostic",
        reference_dir=original,
        prior_run=None,
        cases=cases,
        steps=2,
        revision="a" * 40,
        upstream_revision="b" * 40,
        device="cpu",
        dtype="float32",
        text_guidance_scale=5.0,
        image_guidance_scale=2.0,
        negative_prompt="",
    )
    return trace, case, args


def runner(trace, *, drift=False, fail_at=None):
    calls = []

    def execute(backend, case, seed, args, saved):
        calls.append((args.reference, seed, saved["tensors"]["latents.initial"].copy()))
        if len(calls) == fail_at:
            raise RuntimeError("fixture execution failure")
        result = copy.deepcopy(trace)
        if drift and not args.reference:
            result["tensors"]["condition.positive"] += 1
        return result, Image.new("RGB", (16, 16))

    return execute, calls


def test_one_load_four_ordered_calls_and_exact_adapter_difference(tmp_path):
    trace, case, args = setup_fixture(tmp_path)
    execute, calls = runner(trace, drift=True)
    backend = FixtureBackend()
    result = diagnostic.run_diagnostic(backend, case, args, fixture=True, run_case=execute)
    assert backend.loads == 1
    assert [call[0] for call in calls] == [True, True, False, True]
    assert all(
        call[1] == 42 and np.array_equal(call[2], trace["tensors"]["latents.initial"])
        for call in calls
    )
    assert (
        result["status"] == "completed"
    )  # A numerical difference is a recorded diagnostic result.
    assert result["comparisons"]["native_repeat"]["passed"]
    assert not result["comparisons"]["adapter_vs_native"]["passed"]
    assert result["comparisons"]["native_after_adapter"]["passed"]
    assert result["evidence_kind"] == "fixture_only"
    assert result["p0_p1_acceptance"] == "not_evaluated"
    assert result["tolerances"] == {}
    assert any("wrapper, call-order" in finding for finding in result["findings"])
    for name in diagnostic.CALLS:
        assert (args.output_dir / f"{name}.png").is_file()
        restored = diagnostic.validation.read_trace(args.output_dir / "traces" / name)
        assert restored["provenance"]["real_checkpoint"] is False
        settings = json.loads((args.output_dir / f"{name}-settings-before.json").read_text())
        assert settings["module_training_flags"] == {"": False}
        assert "deterministic_algorithms" in settings


def test_prior_process_comparison_and_runtime_identity_binding(tmp_path):
    trace, case, args = setup_fixture(tmp_path)
    execute, _ = runner(trace)
    diagnostic.run_diagnostic(FixtureBackend(), case, args, fixture=True, run_case=execute)
    first = args.output_dir
    args.output_dir = tmp_path / "second"
    args.prior_run = first
    result = diagnostic.run_diagnostic(FixtureBackend(), case, args, fixture=True, run_case=execute)
    assert result["status"] == "completed"
    assert result["comparisons"]["prior_process"]["passed"]
    args.output_dir = tmp_path / "wrong-sampling"
    args.steps = 3
    result = diagnostic.run_diagnostic(FixtureBackend(), case, args, fixture=True, run_case=execute)
    assert result["status"] == "failed"
    assert "identities differ" in result["errors"][0]
    assert not result["calls"]


def test_execution_failure_recorded_without_hiding_other_attempts(tmp_path):
    trace, case, args = setup_fixture(tmp_path)
    execute, calls = runner(trace, fail_at=2)
    result = diagnostic.run_diagnostic(FixtureBackend(), case, args, fixture=True, run_case=execute)
    assert result["status"] == "failed"
    assert len(calls) == 4
    assert result["calls"][1]["status"] == "failed"
    assert "fixture execution failure" in result["calls"][1]["error"]
    assert result["calls"][3]["status"] == "completed"


def test_nonfixture_injection_and_excess_budget_rejected(tmp_path):
    trace, case, args = setup_fixture(tmp_path)
    execute, _ = runner(trace)
    with pytest.raises(ValueError, match="only for explicitly labelled fixtures"):
        diagnostic.run_diagnostic(FixtureBackend(), case, args, run_case=execute)
    args.steps = 5
    with pytest.raises(ValueError, match="one to four steps"):
        diagnostic.run_diagnostic(FixtureBackend(), case, args, fixture=True, run_case=execute)


def test_cli_blocks_before_model_import_on_missing_assets_and_bad_budget(tmp_path):
    _, _, args = setup_fixture(tmp_path)
    options = {
        "--checkpoint": tmp_path / "missing-checkpoint",
        "--upstream": tmp_path / "missing-source",
        "--cases": args.cases,
        "--reference-dir": args.reference_dir,
        "--device": "xpu",
        "--dtype": "bfloat16",
        "--output-dir": tmp_path / "blocked",
    }
    argv = [part for key, value in options.items() for part in (key, str(value))]
    assert diagnostic.main(argv) == 2
    report = json.loads((tmp_path / "blocked" / "manifest.json").read_text())
    assert report["status"] == "blocked"
    assert report["p0_p1_acceptance"] == "not_evaluated"
    options["--output-dir"] = tmp_path / "too-long"
    argv = [part for key, value in options.items() for part in (key, str(value))]
    assert diagnostic.main([*argv, "--steps", "5"]) == 2


def test_deterministic_math_policy_precedes_load_and_restores_settings(tmp_path):
    trace, case, args = setup_fixture(tmp_path)
    args.deterministic = True
    args.attention_backend = "math"
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )

    class CheckedBackend(FixtureBackend):
        def ensure_loaded(self):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
            super().ensure_loaded()

    execute, _ = runner(trace)
    result = diagnostic.run_diagnostic(CheckedBackend(), case, args, fixture=True, run_case=execute)
    assert result["status"] == "completed"
    assert result["identity"]["numerical_policy"] == {
        "deterministic_algorithms": True,
        "attention_backend": "math",
    }
    assert before == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


def test_precision_policy_changes_are_exploratory_and_runtime_checks_remain_strict():
    runtime = {"device_type": "xpu", "device_name": "fixture", "torch_version": "fixture"}
    provenance = {"kernel_policy": {"policy": "fixture"}}
    reference = {"identity": {**runtime, **provenance, "dtype": "bfloat16"}}
    args = argparse.Namespace(dtype="float32", deterministic=True, attention_backend="math")
    with pytest.raises(diagnostic.validation.ValidationBlocked, match="exploratory"):
        diagnostic.bind_diagnostic_identity({"identity": {}}, reference, provenance, runtime, args)
    args.allow_precision_policy_comparison = True
    manifest = {"identity": {}}
    diagnostic.bind_diagnostic_identity(manifest, reference, provenance, runtime, args)
    assert manifest["reference_comparison_scope"] == "precision_policy_exploratory"
    assert manifest["reference_dtype"] == "bfloat16"
    runtime["torch_version"] = "different"
    with pytest.raises(diagnostic.validation.ValidationBlocked, match="torch_version"):
        diagnostic.bind_diagnostic_identity({"identity": {}}, reference, provenance, runtime, args)


def test_same_dtype_with_changed_attention_policy_is_also_exploratory():
    runtime = {"device_type": "xpu", "device_name": "fixture", "torch_version": "fixture"}
    provenance = {"kernel_policy": {"policy": "fixture"}}
    reference = {"identity": {**runtime, **provenance, "dtype": "bfloat16"}}
    args = argparse.Namespace(
        dtype="bfloat16",
        deterministic=True,
        attention_backend="math",
        allow_precision_policy_comparison=True,
    )
    manifest = {"identity": {}}
    diagnostic.bind_diagnostic_identity(manifest, reference, provenance, runtime, args)
    assert manifest["reference_comparison_scope"] == "precision_policy_exploratory"
