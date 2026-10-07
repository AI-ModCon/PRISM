"""CLI error/report tests only; no models, assets, or downloads are used."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "tools/validate_image_decoder.py"


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-B", str(CLI), *map(str, args)],
        text=True,
        capture_output=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )


def test_missing_preflight_assets_have_nonzero_exit_and_report(tmp_path):
    root = tmp_path / "run"
    result = invoke("--preflight", "--output-dir", root)
    assert result.returncode == 2, result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "blocked"
    assert manifest["preflight"]["model_loaded"] is False
    assert manifest["real_checkpoint"] is False
    assert manifest["numerical_parity"] is False
    assert (root / "report.md").exists()
    assert (root / "gate_report.json").exists()


def test_output_directory_never_overwritten(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "manifest.json"
    marker.write_text("preserve")
    result = invoke("--preflight", "--output-dir", root)
    assert result.returncode == 2
    assert marker.read_text() == "preserve"


def test_generation_without_cases_blocks_before_importing_models(tmp_path):
    root = tmp_path / "reference"
    result = invoke("--reference", "--output-dir", root)
    assert result.returncode == 2
    manifest = json.loads((root / "manifest.json").read_text())
    assert "--cases is required" in manifest["errors"][0]
    assert not manifest["records"]
    assert not manifest["real_checkpoint"]


def test_parity_report_explicitly_leaves_broader_p1_acceptance_unproven(tmp_path):
    root = tmp_path / "parity"
    result = invoke("--parity", "--output-dir", root)
    assert result.returncode == 2
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["validation_scope"] == "reference_adapter_numerical_parity"
    assert manifest["p1_acceptance"]["status"] == "unproven"
    assert (
        "target_free_unified_transformer" in manifest["p1_acceptance"]["required_companion_checks"]
    )
    gate = json.loads((root / "gate_report.json").read_text())
    assert gate["p1_acceptance"] == manifest["p1_acceptance"]
    assert "P1 acceptance: **unproven**" in (root / "report.md").read_text()


def test_smoke_report_can_never_authorize_qualification(tmp_path):
    spec = importlib.util.spec_from_file_location("validate_image_cli_fixture", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    manifest = {
        "status": "completed",
        "smoke": True,
        "stage": "P1",
        "mode": "full_pipeline",
        "numerical_parity": False,
        "records": [],
        "errors": [],
        "claim": "fixture only",
    }
    cli._report(tmp_path, manifest)
    gate = json.loads((tmp_path / "gate_report.json").read_text())
    assert gate["status"] == "blocked"
    assert gate["smoke"] is True
    assert gate["numerical_parity"] is False


def test_seeded_adapter_does_not_consume_source_vae_rng():
    """Small fake backend: proves RNG plumbing, never checkpoint acceptance."""
    import argparse

    import torch
    from PIL import Image

    spec = importlib.util.spec_from_file_location("validate_image_cli_rng_fixture", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class FixtureBackend(torch.nn.Module):
        conditioning_dim = 2

        def provenance(self):
            return {"backend": "fixture"}

        def generate_reference(self, native_context, **options):
            source = torch.randn(2)  # Mimics the upstream global source-VAE RNG.
            latent = options.get("latents")
            if latent is None:
                latent = torch.randn(1, 2, generator=options["generator"])
            self.last_trace = {"latents.initial": latent.clone(), "reference.0.0": source}
            return [Image.new("RGB", (16, 16))]

    case = cli.validation.ImageValidationCase(
        "fixture", "text_to_image", "cube", (), "subject", "test", (1, 2, 3), width=16, height=16
    )
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        reference=True,
        parity_mode="full_pipeline",
        steps=2,
        text_guidance_scale=5.0,
        image_guidance_scale=2.0,
        negative_prompt="",
    )
    before = torch.random.get_rng_state().clone()
    reference, _ = cli._run_seeded_backend(FixtureBackend(), case, 1, args)
    args.reference = False
    candidate, _ = cli._run_seeded_backend(FixtureBackend(), case, 1, args, reference)
    assert torch.equal(reference["tensors"]["reference.0.0"], candidate["tensors"]["reference.0.0"])
    assert torch.equal(before, torch.random.get_rng_state())
    assert (
        candidate["provenance"]["execution_path"] == "prism_ImageDecoder_reference_condition_route"
    )


def test_xpu_runtime_and_peak_memory_are_recorded_with_mock_device():
    """Metadata plumbing fixture; no XPU availability/performance claim."""
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("validate_image_cli_xpu_fixture", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    reset = []
    fake_xpu = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(total_memory=64000),
        get_device_name=lambda index: "Fixture Intel GPU",
        device_count=lambda: 2,
        reset_peak_memory_stats=lambda index: reset.append(index),
        max_memory_allocated=lambda index: 1234,
        max_memory_reserved=lambda index: 2345,
    )
    fake_torch = SimpleNamespace(
        __version__="2.10.0+xpu-fixture",
        version=SimpleNamespace(git_version="fixture-commit", xpu="fixture-runtime"),
        xpu=fake_xpu,
    )
    runtime = cli._runtime_metadata(fake_torch, "xpu")
    assert runtime["device_type"] == "xpu"
    assert runtime["device_index"] == 1
    assert runtime["device_name"] == "Fixture Intel GPU"
    assert runtime["torch_version"] == "2.10.0+xpu-fixture"
    assert runtime["total_memory_bytes"] == 64000
    assert reset == [1]
    assert cli._runtime_peak_memory(fake_torch, runtime) == {
        "allocated_bytes": 1234,
        "reserved_bytes": 2345,
    }


def test_parity_rejects_kernel_device_and_runtime_mismatches():
    import copy

    import pytest

    spec = importlib.util.spec_from_file_location("validate_image_cli_policy_fixture", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    provenance = {
        "kernel_policy": {"policy": "upstream_torch_fallback", "cross_kernel_parity": "not_run"}
    }
    runtime = {
        "device_type": "xpu",
        "device_name": "Fixture Intel GPU",
        "torch_version": "fixture-2.10",
    }
    reference = {"identity": {}}
    cli._bind_runtime_identity(reference, None, provenance, runtime, "bfloat16")
    candidate = {"identity": {}}
    cli._bind_runtime_identity(candidate, reference, provenance, runtime, "bfloat16")
    assert candidate == reference
    for field in ("kernel_policy", "device_type", "device_name", "torch_version", "dtype"):
        mismatched = copy.deepcopy(reference)
        mismatched["identity"][field] = "different fixture"
        with pytest.raises(cli.validation.ValidationBlocked, match=field):
            cli._bind_runtime_identity(
                {"identity": {}}, mismatched, provenance, runtime, "bfloat16"
            )
    with pytest.raises(cli.validation.ValidationBlocked, match="kernel policy"):
        cli._bind_runtime_identity({"identity": {}}, None, {}, runtime, "bfloat16")
