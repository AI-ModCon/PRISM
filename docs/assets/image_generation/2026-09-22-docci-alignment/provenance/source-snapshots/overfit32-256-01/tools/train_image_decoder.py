"""Bounded, local-only connector training for the image output branch.

This runner does not certify P2 or scientific capability. Its fixture API exists
only for offline engineering tests; the CLI always requires real P0/P1 evidence
and measured input alignment tied to the selected local checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P1_ACCEPTANCE_CHECKS = (
    "checkpoint_reload",
    "padding_batch_invariance",
    "text_checkpoint_regression",
    "target_free_unified_transformer",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_training_evidence(
    gate_report: str | Path,
    alignment_report: str | Path,
    *,
    reference_report: str | Path,
    checkpoint: str | Path,
    model_config: str | Path,
    acceptance_report: str | Path | None = None,
) -> dict[str, Any]:
    """Reject generic pass flags, fixture reports, stale alignment, and missing files.

    Consume the manifests (or verified sibling gate-report aliases) written by
    ``tools/validate_image_decoder.py``. Alignment schema:
    ``evidence_kind=measured_input_alignment``, ``status=passed``, checkpoint,
    model-config, tokenizer and source-processor SHA256 fields, and a nonempty
    ``metrics`` object.
    """
    checkpoint = Path(checkpoint).resolve(strict=True)
    model_config = Path(model_config).resolve(strict=True)
    if not checkpoint.is_file() or not model_config.is_file():
        raise ValueError("Checkpoint and model configuration must be local files")
    reference, reference_hash = _validated_gate_report(reference_report, "P0", "reference")
    parity, parity_hash = _validated_gate_report(gate_report, "P1", "full_pipeline")
    alignment = _read_json(alignment_report)
    if parity.get("numerical_parity") is not True:
        raise ValueError("P1 must establish full-pipeline numerical parity")
    reference_identity = reference["identity"]
    parity_identity = parity["identity"]
    for field in (
        "checkpoint_revision",
        "upstream_revision",
        "case_manifest_sha256",
        "reference_checkpoint_sha256",
    ):
        if not reference_identity.get(field) or reference_identity.get(
            field
        ) != parity_identity.get(field):
            raise ValueError(f"P0/P1 checkpoint, code or case identity mismatch: {field}")
    if parity_identity.get("reference_manifest_sha256") != reference_hash:
        raise ValueError("P1 must reference the exact passed P0 manifest SHA256")
    reference_checkpoint_hash = parity_identity.get("reference_checkpoint_sha256", "")
    if (
        not isinstance(reference_checkpoint_hash, str)
        or len(reference_checkpoint_hash) != 64
        or any(c not in "0123456789abcdef" for c in reference_checkpoint_hash)
    ):
        raise ValueError("P0/P1 report must identify the reference checkpoint by SHA256")
    if (
        alignment.get("evidence_kind") != "measured_input_alignment"
        or alignment.get("status") != "passed"
        or not isinstance(alignment.get("metrics"), dict)
        or not alignment["metrics"]
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in alignment["metrics"].values()
        )
    ):
        raise ValueError("A passed, measured input-alignment report with metrics is required")
    checkpoint_hash = sha256_file(checkpoint)
    config_hash = sha256_file(model_config)
    if alignment.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Input-alignment checkpoint SHA256 does not match the selected checkpoint")
    if alignment.get("model_config_sha256") != config_hash:
        raise ValueError("Input-alignment model configuration SHA256 does not match")
    for field in ("tokenizer_sha256", "source_processor_sha256"):
        fingerprint = alignment.get(field)
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError(f"Input-alignment report must bind preprocessing identity: {field}")
    acceptance = _validate_p1_acceptance(
        acceptance_report,
        parity_manifest_sha256=parity_hash,
        parent_checkpoint_sha256=checkpoint_hash,
        model_config_sha256=config_hash,
        reference_checkpoint_sha256=reference_checkpoint_hash,
    )
    return {
        "evidence_kind": "real_checkpoint",
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": checkpoint_hash,
        "model_config": str(model_config),
        "model_config_sha256": config_hash,
        "tokenizer_sha256": alignment["tokenizer_sha256"],
        "source_processor_sha256": alignment["source_processor_sha256"],
        "reference_checkpoint_sha256": reference_checkpoint_hash,
        "reference_report": str(Path(reference_report).resolve()),
        "reference_report_sha256": sha256_file(reference_report),
        "gate_report": str(Path(gate_report).resolve()),
        "gate_report_sha256": sha256_file(gate_report),
        "alignment_report": str(Path(alignment_report).resolve()),
        "alignment_report_sha256": sha256_file(alignment_report),
        **acceptance,
    }


def _validate_p1_acceptance(report_path: str | Path | None, **identity: str) -> dict[str, Any]:
    """Validate real regression evidence, not just an adapter-parity pass flag.

    Each named check has status/evidence_kind, finite numeric metrics, and one or
    more existing artifacts with their SHA256. Paths are relative to the report.
    This verifies the evidence contract and bytes; it does not invent test results.
    """
    if report_path is None:
        raise ValueError("A separate real-checkpoint P1 acceptance report is required")
    path = Path(report_path).resolve(strict=True)
    report = _read_json(path)
    if (
        report.get("schema_version") != 1
        or report.get("stage") != "P1_acceptance"
        or report.get("status") != "passed"
        or report.get("evidence_kind") != "real_checkpoint"
        or report.get("fixture", False) is not False
        or report.get("smoke", False) is not False
    ):
        raise ValueError(
            "P1 acceptance requires a passed real-checkpoint report, not fixture/smoke evidence"
        )
    for field, expected in identity.items():
        if report.get(field) != expected:
            raise ValueError(f"P1 acceptance identity mismatch: {field}")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("P1 acceptance requires named real-checkpoint checks")
    artifact_hashes = {}
    for name in P1_ACCEPTANCE_CHECKS:
        check = checks.get(name)
        if (
            not isinstance(check, dict)
            or check.get("status") != "passed"
            or check.get("evidence_kind") != "real_checkpoint"
            or check.get("fixture", False) is not False
            or check.get("smoke", False) is not False
        ):
            raise ValueError(
                f"P1 acceptance check requires passed real-checkpoint evidence: {name}"
            )
        metrics = check.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in metrics.values()
            )
        ):
            raise ValueError(f"P1 acceptance check requires finite numeric metrics: {name}")
        artifacts = check.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"P1 acceptance check requires hashed artifacts: {name}")
        for artifact in artifacts:
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("path"), str)
                or not artifact["path"]
            ):
                raise ValueError(f"P1 acceptance artifact path missing: {name}")
            artifact_path = (path.parent / artifact["path"]).resolve()
            if not artifact_path.is_file():
                raise ValueError(f"P1 acceptance artifact missing: {name}: {artifact_path}")
            actual = sha256_file(artifact_path)
            if artifact.get("sha256") != actual:
                raise ValueError(f"P1 acceptance artifact SHA256 mismatch: {name}: {artifact_path}")
            if artifact_path.suffix.lower() == ".json":
                with artifact_path.open(encoding="utf-8") as stream:
                    artifact_metadata = json.load(stream)
                if isinstance(artifact_metadata, dict) and (
                    artifact_metadata.get("evidence_kind") in ("fixture", "fixture_only", "mock")
                    or artifact_metadata.get("fixture", False) is not False
                    or artifact_metadata.get("smoke", False) is not False
                    or (
                        "real_checkpoint" in artifact_metadata
                        and artifact_metadata["real_checkpoint"] is not True
                    )
                ):
                    raise ValueError(f"P1 acceptance artifact is fixture/smoke evidence: {name}")
            artifact_hashes[str(artifact_path)] = actual
    return {
        "acceptance_report": str(path),
        "acceptance_report_sha256": sha256_file(path),
        "p1_acceptance": "passed",
        "acceptance_artifact_sha256": artifact_hashes,
    }


def _validated_gate_report(path: str | Path, stage: str, mode: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    report = _read_json(path)
    manifest_path = path
    if "manifest_sha256" in report:
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file() or sha256_file(manifest_path) != report["manifest_sha256"]:
            raise ValueError(f"{stage} gate alias does not match its complete manifest")
        manifest = _read_json(manifest_path)
        for field in (
            "schema_version",
            "stage",
            "mode",
            "status",
            "real_checkpoint",
            "numerical_parity",
            "validation_scope",
            "p1_acceptance",
            "identity",
            "suite",
        ):
            if report.get(field) != manifest.get(field):
                raise ValueError(f"{stage} gate alias disagrees with its manifest: {field}")
        report = manifest
    if report.get("real_checkpoint") is not True:
        raise ValueError(
            "P0/P1 require real_checkpoint evidence; fixtures cannot authorize training"
        )
    if (
        report.get("schema_version") != 1
        or report.get("stage") != stage
        or report.get("mode") != mode
    ):
        raise ValueError(f"{stage} requires a version-1 {mode} report")
    if report.get("status") != "passed":
        raise ValueError(f"{stage} must have passed before connector training")
    suite = report.get("suite", {})
    records = report.get("records", [])
    if (
        suite.get("complete_prespecified_size") is not True
        or suite.get("output_count") != 90
        or suite.get("case_count") != 30
        or not isinstance(records, list)
        or len(records) != 90
        or any(not isinstance(row, dict) or row.get("status") != "passed" for row in records)
    ):
        raise ValueError(f"{stage} requires the complete 30-case, 90-output passed reference suite")
    if len({(row.get("case_id"), row.get("seed")) for row in records}) != 90:
        raise ValueError(f"{stage} contains duplicate or missing case/seed records")
    if not isinstance(report.get("identity"), dict):
        raise ValueError(f"{stage} must identify checkpoint, source code and cases")
    return report, sha256_file(manifest_path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        state["torch_xpu"] = torch.xpu.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    try:
        import numpy as np

        algorithm, keys, pos, has_gauss, cached_gauss = np.random.get_state()
        state["numpy"] = (algorithm, keys.tolist(), pos, has_gauss, cached_gauss)
    except ImportError:
        pass
    return state


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _inside(name: str, modules: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in modules)


def _configure_connector_only(model: nn.Module, modules: Sequence[str]) -> dict[str, nn.Parameter]:
    # A P2 artifact must contain the complete connector so strict connector
    # reload is possible; partial-connector experiments need a separate contract.
    if list(modules) != ["decoders.image.connector"]:
        raise ValueError(
            "Only the complete image output connector decoders.image.connector may be trained"
        )
    named_modules = dict(model.named_modules())
    for name in modules:
        if name not in named_modules:
            raise ValueError(f"Unknown connector module: {name}")
        if not list(named_modules[name].parameters()):
            raise ValueError(f"Connector module has no parameters: {name}")
    selected_ids = {
        id(parameter) for name in modules for parameter in named_modules[name].parameters()
    }
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in selected_ids and not _inside(name, modules):
            raise ValueError(f"Connector parameter is shared with a frozen module: {name}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
        parameter.grad = None
    for name in modules:
        named_modules[name].train()
    selected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not selected:
        raise ValueError("No output connector parameters selected")
    return selected


def _tensor_hash(tensor: torch.Tensor) -> str:
    # Byte view preserves BF16 and integer buffers without NumPy dtype conversion.
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(raw)
    return digest.hexdigest()


def frozen_state_hashes(model: nn.Module, connector_modules: Sequence[str]) -> dict[str, str]:
    """Hash frozen parameters *and buffers*, including non-persistent buffers."""
    values = list(model.named_parameters()) + list(model.named_buffers())
    return {
        name: _tensor_hash(value) for name, value in values if not _inside(name, connector_modules)
    }


def run_connector_training(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    max_steps: int,
    connector_modules: Sequence[str] = ("decoders.image.connector",),
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    decoder_dtype: str = "bfloat16",
    max_grad_norm: float = 1.0,
    provenance: Mapping[str, Any] | None = None,
    fixture: bool = False,
    decoder_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a bounded number of steps and save an explicitly scoped artifact.

    ``fixture=True`` is available only to injected programmatic tests. It is
    always recorded and never constitutes model, P0/P1, or P2 validation.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be an explicit positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if decoder_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("decoder_dtype must be float32, float16 or bfloat16")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; refusing to overwrite a run")
    provenance = dict(provenance or {})
    if not fixture:
        required = (
            "reference_report",
            "gate_report",
            "alignment_report",
            "acceptance_report",
            "parent_checkpoint",
            "model_config",
        )
        if any(not provenance.get(key) for key in required):
            raise ValueError("Real training requires validated P0/P1 and alignment evidence")
        validated = validate_training_evidence(
            provenance["gate_report"],
            provenance["alignment_report"],
            reference_report=provenance["reference_report"],
            checkpoint=provenance["parent_checkpoint"],
            model_config=provenance["model_config"],
            acceptance_report=provenance["acceptance_report"],
        )
        if any(provenance.get(key) != value for key, value in validated.items()):
            raise ValueError("Training provenance changed after prerequisite validation")
        # Materialize lazy generator components before selecting parameters and
        # hashing frozen state. Never trust an identity copied out of a report.
        backend = model.decoders["image"].backend
        backend.to(device=device, dtype=getattr(torch, decoder_dtype))
        backend.ensure_loaded()
        identity = backend.checkpoint_manifest()
        if identity.get("manifest_sha256") != validated["reference_checkpoint_sha256"]:
            raise ValueError("Loaded reference checkpoint does not match the passed P0/P1 evidence")
    _seed_everything(seed)
    target_device = torch.device(device)
    model.to(target_device)
    trainable = _configure_connector_only(model, connector_modules)
    before = frozen_state_hashes(model, connector_modules)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=learning_rate, weight_decay=0.0)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "evidence_kind": "fixture_only" if fixture else "real_checkpoint_training",
        "status": "running",
        "p2_gate": "not_evaluated",
        "seed": seed,
        "max_steps": max_steps,
        "device": str(target_device),
        "decoder_dtype": decoder_dtype if not fixture else "fixture_model_dtypes",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "runner_sha256": sha256_file(__file__),
        "connector_modules": list(connector_modules),
        "trainable_parameters": {name: list(p.shape) for name, p in trainable.items()},
        "trainable_parameter_dtypes": {name: str(p.dtype) for name, p in trainable.items()},
        "trainable_parameter_count": sum(p.numel() for p in trainable.values()),
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 0.0},
        "max_grad_norm": max_grad_norm,
        "provenance": provenance,
        "frozen_hashes_before": before,
    }
    _json_write(out / "run.json", run)
    data_generator = getattr(batches, "generator", None)
    initial_rng = _rng_state()
    if isinstance(data_generator, torch.Generator):
        initial_rng["data_order"] = data_generator.get_state()
    torch.save(initial_rng, out / "initial_rng.pt")
    nonzero_names: set[str] = set()
    iterator = iter(batches)
    completed_steps = 0
    try:
        with (out / "steps.jsonl").open("x", encoding="utf-8") as log:
            for step in range(1, max_steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            "Training batches are empty or cannot be restarted"
                        ) from exc
                batch = _to_device(batch, target_device)
                if set(batch.get("targets", {})) != {"image"}:
                    raise ValueError("Image-only training requires exactly targets['image']")
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_outputs(
                    batch["inputs"],
                    targets=batch["targets"],
                    requested_outputs=["image"],
                    output_specs=batch.get("output_specs", {}),
                    native_context=batch.get("native_context", {}),
                    decoder_kwargs=dict(decoder_kwargs or {}),
                )
                if any(
                    p.requires_grad and name not in trainable
                    for name, p in model.named_parameters()
                ):
                    raise RuntimeError(
                        "Forward registered trainable parameters outside the output connector"
                    )
                losses = result.losses
                if "image" not in losses:
                    raise ValueError("DecoderResult must expose a named image training loss")
                for name, loss_value in losses.items():
                    if not torch.is_tensor(loss_value) or loss_value.numel() != 1:
                        raise ValueError(f"Named loss must be a scalar tensor: {name}")
                    if not torch.isfinite(loss_value).all():
                        raise FloatingPointError(f"Nonfinite training loss: {name}")
                loss = losses["image"]
                if not loss.requires_grad:
                    raise RuntimeError("Image loss is detached from the output connector")
                loss.backward()
                norms = {}
                for name, parameter in trainable.items():
                    if parameter.grad is None:
                        raise RuntimeError(f"Output connector received no gradient: {name}")
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Nonfinite connector gradient: {name}")
                    norm = float(parameter.grad.detach().float().norm())
                    norms[name] = norm
                    if norm > 0:
                        nonzero_names.add(name)
                if not any(value > 0 for value in norms.values()):
                    raise RuntimeError("All output connector gradients are zero")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in trainable.values()):
                    raise FloatingPointError("Optimizer produced nonfinite connector parameters")
                completed_steps = step
                entry = {
                    "step": step,
                    "losses": {name: float(value.detach()) for name, value in losses.items()},
                    "gradient_norm_before_clip": float(grad_norm),
                    "parameter_gradient_norms": norms,
                    "metadata": batch.get("metadata", {}),
                }
                log.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                log.flush()
        never_nonzero = set(trainable) - nonzero_names
        if never_nonzero:
            raise RuntimeError(
                f"Connector parameters never had nonzero gradients: {sorted(never_nonzero)}"
            )
        after = frozen_state_hashes(model, connector_modules)
        if before != after:
            changed = sorted(
                key for key in set(before) | set(after) if before.get(key) != after.get(key)
            )
            raise RuntimeError(f"Frozen parameters or buffers changed: {changed}")
        state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if _inside(name, connector_modules)
        }
        checkpoint_path = out / "connector.pt"
        final_rng = _rng_state()
        if isinstance(data_generator, torch.Generator):
            final_rng["data_order"] = data_generator.get_state()
        torch.save(
            {
                "schema_version": 1,
                "evidence_kind": run["evidence_kind"],
                "step": completed_steps,
                "connector_modules": list(connector_modules),
                "connector_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": final_rng,
                "provenance": provenance,
                "frozen_hashes": before,
            },
            checkpoint_path,
        )
        run.update(
            status="completed",
            frozen_state_unchanged=True,
            connector_checkpoint=str(checkpoint_path.resolve()),
            connector_checkpoint_sha256=sha256_file(checkpoint_path),
        )
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        after = frozen_state_hashes(model, connector_modules)
        run.update(
            completed_steps=completed_steps,
            duration_seconds=time.monotonic() - started,
            frozen_hashes_after=after,
            frozen_state_unchanged=before == after,
        )
        _json_write(out / "run.json", run)
    return run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--source-processor", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--alignment-report", required=True, type=Path)
    parser.add_argument("--acceptance-report", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--connector-module", action="append", default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--decoder-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--bundle-factory",
        default="src.decoders.loading:load_image_training_bundle",
        help="Local module:function loader; does not bypass evidence gates",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("Steps, batch size, and image dimensions must be positive")
    evidence = validate_training_evidence(
        args.gate_report,
        args.alignment_report,
        reference_report=args.reference_report,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        acceptance_report=args.acceptance_report,
    )
    # Force offline before importing Transformers or loading any components.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    for name in ("tokenizer", "source_processor", "manifest"):
        getattr(args, name).resolve(strict=True)
    _seed_everything(args.seed)
    module_name, separator, function_name = args.bundle_factory.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("bundle-factory must be module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    bundle = factory(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        source_processor=args.source_processor,
    )
    provenance = dict(bundle["provenance"])
    if provenance.get("parent_checkpoint_sha256") != evidence["parent_checkpoint_sha256"]:
        raise ValueError("Loaded PRISM checkpoint does not match the input-alignment evidence")
    if provenance.get("model_config_sha256") != evidence["model_config_sha256"]:
        raise ValueError("Loaded model configuration does not match the input-alignment evidence")
    for field in ("tokenizer_sha256", "source_processor_sha256"):
        if provenance.get(field) != evidence[field]:
            raise ValueError(
                f"Loaded preprocessing does not match input-alignment evidence: {field}"
            )
    provenance.update(evidence)
    provenance["manifest_sha256"] = sha256_file(args.manifest)
    provenance["manifest"] = str(args.manifest.resolve())
    provenance["bundle_factory"] = args.bundle_factory
    from src.data.image_generation import ImageGenerationCollator, ImageGenerationDataset

    dataset = ImageGenerationDataset(
        args.manifest,
        source_transform=bundle["source_transform"],
        target_size=(args.height, args.width),
        split="train",
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ImageGenerationCollator(bundle["tokenizer"]),
        generator=generator,
    )
    provenance["sample_count"] = len(dataset)
    provenance["data_fingerprint"] = dataset.data_fingerprint
    provenance["source_and_target_image_count"] = len(dataset.image_fingerprints)
    provenance["data_order_seed"] = args.seed
    run = run_connector_training(
        bundle["model"],
        loader,
        output_dir=args.output_dir,
        max_steps=args.steps,
        connector_modules=args.connector_module or ["decoders.image.connector"],
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        decoder_dtype=args.decoder_dtype,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                key: run[key]
                for key in ("status", "completed_steps", "p2_gate", "connector_checkpoint")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
