#!/usr/bin/env python3
"""Run pinned, offline OmniGen2 reference/parity validation into a new directory.

Preflight checks metadata and file hashes without loading models. Model generation is never started by --preflight;
other modes fail closed when prerequisites are missing. No mode downloads models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Avoid importing src.eval.__init__ (and torch) during file-integrity preflight.
ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prism_image_validation", ROOT / "src/eval/image_generation.py"
)
validation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validation
_SPEC.loader.exec_module(validation)

DEFAULT_REVISION = "df5dca8a981d74e6c3af214c145f5c735fe72367"
DEFAULT_UPSTREAM_REVISION = "18e6f9d5271b517fcb32e999f10df943ae9b8f20"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--parity", action="store_true")
    mode.add_argument("--evaluate", metavar="EVALUATOR_JSON")
    result.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New run directory; existing directories are never overwritten",
    )
    result.add_argument(
        "--checkpoint", type=Path, help="Existing local pinned HF snapshot directory"
    )
    result.add_argument("--revision", default=DEFAULT_REVISION)
    result.add_argument("--upstream", type=Path, help="Clean local checkout of official OmniGen2")
    result.add_argument("--upstream-revision", default=DEFAULT_UPSTREAM_REVISION)
    result.add_argument(
        "--cases", type=Path, help="Fixed case JSONL, with all tasks and three seeds/case"
    )
    result.add_argument("--data-root", type=Path)
    result.add_argument("--reference-dir", type=Path)
    result.add_argument(
        "--parity-mode", choices=("replay", "full_pipeline"), default="full_pipeline"
    )
    result.add_argument(
        "--tolerances", type=Path, help="Preregistered boundary tolerances JSON; default exact"
    )
    result.add_argument("--protocol", type=Path, help="Preregistered qualification protocol JSON")
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    result.add_argument("--steps", type=int, default=50)
    result.add_argument("--text-guidance-scale", type=float, default=5.0)
    result.add_argument("--image-guidance-scale", type=float, default=2.0)
    result.add_argument("--negative-prompt", default="")
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a small real run; never pass P0/P1 qualification (one to three seeds/case)",
    )
    return result


def _json(path):
    return json.loads(Path(path).read_text())


def _runtime_metadata(torch, requested_device):
    """Read the selected runtime/device after generation has been authorized."""
    device_type, _, index_text = requested_device.partition(":")
    report = {
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "hip_runtime": getattr(torch.version, "hip", None),
        "xpu_runtime": getattr(torch.version, "xpu", None),
        "requested_device": requested_device,
        "device_type": device_type,
        "device_name": device_type,
    }
    accelerator = getattr(torch, device_type, None)
    if device_type in ("cuda", "xpu"):
        if accelerator is None or not accelerator.is_available():
            raise validation.ValidationBlocked(f"requested {device_type} runtime is unavailable")
        index = int(index_text) if index_text else accelerator.current_device()
        properties = accelerator.get_device_properties(index)
        report.update(
            device_index=index,
            device_name=accelerator.get_device_name(index),
            total_memory_bytes=int(properties.total_memory),
            visible_device_count=accelerator.device_count(),
        )
        # Include checkpoint load and sampling in the run's high-water mark.
        accelerator.reset_peak_memory_stats(index)
    return report


def _runtime_peak_memory(torch, runtime):
    if runtime["device_type"] not in ("cuda", "xpu"):
        return {"allocated_bytes": None, "reserved_bytes": None}
    accelerator = getattr(torch, runtime["device_type"])
    index = runtime["device_index"]
    return {
        "allocated_bytes": int(accelerator.max_memory_allocated(index)),
        "reserved_bytes": int(accelerator.max_memory_reserved(index)),
    }


def _bind_runtime_identity(manifest, reference, backend_provenance, runtime, dtype):
    policy = backend_provenance.get("kernel_policy")
    if not isinstance(policy, dict) or not policy.get("policy"):
        raise validation.ValidationBlocked("loaded backend did not declare its kernel policy")
    fields = {
        "kernel_policy": policy,
        "device_type": runtime["device_type"],
        "device_name": runtime["device_name"],
        "torch_version": runtime["torch_version"],
        "dtype": dtype,
    }
    manifest["identity"].update(fields)
    if reference:
        for field, value in fields.items():
            if reference.get("identity", {}).get(field) != value:
                raise validation.ValidationBlocked(f"reference runtime identity mismatch: {field}")


def _report(root, manifest):
    validation.write_json(root / "manifest.json", manifest)
    validation.write_json(
        root / "metrics.json",
        {
            "status": manifest["status"],
            "errors": manifest.get("errors", []),
            "records": manifest.get("records", []),
            "evaluation": manifest.get("evaluation"),
        },
    )
    gate = {
        key: manifest.get(key)
        for key in (
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
            "smoke",
        )
    }
    if manifest.get("smoke"):
        gate["status"] = "blocked"
        gate["reason"] = "smoke execution does not establish P0/P1 qualification"
    gate["manifest_sha256"] = validation.sha256_file(root / "manifest.json")
    validation.write_json(root / "gate_report.json", gate)
    lines = [
        "# Image decoder validation",
        "",
        f"Status: **{manifest['status']}**",
        "",
        manifest.get("claim", "No checkpoint or model was loaded."),
        "",
    ]
    if manifest.get("stage") == "P1":
        lines.extend(
            [
                "P1 acceptance: **unproven**. Numerical adapter comparisons do not include "
                "the required real-checkpoint reload, padding/batch, text/checkpoint, and "
                "target-free UnifiedTransformer regression evidence.",
                "",
            ]
        )
    lines.extend(f"- {error}" for error in manifest.get("errors", []))
    lines.extend(["", "## Fixed-case outputs (all attempts)", ""])
    for row in manifest.get("records", []):
        lines.append(
            f"- {row['case_id']} / seed {row['seed']}: {row['status']}"
            + (f" — {row['error']}" if row.get("error") else "")
        )
        if row.get("image"):
            lines.extend(["", f"![{row['case_id']} seed {row['seed']}]({row['image']})", ""])
    (root / "report.md").write_text("\n".join(lines) + "\n")


def _native(case):
    from PIL import Image

    # Canonical order is retained, and targets have no path into this function.
    images = []
    for path in case.reference_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return {"prompt": case.prompt, "input_images": images or None}


def _run_backend(backend, case, seed, args, saved=None):
    import numpy as np
    import torch

    native = _native(case)
    options = {
        "height": case.height,
        "width": case.width,
        "num_inference_steps": args.steps,
        "text_guidance_scale": args.text_guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "negative_prompt": args.negative_prompt,
        "generator": torch.Generator(device=args.device).manual_seed(seed),
        "trace": True,
    }
    if saved is not None:
        if "latents.initial" not in saved["tensors"]:
            raise validation.ValidationBlocked("reference initial latent bank missing")
        options["latents"] = torch.as_tensor(
            saved["tensors"]["latents.initial"],
            device=args.device,
            dtype=getattr(torch, args.dtype),
        ).clone()
    if args.reference:
        images = backend.generate_reference(native, **options)
        execution_path = "official_reference"
    elif args.parity_mode == "replay":
        embeds = torch.as_tensor(
            saved["tensors"]["condition.positive"],
            device=args.device,
            dtype=getattr(torch, args.dtype),
        )
        mask = torch.as_tensor(saved["tensors"]["mask.positive"], device=args.device)
        images = backend.generate_conditioned(embeds, mask, native_context=native, **options)
        execution_path = "saved_positive_condition_replay"
    else:
        from src.decoders.image import ImageDecoder
        from src.decoders.types import DecoderCondition

        # This tests PRISM's typed adapter around the unchanged native pipeline.
        # It does not claim an independently rewritten conditioner or Qwen parity.
        # The bypassed random connector must not consume the source VAE's RNG.
        with torch.random.fork_rng(devices=[]):
            decoder = ImageDecoder(d_model=1, backend=backend)
        condition = DecoderCondition(
            hidden_states=torch.zeros(1, 1, 1),
            attention_mask=torch.ones(1, 1, dtype=torch.bool),
            native_context=native,
            output_spec={"mode": "reference"},
            provenance={"mode": "reference"},
        )
        images = decoder.generate_condition(condition, **options)
        execution_path = "prism_ImageDecoder_reference_condition_route"
    raw = dict(backend.last_trace)
    tensors = raw.get("tensors", raw)
    tensors = {
        name: value
        for name, value in tensors.items()
        if isinstance(value, (torch.Tensor, np.ndarray))
    }
    if hasattr(images, "images"):
        images = images.images
    image = images[0] if isinstance(images, (list, tuple)) else images
    if "pixels" not in tensors:
        tensors["pixels"] = np.asarray(image)
    discrete = {
        "prompt": case.prompt,
        "source_order": [validation.sha256_file(p) for p in case.reference_paths],
        "output_spec": {"height": case.height, "width": case.width},
        "sampling": {
            k: v for k, v in options.items() if k not in ("generator", "latents", "trace")
        },
        "seed": seed,
    }
    discrete.update(raw.get("discrete", {}))
    provenance = backend.provenance() if callable(getattr(backend, "provenance", None)) else {}
    provenance.update(
        {
            "execution_path": execution_path,
            "real_checkpoint": True,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    return {"tensors": tensors, "discrete": discrete, "provenance": provenance}, image


def _run_seeded_backend(backend, case, seed, args, saved=None):
    """Upstream source-VAE sampling uses global RNG, separate from latent noise."""
    import random

    import numpy as np
    import torch

    device_type = args.device.split(":")[0]
    device_ids = [torch.device(args.device).index or 0] if device_type in ("cuda", "xpu") else []
    fork_type = device_type if device_ids else "cuda"
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=device_ids, device_type=fork_type):
            torch.manual_seed(seed)
            np.random.seed(seed % 2**32)
            random.seed(seed)
            return _run_backend(backend, case, seed, args, saved)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = args.output_dir.resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Blocked: output directory already exists: {root}", file=sys.stderr)
        return 2
    mode = (
        "preflight"
        if args.preflight
        else "reference"
        if args.reference
        else args.parity_mode
        if args.parity
        else "evaluation"
    )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    validation.write_json(root / "resolved-config.json", config)
    manifest = {
        "schema_version": 1,
        "stage": "P0"
        if args.reference
        else "P1"
        if args.parity
        else "P4"
        if args.evaluate
        else "preflight",
        "mode": mode,
        "smoke": args.smoke,
        "status": "blocked",
        "real_checkpoint": False,
        "numerical_parity": False,
        "validation_scope": ("reference_adapter_numerical_parity" if args.parity else mode),
        "p1_acceptance": {
            "status": "unproven",
            "required_companion_checks": [
                "checkpoint_reload",
                "padding_batch_invariance",
                "text_checkpoint_regression",
                "target_free_unified_transformer",
            ],
        },
        "identity": {
            "checkpoint_revision": args.revision,
            "upstream_revision": args.upstream_revision,
            "config_sha256": validation.sha256_file(root / "resolved-config.json"),
        },
        "records": [],
        "errors": [],
    }
    started = time.monotonic()
    runtime_torch = None
    try:
        if args.smoke and not (args.reference or args.parity):
            raise validation.ValidationBlocked("--smoke is only valid with --reference or --parity")
        if args.cases:
            manifest["identity"]["case_manifest_sha256"] = validation.sha256_file(args.cases)
            cases = validation.load_cases(args.cases, args.data_root, smoke=args.smoke)
            manifest["suite"] = validation.suite_summary(cases, "P4" if args.evaluate else "P0")
            if args.smoke:
                manifest["suite"]["complete_prespecified_size"] = False
            validation.write_json(root / "resolved-cases.json", [c.to_dict() for c in cases])
        elif not args.preflight:
            raise validation.ValidationBlocked("--cases is required")
        if args.evaluate:
            if not args.protocol:
                raise validation.ValidationBlocked("--protocol is required for qualification")
            manifest["identity"].update(
                {
                    "protocol_sha256": validation.sha256_file(args.protocol),
                    "evaluator_results_sha256": validation.sha256_file(args.evaluate),
                }
            )
            manifest["evaluation"] = validation.evaluate_qualification(
                cases, _json(args.evaluate), _json(args.protocol)
            )
            manifest["status"] = manifest["evaluation"]["status"]
            manifest["errors"] = manifest["evaluation"]["failures"]
            manifest["claim"] = (
                "Externally scored image capability gates; generation provenance and engineering gates must be reviewed separately."
            )
        else:
            manifest["preflight"] = validation.environment_preflight(
                args.checkpoint, args.revision, args.upstream, args.upstream_revision
            )
            if manifest["preflight"]["status"] != "ready":
                manifest["errors"].extend(manifest["preflight"]["errors"])
                raise validation.ValidationBlocked("offline reference prerequisites are incomplete")
            if args.preflight:
                manifest["status"] = "ready"
            else:
                if args.steps < 1:
                    raise validation.ValidationBlocked("sampling steps must be positive")
                tolerances = _json(args.tolerances) if args.tolerances else {}
                if args.tolerances:
                    manifest["identity"]["tolerances_sha256"] = validation.sha256_file(
                        args.tolerances
                    )
                reference = None
                if args.parity:
                    if not args.reference_dir:
                        raise validation.ValidationBlocked("--reference-dir is required for parity")
                    reference = _json(args.reference_dir / "manifest.json")
                    manifest["identity"]["reference_manifest_sha256"] = validation.sha256_file(
                        args.reference_dir / "manifest.json"
                    )
                    if reference.get("mode") != "reference" or not reference.get("real_checkpoint"):
                        raise validation.ValidationBlocked(
                            "parity requires a real official-reference artifact"
                        )
                    if not args.smoke and (
                        reference.get("status") != "passed" or reference.get("smoke")
                    ):
                        raise validation.ValidationBlocked(
                            "full parity qualification requires a completed non-smoke P0 reference"
                        )
                    for field in (
                        "case_manifest_sha256",
                        "checkpoint_revision",
                        "upstream_revision",
                    ):
                        if reference.get("identity", {}).get(field) != manifest["identity"].get(
                            field
                        ):
                            raise validation.ValidationBlocked(
                                f"reference identity mismatch: {field}"
                            )
                # Enforce offline mode before importing any model libraries.
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                sys.path[:0] = [str(ROOT), str(args.upstream.resolve())]
                import torch
                from src.decoders.omnigen2_backend import OmniGen2Backend

                runtime_torch = torch
                manifest["runtime"] = _runtime_metadata(torch, args.device)
                backend = OmniGen2Backend(
                    model_id=str(args.checkpoint.resolve()),
                    revision=args.revision,
                    local_files_only=True,
                ).to(device=args.device, dtype=getattr(torch, args.dtype))
                backend.ensure_loaded()
                manifest["backend_provenance"] = backend.provenance()
                _bind_runtime_identity(
                    manifest,
                    reference,
                    manifest["backend_provenance"],
                    manifest["runtime"],
                    args.dtype,
                )
                # Preserve kernel choice even if hashing or the first sample fails.
                _report(root, manifest)
                checkpoint_manifest = backend.checkpoint_manifest()
                validation.write_json(root / "checkpoint_manifest.json", checkpoint_manifest)
                manifest["identity"]["checkpoint_manifest_sha256"] = checkpoint_manifest[
                    "manifest_sha256"
                ]
                manifest["identity"]["reference_checkpoint_sha256"] = checkpoint_manifest[
                    "manifest_sha256"
                ]
                if (
                    reference
                    and reference.get("identity", {}).get("checkpoint_manifest_sha256")
                    != checkpoint_manifest["manifest_sha256"]
                ):
                    raise validation.ValidationBlocked(
                        "actual checkpoint file identity differs from reference"
                    )
                manifest["claim"] = {
                    "reference": "Actual pinned official-reference generation; no learned PRISM capability claim.",
                    "replay": "Saved positive-conditioning replay only; no input-preprocessing or full-pipeline equivalence claim.",
                    "full_pipeline": "PRISM typed ImageDecoder reference-mode adapter parity with unchanged official conditioner and generator; no independently reimplemented conditioner or learned PRISM/Qwen equivalence claim.",
                }[mode]
                for case in cases:
                    for seed in case.seeds:
                        record = {
                            "case_id": case.case_id,
                            "task": case.task,
                            "group": case.group,
                            "seed": seed,
                            "status": "failed",
                        }
                        tick = time.monotonic()
                        try:
                            saved = (
                                validation.read_trace(
                                    args.reference_dir / "comparisons" / case.case_id / str(seed)
                                )
                                if reference
                                else None
                            )
                            trace, output_image = _run_seeded_backend(
                                backend, case, seed, args, saved
                            )
                            directory = root / "comparisons" / case.case_id / str(seed)
                            validation.archive_trace(directory, trace)
                            manifest["real_checkpoint"] = True
                            import numpy as np
                            from PIL import Image

                            if not isinstance(output_image, Image.Image):
                                array = np.asarray(output_image)
                                if np.issubdtype(array.dtype, np.floating):
                                    array = (array.clip(0, 1) * 255).round().astype("uint8")
                                output_image = Image.fromarray(array)
                            image_path = root / "examples" / f"{case.case_id}-{seed}.png"
                            image_path.parent.mkdir(exist_ok=True)
                            output_image.save(image_path)
                            record.update(
                                {
                                    "image": str(image_path.relative_to(root)),
                                    "image_sha256": validation.sha256_file(image_path),
                                    "trace": str(directory.relative_to(root)),
                                }
                            )
                            required = (
                                validation.PARITY_BOUNDARIES
                                if mode == "replay"
                                else validation.FULL_PIPELINE_BOUNDARIES
                            )
                            required += tuple(
                                f"reference.0.{index}" for index in range(len(case.reference_paths))
                            )
                            missing = set(required) - set(trace["tensors"])
                            if missing:
                                raise validation.ValidationBlocked(
                                    f"required capture boundaries missing: {sorted(missing)}"
                                )
                            if saved:
                                comparison_reference, comparison_trace = saved, trace
                                if mode == "replay":
                                    # Native positive input preprocessing was bypassed:
                                    # never compare/claim those uncaptured boundaries.
                                    excluded = (
                                        "token_ids.",
                                        "token_mask.",
                                        "position_ids.",
                                        "rope_deltas.",
                                    )
                                    comparison_reference = {
                                        **saved,
                                        "tensors": {
                                            k: v
                                            for k, v in saved["tensors"].items()
                                            if not k.startswith(excluded)
                                        },
                                    }
                                    comparison_trace = {
                                        **trace,
                                        "tensors": {
                                            k: v
                                            for k, v in trace["tensors"].items()
                                            if not k.startswith(excluded)
                                        },
                                    }
                                    record["excluded_boundaries"] = list(excluded)
                                record["comparison"] = validation.compare_traces(
                                    comparison_reference, comparison_trace, tolerances, required
                                )
                                if not record["comparison"]["passed"]:
                                    raise validation.ValidationBlocked(
                                        "registered numerical comparisons failed"
                                    )
                            record["status"] = "passed"
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: {exc}"
                        record["duration_seconds"] = time.monotonic() - tick
                        manifest["records"].append(record)
                        _report(root, manifest)
                all_passed = all(row["status"] == "passed" for row in manifest["records"])
                complete = manifest["suite"]["complete_prespecified_size"]
                manifest["status"] = (
                    "completed"
                    if args.smoke and all_passed
                    else "passed"
                    if all_passed and complete
                    else "failed"
                    if not all_passed
                    else "incomplete"
                )
                if not complete and not args.smoke:
                    manifest["errors"].append(
                        "partial fixed-case suite: cannot establish P0/P1 gate"
                    )
                if args.smoke:
                    manifest["claim"] += " Smoke run only; P0/P1 qualification remains blocked."
                manifest["numerical_parity"] = bool(
                    args.parity and mode == "full_pipeline" and manifest["status"] == "passed"
                )
    except Exception as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        manifest["status"] = "blocked"
    if runtime_torch is not None and "runtime" in manifest:
        try:
            manifest["peak_memory"] = _runtime_peak_memory(runtime_torch, manifest["runtime"])
            manifest["peak_memory_bytes"] = manifest["peak_memory"]["allocated_bytes"]
        except Exception as exc:
            manifest["peak_memory_error"] = f"{type(exc).__name__}: {exc}"
    manifest["duration_seconds"] = time.monotonic() - started
    _report(root, manifest)
    print(f"{manifest['status']}: {root / 'report.md'}")
    return 0 if manifest["status"] in ("passed", "ready", "completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
