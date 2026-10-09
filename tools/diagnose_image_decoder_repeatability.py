#!/usr/bin/env python3
"""Bounded exact-comparison diagnostic, never a P0/P1 acceptance report.

One loaded backend runs native N0, native N1, adapter A0, then native N2 using
the same archived initial latent. A second process can compare its N0 using
--prior-run. Differences are diagnostic results, not execution failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prism_repeatability_validator", ROOT / "tools/validate_image_decoder.py"
)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)
validation = validator.validation
CALLS = ("N0", "N1", "A0", "N2")


def _log(message):
    print(f"[image-repeatability] {message}", flush=True)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _source_hashes():
    return {
        name: validation.sha256_file(ROOT / name)
        for name in (
            "tools/diagnose_image_decoder_repeatability.py",
            "tools/validate_image_decoder.py",
            "src/eval/image_generation.py",
            "src/decoders/omnigen2_backend.py",
            "src/decoders/image.py",
            "src/decoders/types.py",
        )
    }


def numerical_policy(args):
    return {
        "deterministic_algorithms": bool(getattr(args, "deterministic", False)),
        "attention_backend": getattr(args, "attention_backend", "current"),
    }


@contextmanager
def execution_policy(args):
    """Select and restore numerical policy before loading any checkpoint."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    policy = numerical_policy(args)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    attention = (
        sdpa_kernel(SDPBackend.MATH) if policy["attention_backend"] == "math" else nullcontext()
    )
    try:
        torch.use_deterministic_algorithms(policy["deterministic_algorithms"], warn_only=False)
        with attention:
            yield policy
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


def bind_diagnostic_identity(manifest, reference, backend_provenance, runtime, args):
    """Permit explicitly exploratory precision changes without relaxing P1 checks."""
    validator._bind_runtime_identity(manifest, None, backend_provenance, runtime, args.dtype)
    policy = numerical_policy(args)
    manifest["identity"]["numerical_policy"] = policy
    if reference is None:
        return
    identity = reference.get("identity", {})
    reference_policy = identity.get(
        "numerical_policy",
        {"deterministic_algorithms": False, "attention_backend": "current"},
    )
    exploratory = identity.get("dtype") != args.dtype or reference_policy != policy
    if exploratory and not getattr(args, "allow_precision_policy_comparison", False):
        raise validation.ValidationBlocked(
            "reference precision/numerical policy differs; use "
            "--allow-precision-policy-comparison for explicitly exploratory diagnostics"
        )
    for field in ("kernel_policy", "device_type", "device_name", "torch_version"):
        if identity.get(field) != manifest["identity"][field]:
            raise validation.ValidationBlocked(f"reference runtime identity mismatch: {field}")
    manifest["reference_comparison_scope"] = (
        "precision_policy_exploratory" if exploratory else "same_precision_policy"
    )
    manifest["reference_numerical_policy"] = reference_policy
    manifest["reference_dtype"] = identity.get("dtype")


def _settings(torch, backend, device_type):
    cuda = torch.backends.cuda
    return {
        "module_training_flags": {
            name: bool(module.training) for name, module in backend.named_modules()
        },
        "caller_grad_enabled": torch.is_grad_enabled(),
        "generation_grad_context": "backend and adapter sampling use torch.no_grad",
        "autocast_enabled": torch.is_autocast_enabled(device_type),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": cuda.matmul.allow_tf32,
        "cuda_matmul_allow_fp16_reduced_precision_reduction": cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sdpa_math_enabled": cuda.math_sdp_enabled(),
        "sdpa_flash_enabled": cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient_enabled": cuda.mem_efficient_sdp_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            key: os.environ[key]
            for key in (
                "ZE_AFFINITY_MASK",
                "ZE_FLAT_DEVICE_HIERARCHY",
                "ONEAPI_DEVICE_SELECTOR",
                "SYCL_DEVICE_FILTER",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "ONEDNN_DEFAULT_FPMATH_MODE",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_XPU_ALLOC_CONF",
            )
            if key in os.environ
        },
    }


def _interpret(comparisons):
    findings = []
    passed = {key: value["passed"] for key, value in comparisons.items()}
    if passed.get("native_repeat") is False:
        findings.append(
            "Native calls differ within one loaded backend; this does not establish an adapter-specific error."
        )
    elif passed.get("native_repeat") is True and passed.get("adapter_vs_native") is False:
        findings.append(
            "The native repeat matched, but the adapter call differed; wrapper, call-order, or execution-context effects need further isolation."
        )
    if passed.get("native_after_adapter") is False:
        findings.append(
            "The native call after the adapter differs from N1; state or call-order effects remain possible."
        )
    if all(
        passed.get(key) is True
        for key in ("native_repeat", "adapter_vs_native", "native_after_adapter")
    ):
        findings.append(
            "All four calls matched exactly within this process for this fixed case and latent."
        )
        if passed.get("original_reference") is False:
            findings.append(
                "The original archived run differs despite this within-process agreement; cross-process or runtime variation remains unresolved."
            )
    if passed.get("prior_process") is False:
        findings.append(
            "N0 differs from the previous diagnostic process under the recorded identities; the precise cause is not established."
        )
    if passed.get("prior_process") is True:
        findings.append(
            "N0 matched the previous diagnostic process exactly for this case and latent."
        )
    findings.append(
        "No tolerances were changed. This diagnostic supplies no P0/P1 acceptance or image-quality claim."
    )
    return findings


def _write_report(root, manifest):
    validation.write_json(root / "manifest.json", manifest)
    validation.write_json(root / "comparisons.json", manifest.get("comparisons", {}))
    lines = [
        "# Image decoder repeatability diagnostic",
        "",
        f"Execution: **{manifest['status']}**",
        "",
        "P0/P1 acceptance: **not evaluated**. All numerical comparisons use exact equality.",
        "",
    ]
    for row in manifest.get("calls", []):
        lines.append(
            f"- {row['name']}: {row['status']}" + (f" — {row['error']}" if row.get("error") else "")
        )
    for name, comparison in manifest.get("comparisons", {}).items():
        lines.append(f"- {name}: {'exact match' if comparison['passed'] else 'DIFFERENT'}")
    lines.extend(["", *manifest.get("findings", [])])
    lines.extend(f"- {error}" for error in manifest.get("errors", []))
    (root / "report.md").write_text("\n".join(lines) + "\n")


def run_diagnostic(backend, case, args, *, preflight=None, fixture=False, run_case=None):
    with execution_policy(args):
        return _run_diagnostic(
            backend, case, args, preflight=preflight, fixture=fixture, run_case=run_case
        )


def _run_diagnostic(backend, case, args, *, preflight=None, fixture=False, run_case=None):
    """One-load experiment; injected runners/backends are fixture-only test seams."""
    if run_case is not None and not fixture:
        raise ValueError("an injected run_case is allowed only for explicitly labelled fixtures")
    if len(case.seeds) != 1 or not 1 <= args.steps <= 4:
        raise ValueError("diagnostic requires one sampling seed and one to four steps")
    import numpy as np
    import torch
    from PIL import Image

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "mode": "repeatability_diagnostic",
        "evidence_kind": "fixture_only" if fixture else "real_checkpoint_diagnostic",
        "fixture": fixture,
        "p0_p1_acceptance": "not_evaluated",
        "tolerances": {},
        "hostname": platform.node(),
        "calls": [],
        "comparisons": {},
        "errors": [],
        "preflight": preflight,
        "source_hashes": _source_hashes(),
        "identity": {},
        "case": case.to_dict(),
        "backend_load_count": 0,
    }
    started = time.monotonic()
    _write_report(root, manifest)
    traces = {}
    original = None
    try:
        _log("read original reference manifest and archived fixed initial latent")
        reference_dir = Path(args.reference_dir).resolve()
        reference = _read_json(reference_dir / "manifest.json")
        if not fixture and (
            reference.get("mode") != "reference" or reference.get("real_checkpoint") is not True
        ):
            raise validation.ValidationBlocked(
                "original artifacts must come from an actual official-reference run"
            )
        original = validation.read_trace(
            reference_dir / "comparisons" / case.case_id / str(case.seeds[0])
        )
        if "latents.initial" not in original["tensors"]:
            raise validation.ValidationBlocked("original fixed latent bank missing")
        manifest["identity"] = {
            "reference_manifest_sha256": validation.sha256_file(reference_dir / "manifest.json"),
            "case_manifest_sha256": validation.sha256_file(args.cases),
            "checkpoint_revision": args.revision,
            "upstream_revision": args.upstream_revision,
            "initial_latent_sha256": _read_json(
                reference_dir / "comparisons" / case.case_id / str(case.seeds[0]) / "trace.json"
            )["tensors"]["latents.initial"]["sha256"],
            "sampling": {
                "seed": case.seeds[0],
                "height": case.height,
                "width": case.width,
                "steps": args.steps,
                "text_guidance_scale": args.text_guidance_scale,
                "image_guidance_scale": args.image_guidance_scale,
                "negative_prompt": args.negative_prompt,
            },
        }
        manifest["runtime"] = validator._runtime_metadata(torch, args.device)
        _log("load backend once")
        tick = time.monotonic()
        backend.to(device=args.device, dtype=getattr(torch, args.dtype))
        backend.ensure_loaded()
        manifest["backend_load_count"] = 1
        manifest["load_seconds"] = time.monotonic() - tick
        manifest["backend_provenance"] = backend.provenance()
        bind_diagnostic_identity(
            manifest,
            None if fixture else reference,
            backend.provenance(),
            manifest["runtime"],
            args,
        )
        _log("backend loaded; hash actual checkpoint components")
        tick = time.monotonic()
        checkpoint = backend.checkpoint_manifest()
        manifest["checkpoint_hash_seconds"] = time.monotonic() - tick
        validation.write_json(root / "checkpoint_manifest.json", checkpoint)
        manifest["identity"]["checkpoint_manifest_sha256"] = checkpoint["manifest_sha256"]
        if (
            not fixture
            and reference.get("identity", {}).get("checkpoint_manifest_sha256")
            != checkpoint["manifest_sha256"]
        ):
            raise validation.ValidationBlocked("actual checkpoint differs from original reference")
        _log("checkpoint identities verified; begin N0/N1/A0/N2")
        prior = None
        if args.prior_run:
            prior_root = Path(args.prior_run).resolve()
            prior_manifest = _read_json(prior_root / "manifest.json")
            if (
                prior_manifest.get("status") != "completed"
                or prior_manifest.get("fixture") != fixture
            ):
                raise validation.ValidationBlocked(
                    "prior diagnostic must be completed and have the same evidence kind"
                )
            if (
                prior_manifest.get("identity") != manifest["identity"]
                or prior_manifest.get("source_hashes") != manifest["source_hashes"]
            ):
                raise validation.ValidationBlocked(
                    "prior diagnostic source/checkpoint/runtime/noise identities differ"
                )
            prior = validation.read_trace(prior_root / "traces" / "N0")
            manifest["prior_manifest_sha256"] = validation.sha256_file(prior_root / "manifest.json")
        required = validation.FULL_PIPELINE_BOUNDARIES + tuple(
            f"reference.0.{i}" for i in range(len(case.reference_paths))
        )
        execute = run_case or validator._run_seeded_backend
        for name in CALLS:
            _log(
                f"start {name}: {'typed reference adapter' if name == 'A0' else 'native reference'}; original fixed latent, seed={case.seeds[0]}"
            )
            row = {
                "name": name,
                "status": "running",
                "seed": case.seeds[0],
                "initial_latent_sha256": manifest["identity"]["initial_latent_sha256"],
            }
            manifest["calls"].append(row)
            call_args = argparse.Namespace(**vars(args))
            call_args.reference = name != "A0"
            call_args.parity_mode = "full_pipeline"
            before = _settings(torch, backend, args.device.split(":")[0])
            validation.write_json(root / f"{name}-settings-before.json", before)
            row["settings_before_sha256"] = validation.sha256_file(
                root / f"{name}-settings-before.json"
            )
            _write_report(root, manifest)
            tick = time.monotonic()
            try:
                trace, image = execute(backend, case, case.seeds[0], call_args, original)
                if fixture:
                    trace["provenance"] = {
                        **trace.get("provenance", {}),
                        "real_checkpoint": False,
                        "evidence_kind": "fixture_only",
                    }
                validation.archive_trace(root / "traces" / name, trace)
                if not isinstance(image, Image.Image):
                    array = np.asarray(image)
                    if np.issubdtype(array.dtype, np.floating):
                        array = (array.clip(0, 1) * 255).round().astype("uint8")
                    image = Image.fromarray(array)
                image.save(root / f"{name}.png")
                missing = set(required) - set(trace["tensors"])
                if missing:
                    raise validation.ValidationBlocked(
                        f"required capture boundaries missing: {sorted(missing)}"
                    )
                if not np.array_equal(
                    validation._array(trace["tensors"]["latents.initial"]),
                    original["tensors"]["latents.initial"],
                ):
                    raise validation.ValidationBlocked(
                        "backend did not use the unchanged original latent bank"
                    )
                traces[name] = trace
                row.update(
                    status="completed",
                    image=f"{name}.png",
                    image_sha256=validation.sha256_file(root / f"{name}.png"),
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            row["duration_seconds"] = time.monotonic() - tick
            after = _settings(torch, backend, args.device.split(":")[0])
            validation.write_json(root / f"{name}-settings-after.json", after)
            row["settings_after_sha256"] = validation.sha256_file(
                root / f"{name}-settings-after.json"
            )
            row["execution_settings_unchanged"] = before == after
            _log(f"finish {name}: {row['status']}, {row['duration_seconds']:.3f}s")
            _write_report(root, manifest)
        pairs = {
            "native_repeat": ("N0", "N1"),
            "adapter_vs_native": ("N1", "A0"),
            "native_after_adapter": ("N1", "N2"),
        }
        for label, (left, right) in pairs.items():
            if left in traces and right in traces:
                manifest["comparisons"][label] = {
                    "left": left,
                    "right": right,
                    **validation.compare_traces(traces[left], traces[right], required=required),
                }
        if "N0" in traces:
            manifest["comparisons"]["original_reference"] = validation.compare_traces(
                original, traces["N0"], required=required
            )
            if prior is not None:
                manifest["comparisons"]["prior_process"] = validation.compare_traces(
                    prior, traces["N0"], required=required
                )
        manifest["status"] = "completed" if len(traces) == len(CALLS) else "failed"
        manifest["findings"] = _interpret(manifest["comparisons"])
        if manifest.get("reference_comparison_scope") == "precision_policy_exploratory":
            manifest["findings"].append(
                "Comparison to the archived reference changes precision and/or numerical policy; "
                "it is exploratory, not parity qualification. Within-process calls share one policy."
            )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        _log(manifest["errors"][-1])
    if "runtime" in manifest:
        try:
            manifest["peak_memory"] = validator._runtime_peak_memory(torch, manifest["runtime"])
        except Exception as exc:
            manifest["peak_memory_error"] = str(exc)
    manifest["duration_seconds"] = time.monotonic() - started
    _write_report(root, manifest)
    for label, comparison in manifest["comparisons"].items():
        _log(f"{label}: {'exact match' if comparison['passed'] else 'DIFFERENT'}")
    _log(
        f"diagnostic {manifest['status']}; report={root / 'report.md'}; P0/P1 acceptance not evaluated"
    )
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "upstream", "cases", "output-dir", "reference-dir"):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument("--prior-run", type=Path)
    result.add_argument("--deterministic", action="store_true")
    result.add_argument("--attention-backend", choices=("current", "math"), default="current")
    result.add_argument("--allow-precision-policy-comparison", action="store_true")
    result.add_argument("--data-root", type=Path)
    result.add_argument("--device", required=True)
    result.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    result.add_argument("--revision", default=validator.DEFAULT_REVISION)
    result.add_argument("--upstream-revision", default=validator.DEFAULT_UPSTREAM_REVISION)
    result.add_argument("--steps", type=int, default=2)
    result.add_argument("--text-guidance-scale", type=float, default=5.0)
    result.add_argument("--image-guidance-scale", type=float, default=2.0)
    result.add_argument("--negative-prompt", default="")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.output_dir.resolve()
    if root.exists():
        _log(f"blocked: output directory already exists: {root}")
        return 2
    try:
        if not 1 <= args.steps <= 4:
            raise validation.ValidationBlocked("diagnostic allows one to four steps only")
        cases = validation.load_cases(args.cases, args.data_root, smoke=True)
        if len(cases) != 1 or len(cases[0].seeds) != 1:
            raise validation.ValidationBlocked("diagnostic requires exactly one case and one seed")
        _log("offline metadata and local-file preflight (no models loaded)")
        preflight = validation.environment_preflight(
            args.checkpoint, args.revision, args.upstream, args.upstream_revision
        )
        if preflight["status"] != "ready":
            raise validation.ValidationBlocked("; ".join(preflight["errors"]))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path[:0] = [str(ROOT), str(args.upstream.resolve())]
        from src.decoders.omnigen2_backend import OmniGen2Backend

        backend = OmniGen2Backend(
            model_id=str(args.checkpoint.resolve()), revision=args.revision, local_files_only=True
        )
        manifest = run_diagnostic(backend, cases[0], args, preflight=preflight)
        return 0 if manifest["status"] == "completed" else 2
    except Exception as exc:
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "blocked",
            "mode": "repeatability_diagnostic",
            "p0_p1_acceptance": "not_evaluated",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        _write_report(root, manifest)
        _log(manifest["errors"][0])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
