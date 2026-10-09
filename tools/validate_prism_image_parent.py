#!/usr/bin/env python3
"""Bounded, offline diagnostic of a fully restored PRISM vision-language parent.

This measures input/hidden-state plumbing and text baselines. It never creates
input-alignment, P1 acceptance, or image-capability evidence. The image route is
instrumented at its decoder boundary; OmniGen2 is not loaded or sampled.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "prism_parent_validation", ROOT / "src/eval/image_generation.py"
)
validation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validation
_SPEC.loader.exec_module(validation)


def _log(message):
    print(f"[prism-parent] {message}", flush=True)


def load_cases(path, max_cases=8):
    root = Path(path).resolve().parent
    cases, seen = [], set()
    allowed = {
        "id",
        "case_id",
        "prompt",
        "source_images",
        "comparison_prompt",
        "comparison_source_images",
        "expected_answers",
        "benchmark",
        "group",
    }
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError(
                f"case {line_number}: unsupported fields; target images/latents are forbidden"
            )
        identifier = row.get("id", row.get("case_id"))
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
            or identifier in seen
        ):
            raise ValueError(f"case {line_number}: unique safe id required")
        for key in ("prompt", "comparison_prompt"):
            if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                raise ValueError(f"case {identifier}: {key} must be nonempty text")
        if "prompt" not in row:
            raise ValueError(f"case {identifier}: prompt required")
        resolved = {**row, "id": identifier}
        for key in ("source_images", "comparison_source_images"):
            values = row.get(key, [])
            if (
                not isinstance(values, list)
                or len(values) > 5
                or any(not isinstance(p, str) or "://" in p for p in values)
            ):
                raise ValueError(
                    f"case {identifier}: {key} requires at most five ordered local paths"
                )
            paths = [str((root / p).resolve()) for p in values]
            if any(not Path(p).is_file() for p in paths):
                raise ValueError(f"case {identifier}: source asset missing")
            if key in row or key == "source_images":
                resolved[key] = paths
        answers = row.get("expected_answers", [])
        if not isinstance(answers, list) or any(
            not isinstance(answer, str) or not answer for answer in answers
        ):
            raise ValueError(
                "expected_answers must contain strings; labels never enter model inputs"
            )
        cases.append(resolved)
        seen.add(identifier)
    if not 1 <= len(cases) <= max_cases <= 8:
        raise ValueError("diagnostic requires one to eight fixed cases; no silent subset selection")
    return cases


def _inputs(case, tokenizer, transform, device, dtype, *, prompt=None, source_paths=None):
    import torch
    from PIL import Image, ImageOps

    prompt = case["prompt"] if prompt is None else prompt
    paths = case["source_images"] if source_paths is None else source_paths
    encoded = tokenizer([prompt], padding=True, truncation=False, return_tensors="pt")
    ids, mask = encoded["input_ids"], encoded.get("attention_mask")
    if mask is None or mask.shape != ids.shape or ids.ndim != 2:
        raise ValueError("tokenizer must supply aligned input_ids and attention_mask")
    result = {
        "text": ids.to(device),
        "text_attention_mask": mask.to(device),
        "_metadata": [f"{int(mask[0].sum())} 0"],
    }
    images, values = [], []
    for path in paths:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").copy()
        images.append(image)
        value = transform(image.copy())
        if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all():
            raise ValueError("source processor must return finite [3,H,W] pixels")
        values.append(value)
    if values:
        result["image"] = torch.stack(values)[None].to(device=device, dtype=dtype)
        result["image_mask"] = torch.ones(1, len(values), dtype=torch.bool, device=device)
    return result, {
        "image": {
            "reference_images": [images],
            "source_ids": [[f"{case['id']}:source:{index}" for index in range(len(paths))]],
        }
    }


def _capture_image_boundary(model, inputs, native_context):
    """Run the real model and connector, intercept only the native sampler call."""
    import torch

    decoder = model.decoders["image"]
    previous = decoder.__dict__.get("generate_condition")
    had_override = "generate_condition" in decoder.__dict__
    captured = {"component_shapes": {}}
    hooks = []

    def capture_compiled(module, arguments, kwargs):
        for key in ("inputs_embeds", "attention_mask", "position_ids"):
            value = kwargs.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"parent backbone must receive explicit {key}")
            captured[f"compiled.{key}"] = value.detach().cpu().clone()

    hooks.append(model.backbone.register_forward_pre_hook(capture_compiled, with_kwargs=True))
    for group in ("encoders", "projectors"):
        modules = getattr(model, group, {})
        if "image" in modules:

            def record_shape(module, arguments, output, group=group):
                if isinstance(output, torch.Tensor):
                    captured["component_shapes"][f"{group}.image"] = {
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "finite": bool(torch.isfinite(output).all()),
                    }

            hooks.append(modules["image"].register_forward_hook(record_shape))

    def capture(self, condition, **kwargs):
        connected, connected_mask = self.connect(condition)
        captured.update(
            hidden_states=condition.hidden_states.detach().cpu(),
            attention_mask=condition.attention_mask.detach().cpu(),
            connected_states=connected.detach().cpu(),
            connected_mask=connected_mask.detach().cpu(),
            modality_spans=condition.modality_spans,
            native_keys=sorted(condition.native_context),
            provenance=condition.provenance,
        )
        return {"diagnostic_capture_only": True, "conditioning_shape": list(connected.shape)}

    decoder.generate_condition = MethodType(capture, decoder)
    try:
        result = model.predict(inputs, requested_outputs=["image"], native_context=native_context)
        if result.loss is not None or not result.predictions["image"].get(
            "diagnostic_capture_only"
        ):
            raise ValueError(
                "target-free route unexpectedly returned training loss or sampled images"
            )
    finally:
        for hook in hooks:
            hook.remove()
        if had_override:
            decoder.generate_condition = previous
        else:
            del decoder.generate_condition
    if any(
        f"compiled.{key}" not in captured
        for key in ("inputs_embeds", "attention_mask", "position_ids")
    ):
        raise ValueError("parent backbone input capture is incomplete")
    if "hidden_states" not in captured:
        raise ValueError("UnifiedTransformer did not dispatch an image DecoderCondition")
    return captured


def _hidden_difference(left, right):
    import torch

    a, b = left["hidden_states"].float(), right["hidden_states"].float()
    ma, mb = left["attention_mask"].bool(), right["attention_mask"].bool()
    a_last = a[0, ma[0]][-1]
    b_last = b[0, mb[0]][-1]
    delta = a_last - b_last
    result = {
        "sequence_shapes_equal": a.shape == b.shape,
        "last_valid_max_absolute_error": float(delta.abs().max()),
        "last_valid_relative_l2": float(delta.norm() / a_last.norm().clamp_min(1e-12)),
        "last_valid_exact_match": bool(torch.equal(a_last, b_last)),
        "claim": "representation sensitivity only; not semantic correctness or learned alignment",
    }
    a_valid, b_valid = a[0, ma[0]], b[0, mb[0]]
    if a_valid.shape == b_valid.shape:
        result["all_valid_exact_match"] = bool(torch.equal(a_valid, b_valid))
        result["all_valid_max_absolute_error"] = float((a_valid - b_valid).abs().max())
    result["compiled_inputs"] = {}
    for key in ("inputs_embeds", "attention_mask", "position_ids"):
        x, y = left[f"compiled.{key}"], right[f"compiled.{key}"]
        same_shape = x.shape == y.shape
        result["compiled_inputs"][key] = {
            "shapes_equal": same_shape,
            "exact_match": bool(same_shape and torch.equal(x, y)),
            "max_absolute_error": float((x.float() - y.float()).abs().max())
            if same_shape
            else None,
        }
    return result


def _save_capture(path, capture, *, fixture=False):
    import torch

    validation.archive_trace(
        path,
        {
            "tensors": {
                key: value for key, value in capture.items() if isinstance(value, torch.Tensor)
            },
            "discrete": {
                "modality_spans": capture["modality_spans"],
                "native_keys": capture["native_keys"],
                "component_shapes": capture["component_shapes"],
            },
            "provenance": {
                **capture["provenance"],
                "scope": "parent_capture_only_random_output_connector",
                "evidence_kind": "fixture_only" if fixture else "real_checkpoint_diagnostic",
                "real_checkpoint": not fixture,
            },
        },
    )


def _restoration_check(provenance, fixture):
    if fixture:
        return
    report = provenance.get("restoration")
    if not isinstance(report, dict) or report.get("strict_parent") is not True:
        raise ValueError("real diagnostic requires a strict complete-parent restoration report")
    if report.get("missing_parent_keys") != [] or report.get("unexpected_keys") != []:
        raise ValueError("full parent weights were not restored exactly")
    if type(report.get("loaded_key_count")) is not int or report["loaded_key_count"] < 1:
        raise ValueError("restoration must account for loaded parent tensors")
    if any(
        not key.startswith("decoders.image.connector.")
        for key in report.get("new_connector_keys", [])
    ):
        raise ValueError("only the new image output connector may be initialized")


def run_parent_diagnostic(bundle, cases, args, *, fixture=False):
    from tools.diagnose_image_decoder_repeatability import execution_policy

    with execution_policy(args):
        return _run_parent_diagnostic(bundle, cases, args, fixture=fixture)


def _run_parent_diagnostic(bundle, cases, args, *, fixture=False):
    import torch
    from tools.train_image_decoder import frozen_state_hashes

    if not 1 <= len(cases) <= 8 or not 1 <= args.max_new_tokens <= 32:
        raise ValueError("bounded diagnostic allows at most eight cases and 32 generated tokens")
    _restoration_check(bundle["provenance"], fixture)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    dtype = getattr(torch, args.dtype)
    model.to(device=args.device, dtype=dtype).eval().requires_grad_(False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "mode": "prism_parent_diagnostic",
        "evidence_kind": "fixture_only" if fixture else "real_checkpoint_diagnostic",
        "fixture": fixture,
        "alignment_gate": "not_evaluated",
        "p1_acceptance": "unproven",
        "generator_loaded": False,
        "sampling": "none; image route is capture-only",
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": str(torch.__version__),
        "numerical_policy": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "attention_backend": getattr(args, "attention_backend", "current"),
        },
        "parent_provenance": bundle["provenance"],
        "cases": [],
        "errors": [],
        "source_sha256": {
            name: validation.sha256_file(ROOT / name)
            for name in (
                "tools/validate_prism_image_parent.py",
                "tools/diagnose_image_decoder_repeatability.py",
                "src/model.py",
                "src/decoders/loading.py",
                "src/decoders/conditioning.py",
            )
        },
    }
    validation.write_json(root / "resolved-cases.json", cases)

    def persist():
        validation.write_json(root / "manifest.json", manifest)

    started = time.monotonic()
    persist()
    _log("fingerprint frozen parent parameters and buffers before inference")
    before = frozen_state_hashes(model, ("decoders.image.connector",))
    validation.write_json(root / "frozen-before.json", before)
    manifest["frozen_before_sha256"] = validation.sha256_file(root / "frozen-before.json")
    generation_options = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    try:
        with torch.no_grad():
            for case in cases:
                _log(
                    f"case {case['id']}: real parent routing, text baseline, and input diagnostics"
                )
                row = {
                    "id": case["id"],
                    "status": "running",
                    "source_sha256": {p: validation.sha256_file(p) for p in case["source_images"]},
                }
                manifest["cases"].append(row)
                tick = time.monotonic()
                try:
                    inputs, native = _inputs(
                        case, tokenizer, bundle["source_transform"], args.device, dtype
                    )
                    baseline = _capture_image_boundary(model, inputs, native)
                    hidden = baseline["hidden_states"]
                    connected = baseline["connected_states"]
                    if (
                        hidden.ndim != 3
                        or not torch.isfinite(hidden).all()
                        or not torch.isfinite(connected).all()
                    ):
                        raise ValueError("nonfinite or invalid parent hidden states")
                    width = model.backbone.get_input_embeddings().weight.shape[1]
                    if (
                        hidden.shape[-1] != width
                        or connected.shape[-1] != model.decoders["image"].conditioning_dim
                    ):
                        raise ValueError(
                            "backbone and image connector widths do not match declared dimensions"
                        )
                    _save_capture(
                        root / "traces" / case["id"] / "baseline", baseline, fixture=fixture
                    )
                    row.update(
                        hidden_shape=list(hidden.shape),
                        valid_tokens=int(baseline["attention_mask"].sum()),
                        connector_shape=list(connected.shape),
                        target_fields_in_inputs=False,
                        image_route="capture_only; pretrained generator not executed",
                        component_shapes=baseline["component_shapes"],
                    )
                    row["same_input_repeats"] = []
                    for repeat_index in range(2):
                        repeated = _capture_image_boundary(model, inputs, native)
                        row["same_input_repeats"].append(_hidden_difference(baseline, repeated))
                        _save_capture(
                            root / "traces" / case["id"] / f"same_input_repeat_{repeat_index + 1}",
                            repeated,
                            fixture=fixture,
                        )
                    old_tokens = model.generate(inputs=inputs, **generation_options)
                    new_tokens = model.predict(
                        inputs,
                        requested_outputs=["text"],
                        decoder_kwargs={"text": generation_options},
                    ).predictions["text"]
                    row["text"] = {
                        "legacy_token_ids": old_tokens[0].detach().cpu().tolist(),
                        "new_route_token_ids": new_tokens[0].detach().cpu().tolist(),
                        "exact_route_match": bool(torch.equal(old_tokens, new_tokens)),
                        "legacy_response": tokenizer.decode(
                            old_tokens[0], skip_special_tokens=True
                        ),
                        "new_route_response": tokenizer.decode(
                            new_tokens[0], skip_special_tokens=True
                        ),
                    }
                    answers = case.get("expected_answers", [])
                    if answers:
                        row["supplied_answer_exact_match"] = row["text"][
                            "new_route_response"
                        ].strip().casefold() in {a.strip().casefold() for a in answers}
                    padded = {
                        **inputs,
                        "text": torch.cat(
                            (
                                torch.full(
                                    (1, 3),
                                    tokenizer.pad_token_id,
                                    device=args.device,
                                    dtype=inputs["text"].dtype,
                                ),
                                inputs["text"],
                            ),
                            dim=1,
                        ),
                        "text_attention_mask": torch.cat(
                            (
                                torch.zeros(
                                    (1, 3),
                                    device=args.device,
                                    dtype=inputs["text_attention_mask"].dtype,
                                ),
                                inputs["text_attention_mask"],
                            ),
                            dim=1,
                        ),
                    }
                    padding_capture = _capture_image_boundary(model, padded, native)
                    row["text_padding_comparison"] = _hidden_difference(baseline, padding_capture)
                    _save_capture(
                        root / "traces" / case["id"] / "text_padding",
                        padding_capture,
                        fixture=fixture,
                    )
                    if "image" in inputs:
                        source_padded = {
                            **inputs,
                            "image": torch.cat(
                                (inputs["image"], torch.ones_like(inputs["image"][:, :1])), dim=1
                            ),
                            "image_mask": torch.cat(
                                (
                                    inputs["image_mask"],
                                    torch.zeros((1, 1), dtype=torch.bool, device=args.device),
                                ),
                                dim=1,
                            ),
                        }
                        source_padding_capture = _capture_image_boundary(
                            model, source_padded, native
                        )
                        row["masked_source_padding_comparison"] = _hidden_difference(
                            baseline, source_padding_capture
                        )
                        _save_capture(
                            root / "traces" / case["id"] / "masked_source_padding",
                            source_padding_capture,
                            fixture=fixture,
                        )
                        changed = {**inputs, "image": torch.zeros_like(inputs["image"])}
                        ablated = _capture_image_boundary(model, changed, native)
                        row["encoder_zero_pixels_sensitivity"] = _hidden_difference(
                            baseline, ablated
                        )
                        _save_capture(
                            root / "traces" / case["id"] / "encoder_zero_pixels",
                            ablated,
                            fixture=fixture,
                        )
                    if "comparison_prompt" in case or "comparison_source_images" in case:
                        alternate, alternate_native = _inputs(
                            case,
                            tokenizer,
                            bundle["source_transform"],
                            args.device,
                            dtype,
                            prompt=case.get("comparison_prompt"),
                            source_paths=case.get("comparison_source_images"),
                        )
                        changed = _capture_image_boundary(model, alternate, alternate_native)
                        row["supplied_counterfactual_sensitivity"] = _hidden_difference(
                            baseline, changed
                        )
                        _save_capture(
                            root / "traces" / case["id"] / "counterfactual",
                            changed,
                            fixture=fixture,
                        )
                    row["status"] = "completed"
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["duration_seconds"] = time.monotonic() - tick
                persist()
                _log(f"case {case['id']}: {row['status']}")
    finally:
        _log("fingerprint frozen parent parameters and buffers after inference")
        after = frozen_state_hashes(model, ("decoders.image.connector",))
        validation.write_json(root / "frozen-after.json", after)
        manifest["frozen_after_sha256"] = validation.sha256_file(root / "frozen-after.json")
        manifest["frozen_parent_unchanged"] = before == after
        manifest["changed_frozen_keys"] = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        manifest["generator_loaded"] = (
            getattr(model.decoders["image"].backend, "_pipeline", None) is not None
        )
        manifest["duration_seconds"] = time.monotonic() - started
        manifest["status"] = (
            "completed"
            if all(row["status"] == "completed" for row in manifest["cases"])
            and before == after
            and not manifest["generator_loaded"]
            else "failed"
        )
        scored = [
            row["supplied_answer_exact_match"]
            for row in manifest["cases"]
            if "supplied_answer_exact_match" in row
        ]
        manifest["supplied_case_score"] = {
            "count": len(scored),
            "exact_match_rate": sum(scored) / len(scored) if scored else None,
            "scope": "supplied fixed examples only; not a complete benchmark reproduction",
        }
        persist()
        (root / "report.md").write_text(
            f"# PRISM parent diagnostic\n\nExecution: **{manifest['status']}**. Frozen parent unchanged: {before == after}.\n\nNo input-alignment, P1 acceptance, or image quality claim. The real parent and new connector were exercised; the pretrained image sampler was not executed.\n\n"
            + "\n".join(
                f"- {row['id']}: {row['status']}; text route equality: {row.get('text', {}).get('exact_route_match', 'unavailable')}"
                for row in manifest["cases"]
            )
            + "\n"
        )
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "cases",
        "output-dir",
    ):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument("--device", required=True)
    result.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    result.add_argument("--max-new-tokens", type=int, default=16)
    result.add_argument("--max-cases", type=int, default=8)
    result.add_argument("--deterministic", action="store_true")
    result.add_argument("--attention-backend", choices=("current", "math"), default="current")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.output_dir.resolve()
    if root.exists():
        _log(f"blocked: output directory exists: {root}")
        return 2
    try:
        if not 1 <= args.max_new_tokens <= 32:
            raise ValueError("max-new-tokens must be between one and 32")
        cases = load_cases(args.cases, args.max_cases)
        for name in ("model_config", "checkpoint", "tokenizer", "source_processor"):
            getattr(args, name).resolve(strict=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from src.decoders.loading import load_image_training_bundle

        _log("load the full parent checkpoint and initialize only the new image connector")
        from tools.diagnose_image_decoder_repeatability import execution_policy

        with execution_policy(args):
            bundle = load_image_training_bundle(
                args.model_config, args.checkpoint, args.tokenizer, args.source_processor
            )
            report = run_parent_diagnostic(bundle, cases, args)
        return 0 if report["status"] == "completed" else 2
    except Exception as exc:
        root.mkdir(parents=True, exist_ok=True)
        validation.write_json(
            root / "failure.json",
            {
                "status": "blocked",
                "error": f"{type(exc).__name__}: {exc}",
                "alignment_gate": "not_evaluated",
                "p1_acceptance": "unproven",
            },
        )
        _log(f"blocked: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
