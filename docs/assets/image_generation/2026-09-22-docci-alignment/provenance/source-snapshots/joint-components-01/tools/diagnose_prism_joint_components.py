#!/usr/bin/env python3
"""Frozen 2x2 connector/DiT intervention, never a training or quality benchmark.

A schema-2 region connector is validated against the original DiT before the
completed joint checkpoint is restored. In-memory, detached runtime snapshots
then isolate the two factors. No activation cache or checkpoint is persisted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_prism_image_conditioning import (
    condition_metadata,
    paired_flow_losses,
    select_wrong_caption,
)
from tools.overfit_prism_image_connector import _phase, _write, execution_policy

EVIDENCE_KIND = "real_checkpoint_joint_component_diagnostic"
PHASES = (
    ("original_aligned", "original", "aligned", ("native_original", "aligned_original")),
    ("original_joint", "original", "joint", ("joint_original",)),
    ("joint_aligned", "joint", "aligned", ("native_joint", "aligned_joint")),
    ("joint_joint", "joint", "joint", ("joint_joint",)),
)
ROUTES = tuple(route for _, _, _, routes in PHASES for route in routes)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "alignment-checkpoint",
        "joint-checkpoint",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "connector-checkpoint-sha256",
        "alignment-checkpoint-sha256",
        "joint-checkpoint-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-connector-step", type=int, default=500)
    parser.add_argument("--expected-parent-tensors", type=int, default=526)
    parser.add_argument("--train-probe-count", type=int, default=2)
    parser.add_argument("--validation-probe-count", type=int, default=2)
    parser.add_argument("--flow-timesteps", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--prism-formats", nargs="+", choices=("chat",), default=["chat"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("math",), default="math")
    return parser


def validate_budget(args):
    fixed = {
        "expected_connector_step": 500,
        "expected_parent_tensors": 526,
        "train_probe_count": 2,
        "validation_probe_count": 2,
        "sample_count": 2,
        "sampling_steps": 50,
        "height": 256,
        "width": 256,
        "max_text_length": 1024,
        "seed": 42,
        "flow_timesteps": [0.1, 0.5, 0.9],
        "prism_formats": ["chat"],
        "dtype": "bfloat16",
        "attention_backend": "math",
        "deterministic": True,
    }
    if any(getattr(args, key) != value for key, value in fixed.items()):
        raise ValueError(
            "Component diagnostic requires the fixed reviewed 2/2, three-time, CFG5 protocol"
        )
    for key in (
        "connector_checkpoint_sha256",
        "alignment_checkpoint_sha256",
        "joint_checkpoint_sha256",
    ):
        value = getattr(args, key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("All three checkpoint SHA256 digests are mandatory lowercase hex")


def named_tensors(module):
    """Include nonpersistent buffers, which state_dict silently omits."""
    values = list(module.named_parameters()) + list(module.named_buffers())
    result = dict(values)
    if len(result) != len(values) or not result:
        raise ValueError("Expected distinct nonempty named parameters and buffers")
    return result


def state_hashes(module):
    from tools.train_image_decoder import _tensor_hash

    return {name: _tensor_hash(value) for name, value in named_tensors(module).items()}


def capture_state(module, *, connector=False):
    import torch

    result = {}
    for name, value in named_tensors(module).items():
        if connector and value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError("Connector runtime must retain FP32 weights")
        captured = value.detach().cpu().clone(memory_format=torch.contiguous_format)
        if captured.requires_grad or (
            value.device.type == "cpu" and captured.data_ptr() == value.data_ptr()
        ):
            raise RuntimeError("Runtime snapshot aliases live tensor storage")
        if not torch.isfinite(captured).all():
            raise ValueError("Cannot snapshot nonfinite component state")
        result[name] = captured
    return result


def restore_state(module, saved, expected):
    """Validate every named tensor and digest before any intentional component swap."""
    import torch
    from tools.train_image_decoder import _tensor_hash

    current = named_tensors(module)
    if set(current) != set(saved) or set(saved) != set(expected):
        raise ValueError("Component snapshot has incomplete named parameter/buffer coverage")
    for name, value in saved.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.requires_grad
            or value.shape != current[name].shape
            or value.dtype != current[name].dtype
            or not torch.isfinite(value).all()
            or _tensor_hash(value) != expected[name]
        ):
            raise ValueError("Component snapshot shape, dtype, content or digest differs")
    with torch.no_grad():
        for name, value in saved.items():
            current[name].copy_(value)
    if state_hashes(module) != expected:
        raise RuntimeError("Component restoration changed verified runtime bytes")


def validate_joint_lineage(joint_report, alignment_lineage, *, data_fingerprint, index_sha256):
    """Bind the counterfactual to the exact region initialization, not a similar run."""
    source = joint_report.get("alignment_initialization", {})
    required = (
        "sha256",
        "report_sha256",
        "checkpoint",
        "selection",
        "parent",
        "index_sha256",
        "data_fingerprint",
        "reference_checkpoint_sha256",
        "evidence_kind",
        "schema_version",
    )
    if any(
        key not in alignment_lineage or source.get(key) != alignment_lineage[key]
        for key in required
    ):
        raise ValueError("Joint source alignment lineage differs from verified region checkpoint")
    settings = joint_report.get("settings", {})
    if (
        joint_report.get("status") != "completed"
        or joint_report.get("completed_steps") != 6
        or joint_report.get("frozen_state_unchanged") is not True
        or joint_report.get("data_fingerprint") != data_fingerprint
        or {split: joint_report.get(split + "_index_sha256") for split in ("train", "validation")}
        != index_sha256
        or settings.get("prompt_format") != "chat"
        or settings.get("steps") != 6
        or joint_report.get("train_selection") != alignment_lineage["selection"]["train"]
        or source.get("evidence_kind")
        != "real_checkpoint_connector_native_region_feature_alignment"
        or source.get("schema_version") != 2
    ):
        raise ValueError(
            "Expected completed six-update joint run with the exact region training cohort"
        )


def feature_pair(model, tokenizer, backend, prompt, original_norm, *, args):
    """Re-encode exact caption tokens; evaluate both connectors under ORIGINAL norm."""
    import torch
    from tools.align_prism_image_conditioning import (
        caption_region_masks,
        encode_observed_native,
        validate_pair,
    )
    from tools.prism_image_conditioning import encode_prism_prompt
    from tools.train_image_decoder import _tensor_hash
    from torch.nn import functional as F

    prism = encode_prism_prompt(
        model,
        tokenizer,
        prompt,
        device=args.device,
        mode="chat",
        max_text_length=args.max_text_length,
    )
    native = encode_observed_native(backend, prompt, max_text_length=args.max_text_length)
    raw = tokenizer(
        [prompt], padding=False, truncation=False, add_special_tokens=False, return_tensors="pt"
    )
    if not raw["attention_mask"].bool().all():
        raise ValueError("Raw caption must not contain padding")
    content, _, span = validate_pair(prism, native, raw["input_ids"])
    masks = caption_region_masks(prism["input_attention_mask"], content, span)
    # Match the actual backend conditioning dtype before any hash checks.
    prism["embeds"] = prism["embeds"].to(native["embeds"])
    teacher = original_norm(native["embeds"]).float()
    predicted = original_norm(prism["embeds"].to(native["embeds"])).float()
    if (
        teacher.shape != predicted.shape
        or not torch.isfinite(teacher).all()
        or not torch.isfinite(predicted).all()
    ):
        raise ValueError("Original-norm feature metrics must be aligned and finite")
    parts = {
        region: {
            "mse": float((predicted[mask] - teacher[mask]).square().mean()),
            "cosine": float(F.cosine_similarity(predicted[mask], teacher[mask], dim=-1).mean()),
            "tokens": int(mask.sum()),
        }
        for region, mask in masks.items()
    }
    audit = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "formatted_prompt_sha256": hashlib.sha256(prism["formatted_prompt"].encode()).hexdigest(),
        "input_ids_sha256": _tensor_hash(native["input_ids"]),
        "input_mask_sha256": _tensor_hash(native["input_attention_mask"]),
        "input_token_ids": native["input_ids"].cpu().tolist(),
        "prism_hidden_sha256": _tensor_hash(prism["hidden_states"]),
        "native_features_sha256": _tensor_hash(native["embeds"]),
        "teacher_normalized_sha256": _tensor_hash(teacher),
        "content_span": span,
        "exact_formatted_input_match": True,
        "exact_input_ids_match": True,
        "exact_input_masks_match": True,
        "actual_native_forward_inputs_verified": True,
        "target_pixels_read": False,
    }
    return (
        prism,
        native,
        {"normalization": "captured_original_dit_rmsnorm", "partitions": parts, "audit": audit},
    )


def runtime_condition_metadata(value):
    from tools.train_image_decoder import _tensor_hash

    return condition_metadata(value) | {
        "embeds_sha256": _tensor_hash(value["embeds"]),
        "attention_mask_sha256": _tensor_hash(value["attention_mask"]),
    }


def merge_flow(report_rows, result, *, phase):
    """Require actual cross-phase inputs to match, not merely the requested seed."""
    key = (result["split"], result["index"], result["repeat"])
    if result.get("actual_conditioning_verified") is not True:
        raise RuntimeError("Flow phase has not verified actual conditioning")
    if key not in report_rows:
        report_rows[key] = copy.deepcopy(result) | {"phases": [phase]}
        return
    previous = report_rows[key]
    for field in (
        "split",
        "index",
        "id",
        "wrong_id",
        "repeat",
        "seed",
        "requested_timestep",
        "actual_inputs",
    ):
        if previous.get(field) != result.get(field):
            raise RuntimeError("Component phases changed actual paired inputs or caption identity")
    if set(previous["routes"]) & set(result["routes"]):
        raise RuntimeError("Duplicate component flow route")
    previous["routes"].update(result["routes"])
    previous["phases"].append(phase)


def verify_cfg5_trace(trace, positive, negative):
    """Assert both actual CFG branches and preserve compact tensor trace hashes."""
    import torch
    from tools.train_image_decoder import _tensor_hash

    for label, value, branch in (("positive", positive, 0), ("negative", negative, 1)):
        for condition_key, mask_key in (
            (f"condition.{label}", f"mask.{label}"),
            (f"condition.branch{branch}", f"mask.branch{branch}"),
        ):
            if not isinstance(trace.get(condition_key), torch.Tensor) or not isinstance(
                trace.get(mask_key), torch.Tensor
            ):
                raise RuntimeError("Sampling trace is missing a CFG5 conditioning branch")
            if _tensor_hash(trace[condition_key]) != _tensor_hash(value["embeds"]) or _tensor_hash(
                trace[mask_key]
            ) != _tensor_hash(value["attention_mask"]):
                raise RuntimeError("Sampling changed an audited CFG5 condition or mask")
    if {key for key in trace if key.startswith("condition.branch")} != {
        "condition.branch0",
        "condition.branch1",
    }:
        raise RuntimeError("Expected exactly two text CFG5 branches")
    for key in ("latents.initial", "latents.final", "prediction.step0", "schedule.timesteps"):
        if not isinstance(trace.get(key), torch.Tensor) or not torch.isfinite(trace[key]).all():
            raise RuntimeError("Sampling trace lacks finite latent/prediction tensors")
    return {
        key: _tensor_hash(value) for key, value in trace.items() if isinstance(value, torch.Tensor)
    }


def sample_route(
    backend, prompt, positive, negative, *, route, native, args, case_id, seed, saved_latent
):
    import torch
    from src.decoders.loading import file_sha256
    from tools.train_image_decoder import _seed_everything, _tensor_hash
    from tools.train_prism_image_connector import preserved_rng

    with preserved_rng(), torch.no_grad():
        _seed_everything(seed)
        options = {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.sampling_steps,
            "text_guidance_scale": 5.0,
            "image_guidance_scale": 1.0,
            "negative_prompt": "",
            "max_sequence_length": args.max_text_length,
            "generator": torch.Generator(device=args.device).manual_seed(seed),
            "trace": True,
        }
        if saved_latent is not None:
            options["latents"] = saved_latent.to(
                device=args.device, dtype=getattr(torch, args.dtype)
            ).clone()
        images = (
            backend.generate_reference({"prompt": prompt}, **options)
            if native
            else backend.generate_conditioned(
                positive["embeds"], positive["attention_mask"], **options
            )
        )
        trace = verify_cfg5_trace(backend.last_trace, positive, negative)
        actual = backend.last_trace["latents.initial"].detach().cpu().clone()
        if saved_latent is not None and _tensor_hash(actual) != _tensor_hash(saved_latent):
            raise RuntimeError("Component sampling changed common initial noise")
        images = images.images if hasattr(images, "images") else images
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise RuntimeError("Expected one image for each component route")
        path = args.output_dir / f"sample-{case_id}-{route}.png"
        images[0].save(path)
        backend.last_trace = {}
        return {
            "case_id": case_id,
            "route": route,
            "path": str(path),
            "sha256": file_sha256(path),
            "seed": seed,
            "initial_latent_sha256": _tensor_hash(actual),
            "prompt": prompt,
            "text_guidance_scale": 5.0,
            "sampling_steps": args.sampling_steps,
            "target_free": True,
            "quality_claim": False,
            "actual_condition_verified": True,
            "negative_conditioner": "original_frozen_native",
            "trace_sha256": trace,
        }, actual


def main(argv=None):
    args = _parser().parse_args(argv)
    validate_budget(args)
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Component diagnosis must run in a PBS compute allocation")
    with execution_policy(args):
        return _run(args)


def _run(args):
    import torch
    from src.data.image_generation_webdataset import ImageGenerationWebDataset
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.align_prism_image_conditioning import caption_norm, encode_observed_native
    from tools.prism_image_alignment_checkpoint import restore_alignment_connector
    from tools.prism_image_conditioning import summarize_caption_controls
    from tools.train_image_decoder import _seed_everything, frozen_state_hashes
    from tools.train_prism_image_connector import validate_repeatability_report
    from tools.train_prism_image_diffusion import (
        MODULES,
        configure_joint_scope,
        restore_joint_stage,
        restore_warm_connector,
    )

    for key, value in vars(args).items():
        if isinstance(value, Path) and key != "output_dir":
            setattr(args, key, value.resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("Component diagnostic output directory must be new")
    precheck = validate_repeatability_report(args.repeatability_report, args)
    datasets = {
        split: ImageGenerationWebDataset(index, target_size=(args.height, args.width), split=split)
        for split, index in (("train", args.train_index), ("validation", args.validation_index))
    }
    if (
        any(len(dataset) < 3 for dataset in datasets.values())
        or datasets["train"].data_fingerprint != datasets["validation"].data_fingerprint
        or any(
            row.task != "t2i" or row.source_ids or row.source_paths or row.split != split
            for split, dataset in datasets.items()
            for row in dataset.records
        )
        or {row.id for row in datasets["train"].records}
        & {row.id for row in datasets["validation"].records}
    ):
        raise ValueError("Expected disjoint source-free train/validation T2I conversion")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    started = time.monotonic()
    source_files = (
        "tools/diagnose_prism_image_conditioning.py",
        "tools/prism_image_conditioning.py",
        "tools/align_prism_image_conditioning.py",
        "tools/prism_image_alignment_checkpoint.py",
        "tools/train_prism_image_diffusion.py",
        "tools/train_prism_image_connector.py",
        "tools/train_image_decoder.py",
        "tools/overfit_prism_image_connector.py",
        "src/data/image_generation_webdataset.py",
        "src/model.py",
        "src/decoders/conditioning.py",
        "src/decoders/image.py",
        "src/decoders/omnigen2_backend.py",
        "src/decoders/loading.py",
    )
    report = {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "qualification": "unqualified",
        "status": "running",
        "completed_steps": 0,
        "quality_benchmark": False,
        "training_performed": False,
        "cache_saved": False,
        "checkpoints_saved": False,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runner_sha256": file_sha256(__file__),
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
        "repeatability_precheck": precheck,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "host": platform.node(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "data_fingerprint": datasets["train"].data_fingerprint,
        "index_sha256": {split: file_sha256(dataset.index) for split, dataset in datasets.items()},
        "component_protocol": {
            "routes": list(ROUTES),
            "route_name": "connector_or_native + '_' + DiT",
            "phases": [
                {"name": name, "dit": dit, "connector": connector, "routes": list(routes)}
                for name, dit, connector, routes in PHASES
            ],
            "intentional_runtime_swaps": True,
            "runtime_dit_snapshot_dtype": args.dtype,
            "connector_snapshot_dtype": "float32",
            "optimizer_created": False,
            "feature_normalization": "captured_original_dit_rmsnorm",
            "sampling": "CFG5 with original frozen native negative; matched initial noise",
            "final_state": "original_dit_aligned_connector",
            "teacher_cache_persisted": False,
        },
        "samples": [],
        "flow_controls": [],
        "feature_drift": [],
        "condition_statistics": [],
        "phase_audits": [],
        "final_state_restored": False,
    }
    model = None
    snapshots = {}
    baseline = None
    invariant = None
    try:
        _phase(report, report_path, "loading_parent")
        _seed_everything(args.seed)
        bundle = load_image_training_bundle(
            args.model_config, args.checkpoint, args.tokenizer, args.source_processor
        )
        model, tokenizer = bundle["model"], bundle["tokenizer"]
        report["parent"] = bundle["provenance"]
        if bundle["provenance"].get("fixture"):
            report["evidence_kind"] = "fixture_only"
        restoration = bundle["provenance"].get("restoration", {})
        if (
            restoration.get("strict_parent") is not True
            or restoration.get("loaded_key_count") != args.expected_parent_tensors
            or restoration.get("missing_parent_keys") != []
            or restoration.get("unexpected_keys") != []
        ):
            raise ValueError("Expected strict complete parent restoration")
        model.to(device=args.device, dtype=getattr(torch, args.dtype))
        connector = model.decoders["image"].connector.float()
        backend = model.decoders["image"].backend
        backend.ensure_loaded()
        report["generator"] = backend.checkpoint_manifest()
        report["generator_runtime"] = backend.provenance()
        validate_repeatability_report(
            args.repeatability_report,
            args,
            checkpoint_manifest=report["generator"],
            backend=report["generator_runtime"],
        )
        report["connector_warm_start"] = restore_warm_connector(
            model,
            args.connector_checkpoint,
            parent=report["parent"],
            reference_sha256=report["generator"]["manifest_sha256"],
            data_fingerprint=report["data_fingerprint"],
            frozen_hashes=frozen_state_hashes(model, (MODULES[0],)),
            expected_step=args.expected_connector_step,
            expected_sha256=args.connector_checkpoint_sha256,
            fixture=report["evidence_kind"] == "fixture_only",
        )
        model.eval().requires_grad_(False)
        backend.configure_training(train_diffusion=False, gradient_checkpointing=False)
        _phase(report, report_path, "strict_region_restore_on_original_dit")
        report["alignment_checkpoint"] = restore_alignment_connector(
            model,
            args.alignment_checkpoint,
            parent=report["parent"],
            generator_manifest=report["generator"],
            data_fingerprint=report["data_fingerprint"],
            index_sha256=report["index_sha256"],
            records={split: dataset.records for split, dataset in datasets.items()},
            tokenizer=tokenizer,
            expected_sha256=args.alignment_checkpoint_sha256,
            fixture=report["evidence_kind"] == "fixture_only",
        )
        joint_report = json.loads((args.joint_checkpoint.parent / "report.json").read_text())
        validate_joint_lineage(
            joint_report,
            report["alignment_checkpoint"],
            data_fingerprint=report["data_fingerprint"],
            index_sha256=report["index_sha256"],
        )
        original_norm = copy.deepcopy(caption_norm(backend)).eval().requires_grad_(False)
        norm_hashes = state_hashes(original_norm)
        report["original_caption_norm_sha256"] = norm_hashes
        snapshots["connector_aligned"] = capture_state(connector, connector=True)
        snapshots["dit_original"] = capture_state(backend.transformer)
        report["state_identities"] = {
            "connector_aligned": state_hashes(connector),
            "dit_original": state_hashes(backend.transformer),
        }
        baseline = frozen_state_hashes(model, ())
        invariant = frozen_state_hashes(model, MODULES)
        report["baseline_hashes"] = baseline
        report["invariant_frozen_hashes_before"] = invariant
        _phase(report, report_path, "strict_joint_restore")
        _, report["joint_checkpoint"] = restore_joint_stage(
            model,
            args.joint_checkpoint,
            named_groups=configure_joint_scope(model),
            parent=report["parent"],
            reference_sha256=report["generator"]["manifest_sha256"],
            frozen_hashes=invariant,
            expected_sha256=args.joint_checkpoint_sha256,
            fixture=report["evidence_kind"] == "fixture_only",
            restore_masters=False,
        )
        # configure_joint_scope temporarily enables gradients; no forward occurs before this reset.
        backend.configure_training(train_diffusion=False, gradient_checkpointing=False)
        model.eval().requires_grad_(False)
        snapshots["connector_joint"] = capture_state(connector, connector=True)
        snapshots["dit_joint"] = capture_state(backend.transformer)
        report["state_identities"].update(
            connector_joint=state_hashes(connector), dit_joint=state_hashes(backend.transformer)
        )
        train_pool = [row["index"] for row in report["alignment_checkpoint"]["selection"]["train"]]
        selections = {"train": train_pool[:2], "validation": [0, 1]}
        report["selection"] = {
            split: [{"index": index, "id": datasets[split].records[index].id} for index in indices]
            for split, indices in selections.items()
        }
        flow_rows, sample_latents, feature_audits = {}, {}, {}
        negative_identity, sampling_schedule = None, None
        with torch.no_grad():
            for phase, dit, connector_kind, routes in PHASES:
                _phase(report, report_path, phase)
                for module, key in (
                    (backend.transformer, "dit_" + dit),
                    (connector, "connector_" + connector_kind),
                ):
                    restore_state(module, snapshots[key], report["state_identities"][key])
                if frozen_state_hashes(model, MODULES) != invariant:
                    raise RuntimeError("Intentional swap mutated PRISM/VAE/native conditioner")
                if any(module.training for module in model.modules()) or any(
                    value.requires_grad for value in model.parameters()
                ):
                    raise RuntimeError("All component routes must remain frozen and eval")
                phase_before = frozen_state_hashes(model, ())
                audit = {
                    "phase": phase,
                    "dit": dit,
                    "connector": connector_kind,
                    "routes": list(routes),
                    "hashes_before": phase_before,
                    "all_frozen_eval": True,
                }
                report["phase_audits"].append(audit)
                negative = encode_observed_native(backend, "", max_text_length=args.max_text_length)
                audit["native_negative"] = runtime_condition_metadata(negative)
                if negative_identity is not None and negative_identity != audit["native_negative"]:
                    raise RuntimeError(
                        "Native negative conditioning changed across component phases"
                    )
                negative_identity = audit["native_negative"]
                for split, indices in selections.items():
                    for position, index in enumerate(indices):
                        row = datasets[split].records[index]
                        wrong_index = select_wrong_caption(
                            datasets[split].records,
                            index,
                            train_pool if split == "train" else list(range(len(datasets[split]))),
                        )
                        wrong = datasets[split].records[wrong_index]
                        conditions = {route: {} for route in routes}
                        for kind, caption in (("matched", row), ("wrong", wrong)):
                            prism, native, feature = feature_pair(
                                model, tokenizer, backend, caption.prompt, original_norm, args=args
                            )
                            feature.update(
                                phase=phase,
                                connector=connector_kind,
                                dit=dit,
                                split=split,
                                id=row.id,
                                caption_kind=kind,
                                caption_id=caption.id,
                            )
                            caption_key = (split, caption.id)
                            if (
                                caption_key in feature_audits
                                and feature_audits[caption_key] != feature["audit"]
                            ):
                                raise RuntimeError(
                                    "Frozen caption tokens/features changed across component phases"
                                )
                            feature_audits[caption_key] = feature["audit"]
                            report["feature_drift"].append(feature)
                            for route in routes:
                                conditions[route][kind] = (
                                    native if route.startswith("native_") else prism
                                )
                        report["condition_statistics"].append(
                            {
                                "phase": phase,
                                "split": split,
                                "id": row.id,
                                "routes": {
                                    route: {
                                        kind: runtime_condition_metadata(value)
                                        for kind, value in pair.items()
                                    }
                                    for route, pair in conditions.items()
                                },
                            }
                        )
                        target = datasets[split][index]["target_image"].unsqueeze(0).to(args.device)
                        for repeat, timestep in enumerate(args.flow_timesteps):
                            result = paired_flow_losses(
                                backend,
                                target,
                                conditions,
                                seed=args.seed
                                + 100000
                                + (0 if split == "train" else 50000)
                                + index * 8
                                + repeat,
                                timestep=timestep,
                                device=args.device,
                                verify_conditioning=True,
                            )
                            result.update(
                                split=split,
                                index=index,
                                id=row.id,
                                wrong_id=wrong.id,
                                repeat=repeat,
                            )
                            merge_flow(flow_rows, result, phase=phase)
                        report["flow_controls"] = list(flow_rows.values())
                        if split == "validation":
                            for route in routes:
                                sample, latent = sample_route(
                                    backend,
                                    row.prompt,
                                    conditions[route]["matched"],
                                    negative,
                                    route=route,
                                    native=route.startswith("native_"),
                                    args=args,
                                    case_id=f"validation-{position:02d}",
                                    seed=args.seed + 200000 + position,
                                    saved_latent=sample_latents.get(position),
                                )
                                schedule = sample["trace_sha256"]["schedule.timesteps"]
                                if sampling_schedule is not None and sampling_schedule != schedule:
                                    raise RuntimeError(
                                        "Component sampling changed actual scheduler timesteps"
                                    )
                                sampling_schedule = schedule
                                report["samples"].append(sample)
                                sample_latents[position] = latent
                        _write(report_path, report)
                        del conditions, target
                audit["hashes_after"] = frozen_state_hashes(model, ())
                audit["frozen_state_unchanged"] = audit["hashes_after"] == phase_before
                audit["invariant_frozen_state_unchanged"] = (
                    frozen_state_hashes(model, MODULES) == invariant
                )
                audit["original_norm_unchanged"] = state_hashes(original_norm) == norm_hashes
                if not all(
                    audit[key]
                    for key in (
                        "frozen_state_unchanged",
                        "invariant_frozen_state_unchanged",
                        "original_norm_unchanged",
                    )
                ):
                    raise RuntimeError("Frozen component phase mutated a tensor")
                _write(report_path, report)
        if len(report["flow_controls"]) != 12 or any(
            set(row["routes"]) != set(ROUTES) or len(row["phases"]) != 4
            for row in report["flow_controls"]
        ):
            raise RuntimeError("Incomplete 2x2 paired flow route coverage")
        if len(report["samples"]) != 12 or any(
            {
                row["route"]
                for row in report["samples"]
                if row["case_id"] == f"validation-{case:02d}"
            }
            != set(ROUTES)
            for case in range(2)
        ):
            raise RuntimeError("Incomplete 2x2 image route coverage")
        report["summary"] = summarize_caption_controls(report["flow_controls"])
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if baseline is not None:
            try:
                restore_state(
                    model.decoders["image"].backend.transformer,
                    snapshots["dit_original"],
                    report["state_identities"]["dit_original"],
                )
                restore_state(
                    model.decoders["image"].connector,
                    snapshots["connector_aligned"],
                    report["state_identities"]["connector_aligned"],
                )
                model.decoders["image"].backend.configure_training(
                    train_diffusion=False, gradient_checkpointing=False
                )
                model.eval().requires_grad_(False)
                report["final_hashes"] = frozen_state_hashes(model, ())
                report["final_state_restored"] = report["final_hashes"] == baseline
                report["invariant_frozen_hashes_after"] = frozen_state_hashes(model, MODULES)
                report["invariant_frozen_state_unchanged"] = (
                    report["invariant_frozen_hashes_after"] == invariant
                )
                if (
                    not report["final_state_restored"]
                    or not report["invariant_frozen_state_unchanged"]
                ):
                    raise RuntimeError("Final original-DiT/aligned-connector restoration failed")
            except Exception as restoration_error:
                report.update(
                    status="failed",
                    restoration_error=f"{type(restoration_error).__name__}: {restoration_error}",
                )
                _write(report_path, report)
                raise
        report.update(phase=report["status"], duration_seconds=time.monotonic() - started)
        _write(report_path, report)
    return report


if __name__ == "__main__":
    main()
