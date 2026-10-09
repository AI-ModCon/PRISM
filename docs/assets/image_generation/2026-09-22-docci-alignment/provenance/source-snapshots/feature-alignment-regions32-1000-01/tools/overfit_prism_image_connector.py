#!/usr/bin/env python3
"""Bounded connector overfit diagnostic; never a production/P2 qualification.

The 16-pair experiment deliberately has a separate artifact kind from the gated
training runner. All checkpoints remain unqualified, including after a successful
500-step run. Targets are used only by the explicit flow-loss path. Native,
untrained and trained sample calls share saved initial noise and never see targets.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE_KIND = "real_checkpoint_connector_overfit_diagnostic"
MODULES = ("decoders.image.connector",)


def validate_budget(steps, train_count, validation_count, height, width, sampling_steps):
    if not 1 <= steps <= 1000:
        raise ValueError("Overfit diagnostic requires 1–1000 total optimizer steps")
    if not 1 <= train_count <= 16 or not 1 <= validation_count <= 8:
        raise ValueError("Overfit diagnostic permits 1–16 train and 1–8 validation pairs")
    if any(size < 32 or size > 256 or size % 16 for size in (height, width)):
        raise ValueError("Diagnostic dimensions must be multiples of 16 in [32, 256]")
    if not 1 <= sampling_steps <= 50:
        raise ValueError("Diagnostic sampling requires 1–50 denoising steps")


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _phase(report, path, name, **values):
    report.update(phase=name, **values)
    _write(path, report)
    print(json.dumps({"phase": name, "completed_steps": report["completed_steps"]}), flush=True)


def _restore_rng(state):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_xpu" in state:
        torch.xpu.set_rng_state_all(state["torch_xpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if "numpy" in state:
        import numpy as np

        algorithm, keys, pos, has_gauss, cached_gauss = state["numpy"]
        np.random.set_state(
            (algorithm, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gauss)
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "validation-manifest",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--manifest", "--train-manifest", dest="train_manifest", required=True, type=Path
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=500, help="Total steps including resumed work")
    parser.add_argument(
        "--sample-every", "--checkpoint-every", dest="checkpoint_every", type=int, default=100
    )
    parser.add_argument("--probe-count", type=int, default=16, help="Fixed examples per split")
    parser.add_argument(
        "--sample-count", type=int, default=2, help="Gallery cases per split (1 or 2)"
    )
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--conditioning-ablation",
        action="store_true",
        help="Compare fixed T2I probes to distinct same-split captions at identical noise",
    )
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="default")
    return parser


@contextmanager
def execution_policy(args):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    attention = sdpa_kernel(SDPBackend.MATH) if args.attention_backend == "math" else nullcontext()
    try:
        torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
        with attention:
            yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def main(argv=None):
    args = _parser().parse_args(argv)
    with execution_policy(args):
        return _run(args)


def _run(args):
    for name in (
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "train_manifest",
        "validation_manifest",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("Output directory must be new, including when resuming")
    if not 1 <= args.checkpoint_every <= 100 or not 1 <= args.probe_count <= 16:
        raise ValueError("checkpoint-every must be 1–100 and probe-count 1–16")
    if not 1 <= args.sample_count <= 2:
        raise ValueError("sample-count must be 1–2 cases per split")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if any(
        not math.isfinite(x) or x < 0 for x in (args.text_guidance_scale, args.image_guidance_scale)
    ):
        raise ValueError("Guidance scales must be finite and nonnegative")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import torch
    from src.data.image_generation import (
        ImageGenerationCollator,
        ImageGenerationDataset,
        load_image_generation_manifest,
        validate_manifest_splits,
    )
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.train_image_decoder import (
        _configure_connector_only,
        _rng_state,
        _seed_everything,
        _tensor_hash,
        _to_device,
        frozen_state_hashes,
    )

    train_records = load_image_generation_manifest(args.train_manifest)
    validation_records = load_image_generation_manifest(args.validation_manifest)
    validate_budget(
        args.steps,
        len(train_records),
        len(validation_records),
        args.height,
        args.width,
        args.sampling_steps,
    )
    if any(row.split != "train" for row in train_records):
        raise ValueError("Train manifest must contain only train rows")
    if any(row.split not in {"validation", "val", "diagnostic"} for row in validation_records):
        raise ValueError("Validation rows require validation, val or diagnostic split")
    if len({row.split for row in validation_records}) != 1:
        raise ValueError("Validation manifest must have one consistent split")
    # Validate together: constructing the two datasets separately cannot detect leakage.
    fingerprints = validate_manifest_splits(train_records + validation_records)
    if args.sample_count > min(len(train_records), len(validation_records)):
        raise ValueError("sample-count exceeds available distinct examples")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "qualification": "unqualified",
        "status": "running",
        "p0_p1_gate": "not_established_by_this_diagnostic",
        "p2_gate": "not_evaluated",
        "quality_benchmark": False,
        "parent_heldout_visual_quality_established": False,
        "source_image_conditioning_exercised": any(row.source_paths for row in train_records),
        "completed_steps": 0,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": str(torch.__version__),
        "numerical_runtime": {
            "torch_version": str(torch.__version__),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "environment": {
                name: os.environ[name]
                for name in (
                    "ZE_AFFINITY_MASK",
                    "ZE_FLAT_DEVICE_HIERARCHY",
                    "ONEAPI_DEVICE_SELECTOR",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "ONEDNN_DEFAULT_FPMATH_MODE",
                )
                if name in os.environ
            },
        },
        "python_version": platform.python_version(),
        "host": platform.node(),
        "runner_sha256": file_sha256(__file__),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/model.py",
                "src/decoders/image.py",
                "src/decoders/conditioning.py",
                "src/decoders/types.py",
                "src/decoders/omnigen2_backend.py",
                "src/decoders/loading.py",
                "src/data/image_generation.py",
                "tools/train_image_decoder.py",
            )
        },
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "validation_manifest_sha256": file_sha256(args.validation_manifest),
        "image_content_fingerprints": fingerprints,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "data_order": "manifest order, round robin; global step selects example",
        "probe_protocol": "Reset complete RNG before each forward, including VAE posterior, flow noise and timestep; fixed per-example seeds independent of training seeds",
        "sample_protocol": "Native initial latent tensors replayed exactly for target-free PRISM samples; matched guidance, negative prompt and sampling steps",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "evaluations": [],
        "samples": [],
        "checkpoints": [],
    }
    model = None
    before = None
    nonzero_names = set()
    try:
        _phase(report, report_path, "loading_parent")
        _seed_everything(args.seed)
        bundle = load_image_training_bundle(
            args.model_config, args.checkpoint, args.tokenizer, args.source_processor
        )
        model = bundle["model"]
        report["parent"] = bundle["provenance"]
        model.to(device=args.device, dtype=getattr(torch, args.dtype))
        _phase(report, report_path, "loading_generator")
        backend = model.decoders["image"].backend
        backend.ensure_loaded()
        report["generator"] = backend.checkpoint_manifest()
        report["generator_runtime"] = backend.provenance()
        model.decoders["image"].connector.float()
        trainable = _configure_connector_only(model, MODULES)
        before = frozen_state_hashes(model, MODULES)
        report["frozen_hashes_before"] = before
        report["trainable_parameters"] = {
            name: list(value.shape) for name, value in trainable.items()
        }
        report["trainable_parameter_count"] = sum(value.numel() for value in trainable.values())
        report["trainable_parameter_dtypes"] = {
            name: str(value.dtype) for name, value in trainable.items()
        }
        datasets = {
            split: ImageGenerationDataset(
                manifest,
                source_transform=bundle["source_transform"],
                target_size=(args.height, args.width),
                split=None,
            )
            for split, manifest in (
                ("train", args.train_manifest),
                ("validation", args.validation_manifest),
            )
        }
        records = {"train": train_records, "validation": validation_records}
        report["data_fingerprints"] = {
            split: dataset.data_fingerprint for split, dataset in datasets.items()
        }
        sampling_datasets = {}
        for split, dataset in datasets.items():
            sampling_dataset = copy.copy(dataset)
            sampling_dataset.records = [
                replace(record, target_path=None) for record in dataset.records
            ]
            sampling_datasets[split] = sampling_dataset
        collator = ImageGenerationCollator(bundle["tokenizer"])
        optimizer = torch.optim.AdamW(
            list(trainable.values()), lr=args.learning_rate, weight_decay=0.0
        )
        protocol = {
            "parent": bundle["provenance"],
            "reference_checkpoint_sha256": report["generator"]["manifest_sha256"],
            "data_fingerprints": report["data_fingerprints"],
            "runner_sha256": report["runner_sha256"],
            "numerical_runtime": report["numerical_runtime"],
            "generator_runtime": report["generator_runtime"],
            "source_sha256": report["source_sha256"],
            "settings": {
                key: value
                for key, value in report["settings"].items()
                if key not in {"steps", "output_dir", "resume"}
            },
        }
        report["resume_protocol"] = protocol
        sample_cases = [
            (split, index)
            for index in range(args.sample_count)
            for split in ("train", "validation")
        ]
        initial_latents = {}
        initial_reference_hashes = {}

        def batch_at(split, index):
            return _to_device(collator([datasets[split][index]]), torch.device(args.device))

        def forward(batch):
            return model.forward_outputs(
                batch["inputs"],
                targets=batch["targets"],
                requested_outputs=["image"],
                native_context=batch["native_context"],
                output_specs=batch["output_specs"],
            ).losses["image"]

        def evaluate(step):
            _phase(report, report_path, "fixed_probes")
            evaluation = {"step": step, "splits": {}}
            with torch.no_grad():
                for split_index, split in enumerate(("train", "validation")):
                    entries = []
                    for index in range(min(args.probe_count, len(datasets[split]))):
                        probe_seed = args.seed + 100000 + 1009 * split_index + index
                        item = datasets[split][index]
                        batch = _to_device(collator([item]), torch.device(args.device))
                        _seed_everything(probe_seed)
                        loss = forward(batch)
                        if loss.ndim or not torch.isfinite(loss):
                            raise RuntimeError("Fixed probe requires finite scalar loss")
                        entry = {
                            "id": records[split][index].id,
                            "seed": probe_seed,
                            "loss": float(loss),
                        }
                        if args.conditioning_ablation:
                            record = records[split][index]
                            candidates = [
                                records[split][(index + offset) % len(records[split])]
                                for offset in range(1, len(records[split]))
                            ]
                            other = next(
                                (
                                    candidate
                                    for candidate in candidates
                                    if candidate.task == "t2i" and candidate.prompt != record.prompt
                                ),
                                None,
                            )
                            applicable = record.task == "t2i" and other is not None
                            entry["conditioning_ablation_applicable"] = applicable
                            if applicable:
                                shuffled_batch = _to_device(
                                    collator([dict(item, prompt=other.prompt)]),
                                    torch.device(args.device),
                                )
                                if not torch.equal(
                                    batch["targets"]["image"], shuffled_batch["targets"]["image"]
                                ):
                                    raise RuntimeError(
                                        "Conditioning ablation changed target pixels"
                                    )
                                # Reset the entire forward, including posterior/noise/timestep.
                                _seed_everything(probe_seed)
                                shuffled_loss = forward(shuffled_batch)
                                if shuffled_loss.ndim or not torch.isfinite(shuffled_loss):
                                    raise RuntimeError(
                                        "Caption ablation requires finite scalar loss"
                                    )
                                entry.update(
                                    shuffled_prompt_id=other.id,
                                    shuffled_prompt=other.prompt,
                                    shuffled_loss=float(shuffled_loss),
                                    shuffled_minus_correct=float(shuffled_loss) - float(loss),
                                )
                            else:
                                entry["conditioning_ablation_skip_reason"] = (
                                    "not_t2i" if record.task != "t2i" else "no_distinct_t2i_caption"
                                )
                        entries.append(entry)
                    evaluation["splits"][split] = {
                        "mean_loss": sum(row["loss"] for row in entries) / len(entries),
                        "examples": entries,
                    }
                    if args.conditioning_ablation:
                        applicable_rows = [row for row in entries if "shuffled_loss" in row]
                        evaluation["splits"][split]["conditioning_ablation"] = {
                            "applicable_count": len(applicable_rows),
                            "mean_shuffled_loss": (
                                sum(row["shuffled_loss"] for row in applicable_rows)
                                / len(applicable_rows)
                                if applicable_rows
                                else None
                            ),
                            "mean_shuffled_minus_correct": (
                                sum(row["shuffled_minus_correct"] for row in applicable_rows)
                                / len(applicable_rows)
                                if applicable_rows
                                else None
                            ),
                            "interpretation": "Positive gap favors the matched caption; diagnostic only",
                        }
            report["evaluations"].append(evaluation)
            _write(report_path, report)
            return evaluation

        def sample(stage, step, native=False):
            _phase(report, report_path, "sampling_" + stage)
            for case_number, (split, index) in enumerate(sample_cases):
                record = records[split][index]
                # This separate view never opens target pixels or builds target tensors.
                item = sampling_datasets[split][index]
                batch = _to_device(collator([item]), torch.device(args.device))
                if batch["targets"] or batch["output_specs"]:
                    raise RuntimeError("Sampling batch leaked supervision")
                sample_seed = args.seed + 200000 + case_number
                _seed_everything(sample_seed)
                options = {
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.sampling_steps,
                    "text_guidance_scale": args.text_guidance_scale,
                    "image_guidance_scale": args.image_guidance_scale,
                    "negative_prompt": args.negative_prompt,
                    "generator": torch.Generator(device=args.device).manual_seed(sample_seed),
                    "trace": True,
                }
                if not native:
                    options["latents"] = (
                        initial_latents[str(case_number)]
                        .to(device=args.device, dtype=getattr(torch, args.dtype))
                        .clone()
                    )
                with torch.no_grad():
                    if native:
                        context = dict(batch["native_context"]["image"])
                        context["prompt"] = record.prompt
                        images = backend.generate_reference(context, **options)
                    else:
                        images = model.predict(
                            inputs=batch["inputs"],
                            requested_outputs=["image"],
                            native_context=batch["native_context"],
                            decoder_kwargs={"image": options},
                        ).predictions["image"]
                trace = backend.last_trace
                latent = trace.get("latents.initial")
                if not isinstance(latent, torch.Tensor):
                    raise RuntimeError("Sample trace must capture initial latents")
                latent = latent.detach().cpu().clone()
                if native:
                    initial_latents[str(case_number)] = latent
                elif not torch.equal(latent, initial_latents[str(case_number)]):
                    raise RuntimeError("Sample initial latents differ from native noise bank")
                reference_hashes = {
                    name: _tensor_hash(value)
                    for name, value in trace.items()
                    if name.startswith("reference.") and isinstance(value, torch.Tensor)
                }
                if native:
                    initial_reference_hashes[str(case_number)] = reference_hashes
                elif reference_hashes != initial_reference_hashes[str(case_number)]:
                    raise RuntimeError("Source reference latents differ from native baseline")
                images = images.images if hasattr(images, "images") else images
                if not isinstance(images, (list, tuple)) or not images:
                    raise RuntimeError("Generator did not return an image sequence")
                path = args.output_dir / f"sample-{stage}-{step:06d}-{case_number:02d}.png"
                images[0].save(path)
                report["samples"].append(
                    {
                        "stage": stage,
                        "step": step,
                        "split": split,
                        "id": record.id,
                        "prompt": record.prompt,
                        "seed": sample_seed,
                        "path": str(path),
                        "sha256": file_sha256(path),
                        "initial_latent_sha256": _tensor_hash(latent),
                        "reference_latent_sha256": {
                            name: _tensor_hash(value)
                            for name, value in trace.items()
                            if name.startswith("reference.") and isinstance(value, torch.Tensor)
                        },
                        "sampling_target_free": True,
                        "quality_claim": False,
                    }
                )
                backend.last_trace = {}
                _write(report_path, report)

        def checkpoint(step):
            path = args.output_dir / f"connector-diagnostic-step-{step:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "evidence_kind": EVIDENCE_KIND,
                    "qualification": "unqualified",
                    "step": step,
                    "connector_modules": list(MODULES),
                    "connector_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                        if name.startswith(MODULES[0] + ".")
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": _rng_state(),
                    "protocol": protocol,
                    "frozen_hashes": before,
                    "nonzero_gradient_parameters": sorted(nonzero_names),
                    "initial_latents": initial_latents,
                    "initial_reference_hashes": initial_reference_hashes,
                    "initial_evaluation": report["initial_evaluation"],
                    "origin_baseline_samples": report["origin_baseline_samples"],
                    "previous_report": str(report_path),
                },
                temporary,
            )
            temporary.replace(path)
            report["checkpoints"].append(
                {
                    "step": step,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "qualification": "unqualified",
                }
            )
            _write(report_path, report)

        if args.resume is not None:
            args.resume = args.resume.resolve(strict=True)
            saved = torch.load(args.resume, map_location="cpu", weights_only=True)
            if (
                saved.get("schema_version") != 1
                or saved.get("evidence_kind") != EVIDENCE_KIND
                or saved.get("qualification") != "unqualified"
            ):
                raise ValueError("Resume accepts only unqualified overfit diagnostic checkpoints")
            if saved.get("protocol") != protocol:
                raise ValueError("Resume protocol/data/parent/runtime settings do not match")
            if saved.get("frozen_hashes") != before:
                raise ValueError("Resume frozen parent/generator hashes do not match")
            start_step = saved["step"]
            if not isinstance(start_step, int) or not 0 <= start_step < args.steps:
                raise ValueError("Resume step must precede requested total steps")
            prefix = MODULES[0] + "."
            state = saved["connector_state_dict"]
            expected = {name for name in model.state_dict() if name.startswith(prefix)}
            if set(state) != expected:
                raise ValueError("Resume must contain the complete connector")
            model.decoders["image"].connector.load_state_dict(
                {name[len(prefix) :]: value for name, value in state.items()}, strict=True
            )
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            _restore_rng(saved["rng_state"])
            initial_latents.update(saved["initial_latents"])
            initial_reference_hashes.update(saved["initial_reference_hashes"])
            if set(initial_latents) != {str(index) for index in range(len(sample_cases))}:
                raise ValueError("Resume initial latent bank is incomplete")
            nonzero_names.update(saved["nonzero_gradient_parameters"])
            report.update(
                completed_steps=start_step,
                resumed_from=str(args.resume),
                resume_checkpoint_sha256=file_sha256(args.resume),
                previous_report=saved["previous_report"],
                initial_evaluation=saved["initial_evaluation"],
                origin_baseline_samples=saved["origin_baseline_samples"],
            )
        else:
            start_step = 0
            report["initial_evaluation"] = evaluate(0)
            sample("native", 0, native=True)
            sample("untrained", 0)
            report["origin_baseline_samples"] = list(report["samples"])
            checkpoint(0)
        _phase(report, report_path, "training")
        with (args.output_dir / "steps.jsonl").open("x") as stream:
            for step in range(start_step + 1, args.steps + 1):
                step_started = time.monotonic()
                _seed_everything(args.seed + step - 1)
                index = (step - 1) % len(datasets["train"])
                optimizer.zero_grad(set_to_none=True)
                loss = forward(batch_at("train", index))
                if any(
                    p.requires_grad and name not in trainable
                    for name, p in model.named_parameters()
                ):
                    raise RuntimeError("Forward registered trainable weights outside connector")
                if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:
                    raise RuntimeError("Expected a finite differentiable scalar image loss")
                loss.backward()
                norms = {}
                for name, parameter in trainable.items():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Missing/nonfinite connector gradient: {name}")
                    norms[name] = float(parameter.grad.float().norm())
                    if norms[name] > 0:
                        nonzero_names.add(name)
                if not any(norm > 0 for norm in norms.values()):
                    raise RuntimeError("All connector gradients are zero")
                norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in trainable.values()):
                    raise RuntimeError("Optimizer produced nonfinite connector weights")
                report["completed_steps"] = step
                entry = {
                    "step": step,
                    "id": train_records[index].id,
                    "seed": args.seed + step - 1,
                    "loss": float(loss.detach()),
                    "gradient_norms": norms,
                    "gradient_norm_before_clip": float(norm),
                    "duration_seconds": time.monotonic() - step_started,
                }
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                del loss
                if step % args.checkpoint_every == 0 or step == args.steps:
                    # Persist optimization state first so an expensive gallery cannot lose progress.
                    checkpoint(step)
                    evaluate(step)
                    sample("trained", step)
                    _phase(report, report_path, "training")
                elif step % 10 == 0:
                    _phase(report, report_path, "training")
        if set(trainable) != nonzero_names:
            raise RuntimeError("Some connector parameters never received nonzero gradients")
        report["nonzero_gradient_parameters"] = sorted(nonzero_names)
        report.update(status="completed", phase="completed")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if model is not None and before is not None:
            try:
                after = frozen_state_hashes(model, MODULES)
                report["frozen_hashes_after"] = after
                report["frozen_state_unchanged"] = before == after
                if before != after:
                    report.update(status="failed")
                    report.setdefault(
                        "error", "Frozen parent/generator parameters or buffers changed"
                    )
            except Exception as audit_error:
                report.update(status="failed", frozen_state_unchanged=None)
                report["frozen_state_audit_error"] = f"{type(audit_error).__name__}: {audit_error}"
                report.setdefault("error", "Frozen-state verification could not complete")
        if args.device.startswith("xpu") and torch.xpu.is_available():
            report["peak_allocated_bytes"] = torch.xpu.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.xpu.max_memory_reserved()
        report["duration_seconds"] = time.monotonic() - started
        _write(report_path, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "evidence_kind",
                    "qualification",
                    "completed_steps",
                    "duration_seconds",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
