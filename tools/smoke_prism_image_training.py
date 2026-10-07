#!/usr/bin/env python3
"""Bounded real-parent optimization diagnostic, never a P2 training acceptance.

This deliberately separate entry point leaves train_image_decoder.py's full
qualification gates intact. It permits at most four optimizer steps/examples at
256 pixels, saves an unqualified artifact rejected by the production loader, and
records actual gradients, parent immutability, and target-free image sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_budget(steps, examples, height, width, sampling_steps):
    if not 1 <= steps <= 4 or not 1 <= examples <= 4:
        raise ValueError("Optimization smoke requires 1–4 steps and 1–4 examples")
    if any(size < 32 or size > 256 or size % 16 for size in (height, width)):
        raise ValueError("Smoke image dimensions must be multiples of 16 in [32, 256]")
    if not 1 <= sampling_steps <= 4:
        raise ValueError("Smoke sampling requires 1–4 denoising steps")


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-config",
        "checkpoint",
        "tokenizer",
        "source-processor",
        "manifest",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args(argv)
    # Resolve explicit assets before importing/loading models. No downloads.
    for name in ("model_config", "checkpoint", "tokenizer", "source_processor", "manifest"):
        getattr(args, name).resolve(strict=True)
    if args.output_dir.exists():
        raise ValueError("Output directory must be new")
    import math

    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import torch
    from src.data.image_generation import (
        ImageGenerationCollator,
        ImageGenerationDataset,
        load_image_generation_manifest,
    )
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.train_image_decoder import (
        _configure_connector_only,
        _seed_everything,
        _to_device,
        frozen_state_hashes,
    )

    records = load_image_generation_manifest(args.manifest)
    validate_budget(args.steps, len(records), args.height, args.width, args.sampling_steps)
    if any(record.split != "train" for record in records):
        raise ValueError("Optimization diagnostic accepts only an explicit train manifest")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "evidence_kind": "real_checkpoint_optimization_smoke",
        "status": "running",
        "qualification": "unqualified",
        "p0_p1_gate": "not_established_by_this_diagnostic",
        "p2_gate": "not_evaluated",
        "quality_benchmark": False,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": str(torch.__version__),
        "manifest_sha256": file_sha256(args.manifest),
        "steps": [],
    }
    model = None
    before = None
    modules = ("decoders.image.connector",)
    try:
        _seed_everything(args.seed)
        bundle = load_image_training_bundle(
            args.model_config, args.checkpoint, args.tokenizer, args.source_processor
        )
        model = bundle["model"]
        report["parent"] = bundle["provenance"]
        report["phase"] = "parent_loaded"
        _write(args.output_dir / "report.json", report)
        dtype = getattr(torch, args.dtype)
        model.to(device=args.device, dtype=dtype)
        backend = model.decoders["image"].backend
        backend.ensure_loaded()
        report["generator"] = backend.checkpoint_manifest()
        report["generator_runtime"] = backend.provenance()
        # FP32 connector updates avoid losing small Adam updates to BF16 rounding.
        model.decoders["image"].connector.float()
        trainable = _configure_connector_only(model, modules)
        before = frozen_state_hashes(model, modules)
        report["frozen_hashes_before"] = before
        report["trainable_parameters"] = {
            name: list(value.shape) for name, value in trainable.items()
        }
        report["trainable_parameter_count"] = sum(value.numel() for value in trainable.values())
        report["phase"] = "generator_loaded_and_frozen"
        _write(args.output_dir / "report.json", report)
        dataset = ImageGenerationDataset(
            args.manifest,
            source_transform=bundle["source_transform"],
            target_size=(args.height, args.width),
            split="train",
        )
        report["data_fingerprint"] = dataset.data_fingerprint
        collator = ImageGenerationCollator(bundle["tokenizer"])
        optimizer = torch.optim.AdamW(
            list(trainable.values()), lr=args.learning_rate, weight_decay=0.0
        )

        def batch_at(index):
            return _to_device(collator([dataset[index]]), torch.device(args.device))

        def forward(batch):
            # Targets travel only through the explicit decoder supervision path.
            return model.forward_outputs(
                batch["inputs"],
                targets=batch["targets"],
                requested_outputs=["image"],
                native_context=batch["native_context"],
                output_specs=batch["output_specs"],
            )

        probe = batch_at(0)
        _seed_everything(args.seed)
        with torch.no_grad():
            report["fixed_probe_loss_before"] = float(forward(probe).losses["image"])
        with (args.output_dir / "steps.jsonl").open("x") as stream:
            for step in range(args.steps):
                _seed_everything(args.seed + step)
                batch = batch_at(step % len(dataset))
                optimizer.zero_grad(set_to_none=True)
                loss = forward(batch).losses["image"]
                if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:
                    raise RuntimeError("Expected a finite differentiable scalar image loss")
                loss.backward()
                norms = {}
                for name, parameter in trainable.items():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Missing/nonfinite connector gradient: {name}")
                    norms[name] = float(parameter.grad.float().norm())
                if not any(value > 0 for value in norms.values()):
                    raise RuntimeError("All connector gradients are zero")
                torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in trainable.values()):
                    raise RuntimeError("Optimizer produced nonfinite connector weights")
                entry = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gradient_norms": norms,
                    "example_id": records[step % len(dataset)].id,
                }
                report["steps"].append(entry)
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                _write(args.output_dir / "report.json", report)
                del loss
        _seed_everything(args.seed)
        with torch.no_grad():
            report["fixed_probe_loss_after"] = float(forward(probe).losses["image"])
        # This artifact cannot be accepted by load_image_connector or the P2 gates.
        artifact = args.output_dir / "connector-smoke.pt"
        torch.save(
            {
                "schema_version": 1,
                "evidence_kind": report["evidence_kind"],
                "qualification": "unqualified",
                "connector_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                    if name.startswith("decoders.image.connector.")
                },
                "provenance": bundle["provenance"],
            },
            artifact,
        )
        report["connector_artifact_sha256"] = file_sha256(artifact)
        _seed_everything(args.seed)
        with torch.no_grad():
            generated = model.predict(
                inputs=probe["inputs"],
                requested_outputs=["image"],
                native_context=probe["native_context"],
                decoder_kwargs={
                    "image": {
                        "height": args.height,
                        "width": args.width,
                        "num_inference_steps": args.sampling_steps,
                    }
                },
            ).predictions["image"]
        images = generated.images if hasattr(generated, "images") else generated
        if not isinstance(images, (list, tuple)) or not images:
            raise RuntimeError("Generator did not return an image sequence")
        image_path = args.output_dir / "after-smoke.png"
        images[0].save(image_path)
        report["generated_image"] = {
            "path": image_path.name,
            "sha256": file_sha256(image_path),
            "sampling_target_free": True,
            "quality_claim": False,
        }
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if model is not None and before is not None:
            after = frozen_state_hashes(model, modules)
            report["frozen_hashes_after"] = after
            report["frozen_state_unchanged"] = before == after
            if before != after:
                report.update(
                    status="failed", error="Frozen parent/generator parameters or buffers changed"
                )
        if args.device.startswith("xpu") and torch.xpu.is_available():
            report["peak_allocated_bytes"] = torch.xpu.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.xpu.max_memory_reserved()
        report["duration_seconds"] = time.monotonic() - started
        _write(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "evidence_kind", "qualification", "duration_seconds")
            }
        )
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
