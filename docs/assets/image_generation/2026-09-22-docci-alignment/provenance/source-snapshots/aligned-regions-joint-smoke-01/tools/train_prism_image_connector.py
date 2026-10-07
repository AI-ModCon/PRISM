#!/usr/bin/env python3
"""Bounded, resumable WebDataset connector pilot; artifacts remain unqualified.

Only the new PRISM-to-OmniGen2 connector is optimized. This experiment requires
a real numerical repeatability diagnostic, but does not replace the P0/P1/P2
acceptance gates of train_image_decoder.py. Validation images never enter the
optimizer, and sampling builds text-only examples without opening target images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.overfit_prism_image_connector import _phase, _restore_rng, _write, execution_policy

EVIDENCE_KIND = "real_checkpoint_connector_webdataset_pilot"
MODULES = ("decoders.image.connector",)
PRECHECK_SOURCES = (
    "tools/diagnose_image_decoder_repeatability.py",
    "tools/validate_image_decoder.py",
    "src/eval/image_generation.py",
    "src/decoders/omnigen2_backend.py",
    "src/decoders/image.py",
    "src/decoders/types.py",
)


class ShuffledEpochOrder:
    """Reproducible finite epochs with an explicit cursor and no worker prefetch.

    The independent Python RNG cannot be perturbed by evaluation or diffusion
    seeds. Checkpoints are taken at optimizer boundaries, after accumulated
    microbatches, and persist the next unread position exactly.
    """

    def __init__(self, count: int, seed: int):
        if count < 1:
            raise ValueError("Training order requires nonempty data")
        self.count, self.seed = count, seed
        self.epoch, self.cursor, self.examples_seen = 0, 0, 0
        self.rng = random.Random(seed)
        self.order = list(range(count))
        self.rng.shuffle(self.order)

    def take(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("Microbatch must contain at least one example")
        result = []
        for _ in range(count):
            if self.cursor == self.count:
                self.epoch += 1
                self.cursor = 0
                self.order = list(range(self.count))
                self.rng.shuffle(self.order)
            result.append(self.order[self.cursor])
            self.cursor += 1
            self.examples_seen += 1
        return result

    def state_dict(self):
        return {
            "count": self.count,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "examples_seen": self.examples_seen,
            "order": list(self.order),
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        if state.get("count") != self.count or state.get("seed") != self.seed:
            raise ValueError("Resume sampler data count or seed differs")
        order = state.get("order", [])
        cursor, epoch, seen = (state.get(key) for key in ("cursor", "epoch", "examples_seen"))
        if (
            sorted(order) != list(range(self.count))
            or not isinstance(cursor, int)
            or not 0 <= cursor <= self.count
            or not isinstance(epoch, int)
            or epoch < 0
            or seen != epoch * self.count + cursor
        ):
            raise ValueError("Invalid resume sampler permutation or cursor")
        self.order, self.cursor, self.epoch, self.examples_seen = list(order), cursor, epoch, seen
        self.rng.setstate(state["rng"])


def validate_repeatability_report(path, args, *, checkpoint_manifest=None, backend=None):
    """Bind passed real diagnostic comparisons to the actual source and runtime."""
    import torch
    from src.decoders.loading import file_sha256

    report = json.loads(Path(path).read_text())
    if (
        report.get("status") != "completed"
        or report.get("mode") != "repeatability_diagnostic"
        or report.get("evidence_kind") != "real_checkpoint_diagnostic"
        or report.get("fixture") is not False
    ):
        raise ValueError("A completed real-checkpoint repeatability report is required")
    required = ("native_repeat", "adapter_vs_native", "native_after_adapter")
    comparisons = report.get("comparisons", {})
    if any(comparisons.get(key, {}).get("passed") is not True for key in required):
        raise ValueError("Repeatability comparisons did not all pass")
    identity = report.get("identity", {})
    expected = {
        "dtype": args.dtype,
        "torch_version": str(torch.__version__),
        "device_type": args.device.split(":", 1)[0],
        "numerical_policy": {
            "deterministic_algorithms": args.deterministic,
            "attention_backend": args.attention_backend,
        },
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"Repeatability runtime mismatch: {key}")
    if not set(PRECHECK_SOURCES).issubset(report.get("source_hashes", {})):
        raise ValueError("Repeatability source hashes are incomplete")
    for name, expected_hash in report["source_hashes"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Invalid repeatability source path: {name}")
        if file_sha256(source) != expected_hash:
            raise ValueError(f"Repeatability source changed: {name}")
    if checkpoint_manifest is not None:
        if identity.get("checkpoint_manifest_sha256") != checkpoint_manifest["manifest_sha256"]:
            raise ValueError("Repeatability generator checkpoint mismatch")
        if identity.get("kernel_policy") != backend.get("kernel_policy"):
            raise ValueError("Repeatability generator kernel policy mismatch")
        device_type = expected["device_type"]
        device_name = device_type
        if device_type in {"xpu", "cuda"}:
            device_name = getattr(torch, device_type).get_device_name(torch.device(args.device))
        if identity.get("device_name") != device_name:
            raise ValueError("Repeatability device model mismatch")
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path), "identity": identity}


@contextmanager
def preserved_rng():
    from tools.train_image_decoder import _rng_state

    state = _rng_state()
    try:
        yield
    finally:
        _restore_rng(state)


def target_free_item(record):
    """Construct a sampling input from metadata only; never call dataset[index]."""
    if record.task != "t2i" or record.source_ids or record.source_paths:
        raise ValueError("This pilot supports caption-conditioned generation only")
    return {
        "id": record.id,
        "prompt": record.prompt,
        "task": "t2i",
        "split": record.split,
        "group_ids": record.group_ids,
        "source_ids": (),
        "encoder_source_images": [],
        "reference_images": [],
        "target_image": None,
    }


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
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, default=500, help="Total optimizer steps, including resume"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument(
        "--probe-count",
        type=int,
        default=32,
        help="Fixed probes per split; final validation uses all records",
    )
    parser.add_argument(
        "--final-validation-count",
        type=int,
        default=0,
        help="0 evaluates every validation record; explicit positive limit for a short smoke run",
    )
    parser.add_argument(
        "--sample-count", type=int, default=2, help="Samples per split at each gallery milestone"
    )
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument(
        "--native-baseline",
        action="store_true",
        help="Also sample native OmniGen2 on validation gallery cases",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--expected-parent-tensors", type=int, default=526)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="math")
    return parser


def validate_budget(args):
    for key, lower, upper in (
        ("steps", 1, 5000),
        ("batch_size", 1, 4),
        ("gradient_accumulation", 1, 32),
        ("checkpoint_every", 1, 1000),
        ("eval_every", 1, 1000),
        ("sample_every", 1, 5000),
        ("probe_count", 1, 100),
        ("final_validation_count", 0, 100),
        ("sample_count", 1, 2),
        ("sampling_steps", 1, 50),
    ):
        if not lower <= getattr(args, key) <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    if args.expected_parent_tensors < 1:
        raise ValueError("expected-parent-tensors must be positive")
    if any(size < 32 or size > 256 or size % 16 for size in (args.height, args.width)):
        raise ValueError("Pilot dimensions must be multiples of 16 in [32, 256]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if any(
        not math.isfinite(x) or x < 0 for x in (args.text_guidance_scale, args.image_guidance_scale)
    ):
        raise ValueError("Guidance scales must be finite and nonnegative")
    if not args.deterministic or args.attention_backend != "math":
        raise ValueError("This pilot requires --deterministic --attention-backend math")


def main(argv=None):
    args = _parser().parse_args(argv)
    validate_budget(args)
    with execution_policy(args):
        return _run(args)


def _run(args):
    import torch
    from src.data.image_generation import ImageGenerationCollator
    from src.data.image_generation_webdataset import ImageGenerationWebDataset
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.train_image_decoder import (
        _configure_connector_only,
        _rng_state,
        _seed_everything,
        _tensor_hash,
        _to_device,
        frozen_state_hashes,
    )

    for name in (
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "train_index",
        "validation_index",
        "repeatability_report",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("Output directory must be new, including on resume")
    if args.resume is not None:
        args.resume = args.resume.resolve(strict=True)
    precheck = validate_repeatability_report(args.repeatability_report, args)
    datasets = {
        split: ImageGenerationWebDataset(index, target_size=(args.height, args.width), split=split)
        for split, index in (("train", args.train_index), ("validation", args.validation_index))
    }
    if any(len(dataset) < 2 for dataset in datasets.values()):
        raise ValueError("Both splits need at least two examples for caption controls")
    if datasets["train"].data_fingerprint != datasets["validation"].data_fingerprint:
        raise ValueError("Train/validation must use the same validated conversion bundle")
    records = {split: dataset.records for split, dataset in datasets.items()}
    for split, rows in records.items():
        if any(
            row.split != split or row.task != "t2i" or row.source_ids or row.source_paths
            for row in rows
        ):
            raise ValueError("Pilot requires source-free t2i examples in the designated split")
    if {row.id for row in records["train"]} & {row.id for row in records["validation"]}:
        raise ValueError("Training and validation IDs overlap")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "evidence_kind": EVIDENCE_KIND,
        "qualification": "unqualified",
        "status": "running",
        "completed_steps": 0,
        "p0_p1_gate": "not_established_by_this_pilot",
        "p2_gate": "not_evaluated",
        "quality_benchmark": False,
        "source_image_conditioning_exercised": False,
        "parent_heldout_visual_quality_established": False,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": str(torch.__version__),
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
                "src/data/image_generation_webdataset.py",
                "tools/train_image_decoder.py",
                "tools/overfit_prism_image_connector.py",
            )
        },
        "repeatability_precheck": precheck,
        "train_count": len(datasets["train"]),
        "validation_count": len(datasets["validation"]),
        "data_fingerprint": datasets["train"].data_fingerprint,
        "data_validation": datasets["train"].validation_report,
        "train_index_sha256": file_sha256(args.train_index),
        "validation_index_sha256": file_sha256(args.validation_index),
        "data_order": "seeded independent shuffled finite epochs; explicit next-unread cursor; no workers/prefetch",
        "probe_protocol": "Full RNG reset per example fixes VAE posterior, timestep and noise; wrong caption uses identical target/noise. Final validation uses every record unless an explicit smoke limit is configured.",
        "sample_protocol": "Metadata-only text input; saved initial latent replay; no target pixels opened by sampling",
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
        if bundle["provenance"].get("fixture"):
            report["evidence_kind"] = "fixture_only"
        restoration = bundle["provenance"].get("restoration", {})
        if (
            restoration.get("strict_parent") is not True
            or restoration.get("loaded_key_count") != args.expected_parent_tensors
            or restoration.get("missing_parent_keys") != []
            or restoration.get("unexpected_keys") != []
        ):
            raise ValueError("Full expected parent tensor restoration was not established")
        model.to(device=args.device, dtype=getattr(torch, args.dtype))
        _phase(report, report_path, "loading_generator")
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
        collator = ImageGenerationCollator(bundle["tokenizer"])
        optimizer = torch.optim.AdamW(
            list(trainable.values()), lr=args.learning_rate, weight_decay=0.0
        )
        order = ShuffledEpochOrder(len(datasets["train"]), args.seed)
        # Follow the first optimizer examples rather than arbitrary index rows;
        # a bounded pilot may visit only a small fraction of the training pool.
        initial_train_order = list(order.order)
        initial_train_rank = {index: rank for rank, index in enumerate(initial_train_order)}
        selections = {
            "train": initial_train_order[: min(args.probe_count, len(records["train"]))],
            "validation": list(range(min(args.probe_count, len(records["validation"])))),
        }
        report["probe_selection"] = {
            split: [{"index": index, "id": records[split][index].id} for index in indices]
            for split, indices in selections.items()
        }
        protocol = {
            "parent": bundle["provenance"],
            "reference_checkpoint_sha256": report["generator"]["manifest_sha256"],
            "generator_runtime": report["generator_runtime"],
            "data_fingerprint": report["data_fingerprint"],
            "train_index_sha256": report["train_index_sha256"],
            "validation_index_sha256": report["validation_index_sha256"],
            "runner_sha256": report["runner_sha256"],
            "source_sha256": report["source_sha256"],
            "repeatability_precheck": precheck,
            "torch_version": str(torch.__version__),
            "probe_selection": report["probe_selection"],
            "settings": {
                key: value
                for key, value in report["settings"].items()
                if key not in {"steps", "output_dir", "resume"}
            },
        }
        report["resume_protocol"] = protocol
        cases = [
            (split, initial_train_order[position] if split == "train" else position)
            for position in range(args.sample_count)
            for split in ("train", "validation")
        ]
        latents = {}
        reference_hashes = {}

        def batch_for(items):
            return _to_device(collator(items), torch.device(args.device))

        def forward(batch):
            if set(batch["targets"]) != {"image"}:
                raise RuntimeError("Only the image flow objective is allowed")
            loss = model.forward_outputs(
                batch["inputs"],
                targets=batch["targets"],
                requested_outputs=["image"],
                native_context=batch["native_context"],
                output_specs=batch["output_specs"],
            ).losses["image"]
            if loss.ndim or not torch.isfinite(loss):
                raise RuntimeError("Expected finite scalar image loss")
            return loss

        def evaluate(step, final=False):
            _phase(report, report_path, "fixed_validation" if final else "fixed_probes")
            evaluation = {
                "step": step,
                "final": final,
                "full_validation": final
                and (
                    not args.final_validation_count
                    or args.final_validation_count >= len(datasets["validation"])
                ),
                "splits": {},
            }
            with preserved_rng(), torch.no_grad():
                for split_number, split in enumerate(("train", "validation")):
                    count = (
                        len(datasets[split])
                        if final and split == "validation"
                        else min(args.probe_count, len(datasets[split]))
                    )
                    if final and split == "validation" and args.final_validation_count:
                        count = min(count, args.final_validation_count)
                    entries = []
                    for position in range(count):
                        index = selections["train"][position] if split == "train" else position
                        item = datasets[split][index]
                        row = records[split][index]
                        other = next(
                            (
                                records[split][(index + offset) % len(records[split])]
                                for offset in range(1, len(records[split]))
                                if records[split][(index + offset) % len(records[split])].prompt
                                != row.prompt
                            ),
                            None,
                        )
                        if other is None:
                            raise ValueError(
                                "Caption control requires distinct captions in each split"
                            )
                        seed = args.seed + 100000 + 1009 * split_number + position
                        batch = batch_for([item])
                        shuffled = batch_for([dict(item, prompt=other.prompt)])
                        if not torch.equal(batch["targets"]["image"], shuffled["targets"]["image"]):
                            raise RuntimeError("Caption control changed target image")
                        _seed_everything(seed)
                        correct = float(forward(batch))
                        _seed_everything(seed)
                        wrong = float(forward(shuffled))
                        entries.append(
                            {
                                "id": row.id,
                                "index": index,
                                "optimized_before_evaluation": split == "train"
                                and order.examples_seen > initial_train_rank[index],
                                "seed": seed,
                                "loss": correct,
                                "shuffled_prompt_id": other.id,
                                "shuffled_loss": wrong,
                                "shuffled_minus_correct": wrong - correct,
                            }
                        )
                    evaluation["splits"][split] = {
                        "count": count,
                        "mean_loss": sum(row["loss"] for row in entries) / count,
                        "mean_shuffled_loss": sum(row["shuffled_loss"] for row in entries) / count,
                        "mean_shuffled_minus_correct": sum(
                            row["shuffled_minus_correct"] for row in entries
                        )
                        / count,
                        "interpretation": "Positive gap favors matched caption; diagnostic only",
                        "examples": entries,
                    }
            report["evaluations"].append(evaluation)
            with (args.output_dir / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(evaluation, allow_nan=False) + "\n")
            _write(report_path, report)
            return evaluation

        def sample(stage, step, native=False):
            _phase(report, report_path, "sampling_" + stage)
            with preserved_rng(), torch.no_grad():
                for case_number, (split, index) in enumerate(cases):
                    if native and split != "validation":
                        continue
                    row = records[split][index]
                    batch = batch_for([target_free_item(row)])
                    if batch["targets"] or batch["output_specs"]:
                        raise RuntimeError("Sampling leaked target supervision")
                    seed = args.seed + 200000 + case_number
                    _seed_everything(seed)
                    options = {
                        "height": args.height,
                        "width": args.width,
                        "num_inference_steps": args.sampling_steps,
                        "text_guidance_scale": args.text_guidance_scale,
                        "image_guidance_scale": args.image_guidance_scale,
                        "negative_prompt": args.negative_prompt,
                        "generator": torch.Generator(device=args.device).manual_seed(seed),
                        "trace": True,
                    }
                    key = str(case_number)
                    if key in latents:
                        options["latents"] = (
                            latents[key]
                            .to(device=args.device, dtype=getattr(torch, args.dtype))
                            .clone()
                        )
                    if native:
                        images = backend.generate_reference(
                            dict(batch["native_context"]["image"], prompt=row.prompt), **options
                        )
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
                        raise RuntimeError("Sampling must capture initial latent trace")
                    latent = latent.detach().cpu().clone()
                    references = {
                        name: _tensor_hash(value)
                        for name, value in trace.items()
                        if name.startswith("reference.") and isinstance(value, torch.Tensor)
                    }
                    if key not in latents:
                        latents[key], reference_hashes[key] = latent, references
                    elif (
                        not torch.equal(latents[key], latent) or reference_hashes[key] != references
                    ):
                        raise RuntimeError("Sampling noise or reference latents changed")
                    images = images.images if hasattr(images, "images") else images
                    if not isinstance(images, (list, tuple)) or not images:
                        raise RuntimeError("Generator did not return image sequence")
                    path = args.output_dir / f"sample-{stage}-{step:06d}-{case_number:02d}.png"
                    images[0].save(path)
                    report["samples"].append(
                        {
                            "stage": stage,
                            "step": step,
                            "split": split,
                            "id": row.id,
                            "index": index,
                            "optimized_before_sampling": split == "train"
                            and order.examples_seen > initial_train_rank[index],
                            "prompt": row.prompt,
                            "seed": seed,
                            "path": str(path),
                            "sha256": file_sha256(path),
                            "initial_latent_sha256": _tensor_hash(latent),
                            "sampling_target_free": True,
                            "quality_claim": False,
                        }
                    )
                    backend.last_trace = {}
                    _write(report_path, report)

        def checkpoint(step):
            path = args.output_dir / f"connector-pilot-step-{step:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "schema_version": 1,
                    "evidence_kind": report["evidence_kind"],
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
                    "sampler_state": order.state_dict(),
                    "protocol": protocol,
                    "frozen_hashes": before,
                    "nonzero_gradient_parameters": sorted(nonzero_names),
                    "initial_latents": latents,
                    "initial_reference_hashes": reference_hashes,
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
            report["sampler_state"] = {
                key: value
                for key, value in order.state_dict().items()
                if key not in {"order", "rng"}
            }
            _write(report_path, report)

        if args.resume is None:
            start_step = 0
            report["initial_evaluation"] = evaluate(0)
            sample("untrained", 0)
            if args.native_baseline:
                sample("native", 0, native=True)
            report["origin_baseline_samples"] = list(report["samples"])
            checkpoint(0)
        else:
            saved = torch.load(args.resume, map_location="cpu", weights_only=True)
            if (
                saved.get("schema_version") != 1
                or saved.get("evidence_kind") != report["evidence_kind"]
                or saved.get("qualification") != "unqualified"
            ):
                raise ValueError("Resume requires an unqualified checkpoint of this pilot")
            if saved.get("protocol") != protocol or saved.get("frozen_hashes") != before:
                raise ValueError(
                    "Resume parent/data/source/runtime protocol or frozen weights changed"
                )
            start_step = saved.get("step")
            if not isinstance(start_step, int) or not 0 <= start_step < args.steps:
                raise ValueError("Resume step must precede total requested steps")
            prefix = MODULES[0] + "."
            state = saved["connector_state_dict"]
            if set(state) != {name for name in model.state_dict() if name.startswith(prefix)}:
                raise ValueError("Resume connector state is incomplete")
            model.decoders["image"].connector.load_state_dict(
                {name[len(prefix) :]: value for name, value in state.items()}, strict=True
            )
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            order.load_state_dict(saved["sampler_state"])
            if order.examples_seen != start_step * args.batch_size * args.gradient_accumulation:
                raise ValueError("Resume sampler progress and optimizer steps disagree")
            _restore_rng(saved["rng_state"])
            latents.update(saved["initial_latents"])
            reference_hashes.update(saved["initial_reference_hashes"])
            if set(latents) != {str(index) for index in range(len(cases))} or set(
                reference_hashes
            ) != set(latents):
                raise ValueError("Resume sampling latent bank is incomplete")
            nonzero_names.update(saved["nonzero_gradient_parameters"])
            report.update(
                completed_steps=start_step,
                resumed_from=str(args.resume),
                resume_checkpoint_sha256=file_sha256(args.resume),
                previous_report=saved["previous_report"],
                initial_evaluation=saved["initial_evaluation"],
                origin_baseline_samples=saved["origin_baseline_samples"],
            )
        _phase(report, report_path, "training")
        with (args.output_dir / "steps.jsonl").open("x") as stream:
            for step in range(start_step + 1, args.steps + 1):
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                microbatches, losses = [], []
                for microbatch in range(args.gradient_accumulation):
                    indices = order.take(args.batch_size)
                    seed = args.seed + (step - 1) * args.gradient_accumulation + microbatch
                    _seed_everything(seed)
                    loss = forward(batch_for([datasets["train"][index] for index in indices]))
                    if not loss.requires_grad:
                        raise RuntimeError("Image loss is detached from the connector")
                    if any(
                        parameter.requires_grad and name not in trainable
                        for name, parameter in model.named_parameters()
                    ):
                        raise RuntimeError("Forward registered trainable weights outside connector")
                    (loss / args.gradient_accumulation).backward()
                    losses.append(float(loss.detach()))
                    microbatches.append(
                        {"ids": [records["train"][index].id for index in indices], "seed": seed}
                    )
                    del loss
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
                    "loss": sum(losses) / len(losses),
                    "microbatch_losses": losses,
                    "microbatches": microbatches,
                    "examples_seen": order.examples_seen,
                    "epoch": order.epoch,
                    "epoch_cursor": order.cursor,
                    "gradient_norms": norms,
                    "gradient_norm_before_clip": float(norm),
                    "duration_seconds": time.monotonic() - tick,
                }
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                final = step == args.steps
                if final or step % args.checkpoint_every == 0:
                    checkpoint(step)
                if final or step % args.eval_every == 0:
                    evaluate(step, final=final)
                if final or step % args.sample_every == 0:
                    sample("trained", step)
                if final or step % 10 == 0:
                    _phase(report, report_path, "training")
        if set(trainable) != nonzero_names:
            raise RuntimeError("Some connector parameters never received nonzero gradients")
        report["nonzero_gradient_parameters"] = sorted(nonzero_names)
        report.update(status="completed", phase="completed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if model is not None and before is not None:
            try:
                after = frozen_state_hashes(model, MODULES)
                report["frozen_hashes_after"] = after
                report["frozen_state_unchanged"] = before == after
                if before != after:
                    report.update(
                        status="failed",
                        error="Frozen parent/generator parameters or buffers changed",
                    )
            except Exception as audit_error:
                report.update(
                    status="failed",
                    frozen_state_unchanged=None,
                    frozen_state_audit_error=str(audit_error),
                )
        accelerator = getattr(torch, args.device.split(":", 1)[0], None)
        if args.device.split(":", 1)[0] in {"xpu", "cuda"} and accelerator.is_available():
            report["peak_allocated_bytes"] = accelerator.max_memory_allocated()
            report["peak_reserved_bytes"] = accelerator.max_memory_reserved()
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
