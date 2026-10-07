#!/usr/bin/env python3
"""Read-only, PBS-only conditioning diagnosis with real frozen checkpoints.

Paired flow losses use identical target/noise/timestep inputs. Sampling compares
native conditioning, the current mixed-conditioner CFG route, no CFG, and a
PRISM negative anchor. These are sensitivity diagnostics, not accuracy scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.overfit_prism_image_connector import _phase, _write, execution_policy
from tools.prism_image_conditioning import (
    encode_prism_prompt,
    feature_statistics,
    preserved_model_state,
    summarize_caption_controls,
)
from tools.train_prism_image_connector import ShuffledEpochOrder, validate_repeatability_report


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
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--connector-checkpoint-sha256")
    parser.add_argument("--expected-connector-step", type=int, default=500)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--joint-checkpoint-sha256")
    parser.add_argument("--train-probe-count", type=int, default=4)
    parser.add_argument("--validation-probe-count", type=int, default=4)
    parser.add_argument("--flow-timesteps", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument(
        "--prism-formats", nargs="+", choices=("raw", "chat"), default=["raw", "chat"]
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-parent-tensors", type=int, default=526)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="math")
    return parser


def validate_budget(args):
    for key, lower, upper in (
        ("train_probe_count", 1, 64),
        ("validation_probe_count", 1, 100),
        ("sample_count", 0, 2),
        ("sampling_steps", 1, 50),
        ("max_text_length", 1, 2048),
    ):
        if not lower <= getattr(args, key) <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    if not 1 <= len(args.flow_timesteps) <= 8 or any(
        not math.isfinite(value) or not 0 < value < 1 for value in args.flow_timesteps
    ):
        raise ValueError("Use 1–8 finite flow timesteps strictly between zero and one")
    if args.sample_count > args.validation_probe_count:
        raise ValueError("Sampling cases must be included in the validation probe cohort")
    if len(set(args.prism_formats)) != len(args.prism_formats):
        raise ValueError("PRISM formats must be distinct")
    if any(size < 32 or size > 256 or size % 16 for size in (args.height, args.width)):
        raise ValueError("Diagnostic dimensions must be multiples of 16 in [32, 256]")
    if not args.deterministic or args.attention_backend != "math":
        raise ValueError("Diagnostic requires --deterministic --attention-backend math")


def select_wrong_caption(records, index, pool):
    """Use another caption from the same declared data pool (tiny subset included)."""
    if index not in pool:
        raise ValueError("Matched caption is outside the designated caption pool")
    position = pool.index(index)
    for offset in range(1, len(pool)):
        candidate = pool[(position + offset) % len(pool)]
        if records[candidate].prompt != records[index].prompt:
            return candidate
    raise ValueError("Caption pool has no distinct wrong-caption control")


def encode_native_prompt(backend, prompt, *, max_text_length=1024):
    """Use the pinned pipeline's exact native prompt path, proving no truncation."""
    pipe = backend._pipeline
    formatted = pipe._apply_chat_template(prompt)
    values = pipe.processor.tokenizer(
        [formatted], padding=True, truncation=False, return_tensors="pt"
    )
    ids, mask = values["input_ids"], values["attention_mask"]
    if ids.shape[1] > max_text_length:
        raise ValueError("Native prompt exceeds max_text_length; truncation is forbidden")
    embeds, attention_mask, _, _ = pipe.encode_prompt(
        prompt=[prompt],
        do_classifier_free_guidance=False,
        device=pipe.transformer.device,
        max_sequence_length=max_text_length,
    )
    if attention_mask.shape != mask.shape or not attention_mask.cpu().equal(mask.cpu()):
        raise RuntimeError("Native prompt encoding changed token mask or truncated text")
    return {
        "embeds": embeds,
        "attention_mask": attention_mask,
        "input_ids": ids,
        "input_attention_mask": mask,
        "formatted_prompt": formatted,
        "format": "native",
        "empty_anchor": None,
    }


def condition_metadata(value):
    from tools.train_image_decoder import _tensor_hash

    result = {
        "format": value["format"],
        "empty_anchor": value["empty_anchor"],
        "formatted_prompt": value["formatted_prompt"],
        "input_token_ids": value["input_ids"].detach().cpu().tolist(),
        "input_valid_token_counts": value["input_attention_mask"].sum(1).cpu().tolist(),
        "input_attention_mask_sha256": _tensor_hash(value["input_attention_mask"]),
        "features": feature_statistics(value["embeds"], value["attention_mask"]),
    }
    if "hidden_states" in value:
        result["prism_hidden_states"] = feature_statistics(
            value["hidden_states"], value["input_attention_mask"]
        )
    if not result["features"]["finite"] or not result["features"]["valid_prefix"]:
        raise RuntimeError("Nonfinite or non-prefix conditioning features")
    return result


def paired_flow_losses(backend, target, conditions, *, seed, timestep, device):
    """Verify actual noisy DiT inputs are equal across all caption/encoder routes."""
    import torch
    from tools.train_image_decoder import _seed_everything, _tensor_hash
    from tools.train_prism_image_connector import preserved_rng

    captures = []

    def capture(module, args, kwargs):
        captures.append(
            {
                "noisy_latent_sha256": _tensor_hash(kwargs["hidden_states"]),
                "timestep_sha256": _tensor_hash(kwargs["timestep"]),
                "timestep": kwargs["timestep"].detach().float().cpu().tolist(),
            }
        )

    hook = backend.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    expected = None
    result = {}
    try:
        with preserved_rng(), torch.no_grad():
            for route, pair in conditions.items():
                losses, predictions = {}, {}
                for caption_kind in ("matched", "wrong"):
                    value = pair[caption_kind]
                    _seed_everything(seed)
                    captures.clear()
                    prediction, loss = backend.training_step(
                        value["embeds"],
                        value["attention_mask"],
                        target,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        timesteps=torch.tensor([timestep], device=device, dtype=torch.float32),
                    )
                    if loss.ndim or not torch.isfinite(loss):
                        raise RuntimeError("Expected a finite scalar flow loss")
                    if len(captures) != 1:
                        raise RuntimeError("Expected one frozen diffusion forward per flow probe")
                    if expected is None:
                        expected = dict(captures[0])
                    elif captures[0] != expected:
                        raise RuntimeError("Paired control changed actual noisy image or timestep")
                    losses[caption_kind] = float(loss)
                    predictions[caption_kind] = prediction.detach().float()
                losses["wrong_minus_matched"] = losses["wrong"] - losses["matched"]
                losses["prediction_change_mse"] = float(
                    (predictions["matched"] - predictions["wrong"]).square().mean()
                )
                result[route] = losses
    finally:
        hook.remove()
    return {
        "seed": seed,
        "requested_timestep": timestep,
        "actual_inputs": expected,
        "routes": result,
    }


def sample_variants(
    backend,
    prompt,
    positive,
    negative,
    *,
    seed,
    args,
    output_dir,
    case_id,
    native_label="native_pretrained",
    only_native=False,
    saved_latent=None,
    skip_native=False,
):
    """Save small galleries with byte-identical initial latents across all routes."""
    import torch
    from src.decoders.loading import file_sha256
    from tools.train_image_decoder import _seed_everything, _tensor_hash
    from tools.train_prism_image_connector import preserved_rng

    variants = [] if skip_native else [(native_label, None, None, 5.0)]
    if not only_native:
        for mode in positive:
            variants.extend(
                [
                    (f"prism_{mode}_native_negative_cfg5", mode, "native", 5.0),
                    (f"prism_{mode}_cfg1", mode, None, 1.0),
                    (f"prism_{mode}_prism_negative_cfg5", mode, "prism", 5.0),
                ]
            )
    records = []
    with preserved_rng(), torch.no_grad():
        for label, mode, negative_kind, scale in variants:
            _seed_everything(seed)
            options = {
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.sampling_steps,
                "text_guidance_scale": scale,
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
            if mode is None:
                images = backend.generate_reference({"prompt": prompt}, **options)
            else:
                if negative_kind == "prism":
                    options.update(
                        negative_prompt_embeds=negative[mode]["embeds"],
                        negative_prompt_attention_mask=negative[mode]["attention_mask"],
                    )
                images = backend.generate_conditioned(
                    positive[mode]["embeds"], positive[mode]["attention_mask"], **options
                )
            latent = backend.last_trace.get("latents.initial")
            if not isinstance(latent, torch.Tensor):
                raise RuntimeError("No initial latent was captured for sampling replay")
            latent = latent.detach().cpu().clone()
            if saved_latent is None:
                saved_latent = latent
            elif not torch.equal(saved_latent, latent):
                raise RuntimeError("Generation ablation changed initial noise")
            images = images.images if hasattr(images, "images") else images
            if not isinstance(images, (list, tuple)) or not images:
                raise RuntimeError("No generated image sequence was returned")
            path = output_dir / f"sample-{case_id}-{label}.png"
            images[0].save(path)
            records.append(
                {
                    "case_id": case_id,
                    "route": label,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "seed": seed,
                    "initial_latent_sha256": _tensor_hash(latent),
                    "prompt": prompt,
                    "text_guidance_scale": scale,
                    "sampling_steps": args.sampling_steps,
                    "target_free": True,
                    "quality_claim": False,
                }
            )
            backend.last_trace = {}
    return records, saved_latent


def main(argv=None):
    args = _parser().parse_args(argv)
    validate_budget(args)
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Real conditioning diagnosis must run in a PBS compute allocation")
    with execution_policy(args):
        return _run(args)


def _run(args):
    import torch
    from src.data.image_generation_webdataset import ImageGenerationWebDataset
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.train_image_decoder import _seed_everything, frozen_state_hashes
    from tools.train_prism_image_diffusion import MODULES, restore_warm_connector

    for name in (
        "model_config",
        "checkpoint",
        "tokenizer",
        "source_processor",
        "train_index",
        "validation_index",
        "repeatability_report",
        "connector_checkpoint",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("Diagnostic output directory must be new")
    precheck = validate_repeatability_report(args.repeatability_report, args)
    datasets = {
        split: ImageGenerationWebDataset(index, target_size=(args.height, args.width), split=split)
        for split, index in (("train", args.train_index), ("validation", args.validation_index))
    }
    if any(len(dataset) < 2 for dataset in datasets.values()):
        raise ValueError("Both splits require at least two captions")
    if datasets["train"].data_fingerprint != datasets["validation"].data_fingerprint:
        raise ValueError("Train and validation conversion fingerprints differ")
    if any(
        record.task != "t2i" or record.source_ids or record.source_paths or record.split != split
        for split, dataset in datasets.items()
        for record in dataset.records
    ):
        raise ValueError("Diagnostic requires source-free T2I examples in designated splits")
    if {row.id for row in datasets["train"].records} & {
        row.id for row in datasets["validation"].records
    }:
        raise ValueError("Train and validation IDs overlap")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "evidence_kind": "real_checkpoint_conditioning_diagnostic",
        "qualification": "unqualified",
        "status": "running",
        "completed_steps": 0,
        "quality_benchmark": False,
        "training_performed": False,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runner_sha256": file_sha256(__file__),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "tools/prism_image_conditioning.py",
                "tools/train_prism_image_diffusion.py",
                "src/model.py",
                "src/decoders/conditioning.py",
                "src/decoders/image.py",
                "src/decoders/omnigen2_backend.py",
                "src/decoders/loading.py",
            )
        },
        "repeatability_precheck": precheck,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "host": platform.node(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "data_fingerprint": datasets["train"].data_fingerprint,
        "index_sha256": {split: file_sha256(dataset.index) for split, dataset in datasets.items()},
        "flow_protocol": "Fixed explicit timesteps, identical full RNG/generator seeds and verified actual noisy DiT inputs for all captions/routes. VAE target is never a PRISM input.",
        "negative_protocol": "Raw empty PRISM text uses an explicit EOS anchor if its tokenizer emits zero tokens; chat uses a formatted empty user message. Neither is assumed to have been trained as unconditional context.",
        "samples": [],
        "flow_controls": [],
        "condition_statistics": [],
    }
    model = None
    before = None
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
            raise ValueError("Expected strict complete parent restoration was not established")
        model.to(device=args.device, dtype=getattr(torch, args.dtype))
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
        report["connector_warm_start"] = restore_warm_connector(
            model,
            args.connector_checkpoint,
            parent=bundle["provenance"],
            reference_sha256=report["generator"]["manifest_sha256"],
            data_fingerprint=report["data_fingerprint"],
            frozen_hashes=frozen_state_hashes(model, (MODULES[0],)),
            expected_step=args.expected_connector_step,
            expected_sha256=args.connector_checkpoint_sha256,
            fixture=report["evidence_kind"] == "fixture_only",
        )
        train_indices = ShuffledEpochOrder(len(datasets["train"]), args.seed).order
        joint_report = None
        if args.joint_checkpoint is not None:
            args.joint_checkpoint = args.joint_checkpoint.resolve(strict=True)
            joint_report = json.loads((args.joint_checkpoint.parent / "report.json").read_text())
            if joint_report.get("data_fingerprint") != report["data_fingerprint"]:
                raise ValueError("Joint checkpoint uses a different data conversion")
            if joint_report.get("train_selection"):
                train_indices = [entry["index"] for entry in joint_report["train_selection"]]
                for entry in joint_report["train_selection"]:
                    if datasets["train"].records[entry["index"]].id != entry["id"]:
                        raise ValueError("Joint training selection no longer matches data IDs")
        selections = {
            "train": train_indices[: args.train_probe_count],
            "validation": list(
                range(min(args.validation_probe_count, len(datasets["validation"])))
            ),
        }
        report["selection"] = {
            split: [{"index": i, "id": datasets[split].records[i].id} for i in indices]
            for split, indices in selections.items()
        }
        latents = {}
        with preserved_model_state(model), torch.no_grad():
            _phase(report, report_path, "native_pretrained_baseline")
            original_hashes = frozen_state_hashes(model, ())
            for position in range(min(args.sample_count, len(datasets["validation"]))):
                row = datasets["validation"].records[position]
                samples, latent = sample_variants(
                    backend,
                    row.prompt,
                    {},
                    {},
                    seed=args.seed + 200000 + position,
                    args=args,
                    output_dir=args.output_dir,
                    case_id=f"validation-{position:02d}",
                    only_native=True,
                )
                report["samples"].extend(samples)
                latents[position] = latent
                _write(report_path, report)
            if frozen_state_hashes(model, ()) != original_hashes:
                raise RuntimeError("Native baseline mutated a frozen tensor")
            report["native_pretrained_state_unchanged"] = True
            if args.joint_checkpoint is not None:
                from tools.train_prism_image_diffusion import (
                    configure_joint_scope,
                    restore_joint_stage,
                )

                _phase(report, report_path, "restoring_joint_checkpoint")
                _, lineage = restore_joint_stage(
                    model,
                    args.joint_checkpoint,
                    named_groups=configure_joint_scope(model),
                    parent=bundle["provenance"],
                    reference_sha256=report["generator"]["manifest_sha256"],
                    frozen_hashes=frozen_state_hashes(model, MODULES),
                    expected_sha256=args.joint_checkpoint_sha256,
                    fixture=report["evidence_kind"] == "fixture_only",
                    restore_masters=False,
                )
                report["joint_checkpoint"] = lineage
                backend.configure_training(train_diffusion=False, gradient_checkpointing=False)
                model.eval()
                model.requires_grad_(False)
            before = frozen_state_hashes(model, ())
            report["frozen_hashes_before"] = before
            native_label = (
                "native_adapted_diffusion" if args.joint_checkpoint else "native_pretrained"
            )
            negative = {
                mode: encode_prism_prompt(
                    model,
                    tokenizer,
                    "",
                    device=args.device,
                    mode=mode,
                    max_text_length=args.max_text_length,
                )
                for mode in args.prism_formats
            }
            report["negative_condition_statistics"] = {
                mode: condition_metadata(value) for mode, value in negative.items()
            }
            report["native_negative_condition_statistics"] = condition_metadata(
                encode_native_prompt(backend, "", max_text_length=args.max_text_length)
            )
            for split, indices in selections.items():
                for position, index in enumerate(indices):
                    row = datasets[split].records[index]
                    wrong_index = select_wrong_caption(
                        datasets[split].records,
                        index,
                        train_indices if split == "train" else list(range(len(datasets[split]))),
                    )
                    wrong = datasets[split].records[wrong_index]
                    _phase(
                        report, report_path, "flow_controls", active_id=row.id, active_split=split
                    )
                    conditions = {
                        native_label: {
                            kind: encode_native_prompt(
                                backend, prompt, max_text_length=args.max_text_length
                            )
                            for kind, prompt in (("matched", row.prompt), ("wrong", wrong.prompt))
                        }
                    }
                    conditions.update(
                        {
                            f"prism_{mode}": {
                                kind: encode_prism_prompt(
                                    model,
                                    tokenizer,
                                    prompt,
                                    device=args.device,
                                    mode=mode,
                                    max_text_length=args.max_text_length,
                                )
                                for kind, prompt in (
                                    ("matched", row.prompt),
                                    ("wrong", wrong.prompt),
                                )
                            }
                            for mode in args.prism_formats
                        }
                    )
                    report["condition_statistics"].append(
                        {
                            "split": split,
                            "id": row.id,
                            "routes": {
                                route: {
                                    kind: condition_metadata(value) for kind, value in pair.items()
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
                        )
                        result.update(
                            split=split, index=index, id=row.id, wrong_id=wrong.id, repeat=repeat
                        )
                        report["flow_controls"].append(result)
                    _write(report_path, report)
                    if split == "validation" and position < args.sample_count:
                        samples, _ = sample_variants(
                            backend,
                            row.prompt,
                            {
                                mode: conditions[f"prism_{mode}"]["matched"]
                                for mode in args.prism_formats
                            },
                            negative,
                            seed=args.seed + 200000 + position,
                            args=args,
                            output_dir=args.output_dir,
                            case_id=f"validation-{position:02d}",
                            native_label=native_label,
                            saved_latent=latents[position],
                            skip_native=args.joint_checkpoint is None,
                        )
                        report["samples"].extend(samples)
                        _write(report_path, report)
                    del conditions, target
            report["summary"] = summarize_caption_controls(report["flow_controls"])
            report["frozen_hashes_after"] = frozen_state_hashes(model, ())
            report["frozen_state_unchanged"] = report["frozen_hashes_after"] == before
            if not report["frozen_state_unchanged"]:
                raise RuntimeError("Conditioning diagnosis mutated a frozen tensor")
        report.update(
            status="completed", phase="completed", duration_seconds=time.monotonic() - started
        )
        _write(report_path, report)
        return report
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            duration_seconds=time.monotonic() - started,
        )
        if model is not None and before is not None:
            report["frozen_hashes_after"] = frozen_state_hashes(model, ())
            report["frozen_state_unchanged"] = report["frozen_hashes_after"] == before
        _write(report_path, report)
        raise


if __name__ == "__main__":
    main()
