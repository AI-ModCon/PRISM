#!/usr/bin/env python3
"""Experimental full dense diffusion plus connector training from a warm connector.

PRISM, VAE and native conditioner remain frozen. BF16 diffusion compute uses
original FP32 checkpoint masters and CPU AdamW. This is a separate unqualified
training contract; it does not establish P0/P1/P2 acceptance or visual quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.overfit_prism_image_connector import _phase, _restore_rng, _write, execution_policy

EVIDENCE_KIND = "real_checkpoint_connector_diffusion_webdataset_pilot"
MODULES = ("decoders.image.connector", "decoders.image.backend.transformer")
CONNECTOR_PREFIX = MODULES[0] + "."
DIFFUSION_PREFIX = MODULES[1] + "."


from tools.train_prism_image_connector import (
    ShuffledEpochOrder,
    preserved_rng,
    target_free_item,
    validate_repeatability_report,
)


def configure_joint_scope(model):
    """Opt in to dense DiT training while every other branch stays frozen/eval."""
    model.requires_grad_(False)
    decoder = model.decoders["image"]
    decoder.configure_training(train_diffusion=True, gradient_checkpointing=True)
    model.train()
    # A frozen PRISM in training mode could still apply dropout. Do not recurse
    # through parents here: that would overwrite the selected child modes.
    for name, module in model.named_modules():
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in MODULES):
            module.training = False
    groups = {
        "connector": {
            name: value
            for name, value in model.named_parameters()
            if name.startswith(CONNECTOR_PREFIX)
        },
        "diffusion": {
            name: value
            for name, value in model.named_parameters()
            if name.startswith(DIFFUSION_PREFIX)
        },
    }
    selected = {name for values in groups.values() for name in values}
    actual = {name for name, value in model.named_parameters() if value.requires_grad}
    if not all(groups.values()) or actual != selected:
        raise ValueError("Trainable scope must be the complete connector and diffusion transformer")
    ids = {id(value) for values in groups.values() for value in values.values()}
    if any(
        id(value) in ids and name not in selected
        for name, value in model.named_parameters(remove_duplicate=False)
    ):
        raise ValueError("A selected parameter aliases a frozen parameter")
    return groups


@contextmanager
def evaluation_mode(model):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode


def restore_warm_connector(
    model,
    path,
    *,
    parent,
    reference_sha256,
    data_fingerprint,
    frozen_hashes,
    expected_step,
    expected_sha256=None,
    fixture=False,
):
    """Accept a complete, audited connector pilot, never its old Adam moments."""
    import torch
    from src.decoders.loading import file_sha256

    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Connector checkpoint SHA256 does not match the requested warm start")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected_kind = "fixture_only" if fixture else "real_checkpoint_connector_webdataset_pilot"
    if (
        saved.get("schema_version") != 1
        or saved.get("evidence_kind") != expected_kind
        or saved.get("qualification") != "unqualified"
        or saved.get("step") != expected_step
        or saved.get("connector_modules") != [MODULES[0]]
    ):
        raise ValueError("Warm start requires the designated complete connector pilot checkpoint")
    protocol = saved.get("protocol", {})
    if (
        protocol.get("parent") != parent
        or protocol.get("reference_checkpoint_sha256") != reference_sha256
        or protocol.get("data_fingerprint") != data_fingerprint
    ):
        raise ValueError("Warm connector parent, generator or dataset differs")
    if any(
        saved.get("frozen_hashes", {}).get(name) != value for name, value in frozen_hashes.items()
    ):
        raise ValueError("Warm connector frozen parent/native state differs")
    report_path = path.parent / "report.json"
    prior_report = json.loads(report_path.read_text())
    if (
        prior_report.get("status") != "completed"
        or prior_report.get("evidence_kind") != expected_kind
        or prior_report.get("frozen_state_unchanged") is not True
        or prior_report.get("frozen_hashes_before") != prior_report.get("frozen_hashes_after")
        or prior_report.get("completed_steps", -1) < expected_step
        or not any(
            row.get("sha256") == digest and row.get("step") == expected_step
            for row in prior_report.get("checkpoints", [])
        )
    ):
        raise ValueError("Warm connector needs its completed, unchanged-frozen-state report")
    state = saved.get("connector_state_dict", {})
    expected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(CONNECTOR_PREFIX)
    }
    if set(state) != set(expected) or not state:
        raise ValueError("Warm connector state is incomplete")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.shape != expected[name].shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(
                f"Warm connector tensor must be finite full-precision with matching shape: {name}"
            )
    model.decoders["image"].connector.load_state_dict(
        {name[len(CONNECTOR_PREFIX) :]: value for name, value in state.items()}, strict=True
    )
    return {
        "checkpoint": str(path),
        "sha256": digest,
        "step": expected_step,
        "report_sha256": file_sha256(report_path),
        "fresh_optimizer": True,
        "restored_connector_tensors": len(state),
    }


def selected_training_indices(count, subset_size, seed):
    """Keep the full default order, or select IDs from its first seeded epoch."""
    if count < 2 or subset_size < 0 or subset_size == 1 or subset_size > count:
        raise ValueError("Training subset must be 0 (full pool), or between 2 and train count")
    if subset_size == 0:
        return list(range(count))
    return ShuffledEpochOrder(count, seed).order[:subset_size]


def sample_exposure(order, train_indices, records):
    """Exact exposure implied by finite epochs; includes zero-count selected IDs."""
    counts = {records[index].id: order.epoch for index in train_indices}
    for position in order.order[: order.cursor]:
        counts[records[train_indices[position]].id] += 1
    if sum(counts.values()) != order.examples_seen:
        raise RuntimeError("Training exposure and sampler cursor disagree")
    return {
        "examples_seen": order.examples_seen,
        "unique_examples_seen": sum(count > 0 for count in counts.values()),
        "selected_example_count": len(train_indices),
        "full_train_example_count": len(records),
        "selected_equivalent_epochs": order.examples_seen / len(train_indices),
        "full_pool_equivalent_epochs": order.examples_seen / len(records),
        "per_id_counts": counts,
    }


def restore_joint_stage(
    model,
    path,
    *,
    named_groups,
    parent,
    reference_sha256,
    frozen_hashes,
    expected_sha256=None,
    fixture=False,
    restore_masters=True,
):
    """Restore complete trained weights, discarding previous Adam/RNG/data progress.

    The source must be the terminal checkpoint of a completed audited run. It
    need not have the new stage's data, optimizer settings, or source version.
    Original parent, generator asset identity, and frozen tensors must match.
    """
    import torch
    from src.decoders.loading import file_sha256

    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Joint initialization checkpoint SHA256 differs")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected_kind = "fixture_only" if fixture else EVIDENCE_KIND
    step = saved.get("step")
    if (
        saved.get("schema_version") != 1
        or saved.get("evidence_kind") != expected_kind
        or saved.get("qualification") != "unqualified"
        or saved.get("connector_modules") != list(MODULES)
        or saved.get("master_state_includes_full_diffusion") is not True
        or not isinstance(step, int)
        or step < 1
    ):
        raise ValueError("Joint initialization requires a complete joint checkpoint")
    protocol = saved.get("protocol", {})
    if (
        protocol.get("parent") != parent
        or protocol.get("reference_checkpoint_sha256") != reference_sha256
        or saved.get("frozen_hashes") != frozen_hashes
    ):
        raise ValueError("Joint initialization parent, generator or frozen identity differs")
    report_path = path.parent / "report.json"
    prior = json.loads(report_path.read_text())
    if (
        prior.get("status") != "completed"
        or prior.get("evidence_kind") != expected_kind
        or prior.get("completed_steps") != step
        or prior.get("frozen_state_unchanged") is not True
        or prior.get("frozen_hashes_before") != frozen_hashes
        or prior.get("frozen_hashes_after") != frozen_hashes
        or not any(
            row.get("sha256") == digest and row.get("step") == step
            for row in prior.get("checkpoints", [])
        )
    ):
        raise ValueError("Joint initialization needs its completed terminal checkpoint report")
    runtime = {name: value for group in named_groups.values() for name, value in group.items()}
    master_state = saved.get("optimizer_state_dict", {}).get("masters", {})
    if set(master_state) != set(runtime):
        raise ValueError("Joint initialization FP32 masters are incomplete")
    for name, value in master_state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.shape != runtime[name].shape
            or not value.is_contiguous()
            or not torch.isfinite(value).all()
            or not torch.isfinite(value.to(dtype=runtime[name].dtype)).all()
        ):
            raise ValueError(f"Joint initialization master is invalid: {name}")
    connector = saved.get("connector_state_dict", {})
    expected_connector = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(CONNECTOR_PREFIX)
    }
    if set(connector) != set(expected_connector):
        raise ValueError("Joint initialization connector state is incomplete")
    for name, value in connector.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.shape != expected_connector[name].shape
            or not torch.isfinite(value).all()
            or (name in master_state and not torch.equal(value, master_state[name]))
        ):
            raise ValueError("Joint initialization connector disagrees with FP32 masters")
    buffers = dict(model.decoders["image"].backend.transformer.named_buffers())
    buffer_state = saved.get("diffusion_buffer_state")
    if not isinstance(buffer_state, dict) or set(buffers) != set(buffer_state):
        raise ValueError("Joint initialization diffusion buffers are incomplete")
    for name, value in buffer_state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != buffers[name].shape
            or value.dtype != buffers[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Joint initialization diffusion buffer shape, dtype or values differ")
    # Copy only after validating every tensor, and detach mmap storage from the
    # new optimizer. Old Adam moments remain file-backed and are never restored.
    masters = {}
    with torch.no_grad():
        for name, value in master_state.items():
            if restore_masters:
                masters[name] = value.clone(memory_format=torch.contiguous_format)
            runtime[name].copy_(value)
        model.decoders["image"].connector.load_state_dict(
            {name[len(CONNECTOR_PREFIX) :]: value for name, value in connector.items()}, strict=True
        )
        for name, value in buffer_state.items():
            buffers[name].copy_(value)
    return masters, {
        "policy": "completed_joint_checkpoint_fp32_masters_fresh_stage",
        "checkpoint": str(path),
        "sha256": digest,
        "step": step,
        "report_sha256": file_sha256(report_path),
        "source_data_fingerprint": protocol.get("data_fingerprint"),
        "fresh_optimizer": True,
        "fresh_rng": True,
        "fresh_sampler": True,
        "restored_connector_tensors": len(connector),
        "restored_diffusion_buffers": len(buffers),
        "restored_master_tensors": len(master_state),
        "restored_master_parameters": sum(value.numel() for value in master_state.values()),
        "returned_optimizer_masters": bool(restore_masters),
    }


def load_original_masters(directory, runtime_parameters, manifest):
    """Load original FP32 safetensors one tensor at a time, never BF16 upcasts.

    Cloning each mmap-backed tensor transfers writable independent storage to the
    optimizer. At most one extra source tensor is live; the entire checkpoint is
    never loaded twice. Every shard is bound to the loaded backend's file audit.
    """
    import torch
    from safetensors import safe_open
    from src.decoders.loading import file_sha256

    directory = Path(directory).resolve(strict=True)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError("No original diffusion safetensors were found")
    expected = {
        name[len(DIFFUSION_PREFIX) :]: (name, value) for name, value in runtime_parameters.items()
    }
    if not expected or any(not name.startswith(DIFFUSION_PREFIX) for name in runtime_parameters):
        raise ValueError("Master loading requires the full named diffusion scope")
    masters, hashes, source_counts, extras = {}, {}, {}, []
    for path in files:
        relative = "transformer/" + path.name
        digest = file_sha256(path)
        if manifest.get("files", {}).get(relative) != digest:
            raise ValueError(f"Original diffusion shard identity mismatch: {relative}")
        hashes[relative] = digest
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key not in expected:
                    extras.append(key)
                    continue  # Persistent buffers are not optimizer parameters.
                name, runtime = expected[key]
                if name in masters:
                    raise ValueError(f"Duplicate original diffusion parameter: {name}")
                tensor = source.get_tensor(key)
                if tensor.dtype != torch.float32 or tensor.shape != runtime.shape:
                    raise ValueError(
                        f"Original diffusion master must be FP32 with matching shape: {name}"
                    )
                masters[name] = tensor.clone(memory_format=torch.contiguous_format)
                source_counts[name] = tensor.numel()
                del tensor
    if set(masters) != set(runtime_parameters):
        raise ValueError(
            f"Missing original diffusion parameters: {sorted(set(runtime_parameters) - set(masters))}"
        )
    return masters, {
        "policy": "exact_original_fp32_safetensors",
        "files": hashes,
        "tensor_count": len(masters),
        "parameter_count": sum(source_counts.values()),
        "checkpoint_extra_nonparameter_keys": extras,
    }


def validate_update_diagnostics(diagnostics):
    norm = diagnostics.get("gradient_norm_before_clip")
    if not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Joint training requires finite nonzero gradients")
    for group in ("connector", "diffusion"):
        detail = diagnostics.get("groups", {}).get(group, {})
        if not detail.get("nonzero_gradient_names"):
            raise RuntimeError(f"Training group received no nonzero gradient: {group}")


def audit_groups_changed(before, after):
    return {
        group: all(
            before["groups"][group][kind] != after["groups"][group][kind]
            for kind in ("master_sha256", "runtime_sha256")
        )
        for group in ("connector", "diffusion")
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--connector-checkpoint", type=Path)
    source.add_argument(
        "--init-joint-checkpoint",
        type=Path,
        help="Initialize a fresh stage from completed joint weights; do not restore Adam or data progress",
    )
    parser.add_argument("--init-joint-checkpoint-sha256")
    parser.add_argument(
        "--init-alignment-checkpoint",
        type=Path,
        help="After the audited 500-step warm start, initialize aligned connector weights with the original DiT",
    )
    parser.add_argument("--init-alignment-checkpoint-sha256")
    parser.add_argument(
        "--resume", type=Path, help="Exactly resume this initialization source and protocol"
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=0,
        help="0 uses the full training pool; otherwise select this many IDs from its seeded first epoch",
    )
    parser.add_argument("--connector-checkpoint-sha256")
    parser.add_argument("--expected-connector-step", type=int, default=500)
    parser.add_argument("--transformer-checkpoint-dir", type=Path)
    parser.add_argument("--diffusion-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--steps", type=int, default=100, help="Total optimizer steps, including resume"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, help="0 saves only the terminal joint checkpoint"
    )
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
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Connector learning rate")
    parser.add_argument("--text-guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument(
        "--negative-conditioning",
        choices=("native", "prism"),
        default="native",
        help="Sampling-only negative branch; PRISM empty input is an experimental anchor, not trained unconditional conditioning",
    )
    parser.add_argument("--expected-parent-tensors", type=int, default=526)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="math")
    return parser


def validate_budget(args):
    if args.init_alignment_checkpoint is not None:
        if args.init_joint_checkpoint is not None or args.connector_checkpoint is None:
            raise ValueError(
                "Alignment initialization requires --connector-checkpoint and is incompatible with --init-joint-checkpoint"
            )
        if args.prompt_format != "chat":
            raise ValueError("Alignment initialization requires --prompt-format chat")
        if args.expected_connector_step != 500:
            raise ValueError(
                "Alignment initialization requires the audited 500-step connector warm start"
            )
    elif args.init_alignment_checkpoint_sha256 is not None:
        raise ValueError("Alignment checkpoint digest requires --init-alignment-checkpoint")
    for key, lower, upper in (
        ("steps", 1, 50000),
        ("batch_size", 1, 4),
        ("gradient_accumulation", 1, 32),
        ("checkpoint_every", 0, 50000),
        ("eval_every", 1, 50000),
        ("sample_every", 1, 50000),
        ("probe_count", 1, 100),
        ("final_validation_count", 0, 100),
        ("sample_count", 0, 2),
        ("sampling_steps", 1, 50),
    ):
        if not lower <= getattr(args, key) <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    if not math.isfinite(args.diffusion_learning_rate) or args.diffusion_learning_rate <= 0:
        raise ValueError("Diffusion learning rate must be finite and positive")
    if args.train_subset_size < 0 or args.train_subset_size == 1:
        raise ValueError("train_subset_size must be 0 or at least 2")
    if args.init_joint_checkpoint is not None and args.native_baseline:
        raise ValueError(
            "Native-pretrained baseline requires the original diffusion weights, not joint stage initialization"
        )
    if args.expected_connector_step < 1:
        raise ValueError("Expected connector step must be positive")
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
    for name in ("connector_checkpoint", "init_joint_checkpoint", "init_alignment_checkpoint"):
        if getattr(args, name) is not None:
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
    train_indices = selected_training_indices(
        len(records["train"]), args.train_subset_size, args.seed
    )
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
        "training_scope": ["connector", "full_diffusion_transformer"],
        "gradient_policy": "No per-tensor nonzero requirement: text-to-image may leave reference branches inactive; both groups must have finite nonzero gradients.",
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
                "tools/train_prism_image_connector.py",
                "tools/prism_image_conditioning.py",
                "src/training/cpu_master_adamw.py",
            )
        },
        "repeatability_precheck": precheck,
        "train_count": len(datasets["train"]),
        "selected_train_count": len(train_indices),
        "train_selection": [
            {"index": index, "id": records["train"][index].id} for index in train_indices
        ],
        "validation_count": len(datasets["validation"]),
        "data_fingerprint": datasets["train"].data_fingerprint,
        "data_validation": datasets["train"].validation_report,
        "train_index_sha256": file_sha256(args.train_index),
        "validation_index_sha256": file_sha256(args.validation_index),
        "data_order": "seeded independent shuffled finite epochs; explicit next-unread cursor; no workers/prefetch",
        "probe_protocol": "Full RNG reset per example fixes VAE posterior, timestep and noise; wrong caption uses identical target/noise. Final validation uses every record unless an explicit smoke limit is configured.",
        "sample_protocol": "Metadata-only text input; saved initial latent replay; no target pixels opened by sampling",
        "optimizer": {
            "name": "CPUMasterAdamW",
            "connector_learning_rate": args.learning_rate,
            "diffusion_learning_rate": args.diffusion_learning_rate,
            "master_dtype": "torch.float32",
            "diffusion_compute_dtype": args.dtype,
            "state_device": "cpu",
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
        before = frozen_state_hashes(model, MODULES)
        report["frozen_hashes_before"] = before
        warm = None
        if args.connector_checkpoint is not None:
            warm = restore_warm_connector(
                model,
                args.connector_checkpoint,
                parent=bundle["provenance"],
                reference_sha256=report["generator"]["manifest_sha256"],
                data_fingerprint=report["data_fingerprint"],
                frozen_hashes=before,
                expected_step=args.expected_connector_step,
                expected_sha256=args.connector_checkpoint_sha256,
                fixture=report["evidence_kind"] == "fixture_only",
            )
        report["connector_warm_start"] = warm
        if args.init_alignment_checkpoint is not None:
            from tools.prism_image_alignment_checkpoint import restore_alignment_connector

            # The strict adapter verifies original DiT/teacher/parent weights.
            # Restore before enabling DiT gradients or creating FP32 masters.
            model.requires_grad_(False)
            _phase(report, report_path, "restoring_aligned_connector")
            lineage = restore_alignment_connector(
                model,
                args.init_alignment_checkpoint,
                parent=bundle["provenance"],
                generator_manifest=report["generator"],
                data_fingerprint=report["data_fingerprint"],
                index_sha256={
                    "train": report["train_index_sha256"],
                    "validation": report["validation_index_sha256"],
                },
                records=records,
                tokenizer=bundle["tokenizer"],
                expected_sha256=args.init_alignment_checkpoint_sha256,
                fixture=report["evidence_kind"] == "fixture_only",
            )
            report["alignment_initialization"] = dict(
                lineage,
                fresh_optimizer=True,
                fresh_sampler=True,
                fresh_rng=True,
                stage_initialization="aligned_connector_original_diffusion",
            )
            report["source_sha256"].update(
                {
                    name: file_sha256(ROOT / name)
                    for name in (
                        "tools/align_prism_image_conditioning.py",
                        "tools/prism_image_alignment_checkpoint.py",
                    )
                }
            )
        groups = configure_joint_scope(model)
        report["pretrained_generator_runtime"] = report["generator_runtime"]
        report["generator_runtime"] = backend.provenance()
        trainable = {name: value for group in groups.values() for name, value in group.items()}
        report["trainable_parameters"] = {
            name: list(value.shape) for name, value in trainable.items()
        }
        report["trainable_parameter_counts"] = {
            group: sum(value.numel() for value in values.values())
            for group, values in groups.items()
        }
        report["trainable_parameter_count"] = sum(report["trainable_parameter_counts"].values())
        report["trainable_parameter_dtypes"] = {
            name: str(value.dtype) for name, value in trainable.items()
        }
        from src.training.cpu_master_adamw import CPUMasterAdamW

        if args.init_joint_checkpoint is not None:
            master_values, master_source = restore_joint_stage(
                model,
                args.init_joint_checkpoint,
                named_groups=groups,
                parent=bundle["provenance"],
                reference_sha256=report["generator"]["manifest_sha256"],
                frozen_hashes=before,
                expected_sha256=args.init_joint_checkpoint_sha256,
                fixture=report["evidence_kind"] == "fixture_only",
            )
            report["joint_stage_initialization"] = master_source
        else:
            master_values, master_source = load_original_masters(
                args.transformer_checkpoint_dir or Path(backend.model_id) / "transformer",
                groups["diffusion"],
                report["generator"],
            )
            master_values.update(
                {
                    name: value.detach().cpu().float().clone()
                    for name, value in groups["connector"].items()
                }
            )
        report["master_initialization"] = master_source
        _phase(report, report_path, "initializing_cpu_optimizer")
        optimizer = CPUMasterAdamW(
            named_groups=groups,
            master_values=master_values,
            learning_rates={
                "connector": args.learning_rate,
                "diffusion": args.diffusion_learning_rate,
            },
            max_grad_norm=1.0,
            weight_decay=0.0,
        )
        del master_values
        report["trainable_audit_before"] = optimizer.audit()
        collator = ImageGenerationCollator(bundle["tokenizer"])
        order = ShuffledEpochOrder(len(train_indices), args.seed)
        # Follow the first optimizer examples rather than arbitrary index rows;
        # a bounded pilot may visit only a small fraction of the training pool.
        initial_train_order = [train_indices[position] for position in order.order]
        initial_train_rank = {index: rank for rank, index in enumerate(initial_train_order)}
        selections = {
            "train": initial_train_order[: min(args.probe_count, len(train_indices))],
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
            "train_selection": report["train_selection"],
            "connector_warm_start": warm,
            "alignment_initialization": report.get("alignment_initialization"),
            "master_initialization": master_source,
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
        control_indices = {
            "train": train_indices,
            "validation": list(range(len(records["validation"]))),
        }
        control_ranks = {
            split: {index: position for position, index in enumerate(indices)}
            for split, indices in control_indices.items()
        }
        from tools.prism_image_conditioning import encode_prism_prompt, format_items

        def batch_for(items):
            formatted = format_items(items, bundle["tokenizer"], args.prompt_format)
            return _to_device(collator(formatted), torch.device(args.device))

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
            with preserved_rng(), evaluation_mode(model), torch.no_grad():
                for split_number, split in enumerate(("train", "validation")):
                    count = (
                        len(datasets[split])
                        if final and split == "validation"
                        else len(selections[split])
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
                                records[split][candidate]
                                for candidate in (
                                    control_indices[split][
                                        (control_ranks[split][index] + offset)
                                        % len(control_indices[split])
                                    ]
                                    for offset in range(1, len(control_indices[split]))
                                )
                                if records[split][candidate].prompt != row.prompt
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
            with preserved_rng(), evaluation_mode(model), torch.no_grad():
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
                    negative = None
                    if not native and args.negative_conditioning == "prism":
                        negative = encode_prism_prompt(
                            model,
                            bundle["tokenizer"],
                            args.negative_prompt,
                            device=args.device,
                            mode=args.prompt_format,
                        )
                        options["negative_prompt_embeds"] = negative["embeds"]
                        options["negative_prompt_attention_mask"] = negative["attention_mask"]
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
                            "prompt_format": "native" if native else args.prompt_format,
                            "negative_conditioning": "native"
                            if native
                            else args.negative_conditioning,
                            "negative_empty_anchor": negative["empty_anchor"]
                            if negative is not None
                            else None,
                            "quality_claim": False,
                        }
                    )
                    backend.last_trace = {}
                    _write(report_path, report)

        def checkpoint(step):
            path = args.output_dir / f"connector-diffusion-pilot-step-{step:06d}.pt"
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
                    "master_state_includes_full_diffusion": True,
                    "diffusion_buffer_state": {
                        name: value.detach().cpu().clone()
                        for name, value in backend.transformer.named_buffers()
                    },
                    "rng_state": _rng_state(),
                    "sampler_state": order.state_dict(),
                    "sample_exposure": sample_exposure(order, train_indices, records["train"]),
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
            report["sample_exposure"] = sample_exposure(order, train_indices, records["train"])
            _write(report_path, report)

        if args.resume is None:
            start_step = 0
            report["initial_evaluation"] = evaluate(0)
            sample("warm-start", 0)
            if args.native_baseline:
                sample("native-pretrained", 0, native=True)
            report["origin_baseline_samples"] = list(report["samples"])
        else:
            saved = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
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
            if saved.get("master_state_includes_full_diffusion") is not True:
                raise ValueError("Resume checkpoint lacks full diffusion master state")
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            buffers = dict(backend.transformer.named_buffers())
            buffer_state = saved.get("diffusion_buffer_state")
            if not isinstance(buffer_state, dict) or set(buffers) != set(buffer_state):
                raise ValueError("Resume diffusion buffer state is incomplete")
            with torch.no_grad():
                for name, value in buffer_state.items():
                    if value.shape != buffers[name].shape or value.dtype != buffers[name].dtype:
                        raise ValueError("Resume diffusion buffer shape or dtype changed")
                    buffers[name].copy_(value)
            restored_connector = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if name.startswith(CONNECTOR_PREFIX)
            }
            if any(
                not torch.equal(restored_connector[name], value) for name, value in state.items()
            ):
                raise ValueError("Resume connector disagrees with restored optimizer masters")
            order.load_state_dict(saved["sampler_state"])
            if saved.get("sample_exposure") != sample_exposure(
                order, train_indices, records["train"]
            ):
                raise ValueError("Resume per-ID exposure disagrees with sampler state")
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
                optimizer.zero_grad()
                microbatches, losses = [], []
                for microbatch in range(args.gradient_accumulation):
                    indices = [train_indices[position] for position in order.take(args.batch_size)]
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
                missing = [name for name, value in trainable.items() if value.grad is None]
                diagnostics = optimizer.step()
                validate_update_diagnostics(diagnostics)
                for detail in diagnostics["groups"].values():
                    nonzero_names.update(detail["nonzero_gradient_names"])
                    detail["nonzero_gradient_tensors"] = len(detail.pop("nonzero_gradient_names"))
                # Dense BF16 gradients are several GiB; they are unnecessary at
                # optimizer boundaries and must not occupy sampling/eval memory.
                optimizer.zero_grad()
                report["last_optimizer_diagnostics"] = diagnostics
                report["completed_steps"] = step
                entry = {
                    "step": step,
                    "loss": sum(losses) / len(losses),
                    "microbatch_losses": losses,
                    "microbatches": microbatches,
                    "examples_seen": order.examples_seen,
                    "unique_examples_seen": min(order.examples_seen, len(train_indices)),
                    "selected_equivalent_epochs": order.examples_seen / len(train_indices),
                    "full_pool_equivalent_epochs": order.examples_seen / len(records["train"]),
                    "epoch": order.epoch,
                    "epoch_cursor": order.cursor,
                    "optimizer_diagnostics": diagnostics,
                    "missing_gradient_names": missing,
                    "duration_seconds": time.monotonic() - tick,
                }
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                final = step == args.steps
                if final or (args.checkpoint_every and step % args.checkpoint_every == 0):
                    checkpoint(step)
                if final or step % args.eval_every == 0:
                    evaluate(step, final=final)
                if final or step % args.sample_every == 0:
                    sample("trained", step)
                if final or step % 10 == 0:
                    _phase(report, report_path, "training")
        report["sample_exposure"] = sample_exposure(order, train_indices, records["train"])
        report["trainable_audit_after"] = optimizer.audit()
        report["nonzero_gradient_parameters"] = sorted(nonzero_names)
        report["trainable_groups_changed"] = audit_groups_changed(
            report["trainable_audit_before"], report["trainable_audit_after"]
        )
        if not all(report["trainable_groups_changed"].values()):
            raise RuntimeError("Connector and diffusion groups must both change")
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
