#!/usr/bin/env python3
"""Bounded caption-feature alignment, with a frozen native teacher and DiT.

Only the PRISM image connector is optimized. This is a separate experimental
objective, not a flow-training checkpoint or evidence of image quality. Caption
metadata alone builds detached CPU caches; target images are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
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
from tools.train_prism_image_connector import ShuffledEpochOrder, validate_repeatability_report

EVIDENCE_KIND = "real_checkpoint_connector_native_feature_alignment"
FIXTURE_KIND = "fixture_only_connector_native_feature_alignment"
MODULES = ("decoders.image.connector",)
PREFIX = MODULES[0] + "."
OBJECTIVE = {
    "name": "native_caption_rmsnorm_content_mse",
    "formula": "mean_examples(mean_content_tokens(mean_features((N(C(H_prism)) - N(H_native))^2)))",
    "C": "existing FP32 PRISM image connector LayerNorm plus Linear",
    "N": "frozen DiT time_caption_embed.caption_embedder[0], including learned channel weights",
    "precision": "FP32 connector and AdamW; cast connector output to native conditioning dtype before N; FP32 MSE after N",
    "content_span": "unique exact raw-caption token subsequence in identical untruncated full chat IDs",
    "template_positions": "excluded from optimization; reported separately",
    "example_weighting": "equal weight per caption, independent of content token count",
    "negative_anchor_trained": False,
    "flow_loss_used": False,
    "image_quality_established": False,
}
REGIONS_EVIDENCE_KIND = "real_checkpoint_connector_native_region_feature_alignment"
REGIONS_FIXTURE_KIND = "fixture_only_connector_native_region_feature_alignment"
REGIONS_OBJECTIVE = {
    **OBJECTIVE,
    "name": "native_caption_rmsnorm_equal_region_mse_v2",
    "formula": "mean_examples((mean_prefix_tokens(feature_MSE) + mean_content_tokens(feature_MSE) + mean_suffix_tokens(feature_MSE)) / 3)",
    "template_positions": "prefix and suffix each optimized with equal weight to caption content",
    "region_weighting": {"prefix": 1 / 3, "content": 1 / 3, "suffix": 1 / 3},
    "regions": "nonempty disjoint prefix/content/suffix partitions of all valid full-chat positions",
}
FUTURE_ADAPTER_CONTRACT = {
    "implemented": False,
    "evidence_kind_required": EVIDENCE_KIND,
    "must_validate": [
        "completed terminal checkpoint digest and report",
        "parent and original pretrained generator identities",
        "complete finite FP32 connector state and unchanged frozen hashes",
        "chat prompt format, exact token identity and content-span audits",
        "caption RMSNorm state identity and alignment objective",
        "data identity and recorded training/heldout IDs",
    ],
    "restore": "connector weights only, with fresh downstream optimizer; retain alignment provenance",
    "downstream_validation": "paired flow controls and target-free no-CFG generation before any quality claim",
    "must_not": "rename this evidence kind or treat feature loss as flow training or image-generation accuracy",
}


def caption_content_mask(full_ids, full_mask, raw_ids):
    """Require one exact contiguous content span; never guess token alignment."""
    import torch

    if (
        full_ids.ndim != 2
        or full_ids.shape[0] != 1
        or full_mask.shape != full_ids.shape
        or raw_ids.ndim != 2
        or raw_ids.shape[0] != 1
        or raw_ids.shape[1] < 1
        or not torch.all((full_mask == 0) | (full_mask == 1))
    ):
        raise ValueError("Invalid full or raw caption token IDs/mask")
    valid = full_mask.bool()
    tokens = full_ids[0].cpu()
    raw = raw_ids[0].cpu()
    starts = [
        start
        for start in range(tokens.numel() - raw.numel() + 1)
        if torch.equal(tokens[start : start + raw.numel()], raw)
        and bool(valid[0, start : start + raw.numel()].all())
    ]
    if len(starts) != 1:
        raise ValueError(
            "Caption token alignment is missing or ambiguous; refusing feature matching"
        )
    start = starts[0]
    content = torch.zeros_like(full_mask, dtype=torch.bool)
    content[:, start : start + raw.numel()] = True
    if not (valid & ~content).any():
        raise ValueError("Chat input must retain template positions outside the caption")
    return content, [start, start + raw.numel()]


def caption_region_masks(full_mask, content, span):
    """Partition a verified full-chat sequence without dropping any valid position."""
    import torch

    if (
        full_mask.ndim != 2
        or full_mask.shape[0] != 1
        or content.shape != full_mask.shape
        or content.dtype != torch.bool
        or len(span) != 2
        or any(type(value) is not int for value in span)
        or not 0 < span[0] < span[1] < full_mask.shape[1]
        or not torch.all((full_mask == 0) | (full_mask == 1))
    ):
        raise ValueError("Region alignment requires a valid interior content span and binary mask")
    valid = full_mask.bool()
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Region alignment requires right-padded valid prefixes")
    positions = torch.arange(full_mask.shape[1], device=full_mask.device).unsqueeze(0)
    expected_content = valid & (positions >= span[0]) & (positions < span[1])
    if not torch.equal(content.to(full_mask.device), expected_content):
        raise ValueError("Region content mask differs from its exact caption span")
    regions = {
        "prefix": valid & (positions < span[0]),
        "content": expected_content,
        "suffix": valid & (positions >= span[1]),
    }
    if any(not bool(mask.any()) for mask in regions.values()) or not torch.equal(
        sum(mask.int() for mask in regions.values()), valid.int()
    ):
        raise ValueError("Prefix, content and suffix must be nonempty disjoint valid partitions")
    return regions


def validate_pair(prism, native, raw_ids):
    """Prove feature positions refer to identical full-text tokens on both routes."""
    import torch

    if prism.get("formatted_prompt") != native.get("formatted_prompt"):
        raise ValueError("PRISM and native formatted inputs differ")
    for key in ("input_ids", "input_attention_mask", "attention_mask"):
        if not torch.equal(prism[key].cpu(), native[key].cpu()):
            raise ValueError(f"PRISM and native {key} differ")
    ids, mask = prism["input_ids"], prism["input_attention_mask"]
    if not torch.equal(prism["attention_mask"].cpu(), mask.cpu()):
        raise ValueError("Connector feature positions differ from input token positions")
    for features in (prism["hidden_states"], native["embeds"]):
        if (
            features.ndim != 3
            or features.shape[:2] != ids.shape
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Nonfinite or misaligned frozen features")
    content, span = caption_content_mask(ids, mask, raw_ids)
    return content, mask.bool() & ~content, span


def encode_observed_native(backend, prompt, *, max_text_length):
    """Verify the actual native teacher consumed the independently audited IDs."""
    import torch
    from tools.diagnose_prism_image_conditioning import encode_native_prompt

    consumed = []

    def capture(module, positional, keyword):
        # Pinned upstream passes IDs positionally and the attention mask by
        # keyword. Also accept explicit keyword IDs, but never ambiguous inputs.
        if len(positional) > 1 or (positional and "input_ids" in keyword):
            raise RuntimeError("Native teacher input ID arguments are ambiguous")
        ids = positional[0] if positional else keyword.get("input_ids")
        mask = keyword.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise RuntimeError("Native teacher must expose actual input IDs and mask")
        consumed.append(
            {"input_ids": ids.detach().cpu().clone(), "attention_mask": mask.detach().cpu().clone()}
        )

    hook = backend.mllm.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        native = encode_native_prompt(backend, prompt, max_text_length=max_text_length)
    finally:
        hook.remove()
    if len(consumed) != 1:
        raise RuntimeError("Expected exactly one native teacher forward per caption")
    if not torch.equal(consumed[0]["input_ids"], native["input_ids"].cpu()) or not torch.equal(
        consumed[0]["attention_mask"], native["input_attention_mask"].cpu()
    ):
        raise RuntimeError("Actual native teacher token IDs or mask differ from audited text")
    return native


def caption_norm(backend):
    """Select the existing DiT normalization module, with no replacement weights."""
    import torch

    try:
        norm = backend.transformer.time_caption_embed.caption_embedder[0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Pinned DiT caption RMSNorm path is unavailable") from error
    if not isinstance(norm, torch.nn.Module) or "rmsnorm" not in type(norm).__name__.lower():
        raise ValueError("Expected the pinned caption RMSNorm module")
    if any(parameter.requires_grad for parameter in norm.parameters()):
        raise ValueError("Caption RMSNorm must remain frozen")
    return norm


def build_feature_cache(
    model, tokenizer, backend, records, *, device, max_text_length, objective="content"
):
    """Encode only record captions; no dataset lookup, target or reference pixels."""
    import torch
    from tools.prism_image_conditioning import encode_prism_prompt
    from tools.train_image_decoder import _tensor_hash

    if objective not in ("content", "regions"):
        raise ValueError("Unknown native-feature alignment objective")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(
            "Cache construction requires the entire parent/connector/generator frozen and eval"
        )
    norm = caption_norm(backend)
    caches, audits = [], []
    with torch.no_grad():
        for record in records:
            if record.task != "t2i" or record.source_ids or record.source_paths:
                raise ValueError("Alignment cache requires caption-only source-free T2I metadata")
            prism = encode_prism_prompt(
                model,
                tokenizer,
                record.prompt,
                device=device,
                mode="chat",
                max_text_length=max_text_length,
            )
            native = encode_observed_native(backend, record.prompt, max_text_length=max_text_length)
            raw = tokenizer(
                [record.prompt],
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if not raw["attention_mask"].bool().all():
                raise ValueError("Single raw caption unexpectedly contains padding")
            content, template, span = validate_pair(prism, native, raw["input_ids"])
            teacher = norm(native["embeds"]).detach().float()
            if teacher.shape != native["embeds"].shape or not torch.isfinite(teacher).all():
                raise ValueError("Native caption normalization returned invalid features")
            caches.append(
                {
                    "id": record.id,
                    "hidden": prism["hidden_states"][0].detach().cpu().float().clone(),
                    "teacher": teacher[0].cpu().clone(),
                    "content": content[0].detach().cpu().clone(),
                    "template": template[0].detach().cpu().clone(),
                    "conditioning_dtype": native["embeds"].dtype,
                }
            )
            audits.append(
                {
                    "id": record.id,
                    "split": record.split,
                    "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
                    "formatted_prompt_sha256": hashlib.sha256(
                        prism["formatted_prompt"].encode()
                    ).hexdigest(),
                    "input_ids_sha256": _tensor_hash(prism["input_ids"]),
                    "input_mask_sha256": _tensor_hash(prism["input_attention_mask"]),
                    "prism_hidden_sha256": _tensor_hash(prism["hidden_states"]),
                    "native_features_sha256": _tensor_hash(native["embeds"]),
                    "teacher_normalized_sha256": _tensor_hash(teacher),
                    "content_span": span,
                    "content_tokens": int(content.sum()),
                    "template_tokens": int(template.sum()),
                    "total_tokens": prism["input_ids"].shape[1],
                    "conditioning_dtype": str(native["embeds"].dtype),
                    "exact_formatted_input_match": True,
                    "exact_input_ids_match": True,
                    "exact_input_masks_match": True,
                    "actual_native_forward_inputs_verified": True,
                    "target_pixels_read": False,
                }
            )
            if objective == "regions":
                regions = caption_region_masks(prism["input_attention_mask"], content, span)
                for region in ("prefix", "suffix"):
                    caches[-1][region] = regions[region][0].detach().cpu().clone()
                audits[-1]["region_spans"] = {
                    "prefix": [0, span[0]],
                    "content": list(span),
                    "suffix": [span[1], int(prism["input_attention_mask"].sum())],
                }
                audits[-1]["region_token_counts"] = {
                    region: int(mask.sum()) for region, mask in regions.items()
                }
    return caches, audits


def cached_batch(rows, device):
    from torch.nn.utils.rnn import pad_sequence

    if not rows or len({row["conditioning_dtype"] for row in rows}) != 1:
        raise ValueError("Cache batch must be nonempty with a single conditioning dtype")
    if any(row[key].requires_grad for row in rows for key in ("hidden", "teacher")):
        raise ValueError("Cached frozen features must be detached")
    region_keys = [{key for key in ("prefix", "suffix") if key in row} for row in rows]
    if any(keys != region_keys[0] for keys in region_keys) or region_keys[0] not in (
        set(),
        {"prefix", "suffix"},
    ):
        raise ValueError("Cached batches cannot mix content-only and region-alignment rows")
    keys = ("hidden", "teacher", "content", "template")
    if region_keys[0]:
        keys += ("prefix", "suffix")
    return {
        key: pad_sequence([row[key] for row in rows], batch_first=True).to(device) for key in keys
    } | {"conditioning_dtype": rows[0]["conditioning_dtype"]}


def alignment_losses(connector, norm, batch, *, objective="content"):
    """Equal-caption content MSE; template MSE and cosine are diagnostics only."""
    import torch
    from torch.nn import functional as F

    if objective not in ("content", "regions"):
        raise ValueError("Unknown native-feature alignment objective")
    if any(parameter.requires_grad for parameter in norm.parameters()):
        raise ValueError("Alignment cannot train caption normalization weights")
    predicted = norm(connector(batch["hidden"]).to(dtype=batch["conditioning_dtype"])).float()
    target = batch["teacher"]
    if (
        predicted.shape != target.shape
        or target.requires_grad
        or not torch.isfinite(target).all()
        or not torch.isfinite(predicted).all()
    ):
        raise ValueError("Invalid or nonfinite alignment features")
    token_mse = (predicted - target).square().mean(-1)
    token_cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
    result = {}
    if objective == "regions":
        for key in ("prefix", "content", "suffix", "template"):
            if (
                key not in batch
                or batch[key].dtype != torch.bool
                or batch[key].shape != token_mse.shape
            ):
                raise ValueError("Region alignment requires binary region masks with token shape")
        region_count = sum(batch[key].int() for key in ("prefix", "content", "suffix"))
        valid = batch["content"] | batch["template"]
        if (batch["content"] & batch["template"]).any() or not torch.equal(
            region_count, valid.int()
        ):
            raise ValueError(
                "Region masks must cover valid content/template positions exactly once"
            )
        if not torch.equal(batch["prefix"] | batch["suffix"], batch["template"]):
            raise ValueError("Prefix and suffix must exactly cover template positions")
    labels = ("content", "template") + (("prefix", "suffix") if objective == "regions" else ())
    for label in labels:
        mask = batch[label]
        if mask.shape != token_mse.shape or not mask.any(-1).all():
            raise ValueError(f"Every caption must contain valid {label} positions")
        counts = mask.sum(-1)
        result[label + "_mse"] = ((token_mse * mask).sum(-1) / counts).mean()
        result[label + "_cosine"] = ((token_cosine * mask).sum(-1) / counts).mean()
    if objective == "regions":
        result["region_balanced_mse"] = (
            sum(result[region + "_mse"] for region in ("prefix", "content", "suffix")) / 3
        )
    return result


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
    parser.add_argument("--objective", choices=("content", "regions"), default="content")
    parser.add_argument("--init-alignment-checkpoint", type=Path)
    parser.add_argument("--init-alignment-checkpoint-sha256")
    parser.add_argument("--expected-parent-tensors", type=int, default=526)
    parser.add_argument("--train-subset-size", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-text-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--attention-backend", choices=("default", "math"), default="math")
    return parser


def validate_budget(args):
    if args.objective == "regions" and args.init_alignment_checkpoint is None:
        raise ValueError(
            "Region alignment requires an audited content-only --init-alignment-checkpoint"
        )
    if args.objective == "content" and (
        args.init_alignment_checkpoint is not None
        or args.init_alignment_checkpoint_sha256 is not None
    ):
        raise ValueError("Default content-only alignment does not accept an alignment initializer")
    for key, lower, upper in (
        ("train_subset_size", 2, 64),
        ("validation_count", 1, 16),
        ("steps", 1, 5000),
        ("batch_size", 1, 32),
        ("eval_every", 1, 5000),
        ("checkpoint_every", 0, 5000),
        ("max_text_length", 1, 2048),
        ("expected_parent_tensors", 1, 10000),
    ):
        if not lower <= getattr(args, key) <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    if args.expected_connector_step != 500:
        raise ValueError("Alignment starts from the audited 500-step connector baseline")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-2:
        raise ValueError("Alignment learning rate must be finite in (0, 0.01]")
    if not args.deterministic or args.attention_backend != "math":
        raise ValueError("Alignment requires --deterministic --attention-backend math")


def main(argv=None):
    args = _parser().parse_args(argv)
    validate_budget(args)
    if not os.environ.get("PBS_JOBID") or args.device.split(":", 1)[0] != "xpu":
        raise RuntimeError("Real alignment requires an XPU in a PBS compute allocation")
    with execution_policy(args):
        return _run(args)


def _run(args):
    import torch
    from src.data.image_generation_webdataset import ImageGenerationWebDataset
    from src.decoders.loading import file_sha256, load_image_training_bundle
    from tools.train_image_decoder import (
        _rng_state,
        _seed_everything,
        _tensor_hash,
        frozen_state_hashes,
    )
    from tools.train_prism_image_diffusion import restore_warm_connector, selected_training_indices

    regions = args.objective == "regions"
    if regions:
        validate_budget(args)
        args.init_alignment_checkpoint = args.init_alignment_checkpoint.resolve(strict=True)
    objective = REGIONS_OBJECTIVE if regions else OBJECTIVE
    loss_key = "region_balanced_mse" if regions else "content_mse"
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
        raise ValueError("Alignment output directory must be new")
    precheck = validate_repeatability_report(args.repeatability_report, args)
    datasets = {
        split: ImageGenerationWebDataset(index, split=split)
        for split, index in (("train", args.train_index), ("validation", args.validation_index))
    }
    if datasets["train"].data_fingerprint != datasets["validation"].data_fingerprint:
        raise ValueError("Alignment train/validation data identity differs")
    if len(datasets["validation"]) < args.validation_count:
        raise ValueError("Validation pool is smaller than requested heldout count")
    selected = selected_training_indices(len(datasets["train"]), args.train_subset_size, args.seed)
    records = {
        "train": [datasets["train"].records[index] for index in selected],
        "validation": datasets["validation"].records[: args.validation_count],
    }
    if any(row.split != split for split, rows in records.items() for row in rows):
        raise ValueError("Alignment selected IDs do not match designated splits")
    if {row.id for row in records["train"]} & {row.id for row in records["validation"]}:
        raise ValueError("Alignment train and heldout IDs overlap")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    started = time.monotonic()
    report = {
        "schema_version": 2 if regions else 1,
        "evidence_kind": REGIONS_EVIDENCE_KIND if regions else EVIDENCE_KIND,
        "qualification": "unqualified",
        "status": "running",
        "completed_steps": 0,
        "training_scope": list(MODULES),
        "objective": objective,
        "future_validation_adapter_contract": (
            {
                **FUTURE_ADAPTER_CONTRACT,
                "implemented": True,
                "evidence_kind_required": REGIONS_EVIDENCE_KIND,
                "must_validate": FUTURE_ADAPTER_CONTRACT["must_validate"]
                + [
                    "strict completed v1 initializer and audited500 source, equal-region objective and metrics"
                ],
            }
            if regions
            else FUTURE_ADAPTER_CONTRACT
        ),
        "quality_benchmark": False,
        "flow_training_performed": False,
        "target_pixels_read": False,
        "cache_saved": False,
        "prompt_format": "chat",
        "negative_anchor_trained": False,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if regions
            or key
            not in ("objective", "init_alignment_checkpoint", "init_alignment_checkpoint_sha256")
        },
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
                "src/data/image_generation_webdataset.py",
                "tools/train_image_decoder.py",
                "tools/overfit_prism_image_connector.py",
                "tools/train_prism_image_connector.py",
                "tools/train_prism_image_diffusion.py",
                "tools/prism_image_conditioning.py",
                "tools/diagnose_prism_image_conditioning.py",
            )
        },
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "host": platform.node(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "repeatability_precheck": precheck,
        "data_fingerprint": datasets["train"].data_fingerprint,
        "data_validation": datasets["train"].validation_report,
        "index_sha256": {split: file_sha256(dataset.index) for split, dataset in datasets.items()},
        "selection": {
            "train": [{"index": i, "id": datasets["train"].records[i].id} for i in selected],
            "validation": [
                {"index": i, "id": row.id} for i, row in enumerate(records["validation"])
            ],
        },
        "full_train_count": len(datasets["train"]),
        "optimizer": {
            "name": "torch.optim.AdamW",
            "dtype": "torch.float32",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "evaluations": [],
        "checkpoints": [],
    }
    if regions:
        report["source_sha256"]["tools/prism_image_alignment_checkpoint.py"] = file_sha256(
            ROOT / "tools/prism_image_alignment_checkpoint.py"
        )
        report["optimized_loss_key"] = loss_key
        report["stage_initialization"] = {
            "optimizer": "fresh_adamw",
            "sampler": "fresh_seeded_order",
            "rng": "fresh_seed",
            "source_optimizer_restored": False,
            "source_sampler_restored": False,
            "source_rng_restored": False,
        }
    model = before = None
    try:
        _phase(report, report_path, "loading_frozen_parent_and_generator")
        _seed_everything(args.seed)
        bundle = load_image_training_bundle(
            args.model_config, args.checkpoint, args.tokenizer, args.source_processor
        )
        model, tokenizer = bundle["model"], bundle["tokenizer"]
        report["parent"] = bundle["provenance"]
        fixture = bool(bundle["provenance"].get("fixture"))
        if fixture:
            report["evidence_kind"] = REGIONS_FIXTURE_KIND if regions else FIXTURE_KIND
        if (
            regions
            and not fixture
            and (
                args.train_subset_size != 32
                or args.validation_count != 8
                or args.steps != 1000
                or args.batch_size != 4
                or args.learning_rate != 1e-4
            )
        ):
            raise ValueError(
                "Real region alignment is bounded to 32 train/8 heldout, 1000 steps, batch4, LR1e-4"
            )
        restoration = bundle["provenance"].get("restoration", {})
        if (
            restoration.get("strict_parent") is not True
            or restoration.get("loaded_key_count") != args.expected_parent_tensors
            or restoration.get("missing_parent_keys") != []
            or restoration.get("unexpected_keys") != []
        ):
            raise ValueError("Expected strict full PRISM parent restoration")
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
        connector = model.decoders["image"].connector.float()
        model.requires_grad_(False)
        model.eval()
        before = frozen_state_hashes(model, MODULES)
        report["frozen_hashes_before"] = before
        report["connector_warm_start"] = restore_warm_connector(
            model,
            args.connector_checkpoint,
            parent=report["parent"],
            reference_sha256=report["generator"]["manifest_sha256"],
            data_fingerprint=report["data_fingerprint"],
            frozen_hashes=before,
            expected_step=args.expected_connector_step,
            expected_sha256=args.connector_checkpoint_sha256,
            fixture=fixture,
        )
        if regions:
            from tools.prism_image_alignment_checkpoint import restore_alignment_connector

            _phase(report, report_path, "restoring_content_alignment_initializer")
            source_report = json.loads(
                (args.init_alignment_checkpoint.parent / "report.json").read_text()
            )
            if (
                source_report.get("schema_version") != 1
                or source_report.get("evidence_kind")
                != (FIXTURE_KIND if fixture else EVIDENCE_KIND)
                or source_report.get("objective") != OBJECTIVE
                or source_report.get("selection") != report["selection"]
            ):
                raise ValueError(
                    "Region initializer must be audited v1 content alignment with identical train/heldout IDs"
                )
            if not fixture and (
                source_report.get("completed_steps") != 1000
                or source_report.get("settings", {}).get("batch_size") != 4
                or source_report.get("sample_exposure", {}).get("examples_seen") != 4000
            ):
                raise ValueError(
                    "Real region initializer must be the completed 1000-step/4000-exposure v1 run"
                )
            report["warm_connector_hashes_before_initializer"] = {
                name: _tensor_hash(value) for name, value in connector.state_dict().items()
            }
            report["alignment_initialization"] = restore_alignment_connector(
                model,
                args.init_alignment_checkpoint,
                parent=report["parent"],
                generator_manifest=report["generator"],
                data_fingerprint=report["data_fingerprint"],
                index_sha256=report["index_sha256"],
                records={split: dataset.records for split, dataset in datasets.items()},
                tokenizer=tokenizer,
                expected_sha256=args.init_alignment_checkpoint_sha256,
                fixture=fixture,
            )
            if report["alignment_initialization"]["selection"] != report["selection"]:
                raise RuntimeError(
                    "Validated v1 initializer selection differs from the region stage"
                )
            _seed_everything(args.seed)
        norm = caption_norm(backend)
        report["caption_normalization"] = {
            "module": "decoders.image.backend.transformer.time_caption_embed.caption_embedder.0",
            "class": type(norm).__module__ + "." + type(norm).__name__,
            "state_sha256": {key: _tensor_hash(value) for key, value in norm.state_dict().items()},
            "eps": getattr(norm, "eps", None),
            "frozen": True,
        }
        report["connector_hashes_before"] = {
            name: _tensor_hash(value) for name, value in connector.state_dict().items()
        }
        _phase(report, report_path, "caching_frozen_caption_features")
        caches = {}
        report["cache_audit"] = {}
        for split, rows in records.items():
            caches[split], report["cache_audit"][split] = build_feature_cache(
                model,
                tokenizer,
                backend,
                rows,
                device=args.device,
                max_text_length=args.max_text_length,
                objective=args.objective,
            )
            _write(report_path, report)
        if regions:
            for split, audits in report["cache_audit"].items():
                source_audits = source_report.get("cache_audit", {}).get(split, [])
                if len(audits) != len(source_audits) or any(
                    any(audit.get(key) != value for key, value in original.items())
                    for audit, original in zip(audits, source_audits, strict=True)
                ):
                    raise ValueError(
                        "Region cache token or frozen feature audit differs from v1 initializer"
                    )
        report["cache_bytes"] = sum(
            row[key].numel() * row[key].element_size()
            for rows in caches.values()
            for row in rows
            for key in ("hidden", "teacher", "content", "template")
            + (("prefix", "suffix") if regions else ())
        )
        connector.requires_grad_(True)
        connector.train()
        selected_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if selected_names != {
            name for name, _ in model.named_parameters() if name.startswith(PREFIX)
        }:
            raise ValueError("Only complete connector parameters may receive gradients")
        selected_ids = {id(parameter) for parameter in connector.parameters()}
        if any(
            id(parameter) in selected_ids and not name.startswith(PREFIX)
            for name, parameter in model.named_parameters(remove_duplicate=False)
        ):
            raise ValueError("Connector parameter aliases a frozen model parameter")
        if any(parameter.dtype != torch.float32 for parameter in connector.parameters()):
            raise ValueError("Connector optimization requires FP32 weights")
        optimizer = torch.optim.AdamW(
            connector.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        order = ShuffledEpochOrder(len(caches["train"]), args.seed)
        exposure = {row["id"]: 0 for row in caches["train"]}
        protocol = {
            "parent": report["parent"],
            "reference_checkpoint_sha256": report["generator"]["manifest_sha256"],
            "data_fingerprint": report["data_fingerprint"],
            "index_sha256": report["index_sha256"],
            "source_sha256": report["source_sha256"],
            "runner_sha256": report["runner_sha256"],
            "settings": report["settings"],
            "objective": objective,
            "caption_normalization": report["caption_normalization"],
            "selection": report["selection"],
            "cache_audit": report["cache_audit"],
            "connector_warm_start": report["connector_warm_start"],
        }
        if regions:
            for key in (
                "alignment_initialization",
                "warm_connector_hashes_before_initializer",
                "stage_initialization",
                "optimized_loss_key",
            ):
                protocol[key] = report[key]
            if optimizer.state or order.examples_seen != 0:
                raise RuntimeError(
                    "Region alignment must start with fresh optimizer and sampler state"
                )

        def evaluate(step):
            result = {"step": step, "splits": {}}
            with torch.no_grad():
                for split, rows in caches.items():
                    entries = []
                    for row in rows:
                        losses = alignment_losses(
                            connector,
                            norm,
                            cached_batch([row], args.device),
                            objective=args.objective,
                        )
                        entries.append(
                            {
                                "id": row["id"],
                                **{key: float(value) for key, value in losses.items()},
                            }
                        )
                    result["splits"][split] = {
                        "count": len(rows),
                        "examples": entries,
                        **{
                            key: sum(row[key] for row in entries) / len(entries)
                            for key in entries[0]
                            if key != "id"
                        },
                    }
            report["evaluations"].append(result)
            with (args.output_dir / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
            _write(report_path, report)

        def checkpoint(step):
            checkpoint_prefix = (
                "connector-region-feature-alignment" if regions else "connector-feature-alignment"
            )
            path = args.output_dir / f"{checkpoint_prefix}-step-{step:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "schema_version": report["schema_version"],
                    "evidence_kind": report["evidence_kind"],
                    "qualification": "unqualified",
                    "step": step,
                    "connector_modules": list(MODULES),
                    "connector_state_dict": {
                        PREFIX + name: value.detach().cpu().clone()
                        for name, value in connector.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": _rng_state(),
                    "sampler_state": order.state_dict(),
                    "sample_exposure": dict(exposure),
                    "protocol": protocol,
                    "frozen_hashes": before,
                    "previous_report": str(report_path),
                },
                temporary,
            )
            temporary.replace(path)
            report["checkpoints"].append(
                {
                    "path": str(path),
                    "step": step,
                    "sha256": file_sha256(path),
                    "evidence_kind": report["evidence_kind"],
                }
            )
            _write(report_path, report)

        evaluate(0)
        _phase(report, report_path, "training_connector_from_cached_features")
        with (args.output_dir / "steps.jsonl").open("x") as stream:
            for step in range(1, args.steps + 1):
                tick = time.monotonic()
                indices = order.take(args.batch_size)
                rows = [caches["train"][index] for index in indices]
                optimizer.zero_grad(set_to_none=True)
                losses = alignment_losses(
                    connector, norm, cached_batch(rows, args.device), objective=args.objective
                )
                loss = losses[loss_key]
                if loss.ndim or not torch.isfinite(loss) or not loss.requires_grad:
                    raise RuntimeError("Expected finite connected native-feature alignment loss")
                loss.backward()
                gradients = [parameter.grad for parameter in connector.parameters()]
                if any(
                    gradient is None or not torch.isfinite(gradient).all() for gradient in gradients
                ):
                    raise RuntimeError("Connector received incomplete or nonfinite gradients")
                if any(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if not name.startswith(PREFIX)
                ):
                    raise RuntimeError("Alignment produced gradients outside connector")
                norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    connector.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                for row in rows:
                    exposure[row["id"]] += 1
                report["completed_steps"] = step
                report["sample_exposure"] = {
                    "examples_seen": order.examples_seen,
                    "unique_examples_seen": sum(count > 0 for count in exposure.values()),
                    "per_id_counts": dict(exposure),
                    "selected_equivalent_epochs": order.examples_seen / len(exposure),
                }
                entry = {
                    "step": step,
                    "ids": [row["id"] for row in rows],
                    **{key: float(value.detach()) for key, value in losses.items()},
                    "gradient_norm_before_clip": float(norm_before_clip),
                    "examples_seen": order.examples_seen,
                    "duration_seconds": time.monotonic() - tick,
                }
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                if step == args.steps or step % args.eval_every == 0:
                    evaluate(step)
                if step == args.steps or (
                    args.checkpoint_every and step % args.checkpoint_every == 0
                ):
                    checkpoint(step)
                if step % 100 == 0:
                    _write(report_path, report)
        optimizer.zero_grad(set_to_none=True)
        report["connector_hashes_after"] = {
            name: _tensor_hash(value) for name, value in connector.state_dict().items()
        }
        report["connector_state_changed"] = (
            report["connector_hashes_before"] != report["connector_hashes_after"]
        )
        if not report["connector_state_changed"]:
            raise RuntimeError("Alignment did not change connector weights")
        report.update(status="completed", phase="completed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if model is not None and before is not None:
            after = frozen_state_hashes(model, MODULES)
            report["frozen_hashes_after"] = after
            report["frozen_state_unchanged"] = before == after
            if before != after:
                report.update(
                    status="failed", error="Frozen parent/teacher/generator state changed"
                )
        report["duration_seconds"] = time.monotonic() - started
        _write(report_path, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "evidence_kind", "completed_steps", "duration_seconds")
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
