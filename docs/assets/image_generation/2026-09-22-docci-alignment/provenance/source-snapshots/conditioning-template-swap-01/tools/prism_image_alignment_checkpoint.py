"""Strict weights-only admission for the distinct caption-feature alignment kind.

This adapter never presents alignment as flow training, never restores Adam,
RNG or sampler state, and does not establish image quality. Validation completes
before any connector tensor changes. The caller owns model mode/device policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tools.align_prism_image_conditioning import (
    EVIDENCE_KIND,
    FIXTURE_KIND,
    MODULES,
    OBJECTIVE,
    PREFIX,
    caption_content_mask,
    caption_norm,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCES = (
    "src/model.py",
    "src/decoders/image.py",
    "src/decoders/conditioning.py",
    "src/decoders/types.py",
    "src/decoders/omnigen2_backend.py",
    "src/decoders/loading.py",
    "tools/prism_image_conditioning.py",
)


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require(condition, message):
    if not condition:
        raise ValueError("Alignment checkpoint: " + message)


def _validate_data_and_token_audit(protocol, report, records, tokenizer, backend):
    """Re-tokenize metadata without opening images or invoking either encoder."""
    import torch
    from tools.prism_image_conditioning import format_prompt
    from tools.train_image_decoder import _tensor_hash
    from tools.train_prism_image_diffusion import selected_training_indices

    settings = protocol.get("settings", {})
    _require(
        isinstance(settings, dict) and report.get("settings") == settings,
        "settings differ between report and checkpoint",
    )
    for key, lower, upper in (
        ("train_subset_size", 2, 64),
        ("validation_count", 1, 16),
        ("max_text_length", 1, 2048),
        ("batch_size", 1, 32),
        ("steps", 1, 5000),
    ):
        _require(
            type(settings.get(key)) is int and lower <= settings[key] <= upper, f"invalid {key}"
        )
    _require(type(settings.get("seed")) is int, "missing deterministic seed")
    _require(
        settings.get("deterministic") is True and settings.get("attention_backend") == "math",
        "training numerical policy is not deterministic math",
    )
    _require(
        set(records) == {"train", "validation"}, "both complete dataset record lists are required"
    )
    all_ids = {}
    for split, rows in records.items():
        identifiers = [row.id for row in rows]
        _require(len(set(identifiers)) == len(identifiers), f"duplicate {split} record IDs")
        _require(
            all(
                row.split == split
                and row.task == "t2i"
                and not row.source_ids
                and not row.source_paths
                for row in rows
            ),
            "records are not caption-only examples in designated splits",
        )
        all_ids[split] = set(identifiers)
    _require(
        not all_ids["train"] & all_ids["validation"], "training and heldout record IDs overlap"
    )
    _require(
        report.get("full_train_count") == len(records["train"]),
        "full training pool identity differs",
    )
    _require(
        len(records["validation"]) >= settings["validation_count"], "heldout pool is incomplete"
    )
    indices = {
        "train": selected_training_indices(
            len(records["train"]), settings["train_subset_size"], settings["seed"]
        ),
        "validation": list(range(settings["validation_count"])),
    }
    selection = {
        split: [{"index": index, "id": records[split][index].id} for index in values]
        for split, values in indices.items()
    }
    _require(
        protocol.get("selection") == selection and report.get("selection") == selection,
        "training or heldout selection differs from current records",
    )
    audits = protocol.get("cache_audit")
    _require(
        isinstance(audits, dict)
        and set(audits) == set(selection)
        and report.get("cache_audit") == audits,
        "cache audits differ or are incomplete",
    )
    for split, values in indices.items():
        _require(
            isinstance(audits[split], list) and len(audits[split]) == len(values),
            "cache audit cohort size differs",
        )
        for index, audit in zip(values, audits[split], strict=True):
            row = records[split][index]
            _require(isinstance(audit, dict), "invalid cache audit entry")
            _require(
                audit.get("id") == row.id and audit.get("split") == split,
                "cache audit record identity differs",
            )
            formatted = format_prompt(row.prompt, tokenizer, "chat")
            native_formatted = backend._pipeline._apply_chat_template(row.prompt)
            _require(formatted == native_formatted, "current PRISM/native chat formatting differs")
            prism = tokenizer([formatted], padding=True, truncation=False, return_tensors="pt")
            native = backend._pipeline.processor.tokenizer(
                [native_formatted], padding=True, truncation=False, return_tensors="pt"
            )
            raw = tokenizer(
                [row.prompt],
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            _require(
                torch.equal(prism["input_ids"].cpu(), native["input_ids"].cpu()),
                "current PRISM/native token IDs differ",
            )
            _require(
                torch.equal(prism["attention_mask"].cpu(), native["attention_mask"].cpu()),
                "current PRISM/native token masks differ",
            )
            _require(
                prism["input_ids"].shape[1] <= settings["max_text_length"]
                and bool(raw["attention_mask"].bool().all()),
                "current caption would truncate or has raw padding",
            )
            content, span = caption_content_mask(
                prism["input_ids"], prism["attention_mask"], raw["input_ids"]
            )
            expected = {
                "prompt_sha256": hashlib.sha256(row.prompt.encode()).hexdigest(),
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "input_ids_sha256": _tensor_hash(prism["input_ids"]),
                "input_mask_sha256": _tensor_hash(prism["attention_mask"]),
                "content_span": span,
                "content_tokens": int(content.sum()),
                "template_tokens": int((prism["attention_mask"].bool() & ~content).sum()),
                "total_tokens": prism["input_ids"].shape[1],
                "exact_formatted_input_match": True,
                "exact_input_ids_match": True,
                "exact_input_masks_match": True,
                "actual_native_forward_inputs_verified": True,
                "target_pixels_read": False,
            }
            _require(
                all(audit.get(key) == value for key, value in expected.items()),
                "caption token/content audit does not match current data",
            )
            _require(
                audit.get("conditioning_dtype") == "torch." + settings.get("dtype", ""),
                "native conditioning precision differs",
            )
            _require(
                all(
                    _digest(audit.get(key))
                    for key in (
                        "prism_hidden_sha256",
                        "native_features_sha256",
                        "teacher_normalized_sha256",
                    )
                ),
                "missing frozen feature hashes",
            )
    return selection


def restore_alignment_connector(
    model,
    path,
    *,
    parent,
    generator_manifest,
    data_fingerprint,
    index_sha256,
    records,
    tokenizer,
    expected_sha256=None,
    fixture=False,
):
    """Validate a completed alignment artifact, then copy only FP32 connector weights.

    ``records`` contains the current full train/validation metadata lists, not
    dataset instances: this adapter never reads target pixels. The loaded model
    must retain the original DiT/teacher/PRISM weights, with frozen nonconnector
    parameters, no pending gradients, and an FP32 connector. Modes and gradient
    flags are preserved. Only inference/training callers may decide what to run
    next; returned provenance retains the distinct alignment evidence kind.
    """
    import torch
    from src.decoders.loading import file_sha256
    from tools.train_image_decoder import _tensor_hash, frozen_state_hashes
    from tools.train_prism_image_connector import ShuffledEpochOrder

    _require(
        bool(parent.get("fixture", False)) == fixture, "fixture flag differs from loaded parent"
    )
    _require(
        all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith(PREFIX)
        ),
        "nonconnector parameters must be frozen",
    )
    _require(
        all(parameter.grad is None for parameter in model.parameters()),
        "pending gradients must be cleared before weights-only restoration",
    )
    connector = model.decoders["image"].connector
    _require(
        all(parameter.dtype == torch.float32 for parameter in connector.parameters()),
        "current connector must be FP32",
    )
    connector_ids = {id(parameter) for parameter in connector.parameters()}
    _require(
        not any(
            id(parameter) in connector_ids and not name.startswith(PREFIX)
            for name, parameter in model.named_parameters(remove_duplicate=False)
        ),
        "connector aliases a frozen parameter",
    )
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    _require(
        expected_sha256 is None or expected_sha256 == digest, "requested checkpoint digest differs"
    )
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    report_path = path.parent / "report.json"
    report = json.loads(report_path.read_text())
    expected_kind = FIXTURE_KIND if fixture else EVIDENCE_KIND
    _require(
        isinstance(saved, dict) and isinstance(report, dict),
        "checkpoint and report must be mappings",
    )
    _require(
        saved.get("schema_version") == report.get("schema_version") == 1, "schema version differs"
    )
    _require(
        saved.get("evidence_kind") == report.get("evidence_kind") == expected_kind,
        "distinct alignment evidence kind required; flow-pilot masquerading is forbidden",
    )
    _require(
        saved.get("qualification") == report.get("qualification") == "unqualified",
        "invalid qualification",
    )
    step = saved.get("step")
    _require(type(step) is int and 1 <= step <= 5000, "invalid terminal step")
    _require(
        report.get("status") == "completed"
        and report.get("phase") == "completed"
        and report.get("completed_steps") == step,
        "completed terminal report required",
    )
    matches = [
        entry
        for entry in report.get("checkpoints", [])
        if entry.get("step") == step
        and entry.get("sha256") == digest
        and entry.get("evidence_kind") == expected_kind
    ]
    _require(len(matches) == 1, "checkpoint digest is not uniquely bound to completed report")
    _require(
        saved.get("connector_modules") == report.get("training_scope") == list(MODULES),
        "training scope is not connector only",
    )
    protocol = saved.get("protocol", {})
    _require(isinstance(protocol, dict), "missing alignment protocol")
    _require(
        protocol.get("parent") == report.get("parent") == parent, "PRISM parent identity differs"
    )
    _require(
        report.get("generator") == generator_manifest
        and protocol.get("reference_checkpoint_sha256")
        == generator_manifest.get("manifest_sha256"),
        "original pretrained generator identity differs",
    )
    _require(
        _digest(generator_manifest.get("manifest_sha256")), "invalid original generator digest"
    )
    _require(
        protocol.get("data_fingerprint") == report.get("data_fingerprint") == data_fingerprint,
        "data conversion identity differs",
    )
    _require(
        set(index_sha256) == {"train", "validation"}
        and all(_digest(value) for value in index_sha256.values()),
        "both current index digests required",
    )
    _require(
        protocol.get("index_sha256") == report.get("index_sha256") == index_sha256,
        "train or heldout index identity differs",
    )
    _require(
        protocol.get("objective") == report.get("objective") == OBJECTIVE,
        "alignment objective differs",
    )
    _require(report.get("prompt_format") == "chat", "alignment requires chat conditioning")
    _require(
        all(
            report.get(key) is False
            for key in (
                "negative_anchor_trained",
                "target_pixels_read",
                "flow_training_performed",
                "quality_benchmark",
                "cache_saved",
            )
        ),
        "alignment evidence claims an unsupported objective or artifact",
    )
    source = protocol.get("source_sha256", {})
    _require(
        isinstance(source, dict)
        and source == report.get("source_sha256")
        and all(_digest(value) for value in source.values()),
        "source provenance is incomplete or differs",
    )
    _require(
        _digest(protocol.get("runner_sha256"))
        and protocol.get("runner_sha256") == report.get("runner_sha256"),
        "alignment runner identity differs",
    )
    _require(
        all(source.get(name) == file_sha256(ROOT / name) for name in CORE_SOURCES),
        "current core conditioning source differs from alignment source",
    )
    backend = model.decoders["image"].backend
    norm = caption_norm(backend)
    norm_identity = {
        "module": "decoders.image.backend.transformer.time_caption_embed.caption_embedder.0",
        "class": type(norm).__module__ + "." + type(norm).__name__,
        "state_sha256": {name: _tensor_hash(value) for name, value in norm.state_dict().items()},
        "eps": getattr(norm, "eps", None),
        "frozen": True,
    }
    _require(
        protocol.get("caption_normalization")
        == report.get("caption_normalization")
        == norm_identity,
        "actual caption RMSNorm identity differs",
    )
    frozen = frozen_state_hashes(model, MODULES)
    _require(
        report.get("frozen_state_unchanged") is True
        and saved.get("frozen_hashes")
        == report.get("frozen_hashes_before")
        == report.get("frozen_hashes_after")
        == frozen,
        "frozen PRISM/teacher/original generator state differs",
    )
    warm = protocol.get("connector_warm_start", {})
    _require(
        isinstance(warm, dict)
        and warm == report.get("connector_warm_start")
        and warm.get("step") == 500
        and warm.get("fresh_optimizer") is True
        and _digest(warm.get("sha256"))
        and _digest(warm.get("report_sha256")),
        "audited 500-step warm-start lineage is incomplete",
    )
    _require(
        {name: _tensor_hash(value) for name, value in connector.state_dict().items()}
        == report.get("connector_hashes_before"),
        "current connector does not match the audited 500-step starting weights",
    )
    selection = _validate_data_and_token_audit(protocol, report, records, tokenizer, backend)
    settings = protocol["settings"]
    _require(step == settings["steps"], "checkpoint is not the terminal requested step")
    order = ShuffledEpochOrder(len(selection["train"]), settings["seed"])
    order.load_state_dict(saved.get("sampler_state", {}))
    _require(
        order.examples_seen == step * settings["batch_size"],
        "sampler progress differs from completed steps",
    )
    exposure = {entry["id"]: order.epoch for entry in selection["train"]}
    for position in order.order[: order.cursor]:
        exposure[selection["train"][position]["id"]] += 1
    _require(
        saved.get("sample_exposure") == exposure, "per-ID exposure includes wrong IDs or counts"
    )
    _require(
        report.get("sample_exposure")
        == {
            "examples_seen": order.examples_seen,
            "unique_examples_seen": sum(count > 0 for count in exposure.values()),
            "per_id_counts": exposure,
            "selected_equivalent_epochs": order.examples_seen / len(exposure),
        },
        "reported training exposure differs",
    )
    evaluations = report.get("evaluations", [])
    _require(
        bool(evaluations) and evaluations[-1].get("step") == step,
        "terminal alignment evaluation is missing",
    )
    for split, entries in selection.items():
        terminal = evaluations[-1].get("splits", {}).get(split, {})
        _require(
            terminal.get("count") == len(entries)
            and [entry.get("id") for entry in terminal.get("examples", [])]
            == [entry["id"] for entry in entries],
            "terminal evaluation IDs differ from training/heldout cohorts",
        )
        _require(
            all(
                isinstance(terminal.get(key), (int, float)) and math.isfinite(terminal[key])
                for key in ("content_mse", "content_cosine", "template_mse", "template_cosine")
            ),
            "terminal alignment metrics are incomplete or nonfinite",
        )
    expected = {PREFIX + name: value for name, value in connector.state_dict().items()}
    state = saved.get("connector_state_dict", {})
    _require(
        isinstance(state, dict) and bool(state) and set(state) == set(expected),
        "full connector tensor state required",
    )
    for name, value in state.items():
        _require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.layout == torch.strided
            and value.shape == expected[name].shape
            and bool(torch.isfinite(value).all()),
            "invalid finite FP32 connector tensor: " + name,
        )
    hashes = {name[len(PREFIX) :]: _tensor_hash(value) for name, value in state.items()}
    _require(
        hashes == report.get("connector_hashes_after")
        and report.get("connector_state_changed") is True
        and hashes != report.get("connector_hashes_before"),
        "connector weight hashes disagree with report",
    )
    # All checks above are read-only. No optimizer, RNG, cache or sampler state
    # is applied, and the connector load preserves module modes/grad flags.
    connector.load_state_dict(
        {name[len(PREFIX) :]: value for name, value in state.items()}, strict=True
    )
    return {
        "checkpoint": str(path),
        "sha256": digest,
        "report_sha256": file_sha256(report_path),
        "step": step,
        "evidence_kind": expected_kind,
        "qualification": "unqualified",
        "restore_policy": "connector_weights_only_from_native_feature_alignment",
        "restored_connector_tensors": len(state),
        "objective": OBJECTIVE,
        "parent": parent,
        "reference_checkpoint_sha256": generator_manifest["manifest_sha256"],
        "data_fingerprint": data_fingerprint,
        "alignment_warm_start": warm,
        "index_sha256": dict(index_sha256),
        "selection": selection,
        "caption_normalization": norm_identity,
        "current_core_sources_verified": list(CORE_SOURCES),
        "exact_token_and_content_audit_revalidated": True,
        "frozen_state_unchanged": True,
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "sampler_state_restored": False,
        "prompt_format": "chat",
        "negative_anchor_trained": False,
        "quality_benchmark": False,
        "required_followup": "paired flow controls and target-free no-CFG generation; alignment loss is not image quality",
    }
