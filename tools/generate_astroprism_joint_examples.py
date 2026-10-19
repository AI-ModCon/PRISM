#!/usr/bin/env python3
"""Generate held-out image and spectrum captions from a joint AstroPRISM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ModelConfig
from src.model import UnifiedTransformer
from src.train import _build_astroprism_joint_data
from src.training.distributed import load_model_weights_only


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run greedy, prompt-only generation for one held-out image and one "
            "held-out spectrum using a joint AstroPRISM checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Saved Hydra config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory containing model.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--image-index", type=int, default=43)
    parser.add_argument("--spectrum-index", type=int, default=48)
    parser.add_argument("--image-max-new-tokens", type=int, default=320)
    parser.add_argument("--spectrum-max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default=None,
        help="Optional inference-only override; defaults to the saved run setting.",
    )
    return parser.parse_args()


def _model_config_from_saved_run(cfg: Any, *, attn_implementation: str | None) -> ModelConfig:
    raw = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("saved model configuration must resolve to a mapping")
    raw = dict(raw)
    raw["llm_backbone_id"] = raw.pop("backbone_id")
    raw["llm_tokenizer_id"] = raw.pop("tokenizer_id")
    raw["max_merged_seq_length"] = int(cfg.data.max_merged_seq_length)
    raw["text_dropout"] = 0.0
    if attn_implementation is not None:
        raw["attn_implementation"] = attn_implementation
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in raw.items() if key in allowed})


def _prompt_only(sample: dict[str, Any]) -> tuple[torch.Tensor, str]:
    text = sample.get("text")
    labels = sample.get("labels")
    if not isinstance(text, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("sample text and labels must be tensors")
    if text.ndim != 1 or labels.shape != text.shape:
        raise ValueError("sample text and labels must be aligned one-dimensional tensors")

    supervised = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise ValueError("sample has no supervised assistant tokens")
    answer_start = int(supervised[0].item())
    if answer_start <= 0:
        raise ValueError("sample has no masked prompt prefix")
    if not bool(labels[:answer_start].eq(-100).all()):
        raise ValueError("prompt mask is not a contiguous leading prefix")
    if not bool(labels[answer_start:].ne(-100).all()):
        raise ValueError("assistant labels are not a contiguous suffix")
    return text[:answer_start].clone(), "leading_ignore_index_prefix"


def _checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights do not exist: {weights}")
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
    projector_keys = tuple(key for key in keys if key.startswith("projectors."))
    aion_keys = tuple(key for key in keys if key.startswith("encoders.aion."))
    if len(projector_keys) != 12:
        raise RuntimeError(
            f"joint checkpoint must contain 12 projector tensors, found {len(projector_keys)}"
        )
    if not aion_keys:
        raise RuntimeError("joint checkpoint contains no shared encoders.aion tensors")
    return {
        "weights": str(weights.resolve()),
        "tensor_count": len(keys),
        "projector_tensor_count": len(projector_keys),
        "shared_aion_tensor_count": len(aion_keys),
    }


def _move_inputs(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_inputs(item, device) for key, item in value.items()}
    raise TypeError(f"unsupported inference input type: {type(value).__name__}")


def _generation_inputs(
    sample: dict[str, Any],
    *,
    modality: str,
    prompt_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {"text": prompt_ids.unsqueeze(0)}
    if modality == "image":
        image = sample.get("image")
        if not isinstance(image, torch.Tensor) or tuple(image.shape) != (4, 160, 160):
            raise ValueError("image sample must contain float[4,160,160]")
        if image.dtype != torch.float32:
            raise TypeError(f"raw image must remain float32, got {image.dtype}")
        inputs["image"] = image.unsqueeze(0)
    elif modality == "spectrum":
        spectrum = sample.get("spectrum")
        if not isinstance(spectrum, dict):
            raise TypeError("spectrum sample must contain a field mapping")
        expected_dtypes = {
            "flux": torch.float32,
            "ivar": torch.float32,
            "mask": torch.bool,
            "wavelength": torch.float32,
        }
        if set(spectrum) != set(expected_dtypes):
            raise ValueError(f"unexpected spectrum fields: {sorted(spectrum)}")
        for field, dtype in expected_dtypes.items():
            tensor = spectrum[field]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (7781,):
                raise ValueError(f"spectrum.{field} must have shape [7781]")
            if tensor.dtype != dtype:
                raise TypeError(f"spectrum.{field} must remain {dtype}, got {tensor.dtype}")
        inputs["spectrum"] = {
            field: tensor.unsqueeze(0) for field, tensor in spectrum.items()
        }
    else:
        raise ValueError(f"unsupported modality {modality!r}")
    return _move_inputs(inputs, device)


def _generate_one(
    *,
    model: UnifiedTransformer,
    tokenizer: Any,
    sample: dict[str, Any],
    modality: str,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids, prompt_mask_contract = _prompt_only(sample)
    reference_ids = sample["text"][prompt_ids.numel() :]
    inputs = _generation_inputs(
        sample,
        modality=modality,
        prompt_ids=prompt_ids,
        device=device,
    )

    use_autocast = device.type in {"xpu", "cuda"}
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        generated = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started

    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise RuntimeError(
            f"generation must return a rank-2 token tensor, got {type(generated).__name__}"
        )
    generated_ids = generated[0].detach().cpu()
    eos_id = int(tokenizer.eos_token_id)
    hit_eos = bool(generated_ids.eq(eos_id).any())
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        raise RuntimeError(f"{modality} generation decoded to empty text")

    provenance = dict(sample.get("provenance") or {})
    return {
        "modality": modality,
        "candidate_index": int(provenance["candidate_index"]),
        "paired_targetid": str(provenance["paired_targetid"]),
        "caption_object_id": str(provenance["caption_object_id"]),
        "source_group_id": provenance.get("source_group_id"),
        "split": provenance.get("split"),
        "prompt_mask_contract": prompt_mask_contract,
        "reference_supplied_to_model": False,
        "prompt_token_count": int(prompt_ids.numel()),
        "prompt_rendered": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "reference_text": tokenizer.decode(reference_ids, skip_special_tokens=True).strip(),
        "generated_text": generated_text,
        "generated_token_count": int(generated_ids.numel()),
        "generated_token_ids": generated_ids.tolist(),
        "max_new_tokens": int(max_new_tokens),
        "finish_reason": "eos" if hit_eos else "max_new_tokens",
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = _parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"saved Hydra config does not exist: {args.config}")
    if args.image_index < 0 or args.spectrum_index < 0:
        raise ValueError("held-out dataset indices must be non-negative")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    model_config = _model_config_from_saved_run(
        cfg,
        attn_implementation=args.attn_implementation,
    )
    checkpoint_contract = _checkpoint_contract(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_tokenizer_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer must define pad_token_id and eos_token_id")

    model = UnifiedTransformer(model_config)
    checkpoint_step = load_model_weights_only(model, str(args.checkpoint), device="cpu")
    model.tokenizer = tokenizer

    device = torch.device(args.device)
    model.to(device=device)
    model.eval()
    model.refresh_encoder_frozen_cache()
    if set(model.encoders) != {"aion"}:
        raise RuntimeError(f"expected one shared AION owner, found {list(model.encoders)}")
    if model._encoder_key_by_modality != {"image": "aion", "spectrum": "aion"}:
        raise RuntimeError(f"unexpected joint route map: {model._encoder_key_by_modality}")

    dataset, _ = _build_astroprism_joint_data(cfg.data, tokenizer, split=args.split)
    if args.image_index >= len(dataset.image_dataset):
        raise IndexError("image index is outside the held-out dataset")
    if args.spectrum_index >= len(dataset.spectrum_dataset):
        raise IndexError("spectrum index is outside the held-out dataset")

    image_result = _generate_one(
        model=model,
        tokenizer=tokenizer,
        sample=dataset.image_dataset[args.image_index],
        modality="image",
        max_new_tokens=args.image_max_new_tokens,
        device=device,
    )
    spectrum_result = _generate_one(
        model=model,
        tokenizer=tokenizer,
        sample=dataset.spectrum_dataset[args.spectrum_index],
        modality="spectrum",
        max_new_tokens=args.spectrum_max_new_tokens,
        device=device,
    )

    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "checkpoint_contract": checkpoint_contract,
        "saved_config": str(args.config.resolve()),
        "device": str(device),
        "dtype": str(next(model.backbone.parameters()).dtype),
        "attn_implementation": model.backbone.config._attn_implementation,
        "decoding": {"strategy": "greedy", "do_sample": False, "seed": args.seed},
        "dataset": {
            "split": args.split,
            "image_count": len(dataset.image_dataset),
            "spectrum_count": len(dataset.spectrum_dataset),
        },
        "examples": [image_result, spectrum_result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
