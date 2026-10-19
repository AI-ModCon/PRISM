#!/usr/bin/env python3
"""Compose the trained image and spectrum adapters into one small checkpoint.

The Stage 1A and Stage 1B checkpoints contain complete frozen Qwen/AION state,
but a joint run only needs the independently trained projector tensors.  This
tool reads those tensors lazily and writes a compact checkpoint accepted by
``training.resume_weights_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPECTED_SHAPES = {
    "projectors.image.fc1.bias": (2560,),
    "projectors.image.fc1.weight": (2560, 768),
    "projectors.image.fc2.bias": (2560,),
    "projectors.image.fc2.weight": (2560, 2560),
    "projectors.image.modality_embedding": (1, 1, 2560),
    "projectors.spectrum.fc1.bias": (2560,),
    "projectors.spectrum.fc1.weight": (2560, 768),
    "projectors.spectrum.fc2.bias": (2560,),
    "projectors.spectrum.fc2.weight": (2560, 2560),
    "projectors.spectrum.final_norm.bias": (2560,),
    "projectors.spectrum.final_norm.weight": (2560,),
    "projectors.spectrum.modality_embedding": (1, 1, 2560),
}


def _weights_path(checkpoint: Path) -> Path:
    path = checkpoint.expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {path}")
    return path


def _tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_adapter(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key.startswith(prefix):
                if clean_key in selected:
                    raise RuntimeError(f"Duplicate adapter tensor after prefix cleanup: {clean_key}")
                selected[clean_key] = checkpoint.get_tensor(key)
    return selected


def compose_joint_adapters(
    image_checkpoint: str | Path,
    spectrum_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict:
    """Write and return provenance for a compact joint-adapter checkpoint."""

    image_path = _weights_path(Path(image_checkpoint))
    spectrum_path = _weights_path(Path(spectrum_checkpoint))
    output = Path(output_dir).expanduser().resolve()
    weights_out = output / "model.safetensors"
    manifest_out = output / "manifest.json"
    state_out = output / "training_state.json"

    existing = [path for path in (weights_out, manifest_out, state_out) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite joint adapter artifact; pass --force for: "
            + ", ".join(str(path) for path in existing)
        )

    image_tensors = _read_adapter(image_path, "projectors.image.")
    spectrum_tensors = _read_adapter(spectrum_path, "projectors.spectrum.")
    tensors = {**image_tensors, **spectrum_tensors}

    expected_keys = set(EXPECTED_SHAPES)
    actual_keys = set(tensors)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"Adapter tensor contract mismatch; missing={missing}, unexpected={unexpected}"
        )
    for key, expected_shape in EXPECTED_SHAPES.items():
        actual_shape = tuple(tensors[key].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Adapter tensor {key} has shape {actual_shape}; expected {expected_shape}"
            )

    output.mkdir(parents=True, exist_ok=True)
    image_digest = _tensor_digest(image_tensors.items())
    spectrum_digest = _tensor_digest(spectrum_tensors.items())
    save_file(
        tensors,
        weights_out,
        metadata={
            "format": "pt",
            "artifact": "astroprism_joint_adapter_init",
            "image_adapter_sha256": image_digest,
            "spectrum_adapter_sha256": spectrum_digest,
        },
    )
    merged_digest = _tensor_digest(tensors.items())
    manifest = {
        "schema_version": 1,
        "artifact": "astroprism_joint_adapter_init",
        "image_checkpoint": str(image_path),
        "spectrum_checkpoint": str(spectrum_path),
        "image_adapter_sha256": image_digest,
        "spectrum_adapter_sha256": spectrum_digest,
        "merged_adapter_sha256": merged_digest,
        "tensor_count": len(tensors),
        "tensor_keys": sorted(tensors),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    state_out.write_text(
        json.dumps(
            {
                "step": 0,
                "stage": "joint_adapter_initialization",
                "provenance_manifest": str(manifest_out),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-checkpoint", required=True, type=Path)
    parser.add_argument("--spectrum-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = compose_joint_adapters(
        args.image_checkpoint,
        args.spectrum_checkpoint,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
