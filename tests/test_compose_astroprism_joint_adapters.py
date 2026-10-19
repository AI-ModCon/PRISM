from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tools.compose_astroprism_joint_adapters import (
    EXPECTED_SHAPES,
    compose_joint_adapters,
)


def _checkpoint(path: Path, prefix: str, *, fill: float) -> Path:
    path.mkdir()
    tensors = {
        key: torch.full(shape, fill, dtype=torch.float16)
        for key, shape in EXPECTED_SHAPES.items()
        if key.startswith(prefix)
    }
    tensors["backbone.large_frozen_weight"] = torch.zeros(2)
    save_file(tensors, path / "model.safetensors")
    return path


def test_compose_joint_adapters_selects_only_both_projectors(tmp_path):
    image = _checkpoint(tmp_path / "image", "projectors.image.", fill=1.0)
    spectrum = _checkpoint(tmp_path / "spectrum", "projectors.spectrum.", fill=2.0)
    output = tmp_path / "joint"

    manifest = compose_joint_adapters(image, spectrum, output)
    result = load_file(output / "model.safetensors")

    assert set(result) == set(EXPECTED_SHAPES)
    assert torch.all(result["projectors.image.fc1.bias"] == 1)
    assert torch.all(result["projectors.spectrum.fc1.bias"] == 2)
    assert manifest["tensor_count"] == len(EXPECTED_SHAPES)
    assert len(manifest["merged_adapter_sha256"]) == 64
    assert (output / "training_state.json").is_file()


def test_compose_joint_adapters_refuses_overwrite(tmp_path):
    image = _checkpoint(tmp_path / "image", "projectors.image.", fill=1.0)
    spectrum = _checkpoint(tmp_path / "spectrum", "projectors.spectrum.", fill=2.0)
    output = tmp_path / "joint"
    compose_joint_adapters(image, spectrum, output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compose_joint_adapters(image, spectrum, output)
