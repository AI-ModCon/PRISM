"""Tests for the explicit-modality lookup added in fix(#23).

The shipped datasets_config.json must carry a `modality` field on every
entry, and `_get_modality` must raise a clear error when missing.
"""
import json
from pathlib import Path

import pytest
from src.data.multimodal import _get_modality

CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "datasets_config.json"
VALID_MODALITIES = {"text", "image", "table", "time_series", "geometry", "graph", "dna"}


def _all_dataset_entries():
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    for zone, zone_cfg in data.items():
        if isinstance(zone_cfg, dict) and "datasets" in zone_cfg:
            for name, info in zone_cfg["datasets"].items():
                yield zone, name, info


def test_get_modality_returns_explicit_field():
    assert (
        _get_modality({"handler": "image_pixmo", "modality": "image"}, "t") == "image"
    )


def test_get_modality_raises_on_missing():
    with pytest.raises(KeyError, match="modality"):
        _get_modality({"handler": "image_pixmo"}, "no_modality_dataset")


def test_get_modality_raises_on_empty_string():
    with pytest.raises(KeyError, match="modality"):
        _get_modality({"handler": "image_pixmo", "modality": ""}, "empty")


def test_every_shipped_dataset_has_valid_modality():
    """Regression guard: datasets_config.json should never lose a `modality`."""
    missing = []
    invalid = []
    for zone, name, info in _all_dataset_entries():
        m = info.get("modality")
        if not m:
            missing.append(f"{zone}/{name}")
        elif m not in VALID_MODALITIES:
            invalid.append(f"{zone}/{name}={m!r}")
    assert not missing, f"Datasets missing 'modality': {missing}"
    assert not invalid, f"Datasets with unknown 'modality': {invalid}"
