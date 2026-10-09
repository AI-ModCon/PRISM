"""Unit-test the runtime patch that flips skip→False + fallback_dummy=True
when allow_dummy_data=True is passed to StreamingMultimodalDataset.

We don't construct the full dataset (requires tokenizer + network); we just
exercise the activation logic by mocking the manager and bailing out of
init after the activation block runs.
"""
import logging
from dataclasses import dataclass, field
from unittest.mock import patch

from src.data.multimodal import StreamingMultimodalDataset


def _mock_datasets_map():
    return {
        "image_a": {"handler": "image_pixmo", "modality": "image", "skip": True},
        "ts_real": {"handler": "ts_qa", "modality": "time_series", "skip": False},
        "graph_b": {"handler": "graph_captioning", "modality": "graph", "skip": True},
    }


class _FakeManager:
    def __init__(self, datasets_map):
        self._datasets_map = datasets_map

    def get_zone_config(self, zone):
        return {"datasets": self._datasets_map}


@dataclass
class _FakeModelConfig:
    """Stand-in for ModelConfig with only the fields the activation block reads.
    Defaults to all six modalities so the existing tests keep passing."""
    modalities: list = field(
        default_factory=lambda: ["text", "image", "table", "time_series", "geometry", "graph"]
    )


def test_allow_dummy_data_activates_skipped_datasets(caplog):
    datasets_map = _mock_datasets_map()
    with caplog.at_level(logging.INFO, logger="src.data.multimodal"):
        with patch("src.data.multimodal.DatasetManager", return_value=_FakeManager(datasets_map)):
            try:
                StreamingMultimodalDataset(
                    tokenizer=None,
                    allow_dummy_data=True,
                    force_streaming=True,
                    model_config=_FakeModelConfig(),
                )
            except Exception:
                # Init may fail downstream (Tapas tokenizer, network) — that's fine.
                # We just need the activation block (which runs near the top) to have run.
                pass
    # Direct evidence the activation block ran (not just stale state).
    assert any(
        "activated 2 skipped datasets" in rec.message for rec in caplog.records
    ), f"Activation log missing; records: {[r.message for r in caplog.records]}"
    assert datasets_map["image_a"]["skip"] is False
    assert datasets_map["image_a"]["fallback_dummy"] is True
    assert datasets_map["graph_b"]["skip"] is False
    assert datasets_map["graph_b"]["fallback_dummy"] is True
    # Real dataset (skip already False) unchanged
    assert datasets_map["ts_real"]["skip"] is False
    assert "fallback_dummy" not in datasets_map["ts_real"]


def test_allow_dummy_data_false_leaves_skipped_alone(caplog):
    datasets_map = _mock_datasets_map()
    with caplog.at_level(logging.INFO, logger="src.data.multimodal"):
        with patch("src.data.multimodal.DatasetManager", return_value=_FakeManager(datasets_map)):
            try:
                StreamingMultimodalDataset(
                    tokenizer=None,
                    allow_dummy_data=False,
                    force_streaming=True,
                    model_config=_FakeModelConfig(),
                )
            except Exception:
                pass
    # No activation log should be emitted when the feature is off.
    assert not any(
        "activated" in rec.message and "skipped datasets" in rec.message
        for rec in caplog.records
    )
    assert datasets_map["image_a"]["skip"] is True
    assert "fallback_dummy" not in datasets_map["image_a"]


def test_activation_skips_modalities_not_in_model_config(caplog):
    """When model.modalities = [text, image], graph (skip=True) must be left
    alone — flipping it would waste init time loading a stream the model
    can't consume."""
    datasets_map = _mock_datasets_map()
    with caplog.at_level(logging.INFO, logger="src.data.multimodal"):
        with patch("src.data.multimodal.DatasetManager", return_value=_FakeManager(datasets_map)):
            try:
                StreamingMultimodalDataset(
                    tokenizer=None,
                    allow_dummy_data=True,
                    force_streaming=True,
                    model_config=_FakeModelConfig(modalities=["text", "image"]),
                )
            except Exception:
                pass
    # image_a (modality=image, in wanted set) → activated
    assert datasets_map["image_a"]["skip"] is False
    assert datasets_map["image_a"]["fallback_dummy"] is True
    # graph_b (modality=graph, NOT in wanted set) → left skipped
    assert datasets_map["graph_b"]["skip"] is True
    assert "fallback_dummy" not in datasets_map["graph_b"]
    # The "left N skipped datasets untouched" log should mention graph_b
    assert any(
        "left 1 skipped datasets untouched" in rec.message for rec in caplog.records
    ), f"Scoping log missing; records: {[r.message for r in caplog.records]}"
