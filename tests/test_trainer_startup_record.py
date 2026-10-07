"""Unit-tests the startup_param_count perf record shape.

The full trainer integration is exercised by PR-1 Smoke 1 (instrumented
training on Aurora). Here we just confirm that the record dict the
trainer constructs:
  - keys off `count_parameters` correctly
  - serializes through `log_perf_record` without raising
  - includes the projector knobs alongside per-component counts

We bypass the trainer's heavy `__init__` and only mimic the snippet that
emits the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn
from src.utils.perf_log import count_parameters, log_perf_record, model_modalities


class _StubConfig:
    modalities = ["text", "image"]
    projector_hidden_mult = 2
    projector_num_layers = 4
    sweep_id = "TEST-SWEEP"
    preset = "test_preset"


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(64, 64)
        self.encoders = nn.ModuleDict({"image": nn.Linear(8, 8)})
        self.projectors = nn.ModuleDict({"image": nn.Linear(8, 64)})
        self.config = _StubConfig()


def test_startup_record_writes_and_round_trips(tmp_path: Path) -> None:
    model = _StubModel()
    record = {
        "event": "startup_param_count",
        "site": "trainer_zone_a",
        **count_parameters(model),
        "sweep_id": getattr(model.config, "sweep_id", None),
        "preset": getattr(model.config, "preset", None),
        "modalities": model_modalities(model),
        "projector_hidden_mult": getattr(model.config, "projector_hidden_mult", 1),
        "projector_num_layers": getattr(model.config, "projector_num_layers", 2),
    }
    log_perf_record(tmp_path, record)

    line = (tmp_path / "perf.jsonl").read_text().strip().splitlines()
    assert len(line) == 1
    parsed = json.loads(line[0])
    assert parsed["event"] == "startup_param_count"
    assert parsed["backbone"] > 0
    assert parsed["projector_image"] > 0
    assert parsed["encoder_image"] > 0
    assert parsed["total"] > 0
    assert parsed["modalities"] == ["text", "image"]
    assert parsed["projector_hidden_mult"] == 2
    assert parsed["projector_num_layers"] == 4
    assert parsed["sweep_id"] == "TEST-SWEEP"
    assert parsed["preset"] == "test_preset"
