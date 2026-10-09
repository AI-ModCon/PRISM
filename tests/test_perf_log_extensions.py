"""Tests for IsoFLOP additions to src/utils/perf_log.py.

Covers `count_parameters`, `sequence_stats`, and `_FlopCounter`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch.nn as nn
from src.utils.perf_log import _FlopCounter, count_parameters, sequence_stats


class _FakeModel(nn.Module):
    """Mimics UnifiedTransformer's named-submodule layout for counting."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(100, 100, bias=False)  # 10000 params
        self.encoders = nn.ModuleDict({
            "image": nn.Linear(50, 50, bias=False),  # 2500 params
            "time_series": nn.Linear(20, 20, bias=False),  # 400 params
        })
        self.projectors = nn.ModuleDict({
            "image": nn.Linear(50, 100, bias=False),  # 5000 params
            "time_series": nn.Linear(20, 100, bias=False),  # 2000 params
        })


def test_count_parameters_buckets_match_expected() -> None:
    model = _FakeModel()
    counts = count_parameters(model)
    assert counts["backbone"] == 10000
    assert counts["encoder_image"] == 2500
    assert counts["encoder_time_series"] == 400
    assert counts["projector_image"] == 5000
    assert counts["projector_time_series"] == 2000
    expected_total = 10000 + 2500 + 400 + 5000 + 2000
    assert counts["total"] == expected_total
    # By default every param is trainable, so train == total.
    assert counts["train"] == expected_total
    # PR-1 placeholder: active mirrors total.
    assert counts["active"] == expected_total


def test_count_parameters_respects_requires_grad() -> None:
    model = _FakeModel()
    for p in model.backbone.parameters():
        p.requires_grad = False
    counts = count_parameters(model)
    assert counts["train"] == counts["total"] - counts["backbone"]


def test_count_parameters_unwraps_module() -> None:
    """count_parameters must walk past DDP/FSDP `.module` wrappers."""
    inner = _FakeModel()

    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    wrapped = _Wrapper(inner)
    counts = count_parameters(wrapped)
    # Should match the unwrapped counts; if unwrap is broken, "backbone"
    # would be missing entirely.
    assert counts["backbone"] == 10000


def test_sequence_stats_basic() -> None:
    stats = sequence_stats([10, 20, 30, 40])
    assert stats["seq_p50"] == 25.0  # numpy percentile linear interp
    # Padding ratio: 4 samples each padded to max=40, sum=100, denom=160,
    # padding = (160-100)/160 = 0.375
    assert stats["padding_ratio"] == 0.375
    assert stats["seq_max"] == 40.0
    # p95 with 4 samples is between 37 and 40 via linear interpolation;
    # don't pin to a specific decimal — just check the bound.
    assert stats["seq_p95"] >= stats["seq_p50"]
    assert stats["seq_p99"] >= stats["seq_p95"]


def test_sequence_stats_empty_window() -> None:
    stats = sequence_stats([])
    assert stats["seq_p50"] == 0.0
    assert stats["padding_ratio"] == 0.0
    assert stats["seq_max"] == 0.0


def test_sequence_stats_uniform_no_padding() -> None:
    # All samples the same length → no padding.
    stats = sequence_stats([100, 100, 100])
    assert stats["padding_ratio"] == 0.0
    assert stats["seq_p50"] == 100.0
    assert stats["seq_max"] == 100.0


def test_flop_counter_uncalibrated_returns_none() -> None:
    counter = _FlopCounter.from_calibration(None)
    a, b = counter.step()
    assert a is None and b is None
    a, b = counter.step()
    assert a is None and b is None


def test_flop_counter_accumulates_from_json(tmp_path: Path) -> None:
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 1.5e15}))
    counter = _FlopCounter.from_calibration(str(cal))
    a, b = counter.step()
    assert a == 1.5e15
    assert b == 1.5e15
    a, b = counter.step()
    assert a == 1.5e15
    assert b == 3.0e15


def test_flop_counter_bad_json_degrades_gracefully(tmp_path: Path) -> None:
    cal = tmp_path / "bad.json"
    cal.write_text("not json")
    counter = _FlopCounter.from_calibration(str(cal))
    a, b = counter.step()
    assert a is None and b is None


def test_flop_counter_missing_file_is_silent(tmp_path: Path) -> None:
    counter = _FlopCounter.from_calibration(str(tmp_path / "missing.json"))
    a, b = counter.step()
    assert a is None and b is None


def test_flop_counter_step_accepts_n_steps_multiplier(tmp_path: Path) -> None:
    """`.step(n_steps=N)` multiplies cumulative by N — required because
    trainers call .step() once per perf flush (~50 training steps) but the
    counter was previously advancing by one step's worth per call, so
    cumulative_flops undercounted by the flush interval. See PR feedback
    on PR #98.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 1.0e15}))
    counter = _FlopCounter.from_calibration(str(cal))
    # Two flushes of 50 steps each = 100 training steps × 1e15 fps = 1e17 cum.
    per, cum = counter.step(n_steps=50)
    assert per == 1.0e15
    assert cum == 50 * 1.0e15
    per, cum = counter.step(n_steps=50)
    assert per == 1.0e15
    assert cum == 100 * 1.0e15


def test_flop_counter_default_n_steps_is_1_backcompat(tmp_path: Path) -> None:
    """Bare .step() (no n_steps) keeps the original behavior of advancing
    one step's worth. This is the back-compat hatch for unit tests + any
    caller that genuinely advances one step at a time.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 2.0e15}))
    counter = _FlopCounter.from_calibration(str(cal))
    per, cum = counter.step()
    assert per == 2.0e15 and cum == 2.0e15
    per, cum = counter.step()
    assert per == 2.0e15 and cum == 4.0e15


def test_flop_counter_uncalibrated_ignores_n_steps() -> None:
    """When no calibration, .step(n_steps=N) still returns (None, None)."""
    counter = _FlopCounter.from_calibration(None)
    a, b = counter.step(n_steps=42)
    assert a is None and b is None


@pytest.fixture
def _restore_runtime_fps_env():
    """Save/restore RUNTIME_FLOPS_PER_STEP so tests don't leak across runs."""
    prev = os.environ.pop("RUNTIME_FLOPS_PER_STEP", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["RUNTIME_FLOPS_PER_STEP"] = prev
        else:
            os.environ.pop("RUNTIME_FLOPS_PER_STEP", None)


def test_runtime_flops_env_overrides_cal_json(
    tmp_path: Path, _restore_runtime_fps_env
) -> None:
    """RUNTIME_FLOPS_PER_STEP takes precedence over the JSON's flops_per_step.

    Closes the cal/runtime FLOP accounting gap from PR #98 review: without
    this, trainer-side cumulative_flops would undercount the plan's
    budget_flops by `rescale_factor` (~48x default). The plan writes the
    rescaled value; isoflop_launch.py exports it via this env var.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 1.0e15}))  # raw cal
    os.environ["RUNTIME_FLOPS_PER_STEP"] = "4.8e16"  # rescaled (48x)
    counter = _FlopCounter.from_calibration(str(cal))
    per, cum = counter.step(n_steps=10)
    assert per == 4.8e16, "env var must override JSON's flops_per_step"
    assert cum == 10 * 4.8e16


def test_runtime_flops_env_works_without_cal_json(_restore_runtime_fps_env) -> None:
    """RUNTIME_FLOPS_PER_STEP alone (no JSON) is a valid uncalibrated->calibrated path.

    Useful for ad-hoc smokes that want FLOP attribution without needing a
    pre-generated calibration file.
    """
    os.environ["RUNTIME_FLOPS_PER_STEP"] = "2.5e15"
    counter = _FlopCounter.from_calibration(None)
    per, cum = counter.step(n_steps=2)
    assert per == 2.5e15
    assert cum == 5.0e15


def test_runtime_flops_env_ignored_when_zero_or_negative(
    tmp_path: Path, _restore_runtime_fps_env
) -> None:
    """A zero/negative RUNTIME_FLOPS_PER_STEP falls back to the JSON value
    rather than poisoning the counter with an invalid rate.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 1.0e15}))
    os.environ["RUNTIME_FLOPS_PER_STEP"] = "0"
    counter = _FlopCounter.from_calibration(str(cal))
    per, _ = counter.step()
    assert per == 1.0e15


def test_runtime_flops_env_ignored_when_non_numeric(
    tmp_path: Path, _restore_runtime_fps_env
) -> None:
    """Garbage RUNTIME_FLOPS_PER_STEP must not crash the trainer — fall
    back to JSON value with a logged warning.
    """
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"flops_per_step": 1.0e15}))
    os.environ["RUNTIME_FLOPS_PER_STEP"] = "not-a-number"
    counter = _FlopCounter.from_calibration(str(cal))
    per, _ = counter.step()
    assert per == 1.0e15
