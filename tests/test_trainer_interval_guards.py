"""Regression tests for the *_every_n_steps=0 ZeroDivisionError guards.

PR #91 added three guards in src/training/trainer_zone_a.py around
`log_every_n_steps`, `viz_every_n_steps`, and `save_every_n_steps`.
The PRISM-MODALITY-SMOKE design sets all three to 0 to disable
logging/visualization/checkpointing during sweep cells; without the
guards the trainer crashes with ZeroDivisionError on `step % interval`.

We can't instantiate ZoneATrainer without Accelerate + a real model, but
we can:

  1. Pin the predicate shape via source inspection (regex) so a future
     refactor that drops the guard fails this test.
  2. Test the boolean short-circuit predicate directly (mirrors what
     the trainer evaluates inline at each step).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINER = ROOT / "src" / "training" / "trainer_zone_a.py"


# --- Behavioral predicate tests ---
#
# These mirror the inline predicates exactly. If anyone refactors and
# drops a guard, the source-inspection test (below) will catch it; if
# the predicate semantics change, these tests will catch it.

def _should_log(interval, step):
    return bool(interval and step % interval == 0)


def _should_viz_or_save(interval, step):
    """viz_every_n_steps and save_every_n_steps share the `step > 0` guard
    so the first step doesn't fire them."""
    return bool(interval and step > 0 and step % interval == 0)


def test_log_predicate_interval_zero_does_not_fire():
    for step in range(5):
        assert not _should_log(0, step), f"interval=0 must NOT fire at step={step}"


def test_log_predicate_interval_one_fires_every_step():
    for step in range(5):
        assert _should_log(1, step)


def test_log_predicate_normal_interval():
    assert _should_log(10, 0)
    assert not _should_log(10, 1)
    assert _should_log(10, 10)
    assert not _should_log(10, 11)
    assert _should_log(10, 100)


def test_viz_save_predicate_interval_zero_does_not_fire():
    for step in range(1, 10):
        assert not _should_viz_or_save(0, step), f"interval=0 must NOT fire at step={step}"


def test_viz_save_predicate_step_zero_does_not_fire():
    """viz/save guard the first step explicitly with `step > 0` so an
    interval that divides 0 doesn't fire on the loop's first iteration."""
    for interval in (1, 5, 50):
        assert not _should_viz_or_save(interval, 0), (
            f"interval={interval} step=0 must NOT fire (step > 0 guard)"
        )


def test_viz_save_predicate_normal_interval():
    assert not _should_viz_or_save(10, 0)
    assert not _should_viz_or_save(10, 1)
    assert _should_viz_or_save(10, 10)
    assert _should_viz_or_save(10, 100)


# --- Source-inspection guards ---
#
# Static check that the three guards are still present in the trainer.
# Catches a refactor that silently drops the `interval and ...` short-circuit.

def _trainer_source() -> str:
    return TRAINER.read_text()


def test_log_every_n_steps_has_truthy_guard():
    src = _trainer_source()
    # The guard must check log_interval truthy BEFORE the modulo to avoid
    # ZeroDivisionError. We accept either `if log_interval and step % log_interval`
    # form — the regex below is intentionally permissive.
    assert "log_interval = self.config.log_every_n_steps" in src
    assert "if log_interval and step % log_interval == 0" in src, (
        "log_every_n_steps modulo must be guarded by truthy `log_interval` check; "
        "see PR #91 fix for ZeroDivisionError when smoke design sets the interval to 0."
    )


def test_viz_every_n_steps_has_truthy_guard():
    src = _trainer_source()
    assert "viz_interval = self.config.viz_every_n_steps" in src
    assert "if viz_interval and step > 0 and step % viz_interval == 0" in src, (
        "viz_every_n_steps modulo must be guarded by truthy `viz_interval` check"
    )


def test_save_every_n_steps_has_truthy_guard():
    src = _trainer_source()
    assert 'save_interval = getattr(self.config, "save_every_n_steps", 1000)' in src
    assert "if save_interval and step > 0 and step % save_interval == 0" in src, (
        "save_every_n_steps modulo must be guarded by truthy `save_interval` check"
    )


def test_final_step_perf_record_branch_present():
    """The companion fix — `is_final_step` ensures a 50-step smoke (which
    exits at step 49 without ever hitting step % 50 == 0) still emits a
    perf record. Pin the structure so a refactor doesn't drop it."""
    src = _trainer_source()
    assert "is_final_step = (step + 1) >= self.config.max_steps" in src, (
        "is_final_step branch missing; short runs (max_steps=50) lose their "
        "only perf record. See PR #91."
    )
    # The branch must combine periodic OR final-step (vs. AND).
    assert "or is_final_step" in src, "is_final_step must be OR'd with periodic trigger"


def test_trainer_module_compiles():
    """Sanity: source inspection only catches regressions if the file still
    parses as Python."""
    src = _trainer_source()
    ast.parse(src)
