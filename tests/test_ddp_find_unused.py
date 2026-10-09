"""Unit tests for _resolve_find_unused (src/training/distributed.py).

The empty-bucket DDP crash documented in #63 happens specifically in the
projector-only multi-modality regime, where heterogeneous batches across
ranks leave per-modality projectors with no gradient flow on some ranks.
This module verifies the auto-detect logic that flips
`find_unused_parameters=True` for that regime, plus the COMPOSITE
auto-detect that already existed, plus the interleaved-QA auto-detect
added 2026-07-03 (dynamic per-batch modality routing breaks
static_graph=True at multi-node scale — see PR fixing the AB-4N-DDP-TSQA
crash observed on job 8642992).
"""

import pytest
from src.training.distributed import _resolve_find_unused

# --- Default behavior (no signals) ---


def test_default_is_off_for_e2e_single_modality_flat():
    """E2E (>1GB trainable) + single modality + FLAT mode → find_unused=False
    (the throughput-optimal path)."""
    find_unused, mm_proj, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=1,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert find_unused is False
    assert mm_proj is False


def test_default_is_off_for_projector_only_single_modality_flat():
    """Projector-only (<1GB) + single modality + FLAT → find_unused=False.
    Single modality means no heterogeneous batches → no empty-bucket risk."""
    find_unused, mm_proj, _, _ = _resolve_find_unused(
        trainable_mb=12.0,
        num_modalities=1,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert find_unused is False
    assert mm_proj is False


# --- COMPOSITE auto-detect (pre-existing) ---


def test_composite_auto_enables_find_unused():
    """COMPOSITE mode flips find_unused=True regardless of modalities."""
    find_unused, _, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=1,
        is_composite=True,
        is_interleaved=False,
        env_override=None,
    )
    assert find_unused is True


# --- Multi-modality projector auto-detect (pre-existing) ---


def test_multimodal_projector_auto_enables_find_unused():
    """Projector-only (<1GB) + len(modalities)>1 → find_unused=True.
    Regression for the empty-bucket crash (#63)."""
    find_unused, mm_proj, _, _ = _resolve_find_unused(
        trainable_mb=12.0,
        num_modalities=3,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert find_unused is True
    assert mm_proj is True


def test_multimodal_e2e_does_not_auto_enable():
    """E2E (>1GB) + multi-modality (NOT interleaved) → NO projector auto-detect.
    E2E backbones don't have the heterogeneous-batch projector problem in the
    same way (every rank exercises the full backbone every step)."""
    find_unused, mm_proj, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=3,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert find_unused is False
    assert mm_proj is False


def test_threshold_at_1000mb():
    """Boundary check: 999 MB triggers, exactly 1000 MB does not, 1001 MB does not.
    Pins the strict `<` direction of the inequality so a refactor to `<=` is caught."""
    fu_just_below, _, _, _ = _resolve_find_unused(
        trainable_mb=999.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    fu_exact, _, _, _ = _resolve_find_unused(
        trainable_mb=1000.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    fu_just_above, _, _, _ = _resolve_find_unused(
        trainable_mb=1001.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert fu_just_below is True
    assert fu_exact is False
    assert fu_just_above is False


# --- Interleaved-QA auto-detect (added 2026-07-03) ---


def test_interleaved_qa_auto_enables_for_e2e():
    """E2E (>1GB) + interleaved_qa=True → find_unused=True. The multimodal
    projector auto-detect does NOT fire for E2E, but interleaved-QA still
    needs find_unused because the per-batch modality routing means different
    ranks exercise different sub-branches (image encoder vs TS encoder vs
    text-only) across microbatches. Regression for AB-4N-DDP-TSQA crash
    (job 8642992, 2026-07-03): rank-sampling variance at 48 ranks × BS=1
    triggered `Your training graph has changed... static_graph set to True`."""
    find_unused, mm_proj, _, interleaved = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=True,
        env_override=None,
    )
    assert find_unused is True
    assert mm_proj is False  # E2E, not projector-only
    assert interleaved is True


def test_interleaved_qa_auto_enables_for_projector_only():
    """Projector-only + interleaved_qa=True: also enables (both signals fire
    but the result is the same)."""
    find_unused, mm_proj, _, interleaved = _resolve_find_unused(
        trainable_mb=12.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=True,
        env_override=None,
    )
    assert find_unused is True
    assert mm_proj is True
    assert interleaved is True


def test_interleaved_qa_single_modality_still_fires():
    """Interleaved-QA is defined by the model config flag, not by counting
    modalities. If someone sets is_interleaved_qa=True with a single-modality
    config, we still enable find_unused (defensively — the config author
    signaled dynamic routing)."""
    find_unused, _, _, interleaved = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=1,
        is_composite=False,
        is_interleaved=True,
        env_override=None,
    )
    assert find_unused is True
    assert interleaved is True


# --- Env override beats all auto-detects ---


def test_env_override_off_beats_composite():
    find_unused, _, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=1,
        is_composite=True,
        is_interleaved=False,
        env_override="0",
    )
    assert find_unused is False


def test_env_override_off_beats_multimodal_projector():
    find_unused, _, _, _ = _resolve_find_unused(
        trainable_mb=12.0,
        num_modalities=3,
        is_composite=False,
        is_interleaved=False,
        env_override="0",
    )
    assert find_unused is False


def test_env_override_off_beats_interleaved():
    """User can explicitly force static_graph=True on interleaved via env
    override (accepts the risk of the graph-change crash at scale — useful
    for 1-node runs where sampling variance is low enough to be safe)."""
    find_unused, _, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=2,
        is_composite=False,
        is_interleaved=True,
        env_override="0",
    )
    assert find_unused is False


def test_env_override_on_works_without_auto_signal():
    """User can force find_unused=True even when no auto-detect fires."""
    find_unused, _, _, _ = _resolve_find_unused(
        trainable_mb=14_000.0,
        num_modalities=1,
        is_composite=False,
        is_interleaved=False,
        env_override="1",
    )
    assert find_unused is True


# --- Combined signals ---


def test_all_auto_detects_active_returns_true():
    find_unused, mm_proj, composite, interleaved = _resolve_find_unused(
        trainable_mb=12.0,
        num_modalities=3,
        is_composite=True,
        is_interleaved=True,
        env_override=None,
    )
    assert find_unused is True
    assert mm_proj is True
    assert composite is True
    assert interleaved is True


# --- Diagnostic return values ---


@pytest.mark.parametrize(
    "trainable_mb,num_modalities,expected_mm_proj",
    [
        (12.0, 1, False),   # projector-only, single modality
        (12.0, 2, True),    # projector-only, multi-modality (the case)
        (12.0, 6, True),    # projector-only, all 6 modalities
        (14_000.0, 1, False),  # E2E, single
        (14_000.0, 3, False),  # E2E, multi
    ],
)
def test_is_multimodal_projector_diagnostic_flag(
    trainable_mb, num_modalities, expected_mm_proj
):
    """The second return value lets callers attribute the decision in logs."""
    _, mm_proj, _, _ = _resolve_find_unused(
        trainable_mb=trainable_mb,
        num_modalities=num_modalities,
        is_composite=False,
        is_interleaved=False,
        env_override=None,
    )
    assert mm_proj is expected_mm_proj
