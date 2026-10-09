"""Unit tests for the cross-rank train-loss aggregation math in PR #112.

The reduction arithmetic is factored into the module-level pure helper
``src.training.trainer_native.compute_loss_stats`` precisely so it can be
exercised directly here, without a process group (the login node has no XPU
and no torch.distributed backend). The collective inside the trainer's
``_global_loss_stats`` closure does the SUM all-reduce and then calls this same
helper, so testing the helper tests the production math.

The all_reduce(SUM) over ranks is modelled by summing the per-rank tuples,
which is exactly what a SUM all-reduce computes.

Properties asserted:
  * token-weighted mean differs from the equal-rank mean when ranks carry
    unequal token counts (the bug this PR fixes: a rank with few, short
    captions and an unusually low loss should be DOWN-weighted in the
    token-weighted loss but counted equally in the rank mean);
  * a zero-token batch falls back to the rank mean (no divide-by-zero);
  * the single-rank path returns weighted_sum / tokens.
"""

from __future__ import annotations

import math

from src.training.trainer_native import compute_loss_stats


def _reduce(per_rank: list[tuple[float, float, int]]) -> dict[str, float]:
    """Model the SUM all-reduce + helper call the trainer performs.

    Args:
        per_rank: list of (rank_mean_loss, weighted_loss_sum, loss_tokens),
            one tuple per rank. len(per_rank) == world_size.
    """
    world_size = len(per_rank)
    assert world_size >= 1
    return compute_loss_stats(
        summed_rank_mean=sum(float(r[0]) for r in per_rank),
        summed_weighted_loss=sum(float(r[1]) for r in per_rank),
        summed_tokens=sum(max(0, int(r[2])) for r in per_rank),
        world_size=world_size,
    )


def test_token_weighting_differs_from_rank_mean_on_unequal_tokens():
    """The exact bug PR #112 fixes.

    rank0: very few tokens (short captions), unusually LOW loss (1.0)
    rank1: many tokens, higher loss (4.0)

    weighted_loss_sum for a rank = rank_mean_loss * loss_tokens.
    """
    # rank0: mean 1.0 over 10 tokens; rank1: mean 4.0 over 990 tokens.
    r0 = (1.0, 1.0 * 10, 10)
    r1 = (4.0, 4.0 * 990, 990)
    out = _reduce([r0, r1])

    # rank_mean ignores token counts: (1.0 + 4.0) / 2 = 2.5
    assert math.isclose(out["rank_mean"], 2.5, rel_tol=0, abs_tol=1e-9)

    # token_mean weights by tokens: (10 + 3960) / 1000 = 3.97
    expected_token_mean = (10.0 + 3960.0) / 1000.0
    assert math.isclose(out["token_mean"], expected_token_mean, abs_tol=1e-9)
    assert math.isclose(out["token_mean"], 3.97, abs_tol=1e-9)

    # The two MUST differ — rank0's low loss is down-weighted in token_mean.
    assert out["token_mean"] != out["rank_mean"]
    # And token_mean is pulled toward the high-token rank's loss (4.0).
    assert out["token_mean"] > out["rank_mean"]
    assert out["global_tokens"] == 1000.0


def test_equal_tokens_makes_means_agree():
    """Sanity check: when token counts are equal, the two means coincide."""
    r0 = (2.0, 2.0 * 100, 100)
    r1 = (6.0, 6.0 * 100, 100)
    out = _reduce([r0, r1])
    assert math.isclose(out["rank_mean"], 4.0, abs_tol=1e-9)
    assert math.isclose(out["token_mean"], 4.0, abs_tol=1e-9)


def test_zero_tokens_falls_back_to_rank_mean_multi_rank():
    """global_tokens == 0 must not divide by zero; falls back to rank_mean."""
    r0 = (3.0, 0.0, 0)
    r1 = (5.0, 0.0, 0)
    out = _reduce([r0, r1])
    assert out["global_tokens"] == 0.0
    assert math.isclose(out["token_mean"], out["rank_mean"], abs_tol=1e-12)
    assert math.isclose(out["rank_mean"], 4.0, abs_tol=1e-9)  # (3+5)/2


def test_single_rank_returns_weighted_sum_over_tokens():
    """world_size == 1 path: token_mean = weighted_sum / tokens."""
    out = _reduce([(2.5, 7.0 * 200, 200)])
    assert math.isclose(out["token_mean"], 7.0, abs_tol=1e-9)
    assert math.isclose(out["rank_mean"], 2.5, abs_tol=1e-9)
    assert out["global_tokens"] == 200.0


def test_single_rank_zero_tokens_falls_back_to_rank_mean():
    out = _reduce([(9.0, 0.0, 0)])
    assert math.isclose(out["token_mean"], 9.0, abs_tol=1e-9)
    assert math.isclose(out["rank_mean"], 9.0, abs_tol=1e-9)
    assert out["global_tokens"] == 0.0


def test_negative_token_total_is_clamped():
    """Production clamps tokens with max(0, ...); the helper clamps the sum."""
    out = compute_loss_stats(
        summed_rank_mean=1.0,
        summed_weighted_loss=5.0,
        summed_tokens=-3.0,
        world_size=1,
    )
    # negative total clamped to 0 -> fall back to rank_mean.
    assert out["global_tokens"] == 0.0
    assert math.isclose(out["token_mean"], 1.0, abs_tol=1e-9)
