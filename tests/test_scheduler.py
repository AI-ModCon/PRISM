from __future__ import annotations

import pytest
import torch
from src.utils.scheduler import get_wsd_scheduler


def _dummy_optimizer():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.AdamW([param], lr=1.0)


def test_wsd_scheduler_has_warmup_stable_and_decay_phases():
    scheduler = get_wsd_scheduler(
        _dummy_optimizer(),
        num_warmup_steps=10,
        num_training_steps=100,
        min_lr_ratio=0.1,
        decay_ratio=0.2,
    )
    lr_scale = scheduler.lr_lambdas[0]

    assert lr_scale(0) == pytest.approx(0.0)
    assert lr_scale(5) == pytest.approx(0.5)
    assert lr_scale(10) == pytest.approx(1.0)
    assert lr_scale(79) == pytest.approx(1.0)
    assert lr_scale(80) == pytest.approx(1.0)
    assert 0.1 < lr_scale(90) < 1.0
    assert lr_scale(100) == pytest.approx(0.1)


def test_wsd_scheduler_explicit_decay_steps_override_ratio():
    scheduler = get_wsd_scheduler(
        _dummy_optimizer(),
        num_warmup_steps=5,
        num_training_steps=50,
        min_lr_ratio=0.2,
        decay_ratio=0.5,
        decay_steps=10,
    )
    lr_scale = scheduler.lr_lambdas[0]

    assert lr_scale(39) == pytest.approx(1.0)
    assert lr_scale(40) == pytest.approx(1.0)
    assert 0.2 < lr_scale(45) < 1.0
    assert lr_scale(50) == pytest.approx(0.2)


def test_wsd_scheduler_decay_ratio_zero_is_warmup_stable_only():
    scheduler = get_wsd_scheduler(
        _dummy_optimizer(),
        num_warmup_steps=5,
        num_training_steps=50,
        min_lr_ratio=0.2,
        decay_ratio=0.0,
    )
    lr_scale = scheduler.lr_lambdas[0]

    assert lr_scale(0) == pytest.approx(0.0)
    assert lr_scale(4) == pytest.approx(0.8)
    assert lr_scale(5) == pytest.approx(1.0)
    assert lr_scale(50) == pytest.approx(1.0)
