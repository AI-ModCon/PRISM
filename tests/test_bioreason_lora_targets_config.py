"""The BioReason training YAMLs declare lora_target_modules, and the listing tool works.

Declaring the key lets ``training.lora_target_modules=[...]`` override it
without Hydra's ``+`` prefix (struct mode rejects undeclared keys). GRPO ships
``[q_proj, v_proj]``; SFT ships ``null`` (every nn.Linear except lm_head).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn
from hydra import compose, initialize_config_dir
from src.train import _coerce_listconfig
from tools.list_lora_targets import count_targets

pytestmark = pytest.mark.unit

CONF = Path(__file__).resolve().parents[1] / "src" / "conf"


def _targets(*overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        cfg = compose(config_name="config", overrides=["model=prism_olmo_1b", *overrides])
    return _coerce_listconfig(cfg.training.lora_target_modules)


@pytest.mark.parametrize(
    ("training", "expected"),
    [("bioreason_grpo", ["q_proj", "v_proj"]), ("bioreason_sft", None)],
)
def test_shipped_default(training, expected):
    assert _targets(f"training={training}") == expected


@pytest.mark.parametrize("training", ["bioreason_grpo", "bioreason_sft"])
@pytest.mark.parametrize(
    ("override", "expected"),
    [("[k_proj,o_proj]", ["k_proj", "o_proj"]), ("null", None)],
)
def test_plain_override_needs_no_plus(training, override, expected):
    targets = _targets(f"training={training}", f"training.lora_target_modules={override}")
    assert targets == expected
    assert targets is None or type(targets) is list


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.act = nn.GELU()


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block(), _Block()])
        self.lm_head = nn.Linear(8, 16)


def test_count_targets_lists_linear_leaf_names_except_lm_head():
    assert count_targets(_Tiny()) == {"q_proj": 3, "v_proj": 3}
