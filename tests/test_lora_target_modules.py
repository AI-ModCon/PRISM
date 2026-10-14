"""apply_lora_torchtune honors target_modules, and the adapter sidecar records it.

Previously ``training.lora_target_modules`` was written to the adapter sidecar
but never reached ``apply_lora_torchtune``, so every nn.Linear except lm_head was
adapted regardless; and a Hydra-supplied list crashed ``save_lora_adapter``'s
json.dump. Skipped where torchtune is not installed (CI does not install it).
"""

from __future__ import annotations

import json

import pytest
import torch.nn as nn

pytest.importorskip("torchtune")
from src.utils.lora_utils import apply_lora_torchtune, save_lora_adapter  # noqa: E402
from torchtune.modules.peft import LoRALinear  # noqa: E402

pytestmark = pytest.mark.unit


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.lm_head = nn.Linear(8, 16)


def _adapted(model: nn.Module) -> set[str]:
    return {name for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def test_default_adapts_every_linear_except_lm_head():
    model = apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0)
    assert _adapted(model) == {
        f"layers.{i}.{leaf}" for i in range(2) for leaf in ("q_proj", "v_proj", "o_proj")
    }


def test_target_modules_restricts_adaptation_and_trainable_params():
    model = apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0, target_modules=["q_proj", "v_proj"])
    assert _adapted(model) == {f"layers.{i}.{leaf}" for i in range(2) for leaf in ("q_proj", "v_proj")}
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        f"layers.{i}.{leaf}.lora_{ab}.weight"
        for i in range(2)
        for leaf in ("q_proj", "v_proj")
        for ab in ("a", "b")
    }


@pytest.mark.parametrize(
    ("target_modules", "error"),
    [
        (["q_prj"], ValueError),  # typo matches nothing
        (["lm_head"], ValueError),  # excluded
        ([], ValueError),
        ("q_proj", TypeError),  # bare string, not a list
    ],
)
def test_bad_target_modules_raise(target_modules, error):
    with pytest.raises(error):
        apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0, target_modules=target_modules)


def test_sidecar_records_hydra_list(tmp_path):
    omegaconf = pytest.importorskip("omegaconf")
    targets = omegaconf.OmegaConf.create({"t": ["q_proj"]}).t  # as cfg.training.get() returns it
    model = apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0, target_modules=targets)
    path = tmp_path / "lora_adapter.pt"
    save_lora_adapter(model, str(path), rank=2, alpha=4.0, target_modules=targets)
    meta = json.loads((tmp_path / "lora_adapter.pt.meta.json").read_text())
    assert meta == {"rank": 2, "alpha": 4.0, "target_modules": ["q_proj"]}
