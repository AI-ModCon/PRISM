"""apply_lora_torchtune honors target_modules, and the adapter sidecar records it.

Previously ``training.lora_target_modules`` was written to the adapter sidecar
but never reached ``apply_lora_torchtune``, so every nn.Linear except lm_head was
adapted regardless; and a Hydra-supplied list crashed ``save_lora_adapter``'s
json.dump. The layer-selection and name-check tests need no torchtune and run in
CI; the LoRALinear and sidecar tests skip where torchtune is not installed (CI
does not install it).
"""

from __future__ import annotations

import json

import pytest
import torch.nn as nn
from src.utils.lora_utils import _select_lora_targets, apply_lora_torchtune, save_lora_adapter

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


_ALL = {f"layers.{i}.{leaf}" for i in range(2) for leaf in ("q_proj", "v_proj", "o_proj")}
_QV = {f"layers.{i}.{leaf}" for i in range(2) for leaf in ("q_proj", "v_proj")}


def _selected(target_modules) -> set[str]:
    return {name for name, _ in _select_lora_targets(_Tiny(), ("lm_head",), target_modules)}


def test_default_selects_every_linear_except_lm_head():
    assert _selected(None) == _ALL


def test_target_modules_restricts_selection():
    assert _selected(["q_proj", "v_proj"]) == _QV


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
        _selected(target_modules)


@pytest.mark.parametrize(
    ("target_modules", "expected"), [(None, _ALL), (["q_proj", "v_proj"], _QV)]
)
def test_apply_adapts_selected_layers_and_trains_only_lora(target_modules, expected):
    LoRALinear = pytest.importorskip("torchtune.modules.peft").LoRALinear
    model = apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0, target_modules=target_modules)
    assert {
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    } == expected
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {f"{name}.lora_{ab}.weight" for name in expected for ab in ("a", "b")}


def test_sidecar_records_hydra_list(tmp_path):
    pytest.importorskip("torchtune")
    omegaconf = pytest.importorskip("omegaconf")
    targets = omegaconf.OmegaConf.create({"t": ["q_proj"]}).t  # as cfg.training.get() returns it
    model = apply_lora_torchtune(_Tiny(), rank=2, alpha=4.0, target_modules=targets)
    path = tmp_path / "lora_adapter.pt"
    save_lora_adapter(model, str(path), rank=2, alpha=4.0, target_modules=targets)
    meta = json.loads((tmp_path / "lora_adapter.pt.meta.json").read_text())
    assert meta == {"rank": 2, "alpha": 4.0, "target_modules": ["q_proj"]}
