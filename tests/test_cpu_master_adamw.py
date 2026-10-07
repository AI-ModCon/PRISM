"""Dense CPU masters preserve sub-BF16 updates and exact checkpoint continuation."""

import copy

import pytest
import torch
from src.training.cpu_master_adamw import CPUMasterAdamW
from torch import nn


def _optimizer(*, lr=1e-4, clip=10.0, decay=0.0):
    values = {
        "connector.weight": torch.tensor([0.25001, -0.50003]),
        "diffusion.weight": torch.tensor([1.0003, -0.7502]),
        "diffusion.reference": torch.tensor([0.3751]),
    }
    runtime = {
        name: nn.Parameter(
            value.clone().to(torch.float32 if name.startswith("connector") else torch.bfloat16)
        )
        for name, value in values.items()
    }
    optimizer = CPUMasterAdamW(
        named_groups={
            "connector": {"connector.weight": runtime["connector.weight"]},
            "diffusion": {name: p for name, p in runtime.items() if name.startswith("diffusion")},
        },
        master_values=values,
        learning_rates={"connector": lr * 2, "diffusion": lr},
        max_grad_norm=clip,
        weight_decay=decay,
    )
    return optimizer, runtime


def _gradients(runtime, step):
    runtime["connector.weight"].grad = torch.tensor([0.125 * (step + 1), -0.25])
    runtime["diffusion.weight"].grad = torch.tensor(
        [0.5, -0.125 * (step + 1)], dtype=torch.bfloat16
    )
    runtime["diffusion.reference"].grad = None


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_equal(a, b)
    else:
        assert left == right


def test_small_updates_accumulate_in_exact_fp32_master():
    original = torch.tensor([1.0003])
    runtime = nn.Parameter(original.to(torch.bfloat16))
    optimizer = CPUMasterAdamW(
        named_groups={"diffusion": {"weight": runtime}},
        master_values={"weight": original.clone()},
        learning_rates={"diffusion": 1e-4},
        max_grad_norm=10,
    )
    reference = nn.Parameter(original.clone())
    expected = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0, foreach=False, fused=False)
    changed = []
    for _ in range(40):
        optimizer.zero_grad()
        runtime.grad = torch.ones_like(runtime)
        reference.grad = torch.ones_like(reference)
        expected.step()
        changed.append(optimizer.step()["groups"]["diffusion"]["changed_runtime_tensors"])
        assert torch.equal(optimizer.master_parameters["weight"], reference)
        assert torch.equal(runtime, reference.to(torch.bfloat16))
        assert optimizer.master_parameters["weight"].grad is None
    assert changed[0] == 0 and sum(changed) > 0
    assert runtime.item() < original.to(torch.bfloat16).item()
    naive = original.to(torch.bfloat16)
    for _ in range(40):
        naive -= 1e-4
    assert naive.item() == original.to(torch.bfloat16).item()


def test_mixed_groups_global_clipping_match_fp32_reference_and_accumulation():
    optimizer, runtime = _optimizer(clip=0.5)
    reference = {
        name: nn.Parameter(value.detach().clone())
        for name, value in optimizer.master_parameters.items()
    }
    expected = torch.optim.AdamW(
        [
            {"params": [reference[n] for n in optimizer.groups[group]], "lr": rate}
            for group, rate in optimizer.settings["learning_rates"].items()
        ],
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    for step in range(4):
        optimizer.zero_grad()
        _gradients(runtime, step)
        # Two accumulated microbatches are represented by an existing grad plus
        # a second addition, exactly as autograd accumulation does at a boundary.
        for name in ("connector.weight", "diffusion.weight"):
            runtime[name].grad.add_(runtime[name].grad.clone())
        grad_snapshots = {name: p.grad.clone() for name, p in runtime.items() if p.grad is not None}
        norms = []
        for name, p in reference.items():
            p.grad = runtime[name].grad.float().clone() if runtime[name].grad is not None else None
            if p.grad is not None:
                norms.append(torch.linalg.vector_norm(p.grad))
        norm = float(torch.linalg.vector_norm(torch.stack(norms)))
        scale = min(1.0, 0.5 / (norm + 1e-6))
        for p in reference.values():
            if p.grad is not None:
                p.grad.mul_(scale)
        expected.step()
        diagnostics = optimizer.step()
        assert diagnostics["gradient_norm_before_clip"] == norm
        assert diagnostics["clip_scale"] == scale
        assert diagnostics["groups"]["diffusion"]["missing_gradient_names"] == [
            "diffusion.reference"
        ]
        for name, p in reference.items():
            assert torch.equal(optimizer.master_parameters[name], p)
            assert torch.equal(runtime[name], p.to(runtime[name].dtype))
        for name, gradient in grad_snapshots.items():
            assert torch.equal(runtime[name].grad, gradient)
        assert all(p.grad is None for p in optimizer.master_parameters.values())
    assert optimizer.optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.optimizer.param_groups[1]["lr"] == 1e-4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 3e38])
def test_nonfinite_gradient_or_norm_fails_before_any_mutation(bad):
    optimizer, runtime = _optimizer()
    _gradients(runtime, 0)
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    audit = optimizer.audit()
    _gradients(runtime, 1)
    runtime["diffusion.weight"].grad.fill_(bad)
    with pytest.raises(FloatingPointError, match="nonfinite|Nonfinite"):
        optimizer.step()
    _assert_equal(before, optimizer.state_dict())
    assert optimizer.audit() == audit
    assert all(p.grad is None for p in optimizer.master_parameters.values())


def test_missing_gradients_skip_adam_and_zero_gradients_keep_adam_semantics():
    optimizer, runtime = _optimizer(lr=0.01, decay=0.1)
    before = optimizer.audit()
    runtime["connector.weight"].grad = torch.zeros_like(runtime["connector.weight"])
    diagnostics = optimizer.step()
    connector = diagnostics["groups"]["connector"]
    assert connector["zero_gradient_names"] == ["connector.weight"]
    assert connector["nonzero_gradient_names"] == []
    assert connector["changed_runtime_tensors"] == 1  # Decoupled weight decay.
    assert optimizer.audit()["groups"]["diffusion"] == before["groups"]["diffusion"]
    assert len(optimizer.optimizer.state) == 1
    optimizer.zero_grad()
    assert all(p.grad is None for p in runtime.values())
    state = copy.deepcopy(optimizer.state_dict())
    diagnostics = optimizer.step()
    assert diagnostics["gradient_norm_before_clip"] == 0
    _assert_equal(state, optimizer.state_dict())


def test_audit_is_state_free_and_does_not_copy_donated_master_storage():
    value = torch.tensor([0.5001])
    runtime = nn.Parameter(value.to(torch.bfloat16))
    optimizer = CPUMasterAdamW(
        named_groups={"diffusion": {"weight": runtime}},
        master_values={"weight": value},
        learning_rates={"diffusion": 1e-5},
    )
    assert optimizer.master_parameters["weight"].data_ptr() == value.data_ptr()
    audit = optimizer.audit()
    assert len(optimizer.optimizer.state) == 0
    assert audit == optimizer.audit()
    group = audit["groups"]["diffusion"]
    assert group["parameter_tensors"] == group["parameter_scalars"] == 1
    assert len(group["master_sha256"]) == len(group["runtime_sha256"]) == 64


def test_checkpoint_resume_exactly_reproduces_master_runtime_and_adam(tmp_path):
    uninterrupted, runtime = _optimizer(clip=0.5)
    for step in range(3):
        uninterrupted.zero_grad()
        _gradients(runtime, step)
        uninterrupted.step()
    path = tmp_path / "optimizer.pt"
    torch.save(uninterrupted.state_dict(), path)
    resumed, resumed_runtime = _optimizer(clip=0.5)
    restored = torch.load(path, weights_only=True)
    resumed.load_state_dict(restored)
    assert resumed.audit() == uninterrupted.audit()
    for step in range(3, 8):
        uninterrupted.zero_grad()
        resumed.zero_grad()
        _gradients(runtime, step)
        _gradients(resumed_runtime, step)
        assert uninterrupted.step() == resumed.step()
    _assert_equal(uninterrupted.state_dict(), resumed.state_dict())
    assert uninterrupted.audit() == resumed.audit()


@pytest.mark.parametrize(
    "problem", ["scope", "learning_rate", "missing_master", "dtype", "shape", "cast", "alias"]
)
def test_initialization_rejects_wrong_scope_precision_and_alignment(problem):
    runtime = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    values = {"weight": torch.tensor([1.0])}
    groups = {"diffusion": {"weight": runtime}}
    rates = {"diffusion": 1e-5}
    if problem == "scope":
        groups["connector"] = {"weight": runtime}
        rates["connector"] = 1e-5
    elif problem == "learning_rate":
        rates["diffusion"] = float("nan")
    elif problem == "missing_master":
        values.clear()
    elif problem == "dtype":
        values["weight"] = values["weight"].bfloat16()
    elif problem == "shape":
        values["weight"] = torch.ones(2)
    elif problem == "cast":
        values["weight"] += 1
    else:
        runtime = nn.Parameter(torch.ones(1))
        groups["diffusion"]["weight"] = runtime
        values["weight"] = runtime.detach()
    with pytest.raises(ValueError):
        CPUMasterAdamW(named_groups=groups, master_values=values, learning_rates=rates)


@pytest.mark.parametrize("problem", ["settings", "scope", "master", "moment", "groups", "step"])
def test_resume_rejects_corrupt_or_mismatched_state_before_mutation(problem):
    optimizer, runtime = _optimizer()
    _gradients(runtime, 0)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    before = optimizer.audit()
    saved = copy.deepcopy(optimizer.state_dict())
    if problem == "settings":
        state["settings"]["learning_rates"]["diffusion"] = 3e-5
    elif problem == "scope":
        state["scope"]["diffusion"][0]["runtime_dtype"] = "torch.float32"
    elif problem == "master":
        state["masters"]["diffusion.weight"][0] = float("nan")
    elif problem == "moment":
        next(iter(state["optimizer"]["state"].values()))["exp_avg"].fill_(float("inf"))
    elif problem == "groups":
        state["optimizer"]["param_groups"][0]["lr"] = 9e-5
    else:
        next(iter(state["optimizer"]["state"].values()))["step"].fill_(-1)
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)
    assert optimizer.audit() == before
    _assert_equal(optimizer.state_dict(), saved)
