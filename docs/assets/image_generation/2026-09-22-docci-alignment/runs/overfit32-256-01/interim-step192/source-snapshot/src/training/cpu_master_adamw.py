"""Dense AdamW with exact FP32 CPU masters and BF16/FP32 compute parameters.

The caller transfers ownership of the contiguous CPU tensors in ``master_values``.
No second full model copy is made: discard that mapping after construction and do
not mutate its tensors. ``state_dict`` also follows PyTorch's shallow snapshot
convention; serialize it before the next update. Checkpoint loading transfers the
loaded master storage in the same way. None of these APIs initialize Adam state
until a parameter actually receives a gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn


def _positive_finite(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _master_check(value: torch.Tensor, runtime: nn.Parameter, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.shape != runtime.shape
    ):
        raise ValueError(f"{name}: master must be contiguous CPU float32 with runtime shape")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: master contains nonfinite values")


def _update_hash(digest, name: str, value: torch.Tensor) -> None:
    """Hash canonical tensor bytes using at most four MiB of host staging."""
    digest.update(
        (
            json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")) + "\n"
        ).encode()
    )
    flat = value.detach().view(-1)
    chunk_elements = max(1, (4 * 1024 * 1024) // value.element_size())
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements].to(device="cpu").contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())


class CPUMasterAdamW:
    """One optimizer over named dense parameter groups, with CPU FP32 state.

    Runtime parameters must already be selected for training and be BF16 or
    FP32. They remain the autograd leaves. Gradients are copied to CPU FP32,
    validated collectively before mutation, globally clipped, and applied to
    FP32 masters. Missing gradients stay ``None`` (including no weight decay).
    Present zero gradients retain ordinary AdamW momentum/decay semantics.
    CPU gradient buffers are released after every step; call ``zero_grad`` to
    release the runtime gradients at the next accumulation boundary.
    """

    def __init__(
        self,
        *,
        named_groups: Mapping[str, Mapping[str, nn.Parameter]],
        master_values: Mapping[str, torch.Tensor],
        learning_rates: Mapping[str, float],
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not named_groups or set(named_groups) != set(learning_rates):
            raise ValueError("Named groups and learning-rate groups must match and be nonempty")
        if any(not isinstance(group, str) or not group for group in named_groups):
            raise ValueError("Group names must be nonempty strings")
        self.max_grad_norm = _positive_finite(max_grad_norm, "max_grad_norm")
        decay = _positive_finite(weight_decay, "weight_decay", zero=True)
        epsilon = _positive_finite(eps, "eps")
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be two finite values in [0, 1)")
        self.groups: dict[str, tuple[str, ...]] = {}
        self.runtime_parameters: dict[str, nn.Parameter] = {}
        parameter_ids = set()
        for group in sorted(named_groups):
            parameters = named_groups[group]
            if not parameters or any(not isinstance(name, str) or not name for name in parameters):
                raise ValueError("Parameter groups must contain nonempty parameter names")
            self.groups[group] = tuple(sorted(parameters))
            for name in self.groups[group]:
                parameter = parameters[name]
                if name in self.runtime_parameters or id(parameter) in parameter_ids:
                    raise ValueError(f"Duplicate runtime parameter or name: {name}")
                if (
                    not isinstance(parameter, nn.Parameter)
                    or not parameter.requires_grad
                    or parameter.dtype not in (torch.float32, torch.bfloat16)
                    or parameter.layout != torch.strided
                    or not parameter.is_contiguous()
                    or parameter.numel() == 0
                ):
                    raise ValueError(f"{name}: expected a trainable contiguous BF16/FP32 Parameter")
                self.runtime_parameters[name] = parameter
                parameter_ids.add(id(parameter))
        if set(master_values) != set(self.runtime_parameters):
            raise ValueError("Master names must match the complete runtime parameter scope")
        for name, runtime in self.runtime_parameters.items():
            value = master_values[name]
            _master_check(value, runtime, name)
            if runtime.device.type == "cpu" and value.data_ptr() == runtime.data_ptr():
                raise ValueError(f"{name}: CPU master must not alias runtime storage")
            if not torch.equal(
                value.to(device=runtime.device, dtype=runtime.dtype), runtime.detach()
            ):
                raise ValueError(f"{name}: master cast does not exactly match runtime parameter")
        if len({value.data_ptr() for value in master_values.values()}) != len(master_values):
            raise ValueError("Master tensors must not share storage locations")
        self.master_parameters = {
            name: nn.Parameter(master_values[name].detach()) for name in self.runtime_parameters
        }
        rates = {
            group: _positive_finite(learning_rates[group], f"learning_rates[{group}]")
            for group in self.groups
        }
        self.settings = {
            "optimizer": "torch.optim.AdamW",
            "learning_rates": rates,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": decay,
            "betas": tuple(float(b) for b in betas),
            "eps": epsilon,
            "foreach": False,
            "fused": False,
            "amsgrad": False,
        }
        self.scope = {
            group: [
                {
                    "name": name,
                    "shape": list(self.runtime_parameters[name].shape),
                    "runtime_dtype": str(self.runtime_parameters[name].dtype),
                }
                for name in names
            ]
            for group, names in self.groups.items()
        }
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [self.master_parameters[name] for name in names],
                    "lr": rates[group],
                    "name": group,
                }
                for group, names in self.groups.items()
            ],
            betas=betas,
            eps=epsilon,
            weight_decay=decay,
            foreach=False,
            fused=False,
            amsgrad=False,
        )

    def zero_grad(self) -> None:
        for parameter in self.runtime_parameters.values():
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> dict[str, Any]:
        """Update after gradient accumulation; reject all bad gradients first."""
        self.optimizer.zero_grad(set_to_none=True)
        groups = {}
        all_norms = []
        try:
            for group, names in self.groups.items():
                detail = {
                    "parameter_tensors": len(names),
                    "parameter_scalars": sum(self.runtime_parameters[n].numel() for n in names),
                    "present_gradient_tensors": 0,
                    "missing_gradient_names": [],
                    "zero_gradient_names": [],
                    "nonzero_gradient_names": [],
                    "changed_runtime_tensors": 0,
                }
                norms = []
                for name in names:
                    gradient = self.runtime_parameters[name].grad
                    if gradient is None:
                        detail["missing_gradient_names"].append(name)
                        continue
                    if gradient.layout != torch.strided:
                        raise ValueError(f"{name}: sparse gradients are unsupported")
                    # copy=True also isolates CPU FP32 fixture/runtime gradients
                    # so clipping cannot mutate the caller's accumulation.
                    cpu_gradient = gradient.detach().to(
                        device="cpu", dtype=torch.float32, copy=True
                    )
                    if not torch.isfinite(cpu_gradient).all():
                        raise FloatingPointError(
                            f"{name}: nonfinite gradient; optimizer not updated"
                        )
                    norm = torch.linalg.vector_norm(cpu_gradient)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"{name}: nonfinite FP32 gradient norm")
                    self.master_parameters[name].grad = cpu_gradient
                    norms.append(norm)
                    all_norms.append(norm)
                    detail["present_gradient_tensors"] += 1
                    detail[
                        "nonzero_gradient_names" if norm.item() > 0 else "zero_gradient_names"
                    ].append(name)
                group_norm = (
                    torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0.0)
                )
                if not torch.isfinite(group_norm):
                    raise FloatingPointError(f"{group}: nonfinite FP32 group gradient norm")
                detail["gradient_norm_before_clip"] = float(group_norm)
                groups[group] = detail
            total_norm = (
                torch.linalg.vector_norm(torch.stack(all_norms)) if all_norms else torch.tensor(0.0)
            )
            if not torch.isfinite(total_norm):
                raise FloatingPointError(
                    "Nonfinite global FP32 gradient norm; optimizer not updated"
                )
            clip_scale = min(1.0, self.max_grad_norm / (float(total_norm) + 1e-6))
            for parameter in self.master_parameters.values():
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_scale)
            self.optimizer.step()
            for group, names in self.groups.items():
                for name in names:
                    master, runtime = self.master_parameters[name], self.runtime_parameters[name]
                    if master.grad is None:
                        continue
                    if not torch.isfinite(master).all():
                        raise FloatingPointError(
                            f"{name}: optimizer produced nonfinite master weights"
                        )
                    updated = master.to(device=runtime.device, dtype=runtime.dtype)
                    if not torch.isfinite(updated).all():
                        raise FloatingPointError(f"{name}: runtime cast produced nonfinite weights")
                    groups[group]["changed_runtime_tensors"] += int(
                        not torch.equal(runtime, updated)
                    )
                    runtime.copy_(updated)
            return {
                "gradient_norm_before_clip": float(total_norm),
                "clip_scale": clip_scale,
                "groups": groups,
            }
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def audit(self) -> dict[str, Any]:
        """Group hashes/counts without creating gradients or Adam moments."""
        groups = {}
        for group, names in self.groups.items():
            masters, runtime = hashlib.sha256(), hashlib.sha256()
            for name in names:
                _update_hash(masters, name, self.master_parameters[name])
                _update_hash(runtime, name, self.runtime_parameters[name])
            groups[group] = {
                "parameter_tensors": len(names),
                "parameter_scalars": sum(self.master_parameters[name].numel() for name in names),
                "master_sha256": masters.hexdigest(),
                "runtime_sha256": runtime.hexdigest(),
            }
        return {"schema_version": 1, "groups": groups}

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": deepcopy(self.scope),
            "settings": deepcopy(self.settings),
            "masters": {name: value.detach() for name, value in self.master_parameters.items()},
            "optimizer": self.optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("scope") != self.scope
            or state.get("settings") != self.settings
        ):
            raise ValueError("CPU-master checkpoint scope or optimizer settings differ")
        values = state.get("masters", {})
        if set(values) != set(self.master_parameters):
            raise ValueError("CPU-master checkpoint does not contain the complete master scope")
        for name, runtime in self.runtime_parameters.items():
            _master_check(values[name], runtime, name)
            if not torch.isfinite(values[name].to(dtype=runtime.dtype)).all():
                raise ValueError(
                    f"{name}: checkpoint master cannot be represented by runtime dtype"
                )
            if runtime.device.type == "cpu" and values[name].data_ptr() == runtime.data_ptr():
                raise ValueError(f"{name}: checkpoint master aliases runtime storage")
        if len({value.data_ptr() for value in values.values()}) != len(values):
            raise ValueError("Checkpoint masters must not share storage locations")
        saved_optimizer = state.get("optimizer")
        if not isinstance(saved_optimizer, dict) or set(saved_optimizer) != {
            "state",
            "param_groups",
        }:
            raise ValueError("Invalid CPU AdamW checkpoint")
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if saved_optimizer["param_groups"] != expected_groups:
            raise ValueError("CPU AdamW checkpoint parameter groups or settings differ")
        names = list(self.master_parameters)
        allowed_ids = {i for group in expected_groups for i in group["params"]}
        if not set(saved_optimizer["state"]).issubset(allowed_ids):
            raise ValueError("CPU AdamW checkpoint contains unknown parameter states")
        for identifier, entry in saved_optimizer["state"].items():
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Incomplete CPU AdamW parameter state")
            for key in ("exp_avg", "exp_avg_sq"):
                _master_check(
                    entry[key], self.runtime_parameters[names[identifier]], names[identifier]
                )
            step = entry["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.device.type != "cpu"
                or step.dtype != torch.float32
                or step.numel() != 1
                or not torch.isfinite(step).all()
                or float(step) < 0
                or float(step) != int(float(step))
            ):
                raise ValueError("Invalid CPU AdamW step counter")
            if (entry["exp_avg_sq"] < 0).any():
                raise ValueError("CPU AdamW squared moments cannot be negative")
        self.zero_grad()
        self.optimizer.load_state_dict(saved_optimizer)
        for name, master in self.master_parameters.items():
            master.data = values[name].detach()
            runtime = self.runtime_parameters[name]
            runtime.copy_(master.to(device=runtime.device, dtype=runtime.dtype))
