"""Final-state readouts and bridges for decoder conditioning."""

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from .base import ConditioningBridge, Readout
from .types import BackboneFeatures, DecoderContext, ReadoutResult


def _config(value, allowed: set[str], name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {name} configuration fields: {sorted(extra)}")
    return dict(value)


def _dimension(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class FinalStateReadout(Readout):
    """Read all valid final states, retaining padding until backend packing."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = _dimension(d_model, "d_model")

    def forward(self, features: BackboneFeatures) -> ReadoutResult:
        hidden, mask = features.hidden_states, features.attention_mask
        if hidden.shape[-1] != self.d_model:
            raise ValueError("readout hidden_states must have shape (B, L, d_model)")
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[:2])
        return ReadoutResult(
            tokens=hidden,
            attention_mask=mask,
            source_positions=positions.masked_fill(~mask.bool(), -1),
            source_modality_spans=dict(features.modality_spans),
            provenance=dict(features.provenance),
        )


class PooledStateReadout(Readout):
    """Reduce valid final states to one token without introducing parameters.

    Last-token selection retains its original source position. A mean is an
    aggregate rather than a source token, so it deliberately has no position map.
    Modality spans retain the original sequence coordinate system in both cases.
    """

    def __init__(self, d_model: int, pool: str = "last"):
        super().__init__()
        self.d_model = _dimension(d_model, "d_model")
        if pool not in ("last", "mean"):
            raise ValueError("pool must be 'last' or 'mean'")
        self.pool = pool

    def forward(self, features: BackboneFeatures) -> ReadoutResult:
        hidden, mask = features.hidden_states, features.attention_mask
        if hidden.shape[-1] != self.d_model:
            raise ValueError("readout hidden_states must have shape (B, L, d_model)")
        valid = mask.bool()
        if self.pool == "last":
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            positions = positions.expand_as(valid).masked_fill(~valid, -1)
            positions = positions.max(dim=1, keepdim=True).values
            pooled = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, self.d_model))
        else:
            positions = None
            sanitized = hidden.masked_fill(~valid.unsqueeze(-1), 0)
            pooled = sanitized.sum(dim=1, keepdim=True)
            # Keep the count integral: casting a count such as 257 to BF16
            # rounds it to 256 and changes the established masked-pool math.
            pooled = pooled / valid.sum(dim=1, keepdim=True).unsqueeze(-1)
        return ReadoutResult(
            tokens=pooled,
            attention_mask=torch.ones(hidden.shape[0], 1, dtype=torch.bool, device=hidden.device),
            source_positions=positions,
            source_modality_spans=dict(features.modality_spans),
            provenance=dict(features.provenance),
        )


class IdentityBridge(nn.Identity, ConditioningBridge):
    """Retain the readout width and dtype with no trainable state.

    Tensor ``forward`` is an ordinary identity. Typed ``connect`` additionally
    sanitizes padding so non-finite padded states cannot reach a generator.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = _dimension(input_dim, "input_dim")
        self.output_dim = _dimension(output_dim, "output_dim")
        if self.input_dim != self.output_dim:
            raise ValueError("identity bridge requires matching input_dim and output_dim")

    def connect(self, readout: ReadoutResult) -> DecoderContext:
        hidden, mask = readout.tokens, readout.attention_mask
        if hidden.shape[-1] != self.input_dim:
            raise ValueError("bridge input width does not match readout tokens")
        return DecoderContext(
            tokens=hidden.masked_fill(~mask.bool().unsqueeze(-1), 0),
            attention_mask=mask,
            source_positions=readout.source_positions,
            source_modality_spans=dict(readout.source_modality_spans),
            provenance=dict(readout.provenance),
        )


class LayerNormLinearBridge(nn.Sequential, ConditioningBridge):
    """Keep tensor-forward and ``0/1`` state keys used by existing checkpoints.

    ``connect`` is the typed API. Ordinary tensor ``forward`` remains available
    to the existing feature-alignment tools; no extra parameters are introduced.
    """

    def __init__(self, input_dim: int, output_dim: int):
        input_dim = _dimension(input_dim, "input_dim")
        output_dim = _dimension(output_dim, "output_dim")
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim

    def connect(self, readout: ReadoutResult) -> DecoderContext:
        hidden, mask = readout.tokens, readout.attention_mask
        if hidden.shape[-1] != self.input_dim:
            raise ValueError("bridge input width does not match readout tokens")
        # Preserve the old execution order, including masking padded NaNs
        # before LayerNorm and casting to the connector's master dtype.
        hidden = hidden.masked_fill(~mask.bool().unsqueeze(-1), 0)
        # ``__init__`` builds index 1 as the nn.Linear; nn.Sequential.__getitem__
        # widens that to Module, which hides the concrete Tensor weight.
        weight = cast(nn.Linear, self[1]).weight
        connected = self(hidden.to(device=weight.device, dtype=weight.dtype))
        positions = readout.source_positions
        return DecoderContext(
            tokens=connected,
            attention_mask=mask.to(connected.device),
            source_positions=None if positions is None else positions.to(connected.device),
            source_modality_spans=dict(readout.source_modality_spans),
            provenance=dict(readout.provenance),
        )


def build_readout(config: Mapping | None, *, d_model: int) -> Readout:
    config = _config(config, {"type", "layers", "positions", "pool"}, "readout")
    kind = config.get("type", "select")
    if kind not in ("select", "pool"):
        raise ValueError("Unsupported readout type; expected 'select' or 'pool'")
    if config.get("layers", "final") != "final":
        raise ValueError("Only final-layer readout is implemented")
    if kind == "pool":
        config = _config(config, {"type", "layers", "pool"}, "pooled readout")
        return PooledStateReadout(d_model, pool=config.get("pool", "last"))
    config = _config(config, {"type", "layers", "positions"}, "select readout")
    if config.get("positions", "all_valid") != "all_valid":
        raise ValueError("Only all_valid positions are implemented")
    return FinalStateReadout(d_model)


def build_bridge(config: Mapping | None, *, input_dim: int, output_dim: int) -> ConditioningBridge:
    config = _config(config, {"type", "output_dim"}, "bridge")
    kind = config.get("type", "layernorm_linear")
    if kind not in ("layernorm_linear", "identity"):
        raise ValueError("Unsupported bridge type; expected 'layernorm_linear' or 'identity'")
    selected_dim = _dimension(config.get("output_dim", output_dim), "bridge output_dim")
    if selected_dim != output_dim:
        raise ValueError("bridge output_dim does not match generator conditioning_dim")
    if kind == "identity":
        return IdentityBridge(input_dim, output_dim)
    return LayerNormLinearBridge(input_dim, output_dim)


def pack_right_padded(context: DecoderContext) -> DecoderContext:
    """Pack valid tokens in stable order for the OmniGen2 prefix-mask contract.

    Source spans remain in the original source coordinate system; only their
    explicit position map follows the packed tokens. This does not change
    conditioning length except for removing padding.
    """
    connected = context.tokens
    valid = context.attention_mask.bool()
    order = torch.argsort(valid.to(torch.int64), dim=1, descending=True, stable=True)
    length = int(valid.sum(dim=1).max().item())
    connected = connected.gather(1, order.unsqueeze(-1).expand_as(connected))[:, :length]
    mask = torch.arange(length, device=connected.device)[None, :] < valid.sum(1)[:, None]
    positions = context.source_positions
    if positions is not None:
        positions = positions.gather(1, order)[:, :length].masked_fill(~mask, -1)
    return DecoderContext(
        tokens=connected.masked_fill(~mask.unsqueeze(-1), 0),
        attention_mask=mask,
        source_positions=positions,
        source_modality_spans=dict(context.source_modality_spans),
        provenance=dict(context.provenance),
    )
