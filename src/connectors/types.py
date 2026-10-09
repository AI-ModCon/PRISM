"""Tensor contracts between a feature source, readout, and decoder bridge.

These types carry observations and their layout, never supervision targets.
Source positions refer to the original feature sequence, not output geometry.
"""

from dataclasses import dataclass, field

import torch


def _validate_sequence(tokens: torch.Tensor, mask: torch.Tensor) -> None:
    if tokens.ndim != 3 or min(tokens.shape) < 1:
        raise ValueError("tokens must have nonempty shape (B, L, D)")
    if not tokens.is_floating_point():
        raise ValueError("tokens must be floating point")
    if mask.shape != tokens.shape[:2] or mask.device != tokens.device:
        raise ValueError("attention_mask must match tokens' (B, L) and device")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must be binary")
    if not mask.bool().any(dim=1).all():
        raise ValueError("Every sequence must contain at least one valid token")


@dataclass
class BackboneFeatures:
    """The current final-layer feature bank; no implicit layer taps or detaching."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    modality_spans: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        _validate_sequence(self.hidden_states, self.attention_mask)


@dataclass
class ReadoutResult:
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    # Original sequence indices, or None for aggregates without a single source.
    # Padding has index -1. Query slots must never impersonate source positions.
    source_positions: torch.Tensor | None = None
    source_modality_spans: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        _validate_sequence(self.tokens, self.attention_mask)
        if self.source_positions is not None:
            positions = self.source_positions
            if (
                positions.shape != self.attention_mask.shape
                or positions.dtype != torch.long
                or positions.device != self.tokens.device
            ):
                raise ValueError("source_positions must be an int64 (B, L) tensor on token device")
            valid = self.attention_mask.bool()
            if (positions[valid] < 0).any() or (positions[~valid] != -1).any():
                raise ValueError("source_positions need valid indices and -1 at padding")


@dataclass
class DecoderContext(ReadoutResult):
    """Bridged conditioning, with a mask and layout for the generator to consume."""
