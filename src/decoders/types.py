"""Explicit conditioning and result contracts for multimodal output decoding."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DecoderCondition:
    """Conditioning handed to an output decoder.

    The shared input contract for every ``OutputDecoder``: the backbone states
    to decode from, their validity mask, and the side information a decoder
    needs to place or render its output. Validated on construction.

    Attributes:
        hidden_states: backbone states to decode, ``(B, L, D)``.
        attention_mask: binary ``(B, L)`` validity mask over ``hidden_states``.
        modality_spans: per modality, per example, half-open spans in the
            merged sequence. Default: ``{}``.
        native_context: modality-native side inputs a decoder consumes
            directly, such as ``node_indices`` for ``GraphDecoder`` or the
            prompt and reference images for ``ImageDecoder``. Default: ``{}``.
        output_spec: requested output options (for images, ``mode`` plus the
            backend's generation options); per-call keyword arguments override
            these. Default: ``{}``.
        provenance: free-form record of where the condition came from, carried
            through to the decoder context and result. Default: ``{}``.

    Raises:
        ValueError: if ``hidden_states`` is not 3-D, if ``attention_mask`` does
            not have shape ``(B, L)``, if the mask is not binary, or if any
            example has no valid token.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    # Per modality, per example, half-open spans in the merged sequence.
    modality_spans: dict = field(default_factory=dict)
    native_context: dict = field(default_factory=dict)
    output_spec: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (B, L, D)")
        if self.attention_mask.shape != self.hidden_states.shape[:2]:
            raise ValueError("attention_mask must have shape (B, L)")
        if not torch.all((self.attention_mask == 0) | (self.attention_mask == 1)):
            raise ValueError("attention_mask must be binary")
        if not self.attention_mask.bool().any(dim=1).all():
            raise ValueError("Every condition must contain at least one valid token")


@dataclass
class DecoderResult:
    """Outputs collected from one multi-decoder routing pass.

    Attributes:
        predictions: per-decoder prediction, keyed by decoder name. Default: ``{}``.
        losses: per-decoder scalar loss, keyed by decoder name; only decoders
            that were given targets appear here. Default: ``{}``.
        loss: the total the caller accumulates from ``losses`` (each weighted
            by its configured decoder loss weight), or ``None`` when no
            decoder was supervised. Default: ``None``.
        provenance: free-form record carried over from the condition.
            Default: ``{}``.
    """

    predictions: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    loss: torch.Tensor | None = None
    provenance: dict = field(default_factory=dict)


def masked_pool(hidden_states, attention_mask, pool):
    """Pool valid positions only, including left padding and noncontiguous masks."""
    if attention_mask is None or hidden_states.ndim == 2:
        if hidden_states.ndim == 2:
            return hidden_states
        return hidden_states[:, -1] if pool == "last" else hidden_states.mean(1)
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if mask.shape != hidden_states.shape[:2] or not mask.any(1).all():
        raise ValueError("Pooling requires a nonempty (B, L) attention mask")
    if pool == "last":
        indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = indices.masked_fill(~mask, -1).max(1).values
        return hidden_states[torch.arange(mask.shape[0], device=mask.device), last]
    return hidden_states.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


def move_tensors(value, device):
    """Move nested tensor batches while preserving native objects such as PIL images."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_tensors(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tensors(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(v, device) for v in value)
    return value
