"""Image output through a learned connector and a pretrained OmniGen2 generator.

The connector is new PRISM capacity, not a pretrained image head.  ``reference``
mode bypasses it and retains the official native conditioner.  Optional image
dependencies and checkpoint weights are loaded only when a backend is used.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import OutputDecoder
from .types import DecoderCondition


class ImageDecoder(OutputDecoder):
    output_kind = "image"
    loss_kind = "flow_matching"
    response_encoding = "png"

    def __init__(
        self,
        d_model: int,
        backend: nn.Module | None = None,
        model_id: str = "OmniGen2/OmniGen2",
        revision: str | None = None,
        local_files_only: bool = True,
        conditioning_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if backend is None:
            from .omnigen2_backend import DEFAULT_REVISION, OmniGen2Backend

            backend = OmniGen2Backend(
                model_id=model_id,
                revision=revision or DEFAULT_REVISION,
                local_files_only=local_files_only,
                conditioning_dim=conditioning_dim or 2048,
            )
        if not isinstance(backend, nn.Module):
            raise TypeError("image backend must be an nn.Module")
        inferred_dim = getattr(backend, "conditioning_dim", None)
        conditioning_dim = conditioning_dim or inferred_dim
        if not isinstance(conditioning_dim, int) or conditioning_dim < 1:
            raise ValueError("conditioning_dim must be set or provided by the backend")
        if inferred_dim is not None and inferred_dim != conditioning_dim:
            raise ValueError("conditioning_dim does not match the backend")
        self.d_model = d_model
        self.conditioning_dim = conditioning_dim
        self.connector = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, conditioning_dim))
        self.backend = backend
        if getattr(self.backend, "train_diffusion", False):
            self.backend.train(self.training)
        else:
            self.backend.requires_grad_(False)
            self.backend.eval()

    def configure_training(
        self, *, train_diffusion: bool = False, gradient_checkpointing: bool = False
    ) -> ImageDecoder:
        """Train the connector and optionally the full pretrained diffusion model.

        Default construction and ``configure_training()`` retain connector-only
        behavior. No pretrained weights are allocated here. Configure after any
        model-wide freezing and before building the optimizer.
        """
        configure = getattr(self.backend, "configure_training", None)
        if not callable(configure):
            raise TypeError("image backend does not expose explicit training configuration")
        configure(train_diffusion=train_diffusion, gradient_checkpointing=gradient_checkpointing)
        self.connector.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> ImageDecoder:
        super().train(mode)
        # Only explicitly opted-in diffusion weights follow the parent's mode.
        self.backend.train(mode if getattr(self.backend, "train_diffusion", False) else False)
        return self

    def connect(self, condition: DecoderCondition) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve valid-token order and right-pad for OmniGen2's prefix mask API."""
        hidden = condition.hidden_states
        mask = condition.attention_mask
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("image hidden_states must have shape (B, L, d_model)")
        if mask.shape != hidden.shape[:2] or mask.device != hidden.device:
            raise ValueError("image attention_mask must match hidden states' (B, L) and device")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("image attention_mask must contain only 0 or 1")
        valid = mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError("image conditioning cannot contain an all-masked row")
        # Mask before LayerNorm so padded NaNs cannot contaminate parameter grads.
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        weight = self.connector[1].weight
        connected = self.connector(hidden.to(device=weight.device, dtype=weight.dtype))
        valid = valid.to(connected.device)
        order = torch.argsort(valid.to(torch.int64), dim=1, descending=True, stable=True)
        length = int(valid.sum(dim=1).max().item())
        connected = connected.gather(1, order.unsqueeze(-1).expand_as(connected))[:, :length]
        mask = torch.arange(length, device=connected.device)[None, :] < valid.sum(1)[:, None]
        return connected.masked_fill(~mask.unsqueeze(-1), 0), mask

    @staticmethod
    def _options(condition: DecoderCondition, kwargs: dict) -> tuple[str, dict]:
        options = dict(condition.output_spec)
        options.update(kwargs)
        mode = options.pop("mode", "prism")
        if mode not in ("prism", "reference"):
            raise ValueError("image mode must be 'prism' or 'reference'")
        return mode, options

    @torch.no_grad()
    def generate_condition(self, condition: DecoderCondition, **kwargs: Any) -> Any:
        mode, options = self._options(condition, kwargs)
        if mode == "reference":
            return self.backend.generate_reference(dict(condition.native_context), **options)
        embeds, mask = self.connect(condition)
        return self.backend.generate_conditioned(
            embeds, mask, native_context=dict(condition.native_context), **options
        )

    def forward_condition(
        self, condition: DecoderCondition, targets: torch.Tensor | None = None, **kwargs: Any
    ) -> tuple[Any, torch.Tensor | None]:
        if targets is None:
            return self.generate_condition(condition, **kwargs), None
        mode, options = self._options(condition, kwargs)
        if mode != "prism":
            raise ValueError("image connector training requires mode='prism'")
        embeds, mask = self.connect(condition)
        # Autograd trains the connector through the generator, and its weights on opt-in.
        return self.backend.training_step(
            embeds, mask, targets, native_context=dict(condition.native_context), **options
        )

    def forward(self, hidden_states: torch.Tensor, targets=None, **kwargs: Any):
        mask = kwargs.pop("attention_mask", None)
        if mask is None:
            mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool
            )
        condition = DecoderCondition(
            hidden_states=hidden_states,
            attention_mask=mask,
            native_context=kwargs.pop("native_context", {}),
            output_spec=kwargs.pop("output_spec", {}),
        )
        return self.forward_condition(condition, targets=targets, **kwargs)

    @torch.no_grad()
    def generate(self, hidden_states: torch.Tensor, **kwargs: Any):
        return self.forward(hidden_states, **kwargs)[0]
