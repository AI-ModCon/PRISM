"""Regression output decoder.

Generalizes the VLA ``action_head`` (previously a hard-coded
``nn.Sequential`` + inline MSE in ``UnifiedTransformer``) into a reusable
``OutputDecoder``. It owns the MLP head and the canonical loss so callers
share one implementation; the VLA forward path sources its head and loss
from here, keeping behavior bit-identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OutputDecoder


class RegressionDecoder(OutputDecoder):
    """MLP head predicting a continuous vector, trained with MSE.

    Mirrors the original VLA head exactly:
    ``Linear(in, in) -> ReLU -> Linear(in, out)``.
    """

    output_kind = "tensor"
    loss_kind = "mse"
    response_encoding = "tensor_b64"

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Accelerate resumes model state directly, bypassing PRISM's native
        # weights-only loader. Accept the former Sequential keys here so all
        # recursive load_state_dict callers retain the trained action head.
        for name in self.head.state_dict():
            legacy_key = prefix + name
            current_key = prefix + "head." + name
            if legacy_key not in state_dict:
                continue
            if current_key in state_dict:
                error_msgs.append(
                    f"Ambiguous regression checkpoint contains both {legacy_key!r} "
                    f"and {current_key!r}"
                )
                continue
            state_dict[current_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Run the head on already-pooled features ``(B, input_dim)``."""
        return self.head(features)

    @staticmethod
    def loss_terms(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical regression loss.

        Returns ``(loss, per_dim_loss)`` computed the same two-stage way as
        the original VLA path (``mean(dim=0)`` then ``mean()``), so any
        caller — generic ``forward`` or the VLA orchestration — gets
        identical numbers.
        """
        per_dim = F.mse_loss(prediction, target, reduction="none").mean(dim=0)
        loss = per_dim.mean()
        return loss, per_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict from pooled features and (optionally) compute loss.

        Args:
            hidden_states: pooled features ``(B, input_dim)``.
            targets: regression target ``(B, output_dim)``.
        """
        prediction = self.predict(hidden_states)
        if targets is None:
            return prediction, None
        target = targets.to(device=prediction.device, dtype=prediction.dtype)
        if target.shape != prediction.shape:
            raise RuntimeError(
                f"Regression target shape {tuple(target.shape)} != "
                f"prediction shape {tuple(prediction.shape)}"
            )
        loss, _ = self.loss_terms(prediction, target)
        return prediction, loss
