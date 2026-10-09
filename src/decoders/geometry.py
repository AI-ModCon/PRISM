"""Geometry / spatio-temporal field decoder.

The inverse of ``GeometryEncoder`` (Walrus, ``src/encoders/geometry.py``): the
encoder turns a point cloud / physical field into backbone tokens; this decoder
reads backbone hidden states and emits a field tensor ``(B, N, C)`` (N spatial
points/cells, C channels).

Design note: the symmetric ideal would reuse Walrus's native transposed-hMLP
decoder, but that decoder operates in Walrus's *own* latent space and Walrus is an optional dep
(absent in CI); the PRISM forward path instead exposes the backbone's
``d_model`` hidden state, and the input projector (PerceiverResampler) already
compresses geometry to a small set of latents, so per-point correspondence is
not preserved into the backbone. v1 is therefore a **direct field-regression
head** on pooled hidden states — single-forward and AR-safe, like
``TimeSeriesDecoder``. ``native_decoder`` is an optional hook for delegating to
a Walrus-style decoder once a latent bridge exists; it is not required and
defaults off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OutputDecoder
from .types import masked_pool


class GeometryDecoder(OutputDecoder):
    """Direct field-regression head.

    Reads a pooled backbone hidden state and projects it to a field of shape
    ``(B, num_points, num_channels)``. Trained with MSE on fixed-correspondence
    fields (the Walrus rollout setting, where grid positions are fixed).
    """

    output_kind = "tensor"
    loss_kind = "mse"
    response_encoding = "tensor_b64"

    def __init__(
        self,
        d_model: int,
        num_points: int,
        num_channels: int,
        pool: str = "mean",
        native_decoder: nn.Module | None = None,
    ):
        super().__init__()
        if num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {num_points}")
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")

        self.d_model = d_model
        self.num_points = num_points
        self.num_channels = num_channels
        self.pool = pool
        # Optional Walrus-style decoder; when set, predict() delegates to it.
        self.native_decoder = native_decoder
        self.head = nn.Linear(d_model, num_points * num_channels)

    def _pool_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            return hidden_states
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, T, d_model) or (B, d_model), "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if self.pool == "last":
            return hidden_states[:, -1, :]
        return hidden_states.mean(dim=1)

    def predict(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Field forecast ``(B, num_points, num_channels)``."""
        if self.native_decoder is not None:
            # Extension point: delegate to a Walrus-style decoder operating in
            # its own latent space. Caller is responsible for the latent bridge.
            if attention_mask is not None:
                return self.native_decoder(hidden_states, attention_mask=attention_mask)
            return self.native_decoder(hidden_states)
        pooled = self._pool_hidden(hidden_states) if attention_mask is None else masked_pool(hidden_states, attention_mask, self.pool)
        head_dtype = self.head.weight.dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)
        flat = self.head(pooled)
        return flat.view(pooled.shape[0], self.num_points, self.num_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict the field and (optionally) compute the MSE loss.

        Args:
            hidden_states: backbone states ``(B, T, d_model)``, or an
                already-pooled ``(B, d_model)`` vector.
            targets: ground-truth field ``(B, num_points, num_channels)``, or
                ``None`` for inference. Default: ``None``.
            **kwargs: ``attention_mask`` is read to pool only valid positions
                (or forwarded to ``native_decoder`` when one is configured);
                other keys are ignored.

        Returns:
            ``(pred, loss)`` where ``pred`` is the field
            ``(B, num_points, num_channels)`` and ``loss`` is the mean squared
            error against ``targets``, or ``None`` when ``targets`` is ``None``.

        Raises:
            RuntimeError: if ``targets`` does not have the predicted shape.
        """
        pred = self.predict(hidden_states, kwargs.get("attention_mask"))
        if targets is None:
            return pred, None
        if targets.shape != pred.shape:
            raise RuntimeError(
                f"geometry target shape {tuple(targets.shape)} != "
                f"forecast shape {tuple(pred.shape)} (B, N, C)"
            )
        target = targets.to(dtype=pred.dtype)
        loss = F.mse_loss(pred, target)
        return pred, loss
