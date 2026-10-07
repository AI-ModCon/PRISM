"""Output-decoder abstraction.

This is the symmetric, output-side counterpart to ``ModalityEncoder``
(``src/encoders/base.py``). An ``OutputDecoder`` reads backbone hidden
states (or, for text, the backbone logits) and produces a modality-native
output plus its training loss.

The input side has a registry of encoders that turn raw inputs into
``(B, T, d_model)`` embeddings; this module begins the mirror image — a
registry of decoders that turn hidden states back into a concrete output
(tokens, a tensor forecast, an image, ...).

Every decoder is **single-forward and AR-safe**: one forward produces the whole
output, so the autoregressive serving path for text is preserved. Anything
requiring iterative or non-autoregressive decoding (discrete diffusion, de-novo
graph synthesis) is deliberately out of scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from .types import DecoderCondition


class OutputDecoder(nn.Module, ABC):
    """Base class for all output decoders.

    Subclasses declare three descriptive attributes so callers (training
    loop, serving layer) can route generically without hard-coding modality
    names:

    - ``output_kind``: one of ``"text" | "tensor" | "image" | "diffusion"``.
      What kind of thing ``generate`` returns.
    - ``loss_kind``: human-readable loss family (e.g. ``"cross_entropy"``,
      ``"mse"``, ``"nll"``). Diagnostic only.
    - ``response_encoding``: how the serving layer should encode the output
      over the wire (e.g. ``"tokens"``, ``"tensor_b64"``, ``"png"``).
    """

    output_kind: str = "tensor"
    loss_kind: str = "mse"
    response_encoding: str = "tensor_b64"

    def forward_condition(self, condition: DecoderCondition, targets=None, **kwargs):
        """Run the decoder from a typed ``DecoderCondition``.

        Default implementation delegates to ``forward``, forwarding only the
        condition's ``hidden_states`` and ``attention_mask``. Decoders that
        need any of the remaining fields (``modality_spans``,
        ``native_context``, ``output_spec``, ``provenance``) override this.

        Args:
            condition: shared conditioning contract carrying the backbone
                hidden states, their attention mask, and per-modality context.
            targets: optional supervision target, forwarded to ``forward``.
                When ``None`` the decoder runs in inference mode. Default: ``None``.
            **kwargs: extra decoder-specific keyword arguments forwarded to
                ``forward``.

        Returns:
            ``(prediction, loss)`` as returned by ``forward``. ``loss`` is
            ``None`` when ``targets`` is ``None``.
        """
        return self.forward(
            condition.hidden_states, targets=targets,
            attention_mask=condition.attention_mask, **kwargs,
        )

    def generate_condition(self, condition: DecoderCondition, **kwargs):
        """Produce a decoded output from a typed ``DecoderCondition``.

        Default implementation delegates to ``generate``, forwarding only the
        condition's ``hidden_states`` and ``attention_mask``.

        Args:
            condition: shared conditioning contract (see ``forward_condition``).
            **kwargs: extra decoder-specific keyword arguments forwarded to
                ``generate``.

        Returns:
            The modality-native output of ``generate``, whose kind is declared
            by ``output_kind``.
        """
        return self.generate(
            condition.hidden_states, attention_mask=condition.attention_mask, **kwargs,
        )

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the decoder.

        Args:
            hidden_states: backbone output to decode. For most decoders this
                is the last-layer hidden state ``(B, T, d_model)``; the text
                decoder on an HF backbone may instead receive precomputed
                logits (see ``LMHeadDecoder``).
            targets: optional supervision target. When ``None`` the decoder
                runs in inference mode and returns ``loss=None``.

        Returns:
            ``(prediction, loss)``. ``loss`` is ``None`` when ``targets`` is
            ``None``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, hidden_states: torch.Tensor, **kwargs: Any) -> Any:
        """Produce a final, decoded output for serving.

        Default implementation returns the prediction from ``forward`` with
        no target. Decoders that need a sampling/denoising loop (or a
        detokenizer step) override this.
        """
        prediction, _ = self.forward(hidden_states, targets=None, **kwargs)
        return prediction
