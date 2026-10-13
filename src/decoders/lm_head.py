"""Text output decoder.

Wraps PRISM's existing language-modeling output path as an ``OutputDecoder``
so text becomes "just the default decoder." The logic here is a faithful
extraction of the loss computation previously inlined in
``UnifiedTransformer.forward`` (Path A, HF backbone) — see
``src/model.py`` history. It must stay bit-identical to that path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import OutputDecoder


class LMHeadDecoder(OutputDecoder):
    """Next-token cross-entropy over backbone logits.

    On an HF backbone the LM head lives inside the backbone, so this decoder
    does not own any parameters — it receives the backbone's ``logits`` and
    computes the shifted cross-entropy over supervised positions only
    (``-100`` labels are ignored). Under autocast, cross-entropy upcasts its
    input to FP32, so only the gathered supervised rows are upcast.
    """

    output_kind = "text"
    loss_kind = "cross_entropy"
    response_encoding = "tokens"

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits-loss for next-token prediction.

        Args:
            hidden_states: the backbone ``logits`` tensor
                ``(B, T, vocab_size)``. Named ``hidden_states`` for interface
                uniformity; for the HF-backbone path the backbone has already
                applied its LM head.
            targets: ``full_labels`` of shape ``(B, T)`` with ``-100`` at
                ignored positions.

        Returns:
            ``(logits, loss)`` — ``loss`` is ``None`` when ``targets`` is
            ``None``.
        """
        logits = hidden_states

        if targets is None:
            return logits, None

        # Shift logits and labels for next-token prediction.
        shift_labels = targets[..., 1:]
        # Gather only supervised rows: avoids a full (B, T, vocab) contiguous
        # copy and autocast's FP32 upcast of it, which OOMs on long batches.
        supervised = shift_labels != -100
        loss = F.cross_entropy(
            logits[..., :-1, :][supervised],
            shift_labels[supervised],
        )
        return logits, loss
