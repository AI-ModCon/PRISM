"""Graph node-feature / property decoder.

The inverse of ``GraphEncoder`` (GraphMAE2, ``src/encoders/graph.py``) for the
**AR-safe subset only**: node-feature / node-level property prediction on a
*given* graph (the GraphMAE2 reconstruction objective).

Out of scope by design: de-novo graph *structure* generation (variable node
count + edge set) is dominated by discrete diffusion (DiGress), a
non-autoregressive pattern that would break the single-forward, AR-safe
contract every decoder here keeps. This decoder never emits an edge set; it
only labels nodes.

Design note: the symmetric ideal would reuse GraphMAE2's reconstruction
decoder, but that GAT decoder operates in the graph encoder's own latent space and
torch_geometric is an optional dep (absent in CI); the PRISM forward path
exposes the backbone ``d_model`` hidden state instead. v1 is therefore a
**per-token regression/classification head** over the node-token hidden states
— single-forward and AR-safe. ``native_decoder`` is an optional hook for
delegating to a GraphMAE2-style GAT decoder once a latent bridge exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import OutputDecoder


class GraphDecoder(OutputDecoder):
    """Per-node head predicting node features (regression) or labels (CE).

    Reads node-token hidden states ``(B, N, d_model)`` and emits one prediction
    per node: ``(B, N, out_dim)``. ``task="regression"`` uses MSE (feature
    reconstruction); ``task="classification"`` uses per-node cross-entropy.
    """

    output_kind = "tensor"
    response_encoding = "tensor_b64"

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        task: str = "regression",
        native_decoder: nn.Module | None = None,
    ):
        super().__init__()
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {task!r}"
            )
        self.d_model = d_model
        self.out_dim = out_dim
        self.task = task
        self.loss_kind = "mse" if task == "regression" else "cross_entropy"
        self.native_decoder = native_decoder
        self.head = nn.Linear(d_model, out_dim)

    def _node_states(self, condition):
        indices = condition.native_context.get("node_indices")
        if indices is None:
            raise ValueError("Graph output needs explicit node_indices; fused/resampled tokens are not graph nodes")
        indices = torch.as_tensor(indices, device=condition.hidden_states.device)
        if indices.dtype != torch.long or indices.ndim != 2 or indices.shape[0] != condition.hidden_states.shape[0]:
            raise ValueError("node_indices must be a (B, N) int64 tensor")
        if indices.numel() == 0 or (indices < 0).any() or (indices >= condition.hidden_states.shape[1]).any():
            raise ValueError("node_indices must reference valid fused positions")
        if not condition.attention_mask.gather(1, indices).bool().all():
            raise ValueError("node_indices cannot reference padding")
        return condition.hidden_states.gather(1, indices[..., None].expand(-1, -1, condition.hidden_states.shape[-1]))

    def forward_condition(self, condition, targets=None, **kwargs):
        """Gather the node tokens from a typed condition, then run ``forward``.

        Args:
            condition: backbone states whose ``native_context["node_indices"]``
                is a ``(B, N)`` int64 tensor of non-padding positions naming
                the node tokens in the merged sequence.
            targets: per-node supervision, forwarded to ``forward``, or
                ``None`` for inference. Default: ``None``.
            **kwargs: extra keyword arguments forwarded to ``forward``.

        Returns:
            ``(prediction, loss)`` from ``forward``; ``loss`` is ``None`` when
            ``targets`` is ``None``.

        Raises:
            ValueError: if ``node_indices`` is absent, is not a valid ``(B, N)``
                int64 index tensor, or references padding.
        """
        return self.forward(self._node_states(condition), targets=targets, **kwargs)

    def generate_condition(self, condition, **kwargs):
        """Gather the node tokens from a typed condition, then run ``generate``.

        Args:
            condition: backbone states with ``native_context["node_indices"]``
                (see ``forward_condition``).
            **kwargs: extra keyword arguments forwarded to ``generate``.

        Returns:
            The per-node predictions ``(B, N, out_dim)``, gathered over the
            ``N`` positions named by ``node_indices`` rather than the full
            merged sequence.

        Raises:
            ValueError: if ``node_indices`` is absent or invalid.
        """
        return self.generate(self._node_states(condition), **kwargs)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-node predictions ``(B, N, out_dim)``.

        Expects node-token hidden states ``(B, N, d_model)``. A 2D
        ``(N, d_model)`` single-graph input is accepted and a batch dim added.
        """
        if self.native_decoder is not None:
            return self.native_decoder(hidden_states)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, N, d_model) or (N, d_model), "
                f"got shape {tuple(hidden_states.shape)}"
            )
        head_dtype = self.head.weight.dtype
        if hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(dtype=head_dtype)
        return self.head(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict per-node outputs and (optionally) compute the task loss.

        Args:
            hidden_states: node-token states ``(B, N, d_model)``, or
                ``(N, d_model)`` for a single graph.
            targets: for ``task="regression"``, the target features
                ``(B, N, out_dim)``; for ``task="classification"``, node labels
                ``(B, N)`` with ``-100`` marking ignored nodes. ``None`` runs
                inference. Default: ``None``.
            **kwargs: accepted and ignored.

        Returns:
            ``(pred, loss)`` where ``pred`` is ``(B, N, out_dim)`` and ``loss``
            is the MSE (regression) or per-node cross-entropy
            (classification), or ``None`` when ``targets`` is ``None``.

        Raises:
            RuntimeError: if ``targets`` does not match the expected shape for
                the configured task.
        """
        pred = self.predict(hidden_states)  # (B, N, out_dim)
        if targets is None:
            return pred, None

        if self.task == "regression":
            if targets.shape != pred.shape:
                raise RuntimeError(
                    f"graph regression target shape {tuple(targets.shape)} != "
                    f"prediction shape {tuple(pred.shape)} (B, N, out_dim)"
                )
            loss = F.mse_loss(pred, targets.to(dtype=pred.dtype))
        else:
            # classification: targets are node labels (B, N), ignore_index -100.
            if targets.shape != pred.shape[:-1]:
                raise RuntimeError(
                    f"graph classification target shape {tuple(targets.shape)} != "
                    f"node grid {tuple(pred.shape[:-1])} (B, N)"
                )
            loss = F.cross_entropy(
                pred.reshape(-1, self.out_dim),
                targets.reshape(-1).long(),
                ignore_index=-100,
            )
        return pred, loss
