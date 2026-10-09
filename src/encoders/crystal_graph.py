"""Periodic crystal-graph encoder for PRISM materials models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ModalityEncoder


class PeriodicCrystalMessageLayer(nn.Module):
    """Distance-aware message passing over a periodic crystal radius graph."""

    def __init__(self, hidden_dim: int, radial_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, radial: torch.Tensor
    ) -> torch.Tensor:
        """Pass one round of distance-aware messages over the radius graph.

        Args:
            nodes: Atom features, ``(N, hidden_dim)``, concatenated across the
                batch.
            edge_index: Edge endpoints, ``(2, E)``, as ``(source, target)`` rows
                into ``nodes``.
            radial: Radial-basis edge features, ``(E, radial_dim)``.

        Returns:
            Updated atom features, ``(N, hidden_dim)``: a degree-normalized sum
            of incoming messages, applied as a layer-normalized residual.
        """
        source, target = edge_index
        messages = self.message(torch.cat((nodes[target], radial), dim=-1))
        aggregated = nodes.new_zeros(nodes.shape)
        aggregated.index_add_(0, source, messages)
        degree = torch.bincount(source, minlength=nodes.shape[0]).clamp_min(1)
        aggregated = aggregated / degree.to(nodes.dtype).unsqueeze(1)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregated), dim=-1)))


class CrystalGraphTokenEncoder(ModalityEncoder):
    """Encode packed periodic graphs into a fixed number of PRISM tokens.

    Inputs use the packed graph representation emitted by the materials
    collator: atomic numbers and edges are concatenated across the batch, and
    ``graph_batch`` maps every atom to its graph. Learned queries attend to the
    variable-length atom representations to produce fixed-count modality
    tokens suitable for a PRISM projector.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        cutoff: float = 5.0,
        radial_dim: int = 32,
        num_tokens: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if radial_dim < 1:
            raise ValueError("radial_dim must be positive")
        super().__init__(hidden_dim)
        self.cutoff = cutoff
        self.radial_dim = radial_dim
        self.num_tokens = num_tokens
        self.atom_embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            PeriodicCrystalMessageLayer(hidden_dim, radial_dim, dropout)
            for _ in range(num_layers)
        )
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def tokens_per_instance(self) -> int:
        """Return the fixed number of tokens emitted per crystal."""
        return self.num_tokens

    def _radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(0.0, self.cutoff, self.radial_dim, device=distances.device)
        width = self.cutoff / max(1, self.radial_dim - 1)
        gaussian = torch.exp(-0.5 * ((distances[:, None] - centers) / width) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        return gaussian * envelope.clamp_min(0).unsqueeze(1)

    def forward(self, inputs: dict[str, torch.Tensor | int]) -> torch.Tensor:
        """Encode a packed batch of periodic crystals into modality tokens.

        Args:
            inputs: Packed graph batch with ``"atomic_numbers"`` ``(N,)``,
                ``"edge_index"`` ``(2, E)``, ``"edge_distance"`` ``(E,)`` and
                ``"graph_batch"`` ``(N,)`` mapping each atom to its graph.
                ``"batch_size"`` is optional; when absent or ``0`` it is
                inferred from ``graph_batch``.

        Returns:
            Crystal tokens of shape ``(batch_size, num_tokens, hidden_dim)``.

        Raises:
            TypeError: If any of the four tensor fields is not a
                ``torch.Tensor``.
            ValueError: If the batch has no atoms, or if any graph in the
                declared batch contributes none.
        """
        atomic_numbers = inputs["atomic_numbers"]
        edge_index = inputs["edge_index"]
        edge_distance = inputs["edge_distance"]
        graph_batch = inputs["graph_batch"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (atomic_numbers, edge_index, edge_distance, graph_batch)
        ):
            raise TypeError("Crystal graph tensor fields must be torch.Tensor instances")
        if atomic_numbers.numel() == 0 or graph_batch.numel() == 0:
            raise ValueError("Each crystal batch must contain at least one atom")

        explicit_batch_size = inputs.get("batch_size", 0)
        batch_size = int(explicit_batch_size)
        if batch_size == 0:
            batch_size = int(graph_batch.max().item()) + 1

        nodes = self.atom_embedding(atomic_numbers)
        radial = self._radial_basis(edge_distance)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, radial)

        counts = torch.bincount(graph_batch, minlength=batch_size)
        if torch.any(counts == 0):
            raise ValueError("Every graph in the declared batch must contain at least one atom")
        max_nodes = int(counts.max().item())
        padded = nodes.new_zeros((batch_size, max_nodes, nodes.shape[-1]))
        padding_mask = torch.ones(
            (batch_size, max_nodes), dtype=torch.bool, device=nodes.device
        )
        for graph_index in range(batch_size):
            graph_nodes = nodes[graph_batch == graph_index]
            padded[graph_index, : graph_nodes.shape[0]] = graph_nodes
            padding_mask[graph_index, : graph_nodes.shape[0]] = False

        normalized_nodes = self.node_norm(padded)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        graph_tokens, _ = self.cross_attention(
            queries,
            normalized_nodes,
            normalized_nodes,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_norm(queries + graph_tokens)
