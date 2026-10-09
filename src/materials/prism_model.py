"""PRISM-style text + crystal-graph regression.

This module turns a crystal into modality tokens, prepends them to the
language-model tokens, and lets the PRISM LLM backbone perform multimodal
fusion before scalar property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.crystal_graph import CrystalGraphTokenEncoder
from src.modules import ModalityProjector


class PRISMMaterialRegressor(nn.Module):
    """Scalar property head over PRISM-fused Qwen text and crystal tokens."""

    def __init__(
        self,
        prism: nn.Module,
        graph_hidden_dim: int = 128,
        graph_layers: int = 3,
        cutoff: float = 5.0,
        radial_dim: int = 32,
        graph_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if getattr(prism, "backbone", None) is None:
            raise ValueError("Materials PRISM regression requires an HF LLM backbone")
        self.prism = prism
        backbone_dim = int(prism.backbone_dim)
        # Register these inside PRISM's normal modality containers so parameter
        # naming, freezing, and checkpoint inspection follow the core framework.
        self.prism.encoders["graph"] = CrystalGraphTokenEncoder(
            hidden_dim=graph_hidden_dim,
            num_layers=graph_layers,
            cutoff=cutoff,
            radial_dim=radial_dim,
            num_tokens=graph_tokens,
            dropout=dropout,
        )
        self.prism.projectors["graph"] = ModalityProjector(
            graph_hidden_dim, backbone_dim, norm_mode="layernorm"
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen pretrained backbone should remain deterministic and avoid
        # dropout even while the graph adapter and regression head train.
        if not any(parameter.requires_grad for parameter in self.prism.backbone.parameters()):
            self.prism.backbone.eval()
        return self

    def _decoder(self) -> nn.Module:
        """Return the hidden-state model without the large vocabulary head."""
        backbone = self.prism.backbone
        decoder = getattr(backbone, "model", None)
        if decoder is not None:
            return decoder
        get_decoder = getattr(backbone, "get_decoder", None)
        if get_decoder is not None:
            return get_decoder()
        raise TypeError(
            f"{type(backbone).__name__} does not expose a decoder/base model for regression"
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mode: str = "joint",
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mode not in {"joint", "text", "graph"}:
            raise ValueError("mode must be one of: joint, text, graph")

        embeddings = []
        masks = []
        batch_size = batch["text_ids"].shape[0]
        if mode != "text":
            graph_inputs = {
                "atomic_numbers": batch["atomic_numbers"],
                "edge_index": batch["edge_index"],
                "edge_distance": batch["edge_distance"],
                "graph_batch": batch["graph_batch"],
                "batch_size": batch_size,
            }
            graph = self.prism.encoders["graph"](graph_inputs)
            graph = self.prism.projectors["graph"](graph)
            embeddings.append(graph)
            masks.append(torch.ones(graph.shape[:2], dtype=torch.long, device=graph.device))

        if mode != "graph":
            text = self.prism.backbone.get_input_embeddings()(batch["text_ids"])
            embeddings.append(text)
            masks.append(batch["text_mask"].to(dtype=torch.long, device=text.device))

        backbone = self.prism.backbone
        target_device = next(backbone.parameters()).device
        target_dtype = next(backbone.parameters()).dtype
        embeddings = [value.to(device=target_device, dtype=target_dtype) for value in embeddings]
        attention_mask = torch.cat(
            [value.to(device=target_device) for value in masks], dim=1
        )
        fused = torch.cat(embeddings, dim=1)
        outputs = self._decoder()(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        final_index = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(batch_size, device=hidden.device)
        pooled = hidden[batch_index, final_index]
        predictions = self.regression_head(pooled.float()).squeeze(-1)
        loss = None if targets is None else F.smooth_l1_loss(predictions, targets.float())
        return predictions, loss
