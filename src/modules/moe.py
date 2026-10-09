"""Feed-forward experts and the sparse Mixture-of-Experts layer.

Provides two dense MLP variants — ``SwiGLUMLP`` and ``ReLUSquaredMLP`` — and
``MoELayer``, which routes each token to its top-k experts and returns a
load-balancing auxiliary loss alongside the mixed output. ``src/model.py``
uses ``MoELayer`` as the feed-forward sub-layer of every ``TransformerBlock``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward expert.

    Computes ``w2(silu(w1(x)) * w3(x))`` and then dropout, the gated-linear
    MLP used by LLaMA-family models. All three projections are bias-free.

    Args:
        d_model: Width of the input and output features.
        hidden_dim: Width of the gate (``w1``) and up (``w3``) projections.
        dropout: Dropout probability applied to the output. Default: 0.1.

    Shape:
        - Input: (B, T, d_model)
        - Output: (B, T, d_model)
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply the gated feed-forward transform.

        Args:
            x: Input activations of shape (B, T, d_model).

        Returns:
            The SwiGLU output of shape (B, T, d_model), after dropout.
        """
        # SwiGLU: (Swish(Gate) * Up) * Down
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ReLUSquaredMLP(nn.Module):
    """Squared-ReLU feed-forward expert.

    Computes ``c_proj(relu(c_fc(x)) ** 2)`` and then dropout. Unlike
    ``SwiGLUMLP`` both projections carry biases.

    Args:
        d_model: Width of the input and output features.
        hidden_dim: Width of the inner ``c_fc`` projection.
        dropout: Dropout probability applied to the output. Default: 0.1.

    Shape:
        - Input: (B, T, d_model)
        - Output: (B, T, d_model)
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply the squared-ReLU feed-forward transform.

        Args:
            x: Input activations of shape (B, T, d_model).

        Returns:
            The ``ReLU^2`` output of shape (B, T, d_model), after dropout.
        """
        # ReLU^2: ReLU(xW_fc)^2 * W_proj
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2
        x = self.c_proj(x)
        return self.dropout(x)


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_token: int,
        dropout: float = 0.1,
        mlp_type: str = "swiglu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Router (Gating Network)
        self.router = nn.Linear(d_model, num_experts)

        # Experts
        # Hidden dim logic:
        # SwiGLU: 4*d_model (or 8/3*d_model for param match)
        # ReLU^2: 4*d_model (NanoChat uses 4x)
        hidden_dim = 4 * d_model

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            if mlp_type == "swiglu":
                self.experts.append(SwiGLUMLP(d_model, hidden_dim, dropout))
            elif mlp_type == "relusquared":
                self.experts.append(ReLUSquaredMLP(d_model, hidden_dim, dropout))
            else:
                raise ValueError(f"Unknown mlp_type: {mlp_type}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route every token to its top-k experts and mix their outputs.

        The router scores each token over all experts; while ``self.training``
        is set, Gaussian noise scaled by 0.1 is added to the router logits for
        exploration. The top ``num_experts_per_token`` logits are softmaxed
        into mixing weights, and each expert is run once over the tokens that
        selected it (a naive per-expert loop, not a grouped GEMM).

        Args:
            x: Input hidden states of shape (B, T, d_model).

        Returns:
            A tuple ``(output, aux_loss)``. ``output`` is the weight-combined
            expert result of shape (B, T, d_model). ``aux_loss`` is the scalar
            load-balancing term ``num_experts * sum(fraction_of_tokens *
            average_prob)``, where ``fraction_of_tokens`` counts top-k
            selections per expert averaged over tokens and ``average_prob`` is
            the mean full softmax probability per expert.

        Shape:
            - Input: (B, T, d_model)
            - Output: (B, T, d_model), plus a 0-dim ``aux_loss`` tensor.
        """
        # x: (B, T, D_model)
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)

        # Router logits
        router_logits = self.router(flat_x)  # (B*T, Num_Experts)

        # Add noise for exploration during training
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # --- Auxiliary Loss (Load Balancing) ---
        # fraction_of_tokens_assigned: Fraction of tokens assigned to each expert
        # average_routing_probability: Average probability assigned to each expert

        # 1. Fraction of tokens assigned to each expert
        # We check if expert i is in selected_experts for each token
        # selected_experts: (N, k)
        # We want count per expert.
        # One-hot encode selected experts: (N, k, Num_Experts) -> sum over k -> (N, Num_Experts) -> mean over N
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        fraction_of_tokens = expert_mask.sum(dim=1).mean(dim=0)  # (Num_Experts,)

        # 2. Average routing probability per expert
        # We need probabilities for all experts, so we take softmax over all logits
        all_probs = F.softmax(router_logits, dim=-1)  # (N, Num_Experts)
        average_prob = all_probs.mean(dim=0)  # (Num_Experts,)

        # Loss = Num_Experts * sum(fraction * prob)
        aux_loss = self.num_experts * (fraction_of_tokens * average_prob).sum()

        # Process with experts
        # Note: This is a naive implementation (iterating experts).
        # Optimized implementations use scatter/gather or grouped GEMMs.
        final_output = torch.zeros_like(flat_x)

        # For simplicity in this demo, we iterate over tokens (very slow) or experts.
        # Let's iterate over experts to be slightly better.

        # Create a mask for each expert
        # selected_experts: (B*T, k)

        for i in range(self.num_experts):
            # Find tokens that selected this expert
            # (B*T, k) == i -> (B*T, k) boolean
            # expert_mask is (N, k, Num_Experts) from above, we can reuse it?
            # expert_mask[:, :, i] is (N, k) boolean (as float) indicating if expert i was selected at rank k

            # We need indices where expert i was selected
            # expert_indices = (selected_experts == i).nonzero(as_tuple=True)
            # expert_indices is (row_idx, col_idx) where row_idx is token index, col_idx is rank (0..k-1)

            # Let's stick to the previous loop logic but use the weights correctly

            # Find indices where expert i is selected
            # (N, k)
            mask = selected_experts == i
            if mask.any():
                # Get token indices and rank indices
                token_indices, rank_indices = mask.nonzero(as_tuple=True)

                if len(token_indices) > 0:
                    selected_tokens = flat_x[token_indices]
                    expert_out = self.experts[i](selected_tokens)

                    # Weight the output
                    # routing_weights: (N, k)
                    w = routing_weights[token_indices, rank_indices].unsqueeze(1)
                    final_output.index_add_(0, token_indices, w * expert_out)

        return final_output.view(batch_size, seq_len, d_model), aux_loss
