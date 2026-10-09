"""Perceiver-style resampling adapter.

Compresses a variable-length sequence of encoder features into a fixed number
of learnable latent tokens, so a modality contributes a constant token budget
to the backbone regardless of its native sequence length.
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler Adapter.
    Aligns variable-length input features (e.g., image patches) to a fixed number of visual tokens
    using cross-attention with learnable latents.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_latents: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents

        # Learnable latents
        self.latents = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)

        # Input projection if dimensions don't match
        self.input_proj = nn.Identity()
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)

        # Perceiver Layers (Cross-Attention + Self-Attention)
        self.layers = nn.ModuleList(
            [PerceiverLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T_in, input_dim)
        Returns:
            out: Resampled features (B, num_latents, d_model)
        """
        B = x.shape[0]

        # Flatten input if > 3D (e.g. B, H, W, D -> B, H*W, D)
        if x.dim() > 3:
            x = x.view(B, -1, x.shape[-1])

        # Project input
        x = self.input_proj(x)  # (B, T_in, d_model)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, num_latents, d_model)

        # Apply layers
        for layer in self.layers:
            latents = layer(latents, x)

        return self.ln_f(latents)


class PerceiverLayer(nn.Module):
    """One Perceiver block: cross-attention, self-attention, feed-forward.

    Pre-norm residual block. The latents cross-attend to the input features,
    then self-attend, then pass through a 4x GELU MLP; each sub-layer is added
    back to the latents as a residual.

    Args:
        d_model: Width of the latents and of the projected input features.
        num_heads: Number of heads in both attention sub-layers.
        dropout: Dropout probability for both ``nn.MultiheadAttention``
            modules and for the final layer of the feed-forward block.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        # Cross-Attention: Latents attend to Input
        self.ln_latents = nn.LayerNorm(d_model)
        self.ln_input = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Self-Attention: Latents attend to Latents
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed Forward
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update the latents from the input features.

        Args:
            latents: Latent queries of shape (B, num_latents, d_model).
            x: Projected input features of shape (B, T_in, d_model), used as
                keys and values for the cross-attention sub-layer.

        Returns:
            The updated latents of shape (B, num_latents, d_model). The
            attention weights returned by ``nn.MultiheadAttention`` are
            discarded.

        Shape:
            - latents: (B, num_latents, d_model)
            - x: (B, T_in, d_model)
            - Output: (B, num_latents, d_model)
        """
        # Cross Attention
        # Query: latents, Key/Value: x
        q = self.ln_latents(latents)
        k = v = self.ln_input(x)
        attn_out, _ = self.cross_attn(q, k, v)
        latents = latents + attn_out

        # Self Attention
        q_sa = k_sa = v_sa = self.ln_self(latents)
        attn_out_sa, _ = self.self_attn(q_sa, k_sa, v_sa)
        latents = latents + attn_out_sa

        # Feed Forward
        latents = latents + self.ff(self.ln_ff(latents))

        return latents
