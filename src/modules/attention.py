"""Rotary position embeddings and causal self-attention.

Holds the attention primitives used by ``src/model.py``: ``RotaryEmbedding``
caches the RoPE cosine/sine tables, ``rotate_half`` and
``apply_rotary_pos_emb`` apply them to queries and keys, and
``CausalSelfAttention`` is the eager (non-fused) masked attention sub-layer of
every ``TransformerBlock``.
"""

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precompute rotary position embedding cosine/sine tables.

    Builds ``inv_freq`` as ``1 / base ** (arange(0, dim, 2) / dim)``, then
    caches the ``cos`` and ``sin`` of ``cat((freqs, freqs))`` for
    ``max_position_embeddings`` positions. ``inv_freq`` is a persistent
    buffer; ``cos_cached`` and ``sin_cached`` are non-persistent and are
    rebuilt on demand when a longer sequence arrives.

    Args:
        dim: Per-head size the rotation is applied over, usually
            ``d_model // num_heads``.
        max_position_embeddings: Number of positions to precompute.
            Default: 2048.
        base: Base of the inverse-frequency geometric series. Default: 10000.
        device: Device the ``inv_freq`` table is built on. Default: ``None``
            (the default device).

    Attributes:
        max_seq_len_cached: Number of positions currently in the cached
            tables; grown in ``forward`` when a longer sequence arrives.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        """Return the cosine/sine tables for the first ``seq_len`` positions.

        If ``seq_len`` exceeds ``max_seq_len_cached`` the tables are rebuilt at
        the larger length and re-registered before slicing.

        Args:
            x: Reference tensor supplying the dtype the tables are cast to;
                nothing else about it is read, so its rank is unconstrained.
                ``UnifiedTransformer`` passes the (B, T, d_model) hidden
                states, not a per-head tensor.
            seq_len: Number of positions to return. Compared against
                ``max_seq_len_cached``, so a concrete integer must be passed
                despite the ``None`` default.

        Returns:
            A tuple ``(cos, sin)``, each of shape (1, 1, seq_len, dim) and of
            ``x``'s dtype, ready to broadcast over batch and heads in
            ``apply_rotary_pos_emb``.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    Each input is rotated as ``x * cos + rotate_half(x) * sin``.

    Args:
        q: Queries of shape (B, num_heads, T, head_size).
        k: Keys of shape (B, num_heads, T, head_size).
        cos: Cosine table broadcastable to ``q``/``k``, as returned by
            ``RotaryEmbedding``.
        sin: Sine table broadcastable to ``q``/``k``.

    Returns:
        A tuple ``(q_embed, k_embed)`` — the rotated queries and keys, in that
        order, each with the same shape as its input.
    """
    # q, k: [bs, num_attention_heads, seq_len, head_size]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, freqs_cis: tuple = None
    ) -> torch.Tensor:
        """Run masked multi-head self-attention over the sequence.

        Projects ``x`` once into packed queries, keys and values, optionally
        rotates queries and keys with ``freqs_cis``, applies scaled dot-product
        attention, then merges heads and applies the output projection.

        Args:
            x: Input hidden states of shape (B, T, d_model).
            mask: Additive attention mask broadcastable to (B, 1, T, T), with
                ``0`` for positions to keep and a large negative value for
                positions to block. ``UnifiedTransformer`` uses ``-1e9``
                rather than ``-inf``, which would send a fully masked row to
                NaN through the softmax. Default: ``None``, which falls back
                to the registered lower-triangular causal ``bias`` buffer.
            freqs_cis: Tuple ``(cos, sin)`` from ``RotaryEmbedding``, applied
                to queries and keys before attention. Default: ``None`` (no
                rotary embedding).

        Returns:
            The attention output of shape (B, T, d_model), after the output
            projection and residual dropout. Note that the caller, not this
            module, adds the residual connection.

        Shape:
            - Input: (B, T, d_model)
            - Output: (B, T, d_model)
        """
        B, T, C = x.shape

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Causal attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            # mask expected to be (B, 1, T, T) or (1, 1, T, T)
            # 0 or False means masked (ignore), 1 or True means attend
            # If mask is boolean: True=keep, False=mask
            # If mask is additive: 0=keep, -inf=mask

            # Assuming additive mask for flexibility (0 for keep, -inf for mask)
            att = att + mask
        else:
            # Fallback to causal mask
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))
