"""Reusable model building blocks.

Re-exports the ``nn.Module`` pieces that ``src/model.py`` assembles into
``UnifiedTransformer``: the Perceiver-style adapter and the modality
projector, which turn encoder features into backbone tokens on every path,
plus the sparse Mixture-of-Experts feed-forward layer, causal self-attention
and the causal language-modeling head, which are built only when no HF
backbone is configured.
"""

from .adapter import PerceiverResampler
from .attention import CausalSelfAttention
from .heads import CausalLMHead
from .moe import MoELayer
from .projector import ModalityProjector

__all__ = [
    "PerceiverResampler",
    "ModalityProjector",
    "MoELayer",
    "CausalSelfAttention",
    "CausalLMHead",
]
