"""Output head for causal language modeling.

Contains ``CausalLMHead``, the bias-free projection from hidden states to
vocabulary logits. ``UnifiedTransformer`` builds it as ``self.head`` only on
the backbone-free path; when an HF backbone is configured the LM head lives
inside that backbone instead.
"""

import torch
import torch.nn as nn


class CausalLMHead(nn.Module):
    """
    Causal Language Modeling Head.
    Projects the hidden state to the vocabulary size to predict the next token.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            logits: (B, T, vocab_size)
        """
        return self.decoder(x)
