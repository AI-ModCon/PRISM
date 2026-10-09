"""Text modality encoder built on a Hugging Face ``AutoModel`` backbone."""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


class TextEncoder(ModalityEncoder):
    """
    Uses SmolLM3 (or SmolLM2-360M) for text encoding.
    Input: Tokenized Text (B, T_text)
    Output: Features (B, T_text, D_text)
    """

    def __init__(self, d_text: int = 768, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(d_text)
        self.model_name = model_name

        logger.info(f"Loading Text Encoder: {model_name}...")
        # Use local_files_only=True when HF_HUB_OFFLINE is set
        import os

        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        logger.info(f"Successfully loaded Text Encoder: {model_name}")
        # except Exception as e:
        #     logger.warning(f"Warning: Could not load {model_name} ({e}). Fallback to DistilBERT.")
        #     self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        # Project to d_text if necessary
        self.proj = nn.Identity()
        if self.model.config.hidden_size != d_text:
            self.proj = nn.Linear(self.model.config.hidden_size, d_text)

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Embed token ids into text token features.

        Args:
            inputs: Token ids, ``(B, T_text)``.
            attention_mask: Mask marking the non-padding positions of
                ``inputs``, ``(B, T_text)``. Default: ``None``, which assumes
                every position is valid and substitutes an all-ones mask.

        Returns:
            Text features of shape ``(B, T_text, d_text)`` — the backbone's
            ``last_hidden_state`` after ``self.proj``.
        """
        # inputs: (B, T_text)
        if attention_mask is None:
            # Create mask for non-padding tokens (assuming 0 is pad, though tokenizer dependent)
            # For robustness, we should pass mask. If not, we assume all valid.
            attention_mask = torch.ones_like(inputs)

        outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        return self.proj(last_hidden_state)
