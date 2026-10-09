"""Table modality encoder built on TAPAS."""

import logging

import torch
import torch.nn as nn
from transformers import TapasModel

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


class TableEncoder(ModalityEncoder):
    """
    Uses TAPAS model for table encoding.
    Input: Pandas DataFrame or Dict of lists
    Output: Token Features (B, T_table, D_table)
    """

    def __init__(
        self, input_dim: int = 128, d_table: int = 768, model_name: str = "google/tapas-base"
    ):
        super().__init__(d_table)
        self.model_name = model_name

        # We use the base model without heads
        logger.info(f"Loading Table Encoder: {model_name}...")
        self.model = TapasModel.from_pretrained(model_name)
        logger.info(f"Successfully loaded Table Encoder: {model_name}")

        # Project to d_table if necessary
        self.proj = nn.Identity()
        if self.model.config.hidden_size != d_table:
            self.proj = nn.Linear(self.model.config.hidden_size, d_table)

    def forward(self, inputs) -> torch.Tensor:
        """Embed an already-tokenized table into token features.

        Args:
            inputs: Either a dict of pre-tokenized TAPAS tensors read under the
                keys ``"input_ids"``, ``"attention_mask"`` and
                ``"token_type_ids"``, or a ``torch.long`` / ``torch.int``
                tensor treated as ``input_ids`` on its own.

        Returns:
            Table token features of shape ``(B, T_table, d_table)`` — TAPAS's
            ``last_hidden_state`` after ``self.proj``.

        Raises:
            ValueError: If ``inputs`` is neither a dict nor a tensor whose
                dtype is exactly ``torch.long`` or ``torch.int``. Float tensors
                are rejected rather than passed through, and so are narrower
                integer dtypes such as ``torch.int16``.
        """
        # inputs: Expecting a dict with 'input_ids', 'attention_mask', 'token_type_ids'
        # In a real pipeline, a tokenizer would process the raw table before this.
        # Here we assume inputs are already tokenized tensors.

        if isinstance(inputs, dict):
            outputs = self.model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
            )
        elif isinstance(inputs, torch.Tensor) and inputs.dtype in [torch.long, torch.int]:
            # Treat as input_ids
            outputs = self.model(input_ids=inputs)
        else:
            # Fallback for dummy tensor input (B, T, D) - if already embeddings (float)
            # FAIL LOUD: Do not allow bypass
            raise ValueError(
                f"TableEncoder Validation Error: Invalid input type {type(inputs)}. Expected Dict or IntTensor input_ids."
            )
            # return self.proj(inputs)

        last_hidden_state = outputs.last_hidden_state
        return self.proj(last_hidden_state)
