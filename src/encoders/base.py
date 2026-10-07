"""Common base class for the input-modality encoders."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module, ABC):
    """Abstract base class that every input-modality encoder subclasses.

    A concrete encoder maps one raw modality to ``(B, T, output_dim)`` token
    features for the shared backbone. Subclasses implement ``forward`` and, when
    a single instance of the modality spans more than one token, override
    ``tokens_per_instance``.

    Args:
        output_dim: Declared width of the token features, stored on
            ``self.output_dim``.

    Attributes:
        output_dim: Declared width of the token features. Nothing here enforces
            it, and subclasses whose backbone fixes the width (for example
            ``TimeSeriesEncoder`` under ``intern_s2`` or ``timeomni``) reassign
            it after calling ``super().__init__``.
        _last_pad_mask: Bool ``(B, T)`` mask, where True marks padding, read
            back by ``src/model.py`` after each encoder call. Only
            ``TimeSeriesEncoder``'s Intern-S2 path sets it today; it stays
            ``None`` otherwise.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self._last_pad_mask: torch.Tensor | None = None

    def tokens_per_instance(self) -> int:
        """
        Returns the number of tokens that this modality contributes per instance of the modality
        in an input sequence. This is used for interleaving logic.
        """
        return 1  # Default is 1 token per instance, override for modalities that contribute multiple tokens (e.g. time series patches)

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.
        Args:
            inputs: Input tensor specific to the modality.
        Returns:
            features: Tensor of shape (B, T, output_dim)

        Encoders that right-pad variable-length outputs may set
        ``_last_pad_mask`` to a bool ``(B, T)`` mask, where True marks padding.
        """
        pass
