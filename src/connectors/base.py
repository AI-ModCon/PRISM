"""Extension interfaces shared by output decoder conditioning routes."""

from abc import ABC, abstractmethod

from torch import nn

from .types import BackboneFeatures, DecoderContext, ReadoutResult


class Readout(nn.Module, ABC):
    """Choose evidence from a feature bank without accessing prediction targets."""

    @abstractmethod
    def forward(self, features: BackboneFeatures) -> ReadoutResult:
        raise NotImplementedError


class ConditioningBridge(nn.Module, ABC):
    """Translate readout representations into a decoder's conditioning space."""

    @abstractmethod
    def connect(self, readout: ReadoutResult) -> DecoderContext:
        raise NotImplementedError
