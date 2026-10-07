"""Reusable readout/bridge contracts, independently parameterized per route."""

from .base import ConditioningBridge, Readout
from .sequence import (
    FinalStateReadout,
    IdentityBridge,
    LayerNormLinearBridge,
    PooledStateReadout,
    build_bridge,
    build_readout,
    pack_right_padded,
)
from .types import BackboneFeatures, DecoderContext, ReadoutResult

__all__ = [
    "BackboneFeatures",
    "ConditioningBridge",
    "DecoderContext",
    "FinalStateReadout",
    "IdentityBridge",
    "LayerNormLinearBridge",
    "PooledStateReadout",
    "Readout",
    "ReadoutResult",
    "build_bridge",
    "build_readout",
    "pack_right_padded",
]
