"""Input-modality encoders.

Each encoder here turns one raw modality (text, tables, time series, images,
geometry, graphs) into ``(B, T, output_dim)`` token features for the shared
backbone; ``src/encoders/base.py`` holds the common ``ModalityEncoder``
contract. This module only re-exports the encoder classes.
"""

# GraphEncoder and GeometryEncoder both swallow their own optional-dep
# ImportErrors at module level, so importing them here always succeeds. The
# loud failure happens inside their __init__ via require_modality_deps().
from .crystal_graph import CrystalGraphTokenEncoder
from .dna import DNAEncoder
from .geometry import GeometryEncoder
from .graph import GraphEncoder
from .image import ImageEncoder
from .table import TableEncoder
from .text import TextEncoder
from .time_series import TimeSeriesEncoder

__all__ = [
    "TextEncoder",
    "TableEncoder",
    "TimeSeriesEncoder",
    "ImageEncoder",
    "GeometryEncoder",
    "GraphEncoder",
    "CrystalGraphTokenEncoder",
    "DNAEncoder",
]
