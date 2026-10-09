"""Output-decoder registry.

Mirror of ``src/encoders/__init__.py`` for the output side. New decoders
register here by name; ``UnifiedTransformer`` builds its ``self.decoders``
ModuleDict from ``DECODERS``.
"""

from .base import OutputDecoder
from .geometry import GeometryDecoder
from .graph import GraphDecoder
from .image import ImageDecoder
from .lm_head import LMHeadDecoder
from .regression import RegressionDecoder
from .time_series import TimeSeriesDecoder
from .types import DecoderCondition, DecoderResult

# name -> class. Keep keys stable: they are referenced from config
# (``output_decoders``) and checkpoints.
DECODERS: dict[str, type[OutputDecoder]] = {
    "text": LMHeadDecoder,
    "action": RegressionDecoder,
    "regression": RegressionDecoder,
    "time_series": TimeSeriesDecoder,
    "geometry": GeometryDecoder,
    "graph": GraphDecoder,
    "image": ImageDecoder,
}

def remap_legacy_decoder_keys(state_dict: dict) -> dict:
    """Rename pre-decoder-refactor checkpoint keys to the current layout.

    Phase 0 moved the VLA action head from a bare ``nn.Sequential`` to a
    ``RegressionDecoder`` wrapper, so its parameters moved:

        action_head.{0,2}.{weight,bias}  ->  action_head.head.{0,2}.{weight,bias}

    Old VLA checkpoints carry the former; without remapping they land in
    ``unexpected`` keys under ``strict=False`` and the action head silently
    re-initializes. This rewrites those keys so old checkpoints load losslessly.

    Returns a new dict; the input is not mutated. Idempotent — keys already in
    the new layout (``action_head.head.*``) are passed through untouched.
    """
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        # Only the bare-Sequential form (action_head.0.*, action_head.2.*),
        # never the already-migrated action_head.head.* form.
        if key.startswith("action_head.") and not key.startswith("action_head.head."):
            new_key = "action_head.head." + key[len("action_head."):]
        remapped[new_key] = value
    return remapped


__all__ = [
    "OutputDecoder",
    "LMHeadDecoder",
    "RegressionDecoder",
    "TimeSeriesDecoder",
    "GeometryDecoder",
    "GraphDecoder",
    "ImageDecoder",
    "DecoderCondition",
    "DecoderResult",
    "DECODERS",
    "remap_legacy_decoder_keys",
]
