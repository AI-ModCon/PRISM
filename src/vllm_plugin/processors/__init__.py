"""Per-modality processor stack for the PRISM vLLM plugin.

Each `ModalityProcessor` owns one modality's data-plane contract: how raw
input becomes a `BatchFeature` tensor, how the placeholder token expands
into N feature tokens, and how the dummy input builder synthesizes worst-
case inputs for vLLM memory profiling.

Stage A adds the abstraction with `image` as the only implementation; later
stages add `time_series`, `geometry`, `dna`.
"""

# Import for the side effect of registering the time_series factory. Cheap:
# this module only pulls torch (already loaded) and vllm.multimodal types.
# Heavy training-side deps (uni2ts/Moirai) only load when build_encoder is
# called by the model class on a checkpoint where time_series is active.
from . import time_series as _time_series  # noqa: F401
from .base import ModalityProcessor
from .image import ImageModalityProcessor
from .registry import (
    MODALITY_PROCESSORS,
    build_modality_processors,
    register_modality_processor,
)

__all__ = [
    "ImageModalityProcessor",
    "MODALITY_PROCESSORS",
    "ModalityProcessor",
    "build_modality_processors",
    "register_modality_processor",
]
