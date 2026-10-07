"""Back-compat shim. The real classes live under `processors/`.

Historical layout:
    from src.vllm_plugin.processor import (
        PrismMultiModalProcessor, PrismProcessingInfo, PrismDummyInputsBuilder,
        _build_image_transform,
    )

After VLLM-1 these are split across per-modality modules; we keep the names
exported here so PR #41 callers (`src/vllm_plugin/__init__.py`,
`tests/test_vllm_plugin.py`, any downstream users) don't churn.
"""

from .processors.image import (  # noqa: F401
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_IMAGE_TOKENS,
    PRISM_IMAGE_TOKEN,
)
from .processors.image import (
    build_image_transform as _build_image_transform,
)
from .processors.orchestrator import (  # noqa: F401
    PrismDummyInputsBuilder,
    PrismMultiModalProcessor,
    PrismProcessingInfo,
)

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NUM_IMAGE_TOKENS",
    "PRISM_IMAGE_TOKEN",
    "PrismDummyInputsBuilder",
    "PrismMultiModalProcessor",
    "PrismProcessingInfo",
    "_build_image_transform",
]
