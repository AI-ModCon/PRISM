"""Image preprocessing helpers shared by PRISM training datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_ENCODER_ID = "google/siglip2-base-patch16-224"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def _cfg_value(model_config: Any | None, name: str, default: Any) -> Any:
    if model_config is None:
        return default
    return getattr(model_config, name, default)


def _rgb_tuple(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    items = list(value)
    if len(items) != 3:
        raise ValueError(f"Expected 3 RGB values, got {value!r}")
    return tuple(float(x) for x in items)


class _HFImageProcessorTransform:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, image: Any) -> torch.Tensor:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        batch = self.processor(images=image, return_tensors="pt")
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Image processor output did not include pixel_values")
        return pixel_values.squeeze(0)


def _load_auto_image_processor() -> Any:
    from transformers import AutoImageProcessor

    return AutoImageProcessor


def build_image_transform(
    model_config: Any | None = None,
    *,
    processor_id: str | None = None,
    image_size: int | None = None,
    image_mean: tuple[float, float, float] | list[float] | None = None,
    image_std: tuple[float, float, float] | list[float] | None = None,
    local_files_only: bool = True,
    strict_processor: bool | None = None,
) -> Callable[[Any], torch.Tensor] | None:
    """Build the PIL-to-pixel-values transform for the selected image tower.

    If ``processor_id`` or ``model_config.image_processor_id`` is set, this
    first tries the HuggingFace image processor so SigLIP variants get their
    native size, crop, and normalization. If no processor is configured, or the
    processor cannot be loaded and strict mode is disabled, the legacy
    torchvision resize/normalize path is used.
    """

    resolved_processor_id = processor_id
    if resolved_processor_id is None:
        resolved_processor_id = _cfg_value(model_config, "image_processor_id", None)

    if strict_processor is None:
        strict_processor = bool(_cfg_value(model_config, "image_processor_strict", False))

    if resolved_processor_id:
        try:
            AutoImageProcessor = _load_auto_image_processor()
            processor = AutoImageProcessor.from_pretrained(
                resolved_processor_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            return _HFImageProcessorTransform(processor)
        except Exception as exc:
            if strict_processor:
                raise RuntimeError(
                    f"Could not load image processor {resolved_processor_id!r} "
                    f"with local_files_only={local_files_only}"
                ) from exc
            logger.warning(
                "Could not load image processor %s; falling back to torchvision "
                "resize/normalize preprocessing: %s",
                resolved_processor_id,
                exc,
            )

    size = int(image_size or _cfg_value(model_config, "image_size", DEFAULT_IMAGE_SIZE))
    mean = _rgb_tuple(image_mean or _cfg_value(model_config, "image_mean", None), DEFAULT_IMAGE_MEAN)
    std = _rgb_tuple(image_std or _cfg_value(model_config, "image_std", None), DEFAULT_IMAGE_STD)

    try:
        from torchvision import transforms
    except ImportError:
        logger.warning("torchvision not available, images will not be transformed")
        return None

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
