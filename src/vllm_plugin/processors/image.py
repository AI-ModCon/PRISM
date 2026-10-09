"""ImageModalityProcessor — extracted from processor.py without behavior change.

The original processor.py emitted a single {input_ids, pixel_values}
BatchFeature with one PromptReplacement that mapped `<image>` to N image
token ids. This class is that contract, isolated so other modalities can be
added without touching the image path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from vllm.multimodal.inputs import MultiModalFieldConfig

from .base import ModalityProcessor

# SigLIP2-base-patch16-224 -> 14x14 = 196 patch tokens. ImageEncoder.forward
# preserves all tokens (no CLS strip), so this is exactly the projector output
# sequence length per image.
DEFAULT_NUM_IMAGE_TOKENS = 196
DEFAULT_IMAGE_SIZE = 224

# Default placeholder when the exported config doesn't specify one.
PRISM_IMAGE_TOKEN = "<image>"


def build_image_transform(size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    # Mirrors the demo transform in src/ui/app.py:81-85.
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class ImageModalityProcessor(ModalityProcessor):
    """Image data plane: PIL image -> (B, 3, 224, 224) pixel_values."""

    MODALITY_NAME = "image"
    MM_KWARG_KEY = "pixel_values"

    def __init__(
        self,
        *,
        placeholder_token: str = PRISM_IMAGE_TOKEN,
        placeholder_token_id: int,
        prism_subconfig: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            modality=self.MODALITY_NAME,
            placeholder_token=placeholder_token,
            placeholder_token_id=placeholder_token_id,
            mm_kwarg_key=self.MM_KWARG_KEY,
            prism_subconfig=prism_subconfig,
        )
        self.image_size = int(
            self.prism_subconfig.get("image_size", DEFAULT_IMAGE_SIZE)
        )
        self.default_num_tokens = int(
            self.prism_subconfig.get("num_image_tokens", DEFAULT_NUM_IMAGE_TOKENS)
        )

    # ------------------------------------------------------------------
    # ModalityProcessor surface

    def num_tokens(self, item: Any) -> int:
        # PRISM uses a fixed image size and SigLIP2 produces a constant patch
        # count, so item-level dimensions don't change the answer today. We
        # still accept `item` because PromptReplacement passes the raw item
        # to its replacement callback per-index.
        return self.default_num_tokens

    def encode(self, raw: Any) -> torch.Tensor:
        transform = build_image_transform(self.image_size)
        if isinstance(raw, Image.Image):
            images_iter: list[Image.Image] = [raw]
        else:
            images_iter = list(raw)
        return torch.stack(
            [transform(img.convert("RGB")) for img in images_iter], dim=0
        )

    def dummy_item(
        self,
        *,
        mm_options: Mapping[str, object] | None,
        count: int,
    ) -> dict[str, Any]:
        # Dummy generation lives on BaseDummyInputsBuilder; we just hand back
        # the parameters it needs. The builder constructs PIL images itself
        # because it owns the override-merging contract.
        return {
            "size": self.image_size,
            "count": count,
            "overrides": (mm_options or {}).get("image"),
        }

    def field_config(self) -> MultiModalFieldConfig:
        return MultiModalFieldConfig.batched("image")

    def normalize_mm_data_key(self, mm_data: Mapping[str, Any]) -> Any | None:
        # Demo callers historically pass either "image" or "images".
        if "images" in mm_data:
            return mm_data["images"]
        return mm_data.get("image")

    # ------------------------------------------------------------------
    # Encoder construction (called by PrismForConditionalGeneration.__init__)

    def build_encoder(self) -> tuple[Any, int, Any]:
        """Construct SigLIP2 vision_model from config (no weight download).

        PRISM's ImageEncoder stores SigLIP under `.model` and an Identity
        `.proj` — we mirror that layout. AutoWeightsLoader fills in real
        weights from the exported safetensors via the
        `vision_tower.model.*` key rename done by checkpoint_export.py.
        """
        from transformers import AutoConfig, AutoModel

        encoder_model = (
            self.prism_subconfig.get("encoder_model")
            or self.prism_subconfig.get("image_encoder_model")
        )
        if not encoder_model:
            raise ValueError(
                "ImageModalityProcessor.build_encoder: prism_subconfig must "
                "carry 'encoder_model' (or legacy 'image_encoder_model')."
            )
        cfg = AutoConfig.from_pretrained(encoder_model)
        inner = AutoModel.from_config(cfg)
        if hasattr(inner, "vision_model"):
            inner = inner.vision_model

        # SigLIP2 wraps the actual transformer under `vision_model`; the
        # `vision_config` block carries the relevant hidden size.
        if hasattr(cfg, "vision_config"):
            cfg = cfg.vision_config
        hidden = getattr(cfg, "hidden_size", None) or int(
            self.prism_subconfig.get("d_img", 0)
        )

        def _forward(model, x):
            outputs = model(pixel_values=x)
            return (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs[0]
            )

        return inner, int(hidden), _forward


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NUM_IMAGE_TOKENS",
    "ImageModalityProcessor",
    "PRISM_IMAGE_TOKEN",
    "build_image_transform",
]
