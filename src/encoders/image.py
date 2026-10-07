"""Image modality encoder built on a Hugging Face vision backbone."""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


class ImageEncoder(ModalityEncoder):
    """
    Uses SigLIP2 (google/siglip2-base-patch16-224) for image encoding.
    Input: RGB Images (B, 3, H, W)
    Output: Patch Features (B, T_img_patches, D_img)
    """

    def __init__(self, d_img: int = 512, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__(d_img)
        self.model_name = model_name

        # try:
        logger.info(f"Loading Image Encoder: {model_name}...")
        # SigLIP2 models are usually loaded with AutoModel.
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)

        # If loaded model is a full SigLIP model (multimodal), use only the vision encoder
        if hasattr(self.model, "vision_model"):
            self.model = self.model.vision_model
        logger.info(f"Successfully loaded Image Encoder: {model_name}")

        # except Exception as e:
        #     logger.warning(f"Warning: Could not load {model_name} ({e}). Fallback to ResNet18.")
        #     self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #     # Remove fc and avgpool for ResNet fallback
        #     self.model = nn.Sequential(*list(self.model.children())[:-2])

        # Project to d_img if necessary
        self.proj = nn.Identity()

        # Determine hidden size
        if isinstance(self.model, nn.Sequential):  # ResNet Fallback
            hidden_size = 512
        else:
            # SigLIP config usually has vision_config
            if hasattr(self.model.config, "vision_config"):
                hidden_size = self.model.config.vision_config.hidden_size
            elif hasattr(self.model.config, "hidden_size"):
                hidden_size = self.model.config.hidden_size
            else:
                # Fallback or print dir to debug
                logger.warning(
                    f"Warning: Could not find hidden_size in config: {dir(self.model.config)}"
                )
                hidden_size = 768  # Default for base model

        if hidden_size != d_img:
            logger.warning(
                f"⚠️ [ImageEncoder] Dimension Mismatch: Encoder ({hidden_size}) != Config ({d_img})"
            )
            logger.warning(
                f"   -> Initializing RANDOM Linear Projection ({hidden_size} -> {d_img})"
            )
            logger.warning(
                "   -> CRITICAL: Ensure 'freeze_encoders' is FALSE, otherwise this layer remains random!"
            )
            self.proj = nn.Linear(hidden_size, d_img)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed a batch of images into patch features.

        Args:
            inputs: Pixel values, ``(B, 3, H, W)``.

        Returns:
            Patch features of shape ``(B, T_img_patches, d_img)``, taken from
            the backbone's ``last_hidden_state`` (or ``outputs[0]`` when the
            output object exposes no ``last_hidden_state``) and passed through
            ``self.proj``. No position is dropped, so any leading pooled/CLS
            position the backbone emits is passed through as a token. For the
            ``nn.Sequential`` (ResNet) path the spatial map is flattened to
            ``(B, H' * W', d_img)`` instead.
        """
        # inputs: (B, 3, H, W)

        if isinstance(self.model, nn.Sequential):  # ResNet Fallback
            x = self.model(inputs)  # (B, 512, H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, T, 512)
            return self.proj(x)

        # SigLIP2 / ViT
        # Assuming inputs are pixel_values
        outputs = self.model(pixel_values=inputs)

        # We want patch features (last_hidden_state)
        # Some CLIP vision models return (pooled_output, last_hidden_state) or similar.
        # AutoModel usually returns BaseModelOutputWithPooling

        if hasattr(outputs, "last_hidden_state"):
            x = outputs.last_hidden_state  # (B, T, H)
            # Remove CLS token if present? Usually index 0.
            # For alignment, keeping it or not is a choice. Let's keep all.
        else:
            # Fallback if output format is weird
            x = outputs[0]

        return self.proj(x)
