"""AION raw-image encoder for AstroPRISM.

This module deliberately imports AION only when :class:`AIONImageEncoder` is
constructed.  Importing PRISM therefore remains possible in environments that
do not have the optional ``polymathic-aion`` package installed.

The image path matches the released AION feature pipeline:

``calibrated g/r/i/z flux -> ImageCodec -> 576 image token ids -> AION.encode``

The image codec uses hard finite-scalar quantization.  It is consequently kept
in float32, frozen, and in evaluation mode.  The AION image-token embedding and
shared transformer encoder can be unfrozen later through
``set_shared_encoder_trainable``; the codec is never made trainable by that
hook.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .base import ModalityEncoder

AION_REPO_ID = "polymathic-ai/aion-base"
AION_MODEL_REVISION = "40541618104bab0fa85c8af68daeb867a720bb8c"
AION_IMAGE_CODEC_REVISION = AION_MODEL_REVISION

AION_IMAGE_TOKENS = 576
AION_IMAGE_WIDTH = 768
AION_RAW_IMAGE_SIZE = (160, 160)
AION_LEGACY_SURVEY_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")


def _load_aion_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Import the optional AION classes only when the encoder is requested."""

    try:
        from aion import AION
        from aion.codecs import ImageCodec
        from aion.modalities import LegacySurveyImage
    except ImportError as exc:  # pragma: no cover - exact message is environment-dependent
        raise ImportError(
            "AIONImageEncoder requires the optional 'polymathic-aion' package "
            "(and its dependencies). Install a pinned AION source checkout before "
            "selecting the AION image encoder."
        ) from exc
    return AION, ImageCodec, LegacySurveyImage


class AIONImageEncoder(ModalityEncoder):
    """Encode calibrated Legacy Survey images with the pinned AION image path.

    Args:
        model_name_or_path: A Hugging Face repository id or a complete local
            AION model directory.
        model_revision: Explicit revision for the AION transformer weights.
        codec_revision: Explicit revision for the image codec weights.
        cache_dir: Optional Hugging Face cache containing the pinned snapshots.
        local_files_only: Refuse network access when loading model assets.
        bands: Input channel names.  Only the verified Legacy Survey g/r/i/z
            ordering is accepted.
        raw_image_size: Expected calibrated cutout size before AION's official
            96x96 center crop.

    Input:
        Floating tensor ``(B, 4, 160, 160)`` in calibrated Legacy Survey flux
        units and g/r/i/z channel order.

    Output:
        AION encoder features ``(B, 576, 768)``.
    """

    def __init__(
        self,
        model_name_or_path: str = AION_REPO_ID,
        *,
        model_revision: str = AION_MODEL_REVISION,
        codec_revision: str = AION_IMAGE_CODEC_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        bands: Sequence[str] = AION_LEGACY_SURVEY_BANDS,
        raw_image_size: tuple[int, int] = AION_RAW_IMAGE_SIZE,
    ) -> None:
        super().__init__(AION_IMAGE_WIDTH)

        resolved_bands = tuple(str(band) for band in bands)
        if resolved_bands != AION_LEGACY_SURVEY_BANDS:
            raise ValueError(
                "AION raw-image input must use calibrated Legacy Survey g/r/i/z channels "
                f"in {AION_LEGACY_SURVEY_BANDS!r} order; got {resolved_bands!r}."
            )
        if len(raw_image_size) != 2 or any(int(size) <= 0 for size in raw_image_size):
            raise ValueError(
                f"raw_image_size must contain two positive integers, got {raw_image_size!r}"
            )

        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = str(model_revision)
        self.codec_revision = str(codec_revision)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.bands = resolved_bands
        self.raw_image_size = tuple(int(size) for size in raw_image_size)
        self.num_image_tokens = AION_IMAGE_TOKENS
        self.hidden_dim = AION_IMAGE_WIDTH

        AION, ImageCodec, LegacySurveyImage = _load_aion_types()
        self._legacy_survey_image_type = LegacySurveyImage
        self.image_token_key = str(LegacySurveyImage.token_key)

        common_load_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        self.aion_model = AION.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            **common_load_kwargs,
        )
        self.image_codec = ImageCodec.from_pretrained(
            self.model_name_or_path,
            modality=LegacySurveyImage,
            revision=self.codec_revision,
            **common_load_kwargs,
        )

        self._validate_aion_interface()
        self._shared_encoder_trainable = False
        self.set_shared_encoder_trainable(False)

    def _validate_aion_interface(self) -> None:
        required = ("encoder_embeddings", "encoder", "encoder_norm", "decoder_proj_context")
        missing = [name for name in required if not hasattr(self.aion_model, name)]
        if missing:
            raise RuntimeError(f"Loaded AION model is missing encoder components: {missing}")
        if self.image_token_key not in self.aion_model.encoder_embeddings:
            raise RuntimeError(
                f"Loaded AION model has no encoder embedding for {self.image_token_key!r}"
            )

    def _shared_encoder_modules(self) -> tuple[nn.Module, ...]:
        """Modules used by ``AION.encode`` for an image-only encoder pass."""

        return (
            self.aion_model.encoder_embeddings[self.image_token_key],
            self.aion_model.encoder,
            self.aion_model.encoder_norm,
            self.aion_model.decoder_proj_context,
        )

    def _enforce_trainability(self) -> None:
        # Freeze every AION component first so decoder heads and embeddings that
        # are not used by this image-only wrapper can never enter an optimizer.
        for parameter in self.aion_model.parameters():
            parameter.requires_grad_(False)
        if self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        # Hard quantization breaks a pixel-to-code gradient path.  The codec is
        # always a deterministic, float32 preprocessing/tokenization component.
        self.image_codec.float()
        self.image_codec.eval()
        for parameter in self.image_codec.parameters():
            parameter.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        """Keep the codec in float32 through blanket dtype casts.

        ``nn.Module.to()`` reaches children via ``_apply``, so a child-level
        ``to()`` override never sees it and the trainer's
        ``self.model.encoders.to(dtype=target_dtype)`` rounds the codec.
        Calling ``.float()`` afterwards restores the dtype but not the values,
        and the codec quantizes hard: rounded weights can change token ids.
        Device moves pass through untouched. Mirrors ``DNAEncoder._apply``.
        """
        is_reduced_precision = False
        try:
            probe = fn(torch.zeros(1, dtype=torch.float32))
            if isinstance(probe, torch.Tensor) and probe.dtype not in (
                torch.float32,
                torch.float64,
            ):
                is_reduced_precision = True
        except Exception:
            pass

        if is_reduced_precision:
            for child in self.children():
                if child is not self.image_codec:
                    child._apply(fn, recurse=recurse)
            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)
            return self

        return super()._apply(fn, recurse=recurse)

    @property
    def shared_encoder_trainable(self) -> bool:
        """Whether the differentiable AION image/shared-encoder path is trainable."""

        return self._shared_encoder_trainable

    def set_shared_encoder_trainable(self, trainable: bool) -> AIONImageEncoder:
        """Freeze or unfreeze only the differentiable AION encoding path.

        The image codec remains float32, frozen, and in evaluation mode for both
        settings.
        """

        self._shared_encoder_trainable = bool(trainable)
        self._enforce_trainability()
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> AIONImageEncoder:
        """Set runtime mode while preserving AION codec/freeze invariants."""

        super().train(mode)
        self._enforce_trainability()

        # Start from deterministic evaluation behavior for the full AION model.
        # If encoder fine-tuning is explicitly enabled, only modules on the
        # image/shared-encoder path enter training mode.
        self.aion_model.eval()
        if mode and self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                module.train(True)
        self.image_codec.eval()
        return self

    def tokens_per_instance(self) -> int:
        """Return the fixed number of image tokens the AION codec emits."""
        return self.num_image_tokens

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"AION image input must be a torch.Tensor, got {type(inputs).__name__}")
        expected = (len(self.bands), *self.raw_image_size)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                "AION image input must have shape "
                f"(B, {expected[0]}, {expected[1]}, {expected[2]}), got {tuple(inputs.shape)}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError(f"AION image input must be floating point, got {inputs.dtype}")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("AION image input contains non-finite calibrated flux values")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode calibrated four-band images into ``(B, 576, 768)`` features."""
        self._validate_inputs(inputs)
        self._enforce_trainability()

        # AION's Clamp operates in place on a view returned by CenterCrop.  Clone
        # here so encoding never mutates the caller's training batch.  The hard
        # codec is deliberately outside autograd and outside mixed precision.
        with torch.no_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
            codec_flux = inputs.detach().to(dtype=torch.float32).clone()
            image = self._legacy_survey_image_type(flux=codec_flux, bands=list(self.bands))
            image_token_ids = self.image_codec.encode(image)

        expected_tokens = (inputs.shape[0], self.num_image_tokens)
        if tuple(image_token_ids.shape) != expected_tokens:
            raise RuntimeError(
                f"AION ImageCodec must emit token ids with shape {expected_tokens}, "
                f"got {tuple(image_token_ids.shape)}"
            )

        encoded_inputs = {self.image_token_key: image_token_ids}
        if self._shared_encoder_trainable:
            features = self.aion_model.encode(
                encoded_inputs,
                num_encoder_tokens=self.num_image_tokens,
            )
        else:
            with torch.no_grad():
                features = self.aion_model.encode(
                    encoded_inputs,
                    num_encoder_tokens=self.num_image_tokens,
                )

        expected_features = (inputs.shape[0], self.num_image_tokens, self.output_dim)
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                f"AION shared encoder must emit features with shape {expected_features}, "
                f"got {tuple(features.shape)}"
            )
        return features


__all__ = [
    "AIONImageEncoder",
    "AION_IMAGE_CODEC_REVISION",
    "AION_IMAGE_TOKENS",
    "AION_IMAGE_WIDTH",
    "AION_LEGACY_SURVEY_BANDS",
    "AION_MODEL_REVISION",
    "AION_RAW_IMAGE_SIZE",
    "AION_REPO_ID",
]
