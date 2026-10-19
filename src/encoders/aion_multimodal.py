"""Joint AION image and spectrum encoder with one shared transformer owner.

The standalone :class:`AIONImageEncoder` and :class:`AIONSpectrumEncoder`
each own a complete AION model.  That is the right ownership model for a
single-modality run, but registering both wrappers in one PRISM model would
duplicate AION's shared transformer (and, when unfrozen, let the copies
diverge).  This module provides the joint path instead:

``raw image -> ImageCodec -------+``
``                               AION.encode -> 768-d tokens``
``raw spectrum -> SpectrumCodec -+``

The AION transformer is registered exactly once.  Both hard-quantizing
codecs remain float32, frozen, and in evaluation mode for the lifetime of the
module.  The shared transformer and both active modality token embeddings can
be unfrozen together through :meth:`set_shared_encoder_trainable`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .aion_image import (
    AION_IMAGE_CODEC_REVISION,
    AION_IMAGE_TOKENS,
    AION_IMAGE_WIDTH,
    AION_LEGACY_SURVEY_BANDS,
    AION_MODEL_REVISION,
    AION_RAW_IMAGE_SIZE,
    AION_REPO_ID,
)
from .aion_spectrum import (
    AION_RAW_SPECTRUM_LENGTH,
    AION_SPECTRUM_CODEC_REVISION,
    AION_SPECTRUM_FIELDS,
    AION_SPECTRUM_TOKENS,
    AION_SPECTRUM_WIDTH,
)
from .base import ModalityEncoder


def _load_aion_types() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    """Import optional AION classes only when the joint encoder is requested."""

    try:
        from aion import AION
        from aion.codecs import ImageCodec, SpectrumCodec
        from aion.modalities import DESISpectrum, LegacySurveyImage
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "AIONMultimodalEncoder requires the optional 'polymathic-aion' package "
            "(and its dependencies). Install a pinned AION source checkout before "
            "selecting joint AION image+spectrum encoding."
        ) from exc
    return AION, ImageCodec, SpectrumCodec, LegacySurveyImage, DESISpectrum


class AIONMultimodalEncoder(ModalityEncoder):
    """Encode raw Legacy Survey images and DESI spectra through one AION model.

    ``encode_modality("image", tensor)`` accepts ``float[B, 4, 160, 160]``
    calibrated Legacy Survey flux and emits ``float[B, 576, 768]`` features.
    ``encode_modality("spectrum", mapping)`` accepts the four raw DESI arrays
    and emits ``float[B, 273, 768]`` features.
    """

    supported_modalities = frozenset({"image", "spectrum"})

    def __init__(
        self,
        model_name_or_path: str = AION_REPO_ID,
        *,
        model_revision: str = AION_MODEL_REVISION,
        image_codec_revision: str = AION_IMAGE_CODEC_REVISION,
        spectrum_codec_revision: str = AION_SPECTRUM_CODEC_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        bands: Sequence[str] = AION_LEGACY_SURVEY_BANDS,
        raw_image_size: tuple[int, int] = AION_RAW_IMAGE_SIZE,
        raw_spectrum_length: int = AION_RAW_SPECTRUM_LENGTH,
    ) -> None:
        if AION_IMAGE_WIDTH != AION_SPECTRUM_WIDTH:  # pragma: no cover - pinned constants
            raise RuntimeError(
                "Joint AION ownership requires image and spectrum features to share a width"
            )
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
        if int(raw_spectrum_length) <= 0:
            raise ValueError(
                "raw_spectrum_length must be a positive integer, got "
                f"{raw_spectrum_length!r}"
            )

        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = str(model_revision)
        self.image_codec_revision = str(image_codec_revision)
        self.spectrum_codec_revision = str(spectrum_codec_revision)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.bands = resolved_bands
        self.raw_image_size = tuple(int(size) for size in raw_image_size)
        self.raw_spectrum_length = int(raw_spectrum_length)
        self.num_image_tokens = AION_IMAGE_TOKENS
        self.num_spectrum_tokens = AION_SPECTRUM_TOKENS
        self.hidden_dim = AION_IMAGE_WIDTH

        (
            AION,
            ImageCodec,
            SpectrumCodec,
            LegacySurveyImage,
            DESISpectrum,
        ) = _load_aion_types()
        self._legacy_survey_image_type = LegacySurveyImage
        self._desi_spectrum_type = DESISpectrum
        self.image_token_key = str(LegacySurveyImage.token_key)
        self.spectrum_token_key = str(DESISpectrum.token_key)

        common_load_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        # This is the sole registered copy of the AION transformer.
        self.aion_model = AION.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            **common_load_kwargs,
        )
        self.image_codec = ImageCodec.from_pretrained(
            self.model_name_or_path,
            modality=LegacySurveyImage,
            revision=self.image_codec_revision,
            **common_load_kwargs,
        )
        self.spectrum_codec = SpectrumCodec.from_pretrained(
            self.model_name_or_path,
            modality=DESISpectrum,
            revision=self.spectrum_codec_revision,
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
        missing_embeddings = [
            key
            for key in (self.image_token_key, self.spectrum_token_key)
            if key not in self.aion_model.encoder_embeddings
        ]
        if missing_embeddings:
            raise RuntimeError(
                "Loaded AION model is missing joint encoder embeddings: "
                f"{missing_embeddings}"
            )

    def _shared_encoder_modules(self) -> tuple[nn.Module, ...]:
        """Return every differentiable module used by either joint route."""

        return (
            self.aion_model.encoder_embeddings[self.image_token_key],
            self.aion_model.encoder_embeddings[self.spectrum_token_key],
            self.aion_model.encoder,
            self.aion_model.encoder_norm,
            self.aion_model.decoder_proj_context,
        )

    def _enforce_trainability(self) -> None:
        # Decoder heads and embeddings for unrelated AION modalities never
        # enter a PRISM optimizer.
        for parameter in self.aion_model.parameters():
            parameter.requires_grad_(False)
        if self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        for codec in (self.image_codec, self.spectrum_codec):
            codec.float()
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)

    @property
    def shared_encoder_trainable(self) -> bool:
        """Whether the shared AION transformer is currently trainable."""
        return self._shared_encoder_trainable

    def set_shared_encoder_trainable(self, trainable: bool) -> AIONMultimodalEncoder:
        """Toggle the one shared core and both modality embeddings together."""

        self._shared_encoder_trainable = bool(trainable)
        self._enforce_trainability()
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> AIONMultimodalEncoder:
        """Set runtime mode while preserving both codec invariants."""

        super().train(mode)
        self._enforce_trainability()
        self.aion_model.eval()
        if mode and self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                module.train(True)
        self.image_codec.eval()
        self.spectrum_codec.eval()
        return self

    def tokens_per_instance(self, modality: str) -> int:
        """Return the token count this shared owner emits for ``modality``."""
        modality_key = str(modality)
        if modality_key == "image":
            return self.num_image_tokens
        if modality_key == "spectrum":
            return self.num_spectrum_tokens
        raise ValueError(
            f"Unsupported AION modality {modality_key!r}; expected 'image' or 'spectrum'"
        )

    def _validate_image_inputs(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"AION image input must be a torch.Tensor, got {type(inputs).__name__}"
            )
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

    def _validate_spectrum_inputs(self, inputs: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "AION spectrum input must be a mapping of flux/ivar/mask/wavelength "
                f"tensors, got {type(inputs).__name__}"
            )
        missing = set(AION_SPECTRUM_FIELDS).difference(inputs)
        if missing:
            raise ValueError(f"AION spectrum input is missing fields: {sorted(missing)}")

        tensors = {field: inputs[field] for field in AION_SPECTRUM_FIELDS}
        for field, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"AION spectrum field {field!r} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            if tensor.ndim != 2 or tuple(tensor.shape[1:]) != (self.raw_spectrum_length,):
                raise ValueError(
                    f"AION spectrum field {field!r} must have shape "
                    f"(B, {self.raw_spectrum_length}), got {tuple(tensor.shape)}"
                )

        if len({tuple(tensor.shape) for tensor in tensors.values()}) != 1:
            raise ValueError("AION spectrum fields must have identical batch and length shapes")
        if len({tensor.device for tensor in tensors.values()}) != 1:
            raise ValueError("AION spectrum fields must be on the same device")

        for field in ("flux", "ivar", "wavelength"):
            tensor = tensors[field]
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"AION spectrum field {field!r} must be floating point, got {tensor.dtype}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"AION spectrum field {field!r} contains non-finite values")
        if tensors["mask"].dtype != torch.bool:
            raise TypeError(
                f"AION spectrum field 'mask' must be boolean, got {tensors['mask'].dtype}"
            )
        if bool((tensors["ivar"] < 0).any()):
            raise ValueError("AION spectrum inverse variance must be non-negative")
        if bool((torch.diff(tensors["wavelength"], dim=-1) <= 0).any()):
            raise ValueError("AION spectrum wavelength must be strictly increasing")

    def _encode_tokens(
        self,
        *,
        token_key: str,
        token_ids: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        encoded_inputs = {token_key: token_ids}
        if self._shared_encoder_trainable:
            return self.aion_model.encode(
                encoded_inputs,
                num_encoder_tokens=token_count,
            )
        with torch.no_grad():
            return self.aion_model.encode(
                encoded_inputs,
                num_encoder_tokens=token_count,
            )

    def _encode_image(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_image_inputs(inputs)
        with torch.no_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
            codec_flux = inputs.detach().to(dtype=torch.float32).clone()
            image = self._legacy_survey_image_type(
                flux=codec_flux,
                bands=list(self.bands),
            )
            token_ids = self.image_codec.encode(image)

        expected_tokens = (inputs.shape[0], self.num_image_tokens)
        if tuple(token_ids.shape) != expected_tokens:
            raise RuntimeError(
                f"AION ImageCodec must emit token ids with shape {expected_tokens}, "
                f"got {tuple(token_ids.shape)}"
            )
        features = self._encode_tokens(
            token_key=self.image_token_key,
            token_ids=token_ids,
            token_count=self.num_image_tokens,
        )
        expected_features = (inputs.shape[0], self.num_image_tokens, self.output_dim)
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                f"AION shared encoder must emit image features with shape {expected_features}, "
                f"got {tuple(features.shape)}"
            )
        return features

    def _encode_spectrum(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self._validate_spectrum_inputs(inputs)
        reference = inputs["flux"]
        with torch.no_grad(), torch.autocast(
            device_type=reference.device.type,
            enabled=False,
        ):
            spectrum = self._desi_spectrum_type(
                flux=inputs["flux"].detach().to(dtype=torch.float32).clone(),
                ivar=inputs["ivar"].detach().to(dtype=torch.float32).clone(),
                mask=inputs["mask"].detach().clone(),
                wavelength=inputs["wavelength"].detach().to(dtype=torch.float32).clone(),
            )
            token_ids = self.spectrum_codec.encode(spectrum)

        expected_tokens = (reference.shape[0], self.num_spectrum_tokens)
        if tuple(token_ids.shape) != expected_tokens:
            raise RuntimeError(
                f"AION SpectrumCodec must emit token ids with shape {expected_tokens}, "
                f"got {tuple(token_ids.shape)}"
            )
        features = self._encode_tokens(
            token_key=self.spectrum_token_key,
            token_ids=token_ids,
            token_count=self.num_spectrum_tokens,
        )
        expected_features = (
            reference.shape[0],
            self.num_spectrum_tokens,
            self.output_dim,
        )
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                "AION shared encoder must emit spectrum features with shape "
                f"{expected_features}, got {tuple(features.shape)}"
            )
        return features

    def encode_modality(
        self,
        modality: str,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode one homogeneous modality batch through the shared owner."""

        self._enforce_trainability()
        modality_key = str(modality)
        if modality_key == "image":
            return self._encode_image(inputs)  # type: ignore[arg-type]
        if modality_key == "spectrum":
            return self._encode_spectrum(inputs)  # type: ignore[arg-type]
        raise ValueError(
            f"Unsupported AION modality {modality_key!r}; expected 'image' or 'spectrum'"
        )

    def forward(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        modality: str,
    ) -> torch.Tensor:
        """Route one modality through the shared AION encoder."""
        return self.encode_modality(modality, inputs)


__all__ = ["AIONMultimodalEncoder"]
