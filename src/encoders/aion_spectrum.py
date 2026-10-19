"""AION raw-spectrum encoder for AstroPRISM.

The spectrum path mirrors the released AION feature pipeline:

``DESI flux/ivar/mask/wavelength -> SpectrumCodec -> 273 token ids -> AION.encode``

The codec contains hard quantization, so it is always kept in float32,
frozen, and in evaluation mode.  The DESI token embedding and shared AION
transformer can be unfrozen later with ``set_shared_encoder_trainable``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .aion_image import AION_MODEL_REVISION, AION_REPO_ID
from .base import ModalityEncoder

AION_SPECTRUM_CODEC_REVISION = AION_MODEL_REVISION
AION_SPECTRUM_TOKENS = 273
AION_SPECTRUM_WIDTH = 768
AION_RAW_SPECTRUM_LENGTH = 7781
AION_SPECTRUM_FIELDS = ("flux", "ivar", "mask", "wavelength")


def _load_aion_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Import optional AION spectrum classes only when requested."""

    try:
        from aion import AION
        from aion.codecs import SpectrumCodec
        from aion.modalities import DESISpectrum
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "AIONSpectrumEncoder requires the optional 'polymathic-aion' package "
            "(and its dependencies). Install a pinned AION source checkout before "
            "selecting the AION spectrum encoder."
        ) from exc
    return AION, SpectrumCodec, DESISpectrum


class AIONSpectrumEncoder(ModalityEncoder):
    """Encode observed-frame DESI spectra with the pinned AION spectrum path.

    Input is a mapping with ``flux``, ``ivar``, ``mask``, and ``wavelength``
    tensors.  Every tensor must have shape ``(B, 7781)``; the first, second,
    and fourth are floating point and ``mask`` is boolean.

    Output has shape ``(B, 273, 768)``.
    """

    def __init__(
        self,
        model_name_or_path: str = AION_REPO_ID,
        *,
        model_revision: str = AION_MODEL_REVISION,
        codec_revision: str = AION_SPECTRUM_CODEC_REVISION,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        raw_spectrum_length: int = AION_RAW_SPECTRUM_LENGTH,
    ) -> None:
        super().__init__(AION_SPECTRUM_WIDTH)

        if int(raw_spectrum_length) <= 0:
            raise ValueError(
                f"raw_spectrum_length must be a positive integer, got {raw_spectrum_length!r}"
            )

        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = str(model_revision)
        self.codec_revision = str(codec_revision)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.local_files_only = bool(local_files_only)
        self.raw_spectrum_length = int(raw_spectrum_length)
        self.num_spectrum_tokens = AION_SPECTRUM_TOKENS
        self.hidden_dim = AION_SPECTRUM_WIDTH

        AION, SpectrumCodec, DESISpectrum = _load_aion_types()
        self._desi_spectrum_type = DESISpectrum
        self.spectrum_token_key = str(DESISpectrum.token_key)

        common_load_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        self.aion_model = AION.from_pretrained(
            self.model_name_or_path,
            revision=self.model_revision,
            **common_load_kwargs,
        )
        self.spectrum_codec = SpectrumCodec.from_pretrained(
            self.model_name_or_path,
            modality=DESISpectrum,
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
        if self.spectrum_token_key not in self.aion_model.encoder_embeddings:
            raise RuntimeError(
                f"Loaded AION model has no encoder embedding for {self.spectrum_token_key!r}"
            )

    def _shared_encoder_modules(self) -> tuple[nn.Module, ...]:
        """Modules used by ``AION.encode`` for a spectrum-only pass."""

        return (
            self.aion_model.encoder_embeddings[self.spectrum_token_key],
            self.aion_model.encoder,
            self.aion_model.encoder_norm,
            self.aion_model.decoder_proj_context,
        )

    def _enforce_trainability(self) -> None:
        for parameter in self.aion_model.parameters():
            parameter.requires_grad_(False)
        if self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        self.spectrum_codec.float()
        self.spectrum_codec.eval()
        for parameter in self.spectrum_codec.parameters():
            parameter.requires_grad_(False)

    @property
    def shared_encoder_trainable(self) -> bool:
        """Whether the shared AION transformer is currently trainable."""
        return self._shared_encoder_trainable

    def set_shared_encoder_trainable(self, trainable: bool) -> AIONSpectrumEncoder:
        """Toggle only the differentiable AION spectrum/shared-encoder path."""

        self._shared_encoder_trainable = bool(trainable)
        self._enforce_trainability()
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> AIONSpectrumEncoder:
        """Set runtime mode while preserving codec dtype/freeze invariants."""

        super().train(mode)
        self._enforce_trainability()
        self.aion_model.eval()
        if mode and self._shared_encoder_trainable:
            for module in self._shared_encoder_modules():
                module.train(True)
        self.spectrum_codec.eval()
        return self

    def tokens_per_instance(self) -> int:
        """Return the fixed number of spectrum tokens the AION codec emits."""
        return self.num_spectrum_tokens

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor]) -> None:
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
            expected_tail = (self.raw_spectrum_length,)
            if tensor.ndim != 2 or tuple(tensor.shape[1:]) != expected_tail:
                raise ValueError(
                    f"AION spectrum field {field!r} must have shape "
                    f"(B, {self.raw_spectrum_length}), got {tuple(tensor.shape)}"
                )

        shapes = {tuple(tensor.shape) for tensor in tensors.values()}
        if len(shapes) != 1:
            raise ValueError("AION spectrum fields must have identical batch and length shapes")
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
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

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode flux/ivar/mask/wavelength into ``(B, 273, 768)`` features."""
        self._validate_inputs(inputs)
        self._enforce_trainability()

        reference = inputs["flux"]
        with torch.no_grad(), torch.autocast(device_type=reference.device.type, enabled=False):
            spectrum = self._desi_spectrum_type(
                flux=inputs["flux"].detach().to(dtype=torch.float32).clone(),
                ivar=inputs["ivar"].detach().to(dtype=torch.float32).clone(),
                mask=inputs["mask"].detach().clone(),
                wavelength=inputs["wavelength"].detach().to(dtype=torch.float32).clone(),
            )
            spectrum_token_ids = self.spectrum_codec.encode(spectrum)

        expected_tokens = (reference.shape[0], self.num_spectrum_tokens)
        if tuple(spectrum_token_ids.shape) != expected_tokens:
            raise RuntimeError(
                f"AION SpectrumCodec must emit token ids with shape {expected_tokens}, "
                f"got {tuple(spectrum_token_ids.shape)}"
            )

        encoded_inputs = {self.spectrum_token_key: spectrum_token_ids}
        if self._shared_encoder_trainable:
            features = self.aion_model.encode(
                encoded_inputs,
                num_encoder_tokens=self.num_spectrum_tokens,
            )
        else:
            with torch.no_grad():
                features = self.aion_model.encode(
                    encoded_inputs,
                    num_encoder_tokens=self.num_spectrum_tokens,
                )

        expected_features = (reference.shape[0], self.num_spectrum_tokens, self.output_dim)
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                f"AION shared encoder must emit features with shape {expected_features}, "
                f"got {tuple(features.shape)}"
            )
        return features


__all__ = [
    "AIONSpectrumEncoder",
    "AION_MODEL_REVISION",
    "AION_RAW_SPECTRUM_LENGTH",
    "AION_REPO_ID",
    "AION_SPECTRUM_CODEC_REVISION",
    "AION_SPECTRUM_FIELDS",
    "AION_SPECTRUM_TOKENS",
    "AION_SPECTRUM_WIDTH",
]
