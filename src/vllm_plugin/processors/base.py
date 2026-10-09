"""ModalityProcessor ABC.

Owns one modality's vLLM-side contract:

  * `modality` — vLLM dispatch key in `mm_data` (e.g. "image", "time_series").
  * `placeholder_token` / `placeholder_token_id` — single special token the
    LLM tokenizer emits for one input item; PromptReplacement expands it to
    N feature tokens at processing time.
  * `mm_kwarg_key` — the `BatchFeature` field name carrying the encoded
    tensor (e.g. "pixel_values", "time_series").
  * `num_tokens(item)` — feature-token count for one input item. **Must
    match** the encoder's actual output length or vLLM raises a placeholder-
    count mismatch at splice time.
  * `encode(raw, **kwargs)` — raw input → encoded tensor with the contract
    shape the model class's encoder expects.
  * `dummy_item(...)` — worst-case input for memory profiling.
  * `field_config()` — `MultiModalFieldConfig` describing the encoded tensor
    layout (batched, flat, etc).
  * `normalize_mm_data_key(mm_data)` — accept synonyms callers may pass
    ("images" vs "image", "ts" vs "time_series") and dispatch to the canonical
    key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig


class ModalityProcessor(ABC):
    """Single-modality processor (image, time-series, geometry, dna)."""

    modality: str  # vLLM mm_data dispatch key
    mm_kwarg_key: str  # BatchFeature field name for the encoded tensor

    def __init__(
        self,
        modality: str,
        placeholder_token: str,
        placeholder_token_id: int,
        mm_kwarg_key: str,
        prism_subconfig: Mapping[str, Any] | None = None,
    ) -> None:
        self.modality = modality
        self.placeholder_token = placeholder_token
        self.placeholder_token_id = int(placeholder_token_id)
        self.mm_kwarg_key = mm_kwarg_key
        self.prism_subconfig = dict(prism_subconfig or {})

    # ------------------------------------------------------------------
    # Required overrides

    @abstractmethod
    def num_tokens(self, item: Any) -> int:
        """Feature-token count for one input item."""

    @abstractmethod
    def encode(self, raw: Any) -> Any:
        """Raw input → encoded tensor (the BatchFeature value)."""

    @abstractmethod
    def dummy_item(
        self,
        *,
        mm_options: Mapping[str, object] | None,
        count: int,
    ) -> Any:
        """Worst-case input(s) for vLLM's memory profiler."""

    @abstractmethod
    def field_config(self) -> MultiModalFieldConfig:
        """Layout of the encoded tensor inside the BatchFeature."""

    # ------------------------------------------------------------------
    # Encoder construction (model-plane)
    #
    # Called once at engine-boot time. Returns the inputs the model class
    # uses to build `_PrismEncoderWrapper(model=<inner>, proj=<dim_match>,
    # forward_fn=<closure>)`. Keeping this in the per-modality file means
    # the model class never needs an `if modality == "image"` branch.

    def build_encoder(
        self,
    ) -> tuple[Any, int, Any]:
        """Return (inner_module, hidden_size, forward_fn).

        Default raises — subclasses must override before VLLM-2 can wire
        them into the model class.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.build_encoder() not implemented"
        )

    def builds_complete_encoder(self) -> bool:
        """True iff `build_encoder()` returns the FULL encoder module.

        When True, the model class registers the inner module directly under
        `encoders[<modality>]` and skips `_PrismEncoderWrapper` — the wrapper
        would insert an extra `.model` level into the state-dict path that
        has no counterpart on the training side. Image returns False because
        its inner SigLIP2 doesn't expose PRISM's `.model/.proj` layout that
        `ImageEncoder.forward` expects, so the wrapper is required.

        Default False (the historical behavior). Subclasses whose
        `build_encoder().inner` is a complete encoder module override to True.
        """
        return False

    # ------------------------------------------------------------------
    # Optional overrides

    def normalize_mm_data_key(self, mm_data: Mapping[str, Any]) -> Any | None:
        """Return the raw input for this modality, accepting key synonyms."""
        # Default: the canonical key only.
        return mm_data.get(self.modality)

    def populate_batch_feature(
        self, out: dict[str, Any], raw: Any
    ) -> None:
        """Mutate `out` with this modality's encoded tensor under `mm_kwarg_key`.

        Default: store `encode(raw)`. Override only when the modality needs
        to emit multiple keys (e.g. an attention mask alongside the values).
        """
        out[self.mm_kwarg_key] = self.encode(raw)

    # ------------------------------------------------------------------
    # Helpers used by both ImageModalityProcessor and the dummy-inputs builder

    def get_dummy_text(self, count: int) -> str:
        """Concatenate `count` placeholder tokens. Used by the dummy builder."""
        return self.placeholder_token * count

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{type(self).__name__}(modality={self.modality!r}, "
            f"placeholder={self.placeholder_token!r}, id={self.placeholder_token_id})"
        )


__all__ = ["ModalityProcessor", "BatchFeature"]
