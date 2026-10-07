"""Modality-agnostic vLLM pipeline glue.

The three classes vLLM's MULTIMODAL_REGISTRY needs:

  * `PrismProcessingInfo`     — capability metadata + tokenizer access.
  * `PrismDummyInputsBuilder` — worst-case inputs for memory profiling.
  * `PrismMultiModalProcessor` — owns `_call_hf_processor`,
    `_get_mm_fields_config`, `_get_prompt_updates`.

All three dispatch through the per-modality `ModalityProcessor` set built by
`build_modality_processors(prism_cfg)`. The image-only behavior of the
original processor.py is preserved when `active_modalities == ["image"]`,
which is the default for PR #41 exports.

Some image-specific facades on `PrismProcessingInfo` (`get_image_token`,
`get_num_image_tokens`, `get_image_size_with_most_features`,
`get_max_image_tokens`) are kept and dispatch through the image processor;
they're load-bearing for downstream callers that imported them in PR #41.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from .base import ModalityProcessor
from .image import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_IMAGE_TOKENS,
    PRISM_IMAGE_TOKEN,
    ImageModalityProcessor,
)
from .registry import build_modality_processors, get_active_modalities


class PrismProcessingInfo(BaseProcessingInfo):
    """Capability metadata. Reads num image tokens / size from prism_config."""

    # NOTE: this object is reconstructed cheaply by vLLM; we cache the active
    # processor set on the instance once, lazily.

    _processors_cache: dict[str, ModalityProcessor] | None = None

    # ------------------------------------------------------------------
    # Modality processor accessors

    def _prism_config(self) -> dict:
        cfg = self.get_hf_config()
        prism_cfg = getattr(cfg, "prism_config", None)
        if prism_cfg is None:
            raise ValueError("HF config missing 'prism_config' block")
        return dict(prism_cfg)

    def modality_processors(self) -> dict[str, ModalityProcessor]:
        if self._processors_cache is None:
            self._processors_cache = build_modality_processors(self._prism_config())
        return self._processors_cache

    def get_modality_processor(self, modality: str) -> ModalityProcessor:
        return self.modality_processors()[modality]

    def active_modalities(self) -> list[str]:
        return get_active_modalities(self._prism_config())

    # ------------------------------------------------------------------
    # vLLM multimodal limits

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Per-modality limit-of-1 for non-image modalities (vLLM expects an
        # explicit limit; None means unbounded which only image supports today).
        limits: dict[str, int | None] = {}
        for modality in self.active_modalities():
            limits[modality] = None if modality == "image" else 1
        return limits

    # ------------------------------------------------------------------
    # Image-specific facades (load-bearing — see PR #41 callers).

    def get_image_token(self) -> str:
        try:
            return self.get_modality_processor("image").placeholder_token
        except KeyError:
            return self._prism_config().get("image_token", PRISM_IMAGE_TOKEN)

    def get_image_token_id(self) -> int:
        try:
            return int(self.get_modality_processor("image").placeholder_token_id)
        except KeyError:
            return int(self._prism_config()["image_token_id"])

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int
    ) -> int:
        try:
            proc = self.get_modality_processor("image")
        except KeyError:
            return int(
                self._prism_config().get(
                    "num_image_tokens", DEFAULT_NUM_IMAGE_TOKENS
                )
            )
        return int(proc.num_tokens(item=None))

    def get_image_size_with_most_features(self) -> ImageSize:
        try:
            proc = self.get_modality_processor("image")
            assert isinstance(proc, ImageModalityProcessor)
            size = proc.image_size
        except KeyError:
            size = int(self._prism_config().get("image_size", DEFAULT_IMAGE_SIZE))
        return ImageSize(width=size, height=size)

    def get_max_image_tokens(self) -> int:
        size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=size.width, image_height=size.height
        )


class PrismDummyInputsBuilder(BaseDummyInputsBuilder[PrismProcessingInfo]):
    """Worst-case inputs for vLLM memory profiling.

    Iterates over all active modalities and asks each processor for its
    dummy item. For image, we still use vLLM's helper PIL builder
    (`_get_dummy_images`) for compatibility with the size/overrides contract;
    other modalities synthesize their own dummies inside `dummy_item`.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        parts: list[str] = []
        for modality, count in mm_counts.items():
            try:
                proc = self.info.get_modality_processor(modality)
            except KeyError:
                continue
            if count <= 0:
                continue
            parts.append(proc.get_dummy_text(count))
        return "".join(parts)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        out: dict[str, object] = {}
        for modality, count in mm_counts.items():
            try:
                proc = self.info.get_modality_processor(modality)
            except KeyError:
                continue
            if count <= 0:
                continue
            if isinstance(proc, ImageModalityProcessor):
                size = self.info.get_image_size_with_most_features()
                overrides = (mm_options or {}).get("image")
                out["image"] = self._get_dummy_images(
                    width=size.width,
                    height=size.height,
                    num_images=count,
                    overrides=overrides,
                )
            else:
                out[modality] = proc.dummy_item(
                    mm_options=mm_options, count=count
                )
        return out


class PrismDataParser(MultiModalDataParser):
    """vLLM's default parser only knows image/audio/video/embeddings.

    Without this subclass, a raw 3-D time-series tensor would be routed
    through `is_embeddings` (which returns True for ndim==3) and silently
    treated as a pre-encoded embedding — wrong for raw time series and
    similarly fragile for future tensor-shaped modalities (geometry,
    voxel grids). We extend `_get_subparsers` to dispatch each PRISM
    modality key to its processor's `parse_*_data` subparser.
    """

    def _get_subparsers(self):
        subparsers = dict(super()._get_subparsers())
        # Local import keeps orchestrator import order independent of which
        # modality modules have already loaded; processors/__init__.py is
        # the canonical eager-import site for time_series factory registration.
        from . import time_series as _ts

        subparsers["time_series"] = _ts.parse_time_series_data
        return subparsers


class PrismMultiModalProcessor(BaseMultiModalProcessor[PrismProcessingInfo]):
    """Builds {input_ids, <kwarg_key>, ...} BatchFeature for all active modalities."""

    def _get_data_parser(self) -> MultiModalDataParser:
        # See PrismDataParser above; we always install the subclass so the
        # behavior is symmetric across modalities. The `expected_hidden_size`
        # plumbing matches vLLM 0.15's default impl.
        mm_config = self.info.ctx.model_config.get_multimodal_config()
        expected_hidden_size = None
        if getattr(mm_config, "enable_mm_embeds", False):
            expected_hidden_size = (
                self.info.ctx.model_config.get_inputs_embeds_size()
            )
        return PrismDataParser(expected_hidden_size=expected_hidden_size)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Our `_call_hf_processor` only tokenizes — it does NOT expand
        # placeholders into N copies. Returning False tells vLLM to run
        # PromptReplacement on the tokenized output.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        text_inputs = tokenizer(prompt, return_tensors="pt", **tok_kwargs)
        out: dict[str, Any] = {"input_ids": text_inputs["input_ids"]}

        for _modality, proc in self.info.modality_processors().items():
            raw = proc.normalize_mm_data_key(mm_data)
            if raw is None:
                continue
            proc.populate_batch_feature(out, raw)

        return BatchFeature(out)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            proc.mm_kwarg_key: proc.field_config()
            for proc in self.info.modality_processors().values()
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates: list[PromptUpdate] = []
        for modality, proc in self.info.modality_processors().items():
            updates.append(self._build_prompt_update(modality, proc, mm_items))
        return updates

    # ------------------------------------------------------------------
    # Per-modality PromptUpdate construction. Image uses ImageProcessorItems
    # so the replacement count can vary with image dimensions; non-image
    # modalities use a single fixed `num_tokens(item=None)`. Capture `proc`
    # by argument to avoid late-binding closure bugs over the loop variable.

    def _build_prompt_update(
        self,
        modality: str,
        proc: ModalityProcessor,
        mm_items: MultiModalDataItems,
    ) -> PromptUpdate:
        if isinstance(proc, ImageModalityProcessor):
            def get_replacement_image(item_idx: int, _proc=proc) -> list[int]:
                images = mm_items.get_items("image", ImageProcessorItems)
                size = images.get_image_size(item_idx)
                # ImageModalityProcessor.num_tokens currently ignores the size
                # (SigLIP2 is fixed at 196), but we pass it through so
                # subclasses that vary by resolution still work.
                n = _proc.num_tokens(
                    item={"width": size.width, "height": size.height}
                )
                return [_proc.placeholder_token_id] * n

            return PromptReplacement(
                modality="image",
                target=[proc.placeholder_token_id],
                replacement=get_replacement_image,
            )

        # Modalities with a start/end envelope (currently only time_series)
        # match the training-side data shape (src/data/multimodal.py:1348-1354):
        # the prompt template inserts `decode([start_id]) + decode([end_id])`
        # adjacent in input_ids. Training's UnifiedTransformer then drops
        # BOTH the start and end embeddings and replaces the start slot with
        # N feature embeddings (src/model.py:495-540: `slot_size[end_any]=0`,
        # `slot_size[start_mask]=tokens_per`, and `text_mask` excludes start
        # AND end).
        #
        # To replicate this in vLLM, target the adjacent `[start_id, end_id]`
        # pair and replace with `[ts_id]*N` where every position is an
        # embedding slot (no is_embed mask). After the splicer runs,
        # `merge_multimodal_embeddings` overwrites all N positions with the
        # encoder's output — start/end token embeddings never reach the LM,
        # matching training byte-for-byte.
        start_id = getattr(proc, "ts_start_id", None)
        end_id = getattr(proc, "ts_end_id", None)
        if start_id is not None and end_id is not None:
            def get_replacement_envelope(
                item_idx: int,
                _proc=proc,
            ) -> list[int]:
                n = _proc.num_tokens(item=None)
                return [_proc.placeholder_token_id] * n

            return PromptReplacement(
                modality=modality,
                target=[int(start_id), int(end_id)],
                replacement=get_replacement_envelope,
            )

        # Generic non-image, no-envelope path. Replacement count is fixed;
        # item is opaque.
        def get_replacement_generic(item_idx: int, _proc=proc) -> list[int]:
            n = _proc.num_tokens(item=None)
            return [_proc.placeholder_token_id] * n

        return PromptReplacement(
            modality=modality,
            target=[proc.placeholder_token_id],
            replacement=get_replacement_generic,
        )


__all__ = [
    "PrismDummyInputsBuilder",
    "PrismMultiModalProcessor",
    "PrismProcessingInfo",
]
