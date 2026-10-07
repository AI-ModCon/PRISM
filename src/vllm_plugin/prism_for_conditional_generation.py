"""vLLM model class for PRISM. Targets vLLM >= 0.15.

Mirrors the LlavaForConditionalGeneration pattern in vLLM but reuses PRISM's
own per-modality encoders and ModalityProjector verbatim so weights load 1:1
from PRISM checkpoints.

Forward path (image, kept for orientation):
    1. vLLM input processor expands the <image> placeholder into N image
       tokens (count == 196 for SigLIP2-base-patch16-224).
    2. embed_multimodal() runs the per-modality encoder + projector on the
       BatchFeature tensor to produce per-modality token embeddings.
    3. embed_input_ids() (from SupportsMultiModal) splices those into the
       language model's text embeddings at the placeholder positions.
    4. forward() delegates to the language model with merged inputs_embeds.

Multi-modality (VLLM-2+): the encoders and projectors live in nn.ModuleDicts
keyed by modality. For image-only checkpoints we additionally alias
`self.vision_tower` / `self.multi_modal_projector` to the image entries so
PR #41 export weight keys and external probes still work.
"""

from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

# Default placeholder when the exported config doesn't specify one.
PRISM_IMAGE_TOKEN = "<image>"


def _normalize_prism_config(prism_cfg: dict) -> dict:
    """Fold the PR #41 flat-key layout into the per-modality block layout.

    VLLM-1.5+ exports write both shapes (flat + per-modality block) so old
    callers keep working. PR #41 exports only have the flat shape; we
    synthesize an `image` block from them here so the rest of the plugin
    can assume the per-modality layout.

    Idempotent: if the block already exists, nothing is overwritten.
    """
    out = dict(prism_cfg)
    out.setdefault("active_modalities", ["image"])

    if "image" in out["active_modalities"] and "image" not in out:
        image_block: dict = {}
        if "image_token" in out:
            image_block["placeholder_token"] = out["image_token"]
        if "image_token_id" in out:
            image_block["placeholder_token_id"] = int(out["image_token_id"])
        # Mirror the int casts the legacy flat-key path applied at use sites
        # (see PrismForConditionalGeneration.__init__). yaml/json round-trips
        # have been observed to serialize these as strings; doing the cast
        # here means downstream code sees ints either way.
        _int_keys = {"d_img", "num_image_tokens", "image_size"}
        for src_key, dst_key in (
            ("image_encoder_model", "encoder_model"),
            ("d_img", "d_img"),
            ("num_image_tokens", "num_image_tokens"),
            ("image_size", "image_size"),
            ("projector", "projector"),
        ):
            if src_key in out:
                val = out[src_key]
                if src_key in _int_keys:
                    val = int(val)
                image_block[dst_key] = val
        if image_block:
            out["image"] = image_block

    return out


class _PrismEncoderWrapper(nn.Module):
    """One PRISM encoder slot.

    `model` is the inner per-modality module (SigLIP2 vision_model for image,
    TimeSeriesEncoder for time_series, ...). `proj` is an optional Linear /
    Identity used to match the inner hidden size to `d_img`-equivalent — for
    image SigLIP2 this is Identity. `_forward_fn(model, x) -> tensor`
    encapsulates the modality-specific call shape (image expects
    `model(pixel_values=x).last_hidden_state`; time_series expects
    `model(x)`).

    Defined at module scope (not nested) so vLLM's deepcopy passes don't
    blow the recursion limit traversing closures.
    """

    def __init__(
        self,
        model: nn.Module,
        proj: nn.Module,
        forward_fn: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.proj = proj
        # Stored as a plain attribute (not a Parameter / Module / Tensor), so
        # nn.Module.__setattr__ falls through to object.__setattr__ and the
        # closure is not registered as a child module or parameter.
        self._forward_fn = forward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._forward_fn(self.model, x))


# Back-compat alias: PR #41 callers (and our own load_weights documentation)
# referred to `_PrismVisionTower`. Keep the name working at module scope.
_PrismVisionTower = _PrismEncoderWrapper


class PrismForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """vLLM-compatible PRISM model.

    Expects the exported HF config to carry a ``prism_config`` block with
    either the PR #41 flat layout (image-only) or the VLLM-1.5+ per-modality
    layout. `_normalize_prism_config` reconciles them.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return PRISM_IMAGE_TOKEN
        # vLLM convention: return None for unsupported modalities so callers
        # can probe; raising would break engine-side dispatch.
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        # Defer heavy imports until construction so importing this module at
        # registration time is cheap. `no_init_weights` is a private but
        # stable HF helper used by every multi-modal model in vLLM; not part
        # of the public transformers __init__.
        from transformers.modeling_utils import no_init_weights

        from src.modules.projector import ModalityProjector

        from .processors import build_modality_processors
        from .processors.image import ImageModalityProcessor

        config = vllm_config.model_config.hf_config
        prism_cfg = getattr(config, "prism_config", None)
        if prism_cfg is None:
            raw = getattr(config, "_prism_config", None)
            prism_cfg = raw or config.to_dict().get("prism_config")
        if prism_cfg is None:
            raise ValueError(
                "PrismForConditionalGeneration requires a 'prism_config' block "
                "in the HF config (see src/vllm_plugin/checkpoint_export.py)."
            )

        self.config = config
        self.prism_config = _normalize_prism_config(dict(prism_cfg))

        # Materialize per-modality processors from the (now-normalized) config.
        # They own per-modality state (placeholder token + encoder factory) and
        # the model class never needs an `if modality == 'image'` branch from
        # here on.
        self._modality_processors = build_modality_processors(self.prism_config)
        self.active_modalities = list(self._modality_processors.keys())

        # image_token_id stays as a top-level attribute for the few places
        # that probe it (PR #41 callers, smoke runners).
        if "image" in self._modality_processors:
            self.image_token_id = int(
                self._modality_processors["image"].placeholder_token_id
            )

        d_model = int(self.prism_config["d_model"])

        # _mark_tower_model accepts set[str]; only one context manager is
        # needed for the whole tower stage. Image is special-cased in vLLM:
        # the (image, video) set is renamed to `vision_tower`. We use a
        # single-modality set so the marker covers exactly our tower
        # weights and the limit-per-prompt check (= 0 -> skip init) applies
        # per modality, not collectively. See interfaces.py:217.
        # mypy can't see _mark_*_model through SupportsMultiModal +
        # nn.Module multiple inheritance; silence the spurious operator error.
        encoder_modules: dict[str, nn.Module] = {}
        projector_modules: dict[str, nn.Module] = {}

        with self._mark_tower_model(  # type: ignore[operator]
            vllm_config, set(self.active_modalities)
        ), no_init_weights():
            for modality, proc in self._modality_processors.items():
                inner, hidden, forward_fn = proc.build_encoder()
                # SigLIP2 hidden == d_img so proj is Identity; non-matching
                # encoders get a Linear. This mirrors PRISM's own
                # ImageEncoder layout.
                d_target = int(proc.prism_subconfig.get("d_img", hidden))
                needs_proj = hidden != d_target
                # The processor declares whether its inner module is a
                # complete encoder. If so AND no projection is needed, skip
                # the wrapper — otherwise its extra `.model` level breaks
                # state-dict key alignment with training. When a projection
                # IS required, the wrapper is the only place to attach it,
                # so we wrap regardless.
                wrap = needs_proj or not proc.builds_complete_encoder()
                if wrap:
                    proj: nn.Module = (
                        nn.Identity()
                        if not needs_proj
                        else nn.Linear(hidden, d_target)
                    )
                    encoder_modules[modality] = _PrismEncoderWrapper(
                        inner, proj, forward_fn
                    )
                else:
                    encoder_modules[modality] = inner

                proj_kwargs = dict(
                    proc.prism_subconfig.get("projector")
                    or self.prism_config.get("projector")
                    or {}
                )
                proj_kwargs.setdefault("input_dim", d_target)
                proj_kwargs.setdefault("d_model", d_model)
                projector_modules[modality] = ModalityProjector(**proj_kwargs)

        # Register the ModuleDicts AFTER the marker context so the dict
        # itself isn't flagged as a tower stage — only its children are.
        self.encoders = nn.ModuleDict(encoder_modules)
        self.multi_modal_projectors = nn.ModuleDict(projector_modules)

        # Back-compat aliases (load-bearing): the PR #41 model class
        # exposed `self.vision_tower` and `self.multi_modal_projector`
        # as top-level submodules. Down-stream code (tests, smokes, anyone
        # probing the module tree) reads those names. When image is active
        # we point them at the corresponding ModuleDict entries; nn.Module
        # tolerates the same submodule reachable through multiple
        # attributes (it dedupes by id()).
        if isinstance(
            self._modality_processors.get("image"), ImageModalityProcessor
        ):
            self.vision_tower = self.encoders["image"]
            self.multi_modal_projector = self.multi_modal_projectors["image"]

        # Language model: a vLLM-registered backbone (e.g. Olmo2ForCausalLM).
        with self._mark_language_model(vllm_config):  # type: ignore[operator]
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=self.prism_config.get("language_model_arch_override"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ------------------------------------------------------------------
    # Multimodal interface (vLLM 0.15+)

    def _stack_per_modality(self, raw: Any) -> torch.Tensor:
        """Bring per-modality input tensors to a single batched tensor.

        Same logic as the original `_parse_pixel_values`, generalized so the
        time-series / geometry paths can share it.
        """
        if isinstance(raw, list):
            raw = torch.cat(
                [t if t.ndim == 4 else t.unsqueeze(0) for t in raw], dim=0
            )
        if raw.ndim == 5:  # (B, N, C, H, W) -> (B*N, C, H, W) for image
            raw = raw.flatten(0, 1)
        return cast(torch.Tensor, raw)

    def embed_multimodal(self, **kwargs: Any) -> MultiModalEmbeddings:
        """Encode every per-modality tensor present in `kwargs` and return
        the concatenated MultiModalEmbeddings for vLLM to splice.

        Fast path: single-image-only checkpoint returns the
        (N, T, d_model) tensor directly — image parity baseline holds bit-
        identically with PR #41.
        """
        # Single-image fast path (PR #41 behavior).
        if (
            len(self.encoders) == 1
            and "image" in self.encoders
            and "pixel_values" in kwargs
        ):
            pixel_values = kwargs.get("pixel_values")
            if pixel_values is None:
                return []
            pixel_values = self._stack_per_modality(pixel_values)
            tower = self.vision_tower
            target_dtype = next(tower.parameters()).dtype
            target_device = next(tower.parameters()).device
            pixel_values = pixel_values.to(
                dtype=target_dtype, device=target_device
            )
            feats = tower(pixel_values)
            embeds = self.multi_modal_projector(feats)
            return embeds

        # General multi-modality path.
        all_embeds: list[torch.Tensor] = []
        for modality, proc in self._modality_processors.items():
            raw = kwargs.get(proc.mm_kwarg_key)
            if raw is None:
                continue
            raw = self._stack_per_modality(raw)
            enc = self.encoders[modality]
            target_dtype = next(enc.parameters()).dtype
            target_device = next(enc.parameters()).device
            raw = raw.to(dtype=target_dtype, device=target_device)
            feats = enc(raw)
            embeds = self.multi_modal_projectors[modality](feats)
            all_embeds.append(embeds)

        if not all_embeds:
            return []
        # MultiModalEmbeddings is `list[Tensor] | Tensor | tuple[Tensor, ...]`
        # (vllm/model_executor/models/interfaces.py:49). The splicer ultimately
        # calls _flatten_embeddings, which accepts all three shapes. We return
        # a bare tensor in the 1-modality case (cheaper, matches image-only
        # fast path) and a list when multiple modalities are present. Do NOT
        # normalize to "always list" — the single-tensor return is part of the
        # parity contract with PR #41 callers.
        if len(all_embeds) == 1:
            return all_embeds[0]
        return all_embeds

    # ------------------------------------------------------------------
    # Forward / logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # ------------------------------------------------------------------
    # Weight loading
    #
    # checkpoint_export.py remaps PRISM's keys into one of two shapes:
    #
    #   image-only (PR #41 baseline, byte-identical export):
    #     backbone.*           -> language_model.*
    #     encoders.image.*     -> vision_tower.*
    #     projectors.image.*   -> multi_modal_projector.*
    #
    #   multi-modality (VLLM-1.5+):
    #     backbone.*           -> language_model.*
    #     encoders.<m>.*       -> encoders.<m>.*
    #     projectors.<m>.*     -> multi_modal_projectors.<m>.*
    #
    # The back-compat alias (`self.vision_tower = self.encoders["image"]`)
    # makes both shapes load via the SAME loader because AutoWeightsLoader
    # finds the modules under their canonical attribute names.

    # PR #41 exports key weights as `vision_tower.*` / `multi_modal_projector.*`.
    # VLLM-2 made the canonical attribute path `encoders.image.*` /
    # `multi_modal_projectors.image.*` (with the legacy names kept as aliases
    # pointing at the same submodules). AutoWeightsLoader walks
    # `named_modules()` and sees the shared submodule under BOTH paths, so it
    # expects both naming schemes in the checkpoint. We translate the legacy
    # names to canonical here so PR #41 checkpoints continue to load with no
    # re-export required.
    _LEGACY_KEY_RENAMES: tuple[tuple[str, str], ...] = (
        ("vision_tower.", "encoders.image."),
        ("multi_modal_projector.", "multi_modal_projectors.image."),
    )

    def _canonicalize_weight_key(self, name: str) -> str:
        for src, dst in self._LEGACY_KEY_RENAMES:
            if name.startswith(src):
                return dst + name[len(src):]
        return name

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        translated = (
            (self._canonicalize_weight_key(name), tensor)
            for name, tensor in weights
        )
        return loader.load_weights(translated)
