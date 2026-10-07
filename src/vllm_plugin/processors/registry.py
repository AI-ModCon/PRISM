"""Modality processor dispatch.

`MODALITY_PROCESSORS` is a name -> factory map. `build_modality_processors`
reads `prism_config["active_modalities"]` (defaulting to ["image"] for
back-compat with PR #41 exports) and instantiates one processor per active
modality.

Other modules register their processors at import time:

    @register_modality_processor("time_series")
    def _make_time_series(prism_cfg):
        return TimeSeriesModalityProcessor(...)

VLLM-4 wires this for time-series.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import ModalityProcessor
from .image import ImageModalityProcessor

ProcessorFactory = Callable[[Mapping[str, Any]], ModalityProcessor]


MODALITY_PROCESSORS: dict[str, ProcessorFactory] = {}


def register_modality_processor(modality: str) -> Callable[[ProcessorFactory], ProcessorFactory]:
    def decorator(factory: ProcessorFactory) -> ProcessorFactory:
        MODALITY_PROCESSORS[modality] = factory
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Built-in: image
#
# Resolves placeholder_token / placeholder_token_id from either the new
# per-modality block (prism_config["image"]) written by VLLM-1.5, or the
# legacy flat keys (prism_config["image_token"], etc.) so PR #41 exports keep
# loading without re-export.


@register_modality_processor("image")
def _build_image_processor(prism_cfg: Mapping[str, Any]) -> ImageModalityProcessor:
    sub = dict(prism_cfg.get("image", {}))
    placeholder = sub.get("placeholder_token") or prism_cfg.get("image_token") or "<image>"
    placeholder_id = sub.get("placeholder_token_id")
    if placeholder_id is None:
        placeholder_id = prism_cfg["image_token_id"]
    # Inherit legacy flat keys so encode() finds image_size/num_image_tokens.
    for legacy_key in ("image_size", "num_image_tokens"):
        if legacy_key in prism_cfg and legacy_key not in sub:
            sub[legacy_key] = prism_cfg[legacy_key]
    return ImageModalityProcessor(
        placeholder_token=str(placeholder),
        placeholder_token_id=int(placeholder_id),
        prism_subconfig=sub,
    )


def get_active_modalities(prism_cfg: Mapping[str, Any]) -> list[str]:
    """Read the active modality list from prism_config, defaulting to ['image'].

    VLLM-1.5 writes `active_modalities: [...]` into prism_config; until then
    (PR #41 exports) we infer the single image modality.
    """
    active = prism_cfg.get("active_modalities")
    if active:
        return [str(m) for m in active]
    return ["image"]


def build_modality_processors(
    prism_cfg: Mapping[str, Any],
) -> dict[str, ModalityProcessor]:
    """Instantiate one processor per active modality."""
    out: dict[str, ModalityProcessor] = {}
    for modality in get_active_modalities(prism_cfg):
        factory = MODALITY_PROCESSORS.get(modality)
        if factory is None:
            raise KeyError(
                f"No ModalityProcessor registered for {modality!r}; known: "
                f"{sorted(MODALITY_PROCESSORS)}"
            )
        out[modality] = factory(prism_cfg)
    return out


__all__ = [
    "MODALITY_PROCESSORS",
    "ProcessorFactory",
    "build_modality_processors",
    "get_active_modalities",
    "register_modality_processor",
]
