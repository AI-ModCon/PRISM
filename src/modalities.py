"""Central registry for PRISM modality identifiers.

The `Modality` enum is the single source of truth for the strings used as
keys in `config.modalities`, `encoders`/`projectors` dicts, batch examples,
and `datasets_config.json`'s `modality` field.

Subclassing `(str, Enum)` (rather than `enum.StrEnum`, which is 3.11+) keeps
equality with plain strings: `Modality.IMAGE == "image"` is True, and
`d[Modality.IMAGE]` retrieves the same value as `d["image"]`. This means
existing CLI overrides (`model.modalities=[text,image]`) and YAML configs
continue to work without changes.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Modality(str, Enum):
    """The modality identifiers PRISM recognizes.

    Members are the canonical lowercase strings used as keys throughout the
    framework: ``config.modalities``, the ``encoders``/``projectors``
    ModuleDicts on ``UnifiedTransformer``, batch example dicts, and the
    ``modality`` field in ``datasets_config.json``.

    Because the enum subclasses ``str``, a member compares equal to and
    hashes like its value, so ``Modality.IMAGE == "image"`` is True and
    ``d[Modality.IMAGE]`` and ``d["image"]`` reach the same entry. The
    explicit ``__str__`` below -- not the ``str`` base -- is what makes
    ``str(Modality.IMAGE)`` render as ``"image"`` instead of
    ``"Modality.IMAGE"``; removing it would change f-string and log output.

    Use ``parse_modality`` to coerce an arbitrary string; ``ALL_MODALITIES``
    is the tuple of every member, in declaration order.
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    TIME_SERIES = "time_series"
    GEOMETRY = "geometry"
    GRAPH = "graph"
    DNA = "dna"

    def __str__(self) -> str:
        return self.value


ALL_MODALITIES: tuple[Modality, ...] = tuple(Modality)


def parse_modality(value: str | Modality) -> Modality:
    """Coerce a string or Modality to a Modality, raising for unknown values."""
    if isinstance(value, Modality):
        return value
    try:
        return Modality(value)
    except ValueError as e:
        known = ", ".join(m.value for m in Modality)
        raise ValueError(f"Unknown modality {value!r}. Known: {known}") from e


def make_dummy_batch(
    modality: Modality | str,
) -> torch.Tensor | dict[str, torch.Tensor | str] | None:
    """Construct a single dummy sample tensor (or dict) matching the modality's
    on-the-wire shape — the one that StreamingMultimodalDataset places into
    `example[modality]` before tokenization.

    Used by `StreamingMultimodalDataset._dummy_generator` when a dataset's
    real shards are missing and `fallback_dummy=True`. Centralized here so
    shape choices are owned next to the enum, not scattered across the
    handler cascade.

    Returns only the tensor/dict — text captions are owned by the caller
    (the tokenizer path), unlike the `(tensor, text)` pairs returned by
    `_process_image` / `_process_geo_pde` / etc.

    Shapes (what PRISM's per-modality encoders expect at their input):
    - text: handled by tokenizer downstream (returns None here)
    - image: (3, 224, 224) — SigLIP2 input shape, matches `_process_image`
      `default_tensor` in src/data/multimodal.py
    - table: (128,) int64 token ids — TAPAS input length, matches
      `default_ids` in `_process_table_*`
    - time_series: (64, 1) float — Moirai single-channel window (the
      `vals` tensor inside `_ts_generator`'s yielded dict)
    - geometry: (1, 10) float — matches `_process_geo_pde` dummy return
    - graph: dict with 'x' (128, 32) and empty edge_index — Walrus/PyG
      schema used by the graph encoder (`_process_graph` raises on
      missing input rather than returning a dummy, so this shape is
      defined here rather than copied from there)
    - dna: dict of raw strings (reference/variant sequence + question/answer)
      — the DNA encoder tokenizes text inputs itself, so the dummy is a
      placeholder text example rather than a pre-tokenized tensor, matching
      the shape KEGG/variant-effect examples take before encoding
    """
    import torch  # local import keeps modalities.py import-cheap

    m = parse_modality(modality)
    if m is Modality.TEXT:
        return None
    if m is Modality.IMAGE:
        return torch.zeros(3, 224, 224)
    if m is Modality.TABLE:
        return torch.zeros(128, dtype=torch.long)
    if m is Modality.TIME_SERIES:
        return torch.zeros(64, 1)
    if m is Modality.GEOMETRY:
        return torch.zeros(1, 10)
    if m is Modality.GRAPH:
        return {"x": torch.zeros(128, 32), "edge_index": torch.empty((2, 0), dtype=torch.long)}
    if m is Modality.DNA:
        return {
            "reference_sequence": "CTGA",
            "variant_sequence": "CTGA",
            "question": "What disease does this sequence exhibit?",
            "answer": "No DNA sequence information available.",
        }
    raise AssertionError(f"Unhandled modality {m!r}")  # exhaustive
