"""Login-node tests for VLLM-1.5: modality-aware checkpoint export.

Cover the pure functions (key/drop maps, state-dict remap, normalization)
without touching disk or HF downloads. Full re-export of an exported
checkpoint is in the compute-node smoke runner.
"""

from __future__ import annotations

import pytest
import torch
from src.vllm_plugin.checkpoint_export import (
    KNOWN_MODALITIES,
    _validate_active_modalities,
    build_drop_prefixes,
    build_key_prefix_map,
    remap_state_dict,
)


def _normalize_prism_config(cfg: dict) -> dict:
    """Import the plugin's normalizer lazily, skipping if vLLM is absent.

    `checkpoint_export` itself is vLLM-free (argparse/json/torch/safetensors/
    transformers), so the 14 key-map, remap and validate tests below run
    anywhere. Only `_normalize_prism_config` lives in
    `prism_for_conditional_generation`, which does `from vllm.config import
    VllmConfig` at module scope -- so importing it at the top of this file
    made the whole module fail to collect in any environment without vLLM,
    including CI (vLLM is an Aurora/XPU build, not on PyPI for CI's platform).

    Deferring the import to the five tests that actually need it keeps the
    other fourteen running. Sibling suites use the module-level form
    (`vllm = pytest.importorskip("vllm")` in tests/test_vllm_ts_encoder.py and
    friends) because there every test needs the engine; here most do not.
    """
    pytest.importorskip("vllm")
    from src.vllm_plugin.prism_for_conditional_generation import (
        _normalize_prism_config as _impl,
    )

    return _impl(cfg)


# ---------------------------------------------------------------------------
# build_key_prefix_map / build_drop_prefixes


def test_key_prefix_map_image_only_preserves_pr41_aliases():
    """Image-only must rename `encoders.image.` -> `vision_tower.` so
    PR #41 exports stay byte-identical."""
    pairs = build_key_prefix_map(["image"])
    pair_set = set(pairs)
    assert ("backbone.", "language_model.") in pair_set
    assert ("encoders.image.", "vision_tower.") in pair_set
    assert ("projectors.image.", "multi_modal_projector.") in pair_set


def test_key_prefix_map_multi_uses_module_dict_layout():
    """When more than image is active, image too goes into the ModuleDict
    layout that VLLM-2's `self.encoders` / `self.multi_modal_projectors`
    consume."""
    pairs = build_key_prefix_map(["image", "time_series"])
    pair_set = set(pairs)
    assert ("encoders.image.", "encoders.image.") in pair_set
    assert (
        "projectors.image.",
        "multi_modal_projectors.image.",
    ) in pair_set
    assert ("encoders.time_series.", "encoders.time_series.") in pair_set
    assert (
        "projectors.time_series.",
        "multi_modal_projectors.time_series.",
    ) in pair_set
    # Legacy vision_tower rename SHOULD NOT appear in multi-modality mode.
    assert ("encoders.image.", "vision_tower.") not in pair_set


def test_drop_prefixes_drops_text_and_inactive():
    drops = build_drop_prefixes(["image", "time_series"])
    assert "encoders.text." in drops
    assert "encoders.geometry." in drops
    assert "projectors.graph." in drops
    # Active modalities are NOT dropped.
    assert "encoders.image." not in drops
    assert "encoders.time_series." not in drops
    assert "projectors.image." not in drops
    assert "projectors.time_series." not in drops


def test_drop_prefixes_image_only():
    drops = build_drop_prefixes(["image"])
    for inactive in ("time_series", "geometry", "graph", "table"):
        assert f"encoders.{inactive}." in drops
        assert f"projectors.{inactive}." in drops


# ---------------------------------------------------------------------------
# remap_state_dict


def _t(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape)


def test_remap_image_only_matches_pr41():
    sd = {
        "module.backbone.model.embed_tokens.weight": _t((1000, 64)),
        "module.encoders.image.vision_model.layer0.weight": _t((8,)),
        "module.projectors.image.fc.weight": _t((64, 8)),
        "module.encoders.text.embed.weight": _t((1000, 64)),  # dropped
        "module.encoders.time_series.proj.weight": _t((8, 8)),  # dropped
    }
    out = remap_state_dict(sd, ["image"])
    assert "language_model.model.embed_tokens.weight" in out
    assert "vision_tower.vision_model.layer0.weight" in out
    assert "multi_modal_projector.fc.weight" in out
    assert "encoders.text.embed.weight" not in out
    assert "encoders.time_series.proj.weight" not in out


def test_remap_multi_modality_keeps_module_dict_keys():
    sd = {
        "backbone.model.embed_tokens.weight": _t((1000, 64)),
        "encoders.image.vision_model.layer0.weight": _t((8,)),
        "projectors.image.fc.weight": _t((64, 8)),
        "encoders.time_series.proj.weight": _t((8, 8)),
        "projectors.time_series.fc.weight": _t((64, 8)),
        "encoders.geometry.proj.weight": _t((4, 4)),  # dropped (not active)
    }
    out = remap_state_dict(sd, ["image", "time_series"])
    assert "language_model.model.embed_tokens.weight" in out
    assert "encoders.image.vision_model.layer0.weight" in out
    assert "multi_modal_projectors.image.fc.weight" in out
    assert "encoders.time_series.proj.weight" in out
    assert "multi_modal_projectors.time_series.fc.weight" in out
    # Inactive -> dropped.
    assert not any("geometry" in k for k in out)
    # Legacy aliases must NOT have been used.
    assert "vision_tower.vision_model.layer0.weight" not in out


def test_drop_prefixes_rpc_modality_drops_encoder_not_projector():
    """When time_series runs out-of-process (Intern-S2 sidecar), its
    encoder weights must be dropped from the export -- they no longer live
    in this process's nn.Module tree -- but its projector (the small
    trained connector) must still be kept."""
    drops = build_drop_prefixes(["image", "time_series"], rpc_modalities=["time_series"])
    assert "encoders.time_series." in drops
    assert "projectors.time_series." not in drops
    # Other active modalities are unaffected.
    assert "encoders.image." not in drops
    assert "projectors.image." not in drops


def test_drop_prefixes_without_rpc_modalities_unaffected():
    """Non-RPC callers (the default, everywhere today) must see identical
    behavior to before rpc_modalities existed."""
    drops_default = build_drop_prefixes(["image", "time_series"])
    drops_explicit_none = build_drop_prefixes(["image", "time_series"], rpc_modalities=None)
    drops_explicit_empty = build_drop_prefixes(["image", "time_series"], rpc_modalities=[])
    assert drops_default == drops_explicit_none == drops_explicit_empty
    assert "encoders.time_series." not in drops_default


def test_remap_state_dict_drops_time_series_encoder_when_rpc():
    sd = {
        "backbone.model.embed_tokens.weight": _t((1000, 64)),
        "encoders.image.vision_model.layer0.weight": _t((8,)),
        "projectors.image.fc.weight": _t((64, 8)),
        "encoders.time_series.proj.weight": _t((8, 8)),
        "projectors.time_series.fc.weight": _t((64, 8)),
    }
    out = remap_state_dict(sd, ["image", "time_series"], rpc_modalities=["time_series"])
    # Encoder weights for the RPC'd modality are gone.
    assert "encoders.time_series.proj.weight" not in out
    # The trained connector still loads normally -- only the encoder moved.
    assert "multi_modal_projectors.time_series.fc.weight" in out
    # Other modalities are unaffected.
    assert "encoders.image.vision_model.layer0.weight" in out
    assert "multi_modal_projectors.image.fc.weight" in out


def test_remap_multi_modality_keeps_module_dict_keys_without_rpc_flag():
    """The existing non-RPC case (rpc_modalities omitted) must remain
    byte-identical to test_remap_multi_modality_keeps_module_dict_keys --
    encoders.time_series.* survives when no RPC flag is set."""
    sd = {
        "backbone.model.embed_tokens.weight": _t((1000, 64)),
        "encoders.time_series.proj.weight": _t((8, 8)),
        "projectors.time_series.fc.weight": _t((64, 8)),
    }
    out = remap_state_dict(sd, ["time_series"])
    assert "encoders.time_series.proj.weight" in out
    assert "multi_modal_projectors.time_series.fc.weight" in out


def test_remap_strips_orig_mod_prefix():
    sd = {
        "_orig_mod.backbone.model.embed_tokens.weight": _t((1000, 64)),
        "_orig_mod.encoders.image.vision_model.weight": _t((8,)),
    }
    out = remap_state_dict(sd, ["image"])
    assert "language_model.model.embed_tokens.weight" in out
    assert "vision_tower.vision_model.weight" in out


# ---------------------------------------------------------------------------
# _normalize_prism_config


def test_normalize_pr41_flat_layout():
    """PR #41 prism_config has flat keys; normalize must synthesize an
    `image` block + an `active_modalities` list."""
    flat = {
        "image_encoder_model": "google/siglip2-base-patch16-224",
        "d_img": 768,
        "d_model": 2048,
        "image_token": "<image>",
        "image_token_id": 50300,
        "num_image_tokens": 196,
        "image_size": 224,
        "projector": {"norm_mode": "layernorm"},
    }
    out = _normalize_prism_config(flat)
    assert out["active_modalities"] == ["image"]
    assert out["image"]["placeholder_token"] == "<image>"
    assert out["image"]["placeholder_token_id"] == 50300
    assert out["image"]["encoder_model"] == "google/siglip2-base-patch16-224"
    assert out["image"]["d_img"] == 768
    assert out["image"]["num_image_tokens"] == 196
    assert out["image"]["image_size"] == 224
    # Flat keys are preserved (additive).
    assert out["image_token_id"] == 50300


def test_normalize_block_layout_idempotent():
    """If the `image` block already exists (VLLM-1.5 export), do not overwrite."""
    new = {
        "active_modalities": ["image", "time_series"],
        "image": {
            "placeholder_token": "<image>",
            "placeholder_token_id": 50300,
            "encoder_model": "google/siglip2-base-patch16-224",
            "num_image_tokens": 196,
            "image_size": 224,
            "d_img": 768,
        },
        "time_series": {
            "placeholder_token": "<time_series>",
            "placeholder_token_id": 50301,
        },
        "d_model": 2048,
    }
    out = _normalize_prism_config(new)
    assert out["active_modalities"] == ["image", "time_series"]
    assert out["image"]["placeholder_token_id"] == 50300
    assert out["time_series"]["placeholder_token_id"] == 50301
    # Image block was not overwritten.
    assert out["image"] is new["image"] or out["image"] == new["image"]


def test_normalize_handles_no_image():
    """Time-series-only checkpoint: no image flat keys; nothing to synthesize."""
    cfg = {
        "active_modalities": ["time_series"],
        "time_series": {
            "placeholder_token": "<time_series>",
            "placeholder_token_id": 50301,
        },
        "d_model": 2048,
    }
    out = _normalize_prism_config(cfg)
    assert "image" not in out
    assert out["active_modalities"] == ["time_series"]


def test_normalize_coerces_string_ints():
    """yaml/json round-trips can serialize numeric fields as strings; the
    normalized block must come back as ints so downstream `int(...)` casts
    aren't doing the work piecemeal."""
    flat = {
        "image_encoder_model": "google/siglip2-base-patch16-224",
        "d_img": "768",
        "d_model": 2048,
        "image_token": "<image>",
        "image_token_id": "50300",
        "num_image_tokens": "196",
        "image_size": "224",
        "projector": {"norm_mode": "layernorm"},
    }
    out = _normalize_prism_config(flat)
    assert out["image"]["d_img"] == 768
    assert isinstance(out["image"]["d_img"], int)
    assert out["image"]["num_image_tokens"] == 196
    assert isinstance(out["image"]["num_image_tokens"], int)
    assert out["image"]["image_size"] == 224
    assert isinstance(out["image"]["image_size"], int)
    assert out["image"]["placeholder_token_id"] == 50300
    assert isinstance(out["image"]["placeholder_token_id"], int)


# ---------------------------------------------------------------------------
# _validate_active_modalities


def test_validate_rejects_unknown_modality():
    with pytest.raises(ValueError, match="Unknown modality 'foo'"):
        _validate_active_modalities(["image", "foo"])


def test_validate_rejects_text():
    """`text` has no encoder/projector; passing it would silently produce a
    no-op block."""
    with pytest.raises(ValueError, match="not a separately-exportable"):
        _validate_active_modalities(["text", "image"])


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        _validate_active_modalities([])


def test_validate_dedupes_preserving_order():
    assert _validate_active_modalities(
        ["image", "time_series", "image"]
    ) == ["image", "time_series"]


def test_validate_accepts_all_known_non_text():
    expected = [m for m in KNOWN_MODALITIES if m != "text"]
    assert _validate_active_modalities(list(expected)) == expected


def test_validate_accepts_dna_reserved_for_bioreason():
    """`dna` is allowlisted ahead of the BioReason merge so future exports
    don't trip 'unknown modality' the day that branch lands. The actual
    block-builder branch will follow then; until then `export()` itself
    raises NotImplementedError for dna (covered separately)."""
    assert "dna" in KNOWN_MODALITIES
    assert _validate_active_modalities(["dna"]) == ["dna"]


# ---------------------------------------------------------------------------
# image_only set-equality (regression: list==["image"] was strict)


def test_key_prefix_map_duplicated_image_still_image_only():
    """`["image", "image"]` must take the legacy alias path; otherwise a
    sloppy caller would silently produce a multi-modality checkpoint that
    PR #41 readers can't load."""
    pairs = set(build_key_prefix_map(["image", "image"]))
    assert ("encoders.image.", "vision_tower.") in pairs
    assert ("projectors.image.", "multi_modal_projector.") in pairs
    # And it must NOT also emit the ModuleDict version.
    assert ("encoders.image.", "encoders.image.") not in pairs


# ---------------------------------------------------------------------------
# Round-trip: VLLM-1.5 image-only export should normalize to a shape that
# matches the per-modality block we'd build directly.


def test_image_only_block_layout_round_trips():
    """The flat keys VLLM-1.5 mirrors must survive a normalize() round-trip
    to the same per-modality block — proving the back-compat mirror in
    export() and the synthesis path in _normalize_prism_config agree."""
    # Simulate what export() writes for image-only: per-modality block PLUS
    # the flat keys that PR #41 callers may still read.
    written = {
        "active_modalities": ["image"],
        "d_model": 2048,
        "image": {
            "encoder_model": "google/siglip2-base-patch16-224",
            "placeholder_token": "<image>",
            "placeholder_token_id": 50300,
            "d_img": 768,
            "num_image_tokens": 196,
            "image_size": 224,
            "projector": {"norm_mode": "layernorm"},
        },
        "image_encoder_model": "google/siglip2-base-patch16-224",
        "image_token": "<image>",
        "image_token_id": 50300,
        "num_image_tokens": 196,
        "image_size": 224,
        "d_img": 768,
        "projector": {"norm_mode": "layernorm"},
        "language_model_arch_override": None,
    }
    out = _normalize_prism_config(written)
    # The block exists, was not overwritten, and matches the legacy flat keys.
    assert out["image"]["placeholder_token_id"] == out["image_token_id"]
    assert out["image"]["encoder_model"] == out["image_encoder_model"]
    assert out["image"]["d_img"] == out["d_img"]
    assert out["image"]["num_image_tokens"] == out["num_image_tokens"]
    assert out["image"]["image_size"] == out["image_size"]
