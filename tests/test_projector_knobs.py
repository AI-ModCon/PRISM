"""IsoFLOP projector knobs: hidden_mult / num_layers + ckpt compat.

Verifies that:
1. Output shape is (B, T, d_model) for every variant.
2. Parameter count matches the closed form for each variant.
3. The legacy `(hidden_mult=1, num_layers=2)` path keeps the exact
   `fc1.*` / `fc2.*` state_dict keys so existing checkpoints still load.
4. Invalid configs raise.
"""

from __future__ import annotations

import pytest
import torch
from src.modules.projector import ModalityProjector


def _expected_params(input_dim: int, d_model: int, hidden_mult: int, num_layers: int) -> int:
    """Closed-form parameter count (matching the constructor)."""
    h = d_model * hidden_mult
    dims = (
        [(input_dim, h)]
        + [(h, h)] * (num_layers - 2)
        + [(h, d_model)]
    )
    return sum(in_dim * out_dim + out_dim for in_dim, out_dim in dims)  # weight + bias


@pytest.mark.parametrize(
    "hidden_mult,num_layers",
    [(1, 2), (2, 2), (4, 2), (1, 4), (1, 8)],
)
def test_projector_forward_shape_and_param_count(hidden_mult: int, num_layers: int) -> None:
    input_dim, d_model = 768, 2048
    proj = ModalityProjector(
        input_dim=input_dim,
        d_model=d_model,
        norm_mode="none",
        modality_embed_pos="none",
        hidden_mult=hidden_mult,
        num_layers=num_layers,
    )

    # Forward shape
    x = torch.randn(2, 16, input_dim)
    out = proj(x)
    assert out.shape == (2, 16, d_model)

    # Param count: closed form should match the actual Linear count
    # (modality embedding + layer-norm params are excluded by norm_mode=none
    # + embed_pos=none so only the Linears contribute).
    actual = sum(p.numel() for p in proj.parameters())
    expected = _expected_params(input_dim, d_model, hidden_mult, num_layers)
    assert actual == expected, f"hm={hidden_mult} nl={num_layers}: expected {expected}, got {actual}"


def test_base_state_dict_keys_match_legacy() -> None:
    """(1, 2) variant must keep fc1/fc2/act keys for checkpoint compat."""
    proj = ModalityProjector(
        input_dim=768,
        d_model=2048,
        norm_mode="layernorm",
        modality_embed_pos="after_norm",
        hidden_mult=1,
        num_layers=2,
    )
    keys = set(proj.state_dict().keys())
    # Exact match for the legacy keys; presence of `layers.*` would break
    # old checkpoints. Other keys (final_norm, modality_embedding) are
    # legitimate and shape-independent of (hm, nl).
    assert "fc1.weight" in keys
    assert "fc1.bias" in keys
    assert "fc2.weight" in keys
    assert "fc2.bias" in keys
    assert not any(k.startswith("layers.") for k in keys), keys


def test_non_base_uses_modulelist() -> None:
    proj = ModalityProjector(
        input_dim=768,
        d_model=2048,
        norm_mode="none",
        modality_embed_pos="none",
        hidden_mult=2,
        num_layers=2,
    )
    keys = set(proj.state_dict().keys())
    assert any(k.startswith("layers.") for k in keys), keys
    # No legacy fc1/fc2 keys
    assert "fc1.weight" not in keys


def test_variant_map_constants() -> None:
    """The IsoFLOP variant table must contain the 5 documented variants."""
    expected = {"BASE": (1, 2), "W2X": (2, 2), "W4X": (4, 2), "D2X": (1, 4), "D4X": (1, 8)}
    assert ModalityProjector.VARIANT_MAP == expected


@pytest.mark.parametrize(
    "hidden_mult,num_layers",
    [(0, 2), (1, 1), (-1, 2)],
)
def test_invalid_knobs_raise(hidden_mult: int, num_layers: int) -> None:
    with pytest.raises(ValueError):
        ModalityProjector(
            input_dim=768,
            d_model=2048,
            norm_mode="none",
            modality_embed_pos="none",
            hidden_mult=hidden_mult,
            num_layers=num_layers,
        )
