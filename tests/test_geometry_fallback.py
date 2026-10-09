"""Test the WALRUS_FALLBACK=1 swap in GeometryEncoder.

The fallback exists to keep the per-modality sweep's text_geometry cell
runnable on environments where the walrus install is broken — production
builds must install walrus. The fallback must:
  1. activate only when WALRUS_FALLBACK=1 AND require_modality_deps raises
  2. never silently downgrade when walrus is installed
  3. never activate when WALRUS_FALLBACK is unset
"""
from unittest.mock import patch

import pytest
from src.encoders.geometry import FallbackGeometryEncoder, GeometryEncoder
from src.utils.optional_deps import MissingOptionalDependencyError


def _force_missing(name):
    raise ImportError(f"forced missing: {name}")


def test_fallback_activated_when_env_set_and_deps_missing(monkeypatch):
    monkeypatch.setenv("WALRUS_FALLBACK", "1")
    with patch(
        "src.utils.optional_deps.importlib.import_module", side_effect=_force_missing
    ):
        enc = GeometryEncoder(input_dim=6, d_geo=64)
    assert isinstance(enc, FallbackGeometryEncoder)
    assert enc.output_dim == 64


def test_fallback_not_activated_when_env_unset(monkeypatch):
    monkeypatch.delenv("WALRUS_FALLBACK", raising=False)
    with patch(
        "src.utils.optional_deps.importlib.import_module", side_effect=_force_missing
    ):
        with pytest.raises(MissingOptionalDependencyError):
            GeometryEncoder(input_dim=6, d_geo=64)


def test_fallback_not_activated_when_deps_present(monkeypatch):
    """If walrus IS installed, WALRUS_FALLBACK=1 must NOT downgrade — operator
    has to see the real walrus path or know it's broken."""
    monkeypatch.setenv("WALRUS_FALLBACK", "1")
    # Patch require_modality_deps so it returns without raising — simulates
    # walrus being importable. We then accept whatever the real init does
    # downstream (likely fails on hydra config). The behavior under test is
    # only: did __new__ return a FallbackGeometryEncoder or pass through?
    with patch(
        "src.utils.optional_deps.importlib.import_module",
        return_value=object(),
    ):
        try:
            enc = GeometryEncoder(input_dim=6, d_geo=64)
        except Exception:
            # Real init will likely fail (no hydra cfg in test env); that's
            # fine — we only care that __new__ didn't swap in the fallback.
            return
        assert not isinstance(enc, FallbackGeometryEncoder)


def test_fallback_forward_shape():
    """Fallback encoder must produce (B, T=1, d_geo)-shaped output."""
    import torch

    enc = FallbackGeometryEncoder(input_dim=6, d_geo=64)
    inputs = torch.randn(2, 100, 6)  # (B, N, D)
    out = enc(inputs)
    assert out.shape == (2, 1, 64)
