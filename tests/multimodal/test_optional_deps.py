"""Tests for the loud-fail optional-dep gate (src/utils/optional_deps.py)."""
from unittest.mock import patch

import pytest
from src.modalities import Modality
from src.utils.optional_deps import (
    MissingOptionalDependencyError,
    require_modality_deps,
)


def _raise_import_error(name):
    raise ImportError(f"No module named {name!r}")


def test_text_modality_is_noop():
    """Modalities without optional deps should not raise."""
    require_modality_deps(Modality.TEXT)
    require_modality_deps(Modality.IMAGE)
    require_modality_deps(Modality.TABLE)
    require_modality_deps(Modality.TIME_SERIES)


def test_accepts_string_input():
    """API takes either Modality or str."""
    require_modality_deps("text")


def test_graph_raises_when_torch_geometric_missing():
    with patch(
        "src.utils.optional_deps.importlib.import_module",
        side_effect=_raise_import_error,
    ):
        with pytest.raises(MissingOptionalDependencyError, match="torch_geometric"):
            require_modality_deps(Modality.GRAPH)


def test_geometry_raises_lists_all_missing_deps():
    """Error message should enumerate every missing package so the user can install in one go."""
    with patch(
        "src.utils.optional_deps.importlib.import_module",
        side_effect=_raise_import_error,
    ):
        with pytest.raises(MissingOptionalDependencyError) as excinfo:
            require_modality_deps(Modality.GEOMETRY)
        msg = str(excinfo.value)
        for pkg in ("walrus", "hydra", "the_well"):
            assert pkg in msg


def test_error_mentions_install_hint():
    with patch(
        "src.utils.optional_deps.importlib.import_module",
        side_effect=_raise_import_error,
    ):
        with pytest.raises(MissingOptionalDependencyError, match="pip install"):
            require_modality_deps(Modality.GRAPH)


def test_does_not_raise_when_all_present():
    with patch(
        "src.utils.optional_deps.importlib.import_module",
        return_value=object(),
    ):
        require_modality_deps(Modality.GRAPH)
        require_modality_deps(Modality.GEOMETRY)


def test_transitive_import_failure_is_caught():
    """A package whose top-level import raises (e.g. broken install) should
    surface as MissingOptionalDependencyError, not crash through. Underlying
    error message should appear in the raised exception for debuggability."""

    def broken_walrus(name):
        if name == "walrus":
            raise ImportError("libfoo.so: cannot open shared object file")
        return object()

    with patch(
        "src.utils.optional_deps.importlib.import_module",
        side_effect=broken_walrus,
    ):
        with pytest.raises(MissingOptionalDependencyError) as excinfo:
            require_modality_deps(Modality.GEOMETRY)
        assert "libfoo.so" in str(excinfo.value)
        assert "walrus" in str(excinfo.value)
