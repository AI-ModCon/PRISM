"""Loud-fail helpers for optional modality dependencies.

PRISM's graph and geometry encoders depend on packages (`torch_geometric`,
`walrus`, `hydra-core`, `the_well`) that aren't part of the base install.
Previously, a missing package silently set `GraphEncoder = None` and the
model was built without graph support, producing partial-but-broken state.

This module provides a single `require_modality_deps(modality)` entry point
that raises `MissingOptionalDependencyError` with an install hint at the
moment a modality is actually requested.
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

from src.modalities import Modality


class MissingOptionalDependencyError(ImportError):
    """Raised when a modality is requested but its optional deps are missing."""


class _Requirement(NamedTuple):
    package: str
    install_hint: str


_MODALITY_REQUIREMENTS: dict[Modality, tuple[_Requirement, ...]] = {
    Modality.GRAPH: (
        _Requirement(
            package="torch_geometric",
            install_hint="pip install torch_geometric",
        ),
    ),
    Modality.GEOMETRY: (
        _Requirement(
            package="walrus",
            install_hint="pip install -e src/libs/walrus",
        ),
        _Requirement(
            package="hydra",
            install_hint="pip install hydra-core",
        ),
        _Requirement(
            package="the_well",
            install_hint="pip install the_well",
        ),
    ),
}


def _missing(requirements: tuple[_Requirement, ...]) -> list[tuple[_Requirement, str]]:
    # Actually import each package so transitive ImportErrors (e.g. a broken
    # walrus install whose own dependency is missing) surface here too. The
    # alternative — find_spec — only proves the package is locatable on
    # sys.path, not that `import pkg` will succeed.
    failures: list[tuple[_Requirement, str]] = []
    for r in requirements:
        try:
            importlib.import_module(r.package)
        except ImportError as e:
            failures.append((r, str(e)))
    return failures


def require_modality_deps(modality: Modality | str) -> None:
    """Raise MissingOptionalDependencyError if any optional dep is missing.

    Modalities without optional deps (text, image, table, time_series) are
    no-ops; this function is safe to call for all six.
    """
    if isinstance(modality, str):
        modality = Modality(modality)
    requirements = _MODALITY_REQUIREMENTS.get(modality)
    if not requirements:
        return
    failures = _missing(requirements)
    if not failures:
        return
    pkgs = ", ".join(r.package for r, _ in failures)
    hints = "\n  ".join(r.install_hint for r, _ in failures)
    details = "\n  ".join(f"{r.package}: {err}" for r, err in failures)
    raise MissingOptionalDependencyError(
        f"Modality {modality.value!r} requires optional dependencies that "
        f"failed to import: {pkgs}. Install with:\n  {hints}\n"
        f"Underlying import errors:\n  {details}\n"
        f"Or remove {modality.value!r} from model.modalities."
    )
