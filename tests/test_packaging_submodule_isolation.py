"""The wheel's contents must not depend on whether submodules are checked out.

`docs/getting-started.md` tells users to `git clone --recursive`, which
populates `src/libs/walrus` -- the MIT-licensed Polymathic AI submodule. Two
separate setuptools mechanisms will happily sweep that tree into our
distribution:

1. `[tool.setuptools.packages.find]` defaults to `namespaces = true`, so every
   directory under `src/` is discovered as a namespace package.
2. `[tool.setuptools.package-data]` globs are matched against the *directory*,
   not against the discovered package list, so an unanchored `"**/*.yaml"`
   reaches into the submodule even when `packages.find` excludes it.

Either one alone shipped walrus files without their LICENSE, and made the same
`python -m build` produce a different wheel depending on submodule state. This
test is deliberately install-free and parses the config by hand: CI's 3.10 leg
has no `tomllib`, and adding a TOML dependency to make a packaging guard work
is how the guard stops running.
"""

from __future__ import annotations

import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_ROOT, "pyproject.toml")


def _section(name: str) -> str:
    """Return the raw text of a single top-level `[name]` table."""
    with open(_PYPROJECT, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index(f"[{name}]") + len(f"[{name}]")
    rest = text[start:]
    end = re.search(r"^\[", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def test_packages_find_excludes_the_walrus_submodule() -> None:
    """Namespace discovery must not treat `src/libs/walrus` as ours to ship."""
    find = _section("tool.setuptools.packages.find")
    assert "src.libs" in find, (
        "[tool.setuptools.packages.find] has no `src.libs` exclusion. With the "
        "default `namespaces = true`, a recursive clone makes setuptools "
        "discover the walrus submodule as a namespace package and ship 52 of "
        "its .py files in our wheel, without its LICENSE."
    )


def test_package_data_globs_are_anchored() -> None:
    """Unanchored recursive globs walk into submodules regardless of `exclude`."""
    data = _section("tool.setuptools.package-data")
    unanchored = re.findall(r'"\*\*/[^"]+"', data)
    assert not unanchored, (
        f"[tool.setuptools.package-data] has unanchored recursive globs "
        f"{unanchored}. These match against the directory tree rather than the "
        f"discovered package list, so they reach into src/libs/walrus and ship "
        f"88 of the submodule's Hydra configs even though packages.find "
        f"excludes it. Anchor each glob to a directory PRISM actually reads "
        f'from, e.g. "conf/**/*.yaml".'
    )


def test_namespace_discovery_is_not_disabled() -> None:
    """`namespaces = false` would silently drop real PRISM packages."""
    find = _section("tool.setuptools.packages.find")
    assert "namespaces = false" not in find, (
        "packages.find sets `namespaces = false`. src/api, src/data, src/ui "
        "and src/utils have no __init__.py, so disabling namespace discovery "
        "drops real modules from the wheel. Exclude the submodule by name "
        "instead."
    )


def test_conf_tree_is_still_declared_as_package_data() -> None:
    """Anchoring the globs must not cost us the Hydra config tree."""
    data = _section("tool.setuptools.package-data")
    assert "conf/**/*.yaml" in data, (
        "The Hydra config tree is no longer declared as package data. "
        "src/train.py composes from `config_path='conf'`, and src/conf has no "
        "__init__.py, so those 50 yaml files only reach the wheel this way."
    )
