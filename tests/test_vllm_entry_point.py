"""Login-node test for VLLM-3: setuptools entry-point registration.

vLLM's `load_plugins_by_group("vllm.general_plugins")` swallows exceptions
and silently logs them. A typo in pyproject.toml would leave the model
registry empty until engine boot, where it surfaces as an opaque
"model not found." This test fails fast on the UAN before any PBS job
spends node-minutes catching the same regression.

Skipped (not failed) when the PRISM dist-info hasn't been installed yet:
that means the user is on a clean checkout and the imperative
`src.vllm_plugin.register()` safety net is still in play — they should
run `tools/install_prism_entry_point.sh` once.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution, entry_points

import pytest

# `prism` was taken on PyPI, so the distribution installs as `prism-mm` while
# the entry point it registers is still named `prism`. Two different strings
# that used to be one: keep them apart deliberately, because looking the
# distribution up under the old name silently skips this whole module.
DIST_NAME = "prism-mm"
ENTRY_POINT_NAME = "prism"


def test_prism_entry_point_declared_in_metadata():
    try:
        distribution(DIST_NAME)
    except PackageNotFoundError:
        pytest.skip(
            "PRISM dist-info not installed; run "
            "`bash tools/install_prism_entry_point.sh` to enable auto-"
            "registration. The imperative register() fallback in "
            "tools/vllm_*.py still works without it."
        )

    eps = entry_points(group="vllm.general_plugins")
    names = sorted(ep.name for ep in eps)
    assert ENTRY_POINT_NAME in names, (
        f"{ENTRY_POINT_NAME!r} missing from vllm.general_plugins entry points: {names}"
    )

    (ep,) = (e for e in eps if e.name == ENTRY_POINT_NAME)
    assert ep.value == "src.vllm_plugin:register", (
        f"prism entry point points to {ep.value!r}, expected 'src.vllm_plugin:register'"
    )


def test_register_target_is_callable_via_entry_point():
    """Loading the entry point must produce the same callable that imperative
    callers use directly. Catches typos / module rename regressions that
    pyproject.toml validation does not."""
    try:
        distribution(DIST_NAME)
    except PackageNotFoundError:
        pytest.skip("PRISM dist-info not installed (see other test).")

    eps = entry_points(group="vllm.general_plugins")
    (ep,) = (e for e in eps if e.name == ENTRY_POINT_NAME)
    loaded = ep.load()

    import src.vllm_plugin

    assert loaded is src.vllm_plugin.register, (
        "Entry point resolved a different object than "
        "`src.vllm_plugin.register`; the imperative path and the "
        "auto-discovery path will register different models."
    )
