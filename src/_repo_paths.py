"""Resolve the repository checkout that `prism` subcommands shell out into.

Several subcommands shell out to a script path built from this root, which is
derived from `__file__`: it lands on the checkout under `pip install -e .`,
but in `site-packages/` under a plain `pip install .`. Some of those targets
live in `tools/` or `scripts/`, which the wheel never ships, so the subprocess
dies with an opaque FileNotFoundError; the rest are packaged but import torch
at module scope, which a resolving-install-free wheel environment also lacks.
Either way the install mode is the real problem, so say so up front.

Stdlib-only and importing nothing from `src`, so the light CLI modules can
depend on it without pulling in the model stack.
"""

import os

# src/_repo_paths.py -> the directory containing src/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories the wheel never ships; either one proves PROJECT_ROOT is a
# checkout rather than an install prefix.
_CHECKOUT_MARKERS = ("tools", "scripts")


def is_repo_checkout() -> bool:
    """True when PROJECT_ROOT holds a checkout of this repository."""
    return any(os.path.isdir(os.path.join(PROJECT_ROOT, m)) for m in _CHECKOUT_MARKERS)


def require_repo_checkout(subcommand: str) -> None:
    """Exit with an actionable error when `subcommand` has no repo to run from."""
    if is_repo_checkout():
        return
    raise SystemExit(
        f"Error: `{subcommand}` runs from a PRISM repository checkout, but none "
        f"was found at {PROJECT_ROOT}.\n"
        "This looks like a plain (non-editable) wheel install. The wheel ships "
        "the `src` package only -- not tools/ or scripts/, and not the heavyweight\n"
        "training dependencies these subcommands need. Use an editable checkout:\n"
        "    git clone https://github.com/AI-ModCon/BaseMM_PRISM.git\n"
        "    cd BaseMM_PRISM && pip install -e . --no-deps\n"
        "See docs/getting-started.md for the per-platform dependency setup."
    )
