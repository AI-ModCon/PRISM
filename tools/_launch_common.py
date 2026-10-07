"""Shared scaffolding for the Aurora launchers.

The three launchers (`launch_aurora.py`, `launch_aurora_daos.py`,
`launch_aurora_web.py`) duplicate ~80% of their setup code. Memory note
`launcher_consolidation_aspiration` describes the target: one entry with
`--storage {daos,lustre,webdataset-staged}`.

This module is the first step. It exposes helpers the existing launchers
can incrementally adopt — without changing the qsub path or env-var
handling that has a long history of silent failures (see
launcher_smoke_harness_bug, launcher_aurora_web_bugs memory notes).

The full unified launcher lands in a follow-up PR after each helper here
has been adopted by all three launchers and validated via Aurora smokes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE_ENCODER_ID = "google/siglip2-base-patch16-224"


def load_dotenv(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Parse a `.env` file (KEY=VALUE per line, # comments) and mirror values
    into `os.environ` for keys not already present.

    Currently used by `launch_aurora.py` and `launch_aurora_web.py`. The DAOS
    launcher does not call this — adding it there is a follow-up.

    Returns the parsed dict so callers can extract defaults via
    `env_config.get("PRISM_DIR", ...)`.
    """
    env_path = Path(path) if path is not None else Path(os.getcwd()) / ".env"
    env_config: dict[str, str] = {}
    if not env_path.exists():
        return env_config
    print(f"Loading configuration from {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip("'").strip('"')
            env_config[key] = val
            if key not in os.environ:
                os.environ[key] = val
    return env_config


def lookup_experiment(
    design_file: str | os.PathLike,
    design_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Find a design (or variant) in a PRISM design YAML and return
    (target_exp, parent_exp, final_overrides).

    `parent_exp` is the enclosing design when the match is a variant; None
    when the match is a top-level entry. `final_overrides` merges in this
    order (later wins):
        parent.common_overrides → parent.overrides
        → variant.common_overrides → variant.overrides

    This is a superset of the pre-extraction launcher behavior, which
    ignored `variant.common_overrides`. No variant in
    `experiments/prism_designs.yaml` currently defines `common_overrides`,
    so the merged output is identical today; the extra layer is
    future-proofing for variants that want their own shared block.

    Raises SystemExit(1) on missing file or unknown design_id so callers
    can stay direct (no exception bubbling required).
    """
    if not os.path.exists(design_file):
        print(f"Error: Design file '{design_file}' not found.")
        sys.exit(1)
    with open(design_file) as f:
        data = yaml.safe_load(f)

    target_exp: dict[str, Any] | None = None
    parent_exp: dict[str, Any] | None = None
    final_overrides: dict[str, Any] = {}

    for exp in data.get("experiments", []):
        if exp.get("id") == design_id:
            target_exp = exp
            parent_exp = exp
            final_overrides = exp.get("common_overrides", {}).copy()
            final_overrides.update(exp.get("overrides", {}))
            break
        if "variants" in exp:
            parent_overrides = exp.get("common_overrides", {}).copy()
            parent_overrides.update(exp.get("overrides", {}))
            for variant in exp["variants"]:
                if variant.get("id") == design_id:
                    target_exp = variant
                    parent_exp = exp
                    final_overrides = parent_overrides.copy()
                    final_overrides.update(variant.get("common_overrides", {}))
                    final_overrides.update(variant.get("overrides", {}))
                    break
        if target_exp is not None:
            break

    if target_exp is None:
        print(f"Error: Experiment ID '{design_id}' not found in '{design_file}'")
        sys.exit(1)

    return target_exp, parent_exp, final_overrides


def clean_hydra_value(value: Any) -> str | None:
    """Normalize a scalar Hydra override value for launcher-side decisions."""
    if value is None:
        return None
    value = str(value).strip()
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def resolve_hydra_override(
    overrides: dict[str, Any],
    unknown_args: list[str],
    key: str,
    default: Any = None,
) -> str | None:
    """Resolve a key from experiment overrides plus CLI Hydra overrides.

    The launchers need this before Hydra runs so they can stage all requested
    HuggingFace caches. CLI overrides are last-write-wins, matching Hydra.
    """
    value = overrides.get(key, default)
    accepted_keys = {key, f"+{key}", f"++{key}"}
    for arg in unknown_args:
        if "=" not in arg or arg.startswith("--"):
            continue
        raw_key, raw_value = arg.split("=", 1)
        if raw_key.strip("'\"") in accepted_keys:
            value = raw_value
    return clean_hydra_value(value)


def load_model_group_config(
    prism_dir: str | os.PathLike,
    model_group: str | None,
) -> dict[str, Any]:
    """Load ``src/conf/model/<model_group>.yaml`` when a design overrides it."""
    model_group = clean_hydra_value(model_group)
    if not model_group:
        return {}
    model_path = Path(prism_dir) / "src" / "conf" / "model" / f"{model_group}.yaml"
    if not model_path.exists():
        return {}
    with open(model_path) as f:
        return yaml.safe_load(f) or {}


def hf_cache_dir(model_id: str | None) -> str | None:
    """Return the HuggingFace cache directory name for a model ID."""
    model_id = clean_hydra_value(model_id)
    if model_id is None or model_id.startswith("/"):
        return None
    return "models--" + model_id.replace("/", "--")


def unique_hf_cache_dirs(model_ids: list[str | None]) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        model_dir = hf_cache_dir(model_id)
        if model_dir and model_dir not in seen:
            seen.add(model_dir)
            dirs.append(model_dir)
    return dirs
