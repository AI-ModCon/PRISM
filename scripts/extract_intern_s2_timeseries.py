#!/usr/bin/env python3

"""Download and extract Intern-S2 time-series components from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_REPO_ID = "internlm/Intern-S2-Preview"
DEFAULT_CODE_DIR = Path("src/encoders/intern_s2_preview")
DEFAULT_ARTIFACT_NAME = "intern-s2-preview-timeseries"
COMPONENT_PREFIXES = {
    "encoder": "model.time_series.",
    "forecaster": "time_series_forecaster.",
}
METADATA_FILES = (
    "config.json",
    "configuration_interns2_preview.py",
    "modeling_interns2_preview.py",
    "model.safetensors.index.json",
)


def select_component_weights(weight_map: Mapping[str, str], prefix: str) -> dict[str, str]:
    """Return checkpoint entries belonging to one component prefix."""
    return {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}


def build_encoder_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    """Promote the nested Intern-S2 time-series config to a standalone config."""
    config = deepcopy(source_config.get("ts_config", {}))
    if not config:
        raise ValueError("Source config does not contain `ts_config`")
    config["architectures"] = ["InternS2PreviewTimeSeriesModel"]
    config.setdefault(
        "auto_map",
        {
            "AutoConfig": (
                "configuration_interns2_preview.InternS2PreviewTimeSeriesConfig"
            ),
            "AutoModel": "modeling_interns2_preview.InternS2PreviewTimeSeriesModel",
        },
    )
    return config


def build_forecaster_config(source_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the optional forecaster config used by larger Intern-S2 variants."""
    config = deepcopy(source_config.get("ts_forecaster_config"))
    if config is not None:
        config["architectures"] = ["InternS2PreviewTimeSeriesForecaster"]
    return config


def extract_component(
    selected_weights: Mapping[str, str],
    *,
    prefix: str,
    shard_paths: Mapping[str, Path],
    output_dir: Path,
    required: bool = True,
) -> dict[str, Any]:
    """Extract selected tensors into one standalone safetensors checkpoint."""
    source_shards = sorted(set(selected_weights.values()))
    if not selected_weights:
        if required:
            raise ValueError(f"No checkpoint tensors found with prefix {prefix!r}")
        return {"available": False, "tensor_count": 0, "source_shards": []}

    tensors = {}
    for shard_name in source_shards:
        shard_path = shard_paths[shard_name]
        with safe_open(str(shard_path), framework="pt", device="cpu") as source:
            for source_key, source_shard in selected_weights.items():
                if source_shard != shard_name:
                    continue
                output_key = source_key.removeprefix(prefix)
                tensors[output_key] = source.get_tensor(source_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_dir / "model.safetensors"))
    return {
        "available": True,
        "tensor_count": len(tensors),
        "source_shards": source_shards,
    }


def _write_vendored_code(
    code_dir: Path,
    config: Mapping[str, Any],
    downloaded_metadata: Mapping[str, Path],
) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for filename in ("configuration_interns2_preview.py", "modeling_interns2_preview.py"):
        shutil.copy2(downloaded_metadata[filename], code_dir / filename)


def resolve_output_dir(output_dir: Path | None) -> Path:
    """Resolve checkpoint output under HF_HOME unless explicitly overridden."""
    if output_dir is not None:
        return output_dir.expanduser()
    hf_home = os.getenv("HF_HOME")
    if not hf_home:
        raise ValueError("HF_HOME must be set when --output-dir is not provided")
    return Path(hf_home).expanduser() / DEFAULT_ARTIFACT_NAME


def download_and_extract(
    *,
    repo_id: str,
    output_dir: Path,
    code_dir: Path,
    token: str,
    revision: str = "main",
    include_forecaster: bool = True,
) -> dict[str, Any]:
    """Download only shards containing Intern-S2 time-series components."""
    metadata = {
        filename: Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
            )
        )
        for filename in METADATA_FILES
    }
    source_config = json.loads(metadata["config.json"].read_text(encoding="utf-8"))
    index = json.loads(
        metadata["model.safetensors.index.json"].read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]

    selections = {
        "encoder": select_component_weights(weight_map, COMPONENT_PREFIXES["encoder"]),
    }
    if include_forecaster:
        selections["forecaster"] = select_component_weights(
            weight_map, COMPONENT_PREFIXES["forecaster"]
        )

    shard_names = sorted(
        {shard for selected in selections.values() for shard in selected.values()}
    )
    shard_paths = {
        shard: Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=shard,
                revision=revision,
                token=token,
            )
        )
        for shard in shard_names
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_manifest = extract_component(
        selections["encoder"],
        prefix=COMPONENT_PREFIXES["encoder"],
        shard_paths=shard_paths,
        output_dir=output_dir,
    )
    _write_vendored_code(code_dir, build_encoder_config(source_config), metadata)

    forecaster_manifest = {"available": False, "tensor_count": 0, "source_shards": []}
    if include_forecaster:
        forecaster_dir = output_dir / "forecaster"
        forecaster_manifest = extract_component(
            selections["forecaster"],
            prefix=COMPONENT_PREFIXES["forecaster"],
            shard_paths=shard_paths,
            output_dir=forecaster_dir,
            required=False,
        )
        forecaster_config = build_forecaster_config(source_config)
        if forecaster_manifest["available"]:
            if forecaster_config is None:
                raise ValueError("Forecaster tensors exist but `ts_forecaster_config` is absent")
            (code_dir / "forecaster_config.json").write_text(
                json.dumps(forecaster_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    manifest = {
        "source_repo": repo_id,
        "source_revision": revision,
        "transformers_version": source_config.get("transformers_version"),
        "encoder": encoder_manifest,
        "forecaster": forecaster_manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint destination (default: $HF_HOME/intern-s2-preview-timeseries).",
    )
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Do not extract forecaster weights when the source repository has them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=False)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(f"HF_TOKEN is not set in the environment or {args.env_file}")
    try:
        output_dir = resolve_output_dir(args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    manifest = download_and_extract(
        repo_id=args.repo_id,
        output_dir=output_dir,
        code_dir=args.code_dir,
        token=token,
        revision=args.revision,
        include_forecaster=not args.encoder_only,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Extracted Intern-S2 time-series weights to {output_dir.resolve()}")
    print(f"Updated Intern-S2 source code in {args.code_dir.resolve()}")


if __name__ == "__main__":
    main()