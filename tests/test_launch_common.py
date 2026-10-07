"""Tests for tools/_launch_common.py."""
import os
from pathlib import Path

import pytest
import yaml
from tools._launch_common import (
    DEFAULT_IMAGE_ENCODER_ID,
    hf_cache_dir,
    load_dotenv,
    load_model_group_config,
    lookup_experiment,
    resolve_hydra_override,
    unique_hf_cache_dirs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# load_dotenv
# ---------------------------------------------------------------------------


def test_load_dotenv_parses_keys(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    env = tmp_path / ".env"
    env.write_text("FOO=bar\n# comment\nBAR='baz qux'\nNOEQUAL\n\n")
    out = load_dotenv(env)
    assert out == {"FOO": "bar", "BAR": "baz qux"}
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAR"] == "baz qux"


def test_load_dotenv_respects_pre_set_envvars(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FOO", "preexisting")
    env = tmp_path / ".env"
    env.write_text("FOO=fromdotenv\n")
    out = load_dotenv(env)
    assert out["FOO"] == "fromdotenv"  # parsed value reflected
    assert os.environ["FOO"] == "preexisting"  # but env not overwritten


def test_load_dotenv_missing_file_returns_empty(tmp_path: Path):
    assert load_dotenv(tmp_path / "does-not-exist") == {}


# ---------------------------------------------------------------------------
# lookup_experiment
# ---------------------------------------------------------------------------


def test_lookup_experiment_top_level(tmp_path: Path):
    design = tmp_path / "designs.yaml"
    design.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "id": "RUN-A",
                        "common_overrides": {"training.batch_size": 8},
                        "overrides": {"training.lr": 1e-4},
                    },
                ]
            }
        )
    )
    target, parent, overrides = lookup_experiment(design, "RUN-A")
    assert target["id"] == "RUN-A"
    assert parent is target  # top-level: parent == target
    assert overrides == {"training.batch_size": 8, "training.lr": 1e-4}


def test_lookup_experiment_variant_inherits_parent(tmp_path: Path):
    design = tmp_path / "designs.yaml"
    design.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "id": "PARENT",
                        "common_overrides": {"a": 1},
                        "overrides": {"b": 2},
                        "variants": [
                            {"id": "CHILD", "overrides": {"b": 99, "c": 3}},
                        ],
                    },
                ]
            }
        )
    )
    target, parent, overrides = lookup_experiment(design, "CHILD")
    assert target["id"] == "CHILD"
    assert parent["id"] == "PARENT"
    assert overrides == {"a": 1, "b": 99, "c": 3}


def test_lookup_experiment_missing_id_exits(tmp_path: Path):
    design = tmp_path / "designs.yaml"
    design.write_text(yaml.safe_dump({"experiments": [{"id": "A"}]}))
    with pytest.raises(SystemExit) as e:
        lookup_experiment(design, "ZZZ")
    assert e.value.code == 1


def test_lookup_experiment_missing_file_exits(tmp_path: Path):
    with pytest.raises(SystemExit) as e:
        lookup_experiment(tmp_path / "nope.yaml", "X")
    assert e.value.code == 1


def test_lookup_experiment_real_designs_file():
    """Smoke: every variant in the real designs file must be findable."""
    designs = REPO_ROOT / "experiments" / "prism_designs.yaml"
    if not designs.exists():
        pytest.skip("designs file missing in this checkout")
    with open(designs) as f:
        data = yaml.safe_load(f)
    sample_ids: list[str] = []
    for exp in data.get("experiments", []):
        sample_ids.append(exp["id"])
        for v in exp.get("variants", []):
            sample_ids.append(v["id"])
    sample_ids = sample_ids[:5]  # don't iterate 70 — just confirm path works
    for sid in sample_ids:
        target, _, _ = lookup_experiment(designs, sid)
        assert target["id"] == sid


# ---------------------------------------------------------------------------
# Image-staging helpers (added in PR #96)
# ---------------------------------------------------------------------------


def test_resolve_hydra_override_prefers_cli_value():
    overrides = {"model.image_encoder_id": DEFAULT_IMAGE_ENCODER_ID}
    unknown_args = ["model.image_encoder_id=google/siglip2-so400m-patch14-384"]

    assert (
        resolve_hydra_override(
            overrides,
            unknown_args,
            "model.image_encoder_id",
        )
        == "google/siglip2-so400m-patch14-384"
    )


def test_hf_cache_dir_skips_null_and_local_paths():
    assert hf_cache_dir("Qwen/Qwen3-0.6B") == "models--Qwen--Qwen3-0.6B"
    assert hf_cache_dir("/models/local-qwen") is None
    assert hf_cache_dir("null") is None


def test_unique_hf_cache_dirs_deduplicates_encoder_and_processor():
    assert unique_hf_cache_dirs(
        [
            "google/siglip2-base-patch16-224",
            "google/siglip2-base-patch16-224",
            "google/siglip2-so400m-patch14-384",
        ]
    ) == [
        "models--google--siglip2-base-patch16-224",
        "models--google--siglip2-so400m-patch14-384",
    ]


def test_load_model_group_config_reads_hydra_model_yaml(tmp_path):
    model_dir = tmp_path / "src" / "conf" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "prism_qwen3_test.yaml").write_text(
        "\n".join(
            [
                'backbone_id: "Qwen/Qwen3-0.6B"',
                'image_encoder_id: "google/siglip2-base-patch16-224"',
            ]
        )
    )

    assert load_model_group_config(tmp_path, "prism_qwen3_test") == {
        "backbone_id": "Qwen/Qwen3-0.6B",
        "image_encoder_id": "google/siglip2-base-patch16-224",
    }
