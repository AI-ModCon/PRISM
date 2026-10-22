from pathlib import Path
from types import SimpleNamespace

import pytest
import src.train as train_module
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_joint_hydra_presets_preserve_both_adapter_contracts():
    config_dir = str((Path(__file__).parents[1] / "src" / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=prism_astroprism_joint_qwen3_4b",
                "training=astroprism_joint_aion",
                "+data=astroprism_joint_aion",
            ],
        )

    assert cfg.training.task == "astroprism_joint_aion"
    assert cfg.training.max_steps == 200
    assert cfg.training.freeze_llm is True
    assert cfg.training.freeze_vit is True
    assert cfg.training.eval_enabled is True
    assert cfg.training.eval_every_n_steps == 50
    assert cfg.training.validation_batches == 0
    assert set(cfg.model.modalities) == {"text", "image", "spectrum"}
    assert cfg.model.image_projector_norm_mode == "none"
    assert cfg.model.image_projector_modality_embed_pos == "after_norm"
    assert cfg.model.spectrum_projector_norm_mode == "text_scaled_layernorm"
    assert cfg.model.spectrum_projector_target_norm == pytest.approx(1.04)
    assert cfg.model.spectrum_projector_modality_embed_pos == "before_norm"
    assert cfg.data.split == "train"


def test_joint_builder_forces_one_split_into_both_children(monkeypatch):
    captured = {}

    class _Child:
        def __init__(self, modality, split):
            self.active_modalities = {modality}
            self.config = SimpleNamespace(split=split)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"index": index}

    def image_builder(config, tokenizer):
        captured["image"] = config
        return _Child("image", config.split), object()

    def spectrum_builder(config, tokenizer):
        captured["spectrum"] = config
        return _Child("spectrum", config.split), object()

    class _JointCollator:
        def __init__(self, image, spectrum):
            captured["collators"] = (image, spectrum)

    monkeypatch.setattr(train_module, "_build_astroprism_stage1a_data", image_builder)
    monkeypatch.setattr(
        train_module,
        "_build_astroprism_stage1b_spectrum_data",
        spectrum_builder,
    )
    monkeypatch.setattr(train_module, "AstroPRISMJointCollator", _JointCollator)

    data_cfg = OmegaConf.create(
        {
            "split": "train",
            "max_merged_seq_length": 2048,
            "image": {"raw_manifest_path": "image.json"},
            "spectrum": {"raw_manifest_path": "spectrum.json"},
        }
    )
    dataset, _ = train_module._build_astroprism_joint_data(
        data_cfg,
        object(),
        split="validation",
    )

    assert dataset.split == "validation"
    assert captured["image"].split == "validation"
    assert captured["spectrum"].split == "validation"
    assert captured["image"].max_merged_seq_length == 2048
    assert captured["spectrum"].max_merged_seq_length == 2048


def test_joint_model_contract_is_fail_closed():
    valid = SimpleNamespace(
        image_encoder_type="aion",
        spectrum_encoder_type="aion",
        modalities=["text", "image", "spectrum"],
    )
    train_module._validate_astroprism_joint_model_contract(
        "astroprism_joint_aion", valid
    )

    with pytest.raises(RuntimeError, match="requires exactly"):
        train_module._validate_astroprism_joint_model_contract(
            "astroprism_joint_aion",
            SimpleNamespace(
                image_encoder_type="aion",
                spectrum_encoder_type="aion",
                modalities=["text", "image"],
            ),
        )


def test_joint_runtime_rejects_deepspeed_loader_bypass(monkeypatch):
    monkeypatch.setenv("DEEPSPEED_ZERO_STAGE", "2")
    with pytest.raises(RuntimeError, match="does not support DeepSpeed"):
        train_module._validate_astroprism_joint_runtime_environment(
            "astroprism_joint_aion"
        )

    train_module._validate_astroprism_joint_runtime_environment("vlm")
    monkeypatch.delenv("DEEPSPEED_ZERO_STAGE")
    train_module._validate_astroprism_joint_runtime_environment(
        "astroprism_joint_aion"
    )


def test_joint_experiment_tracks_validation_and_final_test_offline():
    [experiment] = yaml.safe_load(
        Path("experiments/astroprism_joint_aion.yaml").read_text(encoding="utf-8")
    )["experiments"]
    overrides = experiment["common_overrides"]

    assert experiment["id"] == "ASTROPRISM-JOINT-AION"
    assert overrides["training.task"] == "astroprism_joint_aion"
    assert overrides["training.max_steps"] == 200
    assert overrides["training.eval_enabled"] is True
    assert overrides["training.validation_batches"] == 0
    assert overrides["training.freeze_llm"] is True
    assert overrides["training.freeze_vit"] is True
    assert overrides["wandb.mode"] == "offline"
    assert "astroprism_joint_aion" in train_module._ASTROPRISM_MAP_STYLE_TASKS
