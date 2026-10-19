from pathlib import Path
from types import SimpleNamespace

import pytest
import src.train as train_module
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


class _Tokenizer:
    pad_token_id = 0


def test_spectrum_reviewed_hydra_presets_compose_to_frozen_safe_defaults():
    config_dir = str((Path(__file__).parents[1] / "src" / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=prism_astroprism_spectrum_qwen3_4b",
                "training=astroprism_stage1b_spectrum",
                "+data=astroprism_stage1b_spectrum",
            ],
        )

    assert cfg.training.task == "astroprism_stage1b_spectrum"
    assert cfg.training.max_steps == 100
    assert cfg.training.freeze_llm is True
    assert cfg.training.freeze_vit is True
    assert cfg.training.data_num_workers == 0
    assert cfg.training.eval_enabled is False
    assert cfg.data.raw_manifest_path == "_local/astroprism/raw_spectra/manifest.json"
    assert cfg.data.captions_manifest_path == (
        "_local/astroprism/stage1b_spectrum_caption_accepted/manifest.json"
    )
    assert cfg.data.captions_path is None
    assert cfg.data.split == "train"
    assert list(cfg.data.accepted_caption_statuses) == ["accepted"]
    assert "allow_unsafe_candidates" not in cfg.data
    assert cfg.data.verify_file_hashes is True
    assert list(cfg.model.modalities) == ["text", "spectrum"]
    assert cfg.model.spectrum_encoder_type == "aion"
    assert cfg.model.d_spectrum == 768
    assert cfg.model.aion_num_spectrum_tokens == 273
    assert cfg.model.aion_raw_spectrum_length == 7781
    assert cfg.model.projector_norm_mode == "text_scaled_layernorm"
    assert cfg.model.projector_target_norm == pytest.approx(1.04)
    assert cfg.model.projector_modality_embed_pos == "before_norm"


def test_spectrum_reviewed_builder_forwards_paths_safety_prompts_and_length(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class _Dataset:
        def __init__(self, config, tokenizer):
            captured["dataset_config"] = config
            captured["dataset_tokenizer"] = tokenizer

    class _Collator:
        def __init__(self, tokenizer, *, max_merged_seq_length):
            captured["collator_tokenizer"] = tokenizer
            captured["collator_max"] = max_merged_seq_length

    monkeypatch.setattr(
        train_module,
        "AstroPRISMStage1BSpectrumDataset",
        _Dataset,
    )
    monkeypatch.setattr(
        train_module,
        "AstroPRISMStage1BSpectrumCollator",
        _Collator,
    )

    tokenizer = _Tokenizer()
    raw_manifest = tmp_path / "raw-spectra" / "manifest.json"
    captions_manifest = tmp_path / "accepted-spectra" / "manifest.json"
    captions = tmp_path / "accepted-spectra" / "captions.jsonl"
    data_cfg = OmegaConf.create(
        {
            "raw_manifest_path": str(raw_manifest),
            "captions_manifest_path": str(captions_manifest),
            "captions_path": str(captions),
            "split": "train",
            "accepted_caption_statuses": ["accepted"],
            "verify_file_hashes": True,
            "max_merged_seq_length": 2048,
        }
    )

    dataset, collator = train_module._build_astroprism_stage1b_spectrum_data(
        data_cfg,
        tokenizer,
    )

    assert isinstance(dataset, _Dataset)
    assert isinstance(collator, _Collator)
    config = captured["dataset_config"]
    assert config.raw_manifest_path == raw_manifest
    assert config.captions_manifest_path == captions_manifest
    assert config.captions_path == captions
    assert config.split == "train"
    assert config.system_prompt == train_module.DEFAULT_SPECTRUM_SYSTEM_PROMPT
    assert config.user_prompt == train_module.DEFAULT_SPECTRUM_USER_PROMPT
    assert config.accepted_caption_statuses == ("accepted",)
    assert config.verify_file_hashes is True
    assert captured["dataset_tokenizer"] is tokenizer
    assert captured["collator_tokenizer"] is tokenizer
    assert captured["collator_max"] == 2048


def test_spectrum_reviewed_builder_requires_its_data_group():
    with pytest.raises(RuntimeError, match="data=astroprism_stage1b_spectrum"):
        train_module._build_astroprism_stage1b_spectrum_data(None, _Tokenizer())


def test_spectrum_reviewed_model_contract_accepts_only_raw_aion_spectrum_text():
    config = SimpleNamespace(
        spectrum_encoder_type="aion",
        modalities=["text", "spectrum"],
    )
    train_module._validate_astroprism_stage1b_spectrum_model_contract(
        "astroprism_stage1b_spectrum",
        config,
    )


@pytest.mark.parametrize(
    "spectrum_encoder_type,modalities,message",
    [
        ("generic", ["text", "spectrum"], "spectrum_encoder_type=aion"),
        ("aion", ["text"], "requires exactly"),
        ("aion", ["text", "image", "spectrum"], "requires exactly"),
    ],
)
def test_spectrum_reviewed_model_contract_fails_closed(
    spectrum_encoder_type,
    modalities,
    message,
):
    config = SimpleNamespace(
        spectrum_encoder_type=spectrum_encoder_type,
        modalities=modalities,
    )
    with pytest.raises(RuntimeError, match=message):
        train_module._validate_astroprism_stage1b_spectrum_model_contract(
            "astroprism_stage1b_spectrum",
            config,
        )


def test_spectrum_reviewed_experiment_is_a_100_step_frozen_accelerate_design():
    path = Path("experiments/astroprism_stage1b_spectrum.yaml")
    [experiment] = yaml.safe_load(path.read_text(encoding="utf-8"))["experiments"]
    overrides = experiment["common_overrides"]

    assert experiment["id"] == "ASTROPRISM-STAGE1B-SPECTRUM"
    assert overrides["training.task"] == "astroprism_stage1b_spectrum"
    assert overrides["training.max_steps"] == 100
    assert overrides["training.freeze_llm"] is True
    assert overrides["training.freeze_vit"] is True
    assert overrides["model.freeze_backbone"] is True
    assert overrides["model.freeze_encoders"] is True
    assert overrides["model.projector_norm_mode"] == "text_scaled_layernorm"
    assert overrides["model.projector_target_norm"] == pytest.approx(1.04)
    assert overrides["model.projector_modality_embed_pos"] == "before_norm"
    assert overrides["data.split"] == "train"
    assert overrides["data.accepted_caption_statuses"] == ["accepted"]
    assert "data.allow_unsafe_candidates" not in overrides
    assert "astroprism_stage1b_spectrum" in train_module._ASTROPRISM_MAP_STYLE_TASKS
