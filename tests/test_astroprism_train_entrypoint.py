from pathlib import Path
from types import SimpleNamespace

import pytest
import src.train as train_module
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class _Tokenizer:
    pad_token_id = 0


def test_astroprism_hydra_presets_compose_to_safe_raw_image_defaults():
    config_dir = str((Path(__file__).parents[1] / "src" / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=prism_astroprism_qwen3_4b",
                "training=astroprism_stage1a",
                "+data=astroprism_stage1a",
            ],
        )

    assert cfg.training.task == "astroprism_stage1a"
    assert cfg.training.freeze_llm is True
    assert cfg.training.freeze_vit is True
    assert cfg.training.data_num_workers == 0
    assert cfg.data.raw_manifest_path == ("_local/astroprism/raw_images/manifest.json")
    assert cfg.data.candidates_manifest_path == ("_local/astroprism/stage1a_accepted/manifest.json")
    assert cfg.data.split == "train"
    assert cfg.data.allow_unsafe_candidates is False
    assert cfg.data.verify_file_hashes is True
    assert list(cfg.data.accepted_caption_statuses) == [
        "released_reused",
        "released_revised",
    ]
    assert cfg.data.caption_field == "auto"
    assert cfg.data.max_merged_seq_length == 2048
    assert cfg.model.image_encoder_type == "aion"


def test_stage1a_builder_forwards_paths_safety_and_length(monkeypatch, tmp_path):
    captured = {}

    class _Dataset:
        def __init__(self, config, tokenizer):
            captured["dataset_config"] = config
            captured["dataset_tokenizer"] = tokenizer

    class _Collator:
        def __init__(self, tokenizer, *, max_merged_seq_length):
            captured["collator_tokenizer"] = tokenizer
            captured["collator_max"] = max_merged_seq_length

    monkeypatch.setattr(train_module, "AstroPRISMStage1ADataset", _Dataset)
    monkeypatch.setattr(train_module, "AstroPRISMStage1ACollator", _Collator)

    tokenizer = _Tokenizer()
    raw_manifest = tmp_path / "raw" / "manifest.json"
    candidate_manifest = tmp_path / "candidates" / "manifest.json"
    data_cfg = OmegaConf.create(
        {
            "raw_manifest_path": str(raw_manifest),
            "candidates_manifest_path": str(candidate_manifest),
            "candidates_path": None,
            "split": "train",
            "caption_field": "auto",
            "allow_unsafe_candidates": False,
            "accepted_statuses": ["accepted", "approved"],
            "accepted_caption_statuses": ["released_reused", "released_revised"],
            "verify_file_hashes": True,
            "max_merged_seq_length": 2048,
        }
    )

    dataset, collator = train_module._build_astroprism_stage1a_data(
        data_cfg,
        tokenizer,
    )

    assert isinstance(dataset, _Dataset)
    assert isinstance(collator, _Collator)
    config = captured["dataset_config"]
    assert config.raw_manifest_path == raw_manifest
    assert config.candidates_manifest_path == candidate_manifest
    assert config.candidates_path is None
    assert config.split == "train"
    assert config.caption_field == "auto"
    assert config.allow_unsafe_candidates is False
    assert config.accepted_statuses == ("accepted", "approved")
    assert config.accepted_caption_statuses == (
        "released_reused",
        "released_revised",
    )
    assert config.verify_file_hashes is True
    assert captured["dataset_tokenizer"] is tokenizer
    assert captured["collator_tokenizer"] is tokenizer
    assert captured["collator_max"] == 2048


def test_stage1a_builder_requires_data_group():
    with pytest.raises(RuntimeError, match="data=astroprism_stage1a"):
        train_module._build_astroprism_stage1a_data(None, _Tokenizer())


def test_stage1a_model_contract_accepts_only_raw_aion_image_text():
    config = SimpleNamespace(
        image_encoder_type="aion",
        modalities=["text", "image"],
    )
    train_module._validate_astroprism_stage1a_model_contract(
        "astroprism_stage1a",
        config,
    )


@pytest.mark.parametrize(
    "image_encoder_type,modalities,message",
    [
        ("siglip", ["text", "image"], "image_encoder_type=aion"),
        ("aion", ["text"], "requires exactly"),
        ("aion", ["text", "image", "table"], "requires exactly"),
    ],
)
def test_stage1a_model_contract_fails_closed(
    image_encoder_type,
    modalities,
    message,
):
    config = SimpleNamespace(
        image_encoder_type=image_encoder_type,
        modalities=modalities,
    )
    with pytest.raises(RuntimeError, match=message):
        train_module._validate_astroprism_stage1a_model_contract(
            "astroprism_stage1a",
            config,
        )
