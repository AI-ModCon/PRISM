"""Regression tests for ModelConfig.modalities being typed as list[Modality].

After PR #52 added the Modality enum, the field type was tightened from
list[str] to list[Modality] (issue #6 follow-up). YAML/CLI/preset strings
must still be accepted and coerced.
"""
from src.config import ModelConfig
from src.modalities import ALL_MODALITIES, Modality


def test_default_modalities_are_enum_members():
    cfg = ModelConfig()
    assert all(isinstance(m, Modality) for m in cfg.modalities)
    assert set(cfg.modalities) == set(ALL_MODALITIES)


def test_string_init_coerces_to_enum():
    cfg = ModelConfig(modalities=["text", "image"])
    assert cfg.modalities == [Modality.TEXT, Modality.IMAGE]
    assert all(isinstance(m, Modality) for m in cfg.modalities)


def test_mixed_string_and_enum_init():
    cfg = ModelConfig(modalities=[Modality.TEXT, "image", Modality.GRAPH])
    assert cfg.modalities == [Modality.TEXT, Modality.IMAGE, Modality.GRAPH]
    assert all(isinstance(m, Modality) for m in cfg.modalities)


def test_preset_modalities_coerced():
    # prism-auroragpt-2b explicitly sets modalities=["text", "image"]
    cfg = ModelConfig.from_preset("prism-auroragpt-2b")
    assert cfg.modalities == [Modality.TEXT, Modality.IMAGE]
    assert all(isinstance(m, Modality) for m in cfg.modalities)


def test_str_equality_still_works_post_coercion():
    cfg = ModelConfig(modalities=["text", "image"])
    assert "text" in cfg.modalities
    assert Modality.IMAGE in cfg.modalities
    assert "audio" not in cfg.modalities
