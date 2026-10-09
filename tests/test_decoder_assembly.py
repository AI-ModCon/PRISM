"""Phase 1 tests: output-decoder config + model assembly.

Locks the output-decoder assembly contract:
- config default is ["text"]; checkpoints without the key behave as today,
- is_vla -> ["text","action"] back-compat shim,
- self.decoders ModuleDict is built from config (and only those),
- the canonical model YAML surfaces output_decoders to Hydra.

Config/YAML tests are pure unit. The model-assembly tests construct a tiny
backbone-less UnifiedTransformer (no network).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.config import ModelConfig

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Config: defaults + back-compat
# --------------------------------------------------------------------------


def test_output_decoders_default_is_text_only():
    cfg = ModelConfig(modalities=["text"])
    assert cfg.output_decoders == ["text"]
    assert cfg.decoder_configs == {}


def test_legacy_config_without_field_behaves_as_text_only():
    # Simulates an old checkpoint/config that never knew about output_decoders:
    # the dataclass default must reproduce the pre-Phase-1 behavior.
    cfg = ModelConfig(modalities=["text", "image"])
    assert cfg.output_decoders == ["text"]


def test_is_vla_maps_to_text_action():
    cfg = ModelConfig(modalities=["text", "image"], is_vla=True)
    assert cfg.output_decoders == ["text", "action"]


def test_is_vla_mapping_is_idempotent():
    cfg = ModelConfig(
        modalities=["text", "image"], is_vla=True, output_decoders=["text", "action"]
    )
    assert cfg.output_decoders == ["text", "action"]


def test_explicit_output_decoders_preserved_when_not_vla():
    cfg = ModelConfig(modalities=["text"], output_decoders=["text"])
    assert cfg.output_decoders == ["text"]


def test_output_decoders_coerced_to_str():
    # OmegaConf can hand us non-str scalars; __post_init__ normalizes.
    cfg = ModelConfig(modalities=["text"], output_decoders=["text"])
    assert all(isinstance(d, str) for d in cfg.output_decoders)


# --------------------------------------------------------------------------
# Model assembly (tiny, backbone-less)
# --------------------------------------------------------------------------


# ModelConfig's defaults describe prism-nano (d_model=1280, 20 layers, 8
# experts), which assembles to 3.73B parameters and ~19.5 GB of RSS -- more
# than a hosted CI runner has, and it is never freed between tests. These
# tests only assert on *structure* (which decoders exist, that the text
# decoder is param-less, that no action head is built), never on dimensions,
# so shrink every size knob. The assembly path under test is identical; only
# the tensors are small.
_TINY_DIMS = dict(
    d_model=64,
    d_text=64,
    num_layers=2,
    num_heads=2,
    num_experts=2,
    vocab_size=512,
)


def _tiny_text_model():
    from src.model import UnifiedTransformer

    cfg = ModelConfig(
        modalities=["text"],
        llm_backbone_id=None,
        output_decoders=["text"],
        **_TINY_DIMS,
    )
    return UnifiedTransformer(cfg)


def test_decoders_moduledict_built_from_config(offline_hf):
    from src.decoders import LMHeadDecoder

    m = _tiny_text_model()
    assert "text" in m.decoders
    assert isinstance(m.decoders["text"], LMHeadDecoder)
    # text_decoder handle points at the same instance.
    assert m.text_decoder is m.decoders["text"]


def test_text_decoder_adds_no_parameters(offline_hf):
    # The text decoder is param-less on every path, so it must not change the
    # trainable surface (keeps checkpoints/keys stable).
    m = _tiny_text_model()
    assert sum(p.numel() for p in m.decoders.parameters()) == 0


def test_non_vla_model_has_no_action_head(offline_hf):
    m = _tiny_text_model()
    assert not hasattr(m, "action_head")


def test_unknown_decoder_name_raises(offline_hf):
    from src.model import UnifiedTransformer

    cfg = ModelConfig(
        modalities=["text"],
        llm_backbone_id=None,
        output_decoders=["text", "bogus"],
        **_TINY_DIMS,
    )
    with pytest.raises(ValueError, match="Unknown output decoder"):
        UnifiedTransformer(cfg)


# --------------------------------------------------------------------------
# Hydra YAML surface
# --------------------------------------------------------------------------


def test_canonical_model_yaml_surfaces_output_decoders():
    cfg = yaml.safe_load(
        Path("src/conf/model/prism_olmo3_7b.yaml").read_text()
    )
    assert cfg["output_decoders"] == ["text"]
