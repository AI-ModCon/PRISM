import json

import pytest
import torch
from applications.timeseries.extract_intern_s2_timeseries import (
    COMPONENT_PREFIXES,
    build_encoder_config,
    extract_component,
    resolve_output_dir,
    select_component_weights,
)
from safetensors.torch import load_file, save_file

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]

def test_select_component_weights_uses_only_matching_prefix():
    weight_map = {
        "model.language_model.layer.weight": "model-00001.safetensors",
        "model.time_series.encoder.weight": "model-00023.safetensors",
        "model.time_series.projector.bias": "model-00023.safetensors",
        "time_series_forecaster.head.weight": "model-00024.safetensors",
    }

    selected = select_component_weights(weight_map, COMPONENT_PREFIXES["encoder"])

    assert selected == {
        "model.time_series.encoder.weight": "model-00023.safetensors",
        "model.time_series.projector.bias": "model-00023.safetensors",
    }


def test_build_encoder_config_promotes_ts_subconfig():
    source = {
        "model_type": "intern_s2_preview",
        "ts_config": {
            "model_type": "interns2_preview_time_series",
            "out_hidden_size": 2048,
            "auto_map": {
                "AutoConfig": "configuration_interns2_preview.InternS2PreviewTimeSeriesConfig",
                "AutoModel": "modeling_interns2_preview.InternS2PreviewTimeSeriesModel",
            },
        },
    }

    config = build_encoder_config(source)

    assert config["model_type"] == "interns2_preview_time_series"
    assert config["architectures"] == ["InternS2PreviewTimeSeriesModel"]
    assert config["out_hidden_size"] == 2048


def test_extract_component_writes_stripped_standalone_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "encoder"
    source_dir.mkdir()
    shard = source_dir / "model-00023.safetensors"
    save_file(
        {
            "model.time_series.encoder.weight": torch.arange(4).reshape(2, 2),
            "model.language_model.weight": torch.ones(1),
        },
        str(shard),
    )
    selected = {
        "model.time_series.encoder.weight": shard.name,
    }

    manifest = extract_component(
        selected,
        prefix=COMPONENT_PREFIXES["encoder"],
        shard_paths={shard.name: shard},
        output_dir=output_dir,
    )

    extracted = load_file(str(output_dir / "model.safetensors"))
    assert list(extracted) == ["encoder.weight"]
    assert torch.equal(extracted["encoder.weight"], torch.arange(4).reshape(2, 2))
    assert manifest["tensor_count"] == 1
    assert manifest["source_shards"] == [shard.name]


def test_extract_component_skips_absent_optional_forecaster(tmp_path):
    manifest = extract_component(
        {},
        prefix=COMPONENT_PREFIXES["forecaster"],
        shard_paths={},
        output_dir=tmp_path / "forecaster",
        required=False,
    )

    assert manifest == {"available": False, "tensor_count": 0, "source_shards": []}
    assert not (tmp_path / "forecaster").exists()


def test_encoder_config_is_json_serializable():
    config = build_encoder_config({"ts_config": {"d_model": 768}})
    assert json.loads(json.dumps(config))["d_model"] == 768


def test_resolve_output_dir_uses_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))

    assert resolve_output_dir(None) == (
        tmp_path / "huggingface" / "intern-s2-preview-timeseries"
    )


def test_resolve_output_dir_requires_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)

    with pytest.raises(ValueError, match="HF_HOME"):
        resolve_output_dir(None)