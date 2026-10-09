import pytest
from tools.universal_evaluator import _load_resolved_model_config

pytestmark = pytest.mark.unit


def test_load_resolved_model_config_expands_environment_only_in_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/shared/huggingface")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  ts_encoder_id: ${oc.env:HF_HOME}/intern-s2/model.safetensors
training:
  output_dir: ${hydra:runtime.output_dir}
"""
    )

    model_config = _load_resolved_model_config(str(config_path))

    assert model_config["ts_encoder_id"] == "/shared/huggingface/intern-s2/model.safetensors"