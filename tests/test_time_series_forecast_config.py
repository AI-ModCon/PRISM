"""Decoder-only composition and offline Qwen3 forecasting API checks.

The real Transformers Qwen3 fixture retains Qwen3-0.6B's 1024-wide hidden
states but uses one random layer and a tiny vocabulary. It does not load the
pretrained Qwen checkpoint or execute the separate Intern-S2 input encoder.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.config import ModelConfig

pytestmark = pytest.mark.unit

CONF = Path(__file__).resolve().parents[1] / "src" / "conf"


def _decoder_overlay():
    return OmegaConf.load(CONF / "decoder" / "time_series_direct.yaml")


def test_overlay_preserves_referenced_intern_s2_input_model():
    """Model fields from the user's pinned 9a2f328 reference stay untouched."""
    base = OmegaConf.create(
        {
            "backbone_id": "Qwen/Qwen3-0.6B",
            "tokenizer_id": "Qwen/Qwen3-0.6B",
            "modalities": ["text", "time_series"],
            "freeze_backbone": True,
            "freeze_encoders": True,
            "freeze_vit": True,
            "d_text": 1024,
            "is_timeseries": True,
            "ts_variates": 1,
            "ts_projector": "intern_s2_397b",
            "ts_encoder_id": (
                "${oc.env:HF_HOME}/intern-s2-preview-397b-timeseries/model.safetensors"
            ),
            "ts_load_pretrained": True,
            "d_ts": 4096,
            "max_ts_length": 512,
            "is_interleaved_qa": False,
            "normalize_ts_in_encoder": False,
            "projector_hidden_mult": 1,
            "projector_num_layers": 2,
        }
    )
    before = OmegaConf.to_container(base, resolve=False)
    merged = OmegaConf.to_container(OmegaConf.merge(base, _decoder_overlay()), resolve=False)
    assert {key: merged[key] for key in before} == before
    assert set(merged) - set(before) == {
        "output_decoders", "decoder_configs", "decoder_loss_weights"
    }
    assert merged["output_decoders"] == ["text", "time_series"]
    preset = OmegaConf.to_container(
        OmegaConf.load(CONF / "model" / "prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml"),
        resolve=False,
    )
    assert {key: preset[key] for key in before} == before
    overlay_decoder = merged["decoder_configs"]["time_series"]
    preset_decoder = preset["decoder_configs"]["time_series"]
    assert preset_decoder == {
        **overlay_decoder,
        "bridge": {"type": "identity", "output_dim": 1024},
    }


def test_complete_forecast_preset_composes_nested_connector_contract():
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        merged = compose(
            config_name="config",
            overrides=["model=prism_qwen3_0_6b_intern_s2_397b_ts_forecast"],
        )
    from src.decoders import TimeSeriesDecoder

    decoder = TimeSeriesDecoder(
        merged.model.d_text,
        **OmegaConf.to_container(merged.model.decoder_configs.time_series),
    )
    assert merged.model.backbone_id == "Qwen/Qwen3-0.6B"
    assert merged.model.ts_projector == "intern_s2_397b"
    assert merged.model.output_decoders == ["text", "time_series"]
    assert decoder.conditioning_contract()["bridge"] == {
        "type": "identity", "input_dim": 1024, "output_dim": 1024,
    }
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 295_200


@pytest.mark.parametrize("input_fields, expected", [
    ({"ts_projector": "intern_s2_397b", "ts_forecast_horizon": 24},
     {"ts_projector": "intern_s2_397b", "ts_forecast_horizon": 24}),
    ({}, {"ts_projector": "linear", "ts_forecast_horizon": 96}),
])
def test_train_constructor_forwards_time_series_selection_and_horizon(input_fields, expected):
    """Evaluate the actual keyword expressions without importing training jobs.

    A missing wire-through previously silently replaced an Intern-S2 selection
    with the dataclass's linear default. Testing the expressions also checks
    legacy config fallback values rather than only grepping field names.
    """
    source_path = CONF.parent / "train.py"
    source = ast.parse(source_path.read_text())
    constructors = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelConfig"
    ]
    assert len(constructors) == 1
    keywords = {keyword.arg: keyword.value for keyword in constructors[0].keywords}
    cfg = OmegaConf.create({"model": input_fields})
    evaluated = {}
    for key in expected:
        assert key in keywords, f"ModelConfig constructor must forward {key}"
        expression = ast.fix_missing_locations(ast.Expression(keywords[key]))
        evaluated[key] = eval(
            compile(expression, str(source_path), "eval"),
            {"__builtins__": {}}, {"cfg": cfg},
        )
    configured = ModelConfig(modalities=["text"], **evaluated)
    assert {key: getattr(configured, key) for key in expected} == expected


def test_intern_s2_input_preset_requires_its_encoder_implementation(monkeypatch):
    """The current branch must fail clearly instead of substituting a linear encoder.

    Needs the Intern-S2 dependency floor: below `transformers>=5.2.0` the encoder
    raises `RuntimeError` about the pin before it can reach the `ValueError` this
    asserts. CI pins 4.57.6 deliberately (the `--intern-s2` build is a separate
    overlay), so this skips there rather than failing.
    """
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.2.0"):
        pytest.skip(
            "Intern-S2 preview needs transformers>=5.2.0; "
            f"found {transformers.__version__}"
        )

    from src.encoders.time_series import TimeSeriesEncoder

    preset = OmegaConf.load(CONF / "model" / "prism_qwen3_0_6b_intern_s2_397b_ts_forecast.yaml")
    monkeypatch.setattr(torch.nn, "Linear", lambda *a, **k: pytest.fail("silent linear fallback"))
    with pytest.raises(ValueError, match="Unsupported encoder type: intern_s2_397b"):
        TimeSeriesEncoder(
            encoder_type=preset.ts_projector,
            num_vars=preset.ts_variates,
            d_ts=preset.d_ts,
            max_ts_length=preset.max_ts_length,
        )


def test_hydra_decoder_group_preserves_selected_input_model():
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        base = compose(config_name="config", overrides=["model=prism_qwen3_0_6b_image_only"])
        merged = compose(
            config_name="config",
            overrides=["model=prism_qwen3_0_6b_image_only", "+decoder=time_series_direct"],
        )
    base_model = OmegaConf.to_container(base.model, resolve=False)
    merged_model = OmegaConf.to_container(merged.model, resolve=False)
    assert {key: merged_model[key] for key in base_model} == base_model
    assert "decoder" not in merged  # The group is packaged into model, not a sibling.
    assert merged_model["output_decoders"] == ["text", "time_series"]
    assert merged_model["decoder_configs"]["time_series"]["generator"]["horizon"] == 96
    assert merged_model["decoder_configs"]["time_series"]["readout"]["pool"] == "last"
    assert merged_model["decoder_configs"]["time_series"]["bridge"] == {"type": "identity"}


def test_hydra_forecast_options_can_be_overridden():
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        merged = compose(
            config_name="config",
            overrides=[
                "model=prism_qwen3_0_6b_image_only",
                "+decoder=time_series_direct",
                "model.decoder_configs.time_series.generator.horizon=24",
                "model.decoder_configs.time_series.generator.num_vars=3",
                "model.decoder_configs.time_series.readout.pool=mean",
                "model.decoder_loss_weights.time_series=0.25",
            ],
        )
    from src.decoders import TimeSeriesDecoder

    head = TimeSeriesDecoder(1024, **OmegaConf.to_container(merged.model.decoder_configs.time_series))
    prediction, loss = head(torch.zeros(2, 3, 1024), targets=torch.ones(2, 24, 3))
    assert prediction.shape == (2, 24, 3, 3)
    assert loss is not None and torch.isfinite(loss)
    assert merged.model.decoder_loss_weights.time_series == 0.25


@pytest.mark.parametrize("use_model_defaults", [False, True])
def test_real_qwen_width_forecast_routes_targets_and_updates_only_head(offline_hf, monkeypatch, use_model_defaults):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("Qwen3 requires a recent Transformers release")
    from src.model import UnifiedTransformer

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(937)
        qwen_config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=1024,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=32,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=None,
        )
        qwen_config._attn_implementation = "eager"
        backbone = transformers.Qwen3ForCausalLM(qwen_config)
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: backbone
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *a, **kw: SimpleNamespace(pad_token_id=0, eos_token_id=None),
        )
        overlay = OmegaConf.to_container(_decoder_overlay())
        horizon, num_vars = (24, 3) if use_model_defaults else (96, 1)
        if use_model_defaults:
            # Assembly must fill missing nested values from ModelConfig without
            # introducing conflicting flat generator keys or mutating input.
            del overlay["decoder_configs"]["time_series"]["generator"]["horizon"]
            del overlay["decoder_configs"]["time_series"]["generator"]["num_vars"]
        config = ModelConfig(
            modalities=["text"],
            llm_backbone_id="offline-random-qwen3-width-fixture",
            freeze_backbone=True,
            d_model=64,  # Assembly must use the loaded backbone width instead.
            max_merged_seq_length=32,
            ts_forecast_horizon=horizon,
            ts_variates=num_vars,
            **overlay,
        )
        model = UnifiedTransformer(config).eval()
        if use_model_defaults:
            assert "horizon" not in config.decoder_configs["time_series"]["generator"]
            assert "num_vars" not in config.decoder_configs["time_series"]["generator"]

    head = model.decoders["time_series"]
    assert head.head.in_features == 1024
    assert head.head.out_features == horizon * num_vars * 3
    assert head.conditioning_contract()["generator"]["horizon"] == horizon
    assert head.conditioning_contract()["generator"]["num_vars"] == num_vars
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(
        name.startswith("decoders.time_series.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )

    inputs = {
        "text": torch.tensor([[0, 4, 5], [6, 7, 8]]),
        "text_attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
    }
    initial = model.forward_outputs(
        inputs,
        targets={"time_series": torch.zeros(2, horizon, num_vars)},
        requested_outputs=["time_series"],
    )
    changed_target = model.forward_outputs(
        inputs,
        targets={"time_series": torch.full((2, horizon, num_vars), 100.0)},
        requested_outputs=["time_series"],
    )
    assert set(initial.predictions) == {"time_series"}
    assert set(initial.losses) == {"time_series"}
    assert initial.predictions["time_series"].shape == (2, horizon, num_vars, 3)
    assert initial.loss is not None and torch.isfinite(initial.loss)
    torch.testing.assert_close(
        initial.predictions["time_series"], changed_target.predictions["time_series"],
        atol=0, rtol=0,
    )
    assert not torch.isclose(initial.loss, changed_target.loss)

    generated = model.predict(inputs, requested_outputs=["time_series"])
    assert generated.loss is None
    assert generated.predictions["time_series"].shape == (2, horizon, num_vars)
    torch.testing.assert_close(
        generated.predictions["time_series"], initial.predictions["time_series"][..., 1]
    )

    backbone_before = {key: value.clone() for key, value in backbone.state_dict().items()}
    head_before = head.head.weight.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)
    optimizer.zero_grad()
    initial.loss.backward()
    assert head.head.weight.grad is not None
    assert torch.isfinite(head.head.weight.grad).all()
    assert torch.count_nonzero(head.head.weight.grad) > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())
    optimizer.step()
    assert not torch.equal(head.head.weight, head_before)
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_before[key], atol=0, rtol=0)
