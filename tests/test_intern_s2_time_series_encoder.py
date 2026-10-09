"""
Unit tests for Intern-S2 time series encoders.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from src.encoders.time_series import TimeSeriesEncoder
from torch import nn

pytestmark = [pytest.mark.unit, pytest.mark.timeseries]


class _FakeInternS2(nn.Module):
    """Emits one token per 8 real timesteps (per-sample), padded to batch max."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(out_hidden_size=2048)
        self.forward_kwargs = None
        self.encoder_embed = SimpleNamespace(transformer_encoder=None)

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        ts_lens = kwargs["ts_lens"]
        batch = kwargs["time_series_signals"].shape[0]
        token_counts = torch.div(ts_lens, 8, rounding_mode="floor").clamp(min=1)
        n = int(token_counts.max())
        embeds = torch.arange(batch * n * 2048).reshape(batch, n, 2048).float()
        pad_mask = torch.arange(n).unsqueeze(0) >= token_counts.unsqueeze(1)
        return embeds, pad_mask


class _FakeInternS2_397B_LengthAware(nn.Module):
    """397B fake whose token count scales with real `ts_lens`."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(out_hidden_size=4096)
        self.forward_kwargs = None
        self.encoder_embed = SimpleNamespace(transformer_encoder=None)

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        ts_lens = kwargs["ts_lens"]
        batch = kwargs["time_series_signals"].shape[0]
        token_counts = torch.div(ts_lens, 16, rounding_mode="floor").clamp(min=1)
        n = int(token_counts.max())
        embeds = torch.arange(batch * n * 4096).reshape(batch, n, 4096).float()
        pad_mask = torch.arange(n).unsqueeze(0) >= token_counts.unsqueeze(1)
        ts_encoder_embedding = torch.zeros(batch, n, 1024)
        return embeds, pad_mask, ts_encoder_embedding


@pytest.fixture
def fake_intern_s2():
    return _FakeInternS2()


@pytest.fixture
def fake_intern_s2_397b():
    return _FakeInternS2_397B_LengthAware()


def test_intern_s2_loads_extracted_model_and_pads_to_fixed_width(
    monkeypatch, fake_intern_s2
):
    load_model_calls = []

    def fake_load_model(
        checkpoint_path, package="src.encoders.intern_s2_preview", load_pretrained=True
    ):
        load_model_calls.append((checkpoint_path, package))
        return fake_intern_s2

    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model", fake_load_model
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2",
        model_name="/hf-home/intern-s2-preview-timeseries/model.safetensors",
        max_ts_length=512,
        intern_s2_sampling_rate=1.0,
    )

    output = encoder(torch.zeros(2, 512, 4))

    assert load_model_calls == [
        (
            "/hf-home/intern-s2-preview-timeseries/model.safetensors",
            "src.encoders.intern_s2_preview",
        )
    ]
    assert output.shape == (2, encoder.tokens_per_instance(), 2048)
    assert encoder.hidden_dim == 2048
    assert encoder.model.forward_kwargs["ts_lens"].tolist() == [512, 512]
    assert encoder.model.forward_kwargs["channels"].tolist() == [4, 4]
    assert encoder.model.forward_kwargs["sr"].tolist() == [1.0, 1.0]


def test_intern_s2_accepts_heterogeneous_length_list_input(
    monkeypatch, fake_intern_s2
):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2,
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2",
        model_name="unused",
        max_ts_length=512,
        intern_s2_sampling_rate=1.0,
    )

    inputs = [torch.zeros(64, 1), torch.zeros(512, 1), torch.zeros(200, 1)]
    output = encoder(inputs)

    assert output.shape == (3, encoder.tokens_per_instance(), 2048)
    assert encoder.model.forward_kwargs["ts_lens"].tolist() == [64, 512, 200]
    assert encoder.model.forward_kwargs["channels"].tolist() == [1, 1, 1]


def test_intern_s2_tokens_match_fixed_length_subsampling(monkeypatch, fake_intern_s2):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2,
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2",
        model_name="unused",
        max_ts_length=512,
        intern_s2_sampling_rate=1.0,
    )

    assert encoder.tokens_per_instance() == 64


def test_load_intern_s2_model_uses_vendored_code_and_strict_weights(
    tmp_path, monkeypatch
):
    from src.encoders.time_series import _load_intern_s2_model

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    fake_config = SimpleNamespace(out_hidden_size=2048)

    state_dict_loaded = {}
    strict_flag = {}

    class FakeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            transformer = nn.Linear(10, 10)
            self.encoder_embed = nn.Module()
            self.encoder_embed.transformer_encoder = transformer
            self.encoder_embed.forward = lambda *args, **kwargs: (
                torch.zeros(1, 10, dtype=torch.bfloat16),
                None,
            )

        def load_state_dict(self, state_dict, strict=True):
            state_dict_loaded.update(state_dict)
            strict_flag["strict"] = strict

    config_class = Mock()
    config_class.from_json_file.return_value = fake_config
    state_dict = {"encoder.weight": torch.ones(1)}
    configuration_module = SimpleNamespace(
        InternS2PreviewTimeSeriesConfig=config_class
    )
    modeling_module = SimpleNamespace(InternS2PreviewTimeSeriesModel=FakeModel)

    def fake_import(name):
        if name.endswith("configuration_interns2_preview"):
            return configuration_module
        if name.endswith("modeling_interns2_preview"):
            return modeling_module
        raise ImportError(f"Unknown module {name}")

    monkeypatch.setattr(
        "src.encoders.time_series.load_safetensors",
        lambda path, device="cpu": state_dict,
    )
    monkeypatch.setattr(
        "src.encoders.time_series.importlib.import_module", fake_import
    )

    result = _load_intern_s2_model(checkpoint)

    assert isinstance(result, FakeModel)
    assert result.config is fake_config
    assert state_dict_loaded == state_dict
    assert strict_flag.get("strict") is True
    assert result.encoder_embed.transformer_encoder.weight.dtype == torch.bfloat16
    output, _ = result.encoder_embed()
    assert output.dtype == torch.float32


def test_intern_s2_397b_loads_and_pads_to_fixed_width(
    monkeypatch, fake_intern_s2_397b
):
    load_model_calls = []

    def fake_load_model(
        checkpoint_path, package="src.encoders.intern_s2_preview", load_pretrained=True
    ):
        load_model_calls.append((checkpoint_path, package))
        return fake_intern_s2_397b

    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model", fake_load_model
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b",
        model_name="/hf-home/intern-s2-preview-397b-timeseries/model.safetensors",
        max_ts_length=512,
    )

    output = encoder(torch.zeros(2, 512, 4))

    assert load_model_calls == [
        (
            "/hf-home/intern-s2-preview-397b-timeseries/model.safetensors",
            "src.encoders.intern_s2_preview_397b",
        )
    ]
    assert output.shape == (2, encoder.tokens_per_instance(), 4096)
    assert encoder.hidden_dim == 4096
    assert encoder.model.forward_kwargs["ts_lens"].tolist() == [512, 512]
    assert encoder.model.forward_kwargs["channels"].tolist() == [4, 4]


def test_intern_s2_397b_accepts_heterogeneous_length_and_variable_channels(
    monkeypatch, fake_intern_s2_397b
):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2_397b,
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b",
        model_name="unused",
        max_ts_length=512,
    )

    inputs = [torch.zeros(48, 2), torch.zeros(512, 1)]
    output = encoder(inputs)

    assert output.shape == (2, encoder.tokens_per_instance(), 4096)
    assert encoder.model.forward_kwargs["ts_lens"].tolist() == [48, 512]
    assert encoder.model.forward_kwargs["channels"].tolist() == [2, 1]
    assert encoder.model.forward_kwargs["time_series_signals"].shape == (2, 512, 2)


def test_intern_s2_397b_tokens_per_instance_is_probed_empirically(
    monkeypatch, fake_intern_s2_397b
):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2_397b,
    )

    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b",
        model_name="unused",
        max_ts_length=512,
    )

    assert encoder.tokens_per_instance() == 32


def test_load_intern_s2_model_uses_package_specific_vendored_code(
    tmp_path, monkeypatch
):
    from src.encoders.time_series import _load_intern_s2_model

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    fake_config = SimpleNamespace(out_hidden_size=4096)

    class Fake397BModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.encoder_embed = SimpleNamespace(transformer_encoder=None)

        def load_state_dict(self, state_dict, strict=True):
            pass

    config_class = Mock()
    config_class.from_json_file.return_value = fake_config
    configuration_module = SimpleNamespace(
        InternS2PreviewTimeSeriesConfig=config_class
    )
    modeling_module = SimpleNamespace(InternS2PreviewTimeSeriesModel=Fake397BModel)

    imported_modules = []

    def fake_import(name):
        imported_modules.append(name)
        if name.endswith("configuration_interns2_preview"):
            return configuration_module
        if name.endswith("modeling_interns2_preview"):
            return modeling_module
        raise ImportError(f"Unknown module {name}")

    monkeypatch.setattr(
        "src.encoders.time_series.load_safetensors",
        lambda path, device="cpu": {},
    )
    monkeypatch.setattr(
        "src.encoders.time_series.importlib.import_module", fake_import
    )

    result = _load_intern_s2_model(
        checkpoint, package="src.encoders.intern_s2_preview_397b"
    )

    assert isinstance(result, Fake397BModel)
    assert imported_modules == [
        "src.encoders.intern_s2_preview_397b.configuration_interns2_preview",
        "src.encoders.intern_s2_preview_397b.modeling_interns2_preview",
    ]


def test_load_intern_s2_model_skips_weight_loading_when_not_pretrained(monkeypatch):
    from src.encoders.time_series import _load_intern_s2_model

    fake_config = SimpleNamespace(out_hidden_size=2048)
    load_state_dict_calls = []

    class FakeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.encoder_embed = SimpleNamespace(transformer_encoder=None)

        def load_state_dict(self, state_dict, strict=True):
            load_state_dict_calls.append((state_dict, strict))

    config_class = Mock()
    config_class.from_json_file.return_value = fake_config
    configuration_module = SimpleNamespace(
        InternS2PreviewTimeSeriesConfig=config_class
    )
    modeling_module = SimpleNamespace(InternS2PreviewTimeSeriesModel=FakeModel)

    def fake_import(name):
        if name.endswith("configuration_interns2_preview"):
            return configuration_module
        if name.endswith("modeling_interns2_preview"):
            return modeling_module
        raise ImportError(f"Unknown module {name}")

    def fail_load_safetensors(path, device="cpu"):
        raise AssertionError("safetensors should not be loaded when load_pretrained=False")

    monkeypatch.setattr(
        "src.encoders.time_series.load_safetensors", fail_load_safetensors
    )
    monkeypatch.setattr(
        "src.encoders.time_series.importlib.import_module", fake_import
    )

    # No checkpoint path exists, but this must not raise FileNotFoundError.
    result = _load_intern_s2_model(
        "/nonexistent/checkpoint", load_pretrained=False
    )

    assert isinstance(result, FakeModel)
    assert load_state_dict_calls == []


def test_time_series_encoder_intern_s2_397b_passes_load_pretrained_through(
    monkeypatch, fake_intern_s2_397b
):
    load_model_kwargs = {}

    def fake_load_model(model_name, package="src.encoders.intern_s2_preview", load_pretrained=True):
        load_model_kwargs["load_pretrained"] = load_pretrained
        return fake_intern_s2_397b

    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model", fake_load_model
    )

    TimeSeriesEncoder(
        encoder_type="intern_s2_397b",
        model_name="unused",
        max_ts_length=512,
        load_pretrained=False,
    )

    assert load_model_kwargs["load_pretrained"] is False


def test_load_moirai_model_skips_weight_download_when_not_pretrained(monkeypatch):
    import sys
    import types

    from src.encoders.time_series import _load_moirai_model

    fake_module_cls = Mock()
    fake_module_cls.from_pretrained = Mock(
        side_effect=AssertionError("from_pretrained should not be called")
    )
    fake_config = {"d_model": 64, "num_layers": 1}

    # Pre-seed sys.modules so `from uni2ts.model.moirai2 import Moirai2Module`
    # resolves to our fake without exercising uni2ts's real (heavier) import chain.
    fake_moirai2_module = types.ModuleType("uni2ts.model.moirai2")
    fake_moirai2_module.Moirai2Module = fake_module_cls
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai2", fake_moirai2_module)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename: "/tmp/fake_config.json",
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: __import__("io").StringIO('{"d_model": 64, "num_layers": 1}'),
    )

    _load_moirai_model("Salesforce/moirai-2.0-R-small", load_pretrained=False)

    fake_module_cls.assert_called_once_with(**fake_config)
    fake_module_cls.from_pretrained.assert_not_called()
