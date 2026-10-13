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


def test_intern_s2_boundary_sanitizes_inputs_and_model_embeddings(
    monkeypatch, fake_intern_s2_397b
):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2_397b,
    )
    original_forward = fake_intern_s2_397b.forward

    def forward_with_nonfinite_embeddings(**kwargs):
        embeddings, pad_mask, ts_embeddings = original_forward(**kwargs)
        embeddings[0, 0, 0] = float("nan")
        embeddings[0, 0, 1] = float("inf")
        return embeddings, pad_mask, ts_embeddings

    fake_intern_s2_397b.forward = forward_with_nonfinite_embeddings
    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b", model_name="unused", max_ts_length=512
    )
    history = torch.tensor([[[1.0], [float("nan")], [float("-inf")]]])

    output = encoder(history)

    assert torch.isfinite(
        fake_intern_s2_397b.forward_kwargs["time_series_signals"]
    ).all()
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
@pytest.mark.parametrize("all_masked", [False, True])
def test_mrqformer_zeroes_fully_masked_rows_without_mutating_mask(
    mask_dtype, all_masked
):
    from src.encoders.intern_s2_preview_397b.modeling_interns2_preview import MRQFormer

    class RecordingTransformer(nn.Module):
        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            mask = kwargs["encoder_attention_mask"]
            hidden = kwargs["inputs_embeds"] + kwargs["encoder_hidden_states"].mean(
                dim=1, keepdim=True
            )
            hidden = hidden.masked_fill(~mask.bool().any(dim=1)[:, None, None], float("nan"))
            return SimpleNamespace(last_hidden_state=hidden)

    qformer = MRQFormer([2], hidden_size=16, output_size=8).eval()
    qformer.transformer = RecordingTransformer()
    qformer.proj_out = nn.Linear(16, 8)
    with torch.no_grad():
        qformer.proj_out.weight.fill_(0.1)
        qformer.proj_out.bias.fill_(1.0)
    features = torch.randn(3, 4, 16, requires_grad=True)
    mask = torch.tensor(
        [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=mask_dtype
    )
    if all_masked:
        mask.zero_()
    original_mask = mask.clone()
    fully_masked = ~mask.bool().any(dim=1)

    output = qformer(features, attention_mask=mask)

    assert torch.equal(mask, original_mask)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[fully_masked]) == 0
    forwarded = qformer.transformer.forward_kwargs
    assert forwarded["use_cache"] is False
    assert forwarded["encoder_attention_mask"].bool().any(dim=1).all()
    assert torch.equal(
        forwarded["encoder_attention_mask"][~fully_masked], mask[~fully_masked]
    )
    expected = qformer.proj_out(
        qformer.query_token_list[0].expand(3, -1, -1)
        + features.mean(dim=1, keepdim=True)
    )
    torch.testing.assert_close(output[~fully_masked], expected[~fully_masked])

    output.sum().backward()

    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[fully_masked]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in qformer.parameters()
    )


@pytest.fixture
def small_intern_s2_397b_model():
    from src.encoders.intern_s2_preview_397b.configuration_interns2_preview import (
        InternS2PreviewTimeSeriesConfig,
    )
    from src.encoders.intern_s2_preview_397b.modeling_interns2_preview import (
        InternS2PreviewTimeSeriesModel,
    )

    config = InternS2PreviewTimeSeriesConfig(
        chunk_size=8,
        chunk_step=8,
        subsampling_hidden_dim=32,
        subsampling_num_query=2,
        subsampling_num_conv_layers=0,
        subsampling_selfatt=False,
        d_model=16,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        encoder_layers=1,
        ts_adapt_in_dim=256,
        ts_adapt_out_dim=16,
        ts_hidden_dim=16,
        out_hidden_size=8,
    )
    return InternS2PreviewTimeSeriesModel(config).eval()


@pytest.mark.integration
@pytest.mark.parametrize("length", [8, 9, 17])
def test_intern_s2_397b_chunk_boundaries_produce_finite_raw_embeddings(
    length, small_intern_s2_397b_model
):
    model = small_intern_s2_397b_model

    with torch.no_grad():
        embeddings, pad_mask, hidden = model(
            time_series_signals=torch.randn(1, length, 2),
            ts_lens=torch.tensor([length]),
            sr=torch.ones(1),
            channels=torch.tensor([2]),
        )

    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(hidden).all()
    assert embeddings.shape[:2] == pad_mask.shape
    assert (~pad_mask).any()


@pytest.mark.integration
@pytest.mark.parametrize("replacement", [1.0, float("nan")])
def test_intern_s2_397b_invalid_qformer_rows_cannot_change_valid_embeddings(
    replacement, small_intern_s2_397b_model
):
    model = small_intern_s2_397b_model
    inputs = dict(
        time_series_signals=torch.randn(2, 17, 2),
        ts_lens=torch.tensor([9, 17]),
        sr=torch.ones(2),
        channels=torch.tensor([1, 2]),
    )
    with torch.no_grad():
        baseline, baseline_mask, _ = model(**inputs)
    invalid_counts = []

    def perturb_invalid(module, args, kwargs, output):
        invalid = ~kwargs["attention_mask"].bool().any(dim=-1)
        invalid_counts.append(int(invalid.sum()))
        return output.masked_fill(invalid[:, None, None], replacement)

    handle = model.encoder_embed.mrqformer.register_forward_hook(
        perturb_invalid, with_kwargs=True
    )
    try:
        with torch.no_grad():
            perturbed, perturbed_mask, hidden = model(**inputs)
    finally:
        handle.remove()

    assert sum(invalid_counts) > 0
    assert torch.equal(baseline_mask, perturbed_mask)
    assert torch.isfinite(perturbed).all()
    assert torch.isfinite(hidden).all()
    torch.testing.assert_close(perturbed[~perturbed_mask], baseline[~baseline_mask])
    assert torch.count_nonzero(perturbed[perturbed_mask]) == 0


def test_intern_s2_397b_channel_mixing_excludes_invalid_rows_and_gradients(
    small_intern_s2_397b_model
):
    frontend = small_intern_s2_397b_model.encoder_embed
    qformer_outputs = torch.randn(4, 2, 32, requires_grad=True)
    frontend.mrqformer.forward = lambda *args, **kwargs: qformer_outputs
    inputs = torch.randn(2, 2, 2, 32)
    mask = torch.tensor([[[1, 0], [1, 0]], [[0, 0], [0, 0]]])
    output = frontend.forward_encoder(inputs, mask)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[1]) == 0
    output.square().sum().backward()
    assert torch.isfinite(qformer_outputs.grad).all()
    assert torch.count_nonzero(qformer_outputs.grad[1:]) == 0
    assert torch.count_nonzero(qformer_outputs.grad[0]) > 0


def test_intern_s2_397b_channel_mixing_compacts_padding_without_attention_mask(
    small_intern_s2_397b_model,
):
    frontend = small_intern_s2_397b_model.encoder_embed
    valid = torch.tensor(
        [[1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )
    features = torch.randn(3, 5, 32)
    calls = []

    def record_mask(module, args, kwargs):
        calls.append(kwargs.get("src_key_padding_mask"))

    handle = frontend.channel_encoder.register_forward_pre_hook(
        record_mask, with_kwargs=True
    )
    try:
        actual_input = features.clone().requires_grad_()
        actual = frontend._mix_valid_channels(actual_input, valid)
        (actual_grad,) = torch.autograd.grad(actual.square().sum(), actual_input)
    finally:
        handle.remove()

    reference_input = features.clone().requires_grad_()
    padding_mask = ~valid.T
    fully_padded = padding_mask.all(dim=-1)
    padding_mask[fully_padded, 0] = False
    reference = frontend.channel_encoder(
        reference_input, src_key_padding_mask=padding_mask
    )
    reference = reference.masked_fill(~valid.unsqueeze(-1), 0.0)
    reference = reference.sum(dim=0) / valid.sum(dim=0).clamp_min(1).unsqueeze(-1)
    (reference_grad,) = torch.autograd.grad(reference.square().sum(), reference_input)

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)
    assert all(mask is None for mask in calls)


def test_intern_s2_397b_channel_mixing_casts_attention_output_to_input_dtype(
    small_intern_s2_397b_model,
):
    frontend = small_intern_s2_397b_model.encoder_embed
    frontend.channel_encoder.forward = lambda src, *args, **kwargs: src.float()
    features = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    valid = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    mixed = frontend._mix_valid_channels(features, valid)

    assert mixed.dtype == features.dtype
    assert torch.isfinite(mixed).all()


@pytest.mark.parametrize("causal", [False, True])
def test_intern_s2_397b_temporal_padding_cannot_change_valid_tokens(
    causal, small_intern_s2_397b_model
):
    encoder = small_intern_s2_397b_model.encoder
    inputs = torch.randn(2, 8, 256, requires_grad=True)
    lengths = torch.tensor([3, 8])
    mask = torch.arange(8).unsqueeze(0) >= lengths.unsqueeze(1)
    output_mask = torch.arange(4).unsqueeze(0) >= ((lengths + 1) // 2).unsqueeze(1)
    baseline = encoder(inputs, lengths, mask, causal=causal).last_hidden_state
    perturbed_inputs = inputs.detach().masked_fill(mask.unsqueeze(-1), float("nan"))
    perturbed = encoder(perturbed_inputs, lengths, mask, causal=causal).last_hidden_state

    assert torch.isfinite(perturbed).all()
    torch.testing.assert_close(perturbed[~output_mask], baseline[~output_mask])
    assert torch.count_nonzero(perturbed[output_mask]) == 0
    baseline[~output_mask].sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[mask]) == 0


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


def test_intern_s2_397b_passes_long_series_through_for_internal_chunking(
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
    inputs = [torch.zeros(48, 2), torch.zeros(2048, 1)]

    output = encoder(inputs)

    assert encoder.model.forward_kwargs["ts_lens"].tolist() == [48, 2048]
    assert encoder.model.forward_kwargs["time_series_signals"].shape == (2, 2048, 2)
    assert output.shape[1] == encoder.tokens_per_instance()
    assert output.shape[1] <= encoder.max_ts_length


@pytest.mark.parametrize("as_list", [True, False])
def test_intern_s2_397b_caps_history_to_most_recent_steps(
    monkeypatch, fake_intern_s2_397b, as_list
):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2_397b,
    )
    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b", model_name="unused", max_ts_length=512
    )
    cap = TimeSeriesEncoder.INTERN_S2_397B_MAX_HISTORY
    assert cap == 300_000
    long_series = torch.arange(cap + 1, dtype=torch.float32).unsqueeze(-1)
    inputs = [long_series, torch.zeros(cap, 1)] if as_list else long_series.unsqueeze(0)

    encoder(inputs)

    kwargs = encoder.model.forward_kwargs
    expected_lens = [cap, cap] if as_list else [cap]
    assert kwargs["ts_lens"].tolist() == expected_lens
    assert kwargs["time_series_signals"].shape[1] == cap
    assert kwargs["time_series_signals"][0, 0, 0] == 1.0
    assert kwargs["time_series_signals"][0, -1, 0] == float(cap)


def test_intern_s2_397b_honors_configured_max_history(monkeypatch, fake_intern_s2_397b):
    monkeypatch.setattr(
        "src.encoders.time_series._load_intern_s2_model",
        lambda *args, **kwargs: fake_intern_s2_397b,
    )
    encoder = TimeSeriesEncoder(
        encoder_type="intern_s2_397b", model_name="unused", max_ts_length=512, max_history=512
    )
    series = torch.arange(2048, dtype=torch.float32).unsqueeze(-1)

    encoder([series, torch.zeros(100, 1)])

    kwargs = encoder.model.forward_kwargs
    assert kwargs["ts_lens"].tolist() == [512, 100]
    assert kwargs["time_series_signals"][0, 0, 0] == 1536.0
    assert kwargs["time_series_signals"][0, 511, 0] == 2047.0


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
