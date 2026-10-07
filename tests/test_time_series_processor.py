"""Login-node tests for VLLM-4: TimeSeriesModalityProcessor (data plane).

Encoder wiring (VLLM-5) and the full HF/vLLM parity oracle live on compute
nodes; here we cover the contract: padding, truncation, num_tokens, key
synonyms, data parser dispatch, and the PromptUpdate vLLM will emit.

No XPU, no engine boot.

"""

from __future__ import annotations

import pytest
import torch

vllm = pytest.importorskip("vllm")


# ---------------------------------------------------------------------------
# Shape / padding / truncation


@pytest.mark.parametrize("input_t", [1, 50, 100, 256, 511])
def test_encode_pads_up_to_max_length(input_t: int):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    out = proc.encode(torch.randn(input_t, 1))
    assert tuple(out.shape) == (1, 512, 1)
    # The first `input_t` rows are the original data; the rest are zero pad.
    if input_t < 512:
        torch.testing.assert_close(
            out[0, input_t:, :], torch.zeros(512 - input_t, 1)
        )


@pytest.mark.parametrize("input_t", [513, 1000, 4096])
def test_encode_truncates_to_max_length(input_t: int):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    out = proc.encode(torch.randn(input_t, 1))
    assert tuple(out.shape) == (1, 512, 1)


def test_encode_preserves_batch_axis():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    out = proc.encode(torch.randn(4, 300, 1))
    assert tuple(out.shape) == (4, 512, 1)


def test_encode_accepts_numpy():
    import numpy as np
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    out = proc.encode(np.random.randn(100, 1).astype("float32"))
    assert tuple(out.shape) == (1, 512, 1)
    assert out.dtype == torch.float32


def test_encode_rejects_wrong_ndim():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    with pytest.raises(ValueError, match="ndim"):
        proc.encode(torch.randn(10))  # 1-D
    with pytest.raises(ValueError, match="ndim"):
        proc.encode(torch.randn(2, 3, 4, 5))  # 4-D


def test_encode_rejects_num_vars_mismatch():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"num_vars": 3},
    )
    with pytest.raises(ValueError, match="num_vars"):
        proc.encode(torch.randn(100, 1))  # only 1 var, expected 3


def test_intern_s2_397b_accepted_as_encoder_type():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"encoder_type": "intern_s2_397b", "ts_tokens_per_instance": 64},
    )
    assert proc.num_tokens(None) == 64


def test_intern_s2_397b_requires_ts_tokens_per_instance():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"encoder_type": "intern_s2_397b"},
    )
    with pytest.raises(ValueError, match="ts_tokens_per_instance"):
        proc.num_tokens(None)


def test_timeomni_encode_preserves_multivariate_variable_length():
    """TimeOmni must serialize the supplied values, not fixed-pad its T axis."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "timeomni",
            "num_vars": 3,
            "max_ts_length": 512,
            "timeomni_max_patches": 32,
        },
    )
    raw = torch.arange(45, dtype=torch.float32).reshape(15, 3)

    encoded = proc.encode(raw)

    assert torch.equal(encoded, raw.unsqueeze(0))
    assert proc.num_tokens(item=raw) == 32


# ---------------------------------------------------------------------------
# num_tokens parametrized


@pytest.mark.parametrize(
    "max_len,patch_size,num_vars,expected",
    [
        (512, 16, 1, 32),   # baseline Moirai
        (512, 16, 4, 128),  # multi-variate
        (256, 16, 1, 16),
        (1024, 32, 2, 64),
        (256, 8, 3, 96),
    ],
)
def test_num_tokens_moirai(max_len, patch_size, num_vars, expected):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "max_ts_length": max_len,
            "patch_size": patch_size,
            "num_vars": num_vars,
            "encoder_type": "moirai",
        },
    )
    assert proc.num_tokens(item=None) == expected


def test_rejects_unknown_encoder_type():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    with pytest.raises(ValueError, match="encoder_type"):
        TimeSeriesModalityProcessor(
            placeholder_token_id=50301,
            prism_subconfig={"encoder_type": "morai"},  # typo
        )


def test_dummy_item_shape_matches_worst_case():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"max_ts_length": 256, "num_vars": 3},
    )
    out = proc.dummy_item(mm_options=None, count=4)
    assert tuple(out.shape) == (4, 256, 3)
    # count=0 still returns one item so vLLM's profiler has something to size.
    out_zero = proc.dummy_item(mm_options=None, count=0)
    assert tuple(out_zero.shape) == (1, 256, 3)


def test_field_config_is_batched_time_series():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )
    from vllm.multimodal.inputs import MultiModalFieldConfig

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    fc = proc.field_config()
    assert isinstance(fc, MultiModalFieldConfig)


def test_num_tokens_linear_does_not_divide_by_patch():
    """Encoder_type=linear maps each timestep to a token (no patching)."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "max_ts_length": 512,
            "num_vars": 1,
            "encoder_type": "linear",
        },
    )
    assert proc.num_tokens(item=None) == 512


# ---------------------------------------------------------------------------
# Key synonyms


@pytest.mark.parametrize("key", ["time_series", "timeseries", "ts"])
def test_normalize_mm_data_key_accepts_synonyms(key: str):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    t = torch.randn(10, 1)
    assert proc.normalize_mm_data_key({key: t}) is t


def test_normalize_mm_data_key_returns_none_when_missing():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    assert proc.normalize_mm_data_key({"image": object()}) is None


# ---------------------------------------------------------------------------
# Registry integration


def test_factory_registered():
    from src.vllm_plugin.processors import MODALITY_PROCESSORS

    assert "time_series" in MODALITY_PROCESSORS


def test_factory_requires_placeholder_id():
    """Catch silent regressions where a downstream export forgot to write
    placeholder_token_id."""
    from src.vllm_plugin.processors import build_modality_processors

    with pytest.raises(KeyError, match="placeholder_token_id"):
        build_modality_processors(
            {
                "active_modalities": ["time_series"],
                "time_series": {"placeholder_token": "<time_series>"},
            }
        )


def test_factory_round_trip():
    from src.vllm_plugin.processors import build_modality_processors
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    procs = build_modality_processors(
        {
            "active_modalities": ["time_series"],
            "time_series": {
                "placeholder_token": "<time_series>",
                "placeholder_token_id": 50301,
                "max_ts_length": 256,
                "num_vars": 2,
                "patch_size": 16,
            },
        }
    )
    proc = procs["time_series"]
    assert isinstance(proc, TimeSeriesModalityProcessor)
    assert proc.placeholder_token_id == 50301
    assert proc.max_ts_length == 256
    assert proc.num_vars == 2


# ---------------------------------------------------------------------------
# Data parser dispatch
#
# Without PrismDataParser, vLLM would route a 3-D torch.Tensor through
# is_embeddings (True for ndim==3) and treat it as a pre-encoded embedding.
# We assert that the time_series subparser is wired and produces
# TimeSeriesProcessorItems instances.


def test_data_parser_subparser_emits_processor_items():
    from src.vllm_plugin.processors.orchestrator import PrismDataParser
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesProcessorItems,
    )

    parser = PrismDataParser()
    subparsers = parser._get_subparsers()
    assert "time_series" in subparsers

    parsed = subparsers["time_series"](torch.randn(5, 100, 1))
    assert isinstance(parsed, TimeSeriesProcessorItems)
    assert len(parsed) == 5

    parsed_single = subparsers["time_series"](torch.randn(100, 1))
    assert isinstance(parsed_single, TimeSeriesProcessorItems)
    assert len(parsed_single) == 1


def test_data_parser_passes_through_image():
    """Adding time_series must not break image dispatch."""
    from PIL import Image
    from src.vllm_plugin.processors.orchestrator import PrismDataParser

    parser = PrismDataParser()
    subparsers = parser._get_subparsers()
    assert "image" in subparsers
    parsed = subparsers["image"]([Image.new("RGB", (10, 10), (255, 0, 0))])
    assert parsed is not None
    assert len(parsed) == 1


# ---------------------------------------------------------------------------
# PromptUpdate emission
#
# VLLM-4 emits a basic PromptReplacement (placeholder_id -> [placeholder_id]*N);
# the PromptUpdateDetails envelope (ts_start ... ts_end) is VLLM-5's concern
# because it only matters once an encoder is producing real embeddings.


def test_replacement_yields_n_placeholder_ids():
    """The placeholder_id -> [placeholder_id]*N rule is what the
    orchestrator does today. Lock that contract so VLLM-5's
    PromptUpdateDetails change is intentional."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "max_ts_length": 512,
            "patch_size": 16,
            "num_vars": 1,
            "encoder_type": "moirai",
        },
    )
    # Effective replacement is `[placeholder_token_id] * num_tokens(item=None)`.
    n = proc.num_tokens(item=None)
    replacement = [proc.placeholder_token_id] * n
    assert len(replacement) == 32
    assert all(t == 50301 for t in replacement)
