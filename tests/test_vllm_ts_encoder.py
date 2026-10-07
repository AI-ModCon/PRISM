"""Login-node tests for VLLM-5: TimeSeries encoder wiring + envelope.

No engine boot. Covers:
- TimeSeriesModalityProcessor.build_encoder returns (inner, hidden, fn)
- The forward closure is callable with a (B, T, V) tensor
- ts_start_id / ts_end_id round-trip through the per-modality block
- _build_prompt_update emits PromptUpdateDetails when start/end are set
- _build_prompt_update falls back to plain PromptReplacement when not
- Half-configured start/end pair raises
"""

from __future__ import annotations

import pytest
import torch

vllm = pytest.importorskip("vllm")


# ---------------------------------------------------------------------------
# build_encoder


def test_build_encoder_linear_returns_callable_tuple():
    """Linear encoder type is local-only (no uni2ts network round-trip)."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "linear",
            "num_vars": 1,
            "d_ts": 64,
            "max_ts_length": 32,
        },
    )
    inner, hidden, fwd = proc.build_encoder()
    assert isinstance(inner, torch.nn.Module)
    # TimeSeriesEncoder.hidden_dim defaults to d_ts when not Moirai.
    assert hidden == 64
    assert callable(fwd)


def test_build_encoder_forward_closure_runs():
    """forward_fn(model, x) must accept a (B, T, V) tensor and return
    (B, num_patches, hidden). Linear encoder returns (B, T, d_ts) directly
    (one token per timestep)."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "linear",
            "num_vars": 1,
            "d_ts": 64,
            "max_ts_length": 32,
        },
    )
    inner, hidden, fwd = proc.build_encoder()
    inner.eval()

    # Linear encoder: input (B, T, V) -> output (B, T, d_ts).
    x = torch.zeros(2, 32, 1)
    with torch.no_grad():
        out = fwd(inner, x)
    assert out.shape == (2, 32, 64)


def test_build_encoder_raises_when_d_ts_missing():
    """A config without d_ts or d_model is malformed — fail loud rather
    than fall back to a silent default that may mismatch the projector."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"encoder_type": "linear", "max_ts_length": 16},
    )
    with pytest.raises(KeyError, match="requires `d_ts`"):
        proc.build_encoder()


def test_build_encoder_falls_back_to_d_model():
    """If d_ts isn't set but d_model is, use d_model — same convention as
    training's TimeSeriesEncoder default."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "linear",
            "max_ts_length": 16,
            "d_model": 128,
        },
    )
    _inner, hidden, _fwd = proc.build_encoder()
    assert hidden == 128


def test_builds_complete_encoder_flag():
    from src.vllm_plugin.processors.image import ImageModalityProcessor
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    ts = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    assert ts.builds_complete_encoder() is True

    # Image still wraps because the SigLIP2 inner doesn't expose
    # PRISM's .model/.proj layout.
    img = ImageModalityProcessor(placeholder_token_id=50300)
    assert img.builds_complete_encoder() is False


# ---------------------------------------------------------------------------
# intern_s2_rpc process-isolation flag


def test_build_encoder_intern_s2_rpc_returns_stub_module():
    """With intern_s2_rpc=True, build_encoder must return an
    InternS2RPCEncoder without ever importing TimeSeriesEncoder/transformers
    for the real model -- proving the in-process transformers version
    conflict is actually eliminated, not just avoided by luck."""
    from src.vllm_plugin.processors.intern_s2_rpc_encoder import (
        InternS2RPCEncoder,
    )
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "intern_s2",
            "num_vars": 1,
            "max_ts_length": 32,
            "intern_s2_rpc": True,
            "intern_s2_rpc_hidden_dim": 512,
            "intern_s2_rpc_socket_path": "/tmp/does-not-need-to-exist.sock",
        },
    )
    inner, hidden, fwd = proc.build_encoder()
    assert isinstance(inner, InternS2RPCEncoder)
    assert hidden == 512
    assert callable(fwd)


def test_build_encoder_intern_s2_rpc_requires_hidden_dim():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "intern_s2",
            "num_vars": 1,
            "max_ts_length": 32,
            "intern_s2_rpc": True,
        },
    )
    with pytest.raises(KeyError, match="intern_s2_rpc_hidden_dim"):
        proc.build_encoder()


def test_build_encoder_intern_s2_rpc_requires_socket_path(monkeypatch):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    monkeypatch.delenv("PRISM_INTERN_S2_SOCKET", raising=False)
    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "intern_s2",
            "num_vars": 1,
            "max_ts_length": 32,
            "intern_s2_rpc": True,
            "intern_s2_rpc_hidden_dim": 512,
        },
    )
    with pytest.raises(KeyError, match="PRISM_INTERN_S2_SOCKET"):
        proc.build_encoder()


def test_linear_moirai_timeomni_unaffected_by_rpc_flag():
    """intern_s2_rpc is only valid for encoder_type in
    (intern_s2, intern_s2_397b) -- setting it for any other encoder type
    must raise a clear config error at construction time, not silently
    no-op or corrupt the encoder selection."""
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    for encoder_type in ("linear", "moirai", "timeomni"):
        with pytest.raises(ValueError, match="intern_s2_rpc=True is only valid"):
            TimeSeriesModalityProcessor(
                placeholder_token_id=50301,
                prism_subconfig={
                    "encoder_type": encoder_type,
                    "num_vars": 1,
                    "d_ts": 64,
                    "max_ts_length": 32,
                    "intern_s2_rpc": True,
                },
            )


def test_use_rpc_encoder_defaults_to_false():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={
            "encoder_type": "linear",
            "num_vars": 1,
            "d_ts": 64,
            "max_ts_length": 32,
        },
    )
    assert proc.use_rpc_encoder is False


# ---------------------------------------------------------------------------
# start/end envelope config


def test_start_end_ids_round_trip():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(
        placeholder_token_id=50301,
        prism_subconfig={"ts_start_id": 50280, "ts_end_id": 50281},
    )
    assert proc.ts_start_id == 50280
    assert proc.ts_end_id == 50281


def test_start_end_defaults_to_none():
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    proc = TimeSeriesModalityProcessor(placeholder_token_id=50301)
    assert proc.ts_start_id is None
    assert proc.ts_end_id is None


@pytest.mark.parametrize(
    "start,end",
    [
        (50280, None),
        (None, 50281),
    ],
)
def test_half_configured_envelope_raises(start, end):
    from src.vllm_plugin.processors.time_series import (
        TimeSeriesModalityProcessor,
    )

    sub: dict = {}
    if start is not None:
        sub["ts_start_id"] = start
    if end is not None:
        sub["ts_end_id"] = end
    with pytest.raises(ValueError, match="must both be set or both be None"):
        TimeSeriesModalityProcessor(placeholder_token_id=50301, prism_subconfig=sub)


# ---------------------------------------------------------------------------
# Orchestrator: _build_prompt_update emits the right shape


def _make_processor_info_for_test(
    prism_cfg: dict,
):
    """Build a PrismProcessingInfo-compatible stub that returns prism_cfg.

    We bypass the heavy InputProcessingContext construction by hand-crafting
    the bits PrismMultiModalProcessor._build_prompt_update actually reads.
    """
    from src.vllm_plugin.processors.orchestrator import (
        PrismMultiModalProcessor,
    )

    class _StubInfo:
        def __init__(self, cfg):
            self._cfg = cfg
            self._procs_cache = None

        def _prism_config(self):
            return self._cfg

        def modality_processors(self):
            if self._procs_cache is None:
                from src.vllm_plugin.processors.registry import (
                    build_modality_processors,
                )

                self._procs_cache = build_modality_processors(self._cfg)
            return self._procs_cache

    class _StubProcessor(PrismMultiModalProcessor):
        def __init__(self, info):  # noqa: D401 - test stub
            self.info = info

    return _StubProcessor(_StubInfo(prism_cfg))


def test_build_prompt_update_emits_envelope_when_configured():
    """When ts_start_id / ts_end_id are configured, target the adjacent
    [start, end] pair in input_ids and replace with N feature tokens, all
    is_embed=True. This matches training's _construct_input_embeddings,
    which drops both start AND end embeddings and emits N feature embeddings
    in their place (src/model.py:495-540).
    """
    from vllm.multimodal.processing import PromptReplacement

    proc = _make_processor_info_for_test(
        {
            "active_modalities": ["time_series"],
            "time_series": {
                "placeholder_token": "<time_series>",
                "placeholder_token_id": 50301,
                "encoder_type": "linear",
                "max_ts_length": 32,
                "num_vars": 1,
                "patch_size": 16,
                "ts_start_id": 50280,
                "ts_end_id": 50281,
            },
        }
    )
    ts_proc = proc.info.modality_processors()["time_series"]
    update = proc._build_prompt_update("time_series", ts_proc, mm_items=None)

    assert isinstance(update, PromptReplacement)
    # Target is the adjacent [start, end] pair training inserts in input_ids.
    assert update.target == [50280, 50281]
    # Replacement is N copies of the feature token — start/end embeddings
    # never reach the LM, matching training.
    rep = update.replacement(0)
    n = ts_proc.num_tokens(item=None)
    assert rep == [50301] * n


def test_build_prompt_update_falls_back_to_plain_replacement_without_envelope():
    from vllm.multimodal.processing import PromptReplacement

    proc = _make_processor_info_for_test(
        {
            "active_modalities": ["time_series"],
            "time_series": {
                "placeholder_token": "<time_series>",
                "placeholder_token_id": 50301,
                "encoder_type": "linear",
                "max_ts_length": 32,
                "num_vars": 1,
                "patch_size": 16,
                # No ts_start_id / ts_end_id -> plain replacement
            },
        }
    )
    ts_proc = proc.info.modality_processors()["time_series"]
    update = proc._build_prompt_update("time_series", ts_proc, mm_items=None)

    assert isinstance(update, PromptReplacement)
    rep = update.replacement(0)
    n = ts_proc.num_tokens(item=None)
    assert rep == [50301] * n


# ---------------------------------------------------------------------------
# checkpoint_export.py: ts_start_id/ts_end_id propagate to the per-modality block


def test_time_series_block_includes_envelope():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    block = _time_series_block(
        placeholder_token="<time_series>",
        placeholder_token_id=50301,
        encoder_model="Salesforce/moirai-2.0-R-small",
        max_ts_length=512,
        num_vars=1,
        patch_size=16,
        encoder_type="moirai",
        projector_kwargs={"norm_mode": "layernorm"},
        ts_start_id=50280,
        ts_end_id=50281,
    )
    assert block["ts_start_id"] == 50280
    assert block["ts_end_id"] == 50281


def test_time_series_block_without_envelope():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    block = _time_series_block(
        placeholder_token="<time_series>",
        placeholder_token_id=50301,
        encoder_model="Salesforce/moirai-2.0-R-small",
        max_ts_length=512,
        num_vars=1,
        patch_size=16,
        encoder_type="moirai",
        projector_kwargs={"norm_mode": "layernorm"},
    )
    assert "ts_start_id" not in block
    assert "ts_end_id" not in block


def test_time_series_block_d_ts_round_trip():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    block = _time_series_block(
        placeholder_token="<time_series>",
        placeholder_token_id=50301,
        encoder_model="Salesforce/moirai-2.0-R-small",
        max_ts_length=512,
        num_vars=1,
        patch_size=16,
        encoder_type="linear",
        projector_kwargs={"norm_mode": "layernorm"},
        d_ts=2048,
    )
    assert block["d_ts"] == 2048


@pytest.mark.parametrize(
    "start,end",
    [
        (50280, None),
        (None, 50281),
    ],
)
def test_time_series_block_half_envelope_raises(start, end):
    from src.vllm_plugin.checkpoint_export import _time_series_block

    with pytest.raises(ValueError, match="must both be set or both be None"):
        _time_series_block(
            placeholder_token="<time_series>",
            placeholder_token_id=50301,
            encoder_model="Salesforce/moirai-2.0-R-small",
            max_ts_length=512,
            num_vars=1,
            patch_size=16,
            encoder_type="moirai",
            projector_kwargs={},
            ts_start_id=start,
            ts_end_id=end,
        )


def test_time_series_block_includes_intern_s2_rpc():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    block = _time_series_block(
        placeholder_token="<time_series>",
        placeholder_token_id=50301,
        encoder_model="internlm/Intern-S2",
        max_ts_length=512,
        num_vars=1,
        patch_size=16,
        encoder_type="intern_s2",
        projector_kwargs={"norm_mode": "layernorm"},
        intern_s2_rpc=True,
        intern_s2_rpc_hidden_dim=512,
    )
    assert block["intern_s2_rpc"] is True
    assert block["intern_s2_rpc_hidden_dim"] == 512


def test_time_series_block_intern_s2_rpc_requires_hidden_dim():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    with pytest.raises(ValueError, match="requires --ts-intern-s2-rpc-hidden-dim"):
        _time_series_block(
            placeholder_token="<time_series>",
            placeholder_token_id=50301,
            encoder_model="internlm/Intern-S2",
            max_ts_length=512,
            num_vars=1,
            patch_size=16,
            encoder_type="intern_s2",
            projector_kwargs={},
            intern_s2_rpc=True,
        )


def test_time_series_block_intern_s2_rpc_rejects_other_encoder_types():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    with pytest.raises(ValueError, match="intern_s2_rpc=True is only valid"):
        _time_series_block(
            placeholder_token="<time_series>",
            placeholder_token_id=50301,
            encoder_model="Salesforce/moirai-2.0-R-small",
            max_ts_length=512,
            num_vars=1,
            patch_size=16,
            encoder_type="moirai",
            projector_kwargs={},
            intern_s2_rpc=True,
            intern_s2_rpc_hidden_dim=512,
        )


def test_time_series_block_without_intern_s2_rpc():
    from src.vllm_plugin.checkpoint_export import _time_series_block

    block = _time_series_block(
        placeholder_token="<time_series>",
        placeholder_token_id=50301,
        encoder_model="Salesforce/moirai-2.0-R-small",
        max_ts_length=512,
        num_vars=1,
        patch_size=16,
        encoder_type="moirai",
        projector_kwargs={"norm_mode": "layernorm"},
    )
    assert "intern_s2_rpc" not in block
    assert "intern_s2_rpc_hidden_dim" not in block


def test_detect_lm_vocab_size():
    """vocab_size = embed_tokens.weight.shape[0]; used by the time_series
    export branch to refuse envelope IDs that would OOB-fault the LM."""
    from src.vllm_plugin.checkpoint_export import detect_lm_vocab_size

    sd = {
        "language_model.model.embed_tokens.weight": torch.zeros(50304, 2048),
    }
    assert detect_lm_vocab_size(sd) == 50304

    sd_legacy = {
        "backbone.model.embed_tokens.weight": torch.zeros(100352, 4096),
    }
    assert detect_lm_vocab_size(sd_legacy) == 100352


def test_detect_lm_vocab_size_missing_raises():
    from src.vllm_plugin.checkpoint_export import detect_lm_vocab_size

    with pytest.raises(RuntimeError, match="vocab size"):
        detect_lm_vocab_size({"some.other.weight": torch.zeros(1, 1)})
