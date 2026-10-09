"""TimeSeriesModalityProcessor — data plane only (VLLM-4).

Encoder wiring (model class side) is VLLM-5. This module is what vLLM uses
to ingest a `time_series` tensor from `multi_modal_data`, place a single
`<time_series>` token in the prompt, and emit a BatchFeature with a
`time_series` field.

Shapes match PRISM's training-side TimeSeriesEncoder
(src/encoders/time_series.py): inputs are (T, V) or (B, T, V) float tensors.
Linear and Moirai inputs are padded to `(B, max_ts_length, num_vars)`;
TimeOmni preserves raw sequence length so the shared encoder can serialize
values dynamically. Its placeholder expands to `timeomni_max_patches` tokens.

"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ProcessorBatchItems

from .base import ModalityProcessor
from .registry import register_modality_processor

DEFAULT_MAX_TS_LENGTH = 512
DEFAULT_NUM_VARS = 1
DEFAULT_PATCH_SIZE = 16
PRISM_TS_TOKEN = "<time_series>"


class _TimeOmniEncoderKwargs(TypedDict, total=False):
    """TimeOmni-only kwargs forwarded to TimeSeriesEncoder.

    Spelled out as a TypedDict so the conditional `**` splat in
    build_encoder() keeps a per-key type instead of collapsing to the
    join of all the value types.
    """

    timeomni_patch_len: int | list[int]
    timeomni_stride: int | list[int] | None
    timeomni_d_model: int
    timeomni_dropout: float
    timeomni_ts_tokens: int
    timeomni_max_patches: int


class TimeSeriesProcessorItems(ProcessorBatchItems[torch.Tensor]):
    """vLLM data-items wrapper for a list of time-series tensors.

    vLLM's MultiModalDataParser dispatches each modality key in `mm_data`
    to a parser function that returns one of these. Without this subclass,
    a 3-D torch tensor would be silently routed through `is_embeddings`
    (which returns True for ndim==3) and treated as a pre-encoded
    embedding — wrong for raw time series.
    """

    def __init__(self, data: Sequence[torch.Tensor]) -> None:
        super().__init__(data, modality="time_series")


class TimeSeriesModalityProcessor(ModalityProcessor):
    """Time-series data plane (raw values -> padded BatchFeature tensor)."""

    MODALITY_NAME = "time_series"
    MM_KWARG_KEY = "time_series"

    def __init__(
        self,
        *,
        placeholder_token: str = PRISM_TS_TOKEN,
        placeholder_token_id: int,
        prism_subconfig: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            modality=self.MODALITY_NAME,
            placeholder_token=placeholder_token,
            placeholder_token_id=placeholder_token_id,
            mm_kwarg_key=self.MM_KWARG_KEY,
            prism_subconfig=prism_subconfig,
        )
        self.max_ts_length = int(
            self.prism_subconfig.get("max_ts_length", DEFAULT_MAX_TS_LENGTH)
        )
        self.num_vars = int(
            self.prism_subconfig.get("num_vars", DEFAULT_NUM_VARS)
        )
        self.patch_size = int(
            self.prism_subconfig.get("patch_size", DEFAULT_PATCH_SIZE)
        )
        self.encoder_type = str(
            self.prism_subconfig.get("encoder_type", "moirai")
        )
        if self.encoder_type not in {"linear", "moirai", "intern_s2", "intern_s2_397b", "timeomni"}:
            raise ValueError(
                f"Unsupported encoder_type {self.encoder_type!r}; "
                "expected 'linear', 'moirai', 'intern_s2', 'intern_s2_397b', or 'timeomni'."
            )
        self.intern_s2_sampling_rate = float(
            self.prism_subconfig.get("intern_s2_sampling_rate", 1.0)
        )
        # RPC mode: run intern_s2/intern_s2_397b in a separate sidecar
        # process instead of importing TimeSeriesEncoder (and therefore
        # transformers>=5.2.0) into this (vLLM, transformers<5) process. See
        # build_encoder() below and tools/intern_s2_sidecar.py. Defaults to
        # False everywhere, so every existing checkpoint/config is
        # byte-identical in behavior unless the export explicitly opts in.
        self.use_rpc_encoder = bool(
            self.prism_subconfig.get("intern_s2_rpc", False)
        )
        if self.use_rpc_encoder and self.encoder_type not in ("intern_s2", "intern_s2_397b"):
            raise ValueError(
                "intern_s2_rpc=True is only valid for encoder_type in "
                f"('intern_s2', 'intern_s2_397b'); got encoder_type={self.encoder_type!r}. "
                "linear/moirai/timeomni always run in-process."
            )
        self._rpc_hidden_dim: int | None = self.prism_subconfig.get(
            "intern_s2_rpc_hidden_dim"
        )
        if self._rpc_hidden_dim is not None:
            self._rpc_hidden_dim = int(self._rpc_hidden_dim)
        # Socket path is deliberately NOT baked into the exported config —
        # it's a per-job/per-node runtime detail. Resolved from
        # PRISM_INTERN_S2_SOCKET at build_encoder() time; the subconfig
        # override below exists only for tests that need a fixed path.
        self._rpc_socket_path: str | None = self.prism_subconfig.get(
            "intern_s2_rpc_socket_path"
        )
        self._rpc_timeout_s = float(
            self.prism_subconfig.get("intern_s2_rpc_timeout_s", 30.0)
        )
        # intern_s2_397b's Q-former subsampling has no closed-form token-count
        # formula, so num_tokens() reads this value from the export instead
        # (see checkpoint_export.py's --ts-tokens-per-instance).
        self.ts_tokens_per_instance: int | None = self.prism_subconfig.get(
            "ts_tokens_per_instance"
        )
        if self.ts_tokens_per_instance is not None:
            self.ts_tokens_per_instance = int(self.ts_tokens_per_instance)
        # TimeOmni-specific knobs (ignored for other encoder types)
        self.timeomni_patch_len: list[int] = self.prism_subconfig.get(
            "timeomni_patch_len", [16]
        )
        if isinstance(self.timeomni_patch_len, int):
            self.timeomni_patch_len = [self.timeomni_patch_len]
        self.timeomni_stride: list[int] | None = self.prism_subconfig.get(
            "timeomni_stride", None
        )
        if isinstance(self.timeomni_stride, int):
            self.timeomni_stride = [self.timeomni_stride]
        self.timeomni_d_model: int = int(
            self.prism_subconfig.get("timeomni_d_model", 512)
        )
        self.timeomni_dropout: float = float(
            self.prism_subconfig.get("timeomni_dropout", 0.1)
        )
        self.timeomni_ts_tokens: int = int(
            self.prism_subconfig.get("timeomni_ts_tokens", 100)
        )
        self.timeomni_max_patches: int = int(
            self.prism_subconfig.get("timeomni_max_patches", 100)
        )
        # Start/end token IDs that bracket the feature span in the post-update
        # sequence. Training's UnifiedTransformer expects this envelope (see
        # src/model.py:516-520 — start_idx, end_idx must be adjacent in
        # input_ids and the LM sees `[start_emb, *feature_embs, end_emb]`
        # after splice). For vLLM parity we emit a PromptUpdateDetails whose
        # `full` is `[start_id, *[ts_id]*N, end_id]` with `is_embed` masking
        # only the N middle positions.
        #
        # Both `ts_start_id` and `ts_end_id` are optional (None) — when
        # neither is set we fall back to a plain PromptReplacement of
        # `[ts_id]*N`. That keeps the data-plane VLLM-4 tests passing and
        # gives a graceful degradation when an export forgot to write them.
        self.ts_start_id: int | None = self.prism_subconfig.get("ts_start_id")
        if self.ts_start_id is not None:
            self.ts_start_id = int(self.ts_start_id)
        self.ts_end_id: int | None = self.prism_subconfig.get("ts_end_id")
        if self.ts_end_id is not None:
            self.ts_end_id = int(self.ts_end_id)
        if (self.ts_start_id is None) != (self.ts_end_id is None):
            raise ValueError(
                "ts_start_id and ts_end_id must both be set or both be None; "
                f"got start={self.ts_start_id!r} end={self.ts_end_id!r}"
            )
        # d_model is needed to size the linear encoder type's projection and
        # to surface the encoder's hidden dim back to the model class. Read
        # from the per-modality subconfig (preferred) or top-level d_model
        # (fallback) at engine-boot time, not here — prism_subconfig may not
        # carry it. See build_encoder().

    # ------------------------------------------------------------------
    # ModalityProcessor surface

    def num_tokens(self, item: Any) -> int:
        # Must match TimeSeriesEncoder.tokens_per_instance — otherwise vLLM
        # raises a placeholder-count mismatch at splice time. The patch path
        # divides `max_ts_length` by `patch_size`; the linear path keeps
        # each timestep as its own token; timeomni uses a fixed budget of
        # timeomni_max_patches tokens TOTAL — training's _forward_timeomni
        # flattens any multivariate (T, V) sample into a single (V*T, 1)
        # sequence before patching (one span, not V separate spans), so
        # unlike linear/moirai this must NOT be multiplied by num_vars.
        if self.encoder_type == "linear":
            return self.max_ts_length
        if self.encoder_type == "timeomni":
            return self.timeomni_max_patches
        if self.encoder_type == "intern_s2_397b":
            if self.ts_tokens_per_instance is None:
                raise ValueError(
                    "encoder_type='intern_s2_397b' requires "
                    "prism_subconfig['time_series']['ts_tokens_per_instance'] "
                    "(see checkpoint_export.py's --ts-tokens-per-instance)."
                )
            return self.ts_tokens_per_instance
        if self.encoder_type == "intern_s2":
            stride = math.floor(
                160 / ((1 + math.exp(-self.intern_s2_sampling_rate / 100)) ** 6)
            )
            patch_count = math.ceil(
                (self.max_ts_length - 2 * stride) / stride
            ) + 1
            return ((patch_count // 2) + 1) // 2
        return (self.max_ts_length // self.patch_size) * self.num_vars

    def encode(self, raw: Any) -> torch.Tensor:
        """Accept numpy or torch tensor of shape (T, V) or (B, T, V); pad/
        truncate along T to `max_ts_length` and return (B, T, V) float."""
        t = self._to_3d_float_tensor(raw)
        if self.encoder_type == "timeomni":
            # TimeOmni serializes each (T, V) input into a dynamic V*T value
            # stream. Padding here would add synthetic serialized values and
            # change its patch selection; over-budget streams are decimated by
            # TimeSeriesEncoder using the same path as training.
            if t.shape[-1] != self.num_vars:
                raise ValueError(
                    f"time_series last-dim mismatch: expected num_vars={self.num_vars}, "
                    f"got V={t.shape[-1]}"
                )
            return t
        return self._pad_or_truncate(t)

    def dummy_item(
        self,
        *,
        mm_options: Mapping[str, object] | None,
        count: int,
    ) -> torch.Tensor:
        # vLLM's memory profiler asks for `count` worst-case items; we hand
        # back a single concatenated tensor of zeros at the maximum shape.
        return torch.zeros(max(count, 1), self.max_ts_length, self.num_vars)

    def field_config(self) -> MultiModalFieldConfig:
        return MultiModalFieldConfig.batched("time_series")

    def normalize_mm_data_key(self, mm_data: Mapping[str, Any]) -> Any | None:
        # vLLM's `ProcessorBatchItems.get_processor_data` returns
        # `{f"{modality}s": [items]}` (plural with trailing "s"), so the
        # canonical key from vLLM-internal flow is "time_seriess". User-
        # facing callers typically use "time_series" / "timeseries" / "ts".
        # Accept all four.
        for key in ("time_series", "time_seriess", "timeseries", "ts"):
            if key in mm_data:
                return mm_data[key]
        return None

    # ------------------------------------------------------------------
    # Encoder construction (model-plane). Called by
    # PrismForConditionalGeneration.__init__ at engine-boot time to build
    # self.encoders["time_series"]. Returns (inner, hidden, forward_fn) for
    # _PrismEncoderWrapper.
    #
    # We reuse training's TimeSeriesEncoder class wholesale rather than
    # re-implement Moirai's forward — see plan §VLLM-5 "Top risk":
    # Moirai2Module.forward threads scaler/observed_mask/sample_id/variate_id/
    # time_id/packed_causal_attention_mask. Importing the class lets the
    # vLLM model class consume the same nn.Module training produces, and
    # weight key prefixes match (`encoders.time_series.*` straight from the
    # exported checkpoint).

    def builds_complete_encoder(self) -> bool:
        # TimeSeriesEncoder owns its own forward (patching, scaling, Moirai
        # encoder block) — registering it directly under
        # `encoders["time_series"]` matches the training-side state-dict
        # path `encoders.time_series.model.*` and skips the wrapper's
        # extra `.model` level.
        return True

    def build_encoder(self) -> tuple[Any, int, Any]:
        if self.use_rpc_encoder:
            # Intern-S2's vendored config requires transformers>=5.2.0
            # (RopeParameters), incompatible with vLLM serving's
            # transformers<5 floor. Never import TimeSeriesEncoder (or
            # transformers, or Intern-S2's vendored modules) in this
            # process — the encoder runs in tools/intern_s2_sidecar.py's
            # separate venv instead, reached over a Unix domain socket.
            from .intern_s2_rpc_encoder import InternS2RPCEncoder

            if self._rpc_hidden_dim is None:
                raise KeyError(
                    "intern_s2_rpc=True requires "
                    "prism_subconfig['time_series']['intern_s2_rpc_hidden_dim'] "
                    "(see checkpoint_export.py's --ts-intern-s2-rpc-hidden-dim)."
                )
            socket_path = self._rpc_socket_path or os.environ.get(
                "PRISM_INTERN_S2_SOCKET"
            )
            if not socket_path:
                raise KeyError(
                    "intern_s2_rpc=True requires PRISM_INTERN_S2_SOCKET to be "
                    "set (the launcher sets this before starting vLLM; see "
                    "tools/vllm_serve_with_intern_s2.sh)."
                )
            rpc_encoder = InternS2RPCEncoder(
                hidden_dim=self._rpc_hidden_dim,
                socket_path=socket_path,
                timeout_s=self._rpc_timeout_s,
            )
            return rpc_encoder, self._rpc_hidden_dim, lambda model, x: model(x)

        # Lazy import: TimeSeriesEncoder pulls uni2ts/Moirai when
        # encoder_type=="moirai", which is heavy. The model class only calls
        # build_encoder() when "time_series" is in active_modalities, so the
        # image-only path stays free of this dependency. Non-RPC
        # intern_s2*/intern_s2_397b also lands here — RPC mode is opt-in.
        from src.encoders.time_series import TimeSeriesEncoder

        # d_ts (hidden_dim) follows the inner encoder's `d_model` for moirai
        # (Moirai2Module owns it) or the projector input_dim we configure for
        # linear. Prefer the explicit `d_ts`; fall back to the per-modality
        # `d_model` only if present. A missing value is a config bug — fail
        # loud rather than picking a silent default that may mismatch
        # training's projector shape.
        d_ts_val = self.prism_subconfig.get("d_ts")
        if d_ts_val is None:
            d_ts_val = self.prism_subconfig.get("d_model")
        if d_ts_val is None:
            raise KeyError(
                "TimeSeriesModalityProcessor.build_encoder requires `d_ts` "
                "(or `d_model`) in prism_subconfig['time_series']. Re-export "
                "the checkpoint with --ts-d-ts <hidden> set to training's d_ts."
            )
        d_ts = int(d_ts_val)
        encoder_model = str(
            self.prism_subconfig.get(
                "encoder_model", "Salesforce/moirai-2.0-R-small"
            )
        )

        timeomni_kwargs: _TimeOmniEncoderKwargs = (
            {
                "timeomni_patch_len": self.timeomni_patch_len,
                "timeomni_stride": self.timeomni_stride,
                "timeomni_d_model": self.timeomni_d_model,
                "timeomni_dropout": self.timeomni_dropout,
                "timeomni_ts_tokens": self.timeomni_ts_tokens,
                "timeomni_max_patches": self.timeomni_max_patches,
            }
            if self.encoder_type == "timeomni"
            else {}
        )

        inner = TimeSeriesEncoder(
            encoder_type=self.encoder_type,
            num_vars=self.num_vars,
            d_ts=d_ts,
            model_name=encoder_model,
            max_ts_length=self.max_ts_length,
            is_interleaved=self.encoder_type == "timeomni",
            intern_s2_sampling_rate=self.intern_s2_sampling_rate,
            **timeomni_kwargs,
        )
        hidden = int(getattr(inner, "hidden_dim", d_ts))

        def _forward(model: Any, x: Any) -> Any:
            # TimeSeriesEncoder.forward signature is `(inputs: torch.Tensor)`
            # of shape (B, T, V) and returns (B, Num_Patches, Hidden). The
            # _PrismEncoderWrapper.proj (Identity when hidden == d_target)
            # is applied afterward by the wrapper.
            return model(x)

        return inner, hidden, _forward

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _to_3d_float_tensor(raw: Any) -> torch.Tensor:
        # vLLM's data parser hands us a list of per-item (T, V) tensors
        # (each item is one time-series instance). Stack them along the
        # batch axis here. A bare tensor is treated as a single item.
        if isinstance(raw, list | tuple):
            stacked = []
            for item in raw:
                if isinstance(item, torch.Tensor):
                    t_item = item
                else:
                    t_item = torch.as_tensor(item)
                if t_item.ndim == 2:
                    stacked.append(t_item)
                elif t_item.ndim == 3 and t_item.shape[0] == 1:
                    stacked.append(t_item.squeeze(0))
                else:
                    raise ValueError(
                        "list-of-tensors items must be (T, V) each; "
                        f"got ndim={t_item.ndim} shape={tuple(t_item.shape)}"
                    )
            t = torch.stack(stacked, dim=0)
        elif isinstance(raw, torch.Tensor):
            t = raw
        else:
            t = torch.as_tensor(raw)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim != 3:
            raise ValueError(
                "time_series input must be shape (T, V) or (B, T, V); "
                f"got ndim={t.ndim} shape={tuple(t.shape)}"
            )
        return t

    def _pad_or_truncate(self, t: torch.Tensor) -> torch.Tensor:
        """Pad along the T axis with zeros, or right-truncate."""
        b, T, v = t.shape
        if v != self.num_vars:
            raise ValueError(
                f"time_series last-dim mismatch: expected num_vars={self.num_vars}, "
                f"got V={v}"
            )
        if T == self.max_ts_length:
            return t
        if T > self.max_ts_length:
            return t[:, : self.max_ts_length, :]
        pad = torch.zeros(
            b, self.max_ts_length - T, v, dtype=t.dtype, device=t.device
        )
        return torch.cat([t, pad], dim=1)


# ---------------------------------------------------------------------------
# Registry: pulled in by importing this module. The orchestrator triggers
# this import via `from .processors.time_series import ...` at engine boot
# when "time_series" appears in active_modalities.


@register_modality_processor("time_series")
def _build_time_series_processor(prism_cfg: Mapping[str, Any]) -> TimeSeriesModalityProcessor:
    sub = dict(prism_cfg.get("time_series", {}))
    placeholder = sub.get("placeholder_token") or PRISM_TS_TOKEN
    placeholder_id = sub.get("placeholder_token_id")
    if placeholder_id is None:
        raise KeyError(
            "prism_config['time_series']['placeholder_token_id'] is required "
            "for the time_series modality. Re-export the checkpoint with "
            "`--active-modalities image,time_series`."
        )
    return TimeSeriesModalityProcessor(
        placeholder_token=str(placeholder),
        placeholder_token_id=int(placeholder_id),
        prism_subconfig=sub,
    )


# Used by the data parser in orchestrator.py to dispatch a raw time-series
# payload onto a TimeSeriesProcessorItems wrapper.
def parse_time_series_data(
    data: Any,
) -> TimeSeriesProcessorItems | None:
    """vLLM subparser: turn raw user data into TimeSeriesProcessorItems.

    Accepts:
      - torch.Tensor of shape (T, V), (B, T, V), or (N_items, T, V) — each
        first-dim entry is one item;
      - list of any of the above (None entries are silently skipped, matching
        the audio/video subparsers in vLLM's default parser);
      - numpy arrays (converted via torch.as_tensor).

    Returns None for empty input.
    """
    if data is None:
        return None
    items: list[torch.Tensor] = []
    sequence: Sequence[Any]
    if isinstance(data, list | tuple):
        sequence = data
    elif isinstance(data, torch.Tensor) and data.ndim >= 3:
        sequence = [data[i] for i in range(data.shape[0])]
    else:
        sequence = [data]

    for one in sequence:
        if one is None:
            continue
        if isinstance(one, torch.Tensor):
            items.append(one)
        else:
            items.append(torch.as_tensor(one))

    if not items:
        return None
    return TimeSeriesProcessorItems(items)


__all__ = [
    "DEFAULT_MAX_TS_LENGTH",
    "DEFAULT_NUM_VARS",
    "DEFAULT_PATCH_SIZE",
    "PRISM_TS_TOKEN",
    "TimeSeriesModalityProcessor",
    "TimeSeriesProcessorItems",
    "parse_time_series_data",
]
