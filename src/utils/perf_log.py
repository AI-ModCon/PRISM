"""Structured JSONL throughput log for scaling-sweep aggregation.

Mirrors the per-N-step [THROUGHPUT] / [TIMING] log lines emitted by the
trainers into `<output_dir>/perf.jsonl` (one JSON object per line). The
log is rank-0-only (the trainers already gate logging on `is_main`).

Downstream: `tools/perf_aggregate.py` globs perf.jsonl across `outputs/`
and emits a tidy CSV for paper-quality scaling plots.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from src.modalities import Modality

logger = logging.getLogger(__name__)

_LOG_FILENAMES: dict[str, Path | None] = {}  # output_dir -> resolved log path (None if unwritable)


def _resolve_log_path(output_dir: str | os.PathLike) -> Path | None:
    """Resolve and cache the perf.jsonl path for an output dir.

    Returns None if the output dir cannot be created (e.g. read-only fs);
    logging is best-effort and never raises.
    """
    key = str(output_dir)
    if key in _LOG_FILENAMES:
        return _LOG_FILENAMES[key]
    try:
        path = Path(output_dir) / "perf.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILENAMES[key] = path
        return path
    except OSError as e:
        logger.warning(f"perf_log: cannot create {output_dir}: {e}")
        _LOG_FILENAMES[key] = None
        return None


def log_perf_record(output_dir: str | os.PathLike | None, record: dict[str, Any]) -> None:
    """Append one perf record as a JSON line.

    Must be called from rank 0 only — multiple ranks writing to the same
    file will interleave lines and corrupt the JSONL stream. The trainers
    gate every call on `is_main` / `accelerator.is_main_process`.

    If `output_dir` is None or the file cannot be opened, the call is
    silently dropped (perf telemetry must never kill a training job).
    Always adds a 'wall_time' field (unix seconds) so multi-run
    aggregation can sort.
    """
    if output_dir is None:
        return
    path = _resolve_log_path(output_dir)
    if path is None:
        return
    record = {"wall_time": time.time(), **record}
    try:
        line = json.dumps(record, default=str)
    except Exception as e:  # noqa: BLE001
        # `default=str` already handles most non-serializable cases (OmegaConf
        # ListConfig/DictConfig coerce via str()); this catch is a final
        # backstop so a bad record never escapes a training step. Anything
        # the encoder + str() fallback can't handle is a logger problem,
        # not a training-time error.
        logger.warning(f"perf_log: skipping unserializable record ({e})")
        return
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning(f"perf_log: write failed for {path}: {e}")


def _unwrap(model: Any) -> Any:
    """Strip up to 4 layers of DDP / FSDP / DeepSpeed `.module` wrapping.

    Shared by helpers that need the underlying user model (UnifiedTransformer)
    rather than the distributed wrapper. Bounded depth so a malformed model
    can't spin the loop forever; DDP + DeepSpeed nest at most ~2 levels.
    """
    inner = model
    for _ in range(4):
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    return inner


def model_modalities(model: Any) -> list[str] | None:
    """Best-effort grab of model.config.modalities, unwrapping DDP/FSDP/DS `.module`.

    Main's inline `(model.module if hasattr(model, 'module') else model).config.modalities`
    crashes when `config` lacks `.modalities` (e.g. trainer passes a TrainingConfig
    where a ModelConfig was expected) and doesn't unwrap multi-level wrappers
    (DDP + DeepSpeed nest 2 deep). This helper handles both cases — returns None
    rather than raising so a missing config doesn't kill the training job.
    """
    inner = model
    for _ in range(4):  # bounded unwrap; DDP/FSDP rarely nest more than 2 deep
        cfg = getattr(inner, "config", None)
        if cfg is not None and hasattr(cfg, "modalities"):
            # `list(omegaconf.ListConfig)` returns a ListConfig — not JSON-serializable.
            # Coerce each item via str() so OmegaConf scalars and Modality enums
            # alike land as plain Python strings.
            return [str(m) for m in cfg.modalities]
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            break
    return None


# Sourced from the central Modality enum; text is excluded because it is
# accounted for by `batch_token_count` instead. Any future modality added to
# the enum is picked up automatically.
_KNOWN_MODALITY_KEYS = frozenset(str(m) for m in Modality if m is not Modality.TEXT)


def _looks_non_degenerate(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    if isinstance(value, dict):
        return any(_looks_non_degenerate(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return value is not None


def _graph_batch_size(value: dict) -> int | None:
    """Return the number of graphs in a PyG-style collated dict.

    PyG concatenates node features across graphs, so `value['x'].shape[0]`
    is the total node count, not the graph count. The graph count lives in
    `ptr` (len-N+1 offsets) or `batch` (per-node graph index).
    """
    ptr = value.get("ptr")
    if isinstance(ptr, torch.Tensor) and ptr.numel() >= 2:
        return int(ptr.numel() - 1)
    batch_idx = value.get("batch")
    if isinstance(batch_idx, torch.Tensor) and batch_idx.numel() > 0:
        return int(batch_idx.max().item()) + 1
    return None


def batch_modality_counts(batch: dict) -> dict[str, int]:
    """Per-modality sample count for one collated batch.

    Returns `{modality: n_samples}` for each known modality key whose value
    looks non-degenerate. Tensors use `shape[0]`; PyG-style graph dicts use
    `ptr` / `batch` (concatenated node tensors would otherwise report node
    counts as sample counts).
    """
    out: dict[str, int] = {}
    if not isinstance(batch, dict):
        return out
    for key in _KNOWN_MODALITY_KEYS:
        if key not in batch:
            continue
        value = batch[key]
        if not _looks_non_degenerate(value):
            continue
        n: int | None = None
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            n = int(value.shape[0])
        elif isinstance(value, dict):
            n = _graph_batch_size(value)
            if n is None:
                # Non-PyG dict (rare); leading tensor axis is best we can do.
                for v in value.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        n = int(v.shape[0])
                        break
        if n is None:
            try:
                n = len(value)
            except TypeError:
                n = 1
        out[key] = n
    return out


def batch_token_count(batch: dict, pad_id: int | None = None) -> int:
    """Count non-pad text tokens in a collated batch.

    `batch["text"]` is the padded (B, T) tokenized tensor produced by
    `MultimodalCollator`. When `pad_id` is None, falls back to total
    elements (useful when the trainer can't easily reach the tokenizer).
    Returns 0 when there's no text tensor.
    """
    if not isinstance(batch, dict):
        return 0
    text = batch.get("text")
    if not isinstance(text, torch.Tensor) or text.numel() == 0:
        return 0
    if pad_id is None:
        return int(text.numel())
    return int((text != pad_id).sum().item())


def count_parameters(model: Any) -> dict[str, int]:
    """Per-component parameter counts for IsoFLOP scaling-laws bookkeeping.

    Buckets parameters under the UnifiedTransformer's standard submodule
    layout:
      - `backbone`            : `model.backbone` (HF LLM)
      - `encoder_<modality>`  : entries in `model.encoders` (nn.ModuleDict)
      - `projector_<modality>`: entries in `model.projectors` (nn.ModuleDict)
      - `total`               : every parameter on the unwrapped model
      - `train`               : parameters with `requires_grad=True`
      - `active`              : same as `total` for PR-1 (no LoRA / MoE
                                conditional execution today); revisit when
                                a conditional-compute path lands.

    Caller is responsible for unwrapping DDP/FSDP/DeepSpeed wrappers — for
    consistency this helper also calls `_unwrap` defensively. All values
    are plain Python ints (no `torch.Size` / numpy types) so the dict
    round-trips through `json.dumps`.
    """
    unwrapped = _unwrap(model)
    out: dict[str, int] = {}

    def _sum(module: Any) -> int:
        if module is None:
            return 0
        return sum(int(p.numel()) for p in module.parameters())

    backbone = getattr(unwrapped, "backbone", None)
    if backbone is not None:
        out["backbone"] = _sum(backbone)

    encoders = getattr(unwrapped, "encoders", None)
    if encoders is not None:
        for name, submod in encoders.items():
            out[f"encoder_{name}"] = _sum(submod)

    projectors = getattr(unwrapped, "projectors", None)
    if projectors is not None:
        for name, submod in projectors.items():
            out[f"projector_{name}"] = _sum(submod)

    total = sum(int(p.numel()) for p in unwrapped.parameters())
    train = sum(int(p.numel()) for p in unwrapped.parameters() if p.requires_grad)
    out["total"] = total
    out["train"] = train
    # `active` is a placeholder until PRISM grows a conditional-execution
    # path (LoRA, MoE expert routing, …) — at that point `active < total`
    # for sparse models. IsoFLOP analytic FLOPs already use `N_active`.
    out["active"] = total
    return out


def sequence_stats(seq_lens: list[int]) -> dict[str, float]:
    """Per-window sequence-length distribution + padding fraction.

    `seq_lens` is the per-sample non-pad text-token count collected across
    one throughput window (typically 50 steps). Returns:
      seq_p50, seq_p95, seq_p99, seq_max : percentile / max length in tokens
      padding_ratio : (max(L) * N - sum(L)) / (max(L) * N)
                      — what fraction of the right-padded tensor is filler.
                      0.0 when the window is empty.

    Pure function — trivially unit-testable, no torch / DDP dependency.
    """
    if not seq_lens:
        return {
            "seq_p50": 0.0,
            "seq_p95": 0.0,
            "seq_p99": 0.0,
            "seq_max": 0.0,
            "padding_ratio": 0.0,
        }
    import numpy as np  # local import — perf_log is imported by every trainer

    arr = np.asarray(seq_lens, dtype=np.float64)
    seq_max = float(arr.max())
    n = int(arr.size)
    denom = seq_max * n
    padding_ratio = float((denom - arr.sum()) / denom) if denom > 0 else 0.0
    return {
        "seq_p50": float(np.percentile(arr, 50)),
        "seq_p95": float(np.percentile(arr, 95)),
        "seq_p99": float(np.percentile(arr, 99)),
        "seq_max": seq_max,
        "padding_ratio": padding_ratio,
    }


class _FlopCounter:
    """Apply per-step FLOPs from a calibration JSON.

    The IsoFLOP study calibrates `(family, backbone, projector_variant)`
    offline (see `tools/isoflop_calibrate.py`) and writes a small JSON
    with `flops_per_step` and `flops_per_sample`. The trainer constructs
    one of these at startup (when `CALIBRATION_JSON` env var is set) and
    folds the running total into every perf record so the collector can
    flip a row to `done` without re-running the calibration.

    When no calibration is available, both `flops_per_step` and
    `cumulative_flops` come out as `None` so the collector can treat them
    as missing rather than zero.
    """

    def __init__(self, flops_per_step: float | None = None) -> None:
        self.flops_per_step = flops_per_step
        self.cumulative_flops: float = 0.0

    @classmethod
    def from_calibration(cls, path: str | os.PathLike | None) -> _FlopCounter:
        """Build from a calibration JSON, optionally overridden by env var.

        Reads `flops_per_step` from the JSON at `path`. If the
        `RUNTIME_FLOPS_PER_STEP` env var is set to a positive float, it
        takes precedence — this is the channel `tools/isoflop_launch.py`
        uses to inject the plan's *rescaled* FPS (cal config → runtime
        config), so trainer-side `cumulative_flops` matches the plan's
        `budget_flops` instead of undercounting by `rescale_factor`. When
        neither is available, returns an uncalibrated counter.
        """
        fps: float | None = None
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                fps = float(data.get("flops_per_step", 0.0)) or None
            except (OSError, ValueError, TypeError) as e:
                logger.warning(f"perf_log: cannot read calibration {path}: {e}")
        env_fps = os.environ.get("RUNTIME_FLOPS_PER_STEP")
        if env_fps:
            try:
                _override = float(env_fps)
                if _override > 0:
                    if fps is not None and abs(_override - fps) / fps > 0.01:
                        logger.info(
                            f"perf_log: RUNTIME_FLOPS_PER_STEP={_override:.3e} "
                            f"overrides calibration flops_per_step={fps:.3e} "
                            f"(rescale factor ~{_override / fps:.2f}x)"
                        )
                    fps = _override
            except ValueError:
                logger.warning(
                    f"perf_log: RUNTIME_FLOPS_PER_STEP={env_fps!r} not numeric; ignoring"
                )
        return cls(flops_per_step=fps)

    def step(self, n_steps: int = 1) -> tuple[float | None, float | None]:
        """Advance `n_steps` optimizer steps. Returns `(per_step, cumulative)`.

        `n_steps` exists because trainers call `.step()` once per perf-record
        FLUSH (typically every 50 training steps), not once per training step
        — without the multiplier `cumulative_flops` undercounts by the flush
        interval (~50x). Trainers MUST pass the actual training-step count
        consumed since the previous `.step()` call. Default 1 is a back-compat
        hatch for unit tests and callers that genuinely advance one step.

        When uncalibrated, returns `(None, None)` so the perf record contains
        explicit nulls (preferable to a misleading zero).
        """
        if self.flops_per_step is None:
            return None, None
        self.cumulative_flops += self.flops_per_step * float(n_steps)
        return self.flops_per_step, self.cumulative_flops
