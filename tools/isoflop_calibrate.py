#!/usr/bin/env python3
"""Calibrate per-step FLOPs and throughput for one IsoFLOP cell.

Self-contained (no Hydra). For a given `(backbone, projector_variant,
family, regime)`, instantiates `UnifiedTransformer`, runs a small number
of synthetic-batch training steps, and writes a JSON with:
    flops_per_step, flops_per_sample, samples_per_sec,
    n_total_params, n_active_params, n_trainable_params,
    mean_seq_len, p95_seq_len, max_mem_mb,
    backbone_id, projector_variant, projector_hidden_mult, projector_num_layers,
    family, regime, steps, batch_size, seq_len, calibrated_at.

Downstream: `tools/isoflop_plan.py` reads one of these per `(backbone,
variant, family)` to convert a budget in FLOPs into `max_steps` for the
launcher's `--target-flops` knob (PR-2).

Usage:
    python tools/isoflop_calibrate.py \\
        --backbone allenai/OLMo-2-1B-1124-hf \\
        --projector-variant BASE \\
        --family text_image \\
        --regime projector_only \\
        --steps 30 \\
        --output scaling-study/calibration/OLMO3-1B-BASE-text_image-projector_only.json

Idempotent: re-running without --force exits 0 with "already calibrated".
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Path safety — allow `python tools/isoflop_calibrate.py` from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig  # noqa: E402
from src.modalities import Modality  # noqa: E402
from src.modules.projector import ModalityProjector  # noqa: E402
from tools.benchmark_throughput import get_device, get_memory_stats, sync_device  # noqa: E402

logger = logging.getLogger("isoflop_calibrate")


# Map from `--family` flag to (Modality list, encoder-input-dim default).
# Kept narrow on purpose — calibration only needs the families IsoFLOP
# Stage A exercises (text + one non-text modality). Add new families here
# as the scaling-study expands.
_FAMILY_TO_MODALITIES: dict[str, list[Modality]] = {
    "text_image": [Modality.TEXT, Modality.IMAGE],
    "text_ts": [Modality.TEXT, Modality.TIME_SERIES],
    "text_graph": [Modality.TEXT, Modality.GRAPH],
}

# Regimes:
#   projector_only      — freeze everything except `model.projectors`
#   encoder_projector   — also unfreeze `model.encoders`
#   e2e                 — unfreeze the backbone too
_VALID_REGIMES = ("projector_only", "encoder_projector", "e2e")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, help="HF model id for the LLM backbone")
    p.add_argument(
        "--projector-variant",
        required=True,
        choices=list(ModalityProjector.VARIANT_MAP.keys()),
        help="Projector capacity variant (BASE/W2X/W4X/D2X/D4X)",
    )
    p.add_argument(
        "--family",
        required=True,
        choices=list(_FAMILY_TO_MODALITIES.keys()),
        help="Modality family (text_image, text_ts, text_graph)",
    )
    p.add_argument(
        "--regime",
        default="projector_only",
        choices=list(_VALID_REGIMES),
        help="What to leave trainable",
    )
    p.add_argument("--steps", type=int, default=30, help="Timed steps (after warmup)")
    p.add_argument("--warmup-steps", type=int, default=5, help="Untimed warmup steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--output", required=True, help="JSON path to write")
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _make_synthetic_batch(
    family: str,
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict:
    """Build a single synthetic batch keyed by modality.

    Mirrors the keys that `UnifiedTransformer.forward` expects:
      - "text"        : (B, T) integer token tensor
      - "image"       : (B, 3, 224, 224) float tensor (raw pixels)
      - "time_series" : (B, L, V) float tensor
      - "graph"       : PyG-style dict { x, edge_index, batch, ptr }
    Values are random — calibration only needs shape and dtype to be
    consistent across steps; actual loss curves are irrelevant.
    """
    vocab = max(int(model_config.vocab_size), 16)
    batch: dict = {
        "text": torch.randint(0, vocab, (batch_size, seq_len), dtype=torch.long, device=device),
    }
    if family == "text_image":
        batch["image"] = torch.randn(batch_size, 3, 224, 224, device=device)
    elif family == "text_ts":
        ts_len = min(int(model_config.max_ts_length), 256)
        batch["time_series"] = torch.randn(
            batch_size, ts_len, int(model_config.ts_variates), device=device
        )
    elif family == "text_graph":
        # Minimal PyG-style dict: one tiny graph per sample, concatenated.
        nodes_per_graph = 8
        edges_per_graph = 12
        total_nodes = batch_size * nodes_per_graph
        x = torch.randn(total_nodes, 32, device=device)
        edge_index = torch.randint(
            0, total_nodes, (2, batch_size * edges_per_graph), dtype=torch.long, device=device
        )
        ptr = torch.arange(
            0, total_nodes + 1, nodes_per_graph, dtype=torch.long, device=device
        )
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(nodes_per_graph)
        batch["graph"] = {"x": x, "edge_index": edge_index, "ptr": ptr, "batch": batch_idx}
    return batch


def _freeze_for_regime(model, regime: str) -> None:
    """Apply the requested freezing policy in-place on `model`."""
    for p in model.parameters():
        p.requires_grad = False
    # Always unfreeze projectors — they are what the projector_only regime
    # measures and what the higher regimes ALSO update.
    if hasattr(model, "projectors"):
        for p in model.projectors.parameters():
            p.requires_grad = True
    if regime in ("encoder_projector", "e2e") and hasattr(model, "encoders"):
        for p in model.encoders.parameters():
            p.requires_grad = True
    if regime == "e2e" and getattr(model, "backbone", None) is not None:
        for p in model.backbone.parameters():
            p.requires_grad = True


def _backbone_flops_per_step(
    model,
    batch_size: int,
    seq_len: int,
) -> float:
    """Analytic backbone FLOPs (forward + backward) per step.

    Per PaLM/nanochat convention:
        per-token = 6 * (N_active_backbone - N_embed)
                  + 12 * n_layers * n_heads * head_dim * seq_len
        per-step  = per-token * batch_size * seq_len   (×1 if attention term
                    already folds in seq_len — see below for the breakdown)
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0.0
    n_total = sum(p.numel() for p in backbone.parameters())
    # Subtract embedding params — they participate in lookups, not matmuls,
    # so the 6×N rule overstates their FLOP cost.
    n_embed = 0
    for module in backbone.modules():
        if isinstance(module, torch.nn.Embedding):
            n_embed += sum(p.numel() for p in module.parameters())
    cfg = getattr(backbone, "config", None)
    n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0) or 0))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0) or 0))
    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0) or 0))
    head_dim = (hidden // n_heads) if n_heads > 0 else 0

    flops_per_token_matmul = 6 * (n_total - n_embed)
    flops_per_token_attn = 12 * n_layers * n_heads * head_dim * seq_len
    return float((flops_per_token_matmul + flops_per_token_attn) * batch_size * seq_len)


def _projector_flops_per_step(
    model,
    batch_size: int,
    seq_len: int,
) -> float:
    """Analytic projector FLOPs (forward + backward) per step.

    For each linear, fwd = 2*in*out per token, ×3 for fwd+bwd. We use
    `seq_len` as a stand-in for tokens-per-modality — slight overcount
    when modality tokens < text tokens, but conservative for budgeting.
    """
    proj = getattr(model, "projectors", None)
    if proj is None:
        return 0.0
    total = 0.0
    for _name, sub in proj.items():
        for module in sub.modules():
            if isinstance(module, torch.nn.Linear):
                in_dim = int(module.in_features)
                out_dim = int(module.out_features)
                total += 3.0 * 2.0 * in_dim * out_dim * seq_len * batch_size
    return float(total)


def _instantiate_model(args: argparse.Namespace) -> tuple[object, ModelConfig]:
    hidden_mult, num_layers = ModalityProjector.VARIANT_MAP[args.projector_variant]
    modalities = _FAMILY_TO_MODALITIES[args.family]
    config = ModelConfig(
        llm_backbone_id=args.backbone,
        modalities=list(modalities),
        freeze_backbone=True,
        freeze_encoders=True,
        projector_hidden_mult=hidden_mult,
        projector_num_layers=num_layers,
    )
    # Local import — avoids dragging the big import chain when `--help` runs.
    from src.model import UnifiedTransformer

    logger.info(
        "Instantiating UnifiedTransformer: backbone=%s variant=%s (hm=%d, nl=%d) family=%s",
        args.backbone,
        args.projector_variant,
        hidden_mult,
        num_layers,
        args.family,
    )
    model = UnifiedTransformer(config)
    return model, config


def calibrate(args: argparse.Namespace) -> dict:
    device = get_device()
    logger.info("Device: %s", device)

    model, model_config = _instantiate_model(args)
    model = model.to(device)
    _freeze_for_regime(model, args.regime)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    # Per-component param counts.
    n_total = sum(int(p.numel()) for p in model.parameters())
    n_train = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    n_active = n_total  # PR-1 placeholder (no LoRA/MoE conditional execution).

    seq_lens: list[int] = []
    step_times: list[float] = []

    for step in range(args.warmup_steps + args.steps):
        batch = _make_synthetic_batch(
            args.family, model_config, args.batch_size, args.seq_len, device
        )
        seq_lens.append(int(args.seq_len))
        sync_device(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits, loss = model(batch, labels=batch["text"])
        loss.backward()
        optimizer.step()
        sync_device(device)
        dt = time.perf_counter() - t0
        if step >= args.warmup_steps:
            step_times.append(dt)

    mean_step_s = sum(step_times) / max(1, len(step_times))
    samples_per_sec = float(args.batch_size / mean_step_s) if mean_step_s > 0 else 0.0
    alloc_mb, _reserved_mb = get_memory_stats(device)

    # Analytic FLOPs.
    backbone_fps = _backbone_flops_per_step(model, args.batch_size, args.seq_len)
    projector_fps = _projector_flops_per_step(model, args.batch_size, args.seq_len)
    # Encoder FLOPs deliberately omitted in PR-1: projector_only freezes
    # the encoder (cost amortized via cache), and higher regimes are
    # tracked as a TODO (RISKS §2 in the IsoFLOP plan). Collector treats
    # None as "missing", not zero — fits won't bias against this row.
    encoder_fps: float | None = None
    flops_per_step = backbone_fps + projector_fps
    flops_per_sample = flops_per_step / max(1, args.batch_size)

    import numpy as np

    mean_seq_len = float(np.mean(seq_lens)) if seq_lens else 0.0
    p95_seq_len = float(np.percentile(seq_lens, 95)) if seq_lens else 0.0

    return {
        "flops_per_step": flops_per_step,
        "flops_per_sample": flops_per_sample,
        "flops_per_step_backbone": backbone_fps,
        "flops_per_step_projector": projector_fps,
        "flops_per_step_encoder": encoder_fps,
        "samples_per_sec": samples_per_sec,
        "mean_step_s": mean_step_s,
        "n_total_params": n_total,
        "n_active_params": n_active,
        "n_trainable_params": n_train,
        "mean_seq_len": mean_seq_len,
        "p95_seq_len": p95_seq_len,
        "max_mem_mb": alloc_mb,
        "backbone_id": args.backbone,
        "projector_variant": args.projector_variant,
        "projector_hidden_mult": ModalityProjector.VARIANT_MAP[args.projector_variant][0],
        "projector_num_layers": ModalityProjector.VARIANT_MAP[args.projector_variant][1],
        "family": args.family,
        "regime": args.regime,
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        # PR-4: stamp n_ranks so isoflop_plan can rescale flops_per_step to
        # the runtime config. Calibrator is single-process today, so always 1.
        # Plan computes runtime_fps = cal_fps * (rbs/cbs * rsl/csl * rranks/cranks)
        # and uses that to convert target_flops -> max_steps.
        "n_ranks": 1,
        "calibrated_at": _dt.datetime.utcnow().isoformat() + "Z",
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[isoflop_calibrate] already calibrated: {out} (use --force to overwrite)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    record = calibrate(args)
    with open(out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"[isoflop_calibrate] wrote {out}")
    print(f"  flops_per_step={record['flops_per_step']:.3e}  samples_per_sec={record['samples_per_sec']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
