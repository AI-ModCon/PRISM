"""Gate-0 micro-benchmark for the native XPU fused flash-attention kernel.

Confirms whether ``sdpa_kernel([FLASH_ATTENTION])`` actually engages on Aurora
PVC at the shapes PRISM's HF backbone uses, and measures memory + step-time
delta vs. the default (math) backend and eager-manual attention.

Run on a single tile of a held node:

    ZE_AFFINITY_MASK=0 python tools/bench_xpu_flash_sdpa.py

Prints one row per (backend, shape) combination:
  - engaged: did the requested backend actually run (WARN_FOR_UNFUSED_KERNELS)?
  - peak_gib: torch.xpu.max_memory_allocated after fwd+bwd
  - fwd_ms / bwd_ms: mean over N iters (post-warmup)
  - max_abs_err_vs_math: numeric parity of fwd output vs. the math ref
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class Shape:
    label: str
    B: int
    H: int
    S: int
    D: int


# Shapes that match PRISM's Qwen3-0.6B HSDP runs (H=16, D=128) at the seq
# lengths we routinely hit, plus a stress row to see the math cliff.
DEFAULT_SHAPES = [
    # Qwen3-0.6B: H=16, D=64 (GQA kv=8, but we bench post-GQA-expansion so H=Q-heads).
    Shape("qwen3-06b_seq512",   B=1, H=16, S=512,   D=64),
    Shape("qwen3-06b_seq2048",  B=1, H=16, S=2048,  D=64),
    Shape("qwen3-06b_seq4096",  B=1, H=16, S=4096,  D=64),
    Shape("qwen3-06b_seq8192",  B=1, H=16, S=8192,  D=64),
    # SigLIP2-base-patch16-224 vision block: H=12, D=64, seq=196 (14x14 patches).
    # Bidirectional (is_causal=False).
    Shape("siglip2_base_s196",  B=1, H=12, S=196,   D=64),
    # Batch scaling (Qwen3-0.6B at PRISM's typical bs=8 microbatch).
    Shape("qwen3-06b_seq2048_bs8",  B=8, H=16, S=2048,  D=64),
    Shape("qwen3-06b_seq4096_bs4",  B=4, H=16, S=4096,  D=64),
]


def _mk_bshd(B: int, S: int, H: int, D: int, dtype, device, requires_grad: bool):
    """Allocate q/k/v in BSHD memory (contiguous [B,S,H,D], transposed 1<->2)."""
    x = torch.randn(B, S, H, D, dtype=dtype, device=device, requires_grad=requires_grad)
    return x.transpose(1, 2)


def _sync(device: str) -> None:
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_gib(device: str) -> float:
    if device == "xpu":
        return torch.xpu.max_memory_allocated() / 1024**3
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def run_backend(backend: str, shape: Shape, dtype, device: str, iters: int,
                warmup: int, is_causal: bool, seed: int = 0):
    """Run one backend+shape and return dict of measurements."""

    B, H, S, D = shape.B, shape.H, shape.S, shape.D

    # WARN_FOR_UNFUSED_KERNELS prints a warning when a fused backend is
    # requested but rejected (giving the exact reason). We capture warnings
    # instead of just printing so we can report `engaged` cleanly.
    import torch.nn.attention as tna
    tna.WARN_FOR_UNFUSED_KERNELS = True

    def one_step(measure_fwd: bool, measure_bwd: bool, seed_offset: int = 0):
        # Seed per-call so every backend sees IDENTICAL q/k/v tensors on the
        # equivalent iteration — parity comparisons vs. math are meaningless
        # otherwise.
        torch.manual_seed(seed + seed_offset)
        q = _mk_bshd(B, S, H, D, dtype, device, requires_grad=True)
        k = _mk_bshd(B, S, H, D, dtype, device, requires_grad=True)
        v = _mk_bshd(B, S, H, D, dtype, device, requires_grad=True)

        _sync(device)
        t0 = time.perf_counter()

        if backend == "flash":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=is_causal
                )
        elif backend == "math":
            with sdpa_kernel([SDPBackend.MATH]):
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=is_causal
                )
        elif backend == "default":
            # Whatever XPU auto-dispatch picks (expected: math)
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=is_causal
            )
        elif backend == "eager":
            # Manual matmul+softmax reference (fp32 math, bf16 io)
            scale = 1.0 / (D ** 0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
                )
                attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn.float(), dim=-1).to(dtype)
            o = attn @ v
        else:
            raise ValueError(f"unknown backend: {backend}")

        _sync(device)
        fwd_t = time.perf_counter() - t0

        bwd_t = 0.0
        if measure_bwd:
            _sync(device)
            t1 = time.perf_counter()
            loss = o.float().pow(2).mean()
            loss.backward()
            _sync(device)
            bwd_t = time.perf_counter() - t1

        return o.detach(), fwd_t, bwd_t

    # Warmup (use negative offsets so measured iters see a fixed seed sequence)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        engaged = True
        warn_reason = ""
        try:
            for i in range(warmup):
                one_step(True, True, seed_offset=-(i + 1))
        except RuntimeError as e:
            # e.g. "No available kernel" — flash was requested and rejected
            engaged = False
            warn_reason = str(e).splitlines()[0]
            return {
                "backend": backend, "shape": shape.label, "engaged": False,
                "reason": warn_reason, "peak_gib": 0.0,
                "fwd_ms": 0.0, "bwd_ms": 0.0, "err": None,
            }

    # Measured runs — deterministic seed per iteration so backends line up.
    _reset_peak(device)
    fwd_times, bwd_times = [], []
    last_out = None
    for i in range(iters):
        o, ft, bt = one_step(True, True, seed_offset=i)
        fwd_times.append(ft)
        bwd_times.append(bt)
        last_out = o
    _sync(device)

    return {
        "backend": backend,
        "shape": shape.label,
        "engaged": engaged,
        "reason": warn_reason,
        "peak_gib": _peak_gib(device),
        "fwd_ms": 1000.0 * sum(fwd_times) / len(fwd_times),
        "bwd_ms": 1000.0 * sum(bwd_times) / len(bwd_times),
        "out": last_out,  # kept for parity comparison; stripped before print
    }


def parity_vs_math(row, ref_out):
    """max|Δ| between this backend's fwd output and the math reference."""
    if row.get("out") is None or ref_out is None:
        return float("nan")
    with torch.no_grad():
        return (row["out"].float() - ref_out.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false",
                    help="Bidirectional attention (e.g. SigLIP2 vision blocks).")
    ap.add_argument("--shapes", nargs="*", default=None,
                    help="Optional shape labels to restrict (e.g. qwen3-06b_seq512)")
    ap.add_argument(
        "--backends", nargs="+",
        default=["default", "math", "flash", "eager"],
        help="Which backends to test",
    )
    args = ap.parse_args()

    device = args.device
    if device == "xpu" and not torch.xpu.is_available():
        print("ERROR: torch.xpu.is_available() is False; are you on a compute node with XPU?",
              file=sys.stderr)
        return 2

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = DEFAULT_SHAPES
    if args.shapes:
        want = set(args.shapes)
        shapes = [s for s in shapes if s.label in want]
        if not shapes:
            print(f"ERROR: no shapes match {args.shapes}", file=sys.stderr)
            return 2

    print("# Aurora XPU flash-attention Gate 0 micro-bench")
    print(f"# torch={torch.__version__}  device={device}  dtype={args.dtype}"
          f"  iters={args.iters}  warmup={args.warmup}  causal={args.causal}")
    print(f"# ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")
    print(f"# {'shape':<26} {'backend':<9} {'engaged':<7} {'peak GiB':>9} "
          f"{'fwd ms':>9} {'bwd ms':>9} {'err vs math':>12}   note")

    for shape in shapes:
        rows = []
        for backend in args.backends:
            try:
                row = run_backend(
                    backend, shape, dtype, device,
                    iters=args.iters, warmup=args.warmup, is_causal=args.causal,
                )
                rows.append(row)
            except Exception as e:  # noqa: BLE001 — this is a diagnostic tool
                rows.append({
                    "backend": backend, "shape": shape.label,
                    "engaged": False, "reason": f"exception: {e!r}",
                    "peak_gib": 0.0, "fwd_ms": 0.0, "bwd_ms": 0.0,
                    "out": None,
                })

        # Compute parity vs. the math row (if present)
        math_row = next((r for r in rows if r["backend"] == "math" and r.get("out") is not None), None)
        ref_out = math_row["out"] if math_row is not None else None

        for r in rows:
            err = parity_vs_math(r, ref_out)
            note = r.get("reason", "") or ""
            # Truncate long "No available kernel" messages for the table
            if len(note) > 60:
                note = note[:57] + "..."
            print(
                f"  {r['shape']:<26} {r['backend']:<9} "
                f"{'yes' if r['engaged'] else 'NO':<7} "
                f"{r['peak_gib']:>9.3f} "
                f"{r['fwd_ms']:>9.2f} "
                f"{r['bwd_ms']:>9.2f} "
                f"{err:>12.4g}   {note}"
            )
            # Drop the output tensor from the row before it goes out of scope
            # to keep peak-mem readings honest across shapes.
            r["out"] = None

        _sync(device)
        _reset_peak(device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
