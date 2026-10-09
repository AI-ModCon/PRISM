"""Time-series eval via vLLM vs HF baseline.

Builds N requests by tiling synthetic time-series tensors, runs them
through (a) vLLM's batched engine and (b) HF UnifiedTransformer.generate,
reports wall-clock + tokens/sec for each path and the speedup ratio.

Plan §VLLM-6 bar: ≥5× speedup over HF generate. With synthetic random-
weight checkpoints the OUTPUT TOKENS are meaningless — only the
infrastructure throughput is comparable. When a real ts training
checkpoint lands, swap `--vllm-model` and `--demo-checkpoint` for it and
re-run for the real-bar number.

Usage:
    bash tools/_vllm_eval_ts_runner.sh \\
        --vllm-model exported/<synthetic-or-real-ts> \\
        --demo-checkpoint <training-checkpoint-dir-with-encoders.time_series.*> \\
        --n 50 --max-tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    _PROJECT_ROOT if not _pp else f"{_PROJECT_ROOT}{os.pathsep}{_pp}"
)

import src.vllm_plugin  # noqa: E402

src.vllm_plugin.register()


def _build_requests(
    *, n: int, max_ts_length: int, num_vars: int, prompt: str, seed: int
):
    import torch

    # Same per-request tensor across all N — keeps the comparison apples-
    # to-apples (no random hot-cache effects). Generate once, tile N times.
    torch.manual_seed(seed)
    ts = torch.randn(max_ts_length, num_vars)
    requests = [
        {
            "prompt": prompt,
            "multi_modal_data": {"time_series": ts.clone()},
        }
        for _ in range(n)
    ]
    return requests, ts


def _run_vllm(
    *,
    vllm_model: str,
    requests: list,
    max_tokens: int,
    dtype: str,
) -> tuple[float, int, list[str]]:
    from vllm import LLM, SamplingParams

    print(f"[vllm] booting LLM(model={vllm_model!r}, dtype={dtype})")
    llm = LLM(
        model=vllm_model,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"time_series": 1, "image": 0},
        # Memory budget for TS-only on a single tile. Plan §VLLM-6
        # flagged this — start conservative.
        max_num_batched_tokens=2048,
    )
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Warmup pass so the encoder JIT / kernels are hot.
    llm.generate(requests[:1], sampling)

    t0 = time.time()
    outs = llm.generate(requests, sampling)
    dt = time.time() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    texts = [o.outputs[0].text for o in outs]
    return dt, total_out, texts


def _run_hf_demo(
    *,
    demo_checkpoint: str,
    requests: list,
    max_tokens: int,
    max_ts_length: int,
    num_vars: int,
    dtype_name: str,
) -> tuple[float, int, list[str]]:
    """HF baseline via UnifiedTransformer.generate.

    The HF path is sequential (no batching equivalent to vLLM's PagedAttention),
    so we time each request and sum. Mirrors tools/vllm_parity.py::run_demo_path
    but for ts instead of image, and reused across N requests instead of just one.
    """
    import torch
    from safetensors.torch import load_file
    from src.config import ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

    backbone_id = "allenai/OLMo-1B-0724-hf"

    ckpt_file = os.path.join(demo_checkpoint, "model.safetensors")
    sd = load_file(ckpt_file, device="cpu")
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    d_model = sd["backbone.model.embed_tokens.weight"].shape[1]
    # For the "linear" encoder, encoders.time_series.model is nn.Linear(num_vars, d_ts);
    # weight is (d_ts, num_vars), so shape[0] is d_ts. Other encoder types (moirai)
    # don't expose this key and we fall back to d_model below.
    d_ts = int(sd.get("encoders.time_series.model.weight", torch.zeros(d_model)).shape[0])
    if d_ts == 0:
        d_ts = d_model
    print(f"[hf] backbone={backbone_id} d_model={d_model} d_ts={d_ts}")

    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-warm cache so UnifiedTransformer's local_files_only=True can find it.
    from transformers import AutoModelForCausalLM as _AMC

    _ = _AMC.from_pretrained(backbone_id, trust_remote_code=True)
    del _

    cfg = ModelConfig(
        llm_backbone_id=backbone_id,
        freeze_backbone=True,
        freeze_encoders=True,
        d_model=d_model,
        d_text=d_model,
        d_ts=d_ts,
        modalities=["text", "time_series"],
        is_timeseries=True,
        ts_variates=num_vars,
        max_ts_length=max_ts_length,
        ts_projector="linear",
        normalize_ts_in_encoder=False,
    )
    model = UnifiedTransformer(cfg)
    if model.backbone is None:
        raise RuntimeError("UnifiedTransformer failed to load HF backbone")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    ts_missing = [k for k in missing if "time_series" in k]
    ts_unexpected = [k for k in unexpected if "time_series" in k]
    print(
        f"[hf] state_dict missing={len(missing)} unexpected={len(unexpected)} "
        f"(ts missing={len(ts_missing)} unexpected={len(ts_unexpected)})"
    )

    model.to(device=device, dtype=dtype)
    model.eval()
    model.tokenizer = tokenizer

    texts: list[str] = []
    total_out = 0

    # UnifiedTransformer (non-interleaved path) splices the ts embedding into
    # the prefix BEFORE the text — the `<time_series>` placeholder in the
    # prompt is not expanded by HF demo. Strip it so the HF text prompt is
    # clean. vLLM expands it via PromptUpdateDetails (see VLLM-5). This is
    # the only structural asymmetry between the two paths; per-request
    # `max_tokens` budget makes the throughput comparison fair.
    def _hf_prompt(p: str) -> str:
        for tok in ("<time_series>", "<ts_start>", "<ts_end>"):
            p = p.replace(tok, "")
        return p.strip() or " "

    # Warmup — single forward + generate so the kernels are hot.
    warm_prompt = _hf_prompt(requests[0]["prompt"])
    input_ids = tokenizer(warm_prompt, return_tensors="pt").input_ids.to(device)
    ts0 = requests[0]["multi_modal_data"]["time_series"].unsqueeze(0).to(device, dtype=dtype)
    inputs0 = {"text": input_ids, "time_series": ts0}
    with torch.no_grad():
        _ = model.generate(inputs0, max_new_tokens=max_tokens, do_sample=False)

    t0 = time.time()
    for req in requests:
        hf_prompt = _hf_prompt(req["prompt"])
        input_ids = tokenizer(hf_prompt, return_tensors="pt").input_ids.to(device)
        ts_t = req["multi_modal_data"]["time_series"].unsqueeze(0).to(device, dtype=dtype)
        inputs = {"text": input_ids, "time_series": ts_t}
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=max_tokens, do_sample=False)
        if hasattr(out, "sequences"):
            out = out.sequences[0]
        elif isinstance(out, torch.Tensor):
            out = out[0]
        else:
            out = out[0]
        # Count NEW tokens only (skip the prompt prefix).
        new_tok = max(0, out.shape[0] - input_ids.shape[1])
        total_out += int(new_tok)
        texts.append(tokenizer.decode(out, skip_special_tokens=True))
    dt = time.time() - t0
    return dt, total_out, texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vllm-model",
        required=True,
        help="Exported PRISM dir with active_modalities including time_series.",
    )
    p.add_argument(
        "--demo-checkpoint",
        default=None,
        help="Training checkpoint dir (has encoders.time_series.* keys) for "
        "the HF baseline. Optional — skip to bench vLLM-only.",
    )
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-ts-length", type=int, default=32,
                   help="ts T-axis length; must be <= the export's max_ts_length.")
    p.add_argument("--num-vars", type=int, default=1)
    p.add_argument(
        "--prompt",
        default="<time_series>forecast:",
        help="Prompt for both paths. Must contain <time_series> placeholder.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--mode",
        default="both",
        choices=["both", "vllm", "hf"],
        help="Which path(s) to run. `both` reports the speedup ratio.",
    )
    p.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="If >0, print the first N generated outputs from each path.",
    )
    args = p.parse_args()

    requests, sample_ts = _build_requests(
        n=args.n,
        max_ts_length=args.max_ts_length,
        num_vars=args.num_vars,
        prompt=args.prompt,
        seed=args.seed,
    )
    print(
        f"[eval] n={args.n} max_tokens={args.max_tokens} prompt={args.prompt!r} "
        f"ts_shape={tuple(sample_ts.shape)} dtype={args.dtype}"
    )

    vllm_dt = vllm_out = hf_dt = hf_out = None
    vllm_texts: list[str] = []
    hf_texts: list[str] = []

    if args.mode in ("both", "vllm"):
        print("\n========== VLLM PATH ==========")
        vllm_dt, vllm_out, vllm_texts = _run_vllm(
            vllm_model=args.vllm_model,
            requests=requests,
            max_tokens=args.max_tokens,
            dtype=args.dtype,
        )
        vllm_tps = vllm_out / vllm_dt if vllm_dt > 0 else 0.0
        print(
            f"[vllm] wall={vllm_dt:.2f}s out_tokens={vllm_out} "
            f"tok/s={vllm_tps:.1f} req/s={args.n/vllm_dt:.2f}"
        )

    if args.mode in ("both", "hf"):
        if args.demo_checkpoint is None:
            print("\n[hf] --demo-checkpoint not given; skipping HF baseline")
        else:
            print("\n========== HF PATH ==========")
            hf_dt, hf_out, hf_texts = _run_hf_demo(
                demo_checkpoint=args.demo_checkpoint,
                requests=requests,
                max_tokens=args.max_tokens,
                max_ts_length=args.max_ts_length,
                num_vars=args.num_vars,
                dtype_name=args.dtype,
            )
            hf_tps = hf_out / hf_dt if hf_dt > 0 else 0.0
            print(
                f"[hf]   wall={hf_dt:.2f}s out_tokens={hf_out} "
                f"tok/s={hf_tps:.1f} req/s={args.n/hf_dt:.2f}"
            )

    if args.show_samples > 0:
        print("\n=== samples ===")
        for i in range(min(args.show_samples, args.n)):
            if vllm_texts:
                print(f"[{i}] vllm: {vllm_texts[i]!r}")
            if hf_texts:
                print(f"[{i}] hf:   {hf_texts[i]!r}")

    if vllm_dt is not None and hf_dt is not None:
        if vllm_out == 0 or hf_out == 0 or vllm_dt == 0 or hf_dt == 0:
            print(
                f"\n=== speedup === skipped (vllm_out={vllm_out} hf_out={hf_out} "
                f"vllm_dt={vllm_dt} hf_dt={hf_dt}); one path generated no tokens"
            )
            return
        speedup_tps = (vllm_out / vllm_dt) / (hf_out / hf_dt)
        speedup_wall = hf_dt / vllm_dt
        print(
            f"\n=== speedup === vllm vs hf: tok/s ratio={speedup_tps:.2f}x  "
            f"wall ratio={speedup_wall:.2f}x"
        )
        target = 5.0
        if speedup_tps >= target:
            print(f"[eval] OK: meets plan §VLLM-6 bar (>= {target}x)")
        else:
            print(
                f"[eval] WARN: below plan §VLLM-6 bar (got {speedup_tps:.2f}x < {target}x). "
                "Synthetic checkpoints often run below the real-checkpoint ratio "
                "because batching wins amortize over real encoder cost."
            )


if __name__ == "__main__":
    main()
