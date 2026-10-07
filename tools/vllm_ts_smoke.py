"""Time-series end-to-end smoke for VLLM-5.

Boots vLLM against an exported checkpoint that has `time_series` in
`active_modalities`, posts a tensor through the plugin's data parser +
encoder + projector, asserts a non-empty generation.

No parity claim — synthetic checkpoints have random weights so the output
is gibberish. The point is to prove the PRISM-side ts code path runs
end-to-end inside vLLM's engine (parse_time_series_data ->
TimeSeriesProcessorItems -> encode -> BatchFeature ->
PromptUpdateDetails(envelope) -> embed_multimodal -> language model ->
generate).
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    _PROJECT_ROOT if not _pp else f"{_PROJECT_ROOT}{os.pathsep}{_pp}"
)

import src.vllm_plugin  # noqa: E402

src.vllm_plugin.register()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vllm-model", required=True,
                   help="Exported PRISM dir with active_modalities including "
                   "'time_series'.")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-ts-length", type=int, default=32,
                   help="Time-series length to post; must be <= the export's "
                   "max_ts_length (excess is right-truncated).")
    p.add_argument("--num-vars", type=int, default=1)
    p.add_argument(
        "--prompt-form",
        default="<time_series>forecast:",
        help="Prompt with one <time_series> placeholder per ts item.",
    )
    args = p.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    print(f"[ts-smoke] booting LLM(model={args.vllm_model!r}, enforce_eager=True)")
    llm = LLM(
        model=args.vllm_model,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"time_series": 1, "image": 0},
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    ts = torch.randn(args.max_ts_length, args.num_vars)
    req = {
        "prompt": args.prompt_form,
        "multi_modal_data": {"time_series": ts},
    }
    print(f"[ts-smoke] posting ts shape={tuple(ts.shape)} via {args.prompt_form!r}")
    outs = llm.generate([req], sp)
    if not outs or not outs[0].outputs or not outs[0].outputs[0].text:
        raise SystemExit(
            f"FAILED: empty output from ts request: {outs[0].outputs if outs else outs}"
        )
    text = outs[0].outputs[0].text
    print(f"[ts-smoke] >>> {text!r}")
    print("[ts-smoke] PASSED (ts data plane + encoder ran end-to-end)")


if __name__ == "__main__":
    main()
