"""Assert the vLLM path on the current revision matches a reference vLLM output.

The plan originally specified "first ≥N/M greedy tokens match the demo
(HF UnifiedTransformer.generate) path." On Aurora XPU the HF demo path is
non-deterministic across runs (verified 2026-05-25 against main on
test_images/cat.jpg), so demo-vs-vLLM is an unreliable oracle.

This script instead compares the vLLM path's output against a frozen
reference string (--ref-text) or a reference token list (--ref-tokens).
The reference comes from a known-good revision (e.g., PR #41 main).
This catches refactor regressions in the vLLM model class without
falsely flagging XPU floating-point divergence on the training-side path.

If neither --ref-text nor --ref-tokens is provided, the script runs both
paths and prints the comparison without failing — useful as a manual
sanity check before locking in a new reference.
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


def _tokenize_first_n(model_dir: str, text: str, n: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    return list(ids)[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vllm-model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Optional: run the demo path too and print its output. "
                   "Demo output is informational only — it is NOT used to "
                   "fail the assertion (non-deterministic on XPU).")
    p.add_argument(
        "--prompt-form",
        default="<image>The image shows",
        help="Exact prompt for vLLM (see tools/vllm_parity.py).",
    )
    p.add_argument("--window", type=int, default=20, help="First M tokens compared.")
    p.add_argument("--min-match", type=int, default=18,
                   help="Fail unless N of the first M tokens match the reference.")
    p.add_argument(
        "--ref-text",
        default=None,
        help="Reference vLLM completion text (from a known-good revision). "
        "Compared token-by-token against the new vLLM output.",
    )
    p.add_argument(
        "--ref-tokens",
        default=None,
        help="Comma-separated list of reference token ids (alternative to "
        "--ref-text). Useful when the reference was captured pre-tokenized.",
    )
    args = p.parse_args()

    import importlib.util

    _PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_parity.py")
    _spec = importlib.util.spec_from_file_location("_vllm_parity_inline", _PARITY)
    assert _spec is not None and _spec.loader is not None
    vllm_parity = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(vllm_parity)

    if args.checkpoint:
        print("\n========== DEMO PATH (informational only) ==========")
        demo_text = vllm_parity.run_demo_path(args.checkpoint, args.image)
        print(f"[demo] >>> {demo_text!r}")

    print("\n========== VLLM PATH ==========")
    vllm_text = vllm_parity.run_vllm_path(args.vllm_model, args.image, args.prompt_form)
    print(f"[vllm] >>> {vllm_text!r}")

    vllm_ids = _tokenize_first_n(args.vllm_model, vllm_text, args.window)

    if args.ref_tokens:
        ref_ids = [int(x) for x in args.ref_tokens.split(",")][: args.window]
    elif args.ref_text:
        ref_ids = _tokenize_first_n(args.vllm_model, args.ref_text, args.window)
    else:
        print(
            "\n[parity] no reference supplied (--ref-text / --ref-tokens). "
            "Soft check only — no pass/fail. Re-run with a frozen reference "
            "from a known-good revision to gate."
        )
        print(f"[parity] new vllm tokens (first {args.window}): {vllm_ids}")
        return

    width = min(len(vllm_ids), len(ref_ids), args.window)
    matches = sum(1 for i in range(width) if vllm_ids[i] == ref_ids[i])
    print(f"\n[parity] vllm vs reference: {matches}/{width} match "
          f"(threshold: {args.min_match}/{args.window})")
    if matches < args.min_match:
        raise SystemExit(
            f"PARITY FAILED: {matches}/{width} < {args.min_match}/{args.window}\n"
            f"  vllm_ids:      {vllm_ids}\n  reference_ids: {ref_ids}"
        )
    print("[parity] OK")


if __name__ == "__main__":
    main()
