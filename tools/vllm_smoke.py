"""vLLM plugin smoke test — registration + engine boot + 4-token generation.

Gates every Stage A vLLM PR. Three checks:

  1. After `src.vllm_plugin.register()` runs,
     `"PrismForConditionalGeneration" in ModelRegistry.get_supported_archs()`.
  2. `LLM(model=<exported_dir>, enforce_eager=True)` boots successfully.
  3. `.generate("hello", SamplingParams(max_tokens=4))` returns a non-empty
     completion. (We do not check token content here — that is the parity
     test's job; this gate exists to catch boot-time regressions cheaply.)

Run via `tools/_vllm_smoke_runner.sh --vllm-model <exported_dir>`.
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


def check_registration() -> None:
    from vllm import ModelRegistry

    archs = ModelRegistry.get_supported_archs()
    if "PrismForConditionalGeneration" not in archs:
        raise SystemExit(
            "REGISTRATION FAILED: 'PrismForConditionalGeneration' not in "
            f"ModelRegistry.get_supported_archs() (found {len(archs)} archs)"
        )
    print("[smoke] registration: PrismForConditionalGeneration present")


def check_boot_and_generate(model_dir: str) -> None:
    from vllm import LLM, SamplingParams

    print(f"[smoke] booting LLM(model={model_dir!r}, enforce_eager=True)")
    llm = LLM(model=model_dir, trust_remote_code=True, enforce_eager=True)

    sp = SamplingParams(max_tokens=4, temperature=0.0)
    outs = llm.generate(["hello"], sp)
    if not outs or not outs[0].outputs or not outs[0].outputs[0].text:
        raise SystemExit("GENERATE FAILED: empty output from LLM.generate('hello')")
    print(f"[smoke] generate: {outs[0].outputs[0].text!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vllm-model",
        required=True,
        help="Exported PRISM dir loadable by vLLM (config.json + safetensors).",
    )
    args = p.parse_args()

    check_registration()
    check_boot_and_generate(args.vllm_model)
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
