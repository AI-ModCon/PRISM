"""Pre-flight gate #3: confirm a time-series training checkpoint exists.

VLLM-4's parity oracle compares an HF demo run (UnifiedTransformer with a
trained time_series encoder) against vLLM. Without a checkpoint that contains
`encoders.time_series.*` keys, the oracle has nothing to oracle against.

Run as part of `tools/_vllm_smoke_runner.sh --ts-checkpoint <dir>` to fail
fast before any code in Stage B/VLLM-4 is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Directory containing model.safetensors (and training_state.json).",
    )
    p.add_argument(
        "--key-prefix",
        default="encoders.time_series.",
        help="State-dict key prefix that must be present.",
    )
    args = p.parse_args()

    ckpt_dir = args.checkpoint
    if not ckpt_dir.exists():
        raise SystemExit(f"TS CHECKPOINT MISSING: {ckpt_dir} does not exist")

    sd_file = ckpt_dir / "model.safetensors"
    if not sd_file.exists():
        raise SystemExit(f"TS CHECKPOINT MISSING: {sd_file} not found")

    from safetensors import safe_open

    with safe_open(str(sd_file), framework="pt") as f:
        keys = list(f.keys())

    matching = [k for k in keys if args.key_prefix in k]
    if not matching:
        encoder_keys = [k for k in keys if ".encoders." in k or k.startswith("encoders.")]
        sample = encoder_keys[:8]
        raise SystemExit(
            f"TS CHECKPOINT INCOMPLETE: no {args.key_prefix!r} keys in {sd_file}\n"
            f"  total keys: {len(keys)}\n"
            f"  encoder keys (sample): {sample}"
        )

    print(f"[ts-check] checkpoint: {sd_file}")
    print(f"[ts-check] total keys: {len(keys)}")
    print(f"[ts-check] {args.key_prefix!r} keys: {len(matching)} "
          f"(first 3: {matching[:3]})")

    ts_file = ckpt_dir / "training_state.json"
    if ts_file.exists():
        try:
            meta = json.loads(ts_file.read_text())
            print(f"[ts-check] step: {meta.get('global_step')!r} "
                  f"epoch: {meta.get('epoch')!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ts-check] could not parse training_state.json: {exc!r}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
