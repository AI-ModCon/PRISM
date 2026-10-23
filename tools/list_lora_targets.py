#!/usr/bin/env python3
"""List the layer names ``training.lora_target_modules`` accepts for a model.

LoRA targets are leaf names of the backbone's ``nn.Linear`` layers (e.g.
``q_proj``), and they differ between architectures. This builds the backbone
on the meta device (only its config.json is read, no weights) and prints each
name that apply_lora_torchtune would accept, with how many layers carry it.
The selection is the same ``_select_lora_targets`` the trainer uses.

    python tools/list_lora_targets.py model=prism_olmo_1b
    python tools/list_lora_targets.py --backbone-id Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn as nn
from src.utils.lora_utils import _select_lora_targets

EXCLUDE = ("lm_head",)  # apply_lora_torchtune's default


def count_targets(backbone: nn.Module, exclude: tuple[str, ...] = EXCLUDE) -> Counter[str]:
    """Leaf name -> number of nn.Linear layers that LoRA would adapt by default."""
    return Counter(name.split(".")[-1] for name, _ in _select_lora_targets(backbone, exclude, None))


def _backbone_id_from_hydra(overrides: list[str]) -> str:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "src" / "conf"), version_base="1.2"):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg.model.backbone_id


def _build_on_meta(backbone_id: str) -> nn.Module:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(backbone_id, trust_remote_code=True)
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides selecting the model, e.g. model=prism_olmo_1b"
    )
    parser.add_argument("--backbone-id", help="HF model id or local path; skips Hydra")
    args = parser.parse_args()

    backbone_id = args.backbone_id or _backbone_id_from_hydra(args.overrides)
    counts = count_targets(_build_on_meta(backbone_id))

    print(f"Backbone: {backbone_id}")
    print(f"LoRA-targetable nn.Linear leaf names (excluded: {list(EXCLUDE)}):")
    for leaf, n in sorted(counts.items()):
        print(f"  {leaf:<24} {n:>4} layers")
    print(f"  {'total':<24} {sum(counts.values()):>4} layers")
    print("\nExample: training.lora_target_modules=[q_proj,v_proj]  (null adapts all of the above)")


if __name__ == "__main__":
    main()
