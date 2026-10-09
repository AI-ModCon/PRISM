"""
Minimal test: Is XPU generation deterministic for a single sample run twice?
This isolates XPU-level non-determinism from any batch/cross-sample effects.
"""

import logging
import sys

import pytest
import torch

pytestmark = [pytest.mark.aurora, pytest.mark.gpu, pytest.mark.integration]

# NOTE: Diagnostic test — returns True/False instead of asserting, so a "fail"
# result will still report PASS under pytest. Read the stdout when running
# manually on a compute node.

logging.basicConfig(level=logging.INFO)


def test_single_sample_determinism():
    from src.config import PRISM_CONFIGS, ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    config_dict = PRISM_CONFIGS["prism-auroragpt-2b"].copy()
    config_dict["attn_implementation"] = "eager"
    config = ModelConfig(**config_dict)

    device = "xpu:0"
    print(f"Loading model on {device}")

    model = UnifiedTransformer(config)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.llm_backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    torch.manual_seed(42)
    img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    prompt = "The image"
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)

    results = []
    for i in range(5):
        with torch.no_grad():
            gen = model.generate(
                {"image": img, "text": prompt_ids},
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        results.append((gen[0].tolist(), pred))
        print(f"Run {i}: {repr(pred[:100])}")

    print()
    all_match = all(r[0] == results[0][0] for r in results[1:])
    print(f"All 5 single-sample runs identical: {all_match}")

    if not all_match:
        # Find first divergence
        for i in range(1, len(results)):
            if results[i][0] != results[0][0]:
                ids0 = results[0][0]
                ids_i = results[i][0]
                for pos in range(min(len(ids0), len(ids_i))):
                    if ids0[pos] != ids_i[pos]:
                        print(f"  Run 0 vs Run {i}: diverge at position {pos}")
                        print(
                            f"    Run 0: token {ids0[pos]} = {repr(tokenizer.decode([ids0[pos]]))}"
                        )
                        print(
                            f"    Run {i}: token {ids_i[pos]} = {repr(tokenizer.decode([ids_i[pos]]))}"
                        )
                        break

    # Also test: same image tensor, newly created each time
    print("\n=== Test 2: Fresh tensor each time ===")
    results2 = []
    for i in range(3):
        torch.manual_seed(42)
        img_fresh = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(
                {"image": img_fresh, "text": prompt_ids.clone()},
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        results2.append((gen[0].tolist(), pred))
        print(f"Fresh run {i}: {repr(pred[:100])}")

    all_match2 = all(r[0] == results2[0][0] for r in results2[1:])
    print(f"All 3 fresh-tensor runs identical: {all_match2}")

    # Also test with torch.xpu.synchronize() between runs
    print("\n=== Test 3: With explicit sync between runs ===")
    results3 = []
    for i in range(3):
        torch.xpu.synchronize()
        with torch.no_grad():
            gen = model.generate(
                {"image": img, "text": prompt_ids},
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        torch.xpu.synchronize()
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        results3.append((gen[0].tolist(), pred))
        print(f"Sync run {i}: {repr(pred[:100])}")

    all_match3 = all(r[0] == results3[0][0] for r in results3[1:])
    print(f"All 3 synced runs identical: {all_match3}")

    return all_match and all_match2 and all_match3


if __name__ == "__main__":
    passed = test_single_sample_determinism()
    print()
    if passed:
        print("PASS: Single-sample generation is deterministic on XPU")
    else:
        print("RESULT: XPU shows non-deterministic generation even for single samples")
    sys.exit(0 if passed else 1)
