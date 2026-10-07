"""
Quick test: Does eager attention produce deterministic outputs across batch compositions?
This tests whether the SDPA kernel on XPU is the source of batch-dependent output variation.
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


def test_attention_determinism(attn_impl="eager"):
    from src.config import PRISM_CONFIGS, ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    config_dict = PRISM_CONFIGS["prism-auroragpt-2b"].copy()
    config_dict["attn_implementation"] = attn_impl
    config = ModelConfig(**config_dict)

    device = "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    print(f"Loading model with attn_implementation={attn_impl} on {device}")

    model = UnifiedTransformer(config)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.llm_backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Create 4 distinct synthetic images
    torch.manual_seed(42)
    img_A = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(123)
    img_B = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(456)
    img_C = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(789)
    img_D = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    prompt = "The image"
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)

    # Single
    with torch.no_grad():
        gen_single = model.generate(
            {"image": img_A, "text": prompt_ids},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_single = tokenizer.decode(gen_single[0], skip_special_tokens=True)

    # Batch 1: A,B,C,D
    batch1_imgs = torch.cat([img_A, img_B, img_C, img_D], dim=0)
    batch1_prompts = prompt_ids.expand(4, -1)
    with torch.no_grad():
        gen_batch1 = model.generate(
            {"image": batch1_imgs, "text": batch1_prompts},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_batch1 = tokenizer.batch_decode(gen_batch1, skip_special_tokens=True)

    # Batch 2: A,D,C,B (different order)
    batch2_imgs = torch.cat([img_A, img_D, img_C, img_B], dim=0)
    batch2_prompts = prompt_ids.expand(4, -1)
    with torch.no_grad():
        gen_batch2 = model.generate(
            {"image": batch2_imgs, "text": batch2_prompts},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_batch2 = tokenizer.batch_decode(gen_batch2, skip_special_tokens=True)

    # Batch 3: Run batch1 again to check reproducibility
    with torch.no_grad():
        gen_batch3 = model.generate(
            {"image": batch1_imgs, "text": batch1_prompts},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_batch3 = tokenizer.batch_decode(gen_batch3, skip_special_tokens=True)

    match_single_b1 = torch.equal(gen_single, gen_batch1[0:1])
    match_single_b2 = torch.equal(gen_single, gen_batch2[0:1])
    match_b1_b2 = torch.equal(gen_batch1[0:1], gen_batch2[0:1])
    match_b1_b3 = torch.equal(gen_batch1, gen_batch3)  # Same batch, same order

    print()
    print(f"=== {attn_impl.upper()} ATTENTION RESULTS ===")
    print(f"A alone:     {repr(pred_single[:120])}")
    print(f"A in batch1: {repr(pred_batch1[0][:120])}")
    print(f"A in batch2: {repr(pred_batch2[0][:120])}")
    print(f"A in batch3: {repr(pred_batch3[0][:120])}")
    print()
    print(f"Match (alone vs batch1[0]):  {match_single_b1}")
    print(f"Match (alone vs batch2[0]):  {match_single_b2}")
    print(f"Match (batch1[0] vs batch2[0]): {match_b1_b2}")
    print(f"Match (batch1 vs batch3 - same composition): {match_b1_b3}")

    # Check if different images produce different outputs (sanity)
    all_same = all(p == pred_batch1[0] for p in pred_batch1)
    print(f"All 4 batch1 predictions identical: {all_same}")

    return match_single_b1 and match_single_b2


if __name__ == "__main__":
    attn = sys.argv[1] if len(sys.argv) > 1 else "eager"
    passed = test_attention_determinism(attn)
    print()
    if passed:
        print(f"PASS: {attn} attention is batch-composition-independent")
    else:
        print(f"FAIL: {attn} attention produces batch-dependent outputs")
    sys.exit(0 if passed else 1)
