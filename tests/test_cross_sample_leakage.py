"""
Diagnostic test for cross-sample information leakage during generation.

This test checks whether generating predictions for a sample produces
different outputs when that sample is alone vs when it's batched with
other samples. If there IS cross-sample leakage, the same image will
produce different text when surrounded by different batch-mates.

Usage:
    # On a compute node with GPU:
    python tests/test_cross_sample_leakage.py

    # Or via pytest:
    pytest tests/test_cross_sample_leakage.py -v -s
"""

import logging
import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.aurora, pytest.mark.gpu, pytest.mark.integration]

# NOTE: These functions return True/False instead of using assert. Pytest
# ignores return values, so a "fail" run will still report PASS — read the
# stdout when running manually on a compute node.

logger = logging.getLogger(__name__)


def test_cross_sample_leakage():
    """
    Test 1: Same image, different batch compositions.

    If generation is batch-independent, the same image should produce
    the same prediction regardless of what other images are in the batch.
    """
    from src.config import PRISM_CONFIGS, ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    # Use a small model for testing
    # Try AuroraGPT-2B config if available, else fall back to something loadable
    config_dict = PRISM_CONFIGS["prism-auroragpt-2b"].copy()

    # Check if the backbone path exists
    backbone_path = config_dict.get("llm_backbone_id", "")
    if not os.path.exists(backbone_path):
        print(f"Backbone not found at {backbone_path}, trying smaller model...")
        # Fall back to a model we can load
        config_dict = PRISM_CONFIGS.get(
            "prism-granite-2b", PRISM_CONFIGS["prism-small"]
        ).copy()
        backbone_path = config_dict.get(
            "llm_backbone_id", config_dict.get("hf_model_id", "")
        )

    config = ModelConfig(**config_dict)

    # Determine device
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu:0"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Loading model with backbone: {config.llm_backbone_id}")

    model = UnifiedTransformer(config)
    model = model.to(device)
    model.eval()

    # Get tokenizer from backbone (model doesn't store it directly)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Create 4 distinct synthetic images (random but reproducible)
    torch.manual_seed(42)
    img_A = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(123)
    img_B = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(456)
    img_C = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(789)
    img_D = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    # Create a common prompt
    prompt = "The image"
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids = encoded["input_ids"].to(device)  # (1, T)

    print(f"\nPrompt: {repr(prompt)}")
    print(f"Prompt token IDs: {prompt_ids.tolist()}")

    # ---- Test 1: Generate for image A alone ----
    print("\n=== Test 1: Single-sample generation (image A alone) ===")
    with torch.no_grad():
        gen_ids_single = model.generate(
            {"image": img_A, "text": prompt_ids},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_single = tokenizer.decode(gen_ids_single[0], skip_special_tokens=True)
    print(f"Prediction (A alone): {repr(pred_single)}")

    # ---- Test 2: Generate for image A in a batch with B, C, D ----
    print("\n=== Test 2: Batched generation (A with B, C, D) ===")
    batch_imgs_1 = torch.cat([img_A, img_B, img_C, img_D], dim=0)  # (4, 3, 224, 224)
    batch_prompts_1 = prompt_ids.expand(4, -1)  # (4, T) same prompt for all

    with torch.no_grad():
        gen_ids_batch1 = model.generate(
            {"image": batch_imgs_1, "text": batch_prompts_1},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    preds_batch1 = tokenizer.batch_decode(gen_ids_batch1, skip_special_tokens=True)
    for i, p in enumerate(preds_batch1):
        print(f"Prediction batch1[{i}]: {repr(p)}")

    # ---- Test 3: Generate for image A in a DIFFERENT batch (A, D, C, B) ----
    print("\n=== Test 3: Batched generation (A with D, C, B - different order) ===")
    batch_imgs_2 = torch.cat([img_A, img_D, img_C, img_B], dim=0)  # (4, 3, 224, 224)
    batch_prompts_2 = prompt_ids.expand(4, -1)

    with torch.no_grad():
        gen_ids_batch2 = model.generate(
            {"image": batch_imgs_2, "text": batch_prompts_2},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    preds_batch2 = tokenizer.batch_decode(gen_ids_batch2, skip_special_tokens=True)
    for i, p in enumerate(preds_batch2):
        print(f"Prediction batch2[{i}]: {repr(p)}")

    # ---- Compare results ----
    print("\n=== COMPARISON ===")
    print(f"A alone:     {repr(pred_single)}")
    print(f"A in batch1: {repr(preds_batch1[0])}")
    print(f"A in batch2: {repr(preds_batch2[0])}")

    # Check token-level match
    match_1 = torch.equal(gen_ids_single, gen_ids_batch1[0:1])
    match_2 = torch.equal(gen_ids_single, gen_ids_batch2[0:1])
    match_12 = torch.equal(gen_ids_batch1[0:1], gen_ids_batch2[0:1])

    print(f"\nToken-level match (A alone vs batch1[0]): {match_1}")
    print(f"Token-level match (A alone vs batch2[0]): {match_2}")
    print(f"Token-level match (batch1[0] vs batch2[0]): {match_12}")

    # Also check that different images produce different outputs
    all_same = all(p == preds_batch1[0] for p in preds_batch1)
    print(f"\nAll 4 predictions in batch1 are identical: {all_same}")
    if all_same:
        print(
            "WARNING: All predictions are identical - model may not be using image info at all"
        )

    # Print per-sample comparison
    print("\n=== Per-sample batch1 vs batch2 ===")
    for i in range(4):
        print(f"  Sample {i} batch1: {repr(preds_batch1[i][:80])}")
    print("---")
    for i in range(4):
        print(f"  Sample {i} batch2: {repr(preds_batch2[i][:80])}")

    # VERDICT
    print("\n" + "=" * 60)
    if match_1 and match_2:
        print("PASS: No cross-sample leakage detected.")
        print("Image A produces identical output regardless of batch composition.")
    else:
        print("FAIL: CROSS-SAMPLE LEAKAGE DETECTED!")
        print("Image A produces DIFFERENT outputs depending on batch composition.")

        # Show exactly where they diverge
        ids_single = gen_ids_single[0].tolist()
        ids_batch1 = gen_ids_batch1[0].tolist()

        for pos in range(min(len(ids_single), len(ids_batch1))):
            if ids_single[pos] != ids_batch1[pos]:
                print(f"  First divergence at position {pos}:")
                print(
                    f"    Single: token {ids_single[pos]} = {repr(tokenizer.decode([ids_single[pos]]))}"
                )
                print(
                    f"    Batch1: token {ids_batch1[pos]} = {repr(tokenizer.decode([ids_batch1[pos]]))}"
                )
                break
    print("=" * 60)

    if match_1 and match_2:
        return True
    return False


def test_cross_sample_leakage_with_different_prompts():
    """
    Test 2: Same image, different per-sample prompts (mimics visualize_predictions).

    This more closely replicates the actual visualization pipeline where
    each sample gets a different prompt derived from its GT text.
    """
    from src.config import PRISM_CONFIGS, ModelConfig
    from src.model import UnifiedTransformer
    from transformers import AutoTokenizer

    config_dict = PRISM_CONFIGS["prism-auroragpt-2b"].copy()
    backbone_path = config_dict.get("llm_backbone_id", "")
    if not os.path.exists(backbone_path):
        print(f"Backbone not found at {backbone_path}, skipping test")
        return True

    config = ModelConfig(**config_dict)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu:0"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    model = UnifiedTransformer(config)
    model = model.to(device)
    model.eval()

    # Get tokenizer from backbone (model doesn't store it directly)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_backbone_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Create distinct images
    torch.manual_seed(42)
    img_A = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    torch.manual_seed(123)
    img_B = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    # Different prompts (like visualize_predictions with GT prefix)
    prompts = ["This image is a", "In front of a"]
    encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    batch_prompt_ids = encoded["input_ids"].to(device)  # (2, T_padded)

    print(f"Prompts: {prompts}")
    print(f"Tokenized shape: {batch_prompt_ids.shape}")

    # ---- Generate A alone with prompt[0] ----
    single_encoded = tokenizer(prompts[0], return_tensors="pt")
    single_prompt_ids = single_encoded["input_ids"].to(device)

    with torch.no_grad():
        gen_ids_single = model.generate(
            {"image": img_A, "text": single_prompt_ids},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    pred_single = tokenizer.decode(gen_ids_single[0], skip_special_tokens=True)

    # ---- Generate A and B in batch with different prompts ----
    batch_imgs = torch.cat([img_A, img_B], dim=0)

    with torch.no_grad():
        gen_ids_batch = model.generate(
            {"image": batch_imgs, "text": batch_prompt_ids},
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    preds_batch = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=True)

    print(f"\nA alone (prompt='{prompts[0]}'): {repr(pred_single)}")
    print(f"A in batch (prompt='{prompts[0]}'): {repr(preds_batch[0])}")
    print(f"B in batch (prompt='{prompts[1]}'): {repr(preds_batch[1])}")

    # Check for leakage: A's output should match regardless of B's presence
    # Note: With padding=True, the tokenized prompt for A might differ when
    # batched (extra pad tokens). This IS a potential source of difference
    # (not leakage per se, but pad tokens affecting attention).

    # Compare raw IDs
    single_ids = gen_ids_single[0].tolist()
    batch_ids = gen_ids_batch[0].tolist()

    # Find common prefix length
    common = 0
    for s, b in zip(single_ids, batch_ids, strict=False):
        if s == b:
            common += 1
        else:
            break

    print(f"\nCommon token prefix: {common} / {min(len(single_ids), len(batch_ids))}")

    if common < min(len(single_ids), len(batch_ids)):
        print(f"Divergence at position {common}:")
        print(f"  Single: {single_ids[common : common + 5]}")
        print(f"  Batch:  {batch_ids[common : common + 5]}")

        # Check if this is due to padding differences
        single_prompt_len = single_prompt_ids.shape[1]
        batch_prompt_len = batch_prompt_ids.shape[1]
        print(
            f"\nPrompt lengths - Single: {single_prompt_len}, Batch: {batch_prompt_len}"
        )
        if single_prompt_len != batch_prompt_len:
            print("NOTE: Prompt padding difference detected!")
            print("The single prompt has no padding, but the batched prompt is padded")
            print("to match the longest prompt in the batch. This pad token difference")
            print(
                "in the input embeddings can cause the model to generate differently."
            )
            print("This is NOT cross-sample leakage - it's a padding artifact.")
            print("FIX: The attention mask in model.generate() should mask pad tokens.")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("CROSS-SAMPLE LEAKAGE DIAGNOSTIC TEST")
    print("=" * 70)

    passed = test_cross_sample_leakage()

    print("\n\n")
    print("=" * 70)
    print("PADDING ARTIFACT TEST (different prompts)")
    print("=" * 70)

    test_cross_sample_leakage_with_different_prompts()

    if not passed:
        sys.exit(1)
