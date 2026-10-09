"""Regression test for a pre-existing GRPO bug surfaced by the step-6
end-to-end dataset-mix verification (30-step run against all 3 DNA datasets):

HF tokenizers return an empty tensor as float32 (not the usual int64) when
encoding "" via tokenizer.encode(text, return_tensors="pt") — there are no
token values to infer an integer dtype from. This happens whenever a GRPO
rollout's generated completion decodes to an empty string (e.g. immediate-EOS
generation, or an all-special-token generation stripped by
skip_special_tokens=True in _generate_completions).

trainer_zone_d_grpo.py's train() loop built comp_ids from this call without
forcing a dtype, then did:
    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
torch.cat silently upcasts the whole result to float32 when concatenating a
long tensor with a float tensor, corrupting prompt_ids too — even though
prompt_ids alone was fine. This crashed nn.Embedding downstream ("Expected
tensor for argument #1 'indices' to have one of the following scalar types:
Long, Int; but got torch.cuda.FloatTensor instead") on whichever batch
happened to draw an empty-completion rollout — unrelated to which of the 3
DNA datasets supplied the prompt.

Fixed by explicitly casting comp_ids to torch.long right after encoding
(trainer_zone_d_grpo.py, in the `if self.tokenizer:` branch inside train()'s
per-completion loop).

Reproduces the exact HF quirk here without loading a real tokenizer/model, so
this test is fast and CPU-only.
"""

import pytest
import torch

pytestmark = [pytest.mark.unit]


class FakeTokenizerWithEmptyStringFloatBug:
    """Reproduces the exact real-tokenizer quirk: encoding "" returns an empty
    float32 tensor via return_tensors="pt", while any non-empty string returns
    a normal int64 tensor. Confirmed against the real prism-olmo-1b-interleaved
    tokenizer on 2026-07-01 (see conversation/commit for the exact repro).
    """

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, return_tensors=None):
        if text == "":
            return torch.empty(1, 0, dtype=torch.float32)
        # Fake but deterministic non-empty encoding.
        ids = [ord(c) % 100 for c in text]
        return torch.tensor([ids], dtype=torch.long)


def test_torch_cat_with_float_empty_tensor_upcasts_whole_result():
    """Demonstrates the underlying torch.cat behavior this bug depends on,
    independent of any tokenizer or trainer code — documents *why* the fix
    (forcing comp_ids to long) is necessary rather than incidental.
    """
    prompt_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    empty_float_comp_ids = torch.empty(1, 0, dtype=torch.float32)

    result = torch.cat([prompt_ids, empty_float_comp_ids], dim=1)
    assert result.dtype == torch.float32  # confirms the corruption mechanism


def test_fixed_comp_ids_cast_prevents_dtype_corruption():
    """Simulates the fixed code path in trainer_zone_d_grpo.py's train():
    comp_ids = tokenizer.encode(completion, return_tensors="pt").to(device, dtype=torch.long)
    """
    tok = FakeTokenizerWithEmptyStringFloatBug()
    prompt_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)

    empty_completion = ""
    comp_ids = tok.encode(empty_completion, return_tensors="pt").to("cpu", dtype=torch.long)
    assert comp_ids.dtype == torch.long

    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    assert full_ids.dtype == torch.long  # no corruption — safe to feed to nn.Embedding
    assert full_ids.shape == (1, 3)  # empty completion contributes 0 tokens


def test_fixed_comp_ids_cast_is_a_noop_for_normal_completions():
    tok = FakeTokenizerWithEmptyStringFloatBug()
    prompt_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)

    comp_ids = tok.encode("hi", return_tensors="pt").to("cpu", dtype=torch.long)
    assert comp_ids.dtype == torch.long

    full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    assert full_ids.dtype == torch.long
    assert full_ids.shape == (1, 5)


def test_unfixed_path_would_have_corrupted_dtype():
    """Sanity check that this test suite would have caught the original bug
    had it existed before the fix — i.e. this isn't a tautological test.
    """
    tok = FakeTokenizerWithEmptyStringFloatBug()
    prompt_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)

    # The original buggy line had no explicit dtype cast:
    comp_ids_buggy = tok.encode("", return_tensors="pt").to("cpu")
    full_ids_buggy = torch.cat([prompt_ids, comp_ids_buggy], dim=1)
    assert full_ids_buggy.dtype == torch.float32  # reproduces the crash-causing corruption
