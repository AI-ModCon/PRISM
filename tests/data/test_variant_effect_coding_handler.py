"""Step 3 of the variant_effect dataset integration: handler-level tests for
_process_dna_bioreason_variant_effect_coding in src/data/multimodal.py.

Uses object.__new__(StreamingMultimodalDataset) plus manually-set attributes
rather than the full __init__, since the handler only reads those — the full
constructor needs HF downloads, a DNA tokenizer, and dataset streaming that are
out of scope for this unit-level test (covered later by an end-to-end
StreamingMultimodalDataset smoke test on the real wanglab/variant_effect_coding
stream).

The handler returns the structured dict
{"prompt", "dna_sequences", "answer", "class_weight"}, NOT the generic
(data_tensor, caption, metadata_str) 3-tuple the other modalities use. That
changed in b303043, which deleted the shared _finalize_dna_item return path and
moved rendering, tokenization and metadata construction into the Modality.DNA
branch of __iter__ (which has its own `isinstance(processed_result, dict)`
arm). The old tuple slots now map as:

    data_tensor  -> tokenized in __iter__ from dict["dna_sequences"]
                    (the handler emits raw sequence strings)
    caption      -> rendered in __iter__ via _render_dna_prompt_text(
                    dict["prompt"]) -- the `render()` helper below calls that
                    same real method, so caption assertions still exercise
                    production rendering rather than a test-local copy
    metadata_str -> example["_metadata"], while the RL answer/class_weight it
                    used to carry as formatted substrings are now real typed
                    keys (example["answer"], example["class_weight"]), read
                    directly by trainer_grpo.py. The original code flagged this
                    with `TODO: revisit the exact carrier`; b303043 did it.

`_FakeDnaTokenizer` stands in for the real NT tokenizer (no network access in
this unit test). The handler itself no longer tokenizes, so it is retained only
for the attributes make_dataset() sets.

Requires torch (multimodal.py imports it at module level) — run under the
cluster's prism conda env, same constraint as the other tests/data/ tests here.
"""

import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

# Real sample row shape confirmed against wanglab/variant_effect_coding train
# split (2026-07-01): ID, question, answer, reference_sequence, variant_sequence.
REAL_SAMPLE_ITEM = {
    "ID": "Task1_train_0",
    "question": (
        "The variant affects gene PERM1 (PPARGC1 and ESRR induced regulator, "
        "muscle 1), which is on Chromosome 1. Please evaluate"
    ),
    "answer": "Pathogenic; Renal tubular epithelial cell apoptosis",
    "reference_sequence": "ACGT" * 10,
    "variant_sequence": "ACGA" * 10,
}


class _FakeTokenizer:
    """Minimal stand-in for the LLM tokenizer: whitespace-token count only."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": text.split()})()


class _FakeDnaTokenizer:
    """Minimal stand-in for the NT tokenizer — one id per character, no network."""

    def __call__(self, seq, truncation=True, max_length=1024, return_tensors="pt"):
        ids = torch.tensor([[ord(c) % 97 for c in seq[:max_length]]], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def make_dataset(model_name: str, is_sft: bool) -> StreamingMultimodalDataset:
    ds = object.__new__(StreamingMultimodalDataset)
    ds.model_name = model_name
    ds.is_sft = is_sft
    ds.is_projector_only = False
    ds.class_weights_map = {}
    ds.tokenizer = _FakeTokenizer()
    ds.dna_tokenizer = _FakeDnaTokenizer()
    ds.use_reasoning_traces = True
    ds.model_config = ModelConfig()
    return ds


def render(ds, result):
    """Render a handler result the way __iter__ does, and return its full text.

    Calls the real _render_dna_prompt_text so these assertions cover the same
    rendering production uses, rather than re-implementing it here.
    """
    full_text, _prompt_text = ds._render_dna_prompt_text(result["prompt"])
    return full_text


@pytest.mark.parametrize("model_name", ["llm", "dna-llm"])
def test_answer_is_cleaned_to_pathogenicity_only(model_name):
    ds = make_dataset(model_name, is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    assert "<answer>pathogenic</answer>" in render(ds, result)
    # The cleaned answer is also a typed key now, not only a rendered substring.
    assert result["answer"] == "pathogenic"


def test_dna_llm_mode_uses_separate_dna_slots():
    ds = make_dataset("dna-llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    caption = render(ds, result)
    # The handler hands __iter__ the raw pair; tokenization into
    # dna_reference/dna_variant happens there (see the Modality.DNA branch).
    assert result["dna_sequences"] == [
        REAL_SAMPLE_ITEM["reference_sequence"],
        REAL_SAMPLE_ITEM["variant_sequence"],
    ]
    # Both DNA slots are present in the prompt, so the merge step has somewhere
    # to expand the encoded sequences into.
    slot_types = [c["type"] for c in result["prompt"][0]["content"]]
    assert "dna_reference" in slot_types and "dna_variant" in slot_types
    # Question-only in dna-llm mode — DNA sequences must not be inlined into text.
    assert REAL_SAMPLE_ITEM["reference_sequence"] not in caption
    assert REAL_SAMPLE_ITEM["question"] in caption


def test_llm_mode_inlines_dna_and_empties_dna_sequences():
    ds = make_dataset("llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    caption = render(ds, result)
    # llm mode inlines the sequences as text, so there is no DNA pair to encode.
    assert result["dna_sequences"] == ["", ""]
    assert REAL_SAMPLE_ITEM["reference_sequence"] in caption
    assert REAL_SAMPLE_ITEM["variant_sequence"] in caption


def test_sft_true_appends_answer_tagged_assistant_turn_with_no_reasoning():
    ds = make_dataset("dna-llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    caption = render(ds, result)
    # No reasoning field in this dataset -> reasoning_content must be empty,
    # never "Answer: X" (BioReason's literal format) — see handler docstring.
    assert "<think>" not in caption
    assert "<answer>pathogenic</answer>" in caption
    assert "<|im_start|>assistant" in caption
    # The "promptlen targetlen" split is computed in __iter__ from these two
    # renderings, so assert the property it depends on: the prompt stops at the
    # assistant marker and the answer lands strictly after it.
    full_text, prompt_text = ds._render_dna_prompt_text(result["prompt"])
    assert prompt_text.endswith("<|im_start|>assistant\n")
    assert len(full_text) > len(prompt_text)
    assert "<answer>pathogenic</answer>" not in prompt_text


def test_sft_false_omits_assistant_turn_for_grpo():
    ds = make_dataset("dna-llm", is_sft=False)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    caption = render(ds, result)
    # The generation marker is still emitted (that is what makes it a generation
    # prompt); what must be absent is the answer text after it.
    assert "<|im_start|>assistant" in caption
    assert "<answer>" not in caption  # generation-prompt only, no answer text
    # The answer still surfaces for reward computation, but as a real key --
    # trainer_grpo.py reads batch["answer"] rather than parsing a metadata tag.
    assert result["answer"] == "pathogenic"


def test_class_weight_defaults_to_one_when_unmapped():
    ds = make_dataset("dna-llm", is_sft=False)
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    assert result["class_weight"].item() == pytest.approx(1.0)


def test_class_weight_uses_map_when_present():
    ds = make_dataset("dna-llm", is_sft=False)
    ds.class_weights_map = {"pathogenic": 2.5}
    result = ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
    assert result["class_weight"].item() == pytest.approx(2.5)


def test_none_item_returns_safe_fallback():
    ds = make_dataset("dna-llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_coding(None)
    caption = render(ds, result)
    assert "<answer>No variant information available.</answer>" in caption
    # fallback ref/var = ("CTGA", "CTGA") in dna-llm mode, so there is still a
    # real pair for __iter__ to encode.
    assert result["dna_sequences"] == ["CTGA", "CTGA"]
    assert "<|im_start|>assistant" in caption  # is_sft=True still appends fallback assistant turn


def test_none_item_with_is_sft_false_omits_assistant_turn():
    ds = make_dataset("dna-llm", is_sft=False)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_coding(None))
    assert "<answer>" not in caption


def test_missing_required_field_hits_exception_fallback():
    ds = make_dataset("dna-llm", is_sft=True)
    broken_item = {"question": "test"}  # missing reference_sequence/variant_sequence/answer
    caption = render(ds, ds._process_dna_bioreason_variant_effect_coding(broken_item))
    assert "<answer>No information available.</answer>" in caption


def test_rejects_unsupported_model_name():
    ds = make_dataset("some-other-mode", is_sft=True)
    with pytest.raises(ValueError, match="Unsupported model_name"):
        ds._process_dna_bioreason_variant_effect_coding(REAL_SAMPLE_ITEM)
