"""Step 4 of the variant_effect dataset integration: handler-level tests for
_process_dna_bioreason_variant_effect_non_snv in src/data/multimodal.py.

Same object.__new__(StreamingMultimodalDataset) approach, structured-dict
return contract, and _FakeDnaTokenizer stand-in as
test_variant_effect_coding_handler.py — see that file's module docstring for
the rationale, and for how the old (data_tensor, caption, metadata_str) tuple
slots map onto today's {"prompt", "dna_sequences", "answer", "class_weight"}.

Requires torch (multimodal.py imports it at module level) — run under the
cluster's prism conda env.
"""

import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.data.multimodal import StreamingMultimodalDataset

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

# Real sample row shape confirmed against wanglab/variant_effect_non_snv train
# split (2026-07-01): question, answer, reference_sequence, mutated_sequence,
# cleaned_pathogenicity, __index_level_0__. Note the field is "mutated_sequence",
# not "variant_sequence".
REAL_SAMPLE_ITEM_WITH_TERMS = {
    "question": "Mutation found at chromosome 1 position 1040717, gene AGRN (agrin): benign or pathogenic?",
    "answer": "pathogenic; ['Congenital_myasthenic_syndrome', 'Congenital_myasthenic_syndrome_8']",
    "reference_sequence": "ACGT" * 10,
    "mutated_sequence": "ACGA" * 10,
    "cleaned_pathogenicity": "pathogenic",
    "__index_level_0__": 67,
}

# 228/1000 real rows sampled have this bare shape (no ';', no brackets).
REAL_SAMPLE_ITEM_BARE = {
    "question": "Is this variant benign or pathogenic?",
    "answer": "benign",
    "reference_sequence": "ACGT" * 10,
    "mutated_sequence": "ACGT" * 10,
    "cleaned_pathogenicity": "benign",
    "__index_level_0__": 12,
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


def test_reads_mutated_sequence_field_not_variant_sequence():
    ds = make_dataset("dna-llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_WITH_TERMS)
    # The variant half must come from "mutated_sequence"; this row has no
    # "variant_sequence" key at all, so reading the wrong name would raise and
    # drop us into the exception fallback (["CTGA", "CTGA"]).
    assert result["dna_sequences"] == [
        REAL_SAMPLE_ITEM_WITH_TERMS["reference_sequence"],
        REAL_SAMPLE_ITEM_WITH_TERMS["mutated_sequence"],
    ]


def test_answer_with_bracket_list_is_cleaned():
    ds = make_dataset("dna-llm", is_sft=True)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_WITH_TERMS))
    assert (
        "<answer>pathogenic; Congenital myasthenic syndrome, "
        "Congenital myasthenic syndrome 8</answer>" in caption
    )
    answer_text = caption.split("<answer>")[1].split("</answer>")[0]
    assert "[" not in answer_text
    assert "'" not in answer_text
    assert "_" not in answer_text


def test_bare_pathogenicity_answer_is_unchanged_noop():
    ds = make_dataset("dna-llm", is_sft=True)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_BARE))
    assert "<answer>benign</answer>" in caption


@pytest.mark.parametrize("model_name", ["llm", "dna-llm"])
def test_llm_vs_dna_llm_mode(model_name):
    ds = make_dataset(model_name, is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_non_snv(
        REAL_SAMPLE_ITEM_WITH_TERMS
    )
    caption = render(ds, result)
    if model_name == "llm":
        # Sequences are inlined into the text, so there is no separate DNA
        # modality left for __iter__ to tokenize.
        assert result["dna_sequences"] == ["", ""]
        assert REAL_SAMPLE_ITEM_WITH_TERMS["reference_sequence"] in caption
        assert REAL_SAMPLE_ITEM_WITH_TERMS["mutated_sequence"] in caption
    else:
        assert result["dna_sequences"] == [
            REAL_SAMPLE_ITEM_WITH_TERMS["reference_sequence"],
            REAL_SAMPLE_ITEM_WITH_TERMS["mutated_sequence"],
        ]
        assert REAL_SAMPLE_ITEM_WITH_TERMS["reference_sequence"] not in caption


def test_sft_true_appends_answer_tagged_assistant_turn_with_no_reasoning():
    ds = make_dataset("dna-llm", is_sft=True)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_WITH_TERMS))
    assert "<think>" not in caption
    assert (
        "<answer>pathogenic; Congenital myasthenic syndrome, "
        "Congenital myasthenic syndrome 8</answer>" in caption
    )


def test_sft_false_omits_assistant_turn_for_grpo():
    ds = make_dataset("dna-llm", is_sft=False)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_WITH_TERMS))
    # The generation marker is still emitted (the renderer always opens the
    # assistant turn); what is absent is the answer the model must produce.
    assert "<|im_start|>assistant" in caption
    assert "<answer>" not in caption


def test_class_weight_defaults_to_one_when_unmapped():
    ds = make_dataset("dna-llm", is_sft=False)
    result = ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_BARE)
    assert result["class_weight"].item() == pytest.approx(1.0)


def test_class_weight_uses_map_when_present():
    ds = make_dataset("dna-llm", is_sft=False)
    ds.class_weights_map = {"benign": 3.0}
    result = ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_BARE)
    assert result["class_weight"].item() == pytest.approx(3.0)


def test_none_item_returns_safe_fallback():
    ds = make_dataset("dna-llm", is_sft=True)
    result = ds._process_dna_bioreason_variant_effect_non_snv(None)
    caption = render(ds, result)
    assert "<answer>No variant information available.</answer>" in caption
    # dna-llm still needs a well-formed pair for __iter__ to tokenize.
    assert result["dna_sequences"] == ["CTGA", "CTGA"]
    assert "<|im_start|>assistant" in caption


def test_none_item_with_is_sft_false_omits_assistant_turn():
    ds = make_dataset("dna-llm", is_sft=False)
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(None))
    assert "<answer>" not in caption


def test_missing_required_field_hits_exception_fallback():
    ds = make_dataset("dna-llm", is_sft=True)
    # Missing mutated_sequence (and not variant_sequence, confirming we don't
    # accidentally fall back to reading the wrong key name).
    broken_item = {"question": "test", "reference_sequence": "ACGT", "answer": "benign"}
    caption = render(ds, ds._process_dna_bioreason_variant_effect_non_snv(broken_item))
    assert "<answer>No information available.</answer>" in caption


def test_rejects_unsupported_model_name():
    ds = make_dataset("some-other-mode", is_sft=True)
    with pytest.raises(ValueError, match="Unsupported model_name"):
        ds._process_dna_bioreason_variant_effect_non_snv(REAL_SAMPLE_ITEM_WITH_TERMS)
