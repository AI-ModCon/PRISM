"""Step 2 of the variant_effect dataset integration: answer-cleaning unit tests.

Tests clean_variant_effect_coding_answer and clean_variant_effect_non_snv_answer
in src/data/multimodal.py against real sample answers pulled directly from
wanglab/variant_effect_coding and wanglab/variant_effect_non_snv on HuggingFace
Hub (verified interactively; see PR/commit description for the exact commands).

Distribution facts these tests encode, confirmed against 1000 real rows of each
dataset (2026-07-01):
  - variant_effect_coding: 1000/1000 rows have a ';' separating pathogenicity
    from the free-text description. No exceptions found.
  - variant_effect_non_snv: 228/1000 rows are a bare pathogenicity string with
    no ';' (e.g. "benign"); 772/1000 have "pathogenicity; ['term', ...]"; 0/1000
    have a ';' without a matching bracket list.

Requires torch (multimodal.py imports it at module level), so this only runs
under an environment where torch actually imports (e.g. the cluster's prism
conda env) — same constraint as tests/multimodal/test_modality_contracts.py.
"""

import pytest

torch = pytest.importorskip("torch")

from src.data.multimodal import (
    clean_variant_effect_coding_answer,
    clean_variant_effect_non_snv_answer,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


# Real samples pulled from wanglab/variant_effect_coding train split (streaming,
# first 8 rows) on 2026-07-01.
CODING_REAL_SAMPLES = [
    ("Pathogenic; Renal tubular epithelial cell apoptosis", "pathogenic"),
    ("Pathogenic; Neutrophil inclusion bodies", "pathogenic"),
    ("Pathogenic; Congenital myasthenic syndrome 8", "pathogenic"),
    (
        "Pathogenic; Combined immunodeficiency due to OX40 deficiency",
        "pathogenic",
    ),
    (
        "Pathogenic; Ehlers-Danlos syndrome, spondylodysplastic type, 2",
        "pathogenic",
    ),
    (
        "Pathogenic; Spondyloepimetaphyseal dysplasia with joint laxity",
        "pathogenic",
    ),
]

# Real samples pulled from wanglab/variant_effect_non_snv train split (streaming,
# first 8 rows) on 2026-07-01.
NON_SNV_REAL_SAMPLES = [
    (
        "pathogenic; ['Congenital_myasthenic_syndrome', 'Congenital_myasthenic_syndrome_8']",
        "pathogenic; Congenital myasthenic syndrome, Congenital myasthenic syndrome 8",
    ),
    (
        "pathogenic; ['Congenital_myasthenic_syndrome_8']",
        "pathogenic; Congenital myasthenic syndrome 8",
    ),
    (
        "pathogenic; ['Congenital_myasthenic_syndrome_8', 'Presynaptic_congenital_myasthenic_syndrome']",
        "pathogenic; Congenital myasthenic syndrome 8, Presynaptic congenital myasthenic syndrome",
    ),
    ("benign", "benign"),
    (
        "pathogenic; ['Autosomal_dominant_Robinow_syndrome_2']",
        "pathogenic; Autosomal dominant Robinow syndrome 2",
    ),
]


@pytest.mark.parametrize("raw,expected", CODING_REAL_SAMPLES)
def test_clean_variant_effect_coding_answer_real_samples(raw, expected):
    assert clean_variant_effect_coding_answer(raw) == expected


def test_clean_variant_effect_coding_answer_is_idempotent():
    # Cleaning an already-clean answer should be a no-op (guards against
    # accidentally double-cleaning if this function is ever called twice on
    # the same field).
    once = clean_variant_effect_coding_answer("Pathogenic; Some disease")
    twice = clean_variant_effect_coding_answer(once)
    assert once == twice == "pathogenic"


@pytest.mark.parametrize("raw,expected", NON_SNV_REAL_SAMPLES)
def test_clean_variant_effect_non_snv_answer_real_samples(raw, expected):
    assert clean_variant_effect_non_snv_answer(raw) == expected


def test_clean_variant_effect_non_snv_answer_bare_pathogenicity_is_noop():
    # 228/1000 real rows have no ';' or brackets at all (e.g. "benign").
    # The cleaner must not mangle these.
    assert clean_variant_effect_non_snv_answer("benign") == "benign"
    assert clean_variant_effect_non_snv_answer("pathogenic") == "pathogenic"


def test_clean_variant_effect_non_snv_answer_strips_all_brackets_and_quotes():
    raw = "benign; ['Some_term', 'Another_term']"
    cleaned = clean_variant_effect_non_snv_answer(raw)
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "'" not in cleaned
    assert "_" not in cleaned
    assert cleaned == "benign; Some term, Another term"
