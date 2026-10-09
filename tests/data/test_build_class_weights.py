"""Step 5 of the variant_effect dataset integration: tests for the generalized
_build_class_weights in src/data/multimodal.py, which now merges frequency
counts across dna_bioreason (KEGG), dna_bioreason_variant_effect_coding, and
dna_bioreason_variant_effect_non_snv into one shared class_weights_map.

_count_kegg_answers and _count_streamed_answers (the two new helper methods
that do the actual HF loading) are mocked out here so these tests exercise the
merge/fault-isolation/weight-math logic without touching the network — real
HF loading is covered by the live smoke test in a later step.

Requires torch (multimodal.py imports it at module level) — run under the
cluster's prism conda env.
"""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from src.data.multimodal import StreamingMultimodalDataset

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]


def make_dataset(datasets_map: dict) -> StreamingMultimodalDataset:
    ds = object.__new__(StreamingMultimodalDataset)
    ds.datasets_map = datasets_map
    ds.class_weights_map = {}
    return ds


ALL_THREE_ACTIVE = {
    "dna_bioreason": {"skip": False},
    "dna_bioreason_variant_effect_coding": {"skip": False},
    "dna_bioreason_variant_effect_non_snv": {"skip": False},
}


def test_merges_counts_across_all_three_datasets():
    ds = make_dataset(ALL_THREE_ACTIVE)

    def fake_kegg(self, counts):
        counts["disease a"] += 3
        counts["disease b"] += 1

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        if hf_id == "wanglab/variant_effect_coding":
            counts["pathogenic"] += 4
            counts["benign"] += 4
        elif hf_id == "wanglab/variant_effect_non_snv":
            counts["pathogenic; term x"] += 2

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0)

    # All 5 distinct labels across all 3 datasets should be present in one map.
    assert set(ds.class_weights_map.keys()) == {
        "disease a", "disease b", "pathogenic", "benign", "pathogenic; term x",
    }
    # N=14, C=5 -> w = 14/(5*n). Rarest classes (n=1 or n=2) get the highest weight.
    N, C = 14, 5
    assert ds.class_weights_map["disease b"] == pytest.approx(min(N / (C * 1), 10.0))
    assert ds.class_weights_map["pathogenic; term x"] == pytest.approx(min(N / (C * 2), 10.0))
    assert ds.class_weights_map["disease a"] == pytest.approx(min(N / (C * 3), 10.0))


def test_weight_max_caps_extreme_weights():
    ds = make_dataset(ALL_THREE_ACTIVE)

    def fake_kegg(self, counts):
        counts["rare"] += 1
        counts["common"] += 1000

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        pass

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=5.0)

    assert ds.class_weights_map["rare"] == 5.0  # capped, would be 1001/(2*1) uncapped
    assert ds.class_weights_map["common"] < 5.0


def test_skipped_datasets_are_not_counted():
    datasets_map = {
        "dna_bioreason": {"skip": False},
        "dna_bioreason_variant_effect_coding": {"skip": True},  # skipped
        "dna_bioreason_variant_effect_non_snv": {"skip": False},
    }
    ds = make_dataset(datasets_map)

    called_hf_ids = []

    def fake_kegg(self, counts):
        counts["disease a"] += 1

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        called_hf_ids.append(hf_id)
        counts["x"] += 1

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0)

    assert called_hf_ids == ["wanglab/variant_effect_non_snv"]


def test_missing_dataset_entries_are_not_counted():
    # zone config might not even have the new datasets registered (e.g. an
    # older/partial config) — must not KeyError, just skip them.
    datasets_map = {"dna_bioreason": {"skip": False}}
    ds = make_dataset(datasets_map)

    def fake_kegg(self, counts):
        counts["disease a"] += 1

    with patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg):
        ds._build_class_weights(weight_max=10.0)

    assert ds.class_weights_map == {"disease a": 1.0}


def test_one_dataset_failing_does_not_wipe_out_others():
    ds = make_dataset(ALL_THREE_ACTIVE)

    def fake_kegg(self, counts):
        counts["disease a"] += 5

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        if hf_id == "wanglab/variant_effect_coding":
            raise RuntimeError("simulated network failure")
        counts["pathogenic; term x"] += 1

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0)

    # KEGG succeeded and non_snv succeeded; only coding's contribution is missing.
    assert "disease a" in ds.class_weights_map
    assert "pathogenic; term x" in ds.class_weights_map
    assert len(ds.class_weights_map) == 2


def test_kegg_failure_does_not_wipe_out_variant_effect_counts():
    ds = make_dataset(ALL_THREE_ACTIVE)

    def fake_kegg(self, counts):
        raise RuntimeError("simulated KEGG load failure")

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        counts["pathogenic"] += 1

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0)

    assert ds.class_weights_map  # not empty — variant_effect counts survived
    assert "pathogenic" in ds.class_weights_map


def test_all_datasets_failing_falls_back_to_uniform_empty_map():
    ds = make_dataset(ALL_THREE_ACTIVE)

    def fake_kegg(self, counts):
        raise RuntimeError("fail")

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        raise RuntimeError("fail")

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0)

    assert ds.class_weights_map == {}


def test_streamed_cap_is_passed_through_to_helper():
    ds = make_dataset(ALL_THREE_ACTIVE)
    seen_caps = []

    def fake_kegg(self, counts):
        counts["x"] += 1

    def fake_streamed(self, counts, hf_id, clean_fn, max_rows):
        seen_caps.append(max_rows)

    with (
        patch.object(StreamingMultimodalDataset, "_count_kegg_answers", fake_kegg),
        patch.object(StreamingMultimodalDataset, "_count_streamed_answers", fake_streamed),
    ):
        ds._build_class_weights(weight_max=10.0, streamed_cap=123)

    assert seen_caps == [123, 123]
