"""Step 1 of the variant_effect dataset integration: config-only tests.

Verifies datasets_config.json registers dna_bioreason (KEGG),
dna_bioreason_variant_effect_coding, and dna_bioreason_variant_effect_non_snv
correctly, and that the "dna" modality-group sampling weights normalize the
way src/data/multimodal.py's StreamingMultimodalDataset.__iter__ computes them
(see the "Weighted Sampling Setup (Hierarchical)" block, ~multimodal.py:2564).

No network access and no StreamingMultimodalDataset instantiation here —
that requires HF downloads / a DNA tokenizer and is covered by later steps.
"""

import pytest
from src.data.dataset_manager import DatasetManager

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

DNA_HANDLER_KEYS = {
    "dna_bioreason": "wanglab/kegg",
    "dna_bioreason_variant_effect_coding": "wanglab/variant_effect_coding",
    "dna_bioreason_variant_effect_non_snv": "wanglab/variant_effect_non_snv",
}


@pytest.fixture(scope="module")
def zone_a_datasets():
    manager = DatasetManager()
    return manager.get_zone_config("zone_a")["datasets"]


def test_all_three_dna_datasets_registered(zone_a_datasets):
    for key in DNA_HANDLER_KEYS:
        assert key in zone_a_datasets, f"{key} missing from zone_a datasets"


def test_hf_ids_match_expected_repos(zone_a_datasets):
    for key, expected_hf_id in DNA_HANDLER_KEYS.items():
        assert zone_a_datasets[key]["hf_id"] == expected_hf_id


def test_handler_field_matches_key(zone_a_datasets):
    # multimodal.py dispatches on handler_key.startswith("dna_bioreason") and
    # routes to _process_dna_bioreason / _process_dna_bioreason_variant_effect_coding
    # / _process_dna_bioreason_variant_effect_non_snv based on the exact handler
    # string, so handler must equal the dataset's own key (no typos/mismatches).
    for key in DNA_HANDLER_KEYS:
        assert zone_a_datasets[key]["handler"] == key


def test_none_are_skipped(zone_a_datasets):
    for key in DNA_HANDLER_KEYS:
        assert zone_a_datasets[key]["skip"] is False, f"{key} is marked skip=true"


def test_all_dna_handlers_contain_dna_substring(zone_a_datasets):
    # multimodal.py's modality-group inference does `elif "dna" in handler`
    # (see the Weighted Sampling Setup block) — every DNA dataset's handler
    # string must contain "dna" or it will silently fall through to the
    # "image" default modality group instead of "dna".
    for key in DNA_HANDLER_KEYS:
        assert "dna" in zone_a_datasets[key]["handler"]


def test_required_keys_match_known_hf_schema(zone_a_datasets):
    # Confirmed directly against the real HF datasets (see conversation):
    # wanglab/kegg train: question, answer, reasoning, reference_sequence, variant_sequence
    # wanglab/variant_effect_coding train: ID, question, answer, reference_sequence, variant_sequence
    # wanglab/variant_effect_non_snv train: question, answer, reference_sequence,
    #   mutated_sequence, cleaned_pathogenicity, __index_level_0__
    assert set(zone_a_datasets["dna_bioreason"]["required_keys"]) == {
        "question", "answer", "reasoning", "reference_sequence", "variant_sequence",
    }
    assert set(zone_a_datasets["dna_bioreason_variant_effect_coding"]["required_keys"]) == {
        "question", "answer", "reference_sequence", "variant_sequence",
    }
    assert set(zone_a_datasets["dna_bioreason_variant_effect_non_snv"]["required_keys"]) == {
        "question", "answer", "reference_sequence", "mutated_sequence",
    }


def test_weights_sum_to_one_and_match_kegg_majority_split(zone_a_datasets):
    weights = {key: zone_a_datasets[key]["weight"] for key in DNA_HANDLER_KEYS}
    assert weights["dna_bioreason"] == pytest.approx(0.50)
    assert weights["dna_bioreason_variant_effect_coding"] == pytest.approx(0.25)
    assert weights["dna_bioreason_variant_effect_non_snv"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_modality_group_normalization_matches_multimodal_py():
    """Reproduces the exact grouping/normalization arithmetic from
    StreamingMultimodalDataset.__iter__'s "Weighted Sampling Setup (Hierarchical)"
    block (src/data/multimodal.py ~2564-2610), using the real zone_a config,
    to catch any drift between this test's assumptions and the real code path.
    """
    manager = DatasetManager()
    datasets_map = manager.get_zone_config("zone_a")["datasets"]

    modality_groups: dict[str, list[tuple[str, float]]] = {
        "image": [], "graph": [], "table": [], "time_series": [], "geometry": [], "dna": [],
    }
    for name, info in datasets_map.items():
        if info.get("skip", False):
            continue
        handler = info["handler"]
        w = info.get("weight", 1.0)
        target = "image"
        if "image" in handler:
            target = "image"
        elif "graph" in handler:
            target = "graph"
        elif "table" in handler:
            target = "table"
        elif "ts" in handler or "time" in handler:
            target = "time_series"
        elif "geo" in handler:
            target = "geometry"
        elif "dna" in handler:
            target = "dna"
        modality_groups[target].append((name, w))

    dna_group = dict(modality_groups["dna"])
    assert set(dna_group.keys()) == set(DNA_HANDLER_KEYS.keys())

    total = sum(dna_group.values())
    normalized = {k: v / total for k, v in dna_group.items()}
    assert normalized["dna_bioreason"] == pytest.approx(0.50)
    assert normalized["dna_bioreason_variant_effect_coding"] == pytest.approx(0.25)
    assert normalized["dna_bioreason_variant_effect_non_snv"] == pytest.approx(0.25)
