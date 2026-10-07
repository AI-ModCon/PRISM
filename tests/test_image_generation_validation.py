"""Engineering fixtures only: these tests never load a real image checkpoint."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "src/eval/image_generation.py"
_SPEC = importlib.util.spec_from_file_location("image_validation_test_module", _PATH)
validation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validation
_SPEC.loader.exec_module(validation)


def fixture_case(**changes):
    row = {
        "id": "fixture",
        "task": "text_to_image",
        "prompt": "A red cube",
        "reference_paths": [],
        "group": "object-1",
        "split": "test",
        "seeds": [1, 2, 3],
    }
    row.update(changes)
    return row


def write_cases(tmp_path, rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def fixture_trace():
    return {
        "tensors": {
            name: np.zeros((1, 2), dtype=np.float32) for name in validation.PARITY_BOUNDARIES
        },
        "discrete": {"token_ids": [4, 5], "source_order": [], "output_spec": [512, 512]},
        "provenance": {"execution_path": "fixture", "real_checkpoint": False},
    }


def test_fixed_cases_aliases_and_no_target_conditioning(tmp_path):
    row = fixture_case(task="t2i", source_images=[], group_ids=["identity"])
    del row["reference_paths"], row["group"]
    cases = validation.load_cases(write_cases(tmp_path, [row]))
    assert cases[0].task == "text_to_image"
    assert cases[0].group == "identity"
    assert not validation.suite_summary(cases)["complete_prespecified_size"]
    row["target_image"] = "forbidden.png"
    with pytest.raises(validation.ValidationBlocked, match="targets"):
        validation.load_cases(write_cases(tmp_path, [row]))


def test_one_seed_is_allowed_only_for_explicit_smoke(tmp_path):
    path = write_cases(tmp_path, [fixture_case(seeds=[42])])
    with pytest.raises(validation.ValidationBlocked, match="three distinct"):
        validation.load_cases(path)
    cases = validation.load_cases(path, smoke=True)
    assert cases[0].seeds == (42,)
    assert not validation.suite_summary(cases)["complete_prespecified_size"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"seeds": [1, 1, 2]}, "three distinct"),
        ({"task": "editing"}, "source count"),
        ({"width": 31}, "multiples"),
        ({"id": "../escape"}, "identifier"),
    ],
)
def test_fixed_cases_fail_closed(tmp_path, change, match):
    with pytest.raises(validation.ValidationBlocked, match=match):
        validation.load_cases(write_cases(tmp_path, [fixture_case(**change)]))


def test_missing_assets_duplicates_and_split_groups_rejected(tmp_path):
    with pytest.raises(validation.ValidationBlocked, match="asset missing"):
        validation.load_cases(
            write_cases(tmp_path, [fixture_case(task="editing", reference_paths=["missing.png"])])
        )
    with pytest.raises(validation.ValidationBlocked, match="duplicate"):
        validation.load_cases(write_cases(tmp_path, [fixture_case(), fixture_case()]))
    with pytest.raises(validation.ValidationBlocked, match="leaks"):
        validation.load_cases(
            write_cases(tmp_path, [fixture_case(), fixture_case(id="other", split="train")])
        )


def test_archive_roundtrip_and_hash_tamper(tmp_path):
    trace = fixture_trace()
    directory = tmp_path / "trace"
    record = validation.archive_trace(directory, trace)
    restored = validation.read_trace(directory)
    assert validation.compare_traces(trace, restored)["passed"]
    tensor = directory / record["tensors"]["latents.initial"]["path"]
    tensor.write_bytes(b"corrupted")
    with pytest.raises(validation.ValidationBlocked, match="hash/path"):
        validation.read_trace(directory)


def test_comparator_requires_all_boundaries_and_exact_discrete():
    left, right = fixture_trace(), fixture_trace()
    del right["tensors"]["prediction.step0"]
    right["discrete"]["token_ids"] = [4, 6]
    result = validation.compare_traces(left, right)
    assert not result["passed"]
    assert any("discrete" in failure for failure in result["failures"])
    assert any("prediction.step0" in failure for failure in result["failures"])


def test_tolerances_require_repeatability_and_never_relax_masks_or_noise():
    left, right = fixture_trace(), fixture_trace()
    right["tensors"]["condition.positive"][0, 0] = 1e-6
    assert not validation.compare_traces(left, right)["passed"]
    tolerance = {"condition.positive": {"atol": 1e-5, "rtol": 1e-4}}
    with pytest.raises(validation.ValidationBlocked, match="repeatability"):
        validation.compare_traces(left, right, tolerance)
    tolerance["condition.positive"].update(
        rationale="fixture registered before run", repeatability_artifact="fixture.json"
    )
    assert validation.compare_traces(left, right, tolerance)["passed"]
    tolerance["latents.initial"] = tolerance["condition.positive"]
    with pytest.raises(validation.ValidationBlocked, match="must be exact"):
        validation.compare_traces(left, right, tolerance)


def test_nonfinite_and_dtype_fail_parity():
    left, right = fixture_trace(), fixture_trace()
    right["tensors"]["pixels"][0, 0] = np.nan
    right["tensors"]["latents.final"] = np.zeros((1, 2), dtype=np.float64)
    result = validation.compare_traces(left, right)
    assert not result["passed"]
    assert any("nonfinite" in error for error in result["failures"])
    assert any("dtype" in error for error in result["failures"])


def test_cluster_bootstrap_does_not_treat_seeds_as_independent():
    pairs = [("subject-a", 1, 0), ("subject-a", 1, 0), ("subject-b", 0, 1), ("subject-b", 0, 1)]
    result = validation.clustered_paired_interval(pairs, bootstrap_samples=100, seed=5)
    assert result["clusters"] == 2
    assert result["difference"] == 0
    assert result["lower_one_sided_95"] == -1
    assert result == validation.clustered_paired_interval(pairs, bootstrap_samples=100, seed=5)
    with pytest.raises(validation.ValidationBlocked, match="two independent"):
        validation.clustered_paired_interval(pairs[:2])


def test_partial_evaluation_and_failed_outputs_cannot_qualify(tmp_path):
    cases = validation.load_cases(write_cases(tmp_path, [fixture_case()]))
    evaluation = {
        "evaluator_revision": "judge-v1",
        "records": [
            {
                "case_id": "fixture",
                "seed": 1,
                "training_seed": 7,
                "variant": "full",
                "failed": True,
                "scores": {"success": 1},
            }
        ],
    }
    protocol = {
        "registered_at": "before-results",
        "protocol_id": "fixture",
        "evaluator_revision": "judge-v1",
        "max_statistical_looks": 1,
        "training_seeds": [7, 8],
        "necessary_ablations": {task: ["no_instruction"] for task in validation.TASKS},
    }
    result = validation.evaluate_qualification(cases, evaluation, protocol)
    assert result["status"] == "unproven"
    assert any("200 cases" in error for error in result["failures"])
    assert any("zero success" in error for error in result["failures"])
    assert any("missing score" in error for error in result["failures"])


def test_preflight_is_metadata_only_and_missing_cache_blocks(tmp_path):
    before = set(sys.modules)
    result = validation.environment_preflight(tmp_path / "missing", "main", None, "branch")
    assert result["status"] == "blocked"
    assert result["model_loaded"] is False
    assert result["downloads_allowed"] is False
    assert "torch" not in set(sys.modules) - before
    assert any("40-character" in error for error in result["errors"])


def test_complete_synthetic_scores_exercise_paired_gates_and_missing_row_rejection():
    """Scorer arithmetic only: fabricated scores are not a real qualification run."""
    cases, records = [], []
    for task in validation.TASKS:
        for index in range(200):
            case = validation.ImageValidationCase(
                f"{task}-{index}",
                task,
                "fixture",
                (),
                f"{task}-subject-{index // 10}",
                "sealed",
                (1, 2, 3),
                "natural" if index % 2 else "procedural",
            )
            cases.append(case)
            for seed in case.seeds:
                for training_seed in (7, 8):
                    for variant in ("full", "reference", "native_only", "no_instruction"):
                        value = 1.0 if variant in ("full", "reference") else 0.5
                        records.append(
                            {
                                "case_id": case.case_id,
                                "seed": seed,
                                "training_seed": training_seed,
                                "variant": variant,
                                "scores": {"success": value, "preservation": 1.0, "identity": 1.0},
                            }
                        )
    protocol = {
        "registered_at": "fixture-before-results",
        "protocol_id": "fixture-only",
        "evaluator_revision": "fixture-judge",
        "max_statistical_looks": 1,
        "training_seeds": [7, 8],
        "bootstrap_samples": 100,
        "necessary_ablations": {task: ["no_instruction"] for task in validation.TASKS},
        "auxiliary_gates": {"editing": {"preservation": 0.8}, "multi_reference": {"identity": 0.8}},
    }
    evaluation = {"evaluator_revision": "fixture-judge", "records": records}
    result = validation.evaluate_qualification(cases, evaluation, protocol)
    assert result["status"] == "passed", result["failures"]
    assert result["tasks"]["editing"]["comparisons"]["reference"]["clusters"] == 20
    assert set(result["tasks"]["editing"]["strata"]) == {"natural", "procedural"}
    records.pop()
    assert validation.evaluate_qualification(cases, evaluation, protocol)["status"] == "unproven"
