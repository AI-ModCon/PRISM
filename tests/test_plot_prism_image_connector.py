"""Loss-cohort integrity and scientific artifact rendering tests."""

import copy
import json

import pytest
from PIL import Image
from tools.plot_prism_image_connector import _figure_labels, collect_series, main, plot_series


def _detail(split, losses, *, step, gap=0.2):
    examples = [
        {
            "id": f"{split}-{index}",
            "seed": 1000 + index,
            "shuffled_prompt_id": f"wrong-{split}-{index}",
            "loss": loss,
            "shuffled_loss": loss + gap,
            "shuffled_minus_correct": gap,
            "optimized_before_evaluation": split == "train" and step > 0,
        }
        for index, loss in enumerate(losses)
    ]
    return {
        "count": len(examples),
        "examples": examples,
        "mean_loss": sum(losses) / len(losses),
        "mean_shuffled_loss": sum(row["shuffled_loss"] for row in examples) / len(examples),
        "mean_shuffled_minus_correct": gap,
    }


def _evaluation(step, *, full=False, final=False, validation_losses=None):
    return {
        "step": step,
        "final": final,
        "full_validation": full,
        "splits": {
            "train": _detail("train", [1.2, 2.2], step=step),
            "validation": _detail("validation", validation_losses or [2.0, 4.0], step=step),
        },
    }


def _write_run(root, steps, evaluations, *, protocol=None, initial=None):
    root.mkdir()
    initial = initial or _evaluation(0)
    report = {
        "evidence_kind": "fixture_only",
        "qualification": "unqualified",
        "status": "completed",
        "train_count": 4,
        "validation_count": 3,
        "data_fingerprint": "dataset-fingerprint",
        "completed_steps": max(row["step"] for row in steps),
        "resume_protocol": protocol or {"parent": "fixture", "seed": 42},
        "initial_evaluation": initial,
    }
    (root / "report.json").write_text(json.dumps(report))
    (root / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps))
    (root / "evaluations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evaluations))
    return root


def _steps(*numbers):
    return [
        {
            "step": step,
            "loss": 5 / step,
            "examples_seen": 4 * step,
            "microbatches": [{"ids": [f"train-{index}" for index in range(4)]}],
        }
        for step in numbers
    ]


@pytest.fixture
def pilot(tmp_path):
    initial = _evaluation(0)
    final = _evaluation(2, full=True, final=True, validation_losses=[1.0, 2.0, 90.0])
    return _write_run(tmp_path / "run", _steps(1, 2), [initial, final])


def test_final_full_validation_does_not_change_fixed_cohort_mean(pilot):
    series = collect_series([pilot])
    assert series["fixed_probes"]["validation"][-1]["count"] == 2
    assert series["fixed_probes"]["validation"][-1]["mean_loss"] == 1.5
    assert series["full_validation"][-1]["count"] == 3
    assert series["full_validation"][-1]["mean_loss"] == 31.0
    assert [row["step"] for row in series["fixed_probes"]["validation"]] == [0, 2]
    assert series["coverage"] == {
        "first_logged_step": 1,
        "last_logged_step": 2,
        "logged_optimizer_steps": 2,
        "cumulative_examples_seen": 8,
        "observed_unique_train_ids": 4,
    }
    assert series["fixed_probes"]["train"][0]["optimized_count"] == 0
    assert series["fixed_probes"]["train"][-1]["optimized_count"] == 2
    assert len(series["source_sha256"]) == 3


def test_joint_training_scope_is_preserved_in_series_and_output_names(pilot, tmp_path):
    pytest.importorskip("matplotlib")
    path = pilot / "report.json"
    report = json.loads(path.read_text())
    report["evidence_kind"] = "real_checkpoint_connector_diffusion_webdataset_pilot"
    path.write_text(json.dumps(report))
    series = collect_series([pilot])
    assert series["training_scope"] == "connector_and_diffusion"
    result = plot_series(series, tmp_path / "joint-plots", dpi=100)
    assert result["png"].endswith("connector-diffusion-losses.png")
    assert result["series"].endswith("connector-diffusion-loss-series.json")


def test_smoke_final_subset_is_separate_and_does_not_splice_curve(tmp_path):
    root = _write_run(
        tmp_path / "smoke", _steps(1), [_evaluation(1, final=True, validation_losses=[5.0])]
    )
    series = collect_series([root])
    assert [row["step"] for row in series["fixed_probes"]["validation"]] == [0]
    assert series["alternate_final_validation"][0]["count"] == 1
    assert series["missing_fixed_cohorts"] == [
        {"step": 1, "split": "validation", "expected": 2, "present": 1}
    ]


def test_resume_chain_preserves_global_steps_and_origin_baseline(tmp_path):
    first = _write_run(tmp_path / "first", _steps(1, 2), [_evaluation(0), _evaluation(2)])
    resume = _write_run(
        tmp_path / "resume",
        _steps(3, 4),
        [_evaluation(4, full=True, final=True, validation_losses=[1.0, 2.0, 3.0])],
    )
    combined = collect_series([first, resume])
    assert [row["step"] for row in combined["optimizer"]] == [1, 2, 3, 4]
    assert [row["step"] for row in combined["fixed_probes"]["train"]] == [0, 2, 4]
    assert combined["coverage"]["cumulative_examples_seen"] == 16
    partial = collect_series([resume])
    assert partial["coverage"]["first_logged_step"] == 3
    assert partial["coverage"]["cumulative_examples_seen"] == 16


def test_different_experiments_cannot_be_combined(tmp_path, pilot):
    other = _write_run(
        tmp_path / "other", _steps(3), [_evaluation(3)], protocol={"parent": "different"}
    )
    with pytest.raises(ValueError, match="protocols differ"):
        collect_series([pilot, other])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("seed", "noise/control changed"),
        ("wrong_caption", "noise/control changed"),
        ("nan", "Nonfinite"),
        ("aggregate", "aggregate"),
        ("full_count", "Full-validation count"),
    ],
)
def test_invalid_or_incomparable_measurements_fail(tmp_path, mutation, match):
    final = _evaluation(2, full=True, final=True, validation_losses=[1.0, 2.0, 3.0])
    if mutation == "seed":
        final["splits"]["validation"]["examples"][0]["seed"] += 1
    elif mutation == "wrong_caption":
        final["splits"]["validation"]["examples"][0]["shuffled_prompt_id"] = "changed-caption"
    elif mutation == "nan":
        final["splits"]["validation"]["examples"][0]["loss"] = float("nan")
    elif mutation == "aggregate":
        final["splits"]["validation"]["mean_loss"] = 99.0
    else:
        final["splits"]["validation"] = _detail("validation", [1.0, 2.0], step=2)
    root = _write_run(tmp_path / "bad", _steps(1, 2), [final])
    with pytest.raises(ValueError, match=match):
        collect_series([root])


def test_standalone_png_pdf_and_machine_readable_series_render(pilot, tmp_path):
    pytest.importorskip("matplotlib")
    series = collect_series([pilot])
    original = copy.deepcopy(series)
    result = plot_series(series, tmp_path / "plots", rolling_window=2, dpi=100)
    assert series == original
    with Image.open(result["png"]) as image:
        assert image.size == (1000, 1050)
        assert image.getextrema()[0][0] < 50  # The plot is not an empty white page.
    assert (tmp_path / "plots/connector-losses.pdf").read_bytes().startswith(b"%PDF")
    saved = json.loads((tmp_path / "plots/connector-loss-series.json").read_text())
    assert saved["full_validation"][-1]["mean_loss"] == 31.0
    assert saved["qualification"] == "unqualified"
    assert set(result) == {"png", "pdf", "series"}


def test_cli_accepts_repeated_run_directories(pilot, tmp_path):
    pytest.importorskip("matplotlib")
    assert (
        main(
            [
                "--run-dir",
                str(pilot),
                "--run-dir",
                str(pilot),
                "--output-dir",
                str(tmp_path / "cli"),
                "--dpi",
                "72",
            ]
        )
        == 0
    )


def test_partial_history_labels_do_not_confuse_exposures_with_unique_ids(tmp_path):
    root = _write_run(tmp_path / "partial", _steps(195, 196), [_evaluation(192)])
    path = root / "report.json"
    report = json.loads(path.read_text())
    report.update(train_count=9647, selected_train_count=32, status="running")
    path.write_text(json.dumps(report))
    series = collect_series([root])
    _, _, exposure, pool = _figure_labels(series)
    assert exposure == "Cumulative training exposures: 784 | Unique IDs in shown logs: 4"
    assert pool == "Selected training pool: 32 | Full training pool: 9,647"
    assert "unverified" in series["interpretation"]["gap"]


def test_running_joint_plot_text_fits_and_marks_controls_unverified(pilot, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    series = collect_series([pilot])
    series["training_scope"] = "connector_and_diffusion"
    series["runs"][-1].update(
        status="running", evidence_kind="real_checkpoint_connector_diffusion_webdataset_pilot"
    )
    series.update(train_count=9647, selected_train_count=32)
    series["coverage"].update(cumulative_examples_seen=784, observed_unique_train_ids=32)
    original = Figure.savefig
    observed = []

    def savefig(figure, *args, **kwargs):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = [*figure.texts, *(axis._left_title for axis in figure.axes)]
        for text in texts:
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.width
        observed.extend(text.get_text() for text in texts)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    plot_series(series, tmp_path / "running-joint", dpi=180)
    assert "C  RNG-reset caption control — paired inputs unverified" in observed
    assert "Cumulative training exposures: 784 | Unique IDs in shown logs: 32" in observed
    assert not any("identical target, timestep and noise" in text for text in observed)
