"""Regression checks for documentation validation and advisory URL checks."""

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_doc_links", ROOT / "tools/ci/check_doc_links.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_script_link_fails(tmp_path, monkeypatch, capsys):
    checker = _checker()
    page = tmp_path / "timeseries.md"
    page.write_text("[Analysis](scripts/perlmutter/analyze_ts_data.py)\n")
    monkeypatch.setattr(checker, "tracked_markdown", lambda: iter([str(page)]))

    assert checker.main() == 1
    assert "analyze_ts_data.py" in capsys.readouterr().out


def test_existing_relative_link_passes(tmp_path, monkeypatch):
    checker = _checker()
    page = tmp_path / "index.md"
    page.write_text("[Guide](guide.md)\n")
    (tmp_path / "guide.md").write_text("# Guide\n")
    monkeypatch.setattr(checker, "tracked_markdown", lambda: iter([str(page)]))

    assert checker.main() == 0


def test_documentation_workflow_checks_and_advisory_gate():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/pr_quality.yml").read_text()
    )
    # PyYAML's YAML 1.1 parser reads the GitHub Actions key `on` as True.
    events = workflow[True]
    assert "schedule" not in events
    dispatch = events["workflow_dispatch"]["inputs"]["check_external_urls"]
    assert dispatch["type"] == "boolean"
    assert dispatch["default"] is False

    jobs = workflow["jobs"]
    steps = jobs["doc_links"]["steps"]
    spelling = next(step for step in steps if step.get("name") == "Check spelling")
    assert spelling["with"]["files"] == "${{ steps.markdown.outputs.files }}"
    links = next(
        step for step in steps
        if step.get("name") == "Check local Markdown links and anchors"
    )
    assert "--offline" in links["with"]["args"]
    assert "--include-fragments=anchor-only" in links["with"]["args"]
    assert "--files-from" in links["with"]["args"]

    external = jobs["external_urls"]
    assert external["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.check_external_urls"
    )
    assert external["continue-on-error"] is True
    checker = external["steps"][-1]
    assert checker["continue-on-error"] is True
    assert checker["with"]["fail"] is False
