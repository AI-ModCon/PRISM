"""Pin the local-directory dispatch order in StreamingMultimodalDataset.

The original `if os.path.isdir(local_path):` block at src/data/multimodal.py:363
exited without break/continue/return, so the sibling `elif os.path.isdir(...)`
that held the CSV/JSONL branches was structurally unreachable. Every dataset
whose local_path was a directory full of CSV/JSONL files (graph_captioning,
table_reasoning, ts_qa, ts_instruction) silently fell through to the HF remote
loader — fatal on offline Aurora compute nodes.

These tests assert each table-driven case lands on the correct loader and
load_status string.
"""
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from src.data.multimodal import StreamingMultimodalDataset


@pytest.fixture
def dispatch_dir():
    """A tempdir whose path does NOT contain the substring 'test' — the
    recursive JSONL search filters `if "test" in root`, which matches pytest's
    default tmp_path. Use this fixture to avoid that filter."""
    d = tempfile.mkdtemp(prefix="prism_dispatch_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _FakeManager:
    def __init__(self, datasets_map):
        self._datasets_map = datasets_map

    def get_zone_config(self, zone):
        return {"datasets": self._datasets_map}


@dataclass
class _FakeModelConfig:
    modalities: list = field(
        default_factory=lambda: ["text", "image", "table", "time_series", "geometry", "graph"]
    )


class _LoadDatasetSpy:
    """Records every call to datasets.load_dataset and returns a stub iterable."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        # Record builder name (positional 0) and key kwargs
        self.calls.append({
            "builder": args[0] if args else kwargs.get("path"),
            "data_files": kwargs.get("data_files"),
            "split": kwargs.get("split"),
            "streaming": kwargs.get("streaming"),
        })
        return _StubDataset()


class _StubDataset:
    """Stub of a streaming IterableDataset — only needs .shuffle() to no-op."""

    def shuffle(self, *args, **kwargs):
        return self

    def to_iterable_dataset(self):
        return self


def _run_with_mocked_loader(datasets_map):
    spy = _LoadDatasetSpy()
    with patch("src.data.multimodal.DatasetManager", return_value=_FakeManager(datasets_map)):
        with patch("src.data.multimodal.load_dataset", spy):
            try:
                ds = StreamingMultimodalDataset(
                    tokenizer=None,
                    allow_dummy_data=False,
                    force_streaming=False,
                    model_config=_FakeModelConfig(),
                )
            except Exception:
                ds = None
    return spy, ds


def test_csv_directory_dispatch(dispatch_dir):
    """A directory containing only *.csv files lands on the CSV loader."""
    (dispatch_dir / "shard0.csv").write_text("col_a,col_b\n1,2\n")
    (dispatch_dir / "shard1.csv").write_text("col_a,col_b\n3,4\n")

    datasets_map = {
        "table_reasoning": {
            "name": "table_reasoning",
            "handler": "table_generic",
            "modality": "table",
            "skip": False,
            "local_path": str(dispatch_dir),
            "hf_id": "fake/should-not-be-fetched",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    csv_calls = [c for c in spy.calls if c["builder"] == "csv"]
    assert len(csv_calls) == 1, (
        f"Expected exactly one CSV load_dataset call, got {spy.calls}"
    )
    assert sorted(csv_calls[0]["data_files"]) == sorted(
        [str(dispatch_dir / "shard0.csv"), str(dispatch_dir / "shard1.csv")]
    )
    assert ds is not None
    assert ds.load_status["table_reasoning"] == "Active (Local CSV)"


def test_jsonl_directory_dispatch(dispatch_dir):
    """A directory containing only flat *.jsonl files lands on the JSON loader."""
    (dispatch_dir / "train.jsonl").write_text('{"description": "hi"}\n')
    (dispatch_dir / "more.jsonl").write_text('{"description": "ho"}\n')

    datasets_map = {
        "ts_instruction": {
            "name": "ts_instruction",
            "handler": "ts_instruction",
            "modality": "time_series",
            "skip": False,
            "local_path": str(dispatch_dir),
            "hf_id": "fake/should-not-be-fetched",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    json_calls = [c for c in spy.calls if c["builder"] == "json"]
    assert len(json_calls) == 1, (
        f"Expected exactly one JSON load_dataset call, got {spy.calls}"
    )
    assert sorted(json_calls[0]["data_files"]) == sorted(
        [str(dispatch_dir / "train.jsonl"), str(dispatch_dir / "more.jsonl")]
    )
    assert ds is not None
    assert ds.load_status["ts_instruction"] == "Active (Local JSONL)"


def test_recursive_jsonl_directory_dispatch(dispatch_dir):
    """A directory whose *.jsonl files live in a deeper non-test subdir lands
    on the recursive JSONL loader (TableInstruct/data_v3/*.json pattern)."""
    # The recursive walk filters any `root` containing "test" — guard against
    # a future fixture name change by creating a non-"test" subdir explicitly.
    root = dispatch_dir / "instruct_root"
    root.mkdir()
    nested = root / "data_v3"
    nested.mkdir()
    (nested / "a.json").write_text('{"q": "x"}\n')
    (nested / "b.json").write_text('{"q": "y"}\n')

    # Also drop a file under a filtered subdir to confirm "eval_data" exclusion.
    excluded = root / "eval_data"
    excluded.mkdir()
    (excluded / "leak.json").write_text('{"q": "should not appear"}\n')

    datasets_map = {
        "table_instruct": {
            "name": "table_instruct",
            "handler": "table_generic",
            "modality": "table",
            "skip": False,
            "local_path": str(root),
            "hf_id": "fake/should-not-be-fetched",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    json_calls = [c for c in spy.calls if c["builder"] == "json"]
    assert len(json_calls) == 1, (
        f"Expected exactly one JSON load_dataset call, got {spy.calls}"
    )
    files = json_calls[0]["data_files"]
    assert any(p.endswith("data_v3/a.json") for p in files)
    assert any(p.endswith("data_v3/b.json") for p in files)
    # The filtered subdir's file must not appear
    assert not any("eval_data/leak.json" in p for p in files), (
        f"eval_data/ subdir leaked into data_files: {files}"
    )
    assert ds is not None
    assert ds.load_status["table_instruct"] == "Active (Local JSONL Recursive)"


def test_csv_takes_priority_over_jsonl(dispatch_dir):
    """If a directory contains both CSV and JSONL files, CSV wins (preserves
    pre-fix branch order: csv > json > recursive)."""
    (dispatch_dir / "data.csv").write_text("a,b\n1,2\n")
    (dispatch_dir / "extra.jsonl").write_text('{"a": 1}\n')

    datasets_map = {
        "mixed": {
            "name": "mixed",
            "handler": "table_generic",
            "modality": "table",
            "skip": False,
            "local_path": str(dispatch_dir),
            "hf_id": "fake/should-not-be-fetched",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    csv_calls = [c for c in spy.calls if c["builder"] == "csv"]
    json_calls = [c for c in spy.calls if c["builder"] == "json"]
    assert len(csv_calls) == 1, f"CSV branch should fire: {spy.calls}"
    assert len(json_calls) == 0, f"JSON branch should be skipped: {spy.calls}"
    assert ds is not None
    assert ds.load_status["mixed"] == "Active (Local CSV)"


def test_webdataset_manifest_does_not_dispatch_as_jsonl(dispatch_dir):
    """A WebDataset directory whose tar loader is disabled (e.g. `webdataset`
    not installed) must NOT silently fall into the JSONL branch and load
    manifest.json as training data. Without this filter, every WebDataset
    cell that lacks webdataset library produces IndexError at iteration time.
    Regression caught during Phase 4 sweep: pixmo_cap with
    HAS_WEBDATASET=False loaded /flare/.../manifest.json as data."""
    (dispatch_dir / "manifest.json").write_text('{"shards": ["foo.tar"]}\n')
    (dispatch_dir / "local_manifest.json").write_text('{"shards": ["bar.tar"]}\n')
    # No tar files, no shards subdir — purely a WebDataset metadata directory.

    datasets_map = {
        "pixmo_like": {
            "name": "pixmo_like",
            "handler": "image_generic",
            "modality": "image",
            "skip": False,
            "local_path": str(dispatch_dir),
            "hf_id": "fake/remote",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    json_calls = [c for c in spy.calls if c["builder"] == "json"]
    assert len(json_calls) == 0, (
        f"manifest.json must not be passed to the json loader: {spy.calls}"
    )


def test_empty_directory_does_not_dispatch(dispatch_dir):
    """Empty directories should NOT trigger CSV/JSONL/recursive dispatch — the
    final fallback path (remote HF) should be hit instead."""
    datasets_map = {
        "empty_local": {
            "name": "empty_local",
            "handler": "ts_qa",
            "modality": "time_series",
            "skip": False,
            "local_path": str(dispatch_dir),
            "hf_id": "fake/remote",
            "preferred_source": "local",
        }
    }
    spy, ds = _run_with_mocked_loader(datasets_map)

    # No CSV/JSON local loader fired.
    local_calls = [c for c in spy.calls if c["builder"] in ("csv", "json")]
    assert local_calls == [], (
        f"Empty directory should not dispatch to local loader: {spy.calls}"
    )
    if ds is not None:
        assert ds.load_status.get("empty_local") != "Active (Local CSV)"
        assert ds.load_status.get("empty_local") != "Active (Local JSONL)"
        assert ds.load_status.get("empty_local") != "Active (Local JSONL Recursive)"
