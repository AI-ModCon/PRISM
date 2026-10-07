"""Unit tests for the per-modality sweep harness pure functions.

Covers:
- `tools/perf_aggregate.py::_per_modality_aggregate` — warmup drop,
  sweep_id filter, mismatch propagation from startup_modality_check,
  empty-input path.
- `src/train.py::_resolve_dataset_overrides` — None cfg, missing data
  section, OmegaConf round-trip, plain dict pass-through, type errors.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_perf_aggregate():
    """Load `tools/perf_aggregate.py` as a module (it's not on a package path)."""
    spec = importlib.util.spec_from_file_location(
        "_perf_aggregate_under_test",
        REPO_ROOT / "tools" / "perf_aggregate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_resolve_overrides():
    """Import _resolve_dataset_overrides from src.train without triggering its CLI."""
    from src import train

    return train._resolve_dataset_overrides


# ---------------------------------------------------------------------------
# _per_modality_aggregate
# ---------------------------------------------------------------------------


def _perf_record(preset, step, sps, sweep_id="sw1", run_dir="r1", tps=None, tpb=None, bmc=None):
    rec = {
        "event": "throughput",
        "preset": preset,
        "step": step,
        "samples_per_sec": sps,
        "sweep_id": sweep_id,
        "run_dir": run_dir,
    }
    if tps is not None:
        rec["tokens_per_sec"] = tps
    if tpb is not None:
        rec["tokens_per_batch"] = tpb
    if bmc is not None:
        rec["batch_modality_counts"] = bmc
    return rec


def _startup_record(preset, mismatch, dl=None, mm=None, sweep_id="sw1", run_dir="r1"):
    return {
        "event": "startup_modality_check",
        "preset": preset,
        "run_dir": run_dir,
        "sweep_id": sweep_id,
        "mismatch": mismatch,
        "dataloader_modalities": dl if dl is not None else ["image", "text"],
        "model_modalities": mm if mm is not None else ["image", "text"],
    }


def test_per_modality_aggregate_empty_records_returns_empty():
    mod = _load_perf_aggregate()
    assert mod._per_modality_aggregate([], sweep_id="sw1") == []


def test_per_modality_aggregate_warmup_drop():
    """First N records per (run_dir, preset) are dropped before stats."""
    mod = _load_perf_aggregate()
    # 15 records, warmup=10 → 5 survive. samples_per_sec set so warmup vs
    # post-warmup yield obviously different means.
    recs = [_perf_record("text_image", step=i, sps=1.0) for i in range(10)]
    recs += [_perf_record("text_image", step=10 + i, sps=100.0) for i in range(5)]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=10)
    assert len(out) == 1
    row = out[0]
    assert row["preset"] == "text_image"
    assert row["n_records"] == 5
    assert math.isclose(row["samples_per_sec_mean"], 100.0)


def test_per_modality_aggregate_warmup_zero_keeps_all():
    mod = _load_perf_aggregate()
    recs = [_perf_record("text_image", step=i, sps=float(i)) for i in range(5)]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    assert out[0]["n_records"] == 5
    # mean(0,1,2,3,4) = 2.0
    assert math.isclose(out[0]["samples_per_sec_mean"], 2.0)


def test_per_modality_aggregate_warmup_larger_than_run_keeps_all():
    """Don't drop everything if a run has fewer records than the warmup window."""
    mod = _load_perf_aggregate()
    recs = [_perf_record("text_image", step=i, sps=5.0) for i in range(3)]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=10)
    assert out[0]["n_records"] == 3


def test_per_modality_aggregate_sweep_id_filter():
    mod = _load_perf_aggregate()
    recs = [
        _perf_record("text_image", step=10, sps=10.0, sweep_id="keep"),
        _perf_record("text_image", step=10, sps=99.0, sweep_id="drop"),
        _perf_record("text_image", step=10, sps=20.0, sweep_id="keep"),
    ]
    out = mod._per_modality_aggregate(recs, sweep_id="keep", warmup=0)
    assert out[0]["n_records"] == 2
    # mean(10, 20) = 15.0 — the drop-sweep 99.0 is excluded.
    assert math.isclose(out[0]["samples_per_sec_mean"], 15.0)


def test_per_modality_aggregate_sweep_id_none_keeps_all():
    """sweep_id=None means no filter; mixed sweeps are aggregated."""
    mod = _load_perf_aggregate()
    recs = [
        _perf_record("text_image", step=10, sps=10.0, sweep_id="a"),
        _perf_record("text_image", step=10, sps=20.0, sweep_id="b"),
    ]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    assert out[0]["n_records"] == 2


def test_per_modality_aggregate_mismatch_propagated_from_startup():
    mod = _load_perf_aggregate()
    recs = [
        _startup_record("text_ts", mismatch=True, dl=["image"], mm=["time_series"]),
        _perf_record("text_ts", step=10, sps=5.0),
    ]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    assert len(out) == 1
    row = out[0]
    assert row["mismatch"] is True
    assert json.loads(row["dataloader_modalities"]) == ["image"]
    assert json.loads(row["model_modalities"]) == ["time_series"]


def test_per_modality_aggregate_no_startup_record_yields_null_mismatch():
    mod = _load_perf_aggregate()
    recs = [_perf_record("text_image", step=10, sps=5.0)]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    assert out[0]["mismatch"] is None
    assert json.loads(out[0]["dataloader_modalities"]) == []
    assert json.loads(out[0]["model_modalities"]) == []


def test_per_modality_aggregate_groups_multiple_presets():
    mod = _load_perf_aggregate()
    recs = [
        _perf_record("text_image", step=10, sps=10.0),
        _perf_record("text_image", step=11, sps=12.0),
        _perf_record("text_ts", step=10, sps=5.0),
    ]
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    by_preset = {r["preset"]: r for r in out}
    assert set(by_preset) == {"text_image", "text_ts"}
    assert math.isclose(by_preset["text_image"]["samples_per_sec_mean"], 11.0)
    assert math.isclose(by_preset["text_ts"]["samples_per_sec_mean"], 5.0)


def test_per_modality_aggregate_most_frequent_bmc():
    """batch_modality_counts is rolled up to the most-frequent value across the window."""
    mod = _load_perf_aggregate()
    recs = [
        _perf_record("text_image", step=i, sps=5.0, bmc={"image": 8, "text": 8})
        for i in range(3)
    ]
    recs.append(_perf_record("text_image", step=3, sps=5.0, bmc={"image": 4, "text": 4}))
    out = mod._per_modality_aggregate(recs, sweep_id=None, warmup=0)
    # 3 votes for {image:8,text:8} vs 1 vote for the other.
    parsed = json.loads(out[0]["batch_modality_counts"])
    assert parsed == {"image": 8, "text": 8}


# ---------------------------------------------------------------------------
# _resolve_dataset_overrides
# ---------------------------------------------------------------------------


def test_resolve_overrides_none_cfg():
    fn = _load_resolve_overrides()
    assert fn(None) is None


def test_resolve_overrides_missing_data_section():
    fn = _load_resolve_overrides()
    cfg = OmegaConf.create({"training": {"batch_size": 8}})
    assert fn(cfg) is None


def test_resolve_overrides_missing_dataset_overrides_field():
    fn = _load_resolve_overrides()
    cfg = OmegaConf.create({"data": {"other_field": 1}})
    assert fn(cfg) is None


def test_resolve_overrides_omegaconf_round_trip():
    fn = _load_resolve_overrides()
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset_overrides": {
                    "ts_qa": {"skip": True},
                    "pixmo_cap": {"skip": False, "weight": 0.5},
                }
            }
        }
    )
    out = fn(cfg)
    assert isinstance(out, dict)
    assert out == {
        "ts_qa": {"skip": True},
        "pixmo_cap": {"skip": False, "weight": 0.5},
    }
    # Nested values must also be plain Python types, not OmegaConf containers.
    assert isinstance(out["ts_qa"], dict)


def test_resolve_overrides_plain_dict_pass_through():
    """Tests that hand-build dicts (rather than OmegaConf) also work."""
    fn = _load_resolve_overrides()

    class _PlainCfg:
        def get(self, key, default=None):
            if key == "data":
                return self  # data section is "self" — recurse
            if key == "dataset_overrides":
                return {"ts_qa": {"skip": True}}
            return default

    out = fn(_PlainCfg())
    assert out == {"ts_qa": {"skip": True}}


def test_resolve_overrides_rejects_non_mapping():
    """Misconfigured override yaml (list instead of mapping) must fail loud."""
    fn = _load_resolve_overrides()
    cfg = OmegaConf.create({"data": {"dataset_overrides": ["ts_qa", "pixmo_cap"]}})
    with pytest.raises(TypeError):
        fn(cfg)


def test_resolve_overrides_rejects_scalar():
    fn = _load_resolve_overrides()
    cfg = OmegaConf.create({"data": {"dataset_overrides": "not-a-dict"}})
    with pytest.raises(TypeError):
        fn(cfg)
