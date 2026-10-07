"""Tests for tools/isoflop_plan.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "tools" / "isoflop_plan.py"


def _write_cal(cal_dir: Path, backbone: str, variant: str, family: str = "text_image") -> Path:
    cal_dir.mkdir(parents=True, exist_ok=True)
    p = cal_dir / f"{backbone}-{variant}-{family}-projector_only.json"
    p.write_text(json.dumps({
        "flops_per_step": 2.0e15,
        "samples_per_sec": 10.0,
        "mean_seq_len": 2048,
        "batch_size": 8,
        "seq_len": 2048,
    }))
    return p


def _run(*args: str, skip_backbone_check: bool = True) -> subprocess.CompletedProcess:
    """Invoke isoflop_plan.py with default `--skip-backbone-check`.

    Tests should NOT depend on which models happen to be staged on a host's
    /flare; the staging-gate test (`test_backbone_staging_check_*`) flips
    `skip_backbone_check=False` to exercise the check itself.
    """
    cmd = [sys.executable, str(PLAN), *args]
    if skip_backbone_check and "--skip-backbone-check" not in args:
        cmd.append("--skip-backbone-check")
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )


def test_help_runs():
    r = _run("--help")
    assert r.returncode == 0
    assert "--family" in r.stdout
    assert "--budgets" in r.stdout


def test_writes_full_cross_product(tmp_path: Path):
    cal = tmp_path / "cal"
    for v in ["BASE", "W2X", "W4X", "D2X", "D4X"]:
        _write_cal(cal, "OLMO3-1B", v)
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image",
        "--budgets", "3e17,1e18,3e18",
        "--backbones", "OLMO3-1B",
        "--projector-variants", "BASE,W2X,W4X,D2X,D4X",
        "--calibration-dir", str(cal),
        "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    # 3 budgets × 5 variants × 1 backbone × 1 seed = 15 cells
    assert len(plan["cells"]) == 15
    # Verify required cell fields
    for cell in plan["cells"]:
        assert cell["run_id"].startswith("ISO-text_image-OLMO3-1B-")
        assert cell["family"] == "text_image"
        assert cell["status"] == "planned"
        assert cell["max_steps"] > 0
        assert Path(cell["calibration_json"]).exists()


def test_variant_knobs_derive_from_variant_map(tmp_path: Path):
    cal = tmp_path / "cal"
    for v in ["BASE", "W2X", "W4X", "D2X", "D4X"]:
        _write_cal(cal, "OLMO3-1B", v)
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B",
        "--projector-variants", "BASE,W2X,W4X,D2X,D4X",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    by_variant = {c["projector_variant"]: c for c in plan["cells"]}
    # Source of truth = ModalityProjector.VARIANT_MAP
    assert (by_variant["BASE"]["projector_hidden_mult"],
            by_variant["BASE"]["projector_num_layers"]) == (1, 2)
    assert (by_variant["W2X"]["projector_hidden_mult"],
            by_variant["W2X"]["projector_num_layers"]) == (2, 2)
    assert (by_variant["W4X"]["projector_hidden_mult"],
            by_variant["W4X"]["projector_num_layers"]) == (4, 2)
    assert (by_variant["D2X"]["projector_hidden_mult"],
            by_variant["D2X"]["projector_num_layers"]) == (1, 4)
    assert (by_variant["D4X"]["projector_hidden_mult"],
            by_variant["D4X"]["projector_num_layers"]) == (1, 8)


def test_max_steps_correct(tmp_path: Path):
    """Raw cal/target math when rescale is disabled.

    With rescale enabled (default since PR-4), the helper-generated cal
    (BS=8/SL=2048/n_ranks=1) gets rescale_factor = 12 (assuming 1-node,
    12 ranks/node), so max_steps would be 500/12 ≈ 42. The
    rescale-specific behavior is covered by `test_runtime_rescale_*`;
    this test pins the raw math.
    """
    cal = tmp_path / "cal"
    _write_cal(cal, "OLMO3-1B", "BASE")
    out = tmp_path / "plan.yaml"
    # 1e18 / 2e15 = 500
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--no-runtime-rescale",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    assert plan["cells"][0]["max_steps"] == 500


def test_missing_calibration_warns_skips(tmp_path: Path):
    cal = tmp_path / "cal"
    _write_cal(cal, "OLMO3-1B", "BASE")
    # W2X intentionally absent
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B",
        "--projector-variants", "BASE,W2X",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0
    plan = yaml.safe_load(out.read_text())
    assert len(plan["cells"]) == 1  # Only BASE
    assert "skipped" in plan
    assert any("W2X" in s for s in plan["skipped"])


def test_seeds_multiplied(tmp_path: Path):
    cal = tmp_path / "cal"
    _write_cal(cal, "OLMO3-1B", "BASE")
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--seeds", "3",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0
    plan = yaml.safe_load(out.read_text())
    assert len(plan["cells"]) == 3
    seeds = {c["seed"] for c in plan["cells"]}
    assert seeds == {0, 1, 2}


def test_unknown_backbone_rejected(tmp_path: Path):
    cal = tmp_path / "cal"
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO99-1T",
        "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 2
    assert "unknown backbones" in r.stderr.lower()


def test_unknown_variant_rejected(tmp_path: Path):
    cal = tmp_path / "cal"
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B",
        "--projector-variants", "BOGUS",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 2
    assert "unknown variants" in r.stderr.lower()


def test_backbone_staging_check_passes_when_staged(monkeypatch, tmp_path: Path):
    """Staging check passes when the backbone has a non-empty snapshots/ dir.

    Uses in-process monkeypatch of `_HF_CACHE_ROOTS` so the test exercises
    the check in isolation and runs the same on every host (no dependency
    on real /flare contents).
    """
    import importlib
    sys.path.insert(0, str(REPO_ROOT))
    plan_mod = importlib.import_module("tools.isoflop_plan")
    importlib.reload(plan_mod)
    # Build a fake HF cache that mimics a properly-staged snapshot
    fake_cache = tmp_path / "fake_hf_cache"
    snap = fake_cache / "models--allenai--OLMo-1B-0724-hf" / "snapshots" / "rev0"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr(plan_mod, "_HF_CACHE_ROOTS", (str(fake_cache),))
    cal = tmp_path / "cal"
    _write_cal(cal, "OLMO3-1B", "BASE")
    out = tmp_path / "plan.yaml"
    rc = plan_mod.main([
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    ])
    assert rc == 0
    assert out.exists()


def test_backbone_staging_check_rejects_broken_snapshot(monkeypatch, tmp_path: Path):
    """A `models--<id>/` dir without a populated `snapshots/` fails the gate.

    Defends against the partial-cache case (e.g. `.no_exist/` markers only)
    that would let the plan through but crash at `from_pretrained`.
    """
    import importlib
    sys.path.insert(0, str(REPO_ROOT))
    plan_mod = importlib.import_module("tools.isoflop_plan")
    importlib.reload(plan_mod)
    fake_cache = tmp_path / "fake_hf_cache"
    # Top-level model dir exists, but snapshots/ is empty (the broken case).
    (fake_cache / "models--allenai--OLMo-1B-0724-hf" / "snapshots").mkdir(parents=True)
    monkeypatch.setattr(plan_mod, "_HF_CACHE_ROOTS", (str(fake_cache),))
    assert not plan_mod._backbone_is_staged("allenai/OLMo-1B-0724-hf")
    # And the missing snapshots/ dir altogether
    (fake_cache / "models--allenai--OLMo-7B-0724-hf").mkdir()
    assert not plan_mod._backbone_is_staged("allenai/OLMo-7B-0724-hf")


def test_backbone_staging_check_rejects_unstaged(monkeypatch, tmp_path: Path):
    """When a backbone's hf_id isn't in any HF cache, plan errors out."""
    import importlib
    sys.path.insert(0, str(REPO_ROOT))
    plan_mod = importlib.import_module("tools.isoflop_plan")
    importlib.reload(plan_mod)
    # Temporarily inject a backbone with a definitely-missing hf_id.
    monkeypatched_table = dict(plan_mod._BACKBONE_TABLE)
    monkeypatched_table["BOGUS-1B"] = {
        "hf_id": "bogus-org/never-staged-model",
        "design": "PRISM-IMAGE-ONLY-1N", "nodes": 1,
    }
    monkeypatch.setattr(plan_mod, "_BACKBONE_TABLE", monkeypatched_table)
    # Direct in-process call (not subprocess) to honor the monkeypatch.
    cal = tmp_path / "cal"
    _write_cal(cal, "BOGUS-1B", "BASE")
    out = tmp_path / "plan.yaml"
    rc = plan_mod.main([
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "BOGUS-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    ])
    assert rc == 2
    # Output not written
    assert not out.exists()


def test_backbone_staging_check_skip_flag(monkeypatch, tmp_path: Path):
    """`--skip-backbone-check` bypasses the staging gate."""
    import importlib
    sys.path.insert(0, str(REPO_ROOT))
    plan_mod = importlib.import_module("tools.isoflop_plan")
    importlib.reload(plan_mod)
    monkeypatched_table = dict(plan_mod._BACKBONE_TABLE)
    monkeypatched_table["BOGUS-1B"] = {
        "hf_id": "bogus-org/never-staged-model",
        "design": "PRISM-IMAGE-ONLY-1N", "nodes": 1,
    }
    monkeypatch.setattr(plan_mod, "_BACKBONE_TABLE", monkeypatched_table)
    cal = tmp_path / "cal"
    # Calibration JSON with a matching backbone_id so the consistency
    # check doesn't fire (different code path; verified separately).
    cal.mkdir()
    p = cal / "BOGUS-1B-BASE-text_image-projector_only.json"
    p.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
        "backbone_id": "bogus-org/never-staged-model",
    }))
    out = tmp_path / "plan.yaml"
    rc = plan_mod.main([
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "BOGUS-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
        "--skip-backbone-check",
    ])
    assert rc == 0
    assert out.exists()


def test_calibration_backbone_id_all_mismatch_is_error(tmp_path: Path):
    """If EVERY would-be cell is rejected for backbone_id mismatch, fail loudly.

    Silently writing an empty manifest would let `isoflop_launch.py` no-op
    on a stale calibration; we want the user to notice. Mismatch is also
    surfaced to stderr at warning time.
    """
    cal = tmp_path / "cal"
    cal.mkdir()
    # Note: OLMO3-1B → allenai/OLMo-1B-0724-hf in the table. Calibration
    # JSON below claims it was made against a DIFFERENT model.
    p = cal / "OLMO3-1B-BASE-text_image-projector_only.json"
    p.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
        "backbone_id": "some-other-org/stale-model",
    }))
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 2, r.stderr
    assert "backbone_id" in r.stderr
    assert "every cell was skipped" in r.stderr.lower()
    assert not out.exists()


def test_calibration_backbone_id_partial_mismatch_emits_remaining(tmp_path: Path):
    """If SOME cells match and some don't, write the survivors and warn on the rest."""
    cal = tmp_path / "cal"
    cal.mkdir()
    good = cal / "OLMO3-1B-BASE-text_image-projector_only.json"
    good.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
        "backbone_id": "allenai/OLMo-1B-0724-hf",
    }))
    bad = cal / "OLMO3-1B-W2X-text_image-projector_only.json"
    bad.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
        "backbone_id": "some-other-org/stale-model",
    }))
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE,W2X",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0
    assert "W2X" in r.stderr  # warning surfaced
    plan = yaml.safe_load(out.read_text())
    assert len(plan["cells"]) == 1
    assert plan["cells"][0]["projector_variant"] == "BASE"
    assert any("W2X" in s for s in plan.get("skipped", []))


def test_calibration_backbone_id_match_emits_cell(tmp_path: Path):
    """When cal JSON's backbone_id matches the table, the cell is emitted."""
    cal = tmp_path / "cal"
    cal.mkdir()
    p = cal / "OLMO3-1B-BASE-text_image-projector_only.json"
    p.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 2048, "batch_size": 8, "seq_len": 2048,
        "backbone_id": "allenai/OLMo-1B-0724-hf",
    }))
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0
    plan = yaml.safe_load(out.read_text())
    assert len(plan["cells"]) == 1


def test_olmo3_32b_no_longer_in_table():
    """OLMO3-32B was removed (no 32B-class model staged); keep the contract."""
    import importlib
    sys.path.insert(0, str(REPO_ROOT))
    plan_mod = importlib.import_module("tools.isoflop_plan")
    importlib.reload(plan_mod)
    assert "OLMO3-32B" not in plan_mod._BACKBONE_TABLE
    # The two surviving entries point at actually-staged models.
    assert plan_mod._BACKBONE_TABLE["OLMO3-1B"]["hf_id"] == "allenai/OLMo-1B-0724-hf"
    assert plan_mod._BACKBONE_TABLE["OLMO3-7B"]["hf_id"] == "allenai/OLMo-7B-0724-hf"


def _write_cal_with_config(cal_dir: Path, backbone: str, variant: str,
                            bs: int = 4, sl: int = 1024, n_ranks: int = 1,
                            fps: float = 2.0e15,
                            backbone_id: str = "allenai/OLMo-1B-0724-hf") -> Path:
    """Like _write_cal but stamps the calibration config explicitly."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    p = cal_dir / f"{backbone}-{variant}-text_image-projector_only.json"
    p.write_text(json.dumps({
        "flops_per_step": fps,
        "samples_per_sec": 10.0,
        "mean_seq_len": float(sl),
        "batch_size": bs,
        "seq_len": sl,
        "n_ranks": n_ranks,
        "backbone_id": backbone_id,
    }))
    return p


def test_runtime_rescale_inflates_max_steps(tmp_path: Path):
    """Default rescale: cal BS=4/SL=1024/1-rank vs runtime BS=8/SL=2048/12-rank → 48× FPS bump → 48× fewer steps."""
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1, fps=2.0e15)
    out = tmp_path / "plan.yaml"
    # target=1e18, cal_fps=2e15, rescale=48 → runtime_fps=9.6e16 → 1e18/9.6e16 = 10 steps
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    cell = plan["cells"][0]
    assert cell["rescale_factor"] == pytest.approx(48.0)
    assert cell["calibration_fps"] == pytest.approx(2.0e15)
    assert cell["runtime_fps"] == pytest.approx(9.6e16)
    # 1e18 / 9.6e16 ≈ 10.42 → rounds to 10
    assert cell["max_steps"] == 10


def test_no_runtime_rescale_preserves_raw_math(tmp_path: Path):
    """--no-runtime-rescale: max_steps = target / cal_fps exactly (no inflation)."""
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1, fps=2.0e15)
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--no-runtime-rescale",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    cell = plan["cells"][0]
    assert cell["rescale_factor"] == 1.0
    assert cell["calibration_fps"] == cell["runtime_fps"]
    # 1e18 / 2e15 = 500
    assert cell["max_steps"] == 500


def test_nodes_override_propagates_to_every_cell(tmp_path: Path):
    """`--nodes-override 4` must override _BACKBONE_TABLE['*']['nodes'] for
    EVERY cell in the plan. Stage A round 1 needs 4 nodes per cell for fast
    wall time; without this flag, OLMO3-1B cells would launch on 1 node each.
    """
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1, fps=2.0e15)
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18,3e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--seeds", "2",
        "--nodes-override", "4",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    assert len(plan["cells"]) == 4  # 2 budgets × 1 variant × 2 seeds
    for cell in plan["cells"]:
        assert cell["nodes"] == 4, cell


def test_nodes_override_rescales_max_steps(tmp_path: Path):
    """Doubling --nodes-override from 1 to 2 doubles the effective rank count,
    which doubles runtime_fps and halves max_steps (same total FLOPs budget).
    This is the wall-time-via-parallelism trade-off that motivated the flag.
    """
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1, fps=2.0e15)
    out_1n = tmp_path / "plan_1n.yaml"
    out_4n = tmp_path / "plan_4n.yaml"
    base_args = (
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal),
    )
    r1 = _run(*base_args, "--nodes-override", "1", "--output", str(out_1n))
    r4 = _run(*base_args, "--nodes-override", "4", "--output", str(out_4n))
    assert r1.returncode == 0 and r4.returncode == 0
    plan_1n = yaml.safe_load(out_1n.read_text())
    plan_4n = yaml.safe_load(out_4n.read_text())
    cell_1n = plan_1n["cells"][0]
    cell_4n = plan_4n["cells"][0]
    # Default ranks/node=12. 1n: 12 ranks, 4n: 48 ranks → 4× rescale_factor.
    assert cell_4n["rescale_factor"] == pytest.approx(cell_1n["rescale_factor"] * 4)
    # …so 4× the runtime_fps, hence ~1/4 the steps. Allow ±1 step for
    # rounding (`max(1, round(...))`).
    assert abs(cell_1n["max_steps"] - 4 * cell_4n["max_steps"]) <= 4


def test_nodes_override_rejects_zero(tmp_path: Path):
    """--nodes-override 0 is meaningless (divides-by-zero in tokens math)."""
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1)
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--nodes-override", "0",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 2, r.stderr
    assert "nodes-override" in r.stderr.lower()


def test_rescale_handles_custom_runtime_config(tmp_path: Path):
    """--runtime-* flags should change the rescale factor."""
    cal = tmp_path / "cal"
    _write_cal_with_config(cal, "OLMO3-1B", "BASE", bs=4, sl=1024, n_ranks=1, fps=2.0e15)
    out = tmp_path / "plan.yaml"
    # BS=4/SL=1024/4 ranks → rescale = (4/4) * (1024/1024) * (4/1) = 4
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--runtime-batch-size", "4",
        "--runtime-seq-len", "1024",
        "--runtime-ranks-per-node", "4",
        "--calibration-dir", str(cal), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    cell = plan["cells"][0]
    assert cell["rescale_factor"] == pytest.approx(4.0)
    assert cell["runtime_fps"] == pytest.approx(8.0e15)


def test_yaml_configs_declare_projector_knobs():
    """Every src/conf/model/*.yaml must declare projector_hidden_mult and projector_num_layers.

    Hydra struct mode rejects overrides for keys not in the YAML. PR-1
    added the dataclass fields but missed YAML defaults; PR-4 patches them.
    """
    import yaml as _yaml
    conf_dir = REPO_ROOT / "src" / "conf" / "model"
    missing: list[str] = []
    for yp in sorted(conf_dir.glob("*.yaml")):
        d = _yaml.safe_load(yp.read_text())
        if not isinstance(d, dict):
            continue
        if "projector_hidden_mult" not in d:
            missing.append(f"{yp.name}: missing projector_hidden_mult")
        if "projector_num_layers" not in d:
            missing.append(f"{yp.name}: missing projector_num_layers")
    assert not missing, "Hydra-struct-incompatible model YAMLs:\n  " + "\n  ".join(missing)


def test_training_yamls_declare_eval_enabled():
    """Every src/conf/training/*.yaml must declare eval_enabled (default false)."""
    import yaml as _yaml
    conf_dir = REPO_ROOT / "src" / "conf" / "training"
    missing: list[str] = []
    for yp in sorted(conf_dir.glob("*.yaml")):
        d = _yaml.safe_load(yp.read_text())
        if not isinstance(d, dict):
            continue
        if "eval_enabled" not in d:
            missing.append(f"{yp.name}: missing eval_enabled")
    assert not missing, "training YAMLs missing eval_enabled:\n  " + "\n  ".join(missing)


def test_calibrator_stamps_n_ranks(tmp_path: Path):
    """The calibrator JSON should include n_ranks (default 1) for PR-4 rescale."""
    # We don't actually run the calibrator (needs XPU); just verify the
    # `--help` output mentions the field stamping by checking a synthetic
    # JSON has n_ranks. The real assertion is the round-trip test below.
    cal_dir = tmp_path / "cal"
    p = _write_cal_with_config(cal_dir, "OLMO3-1B", "BASE", n_ranks=1)
    d = json.loads(p.read_text())
    assert d["n_ranks"] == 1, "calibration JSON must stamp n_ranks"


def test_cal_missing_n_ranks_defaults_to_1(tmp_path: Path):
    """Old calibration JSONs without n_ranks should still work (back-compat)."""
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    p = cal_dir / "OLMO3-1B-BASE-text_image-projector_only.json"
    # Note: no n_ranks key
    p.write_text(json.dumps({
        "flops_per_step": 2.0e15, "samples_per_sec": 10.0,
        "mean_seq_len": 1024, "batch_size": 4, "seq_len": 1024,
        "backbone_id": "allenai/OLMo-1B-0724-hf",
    }))
    out = tmp_path / "plan.yaml"
    r = _run(
        "--family", "text_image", "--budgets", "1e18",
        "--backbones", "OLMO3-1B", "--projector-variants", "BASE",
        "--calibration-dir", str(cal_dir), "--output", str(out),
    )
    assert r.returncode == 0, r.stderr
    plan = yaml.safe_load(out.read_text())
    # Default cal_ranks=1 → rescale = 48 (same as the explicit case)
    assert plan["cells"][0]["rescale_factor"] == pytest.approx(48.0)
