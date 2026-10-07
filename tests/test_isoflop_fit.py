"""Tests for tools/isoflop_fit.py.

Two flavors:
1. Subprocess smoke (--help + round-trip on a synthetic CSV).
2. Direct math test: synthetic Chinchilla-like data with planted α = β = 0.5
   must round-trip within 1% (the fit is exact in the noise-free case).
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIT = REPO_ROOT / "tools" / "isoflop_fit.py"

# Single source of truth for the column tuple — keeps test fixtures in
# lockstep with the production schema.
sys.path.insert(0, str(REPO_ROOT))
from tools.isoflop_launch import CSV_COLUMNS  # noqa: E402


def _write_chinchilla_csv(path: Path, alpha_true: float = 0.5) -> None:
    """Write a synthetic experiments.csv with known α = β = alpha_true."""
    N_0 = 1e6
    budgets = [3e17, 1e18, 3e18, 1e19, 3e19]
    variants = ["BASE", "W2X", "W4X", "D2X", "D4X"]
    offsets = [-0.3, -0.15, 0.0, 0.15, 0.3]  # in log10(N)

    rows = []
    for C in budgets:
        log_N_opt = math.log10(N_0 * (C ** alpha_true))
        for v, off in zip(variants, offsets, strict=True):
            N = 10 ** (log_N_opt + off)
            loss = 0.1 * off**2 + 2.0  # parabolic in log10(N) - log_N_opt
            row = {c: "" for c in CSV_COLUMNS}
            row.update({
                "run_id": f"ISO-text_image-OLMO3-1B-{v}-{C:.0e}".replace("+", ""),
                "phase": "M3", "family": "text_image", "regime": "projector_only",
                "backbone": "OLMO3-1B", "projector_variant": v,
                "budget_flops": f"{C:.6e}",
                "seed": "0", "nodes": "1", "batch_size": "8", "max_seq_length": "2048",
                "max_steps": "100", "n_active_params": f"{N:.6e}",
                "loss_main": f"{loss:.6f}", "loss_caption": f"{loss:.6f}",
                "status": "done",
            })
            rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FIT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )


def test_help_runs():
    r = _run("--help")
    assert r.returncode == 0
    assert "--family" in r.stdout
    assert "--bootstrap" in r.stdout


def test_recovers_alpha_beta_05(tmp_path: Path):
    csv_p = tmp_path / "exp.csv"
    _write_chinchilla_csv(csv_p, alpha_true=0.5)
    json_p = tmp_path / "fit.json"
    md_p = tmp_path / "fit.md"
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--backbone", "OLMO3-1B", "--bootstrap", "100",
             "--output-json", str(json_p), "--output-md", str(md_p))
    assert r.returncode == 0, r.stderr
    data = json.loads(json_p.read_text())
    bb = data["backbones"][0]
    # Tolerance: 1% on noise-free data (numpy.polyfit is exact in principle,
    # tiny float drift).
    assert abs(bb["alpha"] - 0.5) < 0.01, f"alpha={bb['alpha']}"
    assert abs(bb["beta"] - 0.5) < 0.01, f"beta={bb['beta']}"
    # MD summary written
    assert "OLMO3-1B" in md_p.read_text()
    assert "α (N exponent)" in md_p.read_text()


def test_parabolas_match_budget_count(tmp_path: Path):
    csv_p = tmp_path / "exp.csv"
    _write_chinchilla_csv(csv_p)
    json_p = tmp_path / "fit.json"
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--backbone", "OLMO3-1B", "--bootstrap", "0",
             "--output-json", str(json_p))
    assert r.returncode == 0
    data = json.loads(json_p.read_text())
    bb = data["backbones"][0]
    # 5 budgets in fixture → 5 parabolas
    assert len(bb["parabolas"]) == 5
    for p in bb["parabolas"]:
        assert p["n_points"] == 5  # 5 variants per budget
        assert math.isfinite(p["N_opt"])
        assert math.isfinite(p["D_opt"])


def test_insufficient_points_warns(tmp_path: Path):
    """A budget with only 2 points must warn + skip (need >= 3 for parabola)."""
    csv_p = tmp_path / "exp.csv"
    # Only 2 variants at one budget
    rows = []
    for v, N in [("BASE", 1e9), ("W2X", 2e9)]:
        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "run_id": f"X-{v}", "phase": "M3", "family": "text_image",
            "regime": "projector_only", "backbone": "OLMO3-1B",
            "projector_variant": v, "budget_flops": "1.000000e+18",
            "seed": "0", "nodes": "1", "batch_size": "8",
            "max_seq_length": "2048", "max_steps": "100",
            "n_active_params": f"{N:.6e}", "loss_main": "2.0",
            "loss_caption": "2.0", "status": "done",
        })
        rows.append(row)
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    json_p = tmp_path / "fit.json"
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--backbone", "OLMO3-1B", "--bootstrap", "0",
             "--output-json", str(json_p))
    assert r.returncode == 0, r.stderr
    data = json.loads(json_p.read_text())
    bb = data["backbones"][0]
    assert bb["parabolas"] == []  # no parabola fit
    assert any("only 2 points" in w for w in bb["warnings"])


def test_no_done_rows_returns_error(tmp_path: Path):
    csv_p = tmp_path / "exp.csv"
    csv_p.write_text(",".join(CSV_COLUMNS) + "\n")
    json_p = tmp_path / "fit.json"
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--bootstrap", "0", "--output-json", str(json_p))
    assert r.returncode == 1
    assert "no rows match" in r.stderr.lower()


def test_inverted_parabola_rejected(tmp_path: Path):
    """a < 0 parabola (loss DECREASES then INCREASES inverted) → N_opt = NaN."""
    csv_p = tmp_path / "exp.csv"
    # Construct a strictly concave-down loss curve: loss(log N) = -k*(log N)^2
    # → a < 0 in polyfit, vertex is a maximum (not a minimum). Same offsets
    # as the chinchilla fixture but with the loss sign flipped relative to
    # a U-shape.
    rows = []
    for v, off in zip(["BASE", "W2X", "W4X", "D2X", "D4X"],
                       [-0.3, -0.15, 0.0, 0.15, 0.3], strict=True):
        N = 10 ** (9 + off)
        loss = -0.1 * off ** 2 + 2.0  # concave-down
        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "run_id": f"INV-{v}", "phase": "M3", "family": "text_image",
            "regime": "projector_only", "backbone": "OLMO3-1B",
            "projector_variant": v, "budget_flops": "1.000000e+18",
            "seed": "0", "nodes": "1", "batch_size": "8",
            "max_seq_length": "2048", "max_steps": "100",
            "n_active_params": f"{N:.6e}", "loss_main": f"{loss:.6f}",
            "loss_caption": f"{loss:.6f}", "status": "done",
        })
        rows.append(row)
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    json_p = tmp_path / "fit.json"
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--backbone", "OLMO3-1B", "--bootstrap", "0",
             "--output-json", str(json_p))
    assert r.returncode == 0, r.stderr
    data = json.loads(json_p.read_text())
    bb = data["backbones"][0]
    assert len(bb["parabolas"]) == 1
    assert math.isnan(bb["parabolas"][0]["N_opt"])
    assert any("inverted" in w for w in bb["warnings"])


def test_regime_filter_excludes_other_regimes(tmp_path: Path):
    """Rows with regime != projector_only are excluded by default."""
    csv_p = tmp_path / "exp.csv"
    rows = []
    # 5 projector_only rows that would fit cleanly...
    for v, off in zip(["BASE", "W2X", "W4X", "D2X", "D4X"],
                       [-0.3, -0.15, 0.0, 0.15, 0.3], strict=True):
        N = 10 ** (9 + off)
        loss = 0.1 * off ** 2 + 2.0
        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "run_id": f"PO-{v}", "phase": "M3", "family": "text_image",
            "regime": "projector_only", "backbone": "OLMO3-1B",
            "projector_variant": v, "budget_flops": "1.000000e+18",
            "seed": "0", "nodes": "1", "batch_size": "8",
            "max_seq_length": "2048", "max_steps": "100",
            "n_active_params": f"{N:.6e}", "loss_main": f"{loss:.6f}",
            "loss_caption": f"{loss:.6f}", "status": "done",
        })
        rows.append(row)
    # ...plus 3 e2e rows that would CONTAMINATE the fit if pooled.
    for v, off in zip(["BASE", "W2X", "W4X"], [0.0, 0.1, 0.2], strict=True):
        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "run_id": f"E2E-{v}", "phase": "M3", "family": "text_image",
            "regime": "e2e", "backbone": "OLMO3-1B",
            "projector_variant": v, "budget_flops": "1.000000e+18",
            "seed": "0", "nodes": "1", "batch_size": "8",
            "max_seq_length": "2048", "max_steps": "100",
            "n_active_params": f"{10**(11+off):.6e}",
            "loss_main": "1.5", "loss_caption": "1.5", "status": "done",
        })
        rows.append(row)
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(rows)
    json_p = tmp_path / "fit.json"
    # Default --regime projector_only — should drop the 3 e2e rows.
    r = _run("--csv", str(csv_p), "--family", "text_image",
             "--backbone", "OLMO3-1B", "--bootstrap", "0",
             "--output-json", str(json_p))
    assert r.returncode == 0, r.stderr
    data = json.loads(json_p.read_text())
    bb = data["backbones"][0]
    assert bb["n_cells"] == 5  # NOT 8

    # With --regime any, all 8 cells should be pooled.
    json_p2 = tmp_path / "fit2.json"
    r2 = _run("--csv", str(csv_p), "--family", "text_image",
              "--backbone", "OLMO3-1B", "--regime", "any",
              "--bootstrap", "0", "--output-json", str(json_p2))
    assert r2.returncode == 0, r2.stderr
    data2 = json.loads(json_p2.read_text())
    assert data2["backbones"][0]["n_cells"] == 8
