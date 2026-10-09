#!/usr/bin/env python3
"""Fit IsoFLOP scaling exponents (α, β) from experiments.csv.

Workflow:
  1. Filter rows by family (+ optional backbone) where status == done
     and loss_main is non-null.
  2. Group by (backbone, budget_flops). For each group with >= 3 points,
     fit a parabola in log-N space:
         loss = a * log10(N)^2 + b * log10(N) + c
     The vertex log10(N_opt) = -b / (2a) gives N_opt = 10^log10(N_opt).
     Groups with < 3 points warn and skip.
  3. D_opt(C) = C / (6 * N_opt(C))  (Chinchilla token budget).
  4. Across budgets within a backbone, fit a power law in log space:
         log10(N_opt) = α * log10(C) + log10(N_0)
     (numpy.polyfit on log-log; analogous for D_opt → β).
  5. Bootstrap confidence intervals via resampling per-budget vertices.
  6. Emit JSON (machine-readable) + Markdown (human-readable) summary.

Usage:
    python tools/isoflop_fit.py \\
      --csv /flare/.../scaling-study/experiments.csv \\
      --family text_image --backbone OLMO3-1B \\
      --bootstrap 1000 \\
      --output-json /tmp/iso-fit.json \\
      --output-md /tmp/iso-fit.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"ERROR: --csv {path} not found")
    with open(path) as f:
        return list(csv.DictReader(f))


def _as_float(s: Any) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _filter_rows(
    rows: list[dict[str, str]],
    family: str,
    backbone: str | None,
    regime: str | None,
) -> list[dict[str, Any]]:
    """Keep only done rows for the requested family / backbone / regime
    with usable N + loss. Pass `regime=None` to disable regime filtering
    (rare — almost always one of projector_only/encoder_projector/e2e)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "done":
            continue
        if r.get("family") != family:
            continue
        if backbone and r.get("backbone") != backbone:
            continue
        if regime and r.get("regime") != regime:
            continue
        n_active = _as_float(r.get("n_active_params"))
        loss = _as_float(r.get("loss_main"))
        budget = _as_float(r.get("budget_flops"))
        if n_active is None or loss is None or budget is None:
            continue
        if n_active <= 0:
            continue
        out.append({
            "run_id": r.get("run_id"),
            "backbone": r.get("backbone"),
            "projector_variant": r.get("projector_variant"),
            "regime": r.get("regime"),
            "budget_flops": budget,
            "n_active_params": n_active,
            "loss_main": loss,
            "seed": _as_float(r.get("seed")) or 0,
        })
    return out


def _fit_parabola(
    log_n: np.ndarray,
    loss: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (a, b, c, log_n_opt).

    log_n_opt is NaN when the parabola has no usable minimum:
    - `a <= 1e-12`: effectively linear / numerically degenerate.
    - `a < 0`: inverted parabola — vertex is a *maximum* of the loss, not
      a minimum. Almost always a sign the budget has too few points or the
      loss is non-monotonic in N. Either way, refusing N_opt is safer than
      reporting a spurious one downstream.
    """
    if log_n.size < 3:
        raise ValueError(f"need >= 3 points to fit parabola, got {log_n.size}")
    a, b, c = np.polyfit(log_n, loss, 2)
    if a <= 1e-12:
        log_n_opt = float("nan")
    else:
        log_n_opt = float(-b / (2 * a))
    return float(a), float(b), float(c), log_n_opt


def _fit_power_law(
    log_c: np.ndarray,
    log_n_opt: np.ndarray,
) -> tuple[float, float]:
    """Return (slope, intercept) for log_n_opt = slope * log_c + intercept."""
    if log_c.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(log_c, log_n_opt, 1)
    return float(slope), float(intercept)


def _bootstrap_alpha_beta(
    log_c: np.ndarray,
    log_n_opt: np.ndarray,
    log_d_opt: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Return 95% CI for α (N exponent) and β (D exponent)."""
    if log_c.size < 2 or n_iter <= 0:
        return {"alpha_ci": (float("nan"), float("nan")),
                "beta_ci": (float("nan"), float("nan"))}
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, log_c.size, size=log_c.size)
        a, _ = _fit_power_law(log_c[idx], log_n_opt[idx])
        b, _ = _fit_power_law(log_c[idx], log_d_opt[idx])
        if not math.isnan(a):
            alphas.append(a)
        if not math.isnan(b):
            betas.append(b)
    if not alphas or not betas:
        return {"alpha_ci": (float("nan"), float("nan")),
                "beta_ci": (float("nan"), float("nan"))}
    return {
        "alpha_ci": (float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))),
        "beta_ci": (float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))),
    }


def fit_backbone(
    rows: list[dict[str, Any]],
    backbone: str,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fit per-budget parabolas + cross-budget power law for one backbone."""
    by_budget: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_budget.setdefault(r["budget_flops"], []).append(r)

    parabolas: list[dict[str, Any]] = []
    warnings: list[str] = []
    for budget, cells in sorted(by_budget.items()):
        if len(cells) < 3:
            warnings.append(
                f"{backbone} budget={budget:.3e}: only {len(cells)} points; need >= 3 for parabola"
            )
            continue
        n_active = np.array([c["n_active_params"] for c in cells], dtype=np.float64)
        loss = np.array([c["loss_main"] for c in cells], dtype=np.float64)
        log_n = np.log10(n_active)
        a, b, c, log_n_opt = _fit_parabola(log_n, loss)
        if not math.isfinite(log_n_opt):
            if a < 0:
                warnings.append(
                    f"{backbone} budget={budget:.3e}: parabola inverted (a={a:.3e}<0); "
                    f"vertex is a loss maximum — N_opt rejected"
                )
            else:
                warnings.append(
                    f"{backbone} budget={budget:.3e}: parabola degenerate (a={a:.3e}); "
                    f"N_opt rejected"
                )
        n_opt = float(10 ** log_n_opt) if math.isfinite(log_n_opt) else float("nan")
        # Chinchilla D_opt(C) = C / (6 * N_opt). The factor 6 matches the
        # FLOPs-per-token convention in the calibrator.
        d_opt = float(budget / (6 * n_opt)) if n_opt and math.isfinite(n_opt) else float("nan")
        parabolas.append({
            "budget_flops": budget,
            "n_points": len(cells),
            "a": a, "b": b, "c": c,
            "log_n_opt": log_n_opt,
            "N_opt": n_opt,
            "D_opt": d_opt,
        })

    valid = [p for p in parabolas
             if math.isfinite(p["N_opt"]) and math.isfinite(p["D_opt"])]
    if len(valid) >= 2:
        log_c = np.array([math.log10(p["budget_flops"]) for p in valid])
        log_n_opt = np.array([math.log10(p["N_opt"]) for p in valid])
        log_d_opt = np.array([math.log10(p["D_opt"]) for p in valid])
        alpha, log_n0 = _fit_power_law(log_c, log_n_opt)
        beta, log_d0 = _fit_power_law(log_c, log_d_opt)
        ci = _bootstrap_alpha_beta(log_c, log_n_opt, log_d_opt, bootstrap, rng)
    else:
        warnings.append(
            f"{backbone}: only {len(valid)} usable parabolas; need >= 2 for α/β fit"
        )
        alpha = beta = log_n0 = log_d0 = float("nan")
        ci = {"alpha_ci": (float("nan"), float("nan")),
              "beta_ci": (float("nan"), float("nan"))}

    return {
        "name": backbone,
        "n_cells": len(rows),
        "parabolas": parabolas,
        "alpha": alpha,
        "alpha_ci": list(ci["alpha_ci"]),
        "beta": beta,
        "beta_ci": list(ci["beta_ci"]),
        "log_n0": log_n0,
        "log_d0": log_d0,
        "warnings": warnings,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# IsoFLOP fit: {result['family']}",
        f"_Generated {result['generated_at']}_",
        "",
    ]
    for bb in result["backbones"]:
        lines.append(f"## Backbone: {bb['name']}")
        lines.append(f"- N cells: {bb['n_cells']}")
        a_lo, a_hi = bb["alpha_ci"]
        b_lo, b_hi = bb["beta_ci"]
        lines.append(
            f"- α (N exponent) = {bb['alpha']:.4f}  "
            f"[95% CI: {a_lo:.4f}, {a_hi:.4f}]"
        )
        lines.append(
            f"- β (D exponent) = {bb['beta']:.4f}  "
            f"[95% CI: {b_lo:.4f}, {b_hi:.4f}]"
        )
        if bb["parabolas"]:
            lines.append("")
            lines.append("| Budget (C) | n_points | N_opt | D_opt | a | b | c |")
            lines.append("|---|---|---|---|---|---|---|")
            for p in bb["parabolas"]:
                lines.append(
                    f"| {p['budget_flops']:.2e} | {p['n_points']} | "
                    f"{p['N_opt']:.3e} | {p['D_opt']:.3e} | "
                    f"{p['a']:.4f} | {p['b']:.4f} | {p['c']:.4f} |"
                )
        if bb["warnings"]:
            lines.append("")
            lines.append("### Warnings")
            for w in bb["warnings"]:
                lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="experiments.csv path")
    p.add_argument("--family", required=True, help="Cell family to fit (e.g. text_image)")
    p.add_argument("--backbone", default=None,
                   help="Fit one backbone only; defaults to all backbones present")
    p.add_argument("--regime", default="projector_only",
                   choices=["projector_only", "encoder_projector", "e2e", "any"],
                   help="Training regime to fit. Defaults to projector_only so "
                        "encoder_projector/e2e rows don't get pooled into a "
                        "projector-only parabola. Pass 'any' to fit across regimes.")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap iterations for α/β CIs (0 disables CIs)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for bootstrap")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows = _load_rows(Path(args.csv))
    regime_filter = None if args.regime == "any" else args.regime
    filtered = _filter_rows(rows, args.family, args.backbone, regime_filter)
    if not filtered:
        print("ERROR: no rows match filter (status=done, loss_main present)",
              file=sys.stderr)
        return 1
    backbones_present = sorted({r["backbone"] for r in filtered})
    backbones_to_fit = [args.backbone] if args.backbone else backbones_present

    results: list[dict[str, Any]] = []
    for bb in backbones_to_fit:
        subset = [r for r in filtered if r["backbone"] == bb]
        if not subset:
            continue
        results.append(fit_backbone(subset, bb, args.bootstrap, rng))

    out = {
        "family": args.family,
        "regime": args.regime,
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "csv": str(args.csv),
        "bootstrap": args.bootstrap,
        "backbones": results,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[isoflop_fit] wrote JSON: {out_json}")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown(out))
        print(f"[isoflop_fit] wrote MD:   {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
