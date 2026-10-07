#!/usr/bin/env python3
"""Render IsoFLOP scaling-laws plots from experiments.csv.

Produces a 2x2 figure per backbone:
  - Top-left:  per-budget IsoFLOP curves (loss vs log10(N_active)) + parabolic
               fits + vertex stars marking N_opt.
  - Top-right: N_opt vs FLOPs (log-log) + power-law fit α.
  - Bottom-left: D_opt vs FLOPs (log-log) + power-law fit β.
               D_opt computed as Chinchilla D = C / (6 N_opt).
  - Bottom-right: variance-floor diagnostic for the lowest-budget BASE@C cells —
               horizontal bar per seed showing loss + mean line. If fewer than
               2 same-(variant, budget) seeds exist, omitted.

Reuses isoflop_fit.py's row loader / parabola fitter so the visualization
and the numerical fit agree on cell filtering.

Usage:
    python tools/isoflop_plot.py \\
      --csv /flare/.../scaling-study/experiments.csv \\
      --family text_image \\
      --outdir /flare/.../scaling-study/results/figs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Matplotlib non-interactive backend — runs fine on UAN with no display.
import matplotlib  # noqa: E402

# Reuse fit-tool primitives so plotting and fit agree on cell filtering
# AND parabola fitting (so plotted vertex matches the JSON/MD report).
from tools.isoflop_fit import _filter_rows, _fit_parabola, _load_rows  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=str(REPO_ROOT / "scaling-study" / "experiments.csv"),
        help="Path to experiments.csv (default: <repo>/scaling-study/experiments.csv)",
    )
    p.add_argument("--family", default="text_image",
                   help="Family to plot (text_image, text_ts, etc.)")
    p.add_argument("--backbone", default=None,
                   help="Backbone to plot. If omitted, makes one figure per "
                        "backbone present in the CSV.")
    p.add_argument("--regime", default="projector_only",
                   choices=["projector_only", "encoder_projector", "e2e", "any"])
    p.add_argument(
        "--outdir", required=True,
        help="Directory to write PNGs into. Created if missing.",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default 150 — readable, not too large).",
    )
    return p.parse_args(argv)


def _fit_parabola_for_plot(
    log_n: np.ndarray, loss: np.ndarray,
) -> tuple[float, float, float, float | None]:
    """Thin wrapper around isoflop_fit._fit_parabola for plotting.

    The fit tool raises on <3 points and returns NaN for log_n_opt when the
    parabola is degenerate. The plotter prefers:
      - None for log_n_opt (so the star can be skipped via truthiness)
      - linear-ish coefficients for 1-2 points (so we can still draw a
        trend line in degenerate cases instead of dropping the budget)

    The 1-2-point branch is plotter-specific (the fit tool refuses), so we
    keep it here, but delegate the >=3 case to the shared helper so vertex
    coordinates exactly match the JSON/MD report.
    """
    if log_n.size < 3:
        slope, intercept = (
            np.polyfit(log_n, loss, 1) if log_n.size >= 2
            else (0.0, float(loss.mean()))
        )
        return 0.0, float(slope), float(intercept), None
    a, b, c, log_n_opt = _fit_parabola(log_n, loss)
    # Translate NaN → None for the plotter's "draw a star?" decision.
    if math.isnan(log_n_opt):
        return a, b, c, None
    return a, b, c, log_n_opt


def plot_backbone(
    rows: list[dict],
    backbone: str,
    family: str,
    outdir: Path,
    dpi: int,
    stability_by_run_id: dict[str, float] | None = None,
) -> dict:
    """Generate the 2x2 figure for one (backbone, family) slice. Returns
    a dict with N_opt/D_opt/alpha/beta — useful for downstream comparison
    against isoflop_fit.py results.
    """
    if not rows:
        print(f"# isoflop_plot: no rows for backbone={backbone}, skipping",
              file=sys.stderr)
        return {}

    budgets = sorted({r["budget_flops"] for r in rows})
    # Viridis palette per budget — matched to nanochat's scaling-laws notebook.
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(budgets)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ---- Panel 1: IsoFLOP curves (loss vs N) per budget ----
    ax = axes[0, 0]
    optimal_points = []   # for panels 2 & 3
    for budget, color in zip(budgets, colors, strict=True):
        cells = [r for r in rows if r["budget_flops"] == budget]
        if not cells:
            continue
        cells.sort(key=lambda r: r["n_active_params"])
        n = np.array([c["n_active_params"] for c in cells])
        loss = np.array([c["loss_main"] for c in cells])
        log_n = np.log10(n)
        # loss_stability (added in PR #105) → error bars when present.
        # Cells from before #105 have no stability value; show as plain dots.
        # _filter_rows from isoflop_fit doesn't propagate loss_stability, so
        # the caller passes in a separate {run_id: stability} dict from the
        # raw CSV row.
        stab_map = stability_by_run_id or {}
        stab = np.array([
            stab_map.get(c.get("run_id"), 0.0) for c in cells
        ])

        # Scatter the data points (+ stability error bars when available)
        label = f"C = {budget:.0e}"
        if (stab > 0).any():
            ax.errorbar(n, loss, yerr=stab, fmt="o", color=color,
                        markersize=8, label=label, capsize=3, capthick=1,
                        elinewidth=1)
        else:
            ax.plot(n, loss, "o", color=color, markersize=8, label=label)

        # Parabolic fit (only if >= 3 points)
        a, b, c_, log_n_opt = _fit_parabola_for_plot(log_n, loss)
        if log_n_opt is not None and a > 0:
            opt_n = 10**log_n_opt
            opt_loss = a * log_n_opt**2 + b * log_n_opt + c_
            in_range = log_n.min() <= log_n_opt <= log_n.max()
            # Only DRAW the fit curve when its vertex is inside the data
            # range AND the predicted vertex-loss is within the data's loss
            # range. Otherwise the parabola is dominated by noise (Stage A
            # round 1's 0.008-dex N-spread routinely yields fits with
            # negative loss at the vertex — visually distracting and
            # scientifically misleading). The vertex itself is still
            # recorded for the α/β panels so the structural unfittability
            # surfaces there.
            loss_lo, loss_hi = float(loss.min()), float(loss.max())
            loss_range = loss_hi - loss_lo
            vertex_loss_ok = (
                (loss_lo - 0.5 * loss_range) <= opt_loss <= (loss_hi + 0.5 * loss_range)
            )
            if in_range and vertex_loss_ok:
                log_fit_x = np.linspace(log_n.min(), log_n.max(), 100)
                fit_y = a * log_fit_x**2 + b * log_fit_x + c_
                ax.plot(10**log_fit_x, fit_y, "--", color=color, linewidth=1.5, alpha=0.7)
                ax.scatter([opt_n], [opt_loss], s=200, color=color,
                           marker="*", zorder=5, edgecolors="black", linewidths=1.2)
            optimal_points.append({
                "budget_flops": budget,
                "N_opt": opt_n,
                "loss_at_opt": opt_loss,
                "D_opt": budget / (6.0 * opt_n),  # Chinchilla approximation
                "vertex_in_data_range": in_range,
                "vertex_loss_ok": vertex_loss_ok,
            })

    ax.set_xscale("log")
    ax.set_xlabel("N_active (params)")
    ax.set_ylabel("loss_main")
    ax.set_title(f"IsoFLOP curves — {backbone} / {family}")
    ax.legend(loc="upper left", fontsize=9, title="FLOP budget")
    ax.grid(True, alpha=0.3)
    # Clip y to the actual loss range (+ small padding). The parabolic fits
    # at narrow N-spread can predict large excursions that hide the data;
    # restrict the view to where the cells actually landed.
    all_losses = [r["loss_main"] for r in rows]
    if all_losses:
        lo, hi = min(all_losses), max(all_losses)
        pad = max(0.1 * (hi - lo), 0.05)
        ax.set_ylim(lo - pad, hi + pad)

    # A vertex is "trusted" only when it falls inside the data's N-range AND
    # its predicted loss is plausible (within ±50% of the loss-range). Suspect
    # vertices still get plotted on the α/β panels (so structural unfittability
    # is visible) but with a hollow x marker so the eye doesn't read them as
    # clean signal driving the power-law slope.
    def _trusted(p): return p.get("vertex_in_data_range") and p.get("vertex_loss_ok")
    trusted = [p for p in optimal_points if _trusted(p)]
    suspect = [p for p in optimal_points if not _trusted(p)]

    def _plot_alpha_beta_panel(ax, y_key, color, sym, fit_label, ylabel, title):
        """Shared rendering for the N_opt vs C and D_opt vs C panels."""
        if not optimal_points:
            ax.text(0.5, 0.5, "No optimal points (no valid parabolas)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            ax.set_xlabel("FLOPs (C)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            return float("nan")
        # Scatter trusted (filled o) and suspect (hollow x) separately.
        if trusted:
            tx = np.array([p["budget_flops"] for p in trusted])
            ty = np.array([p[y_key] for p in trusted])
            ax.loglog(tx, ty, "o", markersize=10, color=color,
                      label="vertex inside data range")
        if suspect:
            sx = np.array([p["budget_flops"] for p in suspect])
            sy = np.array([p[y_key] for p in suspect])
            ax.loglog(sx, sy, "x", markersize=12, color=color,
                      markeredgewidth=2.5, alpha=0.6,
                      label="vertex outside data range\n(extrapolated, suspect)")
        # Fit + slope: power-law over ALL points (suspect included), so the
        # slope matches what isoflop_fit reports; the styling makes the
        # caveat visible.
        C = np.array([p["budget_flops"] for p in optimal_points])
        Y = np.array([p[y_key] for p in optimal_points])
        slope = float("nan")
        if len(optimal_points) >= 2:
            log_c = np.log10(C)
            log_y = np.log10(Y)
            slope_, intercept = np.polyfit(log_c, log_y, 1)
            fit_c = np.logspace(log_c.min() - 0.3, log_c.max() + 0.3, 100)
            fit_y = 10**(intercept + slope_ * np.log10(fit_c))
            label_style = "r--" if trusted else "r:"  # dotted when all suspect
            ax.plot(fit_c, fit_y, label_style, alpha=0.7,
                    label=fit_label.format(slope=slope_))
            slope = slope_
        ax.legend(fontsize=9, loc="best")
        ax.set_xlabel("FLOPs (C)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        return slope

    # ---- Panel 2: N_opt vs C ----
    alpha = _plot_alpha_beta_panel(
        axes[0, 1], "N_opt", "#2ecc71",
        sym="o", fit_label="$N_{{opt}} \\propto C^{{{slope:.3f}}}$",
        ylabel="N_opt (params)",
        title="Optimal model size vs compute (α fit)",
    )
    # ---- Panel 3: D_opt vs C ----
    beta = _plot_alpha_beta_panel(
        axes[1, 0], "D_opt", "#e74c3c",
        sym="o", fit_label="$D_{{opt}} \\propto C^{{{slope:.3f}}}$",
        ylabel="D_opt (tokens) [C / (6·N_opt)]",
        title="Optimal training tokens vs compute (β fit)",
    )

    # ---- Panel 4: variance floor (per-seed scatter at the smallest budget,
    #              fixed variant — BASE by default). Mean ± stdev annotation.
    ax = axes[1, 1]
    min_budget = min(budgets)
    base_cells = [
        r for r in rows
        if r["budget_flops"] == min_budget
        and r.get("projector_variant") == "BASE"
    ]
    if len(base_cells) >= 2:
        # Distinguish missing seed from explicit seed=0. _filter_rows uses
        # `_as_float(...) or 0` for the seed column, so seeds that weren't
        # populated also arrive as 0.0; the dance below replaces any None
        # / missing value with "?" in the label without ambiguity.
        def _seed_for_sort(c):
            raw = c.get("seed")
            return float(raw) if raw not in (None, "") else float("nan")
        def _seed_label(c):
            raw = c.get("seed")
            if raw in (None, ""):
                return "s?"
            try:
                return f"s{int(float(raw))}"
            except (TypeError, ValueError):
                return f"s{raw}"
        sort_keys = [_seed_for_sort(c) for c in base_cells]
        labels_unsorted = [_seed_label(c) for c in base_cells]
        losses = [c["loss_main"] for c in base_cells]
        order = np.argsort(sort_keys, kind="stable")
        labels = [labels_unsorted[i] for i in order]
        losses = [losses[i] for i in order]
        x = np.arange(len(labels))
        ax.scatter(x, losses, s=140, color="#9b59b6", zorder=3)
        mean = float(np.mean(losses))
        stdev = float(np.std(losses, ddof=1)) if len(losses) >= 2 else 0.0
        ax.axhline(mean, color="black", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"mean = {mean:.4f}")
        ax.fill_between(
            [-0.5, len(labels) - 0.5],
            [mean - stdev] * 2, [mean + stdev] * 2,
            color="black", alpha=0.07, label=f"±1σ = ±{stdev:.4f}",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("seed")
        ax.set_ylabel("loss_main")
        ax.set_title(
            f"Variance floor — BASE @ C={min_budget:.0e}  "
            f"(rel σ = {stdev/mean*100:.2f}%)"
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5,
                f"Need ≥ 2 BASE @ C={min_budget:.0e} seed replicas\n"
                f"(have {len(base_cells)})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_axis_off()

    fig.suptitle(
        f"Stage A IsoFLOP — {backbone} / {family}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    safe_backbone = backbone.replace("/", "-")
    out_path = outdir / f"isoflop_{family}_{safe_backbone}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"# isoflop_plot: wrote {out_path}")

    return {
        "backbone": backbone,
        "family": family,
        "n_budgets": len(budgets),
        "n_optimal_points": len(optimal_points),
        "alpha": alpha,
        "beta": beta,
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    rows = _load_rows(csv_path)
    regime = None if args.regime == "any" else args.regime

    # Build a {run_id: loss_stability} side-map from the raw rows so the
    # plotter can render error bars without modifying _filter_rows's API
    # (which other tools may import). Cells from before PR #105 have no
    # loss_stability column; treat missing/empty as 0.0 (no error bar).
    def _stab(v):
        try:
            return float(v) if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0
    stability_by_run_id = {
        r["run_id"]: _stab(r.get("loss_stability"))
        for r in rows if r.get("run_id")
    }

    # Determine backbones to plot
    if args.backbone:
        backbones = [args.backbone]
    else:
        backbones = sorted({
            r.get("backbone") for r in rows
            if r.get("status") == "done" and r.get("family") == args.family
            and r.get("backbone")
        })
    if not backbones:
        print(f"# isoflop_plot: no done rows for family={args.family}",
              file=sys.stderr)
        return 1

    results = []
    for bb in backbones:
        filtered = _filter_rows(rows, args.family, bb, regime)
        if not filtered:
            print(f"# isoflop_plot: skip {bb} (no usable rows)", file=sys.stderr)
            continue
        results.append(plot_backbone(
            filtered, bb, args.family, outdir, args.dpi,
            stability_by_run_id=stability_by_run_id,
        ))

    if not results:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
