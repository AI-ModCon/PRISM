"""Plot measured losses from the completed pilot and its resumed training run."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BLUE = "#176B98"
ORANGE = "#C35A17"

evaluations = {}
updates = []
for run in ("pilot", "overfit-500"):
    report = json.loads((ROOT / run / "report.json").read_text())
    assert report["status"] == "completed"
    for row in report["evaluations"]:
        assert row["step"] not in evaluations
        evaluations[row["step"]] = row
    updates.extend(
        json.loads(line)
        for line in (ROOT / run / "steps.jsonl").read_text().splitlines()
        if line.strip()
    )

assert [row["step"] for row in updates] == list(range(1, 501))
assert all(math.isfinite(row["loss"]) for row in updates)
steps = sorted(evaluations)
assert steps == [0, 20, 100, 200, 300, 400, 500]
means = {
    split: [evaluations[step]["splits"][split]["mean_loss"] for step in steps]
    for split in ("train", "validation")
}
assert all(math.isfinite(value) for values in means.values() for value in values)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
for index, ax in enumerate(axes):
    start = 0 if index == 0 else 1
    for split, label, color in (
        ("train", "Training · 16 pairs", BLUE),
        ("validation", "Validation · 8 pairs", ORANGE),
    ):
        ax.plot(steps[start:], means[split][start:], "o-", color=color,
                linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Mean fixed-probe flow loss")
    ax.grid(alpha=0.18)
    ax.set_xlim((-10, 525) if index == 0 else (10, 525))
    ax.set_xticks([0, 100, 200, 300, 400, 500] if index == 0
                  else [20, 100, 200, 300, 400, 500])
    ax.set_title("Complete run" if index == 0 else "Detail after step 20", pad=12)
axes[0].legend(frameon=False, loc="upper right")
axes[0].text(0.04, 0.46,
             f"Training: {means['train'][0]:.4f} → {means['train'][-1]:.4f}\n"
             f"Validation: {means['validation'][0]:.4f} → {means['validation'][-1]:.4f}",
             transform=axes[0].transAxes, fontsize=11,
             bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
for split, color, offset in (("train", BLUE, -15), ("validation", ORANGE, 12)):
    axes[1].annotate(f"{means[split][-1]:.4f}", (500, means[split][-1]),
                     xytext=(-4, offset), textcoords="offset points", ha="right",
                     color=color, weight="bold")
axes[1].margins(y=0.22)
fig.suptitle("PRISM → OmniGen2 · Connector training and validation loss", fontsize=16,
             fontweight="bold", y=0.99)
fig.text(0.5, 0.91,
         "Fixed examples and fixed randomness at each checkpoint; only the output connector is trained",
         ha="center", fontsize=10.5, color="#555555")
fig.text(0.5, 0.02,
         "Measured at steps 0, 20, 100, 200, 300, 400 and 500. Lines connect measurements; lower loss alone does not establish image quality.",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.05, 1, 0.88))
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"train-validation-loss.{extension}", dpi=180,
                facecolor="white", bbox_inches="tight")
plt.close(fig)

batch_steps = [row["step"] for row in updates]
losses = [row["loss"] for row in updates]
window = 20
rolling_steps = batch_steps[window - 1:]
rolling_loss = [sum(losses[i - window + 1:i + 1]) / window
                for i in range(window - 1, len(losses))]
fig, ax = plt.subplots(figsize=(12.8, 4.8))
ax.plot(batch_steps, losses, color=BLUE, alpha=0.22, linewidth=0.9,
        label="Each optimizer update (batch size 1)")
ax.plot(rolling_steps, rolling_loss, color=BLUE, linewidth=2.2,
        label="Trailing 20-update mean")
ax.set(xlabel="Optimizer step", ylabel="Training batch flow loss", xlim=(0, 505))
ax.set_title("Training loss recorded at all 500 optimizer updates", pad=15)
ax.grid(alpha=0.18)
ax.legend(frameon=False)
fig.text(0.5, 0.02,
         "Examples, noise and diffusion timesteps change between updates. Use the fixed-probe plot for consistent train/validation comparisons.",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.06, 1, 1))
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"training-step-loss.{extension}", dpi=180,
                facecolor="white", bbox_inches="tight")
plt.close(fig)

data = {
    "description": "Actual measured losses; validation exists only at the listed checkpoint steps.",
    "fixed_probes": [{"step": step, "train_loss": means["train"][i],
                      "validation_loss": means["validation"][i]}
                     for i, step in enumerate(steps)],
    "optimizer_updates": [{"step": row["step"], "loss": row["loss"]}
                          for row in updates],
    "rolling_mean_window": window,
    "sources": ["pilot/report.json", "pilot/steps.jsonl",
                "overfit-500/report.json", "overfit-500/steps.jsonl"],
}
(ROOT / "training-loss-data.json").write_text(json.dumps(data, indent=2) + "\n")
print("Saved train-validation-loss and training-step-loss (PNG/SVG), and training-loss-data.json")
