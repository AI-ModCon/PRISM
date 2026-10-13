# SciTS Forecasting with Intern-S2 Preview 397B

A short path from environment setup through training with the numeric
`SciTS-forecast` WebDataset split, including the current evaluation boundary.

This configuration combines a **Qwen3-0.6B** language-model trunk, the
pretrained **[Intern-S2 Preview 397B](https://arxiv.org/html/2608.13505) time-series encoder**, and PRISM's own
trainable linear quantile forecast head. It extracts only the time-series
encoder weights from the 397B checkpoint; it does not train or load the full
397B language model, and it does not use Intern-S2's native forecaster.

## Pages

| Page | What it covers |
|---|---|
| [Environment](environment.md) | Aurora environment, encoder checkpoint extraction, model cache |
| [Data](data.md) | Convert SciTS into train/validation shards with numeric targets and masks |
| [Training](training.md) | Smoke and full `SCITS-FORECAST` runs |
| [Evaluation](evaluation.md) | Training-loss signals and the missing numeric held-out evaluator |

You need an Aurora allocation for training and a local SciTS source tree for
conversion. Run the commands from the repository root. Replace path
placeholders with shared paths available on both the login and compute nodes.
