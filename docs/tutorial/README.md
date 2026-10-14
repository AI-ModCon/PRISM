# SciTS Forecasting with Intern-S2 Preview 397B

A short path from environment setup through training and held-out evaluation
on the numeric `SciTS-forecast` WebDataset split.

This configuration combines a frozen **OLMo-1B** language-model trunk, the
frozen pretrained **[Intern-S2 Preview 397B](https://arxiv.org/html/2608.13505) time-series encoder**, and PRISM's own
trainable linear quantile forecast head. It extracts only the time-series
encoder weights from the 397B checkpoint; it does not train or load the full
397B language model, and it does not use Intern-S2's native forecaster.

## Pages

| Page | What it covers |
|---|---|
| [Environment](environment.md) | Aurora environment, encoder checkpoint extraction, model cache |
| [Data](data.md) | Convert SciTS into train/validation shards with numeric targets and masks |
| [Training](training.md) | Smoke and full `SCITS-FORECAST` runs |
| [Evaluation](evaluation.md) | Training signals, checkpoints, and held-out forecast scoring |
| [Notebook](olmo_scits_forecast.ipynb) | The smoke run and checkpoint evaluation, step by step on a debug node |

You need an Aurora allocation for training. Run the commands from the
repository root. Replace path placeholders with shared paths available on
both the login and compute nodes.

> **Genesis hackathon:** the model cache, extracted encoder, converted
> `SciTS-forecast` data, and a trained `step_2000` checkpoint are pre-staged
> under `/flare/GenesisHackathonOct26/scits`. You can skip the encoder
> extraction and data conversion steps and point `HF_HOME` and
> `PRISM_DATA_ROOT` there (see the [notebook](olmo_scits_forecast.ipynb)).
