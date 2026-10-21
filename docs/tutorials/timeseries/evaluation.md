# 4. Evaluation

Evaluation has three stages: watch the training loss while the job runs,
collect the checkpoints it saves, then score a checkpoint's forecasts against
the held-out `val_shards/`. Only the last stage measures forecast quality.

| Stage | Where | What it tells you |
|---|---|---|
| 1. Training signals | Job log and `checkpoints/perf.jsonl` | Whether optimization is progressing |
| 2. Checkpoints | `checkpoints/step_<N>/model.safetensors` | Which model states you can score |
| 3. Held-out forecast scoring | [Tutorial notebook](../../../applications/timeseries/olmo_scits_forecast.ipynb), section 4 | Forecast accuracy and calibration on unseen series |

All paths below are relative to the run directory,
`outputs/<run-id>/<date>/<time>/`.

## 1. Training signals

The trainer appends one JSON record per log interval to
`checkpoints/perf.jsonl`. Records with `"site": "trainer_native_per_log"` hold
the step, the global training `loss`, and throughput. Every
`eval_every_n_steps` (250 for the full run), it also writes an `"event": "eval"`
record. That record repeats the current running training loss
(`"loss_source": "train_running_mean"`). It is **not** a held-out score.

The job log breaks the loss down by objective on each logged step, for
example from the smoke run:

```text
Step 20: Loss 1.1451 (rank_mean=1.1852, rank0=2.0761, tokens=44985)
  [LOSS] text=1.3194 (n=3153) | time_series=0.3805 (n=95)
```

- `time_series` is the masked pinball (quantile) loss of the forecast head
  over `n` valid target cells. This is the number to watch for forecasting.
- `text` is next-token loss over `n` answer tokens. SciTS also contains
  non-forecast tasks (classification, multiple choice) whose answers are text;
  forecast samples mask their own text labels, so they do not contribute here.

The headline `Loss` combines both terms, so it can fall mostly because the
text term falls. Judge forecasting progress from `time_series`. A healthy
run shows finite values for both terms from the first logged step.

## 2. Checkpoints

The full design saves a checkpoint every 500 steps (`step_500`, `step_1000`,
…); the smoke run saves `step_25` and `step_50`. Each directory holds
`model.safetensors` plus the optimizer and scheduler state needed to resume.
Only `model.safetensors` is needed for evaluation.

## 3. Held-out forecast scoring

Section 4 of the [notebook](../../../applications/timeseries/olmo_scits_forecast.ipynb) scores one
checkpoint end to end:

1. **Load** – build the model from the
   `prism_olmo1b_intern_s2_397b_ts_forecast` preset and load the
   checkpoint's `model.safetensors`.
2. **Read held-out data** – stream `val_shards/` with the same loader the
   trainer uses. The loader caps the history length, decodes the numeric
   target (`.gt_ts.npy`) and its validity mask (`.gt_ts_mask.npy`), and
   computes each channel's history mean and scale.
3. **Forecast** – run the training forward pass and capture the forecast
   head's de-normalized quantiles (horizon × channels × {0.1, 0.5, 0.9}).
4. **Score** – compare the predictions with the target on valid (unmasked)
   cells only, and report the metrics per sample and per SciTS task.

| Metric | Meaning |
|---|---|
| `mae`, `rmse` | Error of the median forecast, in the series' original units |
| `smae` | MAE divided by the channel's history scale, comparable across tasks |
| `smae_mean`, `smae_last` | The same score for two naive baselines: history mean and last observed value |
| `skill_vs_last` | `1 - smae / smae_last`; positive means the model beats persistence |
| `pinball` | Scaled pinball loss over all quantiles (the training objective) |
| `coverage80` | Fraction of targets inside the 0.1–0.9 band; ≈ 0.8 is well calibrated |

Start with `skill_vs_last` and `coverage80`. A model that does not beat
last-value persistence has not yet learned a useful forecast. A coverage far
below 0.8 means the quantile band is too narrow.

To track progress during training, point the notebook's `EVAL_CHECKPOINT` at
successive `step_<N>` directories and compare the tables. The notebook caps
scoring at `N_EVAL` samples (at most `MAX_PER_TASK` per task) for speed;
raise `N_EVAL` to cover all of `val_shards/` before reporting numbers.

## Known limitations

- There is no command-line numeric evaluator yet; held-out forecast scoring
  lives in the notebook.
- `tools/universal_evaluator.py --mode verify_timeseries_scits` scores the
  **text** answer of each SciTS sample. It ignores the numeric target and mask,
  so it does not evaluate the forecast head.

Back to the [tutorial index](README.md).
