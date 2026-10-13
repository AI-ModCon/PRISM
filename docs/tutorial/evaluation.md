# 4. Evaluation

The full `SCITS-FORECAST` experiment enables evaluation logging every 250
steps, but the current path does **not** produce a numeric held-out forecast
score. Under the HSDP strategy used by the launch recipe, the native trainer
skips its rank-0-only held-out forward to avoid unmatched collectives and
records the running training loss instead. That training loss includes the
masked quantile objective, but is not a validation result.

Switching to DDP does not fix the numeric validation gap yet: the native
time-series validation loader currently reads only `ts.npy` and `text` from
each held-out sample. It does not pass `.gt_ts.npy` or `.gt_ts_mask.npy` to the
model, so that forward cannot score the forecast head against numeric truth.

## Text evaluator limitation

`applications/timeseries/launch_capacity_intern_s2_397b.sh` appends
`tools/universal_evaluator.py --mode verify_timeseries_scits` after training.
That evaluator also reads only the time-series history and `.text` prompt /
answer; it does not load `.gt_ts.npy` or `.gt_ts_mask.npy`. SciTS-forecast
examples carry forecast truth in those numeric members, so the text evaluator
is not a numeric forecast score. This branch does not yet provide a separate
post-training numeric metric CLI.

For now, the logged training loss can confirm that optimization is progressing,
but it cannot establish held-out forecast quality. A numeric MAE or quantile
score requires an evaluator that runs the checkpoint on `val_shards/` and
consumes predictions together with the target mask.

Back to the [tutorial index](README.md).