# 3. Training

The experiment design is `PRISM-OLMO-1B-SCITS-FORECAST` in
`experiments/LLM_timeseries_scaling.yaml`. It uses one Aurora node (12 XPUs),
trains for 5,000 steps, and configures a 720-step, 10-channel head with 0.1,
0.5, and 0.9 quantiles. The OLMo-1B trunk and pretrained time-series
encoder are frozen; PRISM's input projector and randomly initialized
forecast head are trained.

The model preset is `prism_olmo1b_intern_s2_397b_ts_forecast`, with
`backbone_id` and `tokenizer_id` set to `allenai/OLMo-1B-0724-hf` and
`d_text: 2048`. Make sure this checkpoint and tokenizer are present in
`$HF_HOME/hub` before launching; the job stages them and runs offline.

The experiment sets `training.batch_size: 2` per rank and
`training.gradient_accumulation_steps: 16`, giving an effective batch of
384 across 12 ranks. Both `training.learning_rate` and
`training.lr_connector` are `1e-3`. Forecast examples use text as conditioning
only: their text labels are masked, and the forecast head uses masked pinball
loss over valid numeric targets.

## Smoke run

With `VENV_PATH`, `HF_HOME`, and `PRISM_DATA_ROOT` set as described on the
previous pages, preview the 50-step run:

```bash
python tools/launch_aurora_web.py \
    --id OLMO-SCITS-FORECAST-SMOKE \
    --file experiments/LLM_timeseries_scaling.yaml \
    --design PRISM-OLMO-1B-SCITS-FORECAST \
    --nodes 1 \
    --project GenesisHackathonOct26 \
    --batch \
    --queue debug \
    --walltime 00:30:00 \
    --dist-strategy hsdp \
    --fsdp-sharding shard_grad_op \
    --max-steps 50 \
    --webdataset-dir "$PRISM_DATA_ROOT/SciTS-forecast" \
    --webdataset-modality time_series \
    --use-shared-venv \
    --hf-home "$HF_HOME" \
    --shared-hf-home "$HF_HOME/hub" \
    training.warmup_steps=5 \
    training.eval_every_n_steps=25 \
    training.save_every_n_steps=25 \
    --dry-run
```

Check the generated job script for the expected data and checkpoint paths.
Remove `--dry-run` to submit the smoke run through `qsub`. Use
`qstat <job-id>` to check its scheduler state.

## Full run

After the smoke run succeeds, use the same command with these changes:

- Set `--id` to `OLMO-SCITS-FORECAST`; keep `--design` set to
  `PRISM-OLMO-1B-SCITS-FORECAST`.
- Set `--queue capacity` and `--walltime 06:00:00`.
- Remove `--max-steps 50` and the three `training.*` smoke overrides to
  restore the full design's training and checkpoint schedule.
- Remove `--dry-run` to submit.

The full design enables periodic evaluation logging and saves checkpoints
every 500 steps under
`outputs/OLMO-SCITS-FORECAST/<date>/<time>/checkpoints/`. It consumes numeric
targets and their masks; text generation is not the forecast output.

`--project` sets the PBS allocation (`#PBS -A`) and defaults to `ModCon`;
use your own allocation outside the hackathon. See
[evaluation](evaluation.md) for how to check the run and score its
checkpoints.

Next: [evaluate the run](evaluation.md).