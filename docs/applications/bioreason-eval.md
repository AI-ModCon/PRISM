# BioReason KEGG evaluation

Evaluates a BioReason SFT or GRPO checkpoint against the KEGG test split.

> [!IMPORTANT]
> `eval_kegg.py` is **not yet on `main`** — it ships with the eval-pipeline PR
> (#133). `scripts/submit_eval_kegg.sh` already invokes it, so both this page
> and that script only work once #133 lands. The invocation below is recorded
> so the arguments are documented; it will not run against a clean checkout
> today.

## Invocation

Paths come from the `PRISM_*` site variables — see
[platforms/site_paths.md](../platforms/site_paths.md). Set `PRISM_OUTPUT_ROOT`
to wherever your training run wrote its checkpoints.

```bash
RUN="$PRISM_OUTPUT_ROOT/GENOME-SFT/<run-timestamp>"

python3 eval_kegg.py \
  --checkpoint_dir "$RUN/checkpoints/step_500" \
  --results_dir "$RUN/eval_kegg" \
  --device auto \
  --truncate_per_side 1024 \
  --max_dna_tokens 768 \
  --max_new_tokens 256
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint_dir` | — | Checkpoint to evaluate; a `step_*` directory from the training run |
| `--results_dir` | `<checkpoint_dir>/../eval_kegg/<model_type>` | Where predictions and the summary are written |
| `--device` | `auto` | `cuda` / `xpu` / `cpu` / `auto` — auto picks the first available |
| `--truncate_per_side` | `1024` | Nucleotides kept either side of the variant |
| `--max_dna_tokens` | `1024` | Cap on DNA tokens handed to the projector |
| `--max_new_tokens` | `256` | Generation budget per answer |
| `--max_examples` | full dataset | Cap on eval examples |
| `--temperature` | `0.0` | `0` is greedy; pair with `--do_sample` to sample |

`--backbone_id`, `--model_type`, `--top_p`, `--do_sample` and the `--wandb_*`
options are also accepted; run `eval_kegg.py --help` once #133 lands.

For the training stages that produce these checkpoints, see
[bioreason_grpo.md](bioreason_grpo.md).
