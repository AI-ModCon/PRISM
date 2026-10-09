---
name: prism-evaluation
description: >
  Evaluate PRISM checkpoints. Use when inspecting training/eval data, running the
  universal evaluator, running a benchmark suite, or checking vLLM parity against
  the reference path. Triggers: "evaluate the checkpoint", "universal_evaluator",
  "run_eval / inspect_train", "vllm parity", "benchmark suite", "BLEU / accuracy
  / MAE per modality", "visualize eval samples".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Evaluation

Two layers: **data/qualitative inspection + quantitative benchmarks** (the
universal evaluator) and **inference-parity checks** (vLLM vs the reference
path). Serving/exporting itself is prism-vllm-inference.

## Universal evaluator

```bash
# On a compute node (via the env-extracting wrapper)
bash tools/run_evaluator.sh          # defaults: inspect_train, limit 5

# Direct
python tools/universal_evaluator.py \
    --mode run_eval --checkpoint <exported_dir> --limit 100
```

| `--mode` | What |
|----------|------|
| `inspect_train` | StreamingMultimodalDataset batches; 1×5 composite viz (Image/Graph/Table/TimeSeries/Geometry) per sample |
| `inspect_eval` | One representative example per modality (ChEBI-20, Time-MMD, Spider, VQAv2, MatBench) |
| `run_eval` | Full suite across benchmarks |

Other flags: `--limit N`, `--visualize`, `--viz_dir`, `--exhaustive`.
`run_evaluator.sh` extracts the packed venv then invokes the evaluator (compute
node).

## Metrics per modality

BLEU (Graph), Accuracy (Time / Table / Vision), MAE (Geometry). See
[`docs/evaluation/evaluation.md`](../../evaluation/evaluation.md) for the benchmark table + status.

## vLLM parity harness

Confirm the vLLM serving path matches the reference `UnifiedTransformer` path —
run after any refactor that touches the model or plugin:

```bash
# Compare demo path vs vLLM (greedy)
python tools/vllm_parity.py --checkpoint <ckpt> --vllm-model <exported> --image <img>

# Assert against a frozen golden (regression gate)
python tools/vllm_parity_assert.py --vllm-model <exported> --image <img> \
    --ref-text "<golden>" --window 20 --min-match 18
```

`vllm_eval.py` / `vllm_eval_time_series.py` run batched evals; `vllm_throughput.py`
gives tok/s. See prism-vllm-inference for the full tool list and export flow.

## See also

- Deep references: [`docs/evaluation/evaluation.md`](../../evaluation/evaluation.md),
  [`docs/evaluation/inference_vllm.md`](../../evaluation/inference_vllm.md).
- [prism-vllm-inference](../prism-vllm-inference/SKILL.md), [prism-scaling-and-isoflop](../prism-scaling-and-isoflop/SKILL.md).
