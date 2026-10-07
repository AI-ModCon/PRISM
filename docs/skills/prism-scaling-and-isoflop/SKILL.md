---
name: prism-scaling-and-isoflop
description: >
  Measure PRISM throughput and run scaling / IsoFLOP studies. Use when
  benchmarking samp/s, running the per-modality sweep, executing the IsoFLOP
  pipeline (calibrate → plan → launch → collect → fit → plot), or reasoning about
  MFU / scaling efficiency. Triggers: "throughput sweep", "isoflop", "samp/s",
  "scaling efficiency", "MFU", "benchmark_throughput", "run_sweep",
  "compile storm", "N_opt / compute-optimal", "calibration json".
metadata:
  version: "1.0"
  project: prism
---

# PRISM — Scaling & IsoFLOP

Two related workflows: **measuring throughput** (per-modality sweep, benchmarks)
and **compute-optimal scaling** (the IsoFLOP pipeline). Before trusting any
scaling exponent, confirm the data pipeline is honest (prism-data-pipeline /
`ml-data-pipeline-correctness`) and that the run wasn't hit by the intermittent
backward collapse (prism-distributed-strategy).

## Throughput measurement

| Tool | Purpose |
|------|---------|
| `tools/benchmark_throughput.py` | Micro-benchmarks: `--test {forward_only,batch_scaling,fwd_bwd,sync_overhead,ddp_overhead,all}`, `--model {1b,7b}`, `--seq-len` |
| `tools/run_sweep.py` | Per-modality sweep harness (`--preset text_image,text_ts,…`, `--designs PRISM-MODALITY-SMOKE-1N`, `--storage {lustre,daos}`, `--sweep-id`) |
| `tools/perf_aggregate.py` | Scan `outputs/` for `perf.jsonl` → CSV (`--filter key=value`, `--per-modality`) |

The per-modality sweep (`PRISM-MODALITY-SMOKE-1N`, presets in
`experiments/modality_presets.yaml`) runs 5 non-VLA cells + a VLA cell. Watch for
the `WEBDATASET_LOCAL_PATH` leak (one cell's shards bleeding into another) —
fixed in PR #73 but easy to reintroduce; verify per-cell shard paths.

## IsoFLOP pipeline (`tools/isoflop_*.py`)

Run in order:

| Step | Tool | Key args |
|------|------|----------|
| 1. Calibrate | `isoflop_calibrate.py` | `--backbone`, `--projector-variant {BASE,W2X,W4X,D2X,D4X}`, `--family {text_image,text_ts,text_graph}` |
| 2. Plan | `isoflop_plan.py` | `--family`, `--budgets <flops,…>`, `--scaling-study-dir`, `--calibration-dir` |
| 3. Launch | `isoflop_launch.py` | `--plan <yaml>`, `--launcher tools/launch_aurora_daos.py`, `--csv` |
| 4. Collect | `isoflop_collect.py` | `--outputs <root>`, `--csv`, `--eval-window 5`, `--force` |
| 5. Fit | `isoflop_fit.py` | `--csv`, `--family`, `--backbone`, `--regime {projector_only,encoder_projector,e2e,any}`, `--bootstrap 1000`, `--output-json/md` |
| 6. Plot | `isoflop_plot.py` | `--csv`, `--family`, `--regime`, `--outdir` |

Calibration JSONs live in `scaling-study/calibration/`. The projector variant
ladder (BASE/W2X/W4X/D2X/D4X = hidden_mult/num_layers) is the axis that spreads
N; see prism-adding-a-modality for where it's defined.

## Hard-won scaling facts (don't re-derive)

- **PRISM runs at ~6% MFU** (vs AGPT-2B's ~27.6%). The bottleneck is low
  arithmetic intensity (~708 tokens/sample), **not** scaling efficiency. There's
  4–5× single-node headroom before multi-node scaling even matters — fix
  intensity first.
- **Full-shard vs HSDP:** for our comm-bound 0.6B/seq-512 regime, HSDP
  `shard_grad_op` beats FSDP full_shard by ~42–47%. AGPT's 94% full_shard number
  does not transfer.
- **10N throughput plateaus.** Follow-up matrices (E15–E20b) falsified the
  "topology lifts throughput" hypothesis; the 20N mechanism is CPU dispatch, not
  AllReduce. Production launcher config is unchanged; only MNIC=global+cxi naming
  gave +1.4%.
- **Compile storm is scale-dependent** (10N+, not 2N); static==dynamic; not
  caused by gradient checkpointing (inert at GCF=1).
- **torch 2.13 vs 2.10:** wins at ≤4N (1N +37%, 2N +33%, 4N +14%), ~break-even at
  10N (gain decays with N per Amdahl, compute-path only).
- **Round-1 IsoFLOP fit was underdetermined by N-spread** (structural, not
  noise): BASE@3e17 stdev≈0.012, between-variant signal ~11× noise. Need 7B+32B
  backbones to spread N.

## See also

- Deep references: [`docs/results/scaling_study.md`](../../results/scaling_study.md),
  [`docs/results/per_modality_sweep.md`](../../results/per_modality_sweep.md).
- [prism-distributed-strategy](../prism-distributed-strategy/SKILL.md), [prism-launching-jobs](../prism-launching-jobs/SKILL.md).
