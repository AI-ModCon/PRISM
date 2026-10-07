# Patrick TSQA A/B — DDP vs HSDP on interleaved OLMo-1B (Aurora, 2026-07-03)

## Question

Patrick's TSQA workflow (`prism_olmo1b_linear_interleaved_ts` + `zone_a_ts`,
seq=4096, projector-only training, interleaved modality routing) has been
run only on 1 node (world_size=12) on Aurora. Recent PR #122 / #124 landed
memory-cliff fixes for the interleaved-QA path. Question: **should Patrick
switch from DDP to HSDP for multi-node runs, and what's the best-tuned
config?**

## TL;DR

- **HSDP dominates DDP by 4–7×** at every scale we tested.
- **HSDP scaling is flat** (2N→4N: 40.4→41.6 samp/s) — comm-bound at
  projector-only trainable footprint.
- **DDP scales positively** (2N→4N: 6.8→11.4 samp/s) but stays 3–6× behind
  HSDP.
- **Modest HSDP tuning headroom** — bumping BS=1→2 and turning off (wasted)
  grad checkpointing gets us to **51.2 samp/s peak (+27%)** vs the 40.4
  baseline. BS≥3 hits the tile ceiling at seq=4096 due to lm_head logits +
  HSDP all-gather buffers.
- **Two latent bugs found en route:** (1) a launcher regression from
  PR #116 that broke all multi-node batch jobs (fixed independently on
  main by PR #125 while this investigation was running), (2) a missing
  auto-detect signal that made E2E interleaved DDP crash at multi-node
  (fix in this branch).

## Config under test

- Model: `prism_olmo1b_linear_interleaved_ts` (OLMo-1B, `<ts>`=50280,
  `is_interleaved_qa: true`, 2 modalities: text + time_series)
- Training: `zone_a_ts` (BS=1 by default, `freeze_llm: true`,
  `freeze_vit: true` — **projector-only trainable, backbone frozen**)
- Data: `per_modality_smoke/text_ts` (local shards at
  `/flare/ModCon/ngetty/data/zone_a/ts_qa/align_256`)
- Tokenizer: `/flare/ModCon/ngetty/BaseMM_PRISM/tokenizers/prism-olmo-1b-interleaved`
- `--max-seq-length 4096`, `--max-steps 150`, `--benchmark-mode`,
  `--prism-disable-perf-probes`, `--grad-norm-interval 0`

## Results

Full throughput matrix, 2 nodes and 4 nodes, seq=4096, 150 steps:

| # | strategy | nodes | BS | grad_ckpt | throughput (final) | peak mem | loss@150 | outcome | job |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DDP (fixed) | 2 | 1 | on | 6.8 samp/s | 22.9 GB | 0.79 | ✅ | 8643008 |
| 2 | HSDP (baseline) | 2 | 1 | on | 40.4 samp/s | 15.8 GB | 0.80 | ✅ | 8642991 |
| 3 | HSDP | 2 | 2 | on | 47.4 samp/s (peak 48.9) | 28.5 GB | 0.81 | ✅ | 8643152 |
| 4 | **HSDP (tuned)** | 2 | 2 | off | **47.4 samp/s (peak 51.2)** | 28.5 GB | 0.78 | ✅ | 8643172 |
| 5 | DDP (fixed) | 4 | 1 | on | 11.4 samp/s | 21.4 GB | 0.67 | ✅ | 8643009 |
| 6 | HSDP (baseline) | 4 | 1 | on | 41.6 samp/s (peak 48.5) | 14.4 GB | 0.82 | ✅ | 8642993 |
| — | HSDP | 2 | 4 | off | — | — | — | ❌ OOM backward | 8643110 |
| — | HSDP | 2 | 8 | on | — | — | — | ❌ OOM lm_head fwd | 8643139 |
| — | HSDP | 2 | 8 | off | — | — | — | ❌ OOM lm_head fwd | 8643093 |

Headline comparisons:

| comparison | speedup |
|---|---|
| 2N HSDP baseline vs 2N DDP | 5.9× |
| 2N HSDP tuned vs 2N DDP | 7.0× (peak 7.5×) |
| 4N HSDP baseline vs 4N DDP | 3.6× |
| HSDP tuning (BS=2 gc=off vs BS=1 gc=on) | +17% final, +27% peak |

## Why HSDP wins so much

Backbone is frozen (`freeze_llm=true`, `freeze_vit=true`), so only the
tiny projector is trainable. Under DDP+interleaved:

- `find_unused_parameters=True` is required for the interleaved modality
  routing (see fix below). Per-forward param scan across the 1B backbone
  is ~90% of step time even with skip-frozen ignore lists.
- `static_graph=False` means no bucket reuse optimization.
- Result: bwd is 88–94% of step time. See timing lines in jobs 8643008/9.

Under HSDP with `shard_grad_op` + `no_sync_accum`:

- Shard-grad-op keeps parameters unsharded at forward (no all-gather
  overhead) and only shards gradients — perfect fit for our comm profile.
- `no_sync_accum` skips grad reduce on non-final microbatches (moot here at
  ga=1 but the flag is validated by scaling-study REPORT.md).
- No `find_unused_parameters` scan.
- Bwd is 55–77% of step time.

## Why the HSDP BS ceiling is BS=2 at seq=4096

BS=8 grad_ckpt=on OOM'd at the exact same line as BS=8 grad_ckpt=off:
`transformers/models/olmo/modeling_olmo.py:450 logits = self.lm_head(...)`
with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.

- logits tensor: `BS × seq × vocab_size × dtype = BS × 4096 × 50304 × 2B`.
  At BS=8, that's ~3.3 GB — cheap in isolation.
- But under HSDP with `shard_grad_op`, all backbone parameters are gathered
  during forward, and the moment the LM head runs there is a transient
  spike combining: unsharded params (~2 GB), full-size grad buffers under
  `no_sync_accum`, the lm_head projection scratch, and the logits.
- The reported "Peak: 15.8 GB" is a *net* value at a stable moment — real
  transient peaks are much higher.
- BS=4 grad_ckpt=off died at backward, not fwd, because keeping all-16
  layers of activations resident (`freeze` means no gradients but forward
  activations still fill under gc=off) pushed us past the tile.

Bigger latent win: **reduce `--max-seq-length`.** Patrick's ts_qa cap
(PR #122) keeps merged length under ~3900 anyway; 4096 is over-provisioned.
Halving seq would move the lm_head bottleneck and unlock BS=4-8.

## Recommended production config for 2N+ HSDP TSQA

```
--dist-strategy hsdp \
--fsdp-sharding shard_grad_op \
--fsdp-no-sync-accum \
--grad-ckpt-freq 0 \
training.batch_size=2
```

For 1N, either strategy works; DDP is still Patrick's default and matches
his validated workflow.

## Bugs found and fixed

### 1. `tools/launch_aurora_web.py:817` — apostrophe in mpiexec heredoc comment

PR #116 (2026-06-20) added the CCL re-export block inside the mpiexec
`bash -lc '...'` heredoc, with a comment reading
`Mirrors launch_aurora_daos.py's post-module-load block`. The apostrophe
in `daos.py's` closes the single-quoted heredoc. Bash parses the truncated
result: `bash -n <generated>` errors with
`unexpected EOF looking for matching '`. Only rank 0 starts;
`TCPStore` rendezvous times out after 601 s with `1/48 clients joined`.
Symptom looks like a hang, is actually a silent multi-node kill.

**Why not caught for 5 weeks:** no batch job had used
`launch_aurora_web.py --batch` since May 29 (scaling-study runs went
through `hold_nodes.sh` + direct mpiexec instead). Same class of bug as
[PR #33](https://github.com/AI-ModCon/BaseMM_PRISM/pull/33) fixed in the
smoke harness in May.

**Fix:** landed independently on main by PR #125 (2026-07-03) while this
investigation was in flight. The independent fix uses the same
"apostrophe-free" wording strategy. Suggested follow-up (not yet done):
add `bash -n <generated_pbs_script>` as a pre-submit self-check to prevent
the same class of bug from recurring silently.

### 2. `src/training/distributed.py::_resolve_find_unused` — missing interleaved signal

E2E DDP training with `is_interleaved_qa=True` crashes on the first
`loss.backward()` at multi-node with:

```
RuntimeError: Your training graph has changed in this iteration, e.g.,
one parameter is unused in first iteration, but then got used in the
second iteration. this is not compatible with static_graph set to True.
```

Root cause: under interleaved training, each rank routes its batch through
one of several modality-specific sub-branches (image encoder / TS encoder /
text-only). Per-rank sampling picks different modalities on different
microbatches, so DDP sees "param X unused in step 0, used in step 1" —
which `static_graph=True` forbids.

The pre-existing `_resolve_find_unused` auto-detect only flipped
`find_unused=True` in the projector-only regime (trainable < 1 GB). E2E
multi-modality with `freeze_backbone: false` was never covered, but the
underlying failure mode also affects Patrick's projector-only shape at
multi-node because the sampling variance widens with rank count.

**Why not caught earlier:** Patrick has only run TSQA on **1 node
(world_size=12)** on Aurora — at 12 ranks with BS=1 and mixed sampling,
the modality-routing variance is small enough that every rank tends to see
every active modality in early steps, so the "graph changed" trigger
doesn't fire. At 48 ranks (4 nodes × 12) the variance widens and the
crash becomes deterministic.

**Fix:** added `is_interleaved` as a third auto-detect signal in
`_resolve_find_unused`. When the model config has `is_interleaved_qa=true`,
DDP now correctly picks `find_unused_parameters=True, static_graph=False`.
19/19 tests pass (4 new tests + 15 existing updated). Log line confirms:

```
[DDP] find_unused_parameters=True, static_graph=False,
gradient_as_bucket_view=True (interleaved-QA auto-detect: dynamic per-batch
modality routing breaks static_graph=True at multi-node scale)
```

**Reproduction:** job 8642992 (AB-4N-DDP-TSQA, pre-fix) — all 48 ranks
crashed simultaneously at step 0 backward.

**Callers updated:** `src/training/trainer_zone_a.py` (Accelerate DDP
path).

### 3. `src/conf/model/prism_olmo1b_linear_interleaved_ts.yaml` — hardcoded Perlmutter tokenizer path (workaround only)

Config points at
`/global/cfs/cdirs/amsc002/pemami/BaseMM_PRISM/tokenizers/prism-olmo-1b-interleaved`
which does not exist on Aurora. Any Aurora user of this model gets:

```
RuntimeError: Could not load tokenizer '/global/cfs/.../prism-olmo-1b-interleaved'
with local_files_only=True.
```

**Workaround for this A/B:** override
`model.tokenizer_id=${PRISM_TOKENIZERS}/prism-olmo-1b-interleaved` in the
design entry. **Not committed.** Since this A/B ran, the tokenizers are
committed under `tokenizers/` in the repo and `PRISM_TOKENIZERS` resolves
the directory — see [../platforms/site_paths.md](../platforms/site_paths.md).
The original run used an absolute path to one user's copy.

## Artifacts

- Logs: `logs/PRISM-OLMO-1B-TSQA/*.OU` (job IDs in the table above)
- Local scratch design entry: `experiments/prism_designs.yaml`
  `PRISM-OLMO-1B-TSQA` — **do not commit**
- Companion hold-script: `tools/hold_2n_debug_ab.sh` — one-off, not
  committed
