# vLLM-Backed Inference

**Last updated**: May 25, 2026

PRISM ships an in-tree vLLM plugin so trained checkpoints serve through
vLLM's batched PagedAttention engine instead of `model.generate()`. On
Aurora XPU this yields **3,336 tok/s for image+text** (PR #41) and
**4,036 tok/s for time-series** (PR #87) on a single tile.

The plugin lives at `src/vllm_plugin/`. Tools that drive it live at
`tools/vllm_*.py`. This doc covers what's supported today, how to
export a checkpoint, how to run eval, and the latest measured
throughput numbers.

---

## What's supported

| Modality | Status | Lands |
|----------|--------|-------|
| `image` + text | ✅ Production | PR #41 (2026-05-15) |
| `time_series` + text | ✅ Smoke + speedup validated | PR #83 / #85 / #86 / #87 (Stage A + B) |
| `geometry` (Walrus) | 🚧 Plan §VLLM-8 | — |
| `dna` (BioReason) | 🚧 Plan §VLLM-10 (waits for BioReason) | — |
| Multi-image / video | ❌ Out of scope today (`limit_mm_per_prompt={"image": 1}`) | — |

The full rollout plan (2026-05-24, `§VLLM-*` section numbers referenced in
this table) has been retired now that the work has landed; see `git log` for it.

vLLM is shipped inside `frameworks/2025.3.1` on Aurora (version
`0.15.0+xpu`). **Do not pip install** — use the module's Python.

---

## Quick start

### 1. Install the entry point (one-time per env)

```bash
module load frameworks/2025.3.1
bash tools/install_prism_entry_point.sh
```

This writes the dist-info metadata that lets vLLM auto-discover
`src.vllm_plugin:register` via the standard `vllm.general_plugins`
setuptools entry-point group. No more `PYTHONPATH` hack in the runner
scripts (PR #82).

### 2. Export a checkpoint

The training-side `model.safetensors` uses PRISM's naming
(`backbone.*`, `encoders.<m>.*`, `projectors.<m>.*`). vLLM needs a
slightly different key layout plus an HF-compatible `config.json`.
`src/vllm_plugin/checkpoint_export.py` does both.

```bash
# Image-only (PR #41 layout — still the path for OLMo-1B + SigLIP2)
python -m src.vllm_plugin.checkpoint_export \
    --checkpoint outputs/SMOKE-TEST/.../checkpoints/step_500 \
    --backbone allenai/OLMo-1B-0724-hf \
    --image-encoder google/siglip2-base-patch16-224 \
    --out exported/prism-olmo1b-image \
    --language-model-arch OlmoForCausalLM

# Multi-modality (VLLM-1.5+): image + time_series
python -m src.vllm_plugin.checkpoint_export \
    --checkpoint outputs/<run>/checkpoints/step_<N> \
    --backbone allenai/OLMo-1B-0724-hf \
    --image-encoder google/siglip2-base-patch16-224 \
    --active-modalities image,time_series \
    --ts-encoder-type linear --ts-num-vars 1 --ts-max-length 32 \
    --ts-start-id 50278 --ts-end-id 50279 \
    --language-model-arch OlmoForCausalLM \
    --out exported/prism-olmo1b-image-ts
```

`--ts-start-id` / `--ts-end-id` must be token IDs that already exist in
the pretrained LM's vocab (their embed_tokens row exists). For OLMo-1B
the vocab is 50,280 tokens — pick low-traffic IDs like
`|||PHONE_NUMBER|||` (50278) and `<|endoftext|>` (50279) for synthetic
testing (these are only used as placeholder markers in the prompt
update, never decoded), or the values from training's
`modality_start_end_token_indices["time_series"]` for a real
checkpoint (training resizes embed_tokens at startup so the model has
rows for those IDs).

### 3. Run eval

#### Image

```bash
bash tools/_vllm_eval_runner.sh \
    --model exported/prism-olmo1b-image \
    --image test_images/ \
    --prompt "The image shows" \
    --limit 5
```

#### Time-series

```bash
bash tools/_vllm_eval_ts_runner.sh \
    --vllm-model exported/prism-olmo1b-image-ts \
    --demo-checkpoint outputs/<ts-run>/checkpoints/step_<N> \
    --mode both --n 50 --max-tokens 64
```

`--mode both` runs both vLLM and HF `UnifiedTransformer.generate`,
prints per-path tok/s and req/s, and asserts the speedup ratio against
the plan's ≥5× bar.

### 4. Serve

For the OpenAI-compatible server (used by lmms-eval-style harnesses):

```bash
# Note: the install_prism_entry_point.sh step above unblocks PYTHONPATH-free use.
# Without it, tools/vllm_serve.py still works via the imperative register() fallback.
bash tools/_vllm_serve_smoke_runner.sh \
    --vllm-model exported/prism-olmo1b-image \
    --port 8000
```

---

## Throughput

All numbers measured on Aurora (single XPU tile of an Intel Max Series
PVC) with `frameworks/2025.3.1` and `enforce_eager=True`. Stage A
(image+text) is the PR #41 production path; Stage B (time-series) ran
on PR #87.

### Image + text — OLMo-1B + SigLIP2-base-patch16-224

| n | max_tokens | Wall | tok/s | req/s |
|---|---|---|---|---|
| 50 | 64 | — | **3,336** | 52 |

Source: [`project_vllm_backend`](https://github.com/AI-ModCon/BaseMM_PRISM/pull/41) memory entry, 2026-04-29.

### Time-series — OLMo-1B + linear encoder (synthetic), T=32, V=1

| n | max_tokens | Path | Wall | tok/s | req/s |
|---|---|---|---|---|---|
| 10 | 32 | vLLM | 0.37s | **860** | 27 |
| 10 | 32 | HF (sequential) | 4.88s | 58 | 2 |
| 50 | 64 | vLLM | 0.79s | **4,036** | 63 |
| 50 | 64 | HF (sequential) | 49.09s | 62 | 1 |

**Speedup**: 14.6× at n=10, **65.0× at n=50**. Plan §VLLM-6 bar (≥5×)
cleared by 13×.

Source: PR #87, PBS job 8507191 on `x4218c0s3b0n0`, 2026-05-25.

**Synthetic-checkpoint caveat**: the ts numbers above use a random-
weight synthetic checkpoint built by
[`tools/build_synthetic_ts_checkpoint.py`](../../tools/build_synthetic_ts_checkpoint.py).
The speedup is dominated by infrastructure (vLLM's PagedAttention
batching + KV cache vs HF's per-request `generate` loop), not by
encoder cost. The ratio will compress on a real ts training checkpoint
where the encoder forward dominates the request budget, but should
still clear the ≥5× bar comfortably. Re-run with `--vllm-model <real>
--demo-checkpoint <real>` once a real ts checkpoint lands.

---

## Architecture

The model class is `src/vllm_plugin/prism_for_conditional_generation.py`.
At engine boot it reads `active_modalities` from the exported
`prism_config` and instantiates one encoder + projector per modality
into `self.encoders: nn.ModuleDict` and
`self.multi_modal_projectors: nn.ModuleDict`. Image keeps a back-compat
alias (`self.vision_tower = self.encoders["image"]`) so PR #41's
exported weight keys still load.

Per-modality processors live in `src/vllm_plugin/processors/`:

```
processors/
├── base.py            ModalityProcessor ABC (data + encoder hooks)
├── image.py           ImageModalityProcessor (SigLIP2-based)
├── time_series.py     TimeSeriesModalityProcessor (Moirai / linear)
├── orchestrator.py    PrismMultiModalProcessor + DataParser + DummyBuilder
└── registry.py        MODALITY_PROCESSORS factory map
```

Each processor implements `num_tokens`, `encode`, `dummy_item`,
`field_config`, `normalize_mm_data_key`, and `build_encoder`. The
orchestrator's `_build_prompt_update` dispatches to the right
`PromptUpdate` shape (image uses `PromptReplacement` keyed off image
size; time-series uses `PromptUpdateDetails` with the
`[start, *[ts_id]*N, end]` envelope when start/end IDs are configured,
plain `PromptReplacement([ts_id]*N)` otherwise).

---

## Known limitations

- **Time-series envelope on synthetic checkpoints**: the
  `PromptUpdateDetails(full=[ts_start_id, ...feat..., ts_end_id])` path
  fails vLLM's text-fallback matcher when the placeholder
  `<time_series>` token is post-vocab (id ≥ vocab_size). The
  no-envelope path (plain `PromptReplacement(target=[ts_id])`) works
  fine — and is what the PR #87 throughput numbers use. The envelope
  becomes usable when a real ts training checkpoint exports envelope
  IDs in the pretrained vocab range (training resizes embed_tokens; the
  synthetic doesn't).
- **`torch.compile` not viable** on Aurora XPU — runners pass
  `enforce_eager=True`. See the Troubleshooting table in `README.md`.
- **Single-image-per-prompt**: `limit_mm_per_prompt={"image": 1}` in
  every runner today. Multi-image + video are out of scope per plan §7.
- **Aurora module pin**: vLLM 0.15.0+xpu is in `frameworks/2025.3.1`.
  Older `frameworks/2025.2.0` shipped 0.10.1rc2 and doesn't have the
  plugin APIs we use. Don't downgrade.

---

## Smoke / parity gates

Every vLLM plugin PR runs three gates before merge:

| Gate | Script | Asserts |
|------|--------|---------|
| Registration + boot + 4-tok generate | `tools/_vllm_smoke_runner.sh` | `PrismForConditionalGeneration` in `ModelRegistry`; `LLM(...).generate("hello")` returns non-empty |
| Image parity vs PR #41 reference | same runner | ≥18/20 first greedy tokens match (frozen tokens for `cat.jpg` baked into runner) |
| Spawn-worker entry-point | `tools/_vllm_serve_smoke_runner.sh` | Server boots with `PYTHONPATH unset`; `/v1/models` lists the PRISM model |

Optional gates:

| Gate | Script |
|------|--------|
| Time-series checkpoint key presence | `tools/vllm_check_ts_checkpoint.py --checkpoint <dir>` |
| Time-series end-to-end | `tools/vllm_ts_smoke.py --vllm-model <dir>` |
| Time-series throughput vs HF | `tools/_vllm_eval_ts_runner.sh --mode both --n 50` |

Run on a Lustre-only hold (`tools/hold_vllm_test_lustre.sh`) — vLLM
smokes only read from `/lus/flare`, so `daos_user_fs` is unnecessary
and strands jobs when DAOS is down.
