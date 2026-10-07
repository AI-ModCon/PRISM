---
name: prism-vllm-inference
description: >
  Serve and evaluate PRISM checkpoints through vLLM. Use when exporting a PRISM
  checkpoint for vLLM, registering the PRISM plugin, running the vLLM serve/smoke/
  parity/throughput tools, or writing a multimodal processor. Triggers:
  "vllm serve prism", "export checkpoint for vllm", "prism vllm plugin",
  "vllm_parity", "vllm_smoke", "/v1/prism/ts", "checkpoint_export",
  "install_prism_entry_point".
compatibility: Aurora XPU (frameworks/2025.3.1 = vLLM 0.15.0+xpu). Compute node for serving.
metadata:
  version: "1.0"
  project: prism
  domain: ml-inference
---

# PRISM — vLLM Inference

This skill covers **PRISM's vLLM integration** — the plugin, checkpoint export,
and the `tools/vllm_*.py` harness. For generic vLLM engine behavior and the
Aurora-XPU internals (enforce-eager, spawn-worker plugin re-registration,
`ZE_AFFINITY_MASK` tile slicing, ALCF proxy bypass for localhost), use the
installed **`vllm`** skill and its `references/aurora-xpu.md`. Don't `pip install`
vLLM — it ships in `frameworks/2025.3.1` (0.15.0+xpu).

## Plugin registration

PRISM registers via a setuptools entry point:

```bash
tools/install_prism_entry_point.sh      # registers src.vllm_plugin:register
```

- Model class: `src/vllm_plugin/prism_for_conditional_generation.py`.
- Under spawn workers the plugin must **re-register** the model class — that's
  handled in the plugin; if you see "spawn workers can't find my model class,"
  check the `vllm` skill's Aurora-XPU reference.
- Olmo-3 is an **Olmo-2 alias** in the in-tree registry.

## Checkpoint export

vLLM needs an HF-layout checkpoint (renamed keys + a `config.json`):

```bash
python -m src.vllm_plugin.checkpoint_export \
    --checkpoint <trained_ckpt> \
    --backbone allenai/OLMo-1B-0724-hf \
    --image-encoder google/siglip2-base-patch16-224 \
    --active-modalities image,time_series \
    --out <exported_dir>
```

`checkpoint_export.py` renames `backbone.*` → vLLM layout and writes a
HF-compatible `config.json`. For time-series, gate first with
`tools/vllm_check_ts_checkpoint.py` (confirms `encoders.time_series.*` keys
exist).

## The tool harness (`tools/vllm_*.py`)

| Tool | Purpose |
|------|---------|
| `vllm_smoke.py` | Registration + engine boot + 4-token gen (gates every vLLM PR) |
| `vllm_serve.py` | OpenAI-compatible server; pre-registers PRISM + custom `/v1/prism/ts` route |
| `vllm_serve_smoke_client.py` | Verify boot + `/v1/models` |
| `vllm_eval.py` | Image+text eval via PagedAttention |
| `vllm_throughput.py` | tokens/sec sanity |
| `vllm_parity.py` | Compare demo `UnifiedTransformer` path vs vLLM (greedy) |
| `vllm_parity_assert.py` | Assert vLLM output matches a frozen golden (catches refactor regressions) |
| `vllm_ts_smoke.py` / `vllm_ts_serve_smoke.py` | Time-series end-to-end (export → boot → POST tensor) |
| `vllm_check_ts_checkpoint.py` | Pre-flight: TS encoder keys present |

Validated throughput (n=50, max_tokens=64): image+text (OLMo-1B + SigLIP2)
~3,336 tok/s; time-series (OLMo-1B + linear) ~4,036 tok/s.

## Multimodal processors (`src/vllm_plugin/processors/`)

One processor per modality (`image.py`, `time_series.py`, + `base.py`,
`orchestrator.py`, `registry.py`). Each implements `num_tokens`, `encode`,
`dummy_item`, `field_config`, `normalize_mm_data_key`, `build_encoder`. Image uses
`PromptReplacement` (keyed off image size); time-series uses `PromptUpdateDetails`
with a `[start, *[ts_id]*N, end]` envelope. See the `vllm` skill's
`references/multimodal-processor.md` for the general processor API.

## See also

- Deep reference: [`docs/evaluation/inference_vllm.md`](../../evaluation/inference_vllm.md).
- [prism-evaluation](../prism-evaluation/SKILL.md) (eval/parity), [prism-platforms](../prism-platforms/SKILL.md).
- Generic installed skill: `vllm` (engine + Aurora-XPU internals).
