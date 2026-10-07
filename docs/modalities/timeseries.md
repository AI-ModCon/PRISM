# Documentation on PRISM's language-timeseries branch

## Dependencies

`soundfile` is an *optional* dependency, needed only if a SciTS source series
has `data_type` `wav` or `flac` (`scripts/convert_scits_to_webdataset.py`
imports it lazily and raises a clear error at conversion time if it's missing
and one of those types is encountered). Most SciTS series are `npy`/`csv` and
don't need it. Install if you hit that error:

```bash
pip install soundfile
```

## Data

Time series is registered as a modality in `src/modalities.py` (`Modality.TIME_SERIES = "time_series"`).

**Datasets** (configured in `src/data/datasets_config.json` and `src/conf/data/lustre_datasets.yaml`):

| Name | Source | Samples | Handler |
|------|--------|---------|---------|
| `ts_qa` | ChatTS Align (Synthetic) | ~105K | `_process_ts_qa` |
| `scits_ts` | SciTS (WebDataset, Lustre) | ~29K | `_process_ts_qa` (via `ts_qa` handler) |
| `ts_caption` | ChatTime / TSQA | ~48K | `_process_ts_caption` |
| `ts_instruction` | TS Reasoning (encoder alignment) | ~16K | `_process_ts_instruction` |
| `ts_weak` | Time-MMD (Economy) | ~3K | `_process_ts_time_mmd` |

`scits_ts` is defined in the `scits` dataset group in `src/conf/data/lustre_datasets.yaml` (29 `.tar` shards at `/lus/flare/projects/ModCon/pemami/data/SciTS-processed`). Each shard sample contains three files: `<key>.ts.npy` (NumPy array, shape `(T, V)`), `<key>.text` (UTF-8 question/answer text in `Question: ...\nAnswer: ...` format), and `<key>.meta.json`.

**Processing pipeline** (in `src/data/multimodal.py`):
- Input: raw series as list, tensor, string, or numpy array (parsed from multiple delimiters)
- Normalization: per-variate mean/std subtraction (when `normalize_ts_in_encoder=False`)
- Shape (linear/moirai path): `(max_ts_length, 1)` — padded/truncated to fixed length (default 512)
- Multivariate (linear/moirai path): flattened to `(T × V, 1)` before encoding
- Interleaved mode (`is_interleaved_qa=True`): injects mean/std statistics between `<ts>` tokens in the text prompt
- SciTS text parsing fallback: when no `instruction`/`question`/`input` key is present, `_process_ts_qa` parses `text` using `Question:` / `Answer:` delimiters.

## Models

**Encoder** (`src/encoders/time_series.py` — `TimeSeriesEncoder`):

Three encoder backends controlled by `ts_projector`:

| Encoder | Supports interleaving multiple instances | Maximum individual-series length | Supports multivariate data | Maximum variates (>1) | Static patching | Dynamic patching | Pretrained weights |
|---------|------------------------------------------|----------------------------------|----------------------------|-----------------------|-----------------|------------------|--------------------|
| `linear` | Yes | `max_ts_length` (default: 512) | Yes | Configurable via `ts_variates` | N | N | N/A |
| `moirai` | Yes | `max_ts_length` (default: 512) | Yes | Configurable via `ts_variates` | Y (`patch_size=16`) | N | [`Salesforce/moirai-2.0-R-small`](https://huggingface.co/Salesforce/moirai-2.0-R-small) |
| `timeomni` | No (exactly one `<ts><ts/>` span) | Flattened `T * V` value count, bounded by `max_ts_length` and the configured patch budget | Yes, serialized into one value stream | Runtime input width | N | Y | N/A |

TimeOmni deliberately serializes each multivariate `(T, V)` tensor in time-major
order into one `(V*T)` value stream, then patches that stream. It supports
multivariate inputs while retaining one fixed feature span per series; patches
are value-stream windows, not guaranteed whole time windows. See [TimeOmni
Dynamic Patching](#timeomni-dynamic-patching) below.

**Projector** (`src/modules/projector.py` — `ModalityProjector`):
- 2-layer MLP: `encoder_hidden → d_model` with GELU activation
- Optional normalization (layernorm, rmsnorm, l2, match_text_stats, etc.)
- Optional learnable modality embedding (scale ~0.02)

## Token interleaving

Implemented in `model.py` → `_merge_text_input_ids_with_modality_embeds`:

1. Text contains `<ts>` / `</ts>` placeholder tokens (IDs configured via `modality_start_end_token_indices`, e.g. `{"time_series": [100278, 100279]}`)
2. At forward time, `<ts>` expands into the full sequence of projected encoder embeddings (e.g., 32 Moirai patch tokens or 512 linear tokens)
3. `</ts>` becomes a separator token
4. Labels are masked (`-100`) for all prompt/timeseries tokens; loss computed only on answer tokens

A custom tokenizer with the special tokens lives at `tokenizers/prism-olmo-3-7b-instruct-interleaved`.

---

## TimeOmni Dynamic Patching

TimeOmni is a variable-patch-size encoder strategy where **patch size is chosen per sample at forward time**, not at collation. This lets the model adapt to wide variation in series length without wasting tokens on short series or truncating long ones.

### Design principles

| Principle | Detail |
|-----------|--------|
| **Batching in the encoder** | Collation skips fixed-length padding (`passthrough_time_series=True`). The encoder receives `list[Tensor(T_i, V)]` with heterogeneous lengths. |
| **Dynamic patch selection** | Each `(T, V)` sample is serialized time-major to one `(V*T)` value stream. `_select_patch_embedding` picks the largest configured `(patch_len, stride)` that keeps its patch count within `timeomni_max_patches`. |
| **Fixed token budget (interleaved mode)** | When `is_interleaved=True`, the encoder left-pads every sample's patch sequence to exactly `tokens_per_instance()` (= `timeomni_max_patches`) tokens, so the interleaved merge contract is stable across the batch. |
| **Fixed token budget (non-interleaved mode)** | Samples are left-padded to the longest encoded sequence in the batch. This padding is not represented in the caller's attention mask, so unequal-length non-interleaved batches remain unsupported. |
| **No reprogramming** | TimeOmni's cross-attention reprogramming layer is excluded. Patch embeddings feed directly into PRISM's existing `ModalityProjector`. |
| **No pre-normalization** | Data loaders preserve raw `(T, V)` tensors; normalization is deferred to the encoder. |

### Patch count formula

`_TimeOmniPatchEmbedding` right-pads by `stride` before `unfold`:

```
num_patches = floor((T + stride − patch_len) / stride) + 1
```

For TimeOmni, `T` in that formula is the serialized value-stream length
`S = time_steps * variates`. A patch can contain values from adjacent
timestamps, so it is not a time-window token. `_select_patch_embedding` keeps
only candidates where `1 ≤ num_patches ≤ timeomni_max_patches`.

**Budget derivation** — for a TimeOmni model, `train.py` derives `max_ts_length` (the data-loader safety ceiling) from the stride paired with the largest configured patch length:

```
max_ts_length = budget_stride × max(1, timeomni_max_patches − 1)
```

Here, `budget_stride` is the stride at the largest `timeomni_patch_len` index. The formula matches the right-padding and unfolding rule when `patch_len == stride`; the encoder still applies the actual candidate-specific patch-count check at forward time. For multivariate input, this ceiling applies to the flattened `T * V` count; the encoder decimates the time axis when needed to preserve the time extent.

### Configuration knobs

All knobs live under `ModelConfig` and are exposed in Hydra YAML:

| Key | Default | Description |
|-----|---------|-------------|
| `timeomni_patch_len` | `16` | Candidate patch lengths (int or list). |
| `timeomni_stride` | same as `timeomni_patch_len` | Strides matching each patch length. |
| `timeomni_d_model` | `512` | Patch-embedding hidden dimension. |
| `timeomni_dropout` | `0.1` | Dropout in patch embedding. |
| `timeomni_ts_tokens` | `100` | Target tokens-per-series used to guide patch-size selection heuristic. |
| `timeomni_max_patches` | `100` | Hard budget: feature tokens for one serialized series, independent of variate count. |

### Data path for SciTS / TimeOmni

`_process_ts_qa` in `src/data/multimodal.py` has a dedicated `is_timeomni` branch:

1. **1D input** (`dim==1`): reshaped to `(T, 1)`, no padding/truncation/normalization.
2. **2D input** (`dim==2`): orientation normalized to `(T, V)` (legacy records often arrive as `(V, T)`).
3. **Budget guard**: in `_forward_timeomni`, checks the flattened multivariate length `V*T` against `max_ts_length`. When over budget, it decimates the time axis by a bounded integer factor using block means, preserving the full time extent while reducing `V*T` to at most the configured ceiling. A sample can still raise `RuntimeError` later if no configured patch setting satisfies the patch budget.
4. **Square tensor warning**: when `T == V`, orientation is ambiguous — a warning is logged and the tensor is left as-is.
5. **Interleaved span enforcement** (`is_interleaved_qa=True`): exactly one `<ts><ts/>` span is required per sample; missing spans are inserted; extra spans raise.


### Key files

| File | Role |
|------|------|
| `src/encoders/time_series.py` | `TimeSeriesEncoder._forward_timeomni`, `_select_patch_embedding`, `_num_timeomni_patches`, `tokens_per_instance()` |
| `src/data/multimodal.py` | `_process_ts_qa` — `is_timeomni` branch preserving raw `(T, V)` and SciTS text fallback |
| `src/data/collate.py` | `MultimodalCollator(passthrough_time_series=True)` |
| `src/train.py` | Budget derivation; BucketedCollator disabled; passthrough flag set |
| `src/training/trainer_native.py` | `_move_batch_to_device` — handles `list[Tensor]` recursively |
| `src/vllm_plugin/processors/time_series.py` | `TimeSeriesModalityProcessor` — `encoder_type="timeomni"` support |

### Tests

| Test file | Coverage |
|-----------|----------|
| `tests/multimodal/test_ts_qa_timeomni_dynamic.py` | `_process_ts_qa` shape preservation, budget handling, span enforcement, orientation transpose, 1D dynamic path |
| `tests/multimodal/test_timeomni_dynamic_batching.py` | Variable-length list input, per-sample patch selection, fixed-budget output (`is_interleaved=True`), infeasible patch rejection |
| `tests/test_collator_timeomni_passthrough.py` | `passthrough_time_series` mode vs. default padding |
| `tests/test_scits_shard_smoke.py` | Opt-in smoke test validating SciTS `.tar` shard integrity and expected payload suffixes |
| `tests/multimodal/test_scits_timeomni_minimal.py` | End-to-end SciTS sample through `_process_ts_qa` -> collator passthrough -> `TimeSeriesEncoder` |
| `tests/test_load_ts_qa.py` | ChatTS fixture path and `_process_ts_instruction` contract through `_process_ts_qa` |
| `tests/multimodal/test_ts_qa_variate_cap.py` | Interleaved variate-cap invariants (issue #120): headroom, 4096-case, `<ts>` pair/tensor-row parity |

### Known limitations

- **Square tensors** (`T == V`): `(T, V)` vs `(V, T)` orientation cannot be determined automatically. Ensure input is already `(T, V)` for these cases.
- **vLLM inference**: `num_tokens` returns `timeomni_max_patches`, matching the one serialized TimeOmni stream and training's fixed token budget. vLLM preserves raw `(T, V)` inputs for TimeOmni; the shared encoder owns decimation and patch selection.
- **No reprogramming**: TimeOmni's vocabulary-remapping cross-attention is not implemented. Patch features go straight to PRISM's MLP projector.
- **Interleaved TimeOmni contract**: exactly one `<ts><ts/>` span per sample is enforced in `_process_ts_qa`.

---

## Training

### ChatTS-Align (synthetic) 

**Config files** in `src/conf/model/`:
- `prism_olmo1b_linear_interleaved_ts.yaml`
- `prism_olmo3_7b_instruct_linear_interleaved_ts.yaml`

**Key settings**:
```yaml
modalities: [text, time_series]
is_timeseries: true
is_interleaved_qa: true
ts_projector: "linear"          # or "moirai"
ts_variates: 1
max_ts_length: 256              # reduced for training efficiency
normalize_ts_in_encoder: false
freeze_backbone: true
freeze_encoders: false          # encoder + projector trained
projector_num_layers: 2
```

- **Loss**: Standard cross-entropy on answer tokens only (prompt/TS embeddings masked)
- **Frozen components**: Backbone LLM is frozen; encoder + projector are trained
- **Data mixing**: Timeseries datasets mixed with other modalities via weighted sampling

#### Interleaved multivariate variate cap (issue #120)

For the linear/moirai interleaved path, each variate row becomes one `<ts><ts/>` span that expands to `max_ts_length` embedding tokens in `_merge_text_input_ids_with_modality_embeds`. If all variates are kept, merged length can exceed `max_seq_length` and trigger OOM.

When `is_interleaved_qa=True` and `StreamingMultimodalDataset.max_seq_length` is set, `_process_ts_qa` applies:

```
text_reserve = max_seq_length // 4
ts_budget = max_seq_length - text_reserve
max_variates = ts_budget // max_ts_length
```

`_TS_QA_TEXT_RESERVE_DIVISOR = 4` reserves 25% headroom for prompt/target text. The cap also truncates matching `<ts><ts/>` placeholders so the invariant `#<ts> pairs == #variate rows` is preserved.

### SciTS

TimeOmni ready-to-use Hydra presets in `src/conf/model/`:

| File | Backbone | `is_interleaved_qa` | `timeomni_max_patches` |
|------|----------|---------------------|------------------------|
| `prism_olmo1b_timeomni_ts.yaml` | OLMo-1B | `false` | `200` |
| `prism_qwen3_0_6b_timeomni_ts.yaml` | Qwen3-0.6B | `false` | `200` |
| `prism_qwen3_1_7b_timeomni_ts.yaml` | Qwen3-1.7B | `false` | `200` |
| `prism_qwen3_4b_timeomni_ts.yaml` | Qwen3-4B | `false` | `200` |
| `prism_qwen3_8b_timeomni_ts.yaml` | Qwen3-8B | `false` | `200` |

Both set `ts_projector: "timeomni"` with candidate patch lengths `[16, 32, 64, 128, 256, 512, 1024, 2048]`.

### Launching a TimeOmni run

SciTS Olmo 1B (non-interleaved):

```bash
python3 tools/launch_aurora_web.py \
	--id PRISM-OLMO-1B-TIMEOMNI-RUN \
	--design PRISM-OLMO-1B-TIMEOMNI \
  --nodes 1 \
  --batch \
  --queue capacity \
  --walltime 06:00:00 \
  --dist-strategy hsdp \
  --fsdp-sharding shard_grad_op \
  --webdataset-dir /flare/ModCon/pemami/data/SciTS-processed \
  --webdataset-modality time_series \
  --wandb-project=YOUR_PROJECT \
  wandb.entity=YOUR_ENTITY \
  training.data_num_workers=1
```

`data_num_workers` is a declared `TrainingConfig` field (`src/config.py`), so it's a
plain override — a leading `+` is only for keys that don't already exist in the
structured config, and Hydra raises an error ("already in struct") if you add it
to one that does.

Change the design to `PRISM-QWEN3-0-6B-TIMEOMNI` for the Qwen3 0.6B backbone.

## Evaluation

**SciTS evaluator** (`src/eval/tasks/modality_tasks.py` — `SciTSEvaluator`, registered as `ts_scits`):

- **Data**: reads validation `.tar` shards from `PRISM_VAL_SHARDS_DIR` (default `/flare/ModCon/pemami/data/SciTS-processed/val_shards`) via stdlib `tarfile`.
- **Process**: decodes `(T, V)` arrays from `.ts.npy`, parses question/answer from `.text`, generates answer text.
- **Metric**: SQuAD-style normalized exact match (headline `accuracy`/`exact_match`) plus token-level F1. An earlier substring rule (`answer in generated` or `generated in answer`) credited any generation that was itself a substring of the answer — SciTS answers are full sentences, so an untrained model emitting "no" or "the" scored as correct.
- Also registered in `tools/universal_evaluator.py` as `("Time (SciTS)", "ts_scits")`.

### Spot checks with universal_evaluator.py

Example command to verify a checkpoint on the first 50 validation samples:
```
export PYTHONNOUSERSITE=1 && unset PYTHONPATH && PRISM_VAL_SHARDS_DIR=/flare/ModCon/pemami/data/SciTS-processed/val_shards python tools/universal_evaluator.py --checkpoint outputs/PRISM-OLMO-1B-TIMEOMNI-TEST/2026-07-30/06-56-32/checkpoints/step_5000/model.safetensors --mode verify_timeseries_scits --validation --limit 50 --backbone allenai/OLMo-1B-0724-hf
```

### Local TS_QA Data Analysis

Use [scripts/perlmutter/analyze_ts_data.py](scripts/perlmutter/analyze_ts_data.py) to scan all JSONL files under a TS_QA root folder and report, per file:
- Longest time series length observed.
- Min/max number of `<ts></ts>` placeholders per sequence.

Example:

```bash
python3 scripts/perlmutter/analyze_ts_data.py \
	--root /flare/ModCon/ngetty/data/zone_a/ts_qa
```

Optional JSON output:

```bash
python3 scripts/perlmutter/analyze_ts_data.py \
	--root /flare/ModCon/ngetty/data/zone_a/ts_qa \
	--output-json outputs/ts_qa_stats.json
```

Output columns:
- `file`: JSONL path
- `records`: number of non-empty lines parsed as records
- `bad_json`: malformed JSON lines
- `max_ts_len`: longest per-series length found in that file
- `min_ts_tags` / `max_ts_tags`: min/max `<ts></ts>` count per sequence in text fields

