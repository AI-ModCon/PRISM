# Adding a Modality — Full Checklist

Work through these in order. Grep an existing modality (e.g. `time_series`) as a
worked reference at every step.

## 1. Encoder — `src/encoders/<mod>.py`

- Subclass `ModalityEncoder` (`src/encoders/base.py`).
- Implement `forward(inputs) -> torch.Tensor` returning `[B, T_mod, output_dim]`.
- Override `tokens_per_instance()` if one instance expands to >1 token slot.
- Device-agnostic: check `torch.xpu` → `torch.cuda` → CPU. No `.item()` /
  data-dependent control flow in `forward` (breaks compile).
- Frozen by default unless the training stage says otherwise.

## 2. Projector — usually no code change

- Reuse `ModalityProjector` (`src/modules/projector.py`) with the modality's
  `input_dim` → `d_model`. Pick a `norm_mode` (start with `layernorm`).

## 3. Enum — `src/modalities.py`

- Add `MYMOD = "mymod"` to the `Modality(str, Enum)`.
- Add it to `ALL_MODALITIES` if it should be on by default.

## 4. Config — `src/config.py`

- Add `d_<mod>` dimension field to `ModelConfig`.
- Add any `projector_*` overrides you need.
- `modalities` is `list[Modality]`; string coercion in `__post_init__` handles
  YAML/CLI names.

## 5. Model wiring — `src/model.py`

- In `UnifiedTransformer.__init__` (~221–354): add a
  `if Modality.MYMOD in config.modalities:` block instantiating encoder +
  projector into `self.encoders` / `self.projectors`.
- Ensure the special-token id(s) for the modality are in
  `config.modality_start_end_token_indices` so
  `_merge_text_input_ids_with_modality_embeds` (~533) knows where to splice.

## 6. Data — dataset group

- Shard the modality's data (see prism-data-pipeline
  `references/webdataset-conversion.md`; extend `shard_modality.py` if needed).
- Add a dataset entry in `src/conf/data/{daos,lustre}_datasets.yaml`.
- Wire a per-modality processor spec into `ModalityAwareWebDatasetWrapper`.

## 7. Tokenizer resize (if the modality adds special tokens)

- Resize embeddings with the **LLM tokenizer id** (`llm_tokenizer_id`), **grow-
  only** — never shrink. Shrinking causes OOB gather → delayed XPU write-fault
  (issue #117).

## 8. Length budgeting

- Cap the modality's token expansion so the **total merged sequence** stays
  under the XPU tile limit (attention ≈ O(T²); cliff ~4608 tokens). Reserve
  headroom for text (issues #120/#122/#123). A per-modality cap alone can be
  inert — budget end-to-end.

## 9. Verify

- `--dry-run` a smoke design, then run a 1-node smoke.
- Confirm tokens land at correct slots, no OOB/segfault, loss decreases.
- Add/extend a pytest under `tests/` (`--timeout=60`).

## Follow-up polish

- Consider typing `ModelConfig.modalities` end-to-end as `list[Modality]` (a
  perf-log str-coercion check surfaces when this lands).
