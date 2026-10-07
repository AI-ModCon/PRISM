# KEGG Dataset Curation Pipeline — Design Doc

This document plans a from-scratch recreation, inside `BaseMM_PRISM`, of the
pipeline BioReason used to build the `wanglab/kegg` biological-reasoning
dataset (1,159 train rows: `question`, `answer`, `reasoning`,
`reference_sequence`, `variant_sequence`). `BaseMM_PRISM` currently only
*consumes* the published dataset (`src/data/multimodal.py`'s
`_process_dna_bioreason`, mirroring `BioReason/bioreason/dataset/kegg.py`'s
`_format_kegg`) — there is no curation/construction code anywhere in this
repo today. This doc is the plan; no pipeline code has been written yet.

**Implementation has since started.** For what's actually been built and
live-tested so far (stage-by-stage, with real data/paths/function
descriptions), see
[kegg_curation_implementation_status.md](kegg_curation_implementation_status.md).
This document remains the original plan/rationale; it is not kept in sync
with implementation details as code lands.

---

## Table of Contents

1. [Source Pipeline Summary](#source-pipeline-summary)
2. [Key Findings From the Original Notebooks](#key-findings-from-the-original-notebooks)
3. [Module Layout](#module-layout)
4. [Stage-by-Stage Plan](#stage-by-stage-plan)
5. [Manual/TODO Checkpoints](#manualtodo-checkpoints)
6. [New Dependencies](#new-dependencies)
7. [Open Questions Resolved](#open-questions-resolved)
8. [Out of Scope](#out-of-scope)

---

## Source Pipeline Summary

BioReason's original pipeline lives entirely in Jupyter notebooks under
`BioReason/data/`, run in this order:

1. `KEGG_Data_1.ipynb` — KEGG network/pathway retrieval, gene-variant token
   extraction, cross-database ID resolution (ClinVar/dbSNP/OMIM/COSMIC),
   variant→network mapping, reference genome download, first-pass sequence
   generation.
2. `BioReasoning_DataCuration_KEGG.ipynb` — Claude API (Batch API,
   `claude-3-7-sonnet-20250219`) generates `question`/`answer`/`reasoning`
   per variant from a structured JSON prompt.
3. `KEGG_Data_2.ipynb` — a cleaner rewrite of the tail of `KEGG_Data_1`,
   re-parses variant/disease metadata and regenerates sequences with a
   different window size (2000nt vs. `KEGG_Data_1`'s 1000nt).
4. `KEGG_Data_3.ipynb` — merges Claude output with sequences, rebuilds the
   final `question` (templated, discarding Claude's own question),
   overwrites `answer` with Claude's structured disease label, applies a
   hand-curated disease-name dictionary, assembles the final `Dataset`, and
   saves it locally (the actual `push_to_hub` call is not present in any
   notebook — it was run manually, out of band).

Full step-by-step trace (30 steps, with exact commands/code snippets and
per-notebook detail) was produced during exploration and is available on
request; this doc summarizes only what's load-bearing for the recreation
plan.

---

## Key Findings From the Original Notebooks

These directly shape the plan below:

1. **Cross-DB ID parsing (Entry → Source + ID) has zero surviving code.**
   The original used a one-off ChatGPT paste-and-parse with no script. This
   is the single largest reproducibility gap.
2. **`Var_ID` assignment is manual (Excel), undocumented in any cell.**
   Everything downstream (sequence files, Claude output files) is keyed by
   this ID via filename, but no code produces it.
3. **Sequence window size is inconsistent across notebooks**: 1000nt
   (`KEGG_Data_1`) → 2000nt (`KEGG_Data_2` config) → 2000nt again
   (`KEGG_Data_3`, final pass, variable named `KEGG_2000`). The final
   published window is almost certainly **2000nt per side**, but this was
   never reconciled in one place in the original.
4. **A dead code branch**: `if variant_allele == "deletion"` never actually
   fires (real `AltAllele` values are nucleotide strings, not the literal
   string `"deletion"`), so the substitution/insertion branch always runs,
   even for true deletions. Repeated unchanged in all three notebooks.
5. **The only LLM-API-cost step is Claude reasoning generation** (~1,449
   variants attempted, `max_tokens=6000` each, Batch API for 50% discount).
   Claude's own `question` field is discarded in `KEGG_Data_3` in favor of a
   template; Claude's `answer` is also overwritten with
   `reasoning.labels.disease[0]`. Only `reasoning.reasoning_steps` survives
   verbatim into the final dataset.
6. **Row-count attrition is never reconciled**: 1,449 variants attempted →
   1,159 published rows (~290 lost). No cell explains the gap; likely a mix
   of Claude JSON parse failures and missing gene/disease/chromosome data
   triggering skips.
7. **`push_to_hub` itself was never scripted** — the pipeline as captured
   stops at a local parquet file.

---

## Module Layout

New module, following the existing `src/data/download_zone_*_data.py`
convention (plain per-stage scripts, not a framework):

```
src/data/kegg_curation/
  __init__.py
  config.py                       # single source of truth: sequence_window,
                                   # paths, model config — the original had
                                   # 3 inconsistent copies of window size
  01_fetch_kegg_networks.py       # KEGG REST + kegg_pull bulk download
  02_extract_variant_ids.py       # regex scrape of gene-variant tokens
  03_resolve_variant_ids.py       # [TODO] cross-DB ID resolution
  04_fetch_variant_coords.py      # NCBI Entrez + COSMIC file lookup → SPDI
  05_merge_and_dedup.py           # merge sources, dedup, auto-assign Var_ID
  06_map_variants_to_networks.py  # variant × pathway/network join
  07_fetch_reference_genome.py    # GRCh38 download + chromosome subset
  08_generate_sequences.py        # extract ref/variant windows
  09_generate_reasoning.py        # pluggable reasoning/Q&A generation
  10_assemble_dataset.py          # merge + assemble final HF Dataset
  checkpoints/                    # intermediate TSV/JSON artifacts (gitignored)
```

Each stage reads one checkpoint file and writes the next, so the pipeline
can be resumed/re-run from any stage without redoing earlier (expensive)
work — mirrors how the original notebooks persisted intermediate TSV/JSON
files between cells.

Final schema matches `wanglab/kegg` exactly so it's a drop-in for
`_process_dna_bioreason` with no changes needed on the consumer side:

| Column | Type | Source |
|---|---|---|
| `question` | str | Stage 10, templated from pathway/gene metadata |
| `answer` | str | Stage 09 (Claude), `reasoning.labels.disease[0]` |
| `reasoning` | str | Stage 09 (Claude), `reasoning.reasoning_steps` joined |
| `reference_sequence` | str | Stage 08 |
| `variant_sequence` | str | Stage 08 |

---

## Stage-by-Stage Plan

| Stage | Recreates (original steps) | Approach |
|---|---|---|
| 01 | Network/pathway listing + bulk download | `kegg_pull` (new dep) for REST calls; direct port, low risk |
| 02 | Gene-variant token extraction | Regex scrape (`[0-9]+v[0-9]+`) over downloaded network entries; direct port |
| 03 | Cross-DB ID resolution | **[TODO]** — see [Manual/TODO Checkpoints](#manualtodo-checkpoints) |
| 04 | ClinVar/dbSNP/OMIM coordinate resolution + COSMIC lookup | `Bio.Entrez` (biopython, new dep) replacing manual `edirect` shell calls, for ClinVar/dbSNP/OMIM SPDI resolution; COSMIC point-mutation matching against a user-supplied licensed TSV (`CompleteTargetedScreensMutant`) — **kept in scope per your answer**, gated behind a config path the user must supply. COSF (fusions) and dbVar are **not** recreated — the original abandoned both for the same reasons documented in the findings above (no reliable path to an exact nt sequence for COSF; dbVar discontinued) |
| 05 | Merge to common schema, dedup, `Var_ID` assignment | Dedup on (Variant ID, Chr, RefAllele, AltAllele) is fully scriptable. `Var_ID` becomes **sequential auto-numbering in processing order** (`KEGG_1, KEGG_2, ...`) — a deliberate, documented deviation from the original's manual Excel assignment, since we're not trying to byte-match `wanglab/kegg`'s specific IDs |
| 06 | Variant→network/pathway mapping | Join on `ENTRY`; direct port |
| 07 | Reference genome acquisition | GRCh38 (GCF_000001405.26) download + `seqkit`-equivalent chromosome subsetting (or biopython `SeqIO` if avoiding a `seqkit` binary dependency is preferred — open question, see below) |
| 08 | Sequence window extraction | Extracts `[start-window, end+window]` and splices in `AltAllele`. **Fixes the dead `deletion`-string-check bug** from the original. **Single configured window** (`config.py`'s `sequence_window`, default 2000nt/side) instead of 3 inconsistent regenerations |
| 09 | Reasoning/Q&A generation | Pluggable `ReasoningGenerator` interface (see below); default backend uses the Anthropic SDK's Batch API, current model, same structured-JSON prompt/schema as the original (`raw_data`, `question`, `answer`, `reasoning` with `reasoning_steps`/`hgvs`/`labels`) |
| 10 | Final assembly | Templated `question` construction (pathway/gene context), `answer`/`reasoning` extraction from stage 09 output, **optional** disease-name standardization hook (starts as a no-op passthrough — see below), `Dataset`/`DatasetDict` assembly, local parquet save. `push_to_hub` documented as a manual final command in the script's docstring, matching the original |

### Pluggable reasoning generation (stage 09)

```python
class ReasoningGenerator(ABC):
    @abstractmethod
    def generate(self, variant_context: dict) -> dict:
        """Returns {"question": ..., "answer": ..., "reasoning": ...}"""

class ClaudeReasoningGenerator(ReasoningGenerator):
    # Batch API, configurable model (defaults to current Claude model,
    # NOT the dated claude-3-7-sonnet-20250219 snapshot the original used)
    ...

class MockReasoningGenerator(ReasoningGenerator):
    # For dry runs / pipeline testing without API cost
    ...
```

Config-selected in `config.py`, so a dry run of the full pipeline (stages
01-10) can execute with zero API cost via the mock backend before committing
to a real Claude run over hundreds of variants.

---

## Manual/TODO Checkpoints

Per your direction, these ship as explicit, documented gaps rather than
best-effort re-automation:

- **Stage 03 (cross-DB ID resolution)**: no original code exists to port.
  Ships as a stub that raises `NotImplementedError` with a docstring
  specifying the exact expected output schema (`Entry, Source, ID` where
  `Source ∈ {OmimVar, ClinVar, dbSNP, COSM}`), so a human can either hand-curate
  this mapping for a small variant set or write a resolver against KEGG's
  variant entry text before running stage 04.
- **Stage 05 (`Var_ID` assignment)**: auto-scriptable (sequential numbering),
  flagged in the script's docstring as an intentional deviation from the
  original's manual process — output `Var_ID`s will not match `wanglab/kegg`'s.
- **Stage 10 (disease-name standardization)**: starts as a no-op passthrough
  function with a hook (`config.py`'s `disease_name_overrides: dict`) a user
  can populate after inspecting the actual set of distinct disease labels
  produced by their run — no attempt to pre-populate the original's ~90-entry
  hand-curated dictionary, since it was specific to the original's exact
  variant set.
- **Final `push_to_hub`**: documented as a manual command in stage 10's
  docstring/`__main__` block, not executed automatically — matches the
  original, which also never scripted this.

---

## New Dependencies

To add to `requirements.txt`:

- `kegg_pull` — KEGG REST API bulk retrieval (stage 01)
- `biopython` — `Bio.Entrez` for ClinVar/dbSNP/OMIM queries, `Bio.SeqIO` for
  FASTA parsing (stages 04, 07, 08)
- `anthropic` — Claude Batch API SDK (stage 09 default backend)

New `.env` entry: `ANTHROPIC_API_KEY` (no existing pattern for this in
`.env.template` today — `HF_TOKEN`/`WANDB_API_KEY` are the current
precedents to follow).

---

## Open Questions Resolved

- **Manual-step handling**: build the full skeleton; flag manual/undocumented
  original steps as explicit TODOs rather than trying to re-automate things
  like the ChatGPT ID-parsing step that has no surviving code.
- **Reasoning generation**: pluggable backend, not hardcoded to one model/approach.
- **Location/schema**: new `src/data/kegg_curation/` module; output schema
  matches `wanglab/kegg` exactly so it's a drop-in for the existing
  `_process_dna_bioreason` consumer.
- **COSMIC**: kept in scope (user has/can get access), gated behind a
  config-supplied path to the licensed COSMIC TSV export.
- **Build order**: design doc first (this document) for review before any
  pipeline code is written.

## Still Open (need input before stage 01 starts)

- **Chromosome extraction tooling**: original used the `seqkit` CLI binary.
  Recreate with `seqkit` (adds a non-Python system dependency) or do the
  chromosome subsetting purely in Python via `Bio.SeqIO` (slower for a
  ~3GB genome FASTA but no extra system dependency)?
- **Target scale for a first run**: the original processed ~1,449 variants
  end-to-end (Claude cost driver). Should the first `BaseMM_PRISM` run
  target a similar scale, or start with a small smoke-scale subset (e.g.
  20-50 variants) to validate the full pipeline mechanically before
  spending on a larger Claude batch run?
- **Stage 03 input**: since there's no original code, do you have a
  preferred approach in mind (e.g. write a regex/heuristic parser against
  KEGG's variant entry text yourself, use an LLM call to replicate what the
  ChatGPT step did, or hand-curate a small mapping for a limited variant set)?

---

## Out of Scope

- COSF (COSMIC fusions) and dbVar — both abandoned in the original for
  documented reasons (no reliable exact-sequence path for COSF; dbVar
  discontinued), contribute zero rows to the final dataset.
- Byte-for-byte reproduction of `wanglab/kegg`'s exact 1,159 rows or its
  specific `Var_ID` numbering — this is a fresh curation run, not a replay.
- The `BioReasoning_DataCuration_KEGG.ipynb` notebook's exploratory/demo
  cells (it only actually executes 20 variants inline; the real ~1,449-variant
  run is inferred from `KEGG_Data_3`'s file-reading range, not captured in
  any single notebook cell).
