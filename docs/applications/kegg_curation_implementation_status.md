# KEGG Curation Pipeline — Implementation Status

Companion to [kegg_curation_pipeline.md](kegg_curation_pipeline.md) (the
design doc, written before any code existed). This document tracks what has
actually been **built and live-tested** against real KEGG data, stage by
stage, with exact file paths, data sources, function-level descriptions,
and real output counts. Update this doc as each new stage lands — it's the
ground truth for "what exists and what it does," the design doc is the
ground truth for "what was planned and why."

All code lives in `src/data/kegg_curation/`. All pipeline **output data**
(checkpoints + downloaded genome) lives outside the repo, under
`$KEGG_CURATION_DATA_ROOT` (env var, defaults to
`/lus/eagle/projects/argonne_tpc/abalaji/modcon/dataset/kegg_curation`) —
never committed to git.

**Update (2026-07-30)**: the full 10-stage pipeline now runs end-to-end
against live data. Final output: 281 real KEGG variant records with the
exact `wanglab/kegg` schema (`question, answer, reasoning,
reference_sequence, variant_sequence`), assembled with the `mock` reasoning
backend (zero API cost) to validate every stage's mechanics — a real
Claude run would replace the `[MOCK] ...`-prefixed question/answer/reasoning
text with genuine generated content but shouldn't otherwise change the
pipeline's behavior. `entrez_client.py` (used by stages 04 and 07) had one
credential bug and three data-correctness bugs found and fixed along the
way — see the Stage 04 section below for the full list; the "under active
parallel development" caveat from the previous version of this note no
longer applies now that stages 04-10 are complete and this doc is caught up.

---

## Status Summary

| Stage | File | Status | Last live-tested |
|---|---|---|---|
| Config | `config.py` | Done | — |
| Shared: KEGG REST | `kegg_rest.py` | Done | 2026-07-28 |
| Shared: NCBI Entrez | `entrez_client.py` | Done (4 bugs found/fixed — see Stage 04) | 2026-07-30 |
| Shared: LLM backends | `reasoning.py` | Done (Mock tested end-to-end; Claude backend code complete but untested live — needs `ANTHROPIC_API_KEY`) | 2026-07-30 (mock only) |
| 01 | `01_fetch_kegg_networks.py` | Done | 2026-07-28 |
| 02 | `02_extract_variant_ids.py` | Done | 2026-07-28 |
| 03 | `03_resolve_variant_ids.py` | Done | 2026-07-28 |
| 04 | `04_fetch_variant_coords.py` | Done | 2026-07-30 |
| 05 | `05_merge_and_dedup.py` | Done | 2026-07-30 |
| 06 | `06_map_variants_to_networks.py` | Done | 2026-07-30 |
| 07 | `07_fetch_reference_genome.py` | Done | 2026-07-30 |
| 08 | `08_generate_sequences.py` | Done | 2026-07-30 |
| 09 | `09_generate_reasoning.py` | Done (mock backend) | 2026-07-30 |
| 10 | `10_assemble_dataset.py` | Done | 2026-07-30 |

---

## `config.py` — Single Source of Truth

`KeggCurationConfig` dataclass, instantiated once as module-level `CONFIG`.
Every stage script imports `CONFIG` from here rather than hardcoding its
own copy of any parameter — this specifically fixes the original BioReason
pipeline's bug of regenerating the sequence window size 3 different times
with 2 different values across 3 notebooks.

**Paths** (all under `DATASET_ROOT`, overridable via
`KEGG_CURATION_DATA_ROOT` env var):
- `checkpoint_dir` — `$DATASET_ROOT/checkpoints/` — every stage's
  intermediate TSV/JSON output.
- `genome_dir` — `$DATASET_ROOT/genome/` — subsetted reference genome
  FASTA (stage 07, not yet built).

**Key parameters**:
- `sequence_window: int = 2000` — nucleotides extracted on each side of a
  variant when building `reference_sequence`/`variant_sequence` (stage 08).
- `id_resolver_llm_fallback: bool = True` — whether stage 03 falls back to
  an LLM for tokens its direct KEGG-field parser can't resolve.
- `cosmic_tsv_path: str | None = None` — user-supplied path to a licensed
  COSMIC "CompleteTargetedScreensMutant" TSV export; COSM-sourced variants
  are skipped (not errored) if unset.
- `reasoning_backend: str = "mock"` — `"claude"` or `"mock"`; controls both
  stage 03's LLM fallback and stage 09's reasoning generation.
- `anthropic_model`, `anthropic_api_key`, `reasoning_max_tokens`,
  `reasoning_temperature` — Claude backend config. `anthropic_api_key`
  reads from the `ANTHROPIC_API_KEY` env var by default — **this is a
  separate credential from `HF_TOKEN`**; get one from an NCBI... no, from
  an Anthropic account (https://console.anthropic.com), not Hugging Face.
- `disease_name_overrides: dict[str, str] = {}` — no-op by default; hook
  for stage 10's optional disease-label standardization.
- `max_variants: int | None = None` — cap for smoke-scale dry runs.

`CONFIG.checkpoint(name)` — helper returning `checkpoint_dir / name`, used
by every stage to read/write its checkpoint file.

---

## `kegg_rest.py` — KEGG REST API Client

Thin, direct wrapper around `https://rest.kegg.jp`, using `requests`
directly rather than the `kegg_pull` PyPI package.

**Why not `kegg_pull`**: `kegg_pull` 3.2.3 (latest on PyPI as of
2026-07-28) is broken against KEGG's *current* live API. Any call into its
Python API (even unrelated ones like `.list("network")`) triggers an
internal bootstrap step (`AbstractKEGGurl.organism_set` property) that
calls `GET https://rest.kegg.jp/list/organism`, which now returns
`400 Bad Request` — confirmed via plain `curl`, independent of any proxy
or kegg_pull code path. KEGG appears to have deprecated the `"organism"`
database name from the `list` operation in favor of `"genome"`
(`GET /list/genome` returns 200 with equivalent data), and `kegg_pull`
hasn't been updated for this. There is no newer `kegg_pull` release.

**Functions**:
- `_get(path, n_tries=3, timeout=30, sleep_time=2.0) -> str` — internal GET
  helper with retry-on-failure/retry-on-non-200 logic.
- `list_database(database: str) -> str` — `GET /list/<database>`, returns
  raw tab-separated (id, description) text.
- `link(target_database: str, source_database: str) -> str` —
  `GET /link/<target>/<source>`, tab-separated (source_id, target_id)
  pairs. **Direction matters**: KEGG's link API can silently return an
  empty body (not an error) for the reverse direction — confirmed
  `link("network", "pathway")` works, `link("disease", "network")` does
  not (empty), but `link("disease", "network")`... (see note in code;
  always verify direction empirically for a new pair before trusting it).
- `get_entry(entry_id: str) -> str` — `GET /get/<entry_id>`, one entry's
  full flat-file text.
- `get_entries(entry_ids: list[str], batch_size=10) -> dict[str, str]` —
  batched `GET /get/<id1>+<id2>+...` (KEGG caps at 10 IDs/request).
  **Important correctness detail**: results are matched back to requested
  IDs by parsing each returned record's own `ENTRY` line (e.g.
  `"ENTRY       N00001                      Network"` → bare ID
  `"N00001"`), NOT by position in the batch. This matters because KEGG
  silently omits missing/obsolete entries from a batch response rather
  than erroring — positional matching would silently misattribute every
  record after the first gap. (This positional-matching bug was present in
  an earlier draft and was caught and fixed before stage 03 was
  live-tested against real data containing such gaps — see stage 03
  below, where 2 of the first ~20 batches had exactly this gap.)

---

## `reasoning.py` — Pluggable LLM Backends

Shared by stage 03 (ID-resolution fallback) and stage 09 (not yet built —
question/answer/reasoning generation).

**`ReasoningBackend` (ABC)** — two methods:
- `complete(prompt, system=None) -> str` — single-turn text completion.
- `generate_variant_reasoning(variant_context: dict) -> dict` — structured
  generation, returns `{"question", "answer", "reasoning"}`.

**`ClaudeBackend`** — real generation via the `anthropic` Python SDK
(`anthropic.Anthropic(api_key=...)`, `client.messages.create(...)`).
Requires `CONFIG.anthropic_api_key` to be set (raises `RuntimeError`
otherwise). Model defaults to `CONFIG.anthropic_model` (`"claude-sonnet-5"`
unless overridden via `ANTHROPIC_MODEL` env var) — **deliberately not**
pinned to the original BioReason pipeline's dated
`claude-3-7-sonnet-20250219` snapshot, which will eventually be
deprecated. **Not yet live-tested** (no `ANTHROPIC_API_KEY` configured in
this environment yet).

**`MockBackend`** — zero-cost, deterministic placeholder. `complete()`
always returns `"[]"` (empty JSON list, matching what stage 03's ID
resolver expects when nothing can be resolved).
`generate_variant_reasoning()` returns synthetic
`[MOCK] ...`-prefixed text referencing the real gene/disease context
passed in, so downstream stages can be validated structurally without any
API cost. **This is the current default** (`CONFIG.reasoning_backend =
"mock"`) and what stages 01-03 were actually tested against.

**`_parse_json_response(raw: str) -> dict`** — multi-tier JSON extraction
mirroring the original notebook's robustness chain: direct `json.loads` →
strip markdown code fences → find first `{`/last `}` substring. Falls back
to `{"error": "unparseable_response", "raw_response": raw}` on total
failure.

**`get_reasoning_backend() -> ReasoningBackend`** — module-level singleton
factory, selects `ClaudeBackend`/`MockBackend` based on
`CONFIG.reasoning_backend`.

---

## Stage 01 — `01_fetch_kegg_networks.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` steps 1a-1b (bulk network
download + variant-type filtering).

**Data source**: KEGG REST API, live (`https://rest.kegg.jp`), no
authentication required.

**Input**: none — fetches fresh from KEGG on every run.

**Output**:
- `checkpoints/network_entries/{network_id}.txt` — one file per KEGG
  network entry, raw flat-file text (e.g. `N00001.txt`,
  `N00002.txt`, ...). Written for **all** networks (not just variant-type),
  so later stages needing non-variant metadata (e.g. stage 06's
  network/pathway parsing) don't have to re-fetch.
- `checkpoints/network_variant_ids.txt` — newline-separated list of
  network IDs whose `TYPE` field equals `"Variant"`.

**Functions**:
- `fetch_all_network_ids() -> list[str]` — `GET /list/network`, parses the
  tab-separated response into a bare ID list.
- `fetch_and_filter_networks(network_ids) -> list[str]` — bulk-fetches
  every network entry via `kegg_rest.get_entries` (batches of 10), saves
  each entry's raw text to `network_entries/`, parses the `TYPE` field
  line-by-line, and returns the subset of IDs where `TYPE == "Variant"`.

**Live-tested result (2026-07-28)**: 1,623 total KEGG network entries
fetched (6.6MB on disk); 298 are `TYPE=Variant`.

**Usage**: `python -m src.data.kegg_curation.01_fetch_kegg_networks`

---

## Stage 02 — `02_extract_variant_ids.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` steps 1c-1d (gene-variant
token extraction), but **improved**: the original regex-scraped only the
`EXPANDED` line (`grep -oE "[0-9]+v[0-9]+"`), discarding any description.
KEGG entries actually carry a structured `VARIANT` field
(e.g. `"VARIANT     25v1 (BCR-ABL)  BCR-ABL1 fusion"`) when present; this
stage prefers that field (keeping the free-text description — useful
context for stage 03's LLM fallback) and only falls back to the
`EXPANDED`-line regex scrape for entries lacking a `VARIANT` field.

**Data source**: `checkpoints/network_entries/*.txt` (stage 01 output,
already on disk — no new network calls).

**Input**:
- `checkpoints/network_variant_ids.txt` (stage 01)
- `checkpoints/network_entries/{id}.txt` (stage 01)

**Output**: `checkpoints/gene_variant_tokens.tsv` — columns
`Network, Token, Description`. One row per (network, gene-variant token)
pair; a token can appear under multiple networks.

**Functions**:
- `TOKEN_RE` — regex `\b(\d+v\d+)\b`, matches KEGG's internal gene-variant
  token notation (e.g. `25v1`, `6654v2`).
- `VARIANT_LINE_RE` — regex matching one `VARIANT` field line: token,
  optional `(ALIAS)` parenthetical, free-text description.
- `_read_field_block(lines, field_name) -> list[str]` — generic KEGG
  flat-file multi-line field reader (field name starts the first line;
  continuation lines are indented with no field name; a differently-named
  field or unindented line ends the block).
- `extract_tokens_from_entry(entry_text) -> list[tuple[str, str]]` — per
  above: tries the structured `VARIANT` field first, falls back to
  regex-scraping the `EXPANDED` field (or, failing that, the whole entry
  text) if no `VARIANT` field is present or it yields nothing.

**Live-tested result (2026-07-28)**: 298 variant-type networks → 328
(Network, Token) rows, 200 unique gene-variant tokens.

**Usage**: `python -m src.data.kegg_curation.02_extract_variant_ids`

---

## Stage 03 — `03_resolve_variant_ids.py`

**Recreates the *intent* of**: BioReason's `KEGG_Data_1.ipynb` step 7
(Entry → cross-DB Source+ID resolution). The original had **zero
surviving code** for this — it was a one-off ChatGPT paste-and-parse of
free NETWORK-entry text with no script saved anywhere.

**Key discovery (not in the original pipeline)**: fetching a gene-variant
token's *own* KEGG entry (`GET /get/hsa_var:<token>`, distinct from the
*network* entry) returns a structured `VARIATION` field with direct
`Source: ID [ID2 ...]` cross-reference lines, e.g.:
```
VARIATION   mutation V600E
            ClinVar: 13961 376069
            dbSNP: rs113488022 rs121913377
```
This is exactly the Entry→Source+ID mapping the original's ChatGPT step
was trying to produce from dirtier text — except it's already structured
data KEGG itself exposes, confirmed by directly fetching and inspecting
multiple real entries (`25v1`, `3845v1`, `673v1`, `1956v1`, `7157v1`)
before writing any parsing code.

**Design** (per user decision 2026-07-28): direct field parser is the
**primary** path (zero cost, high precision); LLM resolution is a
**fallback**, only invoked for tokens whose `hsa_var:` entry is missing
(404) or has no `VARIATION` field.

**Data source**: KEGG REST API (`GET /get/hsa_var:<token>`, batched via
`kegg_rest.get_entries`), live. LLM fallback uses whichever
`reasoning.py` backend is configured (Mock by default).

**Input**: `checkpoints/gene_variant_tokens.tsv` (stage 02)

**Output**: `checkpoints/resolved_variant_ids.tsv` — columns
`Token, Source, ID, ResolvedBy` (`ResolvedBy` ∈ `{"direct", "llm"}`). One
row per (token, source, id) triple — a token can resolve to multiple
sources and multiple IDs per source (e.g. 2+ ClinVar IDs for one
mutation), matching the original's downstream per-source handling.

**Functions**:
- `KNOWN_SOURCES` — dict mapping lowercase source labels
  (`clinvar, dbsnp, dbvar, omimvar, cosf, cosm`) to their canonical KEGG
  spelling. Covers all 6 source types the original pipeline used (COSF
  fusions and dbVar were later abandoned downstream in the original for
  unrelated reasons — see design doc — but are still parsed here since the
  field-level extraction is free).
- `SOURCE_LINE_RE` — regex matching a `"<Source>: <id1> <id2> ..."` line,
  case-insensitive.
- `_read_variation_lines(entry_text) -> list[str]` — KEGG multi-line
  `VARIATION` field reader (same continuation-line pattern as stage 02's
  field reader, but tolerant of a `VARIATION` field repeating multiple
  times per entry, per-mutation, as seen in real entries like `7157v1`
  which has 11 separate `VARIATION` blocks).
- `parse_variation_field(entry_text) -> list[tuple[str, str]]` — the
  direct/primary resolver: extracts all `(Source, ID)` pairs from every
  `VARIATION` field line in one entry.
- `resolve_via_llm(token, description, entry_text) -> list[tuple[str, str]]`
  — fallback resolver; builds a prompt asking the configured
  `reasoning.py` backend to identify cross-references from whatever
  context is available (the full entry text if it exists but lacks a
  `VARIATION` field, or just the token+description from stage 02 if the
  entry 404'd), parses the response as a JSON list of `[source, id]` pairs.
- `main()` — loads stage 02's tokens, batch-fetches all
  `hsa_var:<token>` entries in one `kegg_rest.get_entries` call, tries the
  direct parser per token, falls back to the LLM resolver only if direct
  parsing yields nothing and `CONFIG.id_resolver_llm_fallback` is true,
  writes the combined output, logs a `direct` vs. `llm` vs. `unresolved`
  breakdown.

**Live-tested result (2026-07-28)**: 200 unique tokens → **147 resolved
directly (73.5%), 0 via LLM fallback (mock backend, which always returns
empty — real Claude backend would attempt these), 53 unresolved** → 813
total (Token, Source, ID) rows. Source distribution across those 813 rows:
ClinVar 235, COSM 202, dbSNP 201, COSF 87, OmimVar 60, dbVar 28.

Two of the ~20 batched `get_entries` calls had partial 404s within the
batch (e.g. `hsa_var:25v2` doesn't resolve to its own entry even though it
appears in `25v1`'s network's `VARIANT` field) — handled correctly by
`kegg_rest.get_entries`'s ENTRY-line-based matching (see that section
above); this is exactly the scenario that would have silently corrupted
results under naive positional matching.

**Not yet tested**: the LLM fallback path itself (mock backend never
resolves anything, by design) — needs a real `ANTHROPIC_API_KEY` to
validate end-to-end. Also not yet built: the deferred regex-vs-LLM
accuracy/coverage/cost comparison from the design doc's open questions —
revisit once the LLM fallback has real output to compare against the 147
direct-parse results (which now function as a partial ground truth: at
minimum, any LLM resolver should ideally reproduce a good fraction of
those 147 tokens' known-correct answers to be considered trustworthy on
the remaining 53).

**Usage**: `python -m src.data.kegg_curation.03_resolve_variant_ids`

---

## `entrez_client.py` — NCBI Entrez E-utilities Client

Thin wrapper around NCBI's E-utilities REST endpoints
(`https://eutils.ncbi.nlm.nih.gov/entrez/eutils`), used by stages 04 and 07.
Not built via `Bio.Entrez` — direct `requests` calls instead, for the same
reason `kegg_rest.py` avoids `kegg_pull`: full control over request/response
handling without depending on a library's internal assumptions.

**Functions**:
- `_throttled_get(url, params, n_tries=3, timeout=30) -> str` — internal GET
  helper. Requires `CONFIG.ncbi_entrez_email` (raises `RuntimeError` if
  unset — NCBI's usage policy requires a real contact email on every
  request); adds `CONFIG.ncbi_api_key` when present. Throttles to ~3 req/s
  without a key, ~10 req/s with one.
- `esearch(db, term) -> list[str]` — ESearch, returns matching UIDs.
- `efetch_docsum_batch(db, ids, batch_size=200) -> dict[str, ET.Element]` —
  batched EFetch docsum, returns `{uid: <DocumentSummary> element}`.
- `parse_spdi(doc_summary, db) -> list[str]` — extracts SPDI coordinate
  string(s) from a docsum element (ClinVar: single `canonical_spdi` value;
  dbSNP: comma-separated `SPDI` list, one per possible allele).
- `spdi_to_coords(spdi) -> tuple[str, int, str, str] | None` — parses
  `"seq_id:position:ref:alt"` into `(seq_id, position, ref, alt)`. Position
  is **0-based** (SPDI convention) — this matters directly for stage 08's
  slicing logic, which relies on this being uniform across every source.
- `efetch_fasta(seq_id, seq_start=None, seq_stop=None) -> str` — fetches a
  nucleotide sequence (or coordinate slice) as plain FASTA text. Used by
  stage 07 for whole-chromosome downloads; also supports on-demand
  coordinate-range fetches (confirmed working live, e.g.
  `.../efetch.fcgi?db=nuccore&id=NC_000004.12&seq_start=X&seq_stop=Y`
  returns just that slice) though this pipeline downloads full chromosomes
  once rather than per-variant ranges — a deliberate tradeoff decision
  (avoids one HTTP request per variant at generation time; works offline
  after the initial download).

**Four bugs found and fixed here during live testing (all confirmed with
before/after data), each significant enough to silently corrupt or zero out
a whole source's worth of data if left in place**:

1. **Hardcoded placeholder email** (`"noreply@example.com"`) instead of
   reading `CONFIG.ncbi_entrez_email` — silently violated NCBI's usage
   policy (a real, identifying contact address is required) and would have
   meant `NCBI_ENTREZ_EMAIL` being set had no effect. Fixed to read from
   `CONFIG` and raise clearly if unset.
2. **dbSNP batch docsum parsing crashed on 2+ IDs.** NCBI's docsum response
   shape differs by database: ClinVar wraps records in
   `<eSummaryResult><DocumentSummarySet><DocumentSummary>...`, but a
   multi-ID `db=snp` request returns two or more **sibling**
   `<DocumentSummary>` root elements with no common wrapper at all — which
   is invalid XML (multiple document roots) and raised `ET.ParseError` for
   every dbSNP batch, silently caught and logged as a warning, resulting in
   **0/201 dbSNP rows resolved**. Fixed by stripping any XML
   declaration/DOCTYPE prolog and wrapping the response body in a synthetic
   root element before parsing — confirmed this doesn't break the
   already-working ClinVar path (which does have a real declaration/DOCTYPE
   that would otherwise conflict with the wrapper).
3. **dbSNP ID format mismatch in `04_fetch_variant_coords.py`'s
   `resolve_clinvar_and_dbsnp`.** Stage 03 correctly stores dbSNP IDs *with*
   the `rs` prefix (as KEGG's own `VARIATION` field writes them, e.g.
   `"rs121913506"`), but NCBI's `db=snp` endpoint takes/returns **bare
   numeric IDs** (`efetch_docsum_batch`'s results dict is keyed by
   `doc_summary.get("uid")`, confirmed to never include `rs`). The row
   lookup `docsums.get(row["ID"])` used the `rs`-prefixed string against
   bare-numeric keys — guaranteed miss, every time, on top of bug #2. Fixed
   by querying with the bare ID and mapping back to the original
   `rs`-prefixed ID when building output rows (need both: the bare ID to
   query NCBI, the original ID to write into the output row for provenance).
4. **`04_fetch_variant_coords.py`'s CSV writer `COLUMNS` didn't include
   `ResolvedBy`**, a column stage 03's output (and therefore every row
   flowing through stage 04 via `{**row, ...}`) actually carries — crashed
   with `ValueError: dict contains fields not in fieldnames` on the very
   last line of `main()`, after all the (expensive, rate-limited) NCBI
   resolution work had already completed. Fixed by adding it to `COLUMNS`.

Combined effect of bugs #1-4: the very first live run of stage 04 (before
any of these fixes) produced 185 resolved / 59 ambiguous / 569 unresolved
out of 813 rows, with **dbSNP contributing 0 resolved rows** despite being
one of the two largest source categories (201 rows). After all four fixes:
**216 resolved / 227 ambiguous / 370 unresolved**, with dbSNP correctly
contributing 31 resolved + 168 ambiguous + 2 unresolved.

---

## Stage 04 — `04_fetch_variant_coords.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` steps 8-13 (ClinVar/dbSNP/
OMIM coordinate resolution via NCBI Entrez, COSMIC via a licensed TSV),
using direct E-utilities HTTP calls (`entrez_client.py`) instead of the
original's manual `esearch | efetch` shell pipeline via the `edirect` CLI.

**Data source**: NCBI E-utilities (live), optionally a user-supplied
COSMIC TSV export for COSM IDs.

**Input**: `checkpoints/resolved_variant_ids.tsv` (stage 03)

**Output**: `checkpoints/variant_coordinates.tsv` — columns `Token, Source,
ID, ResolvedBy, SeqID, Position, RefAllele, AltAllele, Status`. `Status` ∈
`{resolved, ambiguous, unresolved}` — see the important semantic note below.

**Functions**:
- `resolve_clinvar_and_dbsnp(rows)` — batches ClinVar and dbSNP IDs
  separately through `entrez_client.efetch_docsum_batch` +
  `parse_spdi` + `spdi_to_coords`. `Status="ambiguous"` here means the
  docsum listed multiple possible SPDI entries (multiple alt alleles for
  one dbSNP rsID) — the first is used, but a **real, usable coordinate is
  always present** for ambiguous ClinVar/dbSNP rows.
- `resolve_omimvar(rows)` — OMIM variant IDs are often gene-level, not
  variant-level; resolves via a ClinVar cross-reference search
  (`esearch("clinvar", f"{omim_id}[mim]")`). If the search returns >5
  ClinVar hits, the row is marked `Status="ambiguous"` **with no
  coordinate at all** (`SeqID`/`Position`/`RefAllele`/`AltAllele` all left
  empty) — matching the original's own documented experience that these
  need manual per-ID resolution. **This is the opposite meaning of
  "ambiguous" from the ClinVar/dbSNP case above** — see stage 05's dedup
  logic, which had to be fixed once this asymmetry was discovered live.
- `resolve_cosmic(rows)` — matches COSM IDs against a user-supplied,
  license-gated COSMIC "CompleteTargetedScreensMutant" TSV
  (`CONFIG.cosmic_tsv_path`); marks all COSM rows unresolved if unset
  (no COSMIC file was available for this pipeline's live testing, so all
  202 COSM rows are currently unresolved — this path is implemented but
  not live-verified against a real COSMIC export).
- `main()` — loads stage 03's output, runs all three resolvers plus a
  pass-through marking dbVar/COSF rows explicitly `unresolved` (never
  resolved — see design doc's "Out of Scope" section for why), writes
  combined output.

**Live-tested result (2026-07-30, after all 4 `entrez_client.py`/stage-04
bugs fixed)**: 813 input rows → **216 resolved, 227 ambiguous, 370
unresolved**. Per-source breakdown: ClinVar 185 resolved / 50 unresolved;
dbSNP 31 resolved / 168 ambiguous / 2 unresolved; OmimVar 59 ambiguous
(no coordinate — see above) / 1 unresolved; COSM 202 unresolved (no COSMIC
file configured); COSF 87 unresolved, dbVar 28 unresolved (both
out-of-scope by design).

**Usage**: `python -m src.data.kegg_curation.04_fetch_variant_coords`
(requires `NCBI_ENTREZ_EMAIL` in the environment)

---

## Stage 05 — `05_merge_and_dedup.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` steps 14-15 (merge to common
schema, dedup, `Var_ID` assignment) — with the two deliberate deviations
already documented in the design doc (`Var_ID` auto-sequential instead of
manual Excel numbering; dedup key is `(Token, SeqID, Position, RefAllele,
AltAllele)` rather than a global coordinate match, so a coincidental
coordinate collision across two different genes/tokens is never merged).

**Key bug found and fixed live**: the original filter
(`row["Status"] in {"resolved", "ambiguous"}`) let through stage 04's
59 OMIM-ambiguous rows, which — per the discovery above — have **no
coordinate data at all** despite the `"ambiguous"` label. This produced 59
downstream variants with empty `SeqID`, which stage 08 then correctly
flagged as `missing_genome` (empty string doesn't match any
`genome/{seq_id}.fasta` file) rather than silently mishandling them, but
the right fix is upstream: filter on whether `row["SeqID"]` is actually
non-empty, not on the `Status` label alone, since `"ambiguous"` means
different things for different sources (see stage 04 notes). Fixed by
adding `if not row["SeqID"]: continue` to the dedup loop.

**Input**: `checkpoints/variant_coordinates.tsv` (stage 04)

**Output**: `checkpoints/variants_deduped.tsv` — columns `Var_ID, Token,
SeqID, Position, RefAllele, AltAllele, Sources, Status`. `Sources` is a
comma-joined list of every cross-DB source that independently confirmed
this exact coordinate (e.g. `"ClinVar,dbSNP"`) — a useful cross-validation
signal kept in the output rather than discarded.

**Functions**:
- `_dedup_key(row)` — `(Token, SeqID, Position, RefAllele, AltAllele)`.
- `merge_and_dedup(rows)` — groups by that key (after the `SeqID`-non-empty
  filter above), tracks the union of contributing `Source` values per
  group, and marks a group `"ambiguous"` overall if ANY contributing row
  was ambiguous (reflects genuine uncertainty about the underlying source
  data, even if another source independently confirmed the same
  coordinate).
- `assign_var_ids(deduped)` — sequential `KEGG_1, KEGG_2, ...` in
  first-seen/processing order.

**Live-tested result (2026-07-30, after the SeqID-filter fix)**: 813
(Token, Source, ID) rows → 443 with real coordinates → **281 distinct
variants** (130 resolved, 151 ambiguous; 85 independently confirmed by 2+
sources — e.g. `KEGG_4`: `2322v1` at `NC_000013.11:28018504 C>A`, confirmed
by both ClinVar and dbSNP). (Before the SeqID-filter fix: 340 variants,
including 59 spurious no-coordinate rows that would have broken stage 08.)

**Usage**: `python -m src.data.kegg_curation.05_merge_and_dedup`

---

## Stage 06 — `06_map_variants_to_networks.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` steps 16-19 (variant →
network/pathway mapping, network metadata parsing).

A single gene-variant token commonly maps to multiple KEGG networks — e.g.
`3815v1` (a KIT kinase domain mutation) appears in both `N00003`
(RAS-ERK signaling) and `N00046` (PI3K signaling), each implicating a
different downstream pathway for the same physical mutation. This stage
keeps **every** (variant, network) pairing as a separate output row rather
than picking one — confirmed each pairing carries genuinely distinct,
non-redundant pathway/disease context by direct inspection of both
`N00003.txt` and `N00046.txt`.

**Input**: `checkpoints/variants_deduped.tsv` (stage 05),
`checkpoints/gene_variant_tokens.tsv` (stage 02, for Token→Network),
`checkpoints/network_entries/{id}.txt` (stage 01, network flat-file text)

**Output**: `checkpoints/variants_with_context.tsv` — columns `Var_ID,
Token, SeqID, Position, RefAllele, AltAllele, Sources, Status, Network,
NetworkName, Definition, Pathway, Disease, Genes`.

**Functions**:
- `_read_field_block` / `_single_line_field` — same generic KEGG flat-file
  field reader pattern used in stages 02/03.
- `parse_network_metadata(entry_text) -> dict` — extracts `NetworkName`
  (`NAME` field), `Definition` (`DEFINITION` + its `EXPANDED` continuation
  line — these get concatenated without a separator since `EXPANDED` isn't
  its own recognized field name to the generic reader; cosmetic clutter,
  not a correctness issue, noted but not fixed), `Pathway`/`Disease`
  (KEGG-ID-prefix stripped, semicolon-joined if multiple), `Genes`
  (gene symbols only, stripped of the `; full name` suffix, comma-joined).
- `main()` — for each deduped variant, looks up all networks its `Token`
  belongs to, parses (and caches) each network's metadata once, emits one
  row per (variant, network) pair.

**Live-tested result (2026-07-30)**: 281 distinct variants → **393
(variant, network) rows, covering all 281/281 variants** (0 unmatched).
Verified against `KEGG_1` (`3815v1`): correctly produced 2 rows, one for
`N00003` (genes: KIT, GRB2, SOS1, SOS2, HRAS, KRAS, NRAS, ARAF, BRAF, RAF1,
MAP2K1, MAP2K2, MAPK1, MAPK3; disease: Acute myeloid leukemia) and one for
`N00046` (genes: KIT, PIK3CA, PIK3CB, PIK3CD, AKT1, AKT2, AKT3, BAD; same
disease, different pathway) — matching the two real KEGG network entries
inspected directly before writing this stage.

**Usage**: `python -m src.data.kegg_curation.06_map_variants_to_networks`

---

## Stage 07 — `07_fetch_reference_genome.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` step 20 (reference genome
acquisition) — but more targeted. The original downloaded the full GRCh38
assembly (~3GB) and used the `seqkit` CLI to subset to needed chromosomes;
this stage instead fetches **only the individual chromosome accessions
actually referenced** by stage 05's deduped variants, directly via
`entrez_client.efetch_fasta` (NCBI Entrez `db=nuccore`, `rettype=fasta`) —
no `seqkit` or other external CLI dependency, matching the design
decision for pure-Python tooling.

An alternative was considered and explicitly rejected during design: NCBI's
Entrez efetch also supports `seq_start`/`seq_stop` range parameters
directly on a chromosome accession (confirmed live —
`.../efetch.fcgi?db=nuccore&id=NC_000004.12&seq_start=X&seq_stop=Y` returns
just that small slice), which would let stage 08 fetch each variant's exact
window on demand with one small request per variant and skip downloading
whole chromosomes entirely. Decided against per-variant range-fetching in
favor of downloading full chromosomes once — works offline afterward, no
per-variant network dependency at sequence-generation time, and `stage 08`
can be re-run cheaply against local files. `efetch_fasta` supports both
modes (range params are optional).

**Input**: `checkpoints/variants_deduped.tsv` (stage 05, to determine the
distinct `SeqID` set actually needed)

**Output**: `genome/{seq_id}.fasta` — one file per needed chromosome, under
`CONFIG.genome_dir`.

**Functions**:
- `needed_chromosomes() -> set[str]` — distinct `SeqID` values from stage
  05's output.
- `main()` — for each needed chromosome not already cached on disk
  (checked via file existence + non-zero size, so re-running is cheap and
  idempotent), fetches via `entrez_client.efetch_fasta` and writes to disk.

**Live-tested result (2026-07-30)**: 18 distinct chromosomes needed
(chr1-7, 9-13, 15, 17, 19-21, X). **All 18/18 downloaded successfully**,
though this took 3 attempts due to a flaky ALCF proxy under sustained
large transfers ("Response ended prematurely" / connection resets on
several of the larger chromosome files, e.g. chr2 at ~240MB, chr4, chr5,
chr11, chrX) — the caching-by-file-existence logic meant each retry only
had to fetch the chromosomes still missing, not re-download everything.
Total downloaded: chr1 (252MB) down to chr21 (47MB), ~2.1GB combined — well
under the ~3GB full-assembly download the original approach would have
required, despite covering the majority of chromosomes.

**Usage**: `python -m src.data.kegg_curation.07_fetch_reference_genome`
(requires `NCBI_ENTREZ_EMAIL`; safe to re-run if some downloads fail —
already-downloaded files are skipped)

---

## Stage 08 — `08_generate_sequences.py`

**Recreates**: BioReason's `KEGG_Data_1.ipynb` step 22 (sequence window
extraction), with two corrections against the original:

1. **Uniform 0-based coordinate system.** The original mixed ENST
   (transcript, sometimes 1-based) and genomic (RefSeq, 0- or 1-based
   depending on source) coordinates, needing a per-row branch to pick the
   right indexing offset, and still logged verification mismatches it
   never auto-corrected. This pipeline resolves every variant to SPDI in
   stage 04, which is uniformly 0-based (confirmed via
   `entrez_client.spdi_to_coords`'s docstring and NCBI's SPDI spec) — so
   Python's native 0-indexed string slicing on the FASTA text needs no
   conversion at all, a single code path for every source.
2. **Fixed the dead `deletion`-string-check bug.** The original checked
   `if variant_allele == "deletion":` to special-case deletion variants,
   but real `AltAllele` values are always nucleotide strings (or, for a
   true deletion, an empty string) — never the literal string
   `"deletion"` — so that branch never executed in any of the three
   original notebooks, and true deletions always fell through to the
   substitution/insertion branch with incorrect results. This stage checks
   `AltAllele == ""` directly, which correctly handles deletions as "splice
   in an empty string" (i.e., remove the ref_allele span) — the same
   splice logic handles substitution, insertion, and deletion uniformly
   without a special case.

**Input**: `checkpoints/variants_with_context.tsv` (stage 06),
`genome/{seq_id}.fasta` (stage 07)

**Output**:
- `checkpoints/sequences/{Var_ID}.json` — one file per variant:
  `{"reference_sequence": ..., "variant_sequence": ...}`.
- `checkpoints/variants_with_sequences.tsv` — stage 06's rows plus a
  `SequenceStatus` column (`ok` / `ref_mismatch` / `missing_genome` /
  `out_of_range`). `ref_mismatch` means the genome's bases at the variant's
  position disagree with the SPDI-reported reference allele — the sequence
  is still written, but stage 09's `!= "ok"` filter skips it, so a
  coordinate error (wrong assembly, stale SPDI, off-by-one) cannot silently
  reach the final dataset. Review those rows before treating them as usable.

**Functions**:
- `_load_chromosome(seq_id) -> str | None` — lazily loads and caches one
  chromosome's sequence as a single uppercase string (FASTA header line
  stripped), from `genome/{seq_id}.fasta`.
- `generate_variant_sequences(chrom_seq, position, ref_allele, alt_allele,
  window) -> tuple[str, str] | None` — the core splice logic: extracts
  `[position - window, position + len(ref_allele) + window]`, verifies the
  genome's actual bases at `position` match the expected `ref_allele`
  (logs a warning but still proceeds using the genome's actual bases if
  not — a stale-SPDI-vs-current-genome-build mismatch is possible but rare
  and shouldn't hard-fail the whole variant), then builds
  `variant_sequence` by replacing the `ref_allele`-length span with
  `alt_allele`. Returns `None` if the window would extend past either end
  of the chromosome (this genuinely happened 0 times in the live run).
- `main()` — processes each **distinct** `Var_ID` once (cached, since a
  variant can appear under multiple network rows from stage 06 — no need
  to regenerate its sequence per network), writes per-variant JSON files
  plus the annotated TSV.

**Live-tested result (2026-07-30)**: **281/281 distinct variants — 100%
success** (0 missing genome file, 0 out of range). Correctness verified
directly on `KEGG_1` (`3815v1`, `NC_000004.12:54733153 G→T`): generated
sequences are 4001bp each (2000 window × 2 + 1bp ref allele, matching
`CONFIG.sequence_window=2000`), differ from each other at **exactly one
position** (index 2000, the exact window center as expected), and that
one difference is `G→T` — matching the deduped variant's `RefAllele`/
`AltAllele` exactly.

**Usage**: `python -m src.data.kegg_curation.08_generate_sequences`

---

## Stage 09 — `09_generate_reasoning.py`

**Recreates**: BioReason's `BioReasoning_DataCuration_KEGG.ipynb` (the
Claude curation step) — generates `question`/`answer`/`reasoning` per
variant via the pluggable `reasoning.py` backend.

**Design note**: unlike the original (which discarded Claude's own
generated `question` in favor of a template in `KEGG_Data_3`, and
overwrote `answer` with `reasoning.labels.disease[0]` rather than Claude's
free-text answer), this stage uses the backend's question/answer directly
as generated — there's no downstream override step, since stage 10 doesn't
currently re-template these fields (see stage 10 below; this is a
reasonable place to reconsider if a real Claude run's output quality
suggests the original's template-override approach was worth keeping).

A variant attached to multiple networks (see stage 06) uses only its
**first** network row as generation context — matching the original's one
Claude-call-per-variant approach, not per (variant, network) pair.

**Input**: `checkpoints/variants_with_sequences.tsv` (stage 08),
`checkpoints/sequences/{Var_ID}.json` (stage 08)

**Output**:
- `checkpoints/reasoning/{Var_ID}.json` — one file per variant:
  `{"question": ..., "answer": ..., "reasoning": ...}`.
- `checkpoints/variants_with_reasoning.tsv` — one row per **distinct**
  variant (deduped from the variant×network input), with a
  `ReasoningStatus` column (`ok` / `error`).

**Functions**:
- `build_variant_context(row, sequences) -> dict` — assembles the context
  dict passed to `reasoning.py`'s `generate_variant_reasoning`: gene(s),
  network name, pathway definition, pathway, disease, chromosome,
  position, ref/alt allele, sources.
- `main()` — picks the first network row per `Var_ID`, calls the
  configured backend, writes per-variant JSON + the summary TSV.

**Live-tested result (2026-07-30, `mock` backend)**: **281/281 succeeded,
0 errors, 0 skipped.** Quality observation worth flagging before a real
Claude run: the `Genes` context field currently lists **every gene in the
pathway** (e.g. `"KIT, GRB2, SOS1, SOS2, HRAS, KRAS, NRAS, ARAF, BRAF,
RAF1, MAP2K1, MAP2K2, MAPK1, MAPK3"` for `KEGG_1`), not just the specific
mutated gene (`KIT`) — this makes the mock-generated question read
awkwardly ("What is the biological effect of this KIT, GRB2, SOS1, ...
variant") and would likely produce a similarly awkward or diluted prompt
for a real Claude call. Worth narrowing `Genes` to the mutated gene
specifically (derivable from `Token`'s originating `GENE` entry) before a
real/paid generation run, rather than the full pathway gene list.

**Not yet tested**: the `ClaudeBackend` path itself — needs
`ANTHROPIC_API_KEY` and `reasoning_backend="claude"` in `config.py` (or via
env var) to validate real generation quality/cost.

**Usage**: `python -m src.data.kegg_curation.09_generate_reasoning`

---

## Stage 10 — `10_assemble_dataset.py`

**Recreates**: BioReason's `KEGG_Data_3.ipynb` steps 27-29 (final tabular
assembly), producing a schema-exact match for `wanglab/kegg` so it's a
drop-in for the existing consumer
(`src/data/multimodal.py`'s `_process_dna_bioreason`).

Matching the original's own behavior: this stage does **not** call
`push_to_hub` — it saves locally and prints the exact command needed to
publish manually once the content is reviewed.

**Input**: `checkpoints/variants_with_reasoning.tsv` (stage 09),
`checkpoints/reasoning/{Var_ID}.json` (stage 09),
`checkpoints/sequences/{Var_ID}.json` (stage 08)

**Output**: `checkpoints/final_dataset/` — a HF `DatasetDict` (parquet +
`dataset_info.json`), single `"train"` split.

**Functions**:
- `standardize_disease_name(answer) -> str` — applies
  `CONFIG.disease_name_overrides` if populated; no-op passthrough
  otherwise (per design doc — the original's ~90-entry hand-curated
  mapping was specific to its own variant set and isn't pre-populated here).
- `main()` — merges stage 09's reasoning + stage 08's sequences per
  variant into the final 5-column schema, builds a `datasets.Dataset`,
  wraps in `DatasetDict({"train": ...})`, saves to disk, prints
  publish instructions.

**Live-tested result (2026-07-30)**: **281 records assembled, 0 skipped.**
Final schema confirmed to match `wanglab/kegg` exactly:
`['question', 'answer', 'reasoning', 'reference_sequence',
'variant_sequence']` (verified via `load_from_disk` + `.features`).

**Usage**: `python -m src.data.kegg_curation.10_assemble_dataset`

---

## End-to-End Pipeline Summary (2026-07-30 live run)

```
1,623 KEGG networks
  → 298 variant-type networks
    → 200 unique gene-variant tokens
      → 813 cross-DB (Source, ID) rows (73.5% resolved via direct KEGG
        field parsing, zero LLM cost)
        → 443 rows with real genomic coordinates (after NCBI resolution)
          → 281 distinct deduped variants (after fixing the OMIM-ambiguous
            no-coordinate bug)
            → 393 (variant, network) context rows
              → 281/281 sequences generated (100% success, correctness verified)
                → 281/281 reasoning generated (mock backend; Claude untested)
                  → 281-row final dataset, exact wanglab/kegg schema match
```

Along the way: dropped the `kegg_pull` dependency (broken against KEGG's
live API), found and fixed 4 bugs in `entrez_client.py`/stage 04 (hardcoded
placeholder email, dbSNP batch XML parsing, dbSNP `rs`-prefix ID mismatch,
missing CSV column), 1 bug in stage 05 (OMIM-ambiguous rows with no actual
coordinate slipping through the dedup filter), survived a flaky ALCF proxy
during stage 07's chromosome downloads via idempotent retry, and confirmed
end-to-end sequence-generation correctness against real GRCh38 data.

**Not yet done**: a real (non-mock) Claude generation run for stage 09 —
recommended next step is narrowing the `Genes` context field (see stage 09
notes) before spending real API cost, then reviewing a sample of real
output before considering `push_to_hub`. Also not yet done: sourcing a
COSMIC TSV export to resolve the 202 currently-unresolved COSM rows, and
the deferred regex-vs-LLM comparison for stage 03's fallback path.

---

## Environment / How to Run the Full Pipeline

```bash
export PATH=/lus/eagle/projects/argonne_tpc/abalaji/conda_env/prism/bin:$PATH
export http_proxy=http://proxy.alcf.anl.gov:3128     # needed on ALCF compute nodes
export https_proxy=http://proxy.alcf.anl.gov:3128
export KEGG_CURATION_DATA_ROOT=/lus/eagle/projects/argonne_tpc/abalaji/modcon/dataset/kegg_curation
export NCBI_ENTREZ_EMAIL=your-email@example.com      # required for stages 04, 07
# export ANTHROPIC_API_KEY=...                        # only if reasoning_backend="claude"
cd /home/abalaji/projects/modcon/genome/BaseMM_PRISM

python -m src.data.kegg_curation.01_fetch_kegg_networks
python -m src.data.kegg_curation.02_extract_variant_ids
python -m src.data.kegg_curation.03_resolve_variant_ids
python -m src.data.kegg_curation.04_fetch_variant_coords
python -m src.data.kegg_curation.05_merge_and_dedup
python -m src.data.kegg_curation.06_map_variants_to_networks
python -m src.data.kegg_curation.07_fetch_reference_genome
python -m src.data.kegg_curation.08_generate_sequences
python -m src.data.kegg_curation.09_generate_reasoning
python -m src.data.kegg_curation.10_assemble_dataset
```

Dependencies (already installed in the `prism` conda env at
`/lus/eagle/projects/argonne_tpc/abalaji/conda_env/prism`):
`biopython`, `anthropic` (both added to `requirements.txt`; `kegg_pull`
was added then removed once found to be broken — see stage 01/`kegg_rest.py`
notes above).

**Credentials needed**:
- `NCBI_ENTREZ_EMAIL` — required for stages 04 and 07 (NCBI usage policy).
  Not a Hugging Face credential — this is specific to NCBI's Entrez system.
- `NCBI_API_KEY` — optional, raises NCBI's rate limit from ~3 to ~10 req/s.
  Register free at https://www.ncbi.nlm.nih.gov/account/.
- `ANTHROPIC_API_KEY` — only needed if `CONFIG.reasoning_backend =
  "claude"` (default is `"mock"`, zero cost). Separate credential from
  `HF_TOKEN` — get one from https://console.anthropic.com, not Hugging Face.
- `CONFIG.cosmic_tsv_path` — optional, for resolving COSM-sourced variants;
  requires a licensed COSMIC "CompleteTargetedScreensMutant" TSV export
  (not fetchable programmatically — COSMIC access is per-user licensed).
