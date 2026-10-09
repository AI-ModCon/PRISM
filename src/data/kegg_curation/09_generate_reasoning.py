"""Stage 09 — Generate question/answer/reasoning for each variant.

Recreates BioReason's BioReasoning_DataCuration_KEGG.ipynb (the Claude
curation step): for each distinct variant, generates a grounded
question/answer/reasoning triple from its gene/pathway/disease context and
sequence, via the pluggable backend in reasoning.py (ClaudeBackend for real
generation, MockBackend for a zero-cost dry run — selected by
CONFIG.reasoning_backend).

Differs from the original in one respect worth noting: the original's final
published `question` field was actually templated in KEGG_Data_3 (Claude's
own generated question was discarded there), while `answer` was overwritten
with Claude's structured `reasoning.labels.disease[0]` rather than Claude's
free-text answer — only `reasoning.reasoning_steps` survived into the final
dataset verbatim. This stage keeps it simpler: the backend's question and
answer are used directly as generated, since there's no equivalent
downstream stage 10 step planned to override them (see stage 10, not yet
built, for whether that should change).

A variant can be attached to multiple KEGG networks (see stage 06); this
stage uses the FIRST network row per variant as the primary context for
generation (matching the original's own approach of one Claude call per
variant, not per variant×network pair) — the other network associations for
that variant remain available in variants_with_context.tsv, they're just
not the ones fed into the reasoning generation prompt.

Input:  checkpoints/variants_with_sequences.tsv   (stage 08)
        checkpoints/sequences/{Var_ID}.json        (stage 08)
Output: checkpoints/reasoning/{Var_ID}.json         one file per variant:
        {"question": ..., "answer": ..., "reasoning": ...}
        checkpoints/variants_with_reasoning.tsv     one row per DISTINCT
        variant (deduped from the variant×network input), with a
        ReasoningStatus column (ok / error)

Usage:
    python -m src.data.kegg_curation.09_generate_reasoning
"""

import csv
import json
import logging

from src.data.kegg_curation.config import CONFIG
from src.data.kegg_curation.reasoning import get_reasoning_backend

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_network_entries_dir = CONFIG.checkpoint("network_entries")
_gene_symbol_cache: dict[str, str] = {}


def resolve_mutated_gene_symbol(token: str, network_id: str) -> str:
    """The `Genes` field on a (variant, network) row lists EVERY gene in
    that KEGG pathway (e.g. 14 genes for a single KIT mutation's RAS-ERK
    network) — not just the gene the variant actually mutates. Passing the
    whole pathway gene list as "the gene" to the reasoning backend produces
    a diluted, awkward prompt/question (confirmed with the mock backend —
    see docs/applications/kegg_curation_implementation_status.md's stage 09 notes).

    KEGG's gene-variant token format is "<gene_id>v<n>" (e.g. "3815v1" for
    KIT, gene ID 3815) — the mutated gene's numeric ID is embedded in the
    token itself. This resolves that ID back to its symbol by matching it
    against the originating network entry's own GENE field lines (e.g.
    "3815  KIT; KIT proto-oncogene receptor tyrosine kinase" -> "KIT").
    """
    if token in _gene_symbol_cache:
        return _gene_symbol_cache[token]

    gene_id = token.split("v")[0]
    symbol = ""
    entry_path = _network_entries_dir / f"{network_id}.txt"
    if entry_path.exists():
        for line in entry_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(gene_id + " ") or stripped.startswith(gene_id + "\t"):
                # "3815  KIT; KIT proto-oncogene receptor tyrosine kinase"
                rest = stripped[len(gene_id):].strip()
                symbol = rest.split(";")[0].strip()
                break

    _gene_symbol_cache[token] = symbol
    return symbol


def build_variant_context(row: dict, sequences: dict) -> dict:
    """Assembles the context dict passed to the reasoning backend from one
    (variant, network) row plus its generated sequences.
    """
    mutated_gene = resolve_mutated_gene_symbol(row["Token"], row["Network"])
    return {
        "gene": mutated_gene or row.get("Genes", ""),
        "pathway_genes": row.get("Genes", ""),
        "network": row.get("NetworkName", ""),
        "pathway_definition": row.get("Definition", ""),
        "pathway": row.get("Pathway", ""),
        "disease": row.get("Disease", ""),
        "chromosome": row.get("SeqID", ""),
        "position": row.get("Position", ""),
        "reference_allele": row.get("RefAllele", ""),
        "alternate_allele": row.get("AltAllele", ""),
        "sources": row.get("Sources", ""),
    }


def main() -> None:
    CONFIG.ensure_dirs()
    in_path = CONFIG.checkpoint("variants_with_sequences.tsv")
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} (variant, network) rows.")

    # First-seen network row per Var_ID becomes the primary generation context.
    first_row_per_variant: dict[str, dict] = {}
    for row in rows:
        first_row_per_variant.setdefault(row["Var_ID"], row)

    logger.info(
        f"{len(first_row_per_variant)} distinct variants to generate "
        f"reasoning for (backend={CONFIG.reasoning_backend})."
    )

    backend = get_reasoning_backend()
    seq_dir = CONFIG.checkpoint("sequences")
    reasoning_dir = CONFIG.checkpoint("reasoning")
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    n_ok = 0
    n_error = 0
    n_skipped_no_sequence = 0

    for var_id, row in first_row_per_variant.items():
        if row.get("SequenceStatus") != "ok":
            n_skipped_no_sequence += 1
            continue

        seq_path = seq_dir / f"{var_id}.json"
        if not seq_path.exists():
            logger.warning(f"{var_id}: SequenceStatus=ok but no sequence file found — skipping.")
            n_skipped_no_sequence += 1
            continue
        sequences = json.loads(seq_path.read_text())

        context = build_variant_context(row, sequences)
        result = backend.generate_variant_reasoning(context)

        if "error" in result:
            n_error += 1
            logger.warning(f"{var_id}: reasoning generation failed: {result.get('error')}")
            out_rows.append({
                "Var_ID": var_id, "Token": row["Token"], "ReasoningStatus": "error",
            })
            continue

        (reasoning_dir / f"{var_id}.json").write_text(json.dumps(result))
        n_ok += 1
        out_rows.append({
            "Var_ID": var_id,
            "Token": row["Token"],
            "SeqID": row["SeqID"],
            "Position": row["Position"],
            "RefAllele": row["RefAllele"],
            "AltAllele": row["AltAllele"],
            "NetworkName": row.get("NetworkName", ""),
            "Disease": row.get("Disease", ""),
            "ReasoningStatus": "ok",
        })

    out_path = CONFIG.checkpoint("variants_with_reasoning.tsv")
    columns = [
        "Var_ID", "Token", "SeqID", "Position", "RefAllele", "AltAllele",
        "NetworkName", "Disease", "ReasoningStatus",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({c: row.get(c, "") for c in columns})

    logger.info(
        f"Generated reasoning for {n_ok} variants ({n_error} errors, "
        f"{n_skipped_no_sequence} skipped for missing sequences). "
        f"Wrote {len(out_rows)} rows to {out_path}"
    )


if __name__ == "__main__":
    main()
