"""Stage 04 — Resolve each cross-DB ID to genomic coordinates.

Recreates BioReason's KEGG_Data_1.ipynb steps 8-11 (ClinVar/dbSNP/OMIM via
NCBI Entrez, COSMIC via a licensed TSV export), using direct E-utilities
HTTP calls (src/data/kegg_curation/entrez_client.py) instead of the
original's manual `edirect` CLI shell commands.

ClinVar and dbSNP both expose an SPDI-format coordinate
("seq_id:position:ref:alt", 0-based) via their docsum XML — this stage
extracts that directly for both, rather than needing per-database parsing.

OmimVar-sourced IDs: the original resolved these by re-querying ClinVar
with an OMIM cross-reference search. In practice OMIM IDs are often
gene-level (not variant-level), producing hundreds of ambiguous ClinVar
matches for a single query — this stage detects that ambiguity (too many
or zero hits) and marks such rows UNRESOLVED rather than guessing, matching
the original's own experience of needing manual per-ID resolution for this
source ("It is being really difficult to run this with a loop...").

dbVar and COSF: NOT resolved here, matching the original's own abandonment
of both (dbVar discontinued; COSF has no reliable path to an exact
sequence). Rows with these sources are marked UNRESOLVED.

COSM (COSMIC): requires a user-supplied, license-gated COSMIC
"CompleteTargetedScreensMutant" TSV export (CONFIG.cosmic_tsv_path). If
unset, COSM rows are marked UNRESOLVED rather than erroring.

Input:  checkpoints/resolved_variant_ids.tsv        (stage 03)
Output: checkpoints/variant_coordinates.tsv           (Token, Source, ID,
        SeqID, Position, RefAllele, AltAllele, Status)
        Status is one of: resolved, unresolved, ambiguous

Usage:
    python -m src.data.kegg_curation.04_fetch_variant_coords
"""

import csv
import logging

from src.data.kegg_curation import entrez_client
from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLUMNS = [
    "Token", "Source", "ID", "ResolvedBy", "SeqID", "Position",
    "RefAllele", "AltAllele", "Status",
]


def resolve_clinvar_and_dbsnp(rows: list[dict]) -> list[dict]:
    """rows: [{"Token": ..., "Source": "ClinVar"|"dbSNP", "ID": ...}, ...]
    Returns rows augmented with SeqID/Position/RefAllele/AltAllele/Status.
    """
    out: list[dict] = []

    for db, source_label in [("clinvar", "ClinVar"), ("snp", "dbSNP")]:
        matching = [r for r in rows if r["Source"] == source_label]
        if not matching:
            continue

        # NCBI's db=snp E-utilities endpoint takes/returns bare numeric IDs
        # (confirmed: efetch_docsum_batch("snp", ...) keys its results dict
        # by doc_summary.get("uid"), which NCBI always renders without the
        # "rs" prefix) — but stage 03 correctly stores dbSNP IDs WITH the
        # "rs" prefix (as they appear in KEGG's own VARIATION field, e.g.
        # "rs121913506"). Query with the bare ID, map back to the original
        # rs-prefixed ID when matching results, so every dbSNP row isn't
        # silently unresolved from an ID-format mismatch.
        query_ids = [r["ID"][2:] if source_label == "dbSNP" and r["ID"].lower().startswith("rs")
                     else r["ID"] for r in matching]
        logger.info(f"Fetching {len(query_ids)} {source_label} docsums...")
        docsums = entrez_client.efetch_docsum_batch(db, query_ids)

        for row, query_id in zip(matching, query_ids, strict=True):
            doc_summary = docsums.get(query_id)
            if doc_summary is None:
                out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                            "AltAllele": "", "Status": "unresolved"})
                continue

            spdi_list = entrez_client.parse_spdi(doc_summary, db=db)
            if not spdi_list:
                out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                            "AltAllele": "", "Status": "unresolved"})
                continue

            # dbSNP can list multiple possible alt alleles for one rsID;
            # take the first (matches the original's behavior of using
            # whichever SPDI entry the docsum listed first — no stronger
            # signal is available to disambiguate without the original
            # KEGG variant's specific allele annotation).
            coords = entrez_client.spdi_to_coords(spdi_list[0])
            if coords is None:
                out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                            "AltAllele": "", "Status": "unresolved"})
                continue

            seq_id, position, ref, alt = coords
            status = "resolved" if len(spdi_list) == 1 else "ambiguous"
            out.append({
                **row, "SeqID": seq_id, "Position": position,
                "RefAllele": ref, "AltAllele": alt, "Status": status,
            })

    return out


def resolve_omimvar(rows: list[dict]) -> list[dict]:
    """OmimVar IDs are often gene-level, not variant-level — resolve via a
    ClinVar cross-reference search, but only accept the result if it's
    unambiguous (a small number of hits, ideally one).
    """
    matching = [r for r in rows if r["Source"] == "OmimVar"]
    if not matching:
        return []

    out: list[dict] = []
    for row in matching:
        try:
            clinvar_ids = entrez_client.esearch("clinvar", f"{row['ID']}[mim]")
        except RuntimeError as e:
            logger.warning(f"OMIM resolution failed for {row['ID']}: {e}")
            out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                        "AltAllele": "", "Status": "unresolved"})
            continue

        if len(clinvar_ids) == 0:
            out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                        "AltAllele": "", "Status": "unresolved"})
        elif len(clinvar_ids) > 5:
            # Gene-level OMIM ID, not variant-specific — matches the
            # original's own finding that these need manual resolution.
            logger.info(
                f"OMIM ID {row['ID']} matched {len(clinvar_ids)} ClinVar "
                "records — too ambiguous to auto-resolve, marking as such."
            )
            out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                        "AltAllele": "", "Status": "ambiguous"})
        else:
            docsums = entrez_client.efetch_docsum_batch("clinvar", clinvar_ids[:1])
            doc_summary = docsums.get(clinvar_ids[0])
            spdi_list = (
                entrez_client.parse_spdi(doc_summary, db="clinvar") if doc_summary else []
            )
            coords = entrez_client.spdi_to_coords(spdi_list[0]) if spdi_list else None
            if coords is None:
                out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                            "AltAllele": "", "Status": "unresolved"})
            else:
                seq_id, position, ref, alt = coords
                out.append({**row, "SeqID": seq_id, "Position": position,
                            "RefAllele": ref, "AltAllele": alt, "Status": "resolved"})

    return out


def resolve_cosmic(rows: list[dict]) -> list[dict]:
    """COSM (COSMIC point mutations) — matched against a user-supplied,
    license-gated COSMIC "CompleteTargetedScreensMutant" TSV export.
    Expected columns (per COSMIC's documented schema): GENOMIC_MUTATION_ID
    or LEGACY_MUTATION_ID (the "COSM<n>" identifier), CHROMOSOME, GENOME_START,
    GENOME_STOP, GENOMIC_WT_ALLELE, GENOMIC_MUT_ALLELE. Column names have
    changed across COSMIC releases — verify against your actual export.
    """
    matching = [r for r in rows if r["Source"] == "COSM"]
    if not matching:
        return []

    if not CONFIG.cosmic_tsv_path:
        logger.info(
            f"{len(matching)} COSM-sourced rows found but CONFIG.cosmic_tsv_path "
            "is not set — marking all as unresolved. Set it to a licensed "
            "COSMIC CompleteTargetedScreensMutant TSV export to resolve these."
        )
        return [{**r, "SeqID": "", "Position": "", "RefAllele": "",
                 "AltAllele": "", "Status": "unresolved"} for r in matching]

    import csv as _csv

    wanted_ids = {r["ID"] for r in matching}
    cosmic_by_id: dict[str, dict] = {}
    with open(CONFIG.cosmic_tsv_path, newline="") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        id_col = next(
            (c for c in ("GENOMIC_MUTATION_ID", "LEGACY_MUTATION_ID", "MUTATION_ID")
             if reader.fieldnames and c in reader.fieldnames),
            None,
        )
        if id_col is None:
            raise RuntimeError(
                f"Could not find a mutation ID column in {CONFIG.cosmic_tsv_path} "
                f"(looked for GENOMIC_MUTATION_ID/LEGACY_MUTATION_ID/MUTATION_ID; "
                f"found columns: {reader.fieldnames})"
            )
        for line in reader:
            raw_id = line.get(id_col, "").replace("COSM", "").strip()
            if raw_id in wanted_ids:
                cosmic_by_id[raw_id] = line

    out: list[dict] = []
    for row in matching:
        hit = cosmic_by_id.get(row["ID"])
        if hit is None:
            out.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                        "AltAllele": "", "Status": "unresolved"})
            continue
        out.append({
            **row,
            "SeqID": hit.get("CHROMOSOME", ""),
            "Position": hit.get("GENOME_START", ""),
            "RefAllele": hit.get("GENOMIC_WT_ALLELE", ""),
            "AltAllele": hit.get("GENOMIC_MUT_ALLELE", ""),
            "Status": "resolved" if hit.get("GENOME_START") else "unresolved",
        })
    return out


def main() -> None:
    CONFIG.ensure_dirs()
    in_path = CONFIG.checkpoint("resolved_variant_ids.tsv")
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} (Token, Source, ID) rows to resolve coordinates for.")

    out_rows: list[dict] = []
    out_rows += resolve_clinvar_and_dbsnp(rows)
    out_rows += resolve_omimvar(rows)
    out_rows += resolve_cosmic(rows)

    # dbVar / COSF — not resolved (see module docstring), pass through as
    # explicitly unresolved so nothing silently disappears from the pipeline.
    unresolved_sources = {"dbVar", "COSF"}
    for row in rows:
        if row["Source"] in unresolved_sources:
            out_rows.append({**row, "SeqID": "", "Position": "", "RefAllele": "",
                              "AltAllele": "", "Status": "unresolved"})

    out_path = CONFIG.checkpoint("variant_coordinates.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    n_resolved = sum(1 for r in out_rows if r["Status"] == "resolved")
    n_ambiguous = sum(1 for r in out_rows if r["Status"] == "ambiguous")
    n_unresolved = sum(1 for r in out_rows if r["Status"] == "unresolved")
    logger.info(
        f"Wrote {len(out_rows)} rows to {out_path}: "
        f"{n_resolved} resolved, {n_ambiguous} ambiguous, {n_unresolved} unresolved."
    )


if __name__ == "__main__":
    main()
