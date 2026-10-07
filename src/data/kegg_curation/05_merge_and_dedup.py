"""Stage 05 — Merge cross-DB coordinate rows into distinct genomic variants,
dedup, and assign a Var_ID to each.

Recreates BioReason's KEGG_Data_1.ipynb steps 14-15: merging ClinVar/dbSNP/
OMIM/COSM rows down to one row per real genomic variant, then deduplicating
and assigning a stable per-variant identifier. Two deliberate deviations
from the original (per design doc decisions, docs/applications/kegg_curation_pipeline.md):

  1. `Var_ID` assignment: the original did this by hand in Excel after a
     manual dedup pass, with no surviving code. This stage auto-assigns
     sequential IDs ("KEGG_1", "KEGG_2", ...) in processing order instead —
     these will NOT match wanglab/kegg's original numbering, which is
     expected and fine; this is a fresh curation run, not a byte-for-byte
     replay.
  2. Dedup key: stage 04 (`variant_coordinates.tsv`) emits one row per
     (Token, Source, ID) triple that successfully resolved to a coordinate
     — so the SAME physical mutation commonly appears multiple times (once
     per cross-referencing source, e.g. both a ClinVar row and a dbSNP row
     pointing at the identical NC_000004.12:54733153:G:A). This stage
     collapses those onto ONE row per (Token, SeqID, Position, RefAllele,
     AltAllele) — the smallest key that actually identifies a distinct
     mutation — while keeping the list of which sources support it, rather
     than deduping globally by coordinate alone (a rare coincidental
     coordinate collision across two different KEGG tokens/genes should NOT
     be merged into one variant).

Only rows with Status in {"resolved", "ambiguous"} are kept — "unresolved"
rows have no coordinate data at all (confirmed empty SeqID/Position/
RefAllele/AltAllele fields) and can't contribute a sequence in stage 08.
"ambiguous" rows (multiple possible SPDI alleles for one dbSNP rsID, or an
OMIM ID matching several ClinVar records) still carry a real, usable
coordinate — the ambiguity is about which specific source entry was
selected, not about the coordinate itself being untrustworthy — so they are
kept, just flagged via the output's Status column for visibility.

Input:  checkpoints/variant_coordinates.tsv      (stage 04)
Output: checkpoints/variants_deduped.tsv          (Var_ID, Token, SeqID,
        Position, RefAllele, AltAllele, Sources, Status)

Usage:
    python -m src.data.kegg_curation.05_merge_and_dedup
"""

import csv
import logging

from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _dedup_key(row: dict) -> tuple[str, str, str, str, str]:
    return (row["Token"], row["SeqID"], row["Position"], row["RefAllele"], row["AltAllele"])


def merge_and_dedup(rows: list[dict]) -> list[dict]:
    """Collapses stage 04's (Token, Source, ID)-keyed rows down to one row
    per (Token, SeqID, Position, RefAllele, AltAllele), tracking which
    sources support each distinct variant. Rows are kept in first-seen
    order so Var_ID assignment (by the caller) is stable/reproducible for a
    given input file.

    "ambiguous" means two different things depending on source, only one of
    which has a usable coordinate: dbSNP-ambiguous rows have a REAL
    coordinate (the ambiguity is which of several possible alt alleles for
    one rsID was picked); OMIM-ambiguous rows (stage 04's resolve_omimvar,
    when an OMIM ID matches >5 ClinVar records) have NO coordinate at all —
    stage 04 explicitly leaves SeqID/Position/RefAllele/AltAllele empty for
    those. So the correct filter is "does this row actually have a
    coordinate", not "is its Status label in some allowed set" — checking
    Status alone let 59 coordinate-less OMIM rows through in an earlier
    version of this stage, which then failed at stage 08 with no genome
    file matching an empty SeqID.
    """
    keep_statuses = {"resolved", "ambiguous"}
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []

    for row in rows:
        if row["Status"] not in keep_statuses:
            continue
        if not row["SeqID"]:
            continue

        key = _dedup_key(row)
        if key not in groups:
            groups[key] = {
                "Token": row["Token"],
                "SeqID": row["SeqID"],
                "Position": row["Position"],
                "RefAllele": row["RefAllele"],
                "AltAllele": row["AltAllele"],
                "Sources": set(),
                # "resolved" only if EVERY contributing row was resolved;
                # a variant supported by even one ambiguous source is
                # itself flagged ambiguous, since that reflects genuine
                # uncertainty about the underlying source data.
                "Status": "resolved",
            }
            order.append(key)

        groups[key]["Sources"].add(row["Source"])
        if row["Status"] == "ambiguous":
            groups[key]["Status"] = "ambiguous"

    deduped = []
    for key in order:
        g = groups[key]
        deduped.append({**g, "Sources": ",".join(sorted(g["Sources"]))})
    return deduped


def assign_var_ids(deduped: list[dict]) -> list[dict]:
    """Sequential Var_ID assignment in processing order — see module
    docstring for why this deviates from the original's manual Excel
    numbering.
    """
    for i, row in enumerate(deduped, start=1):
        row["Var_ID"] = f"KEGG_{i}"
    return deduped


def main() -> None:
    CONFIG.ensure_dirs()
    in_path = CONFIG.checkpoint("variant_coordinates.tsv")
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} (Token, Source, ID) coordinate rows.")

    deduped = merge_and_dedup(rows)
    deduped = assign_var_ids(deduped)

    n_input_with_coords = sum(1 for r in rows if r["Status"] in ("resolved", "ambiguous"))
    n_resolved = sum(1 for r in deduped if r["Status"] == "resolved")
    n_ambiguous = sum(1 for r in deduped if r["Status"] == "ambiguous")
    n_multi_source = sum(1 for r in deduped if "," in r["Sources"])

    logger.info(
        f"{n_input_with_coords} coordinate-bearing rows collapsed to "
        f"{len(deduped)} distinct variants ({n_resolved} resolved, "
        f"{n_ambiguous} ambiguous; {n_multi_source} confirmed by multiple "
        f"independent sources)."
    )

    if CONFIG.max_variants is not None and len(deduped) > CONFIG.max_variants:
        logger.info(
            f"Capping to CONFIG.max_variants={CONFIG.max_variants} "
            f"(smoke-scale run) — dropping {len(deduped) - CONFIG.max_variants} variants."
        )
        deduped = deduped[: CONFIG.max_variants]

    out_path = CONFIG.checkpoint("variants_deduped.tsv")
    columns = ["Var_ID", "Token", "SeqID", "Position", "RefAllele", "AltAllele", "Sources", "Status"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(deduped)

    logger.info(f"Wrote {len(deduped)} deduped variants (Var_ID KEGG_1..KEGG_{len(deduped)}) to {out_path}")


if __name__ == "__main__":
    main()
