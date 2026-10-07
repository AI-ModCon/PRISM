"""Stage 08 — Generate reference/variant nucleotide sequences.

Recreates BioReason's KEGG_Data_1.ipynb step 22 (sequence window
extraction), with two deliberate corrections against the original:

  1. Consistent coordinate system: the original mixed ENST (transcript,
     sometimes 1-based) and genomic (RefSeq, 0- or 1-based depending on
     source) coordinates, needing a per-row branch to decide the indexing
     offset, and still logged verification mismatches it never
     auto-corrected (KEGG_Data_1 step 21). This pipeline instead resolves
     EVERY variant to NCBI's SPDI representation in stage 04 (via
     canonical_spdi/SPDI docsum fields), which is uniformly 0-based — so
     slicing is a single, consistent code path with no per-source branching.
  2. Fixed the dead "deletion" branch bug: the original checked
     `if variant_allele == "deletion":` to special-case deletions, but real
     AltAllele values are always nucleotide strings (never the literal
     string "deletion"), so that branch never actually ran in any of the
     three original notebooks — true deletion variants (AltAllele == "",
     which DOES occur in real SPDI data) always fell through to the
     substitution/insertion branch and got a spliced-in empty string
     instead of correctly-shortened reference sequence. This stage checks
     `AltAllele == ""` directly instead.

Single configured window (CONFIG.sequence_window, default 2000nt per side)
— the original regenerated sequences 3 times across 3 notebooks with 2
different window values (1000, then 2000 twice) with no single place
reconciling them; this pipeline has exactly one.

Input:  checkpoints/variants_with_context.tsv   (stage 06)
        genome/{seq_id}.fasta                    (stage 07)
Output: checkpoints/sequences/{Var_ID}.json       one file per variant:
        {"reference_sequence": ..., "variant_sequence": ...}
        checkpoints/variants_with_sequences.tsv   variants_with_context.tsv
        plus a SequenceStatus column
        (ok / ref_mismatch / missing_genome / out_of_range)

Usage:
    python -m src.data.kegg_curation.08_generate_sequences
"""

import csv
import json
import logging

from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_genome_cache: dict[str, str] = {}


def _load_chromosome(seq_id: str) -> str | None:
    """Lazily loads and caches one chromosome's sequence (as a single
    uppercase string, header line stripped) from genome/{seq_id}.fasta.
    """
    if seq_id in _genome_cache:
        return _genome_cache[seq_id]

    fasta_path = CONFIG.genome_dir / f"{seq_id}.fasta"
    if not fasta_path.exists():
        logger.warning(f"No downloaded genome file for {seq_id} (expected {fasta_path}).")
        return None

    lines = fasta_path.read_text().splitlines()
    # First line is the ">accession description" header; the rest is the
    # sequence, possibly wrapped at a fixed line width.
    seq = "".join(line.strip() for line in lines[1:]).upper()
    _genome_cache[seq_id] = seq
    return seq


def generate_variant_sequences(
    chrom_seq: str, position: int, ref_allele: str, alt_allele: str, window: int
) -> tuple[str, str, bool] | None:
    """Builds the reference and variant sequence windows around one variant.

    `position` is the SPDI 0-based deletion-interval start (i.e. the first
    base of `ref_allele` in `chrom_seq`, 0-indexed) — matching
    entrez_client.spdi_to_coords's documented convention directly, so no
    coordinate-system conversion is needed here.

    Returns (reference_sequence, variant_sequence, ref_matched), where
    `ref_matched` is False when the genome's bases at `position` disagree
    with the SPDI-reported `ref_allele` — the caller records that in
    SequenceStatus so the row is distinguishable downstream.

    Returns None if the requested window falls outside the chromosome's
    bounds (can happen for variants very close to a chromosome end).
    """
    region_start = position - window
    region_end = position + len(ref_allele) + window
    if region_start < 0 or region_end > len(chrom_seq):
        return None

    ref_seq = chrom_seq[region_start:region_end]

    # Position of the variant's ref_allele within the extracted window.
    local_offset = position - region_start
    actual_ref = ref_seq[local_offset : local_offset + len(ref_allele)]
    ref_matched = actual_ref == ref_allele
    if not ref_matched:
        logger.warning(
            f"Reference allele mismatch at position {position}: "
            f"expected {ref_allele!r}, genome has {actual_ref!r}. "
            "Proceeding with genome's actual bases for the reference "
            "sequence, but the variant sequence substitution below still "
            "uses the SPDI-reported ref_allele length to splice in "
            "alt_allele — a mismatch here usually means a stale SPDI vs. "
            "the currently released genome build and is worth flagging."
        )

    # Splice: keep everything before the ref_allele, insert alt_allele,
    # keep everything after where ref_allele ended. AltAllele == "" is a
    # true deletion (previously mishandled — see module docstring); this
    # works correctly for that case too, since splicing in an empty string
    # is exactly "delete this span."
    variant_seq = (
        ref_seq[:local_offset] + alt_allele + ref_seq[local_offset + len(ref_allele) :]
    )

    return ref_seq, variant_seq, ref_matched


def main() -> None:
    CONFIG.ensure_dirs()
    in_path = CONFIG.checkpoint("variants_with_context.tsv")
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} (variant, network) rows.")

    seq_dir = CONFIG.checkpoint("sequences")
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Sequences only need generating once per distinct variant (Var_ID),
    # not once per (variant, network) row — cache by Var_ID so a variant
    # appearing under multiple networks only does the work once.
    sequence_cache: dict[str, tuple[str, str, bool] | None] = {}
    status_cache: dict[str, str] = {}

    for row in rows:
        var_id = row["Var_ID"]
        if var_id in sequence_cache:
            continue

        chrom_seq = _load_chromosome(row["SeqID"])
        if chrom_seq is None:
            sequence_cache[var_id] = None
            status_cache[var_id] = "missing_genome"
            continue

        position = int(row["Position"])
        result = generate_variant_sequences(
            chrom_seq, position, row["RefAllele"], row["AltAllele"], CONFIG.sequence_window
        )
        if result is None:
            sequence_cache[var_id] = None
            status_cache[var_id] = "out_of_range"
            continue

        ref_seq, variant_seq, ref_matched = result
        sequence_cache[var_id] = result
        # A ref-allele mismatch is the one signal that catches an upstream
        # coordinate error (wrong assembly, stale SPDI, off-by-one) — the
        # sequence still looks perfectly well-formed, so if it is only
        # logged it is invisible to every downstream consumer. Record it in
        # SequenceStatus instead of collapsing it into "ok".
        status_cache[var_id] = "ok" if ref_matched else "ref_mismatch"

        (seq_dir / f"{var_id}.json").write_text(
            json.dumps({"reference_sequence": ref_seq, "variant_sequence": variant_seq})
        )

    n_ok = sum(1 for s in status_cache.values() if s == "ok")
    n_ref_mismatch = sum(1 for s in status_cache.values() if s == "ref_mismatch")
    n_missing_genome = sum(1 for s in status_cache.values() if s == "missing_genome")
    n_out_of_range = sum(1 for s in status_cache.values() if s == "out_of_range")
    logger.info(
        f"Generated sequences for {n_ok}/{len(status_cache)} distinct variants "
        f"({n_ref_mismatch} ref-allele mismatch, {n_missing_genome} missing "
        f"genome file, {n_out_of_range} out of range)."
    )
    if n_ref_mismatch:
        logger.warning(
            f"{n_ref_mismatch} variant(s) have SequenceStatus=ref_mismatch: the "
            "genome bases disagree with the SPDI-reported reference allele. "
            "Stage 09 skips them by default; review before treating them as "
            "usable (see the per-variant warnings above)."
        )

    out_path = CONFIG.checkpoint("variants_with_sequences.tsv")
    fieldnames = list(rows[0].keys()) + ["SequenceStatus"] if rows else []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "SequenceStatus": status_cache.get(row["Var_ID"], "unknown")})

    logger.info(f"Wrote {len(rows)} rows (with SequenceStatus) to {out_path}")


if __name__ == "__main__":
    main()
