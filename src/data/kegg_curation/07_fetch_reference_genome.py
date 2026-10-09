"""Stage 07 — Download the reference genome chromosomes needed for sequence
generation.

Recreates BioReason's KEGG_Data_1.ipynb step 20 (reference genome
acquisition), but more targeted: rather than downloading the full GRCh38
assembly (~3GB, includes unplaced scaffolds/mitochondrial DNA/alt
haplotypes never referenced by any KEGG variant) and subsetting locally
with the `seqkit` CLI as the original did, this stage fetches only the
individual chromosome accessions that stage 05's deduped variants actually
reference — directly via NCBI Entrez efetch (db=nuccore, rettype=fasta),
which returns a single chromosome's full FASTA in one request. No `seqkit`
or other external CLI dependency; pure Python (`requests` + `Bio.SeqIO` for
validation), per design decision (see docs/applications/kegg_curation_pipeline.md).

Requires NCBI_ENTREZ_EMAIL to be set (same NCBI usage-policy requirement as
stage 04 — see entrez_client.py).

Input:  checkpoints/variants_deduped.tsv    (stage 05, for the set of
        distinct SeqID chromosome accessions actually needed)
Output: genome/{seq_id}.fasta                one file per needed chromosome,
        under CONFIG.genome_dir

Usage:
    python -m src.data.kegg_curation.07_fetch_reference_genome
"""

import csv
import logging

from src.data.kegg_curation import entrez_client
from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def needed_chromosomes() -> set[str]:
    """Distinct SeqID (RefSeq chromosome accession) values referenced by
    stage 05's deduped variants.
    """
    in_path = CONFIG.checkpoint("variants_deduped.tsv")
    seq_ids: set[str] = set()
    with open(in_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["SeqID"]:
                seq_ids.add(row["SeqID"])
    return seq_ids


def main() -> None:
    CONFIG.ensure_dirs()
    seq_ids = sorted(needed_chromosomes())
    logger.info(f"{len(seq_ids)} distinct chromosome(s) needed: {seq_ids}")

    CONFIG.genome_dir.mkdir(parents=True, exist_ok=True)

    n_downloaded = 0
    n_cached = 0
    n_failed = 0

    for seq_id in seq_ids:
        out_path = CONFIG.genome_dir / f"{seq_id}.fasta"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_cached += 1
            logger.info(f"{seq_id}: already downloaded, skipping.")
            continue

        logger.info(f"Fetching {seq_id}...")
        try:
            text = entrez_client.efetch_fasta(seq_id)
        except RuntimeError as e:
            logger.warning(f"Failed to fetch {seq_id}: {e}")
            n_failed += 1
            continue

        if not text.startswith(">"):
            logger.warning(f"Unexpected response for {seq_id} (doesn't start with '>') — skipping.")
            n_failed += 1
            continue

        out_path.write_text(text)
        n_downloaded += 1
        logger.info(f"{seq_id}: wrote {len(text):,} bytes to {out_path}")

    logger.info(
        f"Done: {n_downloaded} downloaded, {n_cached} already cached, "
        f"{n_failed} failed, out of {len(seq_ids)} needed chromosomes."
    )


if __name__ == "__main__":
    main()
