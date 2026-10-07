"""Stage 06 — Attach pathway/disease/gene context to each deduped variant.

Recreates BioReason's KEGG_Data_1.ipynb steps 16-19: joins each variant back
to the KEGG network(s) its gene-variant token belongs to, then parses each
network entry's NAME/DEFINITION/PATHWAY/DISEASE/GENE fields into the
metadata stage 09 needs to generate a grounded question/answer/reasoning.

A single token commonly maps to multiple networks (e.g. "3815v1" — a KIT
kinase domain mutation — appears in both N00003, a RAS-ERK signaling
network, and N00046, a PI3K signaling network; each network implicates a
different downstream pathway and disease for the same underlying mutation).
This stage keeps ALL (variant, network) pairs rather than picking one, since
each pairing is genuinely distinct context, not a duplicate — one row per
pair, so a variant with N networks contributes N rows to the output.

Input:  checkpoints/variants_deduped.tsv         (stage 05)
        checkpoints/gene_variant_tokens.tsv       (stage 02, Token->Network)
        checkpoints/network_entries/{id}.txt      (stage 01, network text)
Output: checkpoints/variants_with_context.tsv     (Var_ID, Token, SeqID,
        Position, RefAllele, AltAllele, Sources, Status, Network,
        NetworkName, Definition, Pathway, Disease, Genes)

Usage:
    python -m src.data.kegg_curation.06_map_variants_to_networks
"""

import csv
import logging

from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_field_block(lines: list[str], field_name: str) -> list[str]:
    """Same KEGG flat-file multi-line field reader used in stages 02/03:
    field name starts the first line, indented continuation lines follow,
    a differently-named field or an unindented line ends the block.
    """
    block: list[str] = []
    in_field = False
    for line in lines:
        if line.startswith(field_name):
            in_field = True
            block.append(line[len(field_name) :].strip())
        elif in_field and line.startswith((" ", "\t")) and line.strip():
            block.append(line.strip())
        elif in_field:
            break
    return block


def _single_line_field(lines: list[str], field_name: str) -> str:
    block = _read_field_block(lines, field_name)
    return " ".join(block).strip()


def parse_network_metadata(entry_text: str) -> dict:
    """Extracts the fields stage 09 needs from one network entry's text."""
    lines = entry_text.splitlines()
    return {
        "NetworkName": _single_line_field(lines, "NAME"),
        "Definition": _single_line_field(lines, "DEFINITION"),
        # PATHWAY/DISEASE lines look like "hsa05221  Acute myeloid leukemia"
        # / "H00003  Acute myeloid leukemia" — keep the description only,
        # dropping the KEGG ID prefix, joined if multiple are present.
        "Pathway": "; ".join(
            " ".join(line.split(None, 1)[1:]).strip()
            for line in _read_field_block(lines, "PATHWAY")
            if line.split(None, 1)[1:]
        ),
        "Disease": "; ".join(
            " ".join(line.split(None, 1)[1:]).strip()
            for line in _read_field_block(lines, "DISEASE")
            if line.split(None, 1)[1:]
        ),
        # GENE lines look like "3815  KIT; KIT proto-oncogene receptor
        # tyrosine kinase" — keep the gene symbol (before the first ";").
        "Genes": ", ".join(
            line.split(None, 1)[1].split(";")[0].strip()
            for line in _read_field_block(lines, "GENE")
            if len(line.split(None, 1)) > 1
        ),
    }


def main() -> None:
    CONFIG.ensure_dirs()
    variants_path = CONFIG.checkpoint("variants_deduped.tsv")
    with open(variants_path, newline="") as f:
        variants = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(variants)} deduped variants.")

    tokens_path = CONFIG.checkpoint("gene_variant_tokens.tsv")
    token_to_networks: dict[str, list[str]] = {}
    with open(tokens_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token_to_networks.setdefault(row["Token"], []).append(row["Network"])

    entries_dir = CONFIG.checkpoint("network_entries")
    metadata_cache: dict[str, dict] = {}

    out_rows: list[dict] = []
    n_no_network = 0

    for variant in variants:
        networks = token_to_networks.get(variant["Token"], [])
        if not networks:
            n_no_network += 1
            logger.debug(f"No network found for token {variant['Token']} ({variant['Var_ID']}).")
            continue

        for network_id in networks:
            if network_id not in metadata_cache:
                entry_path = entries_dir / f"{network_id}.txt"
                if not entry_path.exists():
                    logger.warning(f"No cached network entry for {network_id} — skipping.")
                    metadata_cache[network_id] = None
                else:
                    metadata_cache[network_id] = parse_network_metadata(entry_path.read_text())

            metadata = metadata_cache[network_id]
            if metadata is None:
                continue

            out_rows.append({**variant, "Network": network_id, **metadata})

    out_path = CONFIG.checkpoint("variants_with_context.tsv")
    columns = [
        "Var_ID", "Token", "SeqID", "Position", "RefAllele", "AltAllele",
        "Sources", "Status", "Network", "NetworkName", "Definition",
        "Pathway", "Disease", "Genes",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    n_unique_variants = len({r["Var_ID"] for r in out_rows})
    logger.info(
        f"Wrote {len(out_rows)} (variant, network) rows covering "
        f"{n_unique_variants}/{len(variants)} variants "
        f"({n_no_network} variants had no matching network) to {out_path}"
    )


if __name__ == "__main__":
    main()
