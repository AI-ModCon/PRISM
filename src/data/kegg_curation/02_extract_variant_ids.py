"""Stage 02 — Extract gene-variant tokens from variant-type network entries.

Recreates BioReason's KEGG_Data_1.ipynb steps 1c-1d: scrapes KEGG's internal
"hsa:GENEv#" variant notation (e.g. "25v1") out of each TYPE=Variant network
entry.

Improvement over the original: the original notebook regex-scraped the
`EXPANDED` line only (`grep -oE "[0-9]+v[0-9]+"`), discarding whatever
free-text description KEGG attaches to each variant token. KEGG entries
actually carry a structured `VARIANT` field for this
(e.g. "VARIANT     25v1 (BCR-ABL)  BCR-ABL1 fusion") when present, so this
stage prefers that field (keeping the description — useful context for
stage 03's ID resolver) and falls back to the `EXPANDED`-line regex scrape
only for entries that lack a `VARIANT` field.

Input:  checkpoints/network_variant_ids.txt         (stage 01)
        checkpoints/network_entries/{id}.txt         (stage 01)
Output: checkpoints/gene_variant_tokens.tsv           (Network, Token, Description)

Usage:
    python -m src.data.kegg_curation.02_extract_variant_ids
"""

import csv
import logging
import re

from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches KEGG's internal gene-variant token, e.g. "25v1", "6654v2".
TOKEN_RE = re.compile(r"\b(\d+v\d+)\b")

# A "VARIANT" field line: token, optional "(ALIAS)", then free-text description.
# e.g. "VARIANT     25v1 (BCR-ABL)  BCR-ABL1 fusion"
#      "            25v2 (BCR-ABL)  First generation TKI-resistant ABL1 mutation"
VARIANT_LINE_RE = re.compile(
    r"^\s*(?:VARIANT\s+)?(\d+v\d+)\s*(?:\([^)]*\))?\s*(.*)$"
)


def _read_field_block(lines: list[str], field_name: str) -> list[str]:
    """KEGG flat-file fields span multiple lines: the field name starts the
    first line, continuation lines are indented with no field name. Returns
    the raw lines belonging to `field_name` (field name stripped from the
    first line, e.g. "VARIANT     25v1 ..." -> "25v1 ...").
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


def extract_tokens_from_entry(entry_text: str) -> list[tuple[str, str]]:
    """Returns [(token, description), ...] for one network entry's text.
    Prefers the structured VARIANT field; falls back to regex-scraping the
    EXPANDED line for entries that lack a VARIANT field.
    """
    lines = entry_text.splitlines()

    variant_block = _read_field_block(lines, "VARIANT")
    if variant_block:
        results = []
        for line in variant_block:
            m = VARIANT_LINE_RE.match(line)
            if m:
                results.append((m.group(1), m.group(2).strip()))
        if results:
            return results

    # Fallback: regex-scrape any gene-variant tokens out of the EXPANDED
    # line (or anywhere else in the entry), with no description available.
    expanded_block = _read_field_block(lines, "EXPANDED")
    search_text = " ".join(expanded_block) if expanded_block else entry_text
    tokens = TOKEN_RE.findall(search_text)
    return [(t, "") for t in dict.fromkeys(tokens)]  # dedupe, preserve order


def main() -> None:
    CONFIG.ensure_dirs()
    variant_ids_path = CONFIG.checkpoint("network_variant_ids.txt")
    network_ids = [
        line.strip() for line in variant_ids_path.read_text().splitlines() if line.strip()
    ]
    logger.info(f"Loaded {len(network_ids)} variant-type network IDs.")

    entries_dir = CONFIG.checkpoint("network_entries")
    rows: list[tuple[str, str, str]] = []
    missing = 0

    for network_id in network_ids:
        entry_path = entries_dir / f"{network_id}.txt"
        if not entry_path.exists():
            missing += 1
            continue
        entry_text = entry_path.read_text()
        for token, description in extract_tokens_from_entry(entry_text):
            rows.append((network_id, token, description))

    if missing:
        logger.warning(f"{missing} network IDs had no cached entry file — skipped.")

    out_path = CONFIG.checkpoint("gene_variant_tokens.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Network", "Token", "Description"])
        writer.writerows(rows)

    n_unique_tokens = len({r[1] for r in rows})
    logger.info(
        f"Wrote {len(rows)} (Network, Token) rows ({n_unique_tokens} unique tokens) to {out_path}"
    )


if __name__ == "__main__":
    main()
