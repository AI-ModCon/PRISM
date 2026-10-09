"""Stage 01 — Fetch KEGG networks, isolate variant-type ones.

Recreates BioReason's KEGG_Data_1.ipynb steps 1a-1b (see
docs/applications/kegg_curation_pipeline.md): bulk-download every KEGG "network" entry,
then filter down to the ones tagged `TYPE Variant` (the network entries that
actually describe a disease-relevant gene variant, as opposed to `TYPE
Reference`, which describes a normal/unperturbed pathway).

Input:  none (fetches fresh from the KEGG REST API)
Output: checkpoints/network_entries/{network_id}.txt  (raw KEGG flat-file text,
        one per network — for ALL networks, not just variant-type, so later
        stages needing non-variant metadata don't have to re-fetch)
        checkpoints/network_variant_ids.txt            (newline-separated list
        of network IDs whose TYPE == Variant)

Usage:
    python -m src.data.kegg_curation.01_fetch_kegg_networks
"""

import logging

from src.data.kegg_curation import kegg_rest
from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_all_network_ids() -> list[str]:
    """GET /list/network — every KEGG network entry ID (~1600+ as of 2026)."""
    text = kegg_rest.list_database("network")
    ids = [line.split("\t")[0] for line in text.splitlines() if line.strip()]
    logger.info(f"Found {len(ids)} total KEGG network entries.")
    return ids


def fetch_and_filter_networks(network_ids: list[str]) -> list[str]:
    """Bulk-fetch every network entry's flat-file text, save each to disk,
    and return the subset of IDs whose TYPE field is "Variant".
    """
    out_dir = CONFIG.checkpoint("network_entries")
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_ids: list[str] = []
    entries = kegg_rest.get_entries(network_ids, batch_size=10)

    for network_id, text in entries.items():
        (out_dir / f"{network_id}.txt").write_text(text)

        entry_type = None
        for line in text.splitlines():
            if line.startswith("TYPE"):
                entry_type = line.split(None, 1)[1].strip()
                break

        if entry_type == "Variant":
            variant_ids.append(network_id)

    logger.info(
        f"Fetched {len(entries)} network entries; {len(variant_ids)} are TYPE=Variant."
    )
    return variant_ids


def main() -> None:
    CONFIG.ensure_dirs()
    network_ids = fetch_all_network_ids()
    variant_ids = fetch_and_filter_networks(network_ids)

    out_path = CONFIG.checkpoint("network_variant_ids.txt")
    out_path.write_text("\n".join(variant_ids) + "\n")
    logger.info(f"Wrote {len(variant_ids)} variant-type network IDs to {out_path}")


if __name__ == "__main__":
    main()
