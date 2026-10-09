"""Stage 03 — Resolve each gene-variant token to cross-database IDs.

Recreates the *intent* of BioReason's KEGG_Data_1.ipynb step 7, which had no
surviving code (a one-off ChatGPT paste parsing free NETWORK-entry text into
Entry/Source/ID columns). This recreation instead uses a source of ground
truth the original didn't: each KEGG "hsa_var:<token>" entry (fetched fresh
from KEGG, not scraped from NETWORK-entry text) carries a structured
VARIATION field with direct cross-reference lines, e.g.:

    VARIATION   mutation V600E
                ClinVar: 13961 376069
                dbSNP: rs113488022 rs121913377

Primary path (zero cost, high precision): parse this field directly.
Fallback (only for tokens whose hsa_var: entry is missing/404 or has no
VARIATION field — see CONFIG.id_resolver_llm_fallback): ask an LLM to infer
Source+ID from the token's NAME/GENE/free-text context, mirroring what the
original's one-off ChatGPT step likely did.

Input:  checkpoints/gene_variant_tokens.tsv   (stage 02)
Output: checkpoints/resolved_variant_ids.tsv  (Token, Source, ID, ResolvedBy)
        one row per (token, source, id) triple — a token can resolve to
        multiple sources/IDs (e.g. both ClinVar and dbSNP for the same
        mutation), matching the original's downstream per-source handling.

Usage:
    python -m src.data.kegg_curation.03_resolve_variant_ids
"""

import csv
import logging
import re

from src.data.kegg_curation import kegg_rest
from src.data.kegg_curation.config import CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cross-DB source labels as they appear in KEGG's VARIATION field, mapped to
# the canonical Source name used downstream (stage 04). Case-sensitive as
# KEGG writes them, but matched case-insensitively below for robustness.
KNOWN_SOURCES = {
    "clinvar": "ClinVar",
    "dbsnp": "dbSNP",
    "dbvar": "dbVar",
    "omimvar": "OmimVar",
    "cosf": "COSF",
    "cosm": "COSM",
}

# e.g. "ClinVar: 13961 376069" or "dbSNP: rs113488022 rs121913377"
SOURCE_LINE_RE = re.compile(
    r"^(" + "|".join(re.escape(k) for k in KNOWN_SOURCES) + r")\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _read_variation_lines(entry_text: str) -> list[str]:
    """Every continuation line under a VARIATION field, KEGG flat-file style
    (field starts a line, continuations are indented with no field name).
    """
    lines: list[str] = []
    in_field = False
    for raw_line in entry_text.splitlines():
        if raw_line.startswith("VARIATION"):
            in_field = True
            lines.append(raw_line[len("VARIATION") :].strip())
        elif in_field and raw_line.startswith((" ", "\t")) and raw_line.strip():
            lines.append(raw_line.strip())
        elif in_field and raw_line.strip() and not raw_line.startswith((" ", "\t")):
            in_field = False
    return lines


def parse_variation_field(entry_text: str) -> list[tuple[str, str]]:
    """Returns [(Source, ID), ...] parsed directly from a hsa_var: entry's
    VARIATION field. Empty if the entry has no VARIATION field or no
    recognized source label within it.
    """
    results: list[tuple[str, str]] = []
    for line in _read_variation_lines(entry_text):
        m = SOURCE_LINE_RE.match(line)
        if not m:
            continue
        source = KNOWN_SOURCES[m.group(1).lower()]
        ids = m.group(2).split()
        for id_ in ids:
            results.append((source, id_))
    return results


def resolve_via_llm(token: str, description: str, entry_text: str | None) -> list[tuple[str, str]]:
    """Fallback resolver for tokens with no direct VARIATION field match.
    Only invoked when CONFIG.id_resolver_llm_fallback is True.
    """
    from src.data.kegg_curation.reasoning import get_reasoning_backend

    backend = get_reasoning_backend()
    context = entry_text or f"KEGG variant token {token}: {description}"
    prompt = (
        "You are a genetics database expert. Given the following KEGG gene "
        "variant entry, identify any cross-references to ClinVar, dbSNP, "
        "dbVar, OMIM (OmimVar), or COSMIC (COSM/COSF) databases. Respond "
        "with ONLY a JSON list of [source, id] pairs, e.g. "
        '[["ClinVar", "12582"], ["dbSNP", "rs121913529"]]. If none are '
        "identifiable, respond with an empty JSON list: []\n\n"
        f"KEGG entry:\n{context}"
    )
    try:
        raw = backend.complete(prompt)
        import json

        pairs = json.loads(raw)
        return [(str(s), str(i)) for s, i in pairs]
    except Exception as e:
        logger.warning(f"LLM fallback failed for token {token}: {e}")
        return []


def main() -> None:
    CONFIG.ensure_dirs()
    tokens_path = CONFIG.checkpoint("gene_variant_tokens.tsv")
    rows: list[dict] = []
    with open(tokens_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    unique_tokens = list(dict.fromkeys(r["Token"] for r in rows))
    descriptions = {r["Token"]: r["Description"] for r in rows}
    logger.info(f"Resolving {len(unique_tokens)} unique gene-variant tokens...")

    entries = kegg_rest.get_entries([f"hsa_var:{t}" for t in unique_tokens])
    # kegg_rest.get_entries keys results by the exact ID string it was given
    # ("hsa_var:<token>"); missing/404'd tokens simply won't be present.

    out_rows: list[tuple[str, str, str, str]] = []
    n_direct = 0
    n_llm = 0
    n_unresolved = 0

    for token in unique_tokens:
        entry_text = entries.get(f"hsa_var:{token}")
        resolved: list[tuple[str, str]] = []

        if entry_text:
            resolved = parse_variation_field(entry_text)

        if resolved:
            n_direct += 1
            for source, id_ in resolved:
                out_rows.append((token, source, id_, "direct"))
            continue

        if CONFIG.id_resolver_llm_fallback:
            resolved = resolve_via_llm(token, descriptions.get(token, ""), entry_text)
            if resolved:
                n_llm += 1
                for source, id_ in resolved:
                    out_rows.append((token, source, id_, "llm"))
                continue

        n_unresolved += 1
        logger.debug(f"Token {token} could not be resolved to any cross-DB ID.")

    out_path = CONFIG.checkpoint("resolved_variant_ids.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Token", "Source", "ID", "ResolvedBy"])
        writer.writerows(out_rows)

    logger.info(
        f"Resolved {n_direct} tokens directly, {n_llm} via LLM fallback, "
        f"{n_unresolved} unresolved. Wrote {len(out_rows)} (Token, Source, ID) rows to {out_path}"
    )


if __name__ == "__main__":
    main()
