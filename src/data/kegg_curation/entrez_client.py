"""Thin wrapper around NCBI E-utilities for ClinVar/dbSNP/OMIM coordinate
resolution — used by stage 04.

Replaces the original notebooks' manual `esearch | efetch` shell pipeline
(via the `edirect` CLI toolkit) with direct HTTP calls to the same
E-utilities endpoints. Both ClinVar and dbSNP expose a `canonical_spdi` /
`SPDI` docsum field in the same `seq_id:position:ref:alt` format (0-based
position), which is the one coordinate representation this module extracts
for both — avoids needing separate parsing paths per database.
"""

import logging
import time
import xml.etree.ElementTree as ET

import requests
from src.data.kegg_curation.config import CONFIG

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI asks for an email/tool identifier on E-utilities requests, and rate-
# limits unauthenticated traffic to ~3 req/s (raised to ~10 req/s with an
# API key). Both come from CONFIG (env vars NCBI_ENTREZ_EMAIL/NCBI_API_KEY)
# — never hardcode a placeholder email here: NCBI's usage policy requires a
# real, identifying contact address on every request.
_NCBI_TOOL = "BaseMM_PRISM-kegg_curation"
_MIN_REQUEST_INTERVAL_NO_KEY = 0.35  # seconds, stays under the ~3req/s NCBI cap
_MIN_REQUEST_INTERVAL_WITH_KEY = 0.11  # stays under the ~10req/s cap

_last_request_time = 0.0


def _throttled_get(url: str, params: dict, *, n_tries: int = 3, timeout: int = 30) -> str:
    global _last_request_time

    if not CONFIG.ncbi_entrez_email:
        raise RuntimeError(
            "NCBI_ENTREZ_EMAIL is not set. NCBI's usage policy requires a "
            "real contact email for all E-utilities requests. Export "
            "NCBI_ENTREZ_EMAIL before running this stage."
        )

    params = {**params, "tool": _NCBI_TOOL, "email": CONFIG.ncbi_entrez_email}
    if CONFIG.ncbi_api_key:
        params["api_key"] = CONFIG.ncbi_api_key
    min_interval = (
        _MIN_REQUEST_INTERVAL_WITH_KEY if CONFIG.ncbi_api_key else _MIN_REQUEST_INTERVAL_NO_KEY
    )

    for attempt in range(1, n_tries + 1):
        elapsed = time.monotonic() - _last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        try:
            resp = requests.get(url, params=params, timeout=timeout)
            _last_request_time = time.monotonic()
        except requests.RequestException as e:
            logger.warning(f"NCBI request failed ({attempt}/{n_tries}): {url} — {e}")
            time.sleep(1.0)
            continue

        if resp.status_code == 200:
            return resp.text
        logger.warning(
            f"NCBI request returned {resp.status_code} ({attempt}/{n_tries}): {url}"
        )
        time.sleep(1.0)

    raise RuntimeError(f"NCBI request failed after {n_tries} tries: {url}")


def esearch(db: str, term: str) -> list[str]:
    """ESearch — returns matching UIDs for `term` in `db`."""
    text = _throttled_get(f"{EUTILS_BASE}/esearch.fcgi", {"db": db, "term": term})
    root = ET.fromstring(text)
    return [id_elem.text for id_elem in root.findall(".//IdList/Id") if id_elem.text]


def efetch_fasta(seq_id: str, *, seq_start: int | None = None, seq_stop: int | None = None) -> str:
    """EFetch a nucleotide sequence (or a coordinate slice of one) as plain
    FASTA text. Used for stage 07's whole-chromosome downloads
    (seq_start/seq_stop omitted) — NCBI's db=nuccore efetch also supports
    range params directly on the accession for on-demand slicing, though
    this pipeline downloads full chromosomes once rather than per-variant
    ranges (see stage 07's module docstring for the tradeoff discussion).
    """
    params = {"db": "nuccore", "id": seq_id, "rettype": "fasta", "retmode": "text"}
    if seq_start is not None:
        params["seq_start"] = seq_start
    if seq_stop is not None:
        params["seq_stop"] = seq_stop
    return _throttled_get(f"{EUTILS_BASE}/efetch.fcgi", params)


def efetch_docsum_batch(db: str, ids: list[str], batch_size: int = 200) -> dict[str, ET.Element]:
    """EFetch docsum for a batch of IDs, returning {uid: <DocumentSummary> element}.

    NCBI's docsum response shape differs by database and isn't always valid
    standalone XML: ClinVar wraps records in
    <eSummaryResult><DocumentSummarySet><DocumentSummary>...</DocumentSummary>...,
    but dbSNP's `db=snp` docsum returns one or more sibling
    <DocumentSummary> elements with NO common wrapping root at all — which
    is multiple root elements, invalid XML on its own, and raises
    ET.ParseError for any batch of 2+ IDs (confirmed directly against the
    live API: a single-ID snp request parses fine since ET treats the lone
    DocumentSummary as the root, but requesting 2 IDs together fails to
    parse). Wrapping the raw text in a synthetic root before parsing handles
    both shapes uniformly regardless of how many top-level records NCBI
    returns.
    """
    results: dict[str, ET.Element] = {}
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        text = _throttled_get(
            f"{EUTILS_BASE}/efetch.fcgi",
            {"db": db, "id": ",".join(batch), "rettype": "docsum"},
        )
        # Strip any XML declaration/DOCTYPE prolog (ClinVar's response has
        # one; dbSNP's doesn't) before wrapping — a prolog is only legal at
        # the very start of a document, not nested inside another element.
        body = text
        if body.lstrip().startswith("<?xml"):
            body = body.split("?>", 1)[1]
        doctype_start = body.find("<!DOCTYPE")
        if doctype_start != -1:
            doctype_end = body.find(">", doctype_start)
            if doctype_end != -1:
                body = body[:doctype_start] + body[doctype_end + 1 :]
        try:
            root = ET.fromstring(f"<KeggCurationWrapper>{body}</KeggCurationWrapper>")
        except ET.ParseError as e:
            logger.warning(f"Failed to parse docsum XML for {db} batch {batch}: {e}")
            continue
        for doc_summary in root.findall(".//DocumentSummary"):
            uid = doc_summary.get("uid")
            if uid:
                results[uid] = doc_summary
    return results


def parse_spdi(doc_summary: ET.Element, *, db: str) -> list[str]:
    """Extract SPDI string(s) from a docsum element.
    ClinVar: single <canonical_spdi> value.
    dbSNP: <SPDI> is a comma-separated list, one entry per possible allele.
    """
    if db == "clinvar":
        elem = doc_summary.find(".//canonical_spdi")
        return [elem.text] if elem is not None and elem.text else []
    elif db == "snp":
        elem = doc_summary.find(".//SPDI")
        if elem is not None and elem.text:
            return [s.strip() for s in elem.text.split(",") if s.strip()]
        return []
    raise ValueError(f"parse_spdi: unsupported db {db!r} (expected 'clinvar' or 'snp')")


def spdi_to_coords(spdi: str) -> tuple[str, int, str, str] | None:
    """Parses "seq_id:position:ref:alt" -> (seq_id, position, ref, alt).
    Position is 0-based, matching SPDI convention (NOT the same as the
    1-based HGVS "g." position NCBI shows elsewhere for the same variant).
    Returns None if the string doesn't have exactly 4 colon-separated parts.
    """
    parts = spdi.split(":")
    if len(parts) != 4:
        logger.warning(f"Malformed SPDI string: {spdi!r}")
        return None
    seq_id, position, ref, alt = parts
    try:
        return seq_id, int(position), ref, alt
    except ValueError:
        logger.warning(f"Malformed SPDI position in: {spdi!r}")
        return None
