"""Thin direct wrapper around the KEGG REST API (https://rest.kegg.jp).

Deliberately NOT using the `kegg_pull` package: as of kegg_pull 3.2.3 (the
latest release on PyPI), every call into its Python API triggers an internal
bootstrap step (`AbstractKEGGurl.organism_set`) that hits
`GET https://rest.kegg.jp/list/organism`, which now returns 400 Bad Request
-- KEGG appears to have deprecated the "organism" database name from the
`list` operation in favor of "genome" (`GET /list/genome` returns 200 with
equivalent data). Confirmed via plain `curl`, independent of any proxy or
kegg_pull code path. There is no newer kegg_pull release that fixes this, so
this module talks to the REST API directly instead.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

KEGG_REST_BASE = "https://rest.kegg.jp"


def _get(path: str, *, n_tries: int = 3, timeout: int = 30, sleep_time: float = 2.0) -> str:
    """GET a KEGG REST endpoint, returning the raw text body.

    Raises RuntimeError on a non-200 response after retries are exhausted.
    """
    url = f"{KEGG_REST_BASE}/{path}"
    last_status = None
    last_error: Exception | None = None

    for attempt in range(1, n_tries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as e:
            last_error = e
            logger.warning(f"KEGG request failed ({attempt}/{n_tries}): {url} — {e}")
            time.sleep(sleep_time)
            continue

        if resp.status_code == 200:
            return resp.text
        last_status = resp.status_code
        logger.warning(
            f"KEGG request returned {resp.status_code} ({attempt}/{n_tries}): {url}"
        )
        time.sleep(sleep_time)

    if last_error is not None:
        raise RuntimeError(f"KEGG request failed after {n_tries} tries: {url}") from last_error
    raise RuntimeError(
        f"KEGG request failed with status {last_status} after {n_tries} tries: {url}"
    )


def list_database(database: str) -> str:
    """GET /list/<database> — tab-separated (id, description) rows."""
    return _get(f"list/{database}")


def link(target_database: str, source_database: str) -> str:
    """GET /link/<target_database>/<source_database> — tab-separated
    (source_id, target_id) rows. KEGG's link direction is order-sensitive;
    the reverse order can silently return an empty body rather than erroring.
    """
    return _get(f"link/{target_database}/{source_database}")


def get_entry(entry_id: str) -> str:
    """GET /get/<entry_id> — the full flat-file text of one KEGG entry."""
    return _get(f"get/{entry_id}")


def get_entries(entry_ids: list[str], batch_size: int = 10) -> dict[str, str]:
    """GET /get/<id1>+<id2>+... in batches (KEGG caps at 10 IDs per request),
    returning {entry_id: entry_text}. When a batch has fewer 404s than
    requested, KEGG silently omits the missing records rather than erroring
    — so records are matched back to IDs by parsing the ENTRY line each
    record itself echoes back (e.g. "ENTRY   N00001   Network"), NOT by
    position. Positional matching would silently misattribute every record
    after the first gap in a batch with any missing entries.
    """
    results: dict[str, str] = {}
    # Map each requested ID's bare form (no "hsa_var:"/"path:" prefix, which
    # KEGG's ENTRY line never includes) back to the exact ID string the
    # caller passed in, so results are keyed identically to entry_ids.
    bare_to_requested = {eid.split(":", 1)[-1]: eid for eid in entry_ids}

    for i in range(0, len(entry_ids), batch_size):
        batch = entry_ids[i : i + batch_size]
        text = _get(f"get/{'+'.join(batch)}")
        records = [r for r in text.split("///") if r.strip()]
        if len(records) != len(batch):
            logger.warning(
                f"Batch get returned {len(records)} records for {len(batch)} "
                f"requested IDs ({batch}) — some entries are missing/obsolete "
                "(matched by ENTRY line, not position, so no misattribution)."
            )
        for record in records:
            first_line = record.strip().splitlines()[0] if record.strip() else ""
            # "ENTRY       N00001                      Network" -> "N00001"
            parts = first_line.split()
            if len(parts) < 2 or parts[0] != "ENTRY":
                logger.warning(f"Could not parse ENTRY line from record: {first_line!r}")
                continue
            bare_id = parts[1]
            requested_id = bare_to_requested.get(bare_id, bare_id)
            results[requested_id] = record.strip() + "\n///"
    return results
