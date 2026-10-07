"""Check that documented counts and capabilities match the code.

``tools/ci/check_doc_links.py`` validates that a link *resolves*; nothing
validates that a sentence is *true*. That gap is how "six modalities" (there
are seven), "24 of 24 configs" (there are 26), and "all six are re-exported
from ``src.encoders``" (eight are) each reached ``main``.

These tests derive the numbers from the code and assert the docs agree, so a
new modality or model config fails here rather than silently dating the prose.
Deliberately narrow: only claims with a single unambiguous source of truth.
"""
import glob
import os
import re

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(relpath: str) -> str:
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as handle:
        return handle.read()


def _model_config_count() -> int:
    return len(glob.glob(os.path.join(_ROOT, "src", "conf", "model", "*.yaml")))


def test_every_modality_appears_in_the_readme_support_table() -> None:
    """A new `Modality` member must gain a row in README's support table."""
    from src.modalities import Modality

    table = _read("README.md").split("### Modality support status", 1)
    assert len(table) == 2, "README lost its 'Modality support status' heading"
    # The table ends at the next h2/h3.
    body = re.split(r"\n#{2,3} ", table[1], maxsplit=1)[0]

    # Only the first cell of each row counts. Matching the whole section would
    # let a modality named once in the prose above the table pass for a row.
    names = {
        row.split("|")[1].strip().strip("*").lower()
        for row in body.splitlines()
        if row.lstrip().startswith("|") and row.count("|") >= 4
    }

    missing = [
        m.value
        for m in Modality
        # "time_series" is spelled "Time series" in prose.
        if m.value.replace("_", " ") not in names
    ]
    assert not missing, (
        f"src.modalities.Modality has {len(list(Modality))} members but README's "
        f"support table never mentions {missing}. Add a row, or the table will "
        f"keep claiming PRISM has fewer modalities than it ships."
    )


def test_readme_shipped_config_counts_use_the_real_denominator() -> None:
    """Every "N of M" in the support table must use the real config count."""
    total = _model_config_count()
    body = _read("README.md").split("### Modality support status", 1)[1]
    body = re.split(r"\n#{2,3} ", body, maxsplit=1)[0]

    wrong = {
        int(d)
        for d in re.findall(r"\b\d+ of (\d+)\b", body)
        if int(d) != total
    }
    assert not wrong, (
        f"README's support table counts out of {sorted(wrong)}, but "
        f"src/conf/model/ holds {total} configs. Update the denominators."
    )


def test_no_doc_claims_timesfm_is_the_time_series_encoder() -> None:
    """TimesFM was documented for a backend that was never implemented."""
    src_hits = [
        path
        for path in glob.glob(os.path.join(_ROOT, "src", "**", "*.py"), recursive=True)
        if "timesfm" in _read(os.path.relpath(path, _ROOT)).lower()
    ]
    if src_hits:
        pytest.skip("TimesFM now exists under src/; this guard is obsolete")

    # Frozen run reports keep their original wording.
    exempt = ("docs/assets/", "docs/reports/", "docs/results/")
    offenders = []
    for path in glob.glob(os.path.join(_ROOT, "docs", "**", "*.md"), recursive=True):
        rel = os.path.relpath(path, _ROOT)
        if rel.startswith(exempt):
            continue
        if "timesfm" in _read(rel).lower():
            offenders.append(rel)
    offenders += [p for p in ("README.md",) if "timesfm" in _read(p).lower()]

    assert not offenders, (
        f"{offenders} name TimesFM as a time-series encoder, but no TimesFM "
        f"code or model id exists under src/. The real backends are linear, "
        f"moirai, intern_s2, intern_s2_397b and timeomni."
    )


def test_encoders_all_count_matches_the_api_page() -> None:
    """`docs/api/encoders.md` states how many encoders `src.encoders` exports."""
    # Parsed rather than imported: `src.encoders` pulls in torch, which would
    # make a docs assertion depend on an accelerator runtime being installed.
    source = _read(os.path.join("src", "encoders", "__init__.py"))
    block = re.search(r"__all__\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert block, "src/encoders/__init__.py no longer declares a literal __all__"
    actual = len(re.findall(r'"([^"]+)"', block.group(1)))

    page = _read("docs/api/encoders.md")

    words = {
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    claimed = {
        words[m.group(1).lower()]
        for m in re.finditer(
            r"\b(five|six|seven|eight|nine|ten)\b(?=[^.]{0,40}\bencoders?\b)",
            page,
            re.IGNORECASE,
        )
    }
    wrong = {n for n in claimed if n != actual}
    assert not wrong, (
        f"docs/api/encoders.md says {sorted(wrong)} encoders; "
        f"src.encoders.__all__ exports {actual}."
    )
