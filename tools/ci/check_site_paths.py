#!/usr/bin/env python3
"""Fail when tracked code gains a new hardcoded site filesystem path.

A path like ``/lus/flare/projects/<real-project>/<real-user>/data`` is
meaningless to everyone but the one account it was written on, and publishing
it leaks the allocation and account names of an internal system. Configs are
supposed to name site roots as ``${PRISM_*}`` (see ``src/site_paths.py``) or as
a sanitized ``<placeholder>`` instead.

This is a RATCHET, not a clean gate. The repository still carries a long tail
of such paths in operational shell scripts and launcher defaults; removing them
all at once would be a large, risky change spread across work nobody is
currently touching. ``site_path_allowlist.json`` records the per-file count as
of the day the gate landed. The gate fails when a file exceeds its recorded
count, or when an unlisted file gains one — so the number can only go down.

Lowering a count is not automatic either: an allowlist entry whose file now has
FEWER occurrences also fails, with the new number to write in. That keeps the
allowlist honest instead of letting it drift upward as a permanent excuse.

Usage::

    python tools/ci/check_site_paths.py            # check, exit 1 on regression
    python tools/ci/check_site_paths.py --detail   # also print every occurrence
    python tools/ci/check_site_paths.py --write    # rewrite the allowlist
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
ALLOWLIST = os.path.join(_HERE, "site_path_allowlist.json")

#: Source trees, plus ``docs/`` minus the exemptions below. Prose is scanned
#: because a "here is how our team sets this up" paragraph is exactly where a
#: real shared path gets written down, and it lands in a public repository the
#: same as any config would.
SCANNED_PREFIXES = ("src/", "tools/", "scripts/", "tests/", "examples/", "docs/")

#: Subtrees within a scanned prefix that are exempt, and why. These are not
#: "not worth fixing" -- they are frozen artifacts where the path IS the
#: content, so rewriting one destroys the record it exists to keep:
#:
#: * ``docs/assets/`` -- provenance snapshots that must stay byte-identical.
#: * ``docs/reports/`` -- date-named run reports whose value is the exact
#:   command and output of one run on one machine.
#: * ``docs/data/public_image_sources/`` -- a dated delivery receipt and its
#:   manifests, same argument.
#:
#: ``docs/results/`` is deliberately NOT exempt, though it was until the paths
#: in it were cleaned up. Unlike ``docs/reports/`` its pages are undated and
#: the index sells them as reusable -- "how to run it", "launch commands" --
#: so a reader copies from them. Its remaining paths are on the ratchet below
#: as ordinary entries: the ones left are observational (what a given run
#: read, an error message quoted verbatim), while every runnable command in
#: it now takes a site variable or a placeholder.
#:
#: Living documentation -- guides, platform setup, skills, modality pages --
#: is NOT exempt. It is edited, it is read as instructions, and a path in it
#: is a path someone will copy.
EXEMPT_PREFIXES = (
    "docs/assets/",
    "docs/reports/",
    "docs/data/public_image_sources/",
)

#: A fragment containing any of these is a template, not a real path:
#: ``<project>``/``<user>`` placeholders, ``${VAR}``/``$VAR`` expansions, or a
#: ``...`` elision. Without this the gate would flag every sanitized example
#: it is meant to encourage.
PLACEHOLDER = re.compile(r"<[^<>/\s]+>|\$\{|\$[A-Za-z_]|\.\.\.")

#: HPC filesystem roots, followed by at least two more segments — one segment
#: (``/flare/ModCon``) is a project name that appears in prose and in the
#: allocation settings, and is not by itself a user's private path.
#:
#: ``pscratch`` is NERSC's: Perlmutter docs leaked ``/pscratch/sd/<i>/<user>``
#: paths past this gate until it was added.
SITE_PATH = re.compile(
    r"/(?:lus|flare|eagle|global|home|raid|pscratch)(?:/[^\s\"'`,;:)\]}]+){2,}"
)


def occurrences(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, fragment)`` for every real site path in ``text``."""
    found = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in SITE_PATH.finditer(line):
            fragment = match.group(0)
            if PLACEHOLDER.search(fragment):
                continue
            found.append((line_no, fragment))
    return found


def tracked_files() -> list[str]:
    """Every tracked path under a scanned prefix, from git rather than a walk."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", *SCANNED_PREFIXES],
        cwd=_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [
        path
        for path in listing.split("\0")
        if path and not path.startswith(EXEMPT_PREFIXES)
    ]


def scan() -> tuple[dict[str, int], dict[str, list[tuple[int, str]]]]:
    """Count real site paths per tracked file.

    Returns:
        ``(counts, detail)`` — counts maps path to number of occurrences and
        omits files with none; detail maps path to the occurrence list.
    """
    counts: dict[str, int] = {}
    detail: dict[str, list[tuple[int, str]]] = {}
    for path in tracked_files():
        try:
            with open(os.path.join(_ROOT, path), encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            # Binary fixtures and anything unreadable carry no reviewable path.
            continue
        found = occurrences(text)
        if found:
            counts[path] = len(found)
            detail[path] = found
    return counts, detail


def load_allowlist() -> dict[str, int]:
    """Read the recorded per-file counts, or an empty ratchet if absent."""
    try:
        with open(ALLOWLIST, encoding="utf-8") as handle:
            return dict(json.load(handle)["files"])
    except FileNotFoundError:
        return {}


def main() -> int:
    """Compare the scan against the allowlist; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", action="store_true", help="print every occurrence")
    parser.add_argument("--write", action="store_true", help="rewrite the allowlist")
    args = parser.parse_args()

    counts, detail = scan()

    if args.detail:
        for path in sorted(detail):
            for line_no, fragment in detail[path]:
                print(f"{path}:{line_no}: {fragment}")

    if args.write:
        payload = {
            "_comment": (
                "Per-file counts of hardcoded site paths, as a ratchet. See "
                "tools/ci/check_site_paths.py. Counts may only decrease; "
                "regenerate with `python tools/ci/check_site_paths.py --write`."
            ),
            "files": dict(sorted(counts.items())),
        }
        with open(ALLOWLIST, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {ALLOWLIST}: {len(counts)} files, {sum(counts.values())} paths")
        return 0

    allowed = load_allowlist()
    regressions = []
    improvements = []
    for path in sorted(set(counts) | set(allowed)):
        now, before = counts.get(path, 0), allowed.get(path, 0)
        if now > before:
            regressions.append((path, before, now))
        elif now < before:
            improvements.append((path, before, now))

    if regressions:
        print("New hardcoded site paths (a path naming a real project/user):", file=sys.stderr)
        for path, before, now in regressions:
            print(f"  {path}: {before} -> {now}", file=sys.stderr)
            for line_no, fragment in detail.get(path, []):
                print(f"      {path}:{line_no}: {fragment}", file=sys.stderr)
        print(
            "\nUse a ${PRISM_*} site variable (src/site_paths.py), an env var with a\n"
            "sanitized <placeholder> default, or a relative path. See\n"
            "docs/platforms/site_paths.md.",
            file=sys.stderr,
        )
        return 1

    if improvements:
        print("Site paths were removed — update the allowlist:", file=sys.stderr)
        for path, before, now in improvements:
            print(f"  {path}: {before} -> {now}", file=sys.stderr)
        print(
            "\nRun `python tools/ci/check_site_paths.py --write` and commit the result.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {len(counts)} allowlisted files, {sum(counts.values())} hardcoded "
        "site paths, none new."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
