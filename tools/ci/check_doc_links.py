#!/usr/bin/env python3
"""Verify that relative links in tracked Markdown files resolve to real paths.

Run from the repository root:

    python tools/ci/check_doc_links.py

Three link forms are checked: inline ``[text](target)`` links and images,
reference-style ``[label]: target`` definitions, and HTML ``src=``/``href=``
attributes. Only relative targets are checked; external URLs, ``mailto:`` and
pure anchor fragments are skipped, as is the fragment part of any link (anchor
text is not validated). Exits non-zero and lists every unresolved target.

Known-broken targets that predate the docs restructure are listed in
``KNOWN_BROKEN`` so the check can gate new breakage without first requiring
unrelated cleanup. Remove entries as the underlying links are fixed or the
referring documents are retired.
"""
import os
import re
import subprocess
import sys
from urllib.parse import unquote

# ](target) and ![alt](target)
INLINE_RE = re.compile(r'\]\(([^)\s]+?)(?:\s+"[^"]*")?\)')
# [label]: target   (reference-style definition, at the start of a line)
REFDEF_RE = re.compile(r'^\[[^\]]+\]:[ \t]*<?([^>\s]+)>?[ \t]*$', re.MULTILINE)
# <img src="..."> / <a href='...'>
HTML_RE = re.compile(r'(?:src|href)=["\']([^"\']+)["\']')
LINK_RES = (INLINE_RE, REFDEF_RE, HTML_RE)
SKIP_RE = re.compile(r'^(https?:|mailto:|ftp:|data:|#|<|\{)')

# (referring file, link target) pairs that are known to be broken and are not
# worth fixing yet. Each points at a path that does not exist in the repository
# at any revision. Keep this as small as possible: the check below fails if an
# entry here stops matching, so a stale allowlist cannot accumulate silently.
KNOWN_BROKEN = {
    # analyze_ts_data.py has never been tracked in this repository.
    ("docs/modalities/timeseries.md", "scripts/perlmutter/analyze_ts_data.py"),
}


def tracked_markdown():
    out = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8")
    for path in out.split("\0"):
        if path.endswith(".md") and os.path.isfile(path):
            yield path


def main():
    broken = []
    stale_allowlist = set(KNOWN_BROKEN)
    checked = 0

    for path in tracked_markdown():
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

        directory = os.path.dirname(path)
        matches = []
        for pattern in LINK_RES:
            matches.extend(pattern.finditer(text))
        matches.sort(key=lambda m: m.start())

        for match in matches:
            raw = match.group(1)
            if SKIP_RE.match(raw):
                continue
            target = raw.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(directory, unquote(target)))
            if os.path.exists(resolved):
                continue
            if (path, raw) in KNOWN_BROKEN:
                stale_allowlist.discard((path, raw))
                continue
            line = text[: match.start()].count("\n") + 1
            broken.append((path, line, raw))

    print(f"Checked {checked} relative links across tracked Markdown files.")

    if broken:
        print(f"\n{len(broken)} unresolved link(s):")
        for path, line, target in sorted(broken):
            print(f"  {path}:{line} -> {target}")

    if stale_allowlist:
        print(f"\n{len(stale_allowlist)} KNOWN_BROKEN entry(ies) no longer match a"
              f" broken link; remove them from {__file__}:")
        for path, target in sorted(stale_allowlist):
            print(f"  {path} -> {target}")

    if broken or stale_allowlist:
        return 1

    print("All relative Markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
