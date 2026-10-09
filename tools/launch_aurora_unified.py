#!/usr/bin/env python3
"""Unified PRISM launcher dispatcher.

Single entry point for the three Aurora launchers. Picks the right one
based on `--storage`, then `exec`s it with all remaining flags forwarded
verbatim. No flag duplication, no risk to the qsub path — the chosen
launcher is invoked exactly as if you'd called it directly.

Usage:
    # Pick a storage backend; everything else is launcher-native
    python tools/launch_aurora_unified.py --storage daos \\
        --id MY-RUN --design PRISM-OLMO3-E2E-PROD --nodes 2 --batch

    python tools/launch_aurora_unified.py --storage webdataset-staged \\
        --id WEB-RUN --design PRISM-IMAGE-ONLY-2N --nodes 2 \\
        --webdataset-dir /flare/<proj>/path/to/shards

    python tools/launch_aurora_unified.py --storage lustre \\
        --id LUSTRE-RUN --design PRISM-IMAGE-ONLY-1N --nodes 1

Storage backends:
    daos               -> tools/launch_aurora_daos.py
                          (DAOS-mounted container; fastest, requires DAOS access)
    webdataset-staged  -> tools/launch_aurora_web.py
                          (WebDataset shards on Lustre, staged to /tmp on each node)
    lustre             -> tools/launch_aurora.py
                          (Plain Lustre, generic)

To see the flags accepted by a specific backend:
    python tools/launch_aurora_unified.py --storage daos --help
    # ^ forwards --help to launch_aurora_daos.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Storage backend -> path to the dedicated launcher (repo-relative).
# These remain the source of truth for storage-specific behavior; this
# dispatcher is intentionally thin so launcher edits don't need a coordinated
# change here.
_LAUNCHERS: dict[str, str] = {
    "daos": "tools/launch_aurora_daos.py",
    "lustre": "tools/launch_aurora.py",
    "webdataset-staged": "tools/launch_aurora_web.py",
}


def _build_parser() -> argparse.ArgumentParser:
    # `--storage` is intentionally NOT required at the argparse level so we
    # can hand-roll the validation: a bare `--help` (no --storage) should
    # show our own usage; `--storage X --help` should forward `--help` to
    # the chosen launcher. argparse `required=True` would short-circuit
    # both flows before we see them.
    parser = argparse.ArgumentParser(
        description="Unified PRISM launcher (dispatches to storage-specific launcher).",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All flags after --storage are passed verbatim to the selected launcher. "
            "Use `--storage <X> --help` to see that launcher's flags."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=tuple(_LAUNCHERS),
        help="Storage backend: daos | lustre | webdataset-staged",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help and exit. Combine with --storage to forward "
        "--help to the chosen launcher.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse --storage, exec the corresponding launcher with everything else."""
    parser = _build_parser()
    args, rest = parser.parse_known_args(argv)

    # Bare --help (or no args at all): show dispatcher help, exit cleanly
    # for --help, exit 2 for empty argv (matching argparse's missing-required
    # behavior).
    if args.storage is None:
        if args.help:
            parser.print_help()
            return 0
        parser.print_usage(sys.stderr)
        print(
            f"{parser.prog}: error: the following arguments are required: --storage",
            file=sys.stderr,
        )
        return 2

    launcher_path = REPO_ROOT / _LAUNCHERS[args.storage]
    if not launcher_path.exists():
        print(f"ERROR: launcher {launcher_path} not found", file=sys.stderr)
        return 1

    # Build the forwarded command. `--storage X --help` should produce
    # `python launcher.py --help` so the user sees the launcher's flag list.
    cmd = [sys.executable, str(launcher_path), *rest]
    if args.help:
        cmd.append("--help")

    # execvp replaces this process — the launcher's stdout/stderr/exit code
    # become ours. Signals (SIGINT from Ctrl-C, SIGTERM from qdel) propagate
    # cleanly without a subprocess.run shim.
    os.execvp(cmd[0], cmd)
    # Unreachable: execvp either succeeds (no return) or raises OSError.


if __name__ == "__main__":
    raise SystemExit(main())
