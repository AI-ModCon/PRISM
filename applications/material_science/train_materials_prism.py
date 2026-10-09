#!/usr/bin/env python3
"""Convenience entry point for PRISM materials regression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.materials.prism_train import main  # noqa: E402

if __name__ == "__main__":
    main()
