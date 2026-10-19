"""Lightweight constants shared by AstroPRISM spectrum data paths.

Keep this module free of extraction-only dependencies such as h5py so the
training runtime can import the mmap dataset from the packed environment.
"""

ASTROPRISM_SPECTRUM_LENGTH = 7781
ASTROPRISM_SPECTRUM_TOKENS = 273

__all__ = [
    "ASTROPRISM_SPECTRUM_LENGTH",
    "ASTROPRISM_SPECTRUM_TOKENS",
]
