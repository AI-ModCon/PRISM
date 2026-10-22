"""Select the pinned AstroPRISM raw DESI spectrum rows.

The 646-object AstroPRISM roster was originally used to extract calibrated
Legacy Survey cutouts from the official AION tutorial HDF5.  This module uses
the same pinned source, object identifiers, coordinate checks, and roster order
to extract the corresponding observed-frame DESI spectra without downloading
the 121 GB source in full.

The four output arrays are mmap-friendly and preserve the inputs expected by
``aion.modalities.DESISpectrum``: flux, inverse variance, mask, and wavelength.
In AION's contract, ``True`` mask values are excluded from normalization.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .astroprism_raw_images import (
    EXPECTED_CANDIDATES,
    RAW_SOURCE_ETAG,
    RAW_SOURCE_LENGTH,
    RAW_SOURCE_URL,
    ROSTER_SHA256,
    RawImageConfig,
    RawImageValidationError,
    _build_raw_row_map,
    _commit_staging,
    _contiguous_runs,
    _crosscheck_metadata,
    _inspect_source,
    _open_hdf5,
    _path_exists,
    _read_roster,
    _require_dataset,
    _validate_schema,
    _verify_source_unchanged,
    sha256_file,
)
from .astroprism_spectrum_contract import (
    ASTROPRISM_SPECTRUM_LENGTH,
    ASTROPRISM_SPECTRUM_TOKENS,
)


class RawSpectrumValidationError(RawImageValidationError):
    """Raised when a source violates the pinned raw-spectrum contract."""


@dataclass(frozen=True)
class _SpectrumDatasetSpec:
    source_path: str
    filename: str
    dtype: np.dtype[Any]


_SPECTRUM_DATASETS = {
    "spectrum_flux": _SpectrumDatasetSpec(
        "desi/spectrum_flux", "spectrum_flux.npy", np.dtype("float32")
    ),
    "spectrum_ivar": _SpectrumDatasetSpec(
        "desi/spectrum_ivar", "spectrum_ivar.npy", np.dtype("float32")
    ),
    "spectrum_mask": _SpectrumDatasetSpec(
        "desi/spectrum_mask", "spectrum_mask.npy", np.dtype("bool")
    ),
    "spectrum_wavelength": _SpectrumDatasetSpec(
        "desi/spectrum_lambda", "spectrum_wavelength.npy", np.dtype("float32")
    ),
}


@dataclass(frozen=True)
class RawSpectrumConfig:
    """Inputs and validation pins for one selective spectrum extraction."""

    roster_path: Path
    output_dir: Path
    source: str | Path = RAW_SOURCE_URL
    overwrite: bool = False
    expected_count: int = EXPECTED_CANDIDATES
    expected_roster_sha256: str | None = ROSTER_SHA256
    expected_source_length: int | None = RAW_SOURCE_LENGTH
    expected_source_etag: str | None = RAW_SOURCE_ETAG
    http_block_size: int = 256 * 1024
    http_timeout_seconds: float = 30.0
    http_probe_attempts: int = 3
    http_probe_backoff_seconds: float = 1.0
    roster_coordinate_tolerance_arcsec: float = 0.1
    desi_coordinate_tolerance_arcsec: float = 1e-4
    legacy_coordinate_tolerance_arcsec: float = 1.0
    caption_legacy_coordinate_tolerance_arcsec: float = 1e-4


def _common_config(config: RawSpectrumConfig) -> RawImageConfig:
    """Adapt to the shared pinned-source and identity-validation helpers."""

    return RawImageConfig(
        roster_path=config.roster_path,
        output_dir=config.output_dir,
        source=config.source,
        overwrite=config.overwrite,
        expected_count=config.expected_count,
        expected_roster_sha256=config.expected_roster_sha256,
        expected_source_length=config.expected_source_length,
        expected_source_etag=config.expected_source_etag,
        http_block_size=config.http_block_size,
        http_timeout_seconds=config.http_timeout_seconds,
        http_probe_attempts=config.http_probe_attempts,
        http_probe_backoff_seconds=config.http_probe_backoff_seconds,
        roster_coordinate_tolerance_arcsec=config.roster_coordinate_tolerance_arcsec,
        desi_coordinate_tolerance_arcsec=config.desi_coordinate_tolerance_arcsec,
        legacy_coordinate_tolerance_arcsec=config.legacy_coordinate_tolerance_arcsec,
        caption_legacy_coordinate_tolerance_arcsec=(
            config.caption_legacy_coordinate_tolerance_arcsec
        ),
    )


def _validate_spectrum_schema(
    handle: Any,
    *,
    source_count: int,
    require_contiguous: bool,
) -> None:
    for spec in _SPECTRUM_DATASETS.values():
        _require_dataset(
            handle,
            spec.source_path,
            shape=(source_count, ASTROPRISM_SPECTRUM_LENGTH),
            dtype=spec.dtype,
            require_contiguous=require_contiguous,
        )


def _validate_spectrum_block(
    key: str,
    block: np.ndarray,
    *,
    raw_start: int,
) -> None:
    if key in {"spectrum_flux", "spectrum_ivar", "spectrum_wavelength"}:
        if not np.isfinite(block).all():
            raise RawSpectrumValidationError(
                f"{_SPECTRUM_DATASETS[key].source_path} contains non-finite values "
                f"in raw rows beginning at {raw_start}"
            )
    if key == "spectrum_ivar" and np.any(block < 0):
        raise RawSpectrumValidationError(
            f"desi/spectrum_ivar contains negative values in raw rows beginning at {raw_start}"
        )
    if key == "spectrum_wavelength" and np.any(np.diff(block, axis=-1) <= 0):
        raise RawSpectrumValidationError(
            "desi/spectrum_lambda must be strictly increasing for every selected spectrum"
        )
    if key == "spectrum_mask" and np.any(np.all(block, axis=-1)):
        raise RawSpectrumValidationError(
            "desi/spectrum_mask marks every wavelength as invalid for a selected spectrum"
        )


def _extract_spectrum_datasets(
    handle: Any,
    runs: list[Any],
    candidate_count: int,
    staging: Path,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for key, spec in _SPECTRUM_DATASETS.items():
        output_path = staging / spec.filename
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=spec.dtype,
            shape=(candidate_count, ASTROPRISM_SPECTRUM_LENGTH),
        )
        try:
            source_dataset = handle[spec.source_path]
            for run in runs:
                block = np.asarray(source_dataset[run.start : run.stop])
                expected_shape = (len(run.candidate_indices), ASTROPRISM_SPECTRUM_LENGTH)
                if block.shape != expected_shape:
                    raise RawSpectrumValidationError(
                        f"short or malformed read from {spec.source_path}: "
                        f"got {block.shape}, expected {expected_shape}"
                    )
                _validate_spectrum_block(key, block, raw_start=run.start)
                destination = np.asarray(run.candidate_indices, dtype=np.intp)
                output[destination] = block
            output.flush()
        finally:
            del output

        outputs[key] = {
            "path": spec.filename,
            "source_dataset": spec.source_path,
            "dtype": str(spec.dtype),
            "shape": [candidate_count, ASTROPRISM_SPECTRUM_LENGTH],
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "mmap_safe": True,
        }
    return outputs


def extract_raw_spectra(config: RawSpectrumConfig) -> dict[str, Any]:
    """Extract the roster's raw spectrum rows into an atomic artifact."""

    if config.http_block_size < 1:
        raise ValueError("http_block_size must be positive")
    if config.http_timeout_seconds <= 0:
        raise ValueError("http_timeout_seconds must be positive")
    if config.http_probe_attempts < 1:
        raise ValueError("http_probe_attempts must be at least 1")
    if (
        not math.isfinite(config.http_probe_backoff_seconds)
        or config.http_probe_backoff_seconds < 0
    ):
        raise ValueError("http_probe_backoff_seconds must be nonnegative")
    for name in (
        "roster_coordinate_tolerance_arcsec",
        "desi_coordinate_tolerance_arcsec",
        "legacy_coordinate_tolerance_arcsec",
        "caption_legacy_coordinate_tolerance_arcsec",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative")

    output_dir = Path(config.output_dir)
    if _path_exists(output_dir) and not config.overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")

    common = _common_config(config)
    rows, roster_ids, roster_sha256 = _read_roster(common)
    source_info = _inspect_source(common)
    staging: Path | None = None
    try:
        with _open_hdf5(common) as handle:
            source_count = _validate_schema(
                handle,
                require_contiguous=source_info.kind == "http",
            )
            _validate_spectrum_schema(
                handle,
                source_count=source_count,
                require_contiguous=source_info.kind == "http",
            )
            raw_rows, source_ids = _build_raw_row_map(handle, roster_ids)
            runs = _contiguous_runs(raw_rows)
            records = _crosscheck_metadata(
                common,
                handle,
                rows,
                roster_ids,
                raw_rows,
                source_ids,
            )

            output_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
            )
            outputs = _extract_spectrum_datasets(handle, runs, len(rows), staging)

        _verify_source_unchanged(common, source_info)
        manifest = {
            "schema_version": 1,
            "artifact_type": "astroprism_raw_spectra",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roster": {
                "path": str(Path(config.roster_path)),
                "sha256": roster_sha256,
                "row_count": len(rows),
            },
            "source": {
                "kind": source_info.kind,
                "location": source_info.location,
                "url": source_info.location if source_info.kind == "http" else None,
                "etag": source_info.etag,
                "length_bytes": source_info.length_bytes,
                "http_range_verified": source_info.range_verified,
                "http_status": source_info.status_code,
                "content_range": source_info.content_range,
                "hdf5_row_count": source_count,
            },
            "selection": {
                "candidate_count": len(rows),
                "roster_order_preserved": True,
                "raw_rows_in_roster_order": raw_rows,
                "raw_row_min": min(raw_rows),
                "raw_row_max": max(raw_rows),
                "contiguous_run_count": len(runs),
                "http_cache_type": "none" if source_info.kind == "http" else None,
            },
            "identity_tolerances_arcsec": {
                "roster_to_provabgs": config.roster_coordinate_tolerance_arcsec,
                "provabgs_to_desi": config.desi_coordinate_tolerance_arcsec,
                "provabgs_to_legacysurvey": config.legacy_coordinate_tolerance_arcsec,
                "caption_to_legacysurvey": (
                    config.caption_legacy_coordinate_tolerance_arcsec
                ),
            },
            "spectrum": {
                "length": ASTROPRISM_SPECTRUM_LENGTH,
                "observed_frame": True,
                "mask_true_means_excluded": True,
                "aion_modality": "DESISpectrum",
                "aion_token_count": ASTROPRISM_SPECTRUM_TOKENS,
            },
            "outputs": outputs,
            "candidates": records,
            "images_included": False,
        }
        assert staging is not None
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _commit_staging(staging, output_dir, overwrite=config.overwrite)
        staging = None
        return {
            "status": "prepared",
            "candidate_count": len(rows),
            "source_length_bytes": source_info.length_bytes,
            "source_etag": source_info.etag,
            "contiguous_run_count": len(runs),
            "output_dir": str(output_dir),
            "manifest": str(output_dir / "manifest.json"),
            "spectrum_length": ASTROPRISM_SPECTRUM_LENGTH,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "ASTROPRISM_SPECTRUM_LENGTH",
    "RawSpectrumConfig",
    "RawSpectrumValidationError",
    "extract_raw_spectra",
]
