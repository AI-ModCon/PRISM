from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import tools.extract_astroprism_raw_spectra as spectrum_cli
from src.data.astroprism_raw_images import sha256_file
from src.data.astroprism_raw_spectra import (
    ASTROPRISM_SPECTRUM_LENGTH,
    RawSpectrumConfig,
    RawSpectrumValidationError,
    extract_raw_spectra,
)

pytestmark = [pytest.mark.unit, pytest.mark.multimodal]

_ROSTER_FIELDS = [
    "paired_targetid",
    "caption_object_id",
    "paired_ra",
    "paired_dec",
    "caption_ra",
    "caption_dec",
]
_SOURCE_IDS = np.asarray([101, 102, 103, 104], dtype=np.int64)
_RAS = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
_DECS = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float64)
_LEGACY_IDS = ["cap-a", "cap-b", "cap-c", "cap-d"]
_ROSTER_SOURCE_ROWS = (3, 0, 1)


def _write_source(path: Path) -> None:
    source_count = len(_SOURCE_IDS)
    row_values = np.arange(1, source_count + 1, dtype=np.float32)
    pixels = np.arange(ASTROPRISM_SPECTRUM_LENGTH, dtype=np.float32)
    flux = np.stack([100.0 * index + pixels / 100.0 for index in range(source_count)])
    ivar = np.broadcast_to(row_values[:, None], flux.shape).copy()
    mask = np.zeros(flux.shape, dtype=np.bool_)
    mask[:, ::257] = True
    wavelength = np.broadcast_to(3500.0 + pixels, flux.shape).copy()

    with h5py.File(path, "w") as handle:
        provabgs = handle.create_group("provabgs_desi_edr_sv3")
        provabgs.create_dataset("object_id", data=_SOURCE_IDS)
        provabgs.create_dataset("ra", data=_RAS)
        provabgs.create_dataset("dec", data=_DECS)

        desi = handle.create_group("desi")
        desi.create_dataset("object_id", data=_SOURCE_IDS)
        desi.create_dataset("ra", data=_RAS)
        desi.create_dataset("dec", data=_DECS)
        desi.create_dataset("spectrum_flux", data=flux)
        desi.create_dataset("spectrum_ivar", data=ivar)
        desi.create_dataset("spectrum_mask", data=mask)
        desi.create_dataset("spectrum_lambda", data=wavelength)

        # The shared source validator checks the complete official schema. These
        # image fields must be present, but the spectrum extractor must not emit them.
        legacy = handle.create_group("legacysurvey")
        legacy.create_dataset("object_id", data=np.asarray(_LEGACY_IDS, dtype="S16"))
        legacy.create_dataset("ra", data=_RAS)
        legacy.create_dataset("dec", data=_DECS)
        legacy.create_dataset(
            "image_array",
            data=np.broadcast_to(
                row_values[:, None, None, None], (source_count, 4, 160, 160)
            ).copy(),
        )
        legacy.create_dataset(
            "image_band",
            data=np.asarray([[b"des-g", b"des-r", b"des-i", b"des-z"]] * source_count),
        )
        legacy.create_dataset(
            "image_ivar",
            data=np.broadcast_to(
                (row_values + 10)[:, None, None, None],
                (source_count, 4, 160, 160),
            ).copy(),
        )
        legacy.create_dataset(
            "image_mask", data=np.zeros((source_count, 160, 160), dtype=np.bool_)
        )
        legacy.create_dataset(
            "image_scale",
            data=np.broadcast_to(row_values[:, None], (source_count, 4)).copy(),
        )
        legacy.create_dataset(
            "object_mask", data=np.zeros((source_count, 160, 160), dtype=np.uint8)
        )


def _write_roster(path: Path) -> None:
    rows = [
        {
            "paired_targetid": str(_SOURCE_IDS[source_row]),
            "caption_object_id": _LEGACY_IDS[source_row],
            "paired_ra": str(_RAS[source_row]),
            "paired_dec": str(_DECS[source_row]),
            "caption_ra": str(_RAS[source_row]),
            "caption_dec": str(_DECS[source_row]),
        }
        for source_row in _ROSTER_SOURCE_ROWS
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROSTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _build_case(tmp_path: Path) -> RawSpectrumConfig:
    source = tmp_path / "source.h5"
    roster = tmp_path / "roster.csv.gz"
    _write_source(source)
    _write_roster(roster)
    return RawSpectrumConfig(
        roster_path=roster,
        output_dir=tmp_path / "output",
        source=source,
        expected_count=len(_ROSTER_SOURCE_ROWS),
        expected_roster_sha256=sha256_file(roster),
        expected_source_length=source.stat().st_size,
        expected_source_etag=None,
        caption_legacy_coordinate_tolerance_arcsec=0.025,
    )


def test_extracts_mmap_spectra_in_roster_order_with_manifest_contract(
    tmp_path: Path,
) -> None:
    config = _build_case(tmp_path)

    summary = extract_raw_spectra(config)

    assert summary["status"] == "prepared"
    assert summary["candidate_count"] == 3
    assert summary["contiguous_run_count"] == 2
    assert summary["spectrum_length"] == ASTROPRISM_SPECTRUM_LENGTH

    manifest = json.loads(
        (config.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_type"] == "astroprism_raw_spectra"
    assert manifest["roster"]["sha256"] == sha256_file(config.roster_path)
    assert manifest["selection"] == {
        "candidate_count": 3,
        "contiguous_run_count": 2,
        "http_cache_type": None,
        "raw_row_max": 3,
        "raw_row_min": 0,
        "raw_rows_in_roster_order": [3, 0, 1],
        "roster_order_preserved": True,
    }
    assert manifest["spectrum"] == {
        "aion_modality": "DESISpectrum",
        "aion_token_count": 273,
        "length": ASTROPRISM_SPECTRUM_LENGTH,
        "mask_true_means_excluded": True,
        "observed_frame": True,
    }
    assert manifest["images_included"] is False
    assert [record["paired_targetid"] for record in manifest["candidates"]] == [
        "104",
        "101",
        "102",
    ]
    assert [record["raw_row"] for record in manifest["candidates"]] == [3, 0, 1]

    expected = {
        "spectrum_flux": ("spectrum_flux.npy", np.dtype("float32")),
        "spectrum_ivar": ("spectrum_ivar.npy", np.dtype("float32")),
        "spectrum_mask": ("spectrum_mask.npy", np.dtype("bool")),
        "spectrum_wavelength": ("spectrum_wavelength.npy", np.dtype("float32")),
    }
    assert set(manifest["outputs"]) == set(expected)
    assert not any("image" in value["path"] for value in manifest["outputs"].values())
    for key, (filename, dtype) in expected.items():
        metadata = manifest["outputs"][key]
        output_path = config.output_dir / filename
        array = np.load(output_path, mmap_mode="r", allow_pickle=False)
        assert isinstance(array, np.memmap)
        assert array.shape == (3, ASTROPRISM_SPECTRUM_LENGTH)
        assert array.dtype == dtype
        assert metadata["shape"] == [3, ASTROPRISM_SPECTRUM_LENGTH]
        assert np.dtype(metadata["dtype"]) == dtype
        assert metadata["mmap_safe"] is True
        assert metadata["size_bytes"] == output_path.stat().st_size
        assert metadata["sha256"] == sha256_file(output_path)

    flux = np.load(config.output_dir / "spectrum_flux.npy", mmap_mode="r")
    ivar = np.load(config.output_dir / "spectrum_ivar.npy", mmap_mode="r")
    assert flux[:, 0].tolist() == [300.0, 0.0, 100.0]
    assert ivar[:, 0].tolist() == [4.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda handle: handle["desi/spectrum_flux"].__setitem__((3, 12), np.nan),
            "spectrum_flux contains non-finite values",
        ),
        (
            lambda handle: handle["desi/spectrum_ivar"].__setitem__((3, 12), -1.0),
            "spectrum_ivar contains negative values",
        ),
        (
            lambda handle: handle["desi/spectrum_lambda"].__setitem__(
                (3, 12), handle["desi/spectrum_lambda"][3, 11]
            ),
            "spectrum_lambda must be strictly increasing",
        ),
        (
            lambda handle: handle["desi/spectrum_mask"].__setitem__(3, True),
            "spectrum_mask marks every wavelength as invalid",
        ),
    ],
)
def test_rejects_invalid_selected_spectrum_and_cleans_staging(
    tmp_path: Path,
    mutate: Callable[[h5py.File], None],
    message: str,
) -> None:
    config = _build_case(tmp_path)
    with h5py.File(config.source, "r+") as handle:
        mutate(handle)

    with pytest.raises(RawSpectrumValidationError, match=message):
        extract_raw_spectra(config)

    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_cli_local_source_does_not_apply_official_http_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.h5"
    args = argparse.Namespace(
        roster=tmp_path / "roster.csv.gz",
        source=str(source),
        output=tmp_path / "output",
        expected_source_length=None,
        expected_source_etag=None,
        http_block_size=4096,
        http_timeout_seconds=7.5,
        http_probe_attempts=2,
        http_probe_backoff_seconds=0.25,
        caption_legacy_coordinate_tolerance_arcsec=0.025,
        overwrite=True,
    )
    captured: dict[str, Any] = {}

    def fake_extract(config: RawSpectrumConfig) -> dict[str, Any]:
        captured["config"] = config
        return {"status": "prepared", "candidate_count": 3}

    monkeypatch.setattr(spectrum_cli, "parse_args", lambda: args)
    monkeypatch.setattr(spectrum_cli, "extract_raw_spectra", fake_extract)

    assert spectrum_cli.main() == 0

    config = captured["config"]
    assert isinstance(config, RawSpectrumConfig)
    assert config.source == str(source)
    assert config.expected_source_length is None
    assert config.expected_source_etag is None
    assert config.http_block_size == 4096
    assert config.overwrite is True
    assert json.loads(capsys.readouterr().out) == {
        "candidate_count": 3,
        "status": "prepared",
    }
