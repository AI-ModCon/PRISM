from __future__ import annotations

import csv
import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import src.data.astroprism_raw_images as raw_images
from src.data.astroprism_raw_images import (
    RawImageConfig,
    RawImageValidationError,
    extract_raw_images,
    sha256_file,
)

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
_SOURCE_BANDS = np.asarray(
    [[b"des-g", b"des-r", b"des-i", b"des-z"]] * 4,
    dtype="S5",
)


def _write_source(path: Path) -> None:
    row_values = np.arange(1, 5, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        provabgs = handle.create_group("provabgs_desi_edr_sv3")
        provabgs.create_dataset("object_id", data=_SOURCE_IDS)
        provabgs.create_dataset("ra", data=_RAS)
        provabgs.create_dataset("dec", data=_DECS)

        desi = handle.create_group("desi")
        desi.create_dataset("object_id", data=_SOURCE_IDS)
        desi.create_dataset("ra", data=_RAS)
        desi.create_dataset("dec", data=_DECS)
        # Deliberately present: the vision-only extractor must never copy spectra.
        desi.create_dataset("spectrum_flux", data=np.arange(28, dtype=np.float32).reshape(4, 7))

        legacy = handle.create_group("legacysurvey")
        legacy.create_dataset("object_id", data=np.asarray(_LEGACY_IDS, dtype="S16"))
        legacy.create_dataset("ra", data=_RAS)
        legacy.create_dataset("dec", data=_DECS)
        legacy.create_dataset(
            "image_array",
            data=np.broadcast_to(row_values[:, None, None, None], (4, 4, 160, 160)).copy(),
        )
        legacy.create_dataset("image_band", data=_SOURCE_BANDS)
        legacy.create_dataset(
            "image_ivar",
            data=np.broadcast_to((row_values + 10)[:, None, None, None], (4, 4, 160, 160)).copy(),
        )
        legacy.create_dataset("image_mask", data=np.zeros((4, 160, 160), dtype=np.bool_))
        legacy.create_dataset(
            "image_scale",
            data=np.broadcast_to(row_values[:, None], (4, 4)).copy(),
        )
        legacy.create_dataset("object_mask", data=np.zeros((4, 160, 160), dtype=np.uint8))


def _write_roster(path: Path, *, fields: list[str] | None = None) -> None:
    rows = []
    for source_row in _ROSTER_SOURCE_ROWS:
        rows.append(
            {
                "paired_targetid": str(_SOURCE_IDS[source_row]),
                "caption_object_id": _LEGACY_IDS[source_row],
                "paired_ra": str(_RAS[source_row]),
                "paired_dec": str(_DECS[source_row]),
                "caption_ra": str(_RAS[source_row]),
                "caption_dec": str(_DECS[source_row]),
            }
        )
    fieldnames = fields or _ROSTER_FIELDS
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_case(tmp_path: Path) -> RawImageConfig:
    source = tmp_path / "source.h5"
    roster = tmp_path / "roster.csv.gz"
    _write_source(source)
    _write_roster(roster)
    return RawImageConfig(
        roster_path=roster,
        output_dir=tmp_path / "output",
        source=source,
        expected_count=3,
        expected_roster_sha256=sha256_file(roster),
        expected_source_length=source.stat().st_size,
        expected_source_etag=None,
        caption_legacy_coordinate_tolerance_arcsec=0.025,
    )


def _manifest(config: RawImageConfig) -> dict[str, Any]:
    return json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))


class _ProbeResponse:
    def __init__(self, *, status: int = 206) -> None:
        self.status = status
        self.headers = {
            "Content-Range": "bytes 0-0/1234",
            "Content-Length": "1",
            "ETag": '"test-etag"',
        }

    def __enter__(self) -> _ProbeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int) -> bytes:
        assert size == 2
        return b"x"

    def geturl(self) -> str:
        return "https://example.test/final.h5"


def test_extracts_all_mmap_outputs_in_roster_order_with_manifest_contract(
    tmp_path: Path,
) -> None:
    config = _build_case(tmp_path)

    summary = extract_raw_images(config)

    assert summary["status"] == "prepared"
    assert summary["candidate_count"] == 3
    assert summary["contiguous_run_count"] == 2
    assert summary["spectra_included"] is False

    manifest = _manifest(config)
    assert manifest["roster"]["sha256"] == sha256_file(config.roster_path)
    assert manifest["selection"] == {
        "acceptance_status": "pending_review",
        "candidate_count": 3,
        "contiguous_run_count": 2,
        "http_cache_type": None,
        "raw_row_max": 3,
        "raw_row_min": 0,
        "raw_rows_in_roster_order": [3, 0, 1],
        "roster_order_preserved": True,
        "split": None,
    }
    assert manifest["identity_tolerances_arcsec"]["caption_to_legacysurvey"] == 0.025
    assert manifest["spectra_included"] is False
    assert [candidate["paired_targetid"] for candidate in manifest["candidates"]] == [
        "104",
        "101",
        "102",
    ]
    assert [candidate["caption_object_id"] for candidate in manifest["candidates"]] == [
        "cap-d",
        "cap-a",
        "cap-b",
    ]
    assert all(
        candidate["acceptance_status"] == "pending_review"
        and candidate["split"] is None
        and candidate["coordinate_crosscheck_arcsec"]["caption_to_legacysurvey"] == 0.0
        for candidate in manifest["candidates"]
    )

    expected_outputs = {
        "image_array",
        "image_band",
        "aion_image_band",
        "image_ivar",
        "image_mask",
        "image_scale",
        "object_mask",
    }
    assert set(manifest["outputs"]) == expected_outputs
    assert not any("spectrum" in item["path"] for item in manifest["outputs"].values())
    assert len({item["path"] for item in manifest["outputs"].values()}) == len(expected_outputs)
    for item in manifest["outputs"].values():
        output_path = config.output_dir / item["path"]
        array = np.load(output_path, mmap_mode="r")
        assert isinstance(array, np.memmap)
        assert item["mmap_safe"] is True
        assert item["sha256"] == sha256_file(output_path)
        assert item["size_bytes"] == output_path.stat().st_size

    image_array = np.load(config.output_dir / "image_array.npy", mmap_mode="r")
    assert image_array[:, 0, 0, 0].tolist() == [4.0, 1.0, 2.0]
    source_bands = np.load(config.output_dir / "image_band.npy", mmap_mode="r")
    aion_bands = np.load(config.output_dir / "aion_image_band.npy", mmap_mode="r")
    assert source_bands.tolist() == _SOURCE_BANDS[[3, 0, 1]].tolist()
    assert aion_bands.tolist() == [
        [b"DES-G", b"DES-R", b"DES-I", b"DES-Z"],
        [b"DES-G", b"DES-R", b"DES-I", b"DES-Z"],
        [b"DES-G", b"DES-R", b"DES-I", b"DES-Z"],
    ]


@pytest.mark.parametrize(
    "missing_field",
    ["caption_object_id", "caption_ra", "caption_dec"],
)
def test_requires_caption_identity_fields(tmp_path: Path, missing_field: str) -> None:
    config = _build_case(tmp_path)
    _write_roster(config.roster_path, fields=[f for f in _ROSTER_FIELDS if f != missing_field])
    config = replace(config, expected_roster_sha256=sha256_file(config.roster_path))

    with pytest.raises(RawImageValidationError, match="missing required columns"):
        extract_raw_images(config)

    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("identity", "does not match caption_object_id"),
        ("coordinate", "caption/Legacy Survey coordinate separation"),
    ],
)
def test_rejects_caption_legacy_mismatch_without_partial_output(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = _build_case(tmp_path)
    if mutation == "identity":
        with h5py.File(config.source, "r+") as handle:
            handle["legacysurvey/object_id"][3] = b"wrong-id"
        config = replace(config, expected_source_length=Path(config.source).stat().st_size)
    else:
        with gzip.open(config.roster_path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["caption_ra"] = str(_RAS[3] + 0.01)
        with gzip.open(config.roster_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_ROSTER_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        config = replace(config, expected_roster_sha256=sha256_file(config.roster_path))

    with pytest.raises(RawImageValidationError, match=message):
        extract_raw_images(config)

    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_rejects_swapped_band_order_and_cleans_staging(tmp_path: Path) -> None:
    config = _build_case(tmp_path)
    with h5py.File(config.source, "r+") as handle:
        handle["legacysurvey/image_band"][3] = np.asarray(
            [b"des-r", b"des-g", b"des-i", b"des-z"], dtype="S5"
        )
    config = replace(config, expected_source_length=Path(config.source).stat().st_size)

    with pytest.raises(RawImageValidationError, match="strict des-g/des-r/des-i/des-z order"):
        extract_raw_images(config)

    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_http_probe_requires_one_byte_pinned_range(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, float]] = []

    def fake_urlopen(request: Any, timeout: float) -> _ProbeResponse:
        calls.append((request, timeout))
        return _ProbeResponse()

    monkeypatch.setattr(raw_images.urllib.request, "urlopen", fake_urlopen)

    info = raw_images._probe_http_source(
        "https://example.test/source.h5",
        expected_length=1234,
        expected_etag='"test-etag"',
        timeout_seconds=7.5,
    )

    request, timeout = calls.pop()
    headers = dict(request.header_items())
    assert request.get_method() == "GET"
    assert headers["Range"] == "bytes=0-0"
    assert headers["If-match"] == '"test-etag"'
    assert timeout == 7.5
    assert info.kind == "http"
    assert info.location == "https://example.test/final.h5"
    assert info.length_bytes == 1234
    assert info.range_verified is True
    assert info.status_code == 206
    assert info.content_range == "bytes 0-0/1234"


def test_http_probe_retries_transport_failures_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[OSError | _ProbeResponse] = [
        raw_images.urllib.error.URLError("temporary DNS failure"),
        TimeoutError("TLS handshake timed out"),
        _ProbeResponse(),
    ]
    sleeps: list[float] = []
    calls = 0

    def fake_urlopen(request: Any, timeout: float) -> _ProbeResponse:
        nonlocal calls
        calls += 1
        outcome = outcomes.pop(0)
        if isinstance(outcome, OSError):
            raise outcome
        return outcome

    monkeypatch.setattr(raw_images.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(raw_images.time, "sleep", sleeps.append)

    info = raw_images._probe_http_source(
        "https://example.test/source.h5",
        expected_length=1234,
        expected_etag='"test-etag"',
        timeout_seconds=7.5,
        attempts=3,
        backoff_seconds=0.25,
    )

    assert info.range_verified is True
    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_http_probe_exhausts_bounded_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    calls = 0

    def fake_urlopen(request: Any, timeout: float) -> _ProbeResponse:
        nonlocal calls
        calls += 1
        raise TimeoutError("TLS handshake timed out")

    monkeypatch.setattr(raw_images.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(raw_images.time, "sleep", sleeps.append)

    with pytest.raises(RawImageValidationError, match="failed .* after 3 attempts"):
        raw_images._probe_http_source(
            "https://example.test/source.h5",
            expected_length=1234,
            expected_etag='"test-etag"',
            timeout_seconds=7.5,
            attempts=3,
            backoff_seconds=0.125,
        )

    assert calls == 3
    assert sleeps == [0.125, 0.25]


def test_http_probe_does_not_retry_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: Any, timeout: float) -> _ProbeResponse:
        nonlocal calls
        calls += 1
        return _ProbeResponse(status=200)

    monkeypatch.setattr(raw_images.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(raw_images.time, "sleep", sleeps.append)

    with pytest.raises(RawImageValidationError, match="requires status 206"):
        raw_images._probe_http_source(
            "https://example.test/source.h5",
            expected_length=1234,
            expected_etag='"test-etag"',
            timeout_seconds=7.5,
            attempts=3,
            backoff_seconds=0.25,
        )

    assert calls == 1
    assert sleeps == []


def test_http_probe_does_not_retry_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: Any, timeout: float) -> _ProbeResponse:
        nonlocal calls
        calls += 1
        raise raw_images.urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", hdrs=None, fp=None
        )

    monkeypatch.setattr(raw_images.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(raw_images.time, "sleep", sleeps.append)

    with pytest.raises(RawImageValidationError, match="returned status 503"):
        raw_images._probe_http_source(
            "https://example.test/source.h5",
            expected_length=1234,
            expected_etag='"test-etag"',
            timeout_seconds=7.5,
            attempts=3,
            backoff_seconds=0.25,
        )

    assert calls == 1
    assert sleeps == []


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = _build_case(tmp_path)
    extract_raw_images(config)
    stale = config.output_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        extract_raw_images(config)
    assert stale.is_file()

    summary = extract_raw_images(replace(config, overwrite=True))

    assert summary["status"] == "prepared"
    assert not stale.exists()
    assert (config.output_dir / "manifest.json").is_file()
