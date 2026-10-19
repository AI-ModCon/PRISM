#!/usr/bin/env python3
"""Build auditable compact requests or a spectrum-caption review bundle.

The workflow follows AstroSpecLM Figure 1 and Appendix A.5: raw DESI spectra
are first distilled into a structured fact report, and prose is generated only
from caption-admissible facts.  This pilot never modifies the Stage 1A image
captions or the current Stage 1B proxy-caption configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.astroprism_spectrum_facts import (  # noqa: E402
    measure_spectrum_facts,
)

PAPER_URL = "https://arxiv.org/pdf/2609.07102"
PAPER_VERSION = "2609.07102v1"
FACT_EXTRACTOR_VERSION = "astroprism-spectrum-facts-v1"
COMPACT_PAYLOAD_SCHEMA_VERSION = 1
COMPACT_POLICY_VERSION = "astroprism-spectrum-caption-conservative-v2"
TENTATIVE_MIN_MEDIAN_PIXEL_SNR = 2.0
DN4000_CLASS_BOUNDARIES = (1.3, 1.6)
DN4000_CLASS_BOUNDARY_SIGMA = 1.0
DEFAULT_CANDIDATE_INDICES = (0, 1, 2, 3, 4)

COMPACT_FLAG_NAMES = {
    "desi_arm_overlap_region": "arm_overlap",
    "telluric_a_band_neighborhood": "telluric",
    "telluric_b_band_neighborhood": "telluric",
    "red_sky_residual_sensitive": "red_sky",
}

# Adapted from the constraints in AstroSpecLM Appendix A.5.  We preserve the
# source citation instead of copying the paper's full prompt into the repo.
ALIGNMENT_CAPTION_PROMPT = """\
Act as an experienced astronomical spectroscopist. Write one concise, natural,
objective caption of the supplied galaxy spectrum using only the compact JSON
facts. Prioritize continuum morphology, a useful Dn4000 value, and two to five
principal features. Include approximate observed positions or rest equivalent
widths only when useful.

The z_cat value and galaxy type are external context, not classifications or a
redshift measured here. A tentative feature is 3-5 sigma and must be hedged; a
detected feature is at least 5 sigma. Any fact with cautious=true must be
hedged or omitted. For a line, cautious means it is tentative or lies in an
arm-overlap, telluric, or red-sky-sensitive region; for continuum or Dn4000 it
means the reported trend or Dn4000 class is uncertain. Ca H can blend with
H-epsilon, and features
named as blends are integrated blends. Do not infer age, dust, star formation, image
morphology, or physical mechanisms. Do not mention JSON, fields, confidence
labels, identifiers, or generation. Return only the caption text.
"""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _round_for_payload(value: Any, digits: int) -> float | None:
    numeric = _finite_float(value)
    if numeric is None:
        return None
    rounded = round(numeric, digits)
    return 0.0 if rounded == 0 else rounded


def _line_is_caption_admissible(line: dict[str, Any]) -> bool:
    return (
        line.get("detection_class") in {"detected", "tentative"}
        and line.get("quality") != "unavailable"
        and "masked_pixels_in_line_window" not in line.get("quality_flags", [])
    )


def _compact_line_exclusion_reason(
    line: dict[str, Any], median_pixel_snr: float | None
) -> str | None:
    if line.get("quality") == "unavailable" or line.get("detection_class") == "unavailable":
        return "unavailable"
    if "masked_pixels_in_line_window" in line.get("quality_flags", []):
        return "masked_line_window"
    if line.get("detection_class") == "not_detected":
        return "not_detected"
    if line.get("detection_class") not in {"detected", "tentative"}:
        return "not_caption_admissible"
    if line["detection_class"] == "tentative" and (
        median_pixel_snr is None
        or not math.isfinite(median_pixel_snr)
        or median_pixel_snr < TENTATIVE_MIN_MEDIAN_PIXEL_SNR
    ):
        return "tentative_suppressed_low_snr"
    return None


def _compact_flags(line: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for flag in line.get("quality_flags", []):
        compact = COMPACT_FLAG_NAMES.get(flag, flag)
        if compact not in result:
            result.append(compact)
    return result


def _dn4000_class_crosses_boundary(value: float, error: float) -> bool:
    if error <= 0 or not math.isfinite(error):
        return True
    lower = value - DN4000_CLASS_BOUNDARY_SIGMA * error
    upper = value + DN4000_CLASS_BOUNDARY_SIGMA * error
    return any(lower < boundary <= upper for boundary in DN4000_CLASS_BOUNDARIES)


def _compact_caption_payload(
    measurement: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, Any]:
    """Return only information needed by the caption language model."""

    median_pixel_snr = _finite_float(measurement["quality"]["median_pixel_snr_rest_3850_7000"])
    payload: dict[str, Any] = {
        "schema_version": COMPACT_PAYLOAD_SCHEMA_VERSION,
        "type": "galaxy",
        "z_cat": _round_for_payload(catalog["redshift"], 6),
    }
    if median_pixel_snr is not None:
        payload["snr"] = _round_for_payload(median_pixel_snr, 2)

    continuum = measurement["continuum"]
    if continuum["status"] == "measured":
        continuum_ratio = float(continuum["red_to_blue_ratio"])
        continuum_error = float(continuum["red_to_blue_ratio_uncertainty"])
        compact_continuum = {
            "class": continuum["label"],
            "ratio": _round_for_payload(continuum_ratio, 3),
            "err": _round_for_payload(continuum_error, 3),
        }
        if continuum_error <= 0 or abs(continuum_ratio - 1.0) / continuum_error < 3.0:
            compact_continuum["cautious"] = True
        payload["cont"] = compact_continuum

    dn4000 = measurement["dn4000"]
    if dn4000["status"] == "measured":
        dn4000_value = float(dn4000["value"])
        dn4000_error = float(dn4000["uncertainty"])
        compact_dn4000 = {
            "class": dn4000["strength_label"],
            "value": _round_for_payload(dn4000_value, 3),
            "err": _round_for_payload(dn4000_error, 3),
        }
        if (
            dn4000_error <= 0
            or (dn4000_value - 1.0) / dn4000_error < 2.0
            or _dn4000_class_crosses_boundary(dn4000_value, dn4000_error)
        ):
            compact_dn4000["cautious"] = True
        payload["dn4000"] = compact_dn4000

    compact_lines: list[dict[str, Any]] = []
    for line in measurement["lines"]:
        if _compact_line_exclusion_reason(line, median_pixel_snr) is not None:
            continue
        flags = _compact_flags(line)
        compact_line = {
            "id": f"spec.line.{line['key']}",
            "name": line["label"],
            "nature": line["nature"],
            "conf": line["detection_class"],
            "sig": _round_for_payload(line.get("significance_sigma"), 2),
            "obs_A": _round_for_payload(line.get("observed_centroid_angstrom"), 1),
            "ew_rest_A": _round_for_payload(line.get("equivalent_width_signed_rest_angstrom"), 2),
            "ew_err_A": _round_for_payload(line.get("equivalent_width_uncertainty_angstrom"), 2),
        }
        if flags:
            compact_line["flags"] = flags
        if line["detection_class"] == "tentative" or flags:
            compact_line["cautious"] = True
        compact_lines.append(compact_line)
    payload["lines"] = compact_lines
    return payload


def _compact_policy_metadata(measurement: dict[str, Any]) -> dict[str, Any]:
    median_pixel_snr = _finite_float(measurement["quality"]["median_pixel_snr_rest_3850_7000"])
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    for line in measurement["lines"]:
        fact_id = f"spec.line.{line['key']}"
        reason = _compact_line_exclusion_reason(line, median_pixel_snr)
        if reason is None:
            included.append(fact_id)
        else:
            excluded.append({"fact_id": fact_id, "reason": reason})
    return {
        "version": COMPACT_POLICY_VERSION,
        "tentative_min_median_pixel_snr": TENTATIVE_MIN_MEDIAN_PIXEL_SNR,
        "dn4000_class_boundaries": list(DN4000_CLASS_BOUNDARIES),
        "dn4000_class_boundary_sigma": DN4000_CLASS_BOUNDARY_SIGMA,
        "included_line_fact_ids": included,
        "excluded_lines": excluded,
    }


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256_text(canonical)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _select_candidate_records(
    accepted_rows: list[dict[str, Any]],
    *,
    requested_indices: list[int] | None,
    train_limit: int | None,
) -> list[dict[str, Any]]:
    if train_limit is not None and requested_indices is not None:
        raise ValueError("--indices and --train-limit are mutually exclusive")

    accepted_by_index = {int(row["candidate_index"]): row for row in accepted_rows}
    if len(accepted_by_index) != len(accepted_rows):
        raise ValueError("accepted candidates contain duplicate candidate_index values")

    if train_limit is not None:
        if train_limit <= 0:
            raise ValueError("--train-limit must be positive")
        records = [row for row in accepted_rows if row["split"] == "train"][:train_limit]
        if len(records) != train_limit:
            raise ValueError(
                f"requested {train_limit} train candidates, but only {len(records)} are available"
            )
        return records

    indices = list(
        requested_indices if requested_indices is not None else DEFAULT_CANDIDATE_INDICES
    )
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("indices must be unique and nonempty")
    if any(index < 0 for index in indices):
        raise ValueError("indices must be nonnegative")
    missing = sorted(set(indices) - set(accepted_by_index))
    if missing:
        raise ValueError(f"requested candidate indices are missing: {missing}")
    return [accepted_by_index[index] for index in indices]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def _load_catalog_rows(
    records: list[dict[str, Any]], paired_root: Path
) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - runtime environment check
        raise RuntimeError("pyarrow is required to read the paired metadata") from exc

    columns = [
        "object_id_provabgs",
        "Z_HP_provabgs",
        "ZERR_provabgs",
        "TSNR2_BGS_provabgs",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["paired_source"]["shard"]].append(record)

    result: dict[int, dict[str, Any]] = {}
    for shard, selected in grouped.items():
        path = paired_root / shard
        if not path.is_file():
            raise FileNotFoundError(f"paired metadata shard is missing: {path}")
        expected_sha256s = {record["paired_source"]["sha256"] for record in selected}
        if len(expected_sha256s) != 1:
            raise ValueError(f"conflicting accepted checksums for paired shard: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 not in expected_sha256s:
            raise ValueError(f"paired metadata checksum mismatch: {path}")

        table = pq.read_table(path, columns=columns)
        for record in selected:
            row_number = int(record["paired_source"]["row"])
            row = table.slice(row_number, 1).to_pylist()[0]
            expected_id = record["paired_targetid"]
            if str(row["object_id_provabgs"]) != expected_id:
                raise ValueError(
                    f"candidate {record['candidate_index']}: paired metadata ID "
                    f"{row['object_id_provabgs']!r} != {expected_id!r}"
                )
            result[int(record["candidate_index"])] = {
                "redshift": float(row["Z_HP_provabgs"]),
                "redshift_error": float(row["ZERR_provabgs"]),
                "tsnr2_bgs": float(row["TSNR2_BGS_provabgs"]),
                "source": {
                    "path": str(path.resolve()),
                    "shard": shard,
                    "row": row_number,
                    "sha256": actual_sha256,
                },
            }
    return result


def _fact(
    fact_id: str,
    name: str,
    value: Any,
    *,
    provenance: str,
    source_reference: str,
    evidence_kind: str,
    required_student_inputs: list[str],
    quality: str,
    status: str,
    evidence_locator: dict[str, Any],
    allowed_uses: list[str],
    uncertainty: Any = None,
    unit: str | None = None,
    restrictions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "name": name,
        "value": value,
        "unit": unit,
        "uncertainty": uncertainty,
        "provenance": provenance,
        "source_reference": source_reference,
        "evidence_kind": evidence_kind,
        "required_student_inputs": required_student_inputs,
        "quality": quality,
        "status": status,
        "evidence_locator": evidence_locator,
        "allowed_uses": allowed_uses,
        "restrictions": restrictions or [],
    }


def _structured_facts(
    measurement: dict[str, Any],
    catalog: dict[str, Any],
    *,
    candidate_index: int,
) -> list[dict[str, Any]]:
    source_reference = f"raw_spectra:candidate_index={candidate_index}"
    facts = [
        _fact(
            "catalog.object_type",
            "object_type",
            "galaxy",
            provenance="catalog_label",
            source_reference="EiffL/desi_x_ls_aion dataset sample definition",
            evidence_kind="external_label",
            required_student_inputs=["spectrum"],
            quality="low_confidence",
            status="supplied",
            evidence_locator={},
            allowed_uses=["caption", "alignment_supervision"],
            restrictions=[
                "Sample-level galaxy label; no row-level DESI class field was available.",
                "Must not be described as a direct spectral classification.",
            ],
        ),
        _fact(
            "catalog.redshift",
            "redshift",
            catalog["redshift"],
            provenance="catalog_label",
            source_reference=(
                f"{catalog['source']['shard']}:{catalog['source']['row']}:Z_HP_provabgs"
            ),
            evidence_kind="external_label",
            required_student_inputs=["spectrum"],
            quality="valid",
            status="supplied",
            evidence_locator={},
            allowed_uses=["caption", "alignment_supervision"],
            uncertainty=catalog["redshift_error"],
            restrictions=[
                "External PROVABGS redshift, not independently recovered here.",
                "All rest-frame measurement windows are conditioned on this value.",
            ],
        ),
        _fact(
            "spec.coverage",
            "wavelength_coverage",
            measurement["coverage"],
            provenance="spectral_measurement",
            source_reference=source_reference,
            evidence_kind="direct",
            required_student_inputs=["spectrum"],
            quality="valid",
            status="measured",
            evidence_locator={
                "observed_wavelength_angstrom": [
                    measurement["coverage"]["observed_min_angstrom"],
                    measurement["coverage"]["observed_max_angstrom"],
                ]
            },
            allowed_uses=["grounded_explanation"],
        ),
        _fact(
            "spec.quality",
            "median_pixel_snr",
            measurement["quality"]["median_pixel_snr_rest_3850_7000"],
            provenance="spectral_measurement",
            source_reference=source_reference,
            evidence_kind="derived",
            required_student_inputs=["spectrum"],
            quality="valid",
            status="measured",
            evidence_locator={"rest_wavelength_angstrom": [3850.0, 7000.0]},
            allowed_uses=["caption", "grounded_explanation"],
            restrictions=[measurement["quality"]["definition"]],
        ),
        _fact(
            "spec.continuum",
            "continuum_morphology",
            measurement["continuum"],
            provenance="spectral_measurement",
            source_reference=source_reference,
            evidence_kind="derived",
            required_student_inputs=["spectrum"],
            quality=(
                "valid" if measurement["continuum"]["status"] == "measured" else "unavailable"
            ),
            status=measurement["continuum"]["status"],
            evidence_locator={
                "blue_rest_wavelength_angstrom": measurement["continuum"][
                    "blue_window_rest_angstrom"
                ],
                "red_rest_wavelength_angstrom": measurement["continuum"][
                    "red_window_rest_angstrom"
                ],
            },
            allowed_uses=(
                ["caption", "grounded_explanation"]
                if measurement["continuum"]["status"] == "measured"
                else []
            ),
            uncertainty=measurement["continuum"]["red_to_blue_ratio_uncertainty"],
            restrictions=[
                "Observed f_lambda morphology only; do not infer age, dust, or star formation."
            ],
        ),
        _fact(
            "spec.dn4000",
            "dn4000",
            measurement["dn4000"],
            provenance="spectral_measurement",
            source_reference=source_reference,
            evidence_kind="derived",
            required_student_inputs=["spectrum"],
            quality=("valid" if measurement["dn4000"]["status"] == "measured" else "unavailable"),
            status=measurement["dn4000"]["status"],
            evidence_locator={
                "blue_rest_wavelength_angstrom": [3850.0, 3950.0],
                "red_rest_wavelength_angstrom": [4000.0, 4100.0],
            },
            allowed_uses=(
                ["caption", "grounded_explanation"]
                if measurement["dn4000"]["status"] == "measured"
                else []
            ),
            uncertainty=measurement["dn4000"]["uncertainty"],
            restrictions=[
                "Do not infer stellar age from Dn4000 alone.",
                "Rest-frame bands are conditioned on the external catalog redshift.",
            ],
        ),
    ]

    for line in measurement["lines"]:
        observed_interval = [
            line["line_window_rest_angstrom"][0] * (1.0 + catalog["redshift"]),
            line["line_window_rest_angstrom"][1] * (1.0 + catalog["redshift"]),
        ]
        caption_allowed = _line_is_caption_admissible(line)
        restrictions = [
            "Line lookup and rest-frame window are conditioned on the external catalog redshift."
        ]
        if line.get("caveat"):
            restrictions.append(line["caveat"])
        restrictions.extend(f"Review flag: {flag}." for flag in line.get("quality_flags", []))
        facts.append(
            _fact(
                f"spec.line.{line['key']}",
                line["label"],
                line,
                provenance="spectral_measurement",
                source_reference=source_reference,
                evidence_kind="derived",
                required_student_inputs=["spectrum"],
                quality=line["quality"],
                status=line["status"],
                evidence_locator={
                    "observed_wavelength_angstrom": observed_interval,
                    "rest_wavelength_angstrom": line["line_window_rest_angstrom"],
                },
                allowed_uses=(
                    ["caption", "grounded_explanation"]
                    if caption_allowed
                    else ["grounded_explanation"]
                ),
                uncertainty=line.get("integrated_strength_uncertainty_source_flux_angstrom"),
                restrictions=restrictions,
            )
        )
    return facts


def _validate_draft(
    draft: dict[str, Any],
    facts: list[dict[str, Any]],
    *,
    candidate_index: int,
) -> None:
    caption = draft.get("student_caption")
    claims = draft.get("claims")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"candidate {candidate_index}: draft caption is empty")
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"candidate {candidate_index}: draft claims are missing")
    facts_by_id = {fact["fact_id"]: fact for fact in facts}
    for claim in claims:
        text = claim.get("text")
        if not isinstance(text, str) or text not in caption:
            raise ValueError(
                f"candidate {candidate_index}: claim text is not present in caption: {text!r}"
            )
        fact_ids = claim.get("fact_ids")
        if not isinstance(fact_ids, list) or not fact_ids:
            raise ValueError(f"candidate {candidate_index}: claim has no fact IDs")
        for fact_id in fact_ids:
            if fact_id not in facts_by_id:
                raise ValueError(f"candidate {candidate_index}: unknown fact ID {fact_id!r}")
            if "caption" not in facts_by_id[fact_id]["allowed_uses"]:
                raise ValueError(
                    f"candidate {candidate_index}: fact {fact_id!r} is not caption-admissible"
                )


def _line_by_key(measurement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {line["key"]: line for line in measurement["lines"]}


def _plot_record(
    path: Path,
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    measurement: dict[str, Any],
    *,
    candidate_index: int,
    target_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = (~mask) & (ivar > 0)
    clean_flux = np.asarray(flux, dtype=np.float64).copy()
    clean_flux[~valid] = np.nan
    interpolated = np.interp(wavelength, wavelength[valid], clean_flux[valid])
    kernel = np.ones(7, dtype=np.float64) / 7.0
    smoothed = np.convolve(interpolated, kernel, mode="same")
    redshift = measurement["catalog_redshift"]
    line_map = _line_by_key(measurement)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    full_axis = axes[0, 0]
    full_axis.plot(wavelength, clean_flux, color="0.72", linewidth=0.45)
    full_axis.plot(wavelength, smoothed, color="black", linewidth=0.9)
    lower, upper = np.nanpercentile(clean_flux, [0.5, 99.5])
    margin = 0.12 * (upper - lower)
    full_axis.set_ylim(lower - margin, upper + margin)
    full_axis.set_xlim(wavelength[0], wavelength[-1])
    full_axis.set_title(f"Candidate {candidate_index} | TARGETID {target_id} | z={redshift:.5f}")
    full_axis.set_xlabel("Observed vacuum wavelength (Angstrom)")
    full_axis.set_ylabel("Flux (source array units)")

    zooms = [
        ("[O II] / Ca H+K", 3650.0, 4050.0, {"oii_3727_3729", "ca_k_3934", "ca_h_3969"}),
        ("H-beta / [O III]", 4770.0, 5050.0, {"hbeta_4863", "oiii_4960", "oiii_5008"}),
        (
            "H-alpha / [N II] / [S II]",
            6480.0,
            6790.0,
            {"nii_6550", "halpha_6565", "nii_6585", "sii_6718", "sii_6733"},
        ),
    ]
    for axis, (title, rest_lo, rest_hi, keys) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]), zooms, strict=False
    ):
        observed_lo = rest_lo * (1.0 + redshift)
        observed_hi = rest_hi * (1.0 + redshift)
        selected = (wavelength >= observed_lo) & (wavelength <= observed_hi)
        axis.plot(wavelength[selected], clean_flux[selected], color="0.65", linewidth=0.6)
        axis.plot(wavelength[selected], smoothed[selected], color="black", linewidth=1.0)
        local_valid = selected & valid
        if np.any(local_valid):
            ylo, yhi = np.nanpercentile(clean_flux[local_valid], [1.0, 99.0])
            ymargin = 0.15 * max(yhi - ylo, 1e-6)
            axis.set_ylim(ylo - ymargin, yhi + ymargin)
        axis.set_xlim(observed_lo, observed_hi)
        axis.set_title(title)
        axis.set_xlabel("Observed vacuum wavelength (Angstrom)")
        axis.set_ylabel("Flux")
        for key in keys:
            line = line_map[key]
            detection = line["detection_class"]
            color = {"detected": "tab:red", "tentative": "tab:orange"}.get(detection, "tab:blue")
            expected = line["expected_observed_wavelength_angstrom"]
            axis.axvline(expected, color=color, alpha=0.65, linewidth=0.9)
            axis.text(
                expected,
                0.98,
                line["label"],
                rotation=90,
                va="top",
                ha="right",
                color=color,
                fontsize=7,
                transform=axis.get_xaxis_transform(),
            )
            centroid = line.get("observed_centroid_angstrom")
            if centroid is not None:
                axis.axvline(centroid, color=color, linestyle=":", alpha=0.8)
        masked = selected & (~valid)
        if np.any(masked):
            axis.scatter(
                wavelength[masked],
                np.full(np.count_nonzero(masked), axis.get_ylim()[0]),
                marker="x",
                color="magenta",
                s=12,
                label="masked/zero ivar",
            )
            axis.legend(loc="lower right", fontsize=7)

    figure.suptitle(
        "AstroPRISM spectrum-caption pilot: gray=raw, black=7-pixel smooth, "
        "red=detected, orange=tentative, blue=not detected",
        fontsize=10,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _review_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# AstroPRISM spectrum-caption review pilot",
        "",
        (
            "This review bundle follows the grounded-fact pipeline in "
            f"[AstroSpecLM Figure 1 and Appendix A.5]({PAPER_URL}). It is a "
            "local pilot, not an accepted training artifact."
        ),
        "",
        (
            "Important split warning: held-out objects are quarantined from training, "
            "and using review feedback to tune generation contaminates them for clean "
            "benchmark claims."
            if any(record["split"] != "train" for record in records)
            else "All selected candidates are from the training split."
        ),
        "",
        "| Candidate | Split | TARGETID | Status |",
        "|---:|---|---|---|",
    ]
    for record in records:
        role = record["training_policy"]["role"]
        lines.append(
            f"| {record['candidate_index']} | {record['split']} | "
            f"`{record['paired_targetid']}` | {role} |"
        )
    lines.extend(["", "All captions remain `pending_review`; none is training-eligible yet.", ""])

    for record in records:
        measurement = record["teacher_reference"]["measurement_summary"]
        continuum_ratio = _format_number(
            measurement["continuum"].get("red_to_blue_ratio"), 3
        )
        continuum_uncertainty = _format_number(
            measurement["continuum"].get("red_to_blue_ratio_uncertainty"), 3
        )
        dn4000_value = _format_number(measurement["dn4000"].get("value"), 3)
        dn4000_uncertainty = _format_number(
            measurement["dn4000"].get("uncertainty"), 3
        )
        lines.extend(
            [
                f"## Candidate {record['candidate_index']} ({record['split']})",
                "",
                f"> {record['spectrum_caption']['original']}",
                "",
                f"![Candidate {record['candidate_index']} spectrum](plots/candidate_{record['candidate_index']:04d}.png)",
                "",
                "Summary facts:",
                "",
                f"- External catalog redshift: `{measurement['catalog_redshift']:.7f}`.",
                f"- Median per-pixel S/N: `{measurement['quality']['median_pixel_snr_rest_3850_7000']:.2f}`.",
                f"- Continuum: `{measurement['continuum']['label']}`; red/blue f_lambda ratio "
                f"`{continuum_ratio} ± {continuum_uncertainty}`.",
                f"- Dn4000: `{dn4000_value} ± {dn4000_uncertainty}` "
                f"(`{measurement['dn4000']['strength_label']}`).",
                "",
                "Detected or tentative line measurements (review flags still apply):",
                "",
                "| Feature | Class | Sigma | Observed centroid (A) | Signed rest EW (A) | Review flags |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for line in measurement["lines"]:
            if line["detection_class"] not in {"detected", "tentative"}:
                continue
            flags = ", ".join(line.get("quality_flags", [])) or "—"
            lines.append(
                f"| {line['label']} | {line['detection_class']} | "
                f"{_format_number(line.get('significance_sigma'))} | "
                f"{_format_number(line.get('observed_centroid_angstrom'), 1)} | "
                f"{_format_number(line.get('equivalent_width_signed_rest_angstrom'))} | "
                f"{flags} |"
            )
        lines.extend(
            [
                "",
                "Claim links:",
                "",
            ]
        )
        for claim in record["audit_sidecar"]["claims"]:
            fact_ids = ", ".join(f"`{item}`" for item in claim["fact_ids"])
            lines.append(f"- {claim['text']} → {fact_ids}")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--train-limit", type=int, help="select the first N training candidates")
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/raw_spectra/manifest.json",
    )
    parser.add_argument(
        "--accepted-candidates",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1a_accepted/candidates.jsonl",
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/hf/datasets/desi_x_ls_aion",
    )
    parser.add_argument(
        "--drafts",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_pilot_drafts.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "_local/astroprism/stage1b_spectrum_caption_pilot",
    )
    parser.add_argument(
        "--requests-only",
        action="store_true",
        help="write compact generation requests and audit facts without requiring drafts",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_manifest = _load_json(args.raw_manifest)
    accepted_rows = _load_jsonl(args.accepted_candidates)
    records = _select_candidate_records(
        accepted_rows,
        requested_indices=args.indices,
        train_limit=args.train_limit,
    )
    indices = [int(record["candidate_index"]) for record in records]

    raw_candidates = raw_manifest["candidates"]
    if len(raw_candidates) != len(accepted_rows):
        raise ValueError("raw and accepted candidate counts differ")
    for record in records:
        index = int(record["candidate_index"])
        raw = raw_candidates[index]
        for key in ("candidate_index", "paired_targetid", "caption_object_id"):
            if raw[key] != record[key]:
                raise ValueError(f"candidate {index}: raw/accepted identity mismatch for {key}")

    raw_dir = args.raw_manifest.parent
    arrays: dict[str, np.ndarray] = {}
    for key in ("spectrum_flux", "spectrum_ivar", "spectrum_mask", "spectrum_wavelength"):
        output = raw_manifest["outputs"][key]
        path = raw_dir / output["path"]
        if sha256_file(path) != output["sha256"]:
            raise ValueError(f"raw array checksum mismatch: {path}")
        arrays[key] = np.load(path, mmap_mode="r")

    catalog_rows = _load_catalog_rows(records, args.paired_root)
    drafts_payload: dict[str, Any] | None = None
    drafts: dict[str, Any] = {}
    if not args.requests_only:
        drafts_payload = _load_json(args.drafts)
        loaded_drafts = drafts_payload.get("captions")
        if not isinstance(loaded_drafts, dict):
            raise ValueError("draft file must contain a captions object")
        drafts = loaded_drafts

    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        if not args.requests_only:
            (staging / "plots").mkdir()
        packaged: list[dict[str, Any]] = []
        generation_requests: list[dict[str, Any]] = []
        request_audits: list[dict[str, Any]] = []
        for accepted in records:
            index = int(accepted["candidate_index"])
            catalog = catalog_rows[index]
            measurement = measure_spectrum_facts(
                arrays["spectrum_wavelength"][index],
                arrays["spectrum_flux"][index],
                arrays["spectrum_ivar"][index],
                arrays["spectrum_mask"][index],
                catalog_redshift=catalog["redshift"],
            )
            measurement = _json_safe(measurement)
            facts = _structured_facts(measurement, catalog, candidate_index=index)
            compact_payload = _compact_caption_payload(measurement, catalog)
            model_input = {
                "system_prompt": ALIGNMENT_CAPTION_PROMPT,
                "compact_spectral_facts": compact_payload,
            }
            compact_payload_sha256 = sha256_json(compact_payload)
            model_input_sha256 = sha256_json(model_input)

            split = accepted["split"]
            is_held_out = split != "train"
            generation_requests.append(
                {
                    "schema_version": 1,
                    "candidate_index": index,
                    "paired_targetid": accepted["paired_targetid"],
                    "split": split,
                    "model_input": model_input,
                    "compact_payload_sha256": compact_payload_sha256,
                    "model_input_sha256": model_input_sha256,
                }
            )
            exposure_policy = _compact_policy_metadata(measurement)
            request_audits.append(
                {
                    "schema_version": 1,
                    "artifact_type": "astroprism_spectrum_caption_request_audit",
                    "candidate_index": index,
                    "paired_targetid": accepted["paired_targetid"],
                    "caption_object_id": accepted["caption_object_id"],
                    "source_group_id": accepted["source_group_id"],
                    "split": split,
                    "model_input_sha256": model_input_sha256,
                    "compact_payload_sha256": compact_payload_sha256,
                    "exposure_policy": exposure_policy,
                    "teacher_reference": {
                        "facts": facts,
                        "measurement_summary": measurement,
                        "catalog_context": catalog,
                    },
                    "training_policy": {
                        "eligible_for_training": False,
                        "role": (
                            "held_out_inspection_only" if is_held_out else "generation_request_only"
                        ),
                    },
                }
            )
            if args.requests_only:
                continue

            draft = drafts.get(str(index))
            if not isinstance(draft, dict):
                raise ValueError(f"missing draft caption for candidate {index}")
            _validate_draft(draft, facts, candidate_index=index)
            assert drafts_payload is not None

            packaged.append(
                {
                    "schema_version": 1,
                    "artifact_type": "astroprism_spectrum_caption_candidate",
                    "candidate_index": index,
                    "paired_targetid": accepted["paired_targetid"],
                    "caption_object_id": accepted["caption_object_id"],
                    "source_group_id": accepted["source_group_id"],
                    "split": split,
                    "student_input": {
                        "modalities": ["spectrum"],
                        "raw_manifest": str(args.raw_manifest.resolve()),
                        "array_index": index,
                        "wavelength_frame": "observed_vacuum",
                        "spectrum_length": int(arrays["spectrum_flux"].shape[1]),
                    },
                    "teacher_reference": {
                        "facts": facts,
                        "measurement_summary": measurement,
                        "catalog_context": catalog,
                    },
                    "spectrum_caption": {
                        "original": draft["student_caption"],
                        "revised": None,
                        "status": "pending_review",
                    },
                    "audit_sidecar": {
                        "active_modalities": ["spectrum"],
                        "claims": draft["claims"],
                        "model_exposure_policy": exposure_policy,
                        "limitations": measurement["limitations"],
                        "spectral_grounding_status": "candidate_unverified",
                    },
                    "generation": {
                        "method": "interactive_language_generation_from_compact_facts",
                        "generator": drafts_payload["generator"],
                        "decoding_settings": drafts_payload.get("decoding_settings"),
                        "paper_source": PAPER_URL,
                        "paper_version": PAPER_VERSION,
                        "prompt_adaptation": "AstroPRISM concise A.5 adaptation",
                        "prompt_sha256": sha256_text(ALIGNMENT_CAPTION_PROMPT),
                        "compact_payload_schema_version": COMPACT_PAYLOAD_SCHEMA_VERSION,
                        "compact_policy_version": COMPACT_POLICY_VERSION,
                        "compact_payload_sha256": compact_payload_sha256,
                        "model_input_sha256": model_input_sha256,
                    },
                    "training_policy": {
                        "eligible_for_training": False,
                        "eligible_after_review": not is_held_out,
                        "role": (
                            "held_out_inspection_only"
                            if is_held_out
                            else "training_candidate_pending_review"
                        ),
                        "restrictions": (
                            [
                                "Never include in alignment, SFT, reward, or prompt-tuning data.",
                                "Review feedback contaminates this object for clean benchmark claims.",
                            ]
                            if is_held_out
                            else ["Requires explicit human acceptance before loader integration."]
                        ),
                    },
                }
            )
            _plot_record(
                staging / "plots" / f"candidate_{index:04d}.png",
                arrays["spectrum_wavelength"][index],
                arrays["spectrum_flux"][index],
                arrays["spectrum_ivar"][index],
                arrays["spectrum_mask"][index],
                measurement,
                candidate_index=index,
                target_id=accepted["paired_targetid"],
            )

        _write_jsonl(staging / "generation_requests.jsonl", generation_requests)
        _write_jsonl(staging / "request_audit.jsonl", request_audits)
        if not args.requests_only:
            _write_jsonl(staging / "captions.jsonl", packaged)
            (staging / "review.md").write_text(_review_markdown(packaged), encoding="utf-8")
        output_entries = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            output_entries[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        manifest = {
            "schema_version": 1,
            "artifact_type": (
                "astroprism_spectrum_caption_requests"
                if args.requests_only
                else "astroprism_spectrum_caption_pilot"
            ),
            "role": (
                "generation_requests_only_not_training_eligible"
                if args.requests_only
                else "review_only_not_training_eligible"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_indices": indices,
            "row_count": len(records),
            "split_counts": {
                split: sum(record["split"] == split for record in records)
                for split in sorted({record["split"] for record in records})
            },
            "held_out_candidate_indices": [
                record["candidate_index"] for record in records if record["split"] != "train"
            ],
            "sources": {
                "raw_manifest": {
                    "path": str(args.raw_manifest.resolve()),
                    "sha256": sha256_file(args.raw_manifest),
                },
                "accepted_candidates": {
                    "path": str(args.accepted_candidates.resolve()),
                    "sha256": sha256_file(args.accepted_candidates),
                },
                **(
                    {
                        "drafts": {
                            "path": str(args.drafts.resolve()),
                            "sha256": sha256_file(args.drafts),
                        }
                    }
                    if not args.requests_only
                    else {}
                ),
                "paired_metadata_root": str(args.paired_root.resolve()),
                "paper": {"url": PAPER_URL, "version": PAPER_VERSION},
            },
            "implementation": {
                "fact_extractor_version": FACT_EXTRACTOR_VERSION,
                "compact_payload_schema_version": COMPACT_PAYLOAD_SCHEMA_VERSION,
                "compact_policy_version": COMPACT_POLICY_VERSION,
                "tentative_min_median_pixel_snr": TENTATIVE_MIN_MEDIAN_PIXEL_SNR,
                "dn4000_class_boundaries": list(DN4000_CLASS_BOUNDARIES),
                "dn4000_class_boundary_sigma": DN4000_CLASS_BOUNDARY_SIGMA,
                "fact_extractor_sha256": sha256_file(
                    REPO_ROOT / "src/data/astroprism_spectrum_facts.py"
                ),
                "packager_sha256": sha256_file(Path(__file__)),
                "prompt_sha256": sha256_text(ALIGNMENT_CAPTION_PROMPT),
            },
            "outputs": output_entries,
        }
        _write_json(staging / "manifest.json", manifest)

        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"output already exists: {output}")
            shutil.rmtree(output)
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    print(
        json.dumps(
            {
                "status": "requests_prepared" if args.requests_only else "prepared",
                "output": str(output),
                "candidate_indices": indices,
                "generation_requests": str(output / "generation_requests.jsonl"),
                "request_audit": str(output / "request_audit.jsonl"),
                "review": None if args.requests_only else str(output / "review.md"),
                "captions": (None if args.requests_only else str(output / "captions.jsonl")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
