from __future__ import annotations

import json

import pytest
from tools.prepare_astroprism_spectrum_caption_pilot import (
    COMPACT_PAYLOAD_SCHEMA_VERSION,
    _compact_caption_payload,
    _compact_policy_metadata,
    _review_markdown,
    _select_candidate_records,
)


def _line(
    key: str,
    *,
    detection_class: str = "detected",
    quality: str = "valid",
    flags: list[str] | None = None,
    significance: float = 5.678,
) -> dict[str, object]:
    return {
        "key": key,
        "label": key.replace("_", " "),
        "nature": "emission",
        "detection_class": detection_class,
        "quality": quality,
        "quality_flags": flags or [],
        "significance_sigma": significance,
        "observed_centroid_angstrom": 5000.06,
        "equivalent_width_signed_rest_angstrom": 2.346,
        "equivalent_width_uncertainty_angstrom": 0.456,
        # Deliberately verbose audit-only fields that must not reach the model.
        "source_reference": "raw_spectra:candidate_index=7",
        "provenance": "spectral_measurement",
        "continuum_windows_rest_angstrom": [[4900.0, 4950.0], [5050.0, 5100.0]],
    }


def _measurement(
    *,
    snr: float = 4.126,
    continuum_ratio: float = 1.3334,
    continuum_error: float = 0.0104,
    dn4000_value: float = 1.5004,
    dn4000_error: float = 0.1004,
    lines: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "quality": {"median_pixel_snr_rest_3850_7000": snr},
        "continuum": {
            "status": "measured",
            "label": "red/rising",
            "red_to_blue_ratio": continuum_ratio,
            "red_to_blue_ratio_uncertainty": continuum_error,
        },
        "dn4000": {
            "status": "measured",
            "strength_label": "moderate",
            "value": dn4000_value,
            "uncertainty": dn4000_error,
        },
        "lines": lines or [],
    }


def test_compact_payload_schema_rounding_and_no_audit_envelope() -> None:
    payload = _compact_caption_payload(
        _measurement(lines=[_line("halpha_6565")]),
        {"redshift": 0.12345678},
    )

    assert payload == {
        "schema_version": COMPACT_PAYLOAD_SCHEMA_VERSION,
        "type": "galaxy",
        "z_cat": 0.123457,
        "snr": 4.13,
        "cont": {"class": "red/rising", "ratio": 1.333, "err": 0.01},
        "dn4000": {"class": "moderate", "value": 1.5, "err": 0.1, "cautious": True},
        "lines": [
            {
                "id": "spec.line.halpha_6565",
                "name": "halpha 6565",
                "nature": "emission",
                "conf": "detected",
                "sig": 5.68,
                "obs_A": 5000.1,
                "ew_rest_A": 2.35,
                "ew_err_A": 0.46,
            }
        ],
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "allowed_uses",
        "continuum_windows_rest_angstrom",
        "evidence_locator",
        "provenance",
        "required_student_inputs",
        "restrictions",
        "source_reference",
    ):
        assert forbidden not in serialized


def test_compact_payload_excludes_unusable_and_low_snr_tentative_lines() -> None:
    measurement = _measurement(
        snr=1.999,
        lines=[
            _line("kept_detected"),
            _line("suppressed_tentative", detection_class="tentative", significance=4.2),
            _line("masked_detected", flags=["masked_pixels_in_line_window"]),
            _line("unavailable", detection_class="unavailable", quality="unavailable"),
            _line("not_detected", detection_class="not_detected"),
        ],
    )

    payload = _compact_caption_payload(measurement, {"redshift": 0.1})
    assert [line["id"] for line in payload["lines"]] == ["spec.line.kept_detected"]

    policy = _compact_policy_metadata(measurement)
    reasons = {item["fact_id"]: item["reason"] for item in policy["excluded_lines"]}
    assert reasons == {
        "spec.line.suppressed_tentative": "tentative_suppressed_low_snr",
        "spec.line.masked_detected": "masked_line_window",
        "spec.line.unavailable": "unavailable",
        "spec.line.not_detected": "not_detected",
    }


def test_tentative_and_observed_frame_flags_are_compact_and_cautious() -> None:
    measurement = _measurement(
        snr=2.0,
        lines=[
            _line("tentative", detection_class="tentative", significance=3.5),
            _line(
                "flagged",
                flags=[
                    "desi_arm_overlap_region",
                    "telluric_a_band_neighborhood",
                    "telluric_b_band_neighborhood",
                    "red_sky_residual_sensitive",
                ],
            ),
        ],
    )

    lines = _compact_caption_payload(measurement, {"redshift": 0.2})["lines"]
    assert lines[0]["cautious"] is True
    assert "flags" not in lines[0]
    assert lines[1]["cautious"] is True
    assert lines[1]["flags"] == ["arm_overlap", "telluric", "red_sky"]


def test_weak_continuum_and_dn4000_significance_are_cautious() -> None:
    payload = _compact_caption_payload(
        _measurement(
            continuum_ratio=1.05,
            continuum_error=0.02,
            dn4000_value=1.1,
            dn4000_error=0.06,
        ),
        {"redshift": 0.3},
    )

    assert payload["cont"]["cautious"] is True
    assert payload["dn4000"]["cautious"] is True


@pytest.mark.parametrize(
    ("value", "error", "cautious"),
    [
        (1.403, 0.145, True),
        (1.845, 0.299, True),
        (1.435, 0.170, True),
        (1.328, 0.071, True),
        (1.850, 0.049, False),
    ],
)
def test_dn4000_class_boundary_crossing_is_cautious(
    value: float, error: float, cautious: bool
) -> None:
    payload = _compact_caption_payload(
        _measurement(dn4000_value=value, dn4000_error=error),
        {"redshift": 0.3},
    )

    assert payload["dn4000"].get("cautious", False) is cautious


def test_candidate_selector_supports_train_limit_and_explicit_order() -> None:
    rows = [
        {"candidate_index": 0, "split": "test"},
        {"candidate_index": 1, "split": "train"},
        {"candidate_index": 2, "split": "validation"},
        {"candidate_index": 3, "split": "train"},
    ]

    selected = _select_candidate_records(rows, requested_indices=None, train_limit=2)
    assert [row["candidate_index"] for row in selected] == [1, 3]
    selected = _select_candidate_records(rows, requested_indices=[3, 0], train_limit=None)
    assert [row["candidate_index"] for row in selected] == [3, 0]

    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_candidate_records(rows, requested_indices=[1], train_limit=1)
    with pytest.raises(ValueError, match="must be positive"):
        _select_candidate_records(rows, requested_indices=None, train_limit=0)
    with pytest.raises(ValueError, match="missing"):
        _select_candidate_records(rows, requested_indices=[99], train_limit=None)


def test_review_markdown_handles_unavailable_continuum_ratio() -> None:
    measurement = _measurement()
    measurement["catalog_redshift"] = 0.4
    measurement["continuum"] = {
        "status": "unavailable",
        "label": "unavailable",
        "red_to_blue_ratio": None,
        "red_to_blue_ratio_uncertainty": None,
    }
    record = {
        "candidate_index": 46,
        "split": "train",
        "paired_targetid": "123",
        "training_policy": {"role": "training_candidate_pending_review"},
        "spectrum_caption": {"original": "A grounded caption."},
        "teacher_reference": {"measurement_summary": measurement},
        "audit_sidecar": {"claims": []},
    }

    review = _review_markdown([record])

    assert "Continuum: `unavailable`; red/blue f_lambda ratio `— ± —`." in review
