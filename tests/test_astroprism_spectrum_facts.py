from __future__ import annotations

import numpy as np
import pytest
from src.data.astroprism_spectrum_facts import (
    SpectralLineDefinition,
    _robust_weighted_linear_fit,
    measure_line,
    measure_spectrum_facts,
)


def _line_definition(nature: str) -> SpectralLineDefinition:
    return SpectralLineDefinition(
        key=f"synthetic_{nature}",
        label=f"synthetic {nature}",
        rest_wavelength_angstrom=5000.0,
        nature=nature,  # type: ignore[arg-type]
        line_window_rest_angstrom=(4997.5, 5002.5),
        continuum_windows_rest_angstrom=((4970.0, 4990.0), (5010.0, 5030.0)),
    )


def test_robust_fit_refits_after_last_clipping_update() -> None:
    x = np.arange(20, dtype=np.float64)
    y = 2.0 + 3.0 * x
    y[10] += 100.0
    weight = np.ones_like(x)

    coefficients, _, _, keep = _robust_weighted_linear_fit(
        x, y, weight, iterations=1, clip_sigma=2.0
    )

    assert not keep[10]
    expected, _, _, _ = np.linalg.lstsq(
        np.column_stack((np.ones(np.count_nonzero(keep)), x[keep])), y[keep], rcond=None
    )
    np.testing.assert_allclose(coefficients, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("nature", "line_delta", "expected_sign"),
    [("emission", 5.0, 1), ("absorption", -5.0, -1)],
)
def test_measure_line_uses_signed_rest_equivalent_width(
    nature: str, line_delta: float, expected_sign: int
) -> None:
    wavelength = np.arange(4950.0, 5051.0)
    flux = np.full(wavelength.shape, 10.0)
    in_line = (wavelength >= 4998.0) & (wavelength <= 5002.0)
    flux[in_line] += line_delta
    ivar = np.full(wavelength.shape, 100.0)
    mask = np.zeros(wavelength.shape, dtype=bool)

    measured = measure_line(
        wavelength,
        flux,
        ivar,
        mask,
        catalog_redshift=0.0,
        definition=_line_definition(nature),
    )

    assert measured["detection_class"] == "detected"
    assert np.sign(measured["equivalent_width_signed_rest_angstrom"]) == expected_sign
    assert abs(measured["equivalent_width_signed_rest_angstrom"]) == pytest.approx(2.5, rel=1e-6)
    assert measured["significance_sigma"] > 5.0


def test_measure_line_marks_masked_feature_for_review() -> None:
    wavelength = np.arange(4950.0, 5051.0)
    flux = np.full(wavelength.shape, 10.0)
    in_line = (wavelength >= 4998.0) & (wavelength <= 5002.0)
    flux[in_line] += 10.0
    ivar = np.full(wavelength.shape, 100.0)
    mask = np.zeros(wavelength.shape, dtype=bool)
    mask[np.flatnonzero(wavelength == 5000.0)] = True

    measured = measure_line(
        wavelength,
        flux,
        ivar,
        mask,
        catalog_redshift=0.0,
        definition=_line_definition("emission"),
    )

    assert measured["detection_class"] == "detected"
    assert measured["quality"] == "low_confidence"
    assert measured["masked_line_pixels"] == 1
    assert "masked_pixels_in_line_window" in measured["quality_flags"]


def test_measure_line_rejects_partially_truncated_continuum_windows() -> None:
    wavelength = np.arange(4980.0, 5041.0)
    flux = np.full(wavelength.shape, 10.0)
    ivar = np.full(wavelength.shape, 100.0)
    mask = np.zeros(wavelength.shape, dtype=bool)

    measured = measure_line(
        wavelength,
        flux,
        ivar,
        mask,
        catalog_redshift=0.0,
        definition=_line_definition("emission"),
    )

    assert measured["status"] == "out_of_coverage"
    assert measured["detection_class"] == "unavailable"


def test_measure_spectrum_facts_uses_narrow_fnu_dn4000() -> None:
    wavelength = np.arange(3600.0, 7201.0)
    # A flat F_nu spectrum has f_lambda proportional to lambda^-2.  Increase
    # F_nu by 50% only in the red Dn4000 band.
    flux = 1.0e8 / wavelength**2
    red_band = (wavelength >= 4000.0) & (wavelength <= 4100.0)
    flux[red_band] *= 1.5
    ivar = np.full(wavelength.shape, 1.0e4)
    mask = np.zeros(wavelength.shape, dtype=bool)

    measured = measure_spectrum_facts(wavelength, flux, ivar, mask, catalog_redshift=0.0)

    assert measured["dn4000"]["status"] == "measured"
    assert measured["dn4000"]["value"] == pytest.approx(1.5, rel=1e-12)
    assert measured["dn4000"]["strength_label"] == "moderate"


def test_dn4000_rejects_partially_truncated_band() -> None:
    wavelength = np.arange(3900.0, 7201.0)
    flux = 1.0e8 / wavelength**2
    ivar = np.full(wavelength.shape, 1.0e4)
    mask = np.zeros(wavelength.shape, dtype=bool)

    measured = measure_spectrum_facts(wavelength, flux, ivar, mask, catalog_redshift=0.0)

    assert measured["dn4000"]["status"] == "unavailable"
    assert np.isnan(measured["dn4000"]["value"])


def test_continuum_rejects_partially_truncated_band() -> None:
    wavelength = np.arange(3600.0, 6301.0)
    flux = 1.0e8 / wavelength**2
    ivar = np.full(wavelength.shape, 1.0e4)
    mask = np.zeros(wavelength.shape, dtype=bool)

    measured = measure_spectrum_facts(wavelength, flux, ivar, mask, catalog_redshift=0.0)

    assert measured["continuum"]["status"] == "unavailable"
    assert np.isnan(measured["continuum"]["red_to_blue_ratio"])


def test_measure_spectrum_facts_rejects_nonmonotonic_wavelength() -> None:
    wavelength = np.array([4000.0, 4001.0, 4001.0, 4002.0])
    flux = np.ones(4)
    ivar = np.ones(4)
    mask = np.zeros(4, dtype=bool)

    with pytest.raises(ValueError, match="strictly increasing"):
        measure_spectrum_facts(wavelength, flux, ivar, mask, catalog_redshift=0.1)
