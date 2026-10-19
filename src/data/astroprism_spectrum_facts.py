"""Deterministic, audit-friendly facts for AstroPRISM DESI spectra.

The fact extractor implements the intermediate stage used by the
AstroSpecLM Figure 1 / Appendix A.5 recipe: numerical spectra are converted
to a compact structured report before any language model writes a caption.

This module intentionally does not generate prose.  It measures a conservative
set of galaxy features using a supplied catalog redshift, retaining that
conditioning in every line record.  The measurements are intended for caption
candidate construction and expert review, not as a replacement for a survey
pipeline or publication-quality line fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Nature = Literal["emission", "absorption"]


@dataclass(frozen=True)
class SpectralLineDefinition:
    """One redshift-conditioned local line measurement."""

    key: str
    label: str
    rest_wavelength_angstrom: float
    nature: Nature
    line_window_rest_angstrom: tuple[float, float]
    continuum_windows_rest_angstrom: tuple[tuple[float, float], tuple[float, float]]
    caveat: str | None = None


# Vacuum wavelengths are used because DESI wavelengths are on a vacuum grid.
# Blend reference wavelengths are descriptive centroids; the full integration
# window, rather than the reference centroid, defines the measurement.
GALAXY_LINE_DEFINITIONS: tuple[SpectralLineDefinition, ...] = (
    SpectralLineDefinition(
        "oii_3727_3729",
        "[O II] 3727/3729 blend",
        3728.48,
        "emission",
        (3718.0, 3739.0),
        ((3653.0, 3710.0), (3745.0, 3800.0)),
        "The doublet is measured as one blended feature.",
    ),
    SpectralLineDefinition(
        "ca_k_3934",
        "Ca K",
        3934.78,
        "absorption",
        (3923.0, 3946.0),
        ((3890.0, 3915.0), (4005.0, 4030.0)),
    ),
    SpectralLineDefinition(
        "ca_h_3969",
        "Ca H",
        3969.59,
        "absorption",
        (3957.0, 3982.0),
        ((3890.0, 3915.0), (4005.0, 4030.0)),
        "Ca H can be blended with H epsilon at this resolution.",
    ),
    SpectralLineDefinition(
        "hbeta_4863",
        "H-beta",
        4862.68,
        "emission",
        (4854.0, 4872.0),
        ((4770.0, 4820.0), (4890.0, 4930.0)),
    ),
    SpectralLineDefinition(
        "oiii_4960",
        "[O III] 4960",
        4960.30,
        "emission",
        (4953.0, 4968.0),
        ((4890.0, 4930.0), (5020.0, 5050.0)),
    ),
    SpectralLineDefinition(
        "oiii_5008",
        "[O III] 5008",
        5008.24,
        "emission",
        (5000.0, 5017.0),
        ((4970.0, 4990.0), (5020.0, 5050.0)),
    ),
    SpectralLineDefinition(
        "mgb_5175",
        "Mg b blend",
        5175.0,
        "absorption",
        (5155.0, 5196.0),
        ((5100.0, 5140.0), (5205.0, 5245.0)),
        "The Mg b triplet is measured as one blended absorption feature.",
    ),
    SpectralLineDefinition(
        "nad_5893",
        "Na D blend",
        5893.0,
        "absorption",
        (5878.0, 5910.0),
        ((5820.0, 5860.0), (5925.0, 5965.0)),
        "The Na D doublet is measured as one blended absorption feature.",
    ),
    SpectralLineDefinition(
        "nii_6550",
        "[N II] 6550",
        6549.86,
        "emission",
        (6543.0, 6556.0),
        ((6480.0, 6525.0), (6605.0, 6640.0)),
    ),
    SpectralLineDefinition(
        "halpha_6565",
        "H-alpha",
        6564.61,
        "emission",
        (6557.0, 6573.0),
        ((6480.0, 6525.0), (6605.0, 6640.0)),
    ),
    SpectralLineDefinition(
        "nii_6585",
        "[N II] 6585",
        6585.27,
        "emission",
        (6578.0, 6593.0),
        ((6480.0, 6525.0), (6605.0, 6640.0)),
    ),
    SpectralLineDefinition(
        "sii_6718",
        "[S II] 6718",
        6718.29,
        "emission",
        (6711.0, 6725.0),
        ((6650.0, 6690.0), (6750.0, 6790.0)),
    ),
    SpectralLineDefinition(
        "sii_6733",
        "[S II] 6733",
        6732.67,
        "emission",
        (6726.0, 6740.0),
        ((6650.0, 6690.0), (6750.0, 6790.0)),
    ),
)


DESI_ARM_OVERLAPS_ANGSTROM: tuple[tuple[float, float], ...] = (
    (5635.0, 5939.0),
    (7445.0, 7731.0),
)

# Broad review flags, not extra mask bits. The extracted boolean mask has
# already collapsed the original DESI bit reasons, and no sky model was
# retained, so these intervals only request human inspection.
REVIEW_SENSITIVE_OBSERVED_INTERVALS_ANGSTROM: tuple[tuple[str, float, float], ...] = (
    ("telluric_b_band_neighborhood", 6860.0, 6950.0),
    ("telluric_a_band_neighborhood", 7580.0, 7700.0),
    ("red_sky_residual_sensitive", 7000.0, 9824.0),
)


def _as_1d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _validated_arrays(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wavelength = _as_1d_float("wavelength", wavelength)
    flux = _as_1d_float("flux", flux)
    ivar = _as_1d_float("ivar", ivar)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {mask.shape}")
    if not (wavelength.shape == flux.shape == ivar.shape == mask.shape):
        raise ValueError("wavelength, flux, ivar, and mask shapes must agree")
    if wavelength.size < 3:
        raise ValueError("a spectrum must contain at least three samples")
    if not np.isfinite(wavelength).all() or np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength must be finite and strictly increasing")
    if not np.isfinite(flux).all():
        raise ValueError("flux must be finite")
    if not np.isfinite(ivar).all() or np.any(ivar < 0):
        raise ValueError("ivar must be finite and nonnegative")
    return wavelength, flux, ivar, mask


def _interval_mask(values: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    return (values >= interval[0]) & (values <= interval[1])


def _robust_weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    iterations: int = 4,
    clip_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit ``y = intercept + slope*x`` with conservative sigma clipping."""

    keep = np.ones(y.size, dtype=bool)
    coefficients = np.array([np.nan, np.nan], dtype=np.float64)
    covariance = np.full((2, 2), np.nan, dtype=np.float64)
    residual_mad = np.nan
    for _ in range(iterations):
        if np.count_nonzero(keep) < 8:
            raise ValueError("too few continuum samples after clipping")
        design = np.column_stack((np.ones(np.count_nonzero(keep)), x[keep]))
        sqrt_weight = np.sqrt(weight[keep])
        weighted_design = design * sqrt_weight[:, None]
        weighted_y = y[keep] * sqrt_weight
        coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        residual = y - (coefficients[0] + coefficients[1] * x)
        center = float(np.median(residual[keep]))
        residual_mad = float(1.4826 * np.median(np.abs(residual[keep] - center)))
        if not np.isfinite(residual_mad) or residual_mad <= 0:
            break
        updated = np.abs(residual - center) <= clip_sigma * residual_mad
        if np.array_equal(updated, keep):
            break
        keep = updated

    if np.count_nonzero(keep) < 8:
        raise ValueError("too few continuum samples after clipping")
    design = np.column_stack((np.ones(np.count_nonzero(keep)), x[keep]))
    sqrt_weight = np.sqrt(weight[keep])
    weighted_design = design * sqrt_weight[:, None]
    weighted_y = y[keep] * sqrt_weight
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    full_residual = y - (coefficients[0] + coefficients[1] * x)
    center = float(np.median(full_residual[keep]))
    residual_mad = float(1.4826 * np.median(np.abs(full_residual[keep] - center)))
    normal = design.T @ (weight[keep, None] * design)
    covariance = np.linalg.pinv(normal)
    residual = full_residual[keep]
    degrees_of_freedom = max(np.count_nonzero(keep) - 2, 1)
    reduced_chi2 = float(np.sum(weight[keep] * residual**2) / degrees_of_freedom)
    covariance *= max(reduced_chi2, 1.0)
    return coefficients, covariance, residual_mad, keep


def measure_line(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    *,
    catalog_redshift: float,
    definition: SpectralLineDefinition,
    detection_sigma: float = 5.0,
    tentative_sigma: float = 3.0,
) -> dict[str, Any]:
    """Measure one line relative to a clipped local linear continuum.

    Equivalent widths use the signed convention: emission is positive and
    absorption is negative. The line lookup is explicitly conditioned on the supplied
    catalog redshift; it is not an independent redshift measurement.
    """

    wavelength, flux, ivar, mask = _validated_arrays(wavelength, flux, ivar, mask)
    if not np.isfinite(catalog_redshift) or catalog_redshift < 0:
        raise ValueError("catalog_redshift must be finite and nonnegative")
    rest_wavelength = wavelength / (1.0 + catalog_redshift)
    base_valid = (~mask) & (ivar > 0)
    line_grid = _interval_mask(rest_wavelength, definition.line_window_rest_angstrom)
    line_valid = line_grid & base_valid
    side_grid = _interval_mask(
        rest_wavelength, definition.continuum_windows_rest_angstrom[0]
    ) | _interval_mask(rest_wavelength, definition.continuum_windows_rest_angstrom[1])
    side_valid = side_grid & base_valid
    line_coverage = float(np.count_nonzero(line_valid) / max(np.count_nonzero(line_grid), 1))
    side_coverage = float(np.count_nonzero(side_valid) / max(np.count_nonzero(side_grid), 1))
    required_min = min(
        definition.line_window_rest_angstrom[0],
        *(window[0] for window in definition.continuum_windows_rest_angstrom),
    )
    required_max = max(
        definition.line_window_rest_angstrom[1],
        *(window[1] for window in definition.continuum_windows_rest_angstrom),
    )
    full_window_coverage = (
        rest_wavelength[0] <= required_min and rest_wavelength[-1] >= required_max
    )

    common: dict[str, Any] = {
        "key": definition.key,
        "label": definition.label,
        "nature": definition.nature,
        "rest_wavelength_angstrom": definition.rest_wavelength_angstrom,
        "expected_observed_wavelength_angstrom": (
            definition.rest_wavelength_angstrom * (1.0 + catalog_redshift)
        ),
        "line_window_rest_angstrom": list(definition.line_window_rest_angstrom),
        "continuum_windows_rest_angstrom": [
            list(window) for window in definition.continuum_windows_rest_angstrom
        ],
        "line_valid_fraction": line_coverage,
        "continuum_valid_fraction": side_coverage,
        "conditioned_on_catalog_redshift": True,
        "caveat": definition.caveat,
    }
    if (
        not full_window_coverage
        or np.count_nonzero(line_grid) == 0
        or np.count_nonzero(side_grid) == 0
    ):
        return {
            **common,
            "status": "out_of_coverage",
            "quality": "unavailable",
            "detection_class": "unavailable",
        }
    if line_coverage < 0.8 or side_coverage < 0.8 or np.count_nonzero(side_valid) < 16:
        return {
            **common,
            "status": "masked",
            "quality": "unavailable",
            "detection_class": "unavailable",
        }

    center = definition.rest_wavelength_angstrom
    side_x = rest_wavelength[side_valid] - center
    side_y = flux[side_valid]
    side_weight = ivar[side_valid]
    try:
        coefficients, covariance, residual_mad, keep = _robust_weighted_linear_fit(
            side_x, side_y, side_weight
        )
    except (ValueError, np.linalg.LinAlgError):
        return {
            **common,
            "status": "not_assessed",
            "quality": "unavailable",
            "detection_class": "unavailable",
        }

    line_x = rest_wavelength[line_valid] - center
    line_flux = flux[line_valid]
    line_ivar = ivar[line_valid]
    line_wavelength = wavelength[line_valid]
    local_continuum = coefficients[0] + coefficients[1] * line_x
    delta_observed = np.gradient(line_wavelength)
    direction = 1.0 if definition.nature == "emission" else -1.0
    signed_residual = line_flux - local_continuum
    integrated_signed_strength = float(np.sum(signed_residual * delta_observed))
    expected_nature_strength = direction * integrated_signed_strength

    pixel_variance = float(np.sum(delta_observed**2 / line_ivar))
    continuum_vector = np.array(
        [np.sum(delta_observed), np.sum(delta_observed * line_x)],
        dtype=np.float64,
    )
    continuum_variance = float(continuum_vector @ covariance @ continuum_vector)
    formal_uncertainty = float(np.sqrt(max(pixel_variance + continuum_variance, 0.0)))
    empirical_uncertainty = float(
        residual_mad * np.sqrt(np.sum(delta_observed**2)) if np.isfinite(residual_mad) else 0.0
    )
    strength_uncertainty = max(formal_uncertainty, empirical_uncertainty)
    significance = (
        expected_nature_strength / strength_uncertainty if strength_uncertainty > 0 else np.nan
    )
    continuum_level = float(np.median(local_continuum))
    if continuum_level > 0:
        equivalent_width = integrated_signed_strength / continuum_level / (1.0 + catalog_redshift)
        equivalent_width_uncertainty = (
            strength_uncertainty / continuum_level / (1.0 + catalog_redshift)
        )
    else:
        equivalent_width = np.nan
        equivalent_width_uncertainty = np.nan

    positive_residual = np.clip(direction * signed_residual, 0.0, None)
    if np.sum(positive_residual) > 0:
        observed_centroid = float(
            np.sum(line_wavelength * positive_residual) / np.sum(positive_residual)
        )
    else:
        observed_centroid = np.nan

    if np.isfinite(significance) and significance >= detection_sigma:
        status = "measured"
        quality = "valid"
        detection_class = "detected"
    elif np.isfinite(significance) and significance >= tentative_sigma:
        status = "measured"
        quality = "low_confidence"
        detection_class = "tentative"
    else:
        status = "not_detected"
        quality = "valid"
        detection_class = "not_detected"

    quality_flags: list[str] = []
    masked_line_pixels = int(np.count_nonzero(line_grid & (~base_valid)))
    if masked_line_pixels:
        quality_flags.append("masked_pixels_in_line_window")
        if quality == "valid":
            quality = "low_confidence"
    observed_lo = definition.line_window_rest_angstrom[0] * (1.0 + catalog_redshift)
    observed_hi = definition.line_window_rest_angstrom[1] * (1.0 + catalog_redshift)
    for overlap_lo, overlap_hi in DESI_ARM_OVERLAPS_ANGSTROM:
        if observed_hi >= overlap_lo and observed_lo <= overlap_hi:
            quality_flags.append("desi_arm_overlap_region")
            break
    for flag, sensitive_lo, sensitive_hi in REVIEW_SENSITIVE_OBSERVED_INTERVALS_ANGSTROM:
        if observed_hi >= sensitive_lo and observed_lo <= sensitive_hi:
            quality_flags.append(flag)

    return {
        **common,
        "status": status,
        "quality": quality,
        "detection_class": detection_class,
        "quality_flags": quality_flags,
        "masked_line_pixels": masked_line_pixels,
        "integrated_signed_flux_source_flux_angstrom": integrated_signed_strength,
        "integrated_strength_uncertainty_source_flux_angstrom": strength_uncertainty,
        "significance_sigma": float(significance),
        "equivalent_width_signed_rest_angstrom": float(equivalent_width),
        "equivalent_width_uncertainty_angstrom": float(equivalent_width_uncertainty),
        "observed_centroid_angstrom": (
            observed_centroid if detection_class != "not_detected" else None
        ),
        "local_continuum_source_flux": continuum_level,
        "continuum_sideband_mad_source_flux": float(residual_mad),
        "continuum_samples_used": int(np.count_nonzero(keep)),
    }


def _band_mean(
    rest_wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
    interval: tuple[float, float],
    *,
    fnu: bool,
) -> tuple[float, float, int, float]:
    grid = _interval_mask(rest_wavelength, interval)
    selected = grid & valid
    count = int(np.count_nonzero(selected))
    coverage = float(count / max(np.count_nonzero(grid), 1))
    if count == 0:
        return np.nan, np.nan, 0, coverage
    scale = rest_wavelength[selected] ** 2 if fnu else np.ones(count)
    values = flux[selected] * scale
    mean = float(np.mean(values))
    uncertainty = float(np.sqrt(np.sum(scale**2 / ivar[selected])) / count)
    return mean, uncertainty, count, coverage


def _continuum_band_location(
    rest_wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
    interval: tuple[float, float],
) -> tuple[float, float, int, float]:
    grid = _interval_mask(rest_wavelength, interval)
    # Remove the predefined diagnostic line windows from broad-band continuum.
    excluded = np.zeros_like(valid)
    for definition in GALAXY_LINE_DEFINITIONS:
        lo, hi = definition.line_window_rest_angstrom
        excluded |= (rest_wavelength >= lo - 4.0) & (rest_wavelength <= hi + 4.0)
    selected = grid & valid & (~excluded)
    total = int(np.count_nonzero(grid & (~excluded)))
    coverage = float(np.count_nonzero(selected) / max(total, 1))
    values = flux[selected]
    variances = 1.0 / ivar[selected]
    if values.size == 0:
        return np.nan, np.nan, 0, coverage
    keep = np.ones(values.size, dtype=bool)
    for _ in range(4):
        median = float(np.median(values[keep]))
        mad = float(1.4826 * np.median(np.abs(values[keep] - median)))
        if not np.isfinite(mad) or mad <= 0:
            break
        updated = np.abs(values - median) <= 4.0 * mad
        if np.array_equal(updated, keep):
            break
        keep = updated
    location = float(np.mean(values[keep]))
    formal = float(np.sqrt(np.sum(variances[keep])) / np.count_nonzero(keep))
    scatter = float(np.std(values[keep], ddof=1) / np.sqrt(np.count_nonzero(keep)))
    return location, max(formal, scatter), int(np.count_nonzero(keep)), coverage


def measure_spectrum_facts(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    *,
    catalog_redshift: float,
) -> dict[str, Any]:
    """Return a compact grounded report for one observed-frame spectrum."""

    wavelength, flux, ivar, mask = _validated_arrays(wavelength, flux, ivar, mask)
    if not np.isfinite(catalog_redshift) or catalog_redshift < 0:
        raise ValueError("catalog_redshift must be finite and nonnegative")
    valid = (~mask) & (ivar > 0)
    if np.count_nonzero(valid) == 0:
        raise ValueError("spectrum has no unmasked pixels with positive inverse variance")
    rest_wavelength = wavelength / (1.0 + catalog_redshift)

    dn_blue, dn_blue_error, dn_blue_count, dn_blue_coverage = _band_mean(
        rest_wavelength, flux, ivar, valid, (3850.0, 3950.0), fnu=True
    )
    dn_red, dn_red_error, dn_red_count, dn_red_coverage = _band_mean(
        rest_wavelength, flux, ivar, valid, (4000.0, 4100.0), fnu=True
    )
    dn_bands_in_range = rest_wavelength[0] <= 3850.0 and rest_wavelength[-1] >= 4100.0
    dn_coverage_valid = dn_bands_in_range and dn_blue_coverage >= 0.9 and dn_red_coverage >= 0.9
    if dn_blue > 0 and dn_red > 0 and dn_coverage_valid:
        dn4000 = float(dn_red / dn_blue)
        dn4000_error = float(
            dn4000 * np.sqrt((dn_red_error / dn_red) ** 2 + (dn_blue_error / dn_blue) ** 2)
        )
    else:
        dn4000 = np.nan
        dn4000_error = np.nan
    if not np.isfinite(dn4000):
        dn4000_label = "unavailable"
    elif dn4000 >= 1.6:
        dn4000_label = "strong"
    elif dn4000 >= 1.3:
        dn4000_label = "moderate"
    else:
        dn4000_label = "weak"

    blue, blue_error, blue_count, blue_coverage = _continuum_band_location(
        rest_wavelength, flux, ivar, valid, (4200.0, 4600.0)
    )
    red, red_error, red_count, red_coverage = _continuum_band_location(
        rest_wavelength, flux, ivar, valid, (6000.0, 6400.0)
    )
    continuum_bands_in_range = rest_wavelength[0] <= 4200.0 and rest_wavelength[-1] >= 6400.0
    continuum_coverage_valid = (
        continuum_bands_in_range and blue_coverage >= 0.9 and red_coverage >= 0.9
    )
    if blue > 0 and red > 0 and continuum_coverage_valid:
        red_blue_ratio = float(red / blue)
        red_blue_error = float(
            red_blue_ratio * np.sqrt((red_error / red) ** 2 + (blue_error / blue) ** 2)
        )
    else:
        red_blue_ratio = np.nan
        red_blue_error = np.nan
    if not np.isfinite(red_blue_ratio):
        continuum_label = "unavailable"
    elif red_blue_ratio >= 1.2:
        continuum_label = "red/rising"
    elif red_blue_ratio >= 1.1:
        continuum_label = "mildly red/rising"
    elif red_blue_ratio <= 0.8:
        continuum_label = "blue/declining"
    elif red_blue_ratio <= 0.9:
        continuum_label = "mildly blue/declining"
    else:
        continuum_label = "approximately flat"

    optical = valid & (rest_wavelength >= 3850.0) & (rest_wavelength <= 7000.0)
    pixel_snr = flux[optical] * np.sqrt(ivar[optical])
    median_pixel_snr = float(np.median(pixel_snr)) if pixel_snr.size else np.nan

    lines = [
        measure_line(
            wavelength,
            flux,
            ivar,
            mask,
            catalog_redshift=catalog_redshift,
            definition=definition,
        )
        for definition in GALAXY_LINE_DEFINITIONS
    ]
    return {
        "measurement_version": "astroprism-spectrum-facts-v1",
        "wavelength_frame": "observed_vacuum",
        "wavelength_unit": "angstrom",
        "flux_unit": "source_array_unit_not_encoded_in_local_artifact",
        "catalog_redshift": float(catalog_redshift),
        "coverage": {
            "observed_min_angstrom": float(wavelength[0]),
            "observed_max_angstrom": float(wavelength[-1]),
            "rest_min_angstrom": float(rest_wavelength[0]),
            "rest_max_angstrom": float(rest_wavelength[-1]),
            "valid_pixels": int(np.count_nonzero(valid)),
            "total_pixels": int(wavelength.size),
            "valid_fraction": float(np.count_nonzero(valid) / wavelength.size),
        },
        "quality": {
            "median_pixel_snr_rest_3850_7000": median_pixel_snr,
            "definition": "median flux*sqrt(ivar) over valid rest-frame pixels",
        },
        "continuum": {
            "label": continuum_label,
            "status": "measured" if np.isfinite(red_blue_ratio) else "unavailable",
            "red_to_blue_ratio": red_blue_ratio,
            "red_to_blue_ratio_uncertainty": red_blue_error,
            "blue_window_rest_angstrom": [4200.0, 4600.0],
            "red_window_rest_angstrom": [6000.0, 6400.0],
            "blue_location_source_flux": blue,
            "red_location_source_flux": red,
            "blue_samples": blue_count,
            "red_samples": red_count,
            "blue_valid_fraction": blue_coverage,
            "red_valid_fraction": red_coverage,
            "method": "four-sigma-clipped mean excluding diagnostic line windows",
        },
        "dn4000": {
            "value": dn4000,
            "uncertainty": dn4000_error,
            "strength_label": dn4000_label,
            "status": "measured" if np.isfinite(dn4000) else "unavailable",
            "definition": "narrow Dn4000 ratio of mean F_nu in 4000-4100 to 3850-3950 angstrom",
            "blue_samples": dn_blue_count,
            "red_samples": dn_red_count,
            "blue_valid_fraction": dn_blue_coverage,
            "red_valid_fraction": dn_red_coverage,
        },
        "lines": lines,
        "limitations": [
            "Line searches and rest-frame windows are conditioned on the external catalog redshift.",
            "Integrated uncertainties use the larger of propagated inverse-variance and local empirical-noise estimates.",
            "Local linear continua and fixed windows are suitable for caption screening, not publication-quality deblending.",
            "Signed rest-frame equivalent width is positive for emission and negative for absorption.",
            "Broad observed-frame quality flags request review because the original DESI mask bits, sky model, and resolution matrix were not retained.",
            "The extracted local arrays do not retain an explicit calibrated flux-unit attribute.",
        ],
    }


__all__ = [
    "GALAXY_LINE_DEFINITIONS",
    "SpectralLineDefinition",
    "measure_line",
    "measure_spectrum_facts",
]
