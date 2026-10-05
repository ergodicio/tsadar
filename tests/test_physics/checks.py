"""Reusable physical oracles for the integrated validation suite.

The helpers live outside the individual cases so their failure modes can be tested
directly.  In particular, :func:`assert_outward_peak_motion` is exercised with a
deliberately reversed response in ``test_collective_modes.py``.
"""

from __future__ import annotations

import numpy as np


def assert_finite_nonnegative_spectrum(spectrum, *, case_id: str) -> None:
    """Require a finite, nonnegative spectral density, up to roundoff."""

    values = np.asarray(spectrum)
    assert np.all(np.isfinite(values)), (
        f"{case_id}: the spectral density contains NaN or infinity; "
        "a physical Thomson spectrum must be finite"
    )

    scale = float(np.max(np.abs(values)))
    roundoff_floor = -1.0e-12 * scale
    assert float(np.min(values)) >= roundoff_floor, (
        f"{case_id}: spectral density became negative (minimum={np.min(values):.6e}, "
        f"allowed roundoff floor={roundoff_floor:.6e})"
    )


def quadratic_peak_wavelength(
    wavelengths_nm,
    spectrum,
    *,
    window_nm: tuple[float, float],
    case_id: str,
) -> float:
    """Locate a peak and refine its position with a three-point parabola."""

    wavelengths = np.asarray(wavelengths_nm)
    values = np.asarray(spectrum)
    in_window = (wavelengths > window_nm[0]) & (wavelengths < window_nm[1])
    candidate_indices = np.flatnonzero(in_window)
    assert (
        candidate_indices.size >= 3
    ), f"{case_id}: peak window {window_nm} nm contains fewer than three samples"

    peak_index = int(candidate_indices[np.argmax(values[in_window])])
    assert (
        0 < peak_index < values.size - 1
    ), f"{case_id}: peak landed at the spectral boundary; widen the wavelength range"
    assert peak_index not in (candidate_indices[0], candidate_indices[-1]), (
        f"{case_id}: maximum landed on the edge of peak window {window_nm} nm; "
        "the requested collective feature was not resolved"
    )

    left, center, right = values[peak_index - 1 : peak_index + 2]
    curvature = left - 2.0 * center + right
    assert (
        curvature < 0.0
    ), f"{case_id}: candidate at {wavelengths[peak_index]:.6f} nm is not a local maximum"

    fractional_index = 0.5 * (left - right) / curvature
    assert (
        abs(float(fractional_index)) <= 1.0
    ), f"{case_id}: unstable quadratic peak interpolation ({fractional_index=:.3g})"
    return float(
        wavelengths[peak_index]
        + fractional_index * (wavelengths[peak_index + 1] - wavelengths[peak_index])
    )


def assert_frequency_shifts_match(
    measured,
    expected,
    *,
    rtol: float,
    case_id: str,
    relation: str,
) -> None:
    """Compare plasma-wave detunings, never the much larger optical carrier."""

    np.testing.assert_allclose(
        np.asarray(measured),
        np.asarray(expected),
        rtol=rtol,
        atol=0.0,
        err_msg=(
            f"{case_id}: peak detunings violate {relation}. The tolerance applies to "
            "|omega_s - omega_L|, not to the optical carrier omega_s."
        ),
    )


def assert_outward_peak_motion(
    lower_density_shifts,
    higher_density_shifts,
    *,
    minimum_relative_increase: float,
    case_id: str,
) -> None:
    """Require increasing density to move both EPW peaks away from the laser line."""

    lower = np.asarray(lower_density_shifts)
    higher = np.asarray(higher_density_shifts)
    required = lower * (1.0 + minimum_relative_increase)
    assert np.all(higher > required), (
        f"{case_id}: higher electron density must move both EPW peaks outward by more "
        f"than {minimum_relative_increase:.1%}; lower-density shifts={lower}, "
        f"higher-density shifts={higher}, required={required}"
    )
