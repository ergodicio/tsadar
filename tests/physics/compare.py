"""The detector-depth (L3) comparison shared by every test that checks the full diagnostic.

The oracle side takes an analytic S(lambda) and pushes it through an independent
detector model: a continuous rectangular notch, a continuous Gaussian spectral
response, an exact average over each pixel, and the fit-range amplitude
normalization.
"""

import numpy as np

from .. import oracles

# Compared inside the fit ranges, away from the notch edges: tsadar applies the rectangular notch on its
# sampled grid, so each edge is placed only to a grid step (up to 6e-3 of peak there, halving with twice
# the points per pixel). The notch itself is checked in test_notch.py. Measured on the fixed code over A1
# and A3: Linf <= 1.0e-3, L1 <= 6e-4, peaks within 1e-3 nm. Before the velocity-grid f' and the Faddeeva
# Z', the A1 EPW case was Linf 2.4e-2 with peaks moved -0.039/-0.030 nm, and IAW Linf 1.2e-3 to 2.4e-3.
LINF_TOL = {"EPW": 1.5e-3, "IAW": 1.0e-3}
L1_TOL = {"EPW": 1.0e-3, "IAW": 1.0e-3}
NOTCH_EDGE_SIGMAS = 2.0
PEAK_TOL_NM = {"EPW": 5.0e-3, "IAW": 5.0e-4}
PEAK_WINDOWS = {"EPW": ((445.0, 520.0), (532.0, 625.0)), "IAW": ((524.6, 526.45), (526.55, 528.4))}


def oracle_detector(det, centers, spectrum):
    """The analytic detector spectrum of ``spectrum`` (a callable of wavelength in nm) on ``det``."""
    transmission = oracles.rectangular_notch(*det.notch) if det.notch else None
    values = oracles.detector_spectrum(spectrum, centers, det.sigma, transmission=transmission)
    return oracles.normalize_in_ranges(centers, values, det.amp_ranges, det.amplitude)


def compared_pixels(det, centers):
    """Pixels inside the fit ranges, excluding NOTCH_EDGE_SIGMAS IRF widths around each notch edge."""
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        mask |= (centers > low) & (centers < high)
    if det.notch:
        center, width, _ = det.notch
        for edge in (center - width / 2, center + width / 2):
            mask &= np.abs(centers - edge) > NOTCH_EDGE_SIGMAS * det.sigma
    return mask


def assert_detector_matches(label, det, centers, model, expected, tol_scale=1.0):
    """Spectrum (Linf, L1) and peak-position agreement between the diagnostic and the oracle."""
    linf, l1 = oracles.relative_errors(model, expected, compared_pixels(det, centers))
    linf_tol, l1_tol = (tol_scale * tol[det.channel] for tol in (LINF_TOL, L1_TOL))
    assert linf < linf_tol and l1 < l1_tol, (
        f"{label}: detector spectrum departs from the analytic one in the fit ranges: "
        f"Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )
    for window in PEAK_WINDOWS[det.channel]:
        shift = oracles.peak_wavelength(centers, model, window) - oracles.peak_wavelength(centers, expected, window)
        assert abs(shift) < PEAK_TOL_NM[det.channel], (
            f"{label}: the detector moved the peak in {window} nm by {shift:+.4f} nm "
            f"(tol {PEAK_TOL_NM[det.channel]:.0e} nm)"
        )
