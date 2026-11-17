"""L3: the full ``ThomsonScatteringDiagnostic`` against the analytic detector spectrum.

The oracle side takes the same analytic S(k, w) as L2 and pushes it through an
independent detector model: a continuous rectangular notch, a continuous Gaussian
spectral response, an exact average over each pixel, and the fit-range amplitude
normalization. Anything between the form factor and the detector that shifts,
scales, or smears the spectrum (IRF centring, pixel grouping, the oversampled grid,
the notch, the amplitude rule) fails here against physics rather than against a
frozen array.
"""

import numpy as np
import pytest

from .. import oracles
from . import harness
from .cases import SPECTRUM_CASES

pytestmark = pytest.mark.physics

# Compared inside the fit ranges, away from the notch edges: tsadar applies the rectangular notch on its
# sampled grid, so each edge is placed only to a grid step (up to 6e-3 of peak there, halving with
# twice the points per pixel). The notch itself is checked in test_notch.py. Measured on the fixed code
# over A1 and A3: Linf <= 1.0e-3, L1 <= 6e-4, peaks within 1e-3 nm. Before the velocity-grid f' and the
# Faddeeva Z', the A1 EPW case was Linf 2.4e-2 with peaks moved -0.039/-0.030 nm, and IAW Linf 1.2e-3 to 2.4e-3.
LINF_TOL = {"EPW": 1.5e-3, "IAW": 1.0e-3}
L1_TOL = {"EPW": 1.0e-3, "IAW": 1.0e-3}
NOTCH_EDGE_SIGMAS = 2.0
PEAK_TOL_NM = {"EPW": 5.0e-3, "IAW": 5.0e-4}
PEAK_WINDOWS = {"EPW": ((445.0, 520.0), (532.0, 625.0)), "IAW": ((524.6, 526.45), (526.55, 528.4))}


def _oracle_detector(case, centers):
    det = case.detector
    transmission = oracles.rectangular_notch(*det.notch) if det.notch else None
    values = oracles.detector_spectrum(
        lambda lam: oracles.spectral_density(lam, case.plasma), centers, det.sigma, transmission=transmission
    )
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


@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_diagnostic_matches_analytic_detector_spectrum(case):
    det = case.detector
    centers, model = harness.detector(case.plasma, det)
    np.testing.assert_allclose(
        centers,
        np.linspace(*det.lam_range, det.pixels),
        rtol=0,
        atol=1e-9,
        err_msg=f"{case}: detector pixel centres moved",
    )
    expected = _oracle_detector(case, centers)
    linf, l1 = oracles.relative_errors(model, expected, compared_pixels(det, centers))
    linf_tol, l1_tol = (case.l3_tol_scale * tol[det.channel] for tol in (LINF_TOL, L1_TOL))
    assert linf < linf_tol and l1 < l1_tol, (
        f"{case.problem}-L3/{case}: detector spectrum departs from the analytic one in the fit ranges: "
        f"Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )
    for window in PEAK_WINDOWS[det.channel]:
        shift = oracles.peak_wavelength(centers, model, window) - oracles.peak_wavelength(centers, expected, window)
        assert abs(shift) < PEAK_TOL_NM[det.channel], (
            f"{case.problem}-L3/{case}: the detector moved the peak in {window} nm by {shift:+.4f} nm "
            f"(tol {PEAK_TOL_NM[det.channel]:.0e} nm)"
        )
