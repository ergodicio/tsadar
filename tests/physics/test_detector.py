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
from .cases import A1_CASES

pytestmark = pytest.mark.physics

# Measured on the fixed code. EPW: Linf <= 3.3e-3 at the notch edges, where tsadar samples the rectangular
# filter on its grid (it halves at 16 points per pixel), L1 <= 8.8e-4, peaks within 1e-3 nm. IAW: Linf
# <= 2.4e-4, L1 <= 1.8e-4. Before the velocity-grid f' and the Faddeeva Z', EPW was Linf 2.4e-2, L1 1.9e-2 with
# peaks moved -0.039/-0.030 nm, and IAW was Linf 1.2e-3 to 2.4e-3.
LINF_TOL = {"EPW": 5.0e-3, "IAW": 1.0e-3}
L1_TOL = {"EPW": 1.5e-3, "IAW": 5.0e-4}
PEAK_TOL_NM = {"EPW": 5.0e-3, "IAW": 5.0e-4}
PEAK_WINDOWS = {"EPW": ((445.0, 520.0), (532.0, 625.0)), "IAW": ((524.6, 526.45), (526.55, 528.4))}


def _oracle_detector(case, centers):
    det = case.detector
    transmission = oracles.rectangular_notch(*det.notch) if det.notch else None
    values = oracles.detector_spectrum(
        lambda lam: oracles.spectral_density(lam, case.plasma), centers, det.sigma, transmission=transmission
    )
    return oracles.normalize_in_ranges(centers, values, det.amp_ranges, det.amplitude)


@pytest.mark.parametrize("case", A1_CASES, ids=str)
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
    in_fit = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        in_fit |= (centers > low) & (centers < high)

    linf, l1 = oracles.relative_errors(model, expected, in_fit)
    linf_tol, l1_tol = LINF_TOL[det.channel], L1_TOL[det.channel]
    assert linf < linf_tol and l1 < l1_tol, (
        f"{case.problem}-L3/{case}: detector spectrum departs from the analytic one inside the fit ranges: "
        f"Linf={linf:.2e} (tol {linf_tol:.0e}), L1={l1:.2e} (tol {l1_tol:.0e})"
    )
    for window in PEAK_WINDOWS[det.channel]:
        shift = oracles.peak_wavelength(centers, model, window) - oracles.peak_wavelength(centers, expected, window)
        assert abs(shift) < PEAK_TOL_NM[det.channel], (
            f"{case.problem}-L3/{case}: the detector moved the peak in {window} nm by {shift:+.4f} nm "
            f"(tol {PEAK_TOL_NM[det.channel]:.0e} nm)"
        )
