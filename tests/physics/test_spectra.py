"""L2: FormFactor against the analytic spectral density."""

import pytest

from tsverify import oracle as oracles
from tsverify.cases import SPECTRUM_CASES

from . import harness
from .compare import L2_TOL

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("npts", [256, 2048])
@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_form_factor_matches_analytic_spectral_density(case, npts):
    lam, model = harness.form_factor(case.plasma, case.detector.lam_range, npts=npts)
    expected = oracles.spectral_density(lam, case.plasma)
    linf, l1 = oracles.relative_errors(model, expected)
    linf_tol, l1_tol = L2_TOL[case.problem]
    assert linf < linf_tol and l1 < l1_tol, (
        f"{case.problem}-L2/{case}: FormFactor departs from the analytic S(k, w): "
        f"peak-normalized Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )


def test_high_z_ion_acoustic_resonance_past_the_z_prime_table():
    """Z Te/Ti = 160 puts the IAW resonance at zeta = 9, past the Z' table: peaks and shape from the series tail."""
    plasma = oracles.Plasma(ne=0.5, Te=1.0, ions=(oracles.Ion(A=197.0, Z=40.0, Ti=0.25),))
    lam, model = harness.form_factor(plasma, (525.0, 528.0), npts=8192)
    expected = oracles.spectral_density(lam, plasma)
    linf, _ = oracles.relative_errors(model, expected)
    assert linf < 1.0e-2, f"high-Z IAW departs from the analytic spectrum by {linf:.2e}"
    for window in ((525.0, 526.45), (526.55, 528.0)):
        shift = oracles.peak_wavelength(lam, model, window) - oracles.peak_wavelength(lam, expected, window)
        assert abs(shift) < 5.0e-5, f"high-Z IAW peak in {window} nm moved by {shift:+.6f} nm"
