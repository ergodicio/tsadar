"""Scheduled lane: the Maxwellian (A1) FormFactor check over a density, temperature and angle matrix.

27 plasmas from k lambda_De = 0.06 to 3 at the EPW peaks, on a 300 nm window with 4096
wavelengths, compared on the EPW wings (the ion feature in an EPW window is checked
by the IAW cases).

Known limit: where the EPW is weakly damped (k lambda_De < 0.25 at a peak), |eps| at the
resonance is below 1e-3 and the spectrum amplifies tsadar's ~1e-4 susceptibility
error, which is set by its 1024-point internal velocity grid. Measured: 8 % at
k lambda_De = 0.17, falling to 0.4 % with 4096 points (and a 4x finer xi2) and 0.1 % with
8192. Those cases are expected to fail until the kernel is refined. The xfail is not
strict because at k lambda_De < 0.1 the resonance can fall between wavelength samples
and pass by luck.
"""

import itertools

import numpy as np
import pytest

from .. import oracles
from ..oracles import ELECTRON_MASS, Ion, Plasma
from . import harness

pytestmark = [pytest.mark.physics, pytest.mark.slow]

LINF_TOL = 2.0e-3
WEAK_DAMPING_K_LAMBDA = 0.25
LAM_RANGE = (400.0, 700.0)


def _min_k_lambda_at_epw_peaks(plasma):
    lam = np.linspace(*LAM_RANGE, 30001)
    spectrum = oracles.spectral_density(lam, plasma)
    values = []
    for side in (lam < plasma.lam - 3.0, lam > plasma.lam + 3.0):
        k, _, _ = oracles.scattering_geometry(lam[np.argmax(np.where(side, spectrum, 0.0))], plasma)
        values.append(k * np.sqrt(plasma.Te / ELECTRON_MASS) / oracles.plasma_frequency(plasma.ne))
    return min(values)


def _cases():
    for ne, te, angle in itertools.product((0.05, 0.2, 0.5), (0.1, 0.6, 2.0), (30.0, 60.0, 120.0)):
        plasma = Plasma(ne=ne, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te / 5),), angle_deg=angle)
        k_lambda = _min_k_lambda_at_epw_peaks(plasma)
        marks = []
        if k_lambda < WEAK_DAMPING_K_LAMBDA:
            marks.append(pytest.mark.xfail(reason=f"weakly damped EPW, k lambda_De = {k_lambda:.2f}", strict=False))
        yield pytest.param(plasma, id=f"ne{ne}-Te{te}-{angle:.0f}deg", marks=marks)


@pytest.mark.parametrize("plasma", list(_cases()))
def test_form_factor_matches_the_analytic_spectrum_across_plasmas(plasma):
    lam, model = harness.form_factor(plasma, LAM_RANGE, npts=4096)
    wings = np.abs(lam - plasma.lam) > 3.0
    linf, _ = oracles.relative_errors(model[wings], oracles.spectral_density(lam[wings], plasma))
    assert linf < LINF_TOL, f"A1-sweep: EPW wings depart from the analytic spectrum by {linf:.2e} (tol {LINF_TOL:.0e})"
