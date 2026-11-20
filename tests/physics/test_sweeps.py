"""Scheduled lane: the A1 FormFactor check over a density, temperature and angle matrix (alpha <= 3.5)."""

import itertools

import numpy as np
import pytest

from tsverify import oracle as oracles
from tsverify.oracle import ELECTRON_MASS, Ion, Plasma

from . import harness

pytestmark = [pytest.mark.physics, pytest.mark.slow]

LINF_TOL = 2.0e-3
MAX_ALPHA = 3.5
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
        if 1.0 / _min_k_lambda_at_epw_peaks(plasma) <= MAX_ALPHA:
            yield pytest.param(plasma, id=f"ne{ne}-Te{te}-{angle:.0f}deg")


@pytest.mark.parametrize("plasma", list(_cases()))
def test_form_factor_matches_the_analytic_spectrum_across_plasmas(plasma):
    lam, model = harness.form_factor(plasma, LAM_RANGE, npts=4096)
    wings = np.abs(lam - plasma.lam) > 3.0
    linf, _ = oracles.relative_errors(model[wings], oracles.spectral_density(lam[wings], plasma))
    assert linf < LINF_TOL, f"A1-sweep: EPW wings depart from the analytic spectrum by {linf:.2e} (tol {LINF_TOL:.0e})"
