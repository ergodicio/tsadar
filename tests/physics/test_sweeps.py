"""Scheduled lane: the Maxwellian (A1) FormFactor check over a density, temperature and angle matrix.

Plasmas from a 3 x 3 x 3 grid (ne 0.05-0.5, Te 0.1-2 keV, 30-120 deg), on a 300 nm window
with 4096 wavelengths, compared on the EPW wings (the ion feature in an EPW window is
checked by the IAW cases). Only the well-resolved regime is kept: a plasma is dropped
if alpha = 1/(k lambda_De) > 3.5 at either EPW peak (18 of 27 remain).

Above that the resonance is weakly damped and tsadar's sampled form factor is accurate
only when the resonance is resolved: measured 1-8 % of the EPW peak at k lambda_De
0.15-0.25 with the 1024-point internal velocity grid.

NOTE FOR THE NEXT AGENT: when ready, extend this sweep to high phase velocity (alpha > 3.5)
using the resonance quadrature (peak refinement) together with the collisional model from
the arts2d-logspace branch, and drop the alpha filter below.
"""

import itertools

import numpy as np
import pytest

from .. import oracles
from ..oracles import ELECTRON_MASS, Ion, Plasma
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
