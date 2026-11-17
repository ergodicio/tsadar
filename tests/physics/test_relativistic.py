"""A12: the relativistic limit of tsadar's form factor.

tsadar carries relativity only through the first-order factor (1 + 2 w / w_L) on a
non-relativistic Maxwellian: Sheffield's O(beta) result, eq. (45) of Palastro et al.,
Phys. Rev. E 81, 036411 (2010). In the non-collective limit its wavelength spectrum
expands to Sheffield's (1 - 7/2 eps) correction, Selden eq. (3).

The reference is the exact non-collective spectrum of a Juttner plasma, Selden, Phys.
Lett. A 79, 405 (1980) (oracles.selden_spectrum). The first test checks that the oracle
reproduces Selden's own tables. The second documents tsadar's validity: the peak-normalized
shape departs from the exact spectrum by about T_e / m_e c^2 (0.7-2.3x over 45-135 deg,
largest in backscatter) and the peak blue shift by at most Selden's second-order term, eq. (4),
as expected of an O(beta) model. A non-relativistic spectrum would be off by O(beta), several percent at
2 keV.

NOTE FOR THE NEXT AGENT: these are non-collective checks. In the collective regime the EPW
phase velocity can approach c at moderate T_e and the first-order factor fails much sooner
(Palastro et al. 2010, figs. 5-9: T_e >= 5 keV at n_e/n_c ~ 0.05). A collective relativistic
test needs Palastro's fully relativistic form factor as the oracle, together with the
resonance quadrature used for the high-alpha regime.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from .. import oracles
from ..oracles import ELECTRON_MASS_KEV, Ion, Plasma
from . import harness

pytestmark = pytest.mark.physics

# Selden Table 1 (T_e = 20 keV, 90 deg): the Zhuravlev-Petrov spectrum S_Zh(eps), normalized to S(0)
SELDEN_TABLE_1 = {
    0.6: 0.0452, 0.5: 0.0819, 0.4: 0.1458, 0.3: 0.2532, 0.2: 0.4238, 0.1: 0.6745, 0.0: 1.0,
    -0.1: 1.3433, -0.2: 1.5703, -0.3: 1.5049, -0.4: 1.0782, -0.5: 0.4959, -0.6: 0.1113, -0.7: 0.0069,
}
# Selden Table 2 (90 deg): T_e in keV at which the peak sits at the given relative blue shift eps_c
SELDEN_TABLE_2 = {3.77: -0.05, 7.80: -0.1, 16.75: -0.2, 27.10: -0.3, 39.21: -0.4, 53.56: -0.5, 71.00: -0.6, 93.27: -0.7}


def test_oracle_reproduces_seldens_tables():
    norm = oracles.selden_spectrum(0.0, 90.0, 20.0)
    for eps, value in SELDEN_TABLE_1.items():
        assert oracles.selden_spectrum(eps, 90.0, 20.0) / norm == pytest.approx(value, abs=1e-4), f"Table 1, eps={eps}"
    for te, eps_c in SELDEN_TABLE_2.items():
        peak = minimize_scalar(
            lambda e: -oracles.selden_spectrum(e, 90.0, te), bounds=(-0.95, 0.5), method="bounded",
            options={"xatol": 1e-10},
        ).x
        assert peak == pytest.approx(eps_c, abs=5e-4), f"Table 2, Te={te} keV"


def _peak(x, y):
    i = int(np.argmax(y))
    left, mid, right = y[i - 1 : i + 2]
    return x[i] + 0.5 * (left - right) / (left - 2.0 * mid + right) * (x[1] - x[0])


# measured: shape Linf / (Te/mc^2) = 0.66-2.31. The peak misses by at most Selden's second-order blue shift,
# eq. (4), eps_c2 = 7 sin^2(theta/2) [1 + 7 sin^2(theta/2)] / (2 alpha^2), which an O(beta) model lacks
# (measured 0.5-0.9 of it).
SHAPE_TOL = 3.0  # in units of Te / m_e c^2
PEAK_TOL = 1.2  # in units of |eps_c2|


@pytest.mark.parametrize("theta", [45.0, 90.0, 135.0])
@pytest.mark.parametrize("te", [0.5, 2.0, 5.0])
def test_first_order_relativistic_accuracy_against_the_exact_spectrum(te, theta, record_property):
    # ne = 1e14 cm^-3: k lambda_De > 100, so the spectrum is non-collective to 1e-4
    plasma = Plasma(ne=1.0e-6, Te=te, ions=(Ion(A=1.0, Z=1.0, Ti=te),), angle_deg=theta)
    lam, model = harness.form_factor(plasma, (0.45 * plasma.lam, 1.6 * plasma.lam), npts=8192)
    eps = lam / plasma.lam - 1.0
    exact = oracles.selden_spectrum(eps, theta, te)
    model, exact = model / model.max(), exact / exact.max()
    beta2 = te / ELECTRON_MASS_KEV
    s2 = np.sin(np.deg2rad(theta) / 2.0) ** 2
    second_order_shift = 14.0 * s2 * (1.0 + 7.0 * s2) * beta2**2  # Selden eq. (4) with 2 alpha = 1 / beta2

    shape = float(np.abs(model - exact).max())
    shift = _peak(eps, model) - _peak(eps, exact)
    record_property("shape error / (Te/mc^2)", float(shape / beta2))
    record_property("peak error / second-order shift", float(shift / second_order_shift))
    assert shape < SHAPE_TOL * beta2, f"A12: shape off the exact spectrum by {shape:.2e} = {shape / beta2:.2f} Te/mc^2"
    assert abs(shift) < PEAK_TOL * second_order_shift, (
        f"A12: peak off by {shift:+.5f} = {shift / second_order_shift:.2f} of the second-order shift"
    )
