"""The oracle itself against physics it does not share code with.

Every other test in this suite trusts ``tests/oracles.py``. These check its
normalization and its collective modes against independent results:

* A6, the fluctuation-dissipation sum rule. For Maxwellian species in equilibrium
  (all temperatures equal) the frequency integral of S(k, w) is the static structure
  factor of a classical weakly coupled plasma, the Debye-Hueckel result
  ``S(k) = (k^2 + sum_s k_s^2) / (k^2 + k_e^2 + sum_s k_s^2)``, ``k_s^2 = 4 pi n_s Z_s^2 e^2 / T``.
  This fixes the absolute scale of both features and the ion weights Z_s^2 n_s / n_e.
* A2, the collective modes in their analytic limits: at small k lambda_De the EPW
  peak follows the kinetic Bohm-Gross expansion, and for cold ions the ion-acoustic
  peak sits at the screened sound speed.
"""

import numpy as np
import pytest

from .. import oracles
from ..oracles import ELECTRON_CHARGE_SQUARED, ELECTRON_MASS, PROTON_MASS, Ion, Plasma

pytestmark = pytest.mark.physics

EQUILIBRIUM = {
    "hydrogen": Plasma(ne=0.2, Te=0.5, ions=(Ion(A=1.0, Z=1.0, Ti=0.5),)),
    "carbon": Plasma(ne=0.5, Te=0.5, ions=(Ion(A=12.0, Z=6.0, Ti=0.5),)),
    "CH": Plasma(ne=0.5, Te=0.5, ions=(Ion(A=12.0, Z=6.0, Ti=0.5, fract=0.5), Ion(A=1.0, Z=1.0, Ti=0.5, fract=0.5))),
}


def _thermal_speed(temperature_kev, mass):
    return np.sqrt(temperature_kev / mass)


@pytest.mark.parametrize("k_lambda_de", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("name", EQUILIBRIUM)
def test_equilibrium_sum_rule(name, k_lambda_de):
    plasma = EQUILIBRIUM[name]
    omega_pe = oracles.plasma_frequency(plasma.ne)
    v_te = _thermal_speed(plasma.Te, ELECTRON_MASS)
    k = k_lambda_de * omega_pe / v_te

    # a wide grid for the electron feature and a fine one across the ion feature
    v_ti = min(_thermal_speed(ion.Ti, ion.A * PROTON_MASS) for ion in plasma.ions)
    c_s = max(_thermal_speed(ion.Z * plasma.Te + 3 * ion.Ti, ion.A * PROTON_MASS) for ion in plasma.ions)
    omega = np.unique(
        np.concatenate(
            [
                np.linspace(-14.0, 14.0, 200001) * k * v_te,
                np.linspace(-1.0, 1.0, 200001) * k * (10.0 * v_ti + 4.0 * c_s),
            ]
        )
    )
    electron_part, ion_part, _, _ = oracles.form_factor_terms(k, omega, plasma)
    total = np.trapezoid(electron_part + ion_part, omega)

    fract = np.array([ion.fract for ion in plasma.ions]) / sum(ion.fract for ion in plasma.ions)
    z = np.array([ion.Z for ion in plasma.ions])
    ne = plasma.ne * 1.0e20
    k_e2 = 4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ne / plasma.Te
    k_i2 = sum(
        4.0 * np.pi * ELECTRON_CHARGE_SQUARED * (f * ne / np.sum(z * fract)) * ion.Z**2 / ion.Ti
        for ion, f in zip(plasma.ions, fract)
    )
    expected = (k**2 + k_i2) / (k**2 + k_e2 + k_i2)
    assert total == pytest.approx(expected, rel=1e-4), (
        f"A6/{name}: integral of S(k, w) = {total:.6f}, Debye-Hueckel S(k) = {expected:.6f}"
    )


def _fixed_k_peak(plasma, k, omega):
    """Peak of the total S(k, w) over ``omega`` (uniform), refined by a parabola."""
    electron_part, ion_part, _, _ = oracles.form_factor_terms(k, omega, plasma)
    values = electron_part + ion_part
    i = np.argmax(values)
    assert 0 < i < values.size - 1, "peak on the edge of the frequency window"
    left, mid, right = values[i - 1 : i + 2]
    return omega[i] + 0.5 * (left - right) / (left - 2 * mid + right) * (omega[1] - omega[0])


@pytest.mark.parametrize("k_lambda_de", [0.1])  # below 0.1 the undamped peak is narrower than the grid
def test_epw_peak_follows_the_kinetic_bohm_gross_expansion(k_lambda_de):
    plasma = Plasma(ne=0.2, Te=0.5, ions=(Ion(A=1.0, Z=1.0, Ti=0.1),))
    omega_pe = oracles.plasma_frequency(plasma.ne)
    k = k_lambda_de * omega_pe / _thermal_speed(plasma.Te, ELECTRON_MASS)
    peak = _fixed_k_peak(plasma, k, np.linspace(0.9, 1.3, 400001) * omega_pe)
    a = k_lambda_de**2
    # (w/w_pe)^2 = 1 + 3 a + 6 a^2 + O(a^3), plus the ion plasma frequency Z me/M. The spectral peak tracks
    # the dispersion root while the damping is negligible, k lambda_De <= 0.1 (residual 10 a^3 there).
    expected = 1.0 + 3.0 * a + 6.0 * a**2 + ELECTRON_MASS / PROTON_MASS
    assert (peak / omega_pe) ** 2 == pytest.approx(expected, abs=20.0 * a**3), (
        f"A2: EPW peak (w/w_pe)^2 = {(peak / omega_pe) ** 2:.7f}, kinetic Bohm-Gross {expected:.7f}"
    )


@pytest.mark.parametrize("k_lambda_de", [0.1, 0.3])
@pytest.mark.parametrize("ti_over_te", [0.02, 0.05, 0.1])
def test_ion_acoustic_peak_sits_at_the_screened_sound_speed(k_lambda_de, ti_over_te):
    plasma = Plasma(ne=0.2, Te=1.0, ions=(Ion(A=12.0, Z=6.0, Ti=ti_over_te),))
    omega_pe = oracles.plasma_frequency(plasma.ne)
    k = k_lambda_de * omega_pe / _thermal_speed(plasma.Te, ELECTRON_MASS)
    c_s = np.sqrt((6.0 * plasma.Te / (1.0 + k_lambda_de**2) + 3.0 * ti_over_te * plasma.Te) / (12.0 * PROTON_MASS))
    peak = _fixed_k_peak(plasma, k, np.linspace(0.5, 1.5, 200001) * k * c_s)
    # warm-fluid sound speed with Debye screening; measured within 1e-3 for Ti/Te <= 0.1
    assert peak / (k * c_s) == pytest.approx(1.0, abs=3e-3), f"A2: ion-acoustic peak at {peak / (k * c_s):.5f} k c_s"
