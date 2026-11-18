"""The scattering parameters reported with every result, and the validity flag built on them."""

import numpy as np
import pytest

from .. import oracles
from ..oracles import Ion, Plasma
from .cases import SPECTRUM_CASES
from .harness import MAXWELLIAN

from tsadar.core.physics.validity import scattering_parameters, validity_message

pytestmark = pytest.mark.physics


def _scattering(plasma):
    ions = {f"ion-{i + 1}": (ion.Z, ion.Ti) for i, ion in enumerate(plasma.ions)}
    return scattering_parameters(plasma.ne, plasma.Te, ions, plasma.lam, [plasma.angle_deg])


def _numeric_species(plasma):
    return [f"ion-{i + 1}" for i, ion in enumerate(plasma.ions) if ion.components != MAXWELLIAN]


@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_alpha_is_the_oracle_value_and_every_physics_case_is_within_the_validity_limits(case):
    plasma = case.plasma
    scattering = _scattering(plasma)
    k, _, _ = oracles.scattering_geometry(plasma.lam, plasma)
    alpha = oracles.plasma_frequency(plasma.ne) / (k * np.sqrt(plasma.Te / oracles.ELECTRON_MASS))
    np.testing.assert_allclose(scattering["alpha"], alpha, rtol=1e-3)
    for i, ion in enumerate(plasma.ions):
        beta2 = ion.Z * plasma.Te / ion.Ti * alpha**2 / (1 + alpha**2)
        np.testing.assert_allclose(scattering[f"beta_ion-{i + 1}"], np.sqrt(beta2), rtol=1e-3)
    assert validity_message(scattering, _numeric_species(plasma)) is None


@pytest.mark.parametrize(
    "plasma, numeric, flagged",
    [
        (Plasma(ne=0.2, Te=0.6, ions=(Ion(1.0, 1.0, 0.12),), angle_deg=25.0), [], "alpha"),
        (Plasma(ne=0.5, Te=1.0, ions=(Ion(A=197.0, Z=40.0, Ti=0.25),)), [], None),
        (Plasma(ne=0.5, Te=1.0, ions=(Ion(A=197.0, Z=40.0, Ti=0.25),)), ["ion-1"], "beta_ion-1"),
    ],
    ids=["epw-alpha-4.8", "high-z-maxwellian", "high-z-numeric"],
)
def test_the_flag_marks_the_measured_breakdown(plasma, numeric, flagged):
    """The limits of the measured breakdown (Test suite page)."""
    message = validity_message(_scattering(plasma), numeric)
    if flagged is None:
        assert message is None
    else:
        assert message is not None and flagged in message
