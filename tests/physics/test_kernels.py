"""L1: tsadar's susceptibility kernel against the analytic Gaussian-mixture susceptibility.

``FormFactor._numeric_chi`` turns a sampled EDF into ``chi(xi)``: the principal value
through the rational-integral operator on the internal velocity grid, and the Landau
term from df/dv on that grid. The oracle is the exact sum of shifted, scaled Faddeeva
terms. Comparing chi directly removes the conditioning of the spectrum (S ~ 1/|eps|^2),
so one tolerance holds for every EDF; the spectrum depths inherit whatever this
kernel error becomes for each case.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from .. import oracles
from . import harness
from .cases import SPECTRUM_CASES

pytestmark = pytest.mark.physics

# Error relative to max|chi| (k lambda_D = 1), set by the 1024-point internal velocity grid. Measured
# 1.1e-4 to 1.4e-4 for every EDF except the two-humped ions, 2.9e-4 (it falls with a 4096-point grid).
# The sampled EDF itself is interpolated to 1e-7.
CHI_TOL = 4.0e-4
F_TOL = 1.0e-6
XI = np.linspace(-7.0, 7.0, 2801)


def _edfs():
    """Every distinct EDF among the spectrum cases, electrons and ions, with the kernel branch it takes."""
    seen = {}
    for case in SPECTRUM_CASES:
        seen.setdefault((case.plasma.electrons, False), f"{case}/electrons")
        for i, ion in enumerate(case.plasma.ions):
            seen.setdefault((ion.components, True), f"{case}/ion-{i + 1}")
    return [pytest.param(components, ion, id=label) for (components, ion), label in seen.items()]


@pytest.fixture(scope="module")
def form_factor():
    return harness.FormFactor(
        [440.0, 630.0],
        npts=256,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([60.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )


@pytest.mark.parametrize("components, ion", _edfs())
def test_numeric_susceptibility_matches_faddeeva_mixture(form_factor, components, ion):
    v = harness.NUMERIC_V
    chi, f_at_xi = form_factor._numeric_chi(
        jnp.asarray(oracles.mixture_density(v, components)),
        jnp.asarray(v),
        jnp.asarray(XI)[None, :, None, None],
        jnp.ones((1, 1, 1, 1)),
        ion=ion,
    )
    chi = np.squeeze(np.asarray(chi))
    expected = oracles.mixture_susceptibility(XI, 1.0, components)

    scale = np.abs(expected).max()
    for part, actual, reference in (("Re", chi.real, expected.real), ("Im", chi.imag, expected.imag)):
        error = np.abs(actual - reference) / scale
        assert error.max() < CHI_TOL, (
            f"A3-L1: {part} chi off by {error.max():.2e} of max|chi| at xi={XI[error.argmax()]:.3f} "
            f"(tol {CHI_TOL:.1e})"
        )
    f_error = np.max(np.abs(np.squeeze(np.asarray(f_at_xi)) - oracles.mixture_density(XI, components)))
    assert f_error < F_TOL, f"A3-L1: f(xi) interpolated with error {f_error:.2e}"
