"""L1: tsadar's susceptibility kernel against the analytic Gaussian-mixture susceptibility.

``FormFactor._numeric_chi`` turns a sampled EDF into ``chi(xi)``: the principal value
through the rational-integral operator on the internal velocity grid, and the Landau
term from df/dv on that grid. The oracle is the exact sum of shifted, scaled Faddeeva
terms. Comparing chi directly removes the conditioning of the spectrum (S ~ 1/|eps|^2),
so one tolerance holds for every EDF; the spectrum depths inherit whatever this
kernel error becomes for each case. For a kappa EDF (A5) the oracle's chi comes from its own
adaptive quadrature instead of the Faddeeva function.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from .. import oracles
from . import harness
from .cases import SPECTRUM_CASES

pytestmark = pytest.mark.physics

# Error relative to max|chi| (k lambda_D = 1), set by the 1024-point internal velocity grid. Measured
# 1.1e-4 to 1.4e-4 for the Gaussian mixtures, 2.0e-4 / 2.9e-4 for kappa 5 / 3, and 2.9e-4 for the two-humped
# ions (it falls with a 4096-point grid).
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
def test_numeric_susceptibility_matches_the_analytic_one(form_factor, components, ion):
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


# --------------------------------------------------------------------------- Z' for Maxwellian ions


def test_zprime_default_is_exact_and_legacy_reproduces_the_old_table():
    from scipy.interpolate import interp1d
    from scipy.special import dawsn

    from tsadar.core.physics.form_factor import _load_zprime_tables, zprimeMaxw

    xi = np.arange(-8.2, 8.2, 0.01)  # FormFactor's xi2 grid
    # Re Z' = -2 (1 - 2 xi F(xi)) with the Dawson function F, an evaluation independent of the Faddeeva function
    exact_real = -2.0 * (1.0 - 2.0 * xi * dawsn(xi))
    exact_imag = -2.0 * np.sqrt(np.pi) * xi * np.exp(-(xi**2))
    default = zprimeMaxw(xi)
    np.testing.assert_allclose(default[0], exact_real, rtol=0, atol=1e-13, err_msg="Re Z' (faddeeva)")
    np.testing.assert_allclose(default[1], exact_imag, rtol=0, atol=1e-13, err_msg="Im Z' (faddeeva)")

    # legacy: the tables, interpolated exactly as before
    rd, idt = _load_zprime_tables()
    legacy = zprimeMaxw(xi, "legacy")
    np.testing.assert_array_equal(legacy[0], interp1d(rd[:, 0], rd[:, 1], "linear")(xi))
    np.testing.assert_array_equal(legacy[1], interp1d(idt[:, 0], idt[:, 1], "linear")(xi))
    # and the known difference between them: up to 1.63e-3 in the real part, 5e-6 in the imaginary part
    assert 1.5e-3 < np.abs(legacy[0] - exact_real).max() < 1.7e-3
    assert np.abs(legacy[1] - exact_imag).max() < 1e-5

    with pytest.raises(ValueError, match="other.zprime"):
        zprimeMaxw(xi, "tabulated")


@pytest.mark.parametrize("source", ["faddeeva", "legacy"])
def test_the_deck_selects_the_zprime_source_for_both_form_factors(source):
    from tsadar.core.physics.form_factor import zprimeMaxw
    from tsadar.core.physics.generate_spectra import FitModel

    case = SPECTRUM_CASES[0]
    cfg = harness.diagnostic_config(case.plasma, case.detector)
    cfg["other"]["zprime"] = source
    model = FitModel(cfg, {"sa": np.array([60.0]), "weights": np.ones((1, 1))})
    for ff in (model.electron_form_factor, model.ion_form_factor):
        np.testing.assert_array_equal(np.asarray(ff.Zpi), zprimeMaxw(np.asarray(ff.xi2), source))
