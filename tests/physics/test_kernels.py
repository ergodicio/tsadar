"""L1: tsadar's susceptibility kernel and Z' against the analytic ones."""

import numpy as np
import pytest
from jax import numpy as jnp

from tsverify import oracle as oracles
from tsverify.cases import SPECTRUM_CASES

from . import harness

pytestmark = pytest.mark.physics

CHI_TOL = 4.0e-4
F_TOL = 1.0e-6
JUMP_TOL = 1.0e-5
TAIL_TOL = 3.0e-2
MAXWELLIAN_TAIL_TOL = 2.0e-4
XI = np.linspace(-7.0, 7.0, 2801)


def _edfs():
    """Every distinct EDF among the spectrum cases, electrons and ions."""
    seen = {}
    for case in SPECTRUM_CASES:
        seen.setdefault(case.plasma.electrons, f"{case}/electrons")
        for i, ion in enumerate(case.plasma.ions):
            seen.setdefault(ion.components, f"{case}/ion-{i + 1}")
    return [pytest.param(components, id=label) for components, label in seen.items()]


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


def _kernel_chi(form_factor, components, xi):
    v = harness.NUMERIC_V
    chi, f_at_xi = form_factor._numeric_chi(
        jnp.asarray(oracles.mixture_density(v, components)),
        jnp.asarray(v),
        jnp.asarray(xi)[None, :, None, None],
        jnp.ones((1, 1, 1, 1)),
    )
    return np.squeeze(np.asarray(chi)), np.squeeze(np.asarray(f_at_xi))


@pytest.mark.parametrize("components", _edfs())
def test_numeric_susceptibility_matches_the_analytic_one(form_factor, components):
    chi, f_at_xi = _kernel_chi(form_factor, components, XI)
    expected = oracles.mixture_susceptibility(XI, 1.0, components)

    scale = np.abs(expected).max()
    for part, actual, reference in (("Re", chi.real, expected.real), ("Im", chi.imag, expected.imag)):
        error = np.abs(actual - reference) / scale
        assert error.max() < CHI_TOL, (
            f"A3-L1: {part} chi off by {error.max():.2e} of max|chi| at xi={XI[error.argmax()]:.3f} "
            f"(tol {CHI_TOL:.1e})"
        )
    f_error = np.max(np.abs(f_at_xi - oracles.mixture_density(XI, components)))
    assert f_error < F_TOL, f"A3-L1: f(xi) interpolated with error {f_error:.2e}"


DRIFTING_BEAM = (oracles.GaussianComponent(0.9, 0.0, 1.0), oracles.GaussianComponent(0.1, 6.0, 0.25))


@pytest.mark.parametrize("components", _edfs() + [pytest.param(DRIFTING_BEAM, id="beam-at-6-vT")])
def test_numeric_susceptibility_continues_past_the_table(form_factor, components):
    """Continuous at both table edges; past them within TAIL_TOL of the analytic response (not checked for the
    beam, whose structure near the edge a moment series cannot follow)."""
    lo, hi = float(form_factor.xi2[0]), float(form_factor.xi2[-1])
    xi = np.array([lo - 1e-6, lo + 1e-6, hi - 1e-6, hi + 1e-6])
    chi, _ = _kernel_chi(form_factor, components, xi)
    for inside, outside in ((1, 0), (2, 3)):
        jump = abs(chi.real[outside] - chi.real[inside]) / abs(chi.real[inside])
        assert jump < JUMP_TOL, f"Re chi jumps by {jump:.1e} at the table edge xi={xi[inside]:.2f}"
    if components == DRIFTING_BEAM:
        return
    far = np.array([lo - 30.0, lo - 4.0, lo - 0.5, hi + 0.5, hi + 4.0, hi + 30.0])
    chi, _ = _kernel_chi(form_factor, components, far)
    expected = oracles.mixture_susceptibility(far, 1.0, components)
    error = np.abs(chi.real - expected.real) / np.abs(expected.real)
    tol = MAXWELLIAN_TAIL_TOL if components == harness.MAXWELLIAN else TAIL_TOL
    assert error.max() < tol, f"Re chi past the table off by {error.max():.1e} at xi={far[error.argmax()]:.1f}"


@pytest.mark.parametrize("source, tol", [("faddeeva", 1e-5), ("legacy", None)])
def test_zprime_tail_past_the_table(source, tol):
    from scipy.special import wofz

    from tsadar.core.physics.form_factor import _zprime_tail

    zeta = np.array([8.2, 9.0, 11.6, 14.0, 20.0, 50.0])
    tail = np.asarray(_zprime_tail(zeta, source))
    if source == "legacy":
        np.testing.assert_array_equal(tail, zeta**-2)
        return
    exact = (-2.0 * (1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta))).real
    np.testing.assert_allclose(tail, exact, rtol=tol, atol=0)


def test_zprime_default_is_exact_and_legacy_reproduces_the_old_table():
    from scipy.interpolate import interp1d
    from scipy.special import dawsn

    from tsadar.core.physics.form_factor import _load_zprime_tables, zprimeMaxw

    xi = np.arange(-8.2, 8.2, 0.01)
    exact_real = -2.0 * (1.0 - 2.0 * xi * dawsn(xi))
    exact_imag = -2.0 * np.sqrt(np.pi) * xi * np.exp(-(xi**2))
    default = zprimeMaxw(xi)
    np.testing.assert_allclose(default[0], exact_real, rtol=0, atol=1e-13, err_msg="Re Z' (faddeeva)")
    np.testing.assert_allclose(default[1], exact_imag, rtol=0, atol=1e-13, err_msg="Im Z' (faddeeva)")

    rd, idt = _load_zprime_tables()
    legacy = zprimeMaxw(xi, "legacy")
    np.testing.assert_array_equal(legacy[0], interp1d(rd[:, 0], rd[:, 1], "linear")(xi))
    np.testing.assert_array_equal(legacy[1], interp1d(idt[:, 0], idt[:, 1], "linear")(xi))
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
