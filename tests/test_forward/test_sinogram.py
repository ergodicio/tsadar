"""Tests for the tabulated-projection (sinogram) path in the 2D form factor.

``calc_chi_vals`` used to rotate the whole 2D EDF at every evaluation point and then
immediately discard all but the 1D projection. It now tabulates that projection over a
fixed grid of angles and interpolates. These tests pin the tabulated path against the
exact per-point rotation it replaced, which is still reachable via ``n_beta=0``.

The angular integration tests skip on CPU because a full ATS forward pass is too slow
without a GPU; these exercise the same kernel directly on a handful of points so the
comparison runs anywhere.
"""

import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from jax import numpy as jnp, grad

from tsadar.core.physics.form_factor import FormFactor


NVX = 128


def _form_factor(n_beta):
    """A FormFactor built only far enough to exercise the susceptibility kernel."""
    return FormFactor(
        lambda_range=[400.0, 700.0],
        npts=64,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([60.0]), "weights": np.array([[1.0]])},
        num_grad_points=1,
        ud_ang=0.0,
        va_ang=0.0,
        calc_gain={"calc": False, "Ipump": 0, "beam_diam_um": 0},
        n_beta=n_beta,
    )


def _distribution():
    """An asymmetric, anisotropic EDF -- a symmetric one would hide angular error."""
    vx = jnp.linspace(-8.0, 8.0, NVX)
    VX, VY = jnp.meshgrid(vx, vx)
    df = jnp.exp(-(VX**2 + 0.4 * VY**2) / 2.0) * (1.0 + 0.3 * jnp.tanh(VX))
    return vx, df


def _inputs(n_points, seed=0):
    rng = np.random.default_rng(seed)
    # beta as built by calc_in_2D lands in (-pi/2, 3*pi/2), not [0, 2*pi)
    beta = jnp.array(rng.uniform(-np.pi / 2, 3 * np.pi / 2, n_points))
    xie_mag = jnp.array(rng.uniform(0.0, 6.0, n_points))
    klde_mag = jnp.array(rng.uniform(0.2, 2.0, n_points))
    return beta, xie_mag, klde_mag


def _rel_err(got, ref):
    return float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))


# chiEI is the interpolated *derivative* of the projection, so it converges a order
# slower than the other two and sets these tolerances.
@pytest.mark.parametrize("n_beta, tol", [(256, 1e-4), (512, 3e-5), (1024, 5e-6)])
def test_sinogram_matches_exact_rotation(n_beta, tol):
    """The tabulated projection reproduces the per-point rotation to the stated tolerance."""
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(48)

    exact = _form_factor(0)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)
    tabulated = _form_factor(n_beta)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)

    for name, ref, got in zip(("fe_vphi", "chiEI", "chiERrat"), exact, tabulated):
        assert got.shape == ref.shape
        assert _rel_err(got, ref) < tol, f"{name} exceeded {tol} at n_beta={n_beta}"


def test_sinogram_converges_with_n_beta():
    """Refining the angle grid monotonically reduces the error against the exact path."""
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(32, seed=1)

    exact = _form_factor(0)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)
    errs = [
        _rel_err(_form_factor(n)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)[0], exact[0])
        for n in (128, 256, 512, 1024)
    ]
    assert all(b < a for a, b in zip(errs, errs[1:])), f"not monotone: {errs}"
    # linear interpolation in the angle -> error should fall roughly 4x per two doublings
    assert errs[0] / errs[-1] > 10.0, f"convergence too slow: {errs}"


def test_interp_beta_wraps_periodically():
    """The angle grid covers exactly one period, so the seam at 2*pi must be continuous."""
    vx, df = _distribution()
    ff = _form_factor(256)
    proj, _ = ff._build_sinogram(vx, df)

    eps = 1e-9
    np.testing.assert_allclose(
        np.asarray(ff._interp_beta(jnp.array(eps), proj)),
        np.asarray(ff._interp_beta(jnp.array(2 * np.pi - eps), proj)),
        rtol=1e-6,
        atol=1e-12,
    )
    # negative angles, which calc_in_2D does produce, fold onto the same grid
    np.testing.assert_allclose(
        np.asarray(ff._interp_beta(jnp.array(-np.pi / 3), proj)),
        np.asarray(ff._interp_beta(jnp.array(2 * np.pi - np.pi / 3), proj)),
        rtol=1e-9,
        atol=1e-12,
    )


def test_gradient_commutes_with_angular_interpolation():
    """`_build_sinogram` differentiates before interpolating; the per-point path does the
    reverse. Both are linear, so the two orders must agree to roundoff."""
    vx, df = _distribution()
    ff = _form_factor(256)
    proj, dproj = ff._build_sinogram(vx, df)

    beta = jnp.array(0.937)
    dproj_then_interp = ff._interp_beta(beta, dproj)
    interp_then_dproj = jnp.gradient(ff._interp_beta(beta, proj), vx[1] - vx[0])

    np.testing.assert_allclose(
        np.asarray(dproj_then_interp), np.asarray(interp_then_dproj), rtol=1e-10, atol=1e-14
    )


def _loss(ff, vx, df, beta, xie_mag, klde_mag):
    out = ff._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)
    return sum(jnp.sum(o**2) for o in out)


def test_gradient_wrt_angle_matches_exact():
    """The angle carries the dependence on ne/Te/ud/Va, so its gradient has to be right.

    This is why `_interp_beta` is cubic and not linear: a linear interpolant's derivative
    in the angle is piecewise constant, which lands ~1e-2 off here however fine the grid.
    """
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(64, seed=3)

    exact = grad(lambda b: _loss(_form_factor(0), vx, df, b, xie_mag, klde_mag))(beta)
    got = grad(lambda b: _loss(_form_factor(1024), vx, df, b, xie_mag, klde_mag))(beta)

    assert _rel_err(got, exact) < 1e-3


def test_gradient_wrt_distribution_matches_exact():
    """Gradients converge more slowly than values -- pin both the magnitude and direction.

    At n_beta=1024 the value error is ~1e-5 but the gradient is ~1e-2 in relative L2. The
    descent *direction* is far better than that (1 - cos ~ 1e-5), which is what actually
    governs the fit, but the magnitude gap is real and worth guarding.
    """
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(128, seed=3)

    exact = grad(lambda d: _loss(_form_factor(0), vx, d, beta, xie_mag, klde_mag))(df)
    got = grad(lambda d: _loss(_form_factor(1024), vx, d, beta, xie_mag, klde_mag))(df)

    rel_l2 = float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))
    cos = float(jnp.sum(got * exact) / (jnp.linalg.norm(got) * jnp.linalg.norm(exact)))
    assert rel_l2 < 2e-2, f"gradient relative L2 error {rel_l2:.2e}"
    assert 1.0 - cos < 1e-4, f"gradient direction off by 1 - cos = {1 - cos:.2e}"


def test_gradient_converges_with_n_beta():
    """Refining the angle grid must improve the gradient, not just the values."""
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(64, seed=5)

    exact = grad(lambda d: _loss(_form_factor(0), vx, d, beta, xie_mag, klde_mag))(df)
    errs = []
    for n in (256, 1024):
        got = grad(lambda d: _loss(_form_factor(n), vx, d, beta, xie_mag, klde_mag))(df)
        errs.append(float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact)))
    assert errs[1] < errs[0] / 4.0, f"expected at least 2nd-order convergence, got {errs}"


def test_interp_beta_v_matches_full_row_interp():
    """The scalar (angle, velocity) gather must agree with interpolating a whole row.

    Includes velocities off both ends of the grid: `jnp.interp` holds the edge value
    there, and the clamped index arithmetic has to do the same.
    """
    vx, df = _distribution()
    ff = _form_factor(256)
    proj, _ = ff._build_sinogram(vx, df)

    beta = jnp.array(2.1)
    row = ff._interp_beta(beta, proj)
    for v in (-50.0, -8.0, -3.3, 0.0, 4.7, 8.0, 50.0):
        np.testing.assert_allclose(
            float(ff._interp_beta_v(beta, vx, jnp.array(v), proj)),
            float(jnp.interp(jnp.array(v), vx, row)),
            rtol=1e-10,
            atol=1e-14,
            err_msg=f"mismatch at v={v}",
        )


def test_projection_matches_rotate_then_sum():
    """`project` is exactly the reduction it replaced, not an approximation of it."""
    vx, df = _distribution()
    ff = _form_factor(0)

    beta = jnp.array(1.234)
    expected = jnp.sum(ff.rotate(vx, df, beta * 180 / jnp.pi, reshape=False), axis=0) * (vx[1] - vx[0])

    np.testing.assert_allclose(
        np.asarray(ff.project(vx, df, beta)), np.asarray(expected), rtol=1e-12, atol=1e-14
    )
