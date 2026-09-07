"""Tests for the tabulated-projection (sinogram) path in the 2D form factor.

``calc_chi_vals`` used to rotate the whole 2D EDF at every evaluation point and then
immediately discard all but the 1D projection. It now tabulates that projection over a
fixed grid of angles and interpolates. These tests pin the tabulated path against the
exact per-point rotation it replaced, which is still reachable via ``n_beta=0``.

The angular integration tests are no help here: they skip when no GPU is visible, and
they cannot run even with one because the config files they open were deleted in #95. So
these drive the kernel directly, plus one end-to-end pass through `calc_in_2D`, sized to
run anywhere.

Deliberately *not* pinned to CPU -- these should exercise whatever backend is present, so
that a GPU-only discrepancy (see the `gav_safe` note in ratintn.py) has somewhere to
surface.
"""

import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp, grad

from tsadar.core.physics.form_factor import FormFactor


NVX = 128


def _form_factor(n_beta, npts=64, sa=None):
    """A FormFactor built only far enough to exercise the susceptibility kernel."""
    sa = np.array([60.0]) if sa is None else sa
    return FormFactor(
        lambda_range=[400.0, 700.0],
        npts=npts,
        lam_shift=0.0,
        scattering_angles={"sa": sa, "weights": np.ones((1, sa.size))},
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
    """Refining the angle grid must reduce the error faster than linear interpolation would.

    Linear interpolation in the angle already converges O(h**2) in *value*, which over this
    8x refinement gives ~64 -- so a bound of 10 would not have distinguished it from the
    cubic this uses. The bound is set above that crossover deliberately; measured ~140.
    """
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(32, seed=1)

    exact = _form_factor(0)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)
    errs = [
        _rel_err(_form_factor(n)._calc_all_chi_vals_(vx, df, beta, xie_mag, klde_mag)[0], exact[0])
        for n in (128, 256, 512, 1024)
    ]
    assert all(b < a for a, b in zip(errs, errs[1:])), f"not monotone: {errs}"
    assert errs[0] / errs[-1] > 60.0, f"convergence consistent with linear interpolation: {errs}"


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


def _params_2d(vx, df, nvx):
    """The parameter tree `calc_in_2D` reads, built directly rather than from a deck."""
    return {
        "electron": {"ne": 0.2, "Te": 0.6, "fe": df, "v": vx},
        "general": {"ne_gradient": 0.0, "Te_gradient": 0.0, "lam": 526.5, "ud": 0.0},
        "ion-1": {"A": 40.0, "Z": 18.0, "Ti": 0.12, "fract": 1.0, "Va": 0.0},
    }


def test_calc_in_2D_matches_exact_rotation():
    """End-to-end guard on the real 2D entrypoint, not just the kernel.

    Every other test here calls `_calc_all_chi_vals_` with hand-made angles. This one goes
    through `calc_in_2D`, so it covers how `beta` is actually built from `k`, the signed
    resonance coordinate, and the shape plumbing around the susceptibilities -- none of
    which the direct-kernel tests touch. The ATS integration tests would have covered it,
    but they are unrunnable (see module docstring).
    """
    nvx = 128  # what the shipped 2D decks use; the error is nvx-sensitive
    vx = jnp.linspace(-8.0, 8.0, nvx)
    VX, VY = jnp.meshgrid(vx, vx)
    df = jnp.exp(-(VX**2 + 0.4 * VY**2) / 2.0) * (1.0 + 0.3 * jnp.tanh(VX))
    params = _params_2d(vx, df, nvx)
    sa = np.linspace(50.0, 70.0, 6)

    exact, lams_exact = _form_factor(0, npts=16, sa=sa).calc_in_2D(params)
    got, lams = _form_factor(1024, npts=16, sa=sa).calc_in_2D(params)

    assert got.shape == exact.shape
    np.testing.assert_allclose(np.asarray(lams), np.asarray(lams_exact), rtol=0, atol=0)
    assert jnp.all(jnp.isfinite(got)), "non-finite values in the 2D spectrum"
    # Looser than the kernel-level tests: this normalizes by the peak of a spectrum
    # spanning several decades, and the EDF is deliberately asymmetric, so the angular
    # structure is harder to interpolate than a super-Gaussian. Measured 4.0e-05.
    assert _rel_err(got, exact) < 1e-4


def _loss(ff, vx, df, beta, xi, klde_mag):
    out = ff._calc_all_chi_vals_(vx, df, beta, xi, klde_mag)
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
    """Refining the angle grid must improve the gradient, not just the values.

    This is the test that actually pins cubic-over-linear: linear converges O(h) in the
    gradient, landing exactly on 4.0 over this 4x refinement, against a measured ~27.
    """
    vx, df = _distribution()
    beta, xie_mag, klde_mag = _inputs(64, seed=5)

    exact = grad(lambda d: _loss(_form_factor(0), vx, d, beta, xie_mag, klde_mag))(df)
    errs = []
    for n in (256, 1024):
        got = grad(lambda d: _loss(_form_factor(n), vx, d, beta, xie_mag, klde_mag))(df)
        errs.append(float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact)))
    assert errs[1] < errs[0] / 10.0, f"convergence consistent with linear interpolation: {errs}"


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


@pytest.mark.parametrize("n_beta", [1, 2, 3])
def test_n_beta_too_short_for_stencil_is_rejected(n_beta):
    """A grid shorter than the 4-wide stencil wraps onto itself; fail loudly, not quietly."""
    with pytest.raises(ValueError, match="at least 4"):
        _form_factor(n_beta)


@pytest.mark.parametrize("n_beta", [4, 10, 31, 33, 100])
def test_n_beta_need_not_divide_the_batch_size(n_beta):
    """`_build_sinogram` batches by BETA_BATCH_SIZE; a ragged final batch must still work."""
    vx, df = _distribution()
    proj, dproj = _form_factor(n_beta)._build_sinogram(vx, df)
    assert proj.shape == (n_beta, vx.size)
    assert dproj.shape == (n_beta, vx.size)
    assert bool(jnp.all(jnp.isfinite(proj))) and bool(jnp.all(jnp.isfinite(dproj)))


def test_projection_matches_rotate_then_sum():
    """`project` maps mathematical beta onto the legacy image-rotation sign convention."""
    vx, df = _distribution()
    ff = _form_factor(0)

    beta = jnp.array(1.234)
    expected = jnp.sum(ff.rotate(vx, df, -beta * 180 / jnp.pi, reshape=False), axis=0) * (vx[1] - vx[0])

    np.testing.assert_allclose(
        np.asarray(ff.project(vx, df, beta)), np.asarray(expected), rtol=1e-12, atol=1e-14
    )
