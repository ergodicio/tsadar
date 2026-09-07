"""Physical normalization and marginalization contracts for ARTS2D EDFs."""

import equinox as eqx
import numpy as np
import pytest
from jax import config, grad, numpy as jnp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import gamma

config.update("jax_enable_x64", True)

from tsadar.core.modules.distribution_functions.base import Arbitrary2V
from tsadar.core.modules.distribution_functions.spherical_harmonics import SphericalHarmonics
from tsadar.core.physics.form_factor import FormFactor


def _harmonic_model(shape, *, nvx=96, nvr=96, nvz=64, dtx=0.0, dty=0.0):
    return SphericalHarmonics(
        {
            "active": True,
            "dim": 2,
            "type": "sphericalharmonic",
            "nvx": nvx,
            "params": {
                "flm_type": "mora-yahi",
                "init_m": shape,
                "dtx": dtx,
                "dty": dty,
                "Nl": 1,
                "nvr": nvr,
                "nvz": nvz,
            },
        }
    )


def _native_marginal(shape, nvx=128):
    return Arbitrary2V(
        {
            "active": True,
            "dim": 2,
            "type": "arbitrary",
            "nvx": nvx,
            "params": {
                "flm_type": "dlm",
                "init_m": shape,
                "learn_log": True,
            },
        }
    )


def _isotropic_3d_constants(shape):
    v0 = np.sqrt(3.0 * gamma(3.0 / shape) / gamma(5.0 / shape))
    normalization = shape / (4.0 * np.pi * v0**3 * gamma(3.0 / shape))
    return v0, normalization


def test_maxwellian_3d_marginal_is_analytic_2d_maxwellian():
    model = _harmonic_model(2.0)
    marginal = np.asarray(model())
    vx, vy = np.meshgrid(np.asarray(model.vx), np.asarray(model.vx))
    analytic = np.exp(-0.5 * (vx**2 + vy**2)) / (2.0 * np.pi)

    assert float(model.get_unnormed_m()) == 2.0
    np.testing.assert_allclose(marginal, analytic, rtol=5e-9, atol=7e-10)


@pytest.mark.parametrize("shape, inward_direction", [(2.0, 1.0), (5.0, -1.0)])
def test_endpoint_shape_initializers_have_nonzero_marginal_gradient(shape, inward_direction):
    model = _harmonic_model(shape, nvx=48, nvr=48, nvz=48)
    vx, vy = jnp.meshgrid(model.vx, model.vx)
    dv = model.vx[1] - model.vx[0]
    fourth_moment_weight = vx**4 + vy**4

    def marginal_observable(normed_m):
        changed = eqx.tree_at(lambda tree: tree.normed_m, model, replace=normed_m)
        return jnp.sum(changed() * fourth_moment_weight) * dv**2

    initial_parameter = model.normed_m
    ad_gradient = grad(marginal_observable)(initial_parameter)
    step = jnp.asarray(1.0e-4)
    fd_gradient = (
        marginal_observable(initial_parameter + step)
        - marginal_observable(initial_parameter - step)
    ) / (2.0 * step)

    assert float(model.get_unnormed_m()) == shape
    assert bool(jnp.isfinite(ad_gradient))
    assert float(jnp.abs(ad_gradient)) > 1.0e-4
    np.testing.assert_allclose(ad_gradient, fd_gradient, rtol=2e-4, atol=1e-8)

    inward_model = eqx.tree_at(
        lambda tree: tree.normed_m,
        model,
        replace=initial_parameter + inward_direction * step,
    )
    assert inward_direction * (float(inward_model.get_unnormed_m()) - shape) > 0.0


@pytest.mark.parametrize("shape, relative_tolerance", [(3.0, 4e-6), (5.0, 5e-8)])
def test_nonmaxwellian_3d_marginal_matches_adaptive_integration(shape, relative_tolerance):
    model = _harmonic_model(shape)
    marginal = np.asarray(model())
    vx = np.asarray(model.vx)
    v0, normalization = _isotropic_3d_constants(shape)

    for iy, ix in ((48, 48), (48, 60), (60, 60), (35, 55)):
        radius_squared = vx[ix] ** 2 + vx[iy] ** 2
        reference = quad(
            lambda vz: normalization
            * np.exp(-((np.sqrt(radius_squared + vz**2) / v0) ** shape)),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        np.testing.assert_allclose(
            marginal[iy, ix], reference, rtol=relative_tolerance, atol=1e-13
        )


def test_anisotropic_marginal_matches_refined_quadrature_and_stays_smoothly_positive():
    model = _harmonic_model(3.0, nvx=64, nvz=128, dtx=4e-4, dty=-7e-4)
    marginal = model()
    f3 = model.get_3d_distribution()
    dv = model.vx[1] - model.vx[0]

    assert bool(jnp.all(jnp.isfinite(f3)))
    assert bool(jnp.all(f3 >= 0))
    np.testing.assert_allclose(jnp.sum(marginal) * dv**2, 1.0, rtol=0, atol=2e-15)

    # Compare the production marginal quadrature with a separately refined rule.
    nodes, weights = leggauss(384)
    refined_vz = jnp.asarray(6.0 * nodes)
    refined_weights = jnp.asarray(6.0 * weights)
    raw_f3 = model._quadrature_3d()
    normalization = jnp.sum(raw_f3 * model.vz_weights[:, None, None]) * dv**2
    for iy, ix in ((32, 32), (32, 42), (42, 42), (20, 37)):
        reference = jnp.sum(
            model._evaluate_3d_unnormalized(model.vx[ix], model.vx[iy], refined_vz)
            * refined_weights
        ) / normalization
        np.testing.assert_allclose(marginal[iy, ix], reference, rtol=6e-6, atol=1e-13)

    # Marginalizing before the in-plane projection is identical to projecting the
    # normalized 3-V hypothesis directly for a cardinal in-plane view.
    projected_from_marginal = jnp.sum(marginal, axis=0) * dv
    projected_from_3d = jnp.sum(
        f3 * model.vz_weights[:, None, None], axis=(0, 1)
    ) * dv
    np.testing.assert_allclose(projected_from_marginal, projected_from_3d, rtol=2e-15, atol=1e-15)

    def x_centroid(dtx):
        changed = eqx.tree_at(lambda tree: tree.flm[1][0].dt, model, replace=dtx)
        return changed.get_in_plane_moments()["mean_vx"]

    coefficient_gradient = grad(x_centroid)(model.flm[1][0].dt)
    assert bool(jnp.isfinite(coefficient_gradient))
    assert float(jnp.abs(coefficient_gradient)) > 1e-5


@pytest.mark.parametrize("shape", [2.0, 3.0, 5.0])
@pytest.mark.parametrize("factory", [_harmonic_model, _native_marginal])
def test_reduced_edf_normalization_and_temperature_moment_do_not_drift(factory, shape):
    model = factory(shape, nvx=128)
    moments = model.get_in_plane_moments()

    np.testing.assert_allclose(moments["density"], 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(moments["mean_vx"], 0.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(moments["mean_vy"], 0.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        moments["thermal_second_moment"], 2.0, rtol=0, atol=2e-6
    )


def test_every_in_plane_projection_matches_direct_3d_integration():
    model = _harmonic_model(3.0, nvx=128)
    marginal = model()
    vx = np.asarray(model.vx)
    dv = float(model.vx[1] - model.vx[0])
    shape = float(model.get_unnormed_m())
    v0, normalization = _isotropic_3d_constants(shape)

    # Directly integrate the declared isotropic 3-V hypothesis over both velocities
    # perpendicular to k. Rotational symmetry makes this reference valid for every
    # in-plane angle while still testing the numerical Radon path used by susceptibility.
    direct_projection = np.asarray(
        [
            2.0
            * np.pi
            * quad(
                lambda radius: normalization
                * radius
                * np.exp(-((np.sqrt(u**2 + radius**2) / v0) ** shape)),
                0.0,
                np.inf,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
            for u in vx
        ]
    )
    form_factor = FormFactor(
        lambda_range=[500.0, 550.0],
        npts=5,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([30.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=0.0,
        va_ang={"ion-1": 0.0},
        calc_gain={"calc": False},
        n_beta=0,
    )

    for angle in (0.0, 0.37, 1.1):
        projected = np.asarray(form_factor.project(model.vx, marginal, angle))
        relative_l1 = np.sum(np.abs(projected - direct_projection)) * dv
        relative_l1 /= np.sum(np.abs(direct_projection)) * dv
        assert relative_l1 < 3e-6
        np.testing.assert_allclose(np.sum(projected) * dv, 1.0, rtol=0, atol=1e-8)
