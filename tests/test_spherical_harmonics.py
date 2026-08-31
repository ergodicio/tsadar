"""Regression tests for the trainable ARTS2D harmonic representation."""

from io import BytesIO

import equinox as eqx
import numpy as np
import pytest
from jax import config, grad, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.modules.distribution_functions.spherical_harmonics import SphericalHarmonics
from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.core.physics.form_factor import FormFactor


def _distribution_config(flm_type="mora-yahi", nl=1, nvx=48, nvr=48):
    return {
        "active": True,
        "dim": 2,
        "type": "sphericalharmonic",
        "nvx": nvx,
        "params": {
            "flm_type": flm_type,
            "init_m": 2.2,
            "dtx": 0.0,
            "dty": 0.0,
            "Nl": nl,
            "nvr": nvr,
        },
    }


def _parameter_config(flm_type="mora-yahi", nl=1):
    def bounded(value, lower, upper, active=False):
        return {"active": active, "lb": lower, "ub": upper, "val": value}

    return {
        "electron": {
            "Te": bounded(0.6, 0.01, 1.5),
            "ne": bounded(0.2, 0.001, 1.0),
            "fe": _distribution_config(flm_type=flm_type, nl=nl, nvx=32, nvr=24),
        },
        "general": {
            "lam": bounded(526.5, 525.0, 528.0),
            "amp1": bounded(1.0, 0.01, 3.75),
            "amp2": bounded(1.0, 0.01, 3.75),
            "amp3": bounded(1.0, 0.01, 3.75),
            "ne_gradient": bounded(0.0, 0.0, 15.0),
            "Te_gradient": bounded(0.0, 0.0, 10.0),
            "ud": bounded(0.0, -100.0, 100.0),
        },
        "ion-1": {
            "A": {"active": False, "val": 40.0},
            "Ti": bounded(0.12, 0.001, 1.0),
            "Va": bounded(0.0, -20.5, 20.5),
            "Z": bounded(14.0, 0.5, 20.0),
            "fract": {"active": False, "val": 1.0},
        },
    }


def _form_factor():
    return FormFactor(
        lambda_range=[500.0, 550.0],
        npts=9,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([35.0, 70.0]), "weights": np.ones((1, 2))},
        num_grad_points=1,
        ud_ang=0.0,
        va_ang={"ion-1": 0.0},
        calc_gain={"calc": False},
        n_beta=0,
    )


def _spectrum(model):
    params = {
        "electron": {"ne": 0.2, "Te": 0.6, "fe": model(), "v": model.vx},
        "general": {"ne_gradient": 0.0, "Te_gradient": 0.0, "lam": 526.5, "ud": 0.0},
        "ion-1": {"A": 40.0, "Z": 14.0, "Ti": 0.12, "fract": 1.0, "Va": 0.0},
    }
    return _form_factor().calc_in_2D(params)[0]


@pytest.mark.parametrize("nvx", [48, 49])
def test_l1_real_harmonics_are_independent_cartesian_modes(nvx):
    model = SphericalHarmonics(_distribution_config(nvx=nvx))
    vx, vy = jnp.meshgrid(model.vx, model.vx)
    x_mode = model._real_harmonic(1, 0)
    y_mode = model._real_harmonic(1, 1)

    scale = float(jnp.max(jnp.abs(x_mode)))
    np.testing.assert_allclose(x_mode, -jnp.flip(x_mode, axis=1), atol=2e-14 * scale)
    np.testing.assert_allclose(x_mode, jnp.flip(x_mode, axis=0), atol=2e-14 * scale)
    np.testing.assert_allclose(y_mode, jnp.flip(y_mode, axis=1), atol=2e-14 * scale)
    np.testing.assert_allclose(y_mode, -jnp.flip(y_mode, axis=0), atol=2e-14 * scale)

    assert float(jnp.sum(vx * x_mode)) > 0
    assert float(jnp.sum(vy * y_mode)) > 0
    np.testing.assert_allclose(jnp.sum(vy * x_mode), 0.0, atol=2e-13)
    np.testing.assert_allclose(jnp.sum(vx * y_mode), 0.0, atol=2e-13)
    np.testing.assert_allclose(jnp.sum(x_mode * y_mode), 0.0, atol=2e-13)

    if nvx % 2:
        center = nvx // 2
        assert x_mode[center, center] == 0.0
        assert y_mode[center, center] == 0.0


@pytest.mark.parametrize("mode, odd_axis, even_axis", [((1, 0), 1, 0), ((1, 1), 0, 1)])
def test_each_mora_yahi_coefficient_perturbs_edf_spectrum_and_gradient(mode, odd_axis, even_axis):
    model = SphericalHarmonics(_distribution_config())
    coefficient = lambda tree: tree.flm[mode[0]][mode[1]].dt

    def with_coefficient(value):
        return eqx.tree_at(coefficient, model, replace=value)

    base_edf = model()
    perturbed_edf = with_coefficient(jnp.asarray(1.0e-4))()
    delta_edf = perturbed_edf - base_edf
    delta_scale = jnp.max(jnp.abs(delta_edf))

    assert bool(jnp.all(jnp.isfinite(delta_edf)))
    assert float(jnp.linalg.norm(delta_edf)) > 0
    np.testing.assert_allclose(
        delta_edf,
        -jnp.flip(delta_edf, axis=odd_axis),
        rtol=1e-6,
        atol=float(delta_scale) * 1e-6,
    )
    np.testing.assert_allclose(
        delta_edf,
        jnp.flip(delta_edf, axis=even_axis),
        rtol=1e-6,
        atol=float(delta_scale) * 1e-6,
    )

    base_spectrum = _spectrum(model)
    perturbed_spectrum = _spectrum(with_coefficient(jnp.asarray(1.0e-4)))
    assert bool(jnp.all(jnp.isfinite(perturbed_spectrum)))
    assert float(jnp.linalg.norm(perturbed_spectrum - base_spectrum)) > 1e-8

    weights = jnp.linspace(0.7, 1.3, base_spectrum.size).reshape(base_spectrum.shape)

    def spectral_loss(value):
        return jnp.sum(_spectrum(with_coefficient(value)) * weights)

    ad_gradient = grad(spectral_loss)(jnp.asarray(0.0))
    step = jnp.asarray(1.0e-5)
    fd_gradient = (spectral_loss(step) - spectral_loss(-step)) / (2 * step)
    assert bool(jnp.isfinite(ad_gradient))
    assert float(jnp.abs(ad_gradient)) > 1e-8
    np.testing.assert_allclose(ad_gradient, fd_gradient, rtol=1e-6, atol=1e-9)


def test_active_mora_yahi_leaves_are_differentiable_and_reported():
    config_params = _parameter_config()
    params = ThomsonParams(config_params, num_params=1, batch=False, activate=False)
    spec = get_filter_spec(config_params, params)

    distribution = params.electron.distribution_functions
    distribution_spec = spec.electron.distribution_functions
    assert distribution.flm[1][0].dt.shape == ()
    assert distribution.flm[1][1].dt.shape == ()
    assert distribution_spec.normed_m is True
    assert distribution_spec.flm[1][0].dt is True
    assert distribution_spec.flm[1][1].dt is True

    diff_params, static_params = eqx.partition(params, spec)
    vx, vy = jnp.meshgrid(distribution.vx, distribution.vx)
    observable_weight = 0.1 * (vx**2 + vy**2) + 0.2 * vx + 0.3 * vy

    def observable(differentiable):
        combined = eqx.combine(differentiable, static_params)
        return jnp.sum(combined.electron.distribution_functions() * observable_weight)

    _, gradients = eqx.filter_value_and_grad(observable)(diff_params)
    distribution_grad = gradients.electron.distribution_functions
    for leaf in (
        distribution_grad.normed_m,
        distribution_grad.flm[1][0].dt,
        distribution_grad.flm[1][1].dt,
    ):
        assert leaf is not None
        assert bool(jnp.isfinite(leaf).all())
        assert float(jnp.linalg.norm(leaf)) > 0

    accessors = (
        lambda tree: tree.electron.distribution_functions.normed_m,
        lambda tree: tree.electron.distribution_functions.flm[1][0].dt,
        lambda tree: tree.electron.distribution_functions.flm[1][1].dt,
    )
    for accessor in accessors:
        initial_value = accessor(params)

        def leaf_observable(value):
            changed = eqx.tree_at(accessor, params, replace=value)
            return jnp.sum(changed.electron.distribution_functions() * observable_weight)

        ad_gradient = grad(leaf_observable)(initial_value)
        step = jnp.asarray(1.0e-5)
        fd_gradient = (
            leaf_observable(initial_value + step) - leaf_observable(initial_value - step)
        ) / (2 * step)
        assert bool(jnp.isfinite(ad_gradient))
        assert float(jnp.abs(ad_gradient)) > 1e-8
        np.testing.assert_allclose(ad_gradient, fd_gradient, rtol=2e-6, atol=1e-8)

    unnormalized = params.get_unnormed_params()["electron"]
    assert set(unnormalized["flm"]) == {0, 1}
    assert set(unnormalized["flm"][1]) == {0, 1}
    assert bool(jnp.isfinite(unnormalized["m"]))
    fitted, _ = params.get_fitted_params(config_params)
    assert "m" in fitted["electron"]
    assert "flm" in fitted["electron"]


def test_nn_filter_includes_every_weight_and_bias_for_nl_greater_than_one():
    config_params = _parameter_config(flm_type="nn", nl=2)
    params = ThomsonParams(config_params, num_params=1, batch=False, activate=False)
    spec = get_filter_spec(config_params, params)

    assert spec.electron.distribution_functions.normed_m is True
    for degree in (1, 2):
        for order in range(degree + 1):
            mode_spec = spec.electron.distribution_functions.flm[degree][order]
            for network_name in ("flm_mag", "flm_sign"):
                for layer in getattr(mode_spec, network_name).layers:
                    assert layer.weight is True
                    assert layer.bias is True


def test_nl_greater_than_one_round_trips_state_and_unnormalized_coefficients():
    model = SphericalHarmonics(_distribution_config(flm_type="arbitrary", nl=2))
    unnormalized = model.get_unnormed_params()
    assert set(unnormalized["flm"]) == {0, 1, 2}
    assert set(unnormalized["flm"][2]) == {0, 1, 2}
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in unnormalized["flm"][2].values())

    state = BytesIO()
    eqx.tree_serialise_leaves(state, model)
    state.seek(0)
    restored = eqx.tree_deserialise_leaves(state, model)
    np.testing.assert_allclose(restored(), model(), rtol=0, atol=0)
