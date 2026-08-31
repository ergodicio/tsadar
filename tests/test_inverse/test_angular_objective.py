"""Focused contracts for the noise-aware ARTS objective from issue #139."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tsadar.inverse.loss_function import LossFunction
from tsadar.core.instrument.irf import AngularIRF, add_ATS_IRF


def _objective(**sections):
    cfg = {
        "optimizer": {"loss_method": "l2", "moment_loss": False},
        "data": {
            "fit_EPWb": True,
            "fit_EPWr": False,
            "fit_rng": {"blue_min": 0.0, "blue_max": 5.0, "red_min": 6.0, "red_max": 10.0},
        },
        "other": {"extraoptions": {"spectype": "angular_full"}},
        "parameters": {"electron": {"fe": {"dim": 2}}},
    }
    loss = LossFunction.__new__(LossFunction)
    loss.cfg = cfg
    loss.is_angular = True
    loss.angular_objective = loss._validated_angular_objective(sections)
    return loss


def test_profiled_row_gain_is_independent_of_measured_peak_and_overall_scale():
    loss = _objective(
        noise={"model": "measured_variance"},
        gain={"mode": "per_row", "smoothness": 0.0},
    )
    signal = jnp.array([[1.0, 2.0, 1.5, 0.5], [0.5, 1.0, 3.0, 2.0]])
    background = jnp.full_like(signal, 0.25)
    lam_axis = jnp.arange(1.0, 5.0)

    for overall_scale in (1.0, 17.0):
        expected_gains = overall_scale * jnp.array([2.0, 0.75])
        batch = {
            "e_data": expected_gains[:, None] * signal + background,
            "noise_e": background,
            "e_variance": jnp.ones_like(signal),
            # A masked high point must not influence a row gain.
            "e_mask": jnp.array([[True, True, True, False], [True, True, True, True]]),
        }
        total, sqdev, fitted, _, diagnostics = loss._angular_data_objective(
            batch, signal + background, lam_axis
        )
        np.testing.assert_allclose(diagnostics["profiled_gains"], expected_gains, rtol=1e-6)
        np.testing.assert_allclose(fitted[batch["e_mask"]], batch["e_data"][batch["e_mask"]], rtol=1e-6)
        assert float(total) < 1e-10
        assert float(jnp.sum(sqdev)) < 1e-10


def test_measured_variance_whitens_residual_and_bad_pixel_mask_excludes_outlier():
    loss = _objective(
        noise={"model": "measured_variance"},
        gain={"mode": "none"},
    )
    theory = jnp.ones((1, 4))
    batch = {
        "e_data": jnp.array([[3.0, 101.0, 1.0, 1.0]]),
        "noise_e": jnp.zeros((1, 4)),
        "e_variance": jnp.full((1, 4), 4.0),
        "e_mask": jnp.array([[True, False, True, True]]),
    }
    total, _, _, terms, diagnostics = loss._angular_data_objective(batch, theory, jnp.arange(1.0, 5.0))
    assert diagnostics["whitened_residual"][0, 0] == pytest.approx(1.0)
    assert np.isnan(diagnostics["whitened_residual"][0, 1])
    # One unit-squared residual across the three valid pixels.
    assert terms["data"] == pytest.approx(1.0 / 3.0)
    assert total == pytest.approx(1.0 / 3.0)


def test_poisson_read_variance_uses_background_and_resolution_unit_averaging():
    loss = _objective(
        noise={
            "model": "poisson_read",
            "read_noise": 3.0,
            "excess_noise_factor": 2.0,
            "background_variance_scale": 4.0,
        }
    )
    loss.cfg["other"].update({"ang_res_unit": 2, "lam_res_unit": 5})
    batch = {
        "e_data": jnp.asarray([[12.0]]),
        "noise_e": jnp.asarray([[2.0]]),
    }

    # (read^2 + F * signal + background factor * background) / 10 averaged pixels
    assert loss._angular_variance(batch)[0, 0] == pytest.approx((9.0 + 20.0 + 8.0) / 10.0)


def test_profiled_objective_is_jittable_and_differentiable():
    loss = _objective(
        noise={"model": "constant", "read_noise": 2.0},
        gain={"mode": "per_row_wing", "smoothness": 0.1},
        robust={"kind": "huber", "iterations": 2},
    )
    theory = jnp.array([[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0]])
    batch = {
        "e_data": 1.5 * theory,
        "noise_e": jnp.zeros_like(theory),
        "e_mask": jnp.ones_like(theory, dtype=bool),
    }

    def objective(candidate):
        return loss._angular_data_objective(batch, candidate, jnp.arange(1.0, 5.0))[0]

    value, gradient = jax.jit(jax.value_and_grad(objective))(theory)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))


def test_profiled_gain_uncertainty_has_calibrated_synthetic_coverage():
    loss = _objective(
        noise={"model": "measured_variance"},
        gain={"mode": "global"},
    )
    signal = jnp.linspace(0.5, 2.0, 20)[None, :]
    variance = jnp.full_like(signal, 0.25)
    lam_axis = jnp.linspace(0.1, 4.9, signal.shape[1])
    true_gain = 2.0
    noise_samples = 0.5 * jax.random.normal(jax.random.PRNGKey(139), (400,) + signal.shape)

    def estimate(noise):
        batch = {
            "e_data": true_gain * signal + noise,
            "noise_e": jnp.zeros_like(signal),
            "e_variance": variance,
        }
        diagnostics = loss._angular_data_objective(batch, signal, lam_axis)[4]
        return diagnostics["profiled_gains"][0], diagnostics["profiled_gain_standard_error"][0]

    estimates, standard_errors = jax.jit(jax.vmap(estimate))(noise_samples)
    coverage = jnp.mean(jnp.abs(estimates - true_gain) <= 1.96 * standard_errors)
    assert float(jnp.mean(estimates)) == pytest.approx(true_gain, abs=0.02)
    assert 0.90 < float(coverage) < 0.99


def test_two_velocity_moments_use_physical_edf_without_exponentiating_it():
    loss = _objective()
    loss.cfg["parameters"]["electron"]["fe"]["velocity"] = jnp.array([-0.5, 0.5])
    physical = {
        "electron": {
            "fe": jnp.full((2, 2), 0.25),
            "v": jnp.array([-0.5, 0.5]),
        }
    }
    density, temperature, momentum = loss._moment_loss_(physical)
    assert density == pytest.approx(0.0)
    assert temperature == pytest.approx((0.5 - 2.0) ** 2)
    assert momentum == pytest.approx(0.0)


def test_configured_physical_edf_regularizers_are_reported_term_by_term():
    loss = _objective(
        regularization={
            "kl_to_maxwellian": 0.5,
            "density": 3.0,
            "temperature": 2.0,
            "momentum": 4.0,
        }
    )
    physical = {
        "electron": {
            "fe": jnp.full((3, 3), 1.0 / 9.0),
            "v": jnp.array([-1.0, 0.0, 1.0]),
        }
    }

    terms = loss._regularization_terms(lambda: physical)

    assert set(terms) == {
        "regularization_radial",
        "regularization_angular",
        "regularization_kl",
        "regularization_density",
        "regularization_temperature",
        "regularization_momentum",
    }
    assert terms["regularization_density"] == pytest.approx(0.0)
    assert terms["regularization_temperature"] == pytest.approx(2.0 * (4.0 / 3.0 - 2.0) ** 2)
    assert terms["regularization_momentum"] == pytest.approx(0.0)
    assert terms["regularization_kl"] > 0.0


def test_profiled_gain_and_edf_regularization_coexist_after_area_preserving_irf():
    loss = _objective(
        noise={"model": "measured_variance"},
        gain={"mode": "per_row", "smoothness": 0.0},
        regularization={"kl_to_maxwellian": 0.5},
    )
    angles = np.array([-1.0, 0.0, 1.0])
    wavelengths = jnp.linspace(0.1, 4.9, 9)
    source = jnp.zeros((3, 9)).at[:, 4].set(jnp.array([1.0, 2.0, 1.5]))
    _, signal = add_ATS_IRF(
        AngularIRF(spect_stddev=0.3, ang_stddev=0.2, ang_axis=angles, normalize=0),
        wavelengths,
        source,
        {"general": {"lam": 2.5, "amp1": 1.0, "amp2": 1.0}},
    )
    expected_gains = jnp.array([0.75, 1.5, 2.0])
    batch = {
        "e_data": expected_gains[:, None] * signal,
        "noise_e": jnp.zeros_like(signal),
        "e_variance": jnp.ones_like(signal),
    }

    data_term, _, _, _, diagnostics = loss._angular_data_objective(batch, signal, wavelengths)
    regularization = loss._regularization_terms(
        lambda: {
            "electron": {
                "fe": jnp.full((3, 3), 1.0 / 9.0),
                "v": jnp.array([-1.0, 0.0, 1.0]),
            }
        }
    )

    np.testing.assert_allclose(diagnostics["profiled_gains"], expected_gains, rtol=2e-5)
    assert data_term == pytest.approx(0.0, abs=1e-10)
    assert regularization["regularization_kl"] > 0.0


def test_angular_diagnostics_exposes_persistable_arrays_and_term_breakdown():
    loss = _objective(
        noise={"model": "measured_variance"},
        gain={"mode": "none"},
    )
    theory = jnp.ones((1, 4))
    loss.multiplex_ang = False
    loss.ts_diag = lambda weights, batch: (theory, 0.0, jnp.arange(1.0, 5.0), jnp.zeros(1))
    batch = {
        "e_data": theory,
        "noise_e": jnp.zeros_like(theory),
        "e_variance": jnp.ones_like(theory),
    }

    arrays, terms = loss.angular_diagnostics(object(), batch)

    assert isinstance(arrays["whitened_residual"], np.ndarray)
    assert set(terms) >= {"data", "gain_prior", "gain_smoothness", "total"}
    assert terms["total"] == pytest.approx(0.0)


def test_unknown_or_negative_angular_regularizer_is_rejected():
    with pytest.raises(ValueError, match="Unknown optimizer.angular_objective.regularization"):
        _objective(regularization={"radial_smothness": 1.0})
    with pytest.raises(ValueError, match="radial_smoothness must be non-negative"):
        _objective(regularization={"radial_smoothness": -1.0})
