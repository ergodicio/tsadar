"""Scientific metric and production smoke contracts for ISS140."""

from dataclasses import replace
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tsadar.benchmarks.arts2d import (
    BenchmarkSpec,
    _blank_batch,
    detector_jacobian,
    fit_model,
    initialize_model,
    make_config,
    run_benchmark,
    summarize_runs,
)
from tsadar.inverse.loss_function import LossFunction
from tsadar.inverse.recovery_benchmark import (
    bootstrap_mean_interval,
    config_hash,
    detector_visible_null_basis,
    edf_moments,
    make_detector_mask,
    noise_whitened_residuals,
    poisson_read_observation,
    project_gain_nuisance,
    subspace_edf_errors,
    synthetic_truth_families,
)


def test_cell_weighted_moments_match_rotated_gaussian():
    axis = np.linspace(-6 + 6 / 128, 6 - 6 / 128, 128)
    truths = synthetic_truth_families(axis, axis, rotation_deg=27)
    for truth in truths.values():
        assert np.all(truth >= 0)
        assert np.sum(truth) * (12 / 128) ** 2 == pytest.approx(1, abs=1e-14)
    actual = edf_moments(truths["elliptic"], axis, axis)
    theta = np.deg2rad(27)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    covariance = rotation @ np.diag([1.15**2, 0.82**2]) @ rotation.T
    np.testing.assert_allclose(
        [
            [actual["pressure_xx"], actual["pressure_xy"]],
            [actual["pressure_xy"], actual["pressure_yy"]],
        ],
        covariance,
        atol=2e-6,
    )
    assert actual["flow_vx"] == pytest.approx(0, abs=1e-14)
    assert actual["temperature_in_plane"] == pytest.approx(
        np.trace(covariance) / 2, abs=2e-6
    )
    assert actual["heat_flux_x_in_plane"] == pytest.approx(0, abs=1e-14)
    skew = edf_moments(truths["skew_tail"], axis, axis)
    assert skew["flow_vx"] > 0 and skew["heat_flux_x_in_plane"] > 0
    assert skew["current_x"] == pytest.approx(-skew["normalization"] * skew["flow_vx"])


def test_splits_and_nonfinite_predictions_cannot_silently_improve_score():
    mask = make_detector_mask(5, 10, heldout_angles=(1, 4), heldout_wedges=((3, 5),))
    assert mask.sum() == 24
    assert not mask[[1, 4]].any() and not mask[:, 3:5].any()
    predicted = np.ones((5, 10))
    np.testing.assert_allclose(
        noise_whitened_residuals(predicted + 2, predicted, 2, mask), 1
    )
    predicted[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        noise_whitened_residuals(np.ones_like(predicted), predicted, 1, mask)
    with pytest.raises(ValueError, match="withhold"):
        make_detector_mask(2, 3)
    with pytest.raises(ValueError, match="positive"):
        noise_whitened_residuals([1.0], [0.0], [0.0])


def test_seeded_poisson_read_noise_has_the_declared_moments():
    signal = np.full(50000, 20.0)
    first, variance = poisson_read_observation(
        signal, seed=41, background=4, read_noise=3
    )
    repeated, _ = poisson_read_observation(signal, seed=41, background=4, read_noise=3)
    np.testing.assert_array_equal(first, repeated)
    assert first.mean() == pytest.approx(24, abs=0.12)
    assert first.var() == pytest.approx(33, rel=0.025)
    np.testing.assert_array_equal(variance, np.full(50000, 33.0))


def test_common_edf_subspaces_remove_gain_and_preserve_error_energy():
    jacobian = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    projected = project_gain_nuisance(jacobian, np.ones(3))
    np.testing.assert_allclose(projected[:, 0], 0, atol=1e-14)
    svd = detector_visible_null_basis(projected, atol=0, rtol=1e-10)
    assert svd["rank"] == 1
    assert svd["visible_basis"].shape == (4, 1)
    result = subspace_edf_errors(
        np.zeros(4),
        np.array([0.0, 3.0, 4.0, 12.0]),
        svd["visible_basis"],
        cell_area=0.25,
    )
    assert result["edf_visible_l2"] == pytest.approx(1.5)
    assert result["edf_l2"] ** 2 == pytest.approx(
        result["edf_visible_l2"] ** 2 + result["edf_null_l2"] ** 2
    )
    assert detector_visible_null_basis(projected, atol=2, rtol=0)["rank"] == 0
    # Underdetermined problems retain an implicit complement, not an N_EDF^2 matrix.
    wide = detector_visible_null_basis(np.eye(3, 1000), atol=0)
    assert wide["visible_basis"].shape == (1000, 3)


def test_common_detector_jacobian_matches_independent_difference():
    matrix = jnp.asarray(
        [[1.0, 0.1, 0.2, 0.5], [0.2, 1.2, 0.3, 0.1], [0.5, 0.4, 0.6, 1.1]]
    )
    forward = lambda f: (matrix @ f.ravel()).reshape(1, 3)
    truth = np.array([[0.2, 0.3], [0.1, 0.4]]) * 4
    sigma = np.array([[0.3, 0.2, 0.4]])
    mask = np.array([[True, True, False]])
    area = 0.25
    gain = 1.15
    actual = detector_jacobian(
        forward, truth, sigma, mask, cell_area=area, gain=gain, batch_size=1
    )
    q0 = truth.ravel() * np.sqrt(area)

    def measurement(q):
        f = q.reshape(2, 2) / np.sqrt(area)
        return gain * np.asarray(forward(f))[mask] / sigma[mask]

    step = 1e-5
    finite_difference = np.column_stack(
        [
            (
                measurement(q0 + step * (e - e.mean()))
                - measurement(q0 - step * (e - e.mean()))
            )
            / (2 * step)
            for e in np.eye(4)
        ]
    )
    expected = project_gain_nuisance(
        finite_difference, np.asarray(forward(truth))[mask] / sigma[mask]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual @ np.ones(4), 0, atol=1e-12)


def test_initialization_activates_the_real_model_and_common_maxwellian():
    spec = BenchmarkSpec.smoke()
    initial_edfs = []
    for name in spec.models:
        config = make_config(spec, model=name)
        initial, diff, static, leaves = initialize_model(
            config, spec.initialization_seed
        )
        f = initial.electron.distribution_functions()
        initial_edfs.append(np.asarray(f))
        assert leaves and all(
            "distribution_functions" in leaf["path"] for leaf in leaves
        )
        assert not any(leaf["path"].endswith(".vx") for leaf in leaves)
        x, _ = jnp.meshgrid(
            initial.electron.distribution_functions.vx,
            initial.electron.distribution_functions.vx,
        )
        gradient = eqx.filter_jit(
            eqx.filter_grad(
                lambda d: jnp.sum(
                    eqx.combine(d, static).electron.distribution_functions() * x
                )
            )
        )(diff)
        norm = sum(
            float(jnp.sum(value**2)) for value in jax.tree_util.tree_leaves(gradient)
        )
        assert (
            np.isfinite(norm) and norm > 0
        ), f"{name} has no trainable directional perturbation"
    for actual in initial_edfs[1:]:
        np.testing.assert_allclose(actual, initial_edfs[0], rtol=1e-6, atol=1e-10)


def test_seed_aggregation_uses_paired_differences_and_no_single_run_interval():
    spec = replace(
        BenchmarkSpec.smoke(),
        seeds=(0, 1, 2),
        models=("arbitrary", "nn-sh"),
        bootstrap_resamples=100,
    )
    records = [
        {
            "truth_family": "elliptic",
            "peak_counts": 1000.0,
            "fit_model": model,
            "seed": seed,
            "best": {"edf_l2": value + offset},
        }
        for model, offset in [("arbitrary", 0), ("nn-sh", 0.3)]
        for seed, value in enumerate([0.2, 0.5, 0.1])
    ]
    summary = summarize_runs(records, spec)
    paired = summary["groups"][1]["paired_difference_from_arbitrary"]["edf_l2"]
    np.testing.assert_allclose(paired["ci95"], [0.3, 0.3], atol=1e-15)
    assert bootstrap_mean_interval([1.0])["ci95"] is None
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        config_hash({"not_a_finite_config": float("nan")})
    with pytest.raises(ValueError, match="exactly one"):
        summarize_runs(records[:-1], spec)
    with pytest.raises(ValueError, match="exactly one"):
        summarize_runs(records + records[:1], spec)


def test_failed_spectrum_cannot_be_hidden_by_production_pixel_masking(tmp_path):
    spec = BenchmarkSpec.smoke()
    config = make_config(spec)
    initial, _, _, _ = initialize_model(config, spec.initialization_seed)
    velocity = np.asarray(initial.electron.distribution_functions.vx)
    shape = (len(spec.angles_deg), spec.detector_bins)
    mask = make_detector_mask(*shape, heldout_angles=spec.heldout_angles)
    batch = dict(
        _blank_batch(spec),
        e_data=jnp.ones(shape),
        e_variance=jnp.ones(shape),
        e_mask=jnp.asarray(mask),
    )
    angles = np.asarray(spec.angles_deg)
    loss = LossFunction(
        config, {"sa": angles, "angAxis": angles, "weights": np.eye(len(angles))}, batch
    )
    bad_spectrum = (
        jnp.ones(shape).at[1, 0].set(jnp.nan)
    )  # even an unobserved model pixel must fail
    with pytest.raises(FloatingPointError, match="nonfinite"):
        fit_model(
            spec,
            config,
            lambda edf: bad_spectrum,
            loss,
            batch,
            {"train": mask, "heldout": ~mask},
            np.asarray(initial.electron.distribution_functions()),
            velocity,
            {"visible_basis": np.empty((spec.nvx**2, 0))},
            tmp_path / "failed-fit",
        )
    assert not (tmp_path / "failed-fit").exists()


def test_production_arts2d_benchmark_smoke(tmp_path):
    """All four real EDFs, refined truth, detector, likelihood, AD, optimizer and SVD."""
    spec = BenchmarkSpec.smoke()
    output = tmp_path / "benchmark"
    records, summary = run_benchmark(spec, output)
    assert len(records) == len(spec.models) == 4
    assert len(summary["groups"]) == 4
    shared = output / "elliptic-counts-1000"
    case = np.load(shared / "case.npz")
    assert case["truth_edf"].shape != case["refined_truth_edf"].shape
    assert np.all(case["train"] != case["heldout"])
    svd = np.load(shared / "subspaces.npz")
    assert np.all(np.isfinite(svd["jacobian"])) and int(svd["rank"]) > 0
    np.testing.assert_allclose(svd["jacobian"].sum(axis=1), 0, atol=1e-10)
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    data = np.load(shared / "seed-0/observations.npy")
    for record in records:
        run = output / record["directory"]
        recovery = np.load(run / "recovery.npz")
        meta = json.loads((run / "provenance.json").read_text())
        assert meta["fit_model"] == record["fit_model"]
        assert meta["active_parameter_count"] > 0
        assert record["gradient_evaluations"] == spec.steps
        assert record["evaluated_states"] == spec.steps + 1
        assert record["best_train_objective"] <= record["initial"]["train_objective"]
        assert record["best_step"] <= spec.steps
        whitened = (data - recovery["best_prediction"]) / np.sqrt(case["variance"])
        # Scores and weights belong to the same evaluated iterate; held-out samples
        # use the training-fitted prediction without a new gain fit.
        assert record["best_train_objective"] == pytest.approx(
            np.mean(whitened[case["train"]] ** 2), rel=1e-8
        )
        assert record["best"]["heldout_whitened_rms"] == pytest.approx(
            np.sqrt(np.mean(whitened[case["heldout"]] ** 2))
        )
        best = record["best"]
        assert best["edf_l2"] ** 2 == pytest.approx(
            best["edf_visible_l2"] ** 2 + best["edf_null_l2"] ** 2
        )
        assert (run / "best_weights.eqx").is_file()
    with pytest.raises(FileExistsError):
        run_benchmark(spec, output)
