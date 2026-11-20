import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import get_filter_spec
from tsadar.data import prepare
from tsadar.inverse.loops import build_batch, one_d_loop
from tsadar.inverse.postprocess import mcmc, mcmc_calibration

from .shared import base_config


def test_check_fe_inactive_raises_when_fe_active():
    cfg = base_config()
    cfg["parameters"]["electron"]["fe"]["active"] = True
    with pytest.raises(NotImplementedError):
        mcmc.check_fe_inactive(cfg["parameters"])


def test_check_fe_inactive_passes_when_fe_inactive():
    cfg = base_config()
    mcmc.check_fe_inactive(cfg["parameters"])  # should not raise


def test_seed_step_scale_from_laplace_regularizes_degenerate_brem_c(fitted_fixture):
    # brem_c is degenerate with Z/Te/ne; starting it at 0.7 gives it a non-positive diagonal Hessian
    # entry at the fitted point, which exercises the eigenvalue-clipping branch of the Laplace seed. The
    # proposal covariance must stay positive-definite, brem_c's variance must come out larger than a
    # well-conditioned parameter's, and the sampler must still run to completion.
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["data"]["background"]["type"] = "brem_model"
    cfg["parameters"]["general"]["brem_amp"] = {"active": True, "lb": 0.0, "ub": 1.0, "val": 0.4}
    cfg["parameters"]["general"]["brem_c"] = {"active": True, "lb": 0.0, "ub": 1.0, "val": 0.7}

    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
    sample_indices = np.arange(cfg["optimizer"]["batch_size"])
    with mlflow.start_run():
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, 1)
    ts_params = fitted_weights[0]
    filter_spec = get_filter_spec(cfg["parameters"], ts_params)
    diff_params, static_params = eqx.partition(ts_params, filter_spec)
    batch = build_batch(all_data, sample_indices, cfg["data"]["background"]["bg_subtract"])

    leaves = jax.tree_util.tree_leaves(diff_params)
    paths = [p for p, _ in jax.tree_util.tree_flatten_with_path(diff_params)[0]]
    brem_c_idx = next(i for i, p in enumerate(paths) if "brem_c" in str(p))
    other_idx = next(i for i in range(len(leaves)) if i != brem_c_idx)

    # Confirm this scenario actually lands brem_c in the non-positive-curvature regime this test means to
    # exercise -- if it didn't, the assertions below wouldn't be testing what this test claims to test.
    def _nll_of_diff(dp):
        weights = eqx.combine(static_params, dp)
        return loss_fn.neg_log_likelihood(weights, batch, per_lineout=False)

    row_value = leaves[brem_c_idx]

    def _nll_wrt_brem_c(value):
        new_leaves = list(leaves)
        new_leaves[brem_c_idx] = value
        return _nll_of_diff(jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(diff_params), new_leaves))

    _, h_ii = jax.jvp(jax.grad(_nll_wrt_brem_c), (row_value,), (jnp.ones_like(row_value),))
    assert np.all(np.asarray(h_ii) <= 0), (
        f"expected brem_c's diagonal Hessian entry to be non-positive at this fitted point (got {h_ii}) -- "
        "this scenario is no longer exercising the intended regularization branch; see this test's "
        "docstring for how val=0.7 was chosen to reproduce that."
    )

    step_scale = mcmc._seed_step_scale_from_laplace(loss_fn, static_params, batch, diff_params)
    n = len(leaves)
    batch_size = cfg["optimizer"]["batch_size"]
    assert step_scale.shape == (batch_size, n, n)

    sigma = np.einsum("bik,bjk->bij", np.asarray(step_scale), np.asarray(step_scale))
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals > 0), "proposal covariance must stay positive-definite even with brem_c's degenerate curvature"

    # brem_c's own marginal variance should come out larger than a well-conditioned leaf's -- a
    # genuinely poorly-constrained parameter reported as such, not silently clamped to a fixed number.
    brem_c_var = sigma[:, brem_c_idx, brem_c_idx]
    other_var = sigma[:, other_idx, other_idx]
    assert np.all(brem_c_var > other_var), (
        f"expected brem_c's marginal variance to exceed a well-conditioned leaf's "
        f"(brem_c={brem_c_var}, other={other_var})"
    )

    # End-to-end sanity: the sampler still runs to completion and returns finite, valid results with this
    # genuinely degenerate parameter active (integration coverage for the regularization path).
    cfg["other"]["mcmc"] = {"num_steps": 500, "burn_in": 200, "thin": 5, "adapt_every": 50, "use_laplace_seed": True}
    key = jax.random.PRNGKey(42)
    samples, _, diagnostics = mcmc.run_mcmc_for_batch(cfg, loss_fn, ts_params, batch, key)
    acceptance_rate = np.asarray(diagnostics["acceptance_rate"])
    assert acceptance_rate.shape == (cfg["optimizer"]["batch_size"],)
    assert np.all(np.isfinite(acceptance_rate))
    assert np.all((acceptance_rate >= 0) & (acceptance_rate <= 1))
    for leaf in jax.tree_util.tree_leaves(samples):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_seed_step_scale_default_has_no_hessian_dependency(fitted_fixture):
    # the default seed needs no loss_fn/batch and returns init_step_scale * I per lineout
    ts_params = fitted_fixture["fitted_weights"][0]
    filter_spec = get_filter_spec(fitted_fixture["config"]["parameters"], ts_params)
    diff_params, _ = eqx.partition(ts_params, filter_spec)
    init_step_scale = 0.05
    step_scale = mcmc._seed_step_scale_default(diff_params, init_step_scale)
    n = len(jax.tree_util.tree_leaves(diff_params))
    batch_size = jax.tree_util.tree_leaves(diff_params)[0].shape[0]
    assert step_scale.shape == (batch_size, n, n)
    expected = init_step_scale * np.eye(n)
    for i in range(batch_size):
        np.testing.assert_allclose(np.asarray(step_scale[i]), expected)


def test_run_mcmc_for_batch_falls_back_when_laplace_seed_disabled(fitted_fixture):
    # use_laplace_seed=False must not touch the Hessian machinery at all, and still produce a valid
    # (if less well-tuned, given the short chain here) chain.
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["mcmc"] = {"num_steps": 100, "burn_in": 50, "thin": 2, "adapt_every": 10, "use_laplace_seed": False}
    key = jax.random.PRNGKey(0)
    samples, _, diagnostics = mcmc.run_mcmc_for_batch(
        cfg, fitted_fixture["loss_fn"], fitted_fixture["fitted_weights"][0], fitted_fixture["batch"], key
    )
    assert np.all(np.isfinite(np.asarray(diagnostics["acceptance_rate"])))


def test_seed_step_scale_from_laplace_matches_full_hessian_inverse(fitted_fixture):
    # the proposal covariance should equal rr_factor^2 * inv(H_reg), with H the Hessian of the negative
    # log-posterior (half the eqx.filter_hessian of the -2*log-likelihood plus the Jacobian term) and
    # H_reg that Hessian after flooring the eigenvalues of its normalized form
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["parameters"]["ion-1"]["Z"]["active"] = True  # exercise more than just electron/general leaves
    # the shared fixture is EPW-only, which leaves Z unconstrained; re-prepare a private copy of the
    # data with the IAW loaded and fit so Z is constrained
    cfg["data"]["load_ion_spec"] = True
    cfg["data"]["fit_IAW"] = True
    with mlflow.start_run():
        all_data, sa, _ = prepare.prepare_data(cfg, cfg["data"]["shotnum"])
    sample_indices = np.arange(cfg["optimizer"]["batch_size"])

    with mlflow.start_run():
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, 1)
    ts_params = fitted_weights[0]
    filter_spec = get_filter_spec(cfg["parameters"], ts_params)
    diff_params, static_params = eqx.partition(ts_params, filter_spec)
    batch = build_batch(all_data, sample_indices, cfg["data"]["background"]["bg_subtract"])

    def _nll_of_diff(dp):
        weights = eqx.combine(static_params, dp)
        return loss_fn.neg_log_likelihood(weights, batch, per_lineout=False)

    full_hess = eqx.filter_hessian(_nll_of_diff)(diff_params)
    target_structure = jax.tree_util.tree_structure(diff_params)
    rows = jax.tree_util.tree_leaves(
        full_hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure
    )
    n = len(jax.tree_util.tree_leaves(diff_params))
    assert len(rows) == n
    batch_size = cfg["optimizer"]["batch_size"]

    # per-lineout Hessian blocks from eqx.filter_hessian; also checks the cross-lineout terms are zero
    H_true = np.zeros((batch_size, n, n))
    for i, row in enumerate(rows):
        row_leaves = jax.tree_util.tree_leaves(row)
        assert len(row_leaves) == n
        for j, block in enumerate(row_leaves):
            block = np.asarray(block)
            off_lineout_diagonal = block - np.diag(np.diag(block))
            assert np.allclose(off_lineout_diagonal, 0.0, atol=1e-6), (
                f"leaf {i} x leaf {j} Hessian block has nonzero cross-lineout entries -- "
                "h_loss_wrt_params_per_lineout's low-memory diagonal-in-lineout trick is not valid here"
            )
            H_true[:, i, j] = np.diagonal(block)

    step_scale = mcmc._seed_step_scale_from_laplace(loss_fn, static_params, batch, diff_params)
    assert step_scale.shape == (batch_size, n, n)

    rr_factor = 2.38 / np.sqrt(n)
    s = 1.0 / (1.0 + np.exp(-np.stack([np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(diff_params)], axis=-1)))
    H_true = 0.5 * H_true + np.stack([np.diag(2.0 * s_i * (1.0 - s_i)) for s_i in s])
    for i in range(batch_size):
        d = 1.0 / np.sqrt(np.abs(np.diagonal(H_true[i])))
        h_norm = H_true[i] * np.outer(d, d)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (h_norm + h_norm.T))
        scaled_eigvecs = eigvecs * d[:, None]
        expected_sigma = (rr_factor**2) * (
            scaled_eigvecs / np.maximum(eigvals, mcmc._LAPLACE_EIGVAL_FLOOR)
        ) @ scaled_eigvecs.T
        if np.all(eigvals > mcmc._LAPLACE_EIGVAL_FLOOR):
            np.testing.assert_allclose(expected_sigma, (rr_factor**2) * np.linalg.inv(H_true[i]), rtol=1e-6)
        actual_sigma = np.asarray(step_scale[i]) @ np.asarray(step_scale[i]).T
        np.testing.assert_allclose(actual_sigma, expected_sigma, rtol=1e-4, atol=1e-8)


def test_calibration_draws_collapse_to_identity_when_unconfigured(fitted_fixture):
    cfg = fitted_fixture["config"]
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(0)
    )
    assert len(draws) == 1
    assert draws[0][0] is cfg
    assert draws[0][1] is fitted_fixture["all_data"]


def test_calibration_draws_repeat_nominal_when_all_sigmas_zero(fitted_fixture):
    # num_draws still drives the chain count even with nothing to perturb calibration-wise: draws
    # collapsing to a single chain here would silently defeat init_dispersion_factor/R-hat, which only
    # need independent chains, not independently-perturbed calibrations.
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 8, "detector_gain_sigma": 0.0, "EPWDispersion_sigma": 0.0}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(0)
    )
    assert len(draws) == 8
    for config_k, all_data_k in draws:
        assert config_k is cfg
        assert all_data_k is fitted_fixture["all_data"]


def test_calibration_draws_perturb_gain_and_rescale_data(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 5, "detector_gain_sigma": 0.05}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(1)
    )
    assert len(draws) == 5
    nominal_gain = cfg["other"]["detector_gain"]
    nominal_e_data = fitted_fixture["all_data"]["e_data"]
    gains = [cfg_k["other"]["detector_gain"] for cfg_k, _ in draws]
    assert len(set(gains)) > 1  # actually different draws, not all collapsed to the nominal value
    for cfg_k, all_data_k in draws:
        expected_scale = nominal_gain / cfg_k["other"]["detector_gain"]
        np.testing.assert_allclose(all_data_k["e_data"], nominal_e_data * expected_scale)


def test_init_dispersion_factor_perturbs_starting_point(fitted_fixture):
    # With very few steps (so the chain has no time to "forget" its start) and use_laplace_seed off (a
    # fixed, deterministic step_scale), a nonzero init_dispersion_factor should visibly shift where the
    # chain's samples land compared to an otherwise-identical zero-dispersion run at the same key.
    cfg = copy.deepcopy(fitted_fixture["config"])
    base_settings = {
        "num_steps": 5, "burn_in": 0, "thin": 1, "adapt_every": 5, "use_laplace_seed": False, "init_step_scale": 0.05,
    }
    key = jax.random.PRNGKey(3)

    cfg["other"]["mcmc"] = {**base_settings, "init_dispersion_factor": 0.0}
    samples_no_disp, _, _ = mcmc.run_mcmc_for_batch(
        cfg, fitted_fixture["loss_fn"], fitted_fixture["fitted_weights"][0], fitted_fixture["batch"], key
    )

    cfg["other"]["mcmc"] = {**base_settings, "init_dispersion_factor": 5.0}
    samples_disp, _, _ = mcmc.run_mcmc_for_batch(
        cfg, fitted_fixture["loss_fn"], fitted_fixture["fitted_weights"][0], fitted_fixture["batch"], key
    )

    leaves_no_disp = jax.tree_util.tree_leaves(samples_no_disp)
    leaves_disp = jax.tree_util.tree_leaves(samples_disp)
    assert len(leaves_no_disp) == len(leaves_disp) > 0
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(leaves_no_disp, leaves_disp))


def test_run_mcmc_pooled_r_hat_is_none_with_a_single_chain(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["mcmc"] = {"num_steps": 100, "burn_in": 50, "thin": 2, "adapt_every": 10}
    key = jax.random.PRNGKey(0)
    batch_size = cfg["optimizer"]["batch_size"]
    ts_params = fitted_fixture["fitted_weights"][0]
    filter_spec = get_filter_spec(cfg["parameters"], ts_params)
    n_active = len(jax.tree_util.tree_leaves(eqx.partition(ts_params, filter_spec)[0]))
    _, _, _, max_r_hat, within_chain_r_hat = mcmc.run_mcmc_pooled(
        cfg,
        [fitted_fixture["loss_fn"]],
        [ts_params],
        [[fitted_fixture["batch"]]],
        key,
    )
    assert max_r_hat is None

    # Unlike max_r_hat, within_chain_r_hat only ever compares a chain to itself, so a single chain (K=1)
    # is not a degenerate case for it the way it is for cross-chain R-hat.
    within_chain_r_hat = np.asarray(within_chain_r_hat)
    assert within_chain_r_hat.shape == (1, 1, batch_size, n_active)
    assert np.all(np.isfinite(within_chain_r_hat))
    assert np.all(within_chain_r_hat >= 0.5)  # see the multi-chain test's assertion for why not a strict >= 1.0


def test_rank_normalize_pooled_uses_symmetric_blom_scores():
    x = np.random.default_rng(202).normal(size=(20, 1))
    z = mcmc._rank_normalize_pooled(x)

    ranks = np.argsort(np.argsort(x[:, 0])) + 1.0
    expected = jax.scipy.stats.norm.ppf((ranks - 0.375) / (x.size + 0.25))
    np.testing.assert_allclose(z[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(np.sort(z[:, 0]), -np.sort(z[:, 0])[::-1], atol=1e-12)


def test_burn_in_shorter_than_a_chunk_is_still_run(fitted_fixture):
    # 25 burn-in steps with adapt_every=50 must not behave like no burn-in at all
    def _run(num_steps, burn_in):
        cfg = copy.deepcopy(fitted_fixture["config"])
        cfg["other"]["mcmc"] = {
            "num_steps": num_steps, "burn_in": burn_in, "adapt_every": 50, "thin": 1, "use_laplace_seed": False,
        }
        _, _, diagnostics = mcmc.run_mcmc_for_batch(
            cfg, fitted_fixture["loss_fn"], fitted_fixture["fitted_weights"][0], fitted_fixture["batch"],
            jax.random.PRNGKey(3),
        )
        return np.asarray(diagnostics["final_step_scale"])

    with pytest.warns(UserWarning):
        adapted = _run(100, 25)
        unadapted = _run(75, 0)
    assert not np.allclose(adapted, unadapted)


def test_covar_loss_function_builds_for_an_electron_only_fit(fitted_fixture):
    from tsadar.inverse.postprocess.mcmc_postprocess import _build_loss_fn_for_draw

    cfg = copy.deepcopy(fitted_fixture["config"])
    assert not cfg["data"]["fit_IAW"]
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = False

    loss_fn = _build_loss_fn_for_draw(cfg, fitted_fixture["sa"], fitted_fixture["all_data"], cfg["optimizer"]["batch_size"])

    assert not hasattr(loss_fn, "covar_iaw_idx")
    assert any(hasattr(loss_fn, name) for name in ("covar_blue_idx", "covar_red_idx"))
    batch = build_batch(fitted_fixture["all_data"], np.arange(cfg["optimizer"]["batch_size"]), False)
    weights = fitted_fixture["fitted_weights"][0]
    nll = np.asarray(loss_fn.neg_log_likelihood(weights, batch, per_lineout=True))
    assert np.all(np.isfinite(nll))

    # Gaussian -2*log-likelihood of the in-range pixels, evaluated independently with numpy:
    # residual^T K^-1 residual + log(det K), summed over the two wings
    ThryE, _, _, _ = loss_fn.ts_diag(weights, batch)
    ThryE, e_data = np.asarray(ThryE), np.asarray(batch["e_data"])
    expected = np.zeros(e_data.shape[0])
    for name in ("blue", "red"):
        idx = np.asarray(getattr(loss_fn, f"covar_{name}_idx"))
        sub = np.asarray(getattr(loss_fn, f"covar_{name}_sub"))
        k = np.asarray(
            loss_fn.calculate_covariance_matrix(jnp.asarray(ThryE[:, idx]), loss_fn.covar_throughput_e[idx])
        )[:, sub][:, :, sub]
        resid = (e_data - ThryE)[:, idx][:, sub]
        quad = np.einsum("bi,bi->b", resid, np.linalg.solve(k, resid[..., None])[..., 0])
        expected += quad + np.linalg.slogdet(k)[1]
    np.testing.assert_allclose(nll, expected, rtol=1e-8)

    # the fitting statistic is differentiated with K held fixed: d/dt [r^T K^-1 r / norm] = -2 K^-1 r / norm
    idx, mask, sub = loss_fn.covar_blue_idx, loss_fn.covar_blue_mask, loss_fn.covar_blue_sub
    e_data_j, thry_j = jnp.asarray(e_data), jnp.asarray(ThryE)
    grad = jax.grad(
        lambda t: loss_fn._feature_error_(
            e_data_j, t, None, None, t, None, False, covar_idx=idx, covar_submask=mask, covar_sub=sub,
            covar_throughput=loss_fn.covar_throughput_e,
        )[0]
    )(thry_j)
    pixels = np.asarray(idx)[np.asarray(sub)]
    k = np.asarray(loss_fn.calculate_covariance_matrix(thry_j[:, np.asarray(idx)], loss_fn.covar_throughput_e[idx]))[
        :, np.asarray(sub)
    ][:, :, np.asarray(sub)]
    resid = (e_data - ThryE)[:, pixels]
    norm = pixels.size * e_data.shape[0] - loss_fn.num_free_params
    expected_grad = np.zeros_like(ThryE)
    expected_grad[:, pixels] = -2.0 * np.linalg.solve(k, resid[..., None])[..., 0] / norm
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-6, atol=1e-12)


def test_covar_loss_function_rejects_an_empty_enabled_fit_range(fitted_fixture):
    from tsadar.inverse.postprocess.mcmc_postprocess import _build_loss_fn_for_draw

    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = False
    cfg["data"]["fit_EPWb"] = True
    cfg["data"]["fit_rng"]["blue_min"], cfg["data"]["fit_rng"]["blue_max"] = 1.0, 2.0

    with pytest.raises(ValueError, match="EPW blue fit range selects no pixels"):
        _build_loss_fn_for_draw(cfg, fitted_fixture["sa"], fitted_fixture["all_data"], cfg["optimizer"]["batch_size"])


def test_calibration_draws_leave_an_unperturbed_zero_irf_width_unchanged(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["detector_specs"]["widIRF"]["spect_stddev_ion"] = 0.0
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 3, "detector_gain_sigma": 0.05}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(4)
    )
    nominal_ele = cfg["other"]["detector_specs"]["widIRF"]["spect_stddev_ele"]
    for cfg_k, _ in draws:
        assert cfg_k["other"]["detector_specs"]["widIRF"]["spect_stddev_ion"] == 0.0
        assert cfg_k["other"]["detector_specs"]["widIRF"]["spect_stddev_ele"] == nominal_ele


def test_log_posterior_adds_the_logit_jacobian_and_the_configured_prior(fitted_fixture, monkeypatch):
    cfg = fitted_fixture["config"]
    loss_fn = fitted_fixture["loss_fn"]
    ts_params = fitted_fixture["fitted_weights"][0]
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(cfg["parameters"], ts_params))
    batch = fitted_fixture["batch"]

    log_likelihood = -0.5 * np.asarray(loss_fn.neg_log_likelihood(ts_params, batch, per_lineout=True))
    z = np.stack([np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(diff_params)], axis=-1)
    s = 1.0 / (1.0 + np.exp(-z))
    log_jacobian = np.sum(np.log(s) + np.log(1.0 - s), axis=-1)

    # default: uniform prior within the bounds, so only the Jacobian of the logit transform is added
    log_post = np.asarray(mcmc._log_posterior(loss_fn, diff_params, static_params, batch))
    np.testing.assert_allclose(log_post, log_likelihood + log_jacobian, rtol=1e-10)

    # a different prior is one registry entry selected by other.mcmc.prior
    monkeypatch.setitem(mcmc._LOG_PRIORS, "te_gaussian", lambda w: -0.5 * (w.get_unnormed_params()["electron"]["Te"] - 1.0) ** 2)
    monkeypatch.setitem(loss_fn.cfg["other"].setdefault("mcmc", {}), "prior", "te_gaussian")
    te = np.asarray(ts_params.get_unnormed_params()["electron"]["Te"])
    log_post = np.asarray(mcmc._log_posterior(loss_fn, diff_params, static_params, batch))
    np.testing.assert_allclose(log_post, log_likelihood + log_jacobian - 0.5 * (te - 1.0) ** 2, rtol=1e-10)

    monkeypatch.setitem(loss_fn.cfg["other"]["mcmc"], "prior", "not-a-prior")
    with pytest.raises(ValueError, match="Unknown MCMC prior"):
        mcmc._log_posterior(loss_fn, diff_params, static_params, batch)


def test_l2_likelihood_sums_the_two_epw_wings(fitted_fixture):
    # a mean-reduced loss averages the blue and red wings; the sum-reduced likelihood adds them
    cfg, loss_fn = fitted_fixture["config"], fitted_fixture["loss_fn"]
    assert cfg["optimizer"]["loss_method"] == "l2" and cfg["data"]["fit_EPWb"] and cfg["data"]["fit_EPWr"]
    weights, batch = fitted_fixture["fitted_weights"][0], fitted_fixture["batch"]

    ThryE, _, lamAxisE, _ = loss_fn.ts_diag(weights, batch)
    ThryE, lamAxisE, e_data = np.asarray(ThryE), np.asarray(lamAxisE), np.asarray(batch["e_data"])
    fr = cfg["data"]["fit_rng"]
    in_range = ((lamAxisE > fr["blue_min"]) & (lamAxisE < fr["blue_max"])) | (
        (lamAxisE > fr["red_min"]) & (lamAxisE < fr["red_max"])
    )
    expected = np.sum(np.where(in_range, (e_data - ThryE) ** 2 / (np.abs(e_data) + 1e-10), 0.0), axis=-1)

    nll = np.asarray(loss_fn.neg_log_likelihood(weights, batch, per_lineout=True))
    np.testing.assert_allclose(nll, expected, rtol=1e-8)
    assert loss_fn._averages_wings(jnp.nanmean) and not loss_fn._averages_wings(jnp.nansum)


def test_log_jacobian_includes_an_active_ion_fraction(fitted_fixture):
    from tsadar.core.modules.ts_params import ThomsonParams

    cfg = copy.deepcopy(fitted_fixture["config"])
    batch_size = cfg["optimizer"]["batch_size"]

    def _diff_params(config):
        ts_params = ThomsonParams(config["parameters"], batch_size, activate=True)
        return eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))[0]

    without = jax.tree_util.tree_leaves(_diff_params(cfg))
    cfg["parameters"]["ion-1"]["fract"]["active"] = True
    diff_params = _diff_params(cfg)
    leaves = jax.tree_util.tree_leaves(diff_params)
    assert len(leaves) == len(without) + 1

    s = 1.0 / (1.0 + np.exp(-np.stack([np.asarray(leaf) for leaf in leaves], axis=-1)))
    np.testing.assert_allclose(np.asarray(mcmc._log_jacobian(diff_params)), np.sum(np.log(s) + np.log(1.0 - s), axis=-1))
    np.testing.assert_allclose(
        np.asarray(mcmc._log_jacobian_curvature(diff_params)), np.stack([np.diag(2.0 * row * (1.0 - row)) for row in s])
    )


def test_covar_noise_covariance_carries_the_throughput_correction(fitted_fixture):
    from tsadar.data.correct_throughput import throughput_correction
    from tsadar.inverse.postprocess.mcmc_postprocess import _build_loss_fn_for_draw

    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = False
    loss_fn = _build_loss_fn_for_draw(cfg, fitted_fixture["sa"], fitted_fixture["all_data"], cfg["optimizer"]["batch_size"])

    # the loss function uses the same correction the electron data were multiplied by, and it is not trivial
    correction = np.asarray(loss_fn.covar_throughput_e)
    expected = throughput_correction(
        cfg["other"]["extraoptions"]["spectype"], np.asarray(fitted_fixture["all_axes"]["epw_y"]), cfg["data"]["shotnum"]
    )
    np.testing.assert_allclose(correction, expected, rtol=1e-10)
    idx = np.asarray(loss_fn.covar_red_idx)
    c = correction[idx]
    assert c.min() > 0 and c.max() / c.min() > 1.05

    # corrected-unit covariance = C K_detector C, with K_detector built from the uncorrected signal
    signal = jnp.asarray(np.linspace(20.0, 60.0, idx.size))[None, :]
    corrected = np.asarray(loss_fn.calculate_covariance_matrix(signal, jnp.asarray(c)))
    detector = np.asarray(loss_fn.calculate_covariance_matrix(signal / jnp.asarray(c)))
    np.testing.assert_allclose(corrected, detector * np.outer(c, c)[None], rtol=1e-10)

    # so the Gaussian quadratic form is the same in corrected and in detector units
    resid = np.sin(np.arange(idx.size))[None, :]
    quad_corrected = np.einsum("bi,bi->b", resid, np.linalg.solve(corrected, resid[..., None])[..., 0])
    quad_detector = np.einsum("bi,bi->b", resid / c, np.linalg.solve(detector, (resid / c)[..., None])[..., 0])
    np.testing.assert_allclose(quad_corrected, quad_detector, rtol=1e-8)


def test_brem_model_background_is_sized_along_the_wavelength_axis():
    from tsadar.data.evaluate_background import get_lineout_bg

    config = {
        "data": {
            "background": {"type": "brem_model"},
            "dpixel": 2,
            "lineouts": {"val": [3, 4]},
            "load_ele_spec": True,
            "load_ion_spec": True,
        }
    }
    image = np.zeros((8, 5))  # 8 wavelength pixels, 5 time pixels
    noiseE, noiseI = get_lineout_bg(config, image, image, 0, 0, [], 0, [1, 2], [1, 2], np.arange(8), np.arange(8))
    assert noiseE.shape == (2, 8) and noiseI.shape == (2, 8)


def test_calibration_draws_keep_the_throughput_correction_applied_to_the_data(fitted_fixture):
    from tsadar.inverse.postprocess.mcmc_postprocess import _build_loss_fn_for_draw

    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = False
    batch_size = cfg["optimizer"]["batch_size"]
    nominal = _build_loss_fn_for_draw(cfg, fitted_fixture["sa"], fitted_fixture["all_data"], batch_size)

    cfg["other"]["calibration_uncertainty"] = {"num_draws": 2, "EPWDispersion_sigma": 0.005}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(7)
    )
    for cfg_k, all_data_k in draws:
        # built on its own, a draw would evaluate the correction on its perturbed wavelength axis
        own = _build_loss_fn_for_draw(cfg_k, fitted_fixture["sa"], all_data_k, batch_size)
        assert not np.allclose(np.asarray(own.covar_throughput_e), np.asarray(nominal.covar_throughput_e))
        # the data were corrected on the nominal axis, so the noise model keeps that correction
        loss_fn_k = _build_loss_fn_for_draw(cfg_k, fitted_fixture["sa"], all_data_k, batch_size, nominal)
        np.testing.assert_array_equal(np.asarray(loss_fn_k.covar_throughput_e), np.asarray(nominal.covar_throughput_e))


def test_covar_throughput_correction_does_not_depend_on_points_per_pixel(fitted_fixture):
    from tsadar.data.correct_throughput import throughput_correction
    from tsadar.inverse.postprocess.mcmc_postprocess import _build_loss_fn_for_draw

    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["optimizer"]["loss_method"] = "covar"
    cfg["data"]["background"]["bg_subtract"] = False
    # oversample the model: its binned wavelength axis is still the detector pixel centers
    cfg["other"]["points_per_pixel"] = 2
    cfg["other"]["npts"] = int(cfg["other"]["CCDsize"][0] * 2)
    loss_fn = _build_loss_fn_for_draw(cfg, fitted_fixture["sa"], fitted_fixture["all_data"], cfg["optimizer"]["batch_size"])

    # the correction is the one applied to the data, on the detector pixel centers
    expected = throughput_correction(
        cfg["other"]["extraoptions"]["spectype"], np.asarray(fitted_fixture["all_axes"]["epw_y"]), cfg["data"]["shotnum"]
    )
    np.testing.assert_allclose(np.asarray(loss_fn.covar_throughput_e), expected, rtol=1e-10)


@pytest.mark.parametrize("points_per_pixel", [1, 2, 5])
def test_binned_model_axis_matches_the_detector_pixel_centers(fitted_fixture, points_per_pixel):
    from tsadar.inverse.loss_function import LossFunction

    cfg = copy.deepcopy(fitted_fixture["config"])
    batch_size = cfg["optimizer"]["batch_size"]
    cfg["other"]["points_per_pixel"] = points_per_pixel
    cfg["other"]["npts"] = int(cfg["other"]["CCDsize"][0] * points_per_pixel)
    sample = {k: v[:batch_size] for k, v in fitted_fixture["all_data"].items()}
    sample = {"noise_e": sample["noiseE"], "noise_i": sample["noiseI"]} | sample
    loss_fn = LossFunction(cfg, fitted_fixture["sa"], sample)

    _, _, lamAxisE, _ = loss_fn.ts_diag(fitted_fixture["fitted_weights"][0], fitted_fixture["batch"])
    np.testing.assert_allclose(np.asarray(lamAxisE)[0], np.asarray(fitted_fixture["all_axes"]["epw_y"]), rtol=0, atol=1e-9)


class _ShiftedGaussianLikelihood:
    def __init__(self, cfg, centre, width):
        self.cfg, self.centre, self.width = cfg, centre, width

    def neg_log_likelihood(self, weights, batch, per_lineout=True):
        te = weights.get_unnormed_params()["electron"]["Te"]
        return ((te - self.centre) / self.width) ** 2


def test_calibration_draws_propagate_into_the_pooled_posterior():
    from tsadar.core.modules.ts_params import ThomsonParams

    from tsverify.cases import HYDROGEN
    from tsverify.oracle import Plasma

    from ..physics import harness

    cfg = harness.diagnostic_config(Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), harness.EPW_DETECTOR)
    cfg["parameters"]["electron"]["Te"]["active"] = True
    cfg["other"]["mcmc"] = {
        "num_steps": 1500, "burn_in": 500, "thin": 2, "adapt_every": 50,
        "use_laplace_seed": False, "init_step_scale": 0.05, "init_dispersion_factor": 0.0,
    }
    ts_params = ThomsonParams(cfg["parameters"], num_params=2, activate=True)
    centres, width = np.array([0.55, 0.65, 0.75]), 0.01
    loss_fns = [_ShiftedGaussianLikelihood(cfg, c, width) for c in centres]
    pooled, static, _, _, _ = mcmc.run_mcmc_pooled(
        cfg, loss_fns, [ts_params], [[{}] for _ in centres], jax.random.PRNGKey(5)
    )
    te_of = eqx.filter_vmap(lambda dp: eqx.combine(dp, static).get_unnormed_params()["electron"]["Te"])
    te = np.asarray(te_of(jax.tree_util.tree_map(lambda x: x[0], pooled)))
    per_draw = te.reshape(len(centres), -1, te.shape[-1])

    np.testing.assert_allclose(per_draw.mean(axis=1), np.repeat(centres[:, None], 2, axis=1), atol=0.4 * width)
    np.testing.assert_allclose(per_draw.std(axis=1), width, rtol=0.3)
    np.testing.assert_allclose(te.std(axis=0), np.sqrt(width**2 + centres.var()), rtol=0.1)
