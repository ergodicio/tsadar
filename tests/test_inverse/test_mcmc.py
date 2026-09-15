import copy
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
import yaml
from flatten_dict import flatten, unflatten
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import get_filter_spec
from tsadar.data import prepare
from tsadar.inverse.loops import build_batch, one_d_loop
from tsadar.inverse.postprocess import mcmc, mcmc_calibration


def _base_config():
    with open("tests/configs/time_test_defaults.yaml") as fi:
        d = yaml.safe_load(fi)
    with open("tests/configs/time_test_inputs.yaml") as fi:
        i = yaml.safe_load(fi)
    flat = flatten(d)
    flat.update(flatten(i))
    cfg = unflatten(flat)
    cfg["parameters"]["electron"]["fe"]["active"] = False  # see mcmc.py's module docstring
    cfg["data"]["launch_data_visualizer"] = False
    cfg["data"]["lineouts"]["val"] = list(
        range(cfg["data"]["lineouts"]["start"], cfg["data"]["lineouts"]["end"], cfg["data"]["lineouts"]["skip"])
    )
    cfg["optimizer"]["num_epochs"] = 10  # fast fit; the tests below only need *a* fitted point, not a good one
    return cfg


@pytest.fixture(scope="module")
def fitted_fixture():
    """Runs one small real fit once and shares it across every test in this file, since re-fitting for
    each test would dominate the file's runtime without adding coverage."""
    cfg = _base_config()
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp()}/mlflow.db")
    mlflow.set_experiment("test-mcmc")
    with mlflow.start_run():
        all_data, sa, all_axes = prepare.prepare_data(cfg, cfg["data"]["shotnum"])
        sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
        num_batches = len(sample_indices) // cfg["optimizer"]["batch_size"] or 1
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, num_batches)
    batch = build_batch(all_data, sample_indices[: cfg["optimizer"]["batch_size"]], cfg["data"]["background"]["bg_subtract"])
    return {
        "config": cfg,
        "all_data": all_data,
        "all_axes": all_axes,
        "sa": sa,
        "loss_fn": loss_fn,
        "fitted_weights": fitted_weights,
        "batch": batch,
    }


def test_check_fe_inactive_raises_when_fe_active():
    cfg = _base_config()
    cfg["parameters"]["electron"]["fe"]["active"] = True
    with pytest.raises(NotImplementedError):
        mcmc.check_fe_inactive(cfg["parameters"])


def test_check_fe_inactive_passes_when_fe_inactive():
    cfg = _base_config()
    mcmc.check_fe_inactive(cfg["parameters"])  # should not raise


def test_run_mcmc_for_batch_acceptance_rate_near_target(fitted_fixture):
    # With the Laplace-seeded proposal scale, acceptance should land close to the configured target on a
    # well-conditioned problem -- this is the sampler's core correctness check: the RWM kernel and its
    # Robbins-Monro step-size adaptation are actually working, not just running without crashing.
    cfg = copy.deepcopy(fitted_fixture["config"])
    target = 0.234
    cfg["other"]["mcmc"] = {
        "num_steps": 2500,
        "burn_in": 2000,
        "thin": 5,
        "adapt_every": 50,
        "target_accept": target,
        "use_laplace_seed": True,
    }
    key = jax.random.PRNGKey(42)
    samples, static_params, diagnostics = mcmc.run_mcmc_for_batch(
        cfg, fitted_fixture["loss_fn"], fitted_fixture["fitted_weights"][0], fitted_fixture["batch"], key
    )
    acceptance_rate = np.asarray(diagnostics["acceptance_rate"])
    assert acceptance_rate.shape == (cfg["optimizer"]["batch_size"],)
    # loose tolerance: this is a stochastic process with a finite adaptation budget, not an exact solve
    assert np.all(np.abs(acceptance_rate - target) < 0.15)

    leaves = jax.tree_util.tree_leaves(samples)
    assert len(leaves) > 0
    for leaf in leaves:
        assert leaf.shape[1] == cfg["optimizer"]["batch_size"]
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_seed_step_scale_default_has_no_hessian_dependency(fitted_fixture):
    ts_params = fitted_fixture["fitted_weights"][0]
    filter_spec = get_filter_spec(fitted_fixture["config"]["parameters"], ts_params)
    diff_params, _ = eqx.partition(ts_params, filter_spec)
    step_scale = mcmc._seed_step_scale_default(diff_params, 0.05)
    for leaf in jax.tree_util.tree_leaves(step_scale):
        assert np.all(np.asarray(leaf) == 0.05)


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


def test_run_mcmc_for_fit_batches_matches_manual_loop_with_multiple_fit_batches(fitted_fixture):
    # run_mcmc_for_fit_batches seeds the Laplace step scale sequentially, one fit-batch at a time, via
    # _seed_step_scale, *before* vmapping the rest of run_mcmc_for_batch across every fit-batch (see
    # _seed_step_scale's docstring: fusing that Hessian computation into the fit-batch vmap itself has
    # been observed to multiply its memory cost by the fit-batch count on real multi-lineout shots).
    # fitted_fixture's own config only ever produces a single fit-batch, so this test builds its own
    # 3-fit-batch fit (reusing the same tiny dataset by wrapping indices) specifically to exercise that
    # n_fit_batches > 1 path, and checks the vmapped result is bit-identical to manually looping
    # run_mcmc_for_batch per fit-batch with the same per-batch PRNG key and a precomputed step_scale --
    # i.e. the precompute-then-vmap refactor changes *how* this is computed, not the result.
    cfg = copy.deepcopy(fitted_fixture["config"])
    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
    batch_size = cfg["optimizer"]["batch_size"]
    n_fit_batches = 3
    n_lineouts = max(len(all_data["e_data"]), len(all_data["i_data"]))
    sample_indices = np.arange(n_fit_batches * batch_size) % n_lineouts

    with mlflow.start_run():
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, n_fit_batches)
    assert len(fitted_weights) == n_fit_batches

    batch_indices = np.reshape(sample_indices, (-1, batch_size))
    background_subtract = cfg["data"]["background"]["bg_subtract"]
    batch_list = [build_batch(all_data, batch_indices[i], background_subtract) for i in range(n_fit_batches)]

    cfg["other"]["mcmc"] = {
        "num_steps": 40, "burn_in": 20, "thin": 2, "adapt_every": 10,
        "use_laplace_seed": True, "adapt_shape": False,
    }
    key = jax.random.PRNGKey(7)

    samples_vmap, _, diag_vmap = mcmc.run_mcmc_for_fit_batches(cfg, loss_fn, fitted_weights, batch_list, key)
    leaves = jax.tree_util.tree_leaves(samples_vmap)
    assert len(leaves) > 0
    for leaf in leaves:
        assert leaf.shape[0] == n_fit_batches
        assert np.all(np.isfinite(np.asarray(leaf)))

    keys = jax.random.split(key, n_fit_batches)
    manual_samples_list, manual_accept_list = [], []
    for i in range(n_fit_batches):
        s, _, diag_i = mcmc.run_mcmc_for_batch(cfg, loss_fn, fitted_weights[i], batch_list[i], keys[i])
        manual_samples_list.append(s)
        manual_accept_list.append(diag_i["acceptance_rate"])
    manual_samples = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_samples_list)
    manual_accept = jnp.stack(manual_accept_list, axis=0)

    manual_leaves = jax.tree_util.tree_leaves(manual_samples)
    assert len(leaves) == len(manual_leaves)
    for a, b in zip(leaves, manual_leaves):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(diag_vmap["acceptance_rate"]), np.asarray(manual_accept), rtol=1e-10)


def test_seed_step_scale_from_laplace_matches_full_hessian_diagonal(fitted_fixture):
    # _seed_step_scale_from_laplace computes each active leaf's diagonal Hessian entry via
    # jvp(grad(loss wrt that leaf alone), ones) -- one extra forward-mode sweep per leaf -- instead of
    # eqx.filter_hessian's full Hessian (which needs every leaf's tangent direction live simultaneously,
    # and has been observed to attempt a large allocation on a real multi-lineout fit even restricted to
    # diff_params and one fit-batch at a time). This is only valid because every off-diagonal entry
    # (cross-lineout *and* cross-parameter) is structurally zero for this model -- verify that
    # assumption directly against the ground-truth full Hessian, and that the two approaches produce
    # numerically identical step scales end-to-end.
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["parameters"]["ion-1"]["Z"]["active"] = True  # exercise more than just electron/general leaves
    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
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
    for i, row in enumerate(rows):
        row_leaves = jax.tree_util.tree_leaves(row)
        assert len(row_leaves) == n
        for j, block in enumerate(row_leaves):
            block = np.asarray(block)
            off_diagonal = block - np.diag(np.diag(block))
            assert np.allclose(off_diagonal, 0.0, atol=1e-6), (
                f"leaf {i} x leaf {j} Hessian block has nonzero off-diagonal entries -- the "
                "grad+jvp(ones) diagonal trick is not valid for this model"
            )

    step_scale = mcmc._seed_step_scale_from_laplace(loss_fn, static_params, batch, diff_params, fallback_scale=0.1)
    step_scale_leaves = jax.tree_util.tree_leaves(step_scale)
    assert len(step_scale_leaves) == n
    for i, row in enumerate(rows):
        row_leaves = jax.tree_util.tree_leaves(row)
        h_ii = np.diagonal(np.asarray(row_leaves[i]))
        expected_var = np.where(h_ii > 0, 1.0 / h_ii, np.nan)
        expected_scale = (2.38 / np.sqrt(n)) * np.sqrt(expected_var)
        valid = np.isfinite(expected_scale) & (expected_scale > 0)
        expected_scale = np.where(valid, expected_scale, 0.1)
        np.testing.assert_allclose(np.asarray(step_scale_leaves[i]), expected_scale, rtol=1e-6, atol=1e-8)


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
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 8, "gain_sigma": 0.0, "EPWDispersion_sigma": 0.0}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(0)
    )
    assert len(draws) == 8
    for config_k, all_data_k in draws:
        assert config_k is cfg
        assert all_data_k is fitted_fixture["all_data"]


def test_calibration_draws_perturb_gain_and_rescale_data(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 5, "gain_sigma": 0.05}
    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(1)
    )
    assert len(draws) == 5
    nominal_gain = cfg["other"]["gain"]
    nominal_e_data = fitted_fixture["all_data"]["e_data"]
    gains = [cfg_k["other"]["gain"] for cfg_k, _ in draws]
    assert len(set(gains)) > 1  # actually different draws, not all collapsed to the nominal value
    for cfg_k, all_data_k in draws:
        expected_scale = nominal_gain / cfg_k["other"]["gain"]
        np.testing.assert_allclose(all_data_k["e_data"], nominal_e_data * expected_scale)


def test_calibration_uncertainty_widens_the_pooled_posterior(fitted_fixture):
    # The whole point of the calibration-draw design: pooling chains run under different calibration
    # realizations should, in expectation, produce a pooled posterior at least as wide as a single chain
    # at the nominal calibration (law of total variance: pooled_var = avg(within-chain var) + between-
    # chain var, and between-chain var >= 0 whenever the draws shift the best fit at all).
    #
    # For gain specifically, that between-chain shift is real but small: LossFunction normalizes each
    # lineout's e_data by its own max (loss_function.py's e_input_norm), which cancels almost all of a
    # pure multiplicative gain perturbation's effect on the recovered amp1 (confirmed by rebuilding a
    # draw's LossFunction from its own perturbed config and getting a bit-for-bit identical result to
    # reusing the nominal one). That leaves this test comparing two noisy std estimates (each from only
    # ~100 post-burn-in samples) whose gap is on the same order as the sampling noise itself -- a single
    # fixed-seed point comparison isn't reliable and did fail for some seeds despite the effect being
    # real and positive on average. Averaging std_with - std_no over several independent (mcmc key,
    # calibration rng) seed pairs is the statistically appropriate fix here, not a bigger gain_sigma
    # (the cancellation above means that wouldn't move the needle much) or much longer chains (would
    # help but is a far more expensive way to buy the same robustness).
    cfg = fitted_fixture["config"]
    sa = fitted_fixture["sa"]
    ts_params = fitted_fixture["fitted_weights"][0]
    batch_size = cfg["optimizer"]["batch_size"]
    mcmc_settings = {"num_steps": 1500, "burn_in": 1000, "thin": 5, "adapt_every": 50, "use_laplace_seed": True}

    def _pooled_amp1_std(num_draws, gain_sigma, mcmc_key_seed, cal_rng_seed):
        cfg_run = copy.deepcopy(cfg)
        cfg_run["other"]["mcmc"] = mcmc_settings
        cfg_run["other"]["calibration_uncertainty"] = {"num_draws": num_draws, "gain_sigma": gain_sigma, "seed": cal_rng_seed}
        draws = mcmc_calibration.draw_calibration_realizations(
            cfg_run, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(cal_rng_seed)
        )
        # Build a real LossFunction for every draw except the one(s) draw_calibration_realizations left
        # untouched (config_k/all_data_k literally the same objects as nominal) -- mirrors
        # mcmc_postprocess.py's reuse_nominal check, not a hardcoded "draw 0" special case (every draw
        # index, including 0, gets its own independent calibration perturbation).
        from tsadar.inverse.loss_function import LossFunction

        loss_fns = []
        for cfg_k, all_data_k in draws:
            reuse_nominal = cfg_k is cfg_run and all_data_k is fitted_fixture["all_data"]
            if reuse_nominal:
                loss_fns.append(fitted_fixture["loss_fn"])
            else:
                sample = {k: v[:batch_size] for k, v in all_data_k.items()}
                sample = {
                    "noise_e": all_data_k["noiseE"][:batch_size],
                    "noise_i": all_data_k["noiseI"][:batch_size],
                } | sample
                loss_fns.append(LossFunction(cfg_k, sa, sample))

        inds = np.arange(batch_size)
        batches = [[build_batch(all_data_k, inds, cfg["data"]["background"]["bg_subtract"])] for _, all_data_k in draws]
        key = jax.random.PRNGKey(mcmc_key_seed)
        pooled, static_params, _, _ = mcmc.run_mcmc_pooled(cfg_run, loss_fns, [ts_params], batches, key)
        filter_spec = get_filter_spec(cfg_run["parameters"], ts_params)
        static_i = jax.tree_util.tree_map(lambda x: x[0], eqx.filter(static_params, eqx.is_array))
        static_i = eqx.combine(static_i, eqx.filter(static_params, eqx.is_array, inverse=True))
        diff_i = jax.tree_util.tree_map(lambda x: x[0], pooled)

        def _unnorm(dp):
            return eqx.combine(static_i, dp).get_unnormed_params()

        physical = eqx.filter_vmap(_unnorm)(diff_i)
        return float(np.std(np.asarray(physical["general"]["amp1"])[:, 0]))

    # Same set of shapes (num_steps/burn_in/thin/adapt_every) every repeat, so _run_window's filter_jit
    # cache is warmed once and every further repeat is cheap -- only the PRNG/calibration seeds vary.
    n_repeats = 5
    gaps = []
    for i in range(n_repeats):
        std_no = _pooled_amp1_std(num_draws=1, gain_sigma=0.0, mcmc_key_seed=100 + i, cal_rng_seed=200 + i)
        std_with = _pooled_amp1_std(num_draws=4, gain_sigma=0.2, mcmc_key_seed=100 + i, cal_rng_seed=200 + i)
        gaps.append(std_with - std_no)

    mean_gap = float(np.mean(gaps))
    assert mean_gap > 0, (
        f"calibration uncertainty should widen the pooled posterior on average across independent seeds; "
        f"got mean gap {mean_gap:.6f} over {n_repeats} repeats: {gaps}"
    )


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


def test_run_mcmc_pooled_reports_r_hat_with_multiple_chains(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["mcmc"] = {
        "num_steps": 300, "burn_in": 200, "thin": 5, "adapt_every": 20,
        "use_laplace_seed": True, "init_dispersion_factor": 3.0,
    }
    cfg["other"]["calibration_uncertainty"] = {"num_draws": 3, "seed": 5}
    ts_params = fitted_fixture["fitted_weights"][0]
    batch_size = cfg["optimizer"]["batch_size"]

    draws = mcmc_calibration.draw_calibration_realizations(
        cfg, fitted_fixture["all_data"], fitted_fixture["all_axes"], np.random.default_rng(5)
    )
    assert len(draws) == 3  # num_draws still drives the chain count with every *_sigma at 0.0

    loss_fns = [fitted_fixture["loss_fn"] for _ in draws]  # every draw shares the identical nominal config/data
    inds = np.arange(batch_size)
    batches = [
        [build_batch(fitted_fixture["all_data"], inds, cfg["data"]["background"]["bg_subtract"])] for _ in draws
    ]

    key = jax.random.PRNGKey(21)
    _, _, _, max_r_hat = mcmc.run_mcmc_pooled(cfg, loss_fns, [ts_params], batches, key)

    assert max_r_hat is not None
    max_r_hat = np.asarray(max_r_hat)
    assert max_r_hat.shape == (1, batch_size)  # one fit-batch
    assert np.all(np.isfinite(max_r_hat))
    assert np.all(max_r_hat >= 1.0 - 1e-6)


def test_run_mcmc_pooled_r_hat_is_none_with_a_single_chain(fitted_fixture):
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["mcmc"] = {"num_steps": 100, "burn_in": 50, "thin": 2, "adapt_every": 10}
    key = jax.random.PRNGKey(0)
    _, _, _, max_r_hat = mcmc.run_mcmc_pooled(
        cfg,
        [fitted_fixture["loss_fn"]],
        [fitted_fixture["fitted_weights"][0]],
        [[fitted_fixture["batch"]]],
        key,
    )
    assert max_r_hat is None
