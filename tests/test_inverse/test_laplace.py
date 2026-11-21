import copy

import equinox as eqx
import jax
import mlflow
import numpy as np
import pytest
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.inverse.loops import build_batch, one_d_loop, unbatch_fitted_params
from tsadar.inverse.postprocess.laplace import (
    get_sandwich_sigmas,
    get_sigmas,
    recalculate_with_chosen_weights,
    sandwich_covariance,
    sigma_method,
)

from tsverify.cases import HYDROGEN
from tsverify.oracle import Plasma

from ..physics import harness
from .shared import base_config


@pytest.mark.parametrize("method", ["laplace", "sandwich"])
def test_calc_sigmas_does_not_fall_back_and_is_fast(fitted_fixture, method):
    # the Hessian is taken with respect to diff_params only, so calc_sigma must complete quickly and
    # must not fall back to calc_sigma=False
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["other"]["sigma_method"] = method
    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
    loss_fn = fitted_fixture["loss_fn"]
    fitted_weights = fitted_fixture["fitted_weights"]
    sample_indices = fitted_fixture["sample_indices"]

    _, num_params = unbatch_fitted_params(cfg, fitted_weights)

    losses, sqdevs, fits, sigmas = recalculate_with_chosen_weights(
        cfg, sa, sample_indices, all_data, loss_fn, True, fitted_weights, num_params
    )

    assert sigmas is not None, "calc_sigma silently fell back to False"
    assert sigmas.shape == (len(sample_indices), num_params)
    assert np.all(np.isfinite(sigmas))


def test_h_loss_wrt_params_per_lineout_matches_full_hessian_diagonal(fitted_fixture):
    # against eqx.filter_hessian: the cross-lineout blocks are zero for every leaf pair, and
    # h_loss_wrt_params_per_lineout reproduces the lineout diagonal of the dense Hessian
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["parameters"]["ion-1"]["Z"]["active"] = True  # exercise a genuine cross-parameter (Te/ne vs Z) pair
    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
    sample_indices = fitted_fixture["sample_indices"]
    batch_size = cfg["optimizer"]["batch_size"]

    with mlflow.start_run():
        num_batches = len(sample_indices) // batch_size or 1
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, num_batches)

    ts_params0 = fitted_weights[0]
    filter_spec = get_filter_spec(cfg["parameters"], ts_params0)
    diff_params0, static_params0 = eqx.partition(ts_params0, filter_spec)
    batch0 = build_batch(all_data, sample_indices[:batch_size], cfg["data"]["background"]["bg_subtract"])

    def _nll_of_diff(dp):
        weights = eqx.combine(static_params0, dp)
        return loss_fn.neg_log_likelihood(weights, batch0, per_lineout=False)

    full_hess = eqx.filter_hessian(_nll_of_diff)(diff_params0)
    target_structure = jax.tree_util.tree_structure(diff_params0)
    dense_rows = jax.tree_util.tree_leaves(
        full_hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure
    )
    n = len(jax.tree_util.tree_leaves(diff_params0))
    assert len(dense_rows) == n

    per_lineout_hess = loss_fn.h_loss_wrt_params_per_lineout(diff_params0, static_params0, batch0)
    per_lineout_rows = jax.tree_util.tree_leaves(
        per_lineout_hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure
    )
    assert len(per_lineout_rows) == n

    for a, dense_row in enumerate(dense_rows):
        dense_blocks = jax.tree_util.tree_leaves(dense_row)
        per_lineout_blocks = jax.tree_util.tree_leaves(per_lineout_rows[a])
        assert len(dense_blocks) == n
        assert len(per_lineout_blocks) == n
        for b in range(n):
            dense_block = np.asarray(dense_blocks[b])
            off_diagonal = dense_block - np.diag(np.diag(dense_block))
            assert np.allclose(off_diagonal, 0.0, atol=1e-6), (
                f"leaf {a} x leaf {b} Hessian block has nonzero off-diagonal (cross-lineout) entries -- "
                "h_loss_wrt_params_per_lineout's diagonal-only shortcut is not valid for this model"
            )
            np.testing.assert_allclose(
                np.asarray(per_lineout_blocks[b]), np.diagonal(dense_block), rtol=1e-6, atol=1e-8
            )


def test_get_sigmas_raises_when_fe_active():
    # an active electron distribution function must raise rather than be mis-computed
    fitted_params = {"electron": {"Te": None, "m": None}, "general": {}, "ion-1": {}}
    with pytest.raises(NotImplementedError):
        get_sigmas(None, None, None, fitted_params, batch_size=2)


MIXED_FREE = (("electron", "ne"), ("electron", "Te"), ("general", "amp1"), ("ion-1", "Ti"))


def _physical_columns(params, columns, lineout):
    unnormed = params.get_unnormed_params()
    return np.array([np.asarray(unnormed[s][k]).ravel()[lineout] for s, k in columns])


def test_get_sigmas_propagates_a_known_hessian_through_mixed_parameters():
    cfg = harness.diagnostic_config(Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), harness.EPW_DETECTOR)
    p = cfg["parameters"]
    for species, key in MIXED_FREE:
        p[species][key]["active"] = True
    params = ThomsonParams(p, num_params=2, activate=True)
    diff, static = eqx.partition(params, get_filter_spec(p, params))
    diff = jax.tree_util.tree_map(lambda x: x + np.array([0.0, 0.4]), diff)
    leaves, treedef = jax.tree_util.tree_flatten(diff)
    n = len(leaves)

    rng = np.random.default_rng(3)
    hessians = []
    for _ in range(2):
        a = rng.normal(size=(n, n))
        hessians.append(a @ a.T + n * np.diag(10.0 ** rng.uniform(-1, 2, n)))
    hess = jax.tree_util.tree_unflatten(
        treedef,
        [
            jax.tree_util.tree_unflatten(treedef, [np.array([h[k1, k2] for h in hessians]) for k2 in range(n)])
            for k1 in range(n)
        ],
    )

    fitted, _ = eqx.combine(static, diff).get_fitted_params(p)
    columns = [(species, key) for species, keys in fitted.items() for key in keys if key not in ("fe", "f", "flm")]
    assert sorted(columns) == sorted(MIXED_FREE)
    sigmas = get_sigmas(hess, diff, static, fitted, batch_size=2)

    step = 1.0e-6
    for i, h in enumerate(hessians):
        jac = np.zeros((len(columns), n))
        for k in range(n):
            shifted = []
            for sign in (1.0, -1.0):
                bumped = [leaf.at[i].add(sign * step) if j == k else leaf for j, leaf in enumerate(leaves)]
                weights = eqx.combine(static, jax.tree_util.tree_unflatten(treedef, bumped))
                shifted.append(_physical_columns(weights, columns, i))
            jac[:, k] = (shifted[0] - shifted[1]) / (2 * step)
        expected = np.sqrt(np.diag(jac @ (2.0 * np.linalg.inv(h)) @ jac.T))
        np.testing.assert_allclose(sigmas[i], expected, rtol=1e-6)
    assert np.all(sigmas[0] != sigmas[1])


def test_a_failed_sigma_calculation_is_reported_as_nan(fitted_fixture, monkeypatch):
    from tsadar.inverse.postprocess import laplace

    def _fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(laplace, "calc_sigmas", _fail)
    cfg = fitted_fixture["config"]
    fitted_weights = fitted_fixture["fitted_weights"]
    _, num_params = unbatch_fitted_params(cfg, fitted_weights)

    _, _, _, sigmas = recalculate_with_chosen_weights(
        cfg, fitted_fixture["sa"], fitted_fixture["sample_indices"], fitted_fixture["all_data"],
        fitted_fixture["loss_fn"], True, fitted_weights, num_params,
    )

    assert sigmas.shape == (len(fitted_fixture["sample_indices"]), num_params)
    assert np.all(np.isnan(sigmas))


def _hat_diagonal(j):
    return np.einsum("np,pq,nq->n", j, np.linalg.inv(j.T @ j), j)


def test_sandwich_covariance_of_leverage_scaled_residuals_is_the_inverse_normal_matrix():
    rng = np.random.default_rng(7)
    jac = rng.normal(size=(3, 40, 4))
    residual = np.stack([2.5 * (1.0 - _hat_diagonal(j)) for j in jac])
    covariance = sandwich_covariance(residual, jac)
    for c, j in zip(covariance, jac):
        np.testing.assert_allclose(c, 2.5**2 * np.linalg.inv(j.T @ j), rtol=1e-12)
    np.testing.assert_allclose(sandwich_covariance(7.0 * residual, 7.0 * jac), covariance, rtol=1e-12)


def test_get_sandwich_sigmas_matches_get_sigmas_for_the_same_covariance():
    cfg = harness.diagnostic_config(Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), harness.EPW_DETECTOR)
    p = cfg["parameters"]
    for species, key in MIXED_FREE:
        p[species][key]["active"] = True
    params = ThomsonParams(p, num_params=2, activate=True)
    diff, static = eqx.partition(params, get_filter_spec(p, params))
    diff = jax.tree_util.tree_map(lambda x: x + np.array([0.0, 0.4]), diff)
    _, treedef = jax.tree_util.tree_flatten(diff)
    n = len(jax.tree_util.tree_leaves(diff))

    rng = np.random.default_rng(11)
    jac = rng.normal(size=(2, 50, n)) * 10.0 ** rng.uniform(-1, 1, n)
    residual = np.stack([1.0 - _hat_diagonal(j) for j in jac])
    normal = [j.T @ j for j in jac]
    hess = jax.tree_util.tree_unflatten(
        treedef,
        [
            jax.tree_util.tree_unflatten(treedef, [np.array([2.0 * a[k1, k2] for a in normal]) for k2 in range(n)])
            for k1 in range(n)
        ],
    )
    fitted, _ = eqx.combine(static, diff).get_fitted_params(p)
    np.testing.assert_allclose(
        get_sandwich_sigmas(residual, jac, diff, static, fitted, batch_size=2),
        get_sigmas(hess, diff, static, fitted, batch_size=2),
        rtol=1e-10,
    )


def test_l2_residual_jacobian_per_lineout_matches_the_residuals_and_their_autodiff(fitted_fixture):
    cfg = fitted_fixture["config"]
    loss_fn, batch = fitted_fixture["loss_fn"], fitted_fixture["batch"]
    ts_params = fitted_fixture["fitted_weights"][0]
    diff, static = eqx.partition(ts_params, get_filter_spec(cfg["parameters"], ts_params))
    residual, jac = loss_fn.l2_residual_jacobian_per_lineout(diff, static, batch)
    batch_size = cfg["optimizer"]["batch_size"]
    leaves, treedef = jax.tree_util.tree_flatten(diff)
    assert residual.shape[0] == batch_size and jac.shape == residual.shape + (len(leaves),)

    flat, _ = loss_fn.residuals(diff, static, batch)
    np.testing.assert_allclose(np.sum(np.asarray(residual) ** 2), np.sum(np.asarray(flat) ** 2), rtol=1e-10)

    def rows_of(stacked):
        weights = eqx.combine(static, jax.tree_util.tree_unflatten(treedef, list(stacked)))
        parts, _ = loss_fn._l2_residual_parts(weights, batch)
        return jax.numpy.concatenate(parts, axis=-1)

    residual, jac = np.asarray(residual), np.asarray(jac)
    np.testing.assert_allclose(residual, np.asarray(rows_of(jax.numpy.stack(leaves))), atol=1e-9 * np.abs(residual).max())
    dense = np.asarray(jax.jacfwd(rows_of)(jax.numpy.stack(leaves)))
    scale = np.abs(dense).max()
    for i in range(batch_size):
        for k in range(batch_size):
            if k != i:
                np.testing.assert_allclose(dense[i, :, :, k], 0.0, atol=1e-12 * scale)
        np.testing.assert_allclose(jac[i], dense[i, :, :, i], atol=1e-9 * scale)


@pytest.mark.parametrize(
    "loss_method, spectype, requested, expected",
    [
        ("l2", "temporal", "auto", "sandwich"),
        ("covar", "temporal", "auto", "laplace"),
        ("l2", "angular_full", "auto", "laplace"),
        ("l2", "temporal", "laplace", "laplace"),
        ("poisson", "temporal", "laplace", "laplace"),
    ],
)
def test_sigma_method_selection(loss_method, spectype, requested, expected):
    cfg = {"optimizer": {"loss_method": loss_method}, "other": {"extraoptions": {"spectype": spectype}}}
    cfg["other"]["sigma_method"] = requested
    assert sigma_method(cfg) == expected


@pytest.mark.parametrize("loss_method, requested", [("covar", "sandwich"), ("l2", "hessian")])
def test_sigma_method_rejects_unsupported_requests(loss_method, requested):
    cfg = {"optimizer": {"loss_method": loss_method}, "other": {"extraoptions": {"spectype": "temporal"}}}
    cfg["other"]["sigma_method"] = requested
    with pytest.raises(ValueError):
        sigma_method(cfg)
