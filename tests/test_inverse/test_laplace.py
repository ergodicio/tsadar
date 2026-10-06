import copy
import tempfile

import equinox as eqx
import jax
import mlflow
import numpy as np
import pytest
import yaml
from flatten_dict import flatten, unflatten
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import get_filter_spec
from tsadar.data import prepare
from tsadar.inverse.loops import build_batch, one_d_loop, unbatch_fitted_params
from tsadar.inverse.postprocess.laplace import get_sigmas, recalculate_with_chosen_weights


def _base_config():
    with open("tests/configs/time_test_defaults.yaml") as fi:
        d = yaml.safe_load(fi)
    with open("tests/configs/time_test_inputs.yaml") as fi:
        i = yaml.safe_load(fi)
    flat = flatten(d)
    flat.update(flatten(i))
    cfg = unflatten(flat)
    cfg["parameters"]["electron"]["fe"]["active"] = False
    cfg["data"]["launch_data_visualizer"] = False
    cfg["data"]["lineouts"]["val"] = list(
        range(cfg["data"]["lineouts"]["start"], cfg["data"]["lineouts"]["end"], cfg["data"]["lineouts"]["skip"])
    )
    cfg["optimizer"]["num_epochs"] = 10  # fast fit; these tests only need *a* fitted point, not a good one
    return cfg


@pytest.fixture(scope="module")
def fitted_fixture():
    """Runs one small real fit once and shares it across every test in this file, since re-fitting for
    each test would dominate the file's runtime without adding coverage. Mirrors test_mcmc.py's
    fitted_fixture."""
    cfg = _base_config()
    mlflow.set_tracking_uri(f"sqlite:///{tempfile.mkdtemp()}/mlflow.db")
    mlflow.set_experiment("test-laplace")
    with mlflow.start_run():
        all_data, sa, all_axes = prepare.prepare_data(cfg, cfg["data"]["shotnum"])
        sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
        num_batches = len(sample_indices) // cfg["optimizer"]["batch_size"] or 1
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, num_batches)
    return {
        "config": cfg,
        "all_data": all_data,
        "all_axes": all_axes,
        "sa": sa,
        "loss_fn": loss_fn,
        "fitted_weights": fitted_weights,
        "sample_indices": sample_indices,
    }


def test_calc_sigmas_does_not_fall_back_and_is_fast(fitted_fixture):
    # the Hessian is taken with respect to diff_params only, so calc_sigma must complete quickly and
    # must not fall back to calc_sigma=False
    cfg = fitted_fixture["config"]
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


def test_get_sigmas_matches_independent_recomputation(fitted_fixture):
    # fitted_params order (electron, general, ions) differs from diff_params order (electron, ions,
    # general), so get_sigmas has to permute; compare against an independent recomputation
    cfg = copy.deepcopy(fitted_fixture["config"])
    cfg["parameters"]["ion-1"]["Z"]["active"] = True
    all_data = fitted_fixture["all_data"]
    sa = fitted_fixture["sa"]
    sample_indices = fitted_fixture["sample_indices"]
    batch_size = cfg["optimizer"]["batch_size"]

    with mlflow.start_run():
        num_batches = len(sample_indices) // batch_size or 1
        fitted_weights, _, loss_fn = one_d_loop(cfg, all_data, sa, sample_indices, num_batches)

    all_params, num_params = unbatch_fitted_params(cfg, fitted_weights)
    ordered = [(species, key) for species, params in all_params.items() for key in params.keys()]

    ts_params0 = fitted_weights[0]
    filter_spec = get_filter_spec(cfg["parameters"], ts_params0)
    diff_params0, static_params0 = eqx.partition(ts_params0, filter_spec)
    batch0 = build_batch(all_data, sample_indices[:batch_size], cfg["data"]["background"]["bg_subtract"])
    hess0 = loss_fn.h_loss_wrt_params_per_lineout(diff_params0, static_params0, batch0)

    fitted_params0, _ = ts_params0.get_fitted_params(cfg["parameters"])
    sigmas = get_sigmas(hess0, diff_params0, static_params0, fitted_params0, batch_size)

    def _leaf(tree, species, key):
        nkey = f"normed_{key}" if key != "fract" else key
        if species.startswith("ion-"):
            return getattr(tree.ions[int(species.split("-")[1]) - 1], nkey)
        return getattr(getattr(tree, species), nkey)

    independent = np.zeros((batch_size, len(ordered)))
    for i in range(batch_size):
        temp = np.zeros((len(ordered), len(ordered)))
        for a, (sp1, k1) in enumerate(ordered):
            outer = _leaf(hess0, sp1, k1)
            for b, (sp2, k2) in enumerate(ordered):
                temp[a, b] = np.asarray(_leaf(outer, sp2, k2))[i]
        # physical covariance: J (2 H^-1) J^T, with J from central differences of the parameter transform
        jac = np.zeros((len(ordered), len(ordered)))
        step = 1e-5
        for b, (sp2, k2) in enumerate(ordered):
            nkey = f"normed_{k2}" if k2 != "fract" else k2
            if sp2.startswith("ion-"):
                where = lambda t, _sp=sp2, _k=nkey: getattr(t.ions[int(_sp.split("-")[1]) - 1], _k)
            else:
                where = lambda t, _sp=sp2, _k=nkey: getattr(getattr(t, _sp), _k)
            plus = eqx.tree_at(where, ts_params0, replace_fn=lambda x: x + step).get_unnormed_params()
            minus = eqx.tree_at(where, ts_params0, replace_fn=lambda x: x - step).get_unnormed_params()
            for a, (sp1, k1) in enumerate(ordered):
                jac[a, b] = (np.asarray(plus[sp1][k1])[i] - np.asarray(minus[sp1][k1])[i]) / (2 * step)
        covariance = jac @ (2.0 * np.linalg.inv(temp)) @ jac.T
        independent[i, :] = np.sign(np.diag(covariance)) * np.sqrt(np.abs(np.diag(covariance)))

    np.testing.assert_allclose(sigmas, independent, rtol=1e-5)


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
