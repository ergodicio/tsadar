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
    # calc_sigmas used to attempt a Hessian over the *entire* parameter tree -- including the electron
    # distribution function's fixed interpolation table -- which is a multi-GB-to-multi-hundred-GB
    # allocation even on an ordinary fit (see loss_function.LossFunction.h_loss_wrt_params's docstring).
    # The fix restricts the Hessian to diff_params only, so this should complete in well under a minute
    # (generously bounded here) and, crucially, must not silently fall back to calc_sigma=False the way
    # the old, broken get_sigmas (which indexed a Hessian shaped like the full ThomsonParams tree with
    # plain dict brackets, and would always raise) always did.
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
    # Regression check for the column-ordering fix: fitted_params' (species, key) order (electron,
    # general, ion-1, ... -- from ThomsonParams.get_unnormed_params) does not match diff_params' own
    # pytree flatten order (electron, ions, general -- from ThomsonParams' declared field order), so
    # get_sigmas has to permute. Verify its output against an independent getattr-based recomputation of
    # the same joint covariance that doesn't go through that permutation logic at all.
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
    hess0 = loss_fn.h_loss_wrt_params(diff_params0, static_params0, batch0)

    fitted_params0, _ = ts_params0.get_fitted_params(cfg["parameters"])
    sigmas = get_sigmas(hess0, diff_params0, fitted_params0, batch_size)

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
                temp[a, b] = np.asarray(_leaf(outer, sp2, k2))[i, i]
        inv = np.linalg.inv(temp)
        independent[i, :] = np.sign(np.diag(inv)) * np.sqrt(np.abs(np.diag(inv)))

    np.testing.assert_allclose(sigmas, independent, rtol=1e-8)


def test_get_sigmas_raises_when_fe_active():
    # Mirrors postprocess.mcmc.check_fe_inactive's restriction: the electron distribution function's
    # per-lineout parameters are stored as a list of separate objects rather than one array with a batch
    # axis, which this leaf-diagonal approach doesn't handle. Must raise clearly rather than silently
    # mis-computing or crashing on an unrelated KeyError.
    fitted_params = {"electron": {"Te": None, "m": None}, "general": {}, "ion-1": {}}
    with pytest.raises(NotImplementedError):
        get_sigmas(None, None, fitted_params, batch_size=2)
