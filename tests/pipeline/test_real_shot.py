"""The whole production pipeline on real data, with as few fits as possible.

Shot 101675 (two EPW lineouts) is fitted through ``runner.run``, the entry point that
``run_tsadar.py`` uses, once per optimizer. Every check below reuses those fits:

* the fitted plasma parameters against the known-good fit of this shot;
* the forced refit path, with the two lineouts in separate batches;
* the postprocess entry points (local folder with defaults/inputs or with the app's
  config.yaml, and the remote run id with either layout), which must all reproduce
  the fit;
* MCMC postprocessing with short chains: artifacts, overrides, R-hat across chains,
  and the remote entry point.

The statistical behaviour of the sampler belongs to the inverse physics tests; here
only the wiring is under test, so the chains are short.
"""

import os
import shutil
from types import SimpleNamespace

import mlflow
import numpy as np
import pytest
import xarray as xr
import yaml
from flatten_dict import flatten, unflatten
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar import runner
from tsadar.mcmc_postprocess_runner import run_mcmc_postprocess_local, run_mcmc_postprocess_remote
from tsadar.postprocess_runner import run_postprocess_local, run_postprocess_remote

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")

# Known-good fit of lineout 0. Values of 3/22/25 (lam updated 4/24/26 for the shot-day calibration), with
# rtol as in the original test_1d_data.
KNOWN_GOOD = {
    "amp1_general": (0.733, 1e-1),
    "amp2_general": (0.520, 1e-1),
    "lam_general": (526.5, 5e-3),
    "Te_electron": (0.641, 1e-1),
    "ne_electron": (0.228, 5e-2),
    "m_electron": (3.20, 15e-2),
}

SHORT_MCMC = {
    "num_steps": 60,
    "burn_in": 20,
    "thin": 2,
    "adapt_every": 10,
    "use_laplace_seed": True,
    "compare_to_laplace": False,
}

# The sampler cannot sample an active EDF; condition on the fitted shape instead.
MCMC_OVERRIDES = {"parameters": {"electron": {"fe": {"active": False}}}}


def _deck(method, **other):
    with open(os.path.join(CONFIGS, "time_test_defaults.yaml")) as fi:
        defaults = yaml.safe_load(fi)
    with open(os.path.join(CONFIGS, "time_test_inputs.yaml")) as fi:
        inputs = yaml.safe_load(fi)
    inputs["mlflow"] = {"experiment": "tsadar-tests", "run": f"test_real_shot-{method}"}
    inputs.setdefault("optimizer", {})["method"] = method
    inputs["other"].update(other)
    return defaults, inputs


def _fit(tmp_path_factory, name, method, batch_size=None, **other):
    folder = tmp_path_factory.mktemp(name)
    defaults, inputs = _deck(method, **other)
    if batch_size is not None:
        inputs["optimizer"]["batch_size"] = batch_size
    for file_name, cfg in (("defaults.yaml", defaults), ("inputs.yaml", inputs)):
        with open(folder / file_name, "w") as fi:
            yaml.dump(cfg, fi)
    run_id = runner.run(str(folder), mode="fit")
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="fitted_weights.eqx", dst_path=str(folder))
    return SimpleNamespace(folder=folder, run_id=run_id, params=run_postprocess_local(str(folder)))


@pytest.fixture(scope="module")
def lbfgs(tmp_path_factory):
    return _fit(tmp_path_factory, "lbfgs", "l-bfgs-b", mcmc=dict(SHORT_MCMC))


@pytest.fixture(scope="module")
def lsq_refit(tmp_path_factory):
    # every lineout flagged bad (losses are non-negative), and the two lineouts in separate batches, so the
    # refit of the second reads its start from, and writes to, a different batch
    return _fit(tmp_path_factory, "lsq_refit", "lsq", batch_size=1, refit=True, refit_thresh=-1.0)


def _assert_two_sane_lineouts(params):
    assert len(params["Te_electron"]) == 2
    for i in range(2):
        assert 0.0 < params["Te_electron"][i] < 5.0
        assert 0.0 < params["ne_electron"][i] < 1.0


@pytest.mark.parametrize("run", ["lbfgs", "lsq_refit"])
def test_fit_reproduces_the_known_good_fit(run, request):
    params = request.getfixturevalue(run).params
    _assert_two_sane_lineouts(params)
    for key, (value, rtol) in KNOWN_GOOD.items():
        np.testing.assert_allclose(params[key][0], value, rtol=rtol, err_msg=f"{run}: {key}")


def _assert_same_fit(actual, expected, label):
    for key in KNOWN_GOOD:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, err_msg=f"{label}: {key}")


def test_every_postprocess_entry_point_reproduces_the_fit(lbfgs):
    reference = lbfgs.params  # local folder, defaults.yaml + inputs.yaml

    # app-originated runs log one merged config.yaml instead of defaults/inputs
    with open(lbfgs.folder / "defaults.yaml") as fi:
        merged = flatten(yaml.safe_load(fi))
    with open(lbfgs.folder / "inputs.yaml") as fi:
        merged.update(flatten(yaml.safe_load(fi)))
    config_only = lbfgs.folder / "config_only"
    config_only.mkdir()
    with open(config_only / "config.yaml", "w") as fi:
        yaml.dump(unflatten(merged), fi)
    shutil.copyfile(lbfgs.folder / "fitted_weights.eqx", config_only / "fitted_weights.eqx")
    _assert_same_fit(run_postprocess_local(str(config_only)), reference, "local config.yaml")

    _assert_same_fit(run_postprocess_remote(lbfgs.run_id), reference, "remote defaults/inputs")
    with mlflow.start_run(run_id=lbfgs.run_id):
        mlflow.log_artifact(str(config_only / "config.yaml"))
    _assert_same_fit(run_postprocess_remote(lbfgs.run_id), reference, "remote config.yaml")


def test_mcmc_postprocess_wiring(lbfgs):
    overrides = {
        **MCMC_OVERRIDES,
        "other": {
            "mcmc": {"save_samples": False, "init_dispersion_factor": 3.0},
            "calibration_uncertainty": {"num_draws": 2},
        },
    }
    params = run_mcmc_postprocess_local(str(lbfgs.folder), overrides=overrides)
    run_id = mlflow.last_active_run().info.run_id
    client = mlflow.tracking.MlflowClient()

    _assert_two_sane_lineouts(params)
    diagnostics = params["mcmc_diagnostics"]
    assert diagnostics["num_calibration_draws"] == 2
    r_hat = np.asarray(diagnostics["max_r_hat"])
    assert np.all(r_hat[np.isfinite(r_hat)] >= 1.0 - 1e-6) and np.isfinite(r_hat).sum() == 2

    top = {f.path for f in client.list_artifacts(run_id)}
    assert {"manifest.json", "sigmas_mcmc.nc", "overrides.yaml"} <= top
    assert "csv/learned_parameters.csv" in {f.path for f in client.list_artifacts(run_id, "csv")}
    binary = {f.path for f in client.list_artifacts(run_id, "binary")}
    assert {"binary/mcmc_covariance.nc", "binary/mcmc_reliability.nc"} <= binary
    assert "binary/mcmc_samples.nc" not in binary  # the override reached the postprocessor

    download = lambda path: mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    with open(download("overrides.yaml")) as fi:
        assert yaml.safe_load(fi) == overrides
    covariance = xr.load_dataset(download("binary/mcmc_covariance.nc"))
    sigmas = xr.load_dataset(download("sigmas_mcmc.nc"))
    assert covariance["covariance"].shape[0] == 2
    for a, name in enumerate(covariance["param_i"].values):
        np.testing.assert_allclose(covariance["covariance"].values[:, a, a], sigmas[str(name)].values ** 2, rtol=1e-8)

    remote = run_mcmc_postprocess_remote(lbfgs.run_id, overrides=MCMC_OVERRIDES)
    _assert_two_sane_lineouts(remote)
