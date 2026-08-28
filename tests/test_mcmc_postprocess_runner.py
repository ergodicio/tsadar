import os
import tempfile

import numpy as np
import yaml
import mlflow
import xarray as xr
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar import runner
from tsadar.mcmc_postprocess_runner import run_mcmc_postprocess_local, run_mcmc_postprocess_remote
from tsadar.postprocess_runner import _load_merged_config, _reconstruct_fit_state


def _mcmc_test_config():
    # Same fixture test_postprocess_runner.py uses, adapted for the MCMC postprocessor: "fe" must be
    # inactive (see inverse/postprocess/mcmc.py's module docstring), and a short chain keeps the test
    # fast -- correctness of the sampler itself (acceptance rate near target, agreement with the
    # Hessian-based sigma on a well-conditioned problem) is covered by tests/test_inverse/test_mcmc.py,
    # not here; this test is about the artifact-producing wiring end to end.
    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults_cfg = yaml.safe_load(fi)
    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs_cfg = yaml.safe_load(fi)
    inputs_cfg["parameters"]["electron"]["fe"]["active"] = False
    inputs_cfg.setdefault("mlflow", {"experiment": "tsadar-tests", "run": "test_mcmc_postprocess_runner"})
    inputs_cfg["other"]["mcmc"] = {
        "num_steps": 300,
        "burn_in": 200,
        "thin": 4,
        "adapt_every": 20,
        "use_laplace_seed": True,
        "compare_to_laplace": False,
    }
    return defaults_cfg, inputs_cfg


def _fit_and_stage_artifacts(td: str) -> str:
    defaults_cfg, inputs_cfg = _mcmc_test_config()

    with open(os.path.join(td, "defaults.yaml"), "w") as fi:
        yaml.dump(defaults_cfg, fi)
    with open(os.path.join(td, "inputs.yaml"), "w") as fi:
        yaml.dump(inputs_cfg, fi)

    run_id = runner.run(td, mode="fit")

    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="fitted_weights.eqx", dst_path=td)
    return run_id


def test_run_mcmc_postprocess_local():
    with tempfile.TemporaryDirectory() as td:
        _fit_and_stage_artifacts(td)
        final_params = run_mcmc_postprocess_local(td)

    assert len(final_params["Te_electron"]) == 2
    for i in range(2):
        assert 0.0 < final_params["Te_electron"][i] < 5.0
        assert 0.0 < final_params["ne_electron"][i] < 1.0
    assert final_params["mcmc_diagnostics"]["num_calibration_draws"] == 1


def test_mcmc_postprocess_writes_expected_artifacts():
    # Guards the wiring (manifest, sigmas_mcmc.nc, mcmc_covariance.nc all present), the same way
    # test_utils/test_manifest.py::test_postprocess_logs_a_manifest_with_the_artifacts guards
    # postprocess.postprocess -- called directly inside a known run context (rather than through
    # run_mcmc_postprocess_local, which opens its own run whose id isn't otherwise returned) so the
    # produced artifacts can be inspected without needing to rediscover which run they landed in.
    from tsadar.inverse.postprocess.mcmc_postprocess import mcmc_postprocess

    with tempfile.TemporaryDirectory() as td:
        _fit_and_stage_artifacts(td)
        config = _load_merged_config(td)

        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            state = _reconstruct_fit_state(config, os.path.join(td, "fitted_weights.eqx"))
            mcmc_postprocess(
                state.config, state.sample_indices, state.all_data, state.all_axes, state.loss_fn,
                state.sa, state.fitted_weights, state.num_params,
            )
            run_id = run.info.run_id

        logged = {f.path for f in client.list_artifacts(run_id)}
        assert "manifest.json" in logged
        assert "sigmas_mcmc.nc" in logged

        csv_files = {f.path for f in client.list_artifacts(run_id, "csv")}
        assert "csv/learned_parameters.csv" in csv_files

        binary_files = {f.path for f in client.list_artifacts(run_id, "binary")}
        assert "binary/mcmc_covariance.nc" in binary_files

        with tempfile.TemporaryDirectory() as dl:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="binary/mcmc_covariance.nc", dst_path=dl
            )
            with xr.open_dataset(local_path) as ds:
                assert ds["covariance"].shape[0] == 2  # 2 lineouts, matching _assert_sane_final_params elsewhere
                assert ds["covariance"].shape[1] == ds["covariance"].shape[2]


def test_run_mcmc_postprocess_local_applies_overrides():
    # Overriding other.mcmc.save_samples to False (without touching the saved deck on disk) should
    # change what gets written -- proving `overrides` actually drives postprocessing behavior rather
    # than being silently ignored in favor of the run's saved config snapshot.
    with tempfile.TemporaryDirectory() as td:
        _fit_and_stage_artifacts(td)
        run_mcmc_postprocess_local(td, overrides={"other": {"mcmc": {"save_samples": False}}})

    run = mlflow.last_active_run()
    client = mlflow.tracking.MlflowClient()
    binary_files = {f.path for f in client.list_artifacts(run.info.run_id, "binary")}
    assert "binary/mcmc_samples.nc" not in binary_files
    assert "binary/mcmc_covariance.nc" in binary_files


def test_run_mcmc_postprocess_local_reports_r_hat_with_multiple_chains():
    # calibration_uncertainty.num_draws is the unified chain-count knob (see mcmc_calibration.py): with
    # every *_sigma at its default 0.0, these 3 chains differ only by init_dispersion_factor's starting-
    # point perturbation and their own independent MH noise, and R-hat should be computable across them.
    with tempfile.TemporaryDirectory() as td:
        _fit_and_stage_artifacts(td)
        final_params = run_mcmc_postprocess_local(
            td,
            overrides={
                "other": {
                    "calibration_uncertainty": {"num_draws": 3},
                    "mcmc": {"init_dispersion_factor": 3.0},
                }
            },
        )

    assert final_params["mcmc_diagnostics"]["num_calibration_draws"] == 3
    max_r_hat = np.asarray(final_params["mcmc_diagnostics"]["max_r_hat"])
    finite = max_r_hat[np.isfinite(max_r_hat)]
    assert len(finite) == 2  # 2 lineouts, both with active parameters
    assert np.all(finite >= 1.0 - 1e-6)


def test_run_mcmc_postprocess_remote():
    with tempfile.TemporaryDirectory() as td:
        run_id = _fit_and_stage_artifacts(td)
    final_params = run_mcmc_postprocess_remote(run_id)
    assert len(final_params["Te_electron"]) == 2


def test_mcmc_postprocess_raises_on_angular():
    # The angular-fit guard is checked before any data is touched, so this can be tested directly
    # against inverse.postprocess.mcmc_postprocess with a minimal fake config, the same way
    # test_utils/test_manifest.py exercises postprocess.postprocess without a real fit.
    import pytest

    from tsadar.inverse.postprocess.mcmc_postprocess import mcmc_postprocess

    config = {"other": {"extraoptions": {"spectype": "angular_full"}}}
    with pytest.raises(NotImplementedError):
        mcmc_postprocess(config, [], {}, {}, None, None, [], 0)


if __name__ == "__main__":
    test_run_mcmc_postprocess_local()
    test_mcmc_postprocess_writes_expected_artifacts()
    test_run_mcmc_postprocess_remote()
    test_mcmc_postprocess_raises_on_angular()
