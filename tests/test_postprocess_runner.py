import os
import tempfile

import yaml
import mlflow
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar import runner
from tsadar.postprocess_runner import run_postprocess_local, run_postprocess_remote, _extract_run_id


def _assert_sane_final_params(final_params):
    assert len(final_params["Te_electron"]) == 2
    for i in range(2):
        assert 0.0 < final_params["Te_electron"][i] < 5.0
        assert 0.0 < final_params["ne_electron"][i] < 1.0
        assert 0.0 < final_params["m_electron"][i] < 10.0


def _fit_and_stage_artifacts(td: str) -> str:
    # Regression fixture shared by both entry-point tests below: runs a real fit through the same
    # runner.run() entry point a real `python run_tsadar.py --cfg ...` invocation would use (rather than
    # hand-rolling the mlflow/config setup here), so defaults.yaml/inputs.yaml/fitted_weights.eqx all end
    # up logged to the run exactly like a production run's artifacts would. `td` doubles as `cfg_path`
    # (runner.run expects literally-named defaults.yaml/inputs.yaml in it) and, after downloading
    # fitted_weights.eqx into it below, as a stand-in for a local copy of the run's artifact folder.
    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults_cfg = yaml.safe_load(fi)
    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs_cfg = yaml.safe_load(fi)
    inputs_cfg.setdefault("mlflow", {"experiment": "tsadar-tests", "run": "test_postprocess_runner"})

    with open(os.path.join(td, "defaults.yaml"), "w") as fi:
        yaml.dump(defaults_cfg, fi)
    with open(os.path.join(td, "inputs.yaml"), "w") as fi:
        yaml.dump(inputs_cfg, fi)

    run_id = runner.run(td, mode="fit")

    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="fitted_weights.eqx", dst_path=td)
    return run_id


def test_run_postprocess_local():
    with tempfile.TemporaryDirectory() as td:
        _fit_and_stage_artifacts(td)
        final_params = run_postprocess_local(td)
    _assert_sane_final_params(final_params)


def test_run_postprocess_remote():
    with tempfile.TemporaryDirectory() as td:
        run_id = _fit_and_stage_artifacts(td)
    # exercises the same mlflow.artifacts.download_artifacts-based path a real continuum.ergodic.io run
    # would use, just against the local file-based tracking store this test session already uses.
    final_params = run_postprocess_remote(run_id)
    _assert_sane_final_params(final_params)


def test_extract_run_id():
    run_id = "0123456789abcdef0123456789abcdef"
    assert _extract_run_id(run_id) == run_id
    assert _extract_run_id(f"  {run_id}  ") == run_id
    assert (
        _extract_run_id(f"https://continuum.ergodic.io/experiments/#/experiments/12/runs/{run_id}") == run_id
    )

    import pytest

    with pytest.raises(ValueError):
        _extract_run_id("not-a-run-id-or-url")


if __name__ == "__main__":
    test_run_postprocess_local()
    test_run_postprocess_remote()
    test_extract_run_id()
