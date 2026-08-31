import json
import os
import tempfile

import yaml
import mlflow
from flatten_dict import flatten, unflatten
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from tsadar import runner
from tsadar.postprocess_runner import run_postprocess_local, run_postprocess_remote, _extract_run_id, _load_merged_config


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


def test_run_postprocess_local_with_config_yaml():
    # App-originated runs (runner.run_for_app) log a single merged config.yaml, never
    # defaults.yaml/inputs.yaml -- run_postprocess_local must work from that layout too.
    with tempfile.TemporaryDirectory() as td:
        run_id = _fit_and_stage_artifacts(td)

        with open(os.path.join(td, "defaults.yaml"), "r") as fi:
            defaults_cfg = yaml.safe_load(fi)
        with open(os.path.join(td, "inputs.yaml"), "r") as fi:
            inputs_cfg = yaml.safe_load(fi)
        merged = flatten(defaults_cfg)
        merged.update(flatten(inputs_cfg))
        os.remove(os.path.join(td, "defaults.yaml"))
        os.remove(os.path.join(td, "inputs.yaml"))
        with open(os.path.join(td, "config.yaml"), "w") as fi:
            yaml.dump(unflatten(merged), fi)

        final_params = run_postprocess_local(td)
    _assert_sane_final_params(final_params)


def test_load_merged_config_prefers_config_yaml_when_present(tmp_path):
    cfg = {"a": 1, "nested": {"b": 2}}
    with open(tmp_path / "config.yaml", "w") as fi:
        yaml.dump(cfg, fi)
    # a stray defaults.yaml/inputs.yaml pair should be ignored once config.yaml exists
    with open(tmp_path / "defaults.yaml", "w") as fi:
        yaml.dump({"a": 999}, fi)

    assert _load_merged_config(str(tmp_path)) == cfg


def test_load_merged_config_falls_back_to_defaults_and_inputs(tmp_path):
    with open(tmp_path / "defaults.yaml", "w") as fi:
        yaml.dump({"a": 1, "b": {"c": 2}}, fi)
    with open(tmp_path / "inputs.yaml", "w") as fi:
        yaml.dump({"b": {"c": 3}}, fi)

    merged = _load_merged_config(str(tmp_path))

    assert merged["a"] == 1
    assert merged["b"]["c"] == 3  # inputs overrides defaults


def test_load_merged_config_applies_checkpoint_refinement_metadata(tmp_path):
    with open(tmp_path / "config.yaml", "w") as fi:
        yaml.dump({"optimizer": {"num_mins": 4}}, fi)
    with open(tmp_path / "checkpoint_metadata.json", "w") as fi:
        json.dump({"format_version": 1, "angular_refinements": 1}, fi)

    config = _load_merged_config(str(tmp_path))

    assert config["optimizer"]["checkpoint_refinements"] == 1


def test_run_postprocess_remote():
    with tempfile.TemporaryDirectory() as td:
        run_id = _fit_and_stage_artifacts(td)
    # exercises the same mlflow.artifacts.download_artifacts-based path a real continuum.ergodic.io run
    # would use, just against the local file-based tracking store this test session already uses.
    final_params = run_postprocess_remote(run_id)
    _assert_sane_final_params(final_params)


def test_run_postprocess_remote_with_config_yaml():
    # App-originated runs (runner.run_for_app) only ever log config.yaml, never defaults.yaml/inputs.yaml
    # -- run_postprocess_remote's config.yaml-first probe must find and prefer it.
    with tempfile.TemporaryDirectory() as td:
        run_id = _fit_and_stage_artifacts(td)

        with open(os.path.join(td, "defaults.yaml"), "r") as fi:
            defaults_cfg = yaml.safe_load(fi)
        with open(os.path.join(td, "inputs.yaml"), "r") as fi:
            inputs_cfg = yaml.safe_load(fi)
        merged = flatten(defaults_cfg)
        merged.update(flatten(inputs_cfg))
        config_only_dir = os.path.join(td, "config_only")
        os.makedirs(config_only_dir)
        with open(os.path.join(config_only_dir, "config.yaml"), "w") as fi:
            yaml.dump(unflatten(merged), fi)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(config_only_dir)

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
