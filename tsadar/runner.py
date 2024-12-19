import time, os
from typing import Dict, Tuple

import mlflow, tempfile, yaml
import multiprocessing as mp
from flatten_dict import flatten, unflatten

from .inverse import fitter
from .forward import calc_series
from .utils import misc

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def load_and_make_folders(cfg_path: str) -> Tuple[str, Dict]:
    """
    This is used to queue runs on NERSC

    Args:
        cfg_path:

    Returns:

    """
    all_configs = {}
    basedir = os.path.join(os.getcwd(), f"{cfg_path}")
    for k in ["defaults", "inputs"]:
        with open(f"{os.path.join(basedir, k)}.yaml", "r") as fi:
            all_configs[k] = yaml.safe_load(fi)

    if "mlflow" in all_configs["inputs"].keys():
        experiment = all_configs["inputs"]["mlflow"]["experiment"]
        run_name = all_configs["inputs"]["mlflow"]["run"]

    else:
        experiment = all_configs["defaults"]["mlflow"]["experiment"]
        run_name = all_configs["defaults"]["mlflow"]["run"]

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

    return mlflow_run.info.run_id, all_configs


def run(cfg_path: str, mode: str) -> str:
    """
    Wrapper for lower level runner

    Args:
        cfg_path:
        mode:

    Returns:

    """
    run_id, all_configs = load_and_make_folders(cfg_path)
    defaults = flatten(all_configs["defaults"])
    defaults.update(flatten(all_configs["inputs"]))
    config = unflatten(defaults)
    with mlflow.start_run(run_id=run_id, log_system_metrics=True) as mlflow_run:
        _run_(config, mode=mode)

    return run_id


def run_for_app(run_id: str) -> str:
    with mlflow.start_run(run_id=run_id, log_system_metrics=True) as mlflow_run:
        # download config
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:

            dest_file_path = misc.download_file(f"config.yaml", mlflow_run.info.artifact_uri, temp_path)
            with open(dest_file_path, "r") as fi:
                config = yaml.safe_load(fi)

            if config["data"]["filenames"]["epw"] is not None:
                config["data"]["filenames"]["epw-local"] = misc.download_file(
                    config["data"]["filenames"]["epw"], mlflow_run.info.artifact_uri, temp_path
                )

            if config["data"]["filenames"]["iaw"] is not None:
                config["data"]["filenames"]["iaw-local"] = misc.download_file(
                    config["data"]["filenames"]["iaw"], mlflow_run.info.artifact_uri, temp_path
                )

            _run_(config, mode="fit")

    return mlflow_run.info.run_id


def _run_(config: Dict, mode: str = "fit"):
    """
    Either performs a forward pass or an entire fitting routine

    Relies on mlflow to log parameters, metrics, and artifacts

    Args:
        config:
        mode:

    Returns:

    """
    misc.log_mlflow(config)
    t0 = time.time()
    if mode.casefold() == "fit":
        fit_results, loss = fitter.fit(config=config)
    elif mode == "forward" or mode == "series":
        calc_series.forward_pass(config=config)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")


def run_job(run_id: str, mode: str, nested: bool):
    """
    This is used to run queued runs on NERSC. It picks up the `run_id` and finds that using MLFlow and does the fitting


    Args:
        run_id:
        nested:

    Returns:

    """
    with mlflow.start_run(run_id=run_id, nested=nested) as run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            all_configs = {}
            for k in ["defaults", "inputs"]:
                dest_file_path = misc.download_file(f"{k}.yaml", run.info.artifact_uri, temp_path)
                with open(f"{os.path.join(temp_path, k)}.yaml", "r") as fi:
                    all_configs[k] = yaml.safe_load(fi)
            defaults = flatten(all_configs["defaults"])
            defaults.update(flatten(all_configs["inputs"]))
            config = unflatten(defaults)

        _run_(config, mode)
