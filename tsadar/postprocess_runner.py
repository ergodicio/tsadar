"""Standalone postprocessor entry point: reruns postprocess() on an already-completed fit's saved results,
loaded either from a local run directory or a remote MLflow run (by id or URL), without redoing the fit.

_reconstruct_fit_state is factored out so mcmc_postprocess_runner.py can reuse the exact same
"reconstruct config/data/skeleton, deserialize fitted_weights.eqx, rebuild LossFunction" logic for the
alternate MCMC uncertainty postprocessor, without duplicating it.
"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

import equinox as eqx
import mlflow
import numpy as np
import yaml
from flatten_dict import flatten, unflatten

from .core.modules.ts_params import ThomsonParams
from .inverse import postprocess
from .inverse.fitter import _validate_inputs_, load_data_for_fitting
from .inverse.loops import advance_refinement_shape, apply_ang_res_unit, unbatch_fitted_params
from .inverse.loss_function import LossFunction
from .utils import misc


@dataclass
class _FitState:
    """Everything a postprocessor (Laplace or MCMC) needs to replay against a saved fit, without
    re-fitting. See _reconstruct_fit_state."""

    config: Dict
    all_data: Dict
    sa: object
    all_axes: Dict
    sample_indices: np.ndarray
    is_angular: bool
    fitted_weights: object
    all_params: Optional[Dict]
    num_params: Optional[int]
    loss_fn: LossFunction

# mlflow's UI route for a specific run: .../experiments/<experiment_id>/runs/<run_id>
_RUN_URL_RE = re.compile(r"experiments/([0-9]+)/runs/([0-9a-f]{32})")
_BARE_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _extract_run_id(run_id_or_url: str) -> str:
    """
    Accepts either a bare mlflow run id or a full run URL (e.g. from continuum.ergodic.io) and returns
    just the run id.
    """
    run_id_or_url = run_id_or_url.strip()
    if _BARE_RUN_ID_RE.match(run_id_or_url):
        return run_id_or_url

    match = _RUN_URL_RE.search(run_id_or_url)
    if match:
        return match.group(2)

    raise ValueError(
        f"Could not extract an mlflow run id from {run_id_or_url!r}. Expected either a bare 32-character "
        "hex run id, or a run URL containing '.../experiments/<experiment_id>/runs/<run_id>'."
    )


def _load_merged_config(dir_path: str) -> Dict:
    """
    Loads the config a fit used, from whichever artifact layout it was saved with: a single config.yaml
    (written by app-originated runs via runner.run_for_app, which never logs defaults.yaml/inputs.yaml) or
    the separate defaults.yaml/inputs.yaml pair (written by runner.load_and_make_folders, used by the
    CLI/cluster entry points).
    """
    config_path = os.path.join(dir_path, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fi:
            return yaml.safe_load(fi)

    all_configs = {}
    for k in ["defaults", "inputs"]:
        with open(os.path.join(dir_path, f"{k}.yaml"), "r") as fi:
            all_configs[k] = yaml.safe_load(fi)
    defaults = flatten(all_configs["defaults"])
    defaults.update(flatten(all_configs["inputs"]))
    return unflatten(defaults)


def _reconstruct_fit_state(config: Dict, fitted_weights_path: str) -> _FitState:
    """
    Reconstructs everything a postprocessor needs to replay against a saved fit -- config validation,
    data reload, a ThomsonParams "skeleton" matching the saved checkpoint's shape, deserializing
    fitted_weights.eqx into it, and rebuilding a LossFunction -- without re-fitting.

    Must be called from inside an active mlflow run: load_data_for_fitting can trigger
    mlflow.log_artifacts calls (e.g. the data visualizer), and mlflow's fluent API implicitly opens its
    own run for those if none is already active, which would collide with a run started by the caller
    right after this returns.

    Args:
        config (Dict): The exact (merged) config the original fit used.
        fitted_weights_path (str): Local path to a fitted_weights.eqx saved by fitter._save_fit_artifacts.

    Returns:
        _FitState: everything postprocess.postprocess (or an alternate postprocessor) needs.
    """
    config = _validate_inputs_(config)
    all_data, sa, all_axes = load_data_for_fitting(config)
    sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))

    is_angular = "angular" in config["other"]["extraoptions"]["spectype"]
    if is_angular:
        # Both mutations mirror side effects multirun_angular_optax applies to config during the
        # original fit, which the freshly-loaded config here never went through: the lineout
        # start/end conversion (needed so the batch built from all_data below matches the range
        # actually fit) and, if multiple minimizations ran, replaying the nvx/window-length growth
        # (needed so the ThomsonParams skeleton has the same shape as the saved checkpoint).
        apply_ang_res_unit(config)
        for _ in range(config["optimizer"]["num_mins"] - 1):
            advance_refinement_shape(config)
        skeleton = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    else:
        num_batches = len(sample_indices) // config["optimizer"]["batch_size"] or 1
        skeleton = [
            ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=True)
            for _ in range(num_batches)
        ]
    fitted_weights = eqx.tree_deserialise_leaves(fitted_weights_path, skeleton)

    if is_angular:
        all_params, num_params = None, None
    else:
        all_params, num_params = unbatch_fitted_params(config, fitted_weights)

    sample = {k: v[: config["optimizer"]["batch_size"]] for k, v in all_data.items()}
    sample = {
        "noise_e": all_data["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": all_data["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    loss_fn = LossFunction(config, sa, sample)

    return _FitState(
        config=config,
        all_data=all_data,
        sa=sa,
        all_axes=all_axes,
        sample_indices=sample_indices,
        is_angular=is_angular,
        fitted_weights=fitted_weights,
        all_params=all_params,
        num_params=num_params,
        loss_fn=loss_fn,
    )


def run_postprocess(config: Dict, fitted_weights_path: str, source_run_id: Optional[str] = None) -> Dict:
    """
    Reconstructs everything postprocess.postprocess() needs from a saved config + fitted_weights.eqx
    (rather than re-fitting), and runs it inside a brand-new mlflow run - this never resumes or mutates
    the run the fit originally came from, it only reads its artifacts.

    Args:
        config (Dict): The exact (merged) config the original fit used.
        fitted_weights_path (str): Local path to a fitted_weights.eqx saved by fitter._save_fit_artifacts.
        source_run_id (Optional[str]): The mlflow run id the artifacts came from, if any - logged as a tag
            on the new run for traceability, but the source run itself is never touched.
    Returns:
        Dict: The final_params produced by postprocess.postprocess.
    """
    mlflow_cfg = config.get("mlflow", {})
    if "experiment" in mlflow_cfg:
        mlflow.set_experiment(mlflow_cfg["experiment"])
    run_name = f"{mlflow_cfg['run']} (postprocess)" if "run" in mlflow_cfg else None

    with mlflow.start_run(run_name=run_name):
        if source_run_id is not None:
            mlflow.set_tag("source_run_id", source_run_id)
        misc.log_mlflow(config)

        state = _reconstruct_fit_state(config, fitted_weights_path)
        final_params = postprocess.postprocess(
            state.config,
            state.sample_indices,
            state.all_data,
            state.all_axes,
            state.loss_fn,
            state.sa,
            state.fitted_weights,
            state.all_params,
            state.num_params,
        )

    return final_params


def run_postprocess_local(dir_path: str) -> Dict:
    """
    Runs postprocess on a fit whose artifacts already sit in a local directory - e.g. a copy of an mlflow
    run's artifact folder. Accepts either config layout _load_merged_config understands: a single
    config.yaml, or defaults.yaml + inputs.yaml. Either way, fitted_weights.eqx must also be present.
    """
    config = _load_merged_config(dir_path)
    fitted_weights_path = os.path.join(dir_path, "fitted_weights.eqx")
    if not os.path.exists(fitted_weights_path):
        raise FileNotFoundError(
            f"No fitted_weights.eqx found in {dir_path} - this fit may predate that artifact, or "
            "postprocessing/saving may have been disabled for it."
        )
    return run_postprocess(config, fitted_weights_path)


def run_postprocess_remote(run_id_or_url: str) -> Dict:
    """
    Runs postprocess on a fit tracked by mlflow, identified by a bare run id or a run URL (e.g. from
    https://continuum.ergodic.io/experiments/...). Only reads the source run's artifacts - the results of
    this replay are logged to a new run, so the source run's record is left untouched.

    Supports both config artifact layouts: a single config.yaml (app-originated runs, via
    runner.run_for_app) or defaults.yaml + inputs.yaml (CLI/cluster runs, via runner.load_and_make_folders).
    Which one a given run has is only known by trying, since app runs never log defaults.yaml/inputs.yaml
    at all - config.yaml is tried first and, only if that's absent, falls back to the defaults/inputs pair.
    """
    run_id = _extract_run_id(run_id_or_url)

    with tempfile.TemporaryDirectory() as td:
        try:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml", dst_path=td)
            config_fnames = ["config.yaml"]
        except Exception:
            config_fnames = ["defaults.yaml", "inputs.yaml"]

        for fname in config_fnames + ["fitted_weights.eqx"]:
            try:
                mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=fname, dst_path=td)
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not download {fname} from run {run_id}: {e}. If this is fitted_weights.eqx, "
                    "the run may predate that artifact, or postprocessing/saving may have been disabled for it."
                ) from e

        config = _load_merged_config(td)
        fitted_weights_path = os.path.join(td, "fitted_weights.eqx")
        return run_postprocess(config, fitted_weights_path, source_run_id=run_id)
