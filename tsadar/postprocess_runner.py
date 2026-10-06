"""Standalone postprocessor entry point: reruns postprocess() on an already-completed fit's saved results,
loaded either from a local run directory or a remote MLflow run (by id or URL), without redoing the fit."""
import json
import os
import posixpath
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import equinox as eqx
import mlflow
import numpy as np
import yaml

from .core.modules.ts_params import ThomsonParams
from .inverse import postprocess
from .inverse.fitter import _validate_inputs_, load_data_for_fitting
from .inverse.loops import (
    advance_refinement_shape,
    apply_ang_res_unit,
    build_angular_batch,
    build_batch,
    unbatch_fitted_params,
)
from .inverse.loss_function import LossFunction
from .utils import misc

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


def _resolve_artifact_uri(run_id: str) -> str:
    """
    Resolves an mlflow run's artifact root URI once, so that downloading several artifacts from the same
    run does not re-resolve the run for every file. Raises if the run does not exist.
    """
    return mlflow.get_run(run_id).info.artifact_uri


def _download_run_artifact(base_artifact_uri: str, fname: str, dst_path: str) -> None:
    """
    Downloads one file from a run's artifacts, given that run's artifact root URI (from
    _resolve_artifact_uri). mlflow artifact paths are POSIX-style on every platform.
    """
    mlflow.artifacts.download_artifacts(artifact_uri=posixpath.join(base_artifact_uri, fname), dst_path=dst_path)


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
            config = yaml.safe_load(fi)
    else:
        all_configs = {}
        for k in ["defaults", "inputs"]:
            with open(os.path.join(dir_path, f"{k}.yaml"), "r") as fi:
                all_configs[k] = yaml.safe_load(fi)
        config = misc.merge_defaults_and_inputs(all_configs["defaults"], all_configs["inputs"])

    metadata_path = os.path.join(dir_path, "checkpoint_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        refinements = metadata.get("angular_refinements")
        if refinements is not None:
            config["optimizer"]["checkpoint_refinements"] = int(refinements)
    return config


@dataclass
class ReconstructedFitState:
    """Everything a postprocessor (postprocess.postprocess or mcmc_postprocess.mcmc_postprocess) needs to
    replay against an already-completed fit's saved artifacts, as built by _reconstruct_fit_state."""

    config: Dict
    is_angular: bool
    sample_indices: np.ndarray
    all_data: Dict
    all_axes: Dict
    sa: Any
    fitted_weights: List
    all_params: Optional[Dict]
    num_params: Optional[int]
    loss_fn: LossFunction


# GeneralParams leaves added after tsadar 0.3.0; checkpoints saved before that do not contain them
_LEGACY_ABSENT_GENERAL_LEAVES = (
    "normed_brem_amp",
    "normed_brem_c",
    "brem_amp_scale",
    "brem_amp_shift",
    "brem_c_scale",
    "brem_c_shift",
)


def _load_fitted_weights(fitted_weights_path: str, skeleton):
    """
    Deserializes a fitted_weights.eqx into skeleton (a ThomsonParams, or a list of them). A checkpoint
    saved before the bremsstrahlung parameters existed is loaded into the older layout, and those
    parameters keep the inactive defaults skeleton was built with.
    """
    try:
        return eqx.tree_deserialise_leaves(fitted_weights_path, skeleton)
    except Exception as current_layout_error:

        def _absent(tree):
            params = tree if isinstance(tree, list) else [tree]
            return [getattr(p.general, name) for p in params for name in _LEGACY_ABSENT_GENERAL_LEAVES]

        defaults = _absent(skeleton)
        legacy_skeleton = eqx.tree_at(_absent, skeleton, replace=[None] * len(defaults))
        try:
            loaded = eqx.tree_deserialise_leaves(fitted_weights_path, legacy_skeleton)
        except Exception:
            raise current_layout_error
        return eqx.tree_at(_absent, loaded, replace=defaults, is_leaf=lambda x: x is None)


def _reconstruct_fit_state(config: Dict, fitted_weights_path: str) -> ReconstructedFitState:
    """
    Reconstructs everything a postprocessor needs from a saved config + fitted_weights.eqx (rather than
    re-fitting): the loaded data/axes, the deserialized fitted weights, and a freshly-built LossFunction.
    Must be called from inside an active mlflow run (load_data_for_fitting can trigger
    mlflow.log_artifacts calls, e.g. the data visualizer).

    Args:
        config (Dict): The exact (merged) config the original fit used.
        fitted_weights_path (str): Local path to a fitted_weights.eqx saved by fitter._save_fit_artifacts.
    Returns:
        ReconstructedFitState
    """
    config = _validate_inputs_(config)
    all_data, sa, all_axes = load_data_for_fitting(config)
    sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))

    is_angular = "angular" in config["other"]["extraoptions"]["spectype"]
    if is_angular:
        # replay the config changes multirun_angular_optax makes during the original fit: unbatched
        # fitting, the lineout start/end conversion, and the nvx/window-length growth
        config["optimizer"]["batch_size"] = 1
        apply_ang_res_unit(config)
        checkpoint_refinements = config["optimizer"].get(
            "checkpoint_refinements", config["optimizer"]["num_mins"] - 1
        )
        for _ in range(checkpoint_refinements):
            advance_refinement_shape(config)
        skeleton = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    else:
        num_batches = len(sample_indices) // config["optimizer"]["batch_size"] or 1
        skeleton = [
            ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=True)
            for _ in range(num_batches)
        ]
    fitted_weights = _load_fitted_weights(fitted_weights_path, skeleton)

    if is_angular:
        all_params, num_params = None, None
    else:
        all_params, num_params = unbatch_fitted_params(config, fitted_weights)

    if is_angular:
        # normalize against the lineout-range slice, as the original fit did
        sample = build_angular_batch(config, all_data)
        if isinstance(config["data"]["shotnum"], list):
            sample = sample["b1"]
    else:
        sample = build_batch(
            all_data, np.arange(config["optimizer"]["batch_size"]), config["data"]["background"]["bg_subtract"]
        )
    loss_fn = LossFunction(config, sa, sample)

    return ReconstructedFitState(
        config=config,
        is_angular=is_angular,
        sample_indices=sample_indices,
        all_data=all_data,
        all_axes=all_axes,
        sa=sa,
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

    # Everything below must run inside the new run's context, not before it: load_data_for_fitting can
    # trigger mlflow.log_artifacts calls (e.g. the data visualizer), and mlflow's fluent API implicitly
    # opens its own run for those if none is already active - which would then collide with start_run below.
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


def run_postprocess_local(dir_path: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Runs postprocess on a fit whose artifacts already sit in a local directory - e.g. a copy of an mlflow
    run's artifact folder. Accepts either config layout _load_merged_config understands: a single
    config.yaml, or defaults.yaml + inputs.yaml. Either way, fitted_weights.eqx must also be present.

    Args:
        dir_path: path to the artifact directory.
        overrides: optional partial config (same nesting as inputs.yaml) deep-merged on top of the saved
            config in memory. Only override fields that affect postprocessing or plotting; fields the
            reconstruction depends on (data.lineouts, optimizer.batch_size, parameters.*.active, ...) must
            match the saved fitted_weights.eqx.
    """
    config = _load_merged_config(dir_path)
    if overrides:
        config = misc.merge_defaults_and_inputs(config, overrides)
    fitted_weights_path = os.path.join(dir_path, "fitted_weights.eqx")
    if not os.path.exists(fitted_weights_path):
        raise FileNotFoundError(
            f"No fitted_weights.eqx found in {dir_path} - this fit may predate that artifact, or "
            "postprocessing/saving may have been disabled for it."
        )
    return run_postprocess(config, fitted_weights_path)


def run_postprocess_remote(run_id_or_url: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Runs postprocess on a fit tracked by mlflow, identified by a bare run id or a run URL (e.g. from
    https://continuum.ergodic.io/experiments/...). Only reads the source run's artifacts - the results of
    this replay are logged to a new run, so the source run's record is left untouched.

    Supports both config artifact layouts: a single config.yaml (tried first) or defaults.yaml + inputs.yaml.

    Args:
        run_id_or_url: mlflow run id or URL of the fit.
        overrides: see run_postprocess_local.
    """
    run_id = _extract_run_id(run_id_or_url)

    with tempfile.TemporaryDirectory() as td:
        base_uri = _resolve_artifact_uri(run_id)
        try:
            _download_run_artifact(base_uri, "config.yaml", td)
            remaining_fnames = ["fitted_weights.eqx"]
        except Exception:
            remaining_fnames = ["defaults.yaml", "inputs.yaml", "fitted_weights.eqx"]

        for fname in remaining_fnames:
            try:
                _download_run_artifact(base_uri, fname, td)
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not download {fname} from run {run_id}: {e}. If this is fitted_weights.eqx, "
                    "the run may predate that artifact, or postprocessing/saving may have been disabled for it."
                ) from e

        # Optional for backward compatibility. New angular checkpoints use this to
        # restore a global best that came from an earlier refinement stage.
        try:
            _download_run_artifact(base_uri, "checkpoint_metadata.json", td)
        except Exception:
            pass

        config = _load_merged_config(td)
        if overrides:
            config = misc.merge_defaults_and_inputs(config, overrides)
        fitted_weights_path = os.path.join(td, "fitted_weights.eqx")
        return run_postprocess(config, fitted_weights_path, source_run_id=run_id)
