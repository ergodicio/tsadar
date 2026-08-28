"""Standalone MCMC postprocessor entry point: mirrors postprocess_runner.py, but runs the alternate
Metropolis-Hastings MCMC uncertainty postprocessor (inverse.postprocess.mcmc_postprocess) instead of the
Hessian/Laplace one, against an already-completed fit's saved results -- loaded either from a local run
directory or a remote MLflow run (by id or URL), without redoing the fit.

Kept as a sibling file to postprocess_runner.py (importing its shared reconstruction/config-loading
helpers) rather than folded into it, so that module's single responsibility -- replaying the original
Laplace postprocessor -- stays focused.
"""
import os
import tempfile
from typing import Dict, Optional

import mlflow

from .inverse.postprocess import mcmc_postprocess as _mcmc_postprocess_module
from .postprocess_runner import _extract_run_id, _load_merged_config, _reconstruct_fit_state
from .utils import misc


def run_mcmc_postprocess(config: Dict, fitted_weights_path: str, source_run_id: Optional[str] = None) -> Dict:
    """
    MCMC analogue of postprocess_runner.run_postprocess: reconstructs everything the MCMC postprocessor
    needs from a saved config + fitted_weights.eqx (rather than re-fitting), and runs it inside a
    brand-new mlflow run - this never resumes or mutates the run the fit originally came from, it only
    reads its artifacts.

    1D (non-angular) fits only, and the electron distribution function ("fe") must be inactive -- see
    inverse/postprocess/mcmc.py's module docstring for why; both raise NotImplementedError immediately.

    Args:
        config (Dict): The exact (merged) config the original fit used.
        fitted_weights_path (str): Local path to a fitted_weights.eqx saved by fitter._save_fit_artifacts.
        source_run_id (Optional[str]): The mlflow run id the artifacts came from, if any - logged as a tag
            on the new run for traceability, but the source run itself is never touched.
    Returns:
        Dict: The final_params produced by inverse.postprocess.mcmc_postprocess.mcmc_postprocess.
    """
    mlflow_cfg = config.get("mlflow", {})
    if "experiment" in mlflow_cfg:
        mlflow.set_experiment(mlflow_cfg["experiment"])
    run_name = f"{mlflow_cfg['run']} (mcmc-postprocess)" if "run" in mlflow_cfg else None

    with mlflow.start_run(run_name=run_name):
        if source_run_id is not None:
            mlflow.set_tag("source_run_id", source_run_id)
        misc.log_mlflow(config)

        state = _reconstruct_fit_state(config, fitted_weights_path)
        if state.is_angular:
            raise NotImplementedError(
                "MCMC postprocessing does not support angular fits; see "
                "inverse.postprocess.mcmc_postprocess.mcmc_postprocess for details."
            )
        final_params = _mcmc_postprocess_module.mcmc_postprocess(
            state.config,
            state.sample_indices,
            state.all_data,
            state.all_axes,
            state.loss_fn,
            state.sa,
            state.fitted_weights,
            state.num_params,
        )

    return final_params


def run_mcmc_postprocess_local(dir_path: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Runs the MCMC postprocessor on a fit whose artifacts already sit in a local directory - e.g. a copy
    of an mlflow run's artifact folder. Accepts either config layout _load_merged_config understands: a
    single config.yaml, or defaults.yaml + inputs.yaml. Either way, fitted_weights.eqx must also be
    present.

    Args:
        overrides: optional partial config (same nesting as inputs.yaml -- typically a small, dedicated
            stub deck containing only the MCMC-relevant keys being changed, e.g.
            config["other"]["mcmc"]) deep-merged on top of the saved config in memory, without touching
            the files on disk. See postprocess_runner.run_postprocess_local's docstring for why this is
            deliberately not sourced from the live repo deck automatically, and for the reconstruction-
            critical fields (data.lineouts, optimizer.batch_size, parameters.*.active, etc.) that should
            not be overridden this way.
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
    return run_mcmc_postprocess(config, fitted_weights_path)


def run_mcmc_postprocess_remote(run_id_or_url: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Runs the MCMC postprocessor on a fit tracked by mlflow, identified by a bare run id or a run URL
    (e.g. from https://continuum.ergodic.io/experiments/...). Only reads the source run's artifacts - the
    results of this replay are logged to a new run, so the source run's record is left untouched.

    Supports both config artifact layouts: a single config.yaml (app-originated runs) or
    defaults.yaml + inputs.yaml (CLI/cluster runs) -- see postprocess_runner.run_postprocess_remote.

    Args:
        overrides: see run_mcmc_postprocess_local -- applied identically here, in memory only.
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
        if overrides:
            config = misc.merge_defaults_and_inputs(config, overrides)
        fitted_weights_path = os.path.join(td, "fitted_weights.eqx")
        return run_mcmc_postprocess(config, fitted_weights_path, source_run_id=run_id)
