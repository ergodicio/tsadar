"""mcmc_postprocess: alternate uncertainty postprocessor that runs a Metropolis-Hastings MCMC chain
(see .mcmc) around each lineout's best-fit solution, optionally pooled across calibration-uncertainty
draws (see .mcmc_calibration), producing the same family of artifacts (learned parameters, per-lineout
sigmas, diagnostic plots, a manifest) the existing Hessian/Laplace postprocessor (.laplace) does -- as a
standalone alternative, not a replacement. Only reachable via mcmc_postprocess_runner.py; never called
from fitter.fit().
"""
import os
import tempfile
import time
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import mlflow
import numpy as np
import xarray as xr

from tsadar.utils import manifest
from tsadar.utils.plotting import plotters
from ..loops import build_batch
from ..loss_function import LossFunction
from . import mcmc, mcmc_calibration
from .laplace import recalculate_with_chosen_weights


def _active_param_keys(cfg_params: Dict) -> List[Tuple[str, str]]:
    """Ordered list of (species, key) pairs that are active fit parameters, excluding the electron
    distribution function ("fe") -- see mcmc.py's module docstring for why fe is out of scope. Order
    matches config["parameters"]'s own dict iteration (insertion) order, the same convention
    plotters.save_sigmas_params/plot_final_params expect for matching sigma columns to parameters.
    """
    keys = []
    for species, params in cfg_params.items():
        for key, p in params.items():
            if key == "fe":
                continue
            if isinstance(p, dict) and p.get("active"):
                keys.append((species, key))
    return keys


def _physical_samples_for_fit_batch(static_array_part, static_nonarray_part, pooled_diff_params, fit_batch_index, active_keys):
    """Reconstructs physical (denormalized) posterior samples for one fit-batch, for exactly the active
    scalar parameters. Returns an array of shape (num_pooled, this_batch_size, n_active), columns ordered
    to match active_keys -- shared by both the mean/std/covariance computation and (optionally) the raw
    sample-saving step, so the reconstruction only has to happen once per fit-batch.
    """
    static_i = eqx.combine(
        jax.tree_util.tree_map(lambda x: x[fit_batch_index], static_array_part),
        static_nonarray_part,
    )
    diff_i = jax.tree_util.tree_map(lambda x: x[fit_batch_index], pooled_diff_params)  # leaves: (num_pooled, this_batch_size)

    def _unnorm(dp):
        return eqx.combine(static_i, dp).get_unnormed_params()

    if not active_keys:
        return np.zeros((0, 0, 0))
    physical = eqx.filter_vmap(_unnorm)(diff_i)  # dict[species][key] -> (num_pooled, this_batch_size)
    return np.stack([np.asarray(physical[species][key_name]) for species, key_name in active_keys], axis=-1)


def _build_loss_fn_for_draw(config_k: Dict, sa, all_data_k: Dict, batch_size: int) -> LossFunction:
    """Builds a fresh LossFunction for one calibration draw's (possibly rescaled/re-ranged) data,
    following the exact same "sample" construction loops.one_d_loop uses to build its own initial
    LossFunction (loops.py:245-250), so normalization factors are derived consistently."""
    sample = {k: v[:batch_size] for k, v in all_data_k.items()}
    sample = {
        "noise_e": all_data_k["noiseE"][:batch_size],
        "noise_i": all_data_k["noiseI"][:batch_size],
    } | sample
    return LossFunction(config_k, sa, sample)


def mcmc_postprocess(
    config: Dict,
    sample_indices: np.ndarray,
    all_data: Dict,
    all_axes: Dict,
    loss_fn: LossFunction,
    sa,
    fitted_weights: List,
    num_params: int,
) -> Dict:
    """
    Alternate uncertainty postprocessor: runs Metropolis-Hastings MCMC (see .mcmc) around each lineout's
    best-fit weights, pooled across K calibration-nuisance draws (see .mcmc_calibration), as an
    alternative to postprocess.laplace.get_sigmas' Hessian/Laplace approximation.

    1D (non-angular) fits only, and the electron distribution function ("fe") must be inactive -- see
    mcmc.py's module docstring for why. Both are checked and raise NotImplementedError immediately
    rather than silently producing a wrong answer.

    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices: indices of the lineouts that were fit
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        all_axes: Dict- calibrated axes and axes labels
        loss_fn: the (nominal, unperturbed-calibration) LossFunction instance used for fitting
        sa: scattering angles and their relative weights
        fitted_weights: List[ThomsonParams], the per-fit-batch best-fit weights returned by the minimizer
        num_params: unused here (kept for signature parity with postprocess.postprocess); the number of
            MCMC-sampled parameters is instead derived from config["parameters"]'s active flags directly
            (see _active_param_keys), since it must exclude "fe" regardless of what num_params counted.

    Returns:
        final_params: Dict- posterior mean of the fitted parameters (same dict layout
        plotters.get_final_params expects), plus an "mcmc_diagnostics" entry with per-draw acceptance
        rates and the number of calibration draws/pooled samples actually used.
    """
    if "angular" in config["other"]["extraoptions"]["spectype"]:
        raise NotImplementedError(
            "MCMC postprocessing does not support angular fits: process_angular_data's batching "
            "(a single non-batched ThomsonParams, loops.build_angular_batch) differs enough from the 1D "
            "path this sampler is built around that it is not attempted here."
        )
    mcmc.check_fe_inactive(config["parameters"])

    t0 = time.time()
    mcmc_cfg = mcmc._mcmc_cfg(config)
    active_keys = _active_param_keys(config["parameters"])
    n_active = len(active_keys)

    calibration_seed = config.get("other", {}).get("calibration_uncertainty", {}).get("seed", 0)
    rng = np.random.default_rng(int(calibration_seed))
    draws = mcmc_calibration.draw_calibration_realizations(config, all_data, all_axes, rng)

    background_subtract = config["data"]["background"]["bg_subtract"]
    batch_size = config["optimizer"]["batch_size"]
    sample_indices = np.sort(np.array(sample_indices))
    batch_indices = np.reshape(sample_indices, (-1, batch_size))
    total_lineouts = len(sample_indices)

    loss_fns_by_draw = []
    batches_by_draw = []
    for draw_index, (config_k, all_data_k) in enumerate(draws):
        # draw 0 always reuses the caller-supplied loss_fn unchanged (config_k is config, all_data_k is
        # all_data, by construction whenever draw_calibration_realizations collapses to K=1) -- this is
        # what guarantees zero extra LossFunction builds on the backward-compatible no-calibration path.
        loss_fn_k = loss_fn if draw_index == 0 else _build_loss_fn_for_draw(config_k, sa, all_data_k, batch_size)
        loss_fns_by_draw.append(loss_fn_k)
        batches_by_draw.append([build_batch(all_data_k, inds, background_subtract) for inds in batch_indices])

    key = jax.random.PRNGKey(int(mcmc_cfg["seed"]))
    pooled_diff_params, static_params, diagnostics_by_draw = mcmc.run_mcmc_pooled(
        config, loss_fns_by_draw, fitted_weights, batches_by_draw, key
    )

    static_array_part = eqx.filter(static_params, eqx.is_array)
    static_nonarray_part = eqx.filter(static_params, eqx.is_array, inverse=True)

    all_params_mean: Dict[str, Dict[str, np.ndarray]] = {}
    all_params_std: Dict[str, Dict[str, np.ndarray]] = {}
    for species, key_name in active_keys:
        all_params_mean.setdefault(species, {})[key_name] = np.full(total_lineouts, np.nan)
        all_params_std.setdefault(species, {})[key_name] = np.full(total_lineouts, np.nan)
    covariance = np.full((total_lineouts, n_active, n_active), np.nan)
    acceptance_rate = np.full(total_lineouts, np.nan)
    # Cached here so the (optional) sample-saving block below can reuse each fit-batch's reconstruction
    # instead of recomputing it.
    physical_by_fit_batch = []

    for b, inds in enumerate(batch_indices):
        stacked = _physical_samples_for_fit_batch(static_array_part, static_nonarray_part, pooled_diff_params, b, active_keys)
        physical_by_fit_batch.append(stacked)

        if n_active > 0:
            means = stacked.mean(axis=0)
            stds = stacked.std(axis=0)
            for lineout_local, lineout_global in enumerate(inds):
                for a, (species, key_name) in enumerate(active_keys):
                    all_params_mean[species][key_name][lineout_global] = means[lineout_local, a]
                    all_params_std[species][key_name][lineout_global] = stds[lineout_local, a]
                covariance[lineout_global] = np.atleast_2d(np.cov(stacked[:, lineout_local, :], rowvar=False))

        rates = np.mean([np.asarray(diag["acceptance_rate"])[b] for diag in diagnostics_by_draw], axis=0)
        acceptance_rate[inds] = rates

    mcmc_sigmas = (
        np.stack([all_params_std[species][key_name] for species, key_name in active_keys], axis=1)
        if n_active > 0
        else np.zeros((total_lineouts, 0))
    )

    laplace_sigmas = None
    if mcmc_cfg["compare_to_laplace"]:
        try:
            _, _, _, laplace_sigmas = recalculate_with_chosen_weights(
                config, sa, sample_indices, all_data, loss_fn, True, fitted_weights, n_active
            )
        except Exception as e:
            print(f"Could not compute Laplace/Hessian sigmas for comparison, skipping: {e}")
            laplace_sigmas = None

    mlflow.log_metrics({"mcmc postprocessing time": round(time.time() - t0, 2)})
    mlflow.set_tag("status", "plotting")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as td:
        _ = [os.makedirs(os.path.join(td, dirname), exist_ok=True) for dirname in ["plots", "binary", "csv"]]

        final_params = plotters.get_final_params(config, all_params_mean, all_axes, td)
        mcmc_sigmas_ds = plotters.save_sigmas_params_mcmc(config, all_params_mean, mcmc_sigmas, all_axes, td)
        plotters.plot_final_params(config, all_params_mean, mcmc_sigmas_ds, td)
        plotters.plot_mcmc_diagnostics(config, acceptance_rate, td)

        laplace_sigmas_ds = None
        if laplace_sigmas is not None:
            laplace_sigmas_ds = plotters.save_sigmas_params(config, all_params_mean, laplace_sigmas, all_axes, td)
        plotters.plot_sigma_comparison(config, all_params_mean, laplace_sigmas_ds, mcmc_sigmas_ds, td)

        param_names = [f"{key_name}_{species}" for species, key_name in active_keys]
        covariance_ds = xr.Dataset(
            {
                "covariance": (
                    ("lineout", "param_i", "param_j"),
                    covariance,
                )
            },
            coords={
                "lineout": np.array(config["data"]["lineouts"]["val"]),
                "param_i": param_names,
                "param_j": param_names,
            },
        )
        covariance_ds.to_netcdf(os.path.join(td, "binary", "mcmc_covariance.nc"))

        if mcmc_cfg["save_samples"] and n_active > 0:
            # Reuses each fit-batch's physical-value reconstruction (physical_by_fit_batch, computed
            # once above) and concatenates them across fit-batches along the lineout axis, to save the
            # full (thinned, pooled) posterior rather than just its moments.
            num_pooled = physical_by_fit_batch[0].shape[0]
            per_param_samples = {name: np.full((num_pooled, total_lineouts), np.nan) for name in param_names}
            for b, inds in enumerate(batch_indices):
                stacked = physical_by_fit_batch[b]
                for a, (species, key_name) in enumerate(active_keys):
                    per_param_samples[f"{key_name}_{species}"][:, inds] = stacked[:, :, a]

            samples_ds = xr.Dataset(
                {name: (("sample", "lineout"), vals) for name, vals in per_param_samples.items()},
                coords={"lineout": np.array(config["data"]["lineouts"]["val"])},
            )
            samples_ds.to_netcdf(os.path.join(td, "binary", "mcmc_samples.nc"))

        manifest.write_manifest(td, mode="mcmc_postprocess")
        mlflow.log_artifacts(td)

    mlflow.log_metrics({"mcmc plotting time": round(time.time() - t0, 2)})
    mlflow.set_tag("status", "done plotting (mcmc)")

    return final_params | {
        "mcmc_diagnostics": {
            "acceptance_rate": acceptance_rate,
            "num_calibration_draws": len(draws),
        }
    }
