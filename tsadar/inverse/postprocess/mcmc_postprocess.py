"""MCMC uncertainty postprocessor: runs a Metropolis-Hastings chain (see .mcmc) around each lineout's
best fit, optionally pooled across calibration draws (see .mcmc_calibration), and writes the same family
of artifacts as the Hessian/Laplace postprocessor (.laplace). Only reachable through
mcmc_postprocess_runner.py.
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
from ..loops import build_batch
from ..loss_function import LossFunction
from . import mcmc, mcmc_calibration
from .laplace import recalculate_with_chosen_weights


def _active_param_keys(cfg_params: Dict) -> List[Tuple[str, str]]:
    """Ordered (species, key) pairs of the active fit parameters, excluding "fe", in config["parameters"]
    iteration order (the order plotters.save_sigmas_params/plot_final_params expect)."""
    keys = []
    for species, params in cfg_params.items():
        for key, p in params.items():
            if key == "fe":
                continue
            if isinstance(p, dict) and p.get("active"):
                keys.append((species, key))
    return keys


def _pytree_active_keys(cfg_params: Dict, diff_params) -> List[Tuple[str, str]]:
    """(species, key) for each leaf of diff_params in pytree traversal order (electron, ions, general),
    which is the parameter order of mcmc._within_chain_r_hat and differs from _active_param_keys.

    Args:
        cfg_params: config["parameters"].
        diff_params: pytree with the active-leaf structure of the sampled parameters.

    Returns:
        List[Tuple[str, str]], one entry per active leaf.
    """
    ion_species_in_order = [species for species in cfg_params if "ion" in species]
    paths = jax.tree_util.tree_flatten_with_path(diff_params)[0]
    keys = []
    for path, _ in paths:
        top = path[0].name
        leaf_name = path[-1].name
        key = leaf_name[len("normed_") :] if leaf_name.startswith("normed_") else leaf_name
        species = ion_species_in_order[path[1].idx] if top == "ions" else top
        keys.append((species, key))
    return keys


def _physical_samples_for_fit_batch(static_array_part, static_nonarray_part, pooled_diff_params, fit_batch_index, active_keys):
    """Physical (denormalized) posterior samples of the active scalar parameters for one fit-batch.
    Returns an array of shape (num_pooled, this_batch_size, n_active) with columns in active_keys order.
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


def _failed_chains(stacked: np.ndarray, num_chains: int, r_hat: np.ndarray, threshold: float) -> np.ndarray:
    """Flags, per chain, lineout and active parameter, the chains that did not converge on their own:
    split R-hat above threshold or not finite, or a chain that never moved.

    Args:
        stacked: (num_chains * num_kept, batch_size, n_active) physical samples, chains concatenated in order.
        num_chains: number of chains pooled into stacked.
        r_hat: (num_chains, batch_size, n_active) split R-hat of each chain.
        threshold: largest acceptable split R-hat.

    Returns:
        failed: (num_chains, batch_size, n_active) boolean.
    """
    total, batch_size, n_active = stacked.shape
    by_chain = stacked.reshape(num_chains, total // num_chains, batch_size, n_active)
    frozen = np.ptp(by_chain, axis=1) == 0
    return frozen | ~(np.asarray(r_hat) <= threshold)


def _reorder_columns(values: np.ndarray, from_keys: List[Tuple[str, str]], to_keys: List[Tuple[str, str]]) -> np.ndarray:
    """Reorders the last axis of values from the parameter order from_keys to the order to_keys."""
    return np.asarray(values)[..., [from_keys.index(key) for key in to_keys]]


def _mad_flagged_chains(stacked: np.ndarray, num_chains: int, mad_scale: float) -> np.ndarray:
    """Flags, per lineout and active parameter, the chains whose posterior mean is more than mad_scale
    robust standard deviations (1.4826 * MAD) from the median of all chains' means.

    Args:
        stacked: (num_chains * num_kept, batch_size, n_active) physical samples, chains concatenated in
            order (see _physical_samples_for_fit_batch).
        num_chains: number of chains pooled into stacked (> 1).
        mad_scale: threshold in robust-SD units.

    Returns:
        flagged: (num_chains, batch_size, n_active) boolean.
    """
    total, batch_size, n_active = stacked.shape
    num_kept = total // num_chains
    by_chain = stacked.reshape(num_chains, num_kept, batch_size, n_active)
    chain_means = by_chain.mean(axis=1)  # (num_chains, batch_size, n_active)

    median = np.median(chain_means, axis=0)  # (batch_size, n_active)
    mad = np.median(np.abs(chain_means - median[None]), axis=0)  # (batch_size, n_active)
    robust_sd = 1.4826 * mad
    # the floor keeps a divergent chain flagged when more than half the chains agree exactly (MAD = 0)
    absolute_deviation = np.abs(chain_means - median[None])
    deviation = absolute_deviation / np.maximum(robust_sd, 1e-300)
    return deviation > mad_scale  # (num_chains, batch_size, n_active)


def _finalize_chain_selection(bad_within: np.ndarray, bad_outlier: np.ndarray, max_drop_fraction: float, num_kept: int):
    """Combines the per-(chain, parameter) non-convergence and outlier flags and decides, per lineout,
    which chains to exclude from the summary statistics.

    A parameter whose own flagged-chain count exceeds the drop budget is excluded from voting. The chains
    flagged by any voting parameter are dropped; if that exceeds the budget the lineout is marked
    unreliable. When no parameter can vote, every chain is kept.

    Args:
        bad_within: (num_chains, batch_size, n_active) boolean, split R-hat above threshold.
        bad_outlier: (num_chains, batch_size, n_active) boolean, from _mad_flagged_chains.
        max_drop_fraction: maximum fraction of chains that may be dropped.
        num_kept: samples per chain.

    Returns:
        keep_mask: (num_chains * num_kept, batch_size) boolean mask of the pooled samples to use. All
            False for an unreliable lineout.
        n_dropped: (batch_size,) number of chains dropped per lineout.
        unreliable: (batch_size,) boolean, True where too many chains were dropped.
        param_unreliable: (batch_size, n_active) boolean, True where the parameter was excluded from voting.
    """
    num_chains, batch_size, n_active = bad_within.shape
    total_bad = bad_within | bad_outlier  # (num_chains, batch_size, n_active)
    max_droppable = int(np.floor(max_drop_fraction * num_chains))

    n_bad_per_param = total_bad.sum(axis=0)  # (batch_size, n_active)
    param_unreliable = n_bad_per_param > max_droppable  # (batch_size, n_active) -- excluded from voting

    voting = ~param_unreliable  # (batch_size, n_active)
    any_voter = voting.any(axis=-1)  # (batch_size,) -- lineouts with at least one parameter allowed to vote

    # lineouts with no voting parameter drop nothing
    total_bad_voted = np.any(total_bad & voting[None, :, :], axis=-1)  # (num_chains, batch_size)

    n_dropped = total_bad_voted.sum(axis=0)  # (batch_size,)
    unreliable = any_voter & (n_dropped > max_droppable)  # (batch_size,)

    keep_chain = ~total_bad_voted  # (num_chains, batch_size)
    keep_chain[:, unreliable] = False  # nothing from an unreliable lineout feeds its own summary stats

    keep_mask = np.repeat(keep_chain, num_kept, axis=0)  # (num_chains*num_kept, batch_size)
    return keep_mask, n_dropped, unreliable, param_unreliable


def _build_loss_fn_for_draw(
    config_k: Dict, sa, all_data_k: Dict, batch_size: int, nominal_loss_fn: LossFunction = None
) -> LossFunction:
    """Builds a LossFunction for one calibration draw's data, using the same sample construction as
    loops.one_d_loop so the normalization factors are consistent.

    The data were throughput-corrected once, on the nominal wavelength axis, and a draw does not
    re-correct them. When nominal_loss_fn is given, the draw's covar noise model therefore keeps the
    nominal throughput correction instead of one evaluated on the draw's perturbed axis."""
    sample = {k: v[:batch_size] for k, v in all_data_k.items()}
    sample = {
        "noise_e": all_data_k["noiseE"][:batch_size],
        "noise_i": all_data_k["noiseI"][:batch_size],
    } | sample
    loss_fn_k = LossFunction(config_k, sa, sample)
    if hasattr(nominal_loss_fn, "covar_throughput_e") and hasattr(loss_fn_k, "covar_throughput_e"):
        loss_fn_k.covar_throughput_e = nominal_loss_fn.covar_throughput_e
    return loss_fn_k


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
    Runs Metropolis-Hastings MCMC (see .mcmc) around each lineout's best-fit weights, pooled across the
    calibration draws (see .mcmc_calibration), and logs the resulting artifacts to mlflow.

    Supports 1D (non-angular) fits with the electron distribution function inactive; raises
    NotImplementedError otherwise.

    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices: indices of the lineouts that were fit
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        all_axes: Dict- calibrated axes and axes labels
        loss_fn: the nominal-calibration LossFunction instance used for fitting
        sa: scattering angles and their relative weights
        fitted_weights: List[ThomsonParams], the per-fit-batch best-fit weights returned by the minimizer
        num_params: unused, kept for signature parity with the Laplace postprocessor

    Returns:
        final_params: Dict- posterior mean of the fitted parameters (same layout as
        plotters.get_final_params), plus an "mcmc_diagnostics" entry.
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
        # the nominal loss_fn is only valid for a draw whose calibration is unperturbed
        reuse_nominal = config_k is config and all_data_k is all_data
        loss_fn_k = (
            loss_fn if reuse_nominal else _build_loss_fn_for_draw(config_k, sa, all_data_k, batch_size, loss_fn)
        )
        loss_fns_by_draw.append(loss_fn_k)
        batches_by_draw.append([build_batch(all_data_k, inds, background_subtract) for inds in batch_indices])

    key = jax.random.PRNGKey(int(mcmc_cfg["seed"]))
    # each draw is checkpointed to this run as soon as it finishes
    active_run = mlflow.active_run()
    checkpoint_run_id = active_run.info.run_id if active_run is not None else None
    pooled_diff_params, static_params, diagnostics_by_draw, max_r_hat_by_batch, within_chain_r_hat_by_batch = (
        mcmc.run_mcmc_pooled(
            config, loss_fns_by_draw, fitted_weights, batches_by_draw, key, checkpoint_run_id=checkpoint_run_id
        )
    )

    if within_chain_r_hat_by_batch is not None and n_active > 0:
        # reorder the R-hat parameter axis from pytree order to active_keys order
        pytree_keys = _pytree_active_keys(config["parameters"], pooled_diff_params)
        reorder = [pytree_keys.index(k) for k in active_keys]
        within_chain_r_hat_by_batch = np.asarray(within_chain_r_hat_by_batch)[..., reorder]

    static_array_part = eqx.filter(static_params, eqx.is_array)
    static_nonarray_part = eqx.filter(static_params, eqx.is_array, inverse=True)

    all_params_mean: Dict[str, Dict[str, np.ndarray]] = {}
    all_params_std: Dict[str, Dict[str, np.ndarray]] = {}
    for species, key_name in active_keys:
        all_params_mean.setdefault(species, {})[key_name] = np.full(total_lineouts, np.nan)
        all_params_std.setdefault(species, {})[key_name] = np.full(total_lineouts, np.nan)
    covariance = np.full((total_lineouts, n_active, n_active), np.nan)
    acceptance_rate = np.full(total_lineouts, np.nan)
    # only defined for >= 2 chains
    max_r_hat = np.full(total_lineouts, np.nan) if max_r_hat_by_batch is not None else None
    num_chains = len(draws)
    n_chains_dropped = np.zeros(total_lineouts, dtype=int) if n_active > 0 else None
    lineout_unreliable = np.zeros(total_lineouts, dtype=bool) if n_active > 0 else None
    # True where a parameter was excluded from chain-selection voting, or its lineout is unreliable
    param_unreliable = np.zeros((total_lineouts, n_active), dtype=bool) if n_active > 0 else None
    # unfiltered pooled samples per fit-batch, reused for mcmc_samples.nc and the corner plots
    physical_by_fit_batch = []
    # per-fit-batch (num_chains * num_kept, this_batch_size) sample mask used for the corner plots
    keep_mask_by_fit_batch = []

    for b, inds in enumerate(batch_indices):
        stacked = _physical_samples_for_fit_batch(static_array_part, static_nonarray_part, pooled_diff_params, b, active_keys)
        physical_by_fit_batch.append(stacked)

        if n_active > 0:
            bad_within = _failed_chains(
                stacked, num_chains, within_chain_r_hat_by_batch[:, b, :, :], mcmc_cfg["within_chain_r_hat_threshold"]
            )
            bad_outlier = (
                _mad_flagged_chains(stacked, num_chains, mcmc_cfg["chain_outlier_mad_scale"])
                if num_chains > 1
                else np.zeros_like(bad_within)
            )
            num_kept = stacked.shape[0] // num_chains
            keep_mask, n_dropped, unreliable, batch_param_unreliable = _finalize_chain_selection(
                bad_within, bad_outlier, mcmc_cfg["max_dropped_chain_fraction"], num_kept
            )
            n_chains_dropped[inds] = n_dropped
            lineout_unreliable[inds] = unreliable
            param_unreliable[inds] = batch_param_unreliable | unreliable[:, None]
            # corner plots of an unreliable lineout show every chain
            keep_mask_for_plotting = np.where(unreliable[None, :], True, keep_mask)
            keep_mask_by_fit_batch.append(keep_mask_for_plotting)

            for lineout_local, lineout_global in enumerate(inds):
                if unreliable[lineout_local]:
                    continue  # mean/std/covariance stay NaN -- see all_params_mean/std/covariance init above
                kept = stacked[keep_mask[:, lineout_local], lineout_local, :]
                means = kept.mean(axis=0)
                stds = kept.std(axis=0, ddof=1)  # same estimator as np.cov below
                for a, (species, key_name) in enumerate(active_keys):
                    all_params_mean[species][key_name][lineout_global] = means[a]
                    all_params_std[species][key_name][lineout_global] = stds[a]
                covariance[lineout_global] = np.atleast_2d(np.cov(kept, rowvar=False))

        rates = np.mean([np.asarray(diag["acceptance_rate"])[b] for diag in diagnostics_by_draw], axis=0)
        acceptance_rate[inds] = rates
        if max_r_hat is not None:
            # worst R-hat across parameters, before any chain filtering, for the diagnostics plot
            max_r_hat[inds] = np.max(np.asarray(max_r_hat_by_batch)[b], axis=-1)

    if n_chains_dropped is not None:
        lineouts_affected = int(np.sum(n_chains_dropped > 0))
        total_dropped = int(np.sum(n_chains_dropped))
        lineouts_unreliable_count = int(np.sum(lineout_unreliable))
        print(
            f"Chain filtering: {total_dropped} chain(s) written off from summary statistics across "
            f"{lineouts_affected}/{total_lineouts} lineout(s) (chain_outlier_mad_scale="
            f"{mcmc_cfg['chain_outlier_mad_scale']}, within_chain_r_hat_threshold="
            f"{mcmc_cfg['within_chain_r_hat_threshold']}, max_dropped_chain_fraction="
            f"{mcmc_cfg['max_dropped_chain_fraction']}); {lineouts_unreliable_count}/{total_lineouts} "
            f"lineout(s) marked unreliable (too many chains written off to trust any subset)."
        )
        metrics = {
            "lineouts_with_dropped_chains": lineouts_affected,
            "total_chains_dropped": total_dropped,
            "lineouts_unreliable": lineouts_unreliable_count,
        }
        # number of lineouts for which each parameter was excluded from chain-selection voting
        param_unreliable_counts = {}
        for a, (species, key_name) in enumerate(active_keys):
            count = int(np.sum(param_unreliable[:, a]))
            param_unreliable_counts[f"{key_name}_{species}"] = count
            metrics[f"param_unreliable.{key_name}_{species}"] = count
        if param_unreliable_counts:
            breakdown = ", ".join(f"{name}: {count}/{total_lineouts}" for name, count in param_unreliable_counts.items())
            print(f"Per-parameter unreliable-lineout counts (excluded from chain-selection voting): {breakdown}")
        mlflow.log_metrics(metrics)

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
            # the Laplace columns follow get_fitted_params order, the MCMC tables follow active_keys
            fitted_params, _ = fitted_weights[0].get_fitted_params(config["parameters"])
            laplace_keys = [
                (species, key)
                for species, params in fitted_params.items()
                for key in params
                if key not in ("fe", "f", "flm")
            ]
            laplace_sigmas = _reorder_columns(laplace_sigmas, laplace_keys, active_keys)
        except Exception as e:
            print(f"Could not compute Laplace/Hessian sigmas for comparison, skipping: {e}")
            laplace_sigmas = None

    mlflow.log_metrics({"mcmc postprocessing time": round(time.time() - t0, 2)})
    mlflow.set_tag("status", "plotting")
    t0 = time.time()

    from tsadar.utils.plotting import plotters

    with tempfile.TemporaryDirectory() as td:
        _ = [os.makedirs(os.path.join(td, dirname), exist_ok=True) for dirname in ["plots", "binary", "csv"]]

        final_params = plotters.get_final_params(config, all_params_mean, all_axes, td)
        mcmc_sigmas_ds = plotters.save_sigmas_params_mcmc(config, all_params_mean, mcmc_sigmas, all_axes, td)
        plotters.plot_final_params(config, all_params_mean, mcmc_sigmas_ds, td)
        plotters.plot_mcmc_diagnostics(
            config, acceptance_rate, td, max_r_hat=max_r_hat, n_chains_dropped=n_chains_dropped,
            lineout_unreliable=lineout_unreliable,
        )

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

        if n_active > 0:
            # which reported values failed the convergence checks, on the same coordinates as the sigmas
            reliability_ds = xr.Dataset(
                {
                    "param_unreliable": (("lineout", "param"), param_unreliable.astype(np.uint8)),
                    "lineout_unreliable": (("lineout",), lineout_unreliable.astype(np.uint8)),
                    "n_chains_dropped": (("lineout",), n_chains_dropped),
                },
                coords={"lineout": np.array(config["data"]["lineouts"]["val"]), "param": param_names},
            )
            reliability_ds.to_netcdf(os.path.join(td, "binary", "mcmc_reliability.nc"))

        if n_active > 0:
            # corner plots for an evenly-spaced subset of lineouts
            n_corner_lineouts = min(8, total_lineouts)
            corner_targets = set(np.linspace(0, total_lineouts - 1, n_corner_lineouts, dtype=int).tolist())
            lineout_vals = np.array(config["data"]["lineouts"]["val"])
            for b, inds in enumerate(batch_indices):
                stacked = physical_by_fit_batch[b]
                keep_mask = keep_mask_by_fit_batch[b]
                for lineout_local, lineout_global in enumerate(inds):
                    if lineout_global in corner_targets:
                        # show the same chains the summary statistics were computed from
                        kept = keep_mask[:, lineout_local]
                        if lineout_unreliable is not None and lineout_unreliable[lineout_global]:
                            num_chains_kept = num_chains
                        elif n_chains_dropped is not None:
                            num_chains_kept = num_chains - int(n_chains_dropped[lineout_global])
                        else:
                            num_chains_kept = num_chains
                        plotters.plot_corner(
                            stacked[kept, lineout_local, :], param_names, lineout_vals[lineout_global], td,
                            num_chains=num_chains_kept,
                        )

        if mcmc_cfg["save_samples"] and n_active > 0:
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

    param_unreliable_by_key = None
    if param_unreliable is not None:
        param_unreliable_by_key = {}
        for a, (species, key_name) in enumerate(active_keys):
            param_unreliable_by_key.setdefault(species, {})[key_name] = param_unreliable[:, a]

    return final_params | {
        "mcmc_diagnostics": {
            "acceptance_rate": acceptance_rate,
            "num_calibration_draws": len(draws),
            "max_r_hat": max_r_hat,
            "n_chains_dropped": n_chains_dropped,
            "lineout_unreliable": lineout_unreliable,
            "param_unreliable": param_unreliable_by_key,
        }
    }
