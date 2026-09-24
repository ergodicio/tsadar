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


def _mad_flagged_chains(stacked: np.ndarray, num_chains: int, mad_scale: float) -> np.ndarray:
    """Identifies, per lineout and per active parameter, which of num_chains independent chains pooled
    into `stacked` settled on a different region of parameter space than the rest -- e.g. one trapped in a
    different local mode along a genuine degeneracy (see mcmc.py's module docstring, and RAM's own
    documented limitation for strongly multimodal targets: "not suitable for strongly multi-modal targets,
    but this is the case for any random walk based approach" -- Vihola 2012, section 6).

    Flags a chain, for one lineout and one active parameter, if that chain's own posterior mean for that
    parameter is more than `mad_scale` robust standard deviations (1.4826 * MAD, the usual normal-
    consistent scaling -- see Iglewicz & Hoaglin's modified z-score) from the median of all chains' means
    for that same parameter. Deliberately kept per-parameter (not reduced to a single "worst case across
    parameters" flag per chain, the way this used to be) for the same reason mcmc._max_r_hat_across_chains
    is: a genuinely weakly-identified parameter's chain means can legitimately scatter widely relative to
    its own (correspondingly tiny) robust-SD, which would flag nearly every chain as an "outlier" on that
    one parameter alone and, under the old whole-chain reduction, get every other -- perfectly
    well-behaved -- parameter's summary statistics written off along with it. See
    _finalize_chain_selection, which is what actually decides, per parameter, whether that parameter's own
    widespread disagreement should be allowed to veto chain selection.

    A genuinely resolved multimodal posterior (several chains each converged to one of a few real,
    well-populated solutions) is deliberately not what this is meant to catch: a well-populated second mode
    inflates its own MAD enough that this rarely flags it.

    Does not decide what to do with flagged chains -- no cap, no truncation, nothing dropped here. See
    _finalize_chain_selection, which combines this with mcmc._within_chain_r_hat's individual-chain
    non-convergence flags under one shared budget and decides, per lineout, whether to exclude specific
    chains, give up on specific parameters, or give up on the whole lineout.

    Args:
        stacked: (num_chains * num_kept, batch_size, n_active) physical samples, chains concatenated in
            order -- as returned by _physical_samples_for_fit_batch (num_kept identical per chain, since
            all chains in a pooled run share the same config["other"]["mcmc"] settings).
        num_chains: K, the number of independent chains pooled into stacked's leading axis (must be > 1;
            outlier detection is meaningless for a single chain).
        mad_scale: threshold in robust-SD units -- see _DEFAULTS["chain_outlier_mad_scale"].

    Returns:
        flagged: (num_chains, batch_size, n_active) boolean, True where that chain's mean for that
            parameter deviates enough to flag.
    """
    total, batch_size, n_active = stacked.shape
    num_kept = total // num_chains
    by_chain = stacked.reshape(num_chains, num_kept, batch_size, n_active)
    chain_means = by_chain.mean(axis=1)  # (num_chains, batch_size, n_active)

    median = np.median(chain_means, axis=0)  # (batch_size, n_active)
    mad = np.median(np.abs(chain_means - median[None]), axis=0)  # (batch_size, n_active)
    robust_sd = 1.4826 * mad
    # MAD (and therefore robust_sd) is itself computed from a majority vote (the median), so it is exactly
    # 0 whenever *more than half* the chains agree precisely -- including the useful case of a tight
    # majority with a genuinely divergent minority, not just the degenerate all-chains-identical case. A
    # tiny floor (not a "treat as untestable" special case) is what's needed here: it makes an agreeing
    # chain's exactly-zero deviation divide to exactly 0 either way, while a genuinely divergent chain's
    # nonzero deviation against a zero MAD divides out to an enormous (correctly, easily flagged) ratio,
    # rather than being silently suppressed to "not tested."
    absolute_deviation = np.abs(chain_means - median[None])
    deviation = absolute_deviation / np.maximum(robust_sd, 1e-300)
    return deviation > mad_scale  # (num_chains, batch_size, n_active)


def _finalize_chain_selection(bad_within: np.ndarray, bad_outlier: np.ndarray, max_drop_fraction: float, num_kept: int):
    """Combines mcmc._within_chain_r_hat's per-(chain, parameter) non-convergence flags with
    _mad_flagged_chains' per-(chain, parameter) outlier flags under one shared drop budget, and decides,
    per lineout, which active parameters may veto chain selection, which chains to exclude from that
    lineout's summary statistics, and whether to give up on the whole lineout anyway.

    A chain is a candidate to be written off (excluded from mean/std/covariance) on a given parameter if it
    failed *either* check for that parameter: one that never reached a stationary distribution within its
    own sampling budget (bad_within) is just as unreliable, for that parameter, as one that settled
    somewhere genuinely different from the rest (bad_outlier), even though they're detected by entirely
    different means (within a chain vs. across chains).

    Unlike the whole-chain, worst-case-across-parameters version this replaced, a parameter is only allowed
    to *drive* chain selection (decide which chains get written off) if doing so would stay within budget
    on its own: a parameter whose own bad-chain count already exceeds max_drop_fraction is "structurally
    unreliable" for this lineout -- no subset of chains would satisfy the cap using that parameter alone --
    and is excluded from voting rather than being allowed to write off (nearly) every chain and, with it,
    every *other* parameter's perfectly good summary statistics. This is exactly the failure mode a
    genuinely weakly-identified parameter (e.g. an ion temperature the data barely constrain) produces: its
    own chains legitimately scatter across whatever the flat posterior direction spans, which is real
    information about that parameter (see mcmc.py's module docstring on block Metropolis-within-Gibbs) --
    it is not evidence that any *other* parameter failed to converge.

    Parameters excluded from voting are NOT hidden -- their own mean/std are still computed and reported
    (using whichever chains the *voting* parameters selected, or every chain if none can vote), simply
    without the ability to exclude chains on their own behalf; the caller is expected to also report which
    parameters were excluded from voting (mcmc_postprocess.mcmc_postprocess does, as `param_unreliable`),
    so a wide reported uncertainty on such a parameter can be told apart from a genuinely tight one.

    If the chains written off by the *voting* parameters alone exceed max_drop_fraction, that lineout is
    still marked unreliable outright (NaN mean/std/covariance for literally everything) rather than
    reporting a value from whatever's left -- silently capping and keeping the least-bad subset was tried
    and found to hide exactly the cases (most chains failing together on parameters that are NOT merely
    weakly-identified) this is supposed to surface, not paper over. When *no* parameter can vote (every one
    is individually structurally unreliable), no chains are written off at all -- there is no informative
    subset to select without a single agreeing parameter to select it by, so every chain is kept and every
    parameter's own (necessarily wide) statistics are reported from the full, unfiltered pool.

    Args:
        bad_within: (num_chains, batch_size, n_active) boolean, from mcmc._within_chain_r_hat > threshold.
        bad_outlier: (num_chains, batch_size, n_active) boolean, from _mad_flagged_chains.
        max_drop_fraction: hard cap on the droppable fraction of chains -- see
            _DEFAULTS["max_dropped_chain_fraction"].
        num_kept: samples per chain, to expand the per-chain keep/drop decision into a per-sample mask.

    Returns:
        keep_mask: (num_chains * num_kept, batch_size) boolean, True where that pooled sample should be
            included in this lineout's summary statistics. All-False for a lineout marked unreliable (the
            caller is expected to NaN that lineout's summary rather than average an empty selection).
        n_dropped: (batch_size,) int, number of chains written off per lineout by the voting parameters
            (capped or not).
        unreliable: (batch_size,) boolean, True where too many chains were written off (by the voting
            parameters) to trust any subset.
        param_unreliable: (batch_size, n_active) boolean, True where that specific parameter was excluded
            from voting for that lineout (its own bad-chain count alone exceeded the cap) -- independent of
            whether the lineout as a whole ended up `unreliable`.
    """
    num_chains, batch_size, n_active = bad_within.shape
    total_bad = bad_within | bad_outlier  # (num_chains, batch_size, n_active)
    max_droppable = int(np.floor(max_drop_fraction * num_chains))

    n_bad_per_param = total_bad.sum(axis=0)  # (batch_size, n_active)
    param_unreliable = n_bad_per_param > max_droppable  # (batch_size, n_active) -- excluded from voting

    voting = ~param_unreliable  # (batch_size, n_active)
    any_voter = voting.any(axis=-1)  # (batch_size,) -- lineouts with at least one parameter allowed to vote

    # OR-reduce total_bad over only the voting parameters, per lineout; lineouts with no voter at all keep
    # every chain (there's nothing informative to select by), matching np.any's all-False-mask convention
    # of returning False when there is nothing to reduce over -- exactly "drop nothing" here.
    total_bad_voted = np.any(total_bad & voting[None, :, :], axis=-1)  # (num_chains, batch_size)

    n_dropped = total_bad_voted.sum(axis=0)  # (batch_size,)
    unreliable = any_voter & (n_dropped > max_droppable)  # (batch_size,)

    keep_chain = ~total_bad_voted  # (num_chains, batch_size)
    keep_chain[:, unreliable] = False  # nothing from an unreliable lineout feeds its own summary stats

    keep_mask = np.repeat(keep_chain, num_kept, axis=0)  # (num_chains*num_kept, batch_size)
    return keep_mask, n_dropped, unreliable, param_unreliable


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
        # Reuses the caller-supplied nominal loss_fn only when config_k/all_data_k are literally the same
        # objects as the nominal config/all_data -- true for every draw whenever
        # draw_calibration_realizations collapses to K=1, and also true for every draw when num_draws > 1
        # but every *_sigma is 0.0 (chains differing only by starting-point dispersion and/or RNG have
        # nothing calibration-wise to rebuild a LossFunction for). draw_index == 0 is NOT special:
        # draw_calibration_realizations draws an independent perturbation for every index including 0, so
        # unconditionally reusing the nominal loss_fn there (as this used to do) would silently evaluate
        # draw 0's genuinely-perturbed calibration against the wrong (nominal) LossFunction -- a no-op for
        # gain (only a data-side rescale, self-cancelled by LossFunction's per-lineout normalization) but
        # a real mismatch for the dispersion/offset/IRF-width fields, which get baked into FormFactor's
        # fixed arrays (lamAxis, IRF kernels) at construction time.
        reuse_nominal = config_k is config and all_data_k is all_data
        loss_fn_k = loss_fn if reuse_nominal else _build_loss_fn_for_draw(config_k, sa, all_data_k, batch_size)
        loss_fns_by_draw.append(loss_fn_k)
        batches_by_draw.append([build_batch(all_data_k, inds, background_subtract) for inds in batch_indices])

    key = jax.random.PRNGKey(int(mcmc_cfg["seed"]))
    # Captured once here (not inside run_mcmc_pooled) so each calibration draw's raw results can be
    # uploaded to THIS run as soon as that draw finishes -- see mcmc._checkpoint_draw's docstring for why:
    # otherwise a run that dies partway through (walltime, OOM, an unrelated crash) loses every completed
    # draw's results along with the incomplete ones.
    active_run = mlflow.active_run()
    checkpoint_run_id = active_run.info.run_id if active_run is not None else None
    pooled_diff_params, static_params, diagnostics_by_draw, max_r_hat_by_batch, within_chain_r_hat_by_batch = (
        mcmc.run_mcmc_pooled(
            config, loss_fns_by_draw, fitted_weights, batches_by_draw, key, checkpoint_run_id=checkpoint_run_id
        )
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
    # Only meaningful with >= 2 independent chains (see mcmc.run_mcmc_pooled/_max_r_hat_across_chains);
    # stays all-NaN (and unplotted) whenever max_r_hat_by_batch is None.
    max_r_hat = np.full(total_lineouts, np.nan) if max_r_hat_by_batch is not None else None
    num_chains = len(draws)
    # n_chains_dropped/lineout_unreliable are meaningful even for a single chain (K=1): a chain can still
    # individually fail to converge within its own sampling budget -- see mcmc._within_chain_r_hat and
    # _finalize_chain_selection. Only n_active == 0 (nothing to have converged or not) skips this entirely.
    n_chains_dropped = np.zeros(total_lineouts, dtype=int) if n_active > 0 else None
    lineout_unreliable = np.zeros(total_lineouts, dtype=bool) if n_active > 0 else None
    # Per-parameter counterpart to lineout_unreliable -- True where that specific parameter was excluded
    # from voting on chain selection for that lineout (mcmc._finalize_chain_selection's param_unreliable),
    # OR'd with lineout_unreliable itself (a lineout marked unreliable overall NaNs every parameter's
    # reported value, this one included). See _finalize_chain_selection's docstring for why a parameter
    # can be excluded from voting without its own mean/std being hidden -- this is what lets a caller tell
    # "reported, but based on chains that individually disagree a lot" apart from "reported, and those
    # chains actually agree."
    param_unreliable = np.zeros((total_lineouts, n_active), dtype=bool) if n_active > 0 else None
    # Cached here so the (optional) sample-saving block below can reuse each fit-batch's reconstruction
    # instead of recomputing it. Always the *full*, unfiltered pooled samples: mcmc_samples.nc (the raw
    # posterior save) always includes every chain, written off or not -- only the summary statistics below
    # and the per-lineout corner plots further down (via keep_mask_by_fit_batch) are filtered.
    physical_by_fit_batch = []
    # Parallel to physical_by_fit_batch -- (num_chains*num_kept, this_batch_size) boolean per fit-batch,
    # reused by the corner-plot loop below so it shows the same filtered chains the quoted mean/std/
    # covariance were actually computed from (or, for a lineout marked unreliable, the full unfiltered
    # pool, since there's no valid "cleaned" subset to show for one).
    keep_mask_by_fit_batch = []

    for b, inds in enumerate(batch_indices):
        stacked = _physical_samples_for_fit_batch(static_array_part, static_nonarray_part, pooled_diff_params, b, active_keys)
        physical_by_fit_batch.append(stacked)

        if n_active > 0:
            bad_within = np.asarray(within_chain_r_hat_by_batch[:, b, :, :]) > mcmc_cfg["within_chain_r_hat_threshold"]
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
            # An unreliable lineout has no valid "cleaned" subset -- fall back to the full unfiltered pool
            # for its corner plot (still useful for diagnosing *why* it's unreliable), rather than the
            # empty selection _finalize_chain_selection deliberately returns for summary-statistics purposes.
            keep_mask_for_plotting = np.where(unreliable[None, :], True, keep_mask)
            keep_mask_by_fit_batch.append(keep_mask_for_plotting)

            for lineout_local, lineout_global in enumerate(inds):
                if unreliable[lineout_local]:
                    continue  # mean/std/covariance stay NaN -- see all_params_mean/std/covariance init above
                kept = stacked[keep_mask[:, lineout_local], lineout_local, :]
                means = kept.mean(axis=0)
                stds = kept.std(axis=0)
                for a, (species, key_name) in enumerate(active_keys):
                    all_params_mean[species][key_name][lineout_global] = means[a]
                    all_params_std[species][key_name][lineout_global] = stds[a]
                covariance[lineout_global] = np.atleast_2d(np.cov(kept, rowvar=False))

        rates = np.mean([np.asarray(diag["acceptance_rate"])[b] for diag in diagnostics_by_draw], axis=0)
        acceptance_rate[inds] = rates
        if max_r_hat is not None:
            # Reduced to one worst-case-across-parameters number per lineout only for this plot's own
            # "before chain filtering" panel (plotters.plot_mcmc_diagnostics) -- an intentionally raw,
            # unfiltered snapshot, unlike the (now per-parameter-aware) chain-selection/reliability
            # decisions above, which no longer let one parameter's own R-hat drive this reduction.
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
        # Per-parameter breakdown of param_unreliable -- how many lineouts had *this* parameter excluded
        # from chain-selection voting (see _finalize_chain_selection), independent of the OR'd-in whole-
        # lineout unreliable count above. Surfaces exactly which parameter(s) are structurally weakly
        # identified across the shot (e.g. an ion temperature) rather than leaving that implicit in a
        # single aggregate count.
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
            # A capped, evenly-spaced subset (matching plotters.plot_ang_lineouts' convention) rather
            # than one corner plot per lineout, since a full fit can have far more lineouts than are
            # useful to eyeball individually. Every lineout's full posterior is still preserved in
            # mcmc_samples.nc below (when save_samples is true, the default), so any lineout not covered
            # here can be corner-plotted after the fact -- see plotters.plot_corner's docstring.
            n_corner_lineouts = min(8, total_lineouts)
            corner_targets = set(np.linspace(0, total_lineouts - 1, n_corner_lineouts, dtype=int).tolist())
            lineout_vals = np.array(config["data"]["lineouts"]["val"])
            for b, inds in enumerate(batch_indices):
                stacked = physical_by_fit_batch[b]
                keep_mask = keep_mask_by_fit_batch[b]
                for lineout_local, lineout_global in enumerate(inds):
                    if lineout_global in corner_targets:
                        # Same filtering the quoted mean/std/covariance were computed from (see
                        # _finalize_chain_selection), so the corner plot shows exactly what the reported
                        # uncertainty is based on rather than the raw, potentially outlier-contaminated
                        # pool -- a written-off chain is still fully preserved in mcmc_samples.nc below,
                        # just not here. A lineout marked unreliable instead falls back to showing every
                        # chain (keep_mask_by_fit_batch already encodes that fallback -- see its
                        # construction above), since there's no valid "cleaned" subset to show for one.
                        # num_chains_kept matches whichever of those two cases applies, so the
                        # chain-coloring in plot_corner lines up with the samples actually shown.
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

    # Nested species -> key_name layout, matching all_params_mean/all_params_std's own convention, so a
    # caller can look up "was ion-1's Ti reliable for this lineout" the same way it looks up Ti's mean.
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
