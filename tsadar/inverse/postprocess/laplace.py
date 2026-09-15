"""postprocess: recomputes final fits/losses/uncertainties after fitting completes, optionally refits
individually poor-fit lineouts, and produces the resulting plots and saved parameter values."""
from typing import Dict, List, Tuple
from collections import defaultdict
from flatten_dict import flatten, unflatten
import json

import time, tempfile, mlflow, os, copy

import numpy as np
import jax
import equinox as eqx

from tsadar.utils import manifest
from ..loss_function import LossFunction
from tsadar.core.modules.ts_params import IonParams, get_filter_spec
from ..loops import one_d_loop, unbatch_fitted_params, build_batch, build_angular_batch
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic


def recalculate_with_chosen_weights(
    config: Dict,
    sa,
    sample_indices,
    all_data: Dict,
    loss_fn: LossFunction,
    calc_sigma: bool,
    fitted_weights: Dict,
    num_params: int,
):
    """
    Gets parameters and the result of the full forward pass i.e. fits


    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices:
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        loss_fn: Instance of the LossFunction class
        fitted_weights: Dict- best values of the parameters returned by the minimizer
        num_params: int- number of active fitted parameters, used to size the sigmas array

    Returns:

    """

    losses = np.zeros_like(sample_indices, dtype=np.float64)
    sample_indices.sort()
    batch_indices = np.reshape(sample_indices, (-1, config["optimizer"]["batch_size"]))

    fits = {
        "ele": {
            "total_spec": np.zeros(all_data["e_data"].shape),
            "IRF": np.zeros(all_data["e_data"].shape),
            "noise": np.zeros(all_data["e_data"].shape),
        },
        "ion": {
            "total_spec": np.zeros(all_data["i_data"].shape),
            "IRF": np.zeros(all_data["i_data"].shape),
            "noise": np.zeros(all_data["i_data"].shape),
        },
    }
    sqdevs = {"ion": np.zeros(all_data["i_data"].shape), "ele": np.zeros(all_data["e_data"].shape)}
    sigmas = None

    for species, data_key in (("ele", "e_data"), ("ion", "i_data")):
        if config["data"][f"load_{species}_spec"]:
            sigmas = np.zeros((all_data[data_key].shape[0], num_params))
            fits[species]["spec_comps"] = np.ones(
                [
                    all_data[data_key].shape[0],
                    max(
                        config["parameters"]["general"]["Te_gradient"]["num_grad_points"],
                        config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
                    ),
                    all_data[data_key].shape[1] * config["other"]["points_per_pixel"],
                    len(sa["sa"]),
                ]
            )
        else:
            fits[species]["spec_comps"] = np.zeros(all_data[data_key].shape)

    background_subtract = config["data"]["background"]["bg_subtract"]
    if config["plotting"]["detailed_breakdown"]:
        ts_diag = ThomsonScatteringDiagnostic(config, sa)
    for i_batch, inds in enumerate(batch_indices):
        batch = build_batch(all_data, inds, background_subtract)

        loss, sqds, ThryE, ThryI, _ = loss_fn.array_loss(fitted_weights[i_batch], batch)

        if config["plotting"]["detailed_breakdown"]:
            # ThryE, ThryI, modlE, modlI, eIRF, iIRF, lamAxisE, lamAxisI = filter_jit(ts_diag.sprectrum_breakdown)(fitted_weights[i_batch], batch)
            ThryE, ThryI, modlE, modlI, eIRF, iIRF, _, _, lamAxisE_raw, lamAxisI_raw = ts_diag.spectrum_breakdown(
                fitted_weights[i_batch], batch
            )
            fits["ele"]["spec_comps"][inds] = modlE
            fits["ion"]["spec_comps"][inds] = modlI
            fits["ele"]["IRF"][inds] = eIRF
            fits["ion"]["IRF"][inds] = iIRF
            fits["ele"]["noise"][inds] = all_data["noiseE"][inds]
            fits["ion"]["noise"][inds] = all_data["noiseI"][inds]
            fits["ele"]["detailed_axis"] = lamAxisE_raw[0]
            fits["ion"]["detailed_axis"] = lamAxisI_raw[0]

        if calc_sigma:
            try:
                # Hessian restricted to only the active fit parameters (diff_params) -- see
                # LossFunction.h_loss_wrt_params's docstring for why the full parameter tree must never
                # be differentiated here (it pulls in every fixed array the model carries, e.g. the
                # electron distribution function's interpolation table, and has been observed to attempt
                # a >150GB allocation on an ordinary fit).
                ts_params = fitted_weights[i_batch]
                filter_spec = get_filter_spec(config["parameters"], ts_params)
                diff_params, static_params = eqx.partition(ts_params, filter_spec)
                hess = loss_fn.h_loss_wrt_params(diff_params, static_params, batch)
                fitted_params_this_batch, _ = ts_params.get_fitted_params(config["parameters"])
                sigmas[inds] = get_sigmas(hess, diff_params, fitted_params_this_batch, config["optimizer"]["batch_size"])
            except Exception as e:
                print(f"Error calculating Hessian, no hessian based uncertainties have been calculated: {e}")
                calc_sigma = False

        losses[inds] = loss

        sqdevs["ele"][inds] = sqds["ele"]
        sqdevs["ion"][inds] = sqds["ion"]

        fits["ele"]["total_spec"][inds] = ThryE
        fits["ion"]["total_spec"][inds] = ThryI

    return losses, sqdevs, fits, sigmas


def _named_diff_leaves(diff_params) -> List[Tuple[str, str]]:
    """Returns the (species, key) name of every active leaf of a diff_params pytree (as produced by
    eqx.partition(ts_params, get_filter_spec(...))), in the same order as jax.tree_util.tree_leaves(
    diff_params) -- i.e. the same order get_sigmas' Hessian `rows` come back in.

    Names mirror get_filter_spec's own species/key convention: species is "electron"/"general"/
    "ion-<n>" (1-based); key strips diff_params' "normed_" prefix (get_filter_spec never prefixes
    "fract"). This is the inverse of get_filter_spec's getattr-based navigation, recovered here via
    tree_flatten_with_path since diff_params carries no other record of which (species, key) each leaf
    came from.
    """
    names = []
    for path, _leaf in jax.tree_util.tree_flatten_with_path(diff_params)[0]:
        top = path[0].name
        if top == "ions":
            species = f"ion-{path[1].idx + 1}"
        else:
            species = top
        attr = path[-1].name
        key = attr[len("normed_") :] if attr.startswith("normed_") else attr
        names.append((species, key))
    return names


def get_sigmas(hess, diff_params, fitted_params: Dict, batch_size: int) -> np.ndarray:
    """
    Calculates parameter uncertainty from a Hessian, using the hessian values as the inverse of the
    covariance matrix and then inverting that. Negatives in the inverse hessian normally indicate
    non-optimal points, to represent this in the final result the uncertainty of those values are
    reported as negative.

    hess must be the Hessian of the loss wrt diff_params ONLY (see LossFunction.h_loss_wrt_params) --
    never the full ThomsonParams tree, which pulls in every fixed array the model carries (e.g. the
    electron distribution function's interpolation table) and has been observed to attempt a >150GB
    allocation on an ordinary fit.

    Args:
        hess: Hessian of the loss wrt diff_params, as returned by LossFunction.h_loss_wrt_params. Has
            diff_params' pytree structure at the outer level; each "leaf" there is itself a
            diff_params-shaped subtree of second derivatives wrt that one leaf.
        diff_params: the same diff_params pytree the Hessian was taken wrt (only active leaves; every
            other leaf is None, per eqx.partition).
        fitted_params: nested dict as returned by ThomsonParams.get_fitted_params(config["parameters"])
            -- gives the exact (species, key) columns and order the returned array must match, since
            that's what plotters.save_sigmas_params/save_sigmas_fe assume of `all_params`.
        batch_size: int- number of lineouts in a batch

    Returns:
        sigmas: batch_size x number_of_parameters array with the uncertainty values for each parameter,
            columns ordered to match fitted_params (species-major, then key, in fitted_params' own
            iteration order).

    Raises:
        NotImplementedError: if any electron distribution-function parameter ("fe"/"f"/"flm"/"m") is
            active. Those are stored as a list of separate per-lineout objects rather than one array with
            a batch axis (see ElectronParams.init_dists), which this leaf-diagonal approach does not
            handle -- mirroring postprocess.mcmc's identical fe-active restriction (see its module
            docstring).
    """
    for species, params in fitted_params.items():
        unsupported = set(params.keys()) & {"fe", "f", "flm", "m"}
        if unsupported:
            raise NotImplementedError(
                f"get_sigmas does not support the electron distribution function as an active fit "
                f"parameter (found {sorted(unsupported)} under {species!r}): its per-lineout parameters "
                "are stored as a list of separate objects rather than one array with a batch axis, which "
                "this leaf-diagonal approach does not handle. Deactivate 'electron.fe.active' to use "
                "calc_sigmas for the remaining (scalar) active parameters."
            )

    ordered_names = [(species, key) for species, params in fitted_params.items() for key in params.keys()]
    num_params = len(ordered_names)
    sigmas = np.full((batch_size, num_params), np.nan)
    if num_params == 0:
        return sigmas

    target_structure = jax.tree_util.tree_structure(diff_params)
    rows = jax.tree_util.tree_leaves(
        hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure
    )
    row_names = _named_diff_leaves(diff_params)
    if len(rows) != len(row_names):
        raise ValueError(f"Unexpected Hessian structure: found {len(rows)} row(s), expected {len(row_names)}")
    name_to_row_index = {name: idx for idx, name in enumerate(row_names)}
    if any(name not in name_to_row_index for name in ordered_names):
        missing = [name for name in ordered_names if name not in name_to_row_index]
        raise ValueError(f"fitted_params names not found among diff_params' active leaves: {missing}")

    blocks = [jax.tree_util.tree_leaves(row) for row in rows]  # blocks[k1][k2] is d^2L/d(leaf k1) d(leaf k2)
    # Permutation from ordered_names' (fitted_params') order into rows'/blocks' (diff_params') order --
    # the two need not agree, since ThomsonParams.get_unnormed_params() (which fitted_params is ultimately
    # derived from) lists species as electron/general/ion-<n>, while diff_params' own pytree flatten order
    # follows ThomsonParams' declared field order, electron/ions/general.
    perm = [name_to_row_index[name] for name in ordered_names]

    for i in range(batch_size):
        temp = np.array(
            [[np.asarray(blocks[perm[k1]][perm[k2]])[i, i] for k2 in range(num_params)] for k1 in range(num_params)]
        )
        inv = np.linalg.inv(temp)
        sigmas[i, :] = np.sign(np.diag(inv)) * np.sqrt(np.abs(np.diag(inv)))

    return sigmas


def postprocess(
    config, sample_indices, all_data: Dict, all_axes: Dict, loss_fn, sa, fitted_weights, all_params=None, num_params=None
):
    """
    Top-level postprocessing entry point, run after a fit completes. For non-angular fits with refitting
    enabled, first refits any lineout whose loss exceeds the configured threshold (see refit_bad_fits) and
    re-unbatches the (possibly updated) fitted weights. Then dispatches to process_angular_data or
    process_data depending on spectype, logs timing/status to mlflow, and returns the final parameters.

    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices: indices of the lineouts that were fit
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        all_axes: Dict- calibrated axes and axes labels
        loss_fn: Instance of the LossFunction class used for fitting
        sa: scattering angles and their relative weights
        fitted_weights: best-fit parameter object(s) returned by the minimizer
        all_params: Dict, optional- unbatched fitted parameters; required unless refitting is enabled
            (in which case it is recomputed here) or the angular path is used (which builds its own)
        num_params: int, optional- number of active fitted parameters; same caveats as all_params

    Returns:
        final_params: Dict- the final fitted parameters and distribution function data, as returned by
        process_data/process_angular_data
    """
    t1 = time.time()

    if config["other"]["extraoptions"]["spectype"] != "angular_full" and config["other"]["refit"]:
        init_losses = refit_bad_fits(config, sa, sample_indices, all_data, loss_fn, fitted_weights, num_params)
        all_params, num_params = unbatch_fitted_params(config, fitted_weights)
    else:
        init_losses = []

    mlflow.log_metrics({"refitting time": round(time.time() - t1, 2)})

    with tempfile.TemporaryDirectory() as td:
        _ = [os.makedirs(os.path.join(td, dirname), exist_ok=True) for dirname in ["plots", "binary", "csv"]]
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            t1, final_params = process_angular_data(
                config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, t1, td
            )

        else:
            t1, final_params = process_data(
                config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, init_losses, t1, td,
                all_params, num_params
            )

        # Written last, so it describes the finished tree rather than a
        # hardcoded list of what should be in it (ergodicio/tsadar#116).
        manifest.write_manifest(td, mode="fit")

        mlflow.log_artifacts(td)
    mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})

    mlflow.set_tag("status", "done plotting")

    return final_params


def refit_bad_fits(config, sa, batch_indices, all_data, loss_fn, fitted_weights, num_params):
    """
    Refits individual lineouts whose loss exceeds config["other"]["refit_thresh"], one lineout at a time
    (batch_size=1), using the previous lineout's fitted weights as the initial guess. If the refit improves
    on the original loss, the corresponding entry in fitted_weights is updated in place; lineout 0 is never
    refit since there is no preceding lineout to initialize from.

    Args:
        config: Dict- configuration dictionary built from input deck
        sa: scattering angles and their relative weights
        batch_indices: np.ndarray- indices specifying how the data was split into batches during fitting
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        loss_fn: Instance of the LossFunction class used for fitting
        fitted_weights: List- per-batch fitted weight objects; mutated in place for any lineout that is
            successfully refit
        num_params: int- number of active fitted parameters, used to size the sigmas array

    Returns:
        losses_init: np.ndarray- the per-lineout losses computed before any refitting was applied
    """
    losses_init, sqdevs, fits, sigmas = recalculate_with_chosen_weights(
        config, sa, batch_indices, all_data, loss_fn, False, fitted_weights, num_params
    )

    # refit bad fits
    # reduced_points = (used_points - num_params)*config["optimizer"]["batch_size"]

    red_losses_init = losses_init  # / (1.1 * reduced_points) by changing losses to mean this is loss per point
    true_batch_size = config["optimizer"]["batch_size"]

    mlflow.log_metrics({"number of fits": len(batch_indices.flatten())})
    mlflow.log_metrics({"number of refits": int(np.sum(red_losses_init > config["other"]["refit_thresh"]))})

    for i in batch_indices.flatten()[red_losses_init > config["other"]["refit_thresh"]]:
        if i == 0:
            continue

        temp_cfg = copy.deepcopy(config)
        temp_cfg["optimizer"]["batch_size"] = 1

        def extract(x):
            # i, true_batch_size would idealy be inputs but i cant figure out how to pass variables
            if isinstance(x, list) or len(np.shape(x)) > 0:
                return x[(i - 1) % true_batch_size]
            else:
                return x

        def insert(x, y):
            # i, true_batch_size
            if isinstance(x, list):
                x[i % true_batch_size] = y[0]
                return x
            elif len(np.shape(x)) > 0:
                x = x.at[i % true_batch_size].set(y[0])
                return x
            else:
                return y

        prev_weights = fitted_weights[(i - 1) // true_batch_size]
        prev_weights = jax.tree.map(
            extract, prev_weights, is_leaf=lambda x: isinstance(x, list) and not isinstance(x[0], IonParams)
        )
        prev_weights = prev_weights.get_unnormed_params()
        prev_weights = jax.tree.map(lambda x: {"val": x}, prev_weights)
        if config["parameters"]["electron"]["fe"]["type"].casefold() == "dlm":
            prev_weights["electron"]["fe"] = {"params": {"m": prev_weights["electron"].pop("m")}}
        else:
            # Arbitrary1V always rebuilds fval from params.init_m and has no config-driven override
            # for "f" (get_unnormed_params()'s key here), so there's nothing to carry over for it.
            prev_weights["electron"].pop("f", None)

        temp_params = flatten(temp_cfg["parameters"])
        temp_params.update(flatten(prev_weights))
        temp_cfg["parameters"] = unflatten(temp_params)
        # temp_cfg["parameters"] = temp_cfg["parameters"] | prev_weights
        new_weights, _, loss_fn = one_d_loop(temp_cfg, all_data, sa, np.array([i]), 1)

        inds = np.array([i])
        batch = build_batch(all_data, inds, config["data"]["background"]["bg_subtract"])
        loss, _, _, _, _ = loss_fn.array_loss(new_weights[0], batch)

        if loss < losses_init[i]:
            fitted_weights[(i - 1) // true_batch_size] = jax.tree.map(
                insert,
                fitted_weights[(i - 1) // true_batch_size],
                new_weights[0],
                is_leaf=lambda x: isinstance(x, list) and not isinstance(x[0], IonParams),
            )
    return losses_init


def process_data(config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, losses_init, t1, td, all_params, num_params):
    """
    Non-angular postprocessing path: recomputes losses, fits, and (if enabled) parameter uncertainties for
    the final fitted weights, then produces the loss-histogram, data-vs-fit, best/worst-lineout comparison
    (detailed or simple, depending on config["plotting"]["detailed_breakdown"]), and final-parameter plots,
    saving them all to td.

    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices: indices of the lineouts that were fit
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        all_axes: Dict- calibrated axes and axes labels
        loss_fn: Instance of the LossFunction class used for fitting
        fitted_weights: best-fit parameter object(s) returned by the minimizer
        sa: scattering angles and their relative weights
        losses_init: np.ndarray- initial (pre-refit) losses, or an empty list if refitting was not performed
        t1: float- timestamp used to measure and log the postprocessing duration
        td: str- temporary directory that will be uploaded to mlflow
        all_params: Dict- unbatched fitted parameters, as returned by unbatch_fitted_params
        num_params: int- number of active fitted parameters, used to size the sigmas array

    Returns:
        tuple:
            t1 (float): updated timestamp, taken after recomputing losses/fits, for timing the plotting step
            final_params (Dict): the final fitted parameters and distribution function data
    """
    from tsadar.utils.plotting import plotters

    losses, sqdevs, fits, sigmas = recalculate_with_chosen_weights(
        config, sa, sample_indices, all_data, loss_fn, config["other"]["calc_sigmas"], fitted_weights, num_params
    )

    reduced_points = 1.0  # (used_points - num_params)*config["optimizer"]["batch_size"]

    if len(losses_init) == 0:
        losses_init = losses
    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    final_params = plotters.get_final_params(config, all_params, all_axes, td)

    red_losses = plotters.plot_loss_hist(config, losses_init, losses, reduced_points, td)
    savedata = plotters.plot_ts_data(config, fits, all_data, all_axes, td)
    if config["plotting"]["detailed_breakdown"]:
        plotters.detailed_lineouts(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td)
    #elif
    else:
        plotters.model_v_actual(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td)
    sigma_ds = plotters.save_sigmas_params(config, all_params, sigmas, all_axes, td)
    plotters.plot_final_params(config, all_params, sigma_ds, td)
    return t1, final_params


def process_angular_data(config, batch_indices, all_data, all_axes, loss_fn, fitted_weights, sa, t1, td):
    """
    Angular postprocessing path: extracts the fitted parameters from the single fitted_weights object,
    builds the angular data batch, computes losses/fits (and, if enabled, parameter uncertainties), and
    produces the angular-specific data-vs-fit, lineout, and distribution-function plots, saving them to td.

    Args:
        config: Dict- configuration dictionary built from input deck
        batch_indices: indices of the lineouts that were fit
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        all_axes: Dict- calibrated axes and axes labels
        loss_fn: Instance of the LossFunction class used for fitting
        fitted_weights: the single fitted-weight object returned by the angular minimizer
        sa: scattering angles and their relative weights
        t1: float- timestamp used to measure and log the postprocessing duration
        td: str- temporary directory that will be uploaded to mlflow

    Returns:
        tuple:
            t1 (float): updated timestamp, taken after recomputing losses/fits, for timing the plotting step
            final_params (Dict): the final fitted parameters and distribution function data
    """
    from tsadar.utils.plotting import plotters

    # Prepare parameter containers
    all_params = {k: defaultdict(list) for k in config["parameters"].keys()}
    batch_fitted_params, num_params = fitted_weights.get_fitted_params(config["parameters"])
    for k in batch_fitted_params:
        for k2 in batch_fitted_params[k]:
            all_params[k][k2].append(batch_fitted_params[k][k2])

   # Prepare batch data
    batch = build_angular_batch(config, all_data)

    # Calculate losses and fits
    losses, sqdevs, fits_ele, _, params = loss_fn.array_loss(fitted_weights, batch)
    fits = {"ele": fits_ele}
    all_params["electron"]["v"] = params["electron"]["v"]

    # Persist the actual likelihood diagnostics, rather than only the unweighted
    # squared-deviation image used by the historical plotting code.
    diagnostic_arrays, objective_terms = loss_fn.angular_diagnostics(fitted_weights, batch)
    np.savez_compressed(os.path.join(td, "angular_objective_diagnostics.npz"), **diagnostic_arrays)
    with open(os.path.join(td, "angular_objective_terms.json"), "w") as file:
        json.dump(objective_terms, file, indent=2, sort_keys=True)
    mlflow.log_metrics({f"arts2d_{key}": value for key, value in objective_terms.items()})

    # Calculate sigmas if needed. Hessian restricted to only the active fit parameters (diff_params) --
    # see LossFunction.h_loss_wrt_params's docstring for why the full parameter tree must never be
    # differentiated here.
    sigmas = None
    if config["other"]["calc_sigmas"]:
        try:
            filter_spec = get_filter_spec(config["parameters"], fitted_weights)
            diff_params, static_params = eqx.partition(fitted_weights, filter_spec)
            hess = loss_fn.h_loss_wrt_params(diff_params, static_params, batch)
            sigmas = get_sigmas(hess, diff_params, batch_fitted_params, config["optimizer"]["batch_size"])
            print(f"Number of 0s in sigma: {np.count_nonzero(sigmas==0)}")
        except Exception as e:
            print(f"Error calculating Hessian, no hessian based uncertainties have been calculated: {e}")
            sigmas = None

    # Logging and plotting
    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    final_params = plotters.get_final_params(config, all_params, all_axes, td)
    sigma_fe = None
    if "fe" in final_params:
        if sigmas is not None:
            sigma_fe = plotters.save_sigmas_fe(final_params, {}, sigmas, td)
        else:
            sigma_fe = np.zeros_like(final_params['fe'])

    savedata = plotters.plot_data_angular(config, fits, all_data, all_axes, td)
    plotters.plot_ang_lineouts(1, sqdevs, losses, all_params, all_axes, savedata, td)
    if config["parameters"]["electron"]["fe"]["type"] != 'dlm':
        plotters.plot_dist(config, final_params, sigma_fe, td)
    
    return t1, final_params
