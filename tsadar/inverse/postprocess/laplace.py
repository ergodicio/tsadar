"""postprocess: recomputes final fits/losses/uncertainties after fitting completes, optionally refits
individually poor-fit lineouts, and produces the resulting plots and saved parameter values."""
from typing import Dict
from collections import defaultdict
from flatten_dict import flatten, unflatten

import time, tempfile, mlflow, os, copy

import numpy as np
import jax

from tsadar.utils import manifest
from tsadar.utils.plotting import plotters
from ..loss_function import LossFunction
from tsadar.core.modules.ts_params import IonParams
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
                hess = loss_fn.h_loss_wrt_params(fitted_weights[i_batch], batch)
            except Exception as e:
                print(f"Error calculating Hessian, no hessian based uncertainties have been calculated: {e}")
                calc_sigma = False

        losses[inds] = loss

        sqdevs["ele"][inds] = sqds["ele"]
        sqdevs["ion"][inds] = sqds["ion"]

        if calc_sigma:
            sigmas[inds] = get_sigmas(hess, config["optimizer"]["batch_size"])
            # print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}") number of negatives?

        fits["ele"]["total_spec"][inds] = ThryE
        fits["ion"]["total_spec"][inds] = ThryI

    return losses, sqdevs, fits, sigmas


def get_sigmas(hess: Dict, batch_size: int) -> Dict:
    """
    Calculates the variance using the hessian with respect to the parameters and then using the hessian values
    as the inverse of the covariance matrix and then inverting that. Negatives in the inverse hessian normally indicate
    non-optimal points, to represent this in the final result the uncertainty of those values are reported as negative.


    Args:
        hess: Hessian dictionary, the field for each fitted parameter has subfields corresponding to each of the other
            fitted parameters. Within each nested subfield is a batch_size x batch_size array with the hessian values
            for that parameter combination and that batch. The cross terms of this array are zero since separate
            lineouts within a batch do not affect each other, they are therefore discarded
        batch_size: int- number of lineouts in a batch

    Returns:
        sigmas: batch_size x number_of_parameters array with the uncertainty values for each parameter
    """
    sizes = {
        key + species: hess[species][key][species][key].shape[1]
        for species in hess.keys()
        for key in hess[species].keys()
    }
    actual_num_params = sum([v for k, v in sizes.items()])
    sigmas = np.zeros((batch_size, actual_num_params))

    for i in range(batch_size):
        temp = np.zeros((actual_num_params, actual_num_params))
        k1 = 0
        for species1 in hess.keys():
            for key1 in hess[species1].keys():
                k2 = 0
                for species2 in hess.keys():
                    for key2 in hess[species2].keys():
                        temp[k1, k2] = np.squeeze(hess[species1][key1][species2][key2])[i, i]
                        k2 += 1
                k1 += 1

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

    # Calculate sigmas if needed
    sigmas = None
    if config["other"]["calc_sigmas"]:
        active_params = loss_fn.spec_calc.get_plasma_parameters(fitted_weights, return_static_params=False)
        hess = loss_fn.h_loss_wrt_params(active_params, batch)
        sigmas = get_sigmas(hess, config["optimizer"]["batch_size"])
        print(f"Number of 0s in sigma: {np.count_nonzero(sigmas==0)}")

    # Logging and plotting
    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    final_params = plotters.get_final_params(config, all_params, all_axes, td)
    sigma_fe = None
    if "fe" in final_params:
        if config["other"]["calc_sigmas"]:
            sigma_fe = plotters.save_sigmas_fe(final_params, {}, sigmas, td)
        else:
            sigma_fe = np.zeros_like(final_params['fe'])

    savedata = plotters.plot_data_angular(config, fits, all_data, all_axes, td)
    plotters.plot_ang_lineouts(1, sqdevs, losses, all_params, all_axes, savedata, td)
    if config["parameters"]["electron"]["fe"]["type"] != 'dlm':
        plotters.plot_dist(config, final_params, sigma_fe, td)
    
    return t1, final_params
