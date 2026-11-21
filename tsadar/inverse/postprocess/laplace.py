"""postprocess: recomputes final fits/losses/uncertainties after fitting completes, optionally refits
individually poor-fit lineouts, and produces the resulting plots and saved parameter values."""
from typing import Dict, List, Tuple
from collections import defaultdict
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
                sigmas[inds] = calc_sigmas(config, loss_fn, fitted_weights[i_batch], batch)
            except Exception as e:
                print(f"Error calculating uncertainties, none have been calculated: {e}")
                calc_sigma = False
                # a failed calculation is reported as NaN, never as zero uncertainty
                sigmas[:] = np.nan

        losses[inds] = loss

        sqdevs["ele"][inds] = sqds["ele"]
        sqdevs["ion"][inds] = sqds["ion"]

        fits["ele"]["total_spec"][inds] = ThryE
        fits["ion"]["total_spec"][inds] = ThryI

    return losses, sqdevs, fits, sigmas


def _check_fe_not_fit(config: Dict) -> None:
    """Raises if the electron distribution function is being fit: Hessian uncertainties are only
    computed for the scalar parameters. The 1D DLM shape parameter m is the one case intended to be
    supported; it is not handled yet."""
    fe = config["parameters"]["electron"]["fe"]
    if fe["active"]:
        if fe["dim"] == 1 and fe["type"].casefold() == "dlm":
            raise NotImplementedError("calc_sigmas does not yet support fitting the DLM shape parameter m.")
        raise NotImplementedError("calc_sigmas does not support a fitted electron distribution function.")


def _named_diff_leaves(diff_params) -> List[Tuple[str, str]]:
    """(species, key) name of every active leaf of a diff_params pytree, in jax.tree_util.tree_leaves
    order. species is "electron"/"general"/"ion-<n>" (1-based) and key has the "normed_" prefix removed.
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


def get_sigmas(hess, diff_params, static_params, fitted_params: Dict, batch_size: int) -> np.ndarray:
    """
    Calculates parameter uncertainty, in physical units, from the Hessian of the -2*log-likelihood. The
    covariance in the fitted (normalized) coordinates is 2 * inverse(Hessian); it is converted to the
    physical parameters with the Jacobian of the parameter transform. Negatives in the covariance
    normally indicate non-optimal points, to represent this in the final result the uncertainty of those
    values are reported as negative.

    Args:
        hess: per-lineout Hessian of the loss wrt diff_params, as returned by
            LossFunction.h_loss_wrt_params_per_lineout.
        diff_params: the diff_params pytree the Hessian was taken with respect to.
        static_params: the complementary partition of the same ThomsonParams.
        fitted_params: nested dict as returned by ThomsonParams.get_fitted_params; sets the columns and
            order of the returned array.
        batch_size: int- number of lineouts in a batch

    Returns:
        sigmas: batch_size x number_of_parameters array with the uncertainty of each parameter in
            physical units, columns ordered to match fitted_params.

    Raises:
        NotImplementedError: if the DLM shape parameter m is among the fitted parameters.
    """
    ordered_names = _sigma_columns(fitted_params)
    num_params = len(ordered_names)
    sigmas = np.full((batch_size, num_params), np.nan)
    if num_params == 0:
        return sigmas

    target_structure = jax.tree_util.tree_structure(diff_params)
    rows = jax.tree_util.tree_leaves(
        hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure
    )
    n_leaves = len(_named_diff_leaves(diff_params))
    if len(rows) != n_leaves:
        raise ValueError(f"Unexpected Hessian structure: found {len(rows)} row(s), expected {n_leaves}")
    perm, jacobian_at = _leaf_to_physical(diff_params, static_params, ordered_names, batch_size)

    blocks = [jax.tree_util.tree_leaves(row) for row in rows]  # blocks[k1][k2][i] is d^2L/d(leaf k1[i]) d(leaf k2[i])

    for i in range(batch_size):
        temp = np.array(
            [[np.asarray(blocks[perm[k1]][perm[k2]]).reshape(-1)[i] for k2 in range(num_params)] for k1 in range(num_params)]
        )
        jac = jacobian_at(i)
        covariance = jac @ (2.0 * np.linalg.inv(temp)) @ jac.T
        sigmas[i, :] = np.sign(np.diag(covariance)) * np.sqrt(np.abs(np.diag(covariance)))

    return sigmas


def _sigma_columns(fitted_params: Dict) -> List[Tuple[str, str]]:
    """(species, key) of every sigma column, in fitted_params order."""
    ordered_names = [
        (species, key)
        for species, params in fitted_params.items()
        for key in params.keys()
        if key not in ("fe", "f", "flm")
    ]
    if any(key == "m" for _, key in ordered_names):
        raise NotImplementedError("get_sigmas does not yet support fitting the DLM shape parameter m.")
    return ordered_names


def _leaf_to_physical(diff_params, static_params, ordered_names, batch_size: int):
    """Positions of ordered_names among diff_params' leaves, and a function of the lineout giving the Jacobian
    d(physical parameter a)/d(leaf perm[b]) in those columns."""
    name_to_row_index = {name: idx for idx, name in enumerate(_named_diff_leaves(diff_params))}
    missing = [name for name in ordered_names if name not in name_to_row_index]
    if missing:
        raise ValueError(f"fitted_params names not found among diff_params' active leaves: {missing}")
    perm = [name_to_row_index[name] for name in ordered_names]
    num_params = len(ordered_names)

    leaves, treedef = jax.tree_util.tree_flatten(diff_params)

    def _physical(flat_leaves):
        unnormed = eqx.combine(static_params, jax.tree_util.tree_unflatten(treedef, flat_leaves)).get_unnormed_params()
        return jax.numpy.stack([jax.numpy.reshape(unnormed[s][k], (-1,)) for s, k in ordered_names])

    jacobian = [np.asarray(j).reshape(num_params, batch_size, -1) for j in jax.jacfwd(_physical)(leaves)]

    def jacobian_at(i):
        return np.array(
            [
                [jacobian[perm[k2]][a, i, i if jacobian[perm[k2]].shape[-1] > 1 else 0] for k2 in range(num_params)]
                for a in range(num_params)
            ]
        )

    return perm, jacobian_at


def sandwich_covariance(residual, jacobian) -> np.ndarray:
    """HC3 sandwich covariance of a least-squares fit, per lineout.

    Args:
        residual: (batch_size, n) residuals at the best fit.
        jacobian: (batch_size, n, p) their Jacobian with respect to the fitted coordinates.

    Returns:
        (batch_size, p, p) covariance of the fitted coordinates.
    """
    residual = np.asarray(residual, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    covariance = np.empty((jacobian.shape[0], jacobian.shape[2], jacobian.shape[2]))
    for i, (r, j) in enumerate(zip(residual, jacobian)):
        bread = np.linalg.inv(j.T @ j)
        leverage = np.einsum("np,pq,nq->n", j, bread, j)
        scores = j * (r / (1.0 - leverage))[:, None]
        covariance[i] = bread @ (scores.T @ scores) @ bread
    return covariance


def get_sandwich_sigmas(residual, jacobian, diff_params, static_params, fitted_params: Dict, batch_size: int):
    """Uncertainty in physical units from the HC3 sandwich covariance of the l2 fit.

    Args:
        residual, jacobian: as returned by LossFunction.l2_residual_jacobian_per_lineout.
        diff_params, static_params, fitted_params, batch_size: as in get_sigmas.

    Returns:
        batch_size x number_of_parameters array, columns ordered to match fitted_params.
    """
    ordered_names = _sigma_columns(fitted_params)
    sigmas = np.full((batch_size, len(ordered_names)), np.nan)
    if not ordered_names:
        return sigmas
    perm, jacobian_at = _leaf_to_physical(diff_params, static_params, ordered_names, batch_size)
    covariance_z = sandwich_covariance(residual, jacobian)
    for i in range(batch_size):
        jac = jacobian_at(i)
        covariance = jac @ covariance_z[i][np.ix_(perm, perm)] @ jac.T
        sigmas[i, :] = np.sqrt(np.diag(covariance))
    return sigmas


SIGMA_METHODS = ("auto", "laplace", "sandwich")


def sigma_method(config: Dict) -> str:
    """The uncertainty method other.sigma_method selects: "auto" is the sandwich for 1D l2 fits, else Laplace."""
    method = config["other"].get("sigma_method", "auto")
    if method not in SIGMA_METHODS:
        raise ValueError(f"other.sigma_method must be one of {SIGMA_METHODS}, got {method!r}")
    least_squares = (
        config["optimizer"]["loss_method"] == "l2"
        and "angular" not in config["other"]["extraoptions"]["spectype"]
    )
    if method == "auto":
        return "sandwich" if least_squares else "laplace"
    if method == "sandwich" and not least_squares:
        raise ValueError("other.sigma_method: sandwich needs optimizer.loss_method: l2 on 1D (non-angular) data")
    return method


def calc_sigmas(config: Dict, loss_fn: LossFunction, ts_params, batch: Dict, fitted_params: Dict = None) -> np.ndarray:
    """Per-lineout uncertainty of the active parameters at ts_params, by the method sigma_method selects.

    Returns:
        batch_size x number_of_parameters array in physical units, columns ordered as get_fitted_params.
    """
    _check_fe_not_fit(config)
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    if fitted_params is None:
        fitted_params, _ = ts_params.get_fitted_params(config["parameters"])
    batch_size = config["optimizer"]["batch_size"]
    if sigma_method(config) == "sandwich":
        residual, jacobian = loss_fn.l2_residual_jacobian_per_lineout(diff_params, static_params, batch)
        return get_sandwich_sigmas(residual, jacobian, diff_params, static_params, fitted_params, batch_size)
    hess = loss_fn.h_loss_wrt_params_per_lineout(diff_params, static_params, batch)
    return get_sigmas(hess, diff_params, static_params, fitted_params, batch_size)


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
            # keeps the batch axis so the result is a valid batch_size=1 ThomsonParams
            prev = (i - 1) % true_batch_size
            if isinstance(x, list) or len(np.shape(x)) > 0:
                return x[prev : prev + 1]
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
        new_weights, _, loss_fn = one_d_loop(temp_cfg, all_data, sa, np.array([i]), 1, previous_weights=[prev_weights])

        inds = np.array([i])
        batch = build_batch(all_data, inds, config["data"]["background"]["bg_subtract"])
        loss, _, _, _, _ = loss_fn.array_loss(new_weights[0], batch)

        if loss < losses_init[i]:
            fitted_weights[i // true_batch_size] = jax.tree.map(
                insert,
                fitted_weights[i // true_batch_size],
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

    final_params = plotters.get_final_params(config, all_params, all_axes, td, sa, fitted_weights)

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

    # Calculate sigmas if needed, from the Hessian with respect to the active fit parameters only
    sigmas = None
    if config["other"]["calc_sigmas"]:
        try:
            sigmas = calc_sigmas(config, loss_fn, fitted_weights, batch, batch_fitted_params)
            print(f"Number of 0s in sigma: {np.count_nonzero(sigmas==0)}")
        except Exception as e:
            print(f"Error calculating uncertainties, none have been calculated: {e}")
            sigmas = None

    # Logging and plotting
    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    final_params = plotters.get_final_params(config, all_params, all_axes, td, sa, fitted_weights)
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
