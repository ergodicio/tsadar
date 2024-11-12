from typing import Dict, Tuple, List
import time
import numpy as np
import pandas as pd
import copy
import scipy.optimize as spopt

import jaxopt, mlflow, optax
from tqdm import trange
from jax.flatten_util import ravel_pytree

from tsadar.distribution_functions.gen_num_dist_func import DistFunc
<<<<<<<< HEAD:tsadar/fitter.py
from tsadar.process import prepare, postprocess
from tsadar.model.TSFitter import TSFitter
========
from tsadar.model.TSFitter import TSFitter
from inverse_thomson_scattering.process import prepare, postprocess
>>>>>>>> cd7e229f1402e57e76dfe949fcc1beaa600b4999:inverse_thomson_scattering/fitter.py


def init_param_norm_and_shift(config: Dict) -> Dict:
    """
    Initializes the dictionary that contains the normalization constants for all the parameters

    The parameters are all normalized from 0 to 1 in order to improve gradient flow

    Args:
        config: Dict

    Returns: Dict

    """
    lb = {}
    ub = {}
    parameters = config["parameters"]
    active_params = {}
    for species in parameters.keys():
        active_params[species] = []
        lb[species] = {}
        ub[species] = {}
        for key in parameters[species].keys():
            if parameters[species][key]["active"]:
                active_params[species].append(key)
                if np.size(parameters[species][key]["val"]) > 1:
                    lb[species][key] = parameters[species][key]["lb"] * np.ones(
                        np.size(parameters[species][key]["val"])
                    )
                    ub[species][key] = parameters[species][key]["ub"] * np.ones(
                        np.size(parameters[species][key]["val"])
                    )
                else:
                    lb[species][key] = parameters[species][key]["lb"]
                    ub[species][key] = parameters[species][key]["ub"]

    norms = {}
    shifts = {}
    if config["optimizer"]["parameter_norm"]:
        for species in active_params.keys():
            norms[species] = {}
            shifts[species] = {}
            for k in active_params[species]:
                norms[species][k] = ub[species][k] - lb[species][k]
                shifts[species][k] = lb[species][k]
    else:
        for species in active_params.keys():
            norms[species] = {}
            shifts[species] = {}
            for k in active_params:
                norms[species][k] = 1.0
                shifts[species][k] = 0.0
    return {"norms": norms, "shifts": shifts, "lb": lb, "ub": ub}


def _validate_inputs_(config: Dict) -> Dict:
    """
    This function adds derived configuration quantities that are necessary for the fitting process

    Args:
        config: Dict

    Returns: Dict

    """
    # get derived quantities
    for species in config["parameters"].keys():
        if "electron" in config["parameters"][species]["type"].keys():
            dist_obj = DistFunc(config["parameters"][species])
            config["parameters"][species]["fe"]["velocity"], config["parameters"][species]["fe"]["val"] = dist_obj(
                config["parameters"][species]["m"]["val"]
            )
            config["parameters"][species]["fe"]["val"] = np.log(config["parameters"][species]["fe"]["val"])[None, :]
            # config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
            Warning("fe length is currently overwritten by v_res")
            config["parameters"][species]["fe"]["length"] = len(config["parameters"][species]["fe"]["val"])
            if config["parameters"][species]["fe"]["symmetric"]:
                Warning("Symmetric EDF has been disabled")
                # config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])
            if config["parameters"][species]["fe"]["dim"] == 2 and config["parameters"][species]["fe"]["active"]:
                Warning("2D EDFs can only be fit for angular data")

            config["parameters"][species]["fe"]["lb"] = np.multiply(
                config["parameters"][species]["fe"]["lb"], np.ones(config["parameters"][species]["fe"]["length"])
            )
            config["parameters"][species]["fe"]["ub"] = np.multiply(
                config["parameters"][species]["fe"]["ub"], np.ones(config["parameters"][species]["fe"]["length"])
            )
        if "dist_obj" in locals():
            ValueError("Only 1 electron species is currently supported")

    # get slices
    config["data"]["lineouts"]["val"] = [
        i
        for i in range(
            config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
        )
    ]

    num_slices = len(config["data"]["lineouts"]["val"])
    batch_size = config["optimizer"]["batch_size"]

    if not num_slices % batch_size == 0:
        print(f"total slices: {num_slices}")
        # print(f"{batch_size=}")
        print(f"batch size = {batch_size} is not a round divisor of the number of lineouts")
        config["data"]["lineouts"]["val"] = config["data"]["lineouts"]["val"][: -(num_slices % batch_size)]
        print(f"final {num_slices % batch_size} lineouts have been removed")

    config["units"] = init_param_norm_and_shift(config)

    return config


def scipy_angular_loop(config: Dict, all_data: Dict, sa) -> Tuple[Dict, float, TSFitter]:
    """
    Performs angular thomson scattering i.e. ARTEMIS fitting exercise using the SciPy optimizer routines


    Args:
        config:
        all_data:
        best_weights:
        all_weights:
        sa:

    Returns:

    """
    print("Running Angular, setting batch_size to 1")
    config["optimizer"]["batch_size"] = 1
    batch = {
        "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": config["other"]["PhysParams"]["noiseE"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
        "noise_i": config["other"]["PhysParams"]["noiseI"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
    }

    ts_fitter = TSFitter(config, sa, batch)
    all_weights = {k: [] for k in ts_fitter.pytree_weights["active"].keys()}

    if config["optimizer"]["num_mins"] > 1:
        print(Warning("multiple num mins doesnt work. only running once"))
    for i in range(1):
        ts_fitter = TSFitter(config, sa, batch)
        init_weights = copy.deepcopy(ts_fitter.flattened_weights)

        res = spopt.minimize(
            ts_fitter.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter.loss,
            init_weights,
            args=batch,
            method=config["optimizer"]["method"],
            jac=True if config["optimizer"]["grad_method"] == "AD" else False,
            bounds=ts_fitter.bounds,
            options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
        )
        these_weights = ts_fitter.unravel_pytree(res["x"])
        for k in all_weights.keys():
            all_weights[k].append(these_weights[k])

    overall_loss = res["fun"]

    return all_weights, overall_loss, ts_fitter


def angular_adam(config, all_data, sa, batch_indices, num_batches):
    """
    This performs an SGD-like fitting routine. It is different than the SciPy routines primarily because we have to
    manage the number of iterations manually whereas the SciPy routines have their own convergence/termination process

    Args:
        config:
        all_data:
        sa:
        batch_indices:
        num_batches:

    Returns:

    """

    config["optimizer"]["batch_size"] = 1
    test_batch = {
        "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": config["other"]["PhysParams"]["noiseE"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
        "noise_i": config["other"]["PhysParams"]["noiseI"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
    }

    ts_fitter = TSFitter(config, sa, test_batch)

    jaxopt_kwargs = dict(
        fun=ts_fitter.vg_loss, maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
    )
    opt = optax.adam(config["optimizer"]["learning_rate"])
    solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

    weights = ts_fitter.pytree_weights["active"]
    # if previous_weights is None:
    #     init_weights = ts_fitter.pytree_weights["active"]
    # else:
    #     init_weights = previous_weights

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights
    opt_state = solver.init_state(weights, batch=test_batch)

    # start train loop
    t1 = time.time()
    print("minimizing")
    mlflow.set_tag("status", "minimizing")

    best_loss = 1e16
    for i_epoch in (pbar := trange(config["optimizer"]["num_epochs"])):
        if config["nn"]["use"]:
            np.random.shuffle(batch_indices)
            # tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
        epoch_loss = 0.0
        weights, opt_state = solver.update(params=weights, state=opt_state, batch=test_batch)
        epoch_loss += opt_state.value
        pbar.set_description(f"Loss {epoch_loss:.2e}")

        # epoch_loss /= num_batches
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = weights

        mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)
    return best_weights, best_loss, ts_fitter


def _1d_adam_loop_(
    config: Dict, ts_fitter: TSFitter, previous_weights: np.ndarray, batch: Dict, tbatch
) -> Tuple[float, Dict]:
    jaxopt_kwargs = dict(
        fun=ts_fitter.vg_loss, maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
    )
    opt = optax.adam(config["optimizer"]["learning_rate"])
    solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

    if previous_weights is None:
        init_weights = ts_fitter.pytree_weights["active"]
    else:
        init_weights = previous_weights

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    opt_state = solver.init_state(init_weights, batch=batch)

    best_loss = 1e16
    epoch_loss = 1e19
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
        # if config["nn"]["use"]:
        #     np.random.shuffle(batch_indices)

        init_weights, opt_state = solver.update(params=init_weights, state=opt_state, batch=batch)
        epoch_loss = opt_state.value
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = init_weights

    return best_loss, best_weights


def _1d_scipy_loop_(config: Dict, ts_fitter: TSFitter, previous_weights: np.ndarray, batch: Dict) -> Tuple[float, Dict]:
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        init_weights = np.copy(ts_fitter.flattened_weights)
    else:
        init_weights = np.array(previous_weights)

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    res = spopt.minimize(
        ts_fitter.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter.loss,
        init_weights,
        args=batch,
        method=config["optimizer"]["method"],
        jac=True if config["optimizer"]["grad_method"] == "AD" else False,
        bounds=ts_fitter.bounds,
        options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
    )

    best_loss = res["fun"]
    best_weights = ts_fitter.unravel_pytree(res["x"])

    return best_loss, best_weights


def one_d_loop(
    config: Dict, all_data: Dict, sa: Tuple, batch_indices: np.ndarray, num_batches: int
) -> Tuple[List, float, TSFitter]:
    """
    This is the higher level wrapper that prepares the data and the fitting code for the 1D fits

    This function branches out into the various optimization routines for fitting.

    For now, this is either running the ADAM loop or the SciPy optimizer loop

    Args:
        config:
        all_data:
        sa:
        batch_indices:
        num_batches:

    Returns:

    """
    sample = {k: v[: config["optimizer"]["batch_size"]] for k, v in all_data.items()}
    sample = {
        "noise_e": config["other"]["PhysParams"]["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": config["other"]["PhysParams"]["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    ts_fitter = TSFitter(config, sa, sample)

    print("minimizing")
    mlflow.set_tag("status", "minimizing")
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    all_weights = []
    overall_loss = 0.0
    previous_weights = None
    with trange(num_batches, unit="batch") as tbatch:
        for i_batch in tbatch:
            inds = batch_indices[i_batch]
            batch = {
                "e_data": all_data["e_data"][inds],
                "e_amps": all_data["e_amps"][inds],
                "i_data": all_data["i_data"][inds],
                "i_amps": all_data["i_amps"][inds],
                "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                "noise_i": config["other"]["PhysParams"]["noiseI"][inds],
            }

            if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
                best_loss, best_weights = _1d_adam_loop_(config, ts_fitter, previous_weights, batch, tbatch)
            else:
                # not sure why this is needed but something needs to be reset, either the weights or the bounds
                ts_fitter = TSFitter(config, sa, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, ts_fitter, previous_weights, batch)

            all_weights.append(best_weights)
            mlflow.log_metrics({"batch loss": float(best_loss)}, step=i_batch)
            overall_loss += best_loss

            # ugly
            if "sequential" in config["optimizer"]:
                if config["optimizer"]["sequential"]:
                    if config["optimizer"]["method"] == "adam":
                        previous_weights = best_weights
                    else:
                        previous_weights, _ = ravel_pytree(best_weights)

    return all_weights, overall_loss, ts_fitter


def fit(config) -> Tuple[pd.DataFrame, float]:
    """
    This function fits the Thomson scattering spectral density function to experimental data, or plots specified spectra. All inputs are derived from the input dictionary config.

    Summary of additional needs:
          A wrapper to allow for multiple lineouts or shots to be analyzed and gradients to be handled
          Better way to handle data finding since the location may change with computer or on a shot day
          Better way to handle shots with multiple types of data
          Way to handle calibrations which change from one to shot day to the next and have to be recalculated frequently (adding a new function to attempt this 8/8/22)
          A way to handle the expanded ion calculation when colapsing the spectrum to pixel resolution
          A way to handle different numbers of points

    Depreciated functions that need to be restored:
       Time axis alignment with fiducials
       interactive confirmation of new table creation
       ability to generate different table names without the default values


    Args:
        config:

    Returns:

    """
    t1 = time.time()
    mlflow.set_tag("status", "preprocessing")
    config = _validate_inputs_(config)

    # prepare data
    all_data, sa, all_axes = prepare.prepare_data(config)
    batch_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
    num_batches = len(batch_indices) // config["optimizer"]["batch_size"] or 1
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    # perform fit
    t1 = time.time()
    mlflow.set_tag("status", "minimizing")
    print("minimizing")

    if "angular" in config["other"]["extraoptions"]["spectype"]:
        fitted_weights, overall_loss, ts_fitter = scipy_angular_loop(config, all_data, sa)
    else:
        fitted_weights, overall_loss, ts_fitter = one_d_loop(config, all_data, sa, batch_indices, num_batches)

    mlflow.log_metrics({"overall loss": float(overall_loss)})
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "postprocessing")
    print("postprocessing")

    final_params = postprocess.postprocess(config, batch_indices, all_data, all_axes, ts_fitter, sa, fitted_weights)

    return final_params, float(overall_loss)
