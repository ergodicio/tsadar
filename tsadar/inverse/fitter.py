from typing import Dict, Tuple, List
import time
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as spopt

import mlflow, optax
import equinox as eqx
from optax import tree_utils as otu
from tqdm import trange
from jax.flatten_util import ravel_pytree

from .loss_function import LossFunction
from ..core.modules import get_filter_spec, ThomsonParams
from ..utils.process import prepare, postprocess


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
    # electron_params = config["parameters"]["electron"]

    # if electron_params["fe"]["type"].casefold() == "arbitrary":
    #     if isinstance(electron_params["fe"]["val"]) in [list, np.array]:
    #         pass
    #     elif isinstance(electron_params["fe"]["val"], str):
    #         if electron_params["fe"]["val"].casefold() == "dlm":
    #             electron_params["fe"]["val"] = DLM1D(electron_params)(electron_params["m"]["val"])
    #         elif "file" in electron_params["fe"]["val"]:  # file-/pscratch/a/.../file.txt
    #             filename = electron_params["fe"]["val"].split("-")[1]
    #         else:
    #             raise NotImplementedError(f"Functional form {electron_params['fe']['val']} not implemented")

    # elif electron_params["fe"]["type"].casefold() == "dlm":
    #     assert electron_params["m"]["val"] >= 2, "DLM requires m >= 2"
    #     assert electron_params["m"]["val"] <= 5, "DLM requires m <= 5"

    # elif electron_params["fe"]["type"].casefold() == "sphericalharmonic":
    #     pass

    # elif electron_params["fe"]["type"].casefold() == "spitzer":
    #     pass  # dont need anything here
    # elif electron_params["fe"]["type"].casefold() == "mydlm":
    #     pass  # don't need anything here
    # else:
    #     raise NotImplementedError(f"Functional form {electron_params['fe']['type']} not implemented")

    # dist_obj = DistFunc(electron_params)
    # electron_params["fe"]["velocity"], electron_params["fe"]["val"] = dist_obj(electron_params["m"]["val"])
    # electron_params["fe"]["val"] = np.log(electron_params["fe"]["val"])[None, :]
    # Warning("fe length is currently overwritten by v_res")
    # electron_params["fe"]["length"] = len(electron_params["fe"]["val"])
    # if electron_params["fe"]["symmetric"]:
    #     Warning("Symmetric EDF has been disabled")
    # if electron_params["fe"]["dim"] == 2 and electron_params["fe"]["active"]:
    #     Warning("2D EDFs can only be fit for angular data")

    # electron_params["fe"]["lb"] = np.multiply(electron_params["fe"]["lb"], np.ones(electron_params["fe"]["length"]))
    # electron_params["fe"]["ub"] = np.multiply(electron_params["fe"]["ub"], np.ones(electron_params["fe"]["length"]))

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

    return config


def angular_optax(config, all_data, sa):
    """
    This performs an fitting routines from the optax packages, different minimizers have different requirements for updating steps

    Args:
        config: Configuration dictionary build from the input decks
        all_data: dictionary of the datasets, amplitudes, and backgrounds as constructed by the prepare.py code
        sa: dictionary of the scattering angles and thier relative weights

    Returns:
        best_weights: best parameter weights as returned by the minimizer
        best_loss: best value of the fit metric found by ther minimizer
        ts_instance: instance of the ThomsonScattering object used for minimization

    """

    config["optimizer"]["batch_size"] = 1
    config["data"]["lineouts"]["start"] = int(config["data"]["lineouts"]["start"] / config["other"]["ang_res_unit"])
    config["data"]["lineouts"]["end"] = int(config["data"]["lineouts"]["end"] / config["other"]["ang_res_unit"])
    batch1 = {
        "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": all_data["noiseE"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "noise_i": all_data["noiseI"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
    }
    if isinstance(config["data"]["shotnum"], list):
        batch2 = {
            "e_data": all_data["e_data_rot"][
                config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
            ],
            "e_amps": all_data["e_amps_rot"][
                config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
            ],
            "noise_e": all_data["noiseE_rot"][
                config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
            ],
            "i_data": all_data["i_data"],
            "i_amps": all_data["i_amps"],
            "noise_i": all_data["noiseI"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        }
        actual_data = {"b1": batch1, "b2": batch2}
    else:
        actual_data = batch1

    loss_fn = LossFunction(config, batch1)
    minimizer = getattr(optax, config["optimizer"]["method"])
    # schedule = optax.schedules.cosine_decay_schedule(config["optimizer"]["learning_rate"], 100, alpha = 0.00001)
    # solver = minimizer(schedule)
    solver = minimizer(config["optimizer"]["learning_rate"])

    weights = loss_fn.pytree_weights["active"]
    opt_state = solver.init(weights)

    # start train loop
    state_weights = {}
    t1 = time.time()
    epoch_loss = 0.0
    best_loss = 100.0
    num_g_wait = 0
    num_b_wait = 0
    for i_epoch in (pbar := trange(config["optimizer"]["num_epochs"])):
        (val, aux), grad = loss_fn.vg_loss(weights, actual_data)
        updates, opt_state = solver.update(grad, opt_state, weights)

        epoch_loss = val
        if epoch_loss < best_loss:
            print(f"delta loss {best_loss - epoch_loss}")
            if best_loss - epoch_loss < 0.000001:
                best_loss = epoch_loss
                num_g_wait += 1
                if num_g_wait > 5:
                    print("Minimizer exited due to change in loss < 1e-6")
                    break
            elif epoch_loss > best_loss:
                num_b_wait += 1
                if num_b_wait > 5:
                    print("Minimizer exited due to increase in loss")
                    break
            else:
                best_loss = epoch_loss
                num_b_wait = 0
                num_g_wait = 0
        pbar.set_description(f"Loss {epoch_loss:.2e}, Learning rate {otu.tree_get(opt_state, 'scale')}")

        weights = optax.apply_updates(weights, updates)

        if config["optimizer"]["save_state"]:
            if i_epoch % config["optimizer"]["save_state_freq"] == 0:
                state_weights[i_epoch] = weights

        mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)

    with open("state_weights.txt", "wb") as file:
        file.write(pickle.dumps(state_weights))

    mlflow.log_artifact("state_weights.txt")
    return weights, epoch_loss, loss_fn


def _1d_adam_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict, tbatch
) -> Tuple[float, Dict]:
    # jaxopt_kwargs = dict(
    #     fun=loss_fn.vg_loss, maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
    # )
    opt = optax.adam(config["optimizer"]["learning_rate"])
    ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"])
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    opt_state = opt.init(diff_params)

    # if previous_weights is None:
    #     init_weights = loss_fn.pytree_weights["active"]
    # else:
    #     init_weights = previous_weights

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    # opt_state = solver.init_state(init_weights, batch=batch)

    best_loss = 1e16
    epoch_loss = 1e19
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
        # if config["nn"]["use"]:
        #     np.random.shuffle(batch_indices)
        (epoch_loss, aux), grad = loss_fn.vg_loss(diff_params, static_params, batch)
        updates, opt_state = opt.update(grad, opt_state)
        diff_params = eqx.apply_updates(diff_params, updates)

        # init_weights, opt_state = solver.update(params=init_weights, state=opt_state, batch=batch)
        # epoch_loss = opt_state.value
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = eqx.combine(diff_params, static_params)

    return best_loss, best_weights


def _1d_scipy_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict
) -> Tuple[float, Dict]:
    # if previous_weights is None:  # if prev, then use that, if not then use flattened weights
    #     init_weights = np.copy(loss_fn.ts_diag.flattened_weights)
    # else:
    #     init_weights = np.array(previous_weights)

    ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"])
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    init_weights, loss_fn.unravel_weights = ravel_pytree(diff_params)

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    res = spopt.minimize(
        loss_fn.vg_loss if config["optimizer"]["grad_method"] == "AD" else loss_fn.loss,
        init_weights,
        args=(static_params, batch),
        method=config["optimizer"]["method"],
        jac=True if config["optimizer"]["grad_method"] == "AD" else False,
        bounds=((0, 1) for _ in range(len(init_weights))),
        options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
    )

    best_loss = res["fun"]
    best_weights = eqx.combine(loss_fn.unravel_weights(res["x"]), static_params)

    return best_loss, best_weights


def one_d_loop(
    config: Dict, all_data: Dict, sa: Tuple, batch_indices: np.ndarray, num_batches: int
) -> Tuple[List, float, LossFunction]:
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
        "noise_e": all_data["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": all_data["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    loss_fn = LossFunction(config, sample)

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
                "noise_e": all_data["noiseE"][inds],
                "noise_i": all_data["noiseI"][inds],
            }

            if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
                best_loss, best_weights = _1d_adam_loop_(config, loss_fn, previous_weights, batch, tbatch)
            else:
                # not sure why this is needed but something needs to be reset, either the weights or the bounds
                loss_fn = LossFunction(config, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, loss_fn, previous_weights, batch)

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

    return all_weights, overall_loss, loss_fn


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
    all_data, sa, all_axes = load_data_for_fitting(config)
    sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
    num_batches = len(sample_indices) // config["optimizer"]["batch_size"] or 1
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    # perform fit
    t1 = time.time()
    mlflow.set_tag("status", "minimizing")
    print("minimizing")

    if "angular" in config["other"]["extraoptions"]["spectype"]:
        fitted_weights, overall_loss, loss_fn = angular_optax(config, all_data, sa)
    else:
        fitted_weights, overall_loss, loss_fn = one_d_loop(config, all_data, sa, sample_indices, num_batches)

    mlflow.log_metrics({"overall loss": float(overall_loss)})
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "postprocessing")
    print("postprocessing")

    final_params = postprocess.postprocess(config, sample_indices, all_data, all_axes, loss_fn, sa, fitted_weights)

    return final_params, float(overall_loss)


def load_data_for_fitting(config):
    if isinstance(config["data"]["shotnum"], list):
        startCCDsize = config["other"]["CCDsize"]
        all_data, sa, all_axes = prepare.prepare_data(config, config["data"]["shotnum"][0])
        config["other"]["CCDsize"] = startCCDsize
        all_data2, _, _ = prepare.prepare_data(config, config["data"]["shotnum"][1])
        all_data.update(
            {
                "e_data_rot": all_data2["e_data"],
                "e_amps_rot": all_data2["e_amps"],
                "rot_angle": config["data"]["shot_rot"],
                "noiseE_rot": all_data2["noiseE"],
            }
        )

        if config["other"]["extraoptions"]["spectype"] != "angular_full":
            raise NotImplementedError("Muliplexed data fitting is only availible for angular data")
    else:
        all_data, sa, all_axes = prepare.prepare_data(config, config["data"]["shotnum"])
    return all_data, sa, all_axes
