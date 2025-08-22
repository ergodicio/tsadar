from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from optax import tree_utils as otu
import equinox as eqx
import scipy.optimize as spopt
from tsadar.inverse.loss_function import LossFunction


import mlflow
import numpy as np
import time
import pickle
from jax.flatten_util import ravel_pytree
from tqdm import trange
import optax


from typing import Dict, List, Tuple


def _1d_scipy_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict
) -> Tuple[float, Dict]:
    """
    Runs a 1D optimization loop using SciPy's minimize function for inverse Thomson scattering.
    Args:
        config (Dict): Configuration dictionary containing optimizer and parameter settings.
        loss_fn (LossFunction): Loss function object with methods for evaluating the loss and its gradient.
        previous_weights (np.ndarray): Previous weights to initialize the optimizer, or None to use default initialization.
        batch (Dict): Batch of data to be used in the loss function.
    Returns:
        Tuple[float, Dict]: A tuple containing the best loss value and the corresponding optimized weights.
    """
    
    _activate = True
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=_activate)
    else:
        ts_params = previous_weights

    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    init_weights, loss_fn.unravel_weights = ravel_pytree(diff_params)

    res = spopt.minimize(
        loss_fn.vg_loss if config["optimizer"]["grad_method"] == "AD" else loss_fn.loss,
        init_weights,
        args=(static_params, batch),
        method=config["optimizer"]["method"],
        jac=True if config["optimizer"]["grad_method"] == "AD" else False,
        bounds=None if _activate else ((0, 1) for _ in range(len(init_weights))),
        options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
    )

    best_loss = res["fun"]
    best_weights = eqx.combine(loss_fn.unravel_weights(res["x"]), static_params)

    return best_loss, best_weights


def _1d_optax_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights: np.ndarray, batch: Dict, tbatch
) -> Tuple[float, Dict]:
    """
    Runs a 1D optimization loop using the Adam optimizer for a specified number of epochs.
    Args:
        config (Dict): Configuration dictionary containing optimizer and parameter settings.
        loss_fn (LossFunction): Loss function object with a `vg_loss` method for computing loss and gradients.
        previous_weights (np.ndarray): Previous weights to initialize the model parameters. If None, initializes new parameters.
        batch (Dict): Batch of data to be used for optimization.
        tbatch: Progress bar or tracker object for displaying epoch progress.
    Returns:
        Tuple[float, Dict]: A tuple containing the best loss achieved and the corresponding model weights.
    """

    minimizer = getattr(optax, config["optimizer"]["method"])
    # schedule = optax.schedules.cosine_decay_schedule(config["optimizer"]["learning_rate"], 100, alpha = 0.00001)
    # solver = minimizer(schedule)
    opt = minimizer(None if config["optimizer"]["method"]=='lbfgs' else config["optimizer"]["learning_rate"])

    #ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    #diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    #opt_state = solver.init(diff_params)

    
    #opt = optax.adam(config["optimizer"]["learning_rate"])
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=True)
    else:
        ts_params = previous_weights
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    opt_state = opt.init(diff_params)

    best_loss = 1e16
    epoch_loss = 1e19
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")

        (epoch_loss, aux), grad = loss_fn.vg_loss(diff_params, static_params, batch)
        #updates, opt_state = opt.update(grad, opt_state)
        updates, opt_state = opt.update(grad, opt_state, diff_params, value = epoch_loss, grad = grad, value_fn = loss_fn._loss_)
        diff_params = eqx.apply_updates(diff_params, updates)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = eqx.combine(diff_params, static_params)

    return best_loss, best_weights


def one_d_loop(
    config: Dict, all_data: Dict, sa: Tuple, batch_indices: np.ndarray, num_batches: int, previous_weights=None,
) -> Tuple[List, float, LossFunction]:
    """
    Higher level wrapper form minimization of 1D fits, preparing data and dispatching to the appropriate optimizer.
    This function prepares batches of data and fits model parameters using either the ADAM optimizer or a SciPy optimizer,
    depending on the configuration. It supports sequential optimization by passing weights between batches if enabled.
        
    Args:    
        config (Dict): Configuration dictionary containing optimizer settings and batch size.
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
        sa (Tuple): Scattering angles and weights used to calculate k-smea r corrections.
        batch_indices (np.ndarray): Array of indices specifying how to split data into batches.
        num_batches (int): Number of batches to process.
        previous_weights (np.ndarray, optional): Weights to initialize the optimizer. If None, initializes new parameters.
    Returns:
        all_weights (List): List of weights from each batch.
        overall_loss (float): Overall accumulated loss across all batches.
        loss_fn (LossFunction): The final LossFunction instance used for fitting.
    Notes: 
        - The function uses a progress bar to display the fitting progress for each batch.
        - The function logs metrics to MLflow for tracking the fitting process.

    """
    sample = {k: v[: config["optimizer"]["batch_size"]] for k, v in all_data.items()}
    sample = {
        "noise_e": all_data["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": all_data["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    loss_fn = LossFunction(config, sa, sample)

    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    all_weights = []
    overall_loss = 0.0
    previous_batch = None
    with trange(num_batches, unit="batch") as tbatch:
        for i_batch in tbatch:
            previous_batch = previous_weights[i_batch] if previous_weights is not None else previous_batch
            inds = batch_indices[i_batch]
            batch = {
                "e_data": all_data["e_data"][inds],
                "e_amps": all_data["e_amps"][inds],
                "i_data": all_data["i_data"][inds],
                "i_amps": all_data["i_amps"][inds],
                "noise_e": all_data["noiseE"][inds],
                "noise_i": all_data["noiseI"][inds],
            }

            if config["optimizer"]["method"] == "l-bfgs-b":  # Stochastic Gradient Descent
                # not sure why this is needed but something needs to be reset, either the weights or the bounds
                loss_fn = LossFunction(config, sa, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, loss_fn, previous_batch, batch)
            else:
                best_loss, best_weights = _1d_optax_loop_(config, loss_fn, previous_batch, batch, tbatch)
                

            all_weights.append(best_weights)
            mlflow.log_metrics({"batch loss": float(best_loss)}, step=i_batch)
            overall_loss += best_loss

            # ugly
            if "sequential" in config["optimizer"]:
                if config["optimizer"]["sequential"]:
                    previous_batch = best_weights
                    # if config["optimizer"]["method"] == "adam":
                    #     previous_weights = best_weights
                    # else:
                    #     previous_weights, _ = ravel_pytree(best_weights)

    return all_weights, overall_loss, loss_fn


def angular_optax(config, sa, loss_fn, actual_data, previous_weights=None, previous_epoch=None):
    """
    This performs an fitting routines from the optax packages, different minimizers have different requirements for updating steps
    Performs parameter optimization using Optax minimizers for angular Thomson scattering data.
    This function sets up and runs a fitting routine using the Optax optimization library, applying the specified minimizer to fit model parameters to experimental data. It handles data batching, optimizer initialization, training loop with early stopping, and logging of metrics and optimizer state.
        
    Args:    
        config (dict): Configuration dictionary built from the input decks, specifying optimizer, data, and parameter settings.
        all_data (dict): Dictionary containing datasets, amplitudes, and backgrounds as constructed by the prepare.py code.
        sa (dict): Dictionary of the scattering angles and their relative weights.
    Returns:
        best_weights (dict): Best parameter weights as returned by the minimizer.
        best_loss (float): Best value of the fit metric found by the minimizer.
        ts_instance (LossFunction): Instance of the LossFunction object used for minimization.
    Notes:
        - Supports early stopping based on loss improvement or degradation.
        - Logs training metrics and optimizer state using mlflow.
        - Handles both single and multiple shot number data configurations for rotated repeats of data.

    """

    minimizer = getattr(optax, config["optimizer"]["method"])
    schedule = optax.schedules.cosine_decay_schedule(config["optimizer"]["learning_rate_init"], np.round(0.75*config["optimizer"]["num_epochs"]), alpha = config["optimizer"]["learning_rate_final"]/config["optimizer"]["learning_rate_init"])
    solver = minimizer(schedule)
    #solver = minimizer(config["optimizer"]["learning_rate"])

    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)
    else:
        ts_params = previous_weights
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    opt_state = solver.init(diff_params)
    # weights = loss_fn.pytree_weights["active"]
    # opt_state = solver.init(weights)

 # start train loop
    state_weights = {}
    t1 = time.time()
    best_weights = {}
    epoch_loss = 0.0
    best_loss = 100.0
    num_g_wait = 0
    num_b_wait = 0
    for i_epoch in (pbar := trange(config["optimizer"]["num_epochs"])):
        (val, aux), grad = loss_fn.vg_loss(diff_params, static_params, actual_data)
        updates, opt_state = solver.update(grad, opt_state)
        diff_params = eqx.apply_updates(diff_params, updates)
        
        epoch_loss = val
        if epoch_loss < best_loss:
            print(f"delta loss {best_loss - epoch_loss}")
            if best_loss - epoch_loss < 0.00000001:
                num_g_wait += 1
                if num_g_wait > 500:
                    print("Minimizer exited due to change in loss < 1e-8")
                    exit_cond = "Change in loss < 1e-8"
                    break
            else:
                num_b_wait = 0
                num_g_wait = 0
            best_loss = epoch_loss
            best_weights = eqx.combine(diff_params, static_params)
                
        elif epoch_loss > best_loss:
            num_b_wait += 1
            if num_b_wait > 500:
                print("Minimizer exited due to increase in loss")
                exit_cond = "Increase in loss"
                break
        
        pbar.set_description(f"Loss {epoch_loss:.2e}, Learning rate {otu.tree_get(opt_state, 'scale')}")
        
        if config["optimizer"]["save_state"]:
            if (previous_epoch+i_epoch) % config["optimizer"]["save_state_freq"] == 0:
                state_weights[previous_epoch + i_epoch] = best_weights.get_unnormed_params()

        mlflow.log_metrics({"epoch loss": float(epoch_loss)}, previous_epoch + i_epoch)

    if i_epoch == config["optimizer"]["num_epochs"] - 1:
        print("Minimizer exited due to reaching max epochs")
        exit_cond = "Reached epoch limit"
        
    with open("state_weights.txt", "wb") as file:
        file.write(pickle.dumps(state_weights))

    mlflow.log_artifact("state_weights.txt")
    return best_weights, best_loss, previous_epoch + i_epoch, loss_fn, exit_cond

def multirun_angular_optax(
    config: Dict, all_data: Dict, sa: Tuple,
) -> Tuple[List, float, LossFunction]:
    """
    Higher level wrapper for angular Thomson scattering data optimization using Optax.

        
    Args:    
        config (Dict): Configuration dictionary containing optimizer settings and batch size.
        all_data (Dict): Dictionary containing all input data arrays required for fitting.
        sa (Tuple): Scattering angles and weights used to calculate k-smea r corrections.
        batch_indices (np.ndarray): Array of indices specifying how to split data into batches.
        num_batches (int): Number of batches to process.
        previous_weights (np.ndarray, optional): Weights to initialize the optimizer. If None, initializes new parameters.
    Returns:
        all_weights (List): List of weights from each batch.
        overall_loss (float): Overall accumulated loss across all batches.
        loss_fn (LossFunction): The final LossFunction instance used for fitting.
    Notes: 
        - The function uses a progress bar to display the fitting progress for each batch.
        - The function logs metrics to MLflow for tracking the fitting process.

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

    previous_weights = None
    total_epochs = 0
    best_loss = 100

    # Run the angular optimization loop num_mins times
    for i_min in range(config["optimizer"]["num_mins"]):
        loss_fn = LossFunction(config, sa, batch1)
        previous_weights, overall_loss, total_epochs, loss_fn, exit_cond = angular_optax(config, sa, loss_fn, actual_data, previous_weights, total_epochs)
        mlflow.set_tag(f"exit cond {i_min}", exit_cond)
        mlflow.log_metrics({"min loss": float(overall_loss)}, step=i_min)
        best_loss = min(best_loss, overall_loss)
        if i_min < config["optimizer"]["num_mins"]-1:
            config["parameters"]["electron"]["fe"]["nvx"]= config["parameters"]["electron"]["fe"]["nvx"]*config["optimizer"]["refine_factor"]
            config["parameters"]["electron"]["fe"]["params"]["window"]["len"]= config["parameters"]["electron"]["fe"]["params"]["window"]["len"]*config["optimizer"]["refine_factor"]+1
            #currently may only work for 1D arbitrary

            new_vx = np.linspace(
                    previous_weights.electron.distribution_functions.vx[0],
                    previous_weights.electron.distribution_functions.vx[-1],
                    config["parameters"]["electron"]["fe"]["nvx"],
                )
            fenorm = np.sum(previous_weights.electron.distribution_functions.fval) * (previous_weights.electron.distribution_functions.vx[1] - previous_weights.electron.distribution_functions.vx[0])
            refined_fe = np.interp(new_vx,
                previous_weights.electron.distribution_functions.vx,
                previous_weights.electron.distribution_functions.fval,
            )
            refined_fe = fenorm*refined_fe / np.sum(refined_fe) / (new_vx[1] - new_vx[0])

            getleaf = lambda t: t.electron.distribution_functions.fval
            previous_weights = eqx.tree_at(getleaf, previous_weights, refined_fe)
            getleaf = lambda t: t.electron.distribution_functions.vx
            previous_weights = eqx.tree_at(getleaf, previous_weights, new_vx)

    return previous_weights, overall_loss, loss_fn