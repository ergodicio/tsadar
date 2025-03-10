from tsadar.core.modules import ThomsonParams, get_filter_spec
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
    config: Dict, loss_fn: LossFunction, previous_weights, batch: Dict
) -> Tuple[float, Dict]:
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"], activate=False)
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
        bounds=((0, 1) for _ in range(len(init_weights))),
        options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
    )

    best_loss = res["fun"]
    best_weights = eqx.combine(loss_fn.unravel_weights(res["x"]), static_params)

    return best_loss, best_weights


def _1d_adam_loop_(
    config: Dict, loss_fn: LossFunction, previous_weights, batch: Dict, tbatch
) -> Tuple[float, Dict]:

    opt = optax.adam(config["optimizer"]["learning_rate"])
    #ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"])
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        ts_params = ThomsonParams(config["parameters"], config["optimizer"]["batch_size"])
    else:
        ts_params = previous_weights
    diff_params, static_params = eqx.partition(ts_params, get_filter_spec(config["parameters"], ts_params))
    opt_state = opt.init(diff_params)

    best_loss = 1e16
    epoch_loss = 1e19
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")

        (epoch_loss, aux), grad = loss_fn.vg_loss(diff_params, static_params, batch)
        updates, opt_state = opt.update(grad, opt_state)
        diff_params = eqx.apply_updates(diff_params, updates)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = eqx.combine(diff_params, static_params)

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
    loss_fn = LossFunction(config, sa, sample)

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
                loss_fn = LossFunction(config, sa, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, loss_fn, previous_weights, batch)

            all_weights.append(best_weights)
            mlflow.log_metrics({"batch loss": float(best_loss)}, step=i_batch)
            overall_loss += best_loss

            # ugly
            if "sequential" in config["optimizer"]:
                if config["optimizer"]["sequential"]:
                    previous_weights = best_weights
                    # if config["optimizer"]["method"] == "adam":
                    #     previous_weights = best_weights
                    # else:
                    #     previous_weights, _ = ravel_pytree(best_weights)

    return all_weights, overall_loss, loss_fn


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

    loss_fn = LossFunction(config, sa, batch1)
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
