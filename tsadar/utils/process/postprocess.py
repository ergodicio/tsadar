from typing import Dict
from collections import defaultdict
from flatten_dict import flatten, unflatten

import time, tempfile, mlflow, os, copy

import numpy as np
import jax
from equinox import filter_jit

from tsadar.utils.plotting import plotters
from tsadar.inverse.loss_function import LossFunction
from tsadar.core.modules.ts_params import IonParams
from tsadar.inverse.loops import one_d_loop
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic


def recalculate_with_chosen_weights(
    config: Dict, sa, sample_indices, all_data: Dict, loss_fn: LossFunction, calc_sigma: bool, fitted_weights: Dict
):
    """
    Gets parameters and the result of the full forward pass i.e. fits


    Args:
        config: Dict- configuration dictionary built from input deck
        sample_indices:
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        loss_fn: Instance of the LossFunction class
        fitted_weights: Dict- best values of the parameters returned by the minimizer

    Returns:

    """

    losses = np.zeros_like(sample_indices, dtype=np.float64)
    sample_indices.sort()
    batch_indices = np.reshape(sample_indices, (-1, config["optimizer"]["batch_size"]))

    # turn list of dictionaries into dictionary of lists
    all_params = {k: defaultdict(list) for k in config["parameters"].keys()}

    for _fw in fitted_weights:
        batch_fitted_params, num_params = _fw.get_fitted_params(config["parameters"])
        for k in batch_fitted_params.keys():
            for k2 in batch_fitted_params[k].keys():
                all_params[k][k2].append(batch_fitted_params[k][k2])

    # concatenate all the lists in the dictionary
    for k in all_params.keys():
        for k2 in all_params[k].keys():
            all_params[k][k2] = np.concatenate(all_params[k][k2])

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
    if config["other"]["extraoptions"]["load_ele_spec"]:
        fits["ele"]["spec_comps"] = np.ones(
            [
                all_data["e_data"].shape[0],
                max(
                    config["parameters"]["general"]["Te_gradient"]["num_grad_points"],
                    config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
                ),
                all_data["e_data"].shape[1] * config["other"]["points_per_pixel"],
                len(sa["sa"]),
            ]
        )
    else:
        fits["ele"]["spec_comps"] = np.zeros(all_data["e_data"].shape)
    if config["other"]["extraoptions"]["load_ion_spec"]:
        fits["ion"]["spec_comps"] = np.ones(
            [
                all_data["i_data"].shape[0],
                max(
                    config["parameters"]["general"]["Te_gradient"]["num_grad_points"],
                    config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
                ),
                all_data["i_data"].shape[1] * config["other"]["points_per_pixel"],
                len(sa["sa"]),
            ]
        )
    else:
        fits["ion"]["spec_comps"] = np.zeros(all_data["i_data"].shape)

    sqdevs = {"ion": np.zeros(all_data["i_data"].shape), "ele": np.zeros(all_data["e_data"].shape)}
    # fits["ion"] = np.zeros(all_data["i_data"].shape)
    # fits["ele"] = np.zeros(all_data["e_data"].shape)

    if config["other"]["extraoptions"]["load_ion_spec"]:
        sigmas = np.zeros((all_data["i_data"].shape[0], num_params))

    if config["other"]["extraoptions"]["load_ele_spec"]:
        sigmas = np.zeros((all_data["e_data"].shape[0], num_params))

    if config["other"]["extraoptions"]["spectype"] == "angular_full":
        batch = {
            "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
            "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
            "i_data": all_data["i_data"],
            "i_amps": all_data["i_amps"],
            "noise_e": all_data["noiseE"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
            "noise_i": all_data["noiseI"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        }
        losses, sqds, used_points, [ThryE, _, params] = loss_fn.array_loss(fitted_weights, batch)
        fits["ele"] = ThryE
        sqdevs["ele"] = sqds["ele"]

        for species in all_params.keys():
            for k in all_params[species].keys():
                if k != "fe":
                    # all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])
                    all_params[species][k] = params[species][k].reshape(-1)
                else:
                    all_params[species][k] = params[species][k]

        if calc_sigma:
            # this line may need to be omited since the weights may be transformed by line 77
            active_params = loss_fn.spec_calc.get_plasma_parameters(fitted_weights, return_static_params=False)
            hess = loss_fn.h_loss_wrt_params(active_params, batch)
            sigmas = get_sigmas(hess, config["optimizer"]["batch_size"])
            print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}")

    else:
        for i_batch, inds in enumerate(batch_indices):
            batch = {
                "e_data": all_data["e_data"][inds],
                "e_amps": all_data["e_amps"][inds],
                "i_data": all_data["i_data"][inds],
                "i_amps": all_data["i_amps"][inds],
                "noise_e": all_data["noiseE"][inds],
                "noise_i": all_data["noiseI"][inds],
            }

            loss, sqds, ThryE, ThryI, params = loss_fn.array_loss(fitted_weights[i_batch], batch)

            if config["plotting"]["detailed_breakdown"]:
                ts_diag = ThomsonScatteringDiagnostic(config, sa)
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
                hess = loss_fn.h_loss_wrt_params(fitted_weights[i_batch], batch)
                try:
                    hess = loss_fn.h_loss_wrt_params(fitted_weights[i_batch], batch)
                except:
                    print("Error calculating Hessian, no hessian based uncertainties have been calculated")
                    calc_sigma = False

            losses[inds] = loss
            sqdevs["ele"][inds] = sqds["ele"]
            sqdevs["ion"][inds] = sqds["ion"]
            if calc_sigma:
                sigmas[inds] = get_sigmas(hess, config["optimizer"]["batch_size"])
                # print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}") number of negatives?

            fits["ele"]["total_spec"][inds] = ThryE
            fits["ion"]["total_spec"][inds] = ThryI

    return losses, sqdevs, num_params, fits, sigmas, all_params


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
    # sizes = {key: hess[key][key].shape[1] for key in keys}
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

        # xc = 0
        # for k1, param in enumerate(keys):
        #     yc = 0
        #     for k2, param2 in enumerate(keys):
        #         if i > 0:
        #             temp[k1, k2] = np.squeeze(hess[param][param2])[i, i]
        #         else:
        #             temp[xc : xc + sizes[param], yc : yc + sizes[param2]] = hess[param][param2][0, :, 0, :]
        #
        #         yc += sizes[param2]
        #     xc += sizes[param]

        # print(temp)
        inv = np.linalg.inv(temp)
        # print(inv)

        sigmas[i, :] = np.sign(np.diag(inv)) * np.sqrt(np.abs(np.diag(inv)))
        # for k1, param in enumerate(keys):
        #     sigmas[i, xc : xc + sizes[param]] = np.diag(
        #         np.sign(inv[xc : xc + sizes[param], xc : xc + sizes[param]])
        #         * np.sqrt(np.abs(inv[xc : xc + sizes[param], xc : xc + sizes[param]]))
        #     )
        # print(sigmas[i, k1])
        # change sigmas into a dictionary?

    return sigmas


def postprocess(config, sample_indices, all_data: Dict, all_axes: Dict, loss_fn, sa, fitted_weights):
    t1 = time.time()

    # calculate used poinsts once right before its used

    for species in config["parameters"].keys():
        if "electron" == species:
            elec_species = species

    if config["other"]["extraoptions"]["spectype"] != "angular_full" and config["other"]["refit"]:
        init_losses = refit_bad_fits(config, sa, sample_indices, all_data, loss_fn, fitted_weights)
    else:
        init_losses = []

    mlflow.log_metrics({"refitting time": round(time.time() - t1, 2)})

    with tempfile.TemporaryDirectory() as td:
        _ = [os.makedirs(os.path.join(td, dirname), exist_ok=True) for dirname in ["plots", "binary", "csv"]]
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            t1 = process_angular_data(
                config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, t1, elec_species, td
            )

        else:
            t1, final_params = process_data(
                config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, init_losses, t1, td
            )

        mlflow.log_artifacts(td)
    mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})

    mlflow.set_tag("status", "done plotting")

    return final_params


def refit_bad_fits(config, sa, batch_indices, all_data, loss_fn, fitted_weights):
    losses_init, sqdevs, num_params, fits, sigmas, all_params = recalculate_with_chosen_weights(
        config, sa, batch_indices, all_data, loss_fn, False, fitted_weights
    )

    # refit bad fits
    # reduced_points = (used_points - num_params)*config["optimizer"]["batch_size"]

    red_losses_init = losses_init  # / (1.1 * reduced_points) by changing losses to mean this is loss per point
    true_batch_size = config["optimizer"]["batch_size"]

    mlflow.log_metrics({"number of fits": len(batch_indices.flatten())})
    mlflow.log_metrics({"number of refits": int(np.sum(red_losses_init > config["other"]["refit_thresh"]))})

    sample_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))

    for i in batch_indices.flatten()[red_losses_init > config["other"]["refit_thresh"]]:
        if i == 0:
            continue

        temp_cfg = copy.deepcopy(config)
        temp_cfg["optimizer"]["batch_size"] = 1

        def func(x):
            # i, true_batch_size
            if hasattr(x, "__len__"):
                return {"val": x[(i - 1) % true_batch_size]}
            else:
                return {"val": x}

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
        prev_weights["electron"]["fe"] = {"m": prev_weights["electron"]["m"]}
        del prev_weights["electron"]["m"]

        temp_params = flatten(temp_cfg["parameters"])
        temp_params.update(flatten(prev_weights))
        temp_cfg["parameters"] = unflatten(temp_params)
        # temp_cfg["parameters"] = temp_cfg["parameters"] | prev_weights
        new_weights, _, loss_fn = one_d_loop(temp_cfg, all_data, sa, sample_indices, 1)

        inds = np.array([i])
        batch = {
            "e_data": all_data["e_data"][inds],
            "e_amps": all_data["e_amps"][inds],
            "i_data": all_data["i_data"][inds],
            "i_amps": all_data["i_amps"][inds],
            "noise_e": all_data["noiseE"][inds],
            "noise_i": all_data["noiseI"][inds],
        }
        loss, _, _, _, _ = loss_fn.array_loss(new_weights[0], batch)

        if loss < losses_init[i]:
            fitted_weights[(i - 1) // true_batch_size] = jax.tree.map(
                insert,
                fitted_weights[(i - 1) // true_batch_size],
                new_weights[0],
                is_leaf=lambda x: isinstance(x, list) and not isinstance(x[0], IonParams),
            )
    return losses_init


def process_data(config, sample_indices, all_data, all_axes, loss_fn, fitted_weights, sa, losses_init, t1, td):
    losses, sqdevs, num_params, fits, sigmas, all_params = recalculate_with_chosen_weights(
        config, sa, sample_indices, all_data, loss_fn, config["other"]["calc_sigmas"], fitted_weights
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
    else:
        plotters.model_v_actual(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td)
    sigma_ds = plotters.save_sigmas_params(config, all_params, sigmas, all_axes, td)
    plotters.plot_final_params(config, all_params, sigma_ds, td)
    return t1, final_params


def process_angular_data(config, batch_indices, all_data, all_axes, loss_fn, fitted_weights, t1, elec_species, td):
    best_weights_val = {}
    best_weights_std = {}
    if config["optimizer"]["num_mins"] > 1:
        for k, v in fitted_weights.items():
            best_weights_val[k] = np.average(v, axis=0)  # [0, :]
            best_weights_std[k] = np.std(v, axis=0)  # [0, :]
    else:
        best_weights_val = fitted_weights

    losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
        config, batch_indices, all_data, loss_fn, config["other"]["calc_sigmas"], best_weights_val
    )

    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    final_params = plotters.get_final_params(config, all_params, all_axes, td)
    if config["other"]["calc_sigmas"]:
        sigma_fe = plotters.save_sigmas_fe(final_params, best_weights_std, sigmas, td)
    else:
        sigma_fe = np.zeros_like(final_params["fe"])
    savedata = plotters.plot_data_angular(config, fits, all_data, all_axes, td)
    plotters.plot_ang_lineouts(used_points, sqdevs, losses, all_params, all_axes, savedata, td)
    # plotters.plot_dist(config, elec_species, final_params, sigma_fe, td)
    return t1
