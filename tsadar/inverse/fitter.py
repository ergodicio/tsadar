from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

import mlflow

from tsadar.inverse.loops import multirun_angular_optax, one_d_loop

from ..utils.process import prepare, postprocess


def _validate_inputs_(config: Dict) -> Dict:
    """
    Validates and augments the configuration dictionary for the fitting process.
    This function checks the boundaries and ordering of lineout and fit ranges, ensures that the electron and ion fitting ranges are contained within the plotting ranges, and generates the list of lineout indices. It also ensures that the number of lineouts is divisible by the batch size, removing excess lineouts if necessary.
        config (Dict): Configuration dictionary containing data, optimizer, and plotting settings.
        Dict: Updated configuration dictionary with validated and derived quantities for lineouts.
        Args:    
        config (Dict): Configuration dictionary containing data and optimizer settings.
    Returns:
        Dict: Updated configuration dictionary with derived quantities for lineouts.
    Raises:
        ValueError: If any of the following conditions are not met:
            - Lineout start is less than lineout end.
            - Lineout end is greater than lineout start plus skip.
            - Blue max is greater than blue min.
            - Red max is greater than red min.
            - IAW fit range is ordered as iaw_min < iaw_cf_min < iaw_cf_max < iaw_max.
            - Electron fitting range is contained within the plotting range.
            - Ion fitting range is contained within the plotting range.

    """

    # check boundries for linouts and fit ranges to ensure they are ordered properly
    if config["data"]["lineouts"]["start"] >= config["data"]["lineouts"]["end"]:
        raise ValueError("Lineout start must be less than lineout end")
    if config["data"]["lineouts"]["end"]- config["data"]["lineouts"]["start"] <= config["data"]["lineouts"]["skip"]:
        raise ValueError("Lineout end must be greater than lineout start + skip")
    
    if config["data"]["fit_rng"]["blue_max"] <= config["data"]["fit_rng"]["blue_min"]:
        raise ValueError("Blue max must be greater than blue min")
    if config["data"]["fit_rng"]["red_max"] <= config["data"]["fit_rng"]["red_min"]:
        raise ValueError("Red max must be greater than red min")
    if not config["data"]["fit_rng"]["iaw_min"]<config["data"]["fit_rng"]["iaw_cf_min"]<config["data"]["fit_rng"]["iaw_cf_max"]<config["data"]["fit_rng"]["iaw_max"]:
        raise ValueError("IAW fit range is not ordered properly, must satisfy: iaw_min < iaw_cf_min < iaw_cf_max < iaw_max")
    
    if config["plotting"]["ele_window_start"] > config["data"]["fit_rng"]["blue_min"] or config["plotting"]["ele_window_end"] < config["data"]["fit_rng"]["red_max"]:
        raise ValueError("Electron fitting range is not contained within the plotting range, please check your inputs")
    if config["plotting"]["ion_window_start"] > config["data"]["fit_rng"]["iaw_min"] or config["plotting"]["ion_window_end"] < config["data"]["fit_rng"]["iaw_max"]:
        raise ValueError("Ion fitting range is not contained within the plotting range, please check your inputs")
    
    # check the distirbution function options are compatible
    if config["parameters"]["electron"]["fe"]["dim"] == 1:
        if config["parameters"]["electron"]["fe"]["type"] not in ["mx", "dlm", "arbitrary"]:
            raise ValueError(f"Electron distribution function type {config['parameters']['electron']['fe']['type']} is not supported for 1D EDFs, please choose one of the allowed types: mx, dlm, arbitrary")
    elif config["parameters"]["electron"]["fe"]["dim"] == 2:
        if config["parameters"]["electron"]["fe"]["type"] not in ["sphericalharmonic", "arbitrary"]:
            raise ValueError(f"Electron distribution function type {config['parameters']['electron']['fe']['type']} is not supported for 2D EDFs, please choose one of the allowed types: sphericalharmonic or arbitrary")
        elif config["parameters"]["electron"]["fe"]["type"] == "sphericalharmonic" and config["parameters"]["electron"]["fe"]["params"]["type"] not in ["Mora-Yahi", "NN", "arbitrary"]:
            raise ValueError(f"Electron distribution function params type {config['parameters']['electron']['fe']['params']['type']} is not supported for spherical harmonic EDFs, please choose one of the allowed flm types: Mora-Yahi, NN, arbitrary")
    if "matte" in config["parameters"]["electron"]["fe"]["params"]["m"] and config["parameters"]["electron"]["fe"]["params"]["m"]["matte"]:
        if config["parameters"]["electron"]["fe"]["type"] != "dlm" or config["parameters"]["electron"]["fe"]["dim"] != 1:
            raise ValueError("Matte based m-values are only supported for 1D DLM electron distribution functions")
        elif config["parameters"]["electron"]["fe"]["active"]:
            raise ValueError("Matte based m-values cannot be fit, please set active to false for fe when using the Matte model")
    

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


def fit(config) -> Tuple[pd.DataFrame, float]:
    """
    Fits the Thomson scattering spectral density function to experimental data or plots specified spectra based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing all necessary parameters for data loading, fitting, and postprocessing.
    Returns:
        Tuple[pd.DataFrame, float]: 
            - A pandas DataFrame containing the final fitted parameters or processed results.
            - A float representing the overall loss value from the fitting procedure.
    Notes:
        - The function logs metrics and status tags to MLflow for experiment tracking.
        - The fitting procedure can handle both angular and 1D spectra, depending on the configuration.
        - Data preparation, fitting, and postprocessing are modularized for clarity and extensibility.
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
        fitted_weights, overall_loss, loss_fn = multirun_angular_optax(config, all_data, sa)
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
