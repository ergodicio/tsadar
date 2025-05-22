from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

import mlflow

from tsadar.inverse.loops import angular_optax, one_d_loop

from ..utils.process import prepare, postprocess


def _validate_inputs_(config: Dict) -> Dict:
    """
    Validates and augments the configuration dictionary for the fitting process by generating the list of lineout indices and ensuring the number of slices is divisible by the batch size.
    Args:    
        config (Dict): Configuration dictionary containing data and optimizer settings.
    Returns:
        Dict: Updated configuration dictionary with derived quantities for lineouts.
    Side Effects:
        - Modifies the 'config' dictionary in place.
        - Prints warnings if the number of lineouts is not divisible by the batch size and removes excess lineouts to ensure divisibility.
    """
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
