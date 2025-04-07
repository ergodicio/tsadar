from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

import mlflow

from tsadar.inverse.loops import angular_optax, one_d_loop

from ..utils.process import prepare, postprocess


def _validate_inputs_(config: Dict) -> Dict:
    """
    This function adds derived configuration quantities that are necessary for the fitting process

    Args:
        config: Dict

    Returns: Dict

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
