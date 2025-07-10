from typing import Dict

import numpy as np
import os

from .evaluate_background import get_shot_bg
from ..data_handling.load_ts_data import loadData
from .correct_throughput import correctThroughput
from ..data_handling.calibration import get_calibrations, get_scattering_angles
from .lineouts import get_lineouts
from ..data_handling.data_visualizer import launch_data_visualizer
from tsadar.utils.process.feature_detector import first_guess


def prepare_data(config: Dict, shotNum: int) -> Dict:
    """
    Prepares and processes experimental data for Thomson scattering analysis.
    This function loads electron and ion spectral data, applies calibrations, background corrections,
    and optionally downsamples the data for angular-resolved measurements. It also updates the configuration
    dictionary with relevant calibration and processing parameters.
    Args:
        config (Dict): Configuration dictionary containing data specifications, processing options, and experiment settings.
        shotNum (int): Shot number identifying the experimental dataset to load.
    Returns:
        Tuple[Dict, Dict, Dict]:
            - all_data (Dict): Dictionary containing processed electron and ion data, amplitudes, and noise estimates.
            - sa (Dict): Dictionary of scattering angles and relative weights.
            - all_axes (Dict): Dictionary of calibrated axes for electron and ion spectra.
    """
    
    # load data
    custom_path = None
    if "filenames" in config["data"].keys():
        if config["data"]["filenames"]["epw"] is not None:
            custom_path = os.path.dirname(config["data"]["filenames"]["epw-local"])

        if config["data"]["filenames"]["iaw"] is not None:
            custom_path = os.path.dirname(config["data"]["filenames"]["iaw-local"])

    [elecData, ionData, xlab, t0, config["other"]["extraoptions"]["spectype"]] = loadData(
        config["data"]["shotnum"], config["data"]["shotDay"], config["other"]["extraoptions"], custom_path=custom_path
    )

    # get scattering angles and weights
    sa = get_scattering_angles(config)

    # Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, stddev] = get_calibrations(
        shotNum, config["other"]["extraoptions"]["spectype"], t0, config["other"]["CCDsize"]
    )
    all_axes = {"epw_x": axisxE, "epw_y": axisyE, "iaw_x": axisxI, "iaw_y": axisyI, "x_label": xlab}

    # turn off ion or electron fitting if the corresponding spectrum was not loaded
    if not config["other"]["extraoptions"]["load_ion_spec"]:
        config["other"]["extraoptions"]["fit_IAW"] = 0
        print("IAW data not loaded, omitting IAW fit")
    if not config["other"]["extraoptions"]["load_ele_spec"]:
        config["other"]["extraoptions"]["fit_EPWb"] = 0
        config["other"]["extraoptions"]["fit_EPWr"] = 0
        print("EPW data not loaded, omitting EPW fit")
    # config["data"]["lineouts"]["start"]=start
    # Correct for spectral throughput
    if config["other"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(elecData, config["other"]["extraoptions"]["spectype"], axisyE, shotNum)
        # temp fix for zeros
        elecData = elecData + 0.1
    if config["other"]["extraoptions"]["load_ion_spec"]:
        ionData = ionData + 0.1

    # load and correct background
    [BGele, BGion] = get_shot_bg(config, shotNum, axisyE, elecData)
     #call feature detector, if the boolean for the featiure detector is true , these can be like  if config["other"]["extraoptions"]["load_ele_spec"]: then call the function which returns some of the outputs 
    #assign each returned variable to the corresponmdent one in the decks
    if config["feature_detector"]["estimate_lineouts_iaw"] and not config["feature_detector"]["estimate_lineouts_epw"]:
        [ lineout_end,lineout_start,iaw_cf_min,iaw_cf_max,iaw_max,iaw_min] = first_guess(elecData, ionData,config)
        config["data"]["lineouts"]["start"] = all_axes["iaw_x"][lineout_start]
        config["data"]["lineouts"]["end"] = all_axes["iaw_x"][lineout_end]
        config["data"]["fit_rng"]["iaw_min"] = all_axes["iaw_y"][iaw_min]
        config["data"]["fit_rng"]["iaw_max"] = all_axes["iaw_y"][iaw_max]
        config["data"]["fit_rng"]["iaw_cf_min"] = all_axes["iaw_y"][int(iaw_cf_min)]
        config["data"]["fit_rng"]["iaw_cf_max"] = all_axes["iaw_y"][int(iaw_cf_max)]
        if config["data"]["lineouts"]["type"] == "pixel":

            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(lineout_start), int(lineout_end), config["data"]["lineouts"]["skip"] 

            )
            ]
        else:
            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(config["data"]["lineouts"]["start"]), int(config["data"]["lineouts"]["end"]), int(config["data"]["lineouts"]["skip"])
            )
            ]
 
    if config["feature_detector"]["estimate_lineouts_epw"] and not config["feature_detector"]["estimate_lineouts_iaw"]:
        [ lineout_end,lineout_start, blue_min, blue_max, red_min, red_max] =first_guess(elecData, ionData, config)
        config["data"]["lineouts"]["start"] = all_axes["epw_x"][lineout_start]
        config["data"]["lineouts"]["end"] = all_axes["epw_x"][lineout_end]
        config["data"]["fit_rng"]["blue_min"] = all_axes["epw_y"][blue_min]
        config["data"]["fit_rng"]["blue_max"] = all_axes["epw_y"][blue_max]
        config["data"]["fit_rng"]["red_min"] = all_axes["epw_y"][red_min]
        config["data"]["fit_rng"]["red_max"] = all_axes["epw_y"][red_max]
        if config["data"]["lineouts"]["type"] == "pixel":

            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(lineout_start), int(lineout_end), config["data"]["lineouts"]["skip"] 

            )
            ]
        else:
            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(config["data"]["lineouts"]["start"]), int(config["data"]["lineouts"]["end"]), int(config["data"]["lineouts"]["skip"])
            )
            ]

    if config["feature_detector"]["estimate_lineouts_epw"] and config["feature_detector"]["estimate_lineouts_iaw"]:
        [ lineout_end, lineout_start, iaw_cf_min, iaw_cf_max, iaw_max, iaw_min, ion_t0_shift, blue_min, blue_max, red_min, red_max] = first_guess(elecData, ionData, config)
        config["data"]["lineouts"]["start"] = all_axes["iaw_x"][lineout_start]
        config["data"]["lineouts"]["end"] = all_axes["iaw_x"][lineout_end]
        config["data"]["fit_rng"]["iaw_min"] = all_axes["iaw_y"][iaw_min]
        config["data"]["fit_rng"]["iaw_max"] = all_axes["iaw_y"][iaw_max]
        config["data"]["fit_rng"]["iaw_cf_min"] = all_axes["iaw_y"][int(iaw_cf_min)]
        config["data"]["fit_rng"]["iaw_cf_max"] = all_axes["iaw_y"][int(iaw_cf_max)]
        config["data"]["ion_t0_shift"] = all_axes["iaw_x"][ion_t0_shift]
        config["data"]["fit_rng"]["blue_min"] = all_axes["epw_y"][blue_min]
        config["data"]["fit_rng"]["blue_max"] = all_axes["epw_y"][blue_max]
        config["data"]["fit_rng"]["red_min"] = all_axes["epw_y"][red_min]
        config["data"]["fit_rng"]["red_max"] = all_axes["epw_y"][red_max]
        if config["data"]["lineouts"]["type"] == "pixel":

            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(lineout_start), int(lineout_end), config["data"]["lineouts"]["skip"] 

            )
            ]
        else:
            config["data"]["lineouts"]["val"] = [
            i
            for i in range(
                int(config["data"]["lineouts"]["start"]), int(config["data"]["lineouts"]["end"]), int(config["data"]["lineouts"]["skip"])
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

    # extract ARTS section
    if (config["data"]["lineouts"]["type"] == "range") & (config["other"]["extraoptions"]["spectype"] == "angular"):
        config["other"]["extraoptions"]["spectype"] = "angular_full"
        # config["other"]["PhysParams"]["amps"] = np.array([np.amax(elecData), 1])
        sa["angAxis"] = axisxE

        # down sample image to resolution units by summation
        ang_res_unit = config["other"]["ang_res_unit"]  # in pixels
        lam_res_unit = config["other"]["lam_res_unit"]  # in pixels

        data_res_unit = np.array(
            [np.average(elecData[i : i + lam_res_unit, :], axis=0) for i in range(0, elecData.shape[0], lam_res_unit)]
        )
        bg_res_unit = np.array(
            [np.average(BGele[i : i + lam_res_unit, :], axis=0) for i in range(0, BGele.shape[0], lam_res_unit)]
        )
        data_res_unit = np.array(
            [
                np.average(data_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, data_res_unit.shape[1], ang_res_unit)
            ]
        )
        bg_res_unit = np.array(
            [
                np.average(bg_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, bg_res_unit.shape[1], ang_res_unit)
            ]
        )

        axisyE = np.array(
            [np.average(axisyE[i : i + lam_res_unit], axis=0) for i in range(0, axisyE.shape[0], lam_res_unit)]
        )
        all_axes["epw_y"] = axisyE.reshape((-1, 1))
        axisxE = np.array(
            [np.average(axisxE[i : i + ang_res_unit], axis=0) for i in range(0, axisxE.shape[0], ang_res_unit)]
        )
        all_axes["epw_x"] = axisxE.reshape((-1, 1))
        all_data = {"e_data": data_res_unit, "e_amps": np.amax(data_res_unit, axis=1, keepdims=True)}
        all_data["i_data"] = all_data["i_amps"] = np.zeros(len(data_res_unit))
        # changed this 8-29-23 not sure how it worked with =0?
        all_data["noiseI"] = np.zeros(np.shape(bg_res_unit))
        all_data["noiseE"] = config["data"]["bgscaleE"] * bg_res_unit + 0.1
        config["other"]["CCDsize"] = np.shape(data_res_unit)
        # config["data"]["lineouts"]["start"] = int(config["data"]["lineouts"]["start"] / ang_res_unit)
        # config["data"]["lineouts"]["end"] = int(config["data"]["lineouts"]["end"] / ang_res_unit)

    else:
        all_data = get_lineouts(
            elecData,
            ionData,
            BGele,
            BGion,
            axisxE,
            axisxI,
            axisyE,
            axisyI,
            config["data"]["ele_t0"],
            config["data"]["ion_t0_shift"],
            xlab,
            sa,
            config,
        )

    # Lauch the data visualizer to show linout selection, not currently interactable
    if config["data"]["launch_data_visualizer"]:
        launch_data_visualizer(elecData, ionData, all_axes, config)

    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [axisyE[0], axisyE[-1]]
    config["other"]["lamrangI"] = [axisyI[0], axisyI[-1]]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

    return all_data, sa, all_axes
