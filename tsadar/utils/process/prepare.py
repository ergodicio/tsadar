from typing import Dict

import numpy as np
import os

from .evaluate_background import get_shot_bg
from ..data_handling.load_ts_data import loadData
from .correct_throughput import correctThroughput
from ..data_handling.calibrations.calibration import get_calibrations, get_scattering_angles
from .lineouts import get_lineouts
from ..data_handling.data_visualizer import launch_data_visualizer


def prepare_data(config: Dict, shotNum: int) -> Dict:
    """
    Loads and preprocesses the data for fitting

    Args:
        config: Configuration dictionary created from input decks
        shotNum: The OMEGA shot number

    Returns:
        all_data: dictionary containing the throughput and warp corrected data at the appropriate resolution unit or lineouts
        sa: scattering angle dictionary containing the scattering angles and thier relative weights
        all_axes: dictionary containing the calibrated axes for all spectra and axis labels

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
    # if config["other"]["extraoptions"]["first_guess"]:
    # run code
    # outs=first_guess(inputs)
    # config["data"]["lineouts"]["start"]=start
    # Correct for spectral throughput
    if config["other"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(elecData, config["other"]["extraoptions"]["spectype"], axisyE, shotNum)
        # temp fix for zeros
        elecData = elecData + 0.1

    # load and correct background
    [BGele, BGion] = get_shot_bg(config, shotNum, axisyE, elecData)

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
