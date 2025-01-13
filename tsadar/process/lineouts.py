from typing import Dict

from collections import defaultdict

import numpy as np
from tsadar.process.evaluate_background import get_lineout_bg


def get_lineouts(
    elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, shift_zero, IAWtime, xlab, sa, config
) -> Dict:
    # Convert lineout locations to pixel
    if config["data"]["lineouts"]["type"] == "ps" or config["data"]["lineouts"]["type"] == "um":
        LineoutPixelE = [np.argmin(abs(axisxE - loc - shift_zero)) for loc in config["data"]["lineouts"]["val"]]
        IAWtime = IAWtime / (axisxI[1]-axisxI[0])  # corrects the iontime to be in the same units as the lineout
        LineoutPixelI = [np.argmin(abs(axisxI - loc - shift_zero)) for loc in config["data"]["lineouts"]["val"]]
    elif config["data"]["lineouts"]["type"] == "pixel":
        LineoutPixelE = config["data"]["lineouts"]["val"]
        LineoutPixelI = config["data"]["lineouts"]["val"]
    else:
        raise NotImplementedError
    LineoutPixelI = np.round(np.array(LineoutPixelI) - IAWtime).astype(int)
    config["data"]["lineouts"]["pixelE"] = LineoutPixelE
    config["data"]["lineouts"]["pixelI"] = LineoutPixelI

    if config["data"]["background"]["type"] == "ps":
        BackgroundPixel = np.argmin(abs(axisxE - config["data"]["background"]["slice"]))
    elif config["data"]["background"]["type"] == "pixel":
        BackgroundPixel = config["data"]["background"]["slice"]
    elif config["data"]["background"]["type"] == "auto":
        BackgroundPixel = LineoutPixelE + 100
    else:
        BackgroundPixel = []

    span = 2 * config["data"]["dpixel"] + 1  # (span must be odd)

    # extract lineouts
    if config["other"]["extraoptions"]["load_ele_spec"]:
        LineoutTSE = [
            np.sum(elecData[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
            for a in LineoutPixelE
        ]
        LineoutTSE_smooth = [
            np.convolve(LineoutTSE[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]
        if config["other"]["extraoptions"]["spectype"] == "angular":
            sa["weights"] = np.array(
                [
                    np.mean(sa["weights"][a - config["data"]["dpixel"] : a + config["data"]["dpixel"], :], axis=0)
                    for a in LineoutPixelE
                ]
            )
            sa["weights"] = sa["weights"][:, np.newaxis, :]
        else:
            sa["weights"] = sa["weights"] * np.ones([len(LineoutPixelE), len(sa["sa"])])
    else:
        LineoutTSE_smooth = []

    if config["other"]["extraoptions"]["load_ion_spec"]:
        LineoutTSI = [
            np.sum(ionData[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
            for a in LineoutPixelI
        ]
        LineoutTSI_smooth = [
            np.convolve(LineoutTSI[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]  # was divided by 10 for some reason (removed 8-9-22)

    # Find background signal combining information from a background shot and background lineout
    [noiseE, noiseI] = get_lineout_bg(
        config, elecData, ionData, BGele, BGion, LineoutTSE_smooth, BackgroundPixel, LineoutPixelE, LineoutPixelI
    )

    # Find data amplitudes
    gain = config["other"]["gain"]
    if config["other"]["extraoptions"]["load_ion_spec"]:
        noiseI = noiseI / gain
        LineoutTSI_norm = [LineoutTSI_smooth[i] / gain for i, _ in enumerate(LineoutPixelI)]
        LineoutTSI_norm = np.array(LineoutTSI_norm)
        ampI = np.amax(
            LineoutTSI_norm[
                :,
                ((config["data"]["fit_rng"]["iaw_min"] < axisyI) & (axisyI < config["data"]["fit_rng"]["iaw_cf_min"]))
                | (
                    (config["data"]["fit_rng"]["iaw_cf_max"] < axisyI) & (axisyI < config["data"]["fit_rng"]["iaw_max"])
                ),
            ],
            axis=1,
        )

    if config["other"]["extraoptions"]["load_ele_spec"]:
        noiseE = noiseE / gain
        LineoutTSE_norm = [LineoutTSE_smooth[i] / gain for i, _ in enumerate(LineoutPixelE)]
        LineoutTSE_norm = np.array(LineoutTSE_norm)
        # ampE = np.amax(LineoutTSE_norm[:, 100:-1] - noiseE[:, 100:-1], axis=1)  # attempts to ignore 3w comtamination
        ampE = np.amax(
            LineoutTSE_norm[
                :,
                ((config["data"]["fit_rng"]["blue_min"] < axisyE) & (axisyE < config["data"]["fit_rng"]["blue_max"]))
                | ((config["data"]["fit_rng"]["red_min"] < axisyE) & (axisyE < config["data"]["fit_rng"]["red_max"])),
            ],
            axis=1,
        )

    config["other"]["PhysParams"]["noiseI"] = noiseI
    config["other"]["PhysParams"]["noiseE"] = noiseE

    all_data = defaultdict(list)

    if config["other"]["extraoptions"]["load_ion_spec"]:
        all_data["i_data"] = LineoutTSI_norm
        all_data["i_amps"] = ampI
    else:
        all_data["i_data"] = all_data["i_amps"] = np.zeros(len(config["data"]["lineouts"]["val"]))
    if config["other"]["extraoptions"]["load_ele_spec"]:
        all_data["e_data"] = LineoutTSE_norm
        all_data["e_amps"] = ampE
    else:
        all_data["e_data"] = all_data["e_amps"] = np.zeros(len(config["data"]["lineouts"]["val"]))

    return all_data
