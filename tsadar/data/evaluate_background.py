"""Computes the background used for fitting/subtraction: per-shot background frames (get_shot_bg) or a
per-lineout background estimate (get_lineout_bg), via the shot/pixel/fit algorithms selected in
config["data"]["background"]."""
from typing import Tuple

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from .load_ts_data import loadData
from .correct_throughput import correctThroughput

def get_shot_bg(config, shotNum, axisyE, elecData):
    """
    Computes the background electron and ion spectra for a given shot based on data from another shot.
    For non-angular data, the function loads background data from a specified shot,
    applies throughput corrections and smoothing. For angular data polynomial model is fit to the data to correct the background.
    If background type is not recognized, returns zeros for both backgrounds.
    Args:
        config (dict): Configuration dictionary containing parameters and processing options.
        shotNum (int): Shot number of data to evaluated for background.
        axisyE (np.ndarray): Array representing the wavelength axis for electron data.
        elecData (np.ndarray): Electron data array used for background fitting in certain modes.
    Returns:
        tuple:
            BGele (np.ndarray or int): Background electron spectrum (array or 0 if not loaded).
            BGion (np.ndarray or int): Background ion spectrum (array or 0 if not loaded).
    """

    # If the background type is Shot, load the data from the specified shot and smooth it.
    if config["data"]["background"]["type"] == "Shot":
        [BGele, BGion, _, _, _] = loadData(
            config["data"]["background"]["slice"], config["data"]["shotDay"], config["other"]["extraoptions"]
        )
        if config["data"]["load_ion_spec"]:
            BGion = conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        else:
            BGion = 0
        if config["data"]["load_ele_spec"]:
            BGele = correctThroughput(
                BGele, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"]
            )
            if config["other"]["extraoptions"]["spectype"] == "angular":
                BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")  # 1/27 for H2 and 1/24 for kr
            else:
                BGele = conv2(BGele, np.ones([5, 3]) / 15, mode="same")
        else:
            BGele = 0
    
    # If the background type is Fit, load the data from the specified shot and apply a polynomial model for correction.
    # This is specifically for angular data.
    elif config["other"]["extraoptions"]["spectype"] == "angular" and config["data"]["background"]["type"] == "Fit":
        [BGele, _, _, _, _] = loadData(
            config["data"]["background"]["slice"], config["data"]["shotDay"], config["other"]["extraoptions"]
        )

        BGele = correctThroughput(BGele, config["other"]["extraoptions"]["spectype"], axisyE, shotNum)

        BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")  # 1/27 for H2 and 1/24 for kr
        xx = np.arange(1024)

        def quadbg(x):
            res = np.sum(
                (elecData[1000, :] - ((x[0] * (xx - x[3]) ** 2 + x[1] * (xx - x[3]) + x[2]) * BGele[1000, :])) ** 2
            )
            return res

        corrfactor = spopt.minimize(quadbg, [0.1, 0.1, 1.15, 300])
        newBG = (
            corrfactor.x[0] * (xx - corrfactor.x[3]) ** 2 + corrfactor.x[1] * (xx - corrfactor.x[3]) + corrfactor.x[2]
        ) * BGele
        BGele = newBG

        print("Angular background corrected with polynomial model")
        print(corrfactor.x)
        BGion = 0
    
    # If the background type is not recognized, return zeros for both backgrounds.
    else:
        BGele = 0
        BGion = 0

    return BGele, BGion


def get_lineout_bg(
    config, elecData, ionData, BGele, BGion, LineoutTSE_smooth, BackgroundPixel, LineoutPixelE, LineoutPixelI, axisyE, axisyI
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates noise or background profiles to based off the data or background data.
    Electron spectra have 3 options "Fit", "pixel", and "brem_model". These specify how foreground data is treated.
    Noise is then the sum of foreground and background noise. Ions only have one background option as the background is
    usually very small

    "Fit" : fits a model (config["data"]["background"]["bg_alg"]) against the edges of the lineout to produce
    a background. This option functions differently for angular data and is handled by the function
    get_shot_bg. This option makes no attempt to remove a background shot, using both can result in double
    counting. This option is best for imaging data. One of the available bg_alg choices, "brem", fits a
    bremsstrahlung emission model; Z and ne are fixed at the input deck's initial plasma conditions (they
    are not separable from the fit's scale parameter) and only the overall scale and offset are fit.

    "brem_model" : defers the bremsstrahlung background entirely to the forward model, which adds it to the
    synthetic spectrum using each fit iteration's own Z/Te/ne rather than a separately pre-fit background
    (see ThomsonScatteringDiagnostic). No background is computed here; a zero background is returned.

    "pixel : the other options "ps" and "auto" are aliases for "pixel" where the background pixel is instead identified
    by a time ("ps") or set to 100 pixels past the lineout ("auto"). This method uses another lint of the data that is
     smoothed to act as the background. If included a background shot is removed to prevent double counting. This option
      is best for time resolved data.

    Args:
        config (dict): Configuration dictionary containing parameters and processing options.
        elecData (np.ndarray): Electron data array used for background fitting in certain modes.
        ionData (np.ndarray): Ion data array used for background fitting in certain modes.
        BGele (np.ndarray): Background electron spectrum from background shot (array or 0 if not loaded).
        BGion (np.ndarray): Background ion spectrum from background shot (array or 0 if not loaded).
        LineoutTSE_smooth (np.ndarray): Smoothed lineout data for electron spectra.
        BackgroundPixel (int): Pixel index for the background region.
        LineoutPixelE (list): List of pixel indices for electron lineouts.
        LineoutPixelI (list): List of pixel indices for ion lineouts.
        axisyE (np.ndarray): Spectral axis for electron data, used as the independent variable for the
            "brem" bg_alg.
        axisyI (np.ndarray): Spectral axis for ion data (unused, kept for signature symmetry).
    """
    span = 2 * config["data"]["dpixel"] + 1  # (span must be odd)

    # Check if the background type is valid
    if config["data"]["background"]["type"].casefold() not in ["fit", "shot", "pixel", "brem_model"]:
        raise NotImplementedError("Background type must be: 'Fit', 'Shot', 'Pixel', or 'brem_model'")

    # brem_model defers the background to the forward model -- nothing to compute here. Shaped like
    # elecData/ionData (not just one value per lineout) when that channel is loaded, so it broadcasts
    # correctly whether or not bg_subtract also happens to be enabled (subtracting all-zero is a no-op
    # either way); matches the zeros-per-lineout convention used elsewhere for an unloaded channel.
    if config["data"]["background"]["type"].casefold() == "brem_model":
        n_lineouts = len(config["data"]["lineouts"]["val"])
        noiseE = np.zeros((n_lineouts, elecData.shape[1])) if config["data"]["load_ele_spec"] else np.zeros(n_lineouts)
        noiseI = np.zeros((n_lineouts, ionData.shape[1])) if config["data"]["load_ion_spec"] else np.zeros(n_lineouts)
        return noiseE, noiseI

    # for electrons, if the background type is "fit" and the data type is not "angular"
    if config["data"]["load_ele_spec"]:
        # fit a background model to the edges of the lineout
        bgfitx = np.hstack([
                    np.arange(config["data"]["background"]["bg_alg_domain"][0],
                               config["data"]["background"]["bg_alg_domain"][1]),
                                 np.arange(config["data"]["background"]["bg_alg_domain"][2],
                                           config["data"]["background"]["bg_alg_domain"][3])])

        if config["data"]["background"]["type"].casefold() == "fit":
            if config["other"]["extraoptions"]["spectype"] != "angular":
                # exp2 bg seems to be the best for some imaging data while rat11 is better in other cases but
                # should be checked in more situations

                def exp2(x, a, b, c, d):
                    return a * np.exp(b * x) + c * np.exp(d * x)

                # [expbg, _] = spopt.curve_fit(exp2,bgfitx,LineoutTSE_smooth[bgfitx])

                def power2(x, a, b, c):
                    return a * x**b + c

                # [pwerbg, _] = spopt.curve_fit(power2,bgfitx,LineoutTSE_smooth[bgfitx])

                def rat21(x, a, b, c, d):
                    return (a * x**2 + b * x + c) / (x + d)

                # [ratbg, _] = spopt.curve_fit(rat21,bgfitx,LineoutTSE_smooth[bgfitx])

                def rat11(x, a, b, c):
                    return (a * x + b) / (x + c)

                def brem(x, a, c):
                    # Full bremsstrahlung model, kept for reference (lam in nm, Te in keV, 1.24 = hc in keV*nm):
                    # lambda lam, a, c, Z, Te, ne: 10**8*Z*ne**2/Te**0.5/lam**2*np.exp(-1.24/(lam*Te))*a + c
                    #
                    # Z, Te, and ne only ever appear multiplied together with the scale a, so curve_fit can't
                    # separate them -- fitting all five leaves the Jacobian singular and curve_fit never
                    # converges. They are fixed at the input deck's initial plasma conditions here; only the
                    # overall scale and offset are actually fit. x is the pixel index (matching the other
                    # bg_alg's), converted to wavelength via axisyE.
                    Z = config["parameters"]["ion-1"]["Z"]["val"]
                    Te = config["parameters"]["electron"]["Te"]["val"]
                    ne = config["parameters"]["electron"]["ne"]["val"]
                    lam = axisyE[np.asarray(x).astype(int)]
                    return 10**8 * Z * ne**2 / Te**0.5 / lam**2 * np.exp(-1.24 / (lam * Te)) * a + c

                methods={"exp2": exp2, "power2": power2, "rat21": rat21, "rat11": rat11, "brem": brem}
                LineoutBGE = []
                bgalg  = methods[config["data"]["background"]["bg_alg"]]
                for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    [pvec, _] = spopt.curve_fit(bgalg, bgfitx, LineoutTSE_smooth[i][bgfitx], config["data"]["background"]["bg_alg_params"])

                    LineoutBGE.append(bgalg(np.arange(1024), *pvec))
        # if not fit use a pixel lineout with smoothing
        else:
            # quantify a background lineout
            LineoutBGE = np.mean(
                (elecData - BGele)[
                    :, BackgroundPixel - config["data"]["dpixel"] : BackgroundPixel + config["data"]["dpixel"]
                ],
                1,
            )
            #smooth the lineout to reduce high frequency noise
            LineoutBGE = np.convolve(LineoutBGE, np.ones(config["data"]["background"]["bg_smoothing_window"]) / (config["data"]["background"]["bg_smoothing_window"]), "same")
            
            
            # replace background lineout with double exponential for extra smoothing
            if config["other"]["extraoptions"]["spectype"] != "angular":
                # rescale background exponential using the edge of each data lineout
                LineoutBGE_rescaled = []
                for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    scale = spopt.minimize_scalar(
                        lambda a: np.sum(abs(LineoutTSE_smooth[i][bgfitx] - a * LineoutBGE[bgfitx]))
                    )

                    LineoutBGE_rescaled.append(scale.x * LineoutBGE)

                LineoutBGE = np.array(LineoutBGE_rescaled)

        # add background from background shot if applicable
        if np.shape(BGele) == tuple(config["other"]["CCDsize"]):
            LineoutBGE2 = [
                np.mean(BGele[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
                for a in LineoutPixelE
            ]
            noiseE = LineoutBGE + np.array(LineoutBGE2)
        else:
            noiseE = LineoutBGE * np.ones((len(LineoutPixelE), 1))

        # constant addition to the background
        noiseE += config["other"]["flatbg"]

    else:
        noiseE = np.zeros(len(config["data"]["lineouts"]["val"]))

    if config["data"]["load_ion_spec"]:
        # Due to the low background associated with IAWs the fitted background is only performed for the EPW
        if config["data"]["background"]["type"].casefold() == "fit":
            BackgroundPixel = config["data"]["background"]["slice"]

        # quantify a uniform background
        noiseI = np.sum(
            (ionData - BGion)[
                :, BackgroundPixel - config["data"]["dpixel"] : BackgroundPixel + config["data"]["dpixel"]
            ],
            1,
        )
        noiseI = np.convolve(noiseI, np.ones(span) / span, "same")
        bgfitx = np.hstack([np.arange(200, 400), np.arange(700, 850)])
        noiseI = np.mean(noiseI[bgfitx])
        noiseI = np.ones(1024) * config["data"]["bgscaleI"] * noiseI

        # add the uniform background to the background from the background shot
        if np.shape(BGion) == tuple(config["other"]["CCDsize"]):
            LineoutBGI = [
                np.mean(BGion[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
                for a in LineoutPixelI
            ]
            noiseI = noiseI + LineoutBGI
        else:
            noiseI = noiseI * np.ones((len(LineoutPixelI), 1))
    else:
        noiseI = np.zeros(len(config["data"]["lineouts"]["val"]))

    return noiseE, noiseI
