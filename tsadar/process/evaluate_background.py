from typing import Tuple

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from tsadar.process.correct_throughput import correctThroughput

from tsadar.utils.data_handling.load_ts_data import loadData
from .correct_throughput import correctThroughput



def get_shot_bg(config, shotNum, axisyE, elecData):
    """
    Quantify the background for full images

    Args:
        config:
        axisyE:
        elecData:

    Returns:

    """
    if config["data"]["background"]["type"] == "Shot":
        [BGele, BGion, _, _, _] = loadData(
            config["data"]["background"]["slice"], config["data"]["shotDay"], config["other"]["extraoptions"]
        )
        if config["other"]["extraoptions"]["load_ion_spec"]:
            BGion = conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        else:
            BGion = 0
        if config["other"]["extraoptions"]["load_ele_spec"]:
            BGele = correctThroughput(
                BGele, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"]
            )
            if config["other"]["extraoptions"]["spectype"] == "angular":
                BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")  # 1/27 for H2 and 1/24 for kr
            else:
                BGele = conv2(BGele, np.ones([5, 3]) / 15, mode="same")
        else:
            BGele = 0
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
    else:
        BGele = 0
        BGion = 0

    return BGele, BGion


def get_lineout_bg(
    config, elecData, ionData, BGele, BGion, LineoutTSE_smooth, BackgroundPixel, LineoutPixelE, LineoutPixelI
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates noise or background profiles to based off the data or background data.
    Electron spectra have 2 options "Fit" and "pixel". These specify how foreground data is treated.
    Noise is then the sum of foreground and background noise. Ions only have one background option as the background is
    usually very small

    "Fit" : fits a rational model against the edges of the lineout to produce a background, the model can be changed.
    This option functions differently for angular data and is handled by the function get_shot_bg. This option makes no
    attempt to remove a background shot, using both can result in double counting. This option is best for imaging data.

    "pixel : the other options "ps" and "auto" are aliases for "pixel" where the background pixel is instead identified
    by a time ("ps") or set to 100 pixels past the lineout ("auto"). This method uses another lint of the data that is
     smoothed to act as the background. If included a background shot is removed to prevent double counting. This option
      is best for time resolved data.
    """
    span = 2 * config["data"]["dpixel"] + 1  # (span must be odd)


    if config["data"]["background"]["type"].casefold() not in ["fit", "shot", "pixel"]:
        raise NotImplementedError("Background type must be: 'Fit', 'Shot', or 'Pixel'")

    if config["other"]["extraoptions"]["load_ele_spec"]:
        if config["data"]["background"]["type"].casefold() == "fit":
            if config["other"]["extraoptions"]["spectype"] != "angular":
                # exp2 bg seems to be the best for some imaging data while rat11 is better in other cases but
                # should be checked in more situations

                bgfitx = np.hstack([
                    np.arange(config["data"]["background"]["bg_alg_domain"][0],
                               config["data"]["background"]["bg_alg_domain"][1]),
                                 np.arange(config["data"]["background"]["bg_alg_domain"][2],
                                           config["data"]["background"]["bg_alg_domain"][3])])

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
                
                def expdecay(t,a,k,b):
                    return a*np.exp(-k*t)+b

                methods={"exp2": exp2, "power2": power2, "rat21": rat21, "rat11": rat11}
                LineoutBGE = []

                for  i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    [ratbg,_]=spopt.curve_fit(rat11,bgfitx,LineoutTSE_smooth[i][bgfitx], [-16, 200000, 170])

                    LineoutBGE.append(rat11(np.arange(1024), *ratbg))
                """for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    [rat1bg, _] = spopt.curve_fit(rat11, bgfitx, LineoutTSE_smooth[i][bgfitx], [-16, 200000, 170], maxfev=2000)
                    # if config["data"]["background"]["show"]:
                    #if i %1 == 0:
            
                    plt.plot(rat11(np.arange(1,1024), *rat1bg))
                    plt.plot(LineoutTSE_smooth[i])
                    plt.ylim([0,5000])
                    plt.show()
                    print(i)

                    LineoutBGE.append(rat11(np.arange(1024), *rat1bg))"""

                bgalg  = methods[config["data"]["background"]["bg_alg"]]
                for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    [pvec, _] = spopt.curve_fit(bgalg, bgfitx, LineoutTSE_smooth[i][bgfitx], [-16, 200000, 170])
                    # if config["data"]["background"]["show"]:
                    #     plt.plot(rat11(np.arange(1024), *rat1bg))
                    #     plt.plot(LineoutTSE_smooth[i])
                    #     plt.show()

                    LineoutBGE.append(bgalg(np.arange(1024), *pvec))

        # if not fit
        else:
            # quantify a background lineout
            LineoutBGE = np.mean(
                (elecData - BGele)[
                    :, BackgroundPixel - config["data"]["dpixel"] : BackgroundPixel + config["data"]["dpixel"]
                ],
                1,
            )
            LineoutBGE = np.convolve(LineoutBGE, np.ones(span) / span, "same")

            # replace background lineout with double exponential for extra smoothing
            if config["other"]["extraoptions"]["spectype"] != "angular":

                def exp2(x, a, b, c, d):
                    return a * np.exp(-b * x) + c * np.exp(-d * x)

                bgfitx = np.hstack(
                    [np.arange(250, 480), np.arange(540, 900)]
                )  # this is specificaly targeted at streaked data, removes the fiducials at top and bottom and notch filter
                bgfitx2 = np.hstack([np.arange(250, 300), np.arange(700, 900)])
                plt.plot(bgfitx, LineoutBGE[bgfitx])

                [expbg, _] = spopt.curve_fit(exp2, bgfitx, LineoutBGE[bgfitx], p0=[200, 0.001, 200, 0.001],maxfev=20000)
                LineoutBGE = config["data"]["bgscaleE"] * exp2(np.arange(1024), *expbg)

                # rescale background exponential using the edge of each data lineout
                LineoutBGE_rescaled = []
                for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                    scale = spopt.minimize_scalar(
                        lambda a: np.sum(abs(LineoutTSE_smooth[i][bgfitx2] - a * LineoutBGE[bgfitx2]))
                    )

                    LineoutBGE_rescaled.append(scale.x * LineoutBGE)

                LineoutBGE = np.array(LineoutBGE_rescaled)
                # if config["data"]["background"]["show"]:
                #     plt.plot(bgfitx, LineoutBGE[bgfitx])
                #     plt.plot(bgfitx, scale.x * LineoutBGE[bgfitx])
                #     plt.plot(bgfitx, exp2(bgfitx, 200, 0.001, 200, 0.001))
                #     lin = np.mean((elecData - BGele)[:, 480 - config["data"]["dpixel"] : 480 + config["data"]["dpixel"]], 1)
                #     plt.plot(bgfitx, lin[bgfitx])
                #     plt.show()

        # add background from background shot is applicable
        if np.shape(BGele) == tuple(config["other"]["CCDsize"]):
            LineoutBGE2 = [
                np.mean(BGele[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
                for a in LineoutPixelE
            ]
            noiseE = LineoutBGE + np.array(LineoutBGE2)
        else:
            #ones_expanded = np.ones((len(LineoutPixelE), 1024))
           #noiseE = LineoutBGE[:20, :] * ones_expanded 
            noiseE = LineoutBGE * np.ones((len(LineoutPixelE), 1))

        # constant addition to the background
        noiseE += config["other"]["flatbg"]

    else:
        noiseE = np.zeros(len(config["data"]["lineouts"]["val"]))

    if config["other"]["extraoptions"]["load_ion_spec"]:
        # Due to the low background associated with IAWs the fitted background is only performed for the EPW

        if config["data"]["background"]["type"] in ("fit", "Fit", "FIT"):
        #if (config["data"]["background"]["type"] == "Fit" or config["data"]["background"]["type"] == "fit" or config["data"]["background"]["type"] =="FIT") :

            BackgroundPixel = config["data"]["background"]["slice"]

        # quantify a uniform background
        noiseI = np.mean(
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
