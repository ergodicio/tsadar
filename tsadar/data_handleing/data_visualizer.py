import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects
import tempfile, mlflow, os



def launch_data_visualizer(elecData, ionData, all_axes, config):
    """
    Plots the raw data with solid lines indicating the beginning and ending of the analysis and dashed lines indicating
    the portions of the spectrum that are included in the analysis.

    Args:
        elecData: Electron data to be plotted, if electron data is not loaded a dummy can be placed here
        ionData: Ion data to be plotted, if ion data is not loaded a dummy can be placed here
        all_axes: A dictionary containing the axes for the data being plotted. If electron data is plotted 'epw_x' and
        'epw_y' are required fields. If ion data is plotted 'iaw_x' and 'iaw_y' are required fields.
        config: Dictionary constructed from input deck

    Returns:

    """
    if config["data"]["lineouts"]["type"] == "ps" or config["data"]["lineouts"]["type"] == "um":
        LineoutPixelE = [
            np.argmin(abs(all_axes["epw_x"] - loc - config["data"]["ele_t0"]))
            for loc in config["data"]["lineouts"]["val"]
        ]
        IAWtime = (
            config["data"]["ion_t0_shift"] / all_axes["iaw_x"][1]
        )  # corrects the iontime to be in the same units as the lineout
        LineoutPixelI = [
            np.argmin(abs(all_axes["iaw_x"] - loc - config["data"]["ele_t0"]))
            for loc in config["data"]["lineouts"]["val"]
        ]
    elif config["data"]["lineouts"]["type"] == "pixel":
        LineoutPixelE = config["data"]["lineouts"]["val"]
        LineoutPixelI = config["data"]["lineouts"]["val"]
        IAWtime = config["data"]["ion_t0_shift"]
    else:
        raise NotImplementedError
    LineoutPixelI = np.round(np.array(LineoutPixelI) - IAWtime).astype(int)

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        # until this can be made interactive this plots all the data regions
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

            print(X)
            print(Y)

            fig, ax = plt.subplots()
            cb = ax.pcolormesh(
                X,
                Y,
                ionData,
                norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
                cmap="turbo_r",
                #norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03, vmin =0, vmax= np.amax(ionData)),
                norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(ionData), vmax= np.amax(ionData)),
                #vmin=np.amin(ionData),
                #vmin=0,
                #vmax= np.amax(ionData),
            )
            (sline,) = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[0]], all_axes["iaw_x"][LineoutPixelI[0]]],
                #[config["data"]["lineouts"]["start"], config["data"]["lineouts"]["start"]]
                [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                lw=2,
                color="w",
            )
            (eline,) = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[-1]], all_axes["iaw_x"][LineoutPixelI[-1]]],
               # [config["data"]["lineouts"]["end"], config["data"]["lineouts"]["end"]],
                [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                lw=2,
                color="w",
            )

            (lamsline,) = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_min"], config["data"]["fit_rng"]["iaw_min"]],
                lw=2,
                color="w",
                linestyle=":",
            )
            (lamsline,) = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_cf_min"], config["data"]["fit_rng"]["iaw_cf_min"]],
                lw=2,
                color="w",
                linestyle="-.",
            )
            (lamsline,) = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_cf_max"], config["data"]["fit_rng"]["iaw_cf_max"]],
                lw=2,
                color="w",
                linestyle="-.",
            )
            (lameline,) = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_max"], config["data"]["fit_rng"]["iaw_max"]],
                #path_effects=[patheffects.withTickedStroke(spacing=7, angle=135)],
                lw=2,
                color="w",
                linestyle="--", 
            )
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength (nm)")
            fig.colorbar(c
                         b)
            fig.savefig(os.path.join(td, "plots", "ion_fit_ranges.png"), bbox_inches="tight")

        if config["other"]["extraoptions"]["load_ele_spec"]:
            X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])
            print("printing x and y")
            print(X)
            print(Y)
            fig, ax = plt.subplots()
            jc= ax.pcolormesh(
                X,
                Y,
                elecData,
                norm=colors.SymLogNorm( linthresh = 0.03, linscale = 0.03,vmin=np.amin(elecData), vmax= np.amax(elecData)),
                cmap="turbo_r",
                shading= "auto",
                #vmin=np.amin(elecData),
                #vmin=0,
                #vmax= np.amax(elecData)
            )
            (sline,) = ax.plot(
                [all_axes["epw_x"][LineoutPixelE[0]], all_axes["epw_x"][LineoutPixelE[0]]],
                [all_axes["epw_y"][0], all_axes["epw_y"][-1]],
                lw=2,
                color="w",
            )
            (eline,) = ax.plot(
                [all_axes["epw_x"][LineoutPixelE[-1]], all_axes["epw_x"][LineoutPixelE[-1]]],
                [all_axes["epw_y"][0], all_axes["epw_y"][-1]],
                lw=2,
                color="w",
            )

            (lamsline,) = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["blue_min"], config["data"]["fit_rng"]["blue_min"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            (lameline,) = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["blue_max"], config["data"]["fit_rng"]["blue_max"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            (lamsline,) = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["red_min"], config["data"]["fit_rng"]["red_min"]],
                lw=2,
                color="w",
                linestyle=":",
            )
            (lameline,) = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["red_max"], config["data"]["fit_rng"]["red_max"]],
                lw=2,
                color="w",
                linestyle=":",
            )
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength (nm)")
            fig.colorbar(jc)
            fig.savefig(os.path.join(td, "plots", "electron_fit_ranges.png"), bbox_inches="tight")

        mlflow.log_artifacts(td)
