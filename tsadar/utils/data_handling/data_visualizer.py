import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects
import tempfile, mlflow, os

from tsadar.utils.process.lineouts import compute_lineout_pixel_indices


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
        None: The function saves the plots to a temporary directory and logs them to MLflow.

    Notes:
        - The function uses a temporary directory to save the plots, which are then logged to MLflow.
        - The function handles different types of lineouts based on the configuration provided.
        - The function uses matplotlib for plotting and color mapping.
        - The function assumes that the data is in a format compatible with numpy and matplotlib.

    """
    LineoutPixelE, LineoutPixelI = compute_lineout_pixel_indices(
        config,
        all_axes["epw_x"],
        all_axes["iaw_x"],
        config["data"]["ele_t0"],
        config["data"]["ion_t0_shift"],
    )

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        # until this can be made interactive this plots all the data regions
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

            fig, ax = plt.subplots()
            cb = ax.pcolormesh(
                X,
                Y,
                ionData,
                cmap="turbo",
                norm=colors.LogNorm(vmin =10, vmax= np.amax(ionData)/2),
            )
            (sline,) = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[0]], all_axes["iaw_x"][LineoutPixelI[0]]],
                [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                lw=2,
                color="w",
            )
            (eline,) = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[-1]], all_axes["iaw_x"][LineoutPixelI[-1]]],
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
            fig.colorbar(cb)
            fig.savefig(os.path.join(td, "plots", "ion_fit_ranges.png"), bbox_inches="tight")

        if config["other"]["extraoptions"]["load_ele_spec"]:
            X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

            fig, ax = plt.subplots()
            jc = ax.pcolormesh(
                X,
                Y,
                elecData,
                cmap="turbo",
                norm=colors.LogNorm(vmin =10, vmax= np.amax(elecData)/2),
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

        
        
        if config["other"]["extraoptions"]["load_ele_spec"] and config["other"]["extraoptions"]["load_ion_spec"]:
            fig = plt.figure()
            plt.plot(all_axes["epw_x"], np.sum(elecData[200:800,:], axis=0), label="Electron Spectrum")
            plt.plot(all_axes["iaw_x"]+config["data"]["ion_t0_shift"], np.sum(ionData[200:800,:], axis=0), label="Ion Spectrum")
            plt.xlabel(all_axes["x_label"])
            plt.ylabel("Integrated counts (a.u.)")
            plt.legend()
            fig.savefig(os.path.join(td, "plots", "temporal comparison.png"), bbox_inches="tight")

        mlflow.log_artifacts(td)
