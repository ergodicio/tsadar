import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle
import tempfile, mlflow, os

from tsadar.utils.process.lineouts import compute_lineout_pixel_indices


def launch_data_visualizer(elecData, ionData, all_data, all_axes, config):
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
        # plot the raw data with solid lines indicating the beginning and ending of the analysis and dashed lines indicating the portions of the spectrum that are included in the analysis
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
            x_start = all_axes["iaw_x"][LineoutPixelI[0]]
            x_end = all_axes["iaw_x"][LineoutPixelI[-1]]
            y_min = config["data"]["fit_rng"]["iaw_min"]
            y_cf_min = config["data"]["fit_rng"]["iaw_cf_min"]
            y_cf_max = config["data"]["fit_rng"]["iaw_cf_max"]
            y_max = config["data"]["fit_rng"]["iaw_max"]
            x_min = all_axes["iaw_x"][0]
            x_max = all_axes["iaw_x"][-1]
            y_lo = all_axes["iaw_y"][0]
            y_hi = all_axes["iaw_y"][-1]

            ax.add_patch(Rectangle((x_min, y_lo), x_start - x_min, y_hi - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_end, y_lo), x_max - x_end, y_hi - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_lo), x_end - x_start, y_min - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_cf_min), x_end - x_start, y_cf_max - y_cf_min, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_max), x_end - x_start, y_hi - y_max, facecolor="black", alpha=0.35))
            # plot line indicating background lineout location if background lineout is not taken from a background shot
            if "pixel" in config["data"]["background"]:
                (bgline,) = ax.plot(
                    [all_axes["iaw_x"][config["data"]["background"]["pixel"]], all_axes["iaw_x"][config["data"]["background"]["pixel"]]],
                    [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                    lw=2,
                    color="k",
                    linestyle="-",
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
            x_start = all_axes["epw_x"][LineoutPixelE[0]]
            x_end = all_axes["epw_x"][LineoutPixelE[-1]]
            y_blue_min = config["data"]["fit_rng"]["blue_min"]
            y_blue_max = config["data"]["fit_rng"]["blue_max"]
            y_red_min = config["data"]["fit_rng"]["red_min"]
            y_red_max = config["data"]["fit_rng"]["red_max"]
            x_min = all_axes["epw_x"][0]
            x_max = all_axes["epw_x"][-1]
            y_lo = all_axes["epw_y"][0]
            y_hi = all_axes["epw_y"][-1]

            ax.add_patch(Rectangle((x_min, y_lo), x_start - x_min, y_hi - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_end, y_lo), x_max - x_end, y_hi - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_lo), x_end - x_start, y_blue_min - y_lo, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_blue_max), x_end - x_start, y_red_min - y_blue_max, facecolor="black", alpha=0.35))
            ax.add_patch(Rectangle((x_start, y_red_max), x_end - x_start, y_hi - y_red_max, facecolor="black", alpha=0.35))
            # plot line indicating background lineout location if background lineout is not taken from a background shot
            if "pixel" in config["data"]["background"]:
                (bgline,) = ax.plot(
                    [all_axes["epw_x"][config["data"]["background"]["pixel"]], all_axes["epw_x"][config["data"]["background"]["pixel"]]],
                    [all_axes["epw_y"][0], all_axes["epw_y"][-1]],
                    lw=2,
                    color="k",
                    linestyle="-",
                )
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength (nm)")
            fig.colorbar(jc)
            fig.savefig(os.path.join(td, "plots", "electron_fit_ranges.png"), bbox_inches="tight")

        
        # Plot temporal comparison of electron and ion spectra if both are loaded to check timing alignment
        if config["other"]["extraoptions"]["load_ele_spec"] and config["other"]["extraoptions"]["load_ion_spec"]:
            fig = plt.figure()
            plt.plot(all_axes["epw_x"], np.sum(elecData[200:800,:], axis=0), label="Electron Spectrum")
            plt.plot(all_axes["iaw_x"]+config["data"]["ion_t0_shift"], np.sum(ionData[200:800,:], axis=0), label="Ion Spectrum")
            plt.xlabel(all_axes["x_label"])
            plt.ylabel("Integrated counts (a.u.)")
            plt.legend()
            

        # Plot lineout of the data with its background to check background subtraction
        if config["data"]["background"]["type"].casefold() in ["fit", "pixel"]:
            #create a figure with 6 subplots, 3 for the electron lineouts and 3 for the ion lineouts, with the lineouts and the backgrounds plotted together
            num_lineouts = len(all_data["e_data"]) if all_data["e_data"].size > 0 else len(all_data["i_data"])
            lineout_indices = [0, num_lineouts // 2, num_lineouts - 1]
            
            fig, ax = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)
            for idx, lineout_idx in enumerate(lineout_indices):
                if config["other"]["extraoptions"]["load_ele_spec"]:
                    ax[0][idx].plot(all_axes["epw_y"][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], all_data["e_data"][lineout_idx][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], label="Electron Lineout")
                    ax[0][idx].plot(all_axes["epw_y"][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], all_data["noiseE"][lineout_idx][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], label="Electron Background")
                    ax[0][idx].set_xlabel("Wavelength (nm)")
                    ax[0][idx].set_ylabel("Counts (a.u.)")
                    ax[0][idx].legend()
                if config["other"]["extraoptions"]["load_ion_spec"]:
                    ax[1][idx].plot(all_axes["iaw_y"][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], all_data["i_data"][lineout_idx][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], label="Ion Lineout")
                    ax[1][idx].plot(all_axes["iaw_y"][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], all_data["noiseI"][lineout_idx][config["data"]["background"]["bg_alg_domain"][0]:config["data"]["background"]["bg_alg_domain"][-1]], label="Ion Background")
                    ax[1][idx].set_xlabel("Wavelength (nm)")
                    ax[1][idx].set_ylabel("Counts (a.u.)")
                    ax[1][idx].legend()
            fig.savefig(os.path.join(td, "plots", "lineouts_with_background.png"), bbox_inches="tight")
        mlflow.log_artifacts(td)
