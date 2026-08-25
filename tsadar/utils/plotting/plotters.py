"""Postprocessing plot library: final-parameter tables/plots, loss histograms, data-vs-fit and best/worst
lineout comparisons (simple and detailed), and the angular-data plotting functions."""
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import mlflow, os, pandas
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import ListedColormap

from tsadar.utils.plotting.lineout_plot import lineout_plot


def white_transition_cmap(base_cmap_name, transition_len=20):
    """
    Builds a version of the named colormap with its first `transition_len` entries replaced by a linear
    transition from white to the color at that boundary index, so low/zero values render as white instead
    of the base colormap's own low-end color.

    Args:
        base_cmap_name: name of a matplotlib colormap (e.g. "gist_ncar") to look up via mpl.colormaps.
        transition_len: number of entries (out of 256) to replace with the white-to-color transition.

    Returns:
        matplotlib.colors.ListedColormap
    """
    base = mpl.colormaps[base_cmap_name]
    colors = base(np.linspace(0, 1, 256))

    lower = np.ones((transition_len, 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the color at the transition boundary
    for i in range(3):
        lower[:, i] = np.linspace(1, colors[transition_len, i], lower.shape[0])

    colors[:transition_len, :] = lower
    return ListedColormap(colors)


def get_final_params(config, best_weights, all_axes, td):
    """
    Formats and saves the final fitted parameters and distribution function.
    This function processes the fitted parameters and distribution functions for all species, formats them into pandas DataFrames, and saves them as CSV files in a specified temporary directory. It handles different parameter structures depending on the species and configuration, and combines the results into a single output dictionary.
    Args:
        config (dict): Configuration dictionary created from the input decks.
        best_weights (dict): Dictionary containing all the fitted parameters for all the species.
        all_axes (dict): Dictionary with calibrated axes and axes labels.
        td (str): Temporary directory path where output files will be saved.
    Returns:
        dict: Dictionary containing all the fitted parameters and distribution function data. The keys are a combination of parameter and species names, and the values are pandas Series or arrays. The output merges the distribution function dictionary with the fitted parameter dictionary.
    """

    all_params = {}
    dist = {}
    fitted_dist = False
    for species in best_weights.keys():
        for k, v in best_weights[species].items():
            if k in ["fe","f"]:
                if config["parameters"][species]["fe"]["active"]:
                    fitted_dist = True
                    dist["fe"] = np.array(v).squeeze()
                    dist["v"] = best_weights["electron"]["v"]
                else:
                    continue
            elif k =="flm":
                fitted_dist = True
                #need to turn this into a lop for when we go to higher orders
                dist["flm0"] = v[0][0][0]
                dist["flm10"] = v[0][1][0]
                dist["flm11"] = v[0][1][1]
                dist["fe"] = v[0]['fvxvy']
                dist["v"] = v[0]['v']
            else:
                all_params[k + "_" + species] = pandas.Series(np.squeeze(v).reshape(-1))
                # if np.shape(v)[1] > 1:
                #     for i in range(np.shape(v)[1]):
                #         all_params[k + str(i)] = pandas.Series(v[:, i].reshape(-1))
                # else:
                #     all_params[k] = pandas.Series(v.reshape(-1))

    final_params = pandas.DataFrame(all_params)
    if config["data"]["load_ion_spec"]:
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelI"])
    elif config["other"]["extraoptions"]["spectype"] != "angular_full":
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelE"])
    final_params.to_csv(os.path.join(td, "csv", "learned_parameters.csv"))

    if fitted_dist:
        if len(np.shape(dist["fe"])) == 1:
            final_dist = pandas.DataFrame({"fe": [l for l in dist["fe"]], "vx": [vx for vx in dist["v"]]})
        elif len(np.shape(dist["fe"])) == 2:
            final_dist = pandas.DataFrame(data=dist["fe"], columns=dist["v"], index=dist["v"])
            if 'flm0' in dist.keys():
                flm_dist = pandas.DataFrame({key: dist[key] for key in dist.keys()-['fe','v']})
                flm_dist.to_csv(os.path.join(td, "csv", "learned_flm.csv"))
            # final_dist = pandas.DataFrame({'fe':[l for l in dist['fe']], 'vx':[vx for vx in dist['v'][0]], 'vy':[vy for vy in dist['v'][1]]})
        final_dist.to_csv(os.path.join(td, "csv", "learned_dist.csv"))

    return all_params | dist


def plot_final_params(config, all_params, sigmas_ds, td):
    """
    Plots the fitted parameters as a function of lineout. These plots include a blue uncertainty region from the hessian
    and a red uncertainty region from the moving average. The plots are saved to files but nothing is returned.


    Args:
        config: configuration dictionary created from the input decks
        all_params: dictionary containing all the fitted parameters for all the species, same as the best_params from
            the function get_final_params
        sigmas_ds: dictionary with uncertainty values for each of the fitted parameters calculated using the hessian
        td: temporary directory that will be uploaded to mlflow
    Returns:
        None: The function saves the plots to a temporary directory and logs them to MLflow.
    

    """
    for species in all_params.keys():
        for param in all_params[species].keys():
            if param not in ["fe","f","v","flm0","flm10","flm11"]:
                vals = pandas.Series(all_params[species][param], dtype=float)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                lineouts = np.array(config["data"]["lineouts"]["val"])
                std = vals.rolling(config["plotting"]["rolling_std_width"], min_periods=1, center=True).std()

                ax.plot(lineouts, vals)
                ax.fill_between(
                    lineouts,
                    (vals.values - config["plotting"]["n_sigmas"] * sigmas_ds[param + "_" + species].values),
                    (vals.values + config["plotting"]["n_sigmas"] * sigmas_ds[param + "_" + species].values),
                    color="b",
                    alpha=0.1,
                )
                ax.fill_between(
                    lineouts,
                    (vals.values - config["plotting"]["n_sigmas"] * std.values),
                    (vals.values + config["plotting"]["n_sigmas"] * std.values),
                    color="r",
                    alpha=0.1,
                )
                ax.set_xlabel("lineout", fontsize=14)
                ax.grid()
                #ax.set_ylim(0.8 * np.min(vals), 1.2 * np.max(vals))
                ax.set_ylabel(param, fontsize=14)
                fig.savefig(
                    os.path.join(td, "plots", "learned_" + param + "_" + species + ".png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
    return


def plot_loss_hist(config, losses_init, losses, reduced_points, td):
    """
    Plots histograms of the raw loss and reduced loss. Each histogram contains 2 data sets, blue for before refitting
    and orange for after refitting. The losses and reduced losses are saved to file as well. Note: A fit metric of
    chi-squared is used and the reduced metric is chi-squared per degree of freedom but this will not necessarily be
    near 1 since Thomson scattering often does not conform to chi-squared statistics.

    With the update of losses to loss/point this may be reduced to just "reduced losses"

    Args:
        config: configuration dictionary created from the input decks
        losses: array of losses with one value per lineout
        all_params: dictionary containing all the fitted parameters for all the species, same as the best_params from
            the function get_final_params
        used_points: int with the number of wavelength points used in each fit, calculated at the same time as loss
        td: temporary directory that will be uploaded to mlflow

    Returns:
        red_losses: array of the losses per degree of freedom for each lineout

    """
    losses[losses > 1e10] = 1e10
    red_losses = losses / (1.1 * reduced_points)
    red_losses_init = losses_init / (1.1 * reduced_points)
    mlflow.log_metrics(
        {"number of fits above threshold after refit": int(np.sum(red_losses > config["other"]["refit_thresh"]))}
    )

    # make histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

    ax[0].hist([red_losses_init, red_losses], 40)
    # ax[0].hist(red_losses, 128)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\chi^2/DOF$")
    ax[0].set_ylabel("Counts")
    ax[0].set_title("Normalized $L^2$ Norm of the Error")
    ax[0].grid()
    ax[0].legend(["Pre-refit Losses", "Post-refit Losses"])
    ax[1].hist([losses_init, losses], 40)
    # ax[1].hist(losses, 128)
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\chi^2$")
    ax[1].set_ylabel("Counts")
    ax[1].set_title("$L^2$ Norm of the Error")
    ax[1].grid()
    fig.savefig(os.path.join(td, "plots", "error_hist.png"), bbox_inches="tight")
    plt.close(fig)

    losses_ds = pandas.DataFrame(
        {
            "initial_losses": losses_init,
            "losses": losses,
            "initial_reduced_losses": red_losses_init,
            "reduced_losses": red_losses,
        }
    )
    losses_ds.to_csv(os.path.join(td, "csv", "losses.csv"))

    return red_losses


def plot_dist(config, final_params, sigma_fe, td):
    """
    Plots the fitted or used distribution function. For 1D distributions plots are does as line plots verse the 1D
    velocity. For 2D distributions a surface plot is shown as a function of the 2 velocities and contours are projected
    onto each plane. In both cases the distribution is plotted in linear spacing, log base 10 spacing, and log base e.

    Args:
        config: configuration dictionary created from the input decks
        final_params: dictionary containing the distribution function as produced by the function get_final_params
        sigma_fe: dictionary with uncertainty values for the distribution function as produced by the function
            save_sigmas_fe
        td: temporary directory that will be uploaded to mlflow

    Returns:
    """

    if config["parameters"]['electron']["fe"]["dim"] == 1:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(final_params["v"], np.log(final_params["fe"]))
        ax[1].plot(final_params["v"], np.log10(final_params["fe"]))
        ax[2].plot(final_params["v"], final_params["fe"])

        if config["other"]["calc_sigmas"]:
            ax[0].fill_between(
                final_params["v"],
                (final_params["fe"] - config["plotting"]["n_sigmas"] * sigma_fe.data),
                (final_params["fe"] + config["plotting"]["n_sigmas"] * sigma_fe.data),
                color="b",
                alpha=0.1,
            )
        ax[0].set_xlabel("v/vth (points)", fontsize=14)
        ax[0].set_ylabel("f_e (ln)")
        ax[0].grid()
        # ax.set_ylim(0.8 * np.min(final_params["ne"]), 1.2 * np.max(final_params["ne"]))
        ax[0].set_title("$f_e$", fontsize=14)
        ax[1].set_xlabel("v/vth (points)", fontsize=14)
        ax[1].set_ylabel("f_e (log)")
        ax[1].grid()
        ax[1].set_ylim(-5, 0)
        ax[1].set_title("$f_e$", fontsize=14)
        ax[2].set_xlabel("v/vth (points)", fontsize=14)
        ax[2].set_ylabel("f_e")
        ax[2].grid()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        x,y = np.meshgrid(final_params["v"], final_params["v"])
        c = ax[0].contourf(x,y, final_params["fe"].T)
        ax[0].set_xlabel("$v_x/v_{th}$", fontsize=14)
        ax[0].set_ylabel("$v_y/v_{th}$", fontsize=14)
        ax[0].set_title("$f_e$", fontsize=14)
        fig.colorbar(c)

        c = ax[1].contourf(x,y, np.log10(final_params["fe"].T))
        ax[1].set_xlabel("$v_x/v_{th}$", fontsize=14)
        ax[1].set_ylabel("$v_y/v_{th}$", fontsize=14)
        ax[1].set_title("log$_{10}(f_e)$", fontsize=14)
        fig.colorbar(c)

        print(np.isnan(final_params["fe"]).any())

        fig.savefig(os.path.join(td, "plots", "fe_contourf.png"), bbox_inches="tight")
        plt.close()

        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        curfe = np.where(np.log(final_params["fe"]) < -50.0, -50.0, np.log(final_params["fe"]))
        ax.plot_surface(
            x,
            y,
            curfe,
            edgecolor="royalblue",
            lw=0.5,
            alpha=0.3,
        )
        ax.set_zlim(-50, 0)
        ax.contour(x,y, curfe, zdir="x", offset=-7.5, cmap="coolwarm")
        ax.contour(x,y, curfe, zdir="y", offset=7.5, cmap="coolwarm")
        ax.contour(x,y, curfe, zdir="z", offset=-50, cmap="coolwarm")
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e (ln)")
        ax = fig.add_subplot(1, 3, 2, projection="3d")
        curfe = np.where(np.log10(final_params["fe"]) < -22.0, -22.0, np.log10(final_params["fe"]))
        ax.plot_surface(
            x,
            y,
            curfe,
            edgecolor="royalblue",
            lw=0.5,
            alpha=0.3,
        )
        ax.set_zlim(-22, 0)
        ax.contour(
            x,
            y,
            curfe,
            zdir="x",
            offset=-7.5,
            cmap="coolwarm",
        )
        ax.contour(
            x,
            y,
            curfe,
            zdir="y",
            offset=7.5,
            cmap="coolwarm",
        )
        ax.contour(
            x,
            y,
            curfe,
            zdir="z",
            offset=-22,
            cmap="coolwarm",
        )
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e (log)")

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        ax.plot_surface(
            x,
            y,
            final_params["fe"],
            edgecolor="royalblue",
            lw=0.5,
            alpha=0.3,
        )
        ax.set_zlim(0.0, 0.15)
        ax.contour(
            x,
            y,
            final_params["fe"],
            zdir="x",
            offset=-7.5,
            cmap="coolwarm",
        )
        ax.contour(
            x,
            y,
            final_params["fe"],
            zdir="y",
            offset=7.5,
            cmap="coolwarm",
        )
        ax.contour(
            x,
            y,
            final_params["fe"],
            zdir="z",
            offset=0.0,
            cmap="coolwarm",
        )
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e")

    # no rolling sigma bc we use a smoothing kernel
    fig.savefig(os.path.join(td, "plots", "fe_final.png"), bbox_inches="tight")
    plt.close(fig)
    return


def save_sigmas_fe(all_params, best_weights_std, sigmas, td):
    """
    Formats and saves the uncertainty values for the distribution function.

    Know Issues:
        THis code has not been updated to reflect the changes for multi-species

    Args:
        all_params: dictionary containing the distribution function as produced by the function get_final_params
        best_weights_std: standard deviations of the fitted parameters over repeated fitting
        sigmas: uncertainty values for the distribution function
        td: temporary directory that will be uploaded to mlflow

    Returns:
        sigma_fe: uncertainty values for the distribution function restructured as a DataArray.

    """
    sigma_params = {}
    sizes = {key: all_params[key].shape[0] for key in all_params.keys()}
    param_ctr = 0
    for i, k in enumerate(all_params.keys()):
        val = sigmas[0, param_ctr : param_ctr + sizes[k]]
        if k == "fe":
            sigma_fe = xr.DataArray(val, coords=(("v", np.linspace(-7, 7, len(val))),))
        else:
            sigma_params[k] = xr.DataArray(val, coords=(("ind", [0]),))
        param_ctr += sizes[k]

    sigma_params = best_weights_std
    sigma_fe.to_netcdf(os.path.join(td, "binary", "sigma-fe.nc"))
    sigma_params = xr.Dataset(sigma_params)
    sigma_params.to_netcdf(os.path.join(td, "binary", "sigma-params.nc"))

    return sigma_fe


def save_sigmas_params(config, all_params, sigmas, all_axes, td, filename="sigmas.nc"):
    """
    Formats and saves the uncertainty values for the fitted parameters.

    Args:
        config: configuration dictionary created from the input decks
        all_params: dictionary containing the distribution function as produced by the function get_final_params
        sigmas: uncertainty values for the fitted parameters
        all_axes: dictionary with calibrated axes and axes labels
        td: temporary directory that will be uploaded to mlflow
        filename: name of the netCDF file to write, relative to td. Defaults to "sigmas.nc" (the existing
            Hessian/Laplace uncertainty artifact); the MCMC postprocessor writes its own uncertainty here
            under a different name (see save_sigmas_params_mcmc) so both can coexist in one run's artifacts.

    Returns:
        sigma_ds: uncertainty values for each of the fitted parameters restructured as a DataArray.

    """
    coords = ((all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]])),)
    sigmas_ds = xr.Dataset(
        {
            k + "_" + series: xr.DataArray(sigmas[:, i], coords=coords)
            for series in all_params.keys()
            for i, k in enumerate(all_params[series].keys())
        }
    )
    sigmas_ds.to_netcdf(os.path.join(td, filename))
    return sigmas_ds


def save_sigmas_params_mcmc(config, all_params, sigmas, all_axes, td):
    """MCMC analogue of save_sigmas_params: identical schema, written to sigmas_mcmc.nc instead of
    sigmas.nc so the MCMC and Hessian/Laplace uncertainty artifacts can coexist unambiguously in one
    run's artifact set."""
    return save_sigmas_params(config, all_params, sigmas, all_axes, td, filename="sigmas_mcmc.nc")


def plot_mcmc_diagnostics(config, acceptance_rate, td):
    """
    Plots a histogram of per-lineout MCMC acceptance rates, so a user can tell at a glance whether the
    sampler's step-size adaptation actually converged (rates clustered near
    config["other"]["mcmc"]["target_accept"], not pinned at 0 or 1) or not.

    Args:
        config: configuration dictionary created from the input decks
        acceptance_rate: array of shape (num_lineouts,), the sampling-phase acceptance rate for each
            lineout, averaged across calibration draws.
        td: temporary directory that will be uploaded to mlflow

    Returns:
        None: the plot is saved to td/plots and logged to MLflow via the usual artifact upload.
    """
    target = config.get("other", {}).get("mcmc", {}).get("target_accept", 0.234)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.hist(np.asarray(acceptance_rate), bins=30)
    ax.axvline(target, color="r", linestyle="--", label=f"target ({target:.3f})")
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel("number of lineouts")
    ax.set_title("MCMC sampling-phase acceptance rate")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(td, "plots", "mcmc_acceptance_rate.png"), bbox_inches="tight")
    plt.close(fig)


def plot_sigma_comparison(config, all_params, laplace_sigmas_ds, mcmc_sigmas_ds, td):
    """
    Plots the Hessian/Laplace-derived sigma against the MCMC-derived sigma for each fitted parameter, as
    a function of lineout, so the two uncertainty methods can be visually compared.

    Args:
        config: configuration dictionary created from the input decks
        all_params: dictionary containing all the fitted parameters for all the species (only used for
            iterating parameter/species names, same as plot_final_params)
        laplace_sigmas_ds: xarray Dataset as returned by save_sigmas_params, or None if the Laplace
            comparison could not be computed (e.g. the Hessian was degenerate or too expensive) -- in
            that case only the MCMC sigma is plotted.
        mcmc_sigmas_ds: xarray Dataset as returned by save_sigmas_params_mcmc
        td: temporary directory that will be uploaded to mlflow

    Returns:
        None: the plots are saved to td/plots and logged to MLflow via the usual artifact upload.
    """
    lineouts = np.array(config["data"]["lineouts"]["val"])
    for species in all_params.keys():
        for param in all_params[species].keys():
            if param not in ["fe", "f", "v", "flm0", "flm10", "flm11"]:
                name = param + "_" + species
                if name not in mcmc_sigmas_ds:
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.plot(lineouts, np.abs(mcmc_sigmas_ds[name].values), label="MCMC")
                if laplace_sigmas_ds is not None and name in laplace_sigmas_ds:
                    ax.plot(lineouts, np.abs(laplace_sigmas_ds[name].values), label="Laplace/Hessian")
                ax.set_xlabel("lineout", fontsize=14)
                ax.set_ylabel(f"sigma({param})", fontsize=14)
                ax.legend()
                ax.grid()
                fig.savefig(
                    os.path.join(td, "plots", "mcmc_sigma_comparison_" + param + "_" + species + ".png"),
                    bbox_inches="tight",
                )
                plt.close(fig)


def plot_data_angular(config, fits, all_data, all_axes, td):
    """
    Plots the resulting spectrum from the fit vs the raw data for angularly resolved data. The data and fit will be
    plotted over the region used in the analysis. The function only creates the grids and structures the data before
    calling the general 2D plotting code.

    Args:
        config: configuration dictionary created from the input decks
        fits: dictionary containing the fitted spectra in a field called 'ele'
        all_data: dictionary containing the raw or processed data must have a field called 'e_data' which contains the
            angular EPW data
        all_axes: dictionary with calibrated axes and axes labels
        td: temporary directory that will be uploaded to mlflow

    Returns:
        savedata: dictionary containing the data and fits as DataArrays
    """
    dat = {
        "fit": fits["ele"],
        "data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
    }
    coords = (all_axes["x_label"], np.squeeze(all_axes["epw_x"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"]])), (
        "Wavelength",
        np.squeeze(all_axes["epw_y"]),
    )
    savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in dat.items()})
    savedata.to_netcdf(os.path.join(td, "binary", "fit_and_data.nc"))
    savedata["data"] = savedata["data"].T
    savedata["fit"] = savedata["fit"].T

    angs, wavs = np.meshgrid(
        all_axes["epw_x"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"]],
        all_axes["epw_y"],
    )

    plot_2D_data_vs_fit(config, angs, wavs, savedata["data"], savedata["fit"], td, xlabel="Angle (degrees)")

    return savedata


def plot_ts_data(config, fits, all_data, all_axes, td):
    """
    Plots the resulting spectrum from the fit vs the raw data for EPW and IAW data. The data and fit will be
    plotted over the region used in the analysis. The function only creates the grids and structures the data before
    calling the general 2D plotting code.

    Args:
        config: configuration dictionary created from the input decks
        fits: dictionary containing the fitted spectra
        all_data: dictionary containing the raw or processed data
        all_axes: dictionary with calibrated axes and axes labels
        td: temporary directory that will be uploaded to mlflow

    Returns:
    """
    background_subtract = config["data"]["background"]["bg_subtract"]
    if background_subtract:
        all_data["e_data"] = all_data["e_data"] - all_data["noiseE"]
        all_data["i_data"] = all_data["i_data"] - all_data["noiseI"]
    if config["data"]["load_ion_spec"]:
        coords_x = all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]])
        coords_y = "Wavelength", all_axes["iaw_y"]
        coords = coords_x, coords_y

        ion_dat = {"fit": fits["ion"]["total_spec"], "data": all_data["i_data"]}
        # fit vs data storage and plot
        ion_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ion_dat.items()})
        ion_savedata.to_netcdf(os.path.join(td, "binary", "ion_fit_and_data.nc"))

        ion_savedata["data"] = ion_savedata["data"].T
        ion_savedata["fit"] = ion_savedata["fit"].T

        x, y = np.meshgrid(
            all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]],
            all_axes["iaw_y"],
        )

        plot_2D_data_vs_fit(
            config,
            x,
            y,
            ion_savedata["data"],
            ion_savedata["fit"],
            td,
            xlabel=all_axes["x_label"],
            name="fit_and_data_ion.png",
        )

    if config["data"]["load_ele_spec"]:
        coords = (all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]])), (
            "Wavelength",
            all_axes["epw_y"],
        )
        ele_dat = {"fit": fits["ele"]["total_spec"], "data": all_data["e_data"]}
        # fit vs data storage and plot
        ele_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ele_dat.items()})
        ele_savedata.to_netcdf(os.path.join(td, "binary", "ele_fit_and_data.nc"))

        ele_savedata["data"] = ele_savedata["data"].T
        ele_savedata["fit"] = ele_savedata["fit"].T

        x, y = np.meshgrid(
            all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]],
            all_axes["epw_y"],
        )

        plot_2D_data_vs_fit(
            config,
            x,
            y,
            ele_savedata["data"],
            ele_savedata["fit"],
            td,
            xlabel=all_axes["x_label"],
            name="fit_and_data_ele.png",
        )


def plot_2D_data_vs_fit(
    config, x, y, data, fit, td, xlabel="Time (ps)", ylabel="Wavelength (nm)", name="fit_and_data.png"
):
    """
    Plots and then saves a set of 2 color plots (each 2D). Mainly used to plot data vs fit images.

    Args:
        config: configuration dictionary created from the input decks
        x: x-axis coordinates from meshgrid
        y: y-axis coordinates from meshgrid
        data: data array
        fit: fit array
        td: temporary directory that will be uploaded to mlflow
        xlabel: label to be used for the x-axis
        ylabel: label to be used for the y-axis
        name: name under which the file will be saved

    Returns:

    """
    newcmp = white_transition_cmap("gist_ncar", transition_len=20)

    # Check if data is all zeros
    data_all_zeros = np.all(data == 0)
    if config["plotting"]["data_cbar_l"] == "data":
        if data_all_zeros:
            vmin = np.amin(fit)
        else:
            vmin = np.amin(data)
    else:
        vmin = config["plotting"]["data_cbar_l"]
    if config["plotting"]["data_cbar_u"] == "data":
        if data_all_zeros:
            vmax = np.amax(fit)
        else:
            vmax = np.amax(data)
    else:
        vmax = config["plotting"]["data_cbar_u"]

    # Create fit and data image
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    pc = ax[0].pcolormesh(x, y, fit, shading="nearest", cmap=newcmp, vmin=vmin, vmax=vmax)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].pcolormesh(x, y, data, shading="nearest", cmap=newcmp, vmin=vmin, vmax=vmax)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    fig.colorbar(pc)
    fig.savefig(os.path.join(td, "plots", name), bbox_inches="tight")
    plt.close(fig)


def plot_ang_lineouts(used_points, sqdevs, losses, all_params, all_axes, savedata, td):
    """
    Plots lineout comparing the fits to the data, but designed for angular data. The value of the fit metric chi^2 per
    point is plotted beneath the data and fit.


    Args:
        used_points: numer of points used in the calculation of the fit metric
        sqdevs: chi^2 per point. Must be the same shape as data
        losses: array of losses with one value per lineout
        all_params: dictionary containing all the fitted parameters for all the species, same as the best_params from
            the function get_final_params
        all_axes: dictionary with the calibrated axes and axes labels
        savedata: dictionary with data and fitted spectra
        td: temporary directory that will be uploaded to mlflow

    Returns:

    """
    used_points = used_points * sqdevs["ele"].shape[1]
    red_losses = np.sum(losses) / (1.1 * (used_points - len(all_params)))
    mlflow.log_metrics({"Total reduced loss": float(red_losses)})

    # Create lineout images
    os.makedirs(os.path.join(td, "lineouts"))
    for i in np.linspace(0, savedata["data"].shape[1] - 1, 8, dtype="int"):
        # plot model vs actual
        titlestr = r"|Error|$^2$" + f" = {losses[i]:.2e}, line out # {i}"
        filename = f"loss={losses[i]:.2e}-lineout={i}.png"
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True, sharex=True)
        ax[0].plot(all_axes["epw_y"], np.squeeze(savedata["data"][:, i]), label="Data")
        ax[0].plot(all_axes["epw_y"], np.squeeze(savedata["fit"][:, i]), label="Fit")
        ax[0].set_title(titlestr, fontsize=14)
        ax[0].set_ylabel("Amplitude (arb. units)")
        ax[0].legend(fontsize=14)
        ax[0].grid()
        ax[1].plot(all_axes["epw_y"], np.squeeze(sqdevs["ele"][i, :]), label="Residual")
        ax[1].set_ylabel(r"$\chi^2_i$")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].grid()
        fig.savefig(os.path.join(td, "lineouts", filename), bbox_inches="tight")
        plt.close(fig)
    return


def model_v_actual(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td):
    """
    Creates a set of plots, up to 8, comparing the best and worst fits to the data. THis function does the sorting and
    the lineout_plot code is used to do the plotting.


    Args:
        config: configuration dictionary created from the input decks
        all_data: dictionary containing the raw or processed data
        fits: dictionary containing the fitted spectra
        losses: array of losses with one value per lineout
        red_losses: array of the losses per lineout divided by the number of degrees of freedom
        sqdevs: chi^2 per point. Must be the same shape as data
        td: temporary directory that will be uploaded to mlflow

    Returns:
    """
    num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

    os.makedirs(os.path.join(td, "worst"))
    os.makedirs(os.path.join(td, "best"))

    loss_inds = losses.flatten().argsort()[::-1]
    sorted_losses = losses[loss_inds]
    sorted_red_losses = red_losses[loss_inds]
    s_ind = []
    e_ind = []
    sorted_data = []
    sorted_fits = []
    sorted_sqdev = []
    yaxis = []

    if config["optimizer"]["loss_method"]=='covar':
        include_data_uncert = True

    if config["data"]["load_ele_spec"]:
        s_ind.append(np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_start"])))
        e_ind.append(np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_end"])))
        sorted_fits.append(fits["ele"]["total_spec"][loss_inds])
        sorted_data.append(all_data["e_data"][loss_inds])
        sorted_sqdev.append(sqdevs["ele"][loss_inds])
        yaxis.append(all_axes["epw_y"])

    if config["data"]["load_ion_spec"]:
        s_ind.append(np.argmin(np.abs(all_axes["iaw_y"] - config["plotting"]["ion_window_start"])))
        e_ind.append(np.argmin(np.abs(all_axes["iaw_y"] - config["plotting"]["ion_window_end"])))
        sorted_fits.append(fits["ion"]["total_spec"][loss_inds])
        sorted_data.append(all_data["i_data"][loss_inds])
        sorted_sqdev.append(sqdevs["ion"][loss_inds])
        yaxis.append(all_axes["iaw_y"])

    for i in range(num_plots):
        # plot model vs actual
        titlestr = (
            r"|Error|$^2$"
            + f" = {sorted_losses[i]:.2e}, line out # {all_axes['iaw_x'][config['data']['lineouts']['pixelI'][loss_inds[i]]]}"
        )
        filename = f"loss={sorted_losses[i]:.2e}-reduced_loss={sorted_red_losses[i]:.2e}-lineout={config['data']['lineouts']['pixelI'][loss_inds[i]]}.png"

        lineout_plot(
            np.array(sorted_data)[:, i, :],
            np.array(sorted_fits)[:, i, :],
            np.array(sorted_sqdev)[:, i, :],
            yaxis,
            (
                None if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
                None if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"],
            ),
            s_ind,
            e_ind,
            titlestr,
            False,
            filename,
            td,
            "worst",
        )

        titlestr = (
            r"|Error|$^2$"
            + f" = {sorted_losses[-1 - i]:.2e}, line out # {all_axes['iaw_x'][config['data']['lineouts']['pixelI'][loss_inds[-1 - i]]]}"
        )
        filename = f"loss={sorted_losses[-1 - i]:.2e}-reduced_loss={sorted_red_losses[-1 - i]:.2e}-lineout={config['data']['lineouts']['pixelI'][loss_inds[-1 - i]]}.png"

        lineout_plot(
            np.array(sorted_data)[:, -1 - i, :],
            np.array(sorted_fits)[:, -1 - i, :],
            np.array(sorted_sqdev)[:, -1 - i, :],
            yaxis,
            (
                None if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
                None if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"],
            ),
            s_ind,
            e_ind,
            titlestr,
            False,
            filename,
            td,
            "best",
        )

def _plot_lineout_comparison(config, all_data, all_axes, fits, sqdevs, idx, titlestr, filename, subfolder, td):
    """
    Shared by detailed_lineouts' "worst" and "best" loops: builds one comparison figure (data vs fit, per-component
    breakdown, and residual, for electron and/or ion features) for a single lineout index and saves it under
    td/subfolder.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), squeeze=False, tight_layout=True, sharex='col')

    if config["data"]["load_ele_spec"]:
        s_ind = np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_start"]))
        e_ind = np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_end"]))
        ax[0][0].plot(
            all_axes["epw_y"][s_ind:e_ind], np.squeeze(all_data["e_data"][idx][s_ind:e_ind]), label="Data"
        )
        ax[0][0].plot(all_axes["epw_y"], np.squeeze(fits["ele"]["total_spec"][idx]), label="Total Fit")
        ax[0][0].plot(all_axes["epw_y"], np.squeeze(fits["ele"]["noise"][idx]), label="Background")
        ax[0][0].plot(
            fits["ele"]["detailed_axis"], np.squeeze(fits["ele"]["spec_comps"][idx, 0, :, 0]), label="First Grad/ Angle"
        )
        ax[0][0].plot(
            fits["ele"]["detailed_axis"], np.squeeze(fits["ele"]["spec_comps"][idx, -1, :, 0]), label="Last Grad"
        )
        ax[0][0].plot(
            fits["ele"]["detailed_axis"], np.squeeze(fits["ele"]["spec_comps"][idx, 0, :, -1]), label="Last angle"
        )
        ax[0][0].plot(all_axes["epw_y"], np.squeeze(fits["ele"]["IRF"][idx]), label="IRF")

        ax[0][0].set_title(titlestr, fontsize=14)
        ax[0][0].set_ylabel("Amp (arb. units)")
        ax[0][0].legend(loc='upper right', bbox_to_anchor=(1.05, 1.05), fontsize=12)
        ax[0][0].grid()
        ax[0][0].set_xlim([config["plotting"]["ele_window_start"], config["plotting"]["ele_window_end"]])
        ax[0][0].set_ylim(
            [None if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
             None if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"]])

        ax[1][0].plot(all_axes["epw_y"], np.squeeze(sqdevs["ele"][idx]), label="Residual")
        ax[1][0].set_xlabel("Wavelength (nm)")
        ax[1][0].set_ylabel(r"$\chi_i^2$")

    if config["data"]["load_ion_spec"]:
        ax[0][1].plot(all_axes["iaw_y"], np.squeeze(all_data["i_data"][idx]), label="Data")
        ax[0][1].plot(all_axes["iaw_y"], np.squeeze(fits["ion"]["total_spec"][idx]), label="Total Fit")
        ax[0][1].plot(all_axes["iaw_y"], np.squeeze(fits["ion"]["noise"][idx]), label="Background")
        ax[0][1].plot(
            fits["ion"]["detailed_axis"], np.squeeze(fits["ion"]["spec_comps"][idx, 0, :, 0]), label="First Grad/ Angle"
        )
        ax[0][1].plot(
            fits["ion"]["detailed_axis"], np.squeeze(fits["ion"]["spec_comps"][idx, -1, :, 0]), label="Last Grad"
        )
        ax[0][1].plot(
            fits["ion"]["detailed_axis"], np.squeeze(fits["ion"]["spec_comps"][idx, 0, :, -1]), label="Last angle"
        )
        ax[0][1].plot(all_axes["iaw_y"], np.squeeze(fits["ion"]["IRF"][idx]), label="IRF")

        ax[0][1].set_title(titlestr, fontsize=14)
        ax[0][1].set_ylabel("Amp (arb. units)")
        ax[0][1].legend(loc='upper right', bbox_to_anchor=(1.05, 1.05), fontsize=12)
        ax[0][1].grid()
        ax[0][1].set_xlim([config["plotting"]["ion_window_start"], config["plotting"]["ion_window_end"]])
        ax[0][1].set_ylim(
            [None if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
             None if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"]])

        ax[1][1].plot(all_axes["iaw_y"], np.squeeze(sqdevs["ion"][idx]), label="Residual")
        ax[1][1].set_xlabel("Wavelength (nm)")
        ax[1][1].set_ylabel(r"$\chi_i^2$")

    fig.savefig(os.path.join(td, subfolder, filename), bbox_inches="tight")
    plt.close(fig)


def detailed_lineouts(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td):
    """
    TODO
    Creates a set of plots, up to 8, comparing the best and worst fits to the data. THis function does the sorting and
    the lineout_plot code is used to do the plotting.


    Args:
        config: configuration dictionary created from the input decks
        all_data: dictionary containing the raw or processed data
        fits: dictionary containing the fitted spectra
        losses: array of losses with one value per lineout
        red_losses: array of the losses per lineout divided by the number of degrees of freedom
        sqdevs: chi^2 per point. Must be the same shape as data
        td: temporary directory that will be uploaded to mlflow

    Returns:
    """
    num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

    os.makedirs(os.path.join(td, "worst"))
    os.makedirs(os.path.join(td, "best"))

    loss_inds = losses.flatten().argsort()[::-1]

    for i in range(num_plots):
        idx = loss_inds[i]
        titlestr = (
            r"|Error|$^2$"
            + f" = {losses[idx]:.2e}, line out # {all_axes['iaw_x'][config['data']['lineouts']['pixelI'][idx]]}"
        )
        filename = f"loss={losses[idx]:.2e}-reduced_loss={red_losses[idx]:.2e}-lineout={config['data']['lineouts']['pixelI'][idx]}.png"
        _plot_lineout_comparison(config, all_data, all_axes, fits, sqdevs, idx, titlestr, filename, "worst", td)

        idx = loss_inds[-1 - i]
        titlestr = (
            r"|Error|$^2$"
            + f" = {losses[idx]:.2e}, line out # {all_axes['iaw_x'][config['data']['lineouts']['pixelI'][idx]]}"
        )
        filename = f"loss={losses[idx]:.2e}-reduced_loss={red_losses[idx]:.2e}-lineout={config['data']['lineouts']['pixelI'][idx]}.png"
        _plot_lineout_comparison(config, all_data, all_axes, fits, sqdevs, idx, titlestr, filename, "best", td)


def TScmap():
    """
    Creates a custom matplotlib colormap based on the 'jet' colormap with a white segment at the lower end.
    The resulting colormap starts with a smooth transition from white to the first color of the 'jet' colormap,
    followed by the standard 'jet' colors. This is useful for visualizations where zero or low values should
    be represented as white.
    Returns:
        matplotlib.colors.ListedColormap: The custom colormap with a white-to-jet transition at the lower end.
    """
    # Define jet colormap with 0=white (this might be moved and just loaded here)
    upper = mpl.cm.jet(np.arange(256))
    lower = np.ones((int(256 / 16), 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    # combine parts of colormap
    cmap = np.vstack((lower, upper))

    # convert to matplotlib colormap
    cmap = mpl.colors.ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])

    return cmap
