from time import time
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import xarray as xr
import pandas

from ..utils.plotting import plotters
from ..core.thomson_diagnostic import ThomsonScatteringDiagnostic
from ..core.modules.ts_params import ThomsonParams
from ..utils.data_handling.calibration import get_scattering_angles, get_calibrations


def forward_pass(config):
    
    """
    Calculates a spectrum or series of spectra from the input configuration, performing a forward pass or a series of forward passes.

    Args:    
        config (dict): Configuration dictionary created from the input deck. For a series of spectra, contains the special
            field 'series', which can have up to 8 subfields: [param1, vals1, param2, vals2, param3, vals3, param4, vals4].
            The param subfields are strings identifying which fields of "parameters" are to be looped over. The vals subfields
            provide the values for each spectrum in the series.
    Returns:
        None: The function does not return any values. Instead, it saves the ion and electron data as NetCDF files, saves plots of
            simulated data, and logs artifacts and metrics to mlflow.    
    Side Effects:
        - Saves ion and electron data as NetCDF files.
        - Saves plots of simulated data.
        - Logs artifacts and metrics to mlflow.
    Notes:
        - The ability to loop over multiple parameters or generate a series of spectra is not working with v0.1+ refactoring.

    """
    is_angular = True if "angular" in config["other"]["extraoptions"]["spectype"] else False
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1

    config["other"]["lamrangE"] = [
        config["data"]["fit_rng"]["forward_epw_start"],
        config["data"]["fit_rng"]["forward_epw_end"],
    ]
    config["other"]["lamrangI"] = [
        config["data"]["fit_rng"]["forward_iaw_start"],
        config["data"]["fit_rng"]["forward_iaw_end"],
    ]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

    sas = get_scattering_angles(config)

    dummy_batch = {
        "i_data": np.array([1]),
        "e_data": np.array([1]),
        "noise_e": np.array([0]),
        "noise_i": np.array([0]),
        "e_amps": np.array([1]),
        "i_amps": np.array([1]),
    }

    if is_angular:
        [axisxE, _, _, _, _, _] = get_calibrations(
            104000, config["other"]["extraoptions"]["spectype"], 0.0, config["other"]["CCDsize"]
        )  # shot number hardcoded to get calibration
        config["other"]["extraoptions"]["spectype"] = "angular_full"

        sas["angAxis"] = axisxE
        dummy_batch["i_data"] = np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1]))
        dummy_batch["e_data"] = np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1]))

    if "series" in config.keys():
        serieslen = len(config["series"]["vals1"])
    else:
        serieslen = 1
    ThryE = [None] * serieslen
    ThryI = [None] * serieslen
    lamAxisE = [None] * serieslen
    lamAxisI = [None] * serieslen

    t_start = time()
    for i in range(serieslen):
        # if "series" in config.keys():
        #     config["parameters"]["species"][config["series"]["param1"]]["val"] = config["series"]["vals1"][i]
        #     if "param2" in config["series"].keys():
        #         config["parameters"]["species"][config["series"]["param2"]]["val"] = config["series"]["vals2"][i]
        #     if "param3" in config["series"].keys():
        #         config["parameters"]["species"][config["series"]["param3"]]["val"] = config["series"]["vals3"][i]
        #     if "param4" in config["series"].keys():
        #         config["parameters"]["species"][config["series"]["param4"]]["val"] = config["series"]["vals4"][i]

        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=not is_angular)
        ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)

        # params = ts_diag.get_plasma_parameters(ts_diag.pytree_weights["active"])
        ThryE[i], ThryI[i], lamAxisE[i], lamAxisI[i] = ts_diag(ts_params, dummy_batch)

    spectime = time() - t_start
    ThryE = np.array(ThryE)
    ThryI = np.array(ThryI)
    lamAxisE = np.array(lamAxisE)
    lamAxisI = np.array(lamAxisI)

    # physical_params = ts_params()
    # fe_val = physical_params["electron"]["fe"][0]
    # velocity = physical_params["electron"]["v"][0]

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        os.makedirs(os.path.join(td, "binary"), exist_ok=True)
        os.makedirs(os.path.join(td, "csv"), exist_ok=True)
        if is_angular:
            physical_params = ts_params()
            fe_val = physical_params["electron"]["fe"]
            velocity = physical_params["electron"]["v"]

            savedata = plotters.plot_data_angular(
                config,
                {"ele": np.squeeze(ThryE)},
                {"e_data": np.zeros((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1]))},
                {"epw_x": sas["angAxis"], "epw_y": lamAxisE, 'x_label': 'Angle'},
                td,
            )
            # plotters.plot_dist(config, "electron", {"fe": np.squeeze(fe_val), "v": velocity}, np.zeros_like(fe_val), td)
            # if len(np.shape(np.squeeze(fe_val))) == 1:
            #     final_dist = pandas.DataFrame({"fe": [l for l in fe_val], "vx": [vx for vx in velocity]})
            # elif len(np.shape(np.squeeze(fe_val))) == 2:
            #     final_dist = pandas.DataFrame(
            #         data=np.squeeze(fe_val),
            #         columns=velocity[0][0],
            #         index=velocity[0][:, 0],
            #     )
            # final_dist.to_csv(os.path.join(td, "csv", "learned_dist.csv"))
        else:
            if config["parameters"]["electron"]["fe"]["dim"] == 2:
                plotters.plot_dist(config, "electron", {"fe": fe_val, "v": velocity}, np.zeros_like(fe_val), td)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
            if config["other"]["extraoptions"]["load_ele_spec"]:
                ax[0].plot(
                    lamAxisE.squeeze().transpose(), ThryE.squeeze().transpose()
                )  # transpose might break single specs?
                ax[0].set_title("Simulated Data", fontsize=14)
                ax[0].set_ylabel("Amp (arb. units)")
                ax[0].set_xlabel("Wavelength (nm)")
                ax[0].grid()

                if "series" in config.keys():
                    ax[0].legend([str(ele) for ele in config["series"]["vals1"]])
                    if config["series"]["param1"] == "fract" or config["series"]["param1"] == "Z":
                        coords_ele = (
                            ("series", np.array(config["series"]["vals1"])[:, 0]),
                            ("Wavelength", lamAxisE[0, :]),
                        )
                    else:
                        coords_ele = (("series", config["series"]["vals1"]), ("Wavelength", lamAxisE[0, :]))
                    ele_dat = {"Sim": ThryE}
                    ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
                else:
                    coords_ele = (("series", [0]), ("Wavelength", lamAxisE[0, :].squeeze()))
                    ele_dat = {"Sim": ThryE.squeeze(0)}
                    ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
                ele_data.to_netcdf(os.path.join(td, "binary", "electron_data.nc"))

            if config["other"]["extraoptions"]["load_ion_spec"]:
                ax[1].plot(lamAxisI.squeeze().transpose(), ThryI.squeeze().transpose())
                ax[1].set_title("Simulated Data", fontsize=14)
                ax[1].set_ylabel("Amp (arb. units)")
                ax[1].set_xlabel("Wavelength (nm)")
                ax[1].grid()

                if "series" in config.keys():
                    ax[1].legend([str(ele) for ele in config["series"]["vals1"]])
                    if config["series"]["param1"] == "fract" or config["series"]["param1"] == "Z":
                        coords_ion = (
                            ("series", np.array(config["series"]["vals1"])[:, 0]),
                            ("Wavelength", lamAxisI[0, :]),
                        )
                    else:
                        coords_ion = (("series", config["series"]["vals1"]), ("Wavelength", lamAxisI[0, :]))
                    ion_dat = {"Sim": ThryI}
                    ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
                else:
                    coords_ion = (("series", [0]), ("Wavelength", lamAxisI[0, :].squeeze()))
                    ion_dat = {"Sim": ThryI.squeeze(0)}
                    ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
                ion_data.to_netcdf(os.path.join(td, "binary", "ion_data.nc"))
            fig.savefig(os.path.join(td, "plots", "simulated_data"), bbox_inches="tight")
        mlflow.log_artifacts(td)
        metrics_dict = {"spectrum_calc_time": spectime}
        mlflow.log_metrics(metrics=metrics_dict)
