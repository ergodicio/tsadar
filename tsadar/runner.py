import time, os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mlflow, tempfile, yaml, pandas
import multiprocessing as mp
import xarray as xr
from tqdm import tqdm
from flatten_dict import flatten, unflatten

from tsadar import fitter
from tsadar.distribution_functions.gen_num_dist_func import DistFunc
from tsadar.loss_function import LossFunction
from tsadar.fitter import init_param_norm_and_shift
from tsadar.misc import utils
from tsadar.plotting import plotters
from tsadar.data_handling.calibrations.calibration import get_calibrations, get_scattering_angles

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def load_and_make_folders(cfg_path: str) -> Tuple[str, Dict]:
    """
    This is used to queue runs on NERSC

    Args:
        cfg_path:

    Returns:

    """
    all_configs = {}
    basedir = os.path.join(os.getcwd(), f"{cfg_path}")
    for k in ["defaults", "inputs"]:
        with open(f"{os.path.join(basedir, k)}.yaml", "r") as fi:
            all_configs[k] = yaml.safe_load(fi)

    if "mlflow" in all_configs["inputs"].keys():
        experiment = all_configs["inputs"]["mlflow"]["experiment"]
        run_name = all_configs["inputs"]["mlflow"]["run"]

    else:
        experiment = all_configs["defaults"]["mlflow"]["experiment"]
        run_name = all_configs["defaults"]["mlflow"]["run"]

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

    return mlflow_run.info.run_id, all_configs


def run(cfg_path: str, mode: str) -> str:
    """
    Wrapper for lower level runner

    Args:
        cfg_path:
        mode:

    Returns:

    """
    run_id, all_configs = load_and_make_folders(cfg_path)
    defaults = flatten(all_configs["defaults"])
    defaults.update(flatten(all_configs["inputs"]))
    config = unflatten(defaults)
    with mlflow.start_run(run_id=run_id, log_system_metrics=True) as mlflow_run:
        _run_(config, mode=mode)

    return run_id


def run_for_app(run_id: str) -> str:
    with mlflow.start_run(run_id=run_id, log_system_metrics=True) as mlflow_run:
        # download config
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:

            dest_file_path = utils.download_file(f"config.yaml", mlflow_run.info.artifact_uri, temp_path)
            with open(dest_file_path, "r") as fi:
                config = yaml.safe_load(fi)

            if config["data"]["filenames"]["epw"] is not None:
                config["data"]["filenames"]["epw-local"] = utils.download_file(
                    config["data"]["filenames"]["epw"], mlflow_run.info.artifact_uri, temp_path
                )

            if config["data"]["filenames"]["iaw"] is not None:
                config["data"]["filenames"]["iaw-local"] = utils.download_file(
                    config["data"]["filenames"]["iaw"], mlflow_run.info.artifact_uri, temp_path
                )

            _run_(config, mode="fit")

    return mlflow_run.info.run_id


def _run_(config: Dict, mode: str = "fit"):
    """
    Either performs a forward pass or an entire fitting routine

    Relies on mlflow to log parameters, metrics, and artifacts

    Args:
        config:
        mode:

    Returns:

    """
    utils.log_params(config)
    t0 = time.time()
    if mode.casefold() == "fit":
        fit_results, loss = fitter.fit(config=config)
    elif mode == "forward" or mode == "series":
        calc_series(config=config)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")


def run_job(run_id: str, mode: str, nested: bool):
    """
    This is used to run queued runs on NERSC. It picks up the `run_id` and finds that using MLFlow and does the fitting


    Args:
        run_id:
        nested:

    Returns:

    """
    with mlflow.start_run(run_id=run_id, nested=nested) as run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            all_configs = {}
            for k in ["defaults", "inputs"]:
                dest_file_path = utils.download_file(f"{k}.yaml", run.info.artifact_uri, temp_path)
                with open(f"{os.path.join(temp_path, k)}.yaml", "r") as fi:
                    all_configs[k] = yaml.safe_load(fi)
            defaults = flatten(all_configs["defaults"])
            defaults.update(flatten(all_configs["inputs"]))
            config = unflatten(defaults)

        _run_(config, mode)


def calc_series(config):
    """
    Calculates a spectrum or series of spectra from the input deck, i.e. performs a forward pass or series of forward
     passes.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck. For series of spectra contains the special
         field 'series'. This field can have up to 8 subfields [param1, vals1, param2, vals2, param3, vals3, param4, vals4].
         the param subfields are a string identifying which fields of "parameters" are to be looped over. The vals subfields
         give the values of that subfield for each spectrum in the series.

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1
    # config["other"]["extraoptions"]["spectype"] = "1d"
    config["other"]["lamrangE"] = [
        config["data"]["fit_rng"]["forward_epw_start"],
        config["data"]["fit_rng"]["forward_epw_end"],
    ]
    config["other"]["lamrangI"] = [
        config["data"]["fit_rng"]["forward_iaw_start"],
        config["data"]["fit_rng"]["forward_iaw_end"],
    ]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

    electron_params = config["parameters"]["electron"]
    dist_obj = DistFunc(electron_params)
    electron_params["fe"]["velocity"], electron_params["fe"]["val"] = dist_obj(electron_params["m"]["val"])
    # electron_params["fe"]["val"] = np.log(electron_params["fe"]["val"])[None, :]

    config["units"] = init_param_norm_and_shift(config)

    sas = get_scattering_angles(config)
    dummy_batch = {
        "i_data": np.array([1]),
        "e_data": np.array([1]),
        "noise_e": np.array([0]),
        "noise_i": np.array([0]),
        "e_amps": np.array([1]),
        "i_amps": np.array([1]),
    }

    if config["other"]["extraoptions"]["spectype"] == "angular":
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

    t_start = time.time()
    for i in tqdm(range(serieslen), total=serieslen):
        if "series" in config.keys():

            config["parameters"]["species"][config["series"]["param1"]]["val"] = config["series"]["vals1"][i]
            if "param2" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param2"]]["val"] = config["series"]["vals2"][i]
            if "param3" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param3"]]["val"] = config["series"]["vals3"][i]
            if "param4" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param4"]]["val"] = config["series"]["vals4"][i]

    if config["other"]["extraoptions"]["spectype"] == "angular":
        [axisxE, _, _, _, _, _] = get_calibrations(
            104000, config["other"]["extraoptions"]["spectype"], config["other"]["CCDsize"]
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

    t_start = time.time()
    for i in range(serieslen):
        if "series" in config.keys():
            config["parameters"]["species"][config["series"]["param1"]]["val"] = config["series"]["vals1"][i]
            if "param2" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param2"]]["val"] = config["series"]["vals2"][i]
            if "param3" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param3"]]["val"] = config["series"]["vals3"][i]
            if "param4" in config["series"].keys():
                config["parameters"]["species"][config["series"]["param4"]]["val"] = config["series"]["vals4"][i]

        ts_fitter = LossFunction(config, sas, dummy_batch)
        params = ts_fitter.weights_to_params(ts_fitter.pytree_weights["active"])
        ThryE[i], ThryI[i], lamAxisE[i], lamAxisI[i] = ts_fitter.spec_calc(params, dummy_batch)

    spectime = time.time() - t_start
    ThryE = np.array(ThryE)
    ThryI = np.array(ThryI)
    lamAxisE = np.array(lamAxisE)
    lamAxisI = np.array(lamAxisI)

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        os.makedirs(os.path.join(td, "binary"), exist_ok=True)
        os.makedirs(os.path.join(td, "csv"), exist_ok=True)
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            savedata = plotters.plot_data_angular(
                config,
                {"ele": np.squeeze(ThryE)},
                {"e_data": np.zeros((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1]))},
                {"epw_x": sas["angAxis"], "epw_y": lamAxisE},
                td,
            )
            plotters.plot_dist(
                config,
                "electron",
                {"fe": np.squeeze(electron_params["fe"]["val"]), "v": electron_params["fe"]["velocity"]},
                np.zeros_like(electron_params["fe"]["val"]),
                td,
            )
            if len(np.shape(np.squeeze(electron_params["fe"]["val"]))) == 1:
                final_dist = pandas.DataFrame(
                    {
                        "fe": [l for l in electron_params["fe"]["val"]],
                        "vx": [vx for vx in electron_params["fe"]["velocity"]],
                    }
                )
            elif len(np.shape(np.squeeze(electron_params["fe"]["val"]))) == 2:
                final_dist = pandas.DataFrame(
                    data=np.squeeze(electron_params["fe"]["val"]),
                    columns=electron_params["fe"]["velocity"][0][0],
                    index=electron_params["fe"]["velocity"][0][:, 0],
                )
            final_dist.to_csv(os.path.join(td, "csv", "learned_dist.csv"))
        else:
            if electron_params["fe"]["dim"] == 2:
                plotters.plot_dist(
                    config,
                    "electron",
                    {"fe": electron_params["fe"]["val"], "v": electron_params["fe"]["velocity"]},
                    np.zeros_like(electron_params["fe"]["val"]),
                    td,
                )

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
