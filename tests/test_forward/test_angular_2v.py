import pytest
from jax import config, devices

config.update("jax_enable_x64", True)

import yaml, mlflow, os, tempfile, time
import numpy as np
from equinox import filter_jit
import matplotlib.pyplot as plt
from flatten_dict import flatten, unflatten

from tsadar.utils import misc
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules.ts_params import ThomsonParams
from tsadar.utils.data_handling.calibration import get_scattering_angles, get_calibrations


def test_arts2d_forward_pass():
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """

    if not any(["gpu" == device.platform for device in devices()]):
        pytest.skip("Takes too long without a GPU")

    mlflow.set_experiment("tsadar-tests")
    with mlflow.start_run(run_name="test_arts2d_fwd") as run:
        with tempfile.TemporaryDirectory() as td:

            t0 = time.time()
            with open("tests/configs/arts2d_test_defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("tests/configs/arts2d_test_inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)

            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(config, fi)

            # get scattering angles and weights
            # config["other"]["lamrangE"] = [
            #     config["data"]["fit_rng"]["forward_epw_start"],
            #     config["data"]["fit_rng"]["forward_epw_end"],
            # ]
            # config["other"]["lamrangI"] = [
            #     config["data"]["fit_rng"]["forward_iaw_start"],
            #     config["data"]["fit_rng"]["forward_iaw_end"],
            # ]
            # config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
            # sas = get_scattering_angles(config)

            # [axisxE, _, _, _, _, _] = get_calibrations(
            #     104000, config["other"]["extraoptions"]["spectype"], 0.0, config["other"]["CCDsize"]
            # )  # shot number hardcoded to get calibration
            # config["other"]["extraoptions"]["spectype"] = "angular_full"

            # sas["angAxis"] = axisxE

            dummy_batch = {
                "i_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
                "e_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
                "noise_e": np.array([0]),
                "noise_i": np.array([0]),
                "e_amps": np.array([1]),
                "i_amps": np.array([1]),
            }

            ts_diag = ThomsonScatteringDiagnostic(config, angular=True, cumulative=False)
            ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
            ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params, dummy_batch)
            # np.save("tests/test_forward/ThryE-arts2v.npy", ThryE)

            ground_truth = np.load("tests/test_forward/ThryE-arts2v.npy")

            misc.log_mlflow(config)

            plot_fwd_vs_ground_truth(td, ts_params, ThryE, ground_truth)
            mlflow.log_artifacts(td)

        mlflow.log_metric("runtime-sec", time.time() - t0)

        np.testing.assert_allclose(ThryE, ground_truth, rtol=1e-4)


def plot_fwd_vs_ground_truth(td, ts_params, ThryE, ground_truth):
    # logging.info("Plotting model vs ground truth")

    fig, ax = plt.subplots(1, 3, figsize=(11, 4), tight_layout=True)
    c = ax[0].contourf(np.squeeze(ThryE).T, levels=np.linspace(0, 2.5, 26))
    fig.colorbar(c)
    c = ax[1].contourf(np.squeeze(ground_truth).T, levels=np.linspace(0, 2.5, 26))
    fig.colorbar(c)
    c = ax[2].contourf((np.squeeze(ground_truth) - np.squeeze(ThryE)).T, levels=np.linspace(0, 2.5, 26))
    fig.colorbar(c)

    ax[0].set_title("Model")
    ax[1].set_title("Ground Truth")
    ax[2].set_title("Model - Ground Truth")
    fig.savefig(os.path.join(td, "ThryE.png"), bbox_inches="tight")

    f00 = ts_params.electron.distribution_functions.get_f00()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
    ax[0].plot(ts_params.electron.distribution_functions.vr, f00)
    ax[0].grid()
    ax[1].semilogy(ts_params.electron.distribution_functions.get_f00())
    ax[1].grid()
    fig.savefig(os.path.join(td, "f00.png"), bbox_inches="tight")

    kwargs = {"m_f0": ts_params.electron.distribution_functions.get_unnormed_m(), "f00": f00}

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.plot(ts_params.electron.distribution_functions.vr, ts_params.electron.distribution_functions.flm[1][0](**kwargs))
    ax.grid()
    fig.savefig(os.path.join(td, "f10.png"), bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.plot(ts_params.electron.distribution_functions.vr, ts_params.electron.distribution_functions.flm[1][1](**kwargs))
    ax.grid()
    fig.savefig(os.path.join(td, "f11.png"), bbox_inches="tight")
    # np.testing.assert_allclose(ThryE, ground_truth["ThryE"], atol=0.01, rtol=0)
