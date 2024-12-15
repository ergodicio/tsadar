from jax import config

config.update("jax_enable_x64", True)

import yaml, mlflow, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from flatten_dict import flatten, unflatten

from tsadar.utils import misc

from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules import ThomsonParams
from tsadar.utils.data_handling.calibration import get_scattering_angles, get_calibrations


def test_arts1d_forward_pass():
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """

    mlflow.set_experiment("tsadar-tests")
    with mlflow.start_run(run_name="test_arts1d_fwd"):
        with open("tests/configs/arts1d_test_defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("tests/configs/arts1d_test_inputs.yaml", "r") as fi:
            inputs = yaml.safe_load(fi)

        defaults = flatten(defaults)
        defaults.update(flatten(inputs))
        config = unflatten(defaults)

        # get scattering angles and weights
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

        [axisxE, _, _, _, _, _] = get_calibrations(
            104000, config["other"]["extraoptions"]["spectype"], 0.0, config["other"]["CCDsize"]
        )  # shot number hardcoded to get calibration
        config["other"]["extraoptions"]["spectype"] = "angular_full"

        sas["angAxis"] = axisxE

        dummy_batch = {
            "i_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
            "e_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
            "noise_e": np.array([0]),
            "noise_i": np.array([0]),
            "e_amps": np.array([1]),
            "i_amps": np.array([1]),
        }

        ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
        ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params, dummy_batch)
        # np.save("tests/test_forward/ThryE-arts1d.npy", ThryE)

        ground_truth = np.load("tests/test_forward/ThryE-arts1d.npy")

        misc.log_mlflow(config)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
        c = ax[0].contourf(ThryE.T)
        ax[0].set_title("Forward Model")
        fig.colorbar(c)

        c = ax[1].contourf(ground_truth.T)
        ax[1].set_title("Ground Truth")
        fig.colorbar(c)
        with tempfile.TemporaryDirectory() as td:
            fig.savefig(os.path.join(td, "ThryE.png"), bbox_inches="tight")
            mlflow.log_artifacts(td)

        np.testing.assert_allclose(ThryE, ground_truth, rtol=1e-4)


if __name__ == "__main__":
    test_arts1d_forward_pass()
