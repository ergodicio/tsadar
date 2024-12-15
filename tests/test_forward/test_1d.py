from jax import config

config.update("jax_enable_x64", True)


import numpy as np
import matplotlib.pyplot as plt
import yaml, os, mlflow, tempfile
from flatten_dict import flatten, unflatten

from tsadar.utils import misc
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules import ThomsonParams
from tsadar.utils.data_handling.calibration import get_scattering_angles


def test_1d_forward_pass():
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """

    mlflow.set_experiment("tsadar-tests")
    with mlflow.start_run(run_name="test_1d_fwd"):
        with open("configs/1d-new/defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("configs/1d-new/inputs.yaml", "r") as fi:
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

        dummy_batch = {
            "i_data": np.array([1]),
            "e_data": np.array([1]),
            "noise_e": np.array([0]),
            "noise_i": np.array([0]),
            "e_amps": np.array([1]),
            "i_amps": np.array([1]),
        }

        ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
        ts_params = ThomsonParams(config["parameters"], num_params=1, batch=True)
        ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params, dummy_batch)

        # np.save("tests/test_forward/ThryE-1d.npy", ThryE)
        ground_truth = np.load("tests/test_forward/ThryE-1d.npy")

        misc.log_mlflow(config)
        with tempfile.TemporaryDirectory() as td:
            fig, ax = plt.subplots(1, 1, figsize=(9, 4), tight_layout=True)
            ax.plot(np.squeeze(lamAxisE), np.squeeze(ThryE), label="Model")
            ax.plot(np.squeeze(lamAxisE), np.squeeze(ground_truth), label="Ground Truth")
            ax.grid()
            ax.legend()
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (arb. units)")
            ax.set_title("Electron Spectrum")
            fig.savefig(os.path.join(td, "ThryE.png"), bbox_inches="tight")
            mlflow.log_artifacts(td)

        np.testing.assert_allclose(ThryE, ground_truth, rtol=1e-4)


if __name__ == "__main__":
    test_1d_forward_pass()
