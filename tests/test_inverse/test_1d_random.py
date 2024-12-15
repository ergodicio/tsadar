from jax import config

config.update("jax_enable_x64", True)

from scipy.optimize import minimize
from jax.flatten_util import ravel_pytree

from jax import numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import yaml, os, mlflow, tempfile, optax, tqdm
from flatten_dict import flatten, unflatten

from tsadar.utils import misc
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules import ThomsonParams, get_filter_spec
from tsadar.utils.data_handling.calibration import get_scattering_angles


def _perturb_params_(rng, params):
    """
    Perturbs the parameters for the forward pass.

    Args:
        params: Dictionary - Parameters to be perturbed

    Returns:
        Dictionary - Perturbed parameters

    """
    # for key in params["electron"].keys():
    #     new_val = (
    #         rng.uniform(0.4, 0.8) * (params["electron"][key]["ub"] - params["electron"][key]["lb"])
    #         + params["electron"][key]["lb"]
    #     )
    #     if key != "fe":
    #         params["electron"][key]["val"] = new_val
    #     else:
    #         params["electron"]["fe"]["params"]["m"]["val"] = new_val

    params["electron"]["fe"]["params"]["m"]["val"] = float(rng.uniform(2.0, 3.5))
    params["electron"]["Te"]["val"] = float(rng.uniform(0.5, 1.5))
    params["electron"]["ne"]["val"] = float(rng.uniform(0.1, 0.7))

    # for key in params["general"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    # for key in params["ion-1"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    return params


def test_1d_inverse():
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """

    mlflow.set_experiment("tsadar-tests")
    with mlflow.start_run(run_name="test_1d_inverse"):
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
        rng = np.random.default_rng()
        ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
        config["parameters"] = _perturb_params_(rng, config["parameters"])
        misc.log_mlflow(config)
        ts_params_gt = ThomsonParams(config["parameters"], num_params=1, batch=True)
        ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params_gt, dummy_batch)
        ground_truth = {"ThryE": ThryE, "lamAxisE": lamAxisE, "ThryI": ThryI, "lamAxisI": lamAxisI}

        ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
        config["parameters"] = _perturb_params_(rng, config["parameters"])
        ts_params_fit = ThomsonParams(config["parameters"], num_params=1, batch=True)
        diff_params, static_params = eqx.partition(
            ts_params_fit, filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=ts_params_fit)
        )

        def loss_fn(_diff_params):
            _all_params = eqx.combine(_diff_params, static_params)
            ThryE, ThryI, _, _ = ts_diag(_all_params, dummy_batch)
            return jnp.sum((ThryE - ground_truth["ThryE"]) ** 2)

        use_optax = False
        if use_optax:
            opt = optax.adam(3e-3)

            opt_state = opt.init(diff_params)
            for i in (pbar := tqdm.tqdm(range(1000))):
                loss, grad_loss = eqx.filter_value_and_grad(loss_fn)(diff_params)
                updates, opt_state = opt.update(grad_loss, opt_state)
                diff_params = eqx.apply_updates(diff_params, updates)
                pbar.set_description(f"Loss: {loss:.4f}")

        else:
            flattened_diff_params, unravel = ravel_pytree(diff_params)

            def _loss_fn(diff_params_flat):
                diff_params_pytree = unravel(diff_params_flat)
                loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_params_pytree)

                return float(loss), np.array(ravel_pytree(grads))

            res = minimize(eqx.filter_value_and_grad(_loss_fn), flattened_diff_params, method="L-BFGS-B", jac=True)

        learned_params = eqx.combine(diff_params, static_params).get_unnormed_params()
        misc.log_mlflow({"loss": loss} | learned_params, which="metrics")

        # np.save("tests/test_forward/ThryE-1d.npy", ThryE)
        # ground_truth = np.load("tests/test_forward/ThryE-1d.npy")
        ThryE, _, _, _ = ts_diag(eqx.combine(diff_params, static_params), dummy_batch)
        # misc.log_params(config)
        with tempfile.TemporaryDirectory() as td:
            fig, ax = plt.subplots(1, 1, figsize=(9, 4), tight_layout=True)
            ax.plot(np.squeeze(lamAxisE), np.squeeze(ThryE), label="Model")
            ax.plot(np.squeeze(lamAxisE), np.squeeze(ground_truth["ThryE"]), label="Ground Truth")
            ax.grid()
            ax.legend()
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (arb. units)")
            ax.set_title("Electron Spectrum")
            fig.savefig(os.path.join(td, "ThryE.png"), bbox_inches="tight")
            mlflow.log_artifacts(td)

        # np.testing.assert_allclose(ThryE, ground_truth, rtol=1e-4)


if __name__ == "__main__":
    test_1d_inverse()
