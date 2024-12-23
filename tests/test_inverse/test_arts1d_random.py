import pytest
from jax import config, block_until_ready

config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import yaml, os, mlflow, tempfile, optax, tqdm, time
from flatten_dict import flatten, unflatten

from tsadar.utils import misc
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules import ThomsonParams, get_filter_spec
from tsadar.utils.data_handling.calibration import get_scattering_angles, get_calibrations


def _perturb_params_(rng, params, arbitrary_distribution: bool = False):
    """
    Perturbs the parameters for the forward pass.

    Args:
        params: Dictionary - Parameters to be perturbed

    Returns:
        Dictionary - Perturbed parameters

    """

    params["electron"]["Te"]["val"] = float(rng.uniform(0.5, 1.5))
    params["electron"]["ne"]["val"] = float(rng.uniform(0.1, 0.7))

    params["general"]["amp1"]["val"] = float(rng.uniform(0.5, 2.5))
    params["general"]["amp2"]["val"] = float(rng.uniform(0.5, 2.5))
    params["general"]["lam"]["val"] = float(rng.uniform(523, 527))

    if arbitrary_distribution:
        params["electron"]["fe"]["params"]["init_m"] = float(rng.uniform(2.0, 3.5))
        params["electron"]["fe"]["type"] = "arbitrary"
    else:
        params["electron"]["fe"]["params"]["m"]["val"] = float(rng.uniform(2.0, 3.5))
        params["electron"]["fe"]["type"] = "dlm"

    # for key in params["general"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    # for key in params["ion-1"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    return params


@pytest.mark.parametrize("arbitrary_distribution", [True, False])
def test_arts1d_inverse(arbitrary_distribution: bool):
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """
    if "CPU_ONLY" in os.environ:
        if os.environ["CPU_ONLY"] == True:
            pytest.skip("Skipping GPU test on CPU-only")
    else:
        pytest.skip("Assuming CPU test - Skipping GPU test")

    _t0 = time.time()
    mlflow.set_experiment("tsadar-tests")
    with mlflow.start_run(run_name="test_arts1d_inverse") as run:
        with open("tests/configs/arts1d_test_defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("tests/configs/arts1d_test_inputs.yaml", "r") as fi:
            inputs = yaml.safe_load(fi)

        defaults = flatten(defaults)
        defaults.update(flatten(inputs))
        config = unflatten(defaults)

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(config, fi)
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

            sas["angAxis"], _, _, _, _, _ = get_calibrations(
                104000, config["other"]["extraoptions"]["spectype"], 0.0, config["other"]["CCDsize"]
            )  # shot number hardcoded to get calibration
            config["other"]["extraoptions"]["spectype"] = "angular_full"

            dummy_batch = {
                "i_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
                "e_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
                "noise_e": np.array([0]),
                "noise_i": np.array([0]),
                "e_amps": np.array([1]),
                "i_amps": np.array([1]),
            }
            rng = np.random.default_rng()
            ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
            config["parameters"] = _perturb_params_(rng, config["parameters"], arbitrary_distribution=False)
            misc.log_mlflow(config)
            ts_params_gt = ThomsonParams(config["parameters"], num_params=1, batch=False, activate=True)

            ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params_gt, dummy_batch)
            ground_truth = {"ThryE": ThryE, "lamAxisE": lamAxisE, "ThryI": ThryI, "lamAxisI": lamAxisI}

            def loss_fn(_diff_params, _static_params):
                _all_params = eqx.combine(_diff_params, _static_params)
                ThryE, ThryI, _, _ = ts_diag(_all_params, dummy_batch)
                return jnp.sum(jnp.mean(jnp.square(ThryE - ground_truth["ThryE"])))

            t0 = time.time()
            jit_vg = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
            diff_params, static_params = perturb_and_split_params(arbitrary_distribution, config, rng)
            temp_out = block_until_ready(jit_vg(diff_params, static_params))
            mlflow.log_metric(f"first run time", time.time() - t0)

            loss = 1
            while np.nan_to_num(loss, nan=1) > 5e-2:
                # ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
                diff_params, static_params = perturb_and_split_params(arbitrary_distribution, config, rng)

                use_optax = False
                if use_optax:
                    opt = optax.adam(0.001)
                    opt_state = opt.init(diff_params)
                    for i in (pbar := tqdm.tqdm(range(25))):
                        t0 = time.time()
                        loss, grad_loss = jit_vg(diff_params, static_params)
                        mlflow.log_metric(f"iteration time", time.time() - t0, step=i)

                        updates, opt_state = opt.update(grad_loss, opt_state)
                        diff_params = eqx.apply_updates(diff_params, updates)
                        pbar.set_description(f"Loss: {loss:.4f}")

                else:
                    flattened_diff_params, unravel = ravel_pytree(diff_params)

                    def scipy_vg_fn(diff_params_flat):
                        diff_params_pytree = unravel(diff_params_flat)
                        loss, grads = jit_vg(diff_params_pytree, static_params)
                        flattened_grads, _ = ravel_pytree(grads)

                        return float(loss), np.array(flattened_grads)

                    res = minimize(
                        scipy_vg_fn, flattened_diff_params, method="L-BFGS-B", jac=True, options={"disp": True}
                    )

                    diff_params = unravel(res["x"])
                    loss = res["fun"]

            params_to_log = {
                "gt": ts_params_gt.get_unnormed_params(),
                "learned": eqx.combine(diff_params, static_params).get_unnormed_params(),
            }

            misc.log_mlflow({"loss": loss} | params_to_log, which="metrics")
            ThryE, _, _, _ = ts_diag(eqx.combine(diff_params, static_params), dummy_batch)

            fig, ax = plt.subplots(1, 3, figsize=(11, 4), tight_layout=True)
            c = ax[0].contourf(np.squeeze(ThryE).T)
            fig.colorbar(c)
            c = ax[1].contourf(np.squeeze(ground_truth["ThryE"]).T)
            fig.colorbar(c)
            c = ax[2].contourf((np.squeeze(ground_truth["ThryE"]) - np.squeeze(ThryE)).T)
            fig.colorbar(c)

            ax[0].set_title("Model")
            ax[1].set_title("Ground Truth")
            ax[2].set_title("Model - Ground Truth")
            fig.savefig(os.path.join(td, "ThryE.png"), bbox_inches="tight")

            mlflow.log_metric("runtime-sec", time.time() - _t0)
            mlflow.log_artifacts(td)

    misc.export_run(run.info.run_id)
    # np.testing.assert_allclose(ThryE, ground_truth["ThryE"], atol=0.01, rtol=0)


def perturb_and_split_params(arbitrary_distribution, config, rng):
    config["parameters"] = _perturb_params_(rng, config["parameters"], arbitrary_distribution=arbitrary_distribution)
    ts_params_fit = ThomsonParams(
        config["parameters"],
        num_params=1,
        batch=False,
        activate=True,
    )
    diff_params, static_params = eqx.partition(
        ts_params_fit, filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=ts_params_fit)
    )

    return diff_params, static_params


if __name__ == "__main__":
    test_arts1d_inverse(arbitrary_distribution=False)
