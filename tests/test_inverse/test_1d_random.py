from jax import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
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

    params["electron"]["fe"]["params"]["m"]["val"] = float(rng.uniform(2.0, 3.5))
    params["electron"]["Te"]["val"] = float(rng.uniform(0.5, 1.5))
    params["electron"]["ne"]["val"] = float(rng.uniform(0.1, 0.7))

    params["general"]["amp1"]["val"] = float(rng.uniform(0.5, 2.5))
    params["general"]["amp2"]["val"] = float(rng.uniform(0.5, 2.5))
    params["general"]["lam"]["val"] = float(rng.uniform(523, 527))

    # for key in params["general"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    # for key in params["ion-1"].keys():
    #     params[key]["val"] *= rng.uniform(0.75, 1.25)

    return params


def _floatify(_params_, prefix="gt"):
    flattened_params = flatten(_params_)
    new_params = {}
    for key in flattened_params.keys():
        new_params[(prefix,) + key] = float(flattened_params[key][0])
    return unflatten(new_params)


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
        with open("tests/configs/1d-defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("tests/configs/1d-inputs.yaml", "r") as fi:
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
        ts_params_gt = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)
        ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params_gt, dummy_batch)
        ground_truth = {"ThryE": ThryE, "lamAxisE": lamAxisE, "ThryI": ThryI, "lamAxisI": lamAxisI}

        loss = 1
        while np.nan_to_num(loss, nan=1) > 1e-3:
            ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
            config["parameters"] = _perturb_params_(rng, config["parameters"])
            ts_params_fit = ThomsonParams(config["parameters"], num_params=1, batch=True, activate=True)
            diff_params, static_params = eqx.partition(
                ts_params_fit, filter_spec=get_filter_spec(cfg_params=config["parameters"], ts_params=ts_params_fit)
            )

            def loss_fn(_diff_params):
                _all_params = eqx.combine(_diff_params, static_params)
                ThryE, ThryI, _, _ = ts_diag(_all_params, dummy_batch)
                return jnp.mean(jnp.square(ThryE - ground_truth["ThryE"]))

            use_optax = False
            if use_optax:
                opt = optax.adam(0.004)

                opt_state = opt.init(diff_params)
                for i in (pbar := tqdm.tqdm(range(1000))):
                    loss, grad_loss = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(diff_params)
                    updates, opt_state = opt.update(grad_loss, opt_state)
                    diff_params = eqx.apply_updates(diff_params, updates)
                    pbar.set_description(f"Loss: {loss:.4f}")

            else:
                flattened_diff_params, unravel = ravel_pytree(diff_params)

                def scipy_vg_fn(diff_params_flat):
                    diff_params_pytree = unravel(diff_params_flat)
                    loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(diff_params_pytree)
                    flattened_grads, _ = ravel_pytree(grads)

                    return float(loss), np.array(flattened_grads)

                res = minimize(scipy_vg_fn, flattened_diff_params, method="L-BFGS-B", jac=True, options={"disp": True})

                diff_params = unravel(res["x"])
                loss = res["fun"]

        gt_params = _floatify(ts_params_gt.get_unnormed_params(), prefix="gt")
        learned_params = _floatify(eqx.combine(diff_params, static_params).get_unnormed_params(), prefix="learned")
        misc.log_mlflow({"loss": loss} | learned_params | gt_params, which="metrics")
        ThryE, _, _, _ = ts_diag(eqx.combine(diff_params, static_params), dummy_batch)

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

        # np.testing.assert_allclose(ThryE, ground_truth["ThryE"], atol=0, rtol=0.2)

        gt_flat = flatten(gt_params)
        learned_flat = flatten(learned_params)

        for key in gt_flat.keys():
            np.testing.assert_allclose(gt_flat[key], learned_flat[("learned",) + key[1:]], atol=0, rtol=0.1)


if __name__ == "__main__":
    test_1d_inverse()
