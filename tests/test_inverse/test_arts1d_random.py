import pytest, os, shutil

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pytest, os, shutil

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config, block_until_ready, devices

config.update("jax_enable_x64", True)

from jax import numpy as jnp, value_and_grad, jit
from jax import numpy as jnp, value_and_grad, jit
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import equinox as eqx, numpy as np, xarray as xr
import equinox as eqx, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import yaml, os, mlflow, tempfile, optax, tqdm, time
from flatten_dict import flatten, unflatten

from tsadar.utils import misc
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.core.modules import ThomsonParams, get_filter_spec
from tsadar.utils.data_handling.calibration import get_scattering_angles, get_calibrations


def _dump_ts_params(td: str, ts_params: ThomsonParams, prefix: str = ""):
    os.makedirs(base_dir := os.path.join(td, "ts_params"), exist_ok=True)
    os.makedirs(params_dir := os.path.join(base_dir, prefix), exist_ok=True)
    os.makedirs(dist_dir := os.path.join(params_dir, "distribution"), exist_ok=True)

    # dump all parameters besides distribution
    unnormed_params = ts_params.get_unnormed_params()

    dist_xr = xr.DataArray(
        ts_params.electron.distribution_functions(),
        coords=(ts_params.electron.distribution_functions.vx,),
        dims=("vx",),
    )
    dist_xr.to_netcdf(os.path.join(dist_dir, "electron-dist.nc"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    dist_xr.plot(ax=ax[0])
    ax[0].grid()
    np.log10(dist_xr).plot(ax=ax[1])
    ax[1].grid()
    fig.savefig(os.path.join(dist_dir, "electron-dist.png"), bbox_inches="tight")
    plt.close()

    for param_key, these_params in unnormed_params.items():
        params_to_dump = {p_key: float(these_params[p_key]) for p_key in set(these_params.keys()) - {"f"}}
        with open(os.path.join(params_dir, f"{param_key}-params.yaml"), "w") as fi:
            yaml.dump(params_to_dump, fi)

    mlflow.log_artifacts(td)
    shutil.rmtree(base_dir)


def _dump_ts_params(td: str, ts_params: ThomsonParams, prefix: str = ""):
    os.makedirs(base_dir := os.path.join(td, "ts_params"), exist_ok=True)
    os.makedirs(params_dir := os.path.join(base_dir, prefix), exist_ok=True)
    os.makedirs(dist_dir := os.path.join(params_dir, "distribution"), exist_ok=True)

    # dump all parameters besides distribution
    unnormed_params = ts_params.get_unnormed_params()

    dist_xr = xr.DataArray(
        ts_params.electron.distribution_functions(),
        coords=(ts_params.electron.distribution_functions.vx,),
        dims=("vx",),
    )
    dist_xr.to_netcdf(os.path.join(dist_dir, "electron-dist.nc"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    dist_xr.plot(ax=ax[0])
    ax[0].grid()
    np.log10(dist_xr).plot(ax=ax[1])
    ax[1].grid()
    fig.savefig(os.path.join(dist_dir, "electron-dist.png"), bbox_inches="tight")
    plt.close()

    for param_key, these_params in unnormed_params.items():
        params_to_dump = {p_key: float(these_params[p_key]) for p_key in set(these_params.keys()) - {"f"}}
        with open(os.path.join(params_dir, f"{param_key}-params.yaml"), "w") as fi:
            yaml.dump(params_to_dump, fi)

    mlflow.log_artifacts(td)
    shutil.rmtree(base_dir)


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


@pytest.mark.parametrize(
    "arbitrary_distribution",
    [
        False,
    ],
)
def test_arts1d_inverse(arbitrary_distribution: bool):
    """
    Runs a forward pass with the Thomson scattering diagnostic and ThomsonParams classes. Saves the results to mlflow.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """
    if not any(["gpu" == device.platform for device in devices()]):
        pytest.skip("Takes too long without a GPU")

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

            active_gt_params, _ = ts_params_gt.get_fitted_params(config["parameters"])
            _dump_ts_params(td, ts_params_gt, prefix="ground_truth")
            ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params_gt, dummy_batch)


            ground_truth = {"ThryE": ThryE, "lamAxisE": lamAxisE, "ThryI": ThryI, "lamAxisI": lamAxisI}

            def loss_fn(_diff_params, _static_params):
                _all_params = eqx.combine(_diff_params, _static_params)
                ThryE, ThryI, _, _ = ts_diag(_all_params, dummy_batch)
                return jnp.mean(jnp.square(ThryE - ground_truth["ThryE"]))

            jit_vg = eqx.filter_jit(value_and_grad(loss_fn))
            # dump ground truth to disk
            loss = 1
            while np.nan_to_num(loss, nan=1) > 5e-2:
                # ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
                diff_params, static_params = perturb_and_split_params(arbitrary_distribution, config, rng)
                use_optax = True
                use_optax = False
                if use_optax:

                    opt = optax.adam(1e-2)  # if arbitrary_distribution else 1e-2)

                    opt = optax.adam(1e-2)  # if arbitrary_distribution else 1e-2)
                    opt_state = opt.init(diff_params)
                    for i in (pbar := tqdm.tqdm(range(1000))):
                    for i in (pbar := tqdm.tqdm(range(1000))):
                        t0 = time.time()
                        loss, grad_loss = jit_vg(diff_params, static_params)
                        mlflow.log_metrics({f"iteration time": time.time() - t0, "loss": float(loss)}, step=i)
                        mlflow.log_metrics({f"iteration time": time.time() - t0, "loss": float(loss)}, step=i)
                        updates, opt_state = opt.update(grad_loss, opt_state)
                        diff_params = eqx.apply_updates(diff_params, updates)
                        pbar.set_description(f"Loss: {loss:.2e}")

                        combined_params = eqx.combine(diff_params, static_params)
                        active_params, _ = combined_params.get_fitted_params(config["parameters"])
                        params_to_log = {"gt": active_gt_params, "learned": active_params}
                        misc.log_mlflow(params_to_log, which="metrics", step=i)

                        # plot f
                        if i % 5 == 0:
                            _dump_ts_params(td, combined_params, prefix=f"step-{i:3d}")
                        pbar.set_description(f"Loss: {loss:.2e}")

                        combined_params = eqx.combine(diff_params, static_params)
                        active_params, _ = combined_params.get_fitted_params(config["parameters"])
                        params_to_log = {"gt": active_gt_params, "learned": active_params}
                        misc.log_mlflow(params_to_log, which="metrics", step=i)

                        # plot f
                        if i % 5 == 0:
                            _dump_ts_params(td, combined_params, prefix=f"step-{i:3d}")

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
                    mlflow.log_metric("loss", loss, step=0)
                    combined_params = eqx.combine(diff_params, static_params)
                    active_params, _ = combined_params.get_fitted_params(config["parameters"])
                    params_to_log = {"gt": active_gt_params, "learned": active_params}
                    mlflow.log_metric("loss", loss, step=0)
                    combined_params = eqx.combine(diff_params, static_params)
                    active_params, _ = combined_params.get_fitted_params(config["parameters"])
                    params_to_log = {"gt": active_gt_params, "learned": active_params}

                    misc.log_mlflow(params_to_log, which="metrics")
                    misc.log_mlflow(params_to_log, which="metrics")

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


def save_electron_distribution_plot(td, i, combined_params):
    f = combined_params.electron.distribution_functions()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.plot(combined_params.electron.distribution_functions.vx, f)
    ax.grid()
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("$f(v_x)$")
    fig.savefig(os.path.join(td, f"evdf-step-{i}.png"), bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifacts(td)


def save_electron_distribution_plot(td, i, combined_params):
    f = combined_params.electron.distribution_functions()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax.plot(combined_params.electron.distribution_functions.vx, f)
    ax.grid()
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("$f(v_x)$")
    fig.savefig(os.path.join(td, f"evdf-step-{i}.png"), bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifacts(td)

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
    test_arts1d_inverse(arbitrary_distribution=True)
