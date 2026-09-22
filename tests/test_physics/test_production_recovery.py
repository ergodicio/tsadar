"""Synthetic recovery through the real diagnostic, loss, activation, and optimizer."""

from pathlib import Path

import equinox as eqx
from jax import config, numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest
import yaml

config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.inverse.loss_function import LossFunction
from tsadar.inverse.loops import _1d_scipy_loop_

pytestmark = pytest.mark.physics


def _recovery_config(feature, *, refined=False):
    """Use the shipped deck, with a small deterministic physical reference case."""
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/1d-defaults.yaml").read_text()
    )
    p = cfg["parameters"]
    p["electron"]["fe"] = {
        "dim": 1,
        "type": "dlm",
        "active": False,
        "nvx": 512 if refined else 256,
        "params": {"m": {"val": 2.0, "lb": 2.0, "ub": 5.0, "matte": False}},
    }
    p["electron"]["Te"]["val"] = 0.6
    p["electron"]["ne"]["val"] = 0.2
    for key, value in {"A": 1.0, "Z": 1.0, "fract": 1.0, "Ti": 0.08, "Va": 1.5}.items():
        p["ion-1"][key]["val"] = value
    p["ion-1"]["Z"].update(lb=0.5, ub=2.0)
    # The central notch prevents an unresolved IAW from setting the EPW
    # peak normalization even when its wavelengths are excluded from the loss.
    cfg["other"].update(
        CCDsize=[128, 128],
        npts=2048 if refined else 1024,
        lamrangE=[440.0, 630.0],
        lamrangI=[524.5, 528.5],
        iawfilter=[True, 4, 24, 526.5],
    )
    cfg["data"].update(
        fit_EPWb=feature == "EPW",
        fit_EPWr=feature == "EPW",
        fit_IAW=feature == "IAW",
        load_ele_spec=feature == "EPW",
        load_ion_spec=feature == "IAW",
    )
    cfg["data"]["fit_rng"].update(
        blue_min=442.0,
        blue_max=522.0,
        red_min=531.0,
        red_max=628.0,
        iaw_min=524.6,
        iaw_max=528.4,
        iaw_cf_min=526.49,
        iaw_cf_max=526.51,
    )
    cfg["optimizer"].update(
        method="l-bfgs-b", num_epochs=80, batch_size=1, y_norm=False
    )
    return cfg


def _blank_batch():
    return {
        "e_data": jnp.ones((1, 128)),
        "i_data": jnp.ones((1, 128)),
        "noise_e": jnp.zeros((1, 128)),
        "noise_i": jnp.zeros((1, 128)),
        "e_amps": jnp.full(1, 100.0),
        "i_amps": jnp.full(1, 100.0),
    }


@pytest.mark.parametrize("feature", ["EPW", "IAW"])
def test_production_detector_recovers_known_parameters(feature, record_property):
    """P-INV-02: recover (ne,Te) or (Va,Ti) from displaced initial conditions.

    Synthetic truth uses twice the wavelength and velocity resolution of the fit.
    Both pass through the production IRF and pixel binning. Truth is interpolated
    to the fit's detector centers because the legacy point-grid binning yields
    slightly different centers at different resolutions. Only the named plasma
    parameters are free; all nuisance parameters and the EDF shape are fixed.
    """
    case_id = f"P-INV-02/{feature}"
    cfg = _recovery_config(feature)
    truth_cfg = _recovery_config(feature, refined=True)
    angles = {"sa": np.array([60.0]), "weights": np.ones((1, 1))}
    batch = _blank_batch()
    truth_params = ThomsonParams(truth_cfg["parameters"], num_params=1, activate=False)
    truth_output = ThomsonScatteringDiagnostic(truth_cfg, angles)(truth_params, batch)
    fit_diagnostic = ThomsonScatteringDiagnostic(cfg, angles)
    fit_output = fit_diagnostic(ThomsonParams(cfg["parameters"], num_params=1), batch)
    channel = 0 if feature == "EPW" else 1
    data_key = "e_data" if feature == "EPW" else "i_data"
    truth_axis = np.asarray(truth_output[channel + 2]).ravel()
    fit_axis = np.asarray(fit_output[channel + 2]).ravel()
    batch[data_key] = jnp.asarray(
        np.interp(fit_axis, truth_axis, np.asarray(truth_output[channel]).ravel())
    )[None, :]
    active = (
        {("electron", "ne"): 0.16, ("electron", "Te"): 0.78}
        if feature == "EPW"
        else {
            ("ion-1", "Va"): -0.5,
            ("ion-1", "Ti"): 0.13,
        }
    )
    for (species, name), start in active.items():
        cfg["parameters"][species][name].update(val=start, active=True)
    initial = ThomsonParams(cfg["parameters"], num_params=1, activate=True)
    loss = LossFunction(cfg, angles, batch)
    diff, static = eqx.partition(initial, get_filter_spec(cfg["parameters"], initial))
    assert (
        ravel_pytree(diff)[0].size == 2
    ), f"{case_id}: incorrect active parameter count"
    initial_loss = float(loss._loss_(diff, static, batch)[0])
    final_loss, fitted = _1d_scipy_loop_(cfg, loss, None, batch)
    record_property("initial_loss", initial_loss)
    record_property("final_loss", float(final_loss))
    assert np.isfinite(final_loss), f"{case_id}: nonfinite fitted objective"
    assert (
        final_loss < initial_loss * 0.01
    ), f"{case_id}: loss {initial_loss} -> {final_loss}"
    expected = truth_params.get_unnormed_params()
    actual = fitted.get_unnormed_params()
    for species, name in active:
        record_property(
            f"{species}.{name}", float(np.asarray(actual[species][name]).item())
        )
        np.testing.assert_allclose(
            actual[species][name],
            expected[species][name],
            rtol=0.03,
            atol=0.01 if name == "Va" else 0,
            err_msg=f"{case_id}: failed physical recovery of {species}.{name}",
        )
    predicted = np.asarray(loss.ts_diag(fitted, batch)[channel])
    target = np.asarray(batch[data_key])
    relative_l2 = np.linalg.norm(predicted - target) / np.linalg.norm(target)
    record_property("detector_relative_l2", float(relative_l2))
    assert relative_l2 < 0.02, f"{case_id}: detector relative L2={relative_l2:.3%}"
