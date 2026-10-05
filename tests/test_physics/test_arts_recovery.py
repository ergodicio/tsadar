"""A bounded ARTS2D production inverse regression for the slow reference lane."""

import equinox as eqx
from jax import config, numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

config.update("jax_enable_x64", True)

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.inverse.loss_function import LossFunction
from tsadar.inverse import loops

from .test_production_recovery import _recovery_config

pytestmark = [pytest.mark.physics, pytest.mark.slow]


def _arts_config(*, refined=False):
    cfg = _recovery_config("EPW", refined=refined)
    cfg["parameters"]["electron"]["fe"] = {
        "active": False,
        "dim": 2,
        "type": "arbitrary",
        "nvx": 256 if refined else 128,
        "params": {"flm_type": "dlm", "init_m": 2.0, "learn_log": True},
    }
    cfg["other"].update(
        iawfilter=[False, 0, 0, 0],
        npts=64,
        CCDsize=[64, 3],
        lamrangE=[440.0, 515.0],
        n_beta=128 if refined else 64,
        ang_res_unit=1,
        lam_res_unit=1,
        resonance_quadrature={
            "enabled": True,
            "root_scan_panels": 512 if refined else 256,
            "integration_panels": 64 if refined else 32,
            "regular_order": 8,
            "root_order": 16,
            "max_roots": 4,
        },
    )
    cfg["other"]["extraoptions"]["spectype"] = "angular_full"
    cfg["other"]["detector_specs"]["widIRF"].update(
        spect_FWHM_ele=1.3, ang_FWHM_ele=1.0
    )
    cfg["data"].update(fit_EPWr=False, lineouts={"start": 0, "end": 3})
    cfg["data"]["fit_rng"].update(blue_min=439.0, blue_max=516.0)
    cfg["optimizer"].update(
        method="adam",
        param_method="adam",
        num_epochs=160,
        learning_rate_init=0.03,
        learning_rate_final=0.003,
        param_learning_rate=0.03,
        save_state=False,
        validate_active_leaves=True,
        angular_objective={
            "noise": {"model": "measured_variance"},
            "gain": {"mode": "none"},
        },
    )
    return cfg


def test_arts2d_production_recovers_density_and_temperature(
    monkeypatch, record_property
):
    """P-INV-03: ARTS2D detector quadrature + noise-aware loss + Optax recovery.

    A fixed Maxwellian 2-V EDF leaves only ne and Te active. Three observed angles
    on the blue EPW wing constrain those two parameters. Truth doubles velocity,
    sinogram, root-scan and integration resolution, retaining identical physical
    detector bins. No projection, quadrature, IRF, loss, or optimizer is mocked.
    Only external MLflow logging is suppressed for an offline regression.
    """
    for name in ("set_tags", "set_tag", "log_metrics"):
        monkeypatch.setattr(loops.mlflow, name, lambda *args, **kwargs: None)
    cfg = _arts_config()
    truth_cfg = _arts_config(refined=True)
    angles = np.array([45.0, 60.0, 75.0])
    geometry = {"sa": angles, "angAxis": angles, "weights": np.eye(3)}
    batch = {
        "e_data": jnp.ones((3, 64)),
        "i_data": jnp.zeros((1, 1)),
        "noise_e": jnp.zeros((3, 64)),
        "noise_i": jnp.zeros((1, 1)),
        "e_amps": jnp.ones((3, 1)),
        "i_amps": jnp.ones(1),
    }
    truth_params = ThomsonParams(truth_cfg["parameters"], num_params=1, batch=False)
    truth_diagnostic = ThomsonScatteringDiagnostic(truth_cfg, geometry)
    batch["e_data"] = eqx.filter_jit(truth_diagnostic)(truth_params, batch)[0]
    batch["e_variance"] = jnp.full_like(
        batch["e_data"], (0.01 * jnp.max(batch["e_data"])) ** 2
    )
    cfg["parameters"]["electron"]["ne"].update(val=0.16, active=True)
    cfg["parameters"]["electron"]["Te"].update(val=0.78, active=True)
    initial = ThomsonParams(cfg["parameters"], num_params=1, batch=False, activate=True)
    loss = LossFunction(cfg, geometry, batch)
    assert loss.ts_diag.model.electron_spectrum_is_detector_binned
    np.testing.assert_array_equal(
        loss.ts_diag.model.electron_detector_edges_nm,
        truth_diagnostic.model.electron_detector_edges_nm,
    )
    diff, static = eqx.partition(initial, get_filter_spec(cfg["parameters"], initial))
    assert ravel_pytree(diff)[0].size == 2, "P-INV-03: unexpected active leaves"
    initial_loss = float(loss._loss_(diff, static, batch)[0])
    fitted, final_loss, _, _, _ = loops.angular_multiple_optax(
        cfg, geometry, loss, batch
    )
    actual = fitted.get_unnormed_params()
    expected = truth_params.get_unnormed_params()
    record_property("initial_loss", initial_loss)
    record_property("final_loss", final_loss)
    for name in ("ne", "Te"):
        record_property(name, float(actual["electron"][name]))
        np.testing.assert_allclose(
            actual["electron"][name],
            expected["electron"][name],
            rtol=0.03,
            atol=0,
            err_msg=f"P-INV-03: failed ARTS2D recovery of {name}",
        )
    assert (
        final_loss < initial_loss * 0.01
    ), f"P-INV-03: loss {initial_loss} -> {final_loss}"
    prediction = eqx.filter_jit(loss.ts_diag)(fitted, batch)[0]
    relative_l2 = float(
        jnp.linalg.norm(prediction - batch["e_data"]) / jnp.linalg.norm(batch["e_data"])
    )
    record_property("detector_relative_l2", relative_l2)
    assert relative_l2 < 0.02, f"P-INV-03: detector relative L2={relative_l2:.3%}"
