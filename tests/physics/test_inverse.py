"""L4: recovery from oracle data (S1), Laplace against Cramer-Rao (S2), MCMC against Laplace (S3)."""

import dataclasses

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.inverse.loops import _1d_scipy_loop_
from tsadar.inverse.loss_function import LossFunction
from tsadar.inverse.postprocess import mcmc
from tsadar.inverse.postprocess.laplace import get_sigmas

from tsverify import oracle as oracles
from tsverify.cases import A1_CASES

from . import harness
from .compare import oracle_detector

pytestmark = pytest.mark.physics

EPW_CASE, _, IAW_CASE, _ = A1_CASES
ANGLES = {"sa": np.array([60.0]), "weights": np.ones((1, 1))}
COUNTS = 1.0e4


def _oracle_data(plasma, det):
    centers = np.linspace(*det.lam_range, det.pixels)
    return oracle_detector(det, centers, lambda lam: oracles.spectral_density(lam, plasma))


def _problem(case, free, starts=None, amplitude=COUNTS):
    """Config, loss and batch for fitting the oracle spectrum of ``case`` with ``free`` parameters active."""
    det = dataclasses.replace(case.detector, amplitude=amplitude)
    cfg = harness.diagnostic_config(case.plasma, det)
    cfg["optimizer"].update(method="l-bfgs-b", num_epochs=200, batch_size=1, y_norm=False, loss_method="l2")
    for (species, name), _ in free.items():
        start = (starts or {}).get((species, name))
        cfg["parameters"][species][name].update(active=True, **({"val": start} if start is not None else {}))
    _, _, batch = harness.build_diagnostic(case.plasma, det)
    channel = "e_data" if det.channel == "EPW" else "i_data"
    batch[channel] = jnp.asarray(_oracle_data(case.plasma, det))[None, :]
    return cfg, LossFunction(cfg, ANGLES, batch), batch


def _physical(params, species, name):
    return float(np.asarray(params.get_unnormed_params()[species][name]).ravel()[0])


S1_CASES = [
    pytest.param(
        EPW_CASE,
        {("electron", "ne"): 0.2, ("electron", "Te"): 0.6},
        {("electron", "ne"): 0.16, ("electron", "Te"): 0.78},
        id="epw-ne-Te",
    ),
    pytest.param(
        IAW_CASE,
        {("ion-1", "Va"): 1.5, ("ion-1", "Ti"): 0.08},
        {("ion-1", "Va"): -0.5, ("ion-1", "Ti"): 0.13},
        id="iaw-Va-Ti",
    ),
]


@pytest.mark.parametrize("case, truth, start", S1_CASES)
def test_noiseless_recovery_from_analytic_data(case, truth, start, record_property):
    cfg, loss, batch = _problem(case, truth, start, amplitude=100.0)
    final_loss, fitted = _1d_scipy_loop_(cfg, loss, None, batch)
    assert np.isfinite(final_loss)
    for (species, name), value in truth.items():
        got = _physical(fitted, species, name)
        record_property(f"{species}.{name}", got)
        np.testing.assert_allclose(
            got, value, rtol=S1_RTOL, atol=S1_ATOL.get(name, 0.0), err_msg=f"S1/{case}: {species}.{name}"
        )


S1_RTOL = 1.0e-3
S1_ATOL = {"Va": 1.0e-3}


EPW_FREE = (("electron", "ne"), ("electron", "Te"))


@pytest.fixture(scope="module")
def epw_at_truth():
    """The A1 EPW problem (ne, Te free) at the values the activated parameters hold, with tsadar's noiseless
    spectrum as data.
    """
    cfg, loss, batch = _problem(EPW_CASE, dict.fromkeys(EPW_FREE))
    params = ThomsonParams(cfg["parameters"], num_params=1, activate=True)
    batch["e_data"] = jnp.asarray(loss.ts_diag(params, batch)[0])
    loss = LossFunction(cfg, ANGLES, batch)
    truth = {name: _physical(params, *name) for name in EPW_FREE}
    diff, static = eqx.partition(params, get_filter_spec(cfg["parameters"], params))
    hess = loss.h_loss_wrt_params_per_lineout(diff, static, batch)
    fitted, _ = params.get_fitted_params(cfg["parameters"])
    columns = [(species, key) for species, keys in fitted.items() for key in keys if key not in ("fe", "f", "flm")]
    sigmas = get_sigmas(hess, diff, static, fitted, 1)[0]
    laplace = np.array([sigmas[columns.index(name)] for name in EPW_FREE])
    return {"cfg": cfg, "loss": loss, "batch": batch, "params": params, "truth": truth, "laplace": laplace}


def _cramer_rao_sigmas(case, truth, data, step=1.0e-3):
    """sqrt(diag(F^-1)), F = J^T diag(1/d) J over the fit-range pixels, J by central differences of the oracle."""
    det = dataclasses.replace(case.detector, amplitude=COUNTS)
    centers = np.linspace(*det.lam_range, det.pixels)
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        mask |= (centers > low) & (centers < high)
    plasma = dataclasses.replace(case.plasma, **{name: value for (_, name), value in truth.items()})
    columns = []
    for (species, name), value in truth.items():
        assert species == "electron", "the finite-difference Jacobian here perturbs electron parameters only"
        h = step * value
        up, down = (dataclasses.replace(plasma, **{name: value + s}) for s in (h, -h))
        columns.append((_oracle_data(up, det) - _oracle_data(down, det)) / (2 * h))
    jacobian = np.stack(columns, axis=1)[mask]
    fisher = jacobian.T @ (jacobian / np.abs(data[mask])[:, None])
    return np.sqrt(np.diag(np.linalg.inv(fisher)))


def test_laplace_uncertainty_equals_the_cramer_rao_bound(epw_at_truth, record_property):
    bound = _cramer_rao_sigmas(EPW_CASE, epw_at_truth["truth"], np.asarray(epw_at_truth["batch"]["e_data"])[0])
    laplace = epw_at_truth["laplace"]
    for (_, name), lap, cr in zip(EPW_FREE, laplace, bound):
        record_property(f"{name}: laplace / cramer-rao", float(lap / cr))
    np.testing.assert_allclose(laplace, bound, rtol=S2_RTOL, err_msg="S2: Laplace sigma != Cramer-Rao bound")


S2_RTOL = 0.01


def test_mcmc_posterior_matches_laplace_at_high_counts(epw_at_truth, record_property):
    cfg = dict(epw_at_truth["cfg"])
    cfg["other"] = dict(cfg["other"]) | {
        "mcmc": {"num_steps": 6000, "burn_in": 1000, "thin": 5, "adapt_every": 50, "use_laplace_seed": True, "seed": 0}
    }
    samples, static, diagnostics = mcmc.run_mcmc_for_batch(
        cfg, epw_at_truth["loss"], epw_at_truth["params"], epw_at_truth["batch"], jax.random.PRNGKey(7)
    )

    def physical(sample):
        unnormed = eqx.combine(static, sample).get_unnormed_params()
        return jnp.stack([jnp.ravel(unnormed[s][n])[0] for s, n in EPW_FREE])

    draws = np.asarray(eqx.filter_vmap(physical)(samples))
    truth = np.array([epw_at_truth["truth"][name] for name in EPW_FREE])
    laplace = epw_at_truth["laplace"]
    mean, std = draws.mean(axis=0), draws.std(axis=0)
    for i, (_, name) in enumerate(EPW_FREE):
        record_property(f"{name}: (mean - truth) / sigma", float((mean[i] - truth[i]) / laplace[i]))
        record_property(f"{name}: std / laplace", float(std[i] / laplace[i]))
    np.testing.assert_array_less(np.abs(mean - truth), S3_MEAN_SIGMAS * laplace, err_msg="S3: posterior mean off truth")
    np.testing.assert_allclose(std, laplace, rtol=S3_STD_RTOL, err_msg="S3: posterior width != Laplace sigma")
    acceptance = float(np.asarray(diagnostics["acceptance_rate"]).ravel()[0])
    record_property("acceptance rate", acceptance)
    assert abs(acceptance - 0.234) < 0.1, f"S3: acceptance rate {acceptance:.3f}, target 0.234"


S3_MEAN_SIGMAS = 0.2
S3_STD_RTOL = 0.10
