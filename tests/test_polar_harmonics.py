"""Contracts for the in-plane polar-harmonic ARTS2D EDF (cos m theta, sin m theta of the in-plane angle)."""

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import config, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.modules.distribution_functions.polar_harmonics import PolarHarmonics
from tsadar.core.modules.distribution_functions.spherical_harmonics import SphericalHarmonics
from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec


def _config(flm_type="arbitrary", modes=2, init_m=3.0, nvx=64):
    return {
        "active": True,
        "dim": 2,
        "type": "polarharmonic",
        "nvx": nvx,
        "params": {"M": modes, "flm_type": flm_type, "init_m": init_m, "m": {"lb": 1.8, "ub": 5.0}, "nvr": 64},
    }


@pytest.mark.parametrize("shape", [2.0, 4.0])
def test_isotropic_part_is_the_in_plane_marginal_of_the_3v_super_gaussian(shape):
    polar = PolarHarmonics(_config(init_m=shape))
    spherical_cfg = _config(init_m=shape) | {"type": "sphericalharmonic"}
    spherical_cfg["params"] = spherical_cfg["params"] | {"Nl": 1, "dtx": 0.0, "dty": 0.0}
    spherical = SphericalHarmonics(spherical_cfg)
    fp, fs = np.asarray(polar()), np.asarray(spherical())
    resolved = fs > 1e-8 * fs.max()
    assert np.max(np.abs(np.log(fp[resolved] / fs[resolved]))) < 2e-4
    v = np.asarray(polar.vx)
    vx, vy = np.meshgrid(v, v)
    dv = v[1] - v[0]
    assert np.sum(fp) * dv**2 == pytest.approx(1.0, rel=1e-12)
    assert np.sum(fp * (vx**2 + vy**2)) * dv**2 == pytest.approx(2.0, rel=1e-4)


def _with_mode(model, m, part, value):
    radial = model.modes[m][part]
    return eqx.tree_at(lambda tree: tree.modes[m][part].coeff, model, jnp.full_like(radial.coeff, value))


@pytest.mark.parametrize(
    "m, part, expect",
    [
        (1, "cos", "x_flow"),
        (1, "sin", "y_flow"),
        (2, "cos", "x_anisotropy"),
        (2, "sin", "tilt"),
    ],
)
def test_each_mode_produces_its_in_plane_moment(m, part, expect):
    # linear regime: exp(c cos theta) also has an O(c^2) cos 2 theta part
    model = _with_mode(PolarHarmonics(_config()), m, part, 1.0e-4)
    f = np.asarray(model())
    v = np.asarray(model.vx)
    vx, vy = np.meshgrid(v, v)
    dv2 = (v[1] - v[0]) ** 2
    ux, uy = np.sum(f * vx) * dv2, np.sum(f * vy) * dv2
    txx, tyy = np.sum(f * vx**2) * dv2, np.sum(f * vy**2) * dv2
    txy = np.sum(f * vx * vy) * dv2
    moments = {"x_flow": ux, "y_flow": uy, "x_anisotropy": txx - tyy, "tilt": txy}
    assert moments[expect] > 1e-6
    for name, value in moments.items():
        if name != expect:
            assert abs(value) < 1e-2 * moments[expect]


@pytest.mark.parametrize("flm_type", ["arbitrary", "nn"])
def test_modes_start_isotropic_with_order_one_gradient_and_are_in_the_filter(flm_type):
    parameters = {
        "electron": {
            "Te": {"active": False, "lb": 0.01, "ub": 1.5, "val": 0.6},
            "ne": {"active": False, "lb": 0.001, "ub": 1.0, "val": 0.2},
            "fe": _config(flm_type=flm_type),
        },
        "general": {name: {"active": False, "lb": 0.0, "ub": 10.0, "val": 1.0}
                    for name in ("lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud")},
        "ion-1": {"A": {"active": False, "val": 1.0}, "Ti": {"active": False, "lb": 0.001, "ub": 1.0, "val": 0.1},
                  "Va": {"active": False, "lb": -1.0, "ub": 1.0, "val": 0.0}, "Z": {"active": False, "lb": 0.5, "ub": 2.0, "val": 1.0},
                  "fract": {"active": False, "val": 1.0}},
    }
    params = ThomsonParams(parameters, num_params=1, batch=False, activate=False)
    edf = params.electron.distribution_functions()
    np.testing.assert_allclose(edf, jnp.flip(edf, axis=1), rtol=1e-10, atol=1e-14 * float(jnp.max(edf)))
    np.testing.assert_allclose(edf, jnp.flip(edf, axis=0), rtol=1e-10, atol=1e-14 * float(jnp.max(edf)))

    spec = get_filter_spec(parameters, params)
    diff, static = eqx.partition(params, spec)
    v = params.electron.distribution_functions.vx
    vx, vy = jnp.meshgrid(v, v)
    weight = vx + vy + vx * vy + (vx**2 - vy**2) + 0.01 * (vx**2 + vy**2) ** 2

    def observable(d):
        return jnp.sum(eqx.combine(d, static).electron.distribution_functions() * weight)

    grads = eqx.filter_grad(observable)(diff).electron.distribution_functions
    norm = lambda tree: float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree))))
    for m in (1, 2):
        for part in ("cos", "sin"):
            assert norm(grads.modes[m][part]) > 1.0
    assert grads.vr is None and grads.rho is None
    assert "f" in params.get_unnormed_params()["electron"]


def test_mode_m_vanishes_like_rho_to_the_m_at_the_origin():
    model = _with_mode(PolarHarmonics(_config(modes=3, nvx=65)), 3, "cos", 1.0)
    anisotropy = np.asarray(model.log_anisotropy())
    rho = np.asarray(model.rho)
    centre = 32
    assert rho[centre, centre] == 0.0 and anisotropy[centre, centre] == 0.0
    # within one cell of the origin the 3-fold term is bounded by rho^3 times its coefficient
    near = (rho > 0) & (rho < 0.3)
    coefficient = np.max(np.abs(np.asarray(model.relative_coefficients()[3]["cos"])))
    assert np.all(np.abs(anisotropy[near]) <= rho[near] ** 3 * coefficient)
    assert bool(np.all(np.isfinite(model()))) and bool(np.all(np.asarray(model()) > 0))


def test_mora_yahi_is_rejected_for_the_polar_model():
    with pytest.raises(NotImplementedError, match="arbitrary or nn"):
        PolarHarmonics(_config(flm_type="mora-yahi"))


def _parameters(fe):
    return {
        "electron": {"Te": {"active": False, "lb": 0.01, "ub": 1.5, "val": 0.6},
                     "ne": {"active": False, "lb": 0.001, "ub": 1.0, "val": 0.2}, "fe": fe},
        "general": {name: {"active": False, "lb": 0.0, "ub": 10.0, "val": 1.0}
                    for name in ("lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud")},
        "ion-1": {"A": {"active": False, "val": 1.0}, "Ti": {"active": False, "lb": 0.001, "ub": 1.0, "val": 0.1},
                  "Va": {"active": False, "lb": -1.0, "ub": 1.0, "val": 0.0},
                  "Z": {"active": False, "lb": 0.5, "ub": 2.0, "val": 1.0}, "fract": {"active": False, "val": 1.0}},
    }


@pytest.mark.parametrize("flm_type", ["arbitrary", "nn"])
def test_f0_is_the_single_trained_isotropic_part(flm_type):
    cfg = _config(flm_type=flm_type)
    model = PolarHarmonics(cfg)
    assert not hasattr(model, "normed_m")
    assert "m" not in model.get_unnormed_params()

    parameters = _parameters(cfg)
    params = ThomsonParams(parameters, num_params=1, batch=False, activate=False)
    spec = get_filter_spec(parameters, params).electron.distribution_functions
    assert any(leaf is True for leaf in jax.tree_util.tree_leaves(spec.f0))
    assert spec.f0.scale is False and spec.log_f_init is False
    # F0's radial function sits outside the bounded exponent and is scaled by 1 + v^2, not 1 + v^4
    np.testing.assert_allclose(np.asarray(model.f0.scale), 1.0 + np.asarray(model.vr) ** 2)

    if flm_type == "arbitrary":
        # a radially varying correction (a constant shift of log f is removed by the normalization)
        bumped = eqx.tree_at(lambda tree: tree.f0.coeff, model, 0.05 * model.vr**2)
        f = np.asarray(bumped())
        # F0 changes the radial shape and keeps the EDF isotropic
        np.testing.assert_allclose(f, np.flip(f, axis=1), rtol=1e-10)
        np.testing.assert_allclose(f, f.T, rtol=1e-10)
        assert not np.allclose(f, np.asarray(model()))
