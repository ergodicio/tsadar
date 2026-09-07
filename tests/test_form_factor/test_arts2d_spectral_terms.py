"""Focused contracts for reusable ARTS2D wavelength-space spectral terms."""

import numpy as np

from jax import config, grad, jit, numpy as jnp

config.update("jax_enable_x64", True)

from tsadar.core.physics.form_factor import FormFactor


def _form_factor():
    return FormFactor(
        lambda_range=[516.0, 524.0],
        npts=7,
        lam_shift=0.0,
        scattering_angles={
            "sa": np.array([38.0, 61.0]),
            "weights": np.ones((1, 2)),
        },
        num_grad_points=2,
        ud_ang=17.0,
        va_ang={"ion-1": -23.0},
        calc_gain={"calc": False},
        n_beta=8,
    )


def _params(ud=0.13):
    vx = jnp.linspace(-7.0, 7.0, 33)
    grid_x, grid_y = jnp.meshgrid(vx, vx)
    fe = jnp.exp(
        -0.5 * ((grid_x - 0.2) / 0.85) ** 2
        - 0.5 * ((grid_y + 0.1) / 1.15) ** 2
    )
    fe /= jnp.sum(fe) * (vx[1] - vx[0]) ** 2
    return {
        "electron": {"ne": 0.24, "Te": 0.62, "fe": fe, "v": vx},
        "general": {
            "ne_gradient": 4.0,
            "Te_gradient": 6.0,
            "lam": 526.5,
            "ud": ud,
        },
        "ion-1": {
            "A": 1.0,
            "Z": 1.0,
            "Ti": 0.09,
            "fract": 1.0,
            "Va": 0.08,
        },
    }


def test_spectral_terms_reconstruct_calc_in_2D_and_support_arbitrary_nodes():
    ff = _form_factor()
    params = _params()
    sinogram = ff.prepare_2D_sinogram(params)

    numerator, epsilon = ff.calc_2D_spectral_terms(
        params,
        ff.lambda_axis_nm,
        sinogram=sinogram,
    )
    spectrum, wavelengths_cm = ff.calc_in_2D(params)

    assert numerator.shape == (ff.npts, ff.num_grad_points, 2)
    assert epsilon.shape == numerator.shape
    np.testing.assert_allclose(
        np.asarray(jnp.transpose(numerator / jnp.abs(epsilon) ** 2, (1, 0, 2))),
        np.asarray(spectrum),
        rtol=2e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        np.squeeze(np.asarray(wavelengths_cm)) * 1.0e7,
        np.asarray(ff.lambda_axis_nm),
        rtol=1e-15,
        atol=0,
    )

    arbitrary_nodes = jnp.array([517.25, 519.875, 523.1])
    single_angle_terms = ff.calc_2D_spectral_terms(
        params,
        arbitrary_nodes,
        sinogram=sinogram,
        scattering_angles=47.0,
    )
    assert single_angle_terms[0].shape == (3, ff.num_grad_points, 1)
    assert bool(jnp.all(jnp.isfinite(single_angle_terms[0])))
    assert bool(jnp.all(jnp.isfinite(single_angle_terms[1])))


def test_pointwise_nodes_match_common_grid_and_gradient_contract():
    ff = _form_factor()
    nodes = jnp.array([517.4, 520.2, 522.7])

    def reconstructed_spectrum(ud, pointwise):
        params = _params(ud)
        sinogram = ff.prepare_2D_sinogram(params)
        if pointwise:
            node_mesh = jnp.broadcast_to(
                nodes[:, None, None],
                (nodes.size, ff.num_grad_points, 2),
            )
            numerator, epsilon = ff.calc_2D_spectral_terms_at_points(
                params,
                node_mesh,
                sinogram=sinogram,
            )
        else:
            numerator, epsilon = ff.calc_2D_spectral_terms(
                params,
                nodes,
                sinogram=sinogram,
            )
        return jnp.sum(numerator / jnp.abs(epsilon) ** 2)

    common_value = reconstructed_spectrum(0.13, False)
    pointwise_value = reconstructed_spectrum(0.13, True)
    common_gradient = grad(reconstructed_spectrum, argnums=0)(0.13, False)
    pointwise_gradient = grad(reconstructed_spectrum, argnums=0)(0.13, True)

    assert bool(jnp.all(jnp.isfinite(jnp.asarray([common_value, common_gradient]))))
    np.testing.assert_allclose(
        np.asarray(pointwise_value), np.asarray(common_value), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(pointwise_gradient), np.asarray(common_gradient), rtol=0, atol=0
    )


def test_prepared_evaluator_builds_sinogram_only_once(monkeypatch):
    ff = _form_factor()
    params = _params()
    build_sinogram = ff._build_sinogram
    calls = 0

    def counted_build(vx, fe):
        nonlocal calls
        calls += 1
        return build_sinogram(vx, fe)

    monkeypatch.setattr(ff, "_build_sinogram", counted_build)
    evaluate = ff.prepare_2D_spectral_evaluator(params, scattering_angles=49.0)
    assert calls == 1

    compiled_evaluate = jit(evaluate)
    numerator, epsilon = compiled_evaluate(jnp.array([518.0, 520.0]))
    assert numerator.shape == epsilon.shape == (2, ff.num_grad_points, 1)
    evaluate(jnp.array([519.0, 521.0, 523.0]))
    assert calls == 1
