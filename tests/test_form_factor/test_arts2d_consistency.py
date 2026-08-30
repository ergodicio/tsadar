"""Physics regressions for the signed ARTS2D longitudinal susceptibility.

These tests deliberately exercise the small geometry and susceptibility kernels rather
than a full diagnostic deck.  That keeps them fast enough for CPU CI while pinning the
Fourier/Landau sign convention, the direction of the Radon coordinate, and the flow
frame independently of detector and fitting code.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import wofz

from jax import config

config.update("jax_enable_x64", True)

from jax import jacfwd, numpy as jnp

from tsadar.core.physics.form_factor import (
    FormFactor,
    _charge_weighted_flow,
    _electron_resonance,
    _principal_value_integral,
)


@pytest.fixture(scope="module")
def form_factor():
    """A small FormFactor shared by the kernel-level tests in this module."""
    return FormFactor(
        lambda_range=[500.0, 501.0],
        npts=4,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([60.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=0.0,
        va_ang=0.0,
        calc_gain={"calc": False},
        n_beta=4,
    )


def _normal_pdf(x, mean=0.0, variance=1.0):
    return jnp.exp(-0.5 * (x - mean) ** 2 / variance) / jnp.sqrt(
        2.0 * jnp.pi * variance
    )


def _isotropic_maxwellian_sinogram(vx, n_beta):
    """Exact angle-independent Radon table for a unit-variance 2-D Maxwellian."""
    projection = _normal_pdf(vx)
    derivative = jnp.gradient(projection, vx[1] - vx[0])
    shape = (n_beta, vx.size)
    return jnp.broadcast_to(projection, shape), jnp.broadcast_to(derivative, shape)


def _chi_values(form_factor, vx, sinogram, beta, xi, klde):
    values = form_factor.calc_chi_vals(vx, sinogram, (beta, xi, klde))
    return jnp.stack(tuple(jnp.squeeze(value) for value in values))


def _faddeeva_susceptibility(xi, klde):
    """Analytic Maxwellian chi for exp(-v^2/2)/sqrt(2*pi).

    The convention is ``exp(-i omega t)`` with the Landau contour: for positive
    phase velocity the imaginary susceptibility is positive.
    """
    zeta = np.asarray(xi) / np.sqrt(2.0)
    plasma_dispersion = 1j * np.sqrt(np.pi) * wofz(zeta)
    return (1.0 + zeta * plasma_dispersion) / klde**2


def test_signed_maxwellian_susceptibility_matches_faddeeva(form_factor):
    """Both signs of xi, including zero, match the analytic Landau response."""
    vx = jnp.linspace(-9.0, 9.0, 1025)
    sinogram = _isotropic_maxwellian_sinogram(vx, form_factor.n_beta)
    xis = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    klde = 0.73

    got = np.stack(
        [
            np.asarray(
                _chi_values(
                    form_factor,
                    vx,
                    sinogram,
                    jnp.asarray(0.731),
                    jnp.asarray(xi),
                    jnp.asarray(klde),
                )
            )
            for xi in xis
        ]
    )
    expected_chi = _faddeeva_susceptibility(xis, klde)

    np.testing.assert_allclose(
        got[:, 0], np.asarray(_normal_pdf(xis)), rtol=5e-4, atol=2e-6
    )
    np.testing.assert_allclose(got[:, 2], expected_chi.real, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(got[:, 1], expected_chi.imag, rtol=5e-4, atol=5e-4)

    # These parity checks make the signed-coordinate requirement explicit instead of
    # relying only on the pointwise analytic comparison above.
    np.testing.assert_allclose(got[:3, 0], got[:3:-1, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[:3, 2], got[:3:-1, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[:3, 1], -got[:3:-1, 1], rtol=1e-12, atol=1e-12)


def test_xi_zero_susceptibility_and_gradient_are_finite(form_factor):
    """A pole on the central grid node has the analytic value and tangent."""
    vx = jnp.linspace(-9.0, 9.0, 1025)
    sinogram = _isotropic_maxwellian_sinogram(vx, form_factor.n_beta)
    beta = jnp.asarray(np.pi / 7.0)
    klde = jnp.asarray(0.73)

    evaluate = lambda xi: _chi_values(form_factor, vx, sinogram, beta, xi, klde)
    value = evaluate(jnp.asarray(0.0))
    tangent = jacfwd(evaluate)(jnp.asarray(0.0))

    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(tangent)))
    expected_chi = _faddeeva_susceptibility(0.0, float(klde))
    step = 1e-6
    expected_chi_gradient = (
        _faddeeva_susceptibility(step, float(klde))
        - _faddeeva_susceptibility(-step, float(klde))
    ) / (2 * step)

    np.testing.assert_allclose(float(value[2]), expected_chi.real, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        float(tangent[2]), expected_chi_gradient.real, rtol=0, atol=5e-5
    )
    np.testing.assert_allclose(
        float(tangent[1]), expected_chi_gradient.imag, rtol=5e-4, atol=5e-4
    )


def test_exact_node_principal_value_tangent_is_dtype_stable():
    """The grid-scale knot tangent is a physics convention, not a dtype cutoff."""

    def evaluate(dtype):
        vx = jnp.linspace(-6.0, 6.0, 129, dtype=dtype)
        distribution = 0.7 * _normal_pdf(
            vx, mean=-0.35, variance=0.8
        ) + 0.3 * _normal_pdf(vx, mean=1.1, variance=0.45)
        derivative = jnp.gradient(distribution, vx[1] - vx[0])
        integral = lambda xi: _principal_value_integral(derivative, vx, xi)
        xi = jnp.asarray(0.0, dtype=dtype)
        return integral(xi), jacfwd(integral)(xi)

    value64, tangent64 = evaluate(jnp.float64)
    value32, tangent32 = evaluate(jnp.float32)

    assert bool(
        jnp.all(jnp.isfinite(jnp.asarray([value64, tangent64, value32, tangent32])))
    )
    np.testing.assert_allclose(float(value32), float(value64), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(float(tangent32), float(tangent64), rtol=2e-4, atol=2e-5)


def test_shipped_decks_use_per_ion_flow_schema():
    """Current-schema defaults, inputs, examples, and fixtures keep Va with each ion."""
    repository = Path(__file__).resolve().parents[2]
    deck_paths = [
        *repository.glob("configs/*/*.yaml"),
        *repository.glob("docs/source/examples/*/*.yaml"),
        repository / "tests/configs/arts1v_test_defaults.yaml",
        repository / "tests/configs/arts1v_test_inputs.yaml",
        repository / "tests/configs/arts2d_test_inputs.yaml",
        repository / "tests/configs/arts2v_test_defaults.yaml",
        repository / "tests/configs/detector_inputs.yaml",
    ]

    for deck_path in deck_paths:
        parameters = yaml.safe_load(deck_path.read_text())["parameters"]
        assert "Va" not in parameters["general"], deck_path
        for species, species_config in parameters.items():
            if species.startswith("ion-"):
                assert "Va" in species_config, f"{deck_path}: {species}"


def test_shifted_anisotropic_gaussian_radon_mean_variance_and_signs(form_factor):
    """`beta` is the conventional (cos beta, sin beta) projection direction."""
    vx = jnp.linspace(-9.0, 9.0, 129)
    grid_x, grid_y = jnp.meshgrid(vx, vx)
    mean = np.array([0.8, -0.55])
    sigma = np.array([0.7, 1.1])
    distribution = jnp.exp(
        -0.5 * ((grid_x - mean[0]) / sigma[0]) ** 2
        - 0.5 * ((grid_y - mean[1]) / sigma[1]) ** 2
    ) / (2.0 * jnp.pi * sigma[0] * sigma[1])

    # The four axes pin both coordinate signs; oblique directions also pin the
    # covariance projection rather than only testing image rotations by 90 degrees.
    for beta in (0.0, np.pi / 2, np.pi, -np.pi / 2, 0.7, -0.9):
        direction = np.array([np.cos(beta), np.sin(beta)])
        expected_mean = float(direction @ mean)
        expected_variance = float(np.sum((direction * sigma) ** 2))
        expected = _normal_pdf(vx, expected_mean, expected_variance)
        projected = form_factor.project(vx, distribution, jnp.asarray(beta))

        norm = jnp.trapezoid(projected, vx)
        measured_mean = jnp.trapezoid(vx * projected, vx) / norm
        measured_variance = (
            jnp.trapezoid((vx - measured_mean) ** 2 * projected, vx) / norm
        )

        # Cubic image interpolation has tiny (~1e-6) ringing in the exponentially
        # small tails; the moments below are the tighter physics assertions.
        np.testing.assert_allclose(
            np.asarray(projected), np.asarray(expected), rtol=2e-5, atol=7e-6
        )
        np.testing.assert_allclose(float(norm), 1.0, rtol=0, atol=5e-8)
        np.testing.assert_allclose(
            float(measured_mean), expected_mean, rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(
            float(measured_variance), expected_variance, rtol=0, atol=1e-7
        )


@pytest.mark.parametrize(
    "k, expected_beta",
    [
        ((2.0, 0.0), 0.0),
        ((0.0, 2.0), np.pi / 2),
        ((0.0, -2.0), -np.pi / 2),
    ],
)
def test_axis_aligned_k_at_xi_zero_has_finite_geometry_gradients(k, expected_beta):
    """`atan2` stays well behaved on either coordinate axis when k is nonzero."""
    k = jnp.asarray(k)
    electron_flow = (jnp.asarray(0.4), jnp.asarray(-0.2))
    omega = k[0] * electron_flow[0] + k[1] * electron_flow[1]  # choose xi == 0
    vte = jnp.asarray(1.3)

    def geometry(k_vector):
        return jnp.stack(
            _electron_resonance((k_vector[0], k_vector[1]), omega, electron_flow, vte)
        )

    value = geometry(k)
    jacobian = jacfwd(geometry)(k)

    np.testing.assert_allclose(float(value[0]), expected_beta, rtol=0, atol=1e-14)
    np.testing.assert_allclose(float(value[1]), 0.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(float(value[2]), 2.0, rtol=0, atol=1e-14)
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_projection_locked_to_k_and_parallel_drift_is_a_translation(form_factor):
    """Perpendicular flow is invisible; parallel flow shifts xi by exactly u/vTe."""
    k = (jnp.asarray(3.0), jnp.asarray(4.0))
    k_hat = (jnp.asarray(0.6), jnp.asarray(0.8))
    perpendicular = (jnp.asarray(-0.8), jnp.asarray(0.6))
    electron_flow = (jnp.asarray(0.4), jnp.asarray(-0.2))
    omega = jnp.asarray(3.7)
    vte = jnp.asarray(1.3)
    perpendicular_speed = jnp.asarray(2.1)
    parallel_speed = jnp.asarray(0.35)

    baseline = _electron_resonance(k, omega, electron_flow, vte)
    with_perpendicular = _electron_resonance(
        k,
        omega,
        tuple(
            electron_flow[i] + perpendicular_speed * perpendicular[i] for i in range(2)
        ),
        vte,
    )
    with_parallel = _electron_resonance(
        k,
        omega,
        tuple(electron_flow[i] + parallel_speed * k_hat[i] for i in range(2)),
        vte,
    )
    translated = _electron_resonance(
        k,
        omega + baseline[2] * parallel_speed,
        tuple(electron_flow[i] + parallel_speed * k_hat[i] for i in range(2)),
        vte,
    )

    np.testing.assert_allclose(
        np.asarray(with_perpendicular), np.asarray(baseline), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        float(with_parallel[0]), float(baseline[0]), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        float(with_parallel[1]),
        float(baseline[1] - parallel_speed / vte),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(translated), np.asarray(baseline), rtol=0, atol=1e-14
    )

    # Feed those resonance coordinates through an anisotropic EDF so this also pins
    # the projected susceptibility, not just the scalar geometry helper.
    vx = jnp.linspace(-9.0, 9.0, 129)
    grid_x, grid_y = jnp.meshgrid(vx, vx)
    distribution = jnp.exp(
        -0.5 * ((grid_x - 0.4) / 0.8) ** 2 - 0.5 * ((grid_y + 0.2) / 1.2) ** 2
    )
    distribution /= jnp.sum(distribution) * (vx[1] - vx[0]) ** 2
    sinogram = form_factor._build_sinogram(vx, distribution)

    susceptibility = lambda resonance: _chi_values(
        form_factor, vx, sinogram, resonance[0], resonance[1], jnp.asarray(0.8)
    )
    np.testing.assert_allclose(
        np.asarray(susceptibility(with_perpendicular)),
        np.asarray(susceptibility(baseline)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(susceptibility(translated)),
        np.asarray(susceptibility(baseline)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_charge_weighted_multispecies_frame_is_order_and_galilean_invariant():
    """1-D and 2-D bulk-flow definitions agree and do not depend on ion order."""
    charge = jnp.asarray([1.0, 6.0, 8.0])
    fraction = jnp.asarray([0.25, 0.35, 0.40])
    ion_flow = (
        jnp.asarray([-1.2, 0.7, 2.4]),
        jnp.asarray([0.5, -1.1, 0.3]),
    )
    weights = np.asarray(charge * fraction)
    expected = np.asarray(
        [np.average(np.asarray(component), weights=weights) for component in ion_flow]
    )

    bulk = _charge_weighted_flow(charge, fraction, ion_flow)
    scalar_bulk_x = _charge_weighted_flow(charge, fraction, ion_flow[0])
    np.testing.assert_allclose(
        np.asarray([component[0] for component in bulk]), expected, rtol=1e-14
    )
    np.testing.assert_allclose(float(scalar_bulk_x[0]), expected[0], rtol=1e-14)

    permutation = jnp.asarray([2, 0, 1])
    reordered = _charge_weighted_flow(
        charge[permutation],
        fraction[permutation],
        tuple(component[permutation] for component in ion_flow),
    )
    np.testing.assert_allclose(
        np.asarray([component[0] for component in reordered]), expected, rtol=1e-14
    )

    relative_drift = np.asarray([0.4, -0.25])
    electron_flow = tuple(bulk[i][0] + relative_drift[i] for i in range(2))
    boost = np.asarray([3.0, -4.0])
    boosted_ions = tuple(component + boost[i] for i, component in enumerate(ion_flow))
    boosted_bulk = _charge_weighted_flow(charge, fraction, boosted_ions)
    boosted_electron_flow = tuple(
        boosted_bulk[i][0] + relative_drift[i] for i in range(2)
    )

    k = (jnp.asarray(2.5), jnp.asarray(-1.2))
    omega = jnp.asarray(1.7)
    vte = jnp.asarray(2.3)
    baseline = _electron_resonance(k, omega, electron_flow, vte)
    boosted = _electron_resonance(
        k,
        omega + k[0] * boost[0] + k[1] * boost[1],
        boosted_electron_flow,
        vte,
    )
    np.testing.assert_allclose(
        np.asarray(boosted), np.asarray(baseline), rtol=1e-14, atol=1e-14
    )


def test_multispecies_2d_spectrum_is_invariant_to_species_order():
    """Per-ion Va and its static direction remain paired when runtime dict order changes."""
    vx = jnp.linspace(-8.0, 8.0, 65)
    grid_x, grid_y = jnp.meshgrid(vx, vx)
    distribution = jnp.exp(
        -0.5 * ((grid_x - 0.25) / 0.9) ** 2 - 0.5 * ((grid_y + 0.15) / 1.1) ** 2
    )
    distribution /= jnp.sum(distribution) * (vx[1] - vx[0]) ** 2

    ion_params = {
        "ion-1": {"A": 1.0, "Z": 1.0, "Ti": 0.08, "fract": 0.4, "Va": 0.30},
        "ion-2": {"A": 12.0, "Z": 6.0, "Ti": 0.16, "fract": 0.6, "Va": 0.18},
    }

    def params(order):
        result = {
            "electron": {"ne": 0.2, "Te": 0.55, "fe": distribution, "v": vx},
            "general": {
                "ne_gradient": 0.0,
                "Te_gradient": 0.0,
                "lam": 526.5,
                "ud": 0.12,
            },
        }
        result.update((species, ion_params[species]) for species in order)
        return result

    ff = FormFactor(
        lambda_range=[520.0, 533.0],
        npts=16,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([55.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=20.0,
        va_ang={"ion-1": 35.0, "ion-2": -50.0},
        calc_gain={"calc": False},
        n_beta=16,
    )
    forward, wavelengths = ff.calc_in_2D(params(("ion-1", "ion-2")))
    reversed_order, reversed_wavelengths = ff.calc_in_2D(params(("ion-2", "ion-1")))

    assert bool(jnp.all(jnp.isfinite(forward))) and bool(
        jnp.all(jnp.isfinite(reversed_order))
    )
    np.testing.assert_allclose(
        np.asarray(reversed_wavelengths), np.asarray(wavelengths), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(reversed_order), np.asarray(forward), rtol=1e-13, atol=1e-13
    )


def test_isotropic_zero_flow_1d_and_2d_spectra_agree():
    """Matched Maxwellians give the same spectrum in the two susceptibility paths.

    The remaining tolerance is set by two intentionally different discretizations:
    the 1-D path differentiates the sampled EDF along the wavelength-derived xi axis,
    while the 2-D path differentiates its finite velocity-grid projection.  On the
    modest CPU grid below their peak-normalized L-infinity difference is about 1.3%.
    """
    vx = jnp.linspace(-9.0, 9.0, 257)
    projection = _normal_pdf(vx)
    grid_x, grid_y = jnp.meshgrid(vx, vx)
    distribution = jnp.exp(-0.5 * (grid_x**2 + grid_y**2)) / (2.0 * jnp.pi)

    common = {
        "general": {"ne_gradient": 0.0, "Te_gradient": 0.0, "lam": 526.5, "ud": 0.0},
        "ion-1": {"A": 1.0, "Z": 1.0, "Ti": 0.1, "fract": 1.0, "Va": 0.0},
    }
    params_1d = {
        **common,
        "electron": {"ne": 0.2, "Te": 0.5, "fe": projection, "v": vx},
    }
    params_2d = {
        **common,
        "electron": {"ne": 0.2, "Te": 0.5, "fe": distribution, "v": vx},
    }

    spectrum_form_factor = FormFactor(
        lambda_range=[510.0, 540.0],
        npts=256,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([60.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=0.0,
        va_ang={"ion-1": 0.0},
        calc_gain={"calc": False},
        n_beta=32,
    )
    spectrum_1d, wavelengths_1d = spectrum_form_factor(params_1d)
    spectrum_2d, wavelengths_2d = spectrum_form_factor.calc_in_2D(params_2d)

    spectrum_1d = np.asarray(spectrum_1d)
    spectrum_2d = np.asarray(spectrum_2d)
    assert np.all(np.isfinite(spectrum_1d)) and np.all(np.isfinite(spectrum_2d))
    np.testing.assert_allclose(
        np.squeeze(np.asarray(wavelengths_2d)),
        np.squeeze(np.asarray(wavelengths_1d)),
        rtol=0,
        atol=0,
    )

    difference = np.abs(spectrum_2d - spectrum_1d)
    relative_linf = float(np.max(difference) / np.max(np.abs(spectrum_1d)))
    relative_l1 = float(np.sum(difference) / np.sum(np.abs(spectrum_1d)))
    assert relative_linf < 0.02, f"peak-normalized L-infinity error {relative_linf:.3%}"
    assert relative_l1 < 0.015, f"relative L1 error {relative_l1:.3%}"
