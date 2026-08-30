"""Focused tests for unresolved-resonance detector quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr as scipy_ndtr

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from tsadar.core.physics.resonance_quadrature import (
    gaussian_bin_probabilities,
    integrate_detector_bins,
    raise_for_diagnostics,
)


ROOT_NM = 474.199678891
HWHM_NM = 3.49399e-4
IRF_SIGMA_NM = 0.035
DETECTOR_EDGES_NM = np.array([473.72, 474.02, 474.15, 474.225, 474.41, 474.76])
SOURCE_BOUNDS_NM = np.array(
    [DETECTOR_EDGES_NM[0] - 6 * IRF_SIGMA_NM, DETECTOR_EDGES_NM[-1] + 6 * IRF_SIGMA_NM]
)


def _lorentz_terms(wavelengths_nm, root_nm, half_width_nm=HWHM_NM):
    numerator = jnp.full_like(wavelengths_nm, half_width_nm / jnp.pi)
    epsilon = wavelengths_nm - root_nm + 1j * half_width_nm
    return numerator, epsilon


def _cauchy_detector_reference(root_nm, half_width_nm=HWHM_NM):
    """High-accuracy reference using the exact Cauchy tan transformation."""

    t_lower = np.arctan((SOURCE_BOUNDS_NM[0] - root_nm) / half_width_nm)
    t_upper = np.arctan((SOURCE_BOUNDS_NM[1] - root_nm) / half_width_nm)
    integrals = []
    for edge_lower, edge_upper in zip(DETECTOR_EDGES_NM[:-1], DETECTOR_EDGES_NM[1:]):

        def transformed_integrand(t):
            wavelength = root_nm + half_width_nm * np.tan(t)
            probability = scipy_ndtr((edge_upper - wavelength) / IRF_SIGMA_NM) - scipy_ndtr(
                (edge_lower - wavelength) / IRF_SIGMA_NM
            )
            # (gamma / pi) / ((lambda-root)^2 + gamma^2) * d lambda / dt = 1 / pi.
            return probability / np.pi

        integral = quad(
            transformed_integrand,
            t_lower,
            t_upper,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=300,
        )[0]
        integrals.append(integral)
    return np.asarray(integrals) / np.diff(DETECTOR_EDGES_NM)


def _unresolved_result(root_nm, *, phase=0.0, panels=256):
    return integrate_detector_bins(
        lambda wavelengths: _lorentz_terms(wavelengths, root_nm),
        jnp.asarray(DETECTOR_EDGES_NM),
        IRF_SIGMA_NM,
        source_bounds_nm=jnp.asarray(SOURCE_BOUNDS_NM),
        root_scan_panels=1024,
        integration_panels=panels,
        regular_order=8,
        root_order=32,
        max_roots=4,
        neighbor_panels=1,
        bisection_iterations=48,
        scan_phase=phase,
    )


def _relative_detector_l1(actual, expected):
    bin_widths = np.diff(DETECTOR_EDGES_NM)
    return np.sum(np.abs(actual - expected) * bin_widths) / np.sum(np.abs(expected) * bin_widths)


def test_gaussian_bin_probabilities_are_exact_cdf_differences():
    wavelengths = np.array([473.9, 474.2, 474.7])
    actual = gaussian_bin_probabilities(
        jnp.asarray(wavelengths), jnp.asarray(DETECTOR_EDGES_NM), IRF_SIGMA_NM
    )
    expected = scipy_ndtr(
        (DETECTOR_EDGES_NM[None, 1:] - wavelengths[:, None]) / IRF_SIGMA_NM
    ) - scipy_ndtr((DETECTOR_EDGES_NM[None, :-1] - wavelengths[:, None]) / IRF_SIGMA_NM)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-14, atol=2e-14)


def test_bin_means_preserve_area_for_nonuniform_detector_edges():
    edges = jnp.asarray([0.0, 0.17, 0.83, 1.9, 2.4])

    result = integrate_detector_bins(
        lambda wavelengths: (jnp.ones_like(wavelengths), jnp.ones_like(wavelengths)),
        edges,
        0.04,
        root_scan_panels=96,
        integration_panels=96,
    )

    raise_for_diagnostics(result)
    # A constant source convolved with a normalized Gaussian remains constant;
    # the default six-sigma truncation loses less than four parts in 1e9.
    np.testing.assert_allclose(np.asarray(result.bin_mean), 1.0, rtol=0, atol=4e-9)
    area = jnp.sum(result.bin_mean * jnp.diff(edges))
    np.testing.assert_allclose(np.asarray(area), 2.4, rtol=0, atol=5e-9)


def test_unresolved_root_is_phase_and_refinement_stable():
    reference = _cauchy_detector_reference(ROOT_NM)
    phases = jnp.arange(8, dtype=jnp.float64) / 8

    def evaluate(phase):
        result = _unresolved_result(ROOT_NM, phase=phase, panels=256)
        return result.bin_mean, result.diagnostics

    values, diagnostics = jax.jit(jax.vmap(evaluate))(phases)
    values = np.asarray(values)

    assert np.all(np.asarray(diagnostics.root_count) == 1)
    assert not np.any(np.asarray(diagnostics.root_overflow))
    assert not np.any(np.asarray(diagnostics.nonfinite))
    assert not np.any(np.asarray(diagnostics.zero_width))
    np.testing.assert_allclose(
        np.asarray(diagnostics.roots_nm)[:, 0], ROOT_NM, rtol=0, atol=3e-12
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.resonance_centers_nm)[:, 0], ROOT_NM, rtol=0, atol=3e-12
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.resonance_half_widths_nm)[:, 0], HWHM_NM, rtol=3e-10, atol=2e-13
    )

    errors = np.asarray([_relative_detector_l1(value, reference) for value in values])
    assert np.max(errors) < 1.0e-3
    assert np.max(np.ptp(values, axis=0) * np.diff(DETECTOR_EDGES_NM)) < 2.0e-5

    coarse = np.asarray(_unresolved_result(ROOT_NM, phase=0.375, panels=128).bin_mean)
    fine = np.asarray(_unresolved_result(ROOT_NM, phase=0.375, panels=256).bin_mean)
    assert _relative_detector_l1(coarse, fine) < 5.0e-4


def test_implicit_root_and_integral_gradients_match_reference():
    bin_weights = jnp.asarray([0.2, -0.4, 0.7, -0.1, 0.3]) * jnp.asarray(
        np.diff(DETECTOR_EDGES_NM)
    )

    def loss(root_nm, phase):
        result = _unresolved_result(root_nm, phase=phase, panels=256)
        return jnp.sum(result.bin_mean * bin_weights)

    phases = jnp.asarray([0.0, 0.375, 0.75])
    gradients = np.asarray(jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0)))(ROOT_NM, phases))

    delta = 2.0e-6

    def reference_loss(root_nm):
        return np.sum(_cauchy_detector_reference(root_nm) * np.asarray(bin_weights))

    reference_gradient = (reference_loss(ROOT_NM + delta) - reference_loss(ROOT_NM - delta)) / (
        2 * delta
    )
    np.testing.assert_allclose(gradients, reference_gradient, rtol=3e-3, atol=2e-5)
    assert np.ptp(gradients) < 3e-3 * abs(reference_gradient) + 2e-5

    root_gradient = jax.grad(lambda root: _unresolved_result(root).diagnostics.roots_nm[0])(
        ROOT_NM
    )
    np.testing.assert_allclose(np.asarray(root_gradient), 1.0, rtol=2e-12, atol=2e-12)


def test_no_root_path_is_jittable_vmap_compatible_and_supports_components():
    edges = jnp.asarray([-1.0, -0.1, 0.25, 1.0])

    def one(amplitude):
        def terms(wavelengths):
            shape = jnp.exp(-(wavelengths / 0.4) ** 2)
            numerator = jnp.stack((amplitude * shape, 2 * amplitude * shape), axis=-1)
            return numerator, jnp.ones_like(wavelengths) + 0.2j

        return integrate_detector_bins(
            terms,
            edges,
            0.06,
            source_bounds_nm=jnp.asarray([-1.4, 1.4]),
            root_scan_panels=64,
            integration_panels=64,
        )

    result = jax.jit(jax.vmap(one))(jnp.asarray([0.5, 1.0, 1.5]))
    assert result.bin_mean.shape == (3, 3, 2)
    assert np.all(np.asarray(result.diagnostics.root_count) == 0)
    assert not np.any(np.asarray(result.diagnostics.nonfinite))
    np.testing.assert_allclose(
        np.asarray(result.bin_mean[..., 1]), 2 * np.asarray(result.bin_mean[..., 0]), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(result.bin_mean[1]), 2 * np.asarray(result.bin_mean[0]), rtol=2e-14, atol=2e-14
    )


def test_root_on_scan_edge_is_counted_once():
    result = integrate_detector_bins(
        lambda wavelengths: (jnp.ones_like(wavelengths), wavelengths - 0.5 + 0.01j),
        jnp.asarray([0.0, 0.4, 0.8, 1.0]),
        0.02,
        source_bounds_nm=jnp.asarray([0.0, 1.0]),
        root_scan_panels=40,
        integration_panels=10,
    )
    assert int(result.diagnostics.root_count) == 1
    np.testing.assert_allclose(np.asarray(result.diagnostics.roots_nm[0]), 0.5, rtol=0, atol=1e-14)


def test_two_close_roots_share_one_coarse_panel_without_phase_error():
    """A fine root scan may assign multiple resonances to one coarse panel."""

    edges = jnp.asarray([-0.25, 0.0, 0.02, 0.05, 0.25])
    source_bounds = jnp.asarray([-0.5, 0.5])
    root_separation = 0.022
    imaginary_epsilon = 4.0e-5
    phases = jnp.asarray([-0.6, 0.0, 0.6])

    def one(shift, phase, integration_panels):
        roots = jnp.asarray([0.012, 0.012 + root_separation]) + shift

        def terms(wavelengths):
            real_epsilon = jnp.prod(wavelengths[:, None] - roots[None, :], axis=1)
            numerator = jnp.full_like(
                wavelengths, imaginary_epsilon * root_separation / jnp.pi
            )
            return numerator, real_epsilon + 1j * imaginary_epsilon

        return integrate_detector_bins(
            terms,
            edges,
            0.018,
            source_bounds_nm=source_bounds,
            root_scan_panels=1024,
            integration_panels=integration_panels,
            regular_order=16,
            root_order=48,
            max_roots=4,
            neighbor_panels=1,
            scan_phase=phase,
        )

    def values_and_gradients(integration_panels):
        def evaluate(phase):
            result = one(0.0, phase, integration_panels)
            gradient = jax.jacrev(lambda shift: one(shift, phase, integration_panels).bin_mean)(
                0.0
            )
            return result, gradient

        return jax.jit(jax.vmap(evaluate))(phases)

    coarse, coarse_gradient = values_and_gradients(16)
    fine, fine_gradient = values_and_gradients(128)
    coarse_values = np.asarray(coarse.bin_mean)
    fine_values = np.asarray(fine.bin_mean)
    coarse_gradient = np.asarray(coarse_gradient)
    fine_gradient = np.asarray(fine_gradient)

    assert np.all(np.asarray(coarse.diagnostics.root_count) == 2)
    assert np.all(np.asarray(fine.diagnostics.root_count) == 2)
    # At phase zero both roots lie in the same 0.0625-nm coarse panel.
    coarse_panel_ids = np.floor((np.array([0.012, 0.034]) + 0.5) / (1.0 / 16)).astype(int)
    assert coarse_panel_ids[0] == coarse_panel_ids[1]
    for phase_index in range(phases.size):
        np.testing.assert_allclose(
            coarse_values[phase_index], fine_values[phase_index], rtol=3e-3, atol=2e-6
        )
        np.testing.assert_allclose(
            coarse_gradient[phase_index], fine_gradient[phase_index], rtol=5e-3, atol=2e-5
        )
    np.testing.assert_allclose(
        coarse_values,
        np.broadcast_to(coarse_values[1], coarse_values.shape),
        rtol=3e-3,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        coarse_gradient,
        np.broadcast_to(coarse_gradient[1], coarse_gradient.shape),
        rtol=5e-3,
        atol=2e-5,
    )


def test_integration_breakpoint_exactly_partitions_a_rectangular_transmission():
    edges = np.asarray([-0.6, -0.1, 0.3, 0.7])
    source_bounds = np.asarray([-1.0, 1.0])
    cutoff = 0.137
    sigma = 0.04

    def terms(wavelengths):
        transmission = jnp.where(wavelengths < cutoff, 1.0, 0.2)
        return transmission, jnp.ones_like(wavelengths) + 0j

    def evaluate(phase):
        return integrate_detector_bins(
            terms,
            jnp.asarray(edges),
            sigma,
            source_bounds_nm=jnp.asarray(source_bounds),
            root_scan_panels=32,
            integration_panels=8,
            integration_breakpoints_nm=jnp.asarray([cutoff]),
            regular_order=16,
            scan_phase=phase,
        )

    phases = jnp.asarray([-0.55, 0.0, 0.55])
    results = jax.jit(jax.vmap(evaluate))(phases)

    reference_integrals = []
    for lower_edge, upper_edge in zip(edges[:-1], edges[1:]):
        probability = lambda wavelength: scipy_ndtr(
            (upper_edge - wavelength) / sigma
        ) - scipy_ndtr((lower_edge - wavelength) / sigma)
        integral = quad(probability, source_bounds[0], cutoff, epsabs=2e-13, epsrel=2e-13)[
            0
        ]
        integral += 0.2 * quad(
            probability, cutoff, source_bounds[1], epsabs=2e-13, epsrel=2e-13
        )[0]
        reference_integrals.append(integral)
    reference = np.asarray(reference_integrals) / np.diff(edges)

    assert not np.any(np.asarray(results.diagnostics.invalid_integration_breakpoints))
    np.testing.assert_allclose(
        np.asarray(results.bin_mean),
        np.broadcast_to(reference, results.bin_mean.shape),
        rtol=0,
        atol=2e-9,
    )


def test_close_breakpoints_share_one_original_panel_exactly_across_phases():
    edges = np.asarray([-0.6, -0.1, 0.3, 0.7])
    source_bounds = np.asarray([-1.0, 1.0])
    filter_lower = 0.031
    filter_upper = 0.047
    attenuation = 0.15
    sigma = 0.04
    breakpoints = np.asarray([filter_lower, filter_upper])

    # Both boundaries select the same nearest edge of the original eight-panel
    # phase-zero grid. They still need two distinct exact integration boundaries.
    original_interior_edges = np.linspace(*source_bounds, 9)[1:-1]
    nearest_edges = np.argmin(
        np.abs(breakpoints[:, None] - original_interior_edges[None, :]), axis=1
    )
    assert nearest_edges[0] == nearest_edges[1]

    def terms(wavelengths):
        inside_filter = (wavelengths > filter_lower) & (wavelengths < filter_upper)
        transmission = jnp.where(inside_filter, attenuation, 1.0)
        return transmission, jnp.ones_like(wavelengths) + 0j

    def evaluate(phase):
        return integrate_detector_bins(
            terms,
            jnp.asarray(edges),
            sigma,
            source_bounds_nm=jnp.asarray(source_bounds),
            root_scan_panels=32,
            integration_panels=8,
            integration_breakpoints_nm=jnp.asarray(breakpoints),
            regular_order=32,
            scan_phase=phase,
        )

    phases = jnp.asarray([-0.55, 0.0, 0.55])
    results = jax.jit(jax.vmap(evaluate))(phases)

    reference_integrals = []
    for lower_edge, upper_edge in zip(edges[:-1], edges[1:]):
        probability = lambda wavelength: scipy_ndtr(
            (upper_edge - wavelength) / sigma
        ) - scipy_ndtr((lower_edge - wavelength) / sigma)
        integral = quad(
            probability,
            source_bounds[0],
            filter_lower,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        integral += attenuation * quad(
            probability,
            filter_lower,
            filter_upper,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        integral += quad(
            probability,
            filter_upper,
            source_bounds[1],
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        reference_integrals.append(integral)
    reference = np.asarray(reference_integrals) / np.diff(edges)

    assert not np.any(np.asarray(results.diagnostics.invalid_integration_breakpoints))
    np.testing.assert_allclose(
        np.asarray(results.bin_mean),
        np.broadcast_to(reference, results.bin_mean.shape),
        rtol=0,
        atol=2e-9,
    )


def test_diagnostics_report_overflow_nonfinite_and_zero_width():
    roots = jnp.asarray([-0.8, -0.4, 0.0, 0.4, 0.8])

    def too_many_terms(wavelengths):
        real_epsilon = jnp.prod(wavelengths[:, None] - roots[None, :], axis=1)
        return jnp.ones_like(wavelengths), real_epsilon + 0.02j

    overflow = integrate_detector_bins(
        too_many_terms,
        jnp.asarray([-0.9, 0.0, 0.9]),
        0.03,
        source_bounds_nm=jnp.asarray([-1.0, 1.0]),
        root_scan_panels=100,
        integration_panels=100,
        max_roots=2,
    )
    assert int(overflow.diagnostics.root_count) == 5
    assert bool(overflow.diagnostics.root_overflow)
    assert np.all(np.isnan(np.asarray(overflow.bin_mean)))
    with pytest.raises(ValueError, match="detected 5 roots"):
        raise_for_diagnostics(overflow)

    zero_width = integrate_detector_bins(
        lambda wavelengths: (jnp.ones_like(wavelengths), wavelengths - 0.13 + 0j),
        jnp.asarray([-0.5, 0.0, 0.5]),
        0.04,
        source_bounds_nm=jnp.asarray([-0.7, 0.7]),
        root_scan_panels=64,
        integration_panels=64,
    )
    assert bool(zero_width.diagnostics.zero_width)
    assert np.all(np.isnan(np.asarray(zero_width.bin_mean)))
    with pytest.raises(ValueError, match="zero local half-width"):
        raise_for_diagnostics(zero_width)

    nonfinite = integrate_detector_bins(
        lambda wavelengths: (
            jnp.ones_like(wavelengths),
            jnp.where(wavelengths > 0, jnp.nan + 0j, 1.0 + 0j),
        ),
        jnp.asarray([-0.5, 0.0, 0.5]),
        0.04,
        source_bounds_nm=jnp.asarray([-0.7, 0.7]),
        root_scan_panels=32,
        integration_panels=32,
    )
    assert bool(nonfinite.diagnostics.nonfinite)
    assert np.all(np.isnan(np.asarray(nonfinite.bin_mean)))

    invalid_edges = integrate_detector_bins(
        lambda wavelengths: (jnp.ones_like(wavelengths), jnp.ones_like(wavelengths) + 0j),
        jnp.asarray([0.0, 0.2, 0.2, 0.5]),
        0.0,
        scan_phase=1.0,
    )
    assert bool(invalid_edges.diagnostics.invalid_bin_width)
    assert bool(invalid_edges.diagnostics.invalid_irf_sigma)
    assert bool(invalid_edges.diagnostics.invalid_scan_phase)
    assert np.all(np.isnan(np.asarray(invalid_edges.bin_mean)))

    @jax.jit
    def invalid_breakpoints(breakpoints):
        return integrate_detector_bins(
            lambda wavelengths: (jnp.ones_like(wavelengths), jnp.ones_like(wavelengths) + 0j),
            jnp.asarray([-0.5, 0.0, 0.5]),
            0.04,
            source_bounds_nm=jnp.asarray([-0.7, 0.7]),
            root_scan_panels=32,
            integration_panels=8,
            integration_breakpoints_nm=breakpoints,
        )

    bad_breakpoint_sets = (
        [0.2, 0.1],
        [0.1, 0.1],
        [-0.8, 0.1],
        [0.1, np.nan],
    )
    bad_breakpoints = None
    for breakpoint_set in bad_breakpoint_sets:
        bad_breakpoints = invalid_breakpoints(jnp.asarray(breakpoint_set))
        assert bool(bad_breakpoints.diagnostics.invalid_integration_breakpoints)
        assert np.all(np.isnan(np.asarray(bad_breakpoints.bin_mean)))
    with pytest.raises(ValueError, match="integration breakpoints"):
        raise_for_diagnostics(bad_breakpoints)
