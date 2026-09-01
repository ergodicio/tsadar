"""Regression for the unresolved small-angle ARTS2D EPW from #124/#134."""

import numpy as np
import pytest

from jax import config

config.update("jax_enable_x64", True)

import jax
from jax import numpy as jnp

from tsadar.core.physics.form_factor import FormFactor
from tsadar.core.physics.generate_spectra import FitModel
from tsadar.core.physics.resonance_quadrature import integrate_detector_bins


pytestmark = [pytest.mark.physics, pytest.mark.slow]


DETECTOR_EDGES_NM = jnp.linspace(468.0, 476.0, 257)
FULL_DETECTOR_EDGES_NM = jnp.linspace(449.0, 670.0, 222)
IRF_SIGMA_NM = 0.9 / 2.3548
PHASES = jnp.asarray([0.0, 0.31, 0.67])
# Includes phases at which a 2048-panel root scan misses the closely spaced
# central pair, plus both extremes of the shifted scan grid.
FULL_RANGE_PHASES = jnp.asarray([-0.95, -0.10, 0.0, 0.90, 0.95])
EXPECTED_FULL_RANGE_ROOTS_NM = np.asarray(
    [
        474.201875844894,
        514.081647830358,
        526.276288254028,
        526.451996163684,
        526.548008175237,
        526.723802255372,
        539.209067615960,
        591.042039725023,
    ]
)
FINE_ROOT_SCAN_PANELS = 4096
MAX_ROOTS = 16


def _small_angle_problem(detector_edges_nm=DETECTOR_EDGES_NM):
    velocity = jnp.linspace(-9.0, 9.0, 129)
    projection = jnp.exp(-0.5 * velocity**2) / jnp.sqrt(2.0 * jnp.pi)
    derivative = jnp.gradient(projection, velocity[1] - velocity[0])
    n_beta = 16
    sinogram = (
        jnp.broadcast_to(projection, (n_beta, velocity.size)),
        jnp.broadcast_to(derivative, (n_beta, velocity.size)),
    )

    vx, vy = jnp.meshgrid(velocity, velocity)
    distribution = jnp.exp(-0.5 * (vx**2 + vy**2)) / (2.0 * jnp.pi)
    detector_edges_nm = jnp.asarray(detector_edges_nm)
    form_factor = FormFactor(
        lambda_range=[float(detector_edges_nm[0]), float(detector_edges_nm[-1])],
        npts=32,
        lam_shift=0.0,
        scattering_angles={"sa": np.asarray([22.0]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        va_ang={"ion-1": 0.0},
        ud_ang=0.0,
        calc_gain={"calc": False},
        n_beta=n_beta,
    )

    def parameters(electron_drift):
        return {
            "electron": {
                "ne": 0.44,
                "Te": 1.1,
                "fe": distribution,
                "v": velocity,
            },
            "general": {
                "ne_gradient": 0.0,
                "Te_gradient": 0.0,
                "lam": 526.5,
                "ud": electron_drift,
            },
            "ion-1": {
                "A": 1.0,
                "Z": 1.0,
                "Ti": 0.03,
                "fract": 1.0,
                "Va": 0.0,
            },
        }

    def integrate(
        electron_drift,
        phase,
        integration_panels,
        root_scan_panels=FINE_ROOT_SCAN_PANELS,
    ):
        params = parameters(electron_drift)

        def terms(wavelengths_nm):
            numerator, epsilon = form_factor.calc_2D_spectral_terms(
                params,
                wavelengths_nm,
                sinogram=sinogram,
            )
            return numerator[:, 0, 0], epsilon[:, 0, 0]

        return integrate_detector_bins(
            terms,
            detector_edges_nm,
            IRF_SIGMA_NM,
            root_scan_panels=root_scan_panels,
            integration_panels=integration_panels,
            regular_order=8,
            root_order=32,
            max_roots=MAX_ROOTS,
            neighbor_panels=1,
            scan_phase=phase,
        )

    return integrate


def _evaluate_values_and_drift_gradients(
    integrate,
    integration_panels,
    *,
    root_scan_panels=FINE_ROOT_SCAN_PANELS,
    phases=PHASES,
):
    def one_phase(phase):
        result = integrate(5.0, phase, integration_panels, root_scan_panels)
        # The input is one scalar while the output has one entry per detector bin,
        # so forward mode forms the complete Jacobian in one tangent evaluation.
        gradient = jax.jacfwd(
            lambda electron_drift: integrate(
                electron_drift,
                phase,
                integration_panels,
                root_scan_panels,
            ).bin_mean
        )(5.0)
        return result, gradient

    return jax.jit(jax.vmap(one_phase))(phases)


def _relative_detector_l1(actual, expected, detector_edges_nm=DETECTOR_EDGES_NM):
    widths = np.diff(np.asarray(detector_edges_nm))
    return np.sum(np.abs(actual - expected) * widths) / np.sum(
        np.abs(expected) * widths
    )


def test_small_angle_unresolved_epw_values_and_gradients_converge():
    """The physical unresolved line is stable to phase and 2x/4x refinement.

    The acceptance tolerance is one percent. This regression retains substantial
    margin while exercising the corrected signed susceptibility from #135, implicit
    root motion with electron drift, exact spectral-IRF bin probabilities, and the
    narrow resonance that the sampled wavelength grid missed in #124.
    """

    integrate = _small_angle_problem()
    panel_counts = (64, 128, 256)
    evaluations = [
        _evaluate_values_and_drift_gradients(integrate, panel_count)
        for panel_count in panel_counts
    ]
    values = [np.asarray(results.bin_mean) for results, _ in evaluations]
    gradients = [np.asarray(gradient) for _, gradient in evaluations]

    for results, gradient in evaluations:
        assert np.all(np.asarray(results.diagnostics.root_count) == 1)
        assert not np.any(np.asarray(results.diagnostics.root_overflow))
        assert np.all(np.isfinite(np.asarray(results.bin_mean)))
        assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(gradients[-1][0]) > 0

    # Retain a focused reverse-mode check representative of inference: one scalar
    # detector loss must have the same drift derivative as contracting the full
    # forward-mode detector Jacobian with its loss weights.
    loss_weights = jnp.linspace(-0.5, 0.75, DETECTOR_EDGES_NM.size - 1)
    loss_weights *= jnp.diff(DETECTOR_EDGES_NM)

    def weighted_detector_loss(electron_drift):
        return jnp.vdot(
            loss_weights,
            integrate(
                electron_drift,
                PHASES[0],
                panel_counts[-1],
                FINE_ROOT_SCAN_PANELS,
            ).bin_mean,
        )

    reverse_loss_gradient = jax.jit(jax.grad(weighted_detector_loss))(5.0)
    forward_loss_gradient = jnp.vdot(loss_weights, evaluations[-1][1][0])
    assert np.isfinite(float(reverse_loss_gradient))
    assert abs(float(reverse_loss_gradient)) > 0
    np.testing.assert_allclose(
        np.asarray(reverse_loss_gradient),
        np.asarray(forward_loss_gradient),
        rtol=2e-10,
        atol=2e-12,
    )

    # The corrected benchmark root lies near 474.2 nm and is roughly one thousand
    # times narrower than the 0.9-nm FWHM instrument response.
    roots = np.asarray(evaluations[-1][0].diagnostics.roots_nm)[:, 0]
    widths = np.asarray(
        evaluations[-1][0].diagnostics.resonance_half_widths_nm
    )[:, 0]
    np.testing.assert_allclose(roots, 474.22, rtol=0, atol=0.03)
    assert np.all(widths < 1.0e-3)

    for phase_index in range(1, PHASES.size):
        assert _relative_detector_l1(values[-1][phase_index], values[-1][0]) < 0.01
        assert (
            np.linalg.norm(gradients[-1][phase_index] - gradients[-1][0])
            / np.linalg.norm(gradients[-1][0])
            < 0.01
        )

    # The same 64-panel baseline is compared with both 2x and 4x integration
    # refinement while the fine root scan remains fixed. This isolates integration
    # convergence from root-discovery topology.
    for refined_index in (1, 2):
        for phase_index in range(PHASES.size):
            assert (
                _relative_detector_l1(
                    values[0][phase_index], values[refined_index][phase_index]
                )
                < 0.01
            )
            assert (
                np.linalg.norm(
                    gradients[0][phase_index] - gradients[refined_index][phase_index]
                )
                / np.linalg.norm(gradients[refined_index][phase_index])
                < 0.01
            )


def test_full_detector_range_finds_every_physical_root_with_finite_gradients():
    """A fine root scan sees the complete small-angle dielectric topology.

    Across the representative 449--670 nm ARTS range the collisionless dielectric
    has narrow blue/red EPW roots, broad crossings, and four closely spaced roots
    around the probe wavelength. Root discovery must be independent of the coarser
    detector-integration grid and its phase; missing the close pairs can silently
    integrate the wrong spectrum even when a narrow-wing regression still passes.
    """

    integrate = _small_angle_problem(FULL_DETECTOR_EDGES_NM)
    results, gradients = _evaluate_values_and_drift_gradients(
        integrate,
        256,
        root_scan_panels=FINE_ROOT_SCAN_PANELS,
        phases=FULL_RANGE_PHASES,
    )

    root_counts = np.asarray(results.diagnostics.root_count)
    roots = np.asarray(results.diagnostics.roots_nm)
    root_masks = np.asarray(results.diagnostics.root_mask)
    values = np.asarray(results.bin_mean)
    gradients = np.asarray(gradients)

    np.testing.assert_array_equal(root_counts, np.full(FULL_RANGE_PHASES.shape, 8))
    assert not np.any(np.asarray(results.diagnostics.root_overflow))
    assert np.all(np.sum(root_masks, axis=1) == 8)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    assert np.all(np.linalg.norm(gradients, axis=1) > 0)

    active_roots = np.stack(
        [
            phase_roots[phase_mask]
            for phase_roots, phase_mask in zip(roots, root_masks)
        ]
    )
    np.testing.assert_allclose(
        active_roots,
        np.broadcast_to(active_roots[0], active_roots.shape),
        rtol=0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        active_roots,
        np.broadcast_to(EXPECTED_FULL_RANGE_ROOTS_NM, active_roots.shape),
        rtol=0,
        atol=2e-6,
    )


def test_fit_model_integrates_each_geometry_then_aperture_weights(monkeypatch):
    """Production plumbing preserves gradient/angle topology without excess memory."""

    detector_edges = np.asarray([0.0, 0.45, 0.9, 1.35, 2.0])
    detector_centers = 0.5 * (detector_edges[:-1] + detector_edges[1:])
    weights = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    scattering_angles = {"sa": np.asarray([1.0, 2.0]), "weights": weights}
    cfg = {
        "data": {
            "load_ele_spec": True,
            "load_ion_spec": False,
            "ele_lam_shift": 0.0,
        },
        "parameters": {
            "general": {
                "Te_gradient": {"num_grad_points": 2},
                "ne_gradient": {"num_grad_points": 2},
                "ud": {"angle": 0.0},
            },
            "electron": {"fe": {"dim": 2}},
            "ion-1": {"Va": {"angle": 0.0}},
        },
        "other": {
            "detector_specs": {
                "electron_wavelength_edges": detector_edges,
                "electron_wavelength_centers": detector_centers,
                "widIRF": {"spect_FWHM_ele": 0.1},
            },
            "extraoptions": {"spectype": "angular_full"},
            "resonance_quadrature": {
                "root_scan_panels": 64,
                "integration_panels": 64,
                "regular_order": 4,
                "root_order": 16,
                "max_roots": MAX_ROOTS,
                "map_batch_size": 1,
            },
            "lamrangE": [0.0, 2.0],
            "lamrangI": [0.0, 2.0],
            "npts": 8,
            "n_beta": 4,
            "iawoff": 0,
            "iawfilter": [0, 0.0, 0.0, 0.0],
        },
    }
    model = FitModel(cfg, scattering_angles)

    monkeypatch.setattr(
        model.electron_form_factor,
        "prepare_2D_sinogram",
        lambda params: None,
    )

    def synthetic_terms(params, wavelengths_nm, sinogram=None, scattering_angles=None):
        del params, sinogram
        angles = jnp.atleast_1d(jnp.asarray(scattering_angles))
        gradient_offsets = 0.03 * jnp.arange(2)[:, None]
        roots = 0.65 + 0.10 * angles[None, :] + gradient_offsets
        epsilon = wavelengths_nm[:, None, None] - roots[None, :, :] + 0.02j
        numerator = jnp.full(epsilon.shape, 0.02 / jnp.pi)
        return numerator, epsilon

    monkeypatch.setattr(
        model.electron_form_factor,
        "calc_2D_spectral_terms",
        synthetic_terms,
    )

    axis, model_bins, raw_bins, diagnostics = model.detector_integrated_electron_spectrum(
        {"general": {"lam": 1.0}}
    )

    assert model.electron_spectrum_is_detector_binned
    assert raw_bins.shape == (2, detector_edges.size - 1, 2)
    assert model_bins.shape == (2, detector_edges.size - 1)
    assert np.all(np.asarray(diagnostics.root_count) == 1)
    assert np.all(np.isfinite(np.asarray(raw_bins)))
    np.testing.assert_array_equal(np.asarray(axis), detector_centers)
    np.testing.assert_allclose(
        np.asarray(model_bins),
        weights @ np.mean(np.asarray(raw_bins), axis=0).T,
        rtol=2e-14,
        atol=2e-14,
    )
