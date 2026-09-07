"""Production wiring regression for source-side ARTS2D notch filtering."""

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from jax import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp

from tsadar.core.physics.generate_spectra import FitModel


DETECTOR_EDGES_NM = np.asarray([500.0, 501.0, 502.0])
IRF_SIGMA_NM = 0.1
NOTCH_LOWER_NM = 500.25
NOTCH_UPPER_NM = 500.75
NOTCH_ATTENUATION = 0.1
TAIL_SIGMA = 6.0


def _config(integration_panels, scan_phase):
    return {
        "data": {
            "load_ele_spec": True,
            "load_ion_spec": False,
            "ele_lam_shift": 0.0,
        },
        "parameters": {
            "general": {
                "Te_gradient": {"num_grad_points": 1},
                "ne_gradient": {"num_grad_points": 1},
                "ud": {"angle": 0.0},
            },
            "electron": {"fe": {"dim": 2}},
            "ion-1": {"Va": {"angle": 0.0}},
        },
        "other": {
            "detector_specs": {
                "electron_wavelength_edges": DETECTOR_EDGES_NM,
                "electron_wavelength_centers": 0.5
                * (DETECTOR_EDGES_NM[:-1] + DETECTOR_EDGES_NM[1:]),
                "widIRF": {"spect_FWHM_ele": IRF_SIGMA_NM * 2.3548},
            },
            "extraoptions": {"spectype": "angular_full"},
            "resonance_quadrature": {
                "enabled": True,
                "root_scan_panels": 64,
                "integration_panels": integration_panels,
                "regular_order": 8,
                "root_order": 16,
                "max_roots": 16,
                "tail_sigma": TAIL_SIGMA,
                "scan_phase": scan_phase,
                "map_batch_size": 1,
            },
            "lamrangE": [500.0, 502.0],
            "lamrangI": [500.0, 502.0],
            "npts": 2,
            "n_beta": 4,
            # OD 1 over [500.25, 500.75] nm: the notch cuts through only part
            # of the first [500, 501] detector bin.
            "iawfilter": [1, 1.0, 0.5, 500.5],
            "iawoff": 0,
        },
    }


def _constant_spectrum_model(integration_panels, scan_phase):
    model = FitModel(
        _config(integration_panels, scan_phase),
        {"sa": np.asarray([30.0]), "weights": np.ones((1, 1))},
    )
    model.electron_form_factor.prepare_2D_sinogram = lambda params: None

    def constant_terms(params, wavelengths_nm, sinogram=None, scattering_angles=None):
        del params, sinogram, scattering_angles
        shape = (wavelengths_nm.size, 1, 1)
        return jnp.ones(shape), jnp.ones(shape, dtype=jnp.complex128)

    model.electron_form_factor.calc_2D_spectral_terms = constant_terms
    return model


def _exact_source_side_bin_means():
    source_lower = DETECTOR_EDGES_NM[0] - TAIL_SIGMA * IRF_SIGMA_NM
    source_upper = DETECTOR_EDGES_NM[-1] + TAIL_SIGMA * IRF_SIGMA_NM

    def transmission(source_nm):
        if NOTCH_LOWER_NM < source_nm < NOTCH_UPPER_NM:
            return NOTCH_ATTENUATION
        return 1.0

    def one_bin(bin_lower, bin_upper):
        def integrand(source_nm):
            probability = ndtr((bin_upper - source_nm) / IRF_SIGMA_NM) - ndtr(
                (bin_lower - source_nm) / IRF_SIGMA_NM
            )
            return transmission(source_nm) * probability

        integral = 0.0
        boundaries = [
            source_lower,
            NOTCH_LOWER_NM,
            NOTCH_UPPER_NM,
            source_upper,
        ]
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            integral += quad(integrand, lower, upper, epsabs=1.0e-13, epsrel=1.0e-13)[0]
        return integral / (bin_upper - bin_lower)

    return np.asarray(
        [
            one_bin(lower, upper)
            for lower, upper in zip(
                DETECTOR_EDGES_NM[:-1], DETECTOR_EDGES_NM[1:]
            )
        ]
    )


def test_partial_bin_notch_is_exact_source_side_and_phase_stable():
    """A notch boundary splits source integration, not a detector-space bin."""

    expected = _exact_source_side_bin_means()
    outputs = []

    for integration_panels, scan_phase in [(32, 0.0), (32, 0.37), (64, -0.41)]:
        model = _constant_spectrum_model(integration_panels, scan_phase)
        np.testing.assert_array_equal(
            np.asarray(
                model.resonance_quadrature_options[
                    "integration_breakpoints_nm"
                ]
            ),
            np.asarray([NOTCH_LOWER_NM, NOTCH_UPPER_NM]),
        )

        _, detector_model, raw_model, diagnostics = (
            model.detector_integrated_electron_spectrum(
                {"general": {"lam": 501.5}}
            )
        )
        output = np.asarray(detector_model[0])
        outputs.append(output)

        np.testing.assert_allclose(output, expected, rtol=2.0e-10, atol=2.0e-10)
        np.testing.assert_allclose(
            np.asarray(raw_model[0, :, 0]),
            expected,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        assert not np.any(np.asarray(diagnostics.invalid_integration_breakpoints))
        assert np.all(np.asarray(diagnostics.root_count) == 0)
        assert np.all(np.isfinite(output))

    # A detector-space mask selected by the first bin's 500.5-nm center would attenuate
    # the entire bin to roughly 0.1. Source-side filtering attenuates only its central
    # half, leaving the correctly blurred/bin-integrated density near 0.55.
    assert expected[0] > 0.5
    np.testing.assert_allclose(
        outputs,
        np.broadcast_to(expected, (3, 2)),
        rtol=2.0e-10,
        atol=2.0e-10,
    )
