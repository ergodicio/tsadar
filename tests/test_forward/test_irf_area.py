"""Detector-coordinate and area-preserving IRF regressions for ARTS."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from jax import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp

from tsadar.core.instrument.irf import (
    AngularIRF,
    _apply_uniform_gaussian_bin_response,
    _gaussian_bin_response,
    add_ATS_IRF,
)
from tsadar.core import thomson_diagnostic
from tsadar.core.thomson_diagnostic import (
    ThomsonScatteringDiagnostic,
    _ensure_angular_detector_edges,
)
from tsadar.data import prepare
from tsadar.data.calibration import (
    detector_edges_from_centers,
    get_calibrations,
    grouped_detector_edges,
)
from tsadar.utils.misc import merge_defaults_and_inputs


def _instrument_params():
    return {"general": {"lam": 0.0, "amp1": 1.0, "amp2": 1.0}}


def test_shipped_arts2d_deck_exposes_detector_irf_and_quadrature():
    repository = Path(__file__).resolve().parents[2]
    defaults = yaml.safe_load((repository / "configs/arts-2d/defaults.yaml").read_text())
    inputs = yaml.safe_load((repository / "configs/arts-2d/inputs.yaml").read_text())

    merged = merge_defaults_and_inputs(defaults, inputs)

    assert merged["other"]["extraoptions"]["spectype"] == "angular"
    assert merged["other"]["detector_specs"]["widIRF"]["spect_FWHM_ele"] == 0.9
    assert merged["other"]["resonance_quadrature"]["enabled"] is True


def test_detector_edges_distinguish_centers_from_finite_pixel_support():
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    np.testing.assert_array_equal(detector_edges_from_centers(centers), np.arange(6.0))
    # Two full two-pixel groups followed by a one-pixel ragged group. Re-inferring
    # edges from the group centers [1, 3, 4.5] would incorrectly move the 4.0 edge.
    np.testing.assert_array_equal(grouped_detector_edges(centers, 2), np.array([0.0, 2.0, 4.0, 5.0]))


@pytest.mark.parametrize("bad_centers", [[1.0], [1.0, 1.0], [2.0, 1.0], [1.0, np.nan]])
def test_detector_edges_reject_invalid_center_axes(bad_centers):
    with pytest.raises(ValueError):
        detector_edges_from_centers(np.asarray(bad_centers))


@pytest.mark.parametrize("bad_group", [0, -1, 1.5, True])
def test_grouped_detector_edges_reject_invalid_group_size(bad_group):
    with pytest.raises(ValueError):
        grouped_detector_edges(np.arange(4.0), bad_group)


def test_prepare_data_retains_exact_arts_spectral_edges(monkeypatch):
    electron_data = np.arange(20.0).reshape(5, 4)
    ion_data = np.zeros_like(electron_data)
    wavelength_centers = np.arange(5.0) + 0.5

    monkeypatch.setattr(
        prepare,
        "loadData",
        lambda *args, **kwargs: [electron_data, ion_data, "Angle", [0.0, 0.0], "angular", [5, 4]],
    )
    monkeypatch.setattr(
        prepare,
        "get_scattering_angles",
        lambda cfg: {"sa": np.arange(4.0), "weights": np.ones((4, 4))},
    )
    monkeypatch.setattr(
        prepare,
        "get_calibrations",
        lambda *args, **kwargs: [
            np.arange(4.0),
            np.arange(4.0),
            wavelength_centers,
            wavelength_centers,
            1.0,
            {
                "spect_stddev_ion": 1.0,
                "spect_stddev_ele": 1.0,
                "spect_FWHM_ele": 1.0,
                "ang_FWHM_ele": 1.0,
            },
        ],
    )
    monkeypatch.setattr(prepare, "correctThroughput", lambda data, *args, **kwargs: data)
    monkeypatch.setattr(
        prepare,
        "get_shot_bg",
        lambda *args, **kwargs: [np.zeros_like(electron_data), np.zeros_like(ion_data)],
    )

    cfg = {
        "data": {
            "lineouts": {"type": "range", "val": [0], "start": 0, "end": 4},
            "load_ele_spec": True,
            "load_ion_spec": False,
            "fit_EPWb": True,
            "fit_EPWr": True,
            "fit_IAW": False,
            "launch_data_visualizer": False,
            "shotDay": False,
            "bgscaleE": 0.0,
        },
        "feature_detector": {"estimate_lineouts_epw": False, "estimate_lineouts_iaw": False},
        "optimizer": {"batch_size": 1},
        "other": {
            "CCDsize": [5, 4],
            "ang_res_unit": 2,
            "lam_res_unit": 2,
            "points_per_pixel": 1,
            "detector_specs": {},
            "extraoptions": {"spectype": "angular"},
        },
    }

    all_data, _, all_axes = prepare.prepare_data(cfg, 1)

    np.testing.assert_array_equal(all_axes["epw_y"].squeeze(), np.array([1.0, 3.0, 4.5]))
    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_edges"], np.array([0.0, 2.0, 4.0, 5.0])
    )
    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_centers"],
        np.array([1.0, 3.0, 4.5]),
    )
    assert all_data["e_data"].shape == (2, 3)
    assert cfg["other"]["CCDsize"] == (2, 3)


def test_forward_arts_bounds_remain_first_and_last_detector_centers():
    cfg = {
        "data": {"load_ele_spec": True},
        "other": {
            "CCDsize": [5, 4],
            "lamrangE": [400.0, 500.0],
            "detector_specs": {},
            "extraoptions": {"spectype": "angular_full"},
        },
    }

    _ensure_angular_detector_edges(cfg)

    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_edges"],
        np.array([387.5, 412.5, 437.5, 462.5, 487.5, 512.5]),
    )
    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_centers"],
        np.array([400.0, 425.0, 450.0, 475.0, 500.0]),
    )


def test_single_bin_forward_arts_uses_the_requested_finite_support():
    cfg = {
        "data": {"load_ele_spec": True},
        "other": {
            "CCDsize": [1, 4],
            "lamrangE": [400.0, 500.0],
            "detector_specs": {},
            "extraoptions": {"spectype": "angular_full"},
        },
    }

    _ensure_angular_detector_edges(cfg)

    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_edges"],
        np.array([400.0, 500.0]),
    )
    np.testing.assert_array_equal(
        cfg["other"]["detector_specs"]["electron_wavelength_centers"],
        np.array([450.0]),
    )


@pytest.mark.parametrize("number_of_points", [400, 801])
def test_normalized_spectral_irf_preserves_unresolved_line_area_and_centroid(
    number_of_points,
):
    wavelengths = jnp.linspace(-10.0, 10.0, number_of_points)
    edges = detector_edges_from_centers(np.asarray(wavelengths))
    widths = np.diff(edges)
    source_index = number_of_points // 2
    unresolved_line = jnp.zeros((1, wavelengths.size)).at[0, source_index].set(
        1.0 / widths[source_index]
    )
    irf = AngularIRF(
        spect_stddev=1.0,
        ang_stddev=1.0,
        ang_axis=np.array([0.0]),
        normalize=0,
    )

    _, blurred = add_ATS_IRF(irf, wavelengths, unresolved_line, _instrument_params())

    blurred = np.asarray(blurred[0])
    area = np.sum(blurred * widths)
    centroid = np.sum(blurred * widths * np.asarray(wavelengths)) / area
    np.testing.assert_allclose(area, 1.0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        centroid,
        float(wavelengths[source_index]),
        rtol=0,
        atol=2e-11,
    )
    assert float(jnp.max(blurred)) < float(jnp.max(unresolved_line))


def test_normalized_ats_irf_keeps_constant_density_away_from_boundaries():
    angles = np.linspace(-5.0, 5.0, 101)
    wavelengths = jnp.linspace(-10.0, 10.0, 201)
    constant = jnp.ones((angles.size, wavelengths.size))
    irf = AngularIRF(spect_stddev=0.5, ang_stddev=0.3, ang_axis=angles, normalize=0)

    _, blurred = add_ATS_IRF(irf, wavelengths, constant, _instrument_params())

    np.testing.assert_allclose(np.asarray(blurred[20:-20, 30:-30]), 1.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("number_of_points", [20, 21, 2050])
def test_uniform_spectral_operator_matches_exact_dense_response(number_of_points):
    wavelengths = jnp.linspace(-3.0, 4.0, number_of_points)
    values = jnp.stack(
        (
            jnp.exp(-((wavelengths + 0.7) / 0.4) ** 2),
            0.3 + jnp.exp(-((wavelengths - 1.2) / 0.8) ** 2),
        )
    )

    actual = _apply_uniform_gaussian_bin_response(
        wavelengths,
        values,
        0.35,
        response_axis=1,
    )
    exact = values @ _gaussian_bin_response(wavelengths, 0.35).T

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(exact),
        rtol=5e-11,
        atol=5e-11,
    )


def test_real_nonuniform_angular_irf_preserves_physical_area_and_center():
    angles, *_ = get_calibrations(104000, "angular", 0.0, [1024, 1024], {})
    edges = detector_edges_from_centers(angles)
    widths = np.diff(edges)
    source_index = angles.size // 2
    line = jnp.zeros((angles.size, 1)).at[source_index, 0].set(
        1.0 / widths[source_index]
    )
    irf = AngularIRF(
        spect_stddev=1.0,
        ang_stddev=1.0 / 2.3548,
        ang_axis=angles,
        normalize=0,
    )

    _, blurred = add_ATS_IRF(
        irf,
        jnp.asarray([500.0]),
        line,
        _instrument_params(),
        apply_spectral_blur=False,
    )

    blurred = np.asarray(blurred[:, 0])
    area = np.sum(blurred * widths)
    centroid = np.sum(blurred * widths * angles) / area
    np.testing.assert_allclose(area, 1.0, rtol=0, atol=2e-11)
    np.testing.assert_allclose(
        centroid,
        angles[source_index],
        rtol=0,
        atol=1e-3,
    )

    constant = np.asarray(
        add_ATS_IRF(
            irf,
            jnp.asarray([500.0]),
            jnp.ones((angles.size, 1)),
            _instrument_params(),
            apply_spectral_blur=False,
        )[1][:, 0]
    )
    interior = (angles > angles[0] + 5 * irf.ang_stddev) & (
        angles < angles[-1] - 5 * irf.ang_stddev
    )
    np.testing.assert_allclose(constant[interior], 1.0, rtol=0, atol=1e-6)


def test_ats_can_skip_spectral_blur_after_detector_bin_quadrature():
    angles = np.linspace(-5.0, 5.0, 101)
    wavelengths = jnp.array([-1.0, 0.0, 1.0])
    detector_bin_means = jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (angles.size, 1))
    irf = AngularIRF(spect_stddev=0.5, ang_stddev=0.3, ang_axis=angles, normalize=0)

    _, angular_only = add_ATS_IRF(
        irf,
        wavelengths,
        detector_bin_means,
        _instrument_params(),
        apply_spectral_blur=False,
    )

    np.testing.assert_allclose(np.asarray(angular_only[angles.size // 2]), np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_diagnostic_does_not_apply_a_second_spectral_irf(monkeypatch):
    diagnostic = ThomsonScatteringDiagnostic.__new__(ThomsonScatteringDiagnostic)
    diagnostic.cfg = {
        "data": {"load_ele_spec": True, "load_ion_spec": False},
        "other": {"extraoptions": {"spectype": "angular_full"}},
    }
    diagnostic.model = SimpleNamespace(electron_spectrum_is_detector_binned=True)
    diagnostic.ats_irf = object()
    called = {}

    def capture_ats_irf(
        irf_description,
        wavelength_axis,
        model,
        instrument_params,
        *,
        apply_spectral_blur,
    ):
        del irf_description, instrument_params
        called["apply_spectral_blur"] = apply_spectral_blur
        return wavelength_axis, model

    monkeypatch.setattr(thomson_diagnostic.irf, "add_ATS_IRF", capture_ats_irf)
    wavelengths = jnp.asarray([1.0, 2.0, 3.0])
    model = jnp.ones((2, 3))
    output, _, output_axis, _ = diagnostic.postprocess_theory(
        model,
        0,
        wavelengths,
        jnp.zeros(1),
        {"e_amps": 1.0, "i_amps": 1.0},
        _instrument_params(),
    )

    assert called["apply_spectral_blur"] is False
    np.testing.assert_array_equal(np.asarray(output), np.asarray(model))
    np.testing.assert_array_equal(np.asarray(output_axis), np.asarray(wavelengths))


def test_detector_binned_arts_reduction_only_reduces_angle():
    diagnostic = ThomsonScatteringDiagnostic.__new__(ThomsonScatteringDiagnostic)
    diagnostic.model = SimpleNamespace(electron_spectrum_is_detector_binned=True)
    diagnostic.cfg = {
        "other": {"CCDsize": [2, 3], "ang_res_unit": 2},
        "data": {"lineouts": {"start": 0, "end": 2}},
    }
    wavelengths = jnp.asarray([1.0, 2.0, 4.0])
    model = jnp.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )
    batch = {"e_data": np.ones((2, 3)), "e_amps": jnp.ones((2, 1))}
    instrument_params = {
        "general": {"lam": 0.0, "amp1": 1.0, "amp2": 1.0}
    }

    reduced, reduced_axis = diagnostic.reduce_ATS_to_resunit(
        model, wavelengths, instrument_params, batch
    )

    assert reduced.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(reduced_axis), np.asarray(wavelengths))


def test_detector_binned_arts_reduction_keeps_a_ragged_angular_group():
    diagnostic = ThomsonScatteringDiagnostic.__new__(ThomsonScatteringDiagnostic)
    diagnostic.model = SimpleNamespace(electron_spectrum_is_detector_binned=True)
    diagnostic.cfg = {
        "other": {"CCDsize": [2, 2], "ang_res_unit": 3},
        "data": {"lineouts": {"start": 0, "end": 2}},
    }
    wavelengths = jnp.asarray([1.0, 2.0])
    model = jnp.arange(10.0).reshape(5, 2) + 1
    batch = {"e_data": np.ones((2, 2)), "e_amps": jnp.ones((2, 1))}
    instrument_params = {
        "general": {"lam": 0.0, "amp1": 1.0, "amp2": 1.0}
    }

    reduced, _ = diagnostic.reduce_ATS_to_resunit(
        model, wavelengths, instrument_params, batch
    )

    assert reduced.shape == (2, 2)
