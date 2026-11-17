"""Detector-coordinate and area-preserving IRF regressions for ARTS."""

import numpy as np
import pytest
from jax import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp

from tsadar.core.instrument.irf import (
    AngularIRF,
    _apply_uniform_gaussian_bin_response,
    _gaussian_bin_response,
    add_ATS_IRF,
)
from tsadar.utils.detector_grid import detector_edges_from_centers, grouped_detector_edges, oversampled_axis
from tsadar.core.thomson_diagnostic import _ensure_angular_detector_edges
from tsadar.data import prepare
from tsadar.data.calibration import get_calibrations


def _instrument_params():
    return {"general": {"lam": 0.0, "amp1": 1.0, "amp2": 1.0}}


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


@pytest.mark.parametrize("points_per_pixel", [1, 2, 5])
def test_oversampled_axis_groups_are_centered_on_the_detector_pixels(points_per_pixel):
    centers = np.linspace(400.0, 700.0, 7)

    fine = oversampled_axis(centers, points_per_pixel)

    assert fine.shape == (centers.size * points_per_pixel,)
    np.testing.assert_allclose(np.diff(fine), 50.0 / points_per_pixel, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fine.reshape(centers.size, -1).mean(axis=1), centers, rtol=0, atol=1e-10)


def test_oversampled_axis_stays_inside_each_pixel_of_a_nonuniform_axis():
    centers = np.array([400.0, 400.5, 401.0, 402.0, 404.0])
    edges = detector_edges_from_centers(centers)

    fine = oversampled_axis(centers, 5).reshape(centers.size, -1)

    assert np.all(np.diff(fine.ravel()) > 0)
    assert np.all(fine > edges[:-1, None]) and np.all(fine < edges[1:, None])
    np.testing.assert_allclose(fine.mean(axis=1), 0.5 * (edges[:-1] + edges[1:]), rtol=0, atol=1e-10)


@pytest.mark.parametrize("bad_points_per_pixel", [0, -1, 1.5, True])
def test_oversampled_axis_rejects_invalid_points_per_pixel(bad_points_per_pixel):
    with pytest.raises(ValueError):
        oversampled_axis(np.arange(4.0), bad_points_per_pixel)


def test_prepare_data_drops_a_ragged_final_arts_spectral_unit_on_the_uniform_model_grid(monkeypatch):
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
        "parameters": {"electron": {"fe": {"dim": 1}}},
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

    # 5 wavelength pixels in units of 2: the uniform model grid drops the ragged fifth pixel.
    centers, edges = np.array([1.0, 3.0]), np.array([0.0, 2.0, 4.0])
    np.testing.assert_array_equal(all_axes["epw_y"].squeeze(), centers)
    np.testing.assert_array_equal(cfg["other"]["detector_specs"]["electron_wavelength_edges"], edges)
    np.testing.assert_array_equal(cfg["other"]["detector_specs"]["electron_wavelength_centers"], centers)
    assert all_data["e_data"].shape == (2, centers.size)
    assert cfg["other"]["CCDsize"] == (2, centers.size)
    assert cfg["other"]["lamrangE"] == [centers[0], centers[-1]]
    assert cfg["other"]["npts"] == centers.size


def test_angular_irf_can_be_jit_compiled():
    import jax

    irf = AngularIRF(spect_stddev=0.5, ang_stddev=1.0, ang_axis=np.linspace(50.0, 70.0, 6), normalize=0)
    wavelengths = jnp.linspace(400.0, 500.0, 16)
    model = jnp.arange(96.0).reshape(6, 16)

    expected = add_ATS_IRF(irf, wavelengths, model, _instrument_params())[1]
    compiled = jax.jit(lambda values: add_ATS_IRF(irf, wavelengths, values, _instrument_params())[1])(model)

    np.testing.assert_allclose(compiled, expected, rtol=1e-12)


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
@pytest.mark.physics
def test_normalized_spectral_irf_preserves_unresolved_line_area_and_centroid(
    number_of_points,
):
    """P-IRF-01: normalized spectral blur preserves line area and centroid, reducing peak height."""
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


@pytest.mark.physics
def test_normalized_ats_irf_keeps_constant_density_away_from_boundaries():
    """P-IRF-02: a normalized spectral/angular response preserves constant interior density."""
    angles = np.linspace(-5.0, 5.0, 101)
    wavelengths = jnp.linspace(-10.0, 10.0, 201)
    constant = jnp.ones((angles.size, wavelengths.size))
    irf = AngularIRF(spect_stddev=0.5, ang_stddev=0.3, ang_axis=angles, normalize=0)

    _, blurred = add_ATS_IRF(irf, wavelengths, constant, _instrument_params())

    np.testing.assert_allclose(np.asarray(blurred[20:-20, 30:-30]), 1.0, rtol=0, atol=1e-9)


def test_angular_irf_rejects_peak_normalization():
    angles = np.linspace(20.0, 30.0, 3)
    wavelengths = jnp.linspace(500.0, 510.0, 5)
    irf = AngularIRF(spect_stddev=0.5, ang_stddev=0.3, ang_axis=angles, normalize=1)

    with pytest.raises(ValueError, match="Peak normalization is not supported for angular spectra"):
        add_ATS_IRF(irf, wavelengths, jnp.ones((3, 5)), _instrument_params())


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


@pytest.mark.physics
def test_real_nonuniform_angular_irf_preserves_physical_area_and_center():
    """P-IRF-03: nonuniform angular calibration preserves physical area and centroid."""
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


