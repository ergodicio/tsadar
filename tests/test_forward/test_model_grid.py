import numpy as np
import pytest

from tsadar.core.physics.form_factor import pixel_centered_grid
from tsadar.core.physics.generate_spectra import points_per_detector_pixel


def _config(spectype, ccd_size, points_per_pixel):
    return {
        "other": {
            "extraoptions": {"spectype": spectype},
            "CCDsize": ccd_size,
            "points_per_pixel": points_per_pixel,
            "npts": int(ccd_size[1] * points_per_pixel),
        }
    }


@pytest.mark.parametrize("points_per_pixel", [1, 2, 5])
def test_grid_groups_average_to_the_pixel_centers(points_per_pixel):
    centers = np.linspace(400.0, 700.0, 64)
    grid = np.asarray(pixel_centered_grid([centers[0], centers[-1]], 64 * points_per_pixel, points_per_pixel))

    np.testing.assert_allclose(grid.reshape(64, points_per_pixel).mean(axis=1), centers, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.diff(grid), (centers[1] - centers[0]) / points_per_pixel, rtol=0, atol=1e-9)


@pytest.mark.parametrize("points_per_pixel", [1, 2, 5])
def test_square_detector_reduces_by_points_per_pixel(points_per_pixel):
    assert points_per_detector_pixel(_config("temporal", [1024, 1024], points_per_pixel)) == points_per_pixel


def test_non_square_detector_uses_the_actual_reduction_factor():
    # npts = 1024 * 2 = 2048 already has one point per wavelength pixel, so the grid must not be extended
    config = _config("temporal", [2048, 1024], 2)
    factor = points_per_detector_pixel(config)
    assert factor == 1

    grid = np.asarray(pixel_centered_grid([400.0, 700.0], config["other"]["npts"], factor))
    np.testing.assert_allclose(grid, np.linspace(400.0, 700.0, 2048), rtol=0, atol=1e-9)


def test_angular_spectra_reduce_by_points_per_pixel():
    # reduced angular data carry CCDsize as (angle units, wavelength units)
    assert points_per_detector_pixel(_config("angular_full", [103, 205], 2)) == 2
