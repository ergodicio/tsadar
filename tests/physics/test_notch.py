"""A13: the rectangular notch filter in the EPW channel.

The oracle applies the notch to the continuous spectrum, transmission 10^-OD inside
the band, before the Gaussian response. tsadar applies it to its sampled wavelength
grid, so each band edge is resolved only to a grid step: the error at the edges is a
discretization error and must shrink as the grid is refined. The detailed breakdown
used for plotting must agree with the fitted model and use the configured
attenuation.
"""

import dataclasses

import numpy as np
import pytest

from .. import oracles
from . import harness
from .cases import A1_CASES

pytestmark = pytest.mark.physics

CASE = A1_CASES[0]  # Maxwellian hydrogen EPW


def _edge_error(points_per_pixel):
    det = dataclasses.replace(CASE.detector, points_per_pixel=points_per_pixel)
    centers, model = harness.detector(CASE.plasma, det)
    expected = oracles.normalize_in_ranges(
        centers,
        oracles.detector_spectrum(
            lambda lam: oracles.spectral_density(lam, CASE.plasma),
            centers,
            det.sigma,
            transmission=oracles.rectangular_notch(*det.notch),
        ),
        det.amp_ranges,
        det.amplitude,
    )
    in_fit = np.zeros(centers.shape, dtype=bool)
    for low, high in det.amp_ranges:
        in_fit |= (centers > low) & (centers < high)
    return oracles.relative_errors(model, expected, in_fit)[0]


def test_notch_edges_converge_with_the_wavelength_grid():
    # measured 3.3e-3 at 8 points per pixel and 1.5e-3 at 16
    coarse, fine = _edge_error(8), _edge_error(16)
    assert coarse < 5.0e-3, f"A13: notch-edge error {coarse:.2e} at 8 points per pixel (tol 5e-3)"
    assert fine < 0.6 * coarse, (
        f"A13: notch-edge error does not converge with the grid: {coarse:.2e} at 8 points per pixel, "
        f"{fine:.2e} at 16"
    )


def test_breakdown_reproduces_the_model_and_the_configured_attenuation():
    det = CASE.detector
    diag, params, batch = harness.build_diagnostic(CASE.plasma, det)
    model = np.asarray(diag(params, batch)[0])
    breakdown = diag.spectrum_breakdown(params, batch)
    np.testing.assert_allclose(
        np.asarray(breakdown[0]), model, rtol=1e-12, atol=1e-12, err_msg="A13: breakdown spectrum != model"
    )

    # the raw spectrum of the breakdown, with and without the notch: their ratio is the transmission
    open_diag, open_params, _ = harness.build_diagnostic(CASE.plasma, dataclasses.replace(det, notch=None))
    filtered, unfiltered = (
        np.asarray(d.spectrum_breakdown(p, batch)[2]).ravel() for d, p in ((diag, params), (open_diag, open_params))
    )
    lam = np.asarray(breakdown[8]).ravel()
    center, width, optical_density = det.notch
    inside = np.abs(lam - center) < width / 2 - 0.01
    outside = np.abs(lam - center) > width / 2 + 0.01
    ratio = filtered / unfiltered
    # each breakdown is scaled to its own maximum, so compare the in-band ratio with the out-of-band one
    np.testing.assert_allclose(
        ratio[inside] / np.median(ratio[outside]),
        10.0**-optical_density,
        rtol=1e-9,
        err_msg="A13: the detailed spectrum does not use the configured notch attenuation",
    )
