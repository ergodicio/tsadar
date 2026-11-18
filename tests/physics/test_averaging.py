"""A10: gradient, aperture and species averaging against the averaged analytic spectrum."""

import dataclasses

import numpy as np
import pytest

from tsverify import oracle as oracles
from tsverify.cases import A1_CASES
from tsverify.oracle import Ion, Plasma

from . import harness
from .compare import assert_detector_matches, oracle_detector

pytestmark = pytest.mark.physics

EPW_CASE, _, IAW_CASE, CH_CASE = A1_CASES
GRADIENTS = harness.Gradients(ne=20.0, Te=20.0, points=5)


def _gradient_plasmas(plasma, gradients):
    scales_ne = np.linspace(1 - gradients.ne / 200, 1 + gradients.ne / 200, gradients.points)
    scales_te = np.linspace(1 - gradients.Te / 200, 1 + gradients.Te / 200, gradients.points)
    return [dataclasses.replace(plasma, ne=plasma.ne * a, Te=plasma.Te * b) for a, b in zip(scales_ne, scales_te)]


@pytest.mark.parametrize("case", [EPW_CASE, IAW_CASE], ids=str)
def test_gradient_average_matches_the_averaged_analytic_spectrum(case):
    det = case.detector
    centers, model = harness.detector(case.plasma, det, gradients=GRADIENTS)
    plasmas = _gradient_plasmas(case.plasma, GRADIENTS)
    expected = oracle_detector(
        det, centers, lambda lam: np.mean([oracles.spectral_density(lam, p) for p in plasmas], axis=0)
    )
    assert_detector_matches(f"A10-gradient/{case}", det, centers, model, expected)


def test_zero_gradient_collapses_to_a_single_point():
    det = EPW_CASE.detector
    _, single = harness.detector(EPW_CASE.plasma, det)
    _, flat = harness.detector(EPW_CASE.plasma, det, gradients=harness.Gradients(points=5))
    np.testing.assert_allclose(flat, single, rtol=1e-12, atol=1e-12, err_msg="A10: zero gradient changed the spectrum")


def test_aperture_weighted_sum_matches_the_analytic_one():
    det = EPW_CASE.detector
    angles, weights = (54.0, 60.0, 66.0), (0.25, 0.5, 0.25)
    centers, model = harness.detector(EPW_CASE.plasma, det, aperture=(angles, weights))
    expected = oracle_detector(
        det,
        centers,
        lambda lam: sum(
            w * oracles.spectral_density(lam, dataclasses.replace(EPW_CASE.plasma, angle_deg=a))
            for a, w in zip(angles, weights)
        ),
    )
    assert_detector_matches("A10-aperture", det, centers, model, expected)


@pytest.mark.parametrize(
    "case, same_plasma",
    [
        (CH_CASE, dataclasses.replace(CH_CASE.plasma, ions=CH_CASE.plasma.ions[::-1])),
        (
            IAW_CASE,
            dataclasses.replace(
                IAW_CASE.plasma,
                ions=tuple(dataclasses.replace(IAW_CASE.plasma.ions[0], fract=f) for f in (0.3, 0.7)),
            ),
        ),
    ],
    ids=["species-order", "species-split"],
)
def test_the_same_plasma_written_differently_gives_the_same_spectrum(case, same_plasma):
    _, reference = harness.detector(case.plasma, case.detector)
    _, rewritten = harness.detector(same_plasma, case.detector)
    np.testing.assert_allclose(
        rewritten, reference, rtol=1e-10, atol=1e-10 * reference.max(), err_msg=f"A10: {case} rewritten"
    )
