"""L3: the full diagnostic against the analytic detector spectrum."""

import numpy as np
import pytest

from tsverify import oracle as oracles
from tsverify.cases import SPECTRUM_CASES

from . import harness
from .compare import assert_detector_matches, oracle_detector

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_diagnostic_matches_analytic_detector_spectrum(case):
    det = case.detector
    centers, model = harness.detector(case.plasma, det)
    np.testing.assert_allclose(
        centers,
        np.linspace(*det.lam_range, det.pixels),
        rtol=0,
        atol=1e-9,
        err_msg=f"{case}: detector pixel centres moved",
    )
    expected = oracle_detector(det, centers, lambda lam: oracles.spectral_density(lam, case.plasma))
    assert_detector_matches(f"{case.problem}-L3/{case}", det, centers, model, expected)
