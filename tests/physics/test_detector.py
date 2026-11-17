"""L3: the full ``ThomsonScatteringDiagnostic`` against the analytic detector spectrum.

The oracle side takes the same analytic S(k, w) as L2 and pushes it through an
independent detector model (see compare.py). Anything between the form factor and
the detector that shifts, scales, or smears the spectrum (IRF centring, pixel
grouping, the oversampled grid, the notch, the amplitude rule) fails here against
physics rather than against a frozen array.
"""

import numpy as np
import pytest

from .. import oracles
from . import harness
from .cases import SPECTRUM_CASES
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
