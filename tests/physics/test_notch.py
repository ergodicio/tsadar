"""A13: ``spectrum_breakdown`` returns the model spectrum."""

import numpy as np
import pytest

from tsverify.cases import A1_CASES

from . import harness

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("case", [A1_CASES[0], A1_CASES[2]], ids=str)
def test_breakdown_reproduces_the_model_spectrum(case):
    diag, params, batch = harness.build_diagnostic(case.plasma, case.detector)
    channel = 0 if case.detector.channel == "EPW" else 1
    model = np.asarray(diag(params, batch)[channel])
    breakdown = np.asarray(diag.spectrum_breakdown(params, batch)[channel])
    np.testing.assert_allclose(breakdown, model, rtol=1e-12, atol=1e-12, err_msg=f"A13/{case}: breakdown != model")
