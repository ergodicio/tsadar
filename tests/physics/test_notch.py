"""A13: the detailed spectral breakdown agrees with the fitted model.

The notch filter itself is not modelled exactly (a rectangular attenuation applied on
the sampled wavelength grid) and its band is excluded from every fit, so no test
asserts its shape: the detector comparisons in compare.py leave out the pixels within
two IRF widths of its edges. What must hold is that ``spectrum_breakdown``, which the
postprocessor plots, returns the same detector spectrum as the forward pass.
"""

import numpy as np
import pytest

from . import harness
from .cases import A1_CASES

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("case", [A1_CASES[0], A1_CASES[2]], ids=str)
def test_breakdown_reproduces_the_model_spectrum(case):
    diag, params, batch = harness.build_diagnostic(case.plasma, case.detector)
    channel = 0 if case.detector.channel == "EPW" else 1
    model = np.asarray(diag(params, batch)[channel])
    breakdown = np.asarray(diag.spectrum_breakdown(params, batch)[channel])
    np.testing.assert_allclose(breakdown, model, rtol=1e-12, atol=1e-12, err_msg=f"A13/{case}: breakdown != model")
