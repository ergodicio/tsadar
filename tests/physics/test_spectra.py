"""L2: FormFactor against the analytic spectral density."""

import pytest

from .. import oracles
from . import harness
from .cases import SPECTRUM_CASES

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("npts", [256, 2048])
@pytest.mark.parametrize("case", SPECTRUM_CASES, ids=str)
def test_form_factor_matches_analytic_spectral_density(case, npts):
    lam, model = harness.form_factor(case.plasma, case.detector.lam_range, npts=npts)
    expected = oracles.spectral_density(lam, case.plasma)
    linf, l1 = oracles.relative_errors(model, expected)
    linf_tol, l1_tol = case.l2_tol
    assert linf < linf_tol and l1 < l1_tol, (
        f"{case.problem}-L2/{case}: FormFactor departs from the analytic S(k, w): "
        f"peak-normalized Linf={linf:.2e} (tol {linf_tol:.1e}), L1={l1:.2e} (tol {l1_tol:.1e})"
    )
