"""L2: the production 1-D ``FormFactor`` against the analytic spectral density.

The oracle evaluates every susceptibility with the Faddeeva function, so this checks
tsadar's tabulated Z', the rational principal-value quadrature of the sampled EDF,
the sampled Landau-damping derivative, and the k, omega, Jacobian bookkeeping, all on
an absolute scale.
"""

import pytest

from .. import oracles
from . import harness
from .cases import A1_CASES

pytestmark = pytest.mark.physics

# Measured on the fixed code (2048 wavelengths, 513-point EDF): Linf <= 5.7e-4, L1 <= 3.0e-4. Before the
# centred f' and the Faddeeva Z' they were up to 2.2e-2 and 8.8e-3.
LINF_TOL = 1.0e-3
L1_TOL = 5.0e-4


@pytest.mark.parametrize("case", A1_CASES, ids=str)
def test_form_factor_matches_analytic_spectral_density(case):
    lam, model = harness.form_factor(case.plasma, case.detector.lam_range, npts=2048)
    expected = oracles.spectral_density(lam, case.plasma)
    linf, l1 = oracles.relative_errors(model, expected)
    assert linf < LINF_TOL and l1 < L1_TOL, (
        f"{case.problem}-L2/{case}: FormFactor departs from the analytic S(k, w): "
        f"peak-normalized Linf={linf:.2e} (tol {LINF_TOL:.0e}), L1={l1:.2e} (tol {L1_TOL:.0e})"
    )
