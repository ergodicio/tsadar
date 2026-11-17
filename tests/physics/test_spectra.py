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

# Measured on the fixed code (513-point EDF): Linf <= 6.2e-4, L1 <= 3.2e-4, the same at 256 and 2048
# wavelengths. Before f' was taken on the velocity grid and Z' from the Faddeeva function they were up to
# 2.2e-2 and 8.8e-3 at 2048 wavelengths, and grew as the wavelength grid coarsened.
LINF_TOL = 1.0e-3
L1_TOL = 5.0e-4


@pytest.mark.parametrize("npts", [256, 2048])
@pytest.mark.parametrize("case", A1_CASES, ids=str)
def test_form_factor_matches_analytic_spectral_density(case, npts):
    lam, model = harness.form_factor(case.plasma, case.detector.lam_range, npts=npts)
    expected = oracles.spectral_density(lam, case.plasma)
    linf, l1 = oracles.relative_errors(model, expected)
    assert linf < LINF_TOL and l1 < L1_TOL, (
        f"{case.problem}-L2/{case}: FormFactor departs from the analytic S(k, w): "
        f"peak-normalized Linf={linf:.2e} (tol {LINF_TOL:.0e}), L1={l1:.2e} (tol {L1_TOL:.0e})"
    )
