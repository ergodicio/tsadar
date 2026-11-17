"""Analytic physics cases shared by every depth of the physics suite.

A case is data: a plasma the oracle can evaluate exactly, and the detector channel it
is observed on. Adding a case adds a kernel/spectrum/detector check at once.

Scope: the well-resolved regime only. Every case keeps alpha = 1/(k lambda_De) <= 3.5 at
the features and has no nearly undamped resonance (min |eps| ~ 0.25 or more). Beyond that,
tsadar's 1-D form factor is accurate only when the resonance is resolved: the ~1e-4
susceptibility error of its fixed internal velocity grid is amplified by 1/|eps|^2
(measured 1-8 % of the EPW peak for k lambda_De 0.15-0.25, and 0.6 % for a positive-slope
bump-on-tail at alpha ~ 2).

NOTE FOR THE NEXT AGENT: when the resonance-aware path is ready, expand these tests to
high phase velocity (alpha > 3.5, weakly damped EPWs, positive-slope bumps) using the
resonance quadrature (peak refinement) together with the collisional model developed on
the arts2d-logspace branch; the plain sampled form factor is not expected to pass there.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..oracles import GaussianComponent as G
from ..oracles import Ion, Plasma
from .harness import EPW_DETECTOR, IAW_DETECTOR, Detector


@dataclass(frozen=True)
class PhysicsCase:
    name: str
    plasma: Plasma
    detector: Detector
    problem: str  # analytic-problem ID from TEST_SUITE_AUDIT.md
    # FormFactor (L2) tolerance: peak-normalized Linf and relative L1
    l2_tol: tuple[float, float] = (1.0e-3, 5.0e-4)

    def __str__(self):
        return self.name


HYDROGEN = Ion(A=1.0, Z=1.0, Ti=0.08)

# A1: Maxwellian electrons and Maxwellian ions, chi from the Faddeeva function.
A1_CASES = [
    PhysicsCase("A1-epw-hydrogen", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), EPW_DETECTOR, "A1"),
    # electron drift relative to the ion fluid: asymmetric EPW wings
    PhysicsCase("A1-epw-electron-drift", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), ud=60.0), EPW_DETECTOR, "A1"),
    PhysicsCase(
        "A1-iaw-hydrogen-flow",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=1.5),)),
        IAW_DETECTOR,
        "A1",
    ),
    # two ion species with a relative drift: fast and slow acoustic modes, charge-weighted electron flow
    PhysicsCase(
        "A1-iaw-CH-relative-drift",
        Plasma(
            ne=0.5,
            Te=1.0,
            ions=(Ion(A=12.0, Z=6.0, Ti=0.2, fract=0.5, Va=1.0), Ion(A=1.0, Z=1.0, Ti=0.2, fract=0.5, Va=3.0)),
        ),
        IAW_DETECTOR,
        "A1",
    ),
]

# A3: Gaussian-mixture EDFs, chi = sum of shifted, scaled Faddeeva terms. tsadar receives them as
# numeric distributions (csv, ``type: numeric``). Narrow components sit on the floor of tsadar's internal
# velocity grid (1024 points): L2 errors up to 1.8e-3 for the two-humped ions, falling to 1.2e-4 with 4096.
A3_L2_TOL = (2.5e-3, 1.5e-3)
A3_CASES = [
    PhysicsCase(
        "A3-epw-two-temperature",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.85, 0.0, 1.0), G(0.15, 0.0, 2.0))),
        EPW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
    # a hot tail: 10 % of the electrons in a broad component at 3 v_Te. A positive-slope bump-on-tail leaves
    # the resonance nearly undamped (min |eps| 0.09) and belongs to the high-resolution work noted above.
    PhysicsCase(
        "A3-epw-hot-tail-shoulder",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.9, 0.0, 1.0), G(0.1, 3.0, 0.8))),
        EPW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
    # a cold beam carrying a return current: asymmetric EPW and IAW
    *[
        PhysicsCase(
            f"A3-{det.channel.lower()}-return-current",
            Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.8, 0.4, 1.0), G(0.2, -1.6, 0.8))),
            det,
            "A3",
            A3_L2_TOL,
        )
        for det in (EPW_DETECTOR, IAW_DETECTOR)
    ],
    PhysicsCase(
        "A3-epw-flat-top",
        Plasma(
            ne=0.2,
            Te=0.6,
            ions=(HYDROGEN,),
            electrons=(G(1 / 3, -1.0, 0.8), G(1 / 3, 0.0, 0.8), G(1 / 3, 1.0, 0.8)),
        ),
        EPW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
    PhysicsCase(
        "A3-iaw-ion-beam",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, components=(G(0.9, 0.0, 1.0), G(0.1, 4.0, 0.7))),)),
        IAW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
    PhysicsCase(
        "A3-iaw-two-humped-ions",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, components=(G(0.5, -0.8, 0.7), G(0.5, 0.8, 0.7))),)),
        IAW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
]

SPECTRUM_CASES = A1_CASES + A3_CASES
