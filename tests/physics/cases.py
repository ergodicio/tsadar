"""Analytic physics cases shared by every depth of the physics suite.

A case is data: a plasma the oracle can evaluate exactly, and the detector channel it
is observed on. Adding a case adds a kernel/spectrum/detector check at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..oracles import Ion, Plasma
from .harness import EPW_DETECTOR, IAW_DETECTOR, Detector


@dataclass(frozen=True)
class PhysicsCase:
    name: str
    plasma: Plasma
    detector: Detector
    problem: str  # analytic-problem ID from TEST_SUITE_AUDIT.md

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
