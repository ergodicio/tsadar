"""Analytic physics cases; each is checked at every depth (kernel, FormFactor, detector)."""

from __future__ import annotations

from dataclasses import dataclass

from ..oracles import GaussianComponent as G
from ..oracles import Ion, KappaEDF, Plasma
from .harness import EPW_DETECTOR, IAW_DETECTOR, Detector


@dataclass(frozen=True)
class PhysicsCase:
    name: str
    plasma: Plasma
    detector: Detector
    problem: str
    l2_tol: tuple[float, float] = (1.0e-3, 5.0e-4)

    def __str__(self):
        return self.name


HYDROGEN = Ion(A=1.0, Z=1.0, Ti=0.08)

A1_CASES = [
    PhysicsCase("A1-epw-hydrogen", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,)), EPW_DETECTOR, "A1"),
    PhysicsCase("A1-epw-electron-drift", Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), ud=60.0), EPW_DETECTOR, "A1"),
    PhysicsCase(
        "A1-iaw-hydrogen-flow",
        Plasma(ne=0.2, Te=0.6, ions=(Ion(A=1.0, Z=1.0, Ti=0.08, Va=1.5),)),
        IAW_DETECTOR,
        "A1",
    ),
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

A3_L2_TOL = (2.5e-3, 1.5e-3)
A5_L2_TOL = (1.0e-3, 5.0e-4)
A3_CASES = [
    PhysicsCase(
        "A3-epw-two-temperature",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.85, 0.0, 1.0), G(0.15, 0.0, 2.0))),
        EPW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
    PhysicsCase(
        "A3-epw-hot-tail-shoulder",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=(G(0.9, 0.0, 1.0), G(0.1, 3.0, 0.8))),
        EPW_DETECTOR,
        "A3",
        A3_L2_TOL,
    ),
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

A5_CASES = [
    PhysicsCase(
        f"A5-{det.channel.lower()}-kappa-{kappa}",
        Plasma(ne=0.2, Te=0.6, ions=(HYDROGEN,), electrons=KappaEDF(float(kappa))),
        det,
        "A5",
        A5_L2_TOL,
    )
    for kappa in (3, 5)
    for det in (EPW_DETECTOR, IAW_DETECTOR)
]

SPECTRUM_CASES = A1_CASES + A3_CASES + A5_CASES
