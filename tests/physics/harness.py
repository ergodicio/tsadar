"""Drive tsadar from an oracle ``Plasma`` at each depth of the code.

This is the only place in the physics suite that knows how tsadar's inputs are laid
out. The tests compare its outputs with :mod:`tests.oracles`.
"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from jax import config

config.update("jax_enable_x64", True)
from jax import numpy as jnp

from tsadar.core.modules.ts_params import ThomsonParams
from tsadar.core.physics.form_factor import FormFactor
from tsadar.core.thomson_diagnostic import ThomsonScatteringDiagnostic

from ..oracles import GaussianComponent, Plasma, mixture_density

DECK = Path(__file__).parents[1] / "configs" / "1d-defaults.yaml"
MAXWELLIAN = (GaussianComponent(),)
# Numeric EDFs are sampled inside the form factor's |v| <= 8.2 table.
NUMERIC_V = np.linspace(-8.0, 8.0, 513)


# --------------------------------------------------------------------------- L2: FormFactor


def form_factor_params(plasma: Plasma, *, vmax=10.0, nv=513):
    """FormFactor parameter dict for ``plasma``.

    The electron EDF is sampled on ``[-vmax, vmax]``. A non-Maxwellian ion reaches FormFactor the way a
    numeric ion EDF does, as ``fi`` on the grid ``vi``.
    """
    v = np.linspace(-vmax, vmax, nv)
    params = {
        "electron": {
            "ne": plasma.ne,
            "Te": plasma.Te,
            "v": jnp.asarray(v),
            "fe": jnp.asarray(mixture_density(v, plasma.electrons)),
        },
        "general": {"lam": plasma.lam, "ud": plasma.ud, "ne_gradient": 0.0, "Te_gradient": 0.0},
    }
    for i, ion in enumerate(plasma.ions):
        params[f"ion-{i + 1}"] = {"A": ion.A, "Z": ion.Z, "Ti": ion.Ti, "fract": ion.fract, "Va": ion.Va}
        if ion.components != MAXWELLIAN:
            params[f"ion-{i + 1}"] |= {
                "vi": jnp.asarray(NUMERIC_V),
                "fi": jnp.asarray(mixture_density(NUMERIC_V, ion.components)),
            }
    return params


def form_factor(plasma: Plasma, lam_range, npts, **edf):
    """Return ``(wavelength_nm, S)`` from the production 1-D ``FormFactor``."""
    ff = FormFactor(
        list(lam_range),
        npts=npts,
        lam_shift=0.0,
        scattering_angles={"sa": np.array([plasma.angle_deg]), "weights": np.ones((1, 1))},
        num_grad_points=1,
        ud_ang=None,
        va_ang=None,
        calc_gain={"calc": False},
    )
    spectrum, lam_cm = ff(form_factor_params(plasma, **edf))
    return np.squeeze(np.asarray(lam_cm)) * 1.0e7, np.squeeze(np.asarray(spectrum))


# --------------------------------------------------------------------------- L3: full diagnostic


@dataclass(frozen=True)
class Detector:
    """One spectrometer channel of the production diagnostic.

    ``lam_range`` is the first and last pixel centre. ``amp_ranges`` are the fit windows
    that set the amplitude; for the ion channel they straddle the excluded line centre.
    """

    channel: str  # "EPW" or "IAW"
    lam_range: tuple[float, float]
    pixels: int
    points_per_pixel: int
    sigma: float
    amp_ranges: tuple[tuple[float, float], ...]
    notch: tuple[float, float, float] | None = None  # (center, width, optical density), EPW only
    amplitude: float = 100.0


EPW_DETECTOR = Detector(
    channel="EPW",
    lam_range=(440.0, 630.0),
    pixels=128,
    points_per_pixel=8,
    sigma=1.3,
    amp_ranges=((442.0, 522.0), (531.0, 628.0)),
    notch=(526.5, 24.0, 4.0),
)

IAW_DETECTOR = Detector(
    channel="IAW",
    lam_range=(524.5, 528.5),
    pixels=128,
    points_per_pixel=8,
    sigma=0.05,
    amp_ranges=((524.6, 526.4), (526.6, 528.4)),
)


@lru_cache(maxsize=1)
def _deck():
    return yaml.safe_load(DECK.read_text())


def _numeric_edf(components, folder, name):
    """Write a Gaussian-mixture EDF as the csv that a ``type: numeric`` distribution loads."""
    path = Path(folder) / f"{name}.csv"
    np.savetxt(
        path,
        np.column_stack([NUMERIC_V, mixture_density(NUMERIC_V, components)]),
        delimiter=",",
        header="v,f",
        comments="",
    )
    return {"dim": 1, "type": "numeric", "file": str(path), "active": False, "nvx": NUMERIC_V.size, "params": {}}


def _ion_block(ion):
    return {
        "Ti": {"val": ion.Ti, "active": False, "same": False, "lb": 1e-3, "ub": 10.0},
        "Z": {"val": ion.Z, "active": False, "lb": 0.5, "ub": 100.0},
        "A": {"val": ion.A, "active": False},
        "fract": {"val": ion.fract, "active": False},
        "Va": {"val": ion.Va, "angle": 0.0, "active": False, "lb": -100.0, "ub": 100.0},
    }


@dataclass(frozen=True)
class Gradients:
    """tsadar's gradient model: ``ne`` and ``Te`` scaled by ``linspace(1 - g/200, 1 + g/200, points)``,
    with the spectra averaged with equal weights. ``ne`` and ``Te`` are in percent, as in the deck."""

    ne: float = 0.0
    Te: float = 0.0
    points: int = 1


def diagnostic_config(plasma: Plasma, det: Detector, edf_folder=None, gradients: Gradients = Gradients()):
    """A complete 1-D input deck that makes the diagnostic model ``plasma`` on ``det``.

    A Maxwellian species uses tsadar's own EDF (DLM m = 2 for electrons, the built-in ion
    Maxwellian). Any other EDF is written to a csv in ``edf_folder`` and loaded as
    ``type: numeric``, the user-facing path for a given distribution.
    """
    cfg = deepcopy(_deck())
    p = cfg["parameters"]
    p["electron"]["Te"]["val"] = plasma.Te
    p["electron"]["ne"]["val"] = plasma.ne
    if plasma.electrons == MAXWELLIAN:
        p["electron"]["fe"] = {
            "dim": 1,
            "type": "dlm",
            "active": False,
            "nvx": 256,
            "params": {"m": {"val": 2.0, "lb": 2.0, "ub": 5.0, "matte": False}},
        }
    else:
        p["electron"]["fe"] = _numeric_edf(plasma.electrons, edf_folder, "fe")
    for key in [k for k in p if k.startswith("ion-")]:
        del p[key]
    for i, ion in enumerate(plasma.ions):
        p[f"ion-{i + 1}"] = _ion_block(ion)
        if ion.components != MAXWELLIAN:
            p[f"ion-{i + 1}"]["fi"] = _numeric_edf(ion.components, edf_folder, f"fi-{i + 1}")
    p["general"]["lam"]["val"] = plasma.lam
    p["general"]["ud"]["val"] = plasma.ud
    for key, value in (("ne_gradient", gradients.ne), ("Te_gradient", gradients.Te)):
        p["general"][key].update(val=value, num_grad_points=gradients.points)

    epw = det.channel == "EPW"
    lam_key = "lamrangE" if epw else "lamrangI"
    other = cfg["other"]
    other["CCDsize"] = [det.pixels, det.pixels]
    other["points_per_pixel"] = det.points_per_pixel
    other["npts"] = det.pixels * det.points_per_pixel
    other[lam_key] = list(det.lam_range)
    if epw:
        other["lamrangI"] = [524.5, 528.5]
        other["detector_specs"]["widIRF"]["spect_stddev_ele"] = det.sigma
        center, width, od = det.notch if det.notch else (526.5, 24.0, 4.0)
        other["iawfilter"] = [det.notch is not None, od, width, center]
    else:
        other["lamrangE"] = [440.0, 630.0]
        other["detector_specs"]["widIRF"]["spect_stddev_ion"] = det.sigma
        other["iawfilter"] = [False, 4, 24, 526.5]
    other["iawoff"] = 0

    data = cfg["data"]
    data.update(fit_EPWb=epw, fit_EPWr=epw, fit_IAW=not epw, load_ele_spec=epw, load_ion_spec=not epw)
    rng = data["fit_rng"]
    if epw:
        (rng["blue_min"], rng["blue_max"]), (rng["red_min"], rng["red_max"]) = det.amp_ranges
    else:
        (rng["iaw_min"], rng["iaw_cf_min"]), (rng["iaw_cf_max"], rng["iaw_max"]) = det.amp_ranges
    return cfg


def build_diagnostic(plasma: Plasma, det: Detector, *, gradients: Gradients = Gradients(), aperture=None):
    """Return ``(diagnostic, params, batch)`` that model ``plasma`` on ``det`` through the production path.

    ``aperture`` is ``(angles_deg, weights)``: the 1-D model sums the spectra at those scattering angles with
    those weights. By default the plasma's single angle with weight one.
    """
    angles_deg, weights = aperture if aperture is not None else ([plasma.angle_deg], [1.0])
    angles = {"sa": np.asarray(angles_deg, dtype=float), "weights": np.asarray(weights, dtype=float)[None, :]}
    with tempfile.TemporaryDirectory() as folder:
        cfg = diagnostic_config(plasma, det, folder, gradients)
        diag = ThomsonScatteringDiagnostic(cfg, angles)
        params = ThomsonParams(cfg["parameters"], num_params=1, activate=False)
    n = det.pixels
    batch = {
        "e_data": jnp.ones((1, n)),
        "i_data": jnp.ones((1, n)),
        "noise_e": jnp.zeros((1, n)),
        "noise_i": jnp.zeros((1, n)),
        "e_amps": jnp.full(1, det.amplitude),
        "i_amps": jnp.full(1, det.amplitude),
    }
    return diag, params, batch


def detector(plasma: Plasma, det: Detector, **geometry):
    """Return ``(pixel_centers_nm, detector_spectrum)`` from ``ThomsonScatteringDiagnostic``.

    ``geometry`` is passed to :func:`build_diagnostic` (``gradients``, ``aperture``).
    """
    diag, params, batch = build_diagnostic(plasma, det, **geometry)
    out = diag(params, batch)
    channel = 0 if det.channel == "EPW" else 1
    return np.asarray(out[channel + 2]).ravel(), np.asarray(out[channel]).ravel()
