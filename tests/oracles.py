"""Analytic oracles for the tsadar physics suite.

Nothing here imports tsadar. Every quantity is computed from closed-form plasma
physics with NumPy/SciPy, so a test that compares tsadar against these functions
checks tsadar's algorithms (tabulated Z', rational principal-value quadrature,
sampled EDF derivatives, IRF convolution, pixel reduction, normalization), not a
copy of them.

The same oracle is meant to be evaluated at several depths of the code:

* L2 ``spectral_density``: compare directly with ``FormFactor``.
* L3 ``detector_spectrum``: push the L2 oracle through an independent detector
  model (continuous Gaussian blur, exact pixel-box average, fit-range
  normalization) and compare with ``ThomsonScatteringDiagnostic``. This is the
  plumbing test: a shift, scale, or smearing introduced anywhere between the form
  factor and the detector shows up as a mismatch against physics.

Conventions follow the collisionless Salpeter form factor (Sheffield et al.,
*Plasma Scattering of Electromagnetic Radiation*, 2nd ed., ch. 3 and 5).
Units match tsadar's input deck: ``ne`` in 1e20 cm^-3, temperatures in keV,
flows in 1e6 cm/s (projected on k), wavelengths in nm. The physical constants are
the same CODATA-style numbers tsadar uses; they are inputs, not algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, wofz

C_CM_PER_S = 2.99792458e10
ELECTRON_MASS_KEV = 510.9896
ELECTRON_MASS = ELECTRON_MASS_KEV / C_CM_PER_S**2  # keV s^2 / cm^2
PROTON_MASS = 1836.1 * ELECTRON_MASS
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179e-13
ELECTRON_CHARGE_SQUARED = ELECTRON_MASS * C_CM_PER_S**2 * CLASSICAL_ELECTRON_RADIUS_CM  # keV cm


# --------------------------------------------------------------------------- plasma description


@dataclass(frozen=True)
class Ion:
    """One Maxwellian ion species. ``Va`` is the flow projected on k, in 1e6 cm/s."""

    A: float
    Z: float
    Ti: float
    fract: float = 1.0
    Va: float = 0.0


@dataclass(frozen=True)
class GaussianComponent:
    """One drifting Maxwellian in the electron EDF, in units of v_Te = sqrt(Te/me).

    ``f(x) = weight / (width sqrt(2 pi)) exp(-(x - shift)^2 / (2 width^2))``. A single
    component ``(1, 0, 1)`` is the Maxwellian at temperature ``Te``.
    """

    weight: float = 1.0
    shift: float = 0.0
    width: float = 1.0


@dataclass(frozen=True)
class Plasma:
    """A homogeneous plasma probed at one scattering angle.

    ``ud`` is the electron drift relative to the charge-weighted ion fluid (tsadar's
    definition), in 1e6 cm/s along k. ``electrons`` is a Gaussian mixture; weights
    must sum to one.
    """

    ne: float
    Te: float
    ions: tuple[Ion, ...]
    angle_deg: float = 60.0
    lam: float = 526.5
    ud: float = 0.0
    electrons: tuple[GaussianComponent, ...] = field(default_factory=lambda: (GaussianComponent(),))

    def __post_init__(self):
        total = sum(c.weight for c in self.electrons)
        if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"electron mixture weights must sum to 1, got {total}")


# --------------------------------------------------------------------------- susceptibilities


def plasma_dispersion(zeta):
    r"""Fried–Conte plasma dispersion function ``Z(zeta) = i sqrt(pi) w(zeta)``."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def maxwellian_response(x):
    r"""``1 + zeta Z(zeta)`` with ``zeta = x / sqrt(2)``.

    For a unit-variance Gaussian ``g``, ``-P.V.∫ g'(t)/(t - x) dt - i pi g'(x)`` equals this
    (Landau prescription), so ``chi = response / (k lambda_D)^2``.
    """
    zeta = np.asarray(x) / np.sqrt(2.0)
    return 1.0 + zeta * plasma_dispersion(zeta)


def mixture_susceptibility(x, k_lambda_d, components):
    r"""``chi`` of a Gaussian-mixture EDF at normalized phase velocity ``x`` (units of v_T).

    For a component of weight ``w``, shift ``u`` and width ``s``, substituting
    ``v = u + s t`` gives ``chi_j = w (1 + zeta_j Z(zeta_j)) / (s^2 (k lambda_D)^2)`` with
    ``zeta_j = (x - u) / (sqrt(2) s)``.
    """
    chi = 0.0
    for c in components:
        chi = chi + c.weight * maxwellian_response((x - c.shift) / c.width) / c.width**2
    return chi / k_lambda_d**2


def mixture_density(x, components):
    """The normalized EDF ``f(x)`` (units of v_T) of a Gaussian mixture."""
    f = 0.0
    for c in components:
        f = f + c.weight * np.exp(-0.5 * ((x - c.shift) / c.width) ** 2) / (c.width * np.sqrt(2.0 * np.pi))
    return f


# --------------------------------------------------------------------------- spectral density (L2)


def _angular_frequency(lam_nm):
    return 2.0 * np.pi * C_CM_PER_S / (np.asarray(lam_nm, dtype=float) * 1.0e-7)


def spectral_density(lam_nm, plasma: Plasma, *, terms=False):
    r"""Collisionless Thomson spectral density per unit wavelength, in tsadar's units.

    ``S(k, w) = (2 pi / k) [ |1 + chi_i|^2 f_e(w/k) + |chi_e|^2 sum_s (Z_s^2 n_s/n_e) f_s(w/k) ] / |eps|^2``
    with every ``chi`` from the Faddeeva function. tsadar's output also carries the
    first-order scattered-power factor ``(1 + 2 w / w_L)``, ``r_e^2 n_e`` and the
    ``dw/dlambda`` Jacobian; those are included so the comparison is absolute.

    With ``terms=True`` returns ``(total, electron_term, ion_term, chi_e, chi_i)``.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    ne = plasma.ne * 1.0e20
    omega_pe = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ne / ELECTRON_MASS)

    omega_s = _angular_frequency(lam_nm)
    omega_l = _angular_frequency(plasma.lam)
    omega = omega_s - omega_l
    k_s = np.sqrt(omega_s**2 - omega_pe**2) / C_CM_PER_S
    k_l = np.sqrt(omega_l**2 - omega_pe**2) / C_CM_PER_S
    theta = np.deg2rad(plasma.angle_deg)
    k = np.sqrt(k_s**2 + k_l**2 - 2.0 * k_s * k_l * np.cos(theta))

    z = np.array([ion.Z for ion in plasma.ions], dtype=float)
    fract = np.array([ion.fract for ion in plasma.ions], dtype=float)
    fract = fract / fract.sum()
    zbar = float(np.sum(z * fract))
    flows = np.array([ion.Va for ion in plasma.ions], dtype=float) * 1.0e6

    # ions
    chi_i = 0.0
    ion_term = 0.0
    for ion, frac in zip(plasma.ions, fract):
        mass = ion.A * PROTON_MASS
        n_s = frac * ne / zbar
        omega_ps = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ion.Z**2 * n_s / mass)
        v_t = np.sqrt(ion.Ti / mass)
        x = (omega - k * ion.Va * 1.0e6) / (k * v_t)
        chi_i = chi_i + maxwellian_response(x) / (v_t * k / omega_ps) ** 2
        ion_term = ion_term + (frac * ion.Z**2 / zbar) * np.exp(-0.5 * x**2) / (np.sqrt(2.0 * np.pi) * v_t)

    # electrons: drift is relative to the charge-weighted ion fluid
    electron_flow = float(np.sum(z * fract * flows) / np.sum(z * fract)) + plasma.ud * 1.0e6
    v_te = np.sqrt(plasma.Te / ELECTRON_MASS)
    k_lambda_de = v_te * k / omega_pe
    x_e = (omega - k * electron_flow) / (k * v_te)
    chi_e = mixture_susceptibility(x_e, k_lambda_de, plasma.electrons)
    f_e = mixture_density(x_e, plasma.electrons) / v_te

    eps2 = np.abs(1.0 + chi_e + chi_i) ** 2
    electron_part = np.abs(1.0 + chi_i) ** 2 * f_e / eps2 / k
    ion_part = np.abs(chi_e) ** 2 * ion_term / eps2 / k

    # tsadar's absolute scale: (1 + 2 w/w_L) r_e^2 n_e, then dw/dlambda = 2 pi c / lambda^2
    scale = (1.0 + 2.0 * omega / omega_l) * CLASSICAL_ELECTRON_RADIUS_CM**2 * ne
    jacobian = 2.0 * np.pi * C_CM_PER_S / (lam_nm * 1.0e-7) ** 2
    electron_part = electron_part * scale * jacobian
    ion_part = ion_part * scale * jacobian
    total = electron_part + ion_part
    if terms:
        return total, electron_part, ion_part, chi_e, chi_i
    return total


# --------------------------------------------------------------------------- detector (L3)


def pixel_edges(centers):
    """Edges of a uniform pixel axis given its centers."""
    centers = np.asarray(centers, dtype=float)
    d = np.diff(centers)
    if not np.allclose(d, d[0], rtol=1e-9, atol=0):
        raise ValueError("pixel_edges expects uniformly spaced centers")
    return np.concatenate([centers - d[0] / 2.0, [centers[-1] + d[0] / 2.0]])


def detector_spectrum(spectrum, centers, sigma, *, transmission=None, samples_per_sigma=40, pad_sigmas=8.0):
    r"""Pixel means of the Gaussian-blurred, filtered continuous spectrum.

    ``D_j = (1/Δ_j) ∫_{e_j}^{e_{j+1}} [(T S) * G_sigma](y) dy
         = (1/Δ_j) ∫ T(x) S(x) [Phi((e_{j+1}-x)/sigma) - Phi((e_j-x)/sigma)] dx``.

    The outer integral is exact (normal CDF); the inner one is a fine trapezoid on a
    grid far finer than tsadar's. ``spectrum`` and ``transmission`` are callables of
    wavelength in nm.
    """
    edges = pixel_edges(centers)
    pixel = edges[1] - edges[0]
    step = min(sigma / samples_per_sigma, pixel / 400.0)
    lo, hi = edges[0] - pad_sigmas * sigma, edges[-1] + pad_sigmas * sigma
    fine = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    source = spectrum(fine)
    if transmission is not None:
        source = source * transmission(fine)
    weights = np.full(fine.size, fine[1] - fine[0])
    weights[[0, -1]] *= 0.5
    weighted = source * weights
    # cumulative response at every edge: I(e) = ∫ S(x) Phi((e - x)/sigma) dx
    cumulative = np.array([np.dot(weighted, ndtr((e - fine) / sigma)) for e in edges])
    return np.diff(cumulative) / np.diff(edges)


def rectangular_notch(center, width, optical_density):
    """Transmission ``10^-OD`` inside ``|lambda - center| < width/2``, one elsewhere."""

    def transmission(lam):
        lam = np.asarray(lam)
        return np.where(np.abs(lam - center) < width / 2.0, 10.0 ** (-optical_density), 1.0)

    return transmission


def normalize_in_ranges(centers, values, ranges, amplitude=100.0):
    """Scale so the maximum over the pixels inside ``ranges`` equals ``amplitude``.

    This is the fit-range amplitude convention of tsadar's detector model
    (``detector_specs.norm == 0``): data and model are both scaled by their maximum
    inside the fit windows, so an unfitted feature never sets the scale.
    """
    centers = np.asarray(centers)
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in ranges:
        mask |= (centers > low) & (centers < high)
    return amplitude * np.asarray(values) / np.max(np.asarray(values)[mask])


# --------------------------------------------------------------------------- comparisons


def relative_errors(actual, expected, mask=None):
    """Peak-normalized L-infinity and relative L1 errors, optionally restricted to ``mask``."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if mask is not None:
        actual, expected = actual[mask], expected[mask]
    diff = np.abs(actual - expected)
    return float(diff.max() / np.abs(expected).max()), float(diff.sum() / np.abs(expected).sum())


def peak_wavelength(lam, values, window):
    """Three-point parabolic peak inside ``window = (lo, hi)``."""
    lam = np.asarray(lam)
    values = np.asarray(values)
    idx = np.flatnonzero((lam > window[0]) & (lam < window[1]))
    i = idx[np.argmax(values[idx])]
    if i in (idx[0], idx[-1]):
        raise ValueError(f"peak sits on the edge of window {window}")
    left, mid, right = values[i - 1 : i + 2]
    return float(lam[i] + 0.5 * (left - right) / (left - 2.0 * mid + right) * (lam[i + 1] - lam[i]))
