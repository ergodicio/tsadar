"""Analytic reference spectra for the physics tests, computed with NumPy and SciPy only (no tsadar imports).

Units follow tsadar's deck: ne in 1e20 cm^-3, temperatures in keV, flows in 1e6 cm/s along k, wavelengths in nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, wofz

C_CM_PER_S = 2.99792458e10
ELECTRON_MASS_KEV = 510.9896
ELECTRON_MASS = ELECTRON_MASS_KEV / C_CM_PER_S**2
PROTON_MASS = 1836.1 * ELECTRON_MASS
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179e-13
ELECTRON_CHARGE_SQUARED = ELECTRON_MASS * C_CM_PER_S**2 * CLASSICAL_ELECTRON_RADIUS_CM


@dataclass(frozen=True)
class GaussianComponent:
    """One drifting Maxwellian of an EDF, in units of the species' v_T: weight, shift and width."""

    weight: float = 1.0
    shift: float = 0.0
    width: float = 1.0


@dataclass(frozen=True)
class Ion:
    """One ion species. ``Va`` is the flow projected on k, in 1e6 cm/s; ``components`` is its EDF as a
    Gaussian mixture in units of sqrt(Ti/mi) (default: Maxwellian)."""

    A: float
    Z: float
    Ti: float
    fract: float = 1.0
    Va: float = 0.0
    components: tuple[GaussianComponent, ...] = (GaussianComponent(),)


@dataclass(frozen=True)
class Plasma:
    """A homogeneous plasma at one scattering angle. ``ud`` is the electron drift relative to the charge-weighted
    ion fluid; ``electrons`` is a Gaussian mixture or a :class:`QuadratureEDF`.
    """

    ne: float
    Te: float
    ions: tuple[Ion, ...]
    angle_deg: float = 60.0
    lam: float = 526.5
    ud: float = 0.0
    electrons: tuple[GaussianComponent, ...] = field(default_factory=lambda: (GaussianComponent(),))

    def __post_init__(self):
        for label, components in [("electron", self.electrons)] + [(f"ion {i}", ion.components) for i, ion in enumerate(self.ions)]:
            if isinstance(components, QuadratureEDF):
                continue
            total = sum(c.weight for c in components)
            if not np.isclose(total, 1.0, rtol=0, atol=1e-12):
                raise ValueError(f"{label} mixture weights must sum to 1, got {total}")


def plasma_dispersion(zeta):
    r"""Fried–Conte plasma dispersion function ``Z(zeta) = i sqrt(pi) w(zeta)``."""
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def maxwellian_response(x):
    """``1 + zeta Z(zeta)``, ``zeta = x / sqrt(2)``: ``(k lambda_D)^2 chi`` of a unit-variance Maxwellian."""
    zeta = np.asarray(x) / np.sqrt(2.0)
    return 1.0 + zeta * plasma_dispersion(zeta)


class QuadratureEDF:
    """An EDF with ``(k lambda_D)^2 chi(x) = -P.V. int f'(t)/(t - x) dt - i pi f'(x)`` by numerical quadrature.

    Subclasses define ``density`` and ``derivative`` on unit-variance velocity.
    """

    X_MAX = 25.0
    X_STEP = 0.02
    LIMIT = 200.0

    def density(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def response(self, x):
        x = np.asarray(x, dtype=float)
        table = _quadrature_response_table(self)
        inside = np.abs(x) <= self.X_MAX
        with np.errstate(divide="ignore", invalid="ignore"):
            tail = -1.0 / x**2 - 3.0 / x**4
        real = np.where(inside, table[0](np.clip(x, -self.X_MAX, self.X_MAX)), tail)
        return real - 1j * np.pi * self.derivative(x)


@lru_cache(maxsize=None)
def _quadrature_response_table(edf):
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline

    a, b = -edf.LIMIT, edf.LIMIT

    def principal_value(x):
        fx = edf.derivative(x)
        smooth = quad(lambda t: (edf.derivative(t) - fx) / (t - x), a, b, points=[x], limit=500, epsabs=1e-13)[0]
        return smooth + fx * np.log((b - x) / (x - a))

    grid = np.arange(-edf.X_MAX, edf.X_MAX + edf.X_STEP / 2, edf.X_STEP)
    return (CubicSpline(grid, -np.array([principal_value(x) for x in grid])),)


@dataclass(frozen=True, eq=True)
class GaussianEDF(QuadratureEDF):
    """A unit-variance Maxwellian through the quadrature path: the check that the path reproduces Faddeeva."""

    def density(self, x):
        return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2.0 * np.pi)

    def derivative(self, x):
        return -np.asarray(x) * self.density(x)


@dataclass(frozen=True, eq=True)
class KappaEDF(QuadratureEDF):
    """1-D kappa distribution: a Student-t with ``nu = 2 kappa - 1``, scaled to unit variance."""

    kappa: float

    @property
    def nu(self):
        return 2.0 * self.kappa - 1.0

    @property
    def scale(self):
        return np.sqrt((self.nu - 2.0) / self.nu)

    def density(self, x):
        from scipy.special import gammaln

        nu = self.nu
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
        u = np.asarray(x) / self.scale
        return np.exp(log_norm - (nu + 1.0) / 2.0 * np.log1p(u**2 / nu)) / self.scale

    def derivative(self, x):
        u = np.asarray(x) / self.scale
        return -(self.nu + 1.0) * u / (self.nu + u**2) * self.density(x) / self.scale


def mixture_susceptibility(x, k_lambda_d, components):
    """``chi`` at normalized phase velocity ``x`` for a Gaussian mixture (sum of shifted, scaled Faddeeva terms)
    or a :class:`QuadratureEDF`.
    """
    if isinstance(components, QuadratureEDF):
        return components.response(x) / k_lambda_d**2
    chi = 0.0
    for c in components:
        chi = chi + c.weight * maxwellian_response((x - c.shift) / c.width) / c.width**2
    return chi / k_lambda_d**2


def mixture_density(x, components):
    """The normalized EDF ``f(x)`` (units of v_T) of a Gaussian mixture or a :class:`QuadratureEDF`."""
    if isinstance(components, QuadratureEDF):
        return components.density(x)
    f = 0.0
    for c in components:
        f = f + c.weight * np.exp(-0.5 * ((x - c.shift) / c.width) ** 2) / (c.width * np.sqrt(2.0 * np.pi))
    return f


def _angular_frequency(lam_nm):
    return 2.0 * np.pi * C_CM_PER_S / (np.asarray(lam_nm, dtype=float) * 1.0e-7)


def plasma_frequency(ne_1e20):
    """Electron plasma frequency in rad/s for a density in 1e20 cm^-3."""
    return np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ne_1e20 * 1.0e20 / ELECTRON_MASS)


def form_factor_terms(k, omega, plasma: Plasma):
    """Electron and ion parts of the collisionless ``S(k, omega)`` (normalized so ``int S d omega`` is the static
    structure factor), with ``chi_e`` and ``chi_i``.
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    ne = plasma.ne * 1.0e20
    omega_pe = plasma_frequency(plasma.ne)

    z = np.array([ion.Z for ion in plasma.ions], dtype=float)
    fract = np.array([ion.fract for ion in plasma.ions], dtype=float)
    fract = fract / fract.sum()
    zbar = float(np.sum(z * fract))
    flows = np.array([ion.Va for ion in plasma.ions], dtype=float) * 1.0e6

    chi_i = 0.0
    ion_term = 0.0
    for ion, frac in zip(plasma.ions, fract):
        mass = ion.A * PROTON_MASS
        n_s = frac * ne / zbar
        omega_ps = np.sqrt(4.0 * np.pi * ELECTRON_CHARGE_SQUARED * ion.Z**2 * n_s / mass)
        v_t = np.sqrt(ion.Ti / mass)
        x = (omega - k * ion.Va * 1.0e6) / (k * v_t)
        chi_i = chi_i + mixture_susceptibility(x, v_t * k / omega_ps, ion.components)
        ion_term = ion_term + (frac * ion.Z**2 / zbar) * mixture_density(x, ion.components) / v_t

    electron_flow = float(np.sum(z * fract * flows) / np.sum(z * fract)) + plasma.ud * 1.0e6
    v_te = np.sqrt(plasma.Te / ELECTRON_MASS)
    k_lambda_de = v_te * k / omega_pe
    x_e = (omega - k * electron_flow) / (k * v_te)
    chi_e = mixture_susceptibility(x_e, k_lambda_de, plasma.electrons)
    f_e = mixture_density(x_e, plasma.electrons) / v_te

    eps2 = np.abs(1.0 + chi_e + chi_i) ** 2
    electron_part = np.abs(1.0 + chi_i) ** 2 * f_e / eps2 / k
    ion_part = np.abs(chi_e) ** 2 * ion_term / eps2 / k
    return electron_part, ion_part, chi_e, chi_i


def scattering_geometry(lam_nm, plasma: Plasma):
    """``(k, omega, omega_l)`` along the spectrometer axis: the frequency shift and the Bragg wavenumber
    of light scattered at ``lam_nm`` through ``plasma.angle_deg``, with the plasma dispersion of both waves."""
    omega_pe = plasma_frequency(plasma.ne)
    omega_s = _angular_frequency(lam_nm)
    omega_l = _angular_frequency(plasma.lam)
    k_s = np.sqrt(omega_s**2 - omega_pe**2) / C_CM_PER_S
    k_l = np.sqrt(omega_l**2 - omega_pe**2) / C_CM_PER_S
    theta = np.deg2rad(plasma.angle_deg)
    k = np.sqrt(k_s**2 + k_l**2 - 2.0 * k_s * k_l * np.cos(theta))
    return k, omega_s - omega_l, omega_l


def spectral_density(lam_nm, plasma: Plasma, *, terms=False):
    """``S`` per unit wavelength in tsadar's absolute units: :func:`form_factor_terms` times ``(1 + 2 omega/omega_L)``,
    ``r_e^2 n_e`` and ``d omega / d lambda``. ``terms=True`` also returns the parts and susceptibilities.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    k, omega, omega_l = scattering_geometry(lam_nm, plasma)
    electron_part, ion_part, chi_e, chi_i = form_factor_terms(k, omega, plasma)

    scale = (1.0 + 2.0 * omega / omega_l) * CLASSICAL_ELECTRON_RADIUS_CM**2 * plasma.ne * 1.0e20
    jacobian = 2.0 * np.pi * C_CM_PER_S / (lam_nm * 1.0e-7) ** 2
    electron_part = electron_part * scale * jacobian
    ion_part = ion_part * scale * jacobian
    total = electron_part + ion_part
    if terms:
        return total, electron_part, ion_part, chi_e, chi_i
    return total


def selden_spectrum(eps, theta_deg, te_kev):
    """Exact non-collective relativistic spectrum shape, Selden, Phys. Lett. A 79, 405 (1980), eq. (1).

    ``eps = lambda_s / lambda_i - 1``; the normalizing constant is omitted.
    """
    eps = np.asarray(eps, dtype=float)
    alpha = ELECTRON_MASS_KEV / te_kev / 2.0
    one_minus_cos = 1.0 - np.cos(np.deg2rad(theta_deg))
    a = (1.0 + eps) ** 3 * np.sqrt(2.0 * one_minus_cos * (1.0 + eps) + eps**2)
    b = np.sqrt(1.0 + eps**2 / (2.0 * one_minus_cos * (1.0 + eps))) - 1.0
    return np.exp(-2.0 * alpha * b) / a


def pixel_edges(centers):
    """Edges of a uniform pixel axis given its centers."""
    centers = np.asarray(centers, dtype=float)
    d = np.diff(centers)
    if not np.allclose(d, d[0], rtol=1e-9, atol=0):
        raise ValueError("pixel_edges expects uniformly spaced centers")
    return np.concatenate([centers - d[0] / 2.0, [centers[-1] + d[0] / 2.0]])


def detector_spectrum(spectrum, centers, sigma, *, transmission=None, samples_per_sigma=40, pad_sigmas=8.0):
    """Pixel means of the Gaussian-blurred spectrum (optionally filtered by ``transmission``), exact over each pixel."""
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
    cumulative = np.array([np.dot(weighted, ndtr((e - fine) / sigma)) for e in edges])
    return np.diff(cumulative) / np.diff(edges)


def rectangular_notch(center, width, optical_density):
    """Transmission ``10^-OD`` inside ``|lambda - center| < width/2``, one elsewhere."""

    def transmission(lam):
        lam = np.asarray(lam)
        return np.where(np.abs(lam - center) < width / 2.0, 10.0 ** (-optical_density), 1.0)

    return transmission


def normalize_in_ranges(centers, values, ranges, amplitude=100.0):
    """Scale so the maximum over the pixels inside ``ranges`` equals ``amplitude`` (tsadar's ``norm: 0`` rule)."""
    centers = np.asarray(centers)
    mask = np.zeros(centers.shape, dtype=bool)
    for low, high in ranges:
        mask |= (centers > low) & (centers < high)
    return amplitude * np.asarray(values) / np.max(np.asarray(values)[mask])


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
