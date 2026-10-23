"""An in-plane polar-harmonic model of the ARTS2D 2-V marginal.

ARTS measures the in-plane marginal f2(vx, vy) = int f3 dvz. A 3-V spherical-harmonic hypothesis carries freedom
the marginal cannot see: from Nl = 2, the out-of-plane temperature Tzz trades against Te with the in-plane EDF, and
the spectrum, unchanged. This model parameterizes the in-plane marginal itself, as a Fourier series in the in-plane
angle theta = atan2(vy, vx). cos(m theta) and sin(m theta) are the eigenfunctions of the 2-D pitch-angle scattering
operator (eigenvalue m^2), the 2-V counterpart of the spherical harmonics of VFP codes (eigenvalue l(l + 1)), and the
expansion is the one used for the gyrophase in magnetized kinetic theory:

    f2(rho, theta) = F0(rho) exp(L tanh(sum_{m=1..M} [c_m(rho) cos m theta + s_m(rho) sin m theta] / L))

F0 is the in-plane marginal of the isotropic 3-V super-Gaussian exp(-(r / v0)^m) with <r^2> = 3, so the shape
parameter and Te mean what they mean in the 3-V models (<vx^2 + vy^2> = 2 for the isotropic part), and an
isotropic 3-V EDF is represented exactly. The radial coefficients are the uncapped (1 + v^4) radial functions of the
spherical-harmonic model (arbitrary or nn), times rho^m / (1 + rho^m) so mode m vanishes like rho^m at the origin,
as the Fourier coefficients of any smooth 2-V function do. m = 1 carries flow and heat flux, m = 2 the in-plane
temperature anisotropy (cos 2 theta) and its tilt (sin 2 theta), m = 3 the third moments; a bi-temperature EDF also
needs m = 4. By default (``free_isotropic: true``) an m = 0 radial function c_0(rho) is added to the exponent, so the
isotropic part is trained as f_0(v) is in VFP codes (the 2-V marginal of a 3-V log-anisotropic EDF has an isotropic part that is
not exactly a super-Gaussian marginal); it trades against Te and m like any free radial shape, so fit it with the
in-plane <v^2> gauge (regularization temperature, target 2).
"""

from collections import defaultdict
from typing import Dict, Optional

import jax
from jax import numpy as jnp, Array
from jax.random import PRNGKey
from jax.scipy.special import gamma
import equinox as eqx
from numpy.polynomial.legendre import leggauss

from .base import DistributionFunction2V
from .spherical_harmonics import ArbitraryVr, FLM_NN, _clip_unit_interval


class PolarHarmonics(DistributionFunction2V):
    """In-plane polar-harmonic (cos m theta, sin m theta) model of the 2-V marginal, m = 1..M.

    Attributes:
        vr (Array): Radial grid of the radial coefficients and of F0.
        rho (Array): In-plane speed on the (vy, vx) grid.
        theta (Array): In-plane angle atan2(vy, vx) on the (vy, vx) grid.
        vz (Array), vz_weights (Array): Gauss-Legendre nodes and weights for F0's integral over vz.
        M (int): Highest angular mode.
        modes (Dict[int, Dict[str, eqx.Module]]): Radial coefficient of cos and sin for each m.
        normed_m (Array): Super-Gaussian order of the isotropic part, as a fraction of its bounds.
    """
    vr: Array
    rho: Array
    theta: Array
    vz: Array
    vz_weights: Array
    M: int
    modes: Dict[int, Dict[str, eqx.Module]]
    free0: Optional[eqx.Module]
    m_scale: float
    m_shift: float
    normed_m: Array
    flm_type: str
    anisotropy_log_limit: float

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        params = dist_cfg["params"]
        vmax = 8.0 * 1.05 * jnp.sqrt(2.0)  # corners of the in-plane grid, with an interpolation guard
        nvr = int(params["nvr"])
        dvr = vmax / nvr
        self.vr = jnp.linspace(dvr / 2, vmax - dvr / 2, nvr)
        vx, vy = jnp.meshgrid(self.vx, self.vx)
        self.rho = jnp.hypot(vx, vy)
        self.theta = jnp.arctan2(vy, vx)
        nvz = int(params.get("nvz", max(64, nvr)))
        if nvz < 2:
            raise ValueError("nvz must contain at least two quadrature nodes")
        nodes, weights = leggauss(nvz)
        self.vz = jnp.asarray(8.0 * nodes)
        self.vz_weights = jnp.asarray(8.0 * weights)
        self.M = int(params["M"])
        if self.M < 1:
            raise ValueError("polarharmonic M must be at least 1")
        self.anisotropy_log_limit = float(params.get("anisotropy_log_limit", 8.0))
        if self.anisotropy_log_limit <= 0:
            raise ValueError("anisotropy_log_limit must be positive")

        m_cfg = params.get("m", {})
        m_lb, m_ub = float(m_cfg.get("lb", 2.0)), float(m_cfg.get("ub", 5.0))
        if not 0.0 < m_lb < m_ub:
            raise ValueError(f"m bounds must satisfy 0 < lb < ub, got lb={m_lb}, ub={m_ub}")
        self.m_scale, self.m_shift = m_ub - m_lb, m_lb
        initial_fraction = (params["init_m"] - self.m_shift) / self.m_scale
        if not 0.0 <= initial_fraction <= 1.0:
            raise ValueError(f"init_m must lie in the closed interval [{m_lb}, {m_ub}] set by the m bounds")
        self.normed_m = jnp.asarray(initial_fraction)

        self.flm_type = params["flm_type"]
        self.modes = defaultdict(dict)
        for m in range(1, self.M + 1):
            for k, part in enumerate(("cos", "sin")):
                if self.flm_type.casefold() == "arbitrary":
                    self.modes[m][part] = ArbitraryVr(self.vr)
                elif self.flm_type.casefold() == "nn":
                    self.modes[m][part] = FLM_NN(self.vr, key=jax.random.fold_in(PRNGKey(1), 10 * m + k))
                else:
                    raise NotImplementedError(f"polarharmonic supports flm_type arbitrary or nn, not {self.flm_type}")
        if bool(params.get("free_isotropic", True)):
            self.free0 = (ArbitraryVr(self.vr) if self.flm_type.casefold() == "arbitrary"
                          else FLM_NN(self.vr, key=jax.random.fold_in(PRNGKey(1), 0)))
        else:
            self.free0 = None

    def get_unnormed_m(self):
        """The physical super-Gaussian order of the isotropic part."""
        return _clip_unit_interval(self.normed_m) * self.m_scale + self.m_shift

    def _log_f0(self, radius):
        """log of the in-plane marginal of exp(-(r / v0)^m), <r^2> = 3, at in-plane speed ``radius`` (unnormalized).

        The vz integral is evaluated on a radial grid 16x finer than ``vr`` (a 1-D integral, cheap next to the 2-D
        grid) so the linear interpolation of the strongly curved super-Gaussian log stays below ~1e-4.
        """
        shape = self.get_unnormed_m()
        v0 = jnp.sqrt(3.0 * gamma(3.0 / shape) / gamma(5.0 / shape))
        fine_r = jnp.linspace(0.0, self.vr[-1], 16 * self.vr.shape[0])
        r3 = jnp.sqrt(fine_r[:, None] ** 2 + self.vz[None, :] ** 2)
        f0_r = jnp.sum(jnp.exp(-((r3 / v0) ** shape)) * self.vz_weights[None, :], axis=1)
        log_f0_r = jnp.log(jnp.maximum(f0_r, jnp.finfo(f0_r.dtype).tiny))
        return jnp.interp(radius, fine_r, log_f0_r)

    def relative_coefficients(self):
        """Radial coefficients r_m(vr) of cos and sin, before the regularity factor: {m: {"cos": .., "sin": ..}}."""
        ones = jnp.ones_like(self.vr)
        return {m: {part: self.modes[m][part](f00=ones) for part in ("cos", "sin")} for m in range(1, self.M + 1)}

    def get_unnormed_params(self):
        return {"m": self.get_unnormed_m(), "f": self()}

    def log_anisotropy(self):
        """c_0(rho) + sum_m rho^m / (1 + rho^m) [c_m(rho) cos m theta + s_m(rho) sin m theta] on the (vy, vx) grid
        (unbounded; c_0 unless free_isotropic is false)."""
        log_anisotropy = jnp.zeros_like(self.rho)
        for m, parts in self.relative_coefficients().items():
            regularity = self.rho**m / (1.0 + self.rho**m)
            cos_r = jnp.interp(self.rho, self.vr, parts["cos"])
            sin_r = jnp.interp(self.rho, self.vr, parts["sin"])
            log_anisotropy += regularity * (cos_r * jnp.cos(m * self.theta) + sin_r * jnp.sin(m * self.theta))
        if self.free0 is not None:
            log_anisotropy += jnp.interp(self.rho, self.vr, self.free0(f00=jnp.ones_like(self.vr)))
        return log_anisotropy

    def __call__(self):
        """Normalized in-plane EDF on the (vy, vx) grid."""
        log_anisotropy = self.log_anisotropy()
        bounded = self.anisotropy_log_limit * jnp.tanh(log_anisotropy / self.anisotropy_log_limit)
        log_f = self._log_f0(self.rho) + bounded
        f = jnp.exp(log_f - jax.lax.stop_gradient(jnp.max(log_f)))
        dv = self.vx[1] - self.vx[0]
        return f / (jnp.sum(f) * dv**2)
