"""An in-plane polar-harmonic model of the ARTS2D 2-V marginal.

ARTS measures the in-plane marginal f2(vx, vy) = int f3 dvz. A 3-V spherical-harmonic hypothesis carries freedom
the marginal cannot see: from Nl = 2, the out-of-plane temperature Tzz trades against Te with the in-plane EDF, and
the spectrum, unchanged. This model parameterizes the in-plane marginal itself, as a Fourier series in the in-plane
angle theta = atan2(vy, vx). cos(m theta) and sin(m theta) are the eigenfunctions of the 2-D pitch-angle scattering
operator (eigenvalue m^2), the 2-V counterpart of the spherical harmonics of VFP codes (eigenvalue l(l + 1)), and the
expansion is the one used for the gyrophase in magnetized kinetic theory:

    f2(rho, theta) = F0(rho) exp(L tanh(sum_{m=1..M} rho^m/(1+rho^m) [c_m(rho) cos m theta + s_m(rho) sin m theta] / L))

F0 is the trained isotropic part, F0(rho) = F_init(rho) exp(g0(rho)), the single source of isotropy, as f_0(v) is in VFP
codes. g0 is a free radial function on the nvr-node radial grid (the arbitrary or nn form scaled by 1 + v^2, zero at
start); F_init, the in-plane marginal of the isotropic 3-V super-Gaussian exp(-(r / v0)^m) with <r^2> = 3 at m =
init_m, only sets the starting shape (no shape parameter is fitted). F0 is free in width as well as shape, so fit it
with the in-plane <v^2> gauge (regularization temperature, target 2), which defines Te as half the in-plane second
moment. The angular coefficients are the same radial functions times rho^m / (1 + rho^m), so mode m vanishes like rho^m
at the origin, as the Fourier coefficients of any smooth 2-V function do. m = 1 carries flow and heat flux, m = 2 the
in-plane temperature anisotropy (cos 2 theta) and its tilt (sin 2 theta), m = 3 the third moments; a bi-temperature EDF
also needs m = 4.
"""

from collections import defaultdict
from typing import Dict

import jax
from jax import numpy as jnp, Array
from jax.random import PRNGKey
from jax.scipy.special import gamma
import equinox as eqx
from numpy.polynomial.legendre import leggauss

from .base import DistributionFunction2V
from .spherical_harmonics import ArbitraryVr, FLM_NN


class PolarHarmonics(DistributionFunction2V):
    """In-plane polar-harmonic (cos m theta, sin m theta) model of the 2-V marginal, m = 1..M, with a trained F0.

    Attributes:
        vr (Array): Radial grid of the radial functions.
        rho (Array): In-plane speed on the (vy, vx) grid.
        theta (Array): In-plane angle atan2(vy, vx) on the (vy, vx) grid.
        log_f_init (Array): log F_init on the (vy, vx) grid (fixed).
        f0 (eqx.Module): Radial function g0 of the trained isotropic part, F0 = F_init exp(g0).
        M (int): Highest angular mode.
        modes (Dict[int, Dict[str, eqx.Module]]): Radial coefficient of cos and sin for each m.
    """
    vr: Array
    rho: Array
    theta: Array
    log_f_init: Array
    f0: eqx.Module
    M: int
    modes: Dict[int, Dict[str, eqx.Module]]
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
        self.log_f_init = _log_super_gaussian_marginal(
            self.rho, float(params["init_m"]), int(params.get("nvz", max(64, nvr)))
        )
        self.M = int(params["M"])
        if self.M < 1:
            raise ValueError("polarharmonic M must be at least 1")
        self.anisotropy_log_limit = float(params.get("anisotropy_log_limit", 8.0))
        if self.anisotropy_log_limit <= 0:
            raise ValueError("anisotropy_log_limit must be positive")

        self.flm_type = params["flm_type"]
        kind = self.flm_type.casefold()
        if kind not in ("arbitrary", "nn"):
            raise NotImplementedError(f"polarharmonic supports flm_type arbitrary or nn, not {self.flm_type}")

        def radial(key_index, scale_power=4):
            if kind == "arbitrary":
                return ArbitraryVr(self.vr, scale_power=scale_power)
            return FLM_NN(self.vr, key=jax.random.fold_in(PRNGKey(1), key_index), scale_power=scale_power)

        # g0 is outside the bounded exponent, so it is scaled by 1 + v^2, not 1 + v^4: with v^4 shared nn weights moved
        # the tail by ~1e4 in log f on the first step and arbitrary nodes random-walked by tens in log f beyond 5 vTe
        # over a fit (Phase 4d session 3); unscaled, F0 cannot reach shapes far from its start (biDLM from m = 3)
        self.f0 = radial(0, scale_power=2)
        self.modes = defaultdict(dict)
        for m in range(1, self.M + 1):
            for k, part in enumerate(("cos", "sin")):
                self.modes[m][part] = radial(10 * m + k)

    def relative_coefficients(self):
        """Radial coefficients r_m(vr) of cos and sin, before the regularity factor: {m: {"cos": .., "sin": ..}}."""
        ones = jnp.ones_like(self.vr)
        return {m: {part: self.modes[m][part](f00=ones) for part in ("cos", "sin")} for m in range(1, self.M + 1)}

    def log_f0(self):
        """log F0 = log F_init + g0 on the (vy, vx) grid (unnormalized)."""
        return self.log_f_init + jnp.interp(self.rho, self.vr, self.f0(f00=jnp.ones_like(self.vr)))

    def get_unnormed_params(self):
        return {"f": self()}

    def log_anisotropy(self):
        """sum_m rho^m / (1 + rho^m) [c_m(rho) cos m theta + s_m(rho) sin m theta] on the (vy, vx) grid (unbounded)."""
        log_anisotropy = jnp.zeros_like(self.rho)
        for m, parts in self.relative_coefficients().items():
            regularity = self.rho**m / (1.0 + self.rho**m)
            cos_r = jnp.interp(self.rho, self.vr, parts["cos"])
            sin_r = jnp.interp(self.rho, self.vr, parts["sin"])
            log_anisotropy += regularity * (cos_r * jnp.cos(m * self.theta) + sin_r * jnp.sin(m * self.theta))
        return log_anisotropy

    def __call__(self):
        """Normalized in-plane EDF on the (vy, vx) grid."""
        log_anisotropy = self.log_anisotropy()
        bounded = self.anisotropy_log_limit * jnp.tanh(log_anisotropy / self.anisotropy_log_limit)
        log_f = self.log_f0() + bounded
        f = jnp.exp(log_f - jax.lax.stop_gradient(jnp.max(log_f)))
        dv = self.vx[1] - self.vx[0]
        return f / (jnp.sum(f) * dv**2)


def _log_super_gaussian_marginal(radius, shape, nvz):
    """log of the in-plane marginal of exp(-(r / v0)^shape), <r^2> = 3, at in-plane speed ``radius`` (unnormalized).

    Evaluated once (it only sets F0's starting shape): Gauss-Legendre over vz on [-8, 8] on a fine radial grid,
    interpolated in log (< ~1e-4 against the direct 3-V marginal for shapes 2-5).
    """
    if nvz < 2:
        raise ValueError("nvz must contain at least two quadrature nodes")
    nodes, weights = leggauss(nvz)
    vz, vz_weights = jnp.asarray(8.0 * nodes), jnp.asarray(8.0 * weights)
    v0 = jnp.sqrt(3.0 * gamma(3.0 / shape) / gamma(5.0 / shape))
    fine_r = jnp.linspace(0.0, float(jnp.max(radius)) * 1.01, 2048)
    r3 = jnp.sqrt(fine_r[:, None] ** 2 + vz[None, :] ** 2)
    f_r = jnp.sum(jnp.exp(-((r3 / v0) ** shape)) * vz_weights[None, :], axis=1)
    log_f_r = jnp.log(jnp.maximum(f_r, jnp.finfo(f_r.dtype).tiny))
    return jnp.interp(radius, fine_r, log_f_r)
