from typing import Dict, Callable
from collections import defaultdict
from functools import partial

from jax import numpy as jnp, vmap, Array
from jax.nn import sigmoid, relu
from jax.random import PRNGKey
from jax.scipy.special import gamma, sph_harm
import equinox as eqx

from .base import DistributionFunction2V, smooth1d


class FLM_NN(eqx.Module):
    flm_mag: eqx.nn.MLP
    flm_sign: eqx.nn.MLP
    vr: Array

    def __init__(self, vr):
        super().__init__()
        self.flm_mag = eqx.nn.MLP(1, 1, 32, 3, final_activation=relu, key=PRNGKey(0))
        self.flm_sign = eqx.nn.MLP(1, 1, 32, 3, final_activation=jnp.tanh, key=PRNGKey(42))
        self.vr = vr

    def __call__(self, **kwargs):
        f00 = kwargs["f00"]
        flm_mag = -vmap(self.flm_mag)(self.vr[:, None])  # from minus inf to 0
        flm_mag = jnp.power(10.0, flm_mag)  # from 0 to 1
        flm_mag *= f00  # from 0 to f00

        flm_sign = vmap(self.flm_sign)(self.vr[:, None])
        flm = flm_mag * flm_sign
        return flm


class FLM_MY(eqx.Module):
    vr: Array
    log_10_LT: float

    def __init__(self, vr: Array, LT: float):
        super().__init__()
        self.vr = vr
        self.log_10_LT = jnp.log10(LT)

    def __call__(self, **kwargs):
        m_f0 = kwargs["m_f0"]
        f00 = kwargs["f00"]

        # Uses eq. 3 from
        # Mora, P. & Yahi, H. Thermal heat-flux reduction in laser-produced plasmas. Phys. Rev. A 26, 2259–2261 (1982).
        v0 = 1.0  # distributions are normalized to vth anyway
        lambda_e = (
            1.0  # this is the thermal mean free path but really, it is just normalizing the gradient scale lengths.
        )
        # So as long as the gradient scale lengths are provided in units of mean free path and just set this to 1.
        ve = gamma(5.0 / m_f0) / 3 / gamma(3.0 / m_f0) * v0

        uu = self.vr / v0
        lambda_v = lambda_e * (self.vr / ve) ** 4.0
        coeff = (
            m_f0 / 2 * uu**m_f0 - 5 * m_f0 / 12 * gamma(8 / m_f0) / gamma(6 / m_f0) * uu ** (m_f0 - 2) - 1.5
        ) * lambda_v

        return coeff / 10**self.log_10_LT * f00


class ArbitraryVr(eqx.Module):
    smooth: Callable
    flm_sign: Array
    flm_mag: Array

    def __init__(self, nvr):
        super().__init__()
        self.smooth = partial(smooth1d, window_size=nvr // 4)
        self.flm_sign = jnp.zeros(nvr)
        self.flm_mag = jnp.zeros(nvr)

    def __call__(self, **kwargs):
        flm_sign = jnp.tanh(self.smooth(self.flm_sign))
        flm_mag = -sigmoid(self.smooth(self.flm_mag)) * 10
        flm = 10**flm_mag * flm_sign

        return flm


class SphericalHarmonics(DistributionFunction2V):
    vr: Array
    th: Array
    phi: Array
    sph_harm: Callable
    vr_vxvy: Array
    Nl: int
    flm: Dict[str, Dict[str, Callable]]
    m_scale: float
    m_shift: float
    act_fun: Callable
    normed_m: Array
    flm_type: str

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)

        vmax = 6.0 * 1.05 * jnp.sqrt(2.0)
        dvr = vmax / dist_cfg["params"]["nvr"]
        self.vr = jnp.linspace(dvr / 2, vmax - dvr / 2, dist_cfg["params"]["nvr"])

        vx, vy = jnp.meshgrid(self.vx, self.vx)
        self.th = jnp.arctan2(vy, vx)
        self.phi = jnp.arccos(vy / jnp.abs(vy))
        self.vr_vxvy = jnp.sqrt(vx**2 + vy**2)
        self.Nl = dist_cfg["params"]["Nl"]

        self.sph_harm = vmap(sph_harm, in_axes=(None, None, 0, 0, None))
        self.flm = defaultdict(dict)

        init_m = dist_cfg["params"]["init_m"]

        self.m_scale = 3.0
        self.m_shift = 2.0
        inv_act_fun = lambda x: jnp.log(1e-2 + x / (1 - x + 1e-2))
        self.act_fun = sigmoid
        self.normed_m = inv_act_fun((init_m - self.m_shift) / self.m_scale)

        self.flm[0][0] = self.get_f00()
        self.flm_type = dist_cfg["params"]["flm_type"]

        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                if self.flm_type.casefold() == "nn":
                    self.flm[i][j] = FLM_NN(self.vr)

                elif self.flm_type.casefold() == "mora-yahi":
                    if i == 1 and j == 0:
                        self.flm[i][j] = FLM_MY(self.vr, dist_cfg["params"]["LTx"])
                    elif i == 1 and j == 1:
                        self.flm[i][j] = FLM_MY(self.vr, dist_cfg["params"]["LTy"])
                    else:
                        raise NotImplementedError("Mora-Yahi only supports l=1, m=0 and l=1, m=1")

                elif self.flm_type.casefold() == "arbitrary":
                    self.flm[i][j] = ArbitraryVr(dist_cfg["params"]["nvr"])

                else:
                    raise NotImplementedError(f"Unknown flm_type: {dist_cfg['params']['flm_type']}")

    def get_unnormed_params(self):
        flm_dict = {0: {0: self.get_f00()}, 1: {}}
        kwargs = {"m_f0": self.get_unnormed_m(), "f00": flm_dict[0][0]}
        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                flm_dict[i][j] = self.flm[i][j](**kwargs)

        return {"flm": flm_dict}

    def get_unnormed_m(self):
        return self.act_fun(self.normed_m) * self.m_scale + self.m_shift

    def get_f00(self):
        unnormed_m = self.get_unnormed_m()

        ve = 1.0
        v0 = ve / jnp.sqrt(gamma(5.0 / unnormed_m) / 3.0 / gamma(3.0 / unnormed_m))
        cst = unnormed_m / (4 * jnp.pi * gamma(3.0 / unnormed_m))
        f00 = cst / v0**3.0 * jnp.exp(-((self.vr / v0) ** unnormed_m))
        f00 /= jnp.sum(f00 * 4 * jnp.pi * self.vr**2.0) * (self.vr[1] - self.vr[0])

        return f00

    def __call__(self):

        f00 = self.get_f00()
        fvxvy = jnp.interp(self.vr_vxvy, self.vr, f00, right=1e-16)

        kwargs = {"m_f0": self.get_unnormed_m(), "f00": f00}

        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                flm = self.flm[i][j](**kwargs)
                _flmvxvy = jnp.interp(self.vr_vxvy, self.vr, flm, right=1e-32)
                _sph_harm = self.sph_harm(
                    jnp.array([j]), jnp.array([i]), self.phi.reshape(-1, order="C"), self.th.reshape(-1, order="C"), 2
                ).reshape(self.vr_vxvy.shape, order="C")
                fvxvy += _flmvxvy * jnp.real(_sph_harm)

        fvxvy = jnp.maximum(fvxvy, 1e-32)
        fvxvy /= jnp.sum(fvxvy) * (self.vx[1] - self.vx[0]) * (self.vx[1] - self.vx[0])

        return fvxvy
