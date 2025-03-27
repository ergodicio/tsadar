import os
from typing import Dict, Callable
from collections import defaultdict
from functools import partial

from jax import numpy as jnp, vmap, Array
from jax.lax import scan
from jax.nn import sigmoid, relu
from jax.random import PRNGKey
from jax.scipy.special import gamma, sph_harm
from scipy.io import loadmat
import equinox as eqx

cwd = os.path.dirname(os.path.realpath(__file__))


def smooth1d(array, window_size):
    # Use a Hanning window
    window = jnp.hanning(window_size)
    window /= window.sum()  # Normalize
    return jnp.convolve(array, window, mode="same")
    # signal = jnp.r_[array[window_size - 1 : 0 : -1], array, array[-2 : -window_size - 1 : -1]]
    # y = jnp.convolve(signal, window, mode="same")
    # return y[(window_size // 2 - 1) : -(window_size // 2)]


def second_order_butterworth(
    signal: Array, f_sampling: int = 100, f_cutoff: int = 15, method: str = "forward_backward"
) -> Array:
    """
    Applies a second order butterworth filter similar to using scipy.signal.butter and scipy.signal.filtfilt

    from https://github.com/jax-ml/jax/issues/17540

    """

    if method == "forward_backward":
        signal = second_order_butterworth(signal, f_sampling, f_cutoff, "forward")
        return second_order_butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError

    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))

    if method == "backward":
        signal = jnp.flip(signal, axis=0)

    return signal


def smooth2d(array, window_size):
    # Use a Hanning window
    window = jnp.outer(jnp.hanning(window_size), jnp.hanning(window_size))
    window /= window.sum()  # Normalize
    return jnp.convolve(array, window, mode="same")


class DistributionFunction1V(eqx.Module):
    vx: Array

    def __init__(self, dist_cfg: Dict):
        super().__init__()
        vmax = 6.0
        dv = 2 * vmax / dist_cfg["nvx"]
        self.vx = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, dist_cfg["nvx"])

    def __call__(self):
        raise NotImplementedError


class Arbitrary1V(DistributionFunction1V):
    fval: Array
    smooth: Callable

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        self.fval = self.init_dlm(dist_cfg["params"]["init_m"])
        self.smooth = partial(second_order_butterworth, f_sampling=100, f_cutoff=6, method="forward_backward")

    def init_dlm(self, m):
        vth_x = 1.0  # jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5.0 / m))
        cst = m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / m))
        fdlm = cst / vth_x**3.0 * jnp.exp(-(jnp.abs(self.vx / alpha / vth_x) ** m))
        fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])
        fdlm = -jnp.log10(fdlm)

        return jnp.sqrt(fdlm) / 7.0

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        fval = (7.0 * self.smooth(self.fval)) ** 2.0
        fval = jnp.power(10.0, -fval)
        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0])


class DLM1V(DistributionFunction1V):
    normed_m: Array
    m_scale: float
    m_shift: float
    act_fun: Callable
    f_vx_m: Array
    interpolate_f_in_m: Callable
    m_ax: Array
    act_fun: Callable

    def __init__(self, dist_cfg, activate=False):
        super().__init__(dist_cfg)
        self.m_scale = 3.0
        self.m_shift = 2.0

        if activate and dist_cfg["active"]:
            inv_act_fun = lambda x: jnp.log(1e-2 + x / (1 - x + 1e-2))
            self.act_fun = sigmoid
        else:
            inv_act_fun = lambda x: x
            self.act_fun = lambda x: x

        self.normed_m = inv_act_fun((dist_cfg["params"]["m"]["val"] - self.m_shift) / self.m_scale)
        projected_distributions = loadmat(
            os.path.join(cwd, "..", "..", "..", "external", "numDistFuncs", "DLM_x_-3_-10_10_m_-1_2_5.mat")
        )["IT"]
        vx_ax = jnp.linspace(-10, 10, 20001)
        self.m_ax = jnp.linspace(2, 5, 31)
        self.f_vx_m = vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(self.vx, vx_ax, projected_distributions)
        self.interpolate_f_in_m = vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)

    def get_unnormed_params(self):
        return {"m": self.act_fun(self.normed_m) * self.m_scale + self.m_shift}

    def __call__(self):
        unnormed_m = self.act_fun(self.normed_m) * self.m_scale + self.m_shift
        vth_x = 1.0  # jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / unnormed_m) / 2.0 / gamma(5.0 / unnormed_m))
        cst = unnormed_m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / unnormed_m))
        fdlm = cst / vth_x**3.0 * jnp.exp(-(jnp.abs(self.vx / alpha / vth_x) ** unnormed_m))
        # fdlm = self.interpolate_f_in_m(unnormed_m, self.m_ax, self.f_vx_m)

        return fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])


class DistributionFunction2V(eqx.Module):
    vx: Array

    def __init__(self, dist_cfg):
        super().__init__()
        vmax = 6.0
        dvx = 2 * vmax / dist_cfg["nvx"]
        self.vx = jnp.linspace(-vmax + dvx / 2, vmax - dvx / 2, dist_cfg["nvx"])

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class Arbitrary2V(DistributionFunction2V):
    fval: Array
    learn_log: bool

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        self.learn_log = dist_cfg["params"]["learn_log"]
        self.fval = self.init_dlm(dist_cfg["params"]["init_m"])

    def init_dlm(self, m):

        vth_x = jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5.0 / m))
        cst = m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / m))
        fdlm = (
            cst
            / vth_x**3.0
            * jnp.exp(-((jnp.sqrt(self.vx[:, None] ** 2.0 + self.vx[None, :] ** 2.0) / alpha / vth_x) ** m))
        )

        fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0]) ** 2.0

        if self.learn_log:
            fdlm = -jnp.log10(fdlm)

        return jnp.sqrt(fdlm)

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        fval = self.fval**2.0
        if self.learn_log:
            fval = jnp.power(10.0, -fval)

        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0]) ** 2.0


def get_distribution_filter_spec(filter_spec: Dict, dist_params: Dict) -> Dict:
    if dist_params["type"].casefold() == "dlm":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].normed_m, filter_spec, replace=True
                )
        else:
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.normed_m, filter_spec, replace=True
            )

    elif dist_params["type"].casefold() == "mx":
        raise Warning("No trainable parameters for Maxwellian distribution")

    elif dist_params["type"].casefold() == "arbitrary":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].fval, filter_spec, replace=True
                )
        else:
            filter_spec = eqx.tree_at(lambda tree: tree.electron.distribution_functions.fval, filter_spec, replace=True)

    elif dist_params["type"].casefold() == "arbitrary-nn":
        df = filter_spec.electron.distribution_functions
        if isinstance(df, list):
            for i in range(len(df)):
                filter_spec = update_distribution_layers(filter_spec, df=df[i])
        else:
            filter_spec = update_distribution_layers(filter_spec, df=df)

    elif dist_params["type"].casefold() == "sphericalharmonic":
        if isinstance(filter_spec.electron.distribution_functions, list):
            raise NotImplementedError

        else:
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.normed_m, filter_spec, replace=True
            )
            if dist_params["params"]["flm_type"].casefold() == "arbitrary":
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].flm_mag, filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].flm_sign, filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].flm_mag, filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].flm_sign, filter_spec, replace=True
                )
            elif dist_params["params"]["flm_type"].casefold() == "mora-yahi":
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].log_10_LT, filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].log_10_LT, filter_spec, replace=True
                )
            elif dist_params["params"]["flm_type"].casefold() == "nn":
                for m in range(2):
                    df = filter_spec.electron.distribution_functions.flm[1][m]
                    for j in range(len(df.flm_mag.layers)):
                        filter_spec = eqx.tree_at(
                            lambda tree: tree.electron.distribution_functions.flm[1][m].flm_mag.layers[j].weight,
                            filter_spec,
                            replace=True,
                        )
                        filter_spec = eqx.tree_at(
                            lambda tree: tree.electron.distribution_functions.flm[1][m].flm_sign.layers[j].weight,
                            filter_spec,
                            replace=True,
                        )
            else:
                raise NotImplementedError(f"Unknown flm_type: {dist_params['flm_type']}")

    else:
        raise NotImplementedError(f"Untrainable distribution type: {dist_params['type']}")

    return filter_spec


def update_distribution_layers(filter_spec, df):
    print(df.f_nn.layers)
    for j in range(len(df.f_nn.layers)):
        if df.f_nn.layers[j].weight:
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.weight, filter_spec, replace=True)
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.bias, filter_spec, replace=True)

    return filter_spec
