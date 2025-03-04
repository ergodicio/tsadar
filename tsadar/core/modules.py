from typing import List, Dict, Union, Callable
from collections import defaultdict
from functools import partial

from jax import Array, numpy as jnp, tree_util as jtu, vmap
from jax.nn import sigmoid, relu
from jax.random import PRNGKey
from jax.scipy.special import gamma, sph_harm
import equinox as eqx


def smooth1d(array, window_size):
    # Use a Hanning window
    window = jnp.hanning(window_size)
    window /= window.sum()  # Normalize
    return jnp.convolve(array, window, mode="same")


def smooth2d(array, window_size):
    # Use a Hanning window
    window = jnp.outer(jnp.hanning(window_size), jnp.hanning(window_size))
    window /= window.sum()  # Normalize
    return jnp.convolve(array, window, mode="same")


class DistributionFunction1D(eqx.Module):
    vx: Array

    def __init__(self, dist_cfg: Dict):
        super().__init__()
        vmax = 6.0
        dv = 2 * vmax / dist_cfg["nv"]
        self.vx = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, dist_cfg["nv"])

    def __call__(self):
        raise NotImplementedError


class Arbitrary1DNN(DistributionFunction1D):
    f_nn: eqx.Module

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        # self.learn_log = dist_cfg["params"]["learn_log"]
        self.f_nn = eqx.nn.MLP(1, 1, 32, 3, final_activation=relu, key=PRNGKey(0))

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        # if self.learn_log:
        #     # bound values between 1e-15 and 10
        #     f_nn = -16 * sigmoid(self.f_nn) + 1
        #     f_nn = jnp.power(10.0, self.f_nn)
        # else:
        # f_nn = sigmoid(f_nn) * 10
        fval = eqx.filter_vmap(self.f_nn)(self.vx[:, None])
        fval = jnp.squeeze(fval)
        # if self.learn_log:
        # fval = jnp.power(10.0, -fval)

        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0])


class Arbitrary1D(DistributionFunction1D):
    fval: Array
    learn_log: bool

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        self.learn_log = dist_cfg["params"]["learn_log"]
        self.fval = self.init_dlm(dist_cfg["params"]["init_m"])
        self.smooth = partial(smooth1d, window_size=dist_cfg["params"]["nvx"] // 8)

    def init_dlm(self, m):
        vth_x = jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5.0 / m))
        cst = m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / m))
        fdlm = cst / vth_x**3.0 * jnp.exp(-(jnp.abs(self.vx / alpha / vth_x) ** m))
        fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])

        if self.learn_log:
            #     # logit function
            # fdlm = 1 / 16 * jnp.log(fdlm / (1 - fdlm)) - 1
            fdlm = -jnp.log10(fdlm)
        # else:
        #     fdlm = 0.1 * jnp.log(fdlm / (1 - fdlm))

        return jnp.sqrt(fdlm)

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        # if self.learn_log:
        #     # bound values between 1e-15 and 10
        #     fval = -16 * sigmoid(self.fval) + 1
        #     fval = jnp.power(10.0, self.fval)
        # else:
        # fval = sigmoid(fval) * 10
        fval = self.fval**2.0
        if self.learn_log:
            fval = jnp.power(10.0, -fval)

        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0])


class DLM1D(DistributionFunction1D):
    normed_m: Array
    m_scale: float
    m_shift: float
    act_fun: Callable

    def __init__(self, dist_cfg, activate=False):
        super().__init__(dist_cfg)
        self.m_scale = 3.0  # dist_cfg["params"]["m"]["ub"] - dist_cfg["params"]["m"]["lb"]
        self.m_shift = 2.0  # dist_cfg["params"]["m"]["lb"]

        if activate:
            inv_act_fun = lambda x: x  # jnp.log(1e-6 + x / (1 - x))
            self.act_fun = sigmoid
        else:
            inv_act_fun = lambda x: x
            self.act_fun = lambda x: x

        self.normed_m = inv_act_fun((dist_cfg["params"]["m"]["val"] - self.m_shift) / self.m_scale)

    def get_unnormed_params(self):
        return {"m": self.act_fun(self.normed_m) * self.m_scale + self.m_shift}

    def __call__(self):
        unnormed_m = self.act_fun(self.normed_m) * self.m_scale + self.m_shift
        vth_x = jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / unnormed_m) / 2.0 / gamma(5.0 / unnormed_m))
        cst = unnormed_m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / unnormed_m))
        fdlm = cst / vth_x**3.0 * jnp.exp(-(jnp.abs(self.vx / alpha / vth_x) ** unnormed_m))

        return fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])


class DistributionFunction2D(eqx.Module):
    vx: Array

    def __init__(self, dist_cfg):
        super().__init__()
        vmax = 6.0
        dvx = 2 * vmax / dist_cfg["nvx"]
        self.vx = jnp.linspace(-vmax + dvx / 2, vmax - dvx / 2, dist_cfg["nvx"])

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class Arbitrary2D(DistributionFunction2D):
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
            #     # logit function
            # fdlm = 1 / 16 * jnp.log(fdlm / (1 - fdlm)) - 1
            fdlm = -jnp.log10(fdlm)
        # else:
        #     fdlm = 0.1 * jnp.log(fdlm / (1 - fdlm))

        return jnp.sqrt(fdlm)

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        # if self.learn_log:
        #     # bound values between 1e-15 and 10
        #     fval = -16 * sigmoid(self.fval) + 1
        #     fval = jnp.power(10.0, self.fval)
        # else:
        # fval = sigmoid(fval) * 10
        fval = self.fval**2.0
        if self.learn_log:
            fval = jnp.power(10.0, -fval)

        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0]) ** 2.0


class SphericalHarmonics(DistributionFunction2D):
    vr: Array
    th: Array
    phi: Array
    sph_harm: Callable
    vr_vxvy: Array
    Nl: int
    flm: Dict[str, Dict[str, Array]]
    m_scale: float
    m_shift: float
    act_fun: Callable
    normed_m: Array
    smooth: Callable

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
        for i in range(self.Nl + 1):
            self.flm[i] = {j: jnp.zeros(dist_cfg["params"]["nvr"]) for j in range(i + 1)}

        init_m = dist_cfg["params"]["init_m"]

        self.m_scale = 3.0  # dist_cfg["params"]["m"]["ub"] - dist_cfg["params"]["m"]["lb"]
        self.m_shift = 2.0  # dist_cfg["params"]["m"]["lb"]
        inv_act_fun = lambda x: jnp.log(1e-6 + x / (1 - x))
        self.act_fun = sigmoid
        self.normed_m = inv_act_fun((init_m - self.m_shift) / self.m_scale)
        self.smooth = partial(smooth1d, window_size=dist_cfg["params"]["nvr"] // 16)

        self.flm[0][0] = self.get_f00()

        # Uses eq. 3 from
        # Mora, P. & Yahi, H. Thermal heat-flux reduction in laser-produced plasmas. Phys. Rev. A 26, 2259–2261 (1982).

        LTx = dist_cfg["params"]["LTx"]  # Provide in units of mean free path
        LTy = dist_cfg["params"]["LTy"]  # Provide in units of mean free path
        v0 = 1.0  # distributions are normalized to vth anyway
        lambda_e = (
            1.0  # this is the thermal mean free path but really, it is just normalizing the gradient scale lengths.
        )
        # So as long as the gradient scale lengths are provided in units of mean free path and just set this to 1.
        ve = gamma(5.0 / init_m) / 3 / gamma(3.0 / init_m) * v0

        uu = self.vr / v0
        lambda_v = lambda_e * (self.vr / ve) ** 4.0
        coeff = (
            init_m / 2 * uu**init_m - 5 * init_m / 12 * gamma(8 / init_m) / gamma(6 / init_m) * uu ** (init_m - 2) - 1.5
        ) * lambda_v

        self.flm[1][0] = coeff / LTx
        self.flm[1][1] = coeff / LTy

        self.flm[0][0] = self.get_f00()

        # Uses eq. 3 from
        # Mora, P. & Yahi, H. Thermal heat-flux reduction in laser-produced plasmas. Phys. Rev. A 26, 2259–2261 (1982).

        LTx = dist_cfg["params"]["LTx"]  # Provide in units of mean free path
        LTy = dist_cfg["params"]["LTy"]  # Provide in units of mean free path
        v0 = 1.0  # distributions are normalized to vth anyway
        lambda_e = (
            1.0  # this is the thermal mean free path but really, it is just normalizing the gradient scale lengths.
        )
        # So as long as the gradient scale lengths are provided in units of mean free path and just set this to 1.
        ve = gamma(5.0 / init_m) / 3 / gamma(3.0 / init_m) * v0

        uu = self.vr / v0
        lambda_v = lambda_e * (self.vr / ve) ** 4.0
        coeff = (
            init_m / 2 * uu**init_m - 5 * init_m / 12 * gamma(8 / init_m) / gamma(6 / init_m) * uu ** (init_m - 2) - 1.5
        ) * lambda_v

        self.flm[1][0] = coeff / LTx
        self.flm[1][1] = coeff / LTy

    def get_unnormed_params(self):
        flm_dict = {0: {0: self.get_f00(self.act_fun(self.normed_m) * self.m_scale + self.m_shift)}, 1: {}}
        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                flm_dict[i][j] = self.smooth(self.flm[i][j])
        return {"flm": flm_dict}

    def get_unnormed_m(self):
        return self.act_fun(self.normed_m) * self.m_scale + self.m_shift

    def get_f00(self):
        unnormed_m = self.get_unnormed_m()

        # m = unnormed_m
        # vth_x = 1.0
        # alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5 / m))
        # cst = m / (4 * jnp.pi * alpha**3.0 * gamma(3 / m))
        # f00 = cst / vth_x**3.0 * jnp.exp(-((self.vr / alpha / vth_x) ** m))
        # f00 /= jnp.sum(f00 * 4 * jnp.pi * self.vr**2.0) * (self.vr[1] - self.vr[0])

        # return f00

        ve = 1.0
        v0 = ve / jnp.sqrt(gamma(5.0 / unnormed_m) / 3.0 / gamma(3.0 / unnormed_m))
        cst = unnormed_m / (4 * jnp.pi * gamma(3.0 / unnormed_m))
        f00 = cst / v0**3.0 * jnp.exp(-((self.vr / v0) ** unnormed_m))
        f00 /= jnp.sum(f00 * 4 * jnp.pi * self.vr**2.0) * (self.vr[1] - self.vr[0])

        return f00

    def __call__(self):

        f00 = self.get_f00()
        fvxvy = jnp.interp(self.vr_vxvy, self.vr, f00, right=1e-16)

        for i in range(1, self.Nl + 1):
            for j in range(i + 1):
                smoothed_flm = self.smooth(self.flm[i][j])
                _flmvxvy = jnp.interp(self.vr_vxvy, self.vr, smoothed_flm * self.flm[0][0], right=1e-16)
                _sph_harm = self.sph_harm(
                    jnp.array([j]), jnp.array([i]), self.phi.reshape(-1, order="C"), self.th.reshape(-1, order="C"), 2
                ).reshape(self.vr_vxvy.shape, order="C")
                fvxvy += _flmvxvy * jnp.real(_sph_harm)

        fvxvy /= jnp.sum(fvxvy) * (self.vx[1] - self.vx[0]) * (self.vx[1] - self.vx[0])
        fvxvy = jnp.maximum(fvxvy, 1e-16)

        return fvxvy


class ElectronParams(eqx.Module):
    normed_Te: Array
    normed_ne: Array
    Te_scale: float
    Te_shift: float
    ne_scale: float
    ne_shift: float
    distribution_functions: Union[
        List[DistributionFunction1D], List[DistributionFunction2D], DistributionFunction1D, DistributionFunction2D
    ]
    batch: bool
    act_fun: Callable

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        super().__init__()

        self.Te_scale = cfg["Te"]["ub"] - cfg["Te"]["lb"]
        self.Te_shift = cfg["Te"]["lb"]
        self.ne_scale = cfg["ne"]["ub"] - cfg["ne"]["lb"]
        self.ne_shift = cfg["ne"]["lb"]
        self.batch = batch

        if activate:
            self.act_fun = sigmoid
            inv_act_fun = lambda x: x  # jnp.log(1e-6 + x / (1 - x))
        else:
            self.act_fun = lambda x: x
            inv_act_fun = lambda x: x

        if batch:
            self.normed_Te = inv_act_fun(jnp.full(batch_size, (cfg["Te"]["val"] - self.Te_shift) / self.Te_scale))
            self.normed_ne = inv_act_fun(jnp.full(batch_size, (cfg["ne"]["val"] - self.ne_shift) / self.ne_scale))
        else:
            self.normed_Te = inv_act_fun((cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
            self.normed_ne = inv_act_fun((cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)

        self.distribution_functions = self.init_dists(cfg["fe"], batch_size, batch, activate)

    def init_dists(self, dist_cfg, batch_size, batch, activate):
        if dist_cfg["dim"] == 1:
            if dist_cfg["type"].casefold() == "dlm":
                if batch:
                    distribution_functions = [DLM1D(dist_cfg, activate) for _ in range(batch_size)]
                else:
                    distribution_functions = DLM1D(dist_cfg, activate)

            elif dist_cfg["type"].casefold() == "mx":
                if batch:
                    distribution_functions = [
                        lambda vx: jnp.exp(-(vx**2 / 2)) / jnp.sum(jnp.exp(-(vx**2 / 2))) / (vx[1] - vx[0])
                    ]
                else:
                    distribution_functions = (
                        lambda vx: jnp.exp(-(vx**2 / 2)) / jnp.sum(jnp.exp(-(vx**2 / 2))) / (vx[1] - vx[0])
                    )
            elif dist_cfg["type"].casefold() == "arbitrary":
                if batch:
                    distribution_functions = [Arbitrary1D(dist_cfg) for _ in range(batch_size)]
                else:
                    distribution_functions = Arbitrary1D(dist_cfg)

            elif dist_cfg["type"].casefold() == "arbitrary-nn":
                if batch:
                    distribution_functions = [Arbitrary1DNN(dist_cfg) for _ in range(batch_size)]
                else:
                    distribution_functions = Arbitrary1DNN(dist_cfg)

            else:
                raise NotImplementedError(f"Unknown 1D distribution type: {dist_cfg['type']}")
        elif dist_cfg["dim"] == 2:
            if batch:
                raise NotImplementedError(
                    "Batch mode not implemented for 2D distributions as a precautionary measure against memory issues"
                )

            if "sph" in dist_cfg["type"].casefold():
                distribution_functions = SphericalHarmonics(dist_cfg)
            elif dist_cfg["type"].casefold() == "arbitrary":
                distribution_functions = Arbitrary2D(dist_cfg)
            else:
                raise NotImplementedError(f"Unknown 2D distribution type: {dist_cfg['type']}")
        else:
            raise NotImplementedError(f"Not implemented distribution dimension: {dist_cfg['dim']}")

        return distribution_functions

    def get_unnormed_params(self):
        unnormed_fe_params = defaultdict(list)
        if isinstance(self.distribution_functions, list):
            for fe in self.distribution_functions:
                for k, v in fe.get_unnormed_params().items():
                    unnormed_fe_params[k].append(v)
            unnormed_fe_params = {k: jnp.array(v) for k, v in unnormed_fe_params.items()}
        else:
            unnormed_fe_params = self.distribution_functions.get_unnormed_params()

        return {
            "Te": self.act_fun(self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_fun(self.normed_ne) * self.ne_scale + self.ne_shift,
        } | unnormed_fe_params

    def __call__(self):
        physical_params = {
            "Te": self.act_fun(self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_fun(self.normed_ne) * self.ne_scale + self.ne_shift,
        }
        if self.batch:
            dist_params = {
                "fe": jnp.concatenate([df()[None, :] for df in self.distribution_functions]),
                "v": jnp.concatenate([df.vx[None, :] for df in self.distribution_functions]),
            }
        else:
            dist_params = {
                "fe": self.distribution_functions(),
                "v": self.distribution_functions.vx,
            }

        return physical_params | dist_params


class IonParams(eqx.Module):
    normed_Ti: Array
    normed_Z: Array
    # normed_A: Array
    fract: Array
    Ti_scale: float
    Ti_shift: float
    Z_scale: float
    Z_shift: float
    # A_scale: float
    # A_shift: float
    A: int
    act_fun: Callable

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        super().__init__()
        self.Ti_scale = cfg["Ti"]["ub"] - cfg["Ti"]["lb"]
        self.Ti_shift = cfg["Ti"]["lb"]
        self.Z_scale = cfg["Z"]["ub"] - cfg["Z"]["lb"]
        self.Z_shift = cfg["Z"]["lb"]

        # self.A_scale = cfg["A"]["ub"] - cfg["A"]["lb"]
        # self.A_shift = cfg["A"]["lb"]

        if activate:
            inv_act_fun = lambda x: x  # jnp.log(1e-6 + x / (1 - x))
            self.act_fun = sigmoid
        else:
            inv_act_fun = lambda x: x
            self.act_fun = lambda x: x

        if batch:
            self.normed_Ti = inv_act_fun(jnp.full(batch_size, (cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale))
            self.normed_Z = inv_act_fun(jnp.full(batch_size, (cfg["Z"]["val"] - self.Z_shift) / self.Z_scale))
            self.A = jnp.full(batch_size, cfg["A"]["val"])
            self.fract = inv_act_fun(jnp.full(batch_size, cfg["fract"]["val"]))
        else:
            self.normed_Ti = inv_act_fun((cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
            self.normed_Z = inv_act_fun((cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
            self.A = cfg["A"]["val"]
            self.fract = float(inv_act_fun(cfg["fract"]["val"]))

    def get_unnormed_params(self):
        return self()

    def __call__(self):

        return {
            "A": self.A,
            "fract": self.act_fun(self.fract),
            "Ti": self.act_fun(self.normed_Ti) * self.Ti_scale + self.Ti_shift,
            "Z": self.act_fun(self.normed_Z) * self.Z_scale + self.Z_shift,
        }


class GeneralParams(eqx.Module):
    normed_lam: Array
    normed_amp1: Array
    normed_amp2: Array
    normed_amp3: Array
    normed_ne_gradient: Array
    normed_Te_gradient: Array
    normed_ud: Array
    normed_vA: Array
    lam_scale: float
    lam_shift: float
    amp1_scale: float
    amp1_shift: float
    amp2_scale: float
    amp2_shift: float
    amp3_scale: float
    amp3_shift: float
    ne_gradient_scale: float
    ne_gradient_shift: float
    Te_gradient_scale: float
    Te_gradient_shift: float
    ud_scale: float
    ud_shift: float
    vA_scale: float
    vA_shift: float
    act_fun: Callable

    def __init__(self, cfg, batch_size: int, batch=True, activate=False):
        super().__init__()
        self.lam_scale = cfg["lam"]["ub"] - cfg["lam"]["lb"]
        self.lam_shift = cfg["lam"]["lb"]
        self.amp1_scale = cfg["amp1"]["ub"] - cfg["amp1"]["lb"]
        self.amp1_shift = cfg["amp1"]["lb"]
        self.amp2_scale = cfg["amp2"]["ub"] - cfg["amp2"]["lb"]
        self.amp2_shift = cfg["amp2"]["lb"]
        self.amp3_scale = cfg["amp3"]["ub"] - cfg["amp3"]["lb"]
        self.amp3_shift = cfg["amp3"]["lb"]
        self.ne_gradient_scale = cfg["ne_gradient"]["ub"] - cfg["ne_gradient"]["lb"]
        self.ne_gradient_shift = cfg["ne_gradient"]["lb"]
        self.Te_gradient_scale = cfg["Te_gradient"]["ub"] - cfg["Te_gradient"]["lb"]
        self.Te_gradient_shift = cfg["Te_gradient"]["lb"]
        self.ud_scale = cfg["ud"]["ub"] - cfg["ud"]["lb"]
        self.ud_shift = cfg["ud"]["lb"]
        self.vA_scale = cfg["Va"]["ub"] - cfg["Va"]["lb"]
        self.vA_shift = cfg["Va"]["lb"]

        if activate:
            inv_act_fun = lambda x: x  # jnp.log(1e-6 + x / (1 - x))
            self.act_fun = sigmoid
        else:
            inv_act_fun = lambda x: x
            self.act_fun = lambda x: x

        if batch:
            self.normed_amp1 = inv_act_fun(
                jnp.full(batch_size, (cfg["amp1"]["val"] - self.amp1_shift) / self.amp1_scale)
            )
            self.normed_amp2 = inv_act_fun(
                jnp.full(batch_size, (cfg["amp2"]["val"] - self.amp2_shift) / self.amp2_scale)
            )
            self.normed_amp3 = inv_act_fun(
                jnp.full(batch_size, (cfg["amp3"]["val"] - self.amp3_shift) / self.amp3_scale)
            )
            self.normed_ne_gradient = inv_act_fun(
                jnp.full(batch_size, (cfg["ne_gradient"]["val"] - self.ne_gradient_shift) / self.ne_gradient_scale)
            )
            self.normed_Te_gradient = inv_act_fun(
                jnp.full(batch_size, (cfg["Te_gradient"]["val"] - self.Te_gradient_shift) / self.Te_gradient_scale)
            )
            self.normed_ud = inv_act_fun(jnp.full(batch_size, (cfg["ud"]["val"] - self.ud_shift) / self.ud_scale))
            self.normed_vA = inv_act_fun(jnp.full(batch_size, (cfg["Va"]["val"] - self.vA_shift) / self.vA_scale))
            self.normed_lam = inv_act_fun(jnp.full(batch_size, (cfg["lam"]["val"] - self.lam_shift) / self.lam_scale))
        else:
            self.normed_amp1 = inv_act_fun((cfg["amp1"]["val"] - self.amp1_shift) / self.amp1_scale)
            self.normed_amp2 = inv_act_fun((cfg["amp2"]["val"] - self.amp2_shift) / self.amp2_scale)
            self.normed_amp3 = inv_act_fun((cfg["amp3"]["val"] - self.amp3_shift) / self.amp3_scale)
            self.normed_ne_gradient = inv_act_fun(
                (cfg["ne_gradient"]["val"] - self.ne_gradient_shift) / self.ne_gradient_scale
            )
            self.normed_Te_gradient = inv_act_fun(
                (cfg["Te_gradient"]["val"] - self.Te_gradient_shift) / self.Te_gradient_scale
            )
            self.normed_ud = inv_act_fun((cfg["ud"]["val"] - self.ud_shift) / self.ud_scale)
            self.normed_vA = inv_act_fun((cfg["Va"]["val"] - self.vA_shift) / self.vA_scale)
            self.normed_lam = inv_act_fun((cfg["lam"]["val"] - self.lam_shift) / self.lam_scale)

    def get_unnormed_params(self):
        return self()

    def __call__(self):
        unnormed_lam = self.act_fun(self.normed_lam) * self.lam_scale + self.lam_shift
        unnormed_amp1 = self.act_fun(self.normed_amp1) * self.amp1_scale + self.amp1_shift
        unnormed_amp2 = self.act_fun(self.normed_amp2) * self.amp2_scale + self.amp2_shift
        unnormed_amp3 = self.act_fun(self.normed_amp3) * self.amp3_scale + self.amp3_shift
        unnormed_ne_gradient = self.act_fun(self.normed_ne_gradient) * self.ne_gradient_scale + self.ne_gradient_shift
        unnormed_Te_gradient = self.act_fun(self.normed_Te_gradient) * self.Te_gradient_scale + self.Te_gradient_shift
        unnormed_ud = self.act_fun(self.normed_ud) * self.ud_scale + self.ud_shift
        unnormed_vA = self.act_fun(self.normed_vA) * self.vA_scale + self.vA_shift

        return {
            "lam": unnormed_lam,
            "amp1": unnormed_amp1,
            "amp2": unnormed_amp2,
            "amp3": unnormed_amp3,
            "ne_gradient": unnormed_ne_gradient,
            "Te_gradient": unnormed_Te_gradient,
            "ud": unnormed_ud,
            "Va": unnormed_vA,
        }


class ThomsonParams(eqx.Module):
    electron: ElectronParams
    ions: List[IonParams]
    general: GeneralParams

    def __init__(self, param_cfg, num_params: int, batch=True, activate=False):
        super().__init__()
        self.electron = ElectronParams(param_cfg["electron"], num_params, batch, activate)
        self.ions = []
        for species in param_cfg.keys():
            if "ion" in species:
                self.ions.append(IonParams(param_cfg[species], num_params, batch, activate))

        assert len(self.ions) > 0, "No ion species found in input deck"
        self.general = GeneralParams(param_cfg["general"], num_params, batch, activate)

    def get_unnormed_params(self):
        return {
            "electron": self.electron.get_unnormed_params(),
            "general": self.general.get_unnormed_params(),
        } | {f"ion-{i+1}": ion.get_unnormed_params() for i, ion in enumerate(self.ions)}

    def __call__(self):
        return {"electron": self.electron(), "general": self.general()} | {
            f"ion-{i+1}": ion() for i, ion in enumerate(self.ions)
        }

    def get_fitted_params(self, param_cfg):
        param_dict = self.get_unnormed_params()
        num_params = 0
        fitted_params = {}
        for k in param_dict.keys():
            fitted_params[k] = {}
            for k2 in param_dict[k].keys():
                if k2 == "m" and param_cfg[k]["fe"]["active"]:
                    fitted_params[k][k2] = param_dict[k][k2]
                    num_params += 1
                elif k2 in ["f", "flm"]:
                    pass
                elif param_cfg[k][k2]["active"]:
                    fitted_params[k][k2] = param_dict[k][k2]
                    num_params += 1

        return fitted_params, num_params


def get_filter_spec(cfg_params: Dict, ts_params: ThomsonParams) -> Dict:
    # Step 2
    filter_spec = jtu.tree_map(lambda _: False, ts_params)
    for species, params in cfg_params.items():
        for key, val in params.items():
            if val["active"]:
                if key == "fe":
                    filter_spec = get_distribution_filter_spec(filter_spec, dist_type=val["type"])
                else:
                    nkey = f"normed_{key}"
                    filter_spec = eqx.tree_at(
                        lambda tree: getattr(getattr(tree, species), nkey),
                        filter_spec,
                        replace=True,
                    )

    return filter_spec


def get_distribution_filter_spec(filter_spec: Dict, dist_type: str) -> Dict:
    if dist_type.casefold() == "dlm":
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

    elif dist_type.casefold() == "mx":
        raise Warning("No trainable parameters for Maxwellian distribution")

    elif dist_type.casefold() == "arbitrary":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].fval, filter_spec, replace=True
                )
        else:
            filter_spec = eqx.tree_at(lambda tree: tree.electron.distribution_functions.fval, filter_spec, replace=True)
    elif dist_type.casefold() == "arbitrary-nn":
        df = filter_spec.electron.distribution_functions
        if isinstance(df, list):
            for i in range(len(df)):
                filter_spec = update_distribution_layers(filter_spec, df=df[i])
        else:
            filter_spec = update_distribution_layers(filter_spec, df=df)
    elif dist_type.casefold() == "sphericalharmonic":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].normed_m, filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].flm[1][0], filter_spec, replace=True
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].flm[1][1], filter_spec, replace=True
                )
        else:

            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.normed_m, filter_spec, replace=True
            )
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.flm[1][0], filter_spec, replace=True
            )
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.flm[1][1], filter_spec, replace=True
            )

    else:
        raise NotImplementedError(f"Untrainable distribution type: {dist_type}")

    return filter_spec


def update_distribution_layers(filter_spec, df):
    print(df.f_nn.layers)
    for j in range(len(df.f_nn.layers)):
        if df.f_nn.layers[j].weight:
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.weight, filter_spec, replace=True)
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.bias, filter_spec, replace=True)

    return filter_spec
