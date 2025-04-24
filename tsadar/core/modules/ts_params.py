from typing import List, Dict, Union, Callable
from collections import defaultdict

from jax import Array, numpy as jnp, tree_util as jtu
from jax.nn import sigmoid
import equinox as eqx

from .distribution_functions.base import (
    DistributionFunction1V,
    DistributionFunction2V,
    DLM1V,
    Arbitrary1V,
    Arbitrary2V,
    get_distribution_filter_spec,
)
from .distribution_functions.spherical_harmonics import SphericalHarmonics


class ElectronParams(eqx.Module):
    normed_Te: Array
    normed_ne: Array
    Te_scale: float
    Te_shift: float
    ne_scale: float
    ne_shift: float
    distribution_functions: Union[
        List[DistributionFunction1V], List[DistributionFunction2V], DistributionFunction1V, DistributionFunction2V
    ]
    batch: bool
    act_funs: Dict[str, Callable]
    inv_act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        super().__init__()

        self.batch = batch

        self.act_funs, self.inv_act_funs = {}, {}
        for param in ["Te", "ne"]:
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])
            self.act_funs[param], self.inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)

        if batch:
            self.normed_Te = self.inv_act_funs["Te"](
                jnp.full(batch_size, (cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
            )
            self.normed_ne = self.inv_act_funs["ne"](
                jnp.full(batch_size, (cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)
            )
        else:
            self.normed_Te = self.inv_act_funs["Te"]((cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
            self.normed_ne = self.inv_act_funs["ne"]((cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)

        self.distribution_functions = self.init_dists(cfg["fe"], batch_size, batch, activate)

    def init_dists(self, dist_cfg, batch_size, batch, activate):
        if dist_cfg["dim"] == 1:
            if dist_cfg["type"].casefold() == "dlm":
                if batch:
                    distribution_functions = [DLM1V(dist_cfg, activate) for _ in range(batch_size)]
                else:
                    distribution_functions = DLM1V(dist_cfg, activate)

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
                    distribution_functions = [Arbitrary1V(dist_cfg) for _ in range(batch_size)]
                else:
                    distribution_functions = Arbitrary1V(dist_cfg)

            else:
                raise NotImplementedError(f"Unknown 1D distribution type: {dist_cfg['type']}")
        elif dist_cfg["dim"] == 2:
            if batch:
                raise NotImplementedError(
                    "Batch mode not implemented for 2D distributions as a precautionary measure against memory issues"
                )
            else:
                if "sph" in dist_cfg["type"].casefold():
                    distribution_functions = SphericalHarmonics(dist_cfg)
                elif dist_cfg["type"].casefold() == "arbitrary":
                    distribution_functions = Arbitrary2V(dist_cfg)
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
            "Te": self.act_funs["Te"](self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_funs["ne"](self.normed_ne) * self.ne_scale + self.ne_shift,
        } | unnormed_fe_params

    def __call__(self):
        physical_params = {
            "Te": self.act_funs["Te"](self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_funs["ne"](self.normed_ne) * self.ne_scale + self.ne_shift,
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
    normed_Va: Array #SB
    fract: Array
    Ti_scale: float
    Ti_shift: float
    Z_scale: float
    Z_shift: float
    Va_scale: float  # SB
    Va_shift: float  # SB
    A: int
    act_funs: Dict[str, Callable]
    inv_act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        super().__init__()
        self.act_funs, self.inv_act_funs = {}, {}
        for param in ["Ti", "Z", "Va"]:  #SB
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])
            self.act_funs[param], self.inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)

        self.act_funs["fract"], self.inv_act_funs["fract"] = get_act_and_inv_act(cfg["fract"], activate)

        if batch:
            self.normed_Ti = self.inv_act_funs["Ti"](
                jnp.full(batch_size, (cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
            )
            self.normed_Z = self.inv_act_funs["Z"](
                jnp.full(batch_size, (cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
            )
            self.normed_Va = self.inv_act_funs["Va"](
                jnp.full(batch_size, (cfg["Va"]["val"] - self.Va_shift) / self.Va_scale)  # SB
            )
            self.A = jnp.full(batch_size, cfg["A"]["val"])
            self.fract = self.inv_act_funs["fract"](jnp.full(batch_size, cfg["fract"]["val"]))
        else:
            self.normed_Ti = self.inv_act_funs["Ti"]((cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
            self.normed_Z = self.inv_act_funs["Z"]((cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
            self.normed_Va = self.inv_act_funs["Va"]((cfg["Va"]["val"] - self.Va_shift) / self.Va_scale)  # SB
            self.A = cfg["A"]["val"]
            self.fract = float(self.inv_act_funs["fract"](cfg["fract"]["val"]))

    def get_unnormed_params(self):
        return self()

    def __call__(self):

        return {
            "A": self.A,
            "fract": self.act_funs["fract"](self.fract),
            "Ti": self.act_funs["Ti"](self.normed_Ti) * self.Ti_scale + self.Ti_shift,
            "Z": self.act_funs["Z"](self.normed_Z) * self.Z_scale + self.Z_shift,
            "Va": self.act_funs["Va"](self.normed_Va) * self.Va_scale + self.Va_shift,
        }


def get_act_and_inv_act(param_cfg: Dict, activate: bool):
    """
    Returns the activation function and its inverse only if the parameter is active i.e.
    it is being fit. If the parameter is not active, the identity function is returned.

    Args:
        param_cfg (Dict): The configuration dictionary for the parameter
        activate (bool): Whether to activate the parameter

    Returns:
        Tuple[Callable, Callable]: The activation function and its inverse


    """

    if param_cfg["active"] and activate:
        inv_act_fun = lambda x: jnp.log(1e-2 + x / (1 - x + 1e-2))  # this is problematic near 0 and 1
        act_fun = sigmoid
    else:
        act_fun = lambda x: x
        inv_act_fun = lambda x: x

    return act_fun, inv_act_fun


class GeneralParams(eqx.Module):
    normed_lam: Array
    normed_amp1: Array
    normed_amp2: Array
    normed_amp3: Array
    normed_ne_gradient: Array
    normed_Te_gradient: Array
    normed_ud: Array
    #normed_Va: Array   # SB
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
    #Va_scale: float    # SB
    #Va_shift: float    # SB
    act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size: int, batch=True, activate=False):
        super().__init__()

        # this is all a bit ugly but we use setattr instead of = to be able to use the for loop
        self.act_funs, inv_act_funs = {}, {}
        for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud"]:   # SB removed Va
            self.act_funs[param], inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])

        # this is where the linear and nonlinear transformations are applied i.e.
        # the rescaling and the activation function
        if batch:
            for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud"]:   # SB removed Va
                setattr(
                    self,
                    "normed_" + param,
                    inv_act_funs[param](
                        jnp.full(
                            batch_size,
                            (cfg[param]["val"] - getattr(self, param + "_shift")) / getattr(self, param + "_scale"),
                        )
                    ),
                )
        else:
            for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud"]:   # SB removed Va
                setattr(
                    self,
                    "normed_" + param,
                    inv_act_funs[param](
                        (cfg[param]["val"] - getattr(self, param + "_shift")) / getattr(self, param + "_scale")
                    ),
                )

    def get_unnormed_params(self):
        return self()

    def __call__(self):
        unnormed_lam = self.act_funs["lam"](self.normed_lam) * self.lam_scale + self.lam_shift
        unnormed_amp1 = self.act_funs["amp1"](self.normed_amp1) * self.amp1_scale + self.amp1_shift
        unnormed_amp2 = self.act_funs["amp2"](self.normed_amp2) * self.amp2_scale + self.amp2_shift
        unnormed_amp3 = self.act_funs["amp3"](self.normed_amp3) * self.amp3_scale + self.amp3_shift
        unnormed_ne_gradient = (
            self.act_funs["ne_gradient"](self.normed_ne_gradient) * self.ne_gradient_scale + self.ne_gradient_shift
        )
        unnormed_Te_gradient = (
            self.act_funs["Te_gradient"](self.normed_Te_gradient) * self.Te_gradient_scale + self.Te_gradient_shift
        )
        unnormed_ud = self.act_funs["ud"](self.normed_ud) * self.ud_scale + self.ud_shift
       # unnormed_Va = self.act_funs["Va"](self.normed_Va) * self.Va_scale + self.Va_shift   # SB

        return {
            "lam": unnormed_lam,
            "amp1": unnormed_amp1,
            "amp2": unnormed_amp2,
            "amp3": unnormed_amp3,
            "ne_gradient": unnormed_ne_gradient,
            "Te_gradient": unnormed_Te_gradient,
            "ud": unnormed_ud,
            #"Va": unnormed_Va,  # SB
        }


class ThomsonParams(eqx.Module):
    electron: ElectronParams
    ions: List[IonParams]
    general: GeneralParams
    param_cfg: Dict

    def __init__(self, param_cfg, num_params: int, batch=True, activate=False):
        super().__init__()

        self.electron = ElectronParams(param_cfg["electron"], num_params, batch, activate)
        self.ions = []
        for ion_index in range(len([species for species in param_cfg.keys() if "ion" in species])):
            self.ions.append(IonParams(param_cfg[f"ion-{ion_index+1}"], num_params, batch, activate))

        assert len(self.ions) > 0, "No ion species found in input deck"
        self.general = GeneralParams(param_cfg["general"], num_params, batch, activate)
        self.param_cfg = param_cfg

    def renormalize_ions(self, tmp_dict):
        fract_sum = 0
        for ion_index in range(len(self.ions)):
            if ion_index > 0 and self.param_cfg[f"ion-{ion_index+1}"]["Ti"]["same"]:
                tmp_dict[f"ion-{ion_index+1}"]["Ti"] = tmp_dict["ion-1"]["Ti"]
            fract_sum += tmp_dict[f"ion-{ion_index+1}"]["fract"]
        for ion_index in range(len(self.ions)):
            tmp_dict[f"ion-{ion_index+1}"]["fract"] /= fract_sum

        return tmp_dict

    def get_unnormed_params(self):
        tmp_dict = {
            "electron": self.electron.get_unnormed_params(),
            "general": self.general.get_unnormed_params(),
        } | {f"ion-{i+1}": ion.get_unnormed_params() for i, ion in enumerate(self.ions)}

        tmp_dict = self.renormalize_ions(tmp_dict)

        return tmp_dict

    def __call__(self):
        tmp_dict = {"electron": self.electron(), "general": self.general()} | {
            f"ion-{i+1}": ion() for i, ion in enumerate(self.ions)
        }
        tmp_dict = self.renormalize_ions(tmp_dict)
        return tmp_dict

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
    ion_num = 0
    for species, params in cfg_params.items():
        if "ion" in species:  # SB
            ion_num += 1      # SB
        for key, _params in params.items():
            if _params["active"]:
                if key == "fe":
                    filter_spec = get_distribution_filter_spec(filter_spec, dist_params=_params)
                else:
                    nkey = f"normed_{key}"
                    if "ion" in species:  # SB
                    #    ion_num += 1      # SB
                        filter_spec = eqx.tree_at(
                            lambda tree: getattr(getattr(tree, "ions")[ion_num - 1], nkey),
                            filter_spec,
                            replace=True,
                        )
                    else:
                        filter_spec = eqx.tree_at(
                            lambda tree: getattr(getattr(tree, species), nkey),
                            filter_spec,
                            replace=True,
                        )

    return filter_spec
