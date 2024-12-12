from typing import List, Dict
from jax import Array, numpy as jnp, tree_util as jtu
from jax.nn import sigmoid
from jax.scipy.special import gamma
import equinox as eqx


class DLM1D(eqx.Module):
    normed_m: Array
    m_scale: float
    m_shift: float

    def __init__(self, dist_cfg):
        super().__init__()
        self.m_scale = dist_cfg["m"]["ub"] - dist_cfg["m"]["lb"]
        self.m_shift = dist_cfg["m"]["lb"]
        self.normed_m = (dist_cfg["m"]["val"] - self.m_shift) / self.m_scale

    def __call__(self, vx):
        unnormed_m = self.normed_m * self.m_scale + self.m_shift
        vth_x = 1.0
        alpha = jnp.sqrt(3.0 * gamma(3.0 / unnormed_m) / 2.0 / gamma(5.0 / unnormed_m))
        cst = unnormed_m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / unnormed_m))
        fdlm = cst / vth_x**3.0 * jnp.exp(-((vx / alpha / vth_x) ** unnormed_m))

        return fdlm / jnp.sum(fdlm) / (vx[1] - vx[0])


class DistributionFunction1D(eqx.Module):
    vx: Array
    fvx: eqx.Module

    def __init__(self, dist_cfg: Dict):
        super().__init__()
        vmax = 8.0
        dv = 2 * vmax / dist_cfg["nv"]
        self.vx = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, dist_cfg["nv"])

        if dist_cfg["type"].casefold() == "dlm":
            self.fvx = DLM1D(dist_cfg["params"])
        elif dist_cfg["type"].casefold() == "mx":
            self.fvx = lambda vx: jnp.exp(-(vx**2 / 2)) / jnp.sum(jnp.exp(-(vx**2 / 2))) / (vx[1] - vx[0])
        else:
            raise NotImplementedError(f"Unknown distribution type: {dist_cfg['type']}")

    def __call__(self):
        return self.fvx(self.vx)


class DistributionFunction2D(eqx.Module):
    vx: Array
    vy: Array
    fvxvy: Array

    def __init__(self, cfg):
        super().__init__()
        self.vx = cfg["vx"]
        self.vy = cfg["vy"]
        self.fvxvy = cfg["fvxvy"]

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class ElectronParams(eqx.Module):
    normed_Te: Array
    normed_ne: Array
    Te_scale: float
    Te_shift: float
    ne_scale: float
    ne_shift: float
    fe: eqx.Module

    def __init__(self, cfg, batch_size):
        super().__init__()

        self.Te_scale = cfg["Te"]["ub"] - cfg["Te"]["lb"]
        self.Te_shift = cfg["Te"]["lb"]
        self.ne_scale = cfg["ne"]["ub"] - cfg["ne"]["lb"]
        self.ne_shift = cfg["ne"]["lb"]
        self.normed_Te = jnp.full(batch_size, (cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
        self.normed_ne = jnp.full(batch_size, (cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)

        if cfg["fe"]["dim"] == 1:
            self.fe = [DistributionFunction1D(cfg["fe"]) for _ in range(batch_size)]
        elif cfg["fe"]["dim"] == 2:
            self.fe = [DistributionFunction2D(cfg["fe"]) for _ in range(batch_size)]
        else:
            raise NotImplementedError(f"Unknown distribution dimension: {cfg['fe']['dim']}")

    def __call__(self):
        unnormed_Te = self.normed_Te * self.Te_scale + self.Te_shift
        unnormed_ne = self.normed_ne * self.ne_scale + self.ne_shift
        return {
            "Te": unnormed_Te,
            "ne": unnormed_ne,
            "fe": jnp.concatenate([fe()[None, :] for fe in self.fe]),
            "v": jnp.concatenate([fe.vx[None, :] for fe in self.fe]),
        }


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

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.Ti_scale = cfg["Ti"]["ub"] - cfg["Ti"]["lb"]
        self.Ti_shift = cfg["Ti"]["lb"]
        self.Z_scale = cfg["Z"]["ub"] - cfg["Z"]["lb"]
        self.Z_shift = cfg["Z"]["lb"]
        # self.A_scale = cfg["A"]["ub"] - cfg["A"]["lb"]
        # self.A_shift = cfg["A"]["lb"]
        self.normed_Ti = jnp.full(batch_size, (cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
        self.normed_Z = jnp.full(batch_size, (cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
        self.A = jnp.full(batch_size, cfg["A"]["val"])
        self.fract = jnp.full(batch_size, cfg["fract"]["val"])

    def __call__(self):
        unnormed_Ti = self.normed_Ti * self.Ti_scale + self.Ti_shift
        unnormed_Z = self.normed_Z * self.Z_scale + self.Z_shift

        return {"Ti": unnormed_Ti, "Z": unnormed_Z, "A": self.A, "fract": self.fract}


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

    def __init__(self, cfg, batch_size: int):
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
        self.normed_amp1 = jnp.full(batch_size, (cfg["amp1"]["val"] - self.amp1_shift) / self.amp1_scale)
        self.normed_amp2 = jnp.full(batch_size, (cfg["amp2"]["val"] - self.amp2_shift) / self.amp2_scale)
        self.normed_amp3 = jnp.full(batch_size, (cfg["amp3"]["val"] - self.amp3_shift) / self.amp3_scale)
        self.normed_ne_gradient = jnp.full(
            batch_size, (cfg["ne_gradient"]["val"] - self.ne_gradient_shift) / self.ne_gradient_scale
        )
        self.normed_Te_gradient = jnp.full(
            batch_size, (cfg["Te_gradient"]["val"] - self.Te_gradient_shift) / self.Te_gradient_scale
        )
        self.normed_ud = jnp.full(batch_size, (cfg["ud"]["val"] - self.ud_shift) / self.ud_scale)
        self.normed_vA = jnp.full(batch_size, (cfg["Va"]["val"] - self.vA_shift) / self.vA_scale)
        self.normed_lam = jnp.full(batch_size, (cfg["lam"]["val"] - self.lam_shift) / self.lam_scale)

    def __call__(self):
        unnormed_lam = self.normed_lam * self.lam_scale + self.lam_shift
        unnormed_amp1 = self.normed_amp1 * self.amp1_scale + self.amp1_shift
        unnormed_amp2 = self.normed_amp2 * self.amp2_scale + self.amp2_shift
        unnormed_amp3 = self.normed_amp3 * self.amp3_scale + self.amp3_shift
        unnormed_ne_gradient = self.normed_ne_gradient * self.ne_gradient_scale + self.ne_gradient_shift
        unnormed_Te_gradient = self.normed_Te_gradient * self.Te_gradient_scale + self.Te_gradient_shift
        unnormed_ud = self.normed_ud * self.ud_scale + self.ud_shift
        unnormed_vA = self.normed_vA * self.vA_scale + self.vA_shift
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

    def __init__(self, param_cfg, num_params: int):
        super().__init__()
        self.electron = ElectronParams(param_cfg["electron"], num_params)
        self.ions = []
        for species in param_cfg.keys():
            if "ion" in species:
                self.ions.append(IonParams(param_cfg[species], num_params))

        assert len(self.ions) > 0, "No ion species found in input deck"
        self.general = GeneralParams(param_cfg["general"], num_params)

    def __call__(self):
        ion_params = {f"ion-{i+1}": ion() for i, ion in enumerate(self.ions)}
        return {"electron": self.electron(), "general": self.general()} | ion_params


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
        num_dists = len(filter_spec.electron.fe)
        for i in range(num_dists):
            filter_spec = eqx.tree_at(lambda tree: tree.electron.fe[i].fvx.normed_m, filter_spec, replace=True)

    elif dist_type.casefold() == "mx":
        pass

    else:
        raise NotImplementedError(f"Untrainable distribution type: {dist_type}")

    return filter_spec
