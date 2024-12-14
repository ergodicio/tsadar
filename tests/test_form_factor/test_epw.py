import numpy as np
from numpy.testing import assert_allclose
from jax import config
from jax import jit
from jax import numpy as jnp
from copy import deepcopy
import yaml
from flatten_dict import flatten, unflatten

config.update("jax_enable_x64", True)

from scipy.signal import find_peaks
from tsadar.core.physics.form_factor import FormFactor
from tsadar.core.modules import ThomsonParams

# from tsadar.distribution_functions.gen_num_dist_func import DistFunc


def test_epw():
    """
    Tests the behaviour of 1D formfactor calculation ensuring it accurately reproduces the EPW dispersion relation

    """
    with open("tests/configs/epw_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/epw_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    # Test #1: Bohm-Gross test, calculate a spectrum and compare the resonance to the Bohm gross dispersion relation
    npts = 2048
    # num_dist_func = DistFunc(config["parameters"]["electron"])
    # vcur, fecur = num_dist_func(config["parameters"]["electron"]["m"]["val"])
    ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
    electron_form_factor = FormFactor(
        [400, 700],
        npts=npts,
        lam_shift=config["data"]["ele_lam_shift"],
        scattering_angles={"sa": np.array([60])},
        num_grad_points=config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
    )

    sa = np.array([60])
    # params = {
    #     "general": {
    #         "Va": config["parameters"]["general"]["Va"]["val"],
    #         "ud": config["parameters"]["general"]["ud"]["val"],
    #     }
    # }

    physical_params = ts_params()
    ThryE, lamAxisE = jit(electron_form_factor)(physical_params)
    ThryE = np.squeeze(ThryE)
    test = deepcopy(np.asarray(ThryE))
    peaks, peak_props = find_peaks(test, height=(0.01, 0.5), prominence=0.05)
    highest_peak_index = peaks[np.argmax(peak_props["peak_heights"])]
    second_highest_peak_index = peaks[np.argsort(peak_props["peak_heights"])[0]]

    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)

    lams = lamAxisE[0, [highest_peak_index, second_highest_peak_index], 0]
    omgs = 2 * jnp.pi * C / lams  # peak frequencies
    omgpe = constants * jnp.sqrt(0.2 * 1e20)
    omgL = 2 * np.pi * 1e7 * C / config["parameters"]["general"]["lam"]["val"]  # laser frequency Rad / s
    ks = jnp.sqrt(omgs**2 - omgpe**2) / C
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C
    k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sa * jnp.pi / 180))
    vTe = jnp.sqrt(0.5 / Me)
    omg = jnp.sqrt(omgpe**2 + 3 * k**2 * vTe**2)
    omgs2 = [omgL + omg[0], omgL - omg[1]]
    assert_allclose(omgs, omgs2, rtol=1e-2)


if __name__ == "__main__":
    test_epw()
