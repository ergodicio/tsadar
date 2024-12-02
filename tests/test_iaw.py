import numpy as np
from jax import config
from jax import jit
from jax import numpy as jnp
from copy import deepcopy
import yaml
from flatten_dict import flatten, unflatten

config.update("jax_enable_x64", True)
from numpy.testing import assert_allclose
from scipy.signal import find_peaks
from tsadar.model.physics.form_factor import FormFactor
from tsadar.distribution_functions.gen_num_dist_func import DistFunc


def test_iaw():
    """
    Test #2: IAW test, calculate a spectrum and compare the resonance to the IAW dispersion relation

    Returns:

    """
    with open("tests/configs/epw_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/epw_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    Mp = Me * 1836.1  # proton mass keV/C^2
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm

    num_dist_func = DistFunc(config["parameters"]["electron"])
    vcur, fecur = num_dist_func(config["parameters"]["electron"]["m"]["val"])

    ion_form_factor = FormFactor(
        [525, 528],
        npts=8192,
        fe_dim=num_dist_func.dim,
        vax=vcur,
    )

    # xie = np.linspace(-7, 7, 1024)
    # ion_form_factor = FormFactor([525, 528], npts=8192)

    sa = np.array([60])
    params = {
        "general": {
            "Va": config["parameters"]["general"]["Va"]["val"],
            "ud": config["parameters"]["general"]["ud"]["val"],
        }
    }
    # num_dist_func = get_num_dist_func({"DLM": []}, xie)
    # fecur = num_dist_func(2.0)
    # lam = 526.5
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)

    # inps = dict(Ti=0.2, Z=1, A=1, fract=1, Va=0, ud=0)
    # cur_ne = np.array([0.2 * 1e20])
    # cur_Te = 0.5

    # ThryI, lamAxisI = jit(ion_form_factor)(inps, cur_ne, cur_Te, sa, (fecur, xie), lam)
    ThryI, lamAxisI = jit(ion_form_factor)(
        params,
        jnp.array(config["parameters"]["electron"]["ne"]["val"] * 1e20).reshape(1, 1),
        jnp.array(config["parameters"]["electron"]["Te"]["val"]).reshape(1, 1),
        config["parameters"]["ion-1"]["A"]["val"],
        config["parameters"]["ion-1"]["Z"]["val"],
        config["parameters"]["ion-1"]["Ti"]["val"],
        config["parameters"]["ion-1"]["fract"]["val"],
        sa,
        (fecur, vcur),
        config["parameters"]["general"]["lam"]["val"],
    )

    ThryI = jnp.real(ThryI)
    ThryI = jnp.mean(ThryI, axis=0)

    ThryI = np.squeeze(ThryI)
    test = deepcopy(np.asarray(ThryI))
    peaks, peak_props = find_peaks(test, height=0.1, prominence=0.2)
    highest_peak_index = peaks[np.argmax(peak_props["peak_heights"])]
    second_highest_peak_index = peaks[np.argpartition(peak_props["peak_heights"], -2)[-2]]

    lams = lamAxisI[0, [highest_peak_index, second_highest_peak_index], 0]
    omgpe = constants * jnp.sqrt(0.2 * 1e20)
    omgL = 2 * np.pi * 1e7 * C / config["parameters"]["general"]["lam"]["val"]  # laser frequency Rad / s
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C

    omgs = 2 * jnp.pi * C / lams  # peak frequencies
    omg = 2 * kL * jnp.sqrt((0.5 + 3 * 0.2) / Mp)
    omgs2 = [omgL + omg, omgL - omg]

    assert_allclose(omgs2, omgs, rtol=1e-2)


if __name__ == "__main__":
    test_iaw()
