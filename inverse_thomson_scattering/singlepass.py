#Alternative code operation mode to fitter. Fitter matches the model to data, this code just calculates the model.
import numpy as np
import haiku as hk
import jax
import matplotlib.pyplot as plt
import os, mlflow, tempfile
from inverse_thomson_scattering.misc.calibration import get_scattering_angles
from inverse_thomson_scattering.model.parameters import TSParameterGenerator
from inverse_thomson_scattering.model.spectrum import SpectrumCalculator


def calc_spec(config):
    
    # get scattering angles and weights
    config["other"]["extraoptions"]["spectype"] = "temporal"
    stddev = dict()
    stddev["spect_stddev_ion"] = 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
    stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [400, 700]
    config["other"]["lamrangI"] = [524, 529]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
    
    
    sas = get_scattering_angles(config)
    dummy_batch = {'i_data': np.array([0]), 'e_data': np.array([0]), 'noise_e': 0, 'noise_i': 0}
    spec = SpectrumCalculator(config, sas, dummy_batch)
    parameterizer = hk.transform(TSParameterGenerator(config))
    params = parametrizer(dict())
    ThryE, ThryI, lamAxisE, lamAxisI = spec(params,[])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), squeeze=False, tight_layout=True, sharex=False)
    ax[0].plot(lamAxisE, ThryE)
    ax[0].set_title("Simulated Data, fontsize=14")
    ax[0].set_ylabel("Amp (arb. units)")
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].grid()
    
    ax[1].plot(lamAxisI, ThryI)
    ax[1].set_title("Simulated Data, fontsize=14")
    ax[1].set_ylabel("Amp (arb. units)")
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].grid()
    
    with tempfile.TemporaryDirectory() as td:
        fig.savefig(os.path.join(td, "simulated_data"), bbox_inches="tight")
        mlflow.log_artifacts(td)
    plt.close(fig)