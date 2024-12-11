from typing import Tuple
from jax import numpy as jnp


def add_ATS_IRF(config, sas, lamAxisE, modlE, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies a 2D gaussian smoothing to angular Thomson data to account for the instrument response of the diagnostic.
    todo: improve doc and typehints

    Args:
        config: Dict- configuration dictionary built from input deck
        sas: Dict- fields 'sa' and 'weights' with sizes (n,) where n is the number of angles where the spectrum is computed. 'sa' contains the scattering angles in degrees and 'weights' included the normalized relative weights each angle contributes to final spectrum.
        lamAxisE: Array- wavelengths the spectrum is computed at in nm
        modlE: Array- synthetic spectra produced by the formfactor routine
        amps: float- maximum amplitude of the data, used to rescale model to the data
        TSins: Dict- Dictionary of the parameters and thier values
        lam: float- probe wavelength in nm

    Returns:

    """
    stddev_lam = config["other"]["PhysParams"]["widIRF"]["spect_FWHM_ele"] / 2.3548
    stddev_ang = config["other"]["PhysParams"]["widIRF"]["ang_FWHM_ele"] / 2.3548
    # Conceptual_origin so the convolution donsn't shift the signal
    origin_lam = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    origin_ang = (jnp.amax(sas["angAxis"]) + jnp.amin(sas["angAxis"])) / 2.0
    inst_func_lam = jnp.squeeze(
        (1.0 / (stddev_lam * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((lamAxisE - origin_lam) ** 2.0) / (2.0 * (stddev_lam) ** 2.0))
    )  # Gaussian
    inst_func_ang = jnp.squeeze(
        (1.0 / (stddev_ang * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((sas["angAxis"] - origin_ang) ** 2.0) / (2.0 * (stddev_ang) ** 2.0))
    )  # Gaussian
    ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    # ThryE = jnp.array([fftconvolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    ThryE = jnp.array([jnp.convolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
    # ThryE = jnp.array([fftconvolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])

    ThryE = jnp.amax(modlE, axis=1, keepdims=True) / jnp.amax(ThryE, axis=1, keepdims=True) * ThryE

    if config["other"]["PhysParams"]["norm"] > 0:
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"],
            TSins["general"]["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < TSins["general"]["lam"]])),
            TSins["general"]["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > TSins["general"]["lam"]])),
        )
    return lamAxisE, ThryE


def add_ion_IRF(config, lamAxisI, modlI, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Ion IRF (Instrument Response Function?)

    todo: improve doc and typehints

    Args:
        config:
        lamAxisI:
        modlI:
        amps:
        TSins:

    Returns:

    """
    stddevI = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
    if stddevI:
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["general"]["amp3"] * amps * ThryI / jnp.amax(ThryI)
            # lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
    else:
        ThryI = modlI

    return lamAxisI, ThryI


def add_electron_IRF(config, lamAxisE, modlE, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    electron IRF (Instrument Response Function?)

    todo: improve doc and typehints

    Args:
        config:
        lamAxisE:
        modlE:
        amps:
        TSins:
        lam:

    Returns:

    """
    stddevE = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
    # Conceptual_origin so the convolution doesn't shift the signal
    originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    inst_funcE = jnp.squeeze(
        (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
    )  # Gaussian
    ThryE = jnp.convolve(modlE, inst_funcE, "same")
    ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

    if config["other"]["PhysParams"]["norm"] > 0:
        ThryE = jnp.where(
            lamAxisE < lam,
            TSins["general"]["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < TSins["general"]["lam"]])),
            TSins["general"]["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > TSins["general"]["lam"]])),
        )

    ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
    if config["other"]["PhysParams"]["norm"] == 0:
        lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
        ThryE = amps * ThryE / jnp.amax(ThryE)
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"], TSins["general"]["amp1"] * ThryE, TSins["general"]["amp2"] * ThryE
        )

    return lamAxisE, ThryE
