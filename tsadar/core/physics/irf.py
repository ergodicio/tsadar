from typing import Tuple
from jax import numpy as jnp


def add_ATS_IRF(config, sas, lamAxisE, modlE, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies a 2D Gaussian smoothing to angular Thomson scattering data to account for the instrument response function (IRF) of the diagnostic.
    This function convolves the synthetic spectra with Gaussian kernels along both the wavelength and angular axes, simulating the broadening effects introduced by the instrument. The resulting spectrum is optionally normalized according to configuration parameters.
    Args:   
        config (dict): Configuration dictionary containing instrument and normalization parameters.
        sas (dict): Dictionary with keys 'sa' (scattering angles in degrees) and 'weights' (normalized relative weights for each angle).
        lamAxisE (jnp.ndarray): Array of wavelengths (in nm) at which the spectrum is computed.
        modlE (jnp.ndarray): Synthetic spectra produced by the formfactor routine, shape (n_angles, n_wavelengths).
        amps (float): Maximum amplitude of the data, used to rescale the model to the data.
        TSins (dict): Dictionary of Thomson scattering instrument parameters and their values.
    Returns:
        lamAxisE (jnp.ndarray): Wavelength axis (in nm).
        ThryE (jnp.ndarray): Smoothed and optionally normalized synthetic spectra, shape (n_angles, n_wavelengths).
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
    Applies an instrumental response function (IRF) to the ion spectral model and optionally normalizes the result.
    Parameters:
        config (dict): Configuration dictionary containing physical parameters, including the standard deviation
            of the Gaussian IRF ('spect_stddev_ion') and normalization flag ('norm').
        lamAxisI (jnp.ndarray): Wavelength axis for the ion spectrum.
        modlI (jnp.ndarray): Theoretical ion spectrum model to which the IRF will be applied.
        amps (float or jnp.ndarray): Amplitude scaling factor(s) for the spectrum.
        TSins (dict): Dictionary containing additional scaling parameters, specifically 'general' -> 'amp3'.
    Returns:
        lamAxisI (jnp.ndarray): The wavelength axis, possibly averaged over batches if the IRF is applied.
        ThryI (jnp.ndarray): The processed ion spectrum after convolution with the IRF and optional normalization.
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
        #print(f"modlI max {jnp.max(modlI)}")
        #print(f"ThryI max {jnp.max(ThryI)}")
        #print(f"amps max {jnp.max(amps)}")

        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["general"]["amp3"] * amps * ThryI / jnp.amax(ThryI)
            # lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
    else:
        ThryI = modlI

    #print(f"final ThryI max {jnp.max(ThryI)}")
    return lamAxisI, ThryI


def add_electron_IRF(config, lamAxisE, modlE, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies an instrumental response function (IRF) to an electron model spectrum and normalizes the result.
    This function convolves the input electron model spectrum (`modlE`) with a Gaussian IRF defined by the configuration,
    normalizes the convolved spectrum according to the provided configuration and signal parameters, and optionally
    averages and rescales the output based on normalization settings.
    Args:
        config (dict): Configuration dictionary containing physical parameters, including the IRF width and normalization settings.
        lamAxisE (jnp.ndarray): Wavelength axis for the electron spectrum.
        modlE (jnp.ndarray): Model electron spectrum to which the IRF will be applied.
        amps (float or jnp.ndarray): Amplitude scaling factor(s) for the output spectrum.
        TSins (dict): Dictionary containing signal parameters, including normalization wavelengths and amplitudes.
    Returns:
        lamAxisE (jnp.ndarray): The wavelength axis, possibly averaged over batches if the IRF is applied.
        ThryE (jnp.ndarray): The processed electron spectrum after convolution with the IRF and optional normalization.
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the (possibly averaged) wavelength axis and the processed, normalized electron spectrum.
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
            lamAxisE < TSins["general"]["lam"],
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
