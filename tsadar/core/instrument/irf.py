"""Instrument response function: spectral (and angular) blur, then binning onto pixels.

This is device-dependent code, not physics. It holds both the value objects describing
*this* detector's response and the routines that apply it, because those two belong to
the same pipeline stage -- sibling stages (aperture weighting, notch filters, pixel
reduction) get their own modules alongside this one.

Nothing in ``tsadar.core.instrument`` may import from ``tsadar.utils`` or read a config
dict. Translating an input deck into these value objects is the caller's job, so that
porting TSADAR to a new diagnostic means constructing them directly rather than
mimicking OMEGA's nested deck layout to reach a constructor.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from jax import numpy as jnp, vmap


@dataclass(frozen=True)
class SpectrometerIRF:
    """Response of a 1D spectrometer: spectral blur, then binning onto detector pixels.

    Every field is required. There are no defaults on purpose: a default for
    ``n_spectral_pixels`` would be silently correct on OMEGA and silently wrong
    elsewhere, which is the failure mode this module exists to prevent.

    This carries only *static* configuration, so it is an ordinary dataclass rather than
    an ``equinox`` module -- nothing here is a traced JAX leaf. The fitted instrument
    nuisance parameters (``amp1``/``amp2``/``amp3``) still arrive separately via the
    parameter tree.

    Args:
        spect_stddev: Gaussian spectral IRF standard deviation, in nm. A falsy value
            means "no spectral IRF" and the convolution is skipped (ion channel only).
        n_spectral_pixels: Number of wavelength pixels on the detector. The convolved
            spectrum is computed on a fine grid and averaged down onto this many bins.
        normalize: Normalization mode. ``0`` scales the spectrum to the measured data
            amplitude; ``> 0`` normalizes each wing to unity and scales it by the fitted
            ``amp1``/``amp2``.
    """

    spect_stddev: float
    n_spectral_pixels: int
    normalize: int


@dataclass(frozen=True, eq=False)
class AngularIRF:
    """Response of a 2D angularly-resolved detector: separable blur in wavelength and angle.

    ``eq=False`` because ``ang_axis`` is an array, and elementwise comparison would make
    a generated ``__eq__`` raise rather than return a bool.

    Args:
        spect_stddev: Gaussian spectral IRF standard deviation, in nm.
        ang_stddev: Gaussian angular IRF standard deviation, in degrees.
        ang_axis: Calibrated angular axis of the detector, in degrees.
        normalize: Normalization mode, as for :class:`SpectrometerIRF`.
    """

    spect_stddev: float
    ang_stddev: float
    ang_axis: np.ndarray
    normalize: int


def add_ATS_IRF(irf: AngularIRF, lamAxisE, modlE, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies a 2D Gaussian smoothing to angular Thomson scattering data to account for the instrument response function (IRF) of the diagnostic.
    This function convolves the synthetic spectra with Gaussian kernels along both the wavelength and angular axes, simulating the broadening effects introduced by the instrument. The resulting spectrum is optionally normalized according to the IRF description.
    Args:
        irf (AngularIRF): Description of the angular detector's response.
        lamAxisE (jnp.ndarray): Array of wavelengths (in nm) at which the spectrum is computed.
        modlE (jnp.ndarray): Synthetic spectra produced by the formfactor routine, shape (n_angles, n_wavelengths).
        TSins (dict): Dictionary of Thomson scattering instrument parameters and their values.
    Returns:
        lamAxisE (jnp.ndarray): Wavelength axis (in nm).
        ThryE (jnp.ndarray): Smoothed and optionally normalized synthetic spectra, shape (n_angles, n_wavelengths).
    """

    stddev_lam = irf.spect_stddev
    stddev_ang = irf.ang_stddev
    # Conceptual_origin so the convolution donsn't shift the signal
    origin_lam = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    origin_ang = (jnp.amax(irf.ang_axis) + jnp.amin(irf.ang_axis)) / 2.0
    inst_func_lam = jnp.squeeze(
        (1.0 / (stddev_lam * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((lamAxisE - origin_lam) ** 2.0) / (2.0 * (stddev_lam) ** 2.0))
    )  # Gaussian
    inst_func_ang = jnp.squeeze(
        (1.0 / (stddev_ang * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((irf.ang_axis - origin_ang) ** 2.0) / (2.0 * (stddev_ang) ** 2.0))
    )  # Gaussian
    # Separable 2D convolution: smooth along the angular axis (axis 0) for every
    # wavelength column, then along the wavelength axis (axis 1) for every angle row.
    # vmap batches each 1D convolution into a single op instead of unrolling a Python
    # loop over thousands of columns/rows into the traced graph (huge XLA compile cost).
    ThryE = vmap(lambda col: jnp.convolve(col, inst_func_ang, "same"), in_axes=1, out_axes=1)(modlE)
    ThryE = vmap(lambda row: jnp.convolve(row, inst_func_lam, "same"), in_axes=0, out_axes=0)(ThryE)

    ThryE = jnp.amax(modlE, axis=1, keepdims=True) / jnp.amax(ThryE, axis=1, keepdims=True) * ThryE

    if irf.normalize > 0:
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"],
            TSins["general"]["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < TSins["general"]["lam"]])),
            TSins["general"]["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > TSins["general"]["lam"]])),
        )
    return lamAxisE, ThryE


def add_ion_IRF(irf: SpectrometerIRF, lamAxisI, modlI, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies an instrumental response function (IRF) to the ion spectral model and optionally normalizes the result.
    Parameters:
        irf (SpectrometerIRF): Description of the ion spectrometer's response.
        lamAxisI (jnp.ndarray): Wavelength axis for the ion spectrum.
        modlI (jnp.ndarray): Theoretical ion spectrum model to which the IRF will be applied.
        amps (float or jnp.ndarray): Amplitude scaling factor(s) for the spectrum.
        TSins (dict): Dictionary containing additional scaling parameters, specifically 'general' -> 'amp3'.
    Returns:
        lamAxisI (jnp.ndarray): The wavelength axis, possibly averaged over batches if the IRF is applied.
        ThryI (jnp.ndarray): The processed ion spectrum after convolution with the IRF and optional normalization.
    """

    stddevI = irf.spect_stddev
    if stddevI:
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(irf.n_spectral_pixels, -1), axis=1)
        #print(f"modlI max {jnp.max(modlI)}")
        #print(f"ThryI max {jnp.max(ThryI)}")
        #print(f"amps max {jnp.max(amps)}")

        if irf.normalize == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(irf.n_spectral_pixels, -1), axis=1)
            ThryI = TSins["general"]["amp3"] * amps * ThryI / jnp.amax(ThryI)
            # lamAxisE = jnp.average(lamAxisE.reshape(irf.n_spectral_pixels, -1), axis=1)
    else:
        ThryI = modlI

    #print(f"final ThryI max {jnp.max(ThryI)}")
    return lamAxisI, ThryI


def add_electron_IRF(irf: SpectrometerIRF, lamAxisE, modlE, amps, TSins) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies an instrumental response function (IRF) to an electron model spectrum and normalizes the result.
    This function convolves the input electron model spectrum (`modlE`) with a Gaussian IRF defined by the IRF
    description, normalizes the convolved spectrum according to that description and the signal parameters, and
    optionally averages and rescales the output based on normalization settings.
    Args:
        irf (SpectrometerIRF): Description of the electron spectrometer's response.
        lamAxisE (jnp.ndarray): Wavelength axis for the electron spectrum.
        modlE (jnp.ndarray): Model electron spectrum to which the IRF will be applied.
        amps (float or jnp.ndarray): Amplitude scaling factor(s) for the output spectrum.
        TSins (dict): Dictionary containing signal parameters, including normalization wavelengths and amplitudes.
    Returns:
        lamAxisE (jnp.ndarray): The wavelength axis, possibly averaged over batches if the IRF is applied.
        ThryE (jnp.ndarray): The processed electron spectrum after convolution with the IRF and optional normalization.
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the (possibly averaged) wavelength axis and the processed, normalized electron spectrum.
    """

    stddevE = irf.spect_stddev
    # Conceptual_origin so the convolution doesn't shift the signal
    originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    inst_funcE = jnp.squeeze(
        (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
    )  # Gaussian
    ThryE = jnp.convolve(modlE, inst_funcE, "same")
    ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

    if irf.normalize > 0:
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"],
            TSins["general"]["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < TSins["general"]["lam"]])),
            TSins["general"]["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > TSins["general"]["lam"]])),
        )

    ThryE = jnp.average(ThryE.reshape(irf.n_spectral_pixels, -1), axis=1)
    if irf.normalize == 0:
        lamAxisE = jnp.average(lamAxisE.reshape(irf.n_spectral_pixels, -1), axis=1)
        ThryE = amps * ThryE / jnp.amax(ThryE)
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"], TSins["general"]["amp1"] * ThryE, TSins["general"]["amp2"] * ThryE
        )

    return lamAxisE, ThryE
