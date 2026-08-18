from jax import numpy as jnp, vmap
from scipy.signal import find_peaks


from .instrument import irf
from .instrument import AngularIRF, SpectrometerIRF
from .modules.ts_params import ThomsonParams
from .physics.generate_spectra import FitModel


def _bin_average(arr, step, axis):
    """Average non-overlapping windows of length ``step`` along ``axis``.

    Vectorized replacement for ``jnp.array([jnp.average(arr[..., i:i+step], axis) for i
    in range(0, n, step)])``. A ragged final window (when ``n`` is not a multiple of
    ``step``) is averaged over its real elements only, matching the original
    list-comprehension semantics. ``step`` and the array shape are static at trace time,
    so this compiles to a single reshape+mean instead of unrolling the loop into the graph.
    """
    n = arr.shape[axis]
    n_bins = -(-n // step)  # ceil division -> matches len(range(0, n, step))
    pad = n_bins * step - n
    if pad:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, pad)
        arr = jnp.pad(arr, pad_width, constant_values=jnp.nan)
    arr = jnp.moveaxis(arr, axis, 0)
    arr = arr.reshape(n_bins, step, *arr.shape[1:])
    arr = jnp.nanmean(arr, axis=1)
    return jnp.moveaxis(arr, 0, axis)


def _irfs_from_config(cfg, scattering_angles):
    """Adapter from the input deck to device-agnostic IRF descriptions.

    This is the only place that knows how instrument-response settings are laid out in
    the deck; ``irf.py`` sees only the value objects. When a per-device factory is
    introduced this function moves there unchanged.

    Returns:
        (ele_irf, ion_irf, ats_irf): each either its value object or ``None`` if that
        channel is not in use. Built under exactly the conditions in which
        ``postprocess_theory`` calls the corresponding routine, so a channel that is
        never used is never described.
    """

    widths = cfg["other"]["detector_specs"]["widIRF"]
    normalize = cfg["other"]["detector_specs"]["norm"]
    spectype = cfg["other"]["extraoptions"]["spectype"]

    # The IRF-convolved spectrum is computed on a fine grid of `npts` points and then
    # averaged down onto the detector's wavelength pixels. That pixel count is the length
    # of the calibrated wavelength axis, which is CCDsize[0] (see
    # calibration.get_calibrations) -- not the 1024 this code used to hardcode. The two
    # agree only because OMEGA's CCD is square.
    n_spectral_pixels = int(cfg["other"]["CCDsize"][0])
    npts = int(cfg["other"]["npts"])

    def _spectrometer_irf(channel, stddev):
        if npts % n_spectral_pixels:
            raise ValueError(
                f"Cannot bin the {channel} spectrum onto the detector: npts={npts} is not a "
                f"multiple of n_spectral_pixels={n_spectral_pixels}. npts is derived from "
                f"CCDsize[1] * points_per_pixel, but the wavelength axis has CCDsize[0] "
                f"pixels, so this fails for a non-square CCD."
            )
        return SpectrometerIRF(
            spect_stddev=stddev, n_spectral_pixels=n_spectral_pixels, normalize=normalize
        )

    ele_irf = ion_irf = ats_irf = None

    if cfg["data"]["load_ion_spec"]:
        ion_irf = _spectrometer_irf("ion", widths["spect_stddev_ion"])

    if cfg["data"]["load_ele_spec"]:
        if spectype == "angular_full":
            # The deck stores these as FWHM while the 1D channels store a standard
            # deviation; normalizing that inconsistency is the adapter's job.
            ats_irf = AngularIRF(
                spect_stddev=widths["spect_FWHM_ele"] / 2.3548,
                ang_stddev=widths["ang_FWHM_ele"] / 2.3548,
                ang_axis=scattering_angles["angAxis"],
                normalize=normalize,
            )
        else:
            ele_irf = _spectrometer_irf("electron", widths["spect_stddev_ele"])

    return ele_irf, ion_irf, ats_irf


class ThomsonScatteringDiagnostic:
    """
    The SpectrumCalculator class wraps the FitModel class adding instrumental effects to the calculated spectrum so it
    can be compared to data.

    Notes:
        This Class will eventually be combined with FitModel in generate_spectra

    Args:
        cfg: Dict- configuration dictionary built from input deck
        scattering_angles: Dict- has fields containing the scattering angles the spectrum will be calculated at and the relative
        weights of each of the scattering angles in the final spectrum
    """

    def __init__(self, cfg, scattering_angles):
        super().__init__()
        self.cfg = cfg
        self.scattering_angles = scattering_angles
        self.model = FitModel(cfg, scattering_angles)
        self.ele_irf, self.ion_irf, self.ats_irf = _irfs_from_config(cfg, scattering_angles)

        if ("angular" in cfg["other"]["extraoptions"]["spectype"] 
            or "_interactive" in cfg["other"]["extraoptions"]["spectype"]):
            pass
        elif (
            "temporal" in cfg["other"]["extraoptions"]["spectype"]
            or "imaging" in cfg["other"]["extraoptions"]["spectype"]
            or "1d" in cfg["other"]["extraoptions"]["spectype"]
        ):
            self.model = vmap(self.model)
            self.postprocess_theory = vmap(self.postprocess_theory)
        else:
            raise NotImplementedError(f"Unknown spectype: {cfg['other']['extraoptions']['spectype']}")

    def postprocess_theory(self, modlE, modlI, lamAxisE, lamAxisI, amps, TSins):
        """
        Adds instrumental broadening to the synthetic Thomson spectrum.

        Args:
            modlE: Synthetic EPW Thomson spectra produced by FitModel
            modlI: Synthetic IAW Thomson spectra produced by FitModel
            lamAxisE: EPW wavelength axis produced by FitModel
            lamAxisI: IAW wavelength axis produced by FitModel
            amps: dictionary containing the scaling facotrs for
            TSins: dictionary of the Thomson scattering parameters

        Returns:
            ThryE: Synthetic Thomson spectrum with instrumental broadening
            ThryI: Synthetic Thomson spectrum with instrumental broadening
            lamAxisE: EPW wavelength axis
            lamAxisI: IAW wavelength axis

        """
        if self.cfg["data"]["load_ion_spec"]:
            lamAxisI, ThryI = irf.add_ion_IRF(self.ion_irf, lamAxisI, modlI, amps["i_amps"], TSins)
        else:
            ThryI = modlI

        if self.cfg["data"]["load_ele_spec"]:
            if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
                lamAxisE, ThryE = irf.add_ATS_IRF(self.ats_irf, lamAxisE, modlE, TSins)
            else:
                lamAxisE, ThryE = irf.add_electron_IRF(self.ele_irf, lamAxisE, modlE, amps["e_amps"], TSins)
        else:
            ThryE = modlE

        return ThryE, ThryI, lamAxisE, lamAxisI

    def reduce_ATS_to_resunit(self, ThryE, lamAxisE, TSins, batch):
        """
        Integrate synthetic angularly resolved Thomson spectra over a resolution unit. A resolution unit is 2D with a width in the spectral and angular domains.

        Args:
            ThryE: Synthetic angularly resolved spectrum
            lamAxisE: calibrated wavelength axis, should have a length equal to one dimension of ThryE
            TSins: dictionary of the Thomson scattering parameters
            batch: dictionary containing the data and amplitudes

        Returns:
            ThryE: The input synthetic angularly resolved spectrum integrated of the resolution unit and correspondingly downsized
            lamAxisE: the input wavelength axis integrated over a wavelngth resolution unit and correspondingly downsized

        """
        lam_step = round(ThryE.shape[1] / batch["e_data"].shape[1])
        ang_step = round(ThryE.shape[0] / self.cfg["other"]["CCDsize"][0])

        ThryE = _bin_average(ThryE, lam_step, axis=1)  # bin the wavelength axis
        ThryE = _bin_average(ThryE, ang_step, axis=0)  # bin the angular axis

        lamAxisE = _bin_average(lamAxisE, lam_step, axis=0)
        ThryE = ThryE[self.cfg["data"]["lineouts"]["start"] : self.cfg["data"]["lineouts"]["end"], :]
        ThryE = batch["e_amps"] * ThryE / jnp.amax(ThryE, axis=1, keepdims=True)
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"], TSins["general"]["amp1"] * ThryE, TSins["general"]["amp2"] * ThryE
        )
        return ThryE, lamAxisE

    def __call__(self, ts_params: ThomsonParams, batch):
        """
        Simulates the Thomson scattering diagnostic for a given set of parameters and input data by adding instrumental effects
        to the calculated spectrum.
        Args:
            ts_params (ThomsonParams): Object containing all the physical and experimental parameters required for the simulation.
            batch (dict): Dictionary containing electron and ion data arrays, amplitude arrays, and noise arrays. 
                Expected keys include:
                    - "e_amps": Electron amplitude array.
                    - "i_amps": Ion amplitude array.
                    - "noise_e": Noise array for electrons.
                    - "noise_i": Noise array for ions.
        Returns:
            ThryE (np.ndarray): Simulated electron spectrum with noise added.
            ThryI (np.ndarray): Simulated ion spectrum with noise added.
            lamAxisE (np.ndarray): Wavelength axis for electron spectrum.
            lamAxisI (np.ndarray): Wavelength axis for ion spectrum.
        Notes:
            - Applies post-processing to theoretical spectra and optionally reduces the spectrum to the resolution unit if specified in the configuration.
            - Adds experimental noise to the simulated spectra before returning.
        """

        physical_params = ts_params()
        modlE, modlI, lamAxisE, lamAxisI = self.model(physical_params)
        ThryE, ThryI, lamAxisE, lamAxisI = self.postprocess_theory(
            modlE, modlI, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, physical_params
        )
        if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
            ThryE, lamAxisE = self.reduce_ATS_to_resunit(ThryE, lamAxisE, physical_params, batch)

        ThryE = ThryE + batch["noise_e"]
        ThryI = ThryI + batch["noise_i"]

        return ThryE, ThryI, lamAxisE, lamAxisI

    def spectrum_breakdown(self, ts_params: ThomsonParams, batch):
        """
        Alternative version of the __call__ method which produces a detailied beakdown of all
        componenets that go into the calculated spectrum. Not intended to be used for angular data.

        Args:
            ts_params (ThomsonParams): Object containing all the physical and experimental parameters required for the simulation.
            batch (dict): Dictionary containing electron and ion data arrays, amplitude arrays, and noise arrays. 
        Returns:
            modlE (np.ndarray): Electron spectrum with instrumental effects applied.
            modlI (np.ndarray): Ion spectrum with instrumental effects applied.
            ThryE (np.ndarray): Raw theoretical electron spectrum.
            ThryI (np.ndarray): Raw theoretical ion spectrum.
            eIRF (np.ndarray): Electron IRF spectrum.
            iIRF (np.ndarray): Ion IRF spectrum.
            lamAxisE (np.ndarray): Wavelength axis for electron spectrum.
            lamAxisI (np.ndarray): Wavelength axis for ion spectrum.
            lamAxisE_raw (np.ndarray): Raw wavelength axis for electron spectrum.
            lamAxisI_raw (np.ndarray): Raw wavelength axis for ion spectrum.

        """

        physical_params = ts_params()
        fmod = FitModel(self.cfg, self.scattering_angles)
        modlE, modlI, ThryE, ThryI, lamAxisE_raw, lamAxisI_raw = vmap(fmod.detailed_spectrum)(physical_params)
        # modlE, modlI, ThryE, ThryI, lamAxisE, lamAxisI = self.model.detailed_spectrum(physical_params)
        modlE, modlI, lamAxisE, lamAxisI = self.postprocess_theory(
            modlE,
            modlI,
            lamAxisE_raw,
            lamAxisI_raw,
            {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]},
            physical_params,
        )
        # add the IRF to a delta function of the peak locations to produce a IRF only plot

        eIRF = jnp.zeros_like(modlE)
        if self.cfg["data"]["load_ele_spec"]:
            for i in range(jnp.shape(modlE)[0]):
                peaksE, propertiesE = find_peaks(modlE[i], prominence=0.1)
                eIRF = eIRF.at[i, peaksE[jnp.argmax(propertiesE["prominences"])]].set(1.0)
                if len(propertiesE["prominences"]) > 1:
                    eIRF = eIRF.at[i, peaksE[jnp.argpartition(propertiesE["prominences"], -2)[-2]]].set(1.0)

        iIRF = jnp.zeros_like(modlI)
        if self.cfg["data"]["load_ion_spec"]:
            for i in range(jnp.shape(modlI)[0]):
                try:
                    peaksI, propertiesI = find_peaks(modlI[i], prominence=0.1)
                    iIRF = iIRF.at[i, peaksI[jnp.argmax(propertiesI["prominences"])]].set(1.0)
                    if len(propertiesI["prominences"]) > 1:
                        iIRF = iIRF.at[i, peaksI[jnp.argpartition(propertiesI["prominences"], -2)[-2]]].set(1.0)
                except BaseException:
                    print("Unable to locate peak IRF may not be plotted")

        eIRF, iIRF, lamAxisE, lamAxisI = self.postprocess_theory(
            eIRF, iIRF, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, physical_params
        )

        if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
            modlE, lamAxisE = self.reduce_ATS_to_resunit(ThryE, lamAxisE, physical_params, batch)

        modlE = modlE + batch["noise_e"]
        modlI = modlI + batch["noise_i"]

        if self.cfg["data"]["load_ele_spec"]:
            ThryE = jnp.reshape(batch["e_amps"], (-1, 1, 1, 1)) * ThryE / jnp.amax(ThryE)
            eIRF = jnp.reshape(batch["e_amps"], (-1, 1)) * eIRF / jnp.amax(eIRF)
        if self.cfg["data"]["load_ion_spec"]:
            ThryI = jnp.reshape(batch["i_amps"], (-1, 1, 1, 1)) * ThryI / jnp.amax(ThryI)
            iIRF = jnp.reshape(batch["i_amps"], (-1, 1)) * iIRF / jnp.amax(iIRF)

        return modlE, modlI, ThryE, ThryI, eIRF, iIRF, lamAxisE, lamAxisI, lamAxisE_raw, lamAxisI_raw
