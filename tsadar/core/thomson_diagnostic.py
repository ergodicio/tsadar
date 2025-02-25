from jax import numpy as jnp, vmap
from scipy.signal import find_peaks


from .modules import ThomsonParams
from .physics import irf
from .physics.generate_spectra import FitModel


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

        if (
            "temporal" in cfg["other"]["extraoptions"]["spectype"]
            or "imaging" in cfg["other"]["extraoptions"]["spectype"]
            or "1d" in cfg["other"]["extraoptions"]["spectype"]
        ):
            self.model = vmap(self.model)
            self.postprocess_theory = vmap(self.postprocess_theory)
        elif "angular" in cfg["other"]["extraoptions"]["spectype"]:
            pass
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
            TSins:

        Returns:

        """
        if self.cfg["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, ThryI = irf.add_ion_IRF(self.cfg, lamAxisI, modlI, amps["i_amps"], TSins)
        else:
            ThryI = modlI

        if self.cfg["other"]["extraoptions"]["load_ele_spec"]:
            if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
                lamAxisE, ThryE = irf.add_ATS_IRF(
                    self.cfg, self.scattering_angles, lamAxisE, modlE, amps["e_amps"], TSins
                )
            else:
                lamAxisE, ThryE = irf.add_electron_IRF(self.cfg, lamAxisE, modlE, amps["e_amps"], TSins)
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

        ThryE = jnp.array([jnp.average(ThryE[:, i : i + lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
        ThryE = jnp.array([jnp.average(ThryE[:, i : i + ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])

        lamAxisE = jnp.array(
            [jnp.average(lamAxisE[i : i + lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)]
        )
        ThryE = ThryE[self.cfg["data"]["lineouts"]["start"] : self.cfg["data"]["lineouts"]["end"], :]
        ThryE = batch["e_amps"] * ThryE / jnp.amax(ThryE, axis=1, keepdims=True)
        ThryE = jnp.where(
            lamAxisE < TSins["general"]["lam"], TSins["general"]["amp1"] * ThryE, TSins["general"]["amp2"] * ThryE
        )
        return ThryE, lamAxisE

    def __call__(self, ts_params: ThomsonParams, batch):
        """
        TODO

        Args:
            params: Dict- contains name value pairs for all the parameters from the input deck
            batch: Dict- contains the electron and ion data arrays as well as their amplitude arrays and noise arrays.

        Returns:

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

    def sprectrum_breakdown(self, ts_params: ThomsonParams, batch):
        """
        Alternaticve version of the __call__ method which produces a detailied beakdown of all
        componenets that go into the calculated spectrum. Not intended to be used for angular data.

        Args:
            ts_params: ThomsonParam- an instance of the ThomsonParams object which contains all the input parameters for 
                a spectrum to be calculated
            batch: Dict- contains the electron and ion data arrays as well as their amplitude arrays and noise arrays.

        Returns:

        """

        physical_params = ts_params()
        fmod= FitModel(self.cfg, self.scattering_angles)
        modlE, modlI, ThryE, ThryI, lamAxisE_raw, lamAxisI_raw=vmap(fmod.detailed_spectrum)(physical_params)
        #modlE, modlI, ThryE, ThryI, lamAxisE, lamAxisI = self.model.detailed_spectrum(physical_params)
        modlE, modlI, lamAxisE, lamAxisI = self.postprocess_theory(
            modlE, modlI, lamAxisE_raw, lamAxisI_raw, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, physical_params
        )
        #add the IRF to a delta function of the peak locations to produce a IRF only plot
        
        eIRF = jnp.zeros_like(modlE)
        if self.cfg["other"]["extraoptions"]["load_ele_spec"]:
            for i in range(jnp.shape(modlE)[0]):
                peaksE, propertiesE = find_peaks(modlE[i], prominence=1)
                eIRF = eIRF.at[i,peaksE[jnp.argmax(propertiesE['prominences'])]].set(1.0)
                if len(propertiesE['prominences'])>1:
                    eIRF = eIRF.at[i,peaksE[jnp.argpartition(propertiesE['prominences'],-2)[-2]]].set(1.0)
        
        iIRF = jnp.zeros_like(modlI)
        if self.cfg["other"]["extraoptions"]["load_ion_spec"]:
            for i in range(jnp.shape(modlI)[0]):
                peaksI, propertiesI = find_peaks(modlI[i], prominence=1)
                iIRF = iIRF.at[i,peaksI[jnp.argmax(propertiesI['prominences'])]].set(1.0)
                iIRF = iIRF.at[i,peaksI[jnp.argpartition(propertiesI['prominences'],-2)[-2]]].set(1.0)
            
        # peaksI, propertiesI = find_peaks(modlI,prominence = 2, width = 4)
        # eIRF= jnp.zeros(1024)
        # eIRF[peaksE[0]] = 1.0
        # eIRF[peaksE[1]] = 1.0
        # iIRF= jnp.zeros(1024)
        # iIRF[peaksI[0]] = 1.0
        # iIRF[peaksI[1]] = 1.0
        eIRF, iIRF, lamAxisE, lamAxisI = self.postprocess_theory(
            eIRF, iIRF, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, physical_params
        )


        if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
            modlE, lamAxisE = self.reduce_ATS_to_resunit(ThryE, lamAxisE, physical_params, batch)

        modlE = modlE + batch["noise_e"]
        modlI = modlI + batch["noise_i"]
        
        if self.cfg["other"]["extraoptions"]["load_ele_spec"]:
            ThryE = jnp.reshape(batch["e_amps"],(-1,1,1,1)) * ThryE / jnp.amax(ThryE)
            eIRF = jnp.reshape(batch["e_amps"],(-1,1)) * eIRF/ jnp.amax(eIRF)
        if self.cfg["other"]["extraoptions"]["load_ion_spec"]:
            ThryI = jnp.reshape(batch["i_amps"],(-1,1,1,1)) * ThryI / jnp.amax(ThryI)
            iIRF = jnp.reshape(batch["i_amps"],(-1,1)) * iIRF

        return modlE, modlI, ThryE, ThryI, eIRF, iIRF, lamAxisE, lamAxisI, lamAxisE_raw, lamAxisI_raw