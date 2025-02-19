from jax import numpy as jnp, vmap

from tsadar.utils.data_handling.calibration import get_scattering_angles, get_calibrations

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

    def __init__(self, cfg, angular=False):

        super().__init__()

        self.cfg, self.scattering_angles = self.initialize_scattering_angles(cfg, angular)

        self.model = FitModel(cfg, self.scattering_angles)

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
    
    def initialize_scattering_angles(self, config, angular):
        """
        Initializes scattering angles and weights.

        Args:
            config: Configuration dictionary

        Returns:
            Updated configuration dictionary with scattering angles and weights
        """
        config["other"]["lamrangE"] = [
            config["data"]["fit_rng"]["forward_epw_start"],
            config["data"]["fit_rng"]["forward_epw_end"],
        ]
        config["other"]["lamrangI"] = [
            config["data"]["fit_rng"]["forward_iaw_start"],
            config["data"]["fit_rng"]["forward_iaw_end"],
        ]
        config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
        scattering_angles = get_scattering_angles(config)

        if angular:
            [axisxE, _, _, _, _, _] = get_calibrations(
                104000, config["other"]["extraoptions"]["spectype"], 0.0, config["other"]["CCDsize"]
            )  # shot number hardcoded to get calibration
            config["other"]["extraoptions"]["spectype"] = "angular_full"

            scattering_angles["angAxis"] = axisxE

        return config, scattering_angles
    
    def get_cfg(self):
        """
        Getter method for the cfg attribute

        Returns:
            The configuration dictionary
        """

        return self.cfg