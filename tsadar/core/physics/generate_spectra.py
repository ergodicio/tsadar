"""FitModel: wraps FormFactor to add finite-aperture/finite-volume instrument effects and produce the
electron (EPW) and ion (IAW) Thomson scattering spectra used for fitting and forward passes."""
from typing import Dict

from .form_factor import DEFAULT_N_BETA, FormFactor

from jax import numpy as jnp


class FitModel:
    """
    FitModel is a class that wraps the FormFactor class to add finite aperture and finite volume effects for generating Thomson scattering spectra. It manages configuration options, handles multiple scattering angles, and supports both electron and ion features, including gradient effects and angular spectra.

    Args: 

        config (Dict): Configuration dictionary built from the input deck, containing all static and runtime parameters for spectrum generation.
        scattering_angles (Dict): Dictionary containing the scattering angles at which the spectrum will be calculated and the relative weights for each angle.

    Methods:

        __call__(all_params: Dict):

            Calculates Thomson spectra corrected for finite aperture and optionally including plasma gradients, based on the current parameter dictionary. Does not compute or return the raw (pre-angle-averaged) theoretical spectra.

                all_params (Dict): Dictionary of current values for all active and static parameters.
                modlE: Electron plasma wave spectrum (array or int 0 if not loaded).
                modlI: Ion acoustic wave spectrum (array or int 0 if not loaded).
                lamAxisE: Wavelength axis for electron plasma wave (array or empty list if not loaded).
                lamAxisI: Wavelength axis for ion acoustic wave (array or empty list if not loaded).

        detailed_spectrum(all_params: Dict):

            Calculates both the total spectrum and all its components for postprocessing, including the raw (pre-angle-averaged) theoretical spectra ThryE/ThryI.

                all_params (Dict): Parameter dictionary.
                modlE: Electron plasma wave spectrum.
                modlI: Ion acoustic wave spectrum.
                ThryE: Detailed electron spectrum components.
                ThryI: Detailed ion spectrum components.
                lamAxisE: Wavelength axis for electron plasma wave.
                lamAxisI: Wavelength axis for ion acoustic wave.

        __call__ and detailed_spectrum share their implementation via the private _ion_spectrum_core/
        _electron_spectrum_core methods, which take a want_thry flag controlling whether the raw theoretical
        spectrum (and, for electrons, its extra filtering step) is computed and returned.

    """

    def __init__(self, config: Dict, scattering_angles: Dict):
        """
        Initializes the FitModel class, setting up static properties required for spectrum generation that remain unchanged across iterations.
        Args:
            config (Dict): Configuration dictionary constructed from the input deck, containing all necessary parameters for spectrum generation.
            scattering_angles (Dict): Dictionary containing the scattering angles at which the spectrum will be calculated, along with the relative weights for each angle in the final spectrum.
        Raises:
            AssertionError: If the number of gradient points for electron temperature (Te) and electron density (ne) are not the same.
        Attributes:
            config (Dict): Stores the provided configuration dictionary.
            scattering_angles (Dict): Stores the provided scattering angles and their weights.
            electron_form_factor (FormFactor): Form factor object for electrons, initialized with relevant parameters from the configuration.
            ion_form_factor (FormFactor): Form factor object for ions, initialized with relevant parameters from the configuration.
        """

        self.config = config
        self.scattering_angles = scattering_angles

        assert (
            config["parameters"]["general"]["Te_gradient"]["num_grad_points"]
            == config["parameters"]["general"]["ne_gradient"]["num_grad_points"]
        ), "Number of gradient points for Te and ne must be the same"
        num_grad_points = config["parameters"]["general"]["Te_gradient"]["num_grad_points"]

        ud_angle = (
            None
            if config["parameters"]["electron"]["fe"]["dim"] < 2
            else config["parameters"]["general"]["ud"]["angle"]
        )
        ion_species = [species for species in config["parameters"] if species.startswith("ion-")]
        va_angle = None
        if config["parameters"]["electron"]["fe"]["dim"] >= 2:
            va_angle = {
                species: config["parameters"][species]["Va"].get("angle", 0.0)
                for species in ion_species
            }

        # Angular resolution of the tabulated projection used by the 2D form factor. Set
        # to 0 to fall back to an exact rotation at every evaluation point, which is far
        # slower but is what the tabulation is validated against.
        n_beta = config["other"].get("n_beta", DEFAULT_N_BETA)

        if 'include_gains' in config["other"] and config["other"]["include_gains"]:
            calc_gain = {'calc': config["other"]["include_gains"], 'Ipump': config["other"]["Ipump_14"], 'beam_diam_um': config["other"]["beam_diam_um"]}
        else:
            calc_gain = {'calc': False, 'Ipump': 0, 'beam_diam_um': 0}

        self.electron_form_factor = FormFactor(
            config["other"]["lamrangE"],
            npts=config["other"]["npts"],
            lam_shift=config["data"]["ele_lam_shift"],
            scattering_angles=self.scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
            calc_gain=calc_gain,
            n_beta=n_beta,
        )
        self.ion_form_factor = FormFactor(
            config["other"]["lamrangI"],
            npts=config["other"]["npts"],
            lam_shift=0,
            scattering_angles=scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
            calc_gain=calc_gain,
            n_beta=n_beta,
        )

    def __call__(self, all_params: Dict):
        """
        Produces Thomson spectra corrected for finite aperture and optionally including gradients in the plasma
        conditions based off the current parameter dictionary. Calling this method will automatically choose the
        appropriate version of the formfactor class based off the dimension and distribute the conditions for
        multiple ion species to their respective inputs.


        Args:

            all_params: Parameter dictionary containing the current values for all active and static parameters. Only a few permanently static properties from the configuration dictionary will be used, everything else must be included in this input.

        Returns:
            
            modlE: calculated electron plasma wave spectrum as an array with length of npts. If an angular spectrum is calculated then it will be 2D. If the EPW is not loaded this is returned as the int 0.
            modlI: calculated ion acoustic wave spectrum as an array with length of npts. If the IAW is not loaded this is returned as the int 0.
            lamAxisE: electron plasma wave wavelength axis as an array with length of npts. If the EPW is not loaded this is returned as an empty list.
            lamAxisI: ion acoustic wave wavelength axis as an array with length of npts. If the IAW is not loaded this is returned as an empty list.
            all_params: The input all_params is returned

            
        """

        lamAxisI, modlI, _ = self._ion_spectrum_core(all_params, want_thry=False)
        lamAxisE, modlE, _ = self._electron_spectrum_core(all_params, want_thry=False)

        return modlE, modlI, lamAxisE, lamAxisI

    def detailed_spectrum(self, all_params: Dict):
        """
        Calculates detailed spectra for both electron plasma waves (EPW) and ion acoustic waves (IAW), including their wavelength axes and theoretical components, for postprocessing analysis. This method produces both the total spectrum and all its components for EPWs and IAWs, using the provided parameter dictionary. It is intended for postprocessing and requires all relevant parameters to be included in the input.
        Args:
            
            all_params (Dict): Dictionary containing current values for all active and static parameters. Most configuration properties must be included in this input, except for a few permanently static ones.
        
        Returns:
            
            modlE (np.ndarray or int): Calculated electron plasma wave spectrum as an array of length npts, or 0 if EPW is not loaded.
            modlI (np.ndarray or int): Calculated ion acoustic wave spectrum as an array of length npts, or 0 if IAW is not loaded.
            ThryE (np.ndarray): Theoretical components of the electron plasma wave spectrum.
            ThryI (np.ndarray): Theoretical components of the ion acoustic wave spectrum.
            lamAxisE (np.ndarray or list): Wavelength axis for the electron plasma wave spectrum, or empty list if EPW is not loaded.
            lamAxisI (np.ndarray or list): Wavelength axis for the ion acoustic wave spectrum, or empty list if IAW is not loaded.
        
        """

        lamAxisI, modlI, ThryI = self._ion_spectrum_core(all_params, want_thry=True)
        lamAxisE, modlE, ThryE = self._electron_spectrum_core(all_params, want_thry=True)

        return modlE, modlI, ThryE, ThryI, lamAxisE, lamAxisI

    def _ion_spectrum_core(self, all_params, want_thry):
        """
        Shared implementation behind ion_spectrum and ion_spectrum_detailed. want_thry controls whether the raw
        (pre-angle-averaged) ThryI is kept and returned; ion_spectrum passes False so callers that only need
        (lamAxisI, modlI) don't retain the larger intermediate array.
        """
        if self.config["data"]["load_ion_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryI, lamAxisI = self.ion_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded
            modlI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(modlI * self.scattering_angles["weights"][0], axis=1)
            raw_thry = ThryI if want_thry else 0
        else:
            modlI = 0
            raw_thry = 0
            lamAxisI = jnp.zeros(1)
        return lamAxisI, modlI, raw_thry

    def _electron_spectrum_core(self, all_params, want_thry):
        """
        Shared implementation behind electron_spectrum and electron_spectrum_detailed. want_thry controls whether
        the raw (pre-angle-averaged) ThryE is filtered and returned. That filtering step is extra work needed only
        for the detailed/postprocessing output, so electron_spectrum passes False to skip it entirely rather than
        computing and discarding it on every forward pass.
        """
        if self.config["data"]["load_ele_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryE, lamAxisE_orig = self.electron_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryE, lamAxisE_orig = self.electron_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisE_orig *= 1e7
            lamAxisE = jnp.squeeze(lamAxisE_orig)  # TODO hardcoded

            modlE = jnp.mean(ThryE, axis=0)
            if self.config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(self.scattering_angles["weights"], modlE.transpose())
            else:
                modlE = jnp.sum(modlE * self.scattering_angles["weights"][0], axis=1)

            lam = all_params["general"]["lam"]
            if self.config["other"]["iawoff"] and (
                self.config["other"]["lamrangE"][0] < lam < self.config["other"]["lamrangE"][1]
            ):
                # set the ion feature to 0 #should be switched to a range about lam
                lamlocb = jnp.argmin(jnp.abs(lamAxisE - lam - 3.0))
                lamlocr = jnp.argmin(jnp.abs(lamAxisE - lam + 3.0))
                modlE = jnp.concatenate(
                    [modlE[:lamlocb], jnp.zeros(lamlocr - lamlocb), modlE[lamlocr:]]
                )  # TODO hardcoded

            if self.config["other"]["iawfilter"][0]:
                filterb = self.config["other"]["iawfilter"][3] - self.config["other"]["iawfilter"][2] / 2
                filterr = self.config["other"]["iawfilter"][3] + self.config["other"]["iawfilter"][2] / 2

                if self.config["other"]["lamrangE"][0] < filterr and self.config["other"]["lamrangE"][1] > filterb:
                    indices = (filterb < lamAxisE) & (filterr > lamAxisE)
                    modlE = jnp.where(indices, modlE * 10 ** (-self.config["other"]["iawfilter"][1]), modlE)

                    if want_thry:
                        indices = (filterb < lamAxisE_orig) & (filterr > lamAxisE_orig)
                        ThryE = jnp.where(indices, ThryE * 10 ** (-9), ThryE)

            raw_thry = ThryE if want_thry else 0
        else:
            modlE = 0
            raw_thry = 0
            lamAxisE = []
        return lamAxisE, modlE, raw_thry
