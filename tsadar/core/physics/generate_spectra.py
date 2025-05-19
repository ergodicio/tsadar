from typing import Dict

from .form_factor import FormFactor

from jax import numpy as jnp


class FitModel:
    """
    FitModel is a class that wraps the FormFactor class to add finite aperture and finite volume effects for generating Thomson scattering spectra. It manages configuration options, handles multiple scattering angles, and supports both electron and ion features, including gradient effects and angular spectra.
    Args:    
        config (Dict): Configuration dictionary built from the input deck, containing all static and runtime parameters for spectrum generation.
        scattering_angles (Dict): Dictionary containing the scattering angles at which the spectrum will be calculated and the relative weights for each angle.
    Methods:
        __call__(all_params: Dict):
            Calculates Thomson spectra corrected for finite aperture and optionally including plasma gradients, based on the current parameter dictionary.
                all_params (Dict): Dictionary of current values for all active and static parameters.
                modlE: Electron plasma wave spectrum (array or int 0 if not loaded).
                modlI: Ion acoustic wave spectrum (array or int 0 if not loaded).
                lamAxisE: Wavelength axis for electron plasma wave (array or empty list if not loaded).
                lamAxisI: Wavelength axis for ion acoustic wave (array or empty list if not loaded).
        ion_spectrum(all_params: Dict):
            Calculates the ion acoustic wave spectrum, applying finite aperture and angular weighting.
                all_params (Dict): Parameter dictionary.
                lamAxisI: Wavelength axis for ion acoustic wave.
                modlI: Ion acoustic wave spectrum.
        electron_spectrum(all_params: Dict):
            Calculates the electron plasma wave spectrum, applying finite aperture, angular weighting, and optional filtering.
                all_params (Dict): Parameter dictionary.
                lamAxisE: Wavelength axis for electron plasma wave.
                modlE: Electron plasma wave spectrum.
        detailed_spectrum(all_params: Dict):
            Calculates both the total spectrum and all its components for postprocessing.
                all_params (Dict): Parameter dictionary.
                modlE: Electron plasma wave spectrum.
                modlI: Ion acoustic wave spectrum.
                ThryE: Detailed electron spectrum components.
                ThryI: Detailed ion spectrum components.
                lamAxisE: Wavelength axis for electron plasma wave.
                lamAxisI: Wavelength axis for ion acoustic wave.
        ion_spectrum_detailed(all_params: Dict):
            Calculates the detailed ion acoustic wave spectrum and its components.
                all_params (Dict): Parameter dictionary.
                lamAxisI: Wavelength axis for ion acoustic wave.
                modlI: Ion acoustic wave spectrum.
                ThryI: Detailed ion spectrum components.
        electron_spectrum_detailed(all_params: Dict):
            Calculates the detailed electron plasma wave spectrum and its components, with optional filtering.
                all_params (Dict): Parameter dictionary.
                lamAxisE: Wavelength axis for electron plasma wave.
                modlE: Electron plasma wave spectrum.
                ThryE: Detailed electron spectrum components.
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
        va_angle = (
            None
            if config["parameters"]["electron"]["fe"]["dim"] < 2
            else config["parameters"]["general"]["Va"]["angle"]
        )
        self.electron_form_factor = FormFactor(
            config["other"]["lamrangE"],
            npts=config["other"]["npts"],
            lam_shift=config["data"]["ele_lam_shift"],
            scattering_angles=self.scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
        )
        self.ion_form_factor = FormFactor(
            config["other"]["lamrangI"],
            npts=config["other"]["npts"],
            lam_shift=0,
            scattering_angles=scattering_angles,
            num_grad_points=num_grad_points,
            va_ang=va_angle,
            ud_ang=ud_angle,
        )

    def __call__(self, all_params: Dict):
        """
        Produces Thomson spectra corrected for finite aperture and optionally including gradients in the plasma
        conditions based off the current parameter dictionary. Calling this method will automatically choose the
        appropriate version of the formfactor class based off the dimension and distribute the conditions for
        multiple ion species to their respective inputs.


        Args:
            all_params: Parameter dictionary containing the current values for all active and static parameters. Only a
                few permanently static properties from the configuration dictionary will be used, everything else must
                be included in this input.

        Returns:
            modlE: calculated electron plasma wave spectrum as an array with length of npts. If an angular spectrum is
                calculated then it will be 2D. If the EPW is not loaded this is returned as the int 0.
            modlI: calculated ion acoustic wave spectrum as an array with length of npts. If the IAW is not loaded this
                is returned as the int 0.
            lamAxisE: electron plasma wave wavelength axis as an array with length of npts. If the EPW is not loaded
                this is returned as an empty list.
            lamAxisI: ion acoustic wave wavelength axis as an array with length of npts. If the IAW is not loaded
                this is returned as an empty list.
            all_params: The input all_params is returned

        """

        lamAxisI, modlI = self.ion_spectrum(all_params)
        lamAxisE, modlE = self.electron_spectrum(all_params)

        return modlE, modlI, lamAxisE, lamAxisI

    def ion_spectrum(self, all_params):
        """
        Computes the ion spectrum based on the provided parameters and configuration.
        Parameters:
            all_params (dict): Dictionary containing all necessary parameters for spectrum calculation.
        Returns:
            tuple:
                lamAxisI (jnp.ndarray): Wavelength axis for the ion spectrum, rescaled to nanometers.
                modlI (jnp.ndarray or int): Computed ion spectrum model. Returns 0 if loading ion spectrum is disabled.
        Notes:
            - If 'load_ion_spec' is enabled in the configuration, the function computes the ion spectrum using the
              appropriate dimensionality (1D or 2D) as specified in the configuration.
            - The wavelength axis is squeezed to remove extra dimensions and rescaled by 1e7 (hardcoded).
            - The spectrum is averaged and weighted by the scattering angles.
            - If 'load_ion_spec' is disabled, returns zeros for both outputs.
        """
        if self.config["other"]["extraoptions"]["load_ion_spec"]:

            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryI, lamAxisI = self.ion_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * self.scattering_angles["weights"][0], axis=1)
        else:
            modlI = 0
            lamAxisI = jnp.zeros(1)
        return lamAxisI, modlI

    def electron_spectrum(self, all_params):
        """
        Computes the electron spectrum based on the provided parameters and configuration.
        This method also applies optional filters or modifications such as
        suppressing the ion feature or applying an IAW (ion-acoustic wave) filter.
        Parameters:
            all_params (dict): Dictionary containing all relevant parameters for spectrum generation,
                including general and electron-specific settings.
        Returns:
            tuple:
                lamAxisE (jnp.ndarray or list): The wavelength axis for the electron spectrum, rescaled to nanometers.
                modlE (jnp.ndarray or int): The processed electron spectrum model. Returns 0 if spectrum loading is disabled.
        """
        if self.config["other"]["extraoptions"]["load_ele_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryE, lamAxisE = self.electron_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryE, lamAxisE = self.electron_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7  # TODO hardcoded

            ThryE = jnp.mean(ThryE, axis=0)
            if self.config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(self.scattering_angles["weights"], ThryE.transpose())
            else:
                modlE = jnp.sum(ThryE * self.scattering_angles["weights"][0], axis=1)

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
        else:
            modlE = 0
            lamAxisE = []
        return lamAxisE, modlE

    def detailed_spectrum(self, all_params: Dict):
        """
        Calculates detailed spectra for both electron plasma waves (EPW) and ion acoustic waves (IAW), including their
        wavelength axes and theoretical components, for postprocessing analysis.
        This method produces both the total spectrum and all its components for EPWs and IAWs, using the provided parameter
        dictionary. It is intended for postprocessing and requires all relevant parameters to be included in the input.
        Args:
            all_params (Dict): Dictionary containing current values for all active and static parameters. Most configuration
                properties must be included in this input, except for a few permanently static ones.
        Returns:
            modlE (np.ndarray or int): Calculated electron plasma wave spectrum as an array of length npts, or 0 if EPW is not loaded.
            modlI (np.ndarray or int): Calculated ion acoustic wave spectrum as an array of length npts, or 0 if IAW is not loaded.
            ThryE (np.ndarray): Theoretical components of the electron plasma wave spectrum.
            ThryI (np.ndarray): Theoretical components of the ion acoustic wave spectrum.
            lamAxisE (np.ndarray or list): Wavelength axis for the electron plasma wave spectrum, or empty list if EPW is not loaded.
            lamAxisI (np.ndarray or list): Wavelength axis for the ion acoustic wave spectrum, or empty list if IAW is not loaded.
        """

        lamAxisI, modlI, ThryI = self.ion_spectrum_detailed(all_params)
        lamAxisE, modlE, ThryE = self.electron_spectrum_detailed(all_params)

        return modlE, modlI, ThryE, ThryI, lamAxisE, lamAxisI
    def ion_spectrum_detailed(self, all_params):
        """
        Computes the detailed ion spectrum based on the provided parameters and configuration.
        This method calculates the ion spectrum using either 1D or 2D form factors, depending on the configuration.
        If the 'load_ion_spec' option is enabled, it computes the theoretical ion spectrum and corresponding wavelength axis.
        The results are processed by removing extra dimensions, rescaling the wavelength axis to nanometers, and averaging
        over the theoretical spectrum. If the option is disabled, it returns zeros.
        Args:
            all_params (dict): Dictionary containing all necessary parameters for spectrum calculation.
        Returns:
            tuple:
                lamAxisI (jnp.ndarray): Wavelength axis for the ion spectrum (in nanometers).
                modlI (jnp.ndarray or int): Processed ion spectrum model or 0 if not loaded.
                ThryI (jnp.ndarray or int): Theoretical ion spectrum or 0 if not loaded.
        """
        if self.config["other"]["extraoptions"]["load_ion_spec"]:
            if self.config["parameters"]["electron"]["fe"]["dim"] == 1:
                ThryI, lamAxisI = self.ion_form_factor(all_params)
            elif self.config["parameters"]["electron"]["fe"]["dim"] == 2:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(all_params)

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded
            modlI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(modlI * self.scattering_angles["weights"][0], axis=1)
        else:
            modlI = 0
            ThryI = 0
            lamAxisI = jnp.zeros(1)
        return lamAxisI, modlI, ThryI

    def electron_spectrum_detailed(self, all_params):
        """
        Computes the detailed electron spectrum based on the provided parameters and configuration.
        This method generates the electron spectrum using either 1D or 2D electron form factors,
        applies various configuration-based modifications (such as angular weighting, ion feature suppression,
        and filtering), and returns the processed wavelength axis, the modeled electron spectrum, and the
        theoretical electron form factor.
        Parameters:
            all_params (dict): Dictionary containing all relevant parameters for spectrum generation,
                including general and electron-specific settings.
        Returns:
            tuple:
                lamAxisE (array-like): The wavelength axis (in nm) for the electron spectrum.
                modlE (array-like or int): The processed/model electron spectrum. Returns 0 if not loaded.
                ThryE (array-like or int): The theoretical electron form factor. Returns 0 if not loaded.
        Notes:
            - The method behavior is controlled by the configuration dictionary (`self.config`), which determines
              whether to load the electron spectrum, the dimensionality of the electron form factor, and various
              spectrum modifications (e.g., angular weighting, ion feature suppression, filtering).
            - Some operations are hardcoded (e.g., wavelength offsets for ion feature suppression).
            - If the spectrum is not loaded (`load_ele_spec` is False), returns zeros and an empty wavelength axis.
        """
        if self.config["other"]["extraoptions"]["load_ele_spec"]:
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

                    indices = (filterb < lamAxisE_orig) & (filterr > lamAxisE_orig)
                    ThryE = jnp.where(indices, ThryE * 10 ** (-9), ThryE)
        else:
            modlE = 0
            ThryE = 0
            lamAxisE = []
        return lamAxisE, modlE, ThryE