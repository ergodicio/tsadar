from typing import Dict

from .form_factor import FormFactor

from jax import numpy as jnp


class FitModel:
    """
    The FitModel Class wraps the FormFactor class adding finite aperture effects and finite volume effects. This class
    also handles the options for calculating the form factor.

    Args:
        config: Dict- configuration dictionary built from input deck
        sa: Dict- has fields containing the scattering angles the spectrum will be calculated at and the relative
        weights of each of the scattering angles in the final spectrum
    """

    def __init__(self, config: Dict, scattering_angles: Dict):
        """
        FitModel class constructor, sets the static properties associated with spectrum generation that will not be
        modified from one iteration of the fitter to the next.

        Args:
            config: Dict- configuration dictionary built from input deck
            sa: Dict- has fields containing the scattering angles the spectrum will be calculated at and the relative
                weights of each of the scattering angles in the final spectrum
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
            else config["parameters"]["general"]["va"]["angle"]
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
        if self.config["other"]["extraoptions"]["load_ion_spec"]:
            if self.num_dist_func.dim == 1:
                ThryI, lamAxisI = self.ion_form_factor(all_params)

            else:
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
