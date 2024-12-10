from typing import Dict

from tsadar.model.physics.form_factor import FormFactor
from tsadar.distribution_functions.gen_num_dist_func import DistFunc

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

    def __init__(self, config: Dict, sa):
        """
        FitModel class constructor, sets the static properties associated with spectrum generation that will not be
        modified from one iteration of the fitter to the next.

        Args:
            config: Dict- configuration dictionary built from input deck
            sa: Dict- has fields containing the scattering angles the spectrum will be calculated at and the relative
                weights of each of the scattering angles in the final spectrum
        """
        self.config = config
        self.sa = sa
        # this will need to be fixed for multi electron
        self.num_ions = 0
        self.num_electrons = 0
        for species in config["parameters"].keys():
            if "electron" == species:
                self.num_dist_func = DistFunc(config["parameters"][species])
                self.num_electrons += 1
            elif "ion" in species:
                self.num_ions += 1
            elif "general" in species:
                pass
            else:
                raise ValueError(f"Unknown species in input: {species}")

        # print(f"{config['other']['npts']=}")
        self.electron_form_factor = FormFactor(
            config["other"]["lamrangE"],
            npts=config["other"]["npts"],
            fe_dim=self.num_dist_func.dim,
            vax=self.num_dist_func.v,
        )
        self.ion_form_factor = FormFactor(
            config["other"]["lamrangI"],
            npts=config["other"]["npts"],
            fe_dim=self.num_dist_func.dim,
            vax=self.num_dist_func.v,
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

        # Add gradients to electron temperature and density just being applied to EPW
        cur_Te, cur_ne, A, Z, Ti, fract = self.populate_plasma_properties(all_params)
        lam = all_params["general"]["lam"]
        fecur, vcur = self.calculate_distribution(all_params, cur_Te, Z, fract)
        lamAxisI, modlI = self.ion_spectrum(all_params, cur_Te, cur_ne, A, Z, Ti, fract, lam, fecur, vcur)
        lamAxisE, modlE = self.electron_spectrum(all_params, cur_Te, cur_ne, A, Z, Ti, fract, lam, fecur, vcur)

        return modlE, modlI, lamAxisE, lamAxisI, all_params

    def calculate_distribution(self, all_params, cur_Te, Z, fract):
        if self.num_dist_func.fe_name.casefold() == "dlm":
            if self.config["parameters"]["electron"]["m"]["matte"]:
                # Intensity should be given in effective 3omega intensity e.i. I*lamda^2/lamda_3w^2 and in units of 10^14 W/cm^2
                alpha = (
                    0.042
                    * self.config["parameters"]["electron"]["m"]["intens"]
                    / 9.0
                    * jnp.sum(Z**2 * fract)
                    / (jnp.sum(Z * fract) * cur_Te)
                )
                mcur = 2.0 + 3.0 / (1.0 + 1.66 / (alpha**0.724))
            else:
                mcur = all_params["electron"]["m"]

            # if self.config["parameters"]["electron"]["m"]["active"]:
            vcur, fecur = self.num_dist_func(mcur)
            # all_params["electron"]["fe"] = jnp.log(all_params["electron"]["fe"])

            # (
            #     self.config["parameters"]["electron"]["fe"]["velocity"],
            #     all_params["electron"]["fe"],
            # ) = self.num_dist_func(mcur.squeeze())
            # all_params["electron"]["fe"] = jnp.log(all_params["electron"]["fe"])
        elif self.num_dist_func.fe_name.casefold() == "sphericalharmonic":
            vcur, fecur = self.num_dist_func(all_params["electron"])

        else:
            raise NotImplementedError(f"Functional form {self.num_dist_func.fe_name.casefold()} not implemented")

        if self.config["parameters"]["electron"]["fe"]["symmetric"]:
            fecur = jnp.concatenate((jnp.flip(fecur[1:]), fecur))
            vcur = jnp.concatenate((-jnp.flip(vcur[1:]), vcur))
        return fecur, vcur

    def populate_plasma_properties(self, all_params):
        cur_Te = jnp.zeros((self.config["parameters"]["general"]["Te_gradient"]["num_grad_points"], self.num_electrons))
        cur_ne = jnp.zeros((self.config["parameters"]["general"]["ne_gradient"]["num_grad_points"], self.num_electrons))
        A = jnp.zeros(self.num_ions)
        Z = jnp.zeros(self.num_ions)
        Ti = jnp.zeros(self.num_ions)
        fract = jnp.zeros(self.num_ions)

        ion_c = 0
        ele_c = 0
        for species in self.config["parameters"].keys():
            if "electron" == species:
                cur_Te = cur_Te.at[:, ele_c].set(
                    jnp.linspace(
                        (1 - all_params["general"]["Te_gradient"] / 200) * all_params[species]["Te"],
                        (1 + all_params["general"]["Te_gradient"] / 200) * all_params[species]["Te"],
                        self.config["parameters"]["general"]["Te_gradient"]["num_grad_points"],
                    ).squeeze()
                )

                cur_ne = cur_ne.at[:, ele_c].set(
                    (
                        jnp.linspace(
                            (1 - all_params["general"]["ne_gradient"] / 200) * all_params[species]["ne"],
                            (1 + all_params["general"]["ne_gradient"] / 200) * all_params[species]["ne"],
                            self.config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
                        )
                        * 1e20
                    ).squeeze()
                )
                ele_c += 1

            elif "ion" in species:
                A = A.at[ion_c].set(all_params[species]["A"].squeeze())
                Z = Z.at[ion_c].set(all_params[species]["Z"].squeeze())
                if self.config["parameters"][species]["Ti"]["same"]:
                    Ti = Ti.at[ion_c].set(Ti[ion_c - 1])
                else:
                    Ti = Ti.at[ion_c].set(all_params[species]["Ti"].squeeze())
                fract = fract.at[ion_c].set(all_params[species]["fract"].squeeze())
                ion_c += 1
        return cur_Te, cur_ne, A, Z, Ti, fract

    def ion_spectrum(self, all_params, cur_Te, cur_ne, A, Z, Ti, fract, lam, fecur, vcur):
        if self.config["other"]["extraoptions"]["load_ion_spec"]:
            if self.num_dist_func.dim == 1:
                ThryI, lamAxisI = self.ion_form_factor(
                    all_params, cur_ne, cur_Te, A, Z, Ti, fract, self.sa["sa"], (fecur, vcur), lam
                )
            else:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(
                    all_params,
                    self.config["parameters"]["general"]["ud"]["angle"],
                    self.config["parameters"]["general"]["ud"]["angle"],
                    cur_ne,
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
                    self.sa["sa"],
                    (fecur, vcur),
                    lam,
                )

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * self.sa["weights"][0], axis=1)
        else:
            modlI = 0
            lamAxisI = []
        return lamAxisI, modlI

    def electron_spectrum(self, all_params, cur_Te, cur_ne, A, Z, Ti, fract, lam, fecur, vcur):
        if self.config["other"]["extraoptions"]["load_ele_spec"]:
            if self.num_dist_func.dim == 1:
                ThryE, lamAxisE = self.electron_form_factor(
                    all_params,
                    cur_ne,
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
                    self.sa["sa"],
                    (fecur, vcur),
                    lam + self.config["data"]["ele_lam_shift"],
                )
            else:
                ThryE, lamAxisE = self.electron_form_factor.calc_in_2D(
                    all_params,
                    self.config["parameters"]["general"]["ud"]["angle"],
                    self.config["parameters"]["general"]["ud"]["angle"],
                    cur_ne,
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
                    self.sa["sa"],
                    (fecur, vcur),
                    lam + self.config["data"]["ele_lam_shift"],
                )

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7  # TODO hardcoded

            ThryE = jnp.mean(ThryE, axis=0)
            if self.config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(self.sa["weights"], ThryE.transpose())
            else:
                modlE = jnp.sum(ThryE * self.sa["weights"][0], axis=1)

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
