from typing import Dict

import copy

import numpy as np
from jax import numpy as jnp, vmap
from jax.flatten_util import ravel_pytree
from interpax import interp2d

from tsadar.model.physics.generate_spectra import FitModel
from tsadar.model.modules import ThomsonParams
from tsadar.distribution_functions.dist_functional_forms import calc_moment
from tsadar.process import irf


def init_weights_and_bounds(config, num_slices):
    """
    this dict form will be unpacked for scipy consumption, we assemble them all in the same way so that we can then
    use ravel pytree from JAX utilities to unpack it
    Args:
        config:
        init_weights:
        num_slices:

    Returns:

    """
    lb = {"active": {}, "inactive": {}}
    ub = {"active": {}, "inactive": {}}
    iw = {"active": {}, "inactive": {}}

    for species in config["parameters"].keys():
        lb["active"][species] = {}
        ub["active"][species] = {}
        iw["active"][species] = {}
        lb["inactive"][species] = {}
        ub["inactive"][species] = {}
        iw["inactive"][species] = {}

    for species in config["parameters"].keys():
        for param_name, param_dict in config["parameters"][species].items():
            if param_dict["active"]:
                active_or_inactive = "active"
            else:
                active_or_inactive = "inactive"

            if param_name == "fe":
                # if config["parameters"][species]["fe"]["type"].casefold() == "dlm":
                # iw[active_or_inactive][species]["m"] = np.ones((num_slices, 1)) * param_dict["params"]["m"]["val"]
                if config["parameters"][species]["fe"]["type"].casefold() == "arbitrary":
                    iw[active_or_inactive][species]["fe"] = np.repeat(param_dict["val"][None, :], num_slices, axis=0)
                # else:
                # raise NotImplementedError(
                # f"Functional form {config['parameters'][species]['fe']['type']} not implemented"
                # )

            else:
                iw[active_or_inactive][species][param_name] = np.ones((num_slices, 1)) * param_dict["val"]

            if param_dict["active"]:
                # shift
                lb[active_or_inactive][species][param_name] = np.zeros(num_slices)
                ub[active_or_inactive][species][param_name] = np.ones(num_slices)

                # normalize
                if param_name == "fe":
                    if config["parameters"][species]["fe"]["type"].casefold() == "dlm":
                        iw[active_or_inactive][species]["m"] = (
                            iw[active_or_inactive][species]["m"] - config["units"]["shifts"][species]["m"]
                        ) / config["units"]["norms"][species]["m"]

                    else:
                        iw[active_or_inactive][species]["fe"] = (
                            iw[active_or_inactive][species]["fe"]
                            - config["units"]["shifts"][species]["fe"].reshape(
                                jnp.shape(iw[active_or_inactive][species]["fe"])
                            )
                        ) / config["units"]["norms"][species]["fe"].reshape(
                            jnp.shape(iw[active_or_inactive][species]["fe"])
                        )
                else:
                    iw[active_or_inactive][species][param_name] = (
                        iw[active_or_inactive][species][param_name] - config["units"]["shifts"][species][param_name]
                    ) / config["units"]["norms"][species][param_name]

    return lb, ub, iw


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

        self.lb, self.ub, init_weights = init_weights_and_bounds(cfg, num_slices=cfg["optimizer"]["batch_size"])
        self.flattened_weights, self.unravel_pytree = ravel_pytree(init_weights["active"])
        self.static_params = init_weights["inactive"]
        self.pytree_weights = init_weights
        self.bounds = self.construct_bounds()

        self.model = FitModel(cfg, scattering_angles)
        self.lam = cfg["parameters"]["general"]["lam"]["val"]

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

    def construct_bounds(self):
        """
        This method construct a bounds zip from the upper and lower bounds. This allows the iterable to be reconstructed
        after being used in a fit.

        Args:

        Returns:

        """
        flattened_lb, _ = ravel_pytree(self.lb)
        flattened_ub, _ = ravel_pytree(self.ub)
        return zip(flattened_lb, flattened_ub)

    def get_plasma_parameters(self, input_weights: Dict, return_static_params: bool = True) -> Dict:
        """
        This function creates the physical parameters used in the TS algorithm from the weights. The input input_weights
        is mapped to these_params causing the input_weights to also be modified.

        This could be a 1:1 mapping, or it could be a linear transformation e.g. "normalized" parameters, or it could
        be something else altogether e.g. a neural network

        Args:
            input_weights: dictionary of weights used or supplied by the minimizer, these are bounded from 0 to 1
            return_static_params: boolean determining if the static parameters (these not modified during fitting) will
            be inculded in the retuned dictionary. This is nessesary for the physics model which requires values for all
            parameters.

        Returns:
            these_params: dictionary of the paramters in physics units

        """
        Te_mult = 1.0
        ne_mult = 1.0
        these_params = copy.deepcopy(input_weights)
        for species in self.cfg["parameters"].keys():
            for param_name, param_config in self.cfg["parameters"][species].items():
                if param_config["active"]:
                    if param_name != "fe":
                        these_params[species][param_name] = (
                            these_params[species][param_name] * self.cfg["units"]["norms"][species][param_name]
                            + self.cfg["units"]["shifts"][species][param_name]
                        )
                    else:
                        fe_shape = jnp.shape(these_params[species][param_name])
                        # convert EDF from 01 bounded log units to unbounded log units
                        # jax.debug.print("these params {a}", a=these_params[species][param_name])

                        fe_cur = jnp.exp(
                            these_params[species][param_name]
                            * self.cfg["units"]["norms"][species][param_name].reshape(fe_shape)
                            + self.cfg["units"]["shifts"][species][param_name].reshape(fe_shape)
                        )
                        # commented out the renormalization to see effect on 2D edfs 9/26/24
                        # jax.debug.print("fe_cur {a}", a=fe_cur)
                        # this only works for 2D edfs and will have to be genralized to 1D
                        # recaclulate the moments of the EDF
                        renorm = jnp.sqrt(
                            calc_moment(jnp.squeeze(fe_cur), self.cfg["parameters"]["electron"]["fe"]["velocity"], 2)
                            / (
                                2
                                * calc_moment(
                                    jnp.squeeze(fe_cur), self.cfg["parameters"]["electron"]["fe"]["velocity"], 0
                                )
                            )
                        )
                        Te_mult = renorm**2

                        vx2 = self.cfg["parameters"]["electron"]["fe"]["velocity"][0][0] / renorm
                        vy2 = self.cfg["parameters"]["electron"]["fe"]["velocity"][0][0] / renorm

                        fe_cur = jnp.exp(
                            interp2d(
                                self.cfg["parameters"]["electron"]["fe"]["velocity"][0].flatten(),
                                self.cfg["parameters"]["electron"]["fe"]["velocity"][1].flatten(),
                                vx2,
                                vy2,
                                jnp.log(jnp.squeeze(fe_cur)),
                                extrap=[-100, -100],
                                method="linear",
                            ).reshape(jnp.shape(self.cfg["parameters"]["electron"]["fe"]["velocity"][0]), order="F")
                        )
                        ne_mult = calc_moment(
                            jnp.squeeze(fe_cur), self.cfg["parameters"]["electron"]["fe"]["velocity"], 0
                        )
                        fe_cur = fe_cur / ne_mult
                        these_params[species][param_name] = jnp.log(fe_cur)

                        if self.cfg["parameters"][species]["fe"]["dim"] == 1:
                            these_params[species]["fe"] = jnp.log(
                                self.smooth(jnp.exp(these_params[species]["fe"][0]))[None, :]
                            )
                        elif self.cfg["dist_fit"]["smooth"]:
                            these_params[species]["fe"] = self.smooth2D(these_params[species]["fe"])

                else:
                    if return_static_params:
                        if param_name == "fe":
                            if param_config["type"].casefold() == "arbitrary":
                                these_params[species][param_name] = self.static_params[species][param_name]
                        else:
                            these_params[species][param_name] = self.static_params[species][param_name]

        # need to confirm this works due to jax imutability
        # jax.debug.print("Temult {total_loss}", total_loss=Te_mult)
        # jax.debug.print("nemult {total_loss}", total_loss=ne_mult)
        # jax.debug.print("Tebefore {total_loss}", total_loss=these_params["electron"]['Te'])
        these_params["electron"]["Te"] *= Te_mult
        these_params["electron"]["ne"] *= ne_mult
        # jax.debug.print("Teafter {total_loss}", total_loss=these_params["electron"]['Te'])
        # jax.debug.print("fe after has NANs {total_loss}", total_loss=jnp.isnan(fe_cur))

        return these_params

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
        ThryE = jnp.where(lamAxisE < self.lam, TSins["general"]["amp1"] * ThryE, TSins["general"]["amp2"] * ThryE)
        return ThryE, lamAxisE

    def __call__(self, params, batch):
        """
        TODO

        Args:
            params: Dict- contains name value pairs for all the parameters from the input deck
            batch: Dict- contains the electron and ion data arrays as well as their amplitude arrays and noise arrays.

        Returns:

        """
        modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = self.model(params)
        ThryE, ThryI, lamAxisE, lamAxisI = self.postprocess_theory(
            modlE, modlI, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, live_TSinputs
        )
        if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
            ThryE, lamAxisE = self.reduce_ATS_to_resunit(ThryE, lamAxisE, live_TSinputs, batch)

        ThryE = ThryE + batch["noise_e"]
        ThryI = ThryI + batch["noise_i"]

        return ThryE, ThryI, lamAxisE, lamAxisI


class ThomsonScatteringDiagnostic2:
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
        # self.lam = cfg["parameters"]["general"]["lam"]["val"]

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
