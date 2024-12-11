import copy
from typing import Dict

import jax
from jax import numpy as jnp


from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import numpy as np

from tsadar.model.thomson_diagnostic import ThomsonScatteringDiagnostic
from tsadar.distribution_functions.dist_functional_forms import trapz
from tsadar.misc.vector_tools import rotate


class LossFunction:
    """
    This class is responsible for handling the forward pass and using that to create a loss function

    """

    def __init__(self, cfg: Dict, scattering_angles, dummy_batch):
        """

        Args:
            cfg: Configuration dictionary constructed from the inputs
            scattering_angles: Dictionary containing the scattering angles and thier relative weights
            dummy_batch: Dictionary of dummy data
        """
        self.cfg = cfg

        if cfg["optimizer"]["y_norm"]:
            self.i_norm = np.amax(dummy_batch["i_data"])
            self.e_norm = np.amax(dummy_batch["e_data"])
        else:
            self.i_norm = self.e_norm = 1.0

        if cfg["optimizer"]["x_norm"] and cfg["nn"]["use"]:
            self.i_input_norm = np.amax(dummy_batch["i_data"])
            self.e_input_norm = np.amax(dummy_batch["e_data"])
        else:
            self.i_input_norm = self.e_input_norm = 1.0

        # boolean used to determine if the analyis is performed twice with rotation of the EDF
        self.multiplex_ang = isinstance(cfg["data"]["shotnum"], list)

        ############

        self.ts_diag = ThomsonScatteringDiagnostic(cfg, scattering_angles=scattering_angles)

        self._loss_ = jit(self.__loss__)
        self._vg_func_ = jit(value_and_grad(self.__loss__, argnums=0, has_aux=True))
        ##this will be replaced with jacobian params jacobian inverse
        self._h_func_ = jit(jax.hessian(self._loss_for_hess_fn_, argnums=0))
        self.array_loss = jit(self.calc_loss)

        # this needs to be rethought and does not work in all cases
        if cfg["parameters"]["electron"]["fe"]["active"]:
            if "dist_fit" in cfg:
                if cfg["parameters"]["electron"]["fe"]["dim"] == 1:
                    self.smooth_window_len = round(
                        cfg["parameters"]["electron"]["fe"]["velocity"].size * cfg["dist_fit"]["window"]["len"]
                    )
                    self.smooth_window_len = self.smooth_window_len if self.smooth_window_len > 1 else 2

                    if cfg["dist_fit"]["window"]["type"] == "hamming":
                        self.w = jnp.hamming(self.smooth_window_len)
                    elif cfg["dist_fit"]["window"]["type"] == "hann":
                        self.w = jnp.hanning(self.smooth_window_len)
                    elif cfg["dist_fit"]["window"]["type"] == "bartlett":
                        self.w = jnp.bartlett(self.smooth_window_len)
                    else:
                        raise NotImplementedError
                else:
                    Warning("Smoothing not enabled for 2D distributions")
            else:
                Warning(
                    "\n !!! Distribution function not fitted !!! Make sure this is what you thought you were running \n"
                )

    def smooth(self, distribution: jnp.ndarray) -> jnp.ndarray:
        """
        This method is used to smooth the distribution function. It sits right in between the optimization algorithm
        that provides the weights/values of the distribution function and the fitting code that uses it.

        Because the optimizer is not constrained to provide a smooth distribution function, this operation smoothens
        the output. This is a differentiable operation and we train/fit our weights through this.

        Args:
            distribution:

        Returns:

        """
        s = jnp.r_[
            distribution[self.smooth_window_len - 1 : 0 : -1],
            distribution,
            distribution[-2 : -self.smooth_window_len - 1 : -1],
        ]
        return jnp.convolve(self.w / self.w.sum(), s, mode="same")[
            self.smooth_window_len - 1 : -(self.smooth_window_len - 1)
        ]

    def smooth2D(self, distribution: jnp.ndarray) -> jnp.ndarray:
        """
        This method is used to smooth the distribution function. It sits right in between the optimization algorithm
        that provides the weights/values of the distribution function and the fitting code that uses it.

        Because the optimizer is not constrained to provide a smooth distribution function, this operation smoothens
        the output. This is a differentiable operation and we train/fit our weights through this.

        Args:
            distribution:

        Returns:

        """

        smoothing_kernel = jnp.outer(jnp.bartlett(5), jnp.bartlett(5))
        smoothing_kernel = smoothing_kernel / jnp.sum(smoothing_kernel)

        return jax.scipy.signal.convolve2d(distribution, smoothing_kernel, "same")

    def _get_normed_batch_(self, batch: Dict):
        """
        Normalizes the batch

        Args:
            batch:

        Returns:

        """
        normed_batch = copy.deepcopy(batch)
        normed_batch["i_data"] = normed_batch["i_data"] / self.i_input_norm
        normed_batch["e_data"] = normed_batch["e_data"] / self.e_input_norm
        return normed_batch

    def vg_loss(self, weights: Dict, batch: Dict):
        """
        This is the primary workhorse high level function. This function returns the value of the loss function which
        is used to assess goodness-of-fit and the gradient of that value with respect to the weights, which is used to
        update the weights

        This function is used by both optimization methods. It performs the necessary pre-/post- processing that is
        needed to work with the optimization software.

        Args:
            weights:
            batch:

        Returns:

        """
        if self.cfg["optimizer"]["method"] == "l-bfgs-b":
            pytree_weights = self.ts_diag.unravel_pytree(weights)
            (value, aux), grad = self._vg_func_(pytree_weights, batch)

            if "fe" in grad:
                grad["fe"] = self.cfg["optimizer"]["grad_scalar"] * grad["fe"]

            for species in self.cfg["parameters"].keys():
                for k, param_dict in self.cfg["parameters"][species].items():
                    if param_dict["active"]:
                        scalar = param_dict["gradient_scalar"] if "gradient_scalar" in param_dict else 1.0
                        grad[species][k] *= scalar

            temp_grad, _ = ravel_pytree(grad)
            flattened_grads = np.array(temp_grad)
            return value, flattened_grads
        else:
            return self._vg_func_(weights, batch)

    def h_loss_wrt_params(self, weights, batch):
        return self._h_func_(weights, batch)

    def _loss_for_hess_fn_(self, weights, batch):
        # params = params | self.static_params
        params = self.ts_diag.get_plasma_parameters(weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(params, batch)
        i_error, e_error, _, _ = self.calc_ei_error(
            batch,
            ThryI,
            lamAxisI,
            ThryE,
            lamAxisE,
            uncert=[jnp.abs(batch["i_data"]) + 1e-10, jnp.abs(batch["e_data"]) + 1e-10],
            reduce_func=jnp.sum,
        )

        return i_error + e_error

    def calc_ei_error(self, batch, ThryI, lamAxisI, ThryE, lamAxisE, uncert, reduce_func=jnp.mean):
        """
        This function calculates the error in the fit of the IAW and EPW

        Args:
            batch: dictionary containing the data
            ThryI: ion theoretical spectrum
            lamAxisI: ion wavelength axis
            ThryE: electron theoretical spectrum
            lamAxisE: electron wavelength axis
            uncert: uncertainty values
            reduce_func: method to combine all lineouts into a single metric

        Returns:

        """
        i_error = 0.0
        e_error = 0.0
        used_points = 0
        i_data = batch["i_data"]
        e_data = batch["e_data"]
        sqdev = {"ele": jnp.zeros(e_data.shape), "ion": jnp.zeros(i_data.shape)}

        if self.cfg["other"]["extraoptions"]["fit_IAW"]:
            _error_ = self.loss_functionals(i_data, ThryI, uncert[0], method=self.cfg["optimizer"]["loss_method"])
            _error_ = jnp.where(
                (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_min"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_cf_min"])
                )
                | (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_cf_max"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_max"])
                ),
                _error_,
                0.0,
            )

            used_points += jnp.sum(
                (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_min"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_cf_min"])
                )
                | (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_cf_max"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_max"])
                )
            )
            # this was temp code to help with 2 species fits
            # _error_ = jnp.where(
            #     (lamAxisI > 526.25) & (lamAxisI < 526.75),
            #     10.0 * _error_,
            #     _error_,
            # )
            sqdev["ion"] = _error_
            i_error += reduce_func(_error_)

        if self.cfg["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = self.loss_functionals(e_data, ThryE, uncert[1], method=self.cfg["optimizer"]["loss_method"])
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"]),
                _error_,
                0.0,
            )
            used_points += jnp.sum(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"])
            )
            e_error += reduce_func(_error_)
            sqdev["ele"] += _error_

        if self.cfg["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = self.loss_functionals(e_data, ThryE, uncert[1], method=self.cfg["optimizer"]["loss_method"])
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"]),
                _error_,
                0.0,
            )
            used_points += jnp.sum(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"])
            )

            e_error += reduce_func(_error_)
            sqdev["ele"] += _error_

        return i_error, e_error, sqdev, used_points

    def calc_loss(self, weights, batch: Dict):
        """
        This function calculates the value of the loss function

        Args:
            params:
            batch:

        Returns:

        """
        params = self.ts_diag.get_plasma_parameters(weights)

        if self.multiplex_ang:
            ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(params, batch["b1"])
            # jax.debug.print("fe size {e_error}", e_error=jnp.shape(params["electron"]['fe']))
            params["electron"]["fe"] = rotate(
                jnp.squeeze(params["electron"]["fe"]), self.cfg["data"]["shot_rot"] * jnp.pi / 180.0
            )

            ThryE_rot, _, _, _ = self.ts_diag(params, batch["b2"])
            i_error1, e_error1, sqdev, used_points = self.calc_ei_error(
                batch["b1"],
                ThryI,
                lamAxisI,
                ThryE,
                lamAxisE,
                denom=[jnp.square(self.i_norm), jnp.square(self.e_norm)],
                reduce_func=jnp.mean,
            )
            i_error2, e_error2, sqdev, used_points = self.calc_ei_error(
                batch["b2"],
                ThryI,
                lamAxisI,
                ThryE_rot,
                lamAxisE,
                denom=[jnp.square(self.i_norm), jnp.square(self.e_norm)],
                reduce_func=jnp.mean,
            )
            i_error = i_error1 + i_error2
            e_error = e_error1 + e_error2

            normed_batch = self._get_normed_batch_(batch["b1"])
        else:
            ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(params, batch)

            i_error, e_error, sqdev, used_points = self.calc_ei_error(
                batch,
                ThryI,
                lamAxisI,
                ThryE,
                lamAxisE,
                uncert=[jnp.square(self.i_norm), jnp.square(self.e_norm)],
                reduce_func=jnp.mean,
            )

            normed_batch = self._get_normed_batch_(batch)

        normed_e_data = normed_batch["e_data"]
        ion_error = self.cfg["data"]["ion_loss_scale"] * i_error

        penalty_error = self.penalties(weights)
        total_loss = ion_error + e_error + penalty_error
        # jax.debug.print("e_error {total_loss}", total_loss=e_error)

        return total_loss, sqdev, used_points, ThryE, ThryI, params
        # return total_loss, [ThryE, params]

    def loss(self, weights, batch: Dict):
        """
        High level function that returns the value of the loss function

        Args:
            weights:
            batch: Dict

        Returns:

        """
        if self.cfg["optimizer"]["method"] == "l-bfgs-b":
            pytree_weights = self.unravel_pytree(weights)
            value, _ = self._loss_(pytree_weights, batch)
            return value
        else:
            return self._loss_(weights, batch)

    def __loss__(self, weights, batch: Dict):
        """
        Output wrapper
        """

        total_loss, sqdev, used_points, ThryE, normed_e_data, params = self.calc_loss(weights, batch)
        return total_loss, [ThryE, params]

    def loss_functionals(self, d, t, uncert, method="l2"):
        """
        This function calculates the error loss metric between d and t for different metrics sepcified by method,
        with the default being the l2 norm

        Args:
            d: data array
            t: theory array
            uncert: uncertainty values
            method: name of the loss metric method, l1, l2, poisson, log-cosh. Currently only l1 and l2 include the uncertainties

        Returns:
            loss: value of the loss metric per slice

        """
        if method == "l1":
            _error_ = jnp.abs(d - t) / uncert
        elif method == "l2":
            _error_ = jnp.square(d - t) / uncert
        elif method == "log-cosh":
            _error_ = jnp.log(jnp.cosh(d - t))
        elif method == "poisson":
            _error_ = t - d * jnp.log(t)
        return _error_

    def penalties(self, weights):
        """
        This function calculates additional penatlities to be added to the loss function

        Args:
            params: parameter weights as supplied to the loss function
            batch:

        Returns:

        """
        param_penalty = 0.0
        # this will need to be modified for the params instead of weights
        for species in weights.keys():
            for k in weights[species].keys():
                if k != "fe":
                    # jax.debug.print("fe size {e_error}", e_error=weights[species][k])
                    param_penalty += jnp.maximum(0.0, jnp.log(jnp.abs(weights[species][k] - 0.5) + 0.5))
        if self.cfg["optimizer"]["moment_loss"]:
            density_loss, temperature_loss, momentum_loss = self._moment_loss_(weights)
            param_penalty = param_penalty + density_loss + temperature_loss + momentum_loss
        else:
            density_loss = 0.0
            temperature_loss = 0.0
            momentum_loss = 0.0
        if self.cfg["parameters"]["electron"]["fe"]["fe_decrease_strict"]:
            gradfe = jnp.sign(self.cfg["velocity"][1:]) * jnp.diff(params["fe"].squeeze())
            vals = jnp.where(gradfe > 0.0, gradfe, 0.0).sum()
            fe_penalty = jnp.tan(jnp.amin(jnp.array([vals, jnp.pi / 2])))
        else:
            fe_penalty = 0.0
        # jax.debug.print("e_err {e_error}", e_error=e_error)
        # jax.debug.print("{density_loss}", density_loss=density_loss)
        # jax.debug.print("{temperature_loss}", temperature_loss=temperature_loss)
        # jax.debug.print("{momentum_loss}", momentum_loss=momentum_loss)
        # jax.debug.print("tot loss {total_loss}", total_loss=total_loss)
        # jax.debug.print("param_penalty {total_loss}", total_loss=jnp.sum(param_penalty))

        return jnp.sum(param_penalty) + fe_penalty + density_loss + temperature_loss + momentum_loss

    def _moment_loss_(self, params):
        """
        This function calculates the loss associated with regularizing the moments of the distribution function i.e.
        the density should be 1, the temperature should be 1, and momentum should be 0.

        Args:
            params:

        Returns:

        """
        if self.cfg["parameters"]["electron"]["fe"]["dim"] == 1:
            dv = (
                self.cfg["parameters"]["electron"]["fe"]["velocity"][1]
                - self.cfg["parameters"]["electron"]["fe"]["velocity"][0]
            )
            if self.cfg["parameters"]["electron"]["fe"]["symmetric"]:
                density_loss = jnp.mean(jnp.square(1.0 - 2.0 * jnp.sum(jnp.exp(params["electron"]["fe"]) * dv, axis=1)))
                temperature_loss = jnp.mean(
                    jnp.square(
                        1.0
                        - 2.0
                        * jnp.sum(
                            jnp.exp(params["electron"]["fe"])
                            * self.cfg["parameters"]["electron"]["fe"]["velocity"] ** 2.0
                            * dv,
                            axis=1,
                        )
                    )
                )
            else:
                density_loss = jnp.mean(jnp.square(1.0 - jnp.sum(jnp.exp(params["electron"]["fe"]) * dv, axis=1)))
                temperature_loss = jnp.mean(
                    jnp.square(
                        1.0
                        - jnp.sum(
                            jnp.exp(params["electron"]["fe"])
                            * self.cfg["parameters"]["electron"]["fe"]["velocity"] ** 2.0
                            * dv,
                            axis=1,
                        )
                    )
                )
            momentum_loss = jnp.mean(
                jnp.square(
                    jnp.sum(
                        jnp.exp(params["electron"]["fe"]) * self.cfg["parameters"]["electron"]["fe"]["velocity"] * dv,
                        axis=1,
                    )
                )
            )
        else:
            fedens = trapz(
                trapz(jnp.exp(params["electron"]["fe"]), self.cfg["parameters"]["electron"]["fe"]["v_res"]),
                self.cfg["parameters"]["electron"]["fe"]["v_res"],
            )
            jax.debug.print("zero moment = {fedens}", fedens=fedens)
            density_loss = jnp.mean(jnp.square(1.0 - fedens))

            # density_loss = jnp.mean(
            #     jnp.square(
            #         1.0
            #         - trapz(
            #             trapz(
            #                 jnp.exp(params["electron"]["fe"]), self.cfg["parameters"]["electron"]["fe"]["v_res"]
            #             ),
            #             self.cfg["parameters"]["electron"]["fe"]["v_res"],
            #         )
            #     )
            # )
            second_moment = trapz(
                trapz(
                    jnp.exp(params["electron"]["fe"])
                    * (
                        self.cfg["parameters"]["electron"]["fe"]["velocity"][0] ** 2
                        + self.cfg["parameters"]["electron"]["fe"]["velocity"][1] ** 2
                    ),
                    self.cfg["parameters"]["electron"]["fe"]["v_res"],
                ),
                self.cfg["parameters"]["electron"]["fe"]["v_res"],
            )
            jax.debug.print("second moment = {fedens}", fedens=second_moment)
            temperature_loss = jnp.mean(jnp.square(1.0 - second_moment / 2))
            # needs to be fixed
            first_moment = second_moment = trapz(
                trapz(
                    jnp.exp(params["electron"]["fe"])
                    * (
                        self.cfg["parameters"]["electron"]["fe"]["velocity"][0] ** 2
                        + self.cfg["parameters"]["electron"]["fe"]["velocity"][1] ** 2
                    )
                    ** (1 / 2),
                    self.cfg["parameters"]["electron"]["fe"]["v_res"],
                ),
                self.cfg["parameters"]["electron"]["fe"]["v_res"],
            )
            jax.debug.print("first moment = {fedens}", fedens=first_moment)
            # momentum_loss = jnp.mean(jnp.square(jnp.sum(jnp.exp(params["fe"]) * self.cfg["velocity"] * dv, axis=1)))
            momentum_loss = 0.0
            # print(temperature_loss)
        return density_loss, temperature_loss, momentum_loss
