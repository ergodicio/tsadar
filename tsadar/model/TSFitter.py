import copy
from typing import Dict

import jax
from jax import numpy as jnp


from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import numpy as np

from tsadar.model.spectrum import SpectrumCalculator
from tsadar.distribution_functions.dist_functional_forms import trapz


class TSFitter:
    """
    This class is responsible for handling the forward pass and using that to create a loss function

    Args:
            cfg: Configuration dictionary
            sas: TODO
            dummy_batch: Dictionary of dummy data

    """

    def __init__(self, cfg: Dict, sas, dummy_batch):
        """

        Args:
            cfg: Configuration dictionary
            sas: TODO
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

        # this will need to be fixed for multi electron
        for species in self.cfg["parameters"].keys():
            if "electron" in self.cfg["parameters"][species]["type"].keys():
                self.e_species = species

        self.spec_calc = SpectrumCalculator(cfg, sas, dummy_batch)

        self._loss_ = jit(self.__loss__)
        self._vg_func_ = jit(value_and_grad(self.__loss__, argnums=0, has_aux=True))
        self._h_func_ = jit(jax.hessian(self._loss_for_hess_fn_, argnums=0))
        self.array_loss = jit(self._array_loss_fn_)

        ############

        if cfg["optimizer"]["method"] == "adam":
            lb, ub, init_weights = init_weights_and_bounds(cfg, num_slices=cfg["optimizer"]["batch_size"])
            self.static_params = init_weights["inactive"]
            self.pytree_weights = init_weights

        else:
            lb, ub, init_weights = init_weights_and_bounds(cfg, num_slices=cfg["optimizer"]["batch_size"])
            self.flattened_weights, self.unravel_pytree = ravel_pytree(init_weights["active"])
            self.static_params = init_weights["inactive"]
            self.pytree_weights = init_weights
            self.lb = lb
            self.ub = ub
            self.construct_bounds()

        # this needs to be rethought and does not work in all cases
        if cfg["parameters"][self.e_species]["fe"]["active"]:
            if "dist_fit" in cfg:
                if cfg["parameters"]["fe"]["dim"] == 1:
                    self.smooth_window_len = round(
                        cfg["parameters"][self.e_species]["fe"]["velocity"].size * cfg["dist_fit"]["window"]["len"]
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

    def construct_bounds(self):
        """
        This method construct a bounds zip from the upper and lower bounds. This allows the iterable to be reconstructed
        after being used in a fit.

        Args:

        Returns:

        """
        flattened_lb, _ = ravel_pytree(self.lb)
        flattened_ub, _ = ravel_pytree(self.ub)
        self.bounds = zip(flattened_lb, flattened_ub)

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

    def weights_to_params(self, input_weights: Dict, return_static_params: bool = True) -> Dict:
        """
        This function creates the physical parameters used in the TS algorithm from the weights. The input these_params
        is directly modified.

        This could be a 1:1 mapping, or it could be a linear transformation e.g. "normalized" parameters, or it could
        be something else altogether e.g. a neural network

        Args:
            these_params:
            return_static_params:

        Returns:

        """
        these_params = copy.deepcopy(input_weights)
        for species in self.cfg["parameters"].keys():
            for param_name, param_config in self.cfg["parameters"][species].items():
                if param_name == "type":
                    continue
                if param_config["active"]:
                    # if self.cfg["optimizer"]["method"] == "adam":
                    #     these_params[param_name] = 0.5 + 0.5 * jnp.tanh(these_params[param_name])

                    these_params[species][param_name] = (
                        these_params[species][param_name] * self.cfg["units"]["norms"][species][param_name]
                        + self.cfg["units"]["shifts"][species][param_name]
                    )
                    if param_name == "fe":
                        these_params[species]["fe"] = jnp.log(
                            self.smooth(jnp.exp(these_params[species]["fe"][0]))[None, :]
                        )
                        # these_params["fe"] = jnp.log(self.smooth(jnp.exp(these_params["fe"])))

                else:
                    if return_static_params:
                        these_params[species][param_name] = self.static_params[species][param_name]

        return these_params

    def _array_loss_fn_(self, weights: Dict, batch: Dict):
        """
        This is similar to the _loss_ function but it calculates the loss per slice without doing a batch-wise mean
        calculation at the end. There might be a better way to write this


        Args:
            weights:
            batch:

        Returns:

        """
        # Used for postprocessing
        params = self.weights_to_params(weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.spec_calc(params, batch)
        used_points = 0
        loss = 0

        i_error, e_error = self.calc_ei_error(
            batch,
            ThryI,
            lamAxisI,
            ThryE,
            lamAxisE,
            denom=[jnp.square(self.i_norm), jnp.square(self.e_norm)],
            reduce_func=jnp.sum,
        )

        i_data = batch["i_data"]
        e_data = batch["e_data"]
        sqdev = {"ele": jnp.zeros(e_data.shape), "ion": jnp.zeros(i_data.shape)}

        if self.cfg["other"]["extraoptions"]["fit_IAW"]:
            sqdev["ion"] = jnp.square(i_data - ThryI) / (jnp.abs(i_data) + 1e-1)
            sqdev["ion"] = jnp.where(
                (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_min"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_cf_min"])
                )
                | (
                    (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_cf_max"])
                    & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_max"])
                ),
                sqdev["ion"],
                0.0,
            )
            loss += jnp.sum(sqdev["ion"], axis=1)
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

        if self.cfg["other"]["extraoptions"]["fit_EPWb"]:
            sqdev_e_b = jnp.square(e_data - ThryE) / jnp.abs(e_data)  # jnp.square(e_norm)
            sqdev_e_b = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"]),
                sqdev_e_b,
                0.0,
            )
            # not sure whether this should be lamAxisE[0,:]  or lamAxisE
            used_points += jnp.sum(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"])
            )

            loss += jnp.sum(sqdev_e_b, axis=1)
            sqdev["ele"] += sqdev_e_b

        if self.cfg["other"]["extraoptions"]["fit_EPWr"]:
            sqdev_e_r = jnp.square(e_data - ThryE) / jnp.abs(e_data)  # jnp.square(e_norm)
            sqdev_e_r = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"]),
                sqdev_e_r,
                0.0,
            )
            used_points += jnp.sum(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"])
            )

            loss += jnp.sum(sqdev_e_r, axis=1)
            sqdev["ele"] += sqdev_e_r

        return loss, sqdev, used_points, [ThryE, ThryI, params]

    def calc_ei_error(self, batch, ThryI, lamAxisI, ThryE, lamAxisE, denom, reduce_func=jnp.mean):
        """
        This function calculates the error in the fit of the IAW and EPW

        Args:
            batch:
            ThryI:
            lamAxisI:
            ThryE:
            lamAxisE:
            denom:
            reduce_func:

        Returns:

        """
        i_error = 0.0
        e_error = 0.0
        i_data = batch["i_data"]
        e_data = batch["e_data"]
        if self.cfg["other"]["extraoptions"]["fit_IAW"]:
            _error_ = jnp.square(i_data - ThryI) / denom[0]
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

            i_error += reduce_func(_error_)

        if self.cfg["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = jnp.square(e_data - ThryE) / denom[1]
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"]),
                _error_,
                0.0,
            )
            e_error += reduce_func(_error_)

        if self.cfg["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = jnp.square(e_data - ThryE) / denom[1]
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"]),
                _error_,
                0.0,
            )
            e_error += reduce_func(_error_)

        return i_error, e_error

    def _moment_loss_(self, params):
        """
        This function calculates the loss associated with regularizing the moments of the distribution function i.e.
        the density should be 1, the temperature should be 1, and momentum should be 0.

        Args:
            params:

        Returns:

        """
        if self.cfg["parameters"][self.e_species]["fe"]["dim"] == 1:
            dv = (
                self.cfg["parameters"][self.e_species]["fe"]["velocity"][1]
                - self.cfg["parameters"][self.e_species]["fe"]["velocity"][0]
            )
            if self.cfg["parameters"][self.e_species]["fe"]["symmetric"]:
                density_loss = jnp.mean(
                    jnp.square(1.0 - 2.0 * jnp.sum(jnp.exp(params[self.e_species]["fe"]) * dv, axis=1))
                )
                temperature_loss = jnp.mean(
                    jnp.square(
                        1.0
                        - 2.0
                        * jnp.sum(
                            jnp.exp(params[self.e_species]["fe"])
                            * self.cfg["parameters"][self.e_species]["fe"]["velocity"] ** 2.0
                            * dv,
                            axis=1,
                        )
                    )
                )
            else:
                density_loss = jnp.mean(jnp.square(1.0 - jnp.sum(jnp.exp(params[self.e_species]["fe"]) * dv, axis=1)))
                temperature_loss = jnp.mean(
                    jnp.square(
                        1.0
                        - jnp.sum(
                            jnp.exp(params[self.e_species]["fe"])
                            * self.cfg["parameters"][self.e_species]["fe"]["velocity"] ** 2.0
                            * dv,
                            axis=1,
                        )
                    )
                )
            momentum_loss = jnp.mean(
                jnp.square(
                    jnp.sum(
                        jnp.exp(params[self.e_species]["fe"])
                        * self.cfg["parameters"][self.e_species]["fe"]["velocity"]
                        * dv,
                        axis=1,
                    )
                )
            )
        else:
            density_loss = jnp.mean(
                jnp.square(
                    1.0
                    - trapz(
                        trapz(
                            jnp.exp(params[self.e_species]["fe"]), self.cfg["parameters"][self.e_species]["fe"]["v_res"]
                        ),
                        self.cfg["parameters"][self.e_species]["fe"]["v_res"],
                    )
                )
            )
            temperature_loss = jnp.mean(
                jnp.square(
                    1.0
                    - trapz(
                        trapz(
                            jnp.exp(params[self.e_species]["fe"])
                            * self.cfg["parameters"][self.e_species]["fe"]["velocity"][0]
                            * self.cfg["parameters"][self.e_species]["fe"]["velocity"][1],
                            self.cfg["parameters"][self.e_species]["fe"]["v_res"],
                        ),
                        self.cfg["parameters"][self.e_species]["fe"]["v_res"],
                    )
                )
            )
            # needs to be fixed
            # momentum_loss = jnp.mean(jnp.square(jnp.sum(jnp.exp(params["fe"]) * self.cfg["velocity"] * dv, axis=1)))
            momentum_loss = 0.0
            print(temperature_loss)
        return density_loss, temperature_loss, momentum_loss

    def calc_other_losses(self, params):
        if self.cfg["parameters"]["fe"]["fe_decrease_strict"]:
            gradfe = jnp.sign(self.cfg["velocity"][1:]) * jnp.diff(params["fe"].squeeze())
            vals = jnp.where(gradfe > 0, gradfe, 0).sum()
            fe_penalty = jnp.tan(jnp.amin(jnp.array([vals, jnp.pi / 2])))
        else:
            fe_penalty = 0

        return fe_penalty

    def _loss_for_hess_fn_(self, weights, batch):
        # params = params | self.static_params
        params = self.weights_to_params(weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.spec_calc(params, batch)
        i_error, e_error = self.calc_ei_error(
            batch,
            ThryI,
            lamAxisI,
            ThryE,
            lamAxisE,
            denom=[jnp.abs(batch["i_data"]) + 1e-10, jnp.abs(batch["e_data"]) + 1e-10],
            reduce_func=jnp.sum,
        )

        return i_error + e_error

    def __loss__(self, weights, batch: Dict):
        """
        This function calculates the value of the loss function

        Args:
            weights:
            batch:

        Returns:

        """
        params = self.weights_to_params(weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.spec_calc(params, batch)
        i_error, e_error = self.calc_ei_error(
            batch,
            ThryI,
            lamAxisI,
            ThryE,
            lamAxisE,
            denom=[jnp.square(self.i_norm), jnp.square(self.e_norm)],
            reduce_func=jnp.mean,
        )
        density_loss, temperature_loss, momentum_loss = self._moment_loss_(params)
        # other_losses = calc_other_losses(params)
        normed_batch = self._get_normed_batch_(batch)
        normed_e_data = normed_batch["e_data"]
        ion_error = self.cfg["data"]["ion_loss_scale"] * i_error

        total_loss = ion_error + e_error + density_loss + temperature_loss + momentum_loss
        return total_loss, [ThryE, normed_e_data, params]

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
            pytree_weights = self.unravel_pytree(weights)
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
        for k, v in config["parameters"][species].items():
            if k == "type":
                continue
            if v["active"]:
                active_or_inactive = "active"
            else:
                active_or_inactive = "inactive"

            if k != "fe":
                iw[active_or_inactive][species][k] = np.array(
                    [config["parameters"][species][k]["val"] for _ in range(num_slices)]
                )[:, None]
            else:
                iw[active_or_inactive][species][k] = np.concatenate(
                    [config["parameters"][species][k]["val"] for _ in range(num_slices)]
                )

            if v["active"]:
                lb[active_or_inactive][species][k] = np.array(
                    [0 * config["units"]["lb"][species][k] for _ in range(num_slices)]
                )
                ub[active_or_inactive][species][k] = np.array(
                    [1.0 + 0 * config["units"]["ub"][species][k] for _ in range(num_slices)]
                )

                iw[active_or_inactive][species][k] = (
                    iw[active_or_inactive][species][k] - config["units"]["shifts"][species][k]
                ) / config["units"]["norms"][species][k]

    return lb, ub, iw
