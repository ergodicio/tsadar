import copy
from typing import Dict

import jax
from jax import numpy as jnp
from equinox import filter_value_and_grad, filter_hessian, filter_jit
from jax.flatten_util import ravel_pytree
import numpy as np
import equinox as eqx

from ..core.thomson_diagnostic import ThomsonScatteringDiagnostic

# from ..core.modules import exchange_params, get_filter_spec
from ..utils.vector_tools import rotate


class LossFunction:
    """
    LossFunction is a class responsible for managing the forward pass and loss computation for inverse Thomson scattering analysis.
    This class encapsulates the logic for:
    - Normalizing input and output data based on configuration.
    - Computing theoretical spectra using a ThomsonScatteringDiagnostic instance.
    - Calculating loss values and gradients for optimization, supporting various loss metrics (L1, L2, log-cosh, Poisson).
    - Handling multiplexed analysis with EDF rotation if required.
    - Applying additional penalties and moment regularization to the loss.
    - Providing interfaces for loss, gradient, and Hessian computation compatible with optimization routines.
    Attributes:
        cfg (Dict): Configuration dictionary constructed from user inputs.
        ts_diag (ThomsonScatteringDiagnostic): Diagnostic object for theoretical spectrum calculation.
        multiplex_ang (bool): Indicates if multiplexed analysis with EDF rotation is enabled.
        i_norm, e_norm (float): Normalization factors for output data.
        i_input_norm, e_input_norm (float): Normalization factors for input data.
        _loss_, _vg_func_, _h_func_ (callable): JIT-compiled loss, value-and-grad, and Hessian functions.
        array_loss (callable): JIT-compiled postprocessing loss function.
    Methods:
        __init__(cfg, scattering_angles, dummy_batch):
            Initializes the LossFunction with configuration, angles, and dummy data for normalization.
        _get_normed_batch_(batch):
            Returns a normalized copy of the input batch.
        vg_loss(diff_weights, static_weights, batch):
            Computes the loss value and gradient with respect to weights for optimization.
        h_loss_wrt_params(weights, batch):
            Computes the Hessian of the loss with respect to parameters.
        _loss_for_hess_fn_(weights, batch):
            Loss function used for Hessian computation.
        calc_ei_error(batch, ThryI, lamAxisI, ThryE, lamAxisE, uncert, reduce_func):
            Calculates the error between experimental and theoretical spectra for IAW and EPW.
        calc_loss(ts_params, batch, denom, reduce_func):
            Computes the total loss, including penalties and normalization, for a given parameter set and batch.
        loss(weights, batch):
            Returns the scalar loss value for a given set of weights and batch.
        __loss__(diff_weights, static_weights, batch):
            Internal loss function wrapper for optimization routines.
        post_loss(weights, batch):
            Computes the loss and additional outputs for postprocessing.
        loss_functionals(d, t, uncert, method="l2"):
            Computes the element-wise loss between data and theory using the specified metric.
        penalties(weights):
            Computes additional penalties (e.g., parameter bounds, moment regularization) to be added to the loss.
        _moment_loss_(params):
            Computes regularization losses for the moments (density, temperature, momentum) of the distribution function.
    Usage:
        Instantiate with configuration, scattering angles, and dummy data. Use `vg_loss` or `loss` for optimization routines.
    """

    def __init__(self, cfg: Dict, scattering_angles, dummy_batch):
        """
        Initializes the loss function class with configuration, scattering angles, and dummy batch data.
            cfg (Dict): Configuration dictionary constructed from the inputs.
            scattering_angles (dict): Dictionary containing the scattering angles and their relative weights.
            dummy_batch (dict): Dictionary of dummy data used for normalization and input scaling.
        Attributes:
            cfg (Dict): Stores the configuration dictionary.
            i_norm (float): Normalization factor for i_data, set to its maximum if y_norm is enabled, otherwise 1.0.
            e_norm (float): Normalization factor for e_data, set to its maximum if y_norm is enabled, otherwise 1.0.
            i_input_norm (float): Input normalization for i_data, set to its maximum if x_norm and nn.use are enabled, otherwise 1.0.
            e_input_norm (float): Input normalization for e_data, set to its maximum if x_norm and nn.use are enabled, otherwise 1.0.
            multiplex_ang (bool): Indicates if analysis is performed twice with rotation of the EDF, based on shotnum type.
            ts_diag (ThomsonScatteringDiagnostic): Instance for Thomson scattering diagnostics.
            _loss_ (Callable): JIT-compiled loss function.
            _vg_func_ (Callable): JIT-compiled value and gradient function for the loss.
            _h_func_ (Callable): JIT-compiled Hessian function for the loss.
            array_loss (Callable): JIT-compiled post-processing loss function.
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

        self._loss_ = filter_jit(self.__loss__)
        self._vg_func_ = filter_jit(filter_value_and_grad(self.__loss__, has_aux=True))
        ## this will be replaced with jacobian params jacobian inverse
        self._h_func_ = filter_jit(filter_hessian(self._loss_for_hess_fn_))
        self.array_loss = filter_jit(self.post_loss)

    def _get_normed_batch_(self, batch: Dict):
        """
        Normalizes the input batch by dividing the 'i_data' and 'e_data' fields by their respective normalization factors.
            
        Args:
            batch (Dict): A dictionary containing at least the keys 'i_data' and 'e_data', representing input data arrays.
        Returns:
            normed_batch (Dict): A deep-copied and normalized version of the input batch, where 'i_data' and 'e_data' are divided by    
            the normalization factors defined in the class instance (self.i_norm and self.e_norm).
        """
        normed_batch = copy.deepcopy(batch)
        normed_batch["i_data"] = normed_batch["i_data"] / self.i_input_norm
        normed_batch["e_data"] = normed_batch["e_data"] / self.e_input_norm
        return normed_batch

    def vg_loss(self, diff_weights, static_weights: Dict, batch: Dict):
        """
        Computes the value of the loss function and its gradient with respect to the weights for optimization.
        This function serves as the main interface for evaluating the loss and its gradient, which are used to assess
        the goodness-of-fit and to update the model weights during optimization. It handles necessary pre- and post-
        processing steps required by the optimization software.
        The behavior of this function depends on the optimizer method specified in the configuration:
          - For "l-bfgs-b", it unravels the weights, computes the loss and gradient, flattens the gradient, and returns
            both the loss value and the flattened gradient.
          - For other methods, it directly returns the result of the internal loss function, which is a PyTree.
        Args:
            diff_weights: The differentiable (trainable) weights to be optimized, possibly in a flattened format.
            static_weights (Dict): The static (non-trainable) weights used in the computation.
            batch (Dict): The batch of data used for evaluating the loss and gradient.
        Returns:
            Tuple[float, np.ndarray] or Any:
                - If using "l-bfgs-b" optimizer: Returns a tuple containing the loss value and the flattened gradient array.
                - Otherwise: Returns the result of the internal loss function, which is a tuple containing the loss value and the structured gradient tree.
        

        """
        if self.cfg["optimizer"]["method"] == "l-bfgs-b":
            # pytree_weights = self.ts_diag.unravel_pytree(weights)

            diff_weights = self.unravel_weights(diff_weights)
            (value, aux), grad = self._vg_func_(diff_weights, static_weights, batch)

            # if "fe" in grad:
            #     grad["fe"] = self.cfg["optimizer"]["grad_scalar"] * grad["fe"]

            # for species in self.cfg["parameters"].keys():
            #     for k, param_dict in self.cfg["parameters"][species].items():
            #         if param_dict["active"]:
            #             scalar = param_dict["gradient_scalar"] if "gradient_scalar" in param_dict else 1.0
            #             grad[species][k] *= scalar

            temp_grad, _ = ravel_pytree(grad)
            flattened_grads = np.array(temp_grad)
            return value, flattened_grads
        else:
            return self._vg_func_(diff_weights, static_weights, batch)

    def h_loss_wrt_params(self, weights, batch):
        return self._h_func_(weights, batch)

    def _loss_for_hess_fn_(self, weights, batch):
        # this function is not being used? if so it has syntax issues
        # params = params | self.static_params
        # params = self.ts_diag.get_plasma_parameters(weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(weights, batch)
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
        Calculates the error metrics for ion and electron spectral fits based on theoretical and experimental data.
        This function computes the error between measured and theoretical spectra for both ion (IAW) and electron (EPW)
        features, applying configurable fitting ranges and loss methods. The errors are reduced using the specified
        reduction function (default is mean), and squared deviations are accumulated for further analysis.
        Args:
            batch (dict): Dictionary containing experimental data arrays with keys "i_data" (ion data) and "e_data" (electron data).
            ThryI (array-like): Theoretical ion spectrum corresponding to i_data.
            lamAxisI (array-like): Wavelength axis for the ion spectrum.
            ThryE (array-like): Theoretical electron spectrum corresponding to e_data.
            lamAxisE (array-like): Wavelength axis for the electron spectrum.
            uncert (tuple or list): Tuple or list containing uncertainty arrays for ion and electron data, respectively.
            reduce_func (callable, optional): Function to reduce the error array to a scalar (e.g., jnp.mean, jnp.sum). Defaults to jnp.mean.
        Returns:
            tuple:
                i_error (float): Reduced error metric for the ion feature (IAW).
                e_error (float): Reduced error metric for the electron feature (EPW).
                sqdev (dict): Dictionary with keys "ion" and "ele" containing arrays of squared deviations for ion and electron data, respectively.
        Notes:
            - The function uses configuration options from self.cfg to determine which features to fit and the wavelength ranges.
            - If both blue and red EPW features are fit, the electron error is averaged accordingly.
            - NaN values are used to mask out-of-range points and are handled with jnp.nan_to_num when accumulating squared deviations.
        """

        i_error = 0.0
        e_error = 0.0
        # used_points = 0
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
                jnp.nan,
            )

            i_error += reduce_func(_error_)
            sqdev["ion"] = jnp.nan_to_num(_error_)

        if self.cfg["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = self.loss_functionals(e_data, ThryE, uncert[1], method=self.cfg["optimizer"]["loss_method"])
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"]),
                _error_,
                jnp.nan,
            )

            e_error += reduce_func(_error_)
            sqdev["ele"] += jnp.nan_to_num(_error_)

        if self.cfg["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = self.loss_functionals(e_data, ThryE, uncert[1], method=self.cfg["optimizer"]["loss_method"])
            _error_ = jnp.where(
                (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"])
                & (lamAxisE < self.cfg["data"]["fit_rng"]["red_max"]),
                _error_,
                jnp.nan,
            )

            e_error += reduce_func(_error_)
            if self.cfg["other"]["extraoptions"]["fit_EPWb"]:
                # the set e_error to the true mean if both sides are fit
                e_error *= 1.0 / 2.0
            sqdev["ele"] += jnp.nan_to_num(_error_)

        return i_error, e_error, sqdev

    def calc_loss(self, ts_params, batch: Dict, denom, reduce_func):
        """
        Calculates the total loss for the inverse Thomson scattering model, including electron and ion errors,
        and applies any necessary penalties. Handles both multiplexed and non-multiplexed angular configurations.
        Args:
            ts_params (dict): Dictionary of Thomson scattering parameters, including electron distribution.
            batch (Dict): Batch of experimental data. If multiplex_ang is True, expects keys "b1" and "b2".
            denom (list or []): Denominator(s) for normalization. If empty, will be set to theoretical values.
            reduce_func (callable): Function to reduce error arrays (e.g., sum, mean).
        Returns:
            tuple:
                total_loss (float): The computed total loss value (sum of scaled ion error, electron error, and penalties).
                sqdev (Any): Squared deviation(s) between theoretical and experimental data.
                ThryE (Any): Theoretical electron spectrum.
                ThryI (Any): Theoretical ion spectrum.
                ts_params (dict): (Possibly updated) Thomson scattering parameters.
        """

        if self.multiplex_ang:
            # params has been replace with the new ts_params but behavior has not been checked 2-20-25
            ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(ts_params, batch["b1"])
            # jax.debug.print("fe size {e_error}", e_error=jnp.shape(params["electron"]['fe']))
            ts_params["electron"]["fe"] = rotate(
                jnp.squeeze(ts_params["electron"]["fe"]), self.cfg["data"]["shot_rot"] * jnp.pi / 180.0
            )

            ThryE_rot, _, _, _ = self.ts_diag(ts_params, batch["b2"])
            i_error1, e_error1, sqdev = self.calc_ei_error(
                batch["b1"],
                ThryI,
                lamAxisI,
                ThryE,
                lamAxisE,
                denom,
                reduce_func,
            )
            i_error2, e_error2, sqdev = self.calc_ei_error(
                batch["b2"],
                ThryI,
                lamAxisI,
                ThryE_rot,
                lamAxisE,
                denom,
                reduce_func,
            )
            i_error = i_error1 + i_error2
            e_error = e_error1 + e_error2

            normed_batch = self._get_normed_batch_(batch["b1"])
        else:
            ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(ts_params, batch)
            if denom == []:
                denom = [ThryI, ThryE]
            i_error, e_error, sqdev = self.calc_ei_error(
                batch,
                ThryI,
                lamAxisI,
                ThryE,
                lamAxisE,
                denom,
                reduce_func,
            )

            normed_batch = self._get_normed_batch_(batch)

        normed_e_data = normed_batch["e_data"]
        ion_error = self.cfg["data"]["ion_loss_scale"] * i_error

        penalty_error = 0.0  # self.penalties(weights)
        total_loss = ion_error + e_error + penalty_error
        # jax.debug.print("e_error {total_loss}", total_loss=e_error)

        return total_loss, sqdev, ThryE, ThryI, ts_params()
        # return total_loss, [ThryE, params]

    def loss(self, weights, batch: Dict):
        """
        High level function that returns the value of the loss function for a given set of weights and a batch of data.
        Depending on the optimizer method specified in the configuration, this function may first
        convert the flat weights array into a pytree structure before computing the loss.
            
        Args:
            weights: The weights to be used in the loss function, either in a flat format or as a pytree.
            batch (Dict): A dictionary containing the data to be used in the loss function.
        Returns:
            float: The computed loss value.

        """
        if self.cfg["optimizer"]["method"] == "l-bfgs-b":
            pytree_weights = self.unravel_pytree(weights)
            value, _ = self._loss_(pytree_weights, batch)
            return value
        else:
            return self._loss_(weights, batch)

    def __loss__(self, diff_weights, static_weights, batch: Dict):
        """
        Output wrapper
        """

        weights = eqx.combine(static_weights, diff_weights)
        total_loss, sqdev, ThryE, normed_e_data, params = self.calc_loss(
            weights, batch, denom=[jnp.square(self.i_norm), jnp.square(self.e_norm)], reduce_func=jnp.nanmean
        )
        return total_loss, [ThryE, params]

    def post_loss(self, weights, batch: Dict):
        """
        Output wrapper for postprocessing
        """

        def nanamean(a):
            return jnp.nanmean(a, axis=1)

        total_loss, sqdev, ThryE, normed_e_data, params = self.calc_loss(weights, batch, denom=[], reduce_func=nanamean)
        return total_loss, sqdev, ThryE, normed_e_data, params

    def loss_functionals(self, d, t, uncert, method="l2"):
        """
        Computes the loss between predicted and target values using various loss functionals.
        
        Parameters
        ----------
        d : array-like
            Data values.
        t : array-like
            Theroetical values.
        uncert : array-like
            Uncertainty values used for normalization.
        method : str, optional
            The loss functional to use. Options are:
                - "l1": Mean absolute error, normalized by uncertainty.
                - "l2": Mean squared error, normalized by uncertainty.
                - "log-cosh": Log-cosh loss.
                - "poisson": Poisson loss.
        Returns
        -------
        _error_ : array-like
            Computed loss values according to the selected method.
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
        Computes the total penalty for the given model parameters (weights), including parameter constraints,
        optional moment losses, and an optional strict penalty on the electron distribution function.
        Args:
            weights (dict): Dictionary containing model parameters for each species. Each species entry is itself
                a dictionary of parameter arrays.
        Returns:
            jnp.ndarray: The total penalty value as a scalar.
        Penalties included:
            - Parameter penalty: Applies a log-based penalty to all parameters except 'fe' for each species.
            - Moment loss: If enabled in the configuration, adds density, temperature, and momentum losses.
            - Electron distribution penalty: If enabled in the configuration, penalizes increases in the electron
              distribution function ('fe') along the velocity axis.
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
            gradfe = jnp.sign(self.cfg["velocity"][1:]) * jnp.diff(weights["fe"].squeeze())
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
        Computes the density, temperature, and momentum loss terms for the electron distribution function
        based on the current model parameters and configuration.
        The loss terms are calculated differently depending on whether the velocity space is 1D or 2D:
        - For 1D velocity space:
            - Density loss enforces normalization of the electron distribution.
            - Temperature loss enforces the correct second moment (temperature) of the distribution.
            - Momentum loss enforces the first moment (mean velocity) to be zero.
            - If the distribution is symmetric, normalization and temperature are doubled.
        - For 2D velocity space:
            - Density loss is based on the sum of the exponentiated distribution times the velocity resolution squared.
            - Temperature loss is based on the second moment of the distribution.
            - Momentum loss is currently set to zero (not implemented).
        Args:
            params (dict): Dictionary containing model parameters, specifically the electron distribution function
                           under 'params["electron"]["fe"]'.
        Returns:
            tuple: (density_loss, temperature_loss, momentum_loss)
                - density_loss (float): Loss term enforcing normalization of the distribution.
                - temperature_loss (float): Loss term enforcing the correct temperature (second moment).
                - momentum_loss (float): Loss term enforcing zero mean velocity (first moment).
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
            fedens = (
                jnp.sum(jnp.exp(params["electron"]["fe"])) * self.cfg["parameters"]["electron"]["fe"]["v_res"] ** 2.0
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
            second_moment = (
                jnp.sum(
                    jnp.exp(params["electron"]["fe"])
                    * (
                        self.cfg["parameters"]["electron"]["fe"]["velocity"][0] ** 2
                        + self.cfg["parameters"]["electron"]["fe"]["velocity"][1] ** 2
                    )
                )
                * self.cfg["parameters"]["electron"]["fe"]["v_res"] ** 2.0
            )
            jax.debug.print("second moment = {fedens}", fedens=second_moment)
            temperature_loss = jnp.mean(jnp.square(1.0 - second_moment / 2))
            # needs to be fixed, was using a custom trapz function not the jax one
            first_moment = second_moment = jnp.trapz(
                jnp.trapz(
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
