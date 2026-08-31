"""LossFunction: computes the forward-model spectra, the fit error/loss between them and the data, and the
gradients/Hessian used by the optimizers in loops.py."""
import copy
from typing import Callable, Dict

import jax
from jax import numpy as jnp
from equinox import filter_value_and_grad, filter_hessian, filter_jit
from jax.flatten_util import ravel_pytree
import numpy as np
import equinox as eqx

from ..core.thomson_diagnostic import ThomsonScatteringDiagnostic

# from ..core.modules import exchange_params, get_filter_spec
from ..utils.vector_tools import rotate


_ANGULAR_OBJECTIVE_DEFAULTS = {
    "noise": {
        "model": "poisson_read",
        "read_noise": 1.0,
        "excess_noise_factor": 1.0,
        "background_variance_scale": 1.0,
        "variance_floor": 1.0e-6,
        "averaged_pixels": "auto",
    },
    "gain": {
        "mode": "per_row",
        "smoothness": 0.01,
        "prior_strength": 0.01,
        "prior_mean": 1.0,
        "minimum": 0.0,
    },
    "robust": {"kind": "gaussian", "threshold": 3.0, "iterations": 3, "dof": 4.0},
    "regularization": {
        "radial_smoothness": 0.0,
        "angular_smoothness": 0.0,
        "kl_to_maxwellian": 0.0,
        "density": 0.0,
        "temperature": 0.0,
        "momentum": 0.0,
        "density_target": 1.0,
        "temperature_target": 2.0,
        "momentum_target": [0.0, 0.0],
    },
}


def _merge_known_options(defaults, supplied, path):
    """Recursively merge a small objective schema, rejecting misspelled settings."""
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown {path} setting(s): {', '.join(unknown)}")
    merged = {}
    for key, default in defaults.items():
        value = supplied.get(key, default)
        merged[key] = _merge_known_options(default, value, f"{path}.{key}") if isinstance(default, dict) else value
    return merged


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
        spectype = cfg.get("other", {}).get("extraoptions", {}).get("spectype", "")
        self.is_angular = "angular" in spectype
        self.angular_objective = None
        if self.is_angular:
            self.angular_objective = self._validated_angular_objective(
                cfg.get("optimizer", {}).get("angular_objective", {})
            )

        if cfg["optimizer"]["y_norm"]:
            self.i_norm = np.amax(dummy_batch["i_amps"])
            self.e_norm = np.amax(dummy_batch["e_amps"])
            # self.i_norm = np.amax(dummy_batch["i_data"])
            # self.e_norm = np.amax(dummy_batch["e_data"])
        else:
            self.i_norm = self.e_norm = 1.0

        if cfg["optimizer"]["x_norm"] and cfg["nn"]["use"]:
            self.i_input_norm = np.amax(dummy_batch["i_data"])
            self.e_input_norm = np.amax(dummy_batch["e_data"])
        else:
            self.i_input_norm = self.e_input_norm = 1.0

        if cfg["optimizer"]["loss_method"] == "covar":
                self.sig_px = 1.0 #this is device specific and can be left hardcoded
                self.sig_rn = 17.0 # this is from the background of the camera and should be derived from the data image
                self.n = 2 * cfg["data"]["dpixel"] + 1
                self.G = 108
                self.F2 = 1.15

                self.num_free_params = sum(
                    p["active"]
                    for category in cfg["parameters"].values()
                    for p in category.values()
                    if isinstance(p, dict) and "active" in p
                )*cfg["optimizer"]["batch_size"]
                # CCD spread function
                a, b = np.meshgrid(np.linspace(-5*self.sig_px, 5*self.sig_px, int(10*self.sig_px+1)), 
                            np.linspace(-5*self.sig_px, 5*self.sig_px, int(10*self.sig_px+1)))
                self.g = 1 / (2 * np.pi * self.sig_px**2) * np.exp(-(a**2 + b**2) / (2 * self.sig_px**2))

        # boolean used to determine if the analyis is performed twice with rotation of the EDF
        self.multiplex_ang = isinstance(cfg["data"]["shotnum"], list)

        ############

        self.ts_diag = ThomsonScatteringDiagnostic(cfg, scattering_angles=scattering_angles)

        # Set by _1d_scipy_loop_ (loops.py) before use, via ravel_pytree(diff_params) -- the unraveling
        # function matching that particular fit's parameter pytree structure. Declared here so it's a
        # known attribute rather than one only ever assigned from outside the class.
        self.unravel_weights: Callable = None  # type: ignore[assignment]

        self._loss_ = filter_jit(self.__loss__)
        self._vg_func_ = filter_jit(filter_value_and_grad(self.__loss__, has_aux=True))
        ## this will be replaced with jacobian params jacobian inverse
        self._h_func_ = filter_jit(filter_hessian(self._loss_for_hess_fn_))
        self.array_loss = filter_jit(self.post_loss)

    def _validated_angular_objective(self, supplied):
        """Return the complete ARTS objective config or reject unsupported choices."""
        options = _merge_known_options(_ANGULAR_OBJECTIVE_DEFAULTS, supplied, "optimizer.angular_objective")
        supplied_gain = supplied.get("gain", {}) if isinstance(supplied, dict) else {}
        if options["gain"]["mode"] in {"none", "global"} and "smoothness" not in supplied_gain:
            options["gain"]["smoothness"] = 0.0
        if options["gain"]["mode"] == "none" and "prior_strength" not in supplied_gain:
            options["gain"]["prior_strength"] = 0.0

        if self.cfg["optimizer"].get("loss_method", "l2") != "l2":
            raise ValueError(
                "The noise-aware ARTS objective requires optimizer.loss_method: l2; "
                "robust contamination is selected with optimizer.angular_objective.robust.kind."
            )

        noise = options["noise"]
        if noise["model"] not in {"poisson_read", "measured_variance", "constant"}:
            raise ValueError(
                "optimizer.angular_objective.noise.model must be poisson_read, measured_variance, or constant"
            )
        for key in ("read_noise", "excess_noise_factor", "background_variance_scale", "variance_floor"):
            if float(noise[key]) < 0.0:
                raise ValueError(f"optimizer.angular_objective.noise.{key} must be non-negative")
        if float(noise["variance_floor"]) <= 0.0:
            raise ValueError("optimizer.angular_objective.noise.variance_floor must be positive")
        if noise["averaged_pixels"] != "auto" and float(noise["averaged_pixels"]) <= 0.0:
            raise ValueError("optimizer.angular_objective.noise.averaged_pixels must be auto or positive")

        gain = options["gain"]
        if gain["mode"] not in {"none", "global", "per_row", "per_row_wing"}:
            raise ValueError(
                "optimizer.angular_objective.gain.mode must be none, global, per_row, or per_row_wing"
            )
        for key in ("smoothness", "prior_strength", "minimum"):
            if float(gain[key]) < 0.0:
                raise ValueError(f"optimizer.angular_objective.gain.{key} must be non-negative")
        if gain["mode"] == "none" and (
            float(gain["smoothness"]) > 0.0 or float(gain["prior_strength"]) > 0.0
        ):
            raise ValueError("ARTS gain smoothness/prior_strength require a profiled gain mode")
        if gain["mode"] == "global" and float(gain["smoothness"]) > 0.0:
            raise ValueError("ARTS gain smoothness requires per_row or per_row_wing gain mode")
        fitted_amplitudes = (
            ("amp1", bool(self.cfg.get("data", {}).get("fit_EPWb", False))),
            ("amp2", bool(self.cfg.get("data", {}).get("fit_EPWr", False))),
        )
        active_amplitudes = [
            name
            for name, fitted in fitted_amplitudes
            if fitted
            and bool(
                self.cfg.get("parameters", {})
                .get("general", {})
                .get(name, {})
                .get("active", False)
            )
        ]
        if (
            gain["mode"] != "none"
            and float(gain["prior_strength"]) == 0.0
            and active_amplitudes
        ):
            raise ValueError(
                "Unanchored profiled ARTS gains are not identifiable with active fitted amplitude "
                f"parameter(s) {', '.join(active_amplitudes)}; set those amplitudes inactive or "
                "configure optimizer.angular_objective.gain.prior_strength > 0."
            )

        robust = options["robust"]
        if robust["kind"] not in {"gaussian", "huber", "student_t"}:
            raise ValueError("optimizer.angular_objective.robust.kind must be gaussian, huber, or student_t")
        if float(robust["threshold"]) <= 0.0 or float(robust["dof"]) <= 0.0:
            raise ValueError("ARTS robust threshold and degrees of freedom must be positive")
        if int(robust["iterations"]) < 1:
            raise ValueError("optimizer.angular_objective.robust.iterations must be at least one")

        regularization = options["regularization"]
        for key in (
            "radial_smoothness",
            "angular_smoothness",
            "kl_to_maxwellian",
            "density",
            "temperature",
            "momentum",
        ):
            if float(regularization[key]) < 0.0:
                raise ValueError(f"optimizer.angular_objective.regularization.{key} must be non-negative")
        momentum_target = regularization["momentum_target"]
        if not isinstance(momentum_target, (list, tuple)) or len(momentum_target) != 2:
            raise ValueError("optimizer.angular_objective.regularization.momentum_target must contain [vx, vy]")

        # Backward compatibility is explicit: the old boolean now activates all physical
        # moment priors instead of silently doing nothing.
        if self.cfg["optimizer"].get("moment_loss", False):
            for key in ("density", "temperature", "momentum"):
                if float(regularization[key]) == 0.0:
                    regularization[key] = 1.0

        fe_config = self.cfg.get("parameters", {}).get("electron", {}).get("fe", {})
        if fe_config.get("fe_decrease_strict", False):
            raise ValueError(
                "fe_decrease_strict is not supported by the noise-aware angular objective; configure the "
                "documented physical-EDF smoothness priors instead."
            )
        return options

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
            self.aux = aux

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
        """
        Computes the Hessian of the loss with respect to the (active) fitted parameters, using the
        JIT-compiled Hessian function built from _loss_for_hess_fn_ in __init__. Used by postprocessing to
        derive parameter uncertainties from the curvature of the loss (see postprocess.get_sigmas).

        Args:
            weights: the parameter values to evaluate the Hessian at (typically the best-fit weights).
            batch (Dict): batch of data to evaluate the loss against.

        Returns:
            The Hessian of the loss with respect to weights, in the same nested structure as weights.
        """
        return self._h_func_(weights, batch)

    def _loss_for_hess_fn_(self, weights, batch):
        if self.is_angular:
            total_loss, _, _, _, _ = self.calc_loss(
                weights, batch, denom=[], reduce_func=jnp.nanmean
            )
            return total_loss

        ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(weights, batch)
        i_error, e_error, _ = self.calc_ei_error(
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

        if self.cfg["data"]["fit_IAW"]:
            mask = (
                (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_min"])
                & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_cf_min"])
            ) | (
                (lamAxisI > self.cfg["data"]["fit_rng"]["iaw_cf_max"])
                & (lamAxisI < self.cfg["data"]["fit_rng"]["iaw_max"])
            )
            # covar_source is i_data here (not ThryI) -- matches the original per-branch behavior below.
            error, sqd = self._feature_error_(i_data, ThryI, uncert[0], mask, i_data, reduce_func)
            i_error += error
            sqdev["ion"] = sqd

        if self.cfg["data"]["fit_EPWb"]:
            mask = (lamAxisE > self.cfg["data"]["fit_rng"]["blue_min"]) & (
                lamAxisE < self.cfg["data"]["fit_rng"]["blue_max"]
            )
            error, sqd = self._feature_error_(e_data, ThryE, uncert[1], mask, ThryE, reduce_func)
            e_error += error
            sqdev["ele"] = sqd

        if self.cfg["data"]["fit_EPWr"]:
            mask = (lamAxisE > self.cfg["data"]["fit_rng"]["red_min"]) & (
                lamAxisE < self.cfg["data"]["fit_rng"]["red_max"]
            )
            error, sqd = self._feature_error_(e_data, ThryE, uncert[1], mask, ThryE, reduce_func)
            e_error += error

            if self.cfg["data"]["fit_EPWb"]:
                # the set e_error to the true mean if both sides are fit
                e_error *= 1.0 / 2.0
            sqdev["ele"] += sqd

        return i_error, e_error, sqdev

    def _feature_error_(self, data, thry, uncert, mask, covar_source, reduce_func):
        """
        Shared per-feature (IAW / EPW-blue / EPW-red) error computation used by calc_ei_error: applies the loss
        functional, masks to the feature's fit range, and reduces either via reduce_func or, for
        loss_method=="covar", via the covariance-weighted quadratic form. covar_source is the array
        calculate_covariance_matrix is built from -- i_data for the ion branch, ThryE for both electron
        branches, matching each branch's original behavior.

        Returns:
            tuple: (error, sqdev) where sqdev is nan_to_num(masked _error_), matching what calc_ei_error
            stored for each branch before this was factored out.
        """
        _error_ = self.loss_functionals(data, thry, uncert, method=self.cfg["optimizer"]["loss_method"])
        _error_ = jnp.where(mask, _error_, jnp.nan)

        if self.cfg["optimizer"]["loss_method"] == "covar":
            k = self.calculate_covariance_matrix(covar_source)
            norm = jnp.sum(jnp.isfinite(_error_)) - self.num_free_params
            print(norm)
            _error_ = jnp.nan_to_num(_error_)
            x = jnp.linalg.solve(k, _error_[..., None]).squeeze(-1)
            error = jnp.sum(jnp.vecdot(_error_, x)) / norm
        else:
            error = reduce_func(_error_)

        return error, jnp.nan_to_num(_error_)

    def _angular_variance(self, batch):
        """Detector variance fixed from measured counts, background, and read noise.

        Keeping this estimate independent of the trial spectrum preserves the linear
        variable-projection problem for the gain nuisance parameters. ``noise_e`` is the
        measured/background-subtraction mean already added by the forward model.
        """
        options = self.angular_objective["noise"]
        data = jnp.asarray(batch["e_data"])
        background = jnp.broadcast_to(jnp.asarray(batch["noise_e"]), data.shape)
        floor = float(options["variance_floor"])
        averaged_pixels = options["averaged_pixels"]
        if averaged_pixels == "auto":
            averaged_pixels = float(self.cfg.get("other", {}).get("ang_res_unit", 1)) * float(
                self.cfg.get("other", {}).get("lam_res_unit", 1)
            )
        else:
            averaged_pixels = float(averaged_pixels)
        if options["model"] == "measured_variance":
            if "e_variance" not in batch:
                raise ValueError(
                    "noise.model=measured_variance requires an e_variance array in the angular batch"
                )
            variance = jnp.broadcast_to(jnp.asarray(batch["e_variance"]), data.shape)
        elif options["model"] == "constant":
            variance = jnp.full_like(data, float(options["read_noise"]) ** 2 / averaged_pixels)
        else:
            signal_counts = jnp.maximum(data - background, 0.0)
            variance = (
                float(options["read_noise"]) ** 2
                + float(options["excess_noise_factor"]) * signal_counts
                + float(options["background_variance_scale"]) * jnp.abs(background)
            ) / averaged_pixels
        return jnp.maximum(variance, floor)

    def _angular_fit_masks(self, lam_axis, shape):
        """Return blue, red, and combined wavelength masks broadcast over detector rows."""
        fr = self.cfg["data"]["fit_rng"]
        lam_axis = jnp.ravel(lam_axis)
        blue_1d = (
            (lam_axis > fr["blue_min"])
            & (lam_axis < fr["blue_max"])
            & bool(self.cfg["data"]["fit_EPWb"])
        )
        red_1d = (
            (lam_axis > fr["red_min"])
            & (lam_axis < fr["red_max"])
            & bool(self.cfg["data"]["fit_EPWr"])
        )
        blue = jnp.broadcast_to(blue_1d[None, :], shape)
        red = jnp.broadcast_to(red_1d[None, :], shape)
        return blue, red, blue | red

    def _solve_gain_series(self, information, rhs):
        """Profile one lower-bounded angular gain series with calibration priors."""
        gain_options = self.angular_objective["gain"]
        information = jnp.asarray(information)
        rhs = jnp.asarray(rhs)
        n_gains = information.shape[0]
        populated = information > 0.0
        info_scale = jnp.sum(information) / jnp.maximum(jnp.sum(populated), 1.0)
        info_scale = jnp.maximum(info_scale, 1.0e-12)
        prior_precision = float(gain_options["prior_strength"]) * info_scale
        smooth_precision = float(gain_options["smoothness"]) * info_scale

        if n_gains > 1:
            difference = jnp.eye(n_gains - 1, n_gains, k=1) - jnp.eye(
                n_gains - 1, n_gains
            )
            laplacian = difference.T @ difference
        else:
            laplacian = jnp.zeros((1, 1), dtype=information.dtype)
        system = (
            jnp.diag(information + prior_precision)
            + smooth_precision * laplacian
            + jnp.eye(n_gains) * info_scale * 1.0e-10
        )
        target = rhs + prior_precision * float(gain_options["prior_mean"])
        minimum = jnp.asarray(float(gain_options["minimum"]), dtype=information.dtype)
        lower = jnp.full_like(information, minimum)

        def equality_solution(active):
            """Minimize the quadratic with the selected gains fixed at the bound."""
            free = ~active
            fixed = jnp.where(active, lower, 0.0)
            free_outer = free[:, None] & free[None, :]
            constrained_system = jnp.where(free_outer, system, 0.0) + jnp.diag(
                active.astype(system.dtype)
            )
            constrained_target = jnp.where(free, target - system @ fixed, lower)
            return (
                jnp.linalg.solve(constrained_system, constrained_target),
                constrained_system,
            )

        # A primal active-set solve is used rather than clipping the unconstrained
        # solution. Clipping is not the constrained minimizer when the smoothness
        # Laplacian couples adjacent gains. The loop has static length for reverse-mode
        # autodiff; completed solves take the no-op branch.
        unconstrained = jnp.linalg.solve(system, target)
        gains = jnp.maximum(unconstrained, lower)
        active = unconstrained <= lower
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.max(jnp.abs(target)),
                jnp.max(jnp.abs(system)) * jnp.max(jnp.abs(lower)),
            ),
        )
        tolerance = 32.0 * jnp.finfo(system.dtype).eps * scale

        def active_set_step(state):
            current, working_set, _ = state
            candidate, _ = equality_solution(working_set)
            free = ~working_set
            direction = candidate - current
            violates = free & (candidate < lower - tolerance)
            ratios = jnp.where(
                free & (direction < -tolerance),
                (current - lower) / jnp.maximum(-direction, tolerance),
                jnp.inf,
            )
            step = jnp.minimum(1.0, jnp.min(ratios))
            hit_index = jnp.argmin(ratios)

            def add_blocking_constraint(_):
                boundary = jnp.maximum(current + step * direction, lower)
                return boundary, working_set.at[hit_index].set(True), jnp.asarray(False)

            def accept_candidate(_):
                feasible = jnp.maximum(candidate, lower)
                gradient = system @ feasible - target
                active_gradient = jnp.where(working_set, gradient, jnp.inf)
                release_index = jnp.argmin(active_gradient)
                release = jnp.min(active_gradient) < -tolerance
                updated_set = jax.lax.cond(
                    release,
                    lambda selected: working_set.at[selected].set(False),
                    lambda selected: working_set,
                    release_index,
                )
                return feasible, updated_set, ~release

            return jax.lax.cond(
                jnp.any(violates),
                add_blocking_constraint,
                accept_candidate,
                operand=None,
            )

        def active_set_iteration(_, state):
            return jax.lax.cond(state[2], lambda value: value, active_set_step, state)

        gains, active, converged = jax.lax.fori_loop(
            0,
            4 * n_gains + 4,
            active_set_iteration,
            (gains, active, jnp.asarray(False)),
        )
        gains = jnp.where(converged, gains, jnp.full_like(gains, jnp.nan))
        _, conditional_system = equality_solution(active)
        conditional_variance = jnp.maximum(
            jnp.diag(jnp.linalg.inv(conditional_system)), 0.0
        )
        # Bound-active gains have no two-sided local Gaussian degree of freedom.
        gain_standard_error = jnp.where(active, 0.0, jnp.sqrt(conditional_variance))
        prior_quadratic = prior_precision * jnp.sum(
            (gains - float(gain_options["prior_mean"])) ** 2
        )
        smooth_quadratic = smooth_precision * jnp.sum(jnp.diff(gains) ** 2)
        return gains, gain_standard_error, prior_quadratic, smooth_quadratic

    def _profile_angular_gains(
        self, signal, target, inverse_variance, valid, blue, red
    ):
        """Analytically profile global, row, or row/wing linear detector gains."""
        mode = self.angular_objective["gain"]["mode"]
        weighted = jnp.where(valid, inverse_variance, 0.0)
        zero = jnp.asarray(0.0, dtype=signal.dtype)
        if mode == "none":
            return (
                signal,
                jnp.ones((1,), dtype=signal.dtype),
                jnp.full((1,), jnp.nan, dtype=signal.dtype),
                zero,
                zero,
            )

        if mode == "global":
            information = jnp.atleast_1d(jnp.sum(weighted * signal**2))
            rhs = jnp.atleast_1d(jnp.sum(weighted * signal * target))
            gains, standard_error, prior, smooth = self._solve_gain_series(
                information, rhs
            )
            return gains[0] * signal, gains, standard_error, prior, smooth

        def profile_rows(group_mask):
            group_weight = jnp.where(group_mask, weighted, 0.0)
            information = jnp.sum(group_weight * signal**2, axis=1)
            rhs = jnp.sum(group_weight * signal * target, axis=1)
            return self._solve_gain_series(information, rhs)

        if mode == "per_row":
            gains, standard_error, prior, smooth = profile_rows(valid)
            return gains[:, None] * signal, gains, standard_error, prior, smooth

        blue_gains, blue_error, blue_prior, blue_smooth = profile_rows(blue & valid)
        red_gains, red_error, red_prior, red_smooth = profile_rows(red & valid)
        fitted_signal = jnp.where(
            blue,
            blue_gains[:, None] * signal,
            jnp.where(red, red_gains[:, None] * signal, signal),
        )
        gains = jnp.stack((blue_gains, red_gains), axis=1)
        standard_error = jnp.stack((blue_error, red_error), axis=1)
        return (
            fitted_signal,
            gains,
            standard_error,
            blue_prior + red_prior,
            blue_smooth + red_smooth,
        )

    def _robust_weights(self, residual):
        robust = self.angular_objective["robust"]
        absolute = jnp.abs(residual)
        if robust["kind"] == "huber":
            return jnp.minimum(
                1.0, float(robust["threshold"]) / jnp.maximum(absolute, 1.0e-12)
            )
        if robust["kind"] == "student_t":
            dof = float(robust["dof"])
            return (dof + 1.0) / (dof + residual**2)
        return jnp.ones_like(residual)

    def _robust_deviance(self, residual):
        robust = self.angular_objective["robust"]
        if robust["kind"] == "huber":
            threshold = float(robust["threshold"])
            absolute = jnp.abs(residual)
            return jnp.where(
                absolute <= threshold,
                residual**2,
                2.0 * threshold * absolute - threshold**2,
            )
        if robust["kind"] == "student_t":
            dof = float(robust["dof"])
            return (dof + 1.0) * jnp.log1p(residual**2 / dof)
        return residual**2

    def _angular_data_objective(self, batch, theory, lam_axis):
        """Profile gains and return the noise-whitened ARTS data term and diagnostics."""
        raw_data = jnp.asarray(batch["e_data"])
        raw_background = jnp.broadcast_to(jnp.asarray(batch["noise_e"]), raw_data.shape)
        raw_signal = theory - raw_background
        raw_target = raw_data - raw_background
        raw_variance = self._angular_variance(batch)
        blue, red, wavelength_mask = self._angular_fit_masks(lam_axis, raw_data.shape)
        supplied_mask = jnp.broadcast_to(
            jnp.asarray(batch.get("e_mask", True), dtype=bool), raw_data.shape
        )
        valid = (
            wavelength_mask
            & supplied_mask
            & jnp.isfinite(raw_data)
            & jnp.isfinite(raw_background)
            & jnp.isfinite(raw_signal)
            & jnp.isfinite(raw_variance)
            & (raw_variance > 0.0)
        )
        # Masking only after NaN arithmetic leaves a finite scalar but can still poison
        # JAX's reverse pass through the inactive branch. Sanitize every operand before
        # gain products, division, and robust-deviance evaluation.
        data = jnp.where(jnp.isfinite(raw_data), raw_data, 0.0)
        background = jnp.where(jnp.isfinite(raw_background), raw_background, 0.0)
        signal = jnp.where(jnp.isfinite(raw_signal), raw_signal, 0.0)
        target = jnp.where(jnp.isfinite(raw_target), raw_target, 0.0)
        variance = jnp.where(
            jnp.isfinite(raw_variance) & (raw_variance > 0.0),
            raw_variance,
            1.0,
        )
        inverse_variance = jnp.where(valid, 1.0 / variance, 0.0)

        robust_weight = jnp.where(valid, 1.0, 0.0)
        iterations = (
            1
            if self.angular_objective["robust"]["kind"] == "gaussian"
            else int(self.angular_objective["robust"]["iterations"])
        )
        for _ in range(iterations):
            fitted_signal, gains, gain_standard_error, gain_prior, gain_smoothness = (
                self._profile_angular_gains(
                    signal, target, inverse_variance * robust_weight, valid, blue, red
                )
            )
            whitened = jnp.where(
                valid,
                (data - (fitted_signal + background)) / jnp.sqrt(variance),
                0.0,
            )
            robust_weight = jnp.where(valid, self._robust_weights(whitened), 0.0)

        valid_count = jnp.sum(valid)
        n_valid = jnp.maximum(valid_count, 1.0)
        deviance = jnp.where(valid, self._robust_deviance(whitened), 0.0)
        terms = {
            "data": jnp.sum(deviance) / n_valid,
            "gain_prior": gain_prior / n_valid,
            "gain_smoothness": gain_smoothness / n_valid,
        }
        total = terms["data"] + terms["gain_prior"] + terms["gain_smoothness"]
        total = jnp.where(valid_count > 0, total, jnp.nan)
        fitted_theory = fitted_signal + background
        diagnostics = {
            "whitened_residual": jnp.where(valid, whitened, jnp.nan),
            "variance": raw_variance,
            "valid_mask": valid,
            "profiled_gains": gains,
            "profiled_gain_standard_error": gain_standard_error,
            "fitted_theory": fitted_theory,
            "raw_theory": theory,
        }
        return (
            total,
            jnp.where(valid, whitened**2, 0.0),
            fitted_theory,
            terms,
            diagnostics,
        )

    def _regularization_terms(self, weights):
        """Evaluate configured priors on the positive, normalized physical EDF."""
        if not self.is_angular:
            return {}
        regularization = self.angular_objective["regularization"]
        term_names = (
            "regularization_radial",
            "regularization_angular",
            "regularization_kl",
            "regularization_density",
            "regularization_temperature",
            "regularization_momentum",
        )
        strength_names = (
            "radial_smoothness",
            "angular_smoothness",
            "kl_to_maxwellian",
            "density",
            "temperature",
            "momentum",
        )
        if not any(float(regularization[key]) > 0.0 for key in strength_names):
            return {name: jnp.asarray(0.0) for name in term_names}

        physical = weights()
        distribution = jnp.asarray(physical["electron"]["fe"])
        velocity = jnp.asarray(physical["electron"]["v"])
        dv = velocity[1] - velocity[0]
        eps = jnp.finfo(distribution.dtype).eps

        if distribution.ndim == 2:
            vx, vy = jnp.meshgrid(velocity, velocity)
            measure = dv**2
            density = jnp.sum(distribution) * measure
            mean_vx = jnp.sum(distribution * vx) * measure / jnp.maximum(density, eps)
            mean_vy = jnp.sum(distribution * vy) * measure / jnp.maximum(density, eps)
            temperature = (
                jnp.sum(distribution * ((vx - mean_vx) ** 2 + (vy - mean_vy) ** 2))
                * measure
                / jnp.maximum(density, eps)
            )
            grad_x = jnp.gradient(distribution, dv, axis=1)
            grad_y = jnp.gradient(distribution, dv, axis=0)
            radius = jnp.sqrt(vx**2 + vy**2)
            safe_radius = jnp.where(radius > 0.0, radius, 1.0)
            radial_derivative = (vx * grad_x + vy * grad_y) / safe_radius
            angular_derivative = -vy * grad_x + vx * grad_y
            norm = jnp.sum(distribution**2) * measure + eps
            radial_roughness = jnp.sum(radial_derivative**2) * measure / norm
            angular_roughness = jnp.sum(angular_derivative**2) * measure / norm
            baseline = jnp.exp(-0.5 * (vx**2 + vy**2))
            baseline = baseline / jnp.sum(baseline) / measure
            momentum_error = (mean_vx - float(regularization["momentum_target"][0])) ** 2 + (
                mean_vy - float(regularization["momentum_target"][1])
            ) ** 2
        else:
            measure = dv
            density = jnp.sum(distribution) * measure
            mean_vx = jnp.sum(distribution * velocity) * measure / jnp.maximum(density, eps)
            temperature = (
                jnp.sum(distribution * (velocity - mean_vx) ** 2) * measure / jnp.maximum(density, eps)
            )
            derivative = jnp.gradient(distribution, dv)
            norm = jnp.sum(distribution**2) * measure + eps
            radial_roughness = jnp.sum(derivative**2) * measure / norm
            angular_roughness = jnp.asarray(0.0, dtype=distribution.dtype)
            baseline = jnp.exp(-0.5 * velocity**2)
            baseline = baseline / jnp.sum(baseline) / measure
            momentum_error = (mean_vx - float(regularization["momentum_target"][0])) ** 2

        kl = jnp.sum(
            distribution
            * (jnp.log(jnp.maximum(distribution, eps)) - jnp.log(jnp.maximum(baseline, eps)))
        ) * measure
        terms = {
            "regularization_radial": float(regularization["radial_smoothness"]) * radial_roughness,
            "regularization_angular": float(regularization["angular_smoothness"]) * angular_roughness,
            "regularization_kl": float(regularization["kl_to_maxwellian"]) * kl,
            "regularization_density": float(regularization["density"])
            * (density - float(regularization["density_target"])) ** 2,
            "regularization_temperature": float(regularization["temperature"])
            * (temperature - float(regularization["temperature_target"])) ** 2,
            "regularization_momentum": float(regularization["momentum"]) * momentum_error,
        }
        return terms

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
            
            ts_params_rot = eqx.tree_at(lambda tree: tree.electron.dist_rot, ts_params, self.cfg["data"]["shot_rot"])

            if denom == []:
                #50 is added to prevent divide by zero errors but should be updated to be more rigorous, this is roughly consistent with the noise
                denom = [ThryI+50.0, ThryE+50.0]

            ThryE_rot, _, _, _ = self.ts_diag(ts_params_rot, batch["b2"])
            if self.is_angular:
                e_error1, ele_sqdev, ThryE, _, _ = self._angular_data_objective(
                    batch["b1"], ThryE, lamAxisE
                )
                e_error2, _, ThryE_rot, _, _ = self._angular_data_objective(
                    batch["b2"], ThryE_rot, lamAxisE
                )
                i_error1 = i_error2 = 0.0
                sqdev = {
                    "ele": ele_sqdev,
                    "ion": jnp.zeros_like(batch["b1"]["i_data"]),
                }
            else:
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
            if self.is_angular:
                e_error, ele_sqdev, ThryE, _, _ = self._angular_data_objective(batch, ThryE, lamAxisE)
                i_error = 0.0
                sqdev = {"ele": ele_sqdev, "ion": jnp.zeros_like(batch["i_data"])}
            else:
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

        penalty_error = sum(self._regularization_terms(ts_params).values(), jnp.asarray(0.0))
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
            pytree_weights = self.unravel_weights(weights)
            value, _ = self._loss_(pytree_weights, batch)
            return value
        else:
            return self._loss_(weights, batch)

    def residuals(self, diff_weights, static_weights, batch: Dict):
        """Least-squares residual vector for the l2 objective (with aux).

        Returns ``(residual, [ThryE, params])``. The residual is the signed, whitened
        deviation such that ``jnp.sum(residual**2)`` reproduces the l2 value of the loss
        exactly: each in-range point is scaled by ``1/sqrt(uncert * N)`` (N = number of
        in-range points for that feature, which folds the mean reduction into a plain sum of
        squares), ``ion_loss_scale`` is folded into the ion residual, and the blue/red
        averaging (factor 1/2 when both EPW sides are fit) into the electron residual.
        Out-of-range points contribute zero. The fit-range masks use the calibrated
        wavelength axes (independent of the fit parameters), so the per-feature counts N are
        constant during optimization.

        This is the single source of truth for the l2 objective: :meth:`__loss__` reduces it
        with ``sum(residual**2)`` for the gradient-based optimizers, and least-squares
        optimizers (Levenberg-Marquardt) consume the residual directly. The ``(value, aux)``
        return follows the equinox/optimistix ``has_aux`` convention so both share one call.
        Only for the non-multiplexed l2 case; other loss methods are not least squares.
        Weighting is intentionally the current diagonal ``1/uncert``; a proper (e.g.
        covariance) whitening is left for a future change.
        """
        weights = eqx.combine(static_weights, diff_weights)
        ThryE, ThryI, lamAxisE, lamAxisI = self.ts_diag(weights, batch)

        fr = self.cfg["data"]["fit_rng"]
        both_epw = self.cfg["data"]["fit_EPWb"] and self.cfg["data"]["fit_EPWr"]
        epw_factor = 2.0 if both_epw else 1.0
        parts = []

        if self.cfg["data"]["fit_IAW"]:
            mask = ((lamAxisI > fr["iaw_min"]) & (lamAxisI < fr["iaw_cf_min"])) | (
                (lamAxisI > fr["iaw_cf_max"]) & (lamAxisI < fr["iaw_max"])
            )
            scale = jnp.sqrt(self.cfg["data"]["ion_loss_scale"] / (jnp.abs(self.i_norm) * jnp.sum(mask)))
            parts.append(jnp.where(mask, (batch["i_data"] - ThryI) * scale, 0.0).ravel())

        if self.cfg["data"]["fit_EPWb"]:
            mask = (lamAxisE > fr["blue_min"]) & (lamAxisE < fr["blue_max"])
            scale = 1.0 / jnp.sqrt(jnp.abs(self.e_norm) * jnp.sum(mask) * epw_factor)
            parts.append(jnp.where(mask, (batch["e_data"] - ThryE) * scale, 0.0).ravel())

        if self.cfg["data"]["fit_EPWr"]:
            mask = (lamAxisE > fr["red_min"]) & (lamAxisE < fr["red_max"])
            scale = 1.0 / jnp.sqrt(jnp.abs(self.e_norm) * jnp.sum(mask) * epw_factor)
            parts.append(jnp.where(mask, (batch["e_data"] - ThryE) * scale, 0.0).ravel())

        return jnp.concatenate(parts), [ThryE, weights()]

    def __loss__(self, diff_weights, static_weights, batch: Dict):
        """
        Output wrapper
        """

        # For the standard (non-angular) l2 fit, derive the scalar loss from the residual
        # vector so the loss and the least-squares residual are a single, consistent source.
        # Angular fits (any shotnum) keep the existing calc_loss path: their 2D
        # reduce_ATS_to_resunit structure is not handled by the 1D residual.
        if (
            self.cfg["optimizer"]["loss_method"] == "l2"
            and "angular" not in self.cfg["other"]["extraoptions"]["spectype"]
        ):
            residual, aux = self.residuals(diff_weights, static_weights, batch)
            return jnp.sum(residual**2), aux

        weights = eqx.combine(static_weights, diff_weights)
        total_loss, sqdev, ThryE, normed_e_data, params = self.calc_loss(
            weights, batch, denom=[jnp.abs(self.i_norm), jnp.abs(self.e_norm)], reduce_func=jnp.nanmean
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

    def angular_diagnostics(self, weights, batch: Dict):
        """Recompute host-persistable ARTS objective terms and whitened residual arrays."""
        if not self.is_angular:
            raise ValueError("angular_diagnostics is only available for angular spectra")

        def evaluate_one(current_weights, current_batch, prefix=""):
            theory, _, lam_axis, _ = self.ts_diag(current_weights, current_batch)
            objective_total, _, _, data_terms, arrays = self._angular_data_objective(
                current_batch, theory, lam_axis
            )
            named_arrays = {f"{prefix}{key}": np.asarray(value) for key, value in arrays.items()}
            return objective_total, data_terms, named_arrays

        if self.multiplex_ang:
            total1, terms1, arrays1 = evaluate_one(weights, batch["b1"], "b1_")
            rotated = eqx.tree_at(
                lambda tree: tree.electron.dist_rot, weights, self.cfg["data"]["shot_rot"]
            )
            total2, terms2, arrays2 = evaluate_one(rotated, batch["b2"], "b2_")
            objective_total = total1 + total2
            terms = {key: terms1[key] + terms2[key] for key in terms1}
            arrays = arrays1 | arrays2
        else:
            objective_total, terms, arrays = evaluate_one(weights, batch)

        regularization_terms = self._regularization_terms(weights)
        terms = terms | regularization_terms
        terms["total"] = objective_total + sum(regularization_terms.values(), jnp.asarray(0.0))
        serializable_terms = {key: float(value) for key, value in terms.items()}
        return arrays, serializable_terms

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
        elif method == "covar":
            _error_ = d - t
            #here the rest of the math is done in the calc_ei_error function because the fit ranges must be applied before the matrix multiplication
        else:
            raise ValueError(f"Unknown loss method: {method!r}")
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
        
        if self.is_angular:
            return sum(self._regularization_terms(weights).values(), jnp.asarray(0.0))

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
        - For 2D velocity space, density, centered in-plane temperature, and both
          momentum components are integrated directly from the positive physical EDF.
        Args:
            params (dict): Dictionary containing model parameters, specifically the electron distribution function
                           under 'params["electron"]["fe"]'.
        Returns:
            tuple: (density_loss, temperature_loss, momentum_loss)
                - density_loss (float): Loss term enforcing normalization of the distribution.
                - temperature_loss (float): Loss term enforcing the correct temperature (second moment).
                - momentum_loss (float): Loss term enforcing zero mean velocity (first moment).
        """
        
        if self.cfg["parameters"]["electron"]["fe"]["dim"] == 2:
            physical = params() if callable(params) else params
            electron = physical["electron"]
            distribution = jnp.asarray(electron["fe"])
            velocity = jnp.asarray(
                electron.get("v", self.cfg["parameters"]["electron"]["fe"]["velocity"])
            )
            dv = velocity[1] - velocity[0]
            cell_area = dv**2
            vx, vy = jnp.meshgrid(velocity, velocity)
            density = jnp.sum(distribution) * cell_area
            safe_density = jnp.maximum(density, jnp.finfo(distribution.dtype).eps)
            mean_vx = jnp.sum(distribution * vx) * cell_area / safe_density
            mean_vy = jnp.sum(distribution * vy) * cell_area / safe_density
            thermal_second_moment = (
                jnp.sum(distribution * ((vx - mean_vx) ** 2 + (vy - mean_vy) ** 2))
                * cell_area
                / safe_density
            )
            density_loss = (density - 1.0) ** 2
            temperature_loss = (thermal_second_moment - 2.0) ** 2
            momentum_loss = mean_vx**2 + mean_vy**2
            return density_loss, temperature_loss, momentum_loss

        physical = params() if callable(params) else params
        electron = physical["electron"]
        distribution = jnp.asarray(electron["fe"])
        velocity = jnp.asarray(
            electron.get("v", self.cfg["parameters"]["electron"]["fe"]["velocity"])
        )
        dv = velocity[1] - velocity[0]
        symmetric_factor = 2.0 if self.cfg["parameters"]["electron"]["fe"].get("symmetric", False) else 1.0
        density = symmetric_factor * jnp.sum(distribution, axis=-1) * dv
        safe_density = jnp.maximum(density, jnp.finfo(distribution.dtype).eps)
        mean_velocity = symmetric_factor * jnp.sum(distribution * velocity, axis=-1) * dv / safe_density
        if symmetric_factor == 2.0:
            mean_velocity = jnp.zeros_like(mean_velocity)
        thermal_second_moment = (
            symmetric_factor
            * jnp.sum(distribution * (velocity - mean_velocity[..., None]) ** 2, axis=-1)
            * dv
            / safe_density
        )
        density_loss = jnp.mean((density - 1.0) ** 2)
        temperature_loss = jnp.mean((thermal_second_moment - 1.0) ** 2)
        momentum_loss = jnp.mean(mean_velocity**2)
        return density_loss, temperature_loss, momentum_loss
    
    def calculate_covariance_matrix(self, data):
        """
        Builds the per-lineout noise-covariance matrix used by the "covar" loss method, following the
        method described in George's RSI. For each lineout, the shot-noise standard deviation is estimated
        from `data` (self.G/self.F2 are device-specific gain/noise-factor constants set in __init__ when
        loss_method=="covar"), placed on the diagonal, convolved with the CCD spread function self.g, and a
        constant readout-noise term (self.n * self.sig_rn**2) is added on the diagonal.

        Args:
            data: array of shape (num_lineouts, num_pixels) used to estimate the shot noise. Note this is
                computed from the signal itself, not the forward model, per the comment below -- a known
                simplification, not yet the model-based noise estimate the method calls for.

        Returns:
            k_noise: array of shape (num_lineouts, num_pixels, num_pixels), the noise-covariance matrix for
            each lineout.
        """
        # Calculate noise (here it is done with the signal but it should be done with the model)
        sig_s = jnp.sqrt(data * self.G * self.F2)

        eye = jnp.eye(jnp.shape(data)[-1])
        #the n in this equation should only be included if the lineouts are summed over n pixels, if they are not summed then n should be 1
        def _slice_noise(sig_s_i):
            return jax.scipy.signal.convolve2d(eye * sig_s_i**2, self.g, mode="same") + eye * self.n * self.sig_rn**2

        k_noise = jax.vmap(_slice_noise)(sig_s)

        return k_noise
