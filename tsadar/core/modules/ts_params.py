from typing import List, Dict, Union, Callable
from collections import defaultdict

from jax import Array, numpy as jnp, tree_util as jtu
from jax.nn import sigmoid
import equinox as eqx

from .distribution_functions.base import (
    DistributionFunction1V,
    DistributionFunction2V,
    DLM1V,
    Arbitrary1V,
    Arbitrary2V,
    get_distribution_filter_spec,
)
from .distribution_functions.spherical_harmonics import SphericalHarmonics


class ElectronParams(eqx.Module):
    """
    ElectronParams encapsulates the parameters and distribution functions for electron populations
    in inverse Thomson scattering fits.
    Attributes:
        normed_Te (Array): Normalized electron temperature(s).
        normed_ne (Array): Normalized electron density(ies).
        Te_scale (float): Scaling factor for electron temperature normalization.
        Te_shift (float): Shift for electron temperature normalization.
        ne_scale (float): Scaling factor for electron density normalization.
        ne_shift (float): Shift for electron density normalization.
        distribution_functions (Union[List[DistributionFunction1V], List[DistributionFunction2V], DistributionFunction1V, DistributionFunction2V]):
            Electron distribution function(s), either 1D or 2D, possibly batched.
        batch (bool): Whether parameters are batched.
        act_funs (Dict[str, Callable]): Activation functions for parameters.
        inv_act_funs (Dict[str, Callable]): Inverse activation functions for parameters.
    Args:
        cfg (dict): Configuration dictionary specifying bounds, values, and distribution settings for parameters.
        batch_size (int): Number of batches (if batch=True).
        batch (bool, optional): Whether to use batched parameters. Defaults to True.
        activate (bool, optional): Whether to apply activation functions. Defaults to False.
    Methods:
        init_dists(dist_cfg, batch_size, batch, activate):
            Initializes the electron distribution function(s) based on configuration.
        get_unnormed_params():
            Returns a dictionary of physical (unnormalized) parameters and distribution function parameters.
        __call__():
            Returns a dictionary of physical parameters and distribution function values (and velocities).
    """
    normed_Te: Array
    normed_ne: Array
    Te_scale: float
    Te_shift: float
    ne_scale: float
    ne_shift: float
    distribution_functions: Union[
        List[DistributionFunction1V], List[DistributionFunction2V], DistributionFunction1V, DistributionFunction2V
    ]
    batch: bool
    act_funs: Dict[str, Callable]
    inv_act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        """
        Initializes the parameter module with configuration values, batch settings, and activation functions.
        Args:
            cfg (dict): Configuration dictionary containing parameter bounds and values for "Te", "ne", and "fe".
            batch_size (int): Number of samples in the batch.
            batch (bool, optional): Whether to operate in batch mode. Defaults to True.
            activate (bool, optional): Whether to apply activation functions. Defaults to False.
        Attributes:
            batch (bool): Indicates if batch mode is enabled.
            act_funs (dict): Dictionary mapping parameter names to their activation functions.
            inv_act_funs (dict): Dictionary mapping parameter names to their inverse activation functions.
            Te_scale (float): Scaling factor for "Te" parameter.
            Te_shift (float): Shift value for "Te" parameter.
            ne_scale (float): Scaling factor for "ne" parameter.
            ne_shift (float): Shift value for "ne" parameter.
            normed_Te (jnp.ndarray or float): Normalized "Te" value(s), possibly batched.
            normed_ne (jnp.ndarray or float): Normalized "ne" value(s), possibly batched.
            distribution_functions: Initialized distribution functions for "fe".
        Raises:
            KeyError: If required keys are missing in the configuration dictionary.
        """
        super().__init__()

        self.batch = batch

        self.act_funs, self.inv_act_funs = {}, {}
        for param in ["Te", "ne"]:
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])
            self.act_funs[param], self.inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)

        if batch:
            self.normed_Te = self.inv_act_funs["Te"](
                jnp.full(batch_size, (cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
            )
            self.normed_ne = self.inv_act_funs["ne"](
                jnp.full(batch_size, (cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)
            )
        else:
            self.normed_Te = self.inv_act_funs["Te"]((cfg["Te"]["val"] - self.Te_shift) / self.Te_scale)
            self.normed_ne = self.inv_act_funs["ne"]((cfg["ne"]["val"] - self.ne_shift) / self.ne_scale)

        self.distribution_functions = self.init_dists(cfg["fe"], batch_size, batch, activate)

    def init_dists(self, dist_cfg, batch_size, batch, activate):
        """
        Initializes distribution functions based on the provided configuration.
        Parameters:
            dist_cfg (dict): Configuration dictionary specifying the distribution type and dimension.
                - "dim" (int): The dimension of the distribution (1 or 2).
                - "type" (str): The type of distribution (e.g., "dlm", "mx", "arbitrary", "sph").
            batch_size (int): Number of distributions to initialize if batch mode is enabled.
            batch (bool): Whether to initialize a batch of distributions.
            activate (callable): Activation function or parameter passed to certain distribution constructors.
        Returns:
            distribution_functions: 
                - For 1D distributions:
                    - If batch is True: list of distribution function instances or callables.
                    - If batch is False: a single distribution function instance or callable.
                - For 2D distributions:
                    - Only single distribution function instance (batch mode not supported).
        Raises:
            NotImplementedError: If the specified distribution type or dimension is not supported,
                or if batch mode is requested for 2D distributions.
        """
        if dist_cfg["dim"] == 1:
            if dist_cfg["type"].casefold() == "dlm":
                if batch:
                    distribution_functions = [DLM1V(dist_cfg, activate) for _ in range(batch_size)]
                else:
                    distribution_functions = DLM1V(dist_cfg, activate)

            elif dist_cfg["type"].casefold() == "mx":
                if batch:
                    distribution_functions = [
                        lambda vx: jnp.exp(-(vx**2 / 2)) / jnp.sum(jnp.exp(-(vx**2 / 2))) / (vx[1] - vx[0])
                    ]
                else:
                    distribution_functions = (
                        lambda vx: jnp.exp(-(vx**2 / 2)) / jnp.sum(jnp.exp(-(vx**2 / 2))) / (vx[1] - vx[0])
                    )
            elif dist_cfg["type"].casefold() == "arbitrary":
                if batch:
                    distribution_functions = [Arbitrary1V(dist_cfg) for _ in range(batch_size)]
                else:
                    distribution_functions = Arbitrary1V(dist_cfg)

            else:
                raise NotImplementedError(f"Unknown 1D distribution type: {dist_cfg['type']}")
        elif dist_cfg["dim"] == 2:
            if batch:
                raise NotImplementedError(
                    "Batch mode not implemented for 2D distributions as a precautionary measure against memory issues"
                )
            else:
                if "sph" in dist_cfg["type"].casefold():
                    distribution_functions = SphericalHarmonics(dist_cfg)
                elif dist_cfg["type"].casefold() == "arbitrary":
                    distribution_functions = Arbitrary2V(dist_cfg)
                else:
                    raise NotImplementedError(f"Unknown 2D distribution type: {dist_cfg['type']}")
        else:
            raise NotImplementedError(f"Not implemented distribution dimension: {dist_cfg['dim']}")

        return distribution_functions

    def get_unnormed_params(self):
        """
        Retrieve the unnormalized parameters for the current object.
        This method collects and returns the unnormalized (physical) parameters for the object,
        including electron temperature ("Te"), electron density ("ne"), and any additional
        parameters from the associated distribution functions.
        Returns:
            dict: A dictionary containing:
                - "Te": Unnormalized electron temperature, computed by applying the activation
                  function to the normalized value, then scaling and shifting.
                - "ne": Unnormalized electron density, computed similarly to "Te".
                - Additional keys and values from the unnormalized parameters of the distribution
                  functions, either as arrays (if multiple distribution functions are present) or
                  as single values.
        Notes:
            - If `self.distribution_functions` is a list, the method aggregates parameters from
              each distribution function and stacks them into arrays.
            - If `self.distribution_functions` is a single object, its unnormalized parameters
              are included directly.
        """
        unnormed_fe_params = defaultdict(list)
        if isinstance(self.distribution_functions, list):
            for fe in self.distribution_functions:
                for k, v in fe.get_unnormed_params().items():
                    unnormed_fe_params[k].append(v)
            unnormed_fe_params = {k: jnp.array(v) for k, v in unnormed_fe_params.items()}
        else:
            unnormed_fe_params = self.distribution_functions.get_unnormed_params()

        return {
            "Te": self.act_funs["Te"](self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_funs["ne"](self.normed_ne) * self.ne_scale + self.ne_shift,
        } | unnormed_fe_params

    def __call__(self):
        physical_params = {
            "Te": self.act_funs["Te"](self.normed_Te) * self.Te_scale + self.Te_shift,
            "ne": self.act_funs["ne"](self.normed_ne) * self.ne_scale + self.ne_shift,
        }
        if self.batch:
            dist_params = {
                "fe": jnp.concatenate([df()[None, :] for df in self.distribution_functions]),
                "v": jnp.concatenate([df.vx[None, :] for df in self.distribution_functions]),
            }
        else:
            dist_params = {
                "fe": self.distribution_functions(),
                "v": self.distribution_functions.vx,
            }

        return physical_params | dist_params


class IonParams(eqx.Module):
    """
    IonParams is a module for handling ion parameter normalization, activation, and denormalization.
    Attributes:
        normed_Ti (Array): Normalized ion temperature(s).
        normed_Z (Array): Normalized ion charge state(s).
        fract (Array): Normalized ion fraction(s).
        Ti_scale (float): Scaling factor for ion temperature normalization.
        Ti_shift (float): Shift for ion temperature normalization.
        Z_scale (float): Scaling factor for ion charge normalization.
        Z_shift (float): Shift for ion charge normalization.
        A (int): Ion mass number (or array if batch).
        act_funs (Dict[str, Callable]): Dictionary of activation functions for each parameter.
        inv_act_funs (Dict[str, Callable]): Dictionary of inverse activation functions for each parameter.
    Args:
        cfg (dict): Configuration dictionary containing parameter bounds, values, and activation settings.
        batch_size (int): Number of samples in the batch.
        batch (bool, optional): Whether to use batch mode (default: True).
        activate (bool, optional): Whether to apply activation functions (default: False).
    Methods:
        get_unnormed_params():
            Returns the unnormalized (physical) parameters as a dictionary.
        __call__():
            Returns a dictionary with the denormalized and activated parameters:
                - "A": Ion mass number(s).
                - "fract": Activated and denormalized ion fraction(s).
                - "Ti": Activated and denormalized ion temperature(s).
                - "Z": Activated and denormalized ion charge state(s).
    """
    normed_Ti: Array
    normed_Z: Array
    fract: Array
    Ti_scale: float
    Ti_shift: float
    Z_scale: float
    Z_shift: float
    A: int
    act_funs: Dict[str, Callable]
    inv_act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size, batch=True, activate=False):
        """
        Initializes the parameter normalization and activation functions for Thomson scattering analysis.
        Args:
            cfg (dict): Configuration dictionary containing parameter bounds and values for "Ti", "Z", "A", and "fract".
            batch_size (int): Number of samples in the batch.
            batch (bool, optional): If True, initializes parameters as batch arrays; otherwise, as scalars. Defaults to True.
            activate (bool, optional): If True, applies activation functions to parameters. Defaults to False.
        Attributes:
            act_funs (dict): Dictionary mapping parameter names to their activation functions.
            inv_act_funs (dict): Dictionary mapping parameter names to their inverse activation functions.
            normed_Ti (jnp.ndarray or float): Normalized ion temperature(s), batch or scalar depending on `batch`.
            normed_Z (jnp.ndarray or float): Normalized ionization state(s), batch or scalar depending on `batch`.
            A (jnp.ndarray or float): Atomic mass number(s), batch or scalar depending on `batch`.
            fract (jnp.ndarray or float): Ion species fractions, batch or scalar depending on `batch`.
            Ti_scale (float): Scaling factor for "Ti" normalization.
            Ti_shift (float): Shift factor for "Ti" normalization.
            Z_scale (float): Scaling factor for "Z" normalization.
            Z_shift (float): Shift factor for "Z" normalization.
        """
        super().__init__()
        self.act_funs, self.inv_act_funs = {}, {}
        for param in ["Ti", "Z"]:
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])
            self.act_funs[param], self.inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)

        self.act_funs["fract"], self.inv_act_funs["fract"] = get_act_and_inv_act(cfg["fract"], activate)

        if batch:
            self.normed_Ti = self.inv_act_funs["Ti"](
                jnp.full(batch_size, (cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
            )
            self.normed_Z = self.inv_act_funs["Z"](
                jnp.full(batch_size, (cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
            )
            self.A = jnp.full(batch_size, cfg["A"]["val"])
            self.fract = self.inv_act_funs["fract"](jnp.full(batch_size, cfg["fract"]["val"]))
        else:
            self.normed_Ti = self.inv_act_funs["Ti"]((cfg["Ti"]["val"] - self.Ti_shift) / self.Ti_scale)
            self.normed_Z = self.inv_act_funs["Z"]((cfg["Z"]["val"] - self.Z_shift) / self.Z_scale)
            self.A = cfg["A"]["val"]
            self.fract = float(self.inv_act_funs["fract"](cfg["fract"]["val"]))

    def get_unnormed_params(self):
        return self()

    def __call__(self):
        """
        Returns a dictionary of unnormalized (physical) parameter values. The parameters are denormalized by applying the activation
        functions and scaling factors defined in the class.
        The returned dictionary contains:
            - "A": The ion mass number.
            - "fract": The ion species fraction.
            - "Ti": Ion temperature.
            - "Z": Ionization state.
        Returns:
            dict: Dictionary with keys "A", "fract", "Ti", and "Z" containing the processed values.
        """

        return {
            "A": self.A,
            "fract": self.act_funs["fract"](self.fract),
            "Ti": self.act_funs["Ti"](self.normed_Ti) * self.Ti_scale + self.Ti_shift,
            "Z": self.act_funs["Z"](self.normed_Z) * self.Z_scale + self.Z_shift,
        }


def get_act_and_inv_act(param_cfg: Dict, activate: bool):
    """
    Returns activation and inverse activation functions for a parameter based on its configuration.
    If the parameter is active (being fit) and activation is requested, returns a sigmoid activation and its inverse (logit).
    Otherwise, returns the identity function for both activation and its inverse.
    Args:
        param_cfg (Dict): Configuration dictionary for the parameter, must contain an "active" key deteriming if the parameter is being fit.
        activate (bool): Whether to use the activation function.
    Returns:
        Tuple[Callable, Callable]: A tuple containing the activation function and its inverse.
    Note:
        The inverse activation function uses a stabilized logit transformation, which may be problematic near 0 and 1.
    """

    if param_cfg["active"] and activate:
        inv_act_fun = lambda x: jnp.log(1e-2 + x / (1 - x + 1e-2))  # this is problematic near 0 and 1
        act_fun = sigmoid
    else:
        act_fun = lambda x: x
        inv_act_fun = lambda x: x

    return act_fun, inv_act_fun


class GeneralParams(eqx.Module):
    """
    GeneralParams is a module for managing and transforming normalized and unnormalized parameters
    used in Thomson scattering analysis, that dont neatly fit with the ion or electron params. It handles parameter normalization, activation functions, and
    provides utilities for converting between normalized and physical parameter values.
    Attributes:
        normed_lam (Array): Normalized probe wavelength parameter.
        normed_amp1 (Array): Normalized amplitude 1 parameter, used for the blue-shifted EPW.
        normed_amp2 (Array): Normalized amplitude 2 parameter, used for the red-shifted EPW.
        normed_amp3 (Array): Normalized amplitude 3 parameter, used for the IAW.
        normed_ne_gradient (Array): Normalized electron density gradient parameter.
        normed_Te_gradient (Array): Normalized electron temperature gradient parameter.
        normed_ud (Array): Normalized drift velocity parameter.
        normed_Va (Array): Normalized fluid velocity parameter.
        lam_scale (float): Scaling factor for wavelength.
        lam_shift (float): Shift for wavelength.
        amp1_scale (float): Scaling factor for amplitude 1.
        amp1_shift (float): Shift for amplitude 1.
        amp2_scale (float): Scaling factor for amplitude 2.
        amp2_shift (float): Shift for amplitude 2.
        amp3_scale (float): Scaling factor for amplitude 3.
        amp3_shift (float): Shift for amplitude 3.
        ne_gradient_scale (float): Scaling factor for electron density gradient.
        ne_gradient_shift (float): Shift for electron density gradient.
        Te_gradient_scale (float): Scaling factor for electron temperature gradient.
        Te_gradient_shift (float): Shift for electron temperature gradient.
        ud_scale (float): Scaling factor for drift velocity.
        ud_shift (float): Shift for drift velocity.
        Va_scale (float): Scaling factor for fluid velocity.
        Va_shift (float): Shift for fluid velocity.
        act_funs (Dict[str, Callable]): Dictionary of activation functions for each parameter.
    Args:
        cfg (dict): Configuration dictionary containing parameter bounds, values, and activation function info.
        batch_size (int): Number of samples in the batch.
        batch (bool, optional): Whether to initialize parameters as batched arrays. Defaults to True.
        activate (bool, optional): Whether to apply activation functions. Defaults to False.
    Methods:
        get_unnormed_params():
            Returns a dictionary of unnormalized (physical) parameter values.
        __call__():
            Returns a dictionary of unnormalized (physical) parameter values, applying activation functions
            and scaling/shifting as necessary.
    """
    normed_lam: Array
    normed_amp1: Array
    normed_amp2: Array
    normed_amp3: Array
    normed_ne_gradient: Array
    normed_Te_gradient: Array
    normed_ud: Array
    normed_Va: Array
    lam_scale: float
    lam_shift: float
    amp1_scale: float
    amp1_shift: float
    amp2_scale: float
    amp2_shift: float
    amp3_scale: float
    amp3_shift: float
    ne_gradient_scale: float
    ne_gradient_shift: float
    Te_gradient_scale: float
    Te_gradient_shift: float
    ud_scale: float
    ud_shift: float
    Va_scale: float
    Va_shift: float
    act_funs: Dict[str, Callable]

    def __init__(self, cfg, batch_size: int, batch=True, activate=False):
        super().__init__()

        # this is all a bit ugly but we use setattr instead of = to be able to use the for loop
        self.act_funs, inv_act_funs = {}, {}
        for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud", "Va"]:
            self.act_funs[param], inv_act_funs[param] = get_act_and_inv_act(cfg[param], activate)
            setattr(self, param + "_scale", cfg[param]["ub"] - cfg[param]["lb"])
            setattr(self, param + "_shift", cfg[param]["lb"])

        # this is where the linear and nonlinear transformations are applied i.e.
        # the rescaling and the activation function
        if batch:
            for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud", "Va"]:
                setattr(
                    self,
                    "normed_" + param,
                    inv_act_funs[param](
                        jnp.full(
                            batch_size,
                            (cfg[param]["val"] - getattr(self, param + "_shift")) / getattr(self, param + "_scale"),
                        )
                    ),
                )
        else:
            for param in ["lam", "amp1", "amp2", "amp3", "ne_gradient", "Te_gradient", "ud", "Va"]:
                setattr(
                    self,
                    "normed_" + param,
                    inv_act_funs[param](
                        (cfg[param]["val"] - getattr(self, param + "_shift")) / getattr(self, param + "_scale")
                    ),
                )

    def get_unnormed_params(self):
        return self()

    def __call__(self):
        """
        Applies the corresponding activation functions and denormalizes each parameter using its scale and shift values.
        Returns:
            dict: A dictionary containing the denormalized values for the following parameters:
                - "lam": Probe wavelength parameter.
                - "amp1": Amplitude 1 parameter.
                - "amp2": Amplitude 2 parameter.
                - "amp3": Amplitude 3 parameter.
                - "ne_gradient": Electron density gradient parameter.
                - "Te_gradient": Electron temperature gradient parameter.
                - "ud": Drift velocity parameter.
                - "Va": Fluid velocity parameter.
        """
        unnormed_lam = self.act_funs["lam"](self.normed_lam) * self.lam_scale + self.lam_shift
        unnormed_amp1 = self.act_funs["amp1"](self.normed_amp1) * self.amp1_scale + self.amp1_shift
        unnormed_amp2 = self.act_funs["amp2"](self.normed_amp2) * self.amp2_scale + self.amp2_shift
        unnormed_amp3 = self.act_funs["amp3"](self.normed_amp3) * self.amp3_scale + self.amp3_shift
        unnormed_ne_gradient = (
            self.act_funs["ne_gradient"](self.normed_ne_gradient) * self.ne_gradient_scale + self.ne_gradient_shift
        )
        unnormed_Te_gradient = (
            self.act_funs["Te_gradient"](self.normed_Te_gradient) * self.Te_gradient_scale + self.Te_gradient_shift
        )
        unnormed_ud = self.act_funs["ud"](self.normed_ud) * self.ud_scale + self.ud_shift
        unnormed_Va = self.act_funs["Va"](self.normed_Va) * self.Va_scale + self.Va_shift

        return {
            "lam": unnormed_lam,
            "amp1": unnormed_amp1,
            "amp2": unnormed_amp2,
            "amp3": unnormed_amp3,
            "ne_gradient": unnormed_ne_gradient,
            "Te_gradient": unnormed_Te_gradient,
            "ud": unnormed_ud,
            "Va": unnormed_Va,
        }


class ThomsonParams(eqx.Module):
    """
    ThomsonParams is an Equinox module that encapsulates the configuration and parameter management for a Thomson scattering fits.
    It manages electron, ion, and general parameters, providing methods for normalization, extraction, and fitting of parameters.
    Attributes:
        electron (ElectronParams): The electron parameter module.
        ions (List[IonParams]): A list of ion parameter modules, one for each ion species.
        general (GeneralParams): The general parameter module.
        param_cfg (Dict): The configuration dictionary for all parameters, specifying parameter values and activation states.
        num_params (int): Number of parameters to generate for each field, same as batch size.
        batch (bool, optional): Whether to operate in batch mode. Defaults to True.
        activate (bool, optional): Whether to activate parameters. Defaults to False.
    Methods:
        renormalize_ions(tmp_dict):
            Renormalizes the ion fractions in the provided dictionary so that their sum is 1.
            Also ensures that temperature parameters marked as "same" are synchronized across ions.
        get_unnormed_params():
            Returns a dictionary of all unnormalized parameters (electron, ions, general), with ion fractions renormalized.
        __call__():
            Returns a dictionary of all current parameters (electron, ions, general), with ion fractions renormalized.
        get_fitted_params(param_cfg):
            Only parameters marked as "active" for fitting in the configuration are included.
            Special handling is applied for certain keys referencing the distribtuion function (e.g., "m", "f", "fe", "flm").
            Returns a tuple of (fitted_params, num_params), where fitted_params is a dictionary of selected parameters,
            and num_params is the count of active fitted parameters.
    Raises:
        AssertionError: If no ion species are found in the input configuration.
    """
    electron: ElectronParams
    ions: List[IonParams]
    general: GeneralParams
    param_cfg: Dict

    def __init__(self, param_cfg, num_params: int, batch=True, activate=False):
        super().__init__()

        self.electron = ElectronParams(param_cfg["electron"], num_params, batch, activate)
        self.ions = []
        for ion_index in range(len([species for species in param_cfg.keys() if "ion" in species])):
            self.ions.append(IonParams(param_cfg[f"ion-{ion_index+1}"], num_params, batch, activate))

        assert len(self.ions) > 0, "No ion species found in input deck"
        self.general = GeneralParams(param_cfg["general"], num_params, batch, activate)
        self.param_cfg = param_cfg

    def renormalize_ions(self, tmp_dict):
        """
        Renormalizes the fractional abundances of ions in the provided dictionary so that their sum equals 1.
        For each ion, if the temperature ("Ti") is set to be the same as the first ion (as specified in the parameter configuration),
        it copies the temperature value from the first ion. Then, it sums the fractional abundances ("fract") of all ions.
        Finally, it normalizes each ion's fractional abundance by dividing by the total sum.
        Args:
            tmp_dict (dict): A dictionary containing ion parameters, where each ion is keyed as "ion-1", "ion-2", etc.,
                             and contains at least the keys "Ti" (temperature) and "fract" (fractional abundance).
        Returns:
            dict: The updated dictionary with normalized fractional abundances and synchronized temperatures where specified.
        """
        fract_sum = 0
        for ion_index in range(len(self.ions)):
            if ion_index > 0 and self.param_cfg[f"ion-{ion_index+1}"]["Ti"]["same"]:
                tmp_dict[f"ion-{ion_index+1}"]["Ti"] = tmp_dict["ion-1"]["Ti"]
            fract_sum += tmp_dict[f"ion-{ion_index+1}"]["fract"]
        for ion_index in range(len(self.ions)):
            tmp_dict[f"ion-{ion_index+1}"]["fract"] /= fract_sum

        return tmp_dict
    
    def set_fe_to_matte(self, tmp_dict):
        """
        I looked into putting this in the distribution function definition but since it need information from the electron and all ions it had to be done last
        """
        zbar = 0
        z2bar = 0
        if "matte" in self.param_cfg['electron']['fe']['params']['m'] and self.param_cfg['electron']['fe']['params']['m']['matte']:
            for ion_index in range(len(self.ions)):
                zbar += tmp_dict[f"ion-{ion_index+1}"]["fract"]*tmp_dict[f"ion-{ion_index+1}"]["Z"]
                z2bar += tmp_dict[f"ion-{ion_index+1}"]["fract"]*tmp_dict[f"ion-{ion_index+1}"]["Z"]**2
            zeff = z2bar/zbar
            lang = 0.042*self.param_cfg['electron']['fe']['params']['m']['intens']*zeff/( tmp_dict['electron']['Te']*9)
            unnormed_m = 2+3/(1+1.66/lang**0.724)
            
            if self.electron.batch:
                 dist_params = {
                    "fe": jnp.concatenate([DLM1V.call_matte(self.electron.distribution_functions[i], unnormed_m[i])[None, :] for i in range(len(self.electron.distribution_functions))]),
                }
            else:
                dist_params = {
                    "fe": DLM1V.call_matte(self.electron.distribution_functions, tmp_dict['electron']['Te'], zeff, self.param_cfg['electron']['fe']['params']['m']['intens']),
                }
            
            tmp_dict["electron"]["fe"] = dist_params["fe"]
            if 'm' in tmp_dict['electron']:
                tmp_dict['electron']['m'] = unnormed_m


        return tmp_dict

    def get_unnormed_params(self):
        """
        Retrieve a dictionary of unnormalized parameters for the electron, general, and ion components.
        This method collects the unnormalized parameters from the electron, general, and each ion object,
        combines them into a single dictionary, and then applies ion renormalization to the result.
        Returns:
            dict: A dictionary containing unnormalized parameters for 'electron', 'general', and each ion
                  (with keys formatted as 'ion-<index>'), after applying ion renormalization.
        """
        tmp_dict = {
            "electron": self.electron.get_unnormed_params(),
            "general": self.general.get_unnormed_params(),
        } | {f"ion-{i+1}": ion.get_unnormed_params() for i, ion in enumerate(self.ions)}

        tmp_dict = self.renormalize_ions(tmp_dict)
        tmp_dict = self.set_fe_to_matte(tmp_dict)

        return tmp_dict

    def __call__(self):
        """
        Aggregates and returns a dictionary of plasma parameters.

        This method constructs a dictionary containing electron and general parameters,
        as well as parameters for each ion species. The keys are:
        - "electron": the result of self.electron()
        - "general": the result of self.general()
        - "ion-<n>": the result of each ion() in self.ions, where <n> is the 1-based index

        The resulting dictionary is then passed through self.renormalize_ions() for further processing
        before being returned.

        Returns:
            dict: A dictionary containing electron, general, and ion parameters, possibly renormalized.
        """
        tmp_dict = {"electron": self.electron(), "general": self.general()} | {
            f"ion-{i+1}": ion() for i, ion in enumerate(self.ions)
        }
        tmp_dict = self.renormalize_ions(tmp_dict)
        tmp_dict = self.set_fe_to_matte(tmp_dict)
        return tmp_dict

    def get_fitted_params(self, param_cfg):
        """
        Extracts and returns the fitted parameters based on the provided parameter configuration.
        This method iterates through the unnormalized parameters and selects those that are marked as "active" for fitting
        in the `param_cfg` dictionary. It constructs a dictionary of fitted parameters and counts the number of
        active parameters.
        Special handling is applied for keys:
            - "m": Included only if `param_cfg[k]["fe"]["active"]` is True.
            - "f", "fe", "flm": Always included. For "flm", additional keys 'fvxvy' and 'v' are set using the result
              of calling `self()`.
        Args:
            param_cfg (dict): Configuration dictionary specifying which parameters are active for fitting.
        Returns:
            tuple:
                fitted_params (dict): Dictionary containing the selected fitted parameters.
                num_params (int): The number of active fitted parameters.
        """
        param_dict = self.get_unnormed_params()
        num_params = 0
        fitted_params = {}
        for k in param_dict.keys():
            fitted_params[k] = {}
            for k2 in param_dict[k].keys():
                if k2 == "m":
                    if param_cfg[k]["fe"]["active"]:
                        fitted_params[k][k2] = param_dict[k][k2]
                        num_params += 1
                    else:
                        pass
                elif k2 in ["f", "fe", "flm"]:
                    fitted_params[k][k2] = param_dict[k][k2]
                    if k2 == 'flm':
                        temp_out = self()
                        fitted_params[k][k2]['fvxvy']=temp_out['electron']['fe']
                        fitted_params[k][k2]['v']=temp_out['electron']['v']
                    pass
                elif param_cfg[k][k2]["active"]:
                    fitted_params[k][k2] = param_dict[k][k2]
                    num_params += 1

        return fitted_params, num_params


def get_filter_spec(cfg_params: Dict, ts_params: ThomsonParams) -> Dict:
    """
    Generates a filter specification dictionary based on the provided configuration and Thomson scattering parameters.
    This function traverses the configuration parameters and updates a filter specification tree to indicate which
    parameters are active and should be included in further processing.
    Args:
        cfg_params (Dict): A dictionary containing configuration parameters for each species. Each species maps to a
            dictionary of parameter keys and their associated settings, including an "active" flag.
        ts_params (ThomsonParams): An object representing the Thomson scattering parameters, structured as a tree.
    Returns:
        Dict: A filter specification dictionary/tree with boolean values indicating which parameters are active (True)
            and should be included in further computations.
    """
    filter_spec = jtu.tree_map(lambda _: False, ts_params)
    ion_num = 0
    for species, params in cfg_params.items():
        if "ion" in species:
            ion_num += 1
        for key, _params in params.items():
            if _params["active"]:
                if key == "fe":
                    filter_spec = get_distribution_filter_spec(filter_spec, dist_params=_params)
                else:
                    nkey = f"normed_{key}"
                    if "ion" in species:
                        filter_spec = eqx.tree_at(
                            lambda tree: getattr(getattr(tree, "ions")[ion_num - 1], nkey),
                            filter_spec,
                            replace=True,
                        )
                    else:
                        filter_spec = eqx.tree_at(
                            lambda tree: getattr(getattr(tree, species), nkey),
                            filter_spec,
                            replace=True,
                        )

    return filter_spec
