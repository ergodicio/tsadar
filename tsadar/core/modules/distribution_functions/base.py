import os
from typing import Dict, Callable, Union
from functools import partial

from jax import numpy as jnp, vmap, Array
from jax.lax import scan
from jax.nn import sigmoid, relu
from jax.random import PRNGKey
from jax.scipy.special import gamma, sph_harm
from scipy.io import loadmat
import equinox as eqx

cwd = os.path.dirname(os.path.realpath(__file__))


def smooth1d(array, window_size):
    """
    Smooths a 1D array using a Hanning window.

    Parameters:
        array (jnp.ndarray): Input 1D array to be smoothed.
        window_size (int): Size of the Hanning window to use for smoothing.

    Returns:
        jnp.ndarray: Smoothed array of the same shape as the input.

    Notes:
        - The function uses a Hanning window for smoothing and applies convolution with 'same' mode.
        - Requires JAX's numpy module (jnp).
    """
    # Use a Hanning window
    window = jnp.hanning(window_size)
    window /= window.sum()  # Normalize
    #v1= jnp.convolve(array, window, mode="same")
    #v2= jnp.convolve(array, window, mode="valid")
    signal = jnp.r_[array[window_size - 1 : 0 : -1], array, array[-2 : -window_size - 1 : -1]]
    y = jnp.convolve(signal, window, mode="same")
    v3 = y[(window_size - 1) : -(window_size - 1)]
    return v3


def second_order_butterworth(
    signal: Array, f_sampling: int = 100, f_cutoff: int = 15, method: str = "forward_backward"
) -> Array:
    """
    Applies a second-order Butterworth filter to a signal using JAX.
    This function implements a digital Butterworth filter, similar to using
    `scipy.signal.butter` and `scipy.signal.filtfilt`, but is compatible with JAX arrays.
    It supports forward, backward, and forward-backward (zero-phase) filtering.
    Args:
        signal (Array): The input signal to be filtered.
        f_sampling (int, optional): The sampling frequency of the signal. Default is 100.
        f_cutoff (int, optional): The cutoff frequency of the filter. Default is 15.
        method (str, optional): The filtering method to use. Can be "forward", "backward",
            or "forward_backward" (default). "forward_backward" applies zero-phase filtering
            by filtering forward and then backward.
    Returns:
        Array: The filtered signal.
    Raises:
        NotImplementedError: If an unsupported method is specified.
    References:
        Adapted from https://github.com/jax-ml/jax/issues/17540

    """

    if method == "forward_backward":
        signal = second_order_butterworth(signal, f_sampling, f_cutoff, "forward")
        return second_order_butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError

    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))

    if method == "backward":
        signal = jnp.flip(signal, axis=0)

    return signal


def smooth2d(array, window_size):
    """
    Smooths a 2D array using a Hanning window of the specified size.

    Parameters:
        array (jnp.ndarray): The 2D input array to be smoothed.
        window_size (int): The size of the Hanning window to use for smoothing.

    Returns:
        jnp.ndarray: The smoothed 2D array, with the same shape as the input.

    Notes:
        - This function applies a 2D Hanning window to the input array and performs convolution.
        - The convolution is performed with 'same' mode, so the output has the same shape as the input.
        - Requires the input array and window size to be compatible with JAX (jnp).
    """
    # Use a Hanning window
    window = jnp.outer(jnp.hanning(window_size), jnp.hanning(window_size))
    window /= window.sum()  # Normalize
    return jnp.convolve(array, window, mode="same")


class DistributionFunction1V(eqx.Module):
    """
    Base class for 1D velocity distribution functions.
    This class represents a distribution function defined over a 1D velocity grid.
    It initializes the velocity grid `vx` based on the configuration provided.
    Attributes:
        vx (Array): 1D array of velocity grid points.
    Args:
        dist_cfg (Dict): Configuration dictionary containing:
            - "nvx" (int): Number of velocity grid points.
    Raises:
        NotImplementedError: If the instance is called directly, as this is an abstract base class.
    """
    vx: Array

    def __init__(self, dist_cfg: Dict):
        """
        Initializes the distribution function object.

        Args:
            dist_cfg (Dict): Configuration dictionary containing distribution parameters.
                Expected to have the key "nvx" specifying the number of velocity grid points.

        Attributes:
            vx (jnp.ndarray): 1D array of velocity grid points, evenly spaced between -vmax and vmax (excluding endpoints),
                where vmax is set to 6.0 and dv is the grid spacing.
        """
        super().__init__()
        vmax = 6.0
        dv = 2 * vmax / dist_cfg["nvx"]
        self.vx = jnp.linspace(-vmax + dv / 2, vmax - dv / 2, dist_cfg["nvx"])

    def __call__(self):
        raise NotImplementedError


class Arbitrary1V(DistributionFunction1V):
    """
    Represents a 1D arbitrary velocity distribution function.
    This class allows for the initialization, smoothing, and evaluation of a custom 1D distribution
    function. The distribution is initialized using a Super-gaussian distribtuion parameterized by a parameter `m`.
    The distribution function is defined in a 1D velocity space and can be smoothed using a second-order
    Butterworth filter.
    Attributes:
        fval (Array): The internal representation of the distribution function values.
        smooth (Callable): A smoothing function (Butterworth filter) applied to the distribution.
    Methods:
        __init__(dist_cfg):
            Initializes the distribution function with configuration parameters, sets up the initial
            distribution and smoothing filter.
        init_dlm(m):
            Initializes the distribution function using the provided shape parameter `m`.
            Returns the processed distribution array.
        get_unnormed_params():
            Returns a dictionary containing the current (unnormalized) distribution function.
        __call__():
            Applies smoothing and normalization to the distribution function and returns the
            normalized distribution array.
    """
    fval: Array
    smooth: Callable

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        self.fval = self.init_dlm(dist_cfg["params"]["init_m"])
        if dist_cfg["params"]["smooth"]:
            if dist_cfg["params"]["window"]["type"] == "butterworth":
                self.smooth = partial(
                    second_order_butterworth, f_sampling=100, f_cutoff=dist_cfg["params"]["window"]["len"], method="forward_backward"
                )
            elif dist_cfg["params"]["window"]["type"] == "hanning":
                self.smooth = partial(smooth1d, window_size=dist_cfg["params"]["window"]["len"])
            else:
                raise NotImplementedError(f"Unknown smoothing type: {dist_cfg['params']['window']['type']}")
        #self.smooth = partial(second_order_butterworth, f_sampling=100, f_cutoff=6, method="forward_backward")
        #self.smooth = partial(smooth1d, window_size=dist_cfg["nvx"] // 4)

    def init_dlm(self, m):
        # vth_x = 1.0  # jnp.sqrt(2.0)
        # alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5.0 / m))
        # cst = m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / m))
        # fdlm = cst / vth_x**3.0 * jnp.exp(-(jnp.abs(self.vx / alpha / vth_x) ** m))
        # fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])
        # fdlm = -jnp.log10(fdlm)
        x0 = jnp.sqrt(3.0 * gamma(3.0 / m) / gamma(5.0 / m))
        fdlm  = jnp.exp(-(jnp.abs(self.vx/x0) ** m))
        fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])
        fdlm = -jnp.log10(fdlm)
        #removed the divide by 7 to put edf on the 0-1 scale
        return jnp.sqrt(fdlm)

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        fval = (self.smooth(self.fval)) ** 2.0
        fval = jnp.power(10.0, -fval)
        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0])


class DLM1V(DistributionFunction1V):
    """
    DLM1V is a 1D distribution function model based on a parameterized "m" shape parameter, with support for activation functions and interpolation over precomputed distributions.
    Attributes:
        normed_m (Array): The normalized "m" parameter, possibly transformed by an activation function.
        m_scale (float): Scaling factor for the "m" parameter normalization.
        m_shift (float): Shift applied during "m" parameter normalization.
        act_fun (Callable): Activation function applied to the normalized "m" parameter.
        f_vx_m (Array): Precomputed distribution values over velocity and "m" axes.
        interpolate_f_in_m (Callable): Interpolation function for evaluating the distribution at arbitrary "m".
        m_ax (Array): Array of "m" values corresponding to precomputed distributions.
    Methods:
        __init__(dist_cfg, activate=False):
            Initializes the DLM1V distribution with configuration, normalization, and activation options.
            Loads precomputed distributions and sets up interpolation.
        get_unnormed_params():
            Returns the unnormalized "m" parameter as a dictionary.
        __call__():
            Evaluates the distribution function for the current "m" parameter, interpolating as needed,
            and returns the normalized distribution over the velocity axis.
    """
    normed_m: Array
    m_scale: float
    m_shift: float
    act_fun: Callable
    f_vx_m: Array
    interpolate_f_in_m: Callable
    m_ax: Array
    act_fun: Callable

    def __init__(self, dist_cfg, activate=False):
        """
        Initializes the distribution function object with configuration parameters and optional activation.
        Args:
            dist_cfg (dict): Configuration dictionary containing distribution parameters and settings.
            activate (bool, optional): If True and dist_cfg["active"] is True, applies activation function to parameters. Defaults to False.
        Attributes:
            m_scale (float): Scaling factor for the 'm' parameter normalization.
            m_shift (float): Shift value for the 'm' parameter normalization.
            act_fun (callable): Activation function applied to parameters.
            normed_m (float): Normalized 'm' parameter, possibly transformed by the inverse activation function.
            m_ax (jnp.ndarray): 'm' axis values of tabulated dirstribution function, used for interpolation of tabulated values.
            f_vx_m (jnp.ndarray): Distribution values from tabulated values interpolated over velocity and 'm' axes.
            interpolate_f_in_m (callable): Vectorized interpolation function for the 'm' axis.
        Raises:
            KeyError: If required keys are missing from dist_cfg.
        """
        super().__init__(dist_cfg)
        self.m_scale = 3.0
        self.m_shift = 2.0

        if activate and dist_cfg["active"]:
            inv_act_fun = lambda x: jnp.log(1e-2 + x / (1 - x + 1e-2))
            self.act_fun = sigmoid
        else:
            inv_act_fun = lambda x: x
            self.act_fun = lambda x: x

        self.normed_m = inv_act_fun((dist_cfg["params"]["m"]["val"] - self.m_shift) / self.m_scale)
        projected_distributions = loadmat(
            os.path.join(cwd, "..", "..", "..", "external", "numDistFuncs", "DLM_x_-4_-10_10_m_-1_2_5.mat")
        )["IT"]
        vx_ax = jnp.linspace(-10, 10, 200001)
        self.m_ax = jnp.linspace(2, 5, 31)
        self.f_vx_m = vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(self.vx, vx_ax, projected_distributions)
        self.interpolate_f_in_m = vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)

    def get_unnormed_params(self):
        return {"m": self.act_fun(self.normed_m) * self.m_scale + self.m_shift}

    def __call__(self):
        """
        Computes the normalized distribution function for the current parameters.
        This method applies the activation function to the normalized parameter `normed_m`, 
        scales and shifts it to obtain `unnormed_m`, and then interpolates the distribution 
        function using `interpolate_f_in_m`. The resulting distribution is normalized such 
        that its sum over the velocity axis `vx` is unity.
        Returns:
            jnp.ndarray: The normalized distribution function evaluated over the velocity grid.
        """
        unnormed_m = self.act_fun(self.normed_m) * self.m_scale + self.m_shift
        fdlm = self.interpolate_f_in_m(unnormed_m, self.m_ax, self.f_vx_m)

        return fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])
    
    def call_matte(self, unnormed_m):
        """
        Computes the normalized distribution function for the current parameters.
        This method applies the activation function to the normalized parameter `normed_m`, 
        scales and shifts it to obtain `unnormed_m`, and then interpolates the distribution 
        function using `interpolate_f_in_m`. The resulting distribution is normalized such 
        that its sum over the velocity axis `vx` is unity.
        Returns:
            jnp.ndarray: The normalized distribution function evaluated over the velocity grid.
        """
        fdlm = self.interpolate_f_in_m(unnormed_m, self.m_ax, self.f_vx_m)

        return fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0])


class DistributionFunction2V(eqx.Module):
    """
    A base class for 2D velocity distribution functions.
    This class initializes a velocity grid for use in distribution function calculations,
    centered around zero and spanning from -vmax to vmax, with a specified number of grid points.
    This velocity grid is symetric in both x and y directions.
    Parameters
    ----------
    dist_cfg : dict
        Configuration dictionary containing:
            - "nvx": int
                Number of velocity grid points along the x-axis.
    Attributes
    ----------
    vx : Array
        1D array of velocity grid points along the x-axis.
    Methods
    -------
    __call__(*args, **kwds)
        Calls the parent class's __call__ method.
    """
    vx: Array

    def __init__(self, dist_cfg):
        """
        Initializes the distribution function with a velocity grid.

        Args:
            dist_cfg (dict): Configuration dictionary containing the key "nvx", which specifies
                the number of velocity grid points.

        Attributes:
            vx (jnp.ndarray): 1D array of velocity grid points, evenly spaced between
                -vmax and vmax (excluding endpoints), where vmax is set to 6.0.
        """
        super().__init__()
        vmax = 6.0
        dvx = 2 * vmax / dist_cfg["nvx"]
        self.vx = jnp.linspace(-vmax + dvx / 2, vmax - dvx / 2, dist_cfg["nvx"])

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class Arbitrary2V(DistributionFunction2V):
    """
    Arbitrary2V is a two-velocity distribution function class that allows for arbitrary initialization and parameterization.
    Attributes:
        fval (Array): The current value of the distribution function parameters.
        learn_log (bool): If True, the logarithm (base 10) of the distribution is learned instead of the distribution itself.
    Methods:
        __init__(dist_cfg):
            Initializes the Arbitrary2V distribution with configuration parameters.
            Args:
                dist_cfg (dict): Configuration dictionary containing initialization parameters.
        init_dlm(m):
            Initializes the distribution function using a generalized Super-gaussian form.
            Args:
                m (float): Super-gaussian order for the distribution.
            Returns:
                Array: The initialized distribution function values.
        get_unnormed_params():
            Returns the current (unnormalized) distribution parameters.
            Returns:
                dict: Dictionary with the current distribution function.
        __call__():
            Computes the normalized distribution function based on current parameters.
            Returns:
                Array: The normalized distribution function.
    """
    fval: Array
    learn_log: bool

    def __init__(self, dist_cfg):
        super().__init__(dist_cfg)
        self.learn_log = dist_cfg["params"]["learn_log"]
        self.fval = self.init_dlm(dist_cfg["params"]["init_m"])

    def init_dlm(self, m):
        """
        Initialize the distribution function using the Dum-Langdon-Matte (DLM) form.
        Parameters
        ----------
        m : float
            The super-gaussian order parameter for the DLM, controlling the shape of the distribution.
        Returns
        -------
        jax.numpy.ndarray
            The square root of the (optionally log-transformed) normalized DLM distribution function
            evaluated on the velocity grid defined by `self.vx`.
        Notes
        -----
        - The function computes the DLM distribution on a 2D velocity grid using the parameter `m`.
        - The distribution is normalized such that its sum over the grid equals one.
        - If `self.learn_log` is True, the function returns the negative base-10 logarithm of the distribution before taking the square root.
        """

        vth_x = jnp.sqrt(2.0)
        alpha = jnp.sqrt(3.0 * gamma(3.0 / m) / 2.0 / gamma(5.0 / m))
        cst = m / (4.0 * jnp.pi * alpha**3.0 * gamma(3.0 / m))
        fdlm = (
            cst
            / vth_x**3.0
            * jnp.exp(-((jnp.sqrt(self.vx[:, None] ** 2.0 + self.vx[None, :] ** 2.0) / alpha / vth_x) ** m))
        )

        fdlm = fdlm / jnp.sum(fdlm) / (self.vx[1] - self.vx[0]) ** 2.0

        if self.learn_log:
            fdlm = -jnp.log10(fdlm)

        return jnp.sqrt(fdlm)

    def get_unnormed_params(self):
        return {"f": self()}

    def __call__(self):
        """
        Evaluates the normalized distribution function.
        This method computes the squared values of `self.fval`, optionally applies a logarithmic transformation
        if `self.learn_log` is True, and then normalizes the result so that the sum over the grid defined by
        `self.vx` integrates to one.
        Returns:
            jnp.ndarray: The normalized distribution function evaluated on the grid.
        """
        fval = self.fval**2.0
        if self.learn_log:
            fval = jnp.power(10.0, -fval)

        return fval / jnp.sum(fval) / (self.vx[1] - self.vx[0]) ** 2.0


def get_distribution_filter_spec(filter_spec: Dict, dist_params: Dict, replace: Union[str, bool] = True) -> Dict:
    """
    Generates a filter for seperating trainable parameters in a distribution function from static parameters, based on the distribution type and parameters.
    This function modifies the `filter_spec` dictionary to indicate which parameters of the electron distribution functions are trainable, depending on the type of distribution specified in `dist_params`. It supports several distribution types, including 'dlm', 'mx', 'arbitrary', 'arbitrary-nn', and 'sphericalharmonic'.
    Parameters:
        filter_spec (Dict): The filter specification dictionary, typically representing the structure of the model or distribution functions.
        dist_params (Dict): Dictionary containing distribution parameters, including the 'type' key and, for some types, additional nested parameters.
    Returns:
        Dict: The updated filter specification dictionary, with trainable parameters marked according to the distribution type.
    Raises:
        Warning: If the distribution type is 'mx' (Maxwellian), indicating no trainable parameters.
        NotImplementedError: If the distribution type or a specific configuration is not supported.
    """
    if dist_params["type"].casefold() == "dlm":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].normed_m, filter_spec, replace=replace
                )
        else:
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.normed_m, filter_spec, replace=replace
            )

    elif dist_params["type"].casefold() == "mx":
        raise Warning("No trainable parameters for Maxwellian distribution")

    elif dist_params["type"].casefold() == "arbitrary":
        if isinstance(filter_spec.electron.distribution_functions, list):
            num_dists = len(filter_spec.electron.distribution_functions)
            for i in range(num_dists):
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions[i].fval, filter_spec, replace=replace
                )
        else:
            filter_spec = eqx.tree_at(lambda tree: tree.electron.distribution_functions.fval, filter_spec, replace=replace)

    elif dist_params["type"].casefold() == "arbitrary-nn":
        df = filter_spec.electron.distribution_functions
        if isinstance(df, list):
            for i in range(len(df)):
                filter_spec = update_distribution_layers(filter_spec, df=df[i])
        else:
            filter_spec = update_distribution_layers(filter_spec, df=df)

    elif dist_params["type"].casefold() == "sphericalharmonic":
        if isinstance(filter_spec.electron.distribution_functions, list):
            raise NotImplementedError

        else:
            filter_spec = eqx.tree_at(
                lambda tree: tree.electron.distribution_functions.normed_m, filter_spec, replace=replace
            )
            if dist_params["params"]["flm_type"].casefold() == "arbitrary":
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].flm_mag, filter_spec, replace=replace
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].flm_sign, filter_spec, replace=replace
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].flm_mag, filter_spec, replace=replace
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].flm_sign, filter_spec, replace=replace
                )
            elif dist_params["params"]["flm_type"].casefold() == "mora-yahi":
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][0].log_10_LT, filter_spec, replace=replace
                )
                filter_spec = eqx.tree_at(
                    lambda tree: tree.electron.distribution_functions.flm[1][1].log_10_LT, filter_spec, replace=replace
                )
            elif dist_params["params"]["flm_type"].casefold() == "nn":
                for m in range(2):
                    df = filter_spec.electron.distribution_functions.flm[1][m]
                    for j in range(len(df.flm_mag.layers)):
                        filter_spec = eqx.tree_at(
                            lambda tree: tree.electron.distribution_functions.flm[1][m].flm_mag.layers[j].weight,
                            filter_spec,
                            replace=replace,
                        )
                        filter_spec = eqx.tree_at(
                            lambda tree: tree.electron.distribution_functions.flm[1][m].flm_sign.layers[j].weight,
                            filter_spec,
                            replace=replace,
                        )
            else:
                raise NotImplementedError(f"Unknown flm_type: {dist_params['flm_type']}")

    else:
        raise NotImplementedError(f"Untrainable distribution type: {dist_params['type']}")

    return filter_spec


def update_distribution_layers(filter_spec, df):
    """
    Updates the filter_spec tree by replacing the weights and biases of each layer in the distribution function neural network.
    Args:
        filter_spec: The filter specification tree to be updated, typically used with Equinox (eqx) models.
        df: An object containing the distribution function neural network (df.f_nn), which is expected to have a 'layers' attribute. Each layer should have 'linear.weight' and 'linear.bias' attributes.
    Returns:
        The updated filter_spec tree with the weights and biases of each layer replaced as specified.
    Note:
        This function assumes that each layer in df.f_nn.layers has 'linear.weight' and 'linear.bias' attributes, and that eqx.tree_at is used for functional updates.
    """
    print(df.f_nn.layers)
    for j in range(len(df.f_nn.layers)):
        if df.f_nn.layers[j].weight:
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.weight, filter_spec, replace=True)
            filter_spec = eqx.tree_at(lambda tree: df.f_nn.layers[j].linear.bias, filter_spec, replace=True)

    return filter_spec
