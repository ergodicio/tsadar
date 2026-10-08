"""Bremsstrahlung continuum emission background, evaluated at the current fit iteration's plasma
conditions so it can be added directly to the synthetic Thomson spectrum during fitting. This is the
forward-model counterpart to the "brem" bg_alg in evaluate_background.get_lineout_bg, which instead
pre-fits a static background using scipy.optimize.curve_fit before the main fit. Fitting Z, Te, and ne as
part of the full spectrum (signal + background) constrains them with the entire spectral shape rather than
just the edges of the lineout, which is what makes them separable from the background's own scale/offset.
"""
from jax import numpy as jnp


def brem_spectrum(lam_nm: jnp.ndarray, Z, Te, ne, amp, offset) -> jnp.ndarray:
    """
    Bremsstrahlung continuum emission vs wavelength.

    Same functional form as the "brem" bg_alg in evaluate_background.py (1.24 = hc in keV*nm), but meant to
    be evaluated with the current fit iteration's Z/Te/ne instead of the input deck's fixed initial values.

    Args:
        lam_nm: Wavelength axis in nm.
        Z: Ion charge state.
        Te: Electron temperature in keV.
        ne: Electron density in units of 1e20 cm^-3.
        amp: Overall scale absorbing the diagnostic's calibration/geometric factors.
        offset: Additive offset.

    Returns:
        Bremsstrahlung emission evaluated at lam_nm, broadcast against Z/Te/ne/amp/offset.
    """
    return 10**8 * Z * ne**2 / Te**0.5 / lam_nm**2 * jnp.exp(-1.24 / (lam_nm * Te)) * amp + offset
