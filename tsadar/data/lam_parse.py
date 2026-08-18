from jax import numpy as jnp


def lamParse(lamrang, lam, npts, fineion=True):
    """
    This function handles the calculation of wavelength grids and their associated frequency grids.
    Uses a minimum and maximum wavelength (string inputs have been depreciated).
    A boolean switch is included to allow a finer griding in the ion spectrum allowing for simultaneous resolution of
    the ion and electron frequencies (currently inoperable working on a solution 11/7/22).

    Args:
        lamrang: list with minimum and maximum wavelengths in nm in that order
        lam: probe wavelength in nm
        npts: number of wavelength points
        fineion: (Boolean) used to produce wavelength grid with finer spacing for the IAW feature, currently
            depreciated

    Returns:
        omgL: laser frequency in Hz
        omgs: array of scattered frequencies
        lamAxis: array of scattered wavelengths
        npts: number of wavelength points used

    """

    c = 2.99792458e10
    min_lam = lamrang[0]
    max_lam = lamrang[1]

    # if fineion and (min_lam < lam and max_lam > lam):
    #     lamAxis = jnp.linspace(min_lam, max_lam, npts)
    #     L = next(i for i in range(len(lamAxis)) if lamAxis[i] >= lam - 2)
    #     # print(L)
    #     # L = find(lamAxis >= lam - 2, 1, 'first');
    #     rlamAxis = lamAxis[::-1]
    #     R = jnp.abs(next(i for i in range(len(lamAxis)) if rlamAxis[i] <= lam + 2) - len(lamAxis))
    #     # R = find(lamAxis <= lam + 2, 1, 'last');
    #     V = jnp.linspace(lamAxis[L], lamAxis[R], npts)
    #     # V = linspace(lamAxis(L), lamAxis(R), npts);
    #     lamAxis = jnp.concatenate((lamAxis[0:L], V, lamAxis[R + 1 : -1]))
    #
    # else:
    lamAxis = jnp.linspace(min_lam, max_lam, npts)

    omgs = 2e7 * jnp.pi * c / lamAxis  # Scattered frequency axis(1 / sec)
    omgL = 2 * jnp.pi * 1e7 * c / lam  # laser frequency Rad / s

    return omgL, omgs, lamAxis, npts
