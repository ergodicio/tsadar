"""Detector wavelength grids: pixel edges, grouped resolution units, and the oversampled model axis.

Every path that needs a wavelength grid builds it here from the detector pixel centers, so the
detector-bin quadrature and the oversampled uniform grid agree on where each pixel is. Which of
those two an electron spectrum uses is decided here as well.

This module depends only on numpy so the data layer can import it without loading the forward model.
"""

import numpy as np


def detector_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Return pixel edges for a strictly increasing one-dimensional center axis.

    Interior edges are the midpoints between neighboring centers. The two outer
    edges use the adjacent half-spacing. This keeps calibrated plotting coordinates
    (centers) distinct from the finite detector support needed by bin quadrature.
    """

    centers = np.asarray(centers)
    if centers.ndim != 1:
        raise ValueError(f"Detector centers must be one-dimensional, got shape {centers.shape}.")
    if centers.size < 2:
        raise ValueError("At least two detector centers are required to infer outer pixel edges.")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Detector centers must all be finite.")

    spacing = np.diff(centers)
    if not np.all(spacing > 0):
        raise ValueError("Detector centers must be strictly increasing.")

    interior = centers[:-1] + 0.5 * spacing
    return np.concatenate(
        (
            centers[:1] - 0.5 * spacing[:1],
            interior,
            centers[-1:] + 0.5 * spacing[-1:],
        )
    )


def grouped_detector_edges(centers: np.ndarray, group_size: int) -> np.ndarray:
    """Return exact outer edges after averaging consecutive detector pixels.

    The last group may be shorter than ``group_size``. Selecting boundaries from
    the original pixel-edge array, rather than re-inferring them from the averaged
    centers, preserves that ragged group's true physical width.
    """

    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}.")

    centers = np.asarray(centers)
    edges = detector_edges_from_centers(centers)
    boundary_indices = np.arange(0, centers.size, int(group_size), dtype=int)
    boundary_indices = np.concatenate((boundary_indices, np.array([centers.size], dtype=int)))
    return edges[boundary_indices]


def oversampled_axis(centers: np.ndarray, points_per_pixel: int) -> np.ndarray:
    """Return the fine model axis with ``points_per_pixel`` points inside every detector pixel.

    Each pixel, bounded by its edges from :func:`detector_edges_from_centers`, is split into
    ``points_per_pixel`` equal cells and sampled at the cell midpoints. Averaging each group of
    ``points_per_pixel`` points therefore returns the middle of the pixel, which on a uniformly
    spaced axis is its center.

    The model only uses this with uniformly spaced centers: the spectral instrument response
    applied to the result assumes a uniform grid.
    """

    if (
        isinstance(points_per_pixel, bool)
        or not isinstance(points_per_pixel, (int, np.integer))
        or points_per_pixel <= 0
    ):
        raise ValueError(f"points_per_pixel must be a positive integer, got {points_per_pixel!r}.")

    edges = detector_edges_from_centers(np.asarray(centers, dtype=float))
    fractions = (np.arange(points_per_pixel) + 0.5) / points_per_pixel
    return (edges[:-1, None] + fractions * np.diff(edges)[:, None]).ravel()


def electron_spectrum_is_detector_binned(config) -> bool:
    """Whether the electron spectrum is integrated directly into detector bins by the resonance quadrature.

    Only angular data can be. A 2D electron distribution function is unless
    ``other.resonance_quadrature.enabled`` is false. A 1D one is only when that flag is set: true, or
    "auto", which the fit's input check resolves to true for full angular data and which counts as
    true here for callers that skip that check. Otherwise the spectrum is evaluated on the oversampled
    uniform grid and binned, which needs every wavelength unit to have the same width.
    """

    fe_dim = config["parameters"]["electron"]["fe"]["dim"]
    return bool(
        config["data"]["load_ele_spec"]
        and fe_dim in (1, 2)
        and config["other"]["extraoptions"]["spectype"] == "angular_full"
        and config["other"].get("resonance_quadrature", {}).get("enabled", fe_dim == 2)
    )
