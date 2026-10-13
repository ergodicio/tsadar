"""Detector wavelength grids: pixel edges, grouped resolution units, and the oversampled model axis.

Every path that needs a wavelength grid builds it here from the detector pixel centers, so the
detector-bin quadrature and the oversampled uniform grid agree on where each pixel is.
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
