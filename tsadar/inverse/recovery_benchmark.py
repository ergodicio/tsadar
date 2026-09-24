"""Numerical contracts for the ISS140 ARTS2D recovery benchmark.

EDFs are cell averages on the uniform, cell-centered production velocity grid.
The common EDF coordinates are q = sqrt(cell_area) * f, so Euclidean errors
and SVD projections measure the discrete integral of squared physical EDF error.
Only in-plane moments of the 2-V marginal are observable here.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import gamma


def config_hash(config) -> str:
    """Hash JSON data without silently stringifying unsupported objects."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def cell_width(axis) -> float:
    """Validate a uniform cell-center axis and return its cell width."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(
            "velocity axis must be a finite vector with at least two cells"
        )
    spacing = np.diff(axis)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("velocity cells must be uniform and increasing")
    return float(spacing[0])


def make_detector_mask(n_angles, n_detector, *, heldout_angles=(), heldout_wedges=()):
    """Return the training mask; wedges are half-open detector index intervals."""
    if n_angles < 1 or n_detector < 1:
        raise ValueError("detector dimensions must be positive")
    mask = np.ones((n_angles, n_detector), dtype=bool)
    for index in heldout_angles:
        if not 0 <= index < n_angles:
            raise ValueError("held-out angle index is outside the detector")
        mask[index] = False
    for lower, upper in heldout_wedges:
        if not 0 <= lower < upper <= n_detector:
            raise ValueError("held-out wedge is outside the detector")
        mask[:, lower:upper] = False
    if not mask.any() or mask.all():
        raise ValueError("the split must retain training samples and withhold samples")
    return mask


def noise_whitened_residuals(observed, predicted, sigma, mask=None):
    """Select residuals without silently dropping failed model predictions."""
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted spectra must have the same shape")
    sigma = np.broadcast_to(np.asarray(sigma), observed.shape)
    mask = np.broadcast_to(
        True if mask is None else np.asarray(mask, dtype=bool), observed.shape
    )
    if not mask.any():
        raise ValueError("the requested split has no samples")
    if not np.all(np.isfinite(observed[mask])) or not np.all(
        np.isfinite(predicted[mask])
    ):
        raise ValueError("selected observations and predictions must be finite")
    if np.any(~np.isfinite(sigma[mask])) or np.any(sigma[mask] <= 0):
        raise ValueError("noise sigma must be finite and positive")
    return (observed[mask] - predicted[mask]) / sigma[mask]


def residual_rms(residuals):
    values = np.asarray(residuals)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must be nonempty and finite")
    return float(np.sqrt(np.mean(values**2)))


def edf_moments(distribution, vx, vy):
    """Cell-integrated 2-V moments in normalized velocity and unit-mass units.

    ``current`` is the electron charge-sign proxy -integral(v*f); multiply by
    e*n_e*vTe for physical current. Pressure is integral(c_i*c_j*f), temperature
    is trace(P)/(2*density), and in-plane heat flux is integral(c*c^2*f)/2.
    No out-of-plane energy, temperature, or heat flux is inferred.
    """
    area = cell_width(vx) * cell_width(vy)
    f = np.asarray(distribution)
    if f.shape != (len(vy), len(vx)) or not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("EDF must be finite, nonnegative, and shaped (vy, vx)")
    density = float(f.sum() * area)
    if density <= 0:
        raise ValueError("EDF density must be positive")
    x, y = np.meshgrid(vx, vy)
    integrate = lambda value: float(np.sum(value * f) * area)
    ux, uy = integrate(x) / density, integrate(y) / density
    cx, cy = x - ux, y - uy
    pxx, pyy, pxy = integrate(cx**2), integrate(cy**2), integrate(cx * cy)
    eigenvalues = np.linalg.eigvalsh([[pxx, pxy], [pxy, pyy]])
    c2 = cx**2 + cy**2
    return {
        "normalization": density,
        "flow_vx": ux,
        "flow_vy": uy,
        "current_x": -density * ux,
        "current_y": -density * uy,
        "energy_in_plane": 0.5 * integrate(x**2 + y**2),
        "temperature_in_plane": (pxx + pyy) / (2 * density),
        "pressure_xx": pxx,
        "pressure_yy": pyy,
        "pressure_xy": pxy,
        "pressure_anisotropy": float(
            (eigenvalues[1] - eigenvalues[0]) / eigenvalues.sum()
        ),
        "heat_flux_x_in_plane": 0.5 * integrate(c2 * cx),
        "heat_flux_y_in_plane": 0.5 * integrate(c2 * cy),
    }


def detector_visible_null_basis(jacobian, *, rtol=1e-6, atol=1.0):
    """Thin SVD of a whitened, nuisance-projected detector Jacobian dS/dq.

    Keep s > max(atol, rtol*s_max). The complementary error includes exact null
    directions and weak directions below this declared sensitivity cutoff. Store
    only the visible basis; forming a full N_EDF by N_EDF null basis is unnecessary.
    Fisher eigenvalues are s**2. The default absolute threshold means sensitivity
    greater than one noise standard deviation for a unit L2 EDF displacement.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if (
        jacobian.ndim != 2
        or min(jacobian.shape) < 1
        or not np.all(np.isfinite(jacobian))
    ):
        raise ValueError("jacobian must be a finite, nonempty matrix")
    if not np.isfinite(rtol + atol) or min(rtol, atol) < 0:
        raise ValueError("SVD tolerances must be finite and nonnegative")
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    threshold = float(max(atol, rtol * singular_values[0]))
    rank = int(np.sum(singular_values > threshold))
    numerical_threshold = max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
    return {
        "singular_values": singular_values,
        "fisher_eigenvalues": singular_values**2,
        "rank": rank,
        "numerical_rank": int(np.sum(singular_values > numerical_threshold)),
        "threshold": threshold,
        "visible_basis": right_vectors[:rank].T,
    }


def project_gain_nuisance(jacobian, whitened_signal):
    """Schur-complement projection for an unanchored fitted global detector gain."""
    jacobian = np.asarray(jacobian, dtype=float)
    signal = np.asarray(whitened_signal, dtype=float).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[0] != signal.size:
        raise ValueError(
            "gain direction and detector Jacobian have incompatible shapes"
        )
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian and gain direction must be finite")
    norm = np.linalg.norm(signal)
    if norm == 0:
        raise ValueError("global gain is unidentifiable for a zero signal")
    direction = signal / norm
    return jacobian - np.outer(direction, direction @ jacobian)


def subspace_edf_errors(truth, reconstruction, visible_basis, *, cell_area=1.0):
    """Score the complete EDF error and its orthogonal visible/null components."""
    truth, reconstruction = np.asarray(truth), np.asarray(reconstruction)
    if truth.shape != reconstruction.shape or not np.all(
        np.isfinite(truth - reconstruction)
    ):
        raise ValueError("EDFs must be equal-shaped and finite")
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("cell area must be positive")
    error = (reconstruction - truth).ravel() * np.sqrt(cell_area)
    basis = np.asarray(visible_basis)
    if (
        basis.ndim != 2
        or basis.shape[0] != error.size
        or not np.all(np.isfinite(basis))
    ):
        raise ValueError("basis must have the common EDF dimension in its first axis")
    coefficients = basis.T @ error
    null_error = error - basis @ coefficients
    return {
        "edf_l2": float(np.linalg.norm(error)),
        "edf_visible_l2": float(np.linalg.norm(coefficients)),
        "edf_null_l2": float(np.linalg.norm(null_error)),
    }


def synthetic_truth_families(vx, vy, *, rotation_deg=27.0):
    """Smooth positive 2-V truths, sampled independently on each requested grid."""
    area = cell_width(vx) * cell_width(vy)
    x, y = np.meshgrid(vx, vy)
    theta = np.deg2rad(rotation_deg)
    parallel = x * np.cos(theta) + y * np.sin(theta)
    transverse = -x * np.sin(theta) + y * np.cos(theta)
    r2 = x**2 + y**2
    shape = 3.0
    scale = np.sqrt(2 * gamma(2 / shape) / gamma(4 / shape))
    truths = {
        "maxwellian": np.exp(-0.5 * r2),
        "elliptic": np.exp(-0.5 * ((parallel / 1.15) ** 2 + (transverse / 0.82) ** 2)),
        "supergaussian": np.exp(-((np.sqrt(r2) / scale) ** shape)),
        "skew_tail": (
            0.95 * np.exp(-0.5 * r2) / (2 * np.pi)
            + 0.05
            * np.exp(-0.5 * ((parallel - 1.7) ** 2 + transverse**2) / 1.3**2)
            / (2 * np.pi * 1.3**2)
        ),
    }
    return {name: value / (value.sum() * area) for name, value in truths.items()}


def poisson_read_observation(signal, *, seed, background=10.0, read_noise=3.0):
    """Photon counts plus additive Gaussian read noise, with known variance."""
    signal = np.asarray(signal, dtype=float)
    if np.any(signal < 0) or not np.all(np.isfinite(signal)):
        raise ValueError("photon signal must be finite and nonnegative")
    if not np.isfinite(background + read_noise) or min(background, read_noise) < 0:
        raise ValueError("background and read noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    mean = signal + background
    variance = mean + read_noise**2
    if np.any(variance <= 0):
        raise ValueError("every detector sample needs positive noise variance")
    data = rng.poisson(mean) + rng.normal(0.0, read_noise, mean.shape)
    return data, variance


def bootstrap_mean_interval(values, *, seed=0, resamples=2000):
    """Seed-level percentile 95% CI; a single run has no estimated interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap inputs must be a finite nonempty vector")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    interval = None
    if values.size > 1:
        rng = np.random.default_rng(seed)
        means = rng.choice(values, (resamples, values.size), replace=True).mean(axis=1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
    return {"n_seeds": int(values.size), "mean": float(values.mean()), "ci95": interval}
