r"""Root-aware wavelength quadrature for unresolved collective resonances.

The detector does not sample a point spectrum.  A source wavelength contributes
to every detector bin according to the Gaussian instrument response, so the
quantity required by the forward model is

.. math::

   \bar S_j = \frac{1}{\Delta\lambda_j}
       \int S(\lambda)\left[\Phi\left(\frac{e_{j+1}-\lambda}{\sigma}\right)
       - \Phi\left(\frac{e_j-\lambda}{\sigma}\right)\right]d\lambda.

For a collective electron-plasma-wave feature, ``S`` can be much narrower than
the detector grid.  This module locates zeros of ``Re(epsilon)`` on a fixed scan,
then uses the local complex-linearized dielectric to tan-map Gauss--Legendre
nodes in coarse root panels and their neighbors. A separate, finer scan locates
roots, and coarse panels containing multiple roots are split at root midpoints.
All other panels use ordinary Gauss--Legendre quadrature.  The topology (which
panels contain roots) is
necessarily nondifferentiable, while roots, mapped nodes, and mapped weights
remain differentiable.

The public kernel handles one dielectric spectrum.  It is deliberately written
so callers can :func:`jax.vmap` it over scattering geometries.  ``terms_fn`` must
accept a one-dimensional wavelength array in nm and return
``(numerator, epsilon)``.  ``epsilon`` has the same one-dimensional shape;
``numerator`` may additionally have trailing component axes.  The intrinsic
spectrum is ``numerator / abs(epsilon)**2``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy.special import ndtr
import numpy as np


Array = jax.Array
SpectralTerms = Callable[[Array], tuple[Array, Array]]


class ResonanceQuadratureDiagnostics(NamedTuple):
    """JAX-compatible status and root information for one integration.

    ``root_overflow``, ``nonfinite``, ``zero_width``, or any invalid-input flag
    means the numerical value must not be used; ``bin_mean`` is poisoned with
    ``nan`` in that case so compiled fit code cannot consume a partial result.
    The fixed-size root arrays are padded with finite values; ``root_mask``
    identifies populated entries.
    """

    root_count: Array
    used_root_count: Array
    root_overflow: Array
    nonfinite: Array
    zero_width: Array
    invalid_bin_width: Array
    nonmonotonic_edges: Array
    invalid_irf_sigma: Array
    invalid_source_bounds: Array
    invalid_scan_phase: Array
    invalid_integration_breakpoints: Array
    roots_nm: Array
    resonance_centers_nm: Array
    resonance_half_widths_nm: Array
    root_mask: Array


class ResonanceQuadratureResult(NamedTuple):
    """Detector-bin mean densities and their numerical diagnostics."""

    bin_mean: Array
    diagnostics: ResonanceQuadratureDiagnostics


@lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def gaussian_bin_probabilities(wavelengths_nm, detector_edges_nm, irf_sigma_nm):
    """Return exact Gaussian probabilities for source wavelengths and bins.

    The result has shape ``wavelengths_nm.shape + (number_of_bins,)``.  This is
    a probability *mass* in each detector bin, not a point-sampled Gaussian.
    ``irf_sigma_nm`` must be strictly positive; the integration result exposes a
    diagnostic flag instead of silently accepting a zero or negative value.
    """

    wavelengths_nm = jnp.asarray(wavelengths_nm)
    detector_edges_nm = jnp.asarray(detector_edges_nm)
    irf_sigma_nm = jnp.asarray(irf_sigma_nm, dtype=wavelengths_nm.dtype)

    z = (detector_edges_nm - wavelengths_nm[..., None]) / irf_sigma_nm
    cdf = ndtr(z)
    return cdf[..., 1:] - cdf[..., :-1]


def _epsilon_real(terms_fn: SpectralTerms, wavelength_nm):
    """Evaluate scalar ``Re(epsilon)`` while retaining a vector callback API."""

    _, epsilon = terms_fn(jnp.reshape(wavelength_nm, (1,)))
    if epsilon.shape != (1,):
        raise ValueError("terms_fn epsilon must have the same 1-D shape as its wavelength input")
    return jnp.real(epsilon[0])


def _fixed_bisection(function, lower, upper, iterations: int):
    """Fixed-work scalar bisection used as the primal custom-root solve."""

    f_lower = function(lower)
    f_upper = function(upper)

    lower_is_root = f_lower == 0
    upper_is_root = f_upper == 0
    collapsed = jnp.where(lower_is_root, lower, jnp.where(upper_is_root, upper, lower))
    lower = jnp.where(lower_is_root | upper_is_root, collapsed, lower)
    upper = jnp.where(lower_is_root | upper_is_root, collapsed, upper)
    f_lower = jnp.where(lower_is_root | upper_is_root, 0, f_lower)

    def body(_, state):
        lo, hi, f_lo = state
        midpoint = lo + 0.5 * (hi - lo)
        f_mid = function(midpoint)
        crosses_left = (f_mid == 0) | (jnp.signbit(f_lo) != jnp.signbit(f_mid))
        next_lo = jnp.where(crosses_left, lo, midpoint)
        next_hi = jnp.where(crosses_left, midpoint, hi)
        next_f_lo = jnp.where(crosses_left, f_lo, f_mid)
        return next_lo, next_hi, next_f_lo

    lower, upper, _ = lax.fori_loop(0, iterations, body, (lower, upper, f_lower))
    return lower + 0.5 * (upper - lower)


def _implicit_bisection(function, lower, upper, iterations: int):
    """Find a root by bisection and differentiate the defining equation.

    Differentiating the bisection decisions would make a root appear locally
    fixed inside its scan panel.  ``lax.custom_root`` instead applies the
    implicit-function tangent ``dx = -df_parameter / df_x``.  Brackets and root
    topology are stopped because changing topology is genuinely discrete.
    """

    lower = lax.stop_gradient(lower)
    upper = lax.stop_gradient(upper)
    initial = lower + 0.5 * (upper - lower)

    def solve(f, _):
        return _fixed_bisection(f, lower, upper, iterations)

    def tangent_solve(linearized_function, right_hand_side):
        slope = linearized_function(jnp.ones_like(right_hand_side))
        safe_slope = jnp.where(slope != 0, slope, jnp.ones_like(slope))
        return right_hand_side / safe_slope

    return lax.custom_root(function, initial, solve, tangent_solve)


def _evaluate_weighted_density(
    terms_fn: SpectralTerms,
    wavelengths_nm,
    detector_edges_nm,
    safe_irf_sigma_nm,
):
    """Evaluate detector-weighted density and a finite flag at flat nodes."""

    wavelengths_nm = jnp.ravel(wavelengths_nm)
    numerator, epsilon = terms_fn(wavelengths_nm)
    numerator = jnp.asarray(numerator)
    epsilon = jnp.asarray(epsilon)

    if epsilon.shape != wavelengths_nm.shape:
        raise ValueError("terms_fn epsilon must have the same 1-D shape as its wavelength input")
    if numerator.ndim == 0 or numerator.shape[0] != wavelengths_nm.shape[0]:
        raise ValueError("terms_fn numerator must have wavelength as its first axis")

    denominator = jnp.abs(epsilon) ** 2
    denominator = denominator.reshape(denominator.shape + (1,) * (numerator.ndim - 1))
    density = numerator / denominator
    probabilities = gaussian_bin_probabilities(wavelengths_nm, detector_edges_nm, safe_irf_sigma_nm)
    probabilities = probabilities.reshape(
        probabilities.shape + (1,) * (numerator.ndim - 1)
    )
    weighted = probabilities * density[:, None, ...]

    finite = jnp.isfinite(wavelengths_nm) & jnp.isfinite(epsilon)
    finite = finite & jnp.all(jnp.isfinite(numerator).reshape((wavelengths_nm.size, -1)), axis=1)
    finite = finite & jnp.all(jnp.isfinite(weighted).reshape((wavelengths_nm.size, -1)), axis=1)
    safe_weighted = jnp.where(jnp.isfinite(weighted), weighted, jnp.zeros_like(weighted))
    return safe_weighted, finite


def _regular_panel_integrals(
    terms_fn: SpectralTerms,
    panel_edges_nm,
    detector_edges_nm,
    safe_irf_sigma_nm,
    order: int,
):
    """Integrate every fixed scan panel with ordinary Gauss--Legendre nodes."""

    numpy_nodes, numpy_weights = _legendre_rule(order)
    nodes = jnp.asarray(numpy_nodes, dtype=panel_edges_nm.dtype)
    weights = jnp.asarray(numpy_weights, dtype=panel_edges_nm.dtype)

    lower = panel_edges_nm[:-1]
    upper = panel_edges_nm[1:]
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    wavelengths = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    weighted, all_finite = _evaluate_weighted_density(
        terms_fn, wavelengths, detector_edges_nm, safe_irf_sigma_nm
    )

    output_shape = (wavelengths.shape[0], wavelengths.shape[1]) + weighted.shape[1:]
    weighted = weighted.reshape(output_shape)
    quadrature_weights = half_width[:, None] * weights[None, :]
    quadrature_weights = quadrature_weights.reshape(
        quadrature_weights.shape + (1,) * (weighted.ndim - 2)
    )
    panel_integrals = jnp.sum(quadrature_weights * weighted, axis=1)
    finite_panels = jnp.all(all_finite.reshape(wavelengths.shape), axis=1)
    return panel_integrals, finite_panels


def _tan_mapped_panel_integrals(
    terms_fn: SpectralTerms,
    lower_nm,
    upper_nm,
    center_nm,
    half_width_nm,
    detector_edges_nm,
    safe_irf_sigma_nm,
    order: int,
):
    """Integrate selected panels after ``lambda = c + gamma tan(t)``."""

    numpy_nodes, numpy_weights = _legendre_rule(order)
    nodes = jnp.asarray(numpy_nodes, dtype=lower_nm.dtype)
    weights = jnp.asarray(numpy_weights, dtype=lower_nm.dtype)

    t_lower = jnp.arctan((lower_nm - center_nm) / half_width_nm)
    t_upper = jnp.arctan((upper_nm - center_nm) / half_width_nm)
    t_midpoint = 0.5 * (t_lower + t_upper)
    t_half_width = 0.5 * (t_upper - t_lower)
    t_nodes = t_midpoint[:, None] + t_half_width[:, None] * nodes[None, :]

    cos_t = jnp.cos(t_nodes)
    wavelengths = center_nm[:, None] + half_width_nm[:, None] * jnp.tan(t_nodes)
    jacobian = half_width_nm[:, None] / (cos_t * cos_t)
    mapped_weights = t_half_width[:, None] * weights[None, :] * jacobian

    weighted, all_finite = _evaluate_weighted_density(
        terms_fn, wavelengths, detector_edges_nm, safe_irf_sigma_nm
    )
    output_shape = (wavelengths.shape[0], wavelengths.shape[1]) + weighted.shape[1:]
    weighted = weighted.reshape(output_shape)
    mapped_weights = mapped_weights.reshape(
        mapped_weights.shape + (1,) * (weighted.ndim - 2)
    )
    panel_integrals = jnp.sum(mapped_weights * weighted, axis=1)
    finite_panels = jnp.all(all_finite.reshape(wavelengths.shape), axis=1)
    return panel_integrals, finite_panels


def _root_crossings(real_epsilon):
    """Mark each root once, including roots exactly on scan-panel boundaries."""

    left = real_epsilon[:-1]
    right = real_epsilon[1:]
    finite = jnp.isfinite(left) & jnp.isfinite(right)
    strict_crossing = (left != 0) & (right != 0) & (jnp.signbit(left) != jnp.signbit(right))

    # A root on a shared scan edge belongs to the panel that starts there.  The
    # right endpoint is included only for the final panel, avoiding duplicates.
    left_root = left == 0
    right_root_at_end = jnp.arange(left.shape[0]) == left.shape[0] - 1
    right_root_at_end = right_root_at_end & (right == 0)
    return finite & (strict_crossing | left_root | right_root_at_end)


def _selected_root_panels(crossings, max_roots: int):
    panel_ids = jnp.arange(crossings.shape[0], dtype=jnp.int32)
    sentinel = jnp.asarray(crossings.shape[0], dtype=jnp.int32)
    candidates = jnp.where(crossings, panel_ids, sentinel)
    selected = jnp.sort(candidates)[:max_roots]
    active = selected < sentinel
    return jnp.minimum(selected, sentinel - 1), active


def _shifted_panel_edges(lower_nm, upper_nm, panel_count: int, phase):
    """Return fixed-count panel edges with a common interior phase shift."""

    uniform_edges_nm = jnp.linspace(lower_nm, upper_nm, panel_count + 1)
    panel_step_nm = (upper_nm - lower_nm) / panel_count
    return jnp.concatenate(
        (
            uniform_edges_nm[:1],
            uniform_edges_nm[1:-1] + phase * panel_step_nm,
            uniform_edges_nm[-1:],
        )
    )


def _integration_panel_edges(
    lower_nm,
    upper_nm,
    integration_panels: int,
    phase,
    integration_breakpoints_nm,
):
    """Build coarse integration edges while retaining exact static breakpoints.

    Breakpoints replace distinct interior phased edges.  Each breakpoint greedily
    removes the nearest base edge that has not already been removed; the remaining
    base edges and exact breakpoints are then sorted together.  This keeps the number
    of integration panels fixed (and therefore keeps JIT topology and memory fixed),
    while allowing arbitrarily close ordered discontinuities to share what was one
    base panel.  Candidate edges that are not strictly monotone are reported as
    invalid.
    """

    base_edges_nm = _shifted_panel_edges(lower_nm, upper_nm, integration_panels, phase)
    if integration_breakpoints_nm is None:
        return base_edges_nm, jnp.asarray(False)

    breakpoints_nm = jnp.asarray(integration_breakpoints_nm, dtype=base_edges_nm.dtype)
    if breakpoints_nm.ndim != 1:
        raise ValueError("integration_breakpoints_nm must be one-dimensional")
    breakpoint_count = breakpoints_nm.shape[0]
    if breakpoint_count == 0:
        return base_edges_nm, jnp.asarray(False)
    if breakpoint_count > integration_panels - 1:
        raise ValueError(
            "integration_breakpoints_nm has more entries than available interior "
            "integration-panel edges"
        )

    finite = jnp.all(jnp.isfinite(breakpoints_nm))
    interior = jnp.all((breakpoints_nm > lower_nm) & (breakpoints_nm < upper_nm))
    ordered = jnp.all(jnp.diff(breakpoints_nm) > 0)

    # Keep all index calculations finite even when a traced breakpoint is invalid.
    fallback_breakpoints = lower_nm + (upper_nm - lower_nm) * (
        jnp.arange(1, breakpoint_count + 1, dtype=base_edges_nm.dtype)
        / (breakpoint_count + 1)
    )
    safe_breakpoints = jnp.where(jnp.isfinite(breakpoints_nm), breakpoints_nm, fallback_breakpoints)
    interior_edges_nm = base_edges_nm[1:-1]

    def remove_nearest_edge(breakpoint_index, removed):
        distances = jnp.abs(interior_edges_nm - safe_breakpoints[breakpoint_index])
        distances = jnp.where(removed, jnp.asarray(jnp.inf, dtype=distances.dtype), distances)
        removal_index = lax.stop_gradient(jnp.argmin(distances))
        return removed.at[removal_index].set(True)

    removed = lax.fori_loop(
        0,
        breakpoint_count,
        remove_nearest_edge,
        jnp.zeros(interior_edges_nm.shape, dtype=bool),
    )
    retained_or_infinite = jnp.where(
        removed,
        jnp.asarray(jnp.inf, dtype=interior_edges_nm.dtype),
        interior_edges_nm,
    )
    candidate_interior_edges_nm = jnp.sort(
        jnp.concatenate((retained_or_infinite, safe_breakpoints))
    )[: integration_panels - 1]
    candidate_edges_nm = jnp.concatenate(
        (base_edges_nm[:1], candidate_interior_edges_nm, base_edges_nm[-1:])
    )
    candidate_monotone = jnp.all(jnp.diff(candidate_edges_nm) > 0)
    invalid = ~(finite & interior & ordered & candidate_monotone)
    return jnp.where(invalid, base_edges_nm, candidate_edges_nm), invalid


def _candidate_root_segments(
    root_panels,
    roots_nm,
    root_mask,
    valid_mapping,
    panel_edges_nm,
    neighbor_panels: int,
):
    """Build disjoint tan-mapped segments for every root/coarse-panel pair.

    Multiple fine-scan roots may occupy one coarse integration panel.  All roots
    whose coarse neighborhoods include that panel are ordered by their already
    sorted root ids, and the panel is split at pairwise root midpoints.  Thus each
    portion of a replaced panel is integrated exactly once without forcing the
    smooth background onto the fine root-scan grid.
    """

    n_panels = panel_edges_nm.shape[0] - 1

    offsets = jnp.arange(-neighbor_panels, neighbor_panels + 1, dtype=jnp.int32)
    raw_panels = root_panels[:, None] + offsets[None, :]
    in_bounds = (raw_panels >= 0) & (raw_panels < n_panels)
    candidate_valid = root_mask[:, None] & valid_mapping[:, None] & in_bounds

    candidate_panels = jnp.clip(raw_panels, 0, n_panels - 1).reshape(-1)
    candidate_valid = candidate_valid.reshape(-1)
    repeats = 2 * neighbor_panels + 1
    candidate_root_ids = jnp.repeat(jnp.arange(root_panels.shape[0], dtype=jnp.int32), repeats)
    candidate_roots_nm = jnp.repeat(roots_nm, repeats)

    same_panel = candidate_panels[:, None] == candidate_panels[None, :]
    valid_other = candidate_valid[None, :]
    previous_other = (
        same_panel
        & valid_other
        & (candidate_root_ids[None, :] < candidate_root_ids[:, None])
    )
    next_other = (
        same_panel
        & valid_other
        & (candidate_root_ids[None, :] > candidate_root_ids[:, None])
    )
    source_span_nm = panel_edges_nm[-1] - panel_edges_nm[0]
    lower_sentinel_nm = panel_edges_nm[0] - source_span_nm
    upper_sentinel_nm = panel_edges_nm[-1] + source_span_nm
    previous_root_nm = jnp.max(
        jnp.where(previous_other, candidate_roots_nm[None, :], lower_sentinel_nm), axis=1
    )
    next_root_nm = jnp.min(
        jnp.where(next_other, candidate_roots_nm[None, :], upper_sentinel_nm), axis=1
    )
    has_previous = jnp.any(previous_other, axis=1)
    has_next = jnp.any(next_other, axis=1)
    safe_previous_root_nm = jnp.where(has_previous, previous_root_nm, candidate_roots_nm)
    safe_next_root_nm = jnp.where(has_next, next_root_nm, candidate_roots_nm)

    panel_lower_nm = panel_edges_nm[candidate_panels]
    panel_upper_nm = panel_edges_nm[candidate_panels + 1]
    segment_lower_nm = jnp.where(
        has_previous,
        jnp.maximum(panel_lower_nm, 0.5 * (safe_previous_root_nm + candidate_roots_nm)),
        panel_lower_nm,
    )
    segment_upper_nm = jnp.where(
        has_next,
        jnp.minimum(panel_upper_nm, 0.5 * (candidate_roots_nm + safe_next_root_nm)),
        panel_upper_nm,
    )
    candidate_use = candidate_valid & (segment_upper_nm > segment_lower_nm)
    return candidate_panels, segment_lower_nm, segment_upper_nm, candidate_use


def integrate_detector_bins(
    terms_fn: SpectralTerms,
    detector_edges_nm,
    irf_sigma_nm,
    *,
    source_bounds_nm=None,
    root_scan_panels: int = 4096,
    integration_panels: int = 256,
    integration_breakpoints_nm=None,
    regular_order: int = 8,
    root_order: int = 32,
    max_roots: int = 16,
    neighbor_panels: int = 1,
    bisection_iterations: int = 48,
    tail_sigma: float = 6.0,
    scan_phase: float = 0.0,
) -> ResonanceQuadratureResult:
    """Integrate one possibly unresolved spectrum into detector bins.

    Args:
        terms_fn: Callable taking a 1-D wavelength array in nm and returning
            ``(numerator, epsilon)``.  The numerator's first axis and epsilon's
            sole axis must match the wavelength array.  The intrinsic spectral
            density is ``numerator / abs(epsilon)**2``.
        detector_edges_nm: Strictly increasing detector-bin edges.  Nonuniform
            bins are supported.
        irf_sigma_nm: Positive Gaussian IRF standard deviation in nm.
        source_bounds_nm: Optional two-element integration interval.  By default
            the detector range is extended by ``tail_sigma * irf_sigma_nm`` on
            each side.
        root_scan_panels: Number of fine fixed panels used only to bracket roots.
        integration_panels: Number of coarse panels used to cover the source
            interval. Root neighborhoods split these panels as needed, so more
            than one root may occupy one integration panel.
        integration_breakpoints_nm: Optional ordered one-dimensional array of
            static source-wavelength discontinuities. Each value replaces a distinct
            nearby coarse interior edge, making rectangular transmission-filter
            boundaries exact without refining the root scan or regular coverage.
        regular_order: Gauss--Legendre order outside root neighborhoods.
        root_order: Even Gauss--Legendre order used on tan-mapped panels.  An
            even order avoids sampling the mapped resonance center exactly.
        max_roots: Static root capacity.  More roots set ``root_overflow``.
        neighbor_panels: Number of coarse integration panels on either side of
            each root panel to tan-map.
        bisection_iterations: Fixed bisection work per root.
        tail_sigma: Default source-domain extension in Gaussian sigmas.
        scan_phase: Shift of every interior fine-scan and coarse-integration
            boundary as a fraction of its respective panel width, while preserving
            the source endpoints and exact integration breakpoints. Values must lie
            strictly between -1 and 1. This is primarily useful for convergence
            checks against arbitrary numerical-grid phase.

    Returns:
        :class:`ResonanceQuadratureResult`. ``bin_mean`` is a spectral density,
        so detector-bin area is ``sum(bin_mean * diff(detector_edges_nm))``.

    Notes:
        Integer controls are compile-time topology.  Close over them when using
        :func:`jax.jit`.  Root-panel selection is nondifferentiable, but root
        locations use implicit differentiation and the tan-mapped nodes and
        weights retain their full derivatives.
    """

    if root_scan_panels < 1:
        raise ValueError("root_scan_panels must be positive")
    if integration_panels < 1:
        raise ValueError("integration_panels must be positive")
    if regular_order < 1:
        raise ValueError("regular_order must be positive")
    if root_order < 2 or root_order % 2:
        raise ValueError("root_order must be a positive even integer")
    if max_roots < 1:
        raise ValueError("max_roots must be positive")
    if neighbor_panels < 0:
        raise ValueError("neighbor_panels must be nonnegative")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    if tail_sigma < 0:
        raise ValueError("tail_sigma must be nonnegative")

    detector_edges_nm = jnp.asarray(detector_edges_nm)
    if detector_edges_nm.ndim != 1 or detector_edges_nm.shape[0] < 2:
        raise ValueError("detector_edges_nm must be a 1-D array with at least two entries")
    dtype = jnp.result_type(detector_edges_nm.dtype, jnp.asarray(irf_sigma_nm).dtype, jnp.float32)
    detector_edges_nm = detector_edges_nm.astype(dtype)
    irf_sigma_nm = jnp.asarray(irf_sigma_nm, dtype=dtype)
    if irf_sigma_nm.ndim != 0:
        raise ValueError("irf_sigma_nm must be scalar")
    scan_phase = jnp.asarray(scan_phase, dtype=dtype)
    if scan_phase.ndim != 0:
        raise ValueError("scan_phase must be scalar")

    bin_widths = jnp.diff(detector_edges_nm)
    invalid_bin_width = jnp.any(bin_widths == 0)
    nonmonotonic_edges = jnp.any(bin_widths < 0)
    invalid_irf_sigma = (~jnp.isfinite(irf_sigma_nm)) | (irf_sigma_nm <= 0)
    invalid_scan_phase = (~jnp.isfinite(scan_phase)) | (jnp.abs(scan_phase) >= 1)
    safe_irf_sigma_nm = jnp.where(invalid_irf_sigma, jnp.asarray(1, dtype=dtype), irf_sigma_nm)
    safe_scan_phase = jnp.where(invalid_scan_phase, jnp.asarray(0, dtype=dtype), scan_phase)

    if source_bounds_nm is None:
        source_lower = detector_edges_nm[0] - tail_sigma * safe_irf_sigma_nm
        source_upper = detector_edges_nm[-1] + tail_sigma * safe_irf_sigma_nm
    else:
        source_bounds_nm = jnp.asarray(source_bounds_nm, dtype=dtype)
        if source_bounds_nm.shape != (2,):
            raise ValueError("source_bounds_nm must contain exactly two entries")
        source_lower, source_upper = source_bounds_nm

    invalid_source_bounds = (
        (~jnp.isfinite(source_lower))
        | (~jnp.isfinite(source_upper))
        | (source_upper <= source_lower)
    )
    safe_source_lower = jnp.where(invalid_source_bounds, detector_edges_nm[0], source_lower)
    fallback_span = jnp.maximum(jnp.abs(detector_edges_nm[-1] - detector_edges_nm[0]), 1)
    safe_source_upper = jnp.where(
        invalid_source_bounds, safe_source_lower + fallback_span, source_upper
    )
    root_scan_edges_nm = _shifted_panel_edges(
        safe_source_lower, safe_source_upper, root_scan_panels, safe_scan_phase
    )
    integration_edges_nm, invalid_integration_breakpoints = _integration_panel_edges(
        safe_source_lower,
        safe_source_upper,
        integration_panels,
        safe_scan_phase,
        integration_breakpoints_nm,
    )

    _, epsilon_scan = terms_fn(root_scan_edges_nm)
    epsilon_scan = jnp.asarray(epsilon_scan)
    if epsilon_scan.shape != root_scan_edges_nm.shape:
        raise ValueError("terms_fn epsilon must have the same 1-D shape as its wavelength input")
    real_epsilon_scan = jnp.real(epsilon_scan)
    crossings = _root_crossings(real_epsilon_scan)
    root_count = jnp.sum(crossings, dtype=jnp.int32)
    used_root_count = jnp.minimum(root_count, max_roots)
    root_overflow = root_count > max_roots
    scan_nonfinite = ~jnp.all(jnp.isfinite(epsilon_scan))

    def no_root_branch(_):
        regular, integration_finite = _regular_panel_integrals(
            terms_fn,
            integration_edges_nm,
            detector_edges_nm,
            safe_irf_sigma_nm,
            regular_order,
        )
        integral = jnp.sum(regular, axis=0)
        roots = jnp.full((max_roots,), safe_source_lower, dtype=dtype)
        centers = roots
        widths = jnp.zeros((max_roots,), dtype=dtype)
        mask = jnp.zeros((max_roots,), dtype=bool)
        return integral, ~jnp.all(integration_finite), False, roots, centers, widths, mask

    def root_branch(_):
        root_scan_panel_ids, root_mask = _selected_root_panels(crossings, max_roots)
        lower = root_scan_edges_nm[root_scan_panel_ids]
        upper = root_scan_edges_nm[root_scan_panel_ids + 1]
        dummy = lower + 0.5 * (upper - lower)

        def solve_one(lo, hi, is_active, dummy_root):
            def equation(wavelength_nm):
                residual = _epsilon_real(terms_fn, wavelength_nm)
                return jnp.where(is_active, residual, wavelength_nm - dummy_root)

            return _implicit_bisection(equation, lo, hi, bisection_iterations)

        roots = jax.vmap(solve_one)(lower, upper, root_mask, dummy)

        def epsilon_scalar(wavelength_nm):
            _, epsilon = terms_fn(jnp.reshape(wavelength_nm, (1,)))
            return epsilon[0]

        epsilon_at_root = jax.vmap(epsilon_scalar)(roots)
        epsilon_slope = jax.vmap(
            lambda root: jax.jvp(epsilon_scalar, (root,), (jnp.ones_like(root),))[1]
        )(roots)
        slope_norm = jnp.abs(epsilon_slope) ** 2
        safe_slope_norm = jnp.where(slope_norm > 0, slope_norm, jnp.ones_like(slope_norm))
        product = epsilon_at_root * jnp.conj(epsilon_slope)
        centers = roots - jnp.real(product) / safe_slope_norm
        widths = jnp.abs(jnp.imag(product)) / safe_slope_norm

        root_finite = (
            jnp.isfinite(roots)
            & jnp.isfinite(epsilon_at_root)
            & jnp.isfinite(epsilon_slope)
            & jnp.isfinite(centers)
            & jnp.isfinite(widths)
        )
        zero_width_roots = root_mask & ((slope_norm == 0) | (widths <= 0))
        valid_mapping = root_mask & root_finite & ~zero_width_roots

        regular, regular_finite = _regular_panel_integrals(
            terms_fn,
            integration_edges_nm,
            detector_edges_nm,
            safe_irf_sigma_nm,
            regular_order,
        )
        # Root-to-coarse-panel assignment is discrete topology. The solved root and
        # all mapped nodes/weights remain differentiable after that assignment.
        integration_root_panels = jnp.searchsorted(
            lax.stop_gradient(integration_edges_nm),
            lax.stop_gradient(roots),
            side="right",
        ) - 1
        integration_root_panels = jnp.clip(
            integration_root_panels, 0, integration_panels - 1
        ).astype(jnp.int32)
        (
            candidate_panels,
            candidate_lower,
            candidate_upper,
            candidate_use,
        ) = _candidate_root_segments(
            integration_root_panels,
            roots,
            root_mask,
            valid_mapping,
            integration_edges_nm,
            neighbor_panels,
        )
        repeats = 2 * neighbor_panels + 1
        candidate_centers = jnp.repeat(centers, repeats)
        candidate_widths = jnp.repeat(widths, repeats)

        # Empty/inactive candidate segments are still evaluated under JAX's static
        # control flow. Give them a well-conditioned full-panel mapping so masked
        # reverse-mode cotangents cannot encounter 0/tiny intermediate derivatives.
        fallback_lower = integration_edges_nm[candidate_panels]
        fallback_upper = integration_edges_nm[candidate_panels + 1]
        safe_candidate_lower = jnp.where(candidate_use, candidate_lower, fallback_lower)
        safe_candidate_upper = jnp.where(candidate_use, candidate_upper, fallback_upper)
        panel_midpoints = 0.5 * (safe_candidate_lower + safe_candidate_upper)
        safe_centers = jnp.where(candidate_use, candidate_centers, panel_midpoints)
        safe_widths = jnp.where(
            candidate_use,
            candidate_widths,
            safe_candidate_upper - safe_candidate_lower,
        )
        mapped, mapped_finite = _tan_mapped_panel_integrals(
            terms_fn,
            safe_candidate_lower,
            safe_candidate_upper,
            safe_centers,
            safe_widths,
            detector_edges_nm,
            safe_irf_sigma_nm,
            root_order,
        )

        component_dims = (1,) * (regular.ndim - 1)
        candidate_mask = candidate_use.reshape(candidate_use.shape + component_dims)
        mapped_by_panel = jnp.zeros_like(regular).at[candidate_panels].add(
            jnp.where(candidate_mask, mapped, jnp.zeros_like(mapped))
        )
        replaced = (
            jnp.zeros((integration_panels,), dtype=jnp.int32)
            .at[candidate_panels]
            .add(candidate_use.astype(jnp.int32))
            > 0
        )
        selected_panels = jnp.where(
            replaced.reshape(replaced.shape + component_dims), mapped_by_panel, regular
        )
        integral = jnp.sum(selected_panels, axis=0)

        regular_nonfinite = jnp.any((~replaced) & ~regular_finite)
        mapped_nonfinite = jnp.any(candidate_use & ~mapped_finite)
        integration_nonfinite = regular_nonfinite | mapped_nonfinite
        root_nonfinite = jnp.any(root_mask & ~root_finite)
        return (
            integral,
            integration_nonfinite | root_nonfinite,
            jnp.any(zero_width_roots),
            roots,
            centers,
            widths,
            root_mask,
        )

    branch_result = lax.cond(root_count > 0, root_branch, no_root_branch, operand=None)
    integral, branch_nonfinite, zero_width, roots, centers, widths, root_mask = branch_result

    safe_bin_widths = jnp.where(bin_widths > 0, bin_widths, jnp.ones_like(bin_widths))
    width_shape = safe_bin_widths.shape + (1,) * (integral.ndim - 1)
    bin_mean = integral / safe_bin_widths.reshape(width_shape)
    output_nonfinite = ~jnp.all(jnp.isfinite(bin_mean))
    nonfinite = (
        scan_nonfinite
        | branch_nonfinite
        | output_nonfinite
        | ~jnp.all(jnp.isfinite(detector_edges_nm))
    )
    fatal = (
        root_overflow
        | nonfinite
        | zero_width
        | invalid_bin_width
        | nonmonotonic_edges
        | invalid_irf_sigma
        | invalid_source_bounds
        | invalid_scan_phase
        | invalid_integration_breakpoints
    )
    bin_mean = jnp.where(fatal, jnp.full_like(bin_mean, jnp.nan), bin_mean)

    diagnostics = ResonanceQuadratureDiagnostics(
        root_count=root_count,
        used_root_count=used_root_count,
        root_overflow=root_overflow,
        nonfinite=nonfinite,
        zero_width=zero_width,
        invalid_bin_width=invalid_bin_width,
        nonmonotonic_edges=nonmonotonic_edges,
        invalid_irf_sigma=invalid_irf_sigma,
        invalid_source_bounds=invalid_source_bounds,
        invalid_scan_phase=invalid_scan_phase,
        invalid_integration_breakpoints=invalid_integration_breakpoints,
        roots_nm=roots,
        resonance_centers_nm=centers,
        resonance_half_widths_nm=widths,
        root_mask=root_mask,
    )
    return ResonanceQuadratureResult(bin_mean=bin_mean, diagnostics=diagnostics)


def raise_for_diagnostics(result: ResonanceQuadratureResult) -> None:
    """Raise a host-side exception if a completed result is not trustworthy.

    This helper is intentionally separate from the JIT-compatible kernel.  Call
    it after transferring a result to the host when fail-fast behavior is useful.
    """

    diagnostics = jax.device_get(result.diagnostics)
    failures = []
    if bool(diagnostics.root_overflow):
        failures.append(
            f"detected {int(diagnostics.root_count)} roots, capacity is {int(diagnostics.used_root_count)}"
        )
    if bool(diagnostics.nonfinite):
        failures.append("non-finite input, dielectric, integrand, or quadrature output")
    if bool(diagnostics.zero_width):
        failures.append("a detected resonance has zero local half-width")
    if bool(diagnostics.invalid_bin_width):
        failures.append("detector edges contain a zero-width bin")
    if bool(diagnostics.nonmonotonic_edges):
        failures.append("detector edges are not increasing")
    if bool(diagnostics.invalid_irf_sigma):
        failures.append("Gaussian IRF sigma must be finite and positive")
    if bool(diagnostics.invalid_source_bounds):
        failures.append("source bounds must be finite and increasing")
    if bool(diagnostics.invalid_scan_phase):
        failures.append("scan phase must be finite and strictly between -1 and 1")
    if bool(diagnostics.invalid_integration_breakpoints):
        failures.append(
            "integration breakpoints must be finite, strictly ordered, interior, and "
            "form strictly monotone integration-panel edges"
        )
    if failures:
        raise ValueError("invalid resonance quadrature: " + "; ".join(failures))


__all__ = [
    "ResonanceQuadratureDiagnostics",
    "ResonanceQuadratureResult",
    "gaussian_bin_probabilities",
    "integrate_detector_bins",
    "raise_for_diagnostics",
]
