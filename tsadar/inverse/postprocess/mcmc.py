"""Metropolis-Hastings MCMC sampler used by mcmc_postprocess.py to estimate per-lineout parameter
uncertainty and covariance, as an alternative to the Hessian/Laplace approximation in `.laplace`.

Proposals are made on the same `diff_params` leaves the optimizer fits, in the same unconstrained
(sigmoid/logit) space, so the [lb, ub] bounds are enforced by that reparametrization. Sampling the
electron distribution function ("fe") is not supported. See docs/source/mcmc.rst and
docs/tsadar_math.tex for the algorithm and its rationale.
"""
import os
import pickle
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import scipy.stats
from jax import random as jr
from mlflow.tracking import MlflowClient
from tqdm import trange

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.inverse.loss_function import LossFunction

_DEFAULTS = {
    "num_steps": 8000,
    "burn_in": 3000,
    "thin": 5,
    # burn-in chunk size (progress-bar granularity only; adaptation runs every step)
    "adapt_every": 50,
    "target_accept": 0.234,
    # RAM vanishing-gain exponent, in (0.5, 1]
    "adapt_gamma": 0.6,
    # flat logit-space proposal step, used when there is no Laplace seed
    "init_step_scale": 0.1,
    "use_laplace_seed": True,
    # multiplier on the seeded step scale used to disperse each chain's starting point (0 = start at best fit)
    "init_dispersion_factor": 0.0,
    "seed": 0,
    "save_samples": True,
    "compare_to_laplace": False,
    # chains whose mean is further than this many robust SDs (1.4826 * MAD) from the median are flagged
    "chain_outlier_mad_scale": 3.5,
    # split R-hat above which a single chain is flagged as not stationary
    "within_chain_r_hat_threshold": 1.1,
    # maximum fraction of chains that may be dropped for a lineout before it is marked unreliable
    "max_dropped_chain_fraction": 0.2,
    # sample substantially-negative-curvature parameters in their own block
    "block_gibbs": True,
    # normalized-Hessian eigenvalue at or below which a direction is assigned to the problem block
    "block_gibbs_eigval_threshold": -0.1,
    # minimum |eigenvector component| for a parameter to count as part of a flagged direction
    "block_gibbs_component_threshold": 0.3,
    # prior over the physical parameters, a key of _LOG_PRIORS
    "prior": "uniform",
}


def _uniform_log_prior(weights) -> float:
    """Uniform within each parameter's [lb, ub]."""
    return 0.0


# Log-prior densities over the physical parameters, selected by config["other"]["mcmc"]["prior"]. Each
# takes the full ThomsonParams (use get_unnormed_params() for physical values) and returns a per-lineout
# array of shape (batch_size,), or a scalar.
_LOG_PRIORS: Dict[str, Callable] = {"uniform": _uniform_log_prior}


def _log_prior_fn(config: Dict) -> Callable:
    name = config.get("other", {}).get("mcmc", {}).get("prior", _DEFAULTS["prior"])
    if name not in _LOG_PRIORS:
        raise ValueError(f"Unknown MCMC prior {name!r}; available priors: {sorted(_LOG_PRIORS)}")
    return _LOG_PRIORS[name]


def _logit_leaves(diff_params) -> List[Tuple[int, jnp.ndarray]]:
    """(index, leaf) of the active leaves sampled in logit coordinates, i.e. every "normed_" leaf. The
    index is the leaf's position in jax.tree_util.tree_leaves(diff_params)."""
    paths = jax.tree_util.tree_flatten_with_path(diff_params)[0]
    return [(i, leaf) for i, (path, leaf) in enumerate(paths) if path[-1].name.startswith("normed_")]


def _log_jacobian(diff_params) -> jnp.ndarray:
    """Per-lineout log-Jacobian of the map from the sampled logit coordinates to the physical parameters,
    up to a constant: sum over parameters of log(s) + log(1 - s), with s = sigmoid(z)."""
    total = 0.0
    for _, z in _logit_leaves(diff_params):
        total = total + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)
    return total


def _log_jacobian_curvature(diff_params) -> jnp.ndarray:
    """Hessian of -_log_jacobian with respect to the stacked leaves: a (batch_size, n_active, n_active)
    diagonal matrix with entries 2 s (1 - s) for the logit-coordinate leaves and 0 otherwise."""
    leaves = jax.tree_util.tree_leaves(diff_params)
    diag = [jnp.zeros_like(leaf) for leaf in leaves]
    for i, z in _logit_leaves(diff_params):
        s = jax.nn.sigmoid(z)
        diag[i] = 2.0 * s * (1.0 - s)
    return jax.vmap(jnp.diag)(jnp.stack(diag, axis=-1))


def _mcmc_cfg(config: Dict) -> Dict:
    """config["other"]["mcmc"] with every field defaulted. Warns about any field left at its default."""
    user_cfg = config.get("other", {}).get("mcmc", {})
    defaulted = sorted(set(_DEFAULTS) - set(user_cfg))
    if defaulted:
        warnings.warn(
            "MCMC config: no override given for field(s) "
            + ", ".join(defaulted)
            + f" under config['other']['mcmc'] -- using built-in default(s) "
            + ", ".join(f"{k}={_DEFAULTS[k]!r}" for k in defaulted)
            + ". If a deck intended to set one of these (e.g. under an old/misspelled key name), that "
            "override is being silently ignored.",
            stacklevel=2,
        )
    return {**_DEFAULTS, **user_cfg}


def check_fe_inactive(cfg_params: Dict) -> None:
    """Raises NotImplementedError if the electron distribution function is an active fit parameter --
    see the module docstring for why this sampler cannot currently handle that case."""
    if cfg_params["electron"]["fe"]["active"]:
        raise NotImplementedError(
            "MCMC sampling of the electron distribution function ('electron.fe.active: true') is not "
            "supported: its per-lineout parameters are stored as a list of separate objects rather than "
            "a single array with a batch axis (see ElectronParams.init_dists), which this sampler's "
            "vectorized random-walk kernel does not handle. Deactivate 'fe' to use MCMC uncertainty for "
            "the remaining (scalar) active parameters, or use the existing Hessian-based uncertainty "
            "(config['other']['calc_sigmas']) instead."
        )


def _broadcast_like(scale_leaf: jnp.ndarray, value_leaf: jnp.ndarray) -> jnp.ndarray:
    """Reshapes a per-lineout leaf (leading axis batch_size) to broadcast against another leaf sharing
    that same leading axis but with extra trailing dims, by appending singleton trailing dims."""
    extra = value_leaf.ndim - scale_leaf.ndim
    return scale_leaf.reshape(scale_leaf.shape + (1,) * extra)


def _propose(key: jax.Array, diff_params, step_scale: jnp.ndarray):
    """One Gaussian random-walk proposal, correlated across the active leaves of diff_params:
    proposal = current + step_scale @ z, z ~ N(0, I), with step_scale the (batch_size, n_active, n_active)
    Cholesky factor of each lineout's proposal covariance. Leaves are stacked in tree_flatten order.
    """
    leaves, treedef = jax.tree_util.tree_flatten(diff_params)
    stacked = jnp.stack(leaves, axis=-1)  # (batch_size, n_active)
    z = jr.normal(key, stacked.shape)
    delta = jnp.einsum("bij,bj->bi", step_scale, z)
    new_stacked = stacked + delta
    new_leaves = [new_stacked[..., i] for i in range(len(leaves))]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _log_posterior(loss_fn: LossFunction, diff_params, static_params, batch: Dict) -> jnp.ndarray:
    """Per-lineout log-posterior density in the sampled coordinates, up to an additive constant:
    log-likelihood + log-prior of the physical parameters + log-Jacobian of the coordinate transform."""
    weights = eqx.combine(static_params, diff_params)
    log_likelihood = -0.5 * loss_fn.neg_log_likelihood(weights, batch, per_lineout=True)
    return log_likelihood + _log_prior_fn(loss_fn.cfg)(weights) + _log_jacobian(diff_params)


def _mh_accept(key: jax.Array, diff_params, log_post: jnp.ndarray, proposal, log_post_proposal: jnp.ndarray):
    """Per-lineout Metropolis-Hastings accept/reject (symmetric proposal, so the ratio is just the
    posterior-density ratio). Returns (new_diff_params, new_log_post, accepted)."""
    u = jr.uniform(key, log_post.shape)
    accept = jnp.log(u) < (log_post_proposal - log_post)

    def _combine_leaf(cur, prop):
        return jnp.where(_broadcast_like(accept, cur), prop, cur)

    new_diff_params = jax.tree_util.tree_map(_combine_leaf, diff_params, proposal)
    new_log_post = jnp.where(accept, log_post_proposal, log_post)
    return new_diff_params, new_log_post, accept


@eqx.filter_jit
def _run_window(
    key: jax.Array,
    loss_fn: LossFunction,
    static_params,
    batch: Dict,
    diff_params,
    log_post: jnp.ndarray,
    step_scale,
    n_steps: int,
    collect: bool,
    thin: int = 1,
):
    """Runs n_steps of propose + accept/reject via jax.lax.scan at a fixed step_scale.

    When collect is True the returned samples are already thinned: an inner uncollected scan of `thin`
    steps is nested in an outer scan that keeps only the last state of each group, so the full unthinned
    history is never held in memory. Requires n_steps % thin == 0.

    Returns (diff_params, log_post, accept_count, collected); collected is None when collect is False.
    """

    def _single_step(carry, key_i):
        diff_params, log_post, accept_count = carry
        k_prop, k_acc = jr.split(key_i)
        proposal = _propose(k_prop, diff_params, step_scale)
        log_post_proposal = _log_posterior(loss_fn, proposal, static_params, batch)
        diff_params, log_post, accept = _mh_accept(k_acc, diff_params, log_post, proposal, log_post_proposal)
        accept_count = accept_count + accept.astype(jnp.int32)
        return (diff_params, log_post, accept_count), diff_params

    init_accept_count = jnp.zeros_like(log_post, dtype=jnp.int32)

    if not collect or thin <= 1:
        keys = jr.split(key, n_steps)

        def _body(carry, key_i):
            carry, diff_params = _single_step(carry, key_i)
            return carry, (diff_params if collect else None)

        (diff_params, log_post, accept_count), collected = jax.lax.scan(
            _body, (diff_params, log_post, init_accept_count), keys
        )
        return diff_params, log_post, accept_count, collected

    assert n_steps % thin == 0, f"_run_window: n_steps ({n_steps}) must be a multiple of thin ({thin})"
    n_groups = n_steps // thin
    group_keys = jr.split(key, n_groups)

    def _group_body(carry, group_key):
        carry, _ = jax.lax.scan(_single_step, carry, jr.split(group_key, thin))
        diff_params, _, _ = carry
        return carry, diff_params  # collect once per thin-sized group, not once per raw step

    (diff_params, log_post, accept_count), collected = jax.lax.scan(
        _group_body, (diff_params, log_post, init_accept_count), group_keys
    )
    return diff_params, log_post, accept_count, collected


def _ram_update(
    step_scale: jnp.ndarray,
    z: jnp.ndarray,
    alpha: jnp.ndarray,
    target_accept: float,
    step_index: jnp.ndarray,
    adapt_gamma: float,
    n_active: int,
) -> jnp.ndarray:
    """One Robust Adaptive Metropolis (Vihola 2012) rank-one update of the proposal Cholesky factor:

        Sigma_i = S_{i-1} (I + eta_i * (alpha_i - target_accept) * z z^T / ||z||^2) S_{i-1}^T
        S_i = cholesky(Sigma_i),    eta_i = min(1, n_active * step_index^-adapt_gamma)

    Args:
        step_scale: S_{i-1}, (batch_size, n_active, n_active) Cholesky factor.
        z: this step's whitened proposal draw, (batch_size, n_active).
        alpha: this step's MH acceptance probability per lineout, (batch_size,), not the 0/1 outcome.
        target_accept: target acceptance rate.
        step_index: number of adaptation steps taken so far including this one (>= 1), counted across
            the whole burn-in.
        adapt_gamma: gain decay exponent.
        n_active: dimension of the proposal.

    Returns:
        S_i: updated (batch_size, n_active, n_active) Cholesky factor.
    """
    eta = jnp.minimum(1.0, n_active * (step_index ** (-adapt_gamma)))
    z_normsq = jnp.sum(z * z, axis=-1)
    coef = eta * (alpha - target_accept) / jnp.maximum(z_normsq, 1e-300)
    eye_n = jnp.broadcast_to(jnp.eye(n_active), step_scale.shape)
    inner = eye_n + coef[:, None, None] * jnp.einsum("bi,bj->bij", z, z)
    sigma_new = jnp.einsum("bij,bjk,blk->bil", step_scale, inner, step_scale)
    sigma_new = 0.5 * (sigma_new + jnp.swapaxes(sigma_new, -1, -2))  # symmetrize away float roundoff
    return jnp.linalg.cholesky(sigma_new)


@eqx.filter_jit
def _run_ram_window(
    key: jax.Array,
    loss_fn: LossFunction,
    static_params,
    batch: Dict,
    diff_params,
    log_post: jnp.ndarray,
    step_scale: jnp.ndarray,
    n_steps: int,
    step_offset: jnp.ndarray,
    target_accept: float,
    adapt_gamma: float,
):
    """Runs n_steps of propose + accept/reject via jax.lax.scan, adapting step_scale every step with
    _ram_update.

    step_offset is the number of adaptation steps taken before this call, so the gain schedule continues
    across chunks. It must be a jnp array; a Python number would retrigger compilation for every chunk.

    Returns (diff_params, log_post, step_scale, accept_count).
    """

    def _single_step(carry, inputs):
        diff_params, log_post, step_scale = carry
        key_i, step_index = inputs
        k_prop, k_acc = jr.split(key_i)

        leaves, treedef = jax.tree_util.tree_flatten(diff_params)
        n_active = len(leaves)
        stacked = jnp.stack(leaves, axis=-1)  # (batch_size, n_active)
        z = jr.normal(k_prop, stacked.shape)
        delta = jnp.einsum("bij,bj->bi", step_scale, z)
        new_stacked = stacked + delta
        proposal = jax.tree_util.tree_unflatten(treedef, [new_stacked[..., i] for i in range(n_active)])

        log_post_proposal = _log_posterior(loss_fn, proposal, static_params, batch)
        log_ratio = log_post_proposal - log_post
        # RAM adapts from the acceptance probability, not the realized accept/reject
        alpha = jnp.exp(jnp.minimum(log_ratio, 0.0))

        u = jr.uniform(k_acc, log_post.shape)
        accept = jnp.log(u) < log_ratio
        accept_b = _broadcast_like(accept, stacked)
        new_stacked = jnp.where(accept_b, new_stacked, stacked)
        new_diff_params = jax.tree_util.tree_unflatten(treedef, [new_stacked[..., i] for i in range(n_active)])
        new_log_post = jnp.where(accept, log_post_proposal, log_post)

        new_step_scale = _ram_update(step_scale, z, alpha, target_accept, step_index, adapt_gamma, n_active)

        return (new_diff_params, new_log_post, new_step_scale), accept

    keys = jr.split(key, n_steps)
    step_indices = step_offset + 1.0 + jnp.arange(n_steps, dtype=step_offset.dtype)
    (diff_params, log_post, step_scale), accept_trace = jax.lax.scan(
        _single_step, (diff_params, log_post, step_scale), (keys, step_indices)
    )
    accept_count = jnp.sum(accept_trace.astype(jnp.int32), axis=0)
    return diff_params, log_post, step_scale, accept_count


# Block Metropolis-within-Gibbs: the active leaves are split into a "well-conditioned" block and a
# "problem" block (leaves in a substantially-negative-curvature eigendirection of the normalized Hessian),
# and each block is proposed, accepted/rejected and RAM-adapted separately within one outer step.
# See docs/tsadar_math.tex (Block Metropolis-within-Gibbs).


def _detect_problem_leaves(H: jnp.ndarray, eigval_threshold: float, component_threshold: float) -> np.ndarray:
    """Flags the active leaves that take part in a negative-curvature eigendirection of the normalized
    Hessian for any lineout in H.

    Args:
        H: per-lineout Hessian, (batch_size, n_active, n_active).
        eigval_threshold: eigenvalues at or below this are flagged.
        component_threshold: minimum |eigenvector component| for a leaf to be part of a flagged direction.

    Returns:
        numpy boolean array of shape (n_active,).
    """
    h_norm, _ = _normalize_hessian(H)
    h_norm = 0.5 * (h_norm + jnp.swapaxes(h_norm, -1, -2))  # symmetrize away float roundoff before eigh
    eigvals, eigvecs = jnp.linalg.eigh(h_norm)  # ascending; (batch_size, n_eig), (batch_size, n_active, n_eig)
    bad = eigvals <= eigval_threshold  # (batch_size, n_eig)
    dominant = jnp.abs(eigvecs) >= component_threshold  # (batch_size, n_active, n_eig)
    flagged_per_lineout = jnp.any(bad[:, None, :] & dominant, axis=-1)  # (batch_size, n_active)
    return np.asarray(jnp.any(flagged_per_lineout, axis=0))  # (n_active,)


def _block_indices_from_hessians(
    hessians: List[jnp.ndarray], eigval_threshold: float, component_threshold: float
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Unions _detect_problem_leaves across every fit-batch into one partition shared by the whole run.
    The partition must be the same for every fit-batch because they are vmapped together.

    Args:
        hessians: one (batch_size, n_active, n_active) Hessian per fit-batch.
        eigval_threshold: see _detect_problem_leaves.
        component_threshold: see _detect_problem_leaves.

    Returns:
        (well_idx, problem_idx): disjoint, sorted tuples of leaf indices (tree_flatten order) covering
        every active leaf. problem_idx is empty when nothing is flagged or when everything is.
    """
    if not hessians:
        return (), ()
    n_active = hessians[0].shape[-1]
    flagged = np.zeros(n_active, dtype=bool)
    for H in hessians:
        flagged |= _detect_problem_leaves(H, eigval_threshold, component_threshold)
    problem_idx = tuple(int(i) for i in np.nonzero(flagged)[0])
    well_idx = tuple(int(i) for i in np.nonzero(~flagged)[0])
    if not problem_idx or not well_idx:
        return tuple(range(n_active)), ()
    return well_idx, problem_idx


def _block_step_scale(H: jnp.ndarray, block_idx: Tuple[int, ...]) -> jnp.ndarray:
    """Regularized Cholesky factor for the Hessian sub-block block_idx, scaled by 2.38/sqrt(block size)."""
    idx = list(block_idx)
    block_n = len(idx)
    H_block = H[:, idx, :][:, :, idx]  # (batch_size, block_n, block_n)
    rr_factor = 2.38 / jnp.sqrt(float(block_n))
    return _regularized_proposal_cholesky(H_block, rr_factor)


def _combine_block_step_scales(
    step_scale_ok: jnp.ndarray,
    well_idx: Tuple[int, ...],
    step_scale_problem: jnp.ndarray,
    problem_idx: Tuple[int, ...],
    n_active: int,
) -> jnp.ndarray:
    """Embeds the two blocks' Cholesky factors into one (batch_size, n_active, n_active) block-diagonal
    matrix. Only used to draw the initial starting-point dispersion of a blocked run.
    """
    batch_size = step_scale_ok.shape[0]
    combined = jnp.zeros((batch_size, n_active, n_active), dtype=step_scale_ok.dtype)
    well = jnp.array(well_idx)
    problem = jnp.array(problem_idx)
    combined = combined.at[:, well[:, None], well[None, :]].set(step_scale_ok)
    combined = combined.at[:, problem[:, None], problem[None, :]].set(step_scale_problem)
    return combined


def _block_step(
    key_i: jax.Array,
    loss_fn: LossFunction,
    static_params,
    batch: Dict,
    diff_params,
    log_post: jnp.ndarray,
    step_scale_block: jnp.ndarray,
    block_idx: Tuple[int, ...],
    adapt: bool,
    step_index: Optional[jnp.ndarray] = None,
    target_accept: Optional[float] = None,
    adapt_gamma: Optional[float] = None,
):
    """One Metropolis-within-Gibbs sub-step on the leaves in block_idx; all other leaves are held fixed.
    When adapt is True, step_scale_block is also updated with _ram_update using this sub-step's acceptance.

    Returns:
        (new_diff_params, new_log_post, new_step_scale_block, accept); new_step_scale_block is None when
        adapt is False.
    """
    k_prop, k_acc = jr.split(key_i)
    leaves, treedef = jax.tree_util.tree_flatten(diff_params)
    stacked = jnp.stack(leaves, axis=-1)  # (batch_size, n_active)
    idx = list(block_idx)
    block_n = len(idx)
    z = jr.normal(k_prop, (stacked.shape[0], block_n))
    delta_block = jnp.einsum("bij,bj->bi", step_scale_block, z)
    new_stacked = stacked.at[:, idx].add(delta_block)
    proposal = jax.tree_util.tree_unflatten(treedef, [new_stacked[..., i] for i in range(len(leaves))])

    log_post_proposal = _log_posterior(loss_fn, proposal, static_params, batch)
    log_ratio = log_post_proposal - log_post
    alpha = jnp.exp(jnp.minimum(log_ratio, 0.0))

    u = jr.uniform(k_acc, log_post.shape)
    accept = jnp.log(u) < log_ratio
    accept_b = _broadcast_like(accept, stacked)
    accepted_stacked = jnp.where(accept_b, new_stacked, stacked)
    new_diff_params = jax.tree_util.tree_unflatten(treedef, [accepted_stacked[..., i] for i in range(len(leaves))])
    new_log_post = jnp.where(accept, log_post_proposal, log_post)

    new_step_scale_block = None
    if adapt:
        new_step_scale_block = _ram_update(
            step_scale_block, z, alpha, target_accept, step_index, adapt_gamma, block_n
        )

    return new_diff_params, new_log_post, new_step_scale_block, accept


@eqx.filter_jit
def _run_block_ram_window(
    key: jax.Array,
    loss_fn: LossFunction,
    static_params,
    batch: Dict,
    diff_params,
    log_post: jnp.ndarray,
    step_scale_ok: jnp.ndarray,
    step_scale_problem: jnp.ndarray,
    well_idx: Tuple[int, ...],
    problem_idx: Tuple[int, ...],
    n_steps: int,
    step_offset: jnp.ndarray,
    target_accept: float,
    adapt_gamma: float,
):
    """Block analogue of _run_ram_window: each outer step updates the well-conditioned block and then the
    problem block via _block_step, each with its own adapted proposal. Both blocks share the outer step
    index for the RAM gain schedule.

    Returns:
        (diff_params, log_post, step_scale_ok, step_scale_problem, accept_count_ok, accept_count_problem)
    """

    def _single_step(carry, inputs):
        diff_params, log_post, step_scale_ok, step_scale_problem = carry
        key_i, step_index = inputs
        k_ok, k_problem = jr.split(key_i)

        diff_params, log_post, step_scale_ok, accept_ok = _block_step(
            k_ok, loss_fn, static_params, batch, diff_params, log_post, step_scale_ok, well_idx,
            adapt=True, step_index=step_index, target_accept=target_accept, adapt_gamma=adapt_gamma,
        )
        diff_params, log_post, step_scale_problem, accept_problem = _block_step(
            k_problem, loss_fn, static_params, batch, diff_params, log_post, step_scale_problem, problem_idx,
            adapt=True, step_index=step_index, target_accept=target_accept, adapt_gamma=adapt_gamma,
        )
        return (diff_params, log_post, step_scale_ok, step_scale_problem), (accept_ok, accept_problem)

    keys = jr.split(key, n_steps)
    step_indices = step_offset + 1.0 + jnp.arange(n_steps, dtype=step_offset.dtype)
    (diff_params, log_post, step_scale_ok, step_scale_problem), (accept_ok_trace, accept_problem_trace) = jax.lax.scan(
        _single_step, (diff_params, log_post, step_scale_ok, step_scale_problem), (keys, step_indices)
    )
    accept_count_ok = jnp.sum(accept_ok_trace.astype(jnp.int32), axis=0)
    accept_count_problem = jnp.sum(accept_problem_trace.astype(jnp.int32), axis=0)
    return diff_params, log_post, step_scale_ok, step_scale_problem, accept_count_ok, accept_count_problem


@eqx.filter_jit
def _run_block_window(
    key: jax.Array,
    loss_fn: LossFunction,
    static_params,
    batch: Dict,
    diff_params,
    log_post: jnp.ndarray,
    step_scale_ok: jnp.ndarray,
    step_scale_problem: jnp.ndarray,
    well_idx: Tuple[int, ...],
    problem_idx: Tuple[int, ...],
    n_steps: int,
    collect: bool,
    thin: int = 1,
):
    """Block analogue of _run_window: the same two sub-steps as _run_block_ram_window with the step
    scales frozen. collect/thin behave as in _run_window.

    Returns (diff_params, log_post, accept_count_ok, accept_count_problem, collected).
    """

    def _single_step(carry, key_i):
        diff_params, log_post, accept_count_ok, accept_count_problem = carry
        k_ok, k_problem = jr.split(key_i)
        diff_params, log_post, _, accept_ok = _block_step(
            k_ok, loss_fn, static_params, batch, diff_params, log_post, step_scale_ok, well_idx, adapt=False,
        )
        diff_params, log_post, _, accept_problem = _block_step(
            k_problem, loss_fn, static_params, batch, diff_params, log_post, step_scale_problem, problem_idx,
            adapt=False,
        )
        accept_count_ok = accept_count_ok + accept_ok.astype(jnp.int32)
        accept_count_problem = accept_count_problem + accept_problem.astype(jnp.int32)
        return (diff_params, log_post, accept_count_ok, accept_count_problem), diff_params

    init_accept_ok = jnp.zeros_like(log_post, dtype=jnp.int32)
    init_accept_problem = jnp.zeros_like(log_post, dtype=jnp.int32)

    if not collect or thin <= 1:
        keys = jr.split(key, n_steps)

        def _body(carry, key_i):
            carry, diff_params = _single_step(carry, key_i)
            return carry, (diff_params if collect else None)

        (diff_params, log_post, accept_count_ok, accept_count_problem), collected = jax.lax.scan(
            _body, (diff_params, log_post, init_accept_ok, init_accept_problem), keys
        )
        return diff_params, log_post, accept_count_ok, accept_count_problem, collected

    assert n_steps % thin == 0, f"_run_block_window: n_steps ({n_steps}) must be a multiple of thin ({thin})"
    n_groups = n_steps // thin
    group_keys = jr.split(key, n_groups)

    def _group_body(carry, group_key):
        carry, _ = jax.lax.scan(_single_step, carry, jr.split(group_key, thin))
        diff_params, _, _, _ = carry
        return carry, diff_params

    (diff_params, log_post, accept_count_ok, accept_count_problem), collected = jax.lax.scan(
        _group_body, (diff_params, log_post, init_accept_ok, init_accept_problem), group_keys
    )
    return diff_params, log_post, accept_count_ok, accept_count_problem, collected


def _seed_step_scale_default(diff_params, init_step_scale: float) -> jnp.ndarray:
    """Flat, uncorrelated proposal seed: init_step_scale * I per lineout.

    Returns:
        L: (batch_size, n_active, n_active) Cholesky factor.
    """
    leaves = jax.tree_util.tree_leaves(diff_params)
    n = len(leaves)
    if n == 0:
        return jnp.zeros((0, 0, 0))
    batch_size = leaves[0].shape[0]
    return init_step_scale * jnp.broadcast_to(jnp.eye(n), (batch_size, n, n))


# floor on the eigenvalues of the normalized Hessian before it is inverted into a proposal covariance
_LAPLACE_EIGVAL_FLOOR = 1e-3


def _normalize_hessian(H: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scales a Hessian to unit diagonal magnitude so that eigenvalue thresholds are comparable across
    parameters.

    Args:
        H: per-lineout Hessian, (batch_size, n_active, n_active).

    Returns:
        (h_norm, d): h_norm = D @ H @ D with D = diag(d), and d = 1/sqrt(|H_ii|), (batch_size, n_active).
    """
    diag = jnp.diagonal(H, axis1=-2, axis2=-1)  # (batch_size, n_active)
    d = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(diag), 1e-300))  # floors an exactly-zero diagonal entry
    h_norm = H * d[:, :, None] * d[:, None, :]  # D @ H @ D per lineout, D = diag(d)
    return h_norm, d


def _stack_hessian(hess, diff_params) -> jnp.ndarray:
    """Flattens the pytree-of-pytrees returned by LossFunction.h_loss_wrt_params_per_lineout into a
    (batch_size, n_active, n_active) array ordered like jax.tree_util.tree_leaves(diff_params)."""
    target_structure = jax.tree_util.tree_structure(diff_params)
    rows = jax.tree_util.tree_leaves(hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure)
    n = len(rows)
    blocks = [jax.tree_util.tree_leaves(row) for row in rows]  # blocks[a][b]: (batch_size,)
    rows_stacked = [jnp.stack([blocks[a][b] for b in range(n)], axis=-1) for a in range(n)]  # each: (batch_size, n)
    return jnp.stack(rows_stacked, axis=-2)  # (batch_size, n_active, n_active); [:, a, b] = d2L/d(a)d(b)


def _regularized_proposal_cholesky(H: jnp.ndarray, rr_factor: float) -> jnp.ndarray:
    """Cholesky factor of a positive-definite proposal covariance rr_factor^2 * H_reg^-1, where H_reg is
    H with the eigenvalues of its normalized form floored at _LAPLACE_EIGVAL_FLOOR. The eigenvectors of
    the normalized Hessian are preserved.

    Args:
        H: per-lineout Hessian, (batch_size, n_active, n_active).
        rr_factor: Roberts-Rosenthal scaling, 2.38/sqrt(dimension).

    Returns:
        L: (batch_size, n_active, n_active) lower-triangular factor with L @ L.T the proposal covariance.
    """
    h_norm, d = _normalize_hessian(H)
    h_norm = 0.5 * (h_norm + jnp.swapaxes(h_norm, -1, -2))  # symmetrize away float roundoff before eigh

    eigvals, eigvecs = jnp.linalg.eigh(h_norm)  # ascending eigvals; (batch_size, n), (batch_size, n, n)
    eigvals_reg = jnp.maximum(eigvals, _LAPLACE_EIGVAL_FLOOR)

    # H^-1 = D @ Q @ diag(1/eigvals) @ Q.T @ D, with H_norm = D @ H @ D = Q @ diag(eigvals) @ Q.T
    scaled_eigvecs = eigvecs * d[:, :, None]  # D @ Q
    sigma = jnp.einsum("bik,bk,bjk->bij", scaled_eigvecs, (rr_factor**2) / eigvals_reg, scaled_eigvecs)
    sigma = 0.5 * (sigma + jnp.swapaxes(sigma, -1, -2))  # symmetrize away float roundoff
    return jnp.linalg.cholesky(sigma)


def _seed_step_scale_from_laplace(loss_fn: LossFunction, static_params, batch: Dict, diff_params) -> jnp.ndarray:
    """Initial per-lineout proposal Cholesky factor from the full per-lineout Hessian of the negative
    log-posterior with respect to diff_params, scaled by 2.38/sqrt(n_active).

    Returns:
        L: (batch_size, n_active, n_active) Cholesky factor of the proposal covariance.
    """
    _, step_scale = _hessian_and_regularized_step_scale(loss_fn, static_params, batch, diff_params)
    return step_scale


def _hessian_and_regularized_step_scale(
    loss_fn: LossFunction, static_params, batch: Dict, diff_params
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the per-lineout Hessian of the negative log-posterior with respect to diff_params once
    and returns it together with its regularized proposal Cholesky factor.

    Returns:
        (H, step_scale): both (batch_size, n_active, n_active).
    """
    n = len(jax.tree_util.tree_leaves(diff_params))
    if n == 0:
        empty = jnp.zeros((0, 0, 0))
        return empty, empty

    hess = loss_fn.h_loss_wrt_params_per_lineout(diff_params, static_params, batch)
    # Hessian of the negative log-posterior: half the -2*log-likelihood Hessian plus the Jacobian term
    # (the curvature of a non-uniform prior is not included)
    H = 0.5 * _stack_hessian(hess, diff_params) + _log_jacobian_curvature(diff_params)
    rr_factor = 2.38 / jnp.sqrt(float(n))
    return H, _regularized_proposal_cholesky(H, rr_factor)


@eqx.filter_jit
def _seed_step_scale(loss_fn: LossFunction, static_params, batch: Dict, diff_params, mcmc_cfg: Dict):
    """Initial proposal step scale for one fit-batch: Laplace-seeded if mcmc_cfg["use_laplace_seed"],
    otherwise (or if the Hessian fails) the flat default. Jitted separately so it can be called once per
    fit-batch before the sampler is vmapped across fit-batches.

    Returns:
        (step_scale, H): H is the Hessian used for the seed, or None when the flat default was used.
    """
    step_scale = None
    H = None
    if mcmc_cfg["use_laplace_seed"]:
        try:
            H, step_scale = _hessian_and_regularized_step_scale(loss_fn, static_params, batch, diff_params)
        except Exception as e:
            print(f"Laplace-seeded step scale failed, falling back to flat init_step_scale: {type(e).__name__}: {e}", flush=True)
            step_scale = None
            H = None
    if step_scale is None:
        step_scale = _seed_step_scale_default(diff_params, mcmc_cfg["init_step_scale"])
    return step_scale, H


def run_mcmc_for_batch(
    config: Dict,
    loss_fn: LossFunction,
    ts_params: ThomsonParams,
    batch: Dict,
    key: jax.Array,
    progress_desc: str = "MCMC",
    pbar_position: int = 0,
    step_scale=None,
    block_step_scale: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    well_idx: Optional[Tuple[int, ...]] = None,
    problem_idx: Optional[Tuple[int, ...]] = None,
) -> Tuple[object, object, Dict]:
    """
    Runs one Metropolis-Hastings chain, vectorized across the lineouts in `batch`, starting from
    `ts_params` (that batch's best-fit weights).

    Args:
        config: configuration dictionary.
        loss_fn: LossFunction providing neg_log_likelihood.
        ts_params: best-fit weights for this batch.
        batch: batch of data, as built by loops.build_batch.
        key: PRNG key.
        progress_desc: label prefix for the progress bar.
        pbar_position: tqdm line offset for this chain's bar.
        step_scale: optional precomputed (batch_size, n_active, n_active) proposal Cholesky factor. When
            given it is used as-is, without blocking. When None the step scale (and block partition) is
            computed here.
        block_step_scale: optional (step_scale_ok, step_scale_problem) pair for a blocked run; takes
            priority over step_scale and requires well_idx/problem_idx.
        well_idx, problem_idx: leaf indices of the two blocks, used with block_step_scale.

    Returns:
        samples: diff_params-shaped pytree; each leaf has shape (num_kept, batch_size, ...), with
            num_kept = ceil((num_steps - burn_in) / thin).
        static_params: the non-sampled partition of ts_params.
        diagnostics: {"acceptance_rate": (batch_size,) array (mean over blocks when blocked),
            "final_step_scale": the adapted Cholesky factor, or a pair of them when blocked}.

    Raises:
        NotImplementedError: if the electron distribution function is an active parameter.
    """
    check_fe_inactive(config["parameters"])
    mcmc_cfg = _mcmc_cfg(config)

    filter_spec = get_filter_spec(config["parameters"], ts_params)
    diff_params, static_params = eqx.partition(ts_params, filter_spec)
    n_active = len(jax.tree_util.tree_leaves(diff_params))

    step_scale_ok = step_scale_problem = None
    if block_step_scale is not None:
        step_scale_ok, step_scale_problem = block_step_scale
    elif step_scale is None:
        step_scale, H = _seed_step_scale(loss_fn, static_params, batch, diff_params, mcmc_cfg)
        well_idx, problem_idx = tuple(range(n_active)), ()
        if mcmc_cfg["block_gibbs"] and H is not None:
            well_idx, problem_idx = _block_indices_from_hessians(
                [H], mcmc_cfg["block_gibbs_eigval_threshold"], mcmc_cfg["block_gibbs_component_threshold"]
            )
        if problem_idx:
            step_scale_ok = _block_step_scale(H, well_idx)
            step_scale_problem = _block_step_scale(H, problem_idx)
    else:
        # an explicitly supplied step_scale is used unblocked
        well_idx, problem_idx = tuple(range(n_active)), ()

    use_blocking = bool(problem_idx)

    # disperse this chain's starting point away from the best fit
    init_dispersion_factor = float(mcmc_cfg.get("init_dispersion_factor", 0.0))
    if init_dispersion_factor > 0:
        key, disperse_key = jr.split(key)
        if use_blocking:
            dispersed_scale = _combine_block_step_scales(
                init_dispersion_factor * step_scale_ok, well_idx,
                init_dispersion_factor * step_scale_problem, problem_idx,
                n_active,
            )
        else:
            dispersed_scale = init_dispersion_factor * step_scale
        diff_params = _propose(disperse_key, diff_params, dispersed_scale)

    log_post = _log_posterior(loss_fn, diff_params, static_params, batch)

    key, burn_key = jr.split(key)
    adapt_every = max(int(mcmc_cfg["adapt_every"]), 1)
    burn_in = max(int(mcmc_cfg["burn_in"]), 0)
    # full adapt_every-sized chunks followed by one shorter chunk for the remainder
    burn_chunks = [adapt_every] * (burn_in // adapt_every) + ([burn_in % adapt_every] if burn_in % adapt_every else [])
    n_sample_steps = max(int(mcmc_cfg["num_steps"]) - int(mcmc_cfg["burn_in"]), 1)
    thin = max(int(mcmc_cfg["thin"]), 1)

    # sampling is done in thin-sized groups so only thinned samples are collected; n_sample_steps is
    # rounded up to a multiple of thin
    num_kept_total = -(-n_sample_steps // thin)  # ceil division
    groups_per_chunk = max(adapt_every // thin, 1)
    total_raw_sample_steps = num_kept_total * thin

    pbar = trange(
        burn_in + total_raw_sample_steps,
        desc=f"{progress_desc} burn-in",
        unit="step",
        leave=False,
        position=pbar_position,
    )
    # RAM adapts every step; adapt_every only sets the chunk size and progress-bar granularity
    steps_done = 0
    for chunk_steps in burn_chunks:
        burn_key, window_key = jr.split(burn_key)
        step_offset = jnp.asarray(float(steps_done))
        if use_blocking:
            diff_params, log_post, step_scale_ok, step_scale_problem, accept_count_ok, accept_count_problem = (
                _run_block_ram_window(
                    window_key, loss_fn, static_params, batch, diff_params, log_post,
                    step_scale_ok, step_scale_problem, well_idx, problem_idx, chunk_steps, step_offset,
                    mcmc_cfg["target_accept"], mcmc_cfg["adapt_gamma"],
                )
            )
        else:
            diff_params, log_post, step_scale, accept_count = _run_ram_window(
                window_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, chunk_steps,
                step_offset, mcmc_cfg["target_accept"], mcmc_cfg["adapt_gamma"],
            )
        steps_done += chunk_steps
        pbar.update(chunk_steps)

    pbar.set_description(f"{progress_desc} sampling")
    key, sample_key = jr.split(key)
    total_accept_count = None
    total_accept_count_ok = total_accept_count_problem = None
    remaining_groups = num_kept_total
    collected_chunks = []
    while remaining_groups > 0:
        sample_key, chunk_key = jr.split(sample_key)
        groups_this_chunk = min(groups_per_chunk, remaining_groups)
        steps_this_chunk = groups_this_chunk * thin
        if use_blocking:
            diff_params, log_post, accept_count_ok, accept_count_problem, chunk_samples = _run_block_window(
                chunk_key, loss_fn, static_params, batch, diff_params, log_post,
                step_scale_ok, step_scale_problem, well_idx, problem_idx, steps_this_chunk,
                collect=True, thin=thin,
            )
            total_accept_count_ok = (
                accept_count_ok if total_accept_count_ok is None else total_accept_count_ok + accept_count_ok
            )
            total_accept_count_problem = (
                accept_count_problem
                if total_accept_count_problem is None
                else total_accept_count_problem + accept_count_problem
            )
        else:
            diff_params, log_post, accept_count, chunk_samples = _run_window(
                chunk_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, steps_this_chunk,
                collect=True, thin=thin,
            )
            total_accept_count = accept_count if total_accept_count is None else total_accept_count + accept_count
        collected_chunks.append(chunk_samples)  # already thinned: (groups_this_chunk, batch_size, ...)
        remaining_groups -= groups_this_chunk
        pbar.update(steps_this_chunk)
    pbar.close()

    thinned_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *collected_chunks)

    if use_blocking:
        acceptance_rate = 0.5 * (
            total_accept_count_ok / total_raw_sample_steps + total_accept_count_problem / total_raw_sample_steps
        )
        final_step_scale = (step_scale_ok, step_scale_problem)
    else:
        acceptance_rate = total_accept_count / total_raw_sample_steps
        final_step_scale = step_scale

    diagnostics = {
        "acceptance_rate": acceptance_rate,
        "final_step_scale": final_step_scale,
    }
    return thinned_samples, static_params, diagnostics


def _stack_ts_params(ts_params_list: List[ThomsonParams]) -> ThomsonParams:
    """Combines a list of structurally-identical ThomsonParams (one per fit-batch, each already batched
    over its own lineouts) into one ThomsonParams-shaped pytree with an extra leading fit-batch axis on
    every array leaf. Static (non-array) fields -- e.g. act_funs, scale/shift constants -- are identical
    across fit-batches by construction (same config), so the first fit-batch's static partition is
    reused unchanged rather than stacked."""
    array_parts = [eqx.filter(tp, eqx.is_array) for tp in ts_params_list]
    static = eqx.filter(ts_params_list[0], eqx.is_array, inverse=True)
    stacked_arrays = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *array_parts)
    return eqx.combine(stacked_arrays, static)


def _stack_batches(batch_list: List[Dict]) -> Dict:
    """Stacks a list of structurally-identical batch dicts (one per fit-batch) into one dict whose
    values have an extra leading fit-batch axis. Batch dicts hold only plain arrays, so this is a plain
    tree_map+stack with no static/array split needed."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batch_list)


def _stack_step_scales(step_scales: List) -> object:
    """Stacks a list of structurally-identical step_scale pytrees (one per fit-batch, as returned by
    _seed_step_scale) into one pytree whose leaves have an extra leading fit-batch axis. Array-only, like
    batch dicts, so a plain tree_map+stack."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *step_scales)


def run_mcmc_for_fit_batches(
    config: Dict,
    loss_fn: LossFunction,
    ts_params_list: List[ThomsonParams],
    batch_list: List[Dict],
    key: jax.Array,
    progress_desc: str = "MCMC",
    pbar_position: int = 0,
) -> Tuple[object, object, Dict]:
    """
    Runs run_mcmc_for_batch across every fit-batch of one calibration draw with a single eqx.filter_vmap.

    The initial step scale (and, when enabled, the block partition) is computed sequentially per
    fit-batch before the vmap, which keeps the Hessian's memory cost from scaling with the number of
    fit-batches. A single block partition is shared by every fit-batch.

    Returns the same three outputs as run_mcmc_for_batch, with an extra leading fit-batch axis on every
    array leaf.
    """
    n_fit_batches = len(ts_params_list)
    mcmc_cfg = _mcmc_cfg(config)

    # per-iteration timing is printed so the one-time compile is distinguishable from per-batch cost
    print(f"{progress_desc}: seeding step scale for {n_fit_batches} fit-batch(es)...", flush=True)
    step_scales = []
    hessians = []
    for i, (ts_params, batch) in enumerate(zip(ts_params_list, batch_list)):
        t0 = time.time()
        filter_spec = get_filter_spec(config["parameters"], ts_params)
        diff_params, static_params = eqx.partition(ts_params, filter_spec)
        step_scale, H = _seed_step_scale(loss_fn, static_params, batch, diff_params, mcmc_cfg)
        step_scales.append(step_scale)
        hessians.append(H)
        jax.block_until_ready(step_scale)
        print(f"{progress_desc}: fit-batch {i + 1}/{n_fit_batches} step scale seeded in {time.time() - t0:.1f}s", flush=True)
    stacked_step_scale = _stack_step_scales(step_scales)

    well_idx, problem_idx = (), ()
    stacked_block_step_scale = None
    if mcmc_cfg["block_gibbs"] and all(H is not None for H in hessians) and hessians:
        well_idx, problem_idx = _block_indices_from_hessians(
            hessians, mcmc_cfg["block_gibbs_eigval_threshold"], mcmc_cfg["block_gibbs_component_threshold"]
        )
        if problem_idx:
            block_step_scales = [
                (_block_step_scale(H, well_idx), _block_step_scale(H, problem_idx)) for H in hessians
            ]
            stacked_ok = _stack_step_scales([bs[0] for bs in block_step_scales])
            stacked_problem = _stack_step_scales([bs[1] for bs in block_step_scales])
            stacked_block_step_scale = (stacked_ok, stacked_problem)
            print(
                f"{progress_desc}: block Metropolis-within-Gibbs active -- "
                f"{len(problem_idx)}/{len(well_idx) + len(problem_idx)} active leaf(es) isolated into their own block",
                flush=True,
            )

    stacked_ts_params = _stack_ts_params(ts_params_list)
    stacked_batch = _stack_batches(batch_list)
    keys = jr.split(key, n_fit_batches)

    def _one(ts_params, batch, k, step_scale, block_step_scale):
        return run_mcmc_for_batch(
            config, loss_fn, ts_params, batch, k, progress_desc=progress_desc, pbar_position=pbar_position,
            step_scale=step_scale, block_step_scale=block_step_scale, well_idx=well_idx, problem_idx=problem_idx,
        )

    return eqx.filter_vmap(_one)(stacked_ts_params, stacked_batch, keys, stacked_step_scale, stacked_block_step_scale)


def _classic_r_hat(x: np.ndarray) -> np.ndarray:
    """Gelman-Rubin R-hat for x shaped (num_kept, num_chains, *extra): sqrt of the ratio of the pooled
    variance estimate to the within-chain variance. Returns an array of shape (*extra,)."""
    num_kept, num_chains = x.shape[0], x.shape[1]
    chain_mean = x.mean(axis=0)
    grand_mean = chain_mean.mean(axis=0, keepdims=True)
    between = num_kept / (num_chains - 1) * np.sum((chain_mean - grand_mean) ** 2, axis=0)
    within = x.var(axis=0, ddof=1).mean(axis=0)
    var_hat = (num_kept - 1) / num_kept * within + between / num_kept
    return np.sqrt(var_hat / within)


def _split_in_half(x: np.ndarray) -> np.ndarray:
    """Splits x, shaped (num_kept, num_chains, *extra), into (num_kept // 2, 2 * num_chains, *extra):
    each chain's first half followed by each chain's second half. Drops one sample if num_kept is odd."""
    num_kept = x.shape[0]
    half = num_kept // 2
    return np.concatenate([x[:half], x[half : 2 * half]], axis=1)


def _rank_normalize_pooled(x: np.ndarray) -> np.ndarray:
    """Rank-normalizes x, shaped (num_kept, num_chains, *extra): values are ranked across both leading
    axes together (ties averaged) and mapped to normal scores, z = Phi^-1((rank - 3/8) / (N + 1/4)).
    See Vehtari et al., Bayesian Analysis 16, 667 (2021)."""
    num_kept, num_chains = x.shape[0], x.shape[1]
    n = num_kept * num_chains
    flat = x.reshape(n, *x.shape[2:])
    ranks = scipy.stats.rankdata(flat, axis=0)
    z = scipy.stats.norm.ppf((ranks - 3.0 / 8.0) / (n + 1.0 / 4.0))
    return z.reshape(x.shape)


def _rank_normalized_r_hat(x: np.ndarray) -> np.ndarray:
    """Rank-normalized, folded, split R-hat (Vehtari et al. 2021): the larger of the bulk R-hat (split,
    rank-normalized chains) and the tail R-hat (the same applied to |x - median|).

    Args:
        x: (num_kept, num_chains, *extra) samples, num_chains >= 1.

    Returns:
        (*extra,) R-hat.
    """
    x = np.asarray(x)
    bulk = _classic_r_hat(_rank_normalize_pooled(_split_in_half(x)))

    grand_median = np.median(x.reshape(-1, *x.shape[2:]), axis=0)
    folded = np.abs(x - grand_median[None, None])
    tail = _classic_r_hat(_rank_normalize_pooled(_split_in_half(folded)))

    return np.maximum(bulk, tail)


def _max_r_hat_across_chains(per_draw_samples: List) -> Optional[np.ndarray]:
    """Rank-normalized R-hat across the chains in per_draw_samples, per fit-batch, lineout and active
    parameter. Each element is a diff_params-shaped pytree with leaves shaped
    (num_fit_batches, num_kept, batch_size, ...).

    Returns:
        (num_fit_batches, batch_size, n_active) array with parameters in tree_leaves order, or None for
        fewer than 2 chains.
    """
    num_chains = len(per_draw_samples)
    if num_chains < 2:
        return None
    leaf_lists = [jax.tree_util.tree_leaves(s) for s in per_draw_samples]
    if not leaf_lists[0]:
        return None
    per_leaf_r_hat = []
    for leaf_idx in range(len(leaf_lists[0])):
        stacked = np.stack([np.asarray(leaf_lists[c][leaf_idx]) for c in range(num_chains)], axis=0)
        stacked = np.moveaxis(stacked, 2, 0)  # (num_kept, num_chains, num_fit_batches, batch_size, ...)
        per_leaf_r_hat.append(_rank_normalized_r_hat(stacked))
    return np.stack(per_leaf_r_hat, axis=-1)  # (num_fit_batches, batch_size, n_active)


def _within_chain_r_hat(per_draw_samples: List) -> np.ndarray:
    """Rank-normalized split R-hat of each chain on its own, per fit-batch, lineout and active parameter.

    Returns:
        (num_chains, num_fit_batches, batch_size, n_active) array with parameters in tree_leaves order.
    """
    per_chain_r_hat = []
    for samples in per_draw_samples:
        leaves = jax.tree_util.tree_leaves(samples)  # each: (num_fit_batches, num_kept, batch_size, ...)
        if not leaves:
            return np.zeros((len(per_draw_samples), 0, 0, 0))
        per_leaf_r_hat = []
        for leaf in leaves:
            x = np.moveaxis(np.asarray(leaf), 1, 0)  # (num_kept, num_fit_batches, batch_size, ...)
            x = x[:, None, ...]  # (num_kept, 1, num_fit_batches, batch_size, ...) -- "1 chain", split inside
            per_leaf_r_hat.append(_rank_normalized_r_hat(x))  # (num_fit_batches, batch_size, ...)
        per_chain_r_hat.append(np.stack(per_leaf_r_hat, axis=-1))  # (num_fit_batches, batch_size, n_active)
    return np.stack(per_chain_r_hat, axis=0)  # (num_chains, num_fit_batches, batch_size, n_active)


def _checkpoint_draw(run_id: Optional[str], draw_index: int, samples, static, diagnostics) -> None:
    """Pickles one draw's samples and diagnostics and uploads them to the mlflow run under
    draw_checkpoints/ as soon as the draw finishes. `static` is not saved (it is not picklable and is
    recoverable from the fitted weights). No-op when run_id is None; failures are reported but not raised.
    """
    if run_id is None:
        return
    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"draw_{draw_index:02d}.pkl")
            with open(path, "wb") as f:
                pickle.dump({"samples": samples, "diagnostics": diagnostics}, f)
            MlflowClient().log_artifact(run_id, path, artifact_path="draw_checkpoints")
        print(f"[checkpoint] draw {draw_index + 1} saved to draw_checkpoints/draw_{draw_index:02d}.pkl", flush=True)
    except Exception as e:
        # checkpointing is best-effort and must not stop the sampler
        print(f"[checkpoint] draw {draw_index + 1} checkpoint FAILED (non-fatal, continuing): "
              f"{type(e).__name__}: {e}", flush=True)


def run_mcmc_pooled(
    config: Dict,
    loss_fns_by_draw: List[LossFunction],
    ts_params_list: List[ThomsonParams],
    batches_by_draw: List[List[Dict]],
    key: jax.Array,
    checkpoint_run_id: Optional[str] = None,
) -> Tuple[object, object, List[Dict], object, object]:
    """
    Runs run_mcmc_for_fit_batches once per chain (one LossFunction and PRNG subkey each) and pools the
    post-burn-in samples along the sample axis. Chains are dispatched to a thread pool, one per worker,
    round-robined across jax.local_devices().

    Args:
        config: configuration dictionary.
        loss_fns_by_draw: length-K list of LossFunction instances, one per chain.
        ts_params_list: the fit-batches' best-fit weights (the starting point of every chain).
        batches_by_draw: length-K list of per-fit-batch batch dict lists.
        key: PRNG key; split once per chain.
        checkpoint_run_id: mlflow run id to upload each finished chain to (see _checkpoint_draw), or None.

    Returns:
        pooled_samples: diff_params-shaped pytree, leaves shaped (num_fit_batches, K * num_kept, batch_size, ...).
        static_params: as returned by run_mcmc_for_fit_batches for chain 0.
        diagnostics_by_draw: list of each chain's diagnostics dict.
        max_r_hat: (num_fit_batches, batch_size, n_active) cross-chain R-hat, or None when K < 2.
        within_chain_r_hat: (K, num_fit_batches, batch_size, n_active) split R-hat of each chain.
    """
    keys = jr.split(key, len(loss_fns_by_draw))
    n_draws = len(loss_fns_by_draw)
    devices = jax.local_devices()

    def _run_one_draw(draw_index, loss_fn, batch_list, draw_key):
        progress_desc = f"MCMC draw {draw_index + 1}/{n_draws}" if n_draws > 1 else "MCMC"
        device = devices[draw_index % len(devices)]
        # default_device is thread-local, so this draw's arrays are allocated on its own device
        with jax.default_device(device):
            samples, static, diagnostics = run_mcmc_for_fit_batches(
                config, loss_fn, ts_params_list, batch_list, draw_key, progress_desc=progress_desc, pbar_position=draw_index
            )
        _checkpoint_draw(checkpoint_run_id, draw_index, samples, static, diagnostics)
        return progress_desc, samples, static, diagnostics

    with ThreadPoolExecutor(max_workers=max(len(devices), 1)) as pool:
        futures = [
            pool.submit(_run_one_draw, draw_index, loss_fn, batch_list, draw_key)
            for draw_index, (loss_fn, batch_list, draw_key) in enumerate(zip(loss_fns_by_draw, batches_by_draw, keys))
        ]
        results = [f.result() for f in futures]  # preserves draw order regardless of completion order

    per_draw_samples = [r[1] for r in results]
    static_params = results[0][2] if results else None
    diagnostics_by_draw = [r[3] for r in results]
    for progress_desc, _, _, diagnostics in results:
        # acceptance rates are only concrete once the vmapped call has returned
        mean_accept = float(jnp.mean(diagnostics["acceptance_rate"]))
        print(f"{progress_desc} done: mean acceptance rate {mean_accept:.3f}")

    max_r_hat = _max_r_hat_across_chains(per_draw_samples)
    within_chain_r_hat = _within_chain_r_hat(per_draw_samples)

    if len(per_draw_samples) == 1:
        pooled_samples = per_draw_samples[0]
    else:
        # pool across draws along the num_kept axis (axis=1)
        pooled_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *per_draw_samples)

    return pooled_samples, static_params, diagnostics_by_draw, max_r_hat, within_chain_r_hat
