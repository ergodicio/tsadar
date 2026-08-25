"""Metropolis-Hastings MCMC sampler used by mcmc_postprocess.py to estimate per-lineout parameter
uncertainty and covariance, as an alternative to the Hessian/Laplace approximation in `.laplace`.

Proposals happen on exactly the same `diff_params` leaves the optimizer fits (see
tsadar.core.modules.ts_params.get_filter_spec), in the same sigmoid/logit-unconstrained space, so the
existing [lb, ub] bounds are enforced for free by that reparametrization -- no separate bounds handling
is needed here.

Scope limitation -- electron distribution function ("fe") sampling is not supported: every other active
leaf (Te, ne, Ti, Z, fract, Va, amp1/2/3, lam, ne_gradient, Te_gradient, ud, brem_amp, brem_c) is stored
as one array with a leading (batch_size,) axis across a fit-batch's lineouts, uniformly vectorizable.
"fe" is different: ElectronParams.distribution_functions is a *list* of batch_size separate,
per-lineout distribution-function objects (see ElectronParams.init_dists) rather than one object with a
batch axis, so its active leaves (e.g. the DLM shape parameter "m") appear as batch_size separate scalar
leaves rather than one (batch_size,)-shaped leaf -- incompatible with this module's
leaf-broadcast-based proposal/accept-reject without a further per-lineout destacking step, which is left
as a documented follow-on. run_mcmc_for_batch raises NotImplementedError if "fe" is active.
"""
from typing import Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr

from tsadar.core.modules.ts_params import ThomsonParams, get_filter_spec
from tsadar.inverse.loss_function import LossFunction

_DEFAULTS = {
    "num_steps": 8000,
    "burn_in": 3000,
    "thin": 5,
    "adapt_every": 50,
    "target_accept": 0.234,
    "adapt_gamma": 0.6,
    "init_step_scale": 0.1,
    "use_laplace_seed": True,
    "seed": 0,
    "save_samples": True,
    # postprocess.laplace.get_sigmas' Hessian is taken w.r.t. the *entire* ts_params pytree, not just the
    # active leaves (unlike _seed_step_scale_from_laplace above, which was deliberately restricted to
    # diff_params after this same full-tree Hessian was observed to attempt a >150GB allocation on an
    # ordinary fit whose electron distribution function carries a sizeable fixed interpolation table,
    # even with "fe" inactive). recalculate_with_chosen_weights already catches that failure and
    # disables calc_sigma gracefully, but the attempt itself costs real time and memory pressure, so the
    # comparison plot defaults to off; opt in with compare_to_laplace: true once you've confirmed your
    # config's Hessian is actually affordable.
    "compare_to_laplace": False,
}


def _mcmc_cfg(config: Dict) -> Dict:
    """config["other"]["mcmc"], with every field defaulted so older decks (or decks that never
    configured this feature) work unchanged."""
    user_cfg = config.get("other", {}).get("mcmc", {})
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


def _propose(key: jax.Array, diff_params, step_scale):
    """One Gaussian random-walk proposal across every leaf of diff_params, scaled per-lineout and
    per-leaf by the matching leaf of step_scale (same pytree structure)."""
    leaves, treedef = jax.tree_util.tree_flatten(diff_params)
    scale_leaves = jax.tree_util.tree_leaves(step_scale)
    keys = list(jr.split(key, max(len(leaves), 1)))
    new_leaves = [
        leaf + _broadcast_like(scale, leaf) * jr.normal(k, leaf.shape)
        for leaf, scale, k in zip(leaves, scale_leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _log_posterior(loss_fn: LossFunction, diff_params, static_params, batch: Dict) -> jnp.ndarray:
    """Per-lineout log-posterior, up to an additive constant: -0.5 * neg_log_likelihood."""
    weights = eqx.combine(static_params, diff_params)
    return -0.5 * loss_fn.neg_log_likelihood(weights, batch, per_lineout=True)


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
):
    """Runs n_steps of propose+accept/reject via jax.lax.scan at a fixed step_scale. When collect is
    True, every step's diff_params is stacked and returned (for the sampling phase); when False, only
    the final state and per-lineout accept counts are computed (for cheap burn-in windows)."""

    def _body(carry, key_i):
        diff_params, log_post, accept_count = carry
        k_prop, k_acc = jr.split(key_i)
        proposal = _propose(k_prop, diff_params, step_scale)
        log_post_proposal = _log_posterior(loss_fn, proposal, static_params, batch)
        diff_params, log_post, accept = _mh_accept(k_acc, diff_params, log_post, proposal, log_post_proposal)
        accept_count = accept_count + accept.astype(jnp.int32)
        out = diff_params if collect else None
        return (diff_params, log_post, accept_count), out

    keys = jr.split(key, n_steps)
    init_accept_count = jnp.zeros_like(log_post, dtype=jnp.int32)
    (diff_params, log_post, accept_count), collected = jax.lax.scan(
        _body, (diff_params, log_post, init_accept_count), keys
    )
    return diff_params, log_post, accept_count, collected


def _adapt_step_scale(step_scale, accept_rate: jnp.ndarray, target_accept: float, window_index: int, adapt_gamma: float):
    """Robbins-Monro proposal-scale update, applied identically to every leaf of step_scale for a given
    lineout (there is only one joint accept/reject decision per lineout per step, so a single per-lineout
    acceptance-rate signal is all that's available to adapt from)."""
    factor = jnp.exp((accept_rate - target_accept) / (window_index + 1.0) ** adapt_gamma)
    return jax.tree_util.tree_map(lambda s: s * _broadcast_like(factor, s), step_scale)


def _seed_step_scale_default(diff_params, fallback_scale: float):
    """Pure heuristic proposal-scale seed: a flat `fallback_scale` per leaf/lineout, in the
    unconstrained/logit space diff_params already lives in. No dependency on the Hessian/Laplace
    machinery -- always available, always succeeds. Used whenever use_laplace_seed is False, or when
    _seed_step_scale_from_laplace fails or produces a non-finite/non-positive scale."""
    return jax.tree_util.tree_map(lambda leaf: jnp.full(leaf.shape, fallback_scale), diff_params)


def _seed_step_scale_from_laplace(loss_fn: LossFunction, static_params, batch: Dict, diff_params):
    """Seeds initial per-lineout, per-leaf proposal scale from a Laplace/Hessian covariance, scaled by
    the standard Roberts-Rosenthal 2.38/sqrt(d) optimal-scaling factor (d = number of active scalar
    leaves). Off-diagonal (cross-parameter) terms are ignored -- only each leaf's own second derivative
    is used, matching postprocess.laplace.get_sigmas' use of only the diagonal blocks (cross-lineout
    terms are structurally zero; cross-parameter terms are dropped here for simplicity, since this seeds
    independent per-leaf proposal scales, not a joint proposal covariance).

    Deliberately differentiates loss_fn.neg_log_likelihood w.r.t. diff_params only (not the full
    ts_params, unlike loss_fn.h_loss_wrt_params/postprocess.laplace.get_sigmas) -- hessian-ing the full
    parameter tree pulls in every fixed array the model carries, including large distribution-function
    lookup tables that are never actually being sampled, and has been observed to attempt a
    multi-hundred-GB allocation on an ordinary fit. Restricting to diff_params keeps this to exactly the
    handful of scalar parameters actually active, which is what run_mcmc_for_batch's fe-active guard
    guarantees are the only leaves present here.

    Raises ValueError (not caught here -- the caller decides whether to fall back) if the Hessian is
    degenerate: non-positive curvature, or a non-finite/non-positive resulting scale, for any
    leaf/lineout.
    """

    def _nll_of_diff(dp):
        weights = eqx.combine(static_params, dp)
        return loss_fn.neg_log_likelihood(weights, batch, per_lineout=False)

    hess = eqx.filter_hessian(_nll_of_diff)(diff_params)
    flat_diff, treedef = jax.tree_util.tree_flatten(diff_params)
    n = len(flat_diff)
    if n == 0:
        return jax.tree_util.tree_unflatten(treedef, [])

    target_structure = jax.tree_util.tree_structure(diff_params)
    # hess has diff_params' structure at the outer level; every "leaf" there is itself a diff_params-
    # shaped pytree (the row of second derivatives wrt that one leaf). Stopping tree_leaves' descent as
    # soon as a subtree's structure matches diff_params' finds exactly those n rows, without needing to
    # know the concrete species/parameter names.
    rows = jax.tree_util.tree_leaves(hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure)
    if len(rows) != n:
        raise ValueError(f"Unexpected Hessian structure: found {len(rows)} row(s), expected {n}")

    rr_factor = 2.38 / jnp.sqrt(float(n))
    scale_leaves = []
    for i, row in enumerate(rows):
        row_leaves = jax.tree_util.tree_leaves(row)
        if len(row_leaves) != n:
            raise ValueError(f"Unexpected Hessian row structure: found {len(row_leaves)} entries, expected {n}")
        h_ii = jnp.diagonal(row_leaves[i])  # (batch_size, batch_size) -> (batch_size,); cross-lineout terms are ~0
        var = jnp.where(h_ii > 0, 1.0 / h_ii, jnp.nan)
        scale_leaves.append(rr_factor * jnp.sqrt(var))

    if not all(bool(jnp.all(jnp.isfinite(s))) and bool(jnp.all(s > 0)) for s in scale_leaves):
        raise ValueError("Laplace-seeded step scale is non-finite or non-positive for at least one lineout/parameter")

    return jax.tree_util.tree_unflatten(treedef, scale_leaves)


def run_mcmc_for_batch(
    config: Dict, loss_fn: LossFunction, ts_params: ThomsonParams, batch: Dict, key: jax.Array
) -> Tuple[object, object, Dict]:
    """
    Runs one Metropolis-Hastings chain, vectorized across the lineouts in `batch`, seeded at
    `ts_params` (that batch's best-fit weights).

    Returns:
        samples: a diff_params-shaped pytree; each leaf has shape (num_kept, batch_size, ...), where
            num_kept = ceil((num_steps - burn_in) / thin).
        static_params: the non-sampled complement of ts_params (eqx.partition's static half), needed by
            the caller to recombine samples into full parameter values via eqx.combine.
        diagnostics: {"acceptance_rate": array (batch_size,), "final_step_scale": step_scale pytree}.

    Raises:
        NotImplementedError: if config["parameters"]["electron"]["fe"]["active"] is true (see module
            docstring).
    """
    check_fe_inactive(config["parameters"])
    mcmc_cfg = _mcmc_cfg(config)

    filter_spec = get_filter_spec(config["parameters"], ts_params)
    diff_params, static_params = eqx.partition(ts_params, filter_spec)

    step_scale = None
    if mcmc_cfg["use_laplace_seed"]:
        try:
            step_scale = _seed_step_scale_from_laplace(loss_fn, static_params, batch, diff_params)
        except Exception:
            step_scale = None
    if step_scale is None:
        step_scale = _seed_step_scale_default(diff_params, mcmc_cfg["init_step_scale"])

    log_post = _log_posterior(loss_fn, diff_params, static_params, batch)

    key, burn_key = jr.split(key)
    adapt_every = max(int(mcmc_cfg["adapt_every"]), 1)
    n_windows = max(int(mcmc_cfg["burn_in"]) // adapt_every, 0) if mcmc_cfg["burn_in"] > 0 else 0
    for window_index in range(n_windows):
        burn_key, window_key = jr.split(burn_key)
        diff_params, log_post, accept_count, _ = _run_window(
            window_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, adapt_every, collect=False
        )
        accept_rate = accept_count / adapt_every
        step_scale = _adapt_step_scale(step_scale, accept_rate, mcmc_cfg["target_accept"], window_index, mcmc_cfg["adapt_gamma"])

    n_sample_steps = max(int(mcmc_cfg["num_steps"]) - int(mcmc_cfg["burn_in"]), 1)
    key, sample_key = jr.split(key)
    diff_params, log_post, accept_count, samples = _run_window(
        sample_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, n_sample_steps, collect=True
    )
    thin = max(int(mcmc_cfg["thin"]), 1)
    thinned_samples = jax.tree_util.tree_map(lambda x: x[::thin], samples)

    diagnostics = {
        "acceptance_rate": accept_count / n_sample_steps,
        "final_step_scale": step_scale,
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


def run_mcmc_for_fit_batches(
    config: Dict, loss_fn: LossFunction, ts_params_list: List[ThomsonParams], batch_list: List[Dict], key: jax.Array
) -> Tuple[object, object, Dict]:
    """
    Runs run_mcmc_for_batch across every fit-batch of a single calibration draw. Every fit-batch shares
    the exact same compiled LossFunction/config -- only the data slice and starting weights differ, both
    ordinary array-valued inputs -- so this is a single eqx.filter_vmap over the fit-batch axis rather
    than a Python loop, avoiding a separate trace/compile per fit-batch.

    Returns the same three outputs as run_mcmc_for_batch, each with an extra leading fit-batch axis on
    every array leaf (size len(ts_params_list)); static_params/diagnostics' non-array leaves are passed
    through unbatched by eqx.filter_vmap since they are identical across fit-batches.
    """
    n_fit_batches = len(ts_params_list)
    stacked_ts_params = _stack_ts_params(ts_params_list)
    stacked_batch = _stack_batches(batch_list)
    keys = jr.split(key, n_fit_batches)

    def _one(ts_params, batch, k):
        return run_mcmc_for_batch(config, loss_fn, ts_params, batch, k)

    return eqx.filter_vmap(_one)(stacked_ts_params, stacked_batch, keys)


def run_mcmc_pooled(
    config: Dict,
    loss_fns_by_draw: List[LossFunction],
    ts_params_list: List[ThomsonParams],
    batches_by_draw: List[List[Dict]],
    key: jax.Array,
) -> Tuple[object, object, List[Dict]]:
    """
    Runs run_mcmc_for_fit_batches independently for each of the K calibration draws (own PRNG subkey
    each), then pools all K draws' post-burn-in samples together along the sample axis -- the union of
    within-chain parameter uncertainty and between-draw calibration-nuisance uncertainty. For K == 1
    this is a no-op concatenation of a single draw's output.

    Each draw's LossFunction is built from different static config (a different FormFactor/IRF per
    draw), so -- unlike the fit-batch axis within one draw -- this loop cannot be vmapped into one
    compiled graph; it stays a Python-level loop over K chains.

    Args:
        loss_fns_by_draw: length-K list of LossFunction instances, one per calibration draw.
        ts_params_list: the fit-batches' best-fit weights (shared starting point for every draw).
        batches_by_draw: length-K list, each a length-num_fit_batches list of batch dicts (one per
            fit-batch, built against that draw's possibly-rescaled data).
        key: PRNG key; split once per draw.

    Returns:
        pooled_samples: diff_params-shaped pytree, leaves shaped (num_fit_batches, K * num_kept, batch_size, ...).
        static_params: as returned by run_mcmc_for_fit_batches (from draw 0; identical in structure/value
            across draws for a fixed config).
        diagnostics_by_draw: list of length K, each draw's diagnostics dict (with the fit-batch axis).
    """
    keys = jr.split(key, len(loss_fns_by_draw))

    per_draw_samples = []
    static_params = None
    diagnostics_by_draw = []
    for loss_fn, batch_list, draw_key in zip(loss_fns_by_draw, batches_by_draw, keys):
        samples, static, diagnostics = run_mcmc_for_fit_batches(config, loss_fn, ts_params_list, batch_list, draw_key)
        per_draw_samples.append(samples)
        diagnostics_by_draw.append(diagnostics)
        if static_params is None:
            static_params = static

    if len(per_draw_samples) == 1:
        pooled_samples = per_draw_samples[0]
    else:
        # each draw's samples have shape (num_fit_batches, num_kept, batch_size, ...); concatenate along
        # the num_kept axis (axis=1) to pool across draws while keeping the fit-batch axis (axis=0) intact.
        pooled_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *per_draw_samples)

    return pooled_samples, static_params, diagnostics_by_draw
