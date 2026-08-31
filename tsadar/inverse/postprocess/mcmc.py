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
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from tqdm import trange

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
    # Multiplier on the (Laplace-seeded or flat init_step_scale) per-lineout/per-parameter step scale,
    # used to perturb each chain's own starting point before burn-in begins (see run_mcmc_for_batch).
    # 0.0 (default) means every chain starts at the exact best fit, matching pre-multi-chain behavior
    # exactly. Set > 0 when running several chains (config["other"]["calibration_uncertainty"]
    # ["num_draws"] > 1) purely for dispersed starts / a meaningful R-hat -- see mcmc.rst.
    "init_dispersion_factor": 0.0,
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
    """Runs n_steps of propose+accept/reject via jax.lax.scan at a fixed step_scale. When collect is
    False (burn-in windows), only the final state and per-lineout accept counts are computed --
    jax.lax.scan's `None` output for every step costs nothing (no leaves to stack).

    @eqx.filter_jit matters here far more than it would for an ordinary function: run_mcmc_for_batch
    calls this once per burn-in window and once per sampling chunk (tens to ~100+ calls per chain), and
    without caching, each of those calls makes JAX rebuild the forward-model trace from scratch even
    though the compiled XLA kernel underneath is reused -- measured at ~800-900ms of pure Python retracing
    per call on a small test fit, i.e. the large majority of total wall time, and identically so whether
    run_mcmc_for_batch is called eagerly or (as in production) traced once inside
    run_mcmc_for_fit_batches' eqx.filter_vmap, since vmap's own one-time trace still calls this bare
    Python function fresh for every window/chunk. filter_jit gives every call after the first (same
    n_steps/collect/thin) a cache hit, skipping the retrace entirely -- ~15x faster on that same fit,
    with no change to the underlying algorithm or outputs.

    When collect is True (sampling), a naive "collect every step, then slice every thin-th one" scan
    would have to hold *all* n_steps' worth of raw diff_params in memory before any thinning ever
    happens -- for a long chain (many thousands of steps) this can be the dominant, and easily
    OOM-triggering, memory cost of the whole sampler, even though only 1/thin of it is ever kept. So
    when thin > 1, this instead nests an inner, uncollected jax.lax.scan of exactly `thin` steps inside
    an outer scan that only collects the *last* state of each inner group -- the collected output is
    already the thinned result (shape (n_steps // thin, batch_size, ...)), with no intermediate buffer
    ever holding more than one thinned sample's worth of history at a time. Requires n_steps % thin == 0
    (run_mcmc_for_batch's chunking guarantees this).
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
    _seed_step_scale_from_laplace fails structurally (see its docstring for the per-entry fallback it
    already applies for individually-degenerate lineouts/parameters)."""
    return jax.tree_util.tree_map(lambda leaf: jnp.full(leaf.shape, fallback_scale), diff_params)


def _seed_step_scale_from_laplace(loss_fn: LossFunction, static_params, batch: Dict, diff_params, fallback_scale: float):
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

    Wherever the Hessian is degenerate for a given leaf/lineout (non-positive curvature, or a resulting
    scale that's non-finite or non-positive), that entry is individually replaced by fallback_scale via
    jnp.where -- deliberately per-entry rather than an all-or-nothing raise/except: this runs under
    run_mcmc_for_fit_batches' eqx.filter_vmap in production, where every value here is a batching tracer,
    so a Python-level bool()/raise on a data-dependent validity check would raise
    TracerBoolConversionError regardless of whether the Hessian was actually degenerate (this was, in
    fact, silently swallowing every real Laplace-seeded scale in production and falling back to a flat
    fallback_scale for every lineout -- see git history/PR discussion for the regression this fixed).
    Structural checks below (row/column counts) stay as ordinary raises since they depend only on pytree
    shape, which is identical across every vmap lane.
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
        scale = rr_factor * jnp.sqrt(var)
        valid = jnp.isfinite(scale) & (scale > 0)
        scale_leaves.append(jnp.where(valid, scale, fallback_scale))

    return jax.tree_util.tree_unflatten(treedef, scale_leaves)


def run_mcmc_for_batch(
    config: Dict,
    loss_fn: LossFunction,
    ts_params: ThomsonParams,
    batch: Dict,
    key: jax.Array,
    progress_desc: str = "MCMC",
    pbar_position: int = 0,
) -> Tuple[object, object, Dict]:
    """
    Runs one Metropolis-Hastings chain, vectorized across the lineouts in `batch`, seeded at
    `ts_params` (that batch's best-fit weights).

    Reports a tqdm step counter (burn-in windows, then sampling windows) as it runs. This function is
    normally invoked through run_mcmc_for_fit_batches' eqx.filter_vmap, so the per-step values (e.g.
    accept rate) are batching tracers that cannot be concretized into the bar's text here without
    breaking the vmap trace -- only a step/window count is shown; the real numeric acceptance rate is
    reported one level up, per calibration draw, once run_mcmc_for_fit_batches' vmapped call has actually
    returned concrete arrays (see run_mcmc_pooled).

    Args:
        progress_desc: prefix for the progress bar's label (e.g. which calibration draw this chain
            belongs to), so nested draws are distinguishable in the terminal.
        pbar_position: tqdm `position` (terminal line offset) for this chain's bar -- run_mcmc_pooled
            runs draws concurrently on separate threads/devices, so each draw needs its own line to avoid
            garbled interleaved output.

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
            step_scale = _seed_step_scale_from_laplace(
                loss_fn, static_params, batch, diff_params, mcmc_cfg["init_step_scale"]
            )
        except Exception:
            step_scale = None
    if step_scale is None:
        step_scale = _seed_step_scale_default(diff_params, mcmc_cfg["init_step_scale"])

    # Nudges this chain's own starting point away from the shared best fit, so that when several
    # independent chains are pooled (config["other"]["calibration_uncertainty"]["num_draws"] > 1) they
    # don't all begin at literally the same point -- see run_mcmc_pooled's R-hat computation, which needs
    # genuinely independent chains to be meaningful.
    init_dispersion_factor = float(mcmc_cfg.get("init_dispersion_factor", 0.0))
    if init_dispersion_factor > 0:
        key, disperse_key = jr.split(key)
        dispersed_scale = jax.tree_util.tree_map(lambda s: init_dispersion_factor * s, step_scale)
        diff_params = _propose(disperse_key, diff_params, dispersed_scale)

    log_post = _log_posterior(loss_fn, diff_params, static_params, batch)

    key, burn_key = jr.split(key)
    adapt_every = max(int(mcmc_cfg["adapt_every"]), 1)
    n_windows = max(int(mcmc_cfg["burn_in"]) // adapt_every, 0) if mcmc_cfg["burn_in"] > 0 else 0
    n_sample_steps = max(int(mcmc_cfg["num_steps"]) - int(mcmc_cfg["burn_in"]), 1)
    thin = max(int(mcmc_cfg["thin"]), 1)

    # Sampling is grouped into thin-sized units so _run_window can collect only the thinned samples
    # directly (see its docstring) rather than every raw step of the whole sampling phase -- for a long
    # chain, holding every raw step in memory before thinning is easily the dominant memory cost and can
    # OOM. Rounds n_sample_steps up to the next multiple of thin if it wasn't already (at most thin-1
    # extra MH steps) so every chunk's step count divides evenly by thin, as _run_window requires.
    num_kept_total = -(-n_sample_steps // thin)  # ceil division
    groups_per_chunk = max(adapt_every // thin, 1)
    total_raw_sample_steps = num_kept_total * thin

    pbar = trange(
        n_windows * adapt_every + total_raw_sample_steps,
        desc=f"{progress_desc} burn-in",
        unit="step",
        leave=False,
        position=pbar_position,
    )
    for window_index in range(n_windows):
        burn_key, window_key = jr.split(burn_key)
        diff_params, log_post, accept_count, _ = _run_window(
            window_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, adapt_every, collect=False
        )
        accept_rate = accept_count / adapt_every
        step_scale = _adapt_step_scale(step_scale, accept_rate, mcmc_cfg["target_accept"], window_index, mcmc_cfg["adapt_gamma"])
        pbar.update(adapt_every)

    pbar.set_description(f"{progress_desc} sampling")
    key, sample_key = jr.split(key)
    total_accept_count = None
    remaining_groups = num_kept_total
    collected_chunks = []
    while remaining_groups > 0:
        sample_key, chunk_key = jr.split(sample_key)
        groups_this_chunk = min(groups_per_chunk, remaining_groups)
        steps_this_chunk = groups_this_chunk * thin
        diff_params, log_post, accept_count, chunk_samples = _run_window(
            chunk_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, steps_this_chunk,
            collect=True, thin=thin,
        )
        collected_chunks.append(chunk_samples)  # already thinned: (groups_this_chunk, batch_size, ...)
        total_accept_count = accept_count if total_accept_count is None else total_accept_count + accept_count
        remaining_groups -= groups_this_chunk
        pbar.update(steps_this_chunk)
    pbar.close()

    thinned_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *collected_chunks)

    diagnostics = {
        "acceptance_rate": total_accept_count / total_raw_sample_steps,
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
    config: Dict,
    loss_fn: LossFunction,
    ts_params_list: List[ThomsonParams],
    batch_list: List[Dict],
    key: jax.Array,
    progress_desc: str = "MCMC",
    pbar_position: int = 0,
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
        return run_mcmc_for_batch(
            config, loss_fn, ts_params, batch, k, progress_desc=progress_desc, pbar_position=pbar_position
        )

    return eqx.filter_vmap(_one)(stacked_ts_params, stacked_batch, keys)


def _gelman_rubin_r_hat(x: jnp.ndarray) -> jnp.ndarray:
    """Classic Gelman-Rubin R-hat for x shaped (num_kept, num_chains, *extra): the ratio of the pooled
    (between + within-chain) variance estimate to the within-chain variance, reduced over the leading two
    axes and broadcast over any remaining ones. Close to 1 when the chains have mixed to the same
    distribution; values well above ~1.01-1.1 indicate they have not."""
    num_kept, num_chains = x.shape[0], x.shape[1]
    chain_mean = x.mean(axis=0)
    grand_mean = chain_mean.mean(axis=0, keepdims=True)
    between = num_kept / (num_chains - 1) * jnp.sum((chain_mean - grand_mean) ** 2, axis=0)
    within = x.var(axis=0, ddof=1).mean(axis=0)
    var_hat = (num_kept - 1) / num_kept * within + between / num_kept
    return jnp.sqrt(var_hat / within)


def _max_r_hat_across_chains(per_draw_samples: List) -> object:
    """Per-(fit-batch, lineout) worst-case (max over active parameters) Gelman-Rubin R-hat across
    len(per_draw_samples) independent chains -- None if fewer than 2 (R-hat is meaningless for a single
    chain). Each element of per_draw_samples is a diff_params-shaped pytree (as returned by
    run_mcmc_for_fit_batches), leaves shaped (num_fit_batches, num_kept, batch_size, ...).

    Reduced to one number per lineout (the worst-mixing active parameter) rather than broken out
    per-parameter, matching acceptance_rate's granularity -- breaking it out per-parameter would need
    re-deriving get_filter_spec's (species, key) attribute-path labeling here, which mcmc_postprocess.py
    already does more naturally via its own active_keys/_physical_samples_for_fit_batch machinery.
    """
    num_chains = len(per_draw_samples)
    if num_chains < 2:
        return None
    leaf_lists = [jax.tree_util.tree_leaves(s) for s in per_draw_samples]
    if not leaf_lists[0]:
        return None
    per_leaf_r_hat = []
    for leaf_idx in range(len(leaf_lists[0])):
        stacked = jnp.stack([leaf_lists[c][leaf_idx] for c in range(num_chains)], axis=0)
        stacked = jnp.moveaxis(stacked, 2, 0)  # (num_kept, num_chains, num_fit_batches, batch_size, ...)
        per_leaf_r_hat.append(_gelman_rubin_r_hat(stacked))
    return jnp.max(jnp.stack(per_leaf_r_hat, axis=0), axis=0)  # (num_fit_batches, batch_size, ...)


def run_mcmc_pooled(
    config: Dict,
    loss_fns_by_draw: List[LossFunction],
    ts_params_list: List[ThomsonParams],
    batches_by_draw: List[List[Dict]],
    key: jax.Array,
) -> Tuple[object, object, List[Dict], object]:
    """
    Runs run_mcmc_for_fit_batches independently for each of the K independent chains (own PRNG subkey
    each), then pools all K chains' post-burn-in samples together along the sample axis. K chains may
    differ by calibration (config["other"]["calibration_uncertainty"]), by starting point
    (config["other"]["mcmc"]["init_dispersion_factor"]), by both, or -- with neither configured -- only
    by their own independent MH random-walk noise from an identical start; all are legitimate independent
    samples of the same overall posterior, so pooling them is valid regardless of which sources of
    variation are active. For K == 1 this is a no-op concatenation of a single chain's output.

    Each draw's LossFunction is built from different static config (a different FormFactor/IRF per
    draw), so -- unlike the fit-batch axis within one draw -- this loop cannot be vmapped into one
    compiled graph. Draws are still independent of each other (each only reads its own loss_fn/
    batch_list/subkey), so instead of a sequential Python loop they are dispatched to a thread pool, one
    draw per worker thread, round-robined across jax.local_devices() via jax.default_device -- on a
    multi-GPU host this is what actually keeps more than one GPU busy at once, since a plain for-loop here
    would run every draw's ~100+ small burn-in/sampling dispatches (see run_mcmc_for_batch's docstring)
    back-to-back on a single device while the rest sit idle. On a single-device host (e.g. CPU-only
    tests) every draw round-robins onto that same one device -- equivalent to (if not quite as fast as)
    a sequential loop; correctness doesn't depend on how many devices are actually available.

    Args:
        loss_fns_by_draw: length-K list of LossFunction instances, one per chain.
        ts_params_list: the fit-batches' best-fit weights (shared starting point for every chain, before
            any per-chain dispersion in run_mcmc_for_batch).
        batches_by_draw: length-K list, each a length-num_fit_batches list of batch dicts (one per
            fit-batch, built against that chain's possibly-rescaled data).
        key: PRNG key; split once per chain.

    Returns:
        pooled_samples: diff_params-shaped pytree, leaves shaped (num_fit_batches, K * num_kept, batch_size, ...).
        static_params: as returned by run_mcmc_for_fit_batches (from chain 0; identical in structure/value
            across chains for a fixed config).
        diagnostics_by_draw: list of length K, each chain's diagnostics dict (with the fit-batch axis).
        max_r_hat: per-(fit-batch, lineout) worst-case Gelman-Rubin R-hat across the K chains, or None
            when K < 2 (see _max_r_hat_across_chains).
    """
    keys = jr.split(key, len(loss_fns_by_draw))
    n_draws = len(loss_fns_by_draw)
    devices = jax.local_devices()

    def _run_one_draw(draw_index, loss_fn, batch_list, draw_key):
        progress_desc = f"MCMC draw {draw_index + 1}/{n_draws}" if n_draws > 1 else "MCMC"
        device = devices[draw_index % len(devices)]
        # default_device is scoped to this worker thread only (JAX's config context vars don't leak
        # across threads), so every array run_mcmc_for_fit_batches creates for this draw -- not just its
        # loss_fn/batch/ts_params inputs -- is allocated directly on `device` rather than migrating there
        # op-by-op.
        with jax.default_device(device):
            samples, static, diagnostics = run_mcmc_for_fit_batches(
                config, loss_fn, ts_params_list, batch_list, draw_key, progress_desc=progress_desc, pbar_position=draw_index
            )
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
        # Only safe to pull a concrete number out here, once run_mcmc_for_fit_batches' vmapped call has
        # actually returned -- doing this inside run_mcmc_for_batch itself (still mid-trace under
        # eqx.filter_vmap there) would raise a tracer-concretization error. Printed after every draw has
        # finished (rather than as each one completes) so concurrent draws don't interleave their lines.
        mean_accept = float(jnp.mean(diagnostics["acceptance_rate"]))
        print(f"{progress_desc} done: mean acceptance rate {mean_accept:.3f}")

    max_r_hat = _max_r_hat_across_chains(per_draw_samples)

    if len(per_draw_samples) == 1:
        pooled_samples = per_draw_samples[0]
    else:
        # each draw's samples have shape (num_fit_batches, num_kept, batch_size, ...); concatenate along
        # the num_kept axis (axis=1) to pool across draws while keeping the fit-batch axis (axis=0) intact.
        pooled_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *per_draw_samples)

    return pooled_samples, static_params, diagnostics_by_draw, max_r_hat
