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
import time
import warnings
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
    # Chunk size burn-in is reported/checkpointed in (tqdm granularity) -- purely cosmetic now that
    # adaptation itself (_ram_update) runs every single step rather than once per window; does not affect
    # the sampler's behavior, only how often the progress bar updates.
    "adapt_every": 50,
    "target_accept": 0.234,
    # Vihola's (2012) Robust Adaptive Metropolis (RAM) vanishing-gain exponent: eta_i =
    # min(1, n_active * step_index^-adapt_gamma) -- see _ram_update. 0.6 is within Vihola's recommended
    # (0.5, 1] range (larger = faster-decaying adaptation, more stable but slower to converge; 0.5 is the
    # slowest-decaying choice consistent with the theory's ergodicity guarantees).
    "adapt_gamma": 0.6,
    # Fallback proposal step used only when there's no usable curvature information at all (see
    # _seed_step_scale_default -- used whenever use_laplace_seed is False, or the whole Laplace Hessian
    # computation raises structurally), applied as-is in the same unconstrained/logit space diff_params
    # already lives in -- i.e. a flat, uncorrelated number shared by every active leaf/lineout, not scaled
    # by each leaf's own position or physical [lb, ub] range. _seed_step_scale_from_laplace no longer has
    # a per-entry version of this fallback: an individually-degenerate leaf's own row/column is instead
    # regularized in place (see _regularized_proposal_cholesky), preserving its measured coupling to every
    # other leaf, rather than replaced outright.
    #
    # A position-aware version of this field (as a dimensionless fraction of each leaf's own physical
    # range, converted internally via sigmoid'(normed_x)) was tried and reverted: it silently changed
    # this field's units under the same name, so a deck carrying forward a tuned flat value (e.g.
    # 0.0001, tuned small deliberately) got reinterpreted as a *fraction of range* instead, inflating the
    # effective logit-space step by ~2-3 orders of magnitude at typical positions and collapsing
    # acceptance to ~0 (see git history/PR discussion for the production regression this caused). Keep
    # this flat: any future revisit of position-awareness needs its own differently-named field,
    # precisely so an old deck's `init_step_scale` can never be silently reinterpreted again.
    "init_step_scale": 0.1,
    "use_laplace_seed": True,
    # Multiplier on the (Laplace-seeded or fallback) per-lineout/per-parameter step scale,
    # used to perturb each chain's own starting point before burn-in begins (see run_mcmc_for_batch).
    # 0.0 (default) means every chain starts at the exact best fit, matching pre-multi-chain behavior
    # exactly. Set > 0 when running several chains (config["other"]["calibration_uncertainty"]
    # ["num_draws"] > 1) purely for dispersed starts / a meaningful R-hat -- see mcmc.rst.
    "init_dispersion_factor": 0.0,
    "seed": 0,
    "save_samples": True,
    # postprocess.laplace.get_sigmas' Hessian is now taken w.r.t. diff_params only (the same restriction
    # _seed_step_scale_from_laplace above already applies), so the multi-hundred-GB allocation this used
    # to risk on an ordinary fit whose electron distribution function carries a sizeable fixed
    # interpolation table is fixed. It still doesn't support "fe" active (get_sigmas raises
    # NotImplementedError in that case, caught by recalculate_with_chosen_weights same as any other
    # failure). The comparison plot defaults to off mainly to keep this postprocessor's own scope
    # minimal, not for memory-safety reasons anymore -- opt in with compare_to_laplace: true freely.
    "compare_to_laplace": False,
}


def _mcmc_cfg(config: Dict) -> Dict:
    """config["other"]["mcmc"], with every field defaulted so older decks (or decks that never
    configured this feature) work unchanged. Warns (once per unique set of defaulted field names, per
    Python's own warnings-dedup) about any field not set in config['other']['mcmc'], since a config
    silently using a built-in default the deck's author didn't actually intend -- a typo, a stale/renamed
    key, or genuinely forgetting to set it -- otherwise has no visible signal at all. See init_step_scale's
    comment in _DEFAULTS for the production regression a silent default fallback caused after exactly this
    kind of mismatch (a renamed config key)."""
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
    """One Gaussian random-walk proposal, jointly correlated across every active leaf of diff_params,
    scaled per lineout by step_scale -- the (batch_size, n_active, n_active) Cholesky factor of that
    lineout's proposal covariance (see _regularized_proposal_cholesky/_seed_step_scale_default). Active
    leaves are flattened into one (batch_size, n_active) block, in jax.tree_util.tree_flatten's own
    leaf order, stepped as proposal = current + step_scale @ z (z ~ N(0, I) per lineout), then
    unflattened back.

    Unlike the independent per-leaf proposal this replaced, a step in one leaf is now drawn correlated
    with every other leaf according to that lineout's actual local covariance -- letting the walk move
    efficiently along real degenerate/correlated ridges (e.g. a near-exact pairwise degeneracy between
    two leaves, or one leaf's own curvature being weak but substantially coupled to another's) instead of
    proposing independent per-axis moves that almost always land off of them.
    """
    leaves, treedef = jax.tree_util.tree_flatten(diff_params)
    stacked = jnp.stack(leaves, axis=-1)  # (batch_size, n_active)
    z = jr.normal(key, stacked.shape)
    delta = jnp.einsum("bij,bj->bi", step_scale, z)
    new_stacked = stacked + delta
    new_leaves = [new_stacked[..., i] for i in range(len(leaves))]
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


def _ram_update(
    step_scale: jnp.ndarray,
    z: jnp.ndarray,
    alpha: jnp.ndarray,
    target_accept: float,
    step_index: jnp.ndarray,
    adapt_gamma: float,
    n_active: int,
) -> jnp.ndarray:
    """One step of Vihola's (2012) Robust Adaptive Metropolis (RAM) rank-one update to the proposal
    Cholesky factor -- replaces the separate Robbins-Monro magnitude update + windowed empirical-shape
    re-estimation this module used previously. RAM adapts every direction of the joint proposal
    independently and automatically from the *continuous* per-step MH acceptance probability, with no
    separate "magnitude" vs "shape" decomposition and no eigenvalue floor or determinant-normalization
    step of its own:

        Sigma_i = S_{i-1} (I + eta_i * (alpha_i - target_accept) * z z^T / ||z||^2) S_{i-1}^T
        S_i = cholesky(Sigma_i)

    where z is the whitened direction actually drawn this step (z ~ N(0, I); the proposal was
    current + S_{i-1} @ z) and eta_i = min(1, n_active * step_index^-adapt_gamma) is a vanishing gain
    sequence. A direction that keeps getting accepted (alpha_i > target_accept) grows on its own; one
    that keeps getting rejected shrinks -- with no shared renormalization coupling it to any other
    direction. The previous determinant-1 shape normalization was found to force a near-degenerate
    direction's regularization to inflate every *other*, genuinely well-constrained direction along with
    it (confirmed both as a production runaway and, separately, as a NaN-producing Cholesky failure on an
    under-sampled window); RAM has no such coupling by construction.

    Guaranteed positive-definite: alpha_i in [0,1] and target_accept in (0,1) bound
    (alpha_i - target_accept) in (-target_accept, 1-target_accept), and eta_i <= 1, so the inner matrix's
    one nontrivial eigenvalue (1 + eta_i*(alpha_i-target_accept)) stays strictly greater than
    (1 - target_accept) > 0 -- no eigenvalue floor or clipping needed the way the Hessian-based Laplace
    seed (_regularized_proposal_cholesky) needs one.

    Never inverts a Hessian, so a saddle-point diagonal entry (e.g. Ti's or brem_c's negative own
    curvature at the reported best fit) is not a concern here -- this only ever consumes an initial S_0
    (e.g. from _seed_step_scale_from_laplace) as a reasonable starting point, not as something that has
    to already be correct.

    Args:
        step_scale: S_{i-1}, (batch_size, n_active, n_active) Cholesky factor.
        z: this step's whitened proposal draw, (batch_size, n_active) -- see _propose.
        alpha: this step's continuous MH acceptance probability per lineout, (batch_size,) -- NOT the
            realized 0/1 accept/reject event; RAM's derivation targets E[alpha] = target_accept.
        step_index: how many RAM adaptation steps have been taken so far including this one (>= 1), as a
            traced array -- continues across burn-in chunks rather than resetting each one, since eta_i's
            vanishing-gain guarantee depends on it increasing monotonically across the whole burn-in.
        n_active: number of active scalar leaves (dimension d in Vihola's eta_i formula).

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
    """Runs n_steps of propose+accept/reject via jax.lax.scan, adapting step_scale every single step via
    the RAM rank-one update (_ram_update). Unlike _run_window (used for the frozen-step_scale sampling
    phase), step_scale here changes step by step rather than once per window, so this cannot share
    _run_window's scan body.

    step_offset lets this be called once per (adapt_every-sized) chunk purely for tqdm progress-bar
    granularity, while RAM's own adaptation stays genuinely continuous across chunks: step_offset is the
    number of RAM steps already taken before this call, so this chunk's steps continue that count
    (step_offset + 1, step_offset + 2, ...) rather than restarting eta_i from step 1 every chunk -- see
    _ram_update's docstring for why that continuity matters. Pass step_offset as a traced jnp array (not
    a bare Python int/float) at the call site: a bare Python value would be treated as a static argument
    by @eqx.filter_jit and trigger a fresh trace/compile on every distinct value, i.e. every chunk.

    @eqx.filter_jit matters here for the same reason documented on _run_window: without it, every
    burn-in chunk's call would retrace the whole forward-model graph from Python even though the
    compiled XLA kernel underneath is reused.

    Returns (diff_params, log_post, step_scale, accept_count) -- accept_count is the number of accepted
    steps per lineout over this chunk (realized 0/1 decisions, matching _run_window's convention), not
    the continuous alpha RAM itself adapts from.
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
        # continuous MH acceptance probability alpha_i = min(1, ratio) -- see _ram_update's docstring for
        # why RAM's derivation needs this, not just the realized 0/1 accept/reject event.
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


def _seed_step_scale_default(diff_params, init_step_scale: float) -> jnp.ndarray:
    """Pure heuristic proposal-scale seed: a flat, uncorrelated `init_step_scale` per leaf/lineout --
    i.e. the Cholesky factor of a diagonal covariance diag(init_step_scale^2) -- in the
    unconstrained/logit space diff_params already lives in. No dependency on the Hessian/Laplace
    machinery -- always available, always succeeds. Used whenever use_laplace_seed is False, or when
    _seed_step_scale_from_laplace fails structurally.

    Returns:
        L: (batch_size, n_active, n_active) Cholesky factor -- see _propose.
    """
    leaves = jax.tree_util.tree_leaves(diff_params)
    n = len(leaves)
    if n == 0:
        return jnp.zeros((0, 0, 0))
    batch_size = leaves[0].shape[0]
    return init_step_scale * jnp.broadcast_to(jnp.eye(n), (batch_size, n, n))


_LAPLACE_EIGVAL_FLOOR = 1e-3
"""Floor applied to the eigenvalues of the *normalized* (correlation-scaled) per-lineout Hessian before
it's inverted into a proposal covariance (see _regularized_proposal_cholesky) -- keeps the result
positive-definite even when the raw Hessian isn't (e.g. a leaf whose own curvature is negative at the
reported best fit, observed in production for weakly-identified parameters). In normalized units, a
well-conditioned, uncorrelated direction has eigenvalue exactly 1 by construction, so 1e-3 means a
clipped direction is treated as roughly 30x wider (sqrt(1/1e-3) ~ 32) than a typical well-conditioned
direction -- deliberately generous rather than conservative, since a genuinely unconstrained parameter's
true uncertainty should be allowed to come out large; sigmoid saturation at the physical [lb, ub] bound
caps how far this can actually push a proposal in physical terms regardless of how large a step this
permits in logit space.

Only needed here, for the Hessian-based Laplace seed (which can be genuinely non-positive-definite at a
saddle) -- the RAM adaptation that takes over from this seed during burn-in (_ram_update) needs no
eigenvalue floor of its own; it's positive-definite by construction."""


def _stack_hessian(hess, diff_params) -> jnp.ndarray:
    """Flattens a per-lineout Hessian (as returned by LossFunction.h_loss_wrt_params_per_lineout -- a
    diff_params-shaped pytree of diff_params-shaped subtrees, each *leaf* of which is itself a
    (batch_size,) array) into one dense (batch_size, n_active, n_active) array, ordered to match
    jax.tree_util.tree_leaves(diff_params). Mirrors postprocess.laplace.get_sigmas' identical
    pytree-to-array flattening."""
    target_structure = jax.tree_util.tree_structure(diff_params)
    rows = jax.tree_util.tree_leaves(hess, is_leaf=lambda node: jax.tree_util.tree_structure(node) == target_structure)
    n = len(rows)
    blocks = [jax.tree_util.tree_leaves(row) for row in rows]  # blocks[a][b]: (batch_size,)
    rows_stacked = [jnp.stack([blocks[a][b] for b in range(n)], axis=-1) for a in range(n)]  # each: (batch_size, n)
    return jnp.stack(rows_stacked, axis=-2)  # (batch_size, n_active, n_active); [:, a, b] = d2L/d(a)d(b)


def _regularized_proposal_cholesky(H: jnp.ndarray, rr_factor: float) -> jnp.ndarray:
    """Turns a per-lineout Hessian block (batch_size, n_active, n_active); see _stack_hessian) into the
    Cholesky factor of a valid (positive-definite) proposal covariance, scaled by the Roberts-Rosenthal
    rr_factor, even when the raw Hessian itself is not positive-definite -- which happens in production
    for weakly-identified parameters (e.g. an ion temperature whose own diagonal curvature was measured
    negative at the reported best fit: a saddle direction, not a true local minimum, because that
    direction is only identifiable jointly with another parameter -- see _seed_step_scale_from_laplace's
    docstring).

    Regularizes by eigenvalue-clipping the *normalized* (correlation-scaled) Hessian rather than the raw
    one: the diagonal entries here can span many orders of magnitude (a real production lineout showed
    roughly -700 for one leaf alongside 6e7 for another), so a single absolute eigenvalue floor applied
    to the raw Hessian has no scale-consistent meaning across leaves. Normalizing first (dividing by each
    leaf's own sqrt(|H_ii|) -- the matrix analogue of turning a covariance into a correlation matrix)
    puts every well-conditioned direction's eigenvalue at O(1) regardless of the leaf's raw curvature
    scale, so one small fixed floor (_LAPLACE_EIGVAL_FLOOR) is meaningful for every leaf simultaneously.

    Crucially, this preserves the Hessian's eigenvectors -- i.e. *which combinations* of parameters form
    a near-degenerate direction -- rather than discarding that structure the way a flat per-entry
    fallback would, since that structure is exactly what a joint proposal needs to move efficiently along
    real degenerate ridges instead of proposing independent per-parameter steps that almost always land
    off of them.

    Args:
        H: per-lineout Hessian block, (batch_size, n_active, n_active), as returned by _stack_hessian.
        rr_factor: the Roberts-Rosenthal 2.38/sqrt(n_active) optimal-scaling factor (same convention as
            the diagonal-only seeding this replaced).

    Returns:
        L: (batch_size, n_active, n_active) lower-triangular Cholesky factor, one per lineout, such that
            L @ L.T is the proposal covariance to draw MH steps from (see _propose).
    """
    diag = jnp.diagonal(H, axis1=-2, axis2=-1)  # (batch_size, n_active)
    d = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(diag), 1e-300))  # floors an exactly-zero diagonal entry
    h_norm = H * d[:, :, None] * d[:, None, :]  # D @ H @ D per lineout, D = diag(d)

    eigvals, eigvecs = jnp.linalg.eigh(h_norm)  # ascending eigvals; (batch_size, n), (batch_size, n, n)
    eigvals_reg = jnp.maximum(eigvals, _LAPLACE_EIGVAL_FLOOR)

    # Sigma = rr_factor^2 * H_reg^-1, built directly from the regularized, normalized eigendecomposition
    # without ever forming H_reg or inverting it explicitly: H_norm = D @ H @ D (D = diag(d)), so
    # H = D^-1 @ H_norm @ D^-1, and therefore H^-1 = D @ H_norm^-1 @ D = D @ Q @ diag(1/eigvals) @ Q.T @ D.
    # Scale the eigenvectors by d itself here (NOT 1/d) -- using 1/d inverts the whole result (a
    # well-constrained leaf's huge curvature would turn into a huge proposal variance instead of a tiny
    # one), which is exactly what production showed: every eigendirection oversized from the very first
    # seed, before any burn-in adaptation ran at all.
    scaled_eigvecs = eigvecs * d[:, :, None]  # D @ Q
    sigma = jnp.einsum("bik,bk,bjk->bij", scaled_eigvecs, (rr_factor**2) / eigvals_reg, scaled_eigvecs)
    sigma = 0.5 * (sigma + jnp.swapaxes(sigma, -1, -2))  # symmetrize away float roundoff
    return jnp.linalg.cholesky(sigma)


def _seed_step_scale_from_laplace(loss_fn: LossFunction, static_params, batch: Dict, diff_params) -> jnp.ndarray:
    """Seeds the initial per-lineout proposal covariance (as a Cholesky factor) from the full per-lineout
    Laplace/Hessian covariance, scaled by the standard Roberts-Rosenthal 2.38/sqrt(d) optimal-scaling
    factor (d = number of active scalar leaves) -- see _regularized_proposal_cholesky.

    Unlike the diagonal-only version this replaced, off-diagonal (cross-parameter) terms are now used
    directly, via LossFunction.h_loss_wrt_params_per_lineout -- the same low-memory per-lineout Hessian
    postprocess.laplace.get_sigmas uses, restricted to diff_params only for the same reason that function
    and h_loss_wrt_params_per_lineout's own docstring document (hessian-ing the full parameter tree pulls
    in every fixed array the model carries, e.g. a large distribution-function lookup table, and has been
    observed to attempt a multi-hundred-GB allocation on an ordinary fit). Reusing that method also means
    this no longer needs its own bespoke diagonal-extraction trick (the jax.lax.map/jvp sweep the
    previous version used) -- h_loss_wrt_params_per_lineout already handles the low-memory computation,
    and this only adds the regularize-and-Cholesky step on top.

    Motivation: two independent, real degeneracies observed in a real production fit made the
    diagonal-only proposal badly inefficient even though each individual leaf's own curvature looked
    reasonable in isolation -- a near-exact pairwise degeneracy between two leaves (off-diagonal Hessian
    entry measured at -0.9998 of sqrt(h_aa * h_bb), i.e. essentially one degenerate direction shared by
    only those two), and a leaf with genuinely poor identifiability on its own (own diagonal curvature
    measured negative -- not just small -- at a real fitted point, but substantially coupled to another
    leaf, and more weakly to two others, through the off-diagonal terms). An independent per-leaf
    proposal can only ever guess randomly at these combinations, landing off the true (correlated) ridge
    almost every time; seeding from -- and proposing along -- the actual joint covariance lets the walk
    move efficiently along it instead, while still correctly reporting a large marginal uncertainty for a
    poorly-identified leaf and comparatively tight ones for the leaves it's coupled to.

    See _regularized_proposal_cholesky for how a Hessian that isn't positive-definite is handled:
    eigenvalue-clipped in normalized (correlation-scaled) space so the fix is meaningful across leaves
    whose raw curvature can differ by many orders of magnitude, while preserving the Hessian's
    eigenvectors -- i.e. which *combinations* of parameters actually form each near-degenerate direction
    -- rather than discarding that structure the way a per-entry flat fallback would. There is
    accordingly no per-entry fallback here any more: an individually-degenerate leaf's own row/column is
    regularized in place, preserving its measured coupling to every other leaf, rather than replaced
    outright.

    Returns:
        L: (batch_size, n_active, n_active) Cholesky factor of the proposal covariance -- see _propose.
    """
    n = len(jax.tree_util.tree_leaves(diff_params))
    if n == 0:
        return jnp.zeros((0, 0, 0))

    hess = loss_fn.h_loss_wrt_params_per_lineout(diff_params, static_params, batch)
    H = _stack_hessian(hess, diff_params)  # (batch_size, n_active, n_active)
    rr_factor = 2.38 / jnp.sqrt(float(n))
    return _regularized_proposal_cholesky(H, rr_factor)


@eqx.filter_jit
def _seed_step_scale(loss_fn: LossFunction, static_params, batch: Dict, diff_params, mcmc_cfg: Dict):
    """Computes one fit-batch's initial proposal step scale: Laplace-seeded (via
    _seed_step_scale_from_laplace) if mcmc_cfg["use_laplace_seed"], falling back to the flat
    _seed_step_scale_default whenever that's off or the Hessian seed fails structurally.

    Deliberately a standalone, @eqx.filter_jit-compiled function (not inlined into run_mcmc_for_batch)
    so run_mcmc_for_fit_batches can call it in a plain Python loop, once per fit-batch, *before*
    vmapping the rest of run_mcmc_for_batch across every fit-batch -- see run_mcmc_for_batch's
    step_scale docstring for why: fusing this Hessian computation into that fit-batch vmap has been
    observed to multiply its (otherwise small, diff_params-only) memory cost by the fit-batch count,
    since nested vmap(hessian(...)) without its own enclosing jit prevents XLA from fusing/reusing
    buffers across the batch. jit-compiling here instead lets every fit-batch in that Python loop reuse
    one compiled executable (same diff_params/static_params shapes every time, only the values differ).
    """
    step_scale = None
    if mcmc_cfg["use_laplace_seed"]:
        try:
            step_scale = _seed_step_scale_from_laplace(loss_fn, static_params, batch, diff_params)
        except Exception as e:
            print(f"Laplace-seeded step scale failed, falling back to flat init_step_scale: {type(e).__name__}: {e}", flush=True)
            step_scale = None
    if step_scale is None:
        step_scale = _seed_step_scale_default(diff_params, mcmc_cfg["init_step_scale"])
    return step_scale


def run_mcmc_for_batch(
    config: Dict,
    loss_fn: LossFunction,
    ts_params: ThomsonParams,
    batch: Dict,
    key: jax.Array,
    progress_desc: str = "MCMC",
    pbar_position: int = 0,
    step_scale=None,
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
        step_scale: optional precomputed initial step scale (the (batch_size, n_active, n_active)
            Cholesky factor of a per-lineout proposal covariance -- see _propose), overriding
            use_laplace_seed/init_step_scale entirely when given. run_mcmc_for_fit_batches passes this in,
            computed sequentially per fit-batch via _seed_step_scale *before* vmapping this function
            across every fit-batch of a shot -- see _seed_step_scale's docstring for why computing the
            Laplace-seeded Hessian *inside* that fit-batch vmap is a memory-blowup risk. Leave None for a
            direct, single-fit-batch call (e.g. tests), which seeds internally exactly as before.

    Returns:
        samples: a diff_params-shaped pytree; each leaf has shape (num_kept, batch_size, ...), where
            num_kept = ceil((num_steps - burn_in) / thin).
        static_params: the non-sampled complement of ts_params (eqx.partition's static half), needed by
            the caller to recombine samples into full parameter values via eqx.combine.
        diagnostics: {"acceptance_rate": array (batch_size,), "final_step_scale": step_scale Cholesky
            factor, (batch_size, n_active, n_active)}.

    Raises:
        NotImplementedError: if config["parameters"]["electron"]["fe"]["active"] is true (see module
            docstring).
    """
    check_fe_inactive(config["parameters"])
    mcmc_cfg = _mcmc_cfg(config)

    filter_spec = get_filter_spec(config["parameters"], ts_params)
    diff_params, static_params = eqx.partition(ts_params, filter_spec)

    if step_scale is None:
        step_scale = _seed_step_scale(loss_fn, static_params, batch, diff_params, mcmc_cfg)

    # Nudges this chain's own starting point away from the shared best fit, so that when several
    # independent chains are pooled (config["other"]["calibration_uncertainty"]["num_draws"] > 1) they
    # don't all begin at literally the same point -- see run_mcmc_pooled's R-hat computation, which needs
    # genuinely independent chains to be meaningful.
    init_dispersion_factor = float(mcmc_cfg.get("init_dispersion_factor", 0.0))
    if init_dispersion_factor > 0:
        key, disperse_key = jr.split(key)
        dispersed_scale = init_dispersion_factor * step_scale
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
    # Burn-in adapts step_scale every single step via RAM (_ram_update/_run_ram_window), not once per
    # window -- adapt_every here only sets how often the progress bar updates and how large each
    # individual jax.lax.scan chunk is, not the adaptation's own behavior. step_offset (passed as a
    # traced array, not a bare Python int -- see _run_ram_window's docstring) keeps RAM's vanishing-gain
    # step counter continuous across chunks.
    for window_index in range(n_windows):
        burn_key, window_key = jr.split(burn_key)
        diff_params, log_post, step_scale, accept_count = _run_ram_window(
            window_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, adapt_every,
            jnp.asarray(float(window_index * adapt_every)), mcmc_cfg["target_accept"], mcmc_cfg["adapt_gamma"],
        )
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
    Runs run_mcmc_for_batch across every fit-batch of a single calibration draw. Every fit-batch shares
    the exact same compiled LossFunction/config -- only the data slice and starting weights differ, both
    ordinary array-valued inputs -- so the sampling itself is a single eqx.filter_vmap over the fit-batch
    axis rather than a Python loop, avoiding a separate trace/compile per fit-batch.

    The initial step_scale is the one exception: it's seeded sequentially, one fit-batch at a time, via
    _seed_step_scale, *before* that vmap -- see _seed_step_scale's docstring for why. Fusing its Hessian
    computation into the fit-batch vmap has been observed to multiply its (otherwise small, diff_params-
    only) memory cost by the fit-batch count on real multi-lineout shots (a nested vmap(hessian(...)) with
    no enclosing jit of its own prevents XLA from fusing/reusing buffers across the batch), even though
    _seed_step_scale_from_laplace's restriction to diff_params already keeps any *single* fit-batch's
    Hessian cheap on its own. _seed_step_scale is itself jit-compiled, so this loop still only compiles
    once (every fit-batch shares the same diff_params/static_params shapes) and just replays that one
    executable per fit-batch.

    Returns the same three outputs as run_mcmc_for_batch, each with an extra leading fit-batch axis on
    every array leaf (size len(ts_params_list)); static_params/diagnostics' non-array leaves are passed
    through unbatched by eqx.filter_vmap since they are identical across fit-batches.
    """
    n_fit_batches = len(ts_params_list)
    mcmc_cfg = _mcmc_cfg(config)

    # No tqdm here deliberately: this loop's first iteration includes _seed_step_scale's one-time JIT
    # compile (which, for an expensive physics forward model, can itself take a while), and every
    # iteration after that should be near-instant (same compiled executable, reused). Printing each
    # iteration's own wall time -- rather than a bar that would sit at 0% through that entire first-call
    # compile -- makes that compile-vs-replay split visible directly, which matters for diagnosing
    # whether a slow startup here is compilation (one-time) or genuinely per-fit-batch cost (recurring).
    print(f"{progress_desc}: seeding step scale for {n_fit_batches} fit-batch(es)...", flush=True)
    step_scales = []
    for i, (ts_params, batch) in enumerate(zip(ts_params_list, batch_list)):
        t0 = time.time()
        filter_spec = get_filter_spec(config["parameters"], ts_params)
        diff_params, static_params = eqx.partition(ts_params, filter_spec)
        step_scales.append(_seed_step_scale(loss_fn, static_params, batch, diff_params, mcmc_cfg))
        jax.block_until_ready(step_scales[-1])
        print(f"{progress_desc}: fit-batch {i + 1}/{n_fit_batches} step scale seeded in {time.time() - t0:.1f}s", flush=True)
    stacked_step_scale = _stack_step_scales(step_scales)

    stacked_ts_params = _stack_ts_params(ts_params_list)
    stacked_batch = _stack_batches(batch_list)
    keys = jr.split(key, n_fit_batches)

    def _one(ts_params, batch, k, step_scale):
        return run_mcmc_for_batch(
            config, loss_fn, ts_params, batch, k, progress_desc=progress_desc, pbar_position=pbar_position,
            step_scale=step_scale,
        )

    return eqx.filter_vmap(_one)(stacked_ts_params, stacked_batch, keys, stacked_step_scale)


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
