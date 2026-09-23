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
from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
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
    # Threshold, in robust-SD units, for flagging one chain's per-lineout posterior mean as an outlier
    # relative to the other independent chains pooled for that lineout (config["other"]
    # ["calibration_uncertainty"]["num_draws"] > 1 required -- meaningless for a single chain) -- see
    # mcmc_postprocess._mad_flagged_chains. A flagged chain's samples are still saved in full (raw
    # per-chain data is never discarded), just excluded from that lineout's quoted mean/std/covariance and
    # summary plots, since a chain that settled on a genuinely different region of parameter space (e.g.
    # a different point along a real degenerate ridge -- see _regularized_proposal_cholesky's docstring)
    # would otherwise silently widen or bias the reported uncertainty. "Robust-SD" here is 1.4826 * MAD
    # (the usual normal-consistent scaling), so 3.5 reproduces Iglewicz & Hoaglin's standard modified
    # z-score outlier threshold rather than an arbitrary number. A genuinely resolved multimodal posterior
    # (several chains each converged to one of a few real, well-populated solutions -- as opposed to one
    # or two chains stuck alone) is deliberately *not* what this is meant to catch: a well-populated second
    # mode inflates its own MAD enough that this rarely flags it. See max_dropped_chain_fraction for what
    # happens when it does anyway (or when too many chains are flagged for another reason).
    "chain_outlier_mad_scale": 3.5,
    # Threshold for _within_chain_r_hat's *split* R-hat: a chain whose own first half and second half of
    # post-burn-in samples don't agree (in the same Gelman-Rubin sense _max_r_hat_across_chains uses
    # between chains) hasn't reached a stationary distribution within its own sampling budget at all -- a
    # different, more fundamental failure than settling somewhere different from the other chains (see
    # chain_outlier_mad_scale), and invisible to any cross-chain check, since two independently-wandering
    # chains can coincidentally look like they agree. Slightly more lenient than the 1.01 typically quoted
    # for many-chain R-hat, since split-R-hat here only ever compares two halves of one chain -- noisier by
    # construction -- not many independent ones.
    "within_chain_r_hat_threshold": 1.1,
    # Hard cap, as a fraction of the total chain count, on how many chains this postprocessor will write
    # off for a single lineout -- for either reason (chain_outlier_mad_scale or
    # within_chain_r_hat_threshold, combined under one shared budget) -- before giving up on that lineout
    # entirely rather than quietly reporting a mean/std/covariance from whatever's left. A genuinely
    # poorly-constrained parameter (e.g. Ti, deliberately sampled with a wide uncertainty rather than an
    # artificially tight one -- the whole point of running MCMC instead of a local Hessian/Laplace
    # approximation) can legitimately spread even *converged* chains' means fairly widely, so this exists
    # to draw the line between "some chains are honestly further out" and "too much of this lineout is
    # unreliable to trust any of it" -- see mcmc_postprocess._finalize_chain_selection. Exceeding this cap
    # marks the whole lineout's mean/std/covariance NaN rather than truncating to the least-bad subset:
    # silently capping was tried and found to hide exactly the cases (e.g. most chains failing
    # within_chain_r_hat_threshold together) this is supposed to surface, not fix.
    "max_dropped_chain_fraction": 0.2,
    # Master switch for block Metropolis-within-Gibbs (see the module-level docstring above
    # _run_block_ram_window). When True (default) and use_laplace_seed succeeds, every active leaf that
    # participates (block_gibbs_component_threshold or more) in some lineout's substantially-negative
    # normalized-Hessian eigendirection (eigenvalue <= block_gibbs_eigval_threshold) is sampled in its own
    # separately-adapted block, isolating it from the rest of that lineout's parameters' own RAM adaptation.
    # False restores the single joint-block proposal this replaced (still eigenvalue-regularized, still
    # RAM-adapted -- just one shared accept/reject decision per step across every active leaf, as before
    # block-Gibbs existed).
    "block_gibbs": True,
    # Eigenvalue threshold (in the same normalized/correlation-scaled Hessian used by
    # _regularized_proposal_cholesky) below which a direction is flagged for block Metropolis-within-Gibbs
    # -- see _detect_problem_leaves. Deliberately a *separate*, stricter, and more negative threshold than
    # _LAPLACE_EIGVAL_FLOOR (1e-3, used for eigenvalue *clipping*): clipping must catch every non-positive-
    # definite or merely-small-positive direction to keep the proposal covariance valid at all, but block
    # Gibbs should trigger only for a genuine saddle (substantially negative curvature, not just "small" or
    # "barely negative") -- a real production case (an ion temperature whose whole-model convergence was
    # confirmed, via a controlled deactivation test, to depend on isolating it) showed normalized
    # eigenvalues around -1.2 to -1.6, while a mildly-negative eigenvalue around -0.02 to -0.05 on an
    # under-converged test fit was found to make block Gibbs converge *more* slowly within a short burn-in
    # budget than the already-regularized joint proposal it would replace, with no corresponding benefit --
    # i.e. clipping alone already handles a merely-marginal direction adequately, and paying block Gibbs'
    # extra complexity (a second independently-adapted proposal, slower to jointly explore correlations
    # between the two blocks) is only worth it for a direction clipping alone measurably fails on. -0.1 sits
    # comfortably between those two observed regimes; revisit together with block_gibbs_component_threshold
    # if a shot shows a genuine problem direction with a milder eigenvalue than -0.1.
    "block_gibbs_eigval_threshold": -0.1,
    # Minimum |eigenvector component| (in the same normalized/correlation-scaled Hessian eigenbasis
    # _regularized_proposal_cholesky uses) for a leaf to be considered part of a flagged near-degenerate
    # direction -- see _detect_problem_leaves. 0.3 is a modest majority-share of an eigenvector's L2-normalized
    # mass (recall sum of squared components = 1, so 0.3^2 = 0.09, i.e. this leaf alone explains at least
    # ~9% of that direction's variance) -- deliberately permissive (flags real participants, e.g. Ti and a
    # correlated ne, rather than only a direction's single most-dominant leaf) since under-flagging leaves a
    # genuine problem parameter in the well-conditioned block, where it can still dominate that block's own
    # shared accept/reject signal exactly the way it dominated the original unblocked one.
    "block_gibbs_component_threshold": 0.3,
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


# ======================================================================================================
# Block Metropolis-within-Gibbs
# ======================================================================================================
# An unblocked joint step (_run_ram_window above) proposes and accepts/rejects every active leaf together,
# from one shared random draw against one shared MH decision. That is fine when every leaf's own local
# curvature is roughly comparable, but production fits showed a real, recurring failure mode when it
# isn't: a leaf with near-zero or negative Hessian curvature (e.g. an ion temperature identifiable only
# weakly, or jointly with another parameter -- see _regularized_proposal_cholesky's docstring) gets a
# hugely oversized Laplace-seeded proposal in that leaf's own direction. Because the *joint* step's
# accept/reject decision is a single coin flip shared by every leaf, that one oversized direction ends up
# dictating whether the whole step is accepted almost regardless of what every other, well-conditioned
# leaf proposed -- starving those other leaves' own RAM adaptation of a meaningful, direction-specific
# learning signal (confirmed directly: a controlled test that simply deactivated the offending leaf(s)
# recovered clean, fast-mixing convergence in every other parameter, on two different lineouts of a real
# shot, one of which had *two* such leaves).
#
# Block Metropolis-within-Gibbs (see e.g. Turek, de Valpine, Paciorek & Anderson-Bergman, "Automated
# Parameter Blocking for Efficient Markov-Chain Monte Carlo Sampling," arXiv:1503.05621 (2015), for
# correlation-based automated blocking in the general MCMC literature) is the standard fix: partition the
# active leaves into blocks, and within one outer MCMC step, propose and accept/reject each block
# *separately*, each against its own MH decision and its own independently RAM-adapted proposal covariance
# (_ram_update, unchanged -- only what set of leaves gets proposed and adapted together changes). This
# module uses exactly two blocks, chosen once per run from the Hessian eigenstructure
# (_block_indices_from_hessians): a "problem" block containing every leaf that meaningfully participates in
# some lineout's near-zero/negative normalized-Hessian eigendirection, and a "well-conditioned" block
# containing everything else. The *detection criterion* (threshold the normalized-Hessian eigenvalues,
# union the flagged leaves' components across a shot) is this module's own synthesis of standard
# ingredients, not itself a technique drawn from the blocking literature above -- Turek et al. use
# posterior-sample correlation to choose blocks online, which this module does not attempt.
#
# Future alternative -- geodesic/Riemannian-manifold-aware sampling: block Gibbs treats the near-degenerate
# direction as "isolate and let it wander" rather than addressing *why* naive Euclidean random-walk steps
# struggle there in the first place -- the underlying issue is that the Hessian/Fisher-information matrix
# defines a natural (Riemannian) metric on parameter space, and that metric's eigenvalue spectrum for this
# kind of physics model is known to span many orders of magnitude with a roughly log-linear decay, a
# phenomenon named and studied as "sloppy models" in the physics/systems-biology literature (Transtrum,
# Machta & Sethna, "Geometry of nonlinear least squares with applications to sloppy models and
# optimization," Phys. Rev. E 83, 036701 (2011); Sethna et al., "Sloppiness and Emergent Theories in
# Physics, Biology, and Beyond," arXiv:1501.07668) -- exactly the signature this module's own Hessian
# eigenvalue spread shows in practice (roughly 1e-9 to 1e-2 across active leaves, ~7 orders of magnitude).
# The principled fix in that framework is to move along *geodesics* of that Riemannian metric rather than
# straight Euclidean lines, e.g. Geodesic Monte Carlo (Byrne & Girolami, "Geodesic Monte Carlo on Embedded
# Manifolds," arXiv:1301.6064 (2013)) or Riemannian-manifold HMC with the SoftAbs metric (which
# symmetrically regularizes indefinite curvature rather than one-sided-clipping it the way
# _regularized_proposal_cholesky's eigenvalue floor does -- Betancourt, "A General Metric for Riemannian
# Manifold Hamiltonian Monte Carlo," arXiv:1212.4693). This is a substantially heavier architectural change
# than block Gibbs -- it requires gradient-based (HMC-style) proposals and, for SoftAbs specifically, third
# derivatives of the loss -- and was set aside in favor of block Gibbs given the latter's direct
# experimental validation already in hand on this codebase's own production data. Left here as a documented
# future direction (see also docs/tsadar_math.tex, S:uq-mcmc) if block Gibbs turns out to be insufficient,
# e.g. for a lineout whose problem block itself remains poorly-mixing even in isolation.
# ======================================================================================================


def _detect_problem_leaves(H: jnp.ndarray, eigval_threshold: float, component_threshold: float) -> np.ndarray:
    """Identifies which active-leaf indices participate in a substantially-negative normalized-Hessian
    eigendirection (a genuine saddle, eigenvalue <= eigval_threshold) for *any* lineout in this fit-batch's
    H -- i.e. candidates for the "problem" block in block Metropolis-within-Gibbs (see the module-level
    docstring above). eigval_threshold is deliberately *not* _LAPLACE_EIGVAL_FLOOR (the looser floor
    _regularized_proposal_cholesky uses to keep the proposal covariance merely valid) -- see
    mcmc_cfg["block_gibbs_eigval_threshold"]'s docstring in _DEFAULTS for why the two thresholds need to
    differ.

    A leaf is flagged if its |component| in some bad eigenvector meets or exceeds component_threshold --
    i.e. it meaningfully participates in that near-degenerate direction, not just a numerically-nonzero but
    physically negligible contribution.

    Args:
        H: per-lineout Hessian block, (batch_size, n_active, n_active), as returned by _stack_hessian.
        eigval_threshold: mcmc_cfg["block_gibbs_eigval_threshold"] (negative).
        component_threshold: mcmc_cfg["block_gibbs_component_threshold"].

    Returns:
        Plain numpy boolean array, shape (n_active,) -- this always runs on concrete (already-computed) H
        outside any jit/vmap trace (see _block_indices_from_hessians, its only caller), so there's no
        reason to keep the result as a traced jax array past this point.
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
    """Unions _detect_problem_leaves' flags across every fit-batch's own Hessian into one shot-wide
    partition -- a single, fixed (well_idx, problem_idx) split shared by every fit-batch in the run, rather
    than one decided independently per fit-batch. This is a deliberate simplification, not just a
    convenience: run_mcmc_for_fit_batches vmaps every fit-batch's sampler together into one compiled graph,
    which requires every fit-batch to carry arrays of the *same* shape -- a per-fit-batch problem-block
    size that varied from one fit-batch to the next would break that vmap outright. A single shot-wide
    union keeps every fit-batch's two blocks the same fixed size, at the cost of possibly including, for
    some individual lineout, a leaf that isn't *itself* near-degenerate there (block-Gibbs on an
    already-well-conditioned block is harmless -- merely a little less efficient than a joint step would
    be -- whereas the reverse, missing a genuine problem direction for even one lineout in the batch, is
    the failure mode this exists to prevent).

    Args:
        hessians: one (batch_size, n_active, n_active) Hessian per fit-batch (same n_active for every
            entry -- guaranteed by construction, since every fit-batch in a run shares the same active-leaf
            set from the same config).
        eigval_threshold: mcmc_cfg["block_gibbs_eigval_threshold"].
        component_threshold: mcmc_cfg["block_gibbs_component_threshold"].

    Returns:
        (well_idx, problem_idx): disjoint, sorted tuples of leaf indices (jax.tree_util.tree_flatten order)
        covering every active leaf between them. problem_idx is empty (and well_idx is range(n_active))
        when nothing was flagged anywhere, or when either resulting block would be empty (every leaf
        flagged) -- both cases fall back to the existing single, unblocked joint proposal.
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
    """Extracts the (batch_size, block_n, block_n) Hessian submatrix for block_idx (a static tuple of leaf
    indices into H's own axes) and regularizes/Choleskys it independently via _regularized_proposal_cholesky
    -- each block's own within-block conditioning is handled on its own terms, using its own
    Roberts-Rosenthal 2.38/sqrt(block_n) scaling (block_n, not the full active-leaf count n_active, since
    each block-restricted sub-step genuinely proposes in only block_n dimensions at a time -- see the
    module-level block-Gibbs docstring above)."""
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
    """Embeds two blocks' own (batch_size, block_n, block_n) Cholesky factors into one full-size
    (batch_size, n_active, n_active) block-diagonal matrix (cross-block entries exactly zero). Used only
    for run_mcmc_for_batch's one-time initial-dispersion draw (init_dispersion_factor) -- the only place a
    blocked run still needs a single joint proposal to draw from, since every other proposal in a blocked
    run goes through _block_step directly, one block at a time.

    Not itself the Cholesky factor of any real covariance estimate (the two blocks' cross terms are exactly
    zero, not "estimated as uncorrelated" -- block Gibbs never estimates those cross terms at all), but
    that distinction doesn't matter here: this is only ever used to *draw* one perturbation via _propose's
    stacked @ z construction, not treated as a covariance to reason about afterward.
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
    """One Metropolis-within-Gibbs sub-step restricted to the active leaves named by block_idx (a static
    tuple of indices into jax.tree_util.tree_flatten(diff_params)'s own leaf order) -- every leaf *not* in
    block_idx is held exactly fixed for this sub-step (its column of the stacked representation is never
    touched). See the module-level block-Gibbs docstring above for why: an oversized proposal along one
    near-degenerate direction otherwise dominates the single, shared accept/reject decision an *unblocked*
    joint step would make, starving every other direction of its own meaningful adaptation signal.

    When adapt is True, also applies one Vihola RAM update (_ram_update) to step_scale_block using only
    this sub-step's own accept/reject signal and this block's own dimension (len(block_idx), not the full
    active-leaf count) -- i.e. each block adapts entirely independently of the other's acceptance behavior,
    exactly the property block Gibbs exists to provide. When adapt is False (the frozen sampling phase,
    mirroring _run_window's step_scale-frozen behavior), step_index/target_accept/adapt_gamma are unused.

    Returns:
        (new_diff_params, new_log_post, new_step_scale_block_or_None, accept) -- new_step_scale_block is
        None when adapt is False.
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
    """Block Metropolis-within-Gibbs analogue of _run_ram_window: each outer step performs TWO sequential
    sub-steps via _block_step rather than one joint proposal -- first the well-conditioned block (well_idx),
    then the problem block (problem_idx), each with its own independently RAM-adapted proposal covariance
    and its own accept/reject decision. See the module-level block-Gibbs docstring above for the motivation
    and citation (Turek et al. 2015).

    well_idx/problem_idx are static Python tuples of leaf indices (disjoint, covering every active leaf
    between them), decided once per run from the Hessian eigenstructure (_block_indices_from_hessians) --
    not re-derived here. Both blocks use the *same* step_index per outer step (this chunk's step_offset + 1,
    +2, ...), so each block's own RAM vanishing-gain schedule counts outer MCMC steps, not its own sub-step
    calls -- consistent with _run_ram_window's unblocked step_index convention, and with each block having
    exactly one sub-step per outer step.

    Returns:
        (diff_params, log_post, step_scale_ok, step_scale_problem, accept_count_ok, accept_count_problem) --
        accept counts are realized 0/1 decisions per block over this chunk, matching _run_ram_window's
        accept_count convention.
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
    """Block Metropolis-within-Gibbs analogue of _run_window for the frozen (post-burn-in) sampling phase
    -- alternates the same two block-restricted sub-steps as _run_block_ram_window (well_idx then
    problem_idx) but with no RAM update, matching _run_window's own frozen-step_scale behavior. See
    _run_window's docstring for the collect/thin memory-management rationale, which applies identically
    here (thin > 1 nests an inner, uncollected scan of `thin` steps so only one thinned sample's worth of
    history is ever held at once).

    Returns (diff_params, log_post, accept_count_ok, accept_count_problem, collected) -- collected is
    diff_params-shaped with a leading (num_kept,) axis when collect is True, else None (matching
    _run_window's convention).
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
eigenvalue floor of its own; it's positive-definite by construction.

Block Metropolis-within-Gibbs (_detect_problem_leaves) deliberately does NOT reuse this floor as its own
detection threshold, even though both operate on the same normalized Hessian -- see
mcmc_cfg["block_gibbs_eigval_threshold"]'s docstring in _DEFAULTS for why a direction that merely needs
clipping (this floor's job, needed for *any* non-positive-definite or small-positive direction to keep the
proposal covariance valid) is not automatically severe enough to be worth block Gibbs' extra complexity."""


def _normalize_hessian(H: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Correlation-scale normalization shared by _regularized_proposal_cholesky and
    _detect_problem_leaves: divides by each leaf's own sqrt(|diagonal|) (the matrix analogue of turning a
    covariance into a correlation matrix), so that a well-conditioned, uncorrelated direction's eigenvalue
    sits at O(1) regardless of that leaf's raw curvature scale -- H's diagonal entries can otherwise span
    many orders of magnitude (a real production lineout showed roughly -700 for one leaf alongside 6e7 for
    another), making a single absolute eigenvalue threshold on the *raw* Hessian meaningless across leaves.

    Args:
        H: per-lineout Hessian block, (batch_size, n_active, n_active).

    Returns:
        (h_norm, d): h_norm is D @ H @ D per lineout (D = diag(d)), symmetrized only by construction (the
            caller should re-symmetrize away float roundoff if it matters, as _regularized_proposal_cholesky
            and _detect_problem_leaves both do before eigh); d is (batch_size, n_active), 1/sqrt(|H_ii|)
            (floored so an exactly-zero diagonal entry doesn't divide by zero).
    """
    diag = jnp.diagonal(H, axis1=-2, axis2=-1)  # (batch_size, n_active)
    d = 1.0 / jnp.sqrt(jnp.maximum(jnp.abs(diag), 1e-300))  # floors an exactly-zero diagonal entry
    h_norm = H * d[:, :, None] * d[:, None, :]  # D @ H @ D per lineout, D = diag(d)
    return h_norm, d


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
    h_norm, d = _normalize_hessian(H)
    h_norm = 0.5 * (h_norm + jnp.swapaxes(h_norm, -1, -2))  # symmetrize away float roundoff before eigh

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
    _, step_scale = _hessian_and_regularized_step_scale(loss_fn, static_params, batch, diff_params)
    return step_scale


def _hessian_and_regularized_step_scale(
    loss_fn: LossFunction, static_params, batch: Dict, diff_params
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shared implementation behind _seed_step_scale_from_laplace and _seed_step_scale: computes the
    per-lineout Hessian restricted to diff_params (see _seed_step_scale_from_laplace's docstring for why)
    once, and returns both the raw Hessian *and* its eigenvalue-regularized Cholesky factor -- block
    Metropolis-within-Gibbs detection (_block_indices_from_hessians) needs the raw H itself (to
    eigendecompose per candidate block, not just the already-jointly-Cholesky'd result), whereas
    _seed_step_scale_from_laplace's own (pre-block-Gibbs) callers only ever wanted the latter. Splitting
    this out avoids computing the Hessian twice for the same fit-batch.

    Returns:
        (H, step_scale): H is the raw (batch_size, n_active, n_active) per-lineout Hessian (see
            _stack_hessian); step_scale is _regularized_proposal_cholesky(H, rr_factor) -- the same joint,
            unblocked proposal Cholesky factor _seed_step_scale_from_laplace has always returned, still
            used as-is whenever block Metropolis-within-Gibbs ends up not applicable (block_gibbs off, or
            nothing flagged -- see _block_indices_from_hessians).
    """
    n = len(jax.tree_util.tree_leaves(diff_params))
    if n == 0:
        empty = jnp.zeros((0, 0, 0))
        return empty, empty

    hess = loss_fn.h_loss_wrt_params_per_lineout(diff_params, static_params, batch)
    H = _stack_hessian(hess, diff_params)  # (batch_size, n_active, n_active)
    rr_factor = 2.38 / jnp.sqrt(float(n))
    return H, _regularized_proposal_cholesky(H, rr_factor)


@eqx.filter_jit
def _seed_step_scale(loss_fn: LossFunction, static_params, batch: Dict, diff_params, mcmc_cfg: Dict):
    """Computes one fit-batch's initial proposal step scale: Laplace-seeded (via
    _hessian_and_regularized_step_scale) if mcmc_cfg["use_laplace_seed"], falling back to the flat
    _seed_step_scale_default whenever that's off or the Hessian seed fails structurally.

    Deliberately a standalone, @eqx.filter_jit-compiled function (not inlined into run_mcmc_for_batch)
    so run_mcmc_for_fit_batches can call it in a plain Python loop, once per fit-batch, *before*
    vmapping the rest of run_mcmc_for_batch across every fit-batch -- see run_mcmc_for_batch's
    step_scale docstring for why: fusing this Hessian computation into that fit-batch vmap has been
    observed to multiply its (otherwise small, diff_params-only) memory cost by the fit-batch count,
    since nested vmap(hessian(...)) without its own enclosing jit prevents XLA from fusing/reusing
    buffers across the batch. jit-compiling here instead lets every fit-batch in that Python loop reuse
    one compiled executable (same diff_params/static_params shapes every time, only the values differ).

    Returns:
        (step_scale, H): H is the raw per-lineout Hessian used to seed step_scale, or None whenever
            use_laplace_seed is off or the Hessian computation failed structurally (the flat fallback
            carries no curvature information to detect a block Metropolis-within-Gibbs partition from --
            see run_mcmc_for_batch/run_mcmc_for_fit_batches, which treat H is None as "no blocking").
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
            use_laplace_seed/init_step_scale entirely when given, and used as-is (no block Metropolis-
            within-Gibbs partitioning) whenever it's explicitly passed by the caller. run_mcmc_for_fit_batches
            passes this in, computed sequentially per fit-batch via _seed_step_scale *before* vmapping this
            function across every fit-batch of a shot -- see _seed_step_scale's docstring for why computing
            the Laplace-seeded Hessian *inside* that fit-batch vmap is a memory-blowup risk. Leave None for
            a direct, single-fit-batch call (e.g. tests), which seeds -- and, unless mcmc_cfg["block_gibbs"]
            is off, detects a block partition -- internally.
        block_step_scale: optional (step_scale_ok, step_scale_problem) pair -- the well-conditioned and
            problem blocks' own initial Cholesky factors (see _block_step_scale) -- passed together with
            well_idx/problem_idx whenever run_mcmc_for_fit_batches has already decided a non-trivial,
            shot-wide block partition (see _block_indices_from_hessians). Mutually exclusive with step_scale
            in practice (run_mcmc_for_fit_batches passes exactly one of the two), though nothing here
            enforces that beyond block_step_scale taking priority when both are given.
        well_idx, problem_idx: static tuples of active-leaf indices for the two blocks (see
            _block_indices_from_hessians) -- required together with block_step_scale; ignored otherwise
            (this function derives its own from the Hessian it computes internally, when step_scale is None
            and mcmc_cfg["block_gibbs"] is on).

    Returns:
        samples: a diff_params-shaped pytree; each leaf has shape (num_kept, batch_size, ...), where
            num_kept = ceil((num_steps - burn_in) / thin).
        static_params: the non-sampled complement of ts_params (eqx.partition's static half), needed by
            the caller to recombine samples into full parameter values via eqx.combine.
        diagnostics: {"acceptance_rate": array (batch_size,) -- the mean of both blocks' own acceptance
            rates when block Metropolis-within-Gibbs is active, else the single joint step's rate, matching
            this key's shape/meaning either way for downstream consumers (e.g. mcmc_postprocess.py) that
            don't need to know which mode produced it; "final_step_scale": step_scale Cholesky factor,
            (batch_size, n_active, n_active) when unblocked, or a (step_scale_ok, step_scale_problem) pair
            of smaller Cholesky factors when blocked -- not currently consumed downstream either way}.

    Raises:
        NotImplementedError: if config["parameters"]["electron"]["fe"]["active"] is true (see module
            docstring).
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
        # step_scale explicitly provided as a single array by a direct caller -- used as-is, unblocked,
        # regardless of mcmc_cfg["block_gibbs"] (there's no Hessian here to detect a partition from).
        well_idx, problem_idx = tuple(range(n_active)), ()

    use_blocking = bool(problem_idx)

    # Nudges this chain's own starting point away from the shared best fit, so that when several
    # independent chains are pooled (config["other"]["calibration_uncertainty"]["num_draws"] > 1) they
    # don't all begin at literally the same point -- see run_mcmc_pooled's R-hat computation, which needs
    # genuinely independent chains to be meaningful.
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
    n_windows = max(int(mcmc_cfg["burn_in"]) // adapt_every, 0) if mcmc_cfg["burn_in"] > 0 else 0
    n_sample_steps = max(int(mcmc_cfg["num_steps"]) - int(mcmc_cfg["burn_in"]), 1)
    thin = max(int(mcmc_cfg["thin"]), 1)

    # Sampling is grouped into thin-sized units so _run_window/_run_block_window can collect only the
    # thinned samples directly (see _run_window's docstring) rather than every raw step of the whole
    # sampling phase -- for a long chain, holding every raw step in memory before thinning is easily the
    # dominant memory cost and can OOM. Rounds n_sample_steps up to the next multiple of thin if it wasn't
    # already (at most thin-1 extra MH steps) so every chunk's step count divides evenly by thin.
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
    # Burn-in adapts step_scale every single step via RAM (_ram_update, through either _run_ram_window or,
    # when blocking, _run_block_ram_window), not once per window -- adapt_every here only sets how often
    # the progress bar updates and how large each individual jax.lax.scan chunk is, not the adaptation's
    # own behavior. step_offset (passed as a traced array, not a bare Python int -- see _run_ram_window's
    # docstring) keeps RAM's vanishing-gain step counter continuous across chunks.
    for window_index in range(n_windows):
        burn_key, window_key = jr.split(burn_key)
        step_offset = jnp.asarray(float(window_index * adapt_every))
        if use_blocking:
            diff_params, log_post, step_scale_ok, step_scale_problem, accept_count_ok, accept_count_problem = (
                _run_block_ram_window(
                    window_key, loss_fn, static_params, batch, diff_params, log_post,
                    step_scale_ok, step_scale_problem, well_idx, problem_idx, adapt_every, step_offset,
                    mcmc_cfg["target_accept"], mcmc_cfg["adapt_gamma"],
                )
            )
        else:
            diff_params, log_post, step_scale, accept_count = _run_ram_window(
                window_key, loss_fn, static_params, batch, diff_params, log_post, step_scale, adapt_every,
                step_offset, mcmc_cfg["target_accept"], mcmc_cfg["adapt_gamma"],
            )
        pbar.update(adapt_every)

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

    That same sequential loop also decides this run's block Metropolis-within-Gibbs partition (see the
    module-level docstring above _run_block_ram_window), when mcmc_cfg["block_gibbs"] is on: each
    fit-batch's own Hessian is kept (already computed, no extra forward-model cost) and, once every
    fit-batch has been seeded, unioned into one shot-wide (well_idx, problem_idx) split via
    _block_indices_from_hessians -- a single, fixed partition shared by every fit-batch, not one decided
    independently per fit-batch (see that function's docstring for why: this loop's own vmap below requires
    every fit-batch's arrays to share one shape, which a per-fit-batch-sized problem block would break).
    When a non-trivial partition results, each fit-batch's own two blocks' step scales are then built from
    its already-computed Hessian (_block_step_scale) -- cheap linear algebra, no further forward-model
    evaluation -- and passed into the vmap alongside the (in that case, otherwise-unused) unblocked
    step_scale.

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
    """Classic Gelman-Rubin R-hat for x shaped (num_kept, num_chains, *extra): the ratio of the pooled
    (between + within-chain) variance estimate to the within-chain variance, reduced over the leading two
    axes and broadcast over any remaining ones. Close to 1 when the chains have mixed to the same
    distribution; values well above ~1.01-1.1 indicate they have not.

    This is the raw Gelman & Rubin (1992) / Brooks & Gelman (1998) formula, over whatever is actually
    passed in -- _rank_normalized_r_hat is what applies it to rank-normalized, folded, split chains per
    Vehtari et al. (2021); nothing here on its own knows about that recipe. Plain numpy (not jnp): every
    caller in this module runs post-hoc on already-concrete sampling output, never under jit/vmap trace,
    and rank-transforming (see _rank_normalize_pooled) needs numpy/scipy regardless.
    """
    num_kept, num_chains = x.shape[0], x.shape[1]
    chain_mean = x.mean(axis=0)
    grand_mean = chain_mean.mean(axis=0, keepdims=True)
    between = num_kept / (num_chains - 1) * np.sum((chain_mean - grand_mean) ** 2, axis=0)
    within = x.var(axis=0, ddof=1).mean(axis=0)
    var_hat = (num_kept - 1) / num_kept * within + between / num_kept
    return np.sqrt(var_hat / within)


def _split_in_half(x: np.ndarray) -> np.ndarray:
    """Splits x, shaped (num_kept, num_chains, *extra), into twice as many half-length chains along the
    existing chain axis: (half, 2*num_chains, *extra), the first num_chains columns being each chain's own
    first half and the last num_chains its second half. Drops one sample if num_kept is odd. Works equally
    well starting from num_chains == 1 (a single chain split into its own 2 halves) as from many.

    This split-in-half step is what lets a per-parameter R-hat computed from the resulting 2*num_chains
    "chains" catch a chain that is individually still drifting (non-stationary within its own sampling
    budget) even when every chain's own *mean* happens to already agree with every other chain's -- the
    same reasoning _within_chain_r_hat has always used, now folded into one shared implementation with
    _max_r_hat_across_chains rather than being that function's own special case."""
    num_kept = x.shape[0]
    half = num_kept // 2
    return np.concatenate([x[:half], x[half : 2 * half]], axis=1)


def _rank_normalize_pooled(x: np.ndarray) -> np.ndarray:
    """Rank-normalizes x, shaped (num_kept, num_chains, *extra): pools *every* value across both the
    num_kept and num_chains axes together (per trailing-extra-dims slice, e.g. per lineout) -- not
    independently per chain, since the point is to compare each chain's values against the distribution
    pooled across every chain -- replaces each with its rank there (ties broken by averaging: a *rejected*
    Metropolis-Hastings step repeats the previous sample's exact float value, so exact ties are common in
    real MCMC output, not a rare edge case to assume away), then maps those ranks through the inverse-
    normal ("Blom") transform onto approximately standard-normal scores:
    z = Phi^-1((rank - 3/8) / (N - 1/4)), N = num_kept * num_chains.

    See Vehtari, Gelman, Simpson, Carpenter & Buerkner, "Rank-normalization, folding, and localization: An
    improved R-hat for assessing convergence of MCMC," Bayesian Analysis 16(2), 667-718 (2021), Sec. 3.
    """
    num_kept, num_chains = x.shape[0], x.shape[1]
    n = num_kept * num_chains
    flat = x.reshape(n, *x.shape[2:])
    ranks = scipy.stats.rankdata(flat, axis=0)
    z = scipy.stats.norm.ppf((ranks - 3.0 / 8.0) / (n - 1.0 / 4.0))
    return z.reshape(x.shape)


def _rank_normalized_r_hat(x: np.ndarray) -> np.ndarray:
    """Rank-normalized, folded, split R-hat (Vehtari et al. 2021 -- see _rank_normalize_pooled's docstring
    for the citation) -- the modern replacement for the classical Gelman-Rubin R-hat (_classic_r_hat) this
    module used previously everywhere R-hat is computed.

    Classical R-hat is explicitly documented (Vehtari et al., Sec. 2) to fail once a chain's variance is
    very large or effectively infinite -- exactly what a genuinely weakly-identified parameter looks like
    in this codebase's own unconstrained/logit sampling space (see _seed_step_scale_from_laplace's
    docstring): its RAM-adapted proposal step keeps growing throughout burn-in with no sign of plateauing,
    since nothing in the likelihood pulls it back. On a real production shot, that alone was measured to
    produce classical cross-chain R-hat values above 1000 for such a parameter -- not because the chains
    disagreed about where the *bulk* of the distribution sits, but because a few chains' own extreme
    excursions dominate the naive variance ratio. Rank-normalizing first (mapping every pooled value to its
    rank, then to a standard-normal score) makes the statistic depend only on chains' relative *ordering*,
    which stays well-defined and comparably scaled regardless of how extreme the raw values get.

    Two R-hats are computed and the worse one kept, per Vehtari et al.'s recommended recipe:
      - "bulk" R-hat: rank-normalize the raw (split) chains directly -- sensitive to chains disagreeing
        about central tendency/location.
      - "tail" R-hat: rank-normalize the *folded* chains (|x - pooled median|) -- sensitive to chains
        disagreeing about spread/scale even when their centers agree (Vehtari et al., Sec. 4).
    Both go through the same split-in-half + rank-normalize + classical-R-hat pipeline; splitting first
    (via _split_in_half) additionally catches within-chain non-stationarity, so a single chain (num_chains
    == 1, as mcmc_postprocess.py's within-chain check passes in) is handled by the exact same code path as
    many chains, with no special-casing needed.

    Args:
        x: (num_kept, num_chains, *extra) raw samples, num_chains >= 1.

    Returns:
        (*extra,) worst-case (max of bulk and tail) R-hat, elementwise.
    """
    x = np.asarray(x)
    bulk = _classic_r_hat(_rank_normalize_pooled(_split_in_half(x)))

    grand_median = np.median(x.reshape(-1, *x.shape[2:]), axis=0)
    folded = np.abs(x - grand_median[None, None])
    tail = _classic_r_hat(_rank_normalize_pooled(_split_in_half(folded)))

    return np.maximum(bulk, tail)


def _max_r_hat_across_chains(per_draw_samples: List) -> Optional[np.ndarray]:
    """Per-(fit-batch, lineout, active parameter) rank-normalized R-hat (_rank_normalized_r_hat) across
    len(per_draw_samples) independent chains -- None if fewer than 2 (R-hat is meaningless for a single
    chain). Each element of per_draw_samples is a diff_params-shaped pytree (as returned by
    run_mcmc_for_fit_batches), leaves shaped (num_fit_batches, num_kept, batch_size, ...).

    Deliberately kept per-parameter rather than reduced to one worst-case number per lineout, the way this
    used to be: a single genuinely weakly-identified parameter (e.g. an ion temperature the data barely
    constrain) would otherwise make a "worst case across parameters" R-hat -- and everything downstream
    that trusts it -- look uniformly bad for a whole lineout even when every *other* parameter converged
    cleanly. See mcmc_postprocess._finalize_chain_selection, which is what actually makes use of the
    per-parameter breakdown (deciding which parameters may veto chain selection for a lineout, rather than
    letting any one of them always do so).

    Returns:
        (num_fit_batches, batch_size, n_active) -- column order matches jax.tree_util.tree_leaves' own
        order over the diff_params pytree, the same order mcmc_postprocess._active_param_keys reconstructs
        independently via config["parameters"]'s own iteration order.
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
    """Per-chain, per-(fit-batch, lineout, active parameter) rank-normalized *split* R-hat
    (_rank_normalized_r_hat, called with a single "chain" -- its own internal _split_in_half step is what
    actually detects non-stationarity here): the standard trick modern R-hat implementations (e.g.
    Stan/ArviZ) use even for a single chain, precisely because ordinary *cross*-chain diagnostics
    (_max_r_hat_across_chains, or a cross-chain outlier check like mcmc_postprocess._mad_flagged_chains)
    only ever compare chains to *each other* -- a chain that never reached a stationary distribution within
    its own sampling budget (still genuinely random-walking, not just settled somewhere different) can
    coincidentally look "fine" by those checks, or drag every other diagnostic down with it, without either
    ever directly testing whether that chain, on its own, actually converged.

    Unlike _max_r_hat_across_chains, this deliberately does NOT reduce across chains -- the caller needs
    to know *which* chain(s) failed to converge individually, not just that some chain did. Also unlike
    _max_r_hat_across_chains, this is kept per-parameter for the same reason described in that function's
    docstring.

    Returns:
        (num_chains, num_fit_batches, batch_size, n_active): worst-of-bulk-and-tail split-R-hat for each
        chain considered on its own (fewer than 2 chains overall is fine here, unlike
        _max_r_hat_across_chains -- this never compares different chains to each other, only a chain to
        itself).
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


def run_mcmc_pooled(
    config: Dict,
    loss_fns_by_draw: List[LossFunction],
    ts_params_list: List[ThomsonParams],
    batches_by_draw: List[List[Dict]],
    key: jax.Array,
) -> Tuple[object, object, List[Dict], object, object]:
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
        max_r_hat: per-(fit-batch, lineout, active parameter) rank-normalized R-hat across the K chains
            (see _max_r_hat_across_chains and _rank_normalized_r_hat), shape (num_fit_batches, batch_size,
            n_active), or None when K < 2.
        within_chain_r_hat: (K, num_fit_batches, batch_size, n_active) rank-normalized *split* R-hat for
            each chain considered on its own -- see _within_chain_r_hat.
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
    within_chain_r_hat = _within_chain_r_hat(per_draw_samples)

    if len(per_draw_samples) == 1:
        pooled_samples = per_draw_samples[0]
    else:
        # each draw's samples have shape (num_fit_batches, num_kept, batch_size, ...); concatenate along
        # the num_kept axis (axis=1) to pool across draws while keeping the fit-batch axis (axis=0) intact.
        pooled_samples = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *per_draw_samples)

    return pooled_samples, static_params, diagnostics_by_draw, max_r_hat, within_chain_r_hat
