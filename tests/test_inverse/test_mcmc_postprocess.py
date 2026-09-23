import numpy as np

from tsadar.inverse.postprocess.mcmc_postprocess import _finalize_chain_selection, _mad_flagged_chains


def _make_stacked(chain_means, num_kept=50, noise=1e-3, seed=0):
    """Builds a (num_chains*num_kept, batch_size, n_active) array: for each chain, num_kept samples drawn
    tightly (std=noise) around that chain's own per-lineout, per-parameter mean, in the same
    concatenation order _mad_flagged_chains expects (chain-major)."""
    rng = np.random.default_rng(seed)
    num_chains, batch_size, n_active = chain_means.shape
    chunks = [
        rng.normal(loc=chain_means[c], scale=noise, size=(num_kept, batch_size, n_active)) for c in range(num_chains)
    ]
    return np.concatenate(chunks, axis=0)


def test_mad_flagged_chains_flags_nothing_when_chains_agree():
    num_chains, batch_size, n_active = 6, 2, 3
    rng = np.random.default_rng(1)
    chain_means = np.broadcast_to(rng.normal(size=(1, batch_size, n_active)), (num_chains, batch_size, n_active))
    stacked = _make_stacked(chain_means, seed=2)

    flagged = _mad_flagged_chains(stacked, num_chains, mad_scale=3.5)

    assert not np.any(flagged)


def test_mad_flagged_chains_flags_a_clear_outlier_only_for_the_lineout_and_parameter_it_affects():
    num_chains, batch_size, n_active = 6, 2, 3
    chain_means = np.zeros((num_chains, batch_size, n_active))
    # Chain 4 is way off, but only for lineout 0's parameter 1 -- every other (lineout, parameter)
    # combination has every chain agreeing.
    outlier_chain = 4
    chain_means[outlier_chain, 0, 1] = 50.0
    stacked = _make_stacked(chain_means, seed=3)

    flagged = _mad_flagged_chains(stacked, num_chains, mad_scale=3.5)

    assert flagged[outlier_chain, 0, 1]
    assert not np.any(np.delete(flagged[:, 0, 1], outlier_chain))  # no one else flagged for this (lineout, param)
    # Kept per-parameter now (not reduced): the same chain is NOT flagged on the *other* parameters at
    # lineout 0, since it only actually diverges on parameter 1.
    assert not np.any(flagged[outlier_chain, 0, [0, 2]])
    assert not np.any(flagged[:, 1, :])  # nothing flagged for lineout 1 at all


def test_mad_flagged_chains_returns_shape_matching_num_chains_batch_and_params():
    num_chains, batch_size, n_active = 4, 3, 2
    chain_means = np.random.default_rng(5).normal(size=(num_chains, batch_size, n_active))
    stacked = _make_stacked(chain_means, seed=6)

    flagged = _mad_flagged_chains(stacked, num_chains, mad_scale=3.5)

    assert flagged.shape == (num_chains, batch_size, n_active)
    assert flagged.dtype == bool


def test_finalize_chain_selection_keeps_everything_when_nothing_flagged():
    num_chains, batch_size, n_active, num_kept = 6, 2, 3, 10
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert np.all(keep_mask)
    assert np.all(n_dropped == 0)
    assert not np.any(unreliable)
    assert not np.any(param_unreliable)


def test_finalize_chain_selection_drops_flagged_chains_under_the_cap():
    # A single active parameter, so this degenerates to the pre-per-parameter behavior: the one parameter
    # stays well under its own cap, so it drives chain selection normally.
    num_chains, batch_size, n_active, num_kept = 10, 1, 1, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier[[2, 5], 0, 0] = True  # 2 of 10 flagged, cap allows floor(0.2*10)=2 -- exactly at the cap

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert n_dropped[0] == 2
    assert not unreliable[0]
    assert not param_unreliable[0, 0]  # under its own cap, so not excluded from voting either
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    assert not keep_by_chain[2, :, 0].any()
    assert not keep_by_chain[5, :, 0].any()
    other_chains = [c for c in range(num_chains) if c not in (2, 5)]
    assert keep_by_chain[other_chains, :, 0].all()


def test_finalize_chain_selection_combines_within_and_outlier_flags():
    num_chains, batch_size, n_active, num_kept = 10, 1, 1, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_within[1, 0, 0] = True  # never converged on its own
    bad_outlier[3, 0, 0] = True  # converged, but to a different answer

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.5, num_kept)

    assert n_dropped[0] == 2  # union of the two failure modes, not just one
    assert not unreliable[0]
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    assert not keep_by_chain[1, :, 0].any()
    assert not keep_by_chain[3, :, 0].any()


def test_finalize_chain_selection_marks_lineout_unreliable_when_voting_parameters_exceed_cap():
    # Two active parameters, BOTH structurally fine on their own (well under the cap individually) but
    # together dropping more chains than the shared budget allows -- the case max_dropped_chain_fraction
    # is meant to catch: not any single parameter's fault, but too much disagreement overall.
    num_chains, batch_size, n_active, num_kept = 10, 1, 2, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_within[[0, 1], 0, 0] = True  # param 0: 2 chains bad, under the cap (floor(0.2*10)=2) on its own
    bad_within[[2, 3], 0, 1] = True  # param 1: 2 different chains bad, also under the cap on its own
    # union across the two voting parameters is 4 distinct chains -- over the shared cap of 2

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert not np.any(param_unreliable)  # neither parameter alone exceeded its own cap
    assert n_dropped[0] == 4  # the true (uncapped) union count is still reported, not truncated
    assert unreliable[0]
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    assert not keep_by_chain[:, :, 0].any()  # nothing feeds this lineout's summary stats


def test_finalize_chain_selection_excludes_a_structurally_unreliable_parameter_from_voting():
    # The core new behavior: one genuinely weakly-identified parameter (most chains disagree on it, e.g.
    # a near-flat marginal posterior -- see mcmc.py's block Metropolis-within-Gibbs docstring) must not be
    # able to veto chain selection -- and therefore the reported mean/std -- for a second, well-behaved
    # parameter that every chain actually agrees on.
    num_chains, batch_size, n_active, num_kept = 10, 1, 2, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_within[[0, 1, 2, 3, 4, 5], 0, 0] = True  # param 0 ("Ti"): 6 of 10 chains disagree -- over the cap
    bad_outlier[7, 0, 1] = True  # param 1 (well-behaved): one genuine outlier chain, well under the cap

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert param_unreliable[0, 0]  # param 0 excluded from voting -- its own disagreement doesn't get "fixed"
    assert not param_unreliable[0, 1]  # param 1 is fine
    assert not unreliable[0]  # the lineout as a whole is still reliable -- param 0 didn't poison it
    assert n_dropped[0] == 1  # driven only by param 1's own outlier chain, not param 0's six
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    assert not keep_by_chain[7, :, 0].any()
    other_chains = [c for c in range(num_chains) if c != 7]
    assert keep_by_chain[other_chains, :, 0].all()


def test_finalize_chain_selection_keeps_every_chain_when_every_parameter_is_structurally_unreliable():
    # If no parameter can vote at all, there's no informative subset to select -- every chain is kept
    # (reporting a necessarily wide, but non-NaN, summary) rather than the whole lineout being discarded.
    num_chains, batch_size, n_active, num_kept = 10, 1, 1, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_within[[0, 1, 2, 3, 4], 0, 0] = True  # the only active parameter: 5 of 10 chains disagree

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert param_unreliable[0, 0]  # flagged, so a caller can tell this parameter's spread is untrustworthy
    assert not unreliable[0]  # but not wholesale-discarded -- there's nothing else to select chains by
    assert n_dropped[0] == 0
    assert np.all(keep_mask[:, 0])  # every chain kept, full pool used for this parameter's own mean/std


def test_finalize_chain_selection_is_per_lineout_independent():
    num_chains, batch_size, n_active, num_kept = 10, 2, 1, 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_within[[0, 1, 2, 3, 4], 0, 0] = True  # lineout 0: the only parameter is structurally unreliable
    bad_outlier[7, 1, 0] = True  # lineout 1: one clean outlier, well under the cap

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, 0.2, num_kept)

    assert param_unreliable[0, 0] and not param_unreliable[1, 0]
    assert not unreliable[0] and not unreliable[1]
    assert n_dropped[0] == 0 and n_dropped[1] == 1
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    assert keep_by_chain[:, :, 0].all()  # lineout 0: nothing to vote, so nothing dropped
    assert not keep_by_chain[7, :, 1].any()
    other_chains = [c for c in range(num_chains) if c != 7]
    assert keep_by_chain[other_chains, :, 1].all()
