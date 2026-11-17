import numpy as np
import pytest

from tsadar.inverse.postprocess.mcmc_postprocess import _finalize_chain_selection, _mad_flagged_chains, mcmc_postprocess


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


# Chain-selection scenarios for _finalize_chain_selection.
# (chains, lineouts, parameters, within-chain flags, outlier flags, max dropped fraction,
#  dropped chains per lineout, unreliable lineouts, (lineout, parameter) excluded from voting)
# Flags are (chain, lineout, parameter). An unreliable lineout keeps no chain at all.
CHAIN_SELECTION = [
    pytest.param(6, 2, 3, [], [], 0.2, [set(), set()], [], [], id="nothing-flagged"),
    # one parameter, 2 of 10 flagged, the cap allows floor(0.2*10) = 2: exactly at the cap
    pytest.param(10, 1, 1, [], [(2, 0, 0), (5, 0, 0)], 0.2, [{2, 5}], [], [], id="drops-under-the-cap"),
    # never converged on its own (within) + converged to a different answer (outlier): the union is dropped
    pytest.param(10, 1, 1, [(1, 0, 0)], [(3, 0, 0)], 0.5, [{1, 3}], [], [], id="within-and-outlier-union"),
    # two parameters each under their own cap, but 4 distinct chains together exceed the shared cap: not any
    # single parameter's fault, too much disagreement overall
    pytest.param(
        10, 1, 2, [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)], [], 0.2, [{0, 1, 2, 3}], [0], [],
        id="voting-parameters-exceed-the-cap",
    ),
    # a weakly identified parameter (6 of 10 chains disagree, e.g. a near-flat marginal) must not veto chain
    # selection for a well-behaved one; it is excluded from voting and the lineout stays reliable
    pytest.param(
        10, 1, 2, [(c, 0, 0) for c in range(6)], [(7, 0, 1)], 0.2, [{7}], [], [(0, 0)],
        id="unreliable-parameter-excluded-from-voting",
    ),
    # if no parameter can vote there is no informative subset: keep every chain, flag the parameter
    pytest.param(10, 1, 1, [(c, 0, 0) for c in range(5)], [], 0.2, [set()], [], [(0, 0)], id="nobody-can-vote"),
    pytest.param(
        10, 2, 1, [(c, 0, 0) for c in range(5)], [(7, 1, 0)], 0.2, [set(), {7}], [], [(0, 0)],
        id="lineouts-are-independent",
    ),
]


@pytest.mark.parametrize(
    "num_chains, batch_size, n_active, within, outlier, cap, dropped, unreliable_lineouts, excluded",
    CHAIN_SELECTION,
)
def test_finalize_chain_selection(num_chains, batch_size, n_active, within, outlier, cap, dropped, unreliable_lineouts, excluded):
    num_kept = 5
    bad_within = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    bad_outlier = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    for flags, array in ((within, bad_within), (outlier, bad_outlier)):
        for c, b, a in flags:
            array[c, b, a] = True

    keep_mask, n_dropped, unreliable, param_unreliable = _finalize_chain_selection(bad_within, bad_outlier, cap, num_kept)

    # the true (uncapped) number of dropped chains is reported, also for an unreliable lineout
    np.testing.assert_array_equal(n_dropped, [len(d) for d in dropped])
    np.testing.assert_array_equal(np.flatnonzero(unreliable), unreliable_lineouts)
    assert {tuple(i) for i in np.argwhere(param_unreliable)} == set(excluded)
    keep_by_chain = keep_mask.reshape(num_chains, num_kept, batch_size)
    for b in range(batch_size):
        kept = set() if b in unreliable_lineouts else set(range(num_chains)) - dropped[b]
        assert {c for c in range(num_chains) if keep_by_chain[c, :, b].all()} == kept
        assert not any(keep_by_chain[c, :, b].any() for c in set(range(num_chains)) - kept)


def test_failed_chains_flags_frozen_and_nonfinite_chains():
    from tsadar.inverse.postprocess.mcmc_postprocess import _failed_chains

    num_chains, batch_size, n_active, num_kept = 3, 1, 2, 20
    stacked = _make_stacked(np.zeros((num_chains, batch_size, n_active)), num_kept=num_kept, seed=5)
    by_chain = stacked.reshape(num_chains, num_kept, batch_size, n_active)
    by_chain[1, :, 0, 0] = 0.7  # chain 1 never moved in parameter 0
    r_hat = np.full((num_chains, batch_size, n_active), 1.0)
    r_hat[1, 0, 0] = 0.99  # a frozen chain can still report an acceptable R-hat
    r_hat[2, 0, 1] = np.nan
    r_hat[0, 0, 1] = 1.5

    failed = _failed_chains(by_chain.reshape(stacked.shape), num_chains, r_hat, threshold=1.1)

    expected = np.zeros((num_chains, batch_size, n_active), dtype=bool)
    expected[1, 0, 0] = expected[2, 0, 1] = expected[0, 0, 1] = True
    np.testing.assert_array_equal(failed, expected)


def test_reorder_columns_matches_parameters_by_name():
    from tsadar.inverse.postprocess.mcmc_postprocess import _reorder_columns

    laplace_keys = [("electron", "Te"), ("general", "lam"), ("general", "amp1"), ("general", "amp2")]
    active_keys = [("electron", "Te"), ("general", "amp1"), ("general", "amp2"), ("general", "lam")]
    sigmas = np.array([[1.0, 3.0, 4.0, 5.0]])

    np.testing.assert_array_equal(_reorder_columns(sigmas, laplace_keys, active_keys), [[1.0, 4.0, 5.0, 3.0]])


def test_mcmc_postprocess_raises_on_angular():
    # the angular guard runs before any data is touched
    config = {"other": {"extraoptions": {"spectype": "angular_full"}}}
    with pytest.raises(NotImplementedError):
        mcmc_postprocess(config, [], {}, {}, None, None, [], 0)
