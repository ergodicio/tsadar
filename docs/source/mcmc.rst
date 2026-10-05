.. _mcmc-postprocessor:

MCMC Uncertainty Postprocessor
========================================

TSADAR normally estimates the uncertainty on fitted parameters from the Hessian of the loss at the
best fit (the ``calc_sigmas`` option described in :doc:`defaults`, computed by
:mod:`tsadar.inverse.postprocess.laplace`). That Laplace/Hessian approximation is fast but assumes the
posterior is locally Gaussian around the best fit, which is not always a good approximation.

As an alternative, TSADAR includes a standalone Metropolis-Hastings MCMC sampler
(:mod:`tsadar.inverse.postprocess.mcmc`) that samples the actual posterior around each lineout's best
fit, rather than approximating it from local curvature. It is a **postprocessor**, not part of a normal
fit: it never runs automatically and has no effect on fitting itself. It is run separately, after a fit
has already completed, against that fit's saved results.

Scope and limitations
----------------------

- **1D (non-angular) fits only.** Angular fits use a different batching scheme
  (:func:`tsadar.inverse.loops.build_angular_batch`) that this sampler does not support; attempting it
  raises ``NotImplementedError``.

- **The electron distribution function ("fe") must be inactive.** Every other active fit parameter
  (``Te``, ``ne``, ``Ti``, ``Z``, ``fract``, ``Va``, ``amp1``/``amp2``/``amp3``, ``lam``,
  ``ne_gradient``, ``Te_gradient``, ``ud``, ``brem_amp``, ``brem_c``) is stored as a single array with a
  leading batch axis, so the sampler can propose and accept/reject across a whole batch of lineouts at
  once. ``fe``'s per-lineout parameters are stored as a list of separate objects instead, which this
  vectorized sampler cannot handle. If ``parameters.electron.fe.active`` is ``true``, the postprocessor
  raises ``NotImplementedError`` immediately rather than silently producing a wrong answer; deactivate
  ``fe`` to use MCMC uncertainty for the remaining parameters, or fall back to ``calc_sigmas``.

How it works
-------------

For each lineout, proposals are Gaussian random walks on the same unconstrained (sigmoid/logit)
parameters the optimizer itself fits, so the existing ``lb``/``ub`` bounds from the input deck are
enforced for free by that reparametrization -- no separate bounds handling is needed. The chain is
vectorized across every lineout in a fit-batch (and across fit-batches, via ``vmap``), so one call
samples every lineout's posterior simultaneously rather than looping over them.

The run proceeds in two phases:

1. **Burn-in.** The per-lineout proposal covariance is adapted on every step with Robust Adaptive
   Metropolis (RAM, Vihola 2012), which grows the proposal along directions that are accepted more often
   than ``target_accept`` and shrinks it along directions that are accepted less often. ``adapt_every``
   only sets the chunk size the progress bar advances in.
2. **Sampling**, with the proposal frozen at what burn-in converged to. Every ``thin``-th post-burn-in
   sample is kept.

Proposals are jointly correlated across a lineout's active parameters. With ``use_laplace_seed: true``
the initial proposal covariance is taken from the full Hessian of the likelihood at the best fit, which
lets the chain move along correlated or degenerate directions from the start. Where that Hessian is not
positive-definite (a weakly identified parameter can have negative curvature at the reported best fit)
its eigenvalues are clipped. Otherwise the proposal starts from the flat ``init_step_scale``.

A parameter with strongly negative curvature gets a very wide proposal, and because a joint step has a
single accept/reject decision, that one parameter can dominate the acceptance of every step. With
``block_gibbs: true`` (the default) such parameters are detected from the Hessian and sampled in their
own block, with their own accept/reject decision and their own adapted proposal, separately from the
remaining parameters.

The derivations, and the reasoning behind these choices, are in the math document linked from
:doc:`math`.

Likelihood
~~~~~~~~~~~

The chains sample the likelihood defined by ``optimizer.loss_method``. ``covar`` is recommended for this
postprocessor even when the fit itself used ``l2``: the ``l2`` likelihood takes its per-pixel variance
from the data, which is unreliable wherever the data are close to zero, while ``covar`` builds a
correlated detector-noise covariance from the model. Set it in the overrides deck (see below). ``covar``
requires ``data.background.bg_subtract: false`` so that the noise model sees the total signal; this is
enforced with a warning.

Multiple chains
~~~~~~~~~~~~~~~~

By default the postprocessor runs a single chain per lineout. Setting
``other.calibration_uncertainty.num_draws`` above 1 runs that many **independent chains** instead, each
run via :func:`tsadar.inverse.postprocess.mcmc.run_mcmc_pooled`, pooling all of their post-burn-in
samples into one posterior. ``num_draws`` is the one knob for *how many* chains; two independent, opt-in
knobs control *how* those chains differ from one another:

- **Calibration.** Instrument calibration values (spectral dispersion/offset, IRF width, detector gain)
  are not fit parameters, so a single chain holds them fixed at their nominal values. To account for
  uncertainty in those values too, each chain can instead be run under its own independently-drawn
  calibration realization (:mod:`tsadar.inverse.postprocess.mcmc_calibration`), via the ``*_sigma``
  fields in ``other.calibration_uncertainty`` (see :doc:`defaults`). Off by default (every ``*_sigma`` at
  0.0) -- chains then all share the nominal calibration.
- **Starting point.** ``other.mcmc.init_dispersion_factor`` perturbs each chain's own starting point
  before burn-in, scaled off the same step scale the sampler already uses. Off by default (``0.0``) --
  chains then all start at the exact best fit.

These compose: with both off, ``num_draws`` chains still run, differing only by their own independent
random-walk noise from an identical start (a legitimate, if weaker, basis for the convergence check
below). With calibration sigmas on, the pooled posterior is the union of within-chain parameter
uncertainty and between-chain calibration uncertainty. With dispersion on, chains explore the posterior
from different starting points, which is also the more standard way of guarding against R-hat
under-detecting non-convergence when chains happen to start from the same point.

Convergence checks
~~~~~~~~~~~~~~~~~~~

Three checks decide which chains contribute to a lineout's reported mean, standard deviation and
covariance. All use the rank-normalized, folded, split R-hat of Vehtari et al. (2021) where an R-hat is
needed.

- **Within-chain.** Each chain's first and second halves are compared. A chain whose split R-hat for a
  parameter exceeds ``within_chain_r_hat_threshold`` has not reached a stationary distribution.
- **Cross-chain outliers** (``num_draws > 1``). A chain whose posterior mean for a parameter is more than
  ``chain_outlier_mad_scale`` robust standard deviations from the median of all chains has settled
  somewhere different from the rest.
- **Drop budget.** Chains flagged by either check are excluded from the summary statistics. If more than
  ``max_dropped_chain_fraction`` of the chains would be excluded, the lineout is marked unreliable and its
  summary is reported as NaN.

The checks are made per parameter. A parameter that is weakly identified, so that most chains disagree on
it, is not allowed to exclude chains on behalf of the other parameters; its own mean and standard
deviation are still reported, and it is listed in ``mcmc_diagnostics["param_unreliable"]`` and in the
``param_unreliable.<param>_<species>`` metrics. The raw samples of every chain are always saved.

With ``num_draws > 1`` the diagnostics plot also shows the per-lineout cross-chain R-hat of the
worst-mixing parameter, computed before any chains are excluded.

Running it
-----------

The postprocessor is invoked from the command line via ``run_mcmc_postprocessor.py``, against a fit
that has already finished (so its ``fitted_weights.eqx`` and input decks are available):

.. code-block:: bash

   # against a local copy of a run's artifact directory
   python run_mcmc_postprocessor.py --dir path/to/run/artifacts

   # against a run already tracked in mlflow, by run id or run URL
   python run_mcmc_postprocessor.py --run <run_id_or_url>

A small YAML deck of settings to change (same nesting as the input deck) can be merged on top of the
fit's saved config with ``--overrides``; ``configs/postprocessor/postprocessor_stub.yaml`` is an example.
Only override settings that affect postprocessing, not ones the saved weights depend on (lineouts, batch
size, which parameters are active). ``queue_mcmc_postprocessor.py`` takes the same arguments and submits
the run as a Slurm job.

.. code-block:: bash

   python run_mcmc_postprocessor.py --run <run_id_or_url> --overrides configs/postprocessor/postprocessor_stub.yaml

Either form reconstructs the original fit's state (config, data, best-fit weights) without re-running
the optimizer, then runs the sampler and logs its results to a **new** mlflow run -- the source run is
only ever read, never modified. The new run is tagged with ``source_run_id`` when starting from
``--run``, for traceability back to the original fit.

Configuration
--------------

All configuration lives under ``other.mcmc`` and ``other.calibration_uncertainty`` in the input deck --
see :doc:`defaults` for the full field-by-field reference. Every field is optional. A field that is
omitted takes the sampler's built-in default (``num_steps: 8000``, ``burn_in: 3000``), and a warning
lists every field that was defaulted. These options are not part of ``configs/1d/defaults.yaml``,
since the postprocessor never runs as part of a fit; ``configs/postprocessor/postprocessor_stub.yaml``
lists all of them and is the deck to copy and pass with ``--overrides``.

Outputs
--------

The postprocessor writes the same family of artifacts a normal fit does (see :doc:`artifacts`), plus a
few MCMC-specific ones, all logged to its own mlflow run:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Artifact
     - Contents
   * - ``sigmas_mcmc.nc``
     - Per-lineout posterior standard deviation of each active parameter -- the MCMC analogue of
       ``sigmas.nc``, kept under a different name so both can coexist if ``compare_to_laplace`` is used.
   * - ``binary/mcmc_covariance.nc``
     - Per-lineout posterior covariance matrix across all active parameters.
   * - ``binary/mcmc_samples.nc``
     - The full thinned, pooled posterior samples for every active parameter, one value per kept sample
       per lineout. Only written when ``save_samples`` is true.
   * - ``plots/mcmc_acceptance_rate.png``
     - Histogram of per-lineout sampling-phase acceptance rates, to check burn-in adaptation actually
       converged near ``target_accept`` rather than pinning at 0 or 1. With several chains, further
       panels show the cross-chain R-hat, the number of chains excluded per lineout, and the number of
       lineouts marked unreliable.
   * - ``plots/corner/corner_lineout_<value>.png``
     - Corner plots of the posterior for an evenly spaced subset of lineouts, showing the chains the
       summary statistics were computed from, colored by chain.
   * - ``draw_checkpoints/draw_<k>.pkl``
     - Each chain's raw samples and diagnostics, uploaded as soon as that chain finishes.
   * - ``overrides.yaml``
     - The overrides deck the run was launched with, if any.
   * - ``plots/mcmc_sigma_comparison_<param>_<species>.png``
     - Per parameter, the MCMC sigma as a function of lineout; also overlaid against the Laplace/Hessian
       sigma when ``compare_to_laplace`` succeeded.

The returned ``final_params`` (posterior mean per parameter) also carries an ``mcmc_diagnostics`` entry
with the per-lineout acceptance rate, the number of chains pooled, the cross-chain R-hat, the number of
chains excluded per lineout, and the per-lineout and per-parameter reliability flags.
