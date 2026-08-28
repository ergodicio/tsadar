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

1. **Burn-in**, split into ``adapt_every``-sized windows. After each window, the per-lineout proposal
   step scale is rescaled (Robbins-Monro adaptation) toward the window's observed acceptance rate,
   targeting ``target_accept``.
2. **Sampling**, at the step scale burn-in converged to. Every ``thin``-th post-burn-in sample is kept.

The initial step scale can optionally be seeded from a Laplace/Hessian approximation at the best fit
(``use_laplace_seed: true``), which tends to reach a well-mixing step scale faster than starting from a
flat guess -- see ``init_step_scale`` and ``use_laplace_seed`` in :doc:`defaults`.

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

Whenever ``num_draws > 1``, the postprocessor also reports a per-lineout Gelman-Rubin R-hat (the
worst-mixing active parameter's R-hat, across all chains) alongside the acceptance-rate diagnostic --
values well above ~1.01-1.1 mean the chains have not mixed to the same distribution, and results for
that lineout should not be trusted without investigation (e.g. more steps, or a larger
``init_dispersion_factor``).

Running it
-----------

The postprocessor is invoked from the command line via ``run_mcmc_postprocessor.py``, against a fit
that has already finished (so its ``fitted_weights.eqx`` and input decks are available):

.. code-block:: bash

   # against a local copy of a run's artifact directory
   python run_mcmc_postprocessor.py --dir path/to/run/artifacts

   # against a run already tracked in mlflow, by run id or run URL
   python run_mcmc_postprocessor.py --run <run_id_or_url>

Either form reconstructs the original fit's state (config, data, best-fit weights) without re-running
the optimizer, then runs the sampler and logs its results to a **new** mlflow run -- the source run is
only ever read, never modified. The new run is tagged with ``source_run_id`` when starting from
``--run``, for traceability back to the original fit.

Configuration
--------------

All configuration lives under ``other.mcmc`` and ``other.calibration_uncertainty`` in the input deck --
see :doc:`defaults` for the full field-by-field reference. Every field is optional and defaults to a
small, fast smoke-test configuration (``configs/1d/defaults.yaml`` ships ``num_steps: 80``,
``burn_in: 30``); for production uncertainty estimates, raise ``num_steps``/``burn_in`` substantially
(the module's own internal defaults, used whenever ``other.mcmc`` is omitted entirely, are
``num_steps: 8000``, ``burn_in: 3000``).

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
       converged near ``target_accept`` rather than pinning at 0 or 1.
   * - ``plots/mcmc_sigma_comparison_<param>_<species>.png``
     - Per parameter, the MCMC sigma as a function of lineout; also overlaid against the Laplace/Hessian
       sigma when ``compare_to_laplace`` succeeded.

The returned ``final_params`` (posterior mean per parameter) also carries an ``mcmc_diagnostics`` entry
with the per-lineout acceptance rate and the number of calibration draws actually pooled.
