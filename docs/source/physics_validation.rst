.. _physics_validation:

Test suite
==========

This page describes every test in ``tests/``: what it checks, how, and why its
tolerances are what they are. The tests themselves carry only short docstrings; this
page is the reference.

Running the tests
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Lane
     - Command
     - When
   * - Pull request
     - ``pytest -m "not slow" tests/``
     - every PR (``.github/workflows/test.yaml``)
   * - Scheduled
     - ``pytest -m "physics and slow" tests/``
     - weekly (``physics-nightly.yaml``)
   * - GPU
     - ``pytest tests/test_form_factor/test_ratintn_gpu.py``
     - opt-in, on a GPU runner

The PR lane runs serially, in one process.

Markers: ``physics`` (a check against physics), ``slow`` (scheduled lane only), ``gpu``.

Session setup
~~~~~~~~~~~~~

``tests/conftest.py`` does two things for every test session.

* **Agg backend.** Tests write figures only to files. On Windows the default backend is
  TkAgg, and MCMC postprocessing draws from worker threads; Tk then fails with "main
  thread is not in main loop" and a 23 s postprocess runs for more than 25 minutes.
* **A throwaway MLflow store.** Runs go to a temporary sqlite store unless
  ``MLFLOW_TRACKING_URI`` is set, so tests never write to ``./mlruns``. Fixtures that
  need their own store restore the previous tracking URI when they finish.

Layout
------

.. code-block:: text

   tests/
     conftest.py             session setup (above)
     physics/                checks against physics, at every depth of the code
       harness.py            builds tsadar inputs from an oracle plasma
       compare.py            the detector-depth comparison and TSADAR's tolerances
       test_kernels.py       L1: susceptibility kernel and Z'
       test_spectra.py       L2: FormFactor
       test_detector.py      L3: the full diagnostic
       test_averaging.py     A10: gradients, aperture, species
       test_notch.py         A13: spectrum breakdown
       test_relativistic.py  A12: relativistic limit
       test_inverse.py       L4: S1-S3
       test_sweeps.py        scheduled: A1 over a plasma matrix
     pipeline/test_real_shot.py   the production pipeline on real data
     test_form_factor/       ratintn numerics (singular nodes, AD branches, GPU)
     test_forward/           detector edges and IRF area/centroid (incl. angular IRF)
     test_inverse/           loops, loss, Laplace, MCMC, angular objective, input validation
     test_runner/            run tags and status lifecycle, tag backfill
     test_utils/             manifest (golden files), plot csv outputs
     test_postprocess_runner.py   postprocess config loading and run-id parsing

Physics suite
-------------

Design: one oracle, four depths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The oracle and the case catalog live in `tsverify <https://github.com/almilder/tsverify>`_,
a separate package (NumPy and SciPy only) pinned in the ``test`` extra. ``tsverify.oracle``
computes the collisionless Thomson spectrum independently of TSADAR. Every susceptibility comes from the Faddeeva function,
:math:`\chi_s = [1 + \zeta_s Z(\zeta_s)]/(k\lambda_{Ds})^2` with
:math:`Z(\zeta) = i\sqrt{\pi}\,w(\zeta)`, for Maxwellian or Gaussian-mixture species (a
mixture's :math:`\chi` is the sum of shifted, scaled Faddeeva terms). Distributions with
no closed form (kappa) use ``QuadratureEDF``: the principal value by singularity
subtraction,
:math:`\mathrm{P}\!\int f'(t)/(t-x)\,dt = \int [f'(t)-f'(x)]/(t-x)\,dt + f'(x)\ln\frac{b-x}{x-a}`,
with adaptive quadrature, tabulated once per distribution for :math:`|x| \le 25` (step
0.02, cubic spline) and the asymptote :math:`-1/x^2 - 3/x^4` beyond. On a Maxwellian
it matches Faddeeva to 2e-7. (QUADPACK's Cauchy-weight rule is not used: it fails when
the pole falls on one of its subdivision points.)

The oracle's spectral density includes TSADAR's absolute scale: the first-order factor
:math:`(1+2\omega/\omega_L)`, :math:`r_e^2 n_e` and :math:`d\omega/d\lambda`, so
comparisons are absolute. The same oracle is compared with TSADAR at four depths:

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Depth
     - TSADAR code
     - Oracle side
   * - L1 kernel
     - ``FormFactor._numeric_chi``, ``zprimeMaxw``
     - :math:`\chi(\xi)` of the same distribution, relative to max :math:`|\chi|`
   * - L2 spectrum
     - ``FormFactor``
     - :math:`S(\lambda)`, peak-normalized :math:`L_\infty` and relative :math:`L_1`
   * - L3 detector
     - ``ThomsonScatteringDiagnostic``: IRF, pixel grouping, oversampled grid, gradients,
       aperture weights, fit-range normalization
     - L2 through an independent detector model: continuous Gaussian blur, exact average
       over each pixel (normal CDF), the same normalization
   * - L4 inverse
     - ``LossFunction``, the fit loop, Laplace, MCMC
     - data made by the oracle

L3 is also the plumbing test: a shift, scale or smear anywhere between the form factor
and the detector fails against physics. Cases are data (``Case`` in
``tsverify.cases``), so a new plasma is checked at every depth. Non-Maxwellian distributions
reach TSADAR the way a user gives one: written to a csv and loaded as
``type: numeric`` (``fe`` for electrons, ``fi`` for ions).

L3 compares the pixels inside the fit ranges and leaves out those within two IRF widths
of the notch edges. The notch is not modelled exactly (a rectangular attenuation on the
sampled grid places each edge only to a grid step) and its band is excluded from fits,
so no test asserts its shape.

Scope: the well-resolved regime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every TSADAR case keeps :math:`\alpha = 1/(k\lambda_{De}) \le 3.5` at its features and
has no nearly undamped resonance (:math:`\min|\epsilon| \gtrsim 0.25`). TSADAR's
susceptibility is accurate to about 1e-4, set by its 1024-point internal velocity grid.
At a weakly damped resonance :math:`S \propto 1/|\epsilon|^2` amplifies that to 1–8 % of
the EPW peak for :math:`k\lambda_{De}` = 0.15–0.25 (0.4 % with 4096 points and a 4×
finer :math:`\xi_2`, 0.1 % with 8192), and to 0.6 % for a positive-slope bump-on-tail at
:math:`\alpha \approx 2`. Those cases are not tested; see *Future work*.

Validity flag
~~~~~~~~~~~~~

Results outside this scope are not rejected. They are reported: every fit and forward run
records :math:`\alpha` and :math:`\beta` per lineout and an MLflow ``validity`` tag (see
:ref:`validity-flag`). The limits come from where the current code breaks down: FormFactor
against the oracle, peak-normalized :math:`L_\infty`, at 1024 and 8192 wavelengths.

.. list-table::
   :header-rows: 1

   * - Scan
     - Error
     - Limit
   * - EPW, Maxwellian electrons (quadrature), :math:`\alpha` from angle and from
       :math:`n_e`
     - 1.4e-4 to 2.3e-3 for :math:`\alpha` = 1.2–3.4; 2.9e-3 to 9e-3 at 4.0–4.1; 6e-3 to
       0.77 at 4.8–5.9 (the larger value at 8192 points, which resolve the narrowing line).
       Above :math:`\alpha \approx 7` the line falls between samples, so the small errors
       there mean nothing.
     - :math:`\alpha \le 3.5`
   * - IAW, Maxwellian ions (Faddeeva :math:`Z'`), :math:`\alpha` = 2.06, :math:`\beta`
       from :math:`Z T_e/T_i`
     - ≤ 7e-4 for :math:`\beta \le 2.9`; 1.3e-3 to 3.9e-3 for 4–6; flat at 4e-3 to 6e-3
       out to :math:`\beta` = 31
     - none
   * - IAW, numeric ions, same plasmas
     - as for Maxwellian ions up to :math:`\beta` = 8; 1.1e-2 at 11, just past the
       :math:`|\xi| = 8.2` table
     - :math:`\beta \le 8`

``test_validity.py`` checks that :math:`\alpha` and :math:`\beta` match the oracle's
:math:`1/(k\lambda_{De})` to 1e-3, that every physics case is inside the limits, and that
the flag marks a case past each limit.

Cases and tolerances
~~~~~~~~~~~~~~~~~~~~

Tolerances are set from measured errors; the measured values are in brackets.

.. list-table::
   :header-rows: 1
   :widths: 8 40 52

   * - ID
     - Cases
     - Checks and tolerances (measured)
   * - A1
     - Maxwellian: H EPW; EPW with electron drift ``ud`` = 60; H IAW with flow; CH IAW
       with a relative ion drift. ne 0.2–0.5, Te 0.6–1 keV, 60°
     - L2 at 256 and 2048 wavelengths: :math:`L_\infty` 1e-3, :math:`L_1` 5e-4 (6.2e-4,
       3.2e-4). L3: EPW 1.5e-3, IAW 1e-3 (1.0e-3, 2.4e-4); peaks 5e-3 / 5e-4 nm (1e-3)
   * - A3
     - Gaussian-mixture EDFs: two-temperature, hot-tail shoulder (10 % at 3
       :math:`v_T`), return-current beam (EPW and IAW), flat-top, ion beam, two-humped
       ions
     - L1 :math:`\chi` 4e-4 of max :math:`|\chi|` (1.1e-4 to 2.9e-4). L2 2.5e-3 / 1.5e-3
       (up to 1.8e-3 for the two-humped ions, 1.2e-4 with a 4096-point grid). L3 as A1
   * - A5
     - Heavy-tailed EDF through the numeric input: kappa 3 and 5 electrons (a Student-t
       with :math:`\nu = 2\kappa - 1`, unit variance) on the EPW and IAW. TSADAR has no
       kappa model; the check is of its numeric-EDF path on power-law tails: the
       principal value of a slowly decaying :math:`f'`, truncation and renormalization
       at :math:`|v| = 8\,v_T`, and interpolation of a non-Gaussian shape
     - As A1: L1 2.9e-4 / 2.0e-4; L2 ≤ 6.0e-4 / 3.9e-4; L3 ≤ 3.5e-4. Kappa ≥ 3 only:
       the mass beyond :math:`8\,v_T` is 1.5e-4 (kappa 3), 8e-6 (kappa 5), 8e-4 (kappa 2)
   * - A10
     - 20 % ne and Te gradient over 5 points (EPW, IAW); zero gradient; a three-angle
       aperture (54°, 60°, 66°; weights 0.25/0.5/0.25); CH species reordered; H split
       into 0.3/0.7
     - Against the oracle averaged the same way, as L3 (the gradient and aperture change
       the spectrum by 2–6 %). Zero gradient = one point and reorder/split to 1e-10
   * - A13
     - ``spectrum_breakdown``
     - Returns the model spectrum to 1e-12
   * - A12
     - Non-collective relativistic limit: ne 1e14 cm⁻³, Te 0.5/2/5 keV, 45°/90°/135°
     - Against Selden's exact spectrum of a Jüttner plasma (Phys. Lett. A 79, 405, 1980,
       eq. 1). Shape within :math:`3\,T_e/m_ec^2` (0.66–2.31); peak within 1.2× Selden's
       second-order blue shift :math:`7\sin^2(\theta/2)[1+7\sin^2(\theta/2)]/2\alpha^2`,
       eq. 4 (0.47–0.93), which a first-order model lacks. Without the
       :math:`(1+2\omega/\omega_L)` factor the shape error is 22–30
       :math:`T_e/m_ec^2`
   * - Z′
     - ``zprimeMaxw`` and ``other.zprime``
     - Default against the Dawson-function form to 1e-13; ``legacy`` equal to the table;
       the known difference (1.5e-3 to 1.7e-3); the deck setting reaches both form factors
   * - Tails
     - :math:`\chi` past the :math:`|\xi| \le 8.2` table: the Maxwellian-ion :math:`Z'` and
       the numeric-EDF principal value, for every A3 distribution and for a beam at 6
       :math:`v_T`; high-Z IAW (Au, Z = 40, :math:`T_i/T_e` = 0.25) at 8192 wavelengths
     - Continuity at both edges 1e-5 (≤ 1.1e-6). At 0.5, 4 and 30 past each edge:
       Maxwellian 2e-4 (9e-5), A3 3e-2 (≤ 1.7e-2); the beam is checked for continuity only
       (up to 15 % there, see *Defects found*). :math:`Z'` series against Faddeeva to 1e-5.
       High-Z: :math:`L_\infty` 1e-2 (5.8e-3), peaks 5e-5 nm (1e-5)
   * - S1
     - Recovery of (ne, Te) from the EPW and (Va, Ti) from the IAW, from displaced starts,
       with the production loss and L-BFGS-B
     - 1e-3 relative, Va 1e-3 absolute (5e-6 to 2.1e-4)
   * - S2
     - Laplace σ (``get_sigmas`` on the Hessian of :math:`-2\ln L = \sum (d-t)^2/|d|`) at
       1e4 counts
     - Equals the Cramér–Rao bound :math:`F = J^T \mathrm{diag}(1/d) J`, :math:`J` by
       finite differences of the oracle, to 1 % (1.0009, 1.0007)
   * - S3
     - MCMC (``run_mcmc_for_batch``, 6000 steps, 1e4 counts)
     - Width / Laplace σ within 10 % (0.972, 0.975); mean within 0.2σ of truth (0.03);
       acceptance within 0.1 of 0.234
   * - Sweep
     - Scheduled lane: A1 at L2 over ne 0.05/0.2/0.5, Te 0.1/0.6/2 keV, 30°/60°/120°,
       keeping the 18 of 27 plasmas with :math:`\alpha \le 3.5`
     - EPW wings :math:`L_\infty` 2e-3

S2 and S3 use TSADAR's own noiseless spectrum at the truth as data. With oracle data at
1e4 counts the 1e-3 model/oracle difference is statistically significant on the
low-count wings (:math:`-2\ln L = 424` at the truth) and the Hessian's residual term
moves the Laplace σ by 8–15 %. The truth is the physical value the activated parameters
hold: the inverse activation in ``ts_params`` is a stabilized logit,
:math:`\log(0.01 + x/(1.01-x))`, a deliberate offset (a deck ne of 0.2 starts a fit at
0.2044).

The oracle against independent physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tsverify.selftest`` checks the oracle itself (``pytest --pyargs tsverify.selftest``):

* A6, equilibrium sum rule: for Maxwellian species at one temperature,
  :math:`\int S\,d\omega = (k^2 + \sum_s k_s^2)/(k^2 + k_e^2 + \sum_s k_s^2)` (Debye–Hückel),
  for H, C and CH at :math:`k\lambda_{De}` = 0.3, 1, 3, to 1e-4.
* A2, dispersion limits: the EPW peak at :math:`k\lambda_{De} = 0.1` follows
  :math:`(\omega/\omega_{pe})^2 = 1 + 3a + 6a^2 + Zm_e/M` (:math:`a = k^2\lambda_{De}^2`)
  within :math:`20a^3` (measured :math:`10a^3`); below 0.1 the undamped peak is narrower
  than the frequency grid. The ion-acoustic peak sits at the screened sound speed to
  3e-3 (1e-3) for :math:`T_i/T_e \le 0.1`.
* The quadrature path against Faddeeva (2e-7), the kappa distribution's normalization
  and unit variance, and Selden's Tables 1 (to 1e-4) and 2 (to 5e-4).

These evaluate only the oracle, which is exact at any :math:`\alpha`; the resolution
limit above does not apply to them.

Defects found
~~~~~~~~~~~~~

* **Electron f′.** The Landau-damping term took :math:`f'` as a forward difference along
  the wavelength samples, i.e. at :math:`\xi_{i+1/2}`, stored at sample *i*. It is now
  interpolated from the internal velocity grid, as for numeric ions, which makes it
  independent of the wavelength sampling (5.0e-4 to 5.7e-4 at 128 to 2048 points, against
  1.9e-2 before at 2048). On shot 101675 (1024 samples over 323–743 nm) the fit moves by
  ne −1.0 %, Te +1.5 %, m +2.8 %, amp2 +12 % at the same loss.
* **Ion Z′ table.** The real part of ``rdWT.txt`` differs from the exact :math:`Z'` by up
  to 1.6e-3 (:math:`\mathrm{Re}\,Z'(1)` = 0.153356 against 0.152318), about 0.65 % on an
  IAW feature. ``other.zprime: faddeeva`` (default) is exact; ``legacy`` restores the
  table for comparisons with codes that use it.
* **Susceptibility past the table.** Past :math:`|\xi| = 8.2` the Maxwellian-ion
  :math:`Z'` was clamped to its edge value and the numeric-EDF principal value switched
  to :math:`1/\xi^2`. Both jumped at the edge (4.6e-2 for Maxwellian ions, 0.58 for a
  beam at 6 :math:`v_T`), which shifted high-Z IAW peaks by 2.5–3e-3 nm
  (:math:`Z T_e/T_i \gtrsim 130` Maxwellian, :math:`\gtrsim 64` numeric; 0.83 of the
  peak on the high-Z test). The Maxwellian tail is now the asymptotic series
  :math:`1/\zeta^2 + 3/2\zeta^4 + 15/4\zeta^6 + 105/8\zeta^8`; the numeric tail is the
  moment series :math:`\sum_n (n+1) M_n/\xi^{n+2}` (moments 0–5 of :math:`f`) plus a
  residual decaying as :math:`(\xi_{edge}/\xi)^5` that matches the table at the edge. A
  distribution with a narrow feature near the edge, such as the 6 :math:`v_T` beam, is
  continuous but still off past the edge: 2.3 % at :math:`\xi` = 8.3, 15 % at 10 and 8 %
  at 15. Its moment series converges slowly there. A resonance that far out needs
  :math:`\alpha \gtrsim 8` for electrons or :math:`\beta \gtrsim 8` for numeric ions.
  Both are past the limits above, so such a result is flagged as likely under-resolved
  rather than corrected.
* **Numeric distributions with zero tails.** Zero density in the csv gave NaN in every
  pixel (log interpolation). Values below :math:`e^{-50}` are now raised to it with a
  warning, and negative values are rejected.

The production pipeline
-----------------------

``tests/pipeline/test_real_shot.py`` fits shot 101675 (two EPW lineouts) through
``runner.run`` three times: with L-BFGS-B, with LSQ plus a forced refit with the lineouts in
separate batches, and with LSQ and ``fe`` inactive in separate batches for MCMC. It reuses
those fits for every check:

* the known-good fit values of lineout 0 for both optimizers;
* the refit path;
* every postprocess entry point (local folder with ``defaults.yaml``/``inputs.yaml`` or
  with ``config.yaml``; remote run id with either layout), which must reproduce the fit
  to 1e-6;
* MCMC postprocessing with 60-step chains and two chains over two fit batches: the logged
  artifacts, the covariance diagonal against the sigmas, overrides reaching the
  postprocessor, R-hat, and the remote entry point. The sampler cannot sample an active
  EDF, so its fit is made with ``fe`` inactive from the start, as :doc:`mcmc` requires;
  deactivating ``fe`` by override on a fit with :math:`m` free is not supported.

The sampler's statistics are checked by S3, not here.

Unit and contract tests
-----------------------

* ``test_inverse/conftest.py`` makes one small fit of shot 101675 (10 epochs, ``fe``
  inactive), shared by the MCMC and Laplace unit tests for the whole session.
* ``test_utils/test_manifest.py`` compares the manifest of four artifact trees (1-D,
  angular, ion-only, no spectrogram) with ``tests/test_utils/golden/manifest_<case>.json``.
  The manifest format is the specification, so golden files are the right reference.
  ``tsadar_version`` and file sizes are masked (they depend on the platform);
  ``coords`` and ``data_vars`` are compared sorted. After an intended format change,
  regenerate with ``TSADAR_REGENERATE_GOLDEN=1 pytest tests/test_utils/test_manifest.py``
  and bump ``SCHEMA_VERSION`` for a rename or removal.
* ``test_runner/test_run_tags.py`` is two tables: config and mode → canonical tags, and
  fit behaviour (completes, raises, cancelled, unknown mode, dies mid-stage) → terminal
  status and error tag.
* ``test_inverse/test_mcmc_postprocess.py`` holds the chain-selection rules as one table
  of flag patterns → dropped chains, unreliable lineouts and parameters excluded from
  voting.
* ``test_inverse/test_laplace.py``: ``get_sigmas`` on a known Hessian with ne and Te
  (electron), ``amp1`` (general) and Ti (ion) active over two lineouts with different
  Hessians. The sigmas must equal :math:`\sqrt{\mathrm{diag}(J\,2H^{-1}J^T)}`, with
  :math:`J` from central differences of the parameter transform, in the columns
  ``get_fitted_params`` reports. It fails if a lineout's Hessian or Jacobian, or the
  column labels, are mixed up.
* ``test_inverse/test_mcmc.py``, calibration propagation: three draws with a Gaussian
  likelihood in Te centred at 0.55, 0.65 and 0.75 (width 0.01) through
  ``run_mcmc_pooled`` from a fit at 0.6. Each draw centres on its own shift with the
  likelihood's width, and the pooled spread is :math:`\sqrt{w^2 + \mathrm{var}}` of the
  shifts. Tolerances from 8 seeds: means 0.4 w (≤ 0.18 w), widths 30 % (≤ 24 %), pooled
  10 % (≤ 1 %).
* Numeric distributions are for forward calculations only (one spectrum cannot determine
  an EDF). ``fitter.fit`` and the fit-replay path reject ``type: numeric`` for ``fe`` or
  any ``fi``; ``calc_vs_data`` and every other distribution type are unaffected
  (``test_inverse/test_validate_inputs.py``). ``test_forward/test_numeric_edf.py`` loads
  electron and ion csvs with zero tails, through the full diagnostic and from the minimal
  numeric block (no ``params``). A numeric ion distribution with a 2-D electron
  distribution raises ``NotImplementedError`` until the 2-D path supports it (#165).

Future work
-----------

These are outside this suite on purpose and should be added with the code they need.

* **High phase velocity** (:math:`\alpha > 3.5`, weakly damped EPWs, positive-slope
  bumps): test with the resonance quadrature (peak refinement) together with the
  collisional model; the plain sampled form factor is not expected to pass there. Drop the
  :math:`\alpha` filter in ``test_sweeps.py`` then.
* **Collective relativistic regime:** the EPW phase velocity can approach :math:`c` at
  moderate :math:`T_e`, and the first-order factor fails much sooner than in the
  non-collective limit (Palastro et al., Phys. Rev. E 81, 036411, 2010, figs. 5–9:
  :math:`T_e \gtrsim 5` keV at :math:`n_e/n_c \sim 0.05`). The oracle needs Palastro's
  fully relativistic form factor.
* **ARTS (angular), with the ARTS / 2-V work** (`#165
  <https://github.com/ergodicio/tsadar/issues/165>`_): ARTS-1V at L3 against the oracle
  summed over angles (the A10 aperture test shows the pattern); 2-D cases A4 (anisotropic
  Gaussian mixtures with an analytic Radon projection) and A11 (l = 1 selection rules);
  the removed 2-D invariants as analytic checks; numeric ions on the 2-D path, which
  currently ignores them.

Adding a test
-------------

* A new plasma: add a ``Case`` to ``tsverify.cases`` (Gaussian mixtures or a
  ``QuadratureEDF`` for the distributions), release tsverify, and bump the pin. It is then
  checked at L1, L2 and L3.
* Oracle changes go to tsverify, with a self-test stating the equation or identity.
* Set tolerances from measured errors and record the measurement and what limits it on
  this page.
* Stay in the well-resolved regime unless the case uses the resonance-aware path.
* Record plain floats with ``record_property``, and mark expensive tests ``slow``.
