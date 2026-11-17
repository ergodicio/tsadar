.. _physics_validation:

Physics-validation suite
========================

Purpose
-------

The physics suite checks that TSADAR computes the right physics, not only that it
runs. Every check compares TSADAR with an analytic answer that shares no code with it.
A frozen spectrum can show that something changed but not which version is right, so
the suite has none.

The suite is selected with the ``physics`` pytest marker. Pull requests run the fast
cases (about a minute); the scheduled workflow adds the cases marked ``slow``.

.. code-block:: console

   pytest tests/ -m "physics and not slow"
   pytest tests/ -m "physics and slow"

Design: one oracle, checked at four depths
------------------------------------------

``tests/oracles.py`` evaluates the collisionless spectral density with NumPy and SciPy
only. Every susceptibility comes from the Faddeeva function,
:math:`\chi_s = [1 + \zeta_s Z(\zeta_s)]/(k\lambda_{Ds})^2` with
:math:`Z(\zeta) = i\sqrt{\pi}\,w(\zeta)`, for Maxwellian or Gaussian-mixture
distributions of every species (a mixture's :math:`\chi` is the sum of shifted, scaled
Faddeeva terms). The same oracle is compared with TSADAR at each depth of the code:

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Depth
     - TSADAR code under test
     - Oracle side
   * - L1 kernel
     - ``FormFactor._numeric_chi``, ``zprimeMaxw``
     - :math:`\chi(\xi)` of the same distribution
   * - L2 spectrum
     - ``FormFactor``, absolute units
     - :math:`S(k,\omega)` with the :math:`(1 + 2\omega/\omega_L)` factor,
       :math:`r_e^2 n_e` and :math:`d\omega/d\lambda`
   * - L3 detector
     - ``ThomsonScatteringDiagnostic``: IRF, pixel grouping, oversampled grid, gradients,
       aperture weights, fit-range normalization
     - the L2 oracle through an independent detector model: continuous Gaussian blur,
       exact average over each pixel, the same normalization rule
   * - L4 inverse
     - ``LossFunction``, the fitting loop, Laplace uncertainty, MCMC
     - data made by the oracle (or, where stated, TSADAR's noiseless spectrum at a known
       truth)

L3 is also the plumbing test: a shift, scale or smearing introduced anywhere between
the form factor and the detector fails against physics. Cases are data
(``tests/physics/cases.py``), so a new plasma adds a check at every depth.

Non-Maxwellian distributions reach TSADAR the way a user gives one: written to a CSV
file and loaded as ``type: numeric`` (``fe`` for electrons, ``fi`` for ions).

Scope: the well-resolved regime
-------------------------------

Every TSADAR case keeps :math:`\alpha = 1/(k\lambda_{De}) \le 3.5` at its features and
has no nearly undamped resonance (:math:`\min|\epsilon| \gtrsim 0.25`). TSADAR's
susceptibility error, about :math:`10^{-4}` and set by its fixed internal velocity grid,
is amplified by :math:`1/|\epsilon|^2` at a weakly damped resonance: measured 1–8 % of
the EPW peak for :math:`k\lambda_{De}` = 0.15–0.25, and 0.6 % for a positive-slope
bump-on-tail. Testing at higher phase velocity needs the resonance quadrature (peak
refinement) together with the collisional model; the test files carry a note to extend
them when that path is ready.

The notch filter is not modelled exactly (a rectangular attenuation applied on the
sampled grid) and its band is excluded from fits. The detector comparisons leave out
the pixels within two IRF widths of its edges, and no test asserts its shape.

Inventory
---------

IDs follow ``TEST_SUITE_AUDIT.md``. Measured values are CPU float64.

.. list-table::
   :header-rows: 1
   :widths: 9 22 39 30

   * - ID
     - File
     - Check
     - Tolerance (measured)
   * - A1
     - ``test_kernels``, ``test_spectra``, ``test_detector``
     - Maxwellian electrons and ions: hydrogen EPW, EPW with electron drift, hydrogen
       IAW with flow, CH IAW with a relative ion drift. L2 at 256 and 2048 wavelengths
     - L2 :math:`L_\infty` 1e-3, :math:`L_1` 5e-4 (6.2e-4, 3.2e-4); L3 EPW 1.5e-3,
       IAW 1e-3 (1.0e-3, 2.4e-4); peaks 5e-3 / 5e-4 nm (1e-3)
   * - A3
     - same
     - Gaussian-mixture EDFs: two-temperature, hot-tail shoulder, return-current beam
       (EPW and IAW), flat-top, ion beam, two-humped ions
     - L1 :math:`\chi` 4e-4 of max :math:`|\chi|` (1.1e-4 to 2.9e-4); L2 2.5e-3 /
       1.5e-3 (up to 1.8e-3 for the two-humped ions, falling to 1.2e-4 with a
       4096-point internal grid)
   * - A5
     - same, ``test_oracle``
     - A heavy-tailed EDF through the numeric input (TSADAR has no kappa model): kappa 3 and 5
       electrons (Student-t, unit variance) on the EPW and IAW. The oracle's :math:`\chi` is its
       own singularity-subtracted quadrature, checked against Faddeeva for a Gaussian
     - As A1: L1 :math:`\chi` 2.0e-4 / 2.9e-4; L2 6.0e-4 / 3.9e-4; L3 3.5e-4. Quadrature vs
       Faddeeva 2e-7
   * - A10
     - ``test_averaging``
     - 20 % ne and Te gradient over 5 points; zero gradient = one point; three-angle
       aperture; species reordering and splitting
     - As L3; the gradient and aperture change the spectrum by 2–6 %. Reorder/split
       to 1e-10
   * - A13
     - ``test_notch``
     - ``spectrum_breakdown`` returns the model spectrum
     - 1e-12
   * - A6
     - ``test_oracle``
     - The oracle alone: for Maxwellian species at one temperature,
       :math:`\int S\,d\omega` equals the Debye–Hückel structure factor
       :math:`(k^2+\sum_s k_s^2)/(k^2+k_e^2+\sum_s k_s^2)` (H, C, CH;
       :math:`k\lambda_{De}` = 0.3, 1, 3)
     - 1e-4
   * - A2
     - ``test_oracle``
     - The oracle alone: EPW peak
       :math:`(\omega/\omega_{pe})^2 = 1 + 3a + 6a^2 + Zm_e/M`
       (:math:`a = k^2\lambda_{De}^2 = 0.01`); ion-acoustic peak at the screened sound
       speed for :math:`T_i/T_e \le 0.1`
     - :math:`20a^3` (:math:`10a^3`); 3e-3 (1e-3)
   * - Z′
     - ``test_kernels``
     - ``zprimeMaxw`` against the Dawson-function form; ``legacy`` reproduces the old
       table; ``other.zprime`` reaches both form factors
     - 1e-13; bit-identical
   * - A12
     - ``test_relativistic``
     - The oracle reproduces Selden's tables (Phys. Lett. A 79, 405, 1980). tsadar's
       first-order relativistic factor :math:`(1 + 2\omega/\omega_L)` against Selden's exact
       non-collective spectrum at 0.5–5 keV, 45–135°
     - Tables to 1e-4 / 5e-4; shape :math:`3\,T_e/m_ec^2` (0.66–2.31); peak within
       1.2× Selden's second-order shift (0.47–0.93). Without the factor: 22–30
       :math:`T_e/m_ec^2`
   * - S1
     - ``test_inverse``
     - Noiseless recovery from oracle data: (ne, Te) from the EPW, (Va, Ti) from the
       IAW, displaced starts, production loss and L-BFGS-B
     - 1e-3 relative (5e-6 to 2.1e-4)
   * - S2
     - ``test_inverse``
     - Laplace :math:`\sigma` equals the Cramér–Rao bound from the oracle's
       finite-difference Jacobian, :math:`F = J^T \mathrm{diag}(1/d) J`
     - 1 % (0.07–0.09 %)
   * - S3
     - ``test_inverse``
     - At :math:`10^4` counts the MCMC posterior is centred on the truth with the Laplace
       width; acceptance near 0.234
     - mean 0.2 :math:`\sigma` (0.03); width 10 % (2.5–2.8 %)
   * - Sweep
     - ``test_sweeps`` (slow)
     - A1 at L2 over ne 0.05–0.5, Te 0.1–2 keV, 30–120°, the 18 of 27 plasmas with
       :math:`\alpha \le 3.5`
     - :math:`L_\infty` 2e-3 on the EPW wings

S2 and S3 use TSADAR's noiseless spectrum at the truth as data. With oracle data at
:math:`10^4` counts the :math:`10^{-3}` model/oracle difference is statistically
significant on the low-count wings and adds a residual term to the Hessian. The truth
is the physical value the activated parameters actually hold: the inverse activation
in ``ts_params`` is a stabilized logit, :math:`\log(0.01 + x/(1.01 - x))`, a deliberate offset
that avoids failures of the exact logit, so a deck value of ne = 0.2 starts a fit at 0.2044.

Defects the suite found
-----------------------

* **Electron f′.** The Landau-damping term took f′ as a forward difference along the
  wavelength samples, i.e. at :math:`\xi_{i+1/2}`, stored at sample i. It is now taken on
  the internal velocity grid, as for numeric ions. On shot 101675 (1024 samples over
  323–743 nm) the fit moves by ne −1.0 %, Te +1.5 %, m +2.8 %, amp2 +12 % at the same
  loss.
* **Ion Z′ table.** The real part of the ``rdWT.txt`` table differs from the exact
  :math:`Z'` by up to 1.6e-3 (:math:`\mathrm{Re}\,Z'(1)` = 0.153356 vs 0.152318),
  about 0.65 % on an IAW feature. The default is now the Faddeeva value;
  ``other.zprime: legacy`` restores the table for comparisons with codes that use it.

Real-data pipeline
------------------

``tests/pipeline/test_real_shot.py`` is not a physics check. It fits shot 101675
through ``runner.run`` with two optimizers and reuses those fits for the known-good fit
values, the refit path, every postprocess entry point and MCMC postprocessing.

Remaining coverage
------------------

1. ARTS-1V (angular, 1-D EDF) at L3, with the ARTS / 2-V work on that code.
2. The high-phase-velocity regime, with the resonance quadrature and collisions; the collective
   relativistic regime with Palastro et al.'s fully relativistic form factor (Phys. Rev. E 81, 036411,
   2010), where the first-order factor fails at :math:`T_e \gtrsim 5` keV for :math:`n_e/n_c \sim 0.05`.
3. The 2-D (ARTS2D) cases, which return with that code.

Adding a case
-------------

* Add a ``PhysicsCase`` to ``tests/physics/cases.py`` when the oracle can describe it;
  it is then checked at L2 and L3, and its EDF at L1.
* State the equation or identity in the docstring and keep the oracle free of TSADAR
  imports.
* Set tolerances from measured errors and record them, with what limits them, next to
  the constant and in this page.
* Stay in the well-resolved regime unless the case uses the resonance-aware path.
* Mark expensive cases ``@pytest.mark.slow``.
