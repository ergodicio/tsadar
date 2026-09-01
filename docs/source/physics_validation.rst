.. _physics_validation:

Physics-validation suite
========================

Purpose
-------

The physics-validation suite catches model changes that execute successfully but
violate Thomson-scattering physics. It complements unit, configuration, and frozen
output tests. A frozen spectrum can identify a code change, but it cannot establish
that either version is physically correct.

The suite is selected with the ``physics`` pytest marker. Pull requests run the fast
cases; the scheduled reference workflow runs cases marked ``physics`` and ``slow``.

.. code-block:: console

   pytest tests/ -m "physics and not slow"
   pytest tests/ -m "physics and slow"

The second command is also the entry point for a NERSC allocation. The allocation,
account, modules, and pre-installed environment must be chosen for the actual system;
they are intentionally not hard-coded in the repository.

Reference plasma and observables
--------------------------------

The initial integrated cases use an isotropic Maxwellian electron distribution, one
hydrogen ion species, a 526.5 nm probe, and 60-degree scattering. They exercise the
production ``FormFactor`` from physical parameters through the wavelength-space
spectrum. MLflow, plotting, detector calibration, and YAML merging are excluded so a
failure is attributable to the model or physical oracle.

Peak tests compare the plasma-wave detuning

.. math::

   \Delta\omega = |\omega_s - \omega_L|,

not the total scattered-light frequency. Since
:math:`\omega_L \gg \Delta\omega`, a percent-level comparison of
:math:`\omega_s` can pass even when the predicted IAW or EPW shift is wrong by order
unity. The former ``test_epw.py`` and ``test_iaw.py`` checks had exactly this failure
mode and were replaced by the cases below.

Implemented inventory
---------------------

Every case ID appears in its test docstring and failure messages.

.. list-table:: Integrated physics cases
   :header-rows: 1
   :widths: 16 23 31 18 12

   * - ID
     - Feature
     - Invariant or reference
     - Tolerance rationale
     - Lane
   * - ``P-MAXWELL-01``
     - Full Maxwellian spectrum
     - Independent Faddeeva evaluation of
       :math:`\chi_s=[1+\zeta_s Z(\zeta_s)]/(k\lambda_{Ds})^2`, dielectric response,
       electron and ion terms, and the wavelength Jacobian
     - Peak-normalized :math:`L_\infty < 5\%` and relative :math:`L_1 < 2.5\%`;
       observed errors are about 2.6% and 1.1%
     - Fast
   * - ``P-MULTI-01``
     - Multispecies bookkeeping
     - Splitting one ion population into two identical species leaves
       :math:`\bar Z`, susceptibility, and spectral density unchanged
     - ``rtol=1e-12``; the algebraic invariant is exact and observed error is near
       machine precision
     - Fast
   * - ``P-EPW-01``
     - Electron feature
     - Bohm--Gross:
       :math:`\Delta\omega^2=\omega_{pe}^2+3k^2T_e/m_e`
     - 5% of detuning; higher-order kinetic corrections reach 4.3% at
       :math:`k\lambda_{De}=0.46\text{--}0.51`, while grid error is below 0.1%
     - Fast
   * - ``P-EPW-02``
     - Density response
     - Since :math:`\omega_{pe}\propto\sqrt{n_e}`, increasing density moves both EPW
       peaks away from the laser line
     - Require at least 10%; the reference change produces more than 20%
     - Fast
   * - ``P-IAW-01``
     - Ion feature
     - Screened warm-ion approximation:
       :math:`\Delta\omega=k\sqrt{[ZT_e/(1+k^2\lambda_{De}^2)+3T_i]/m_i}`;
       zero flow also centers the doublet in frequency
     - 3% of detuning covers a 2.4% kinetic departure; midpoint error must be below
       0.2% of the half-separation
     - Fast
   * - ``P-IAW-02``
     - Ion bulk flow
     - The IAW midpoint obeys :math:`\delta\omega=kV_a`
     - 2%; grid refinement and peak interpolation are much smaller
     - Fast
   * - ``P-INV-01``
     - Parameter recovery
     - A noiseless EPW spectrum has its minimum loss at the density that generated it
     - Absolute density error below 0.002 in units of :math:`10^{20}\,\mathrm{cm}^{-3}`
     - Fast
   * - ``P-ORACLE-01``
     - Deliberate inconsistency
     - Reversing the measured low- and high-density responses must be rejected by the
       outward-motion oracle
     - Uses the same 10% margin as ``P-EPW-02``
     - Fast
   * - ``P-ORACLE-02``
     - Deliberate inconsistency
     - Giving one member of an otherwise identical species split a different ion
       temperature must be rejected by the exact split/merge oracle
     - Uses the ``1e-12`` split-invariance threshold; the mutation changes the spectrum
       by order unity
     - Fast
   * - ``P-ORACLE-03``
     - Known inconsistency
     - The retired :math:`k=2k_L` backscatter approximation at 60 degrees passes a 1%
       carrier-frequency comparison but must fail a 3% detuning comparison
     - The mutant's detuning error exceeds 100%
     - Fast
   * - ``P-EPW-REF-01``
     - Density/angle matrix
     - Bohm--Gross over three densities, three angles, and a 4096-point wavelength grid
     - 6% bounds the largest kinetic correction in the matrix; grid error is below 0.1%
     - Slow/nightly

The equations use the Gaussian-cgs conventions implemented by TSADAR. The standard
kinetic spectral-density and dispersion references are Sheffield, Froula, Glenzer, and
Luhmann, *Plasma Scattering of Electromagnetic Radiation* (2010), also cited in
:ref:`ts_fundamentals`.

Existing focused coverage
-------------------------

Several focused tests support the integrated battery without being its primary
end-to-end reference:

* ``tests/test_form_factor/test_arts2d_consistency.py`` compares the Maxwellian
  susceptibility with the analytic Faddeeva function and checks parity, Galilean
  invariance, species ordering, Radon moments, and isotropic 1D/2D agreement.
* ``tests/test_forward/test_irf_area.py`` checks area and centroid preservation through
  instrument-response corrections.
* ``tests/test_forward/test_unresolved_arts2d.py`` exercises unresolved-root topology
  and runs in the slow reference lane.

High-value additions
--------------------

The following cases are the recommended implementation backlog, in priority order.

1. Add a detector-level synthetic recovery through ``ThomsonScatteringDiagnostic`` and
   the production loss path. Recover ``(n_e,T_e)`` from an EPW spectrum and
   ``(V_a,T_i)`` from an IAW spectrum with all nuisance parameters fixed.
2. Add the non-collective limit. For :math:`k\lambda_{De}\gg1`, susceptibilities and
   ion scattering vanish and the corrected spectral shape must approach the projected
   electron distribution. A shifted anisotropic Gaussian gives an analytic reference.
3. Add exact aperture and gradient invariants: splitting one angular weight among
   duplicate angles must not change a detector spectrum, and reversing a symmetric
   density/temperature gradient must not change its averaged spectrum.
4. Add 2D mirror covariance. Reflecting
   :math:`f(v_x,v_y)\rightarrow f(v_x,-v_y)` while reflecting the scattering and flow
   angles must leave the complete spectrum invariant.
5. Add a slow resolution/backend matrix over wavelength, velocity, angular-projection,
   and root-scan grids, including values, peak roots, integrated areas, and AD gradients.
6. Add noisy, multi-seed combined EPW/IAW recovery generated on a finer grid than the
   fit model. Keep this on NERSC/nightly because it is a stress test, not a pull-request
   gate.

Adding a case
-------------

New physics tests belong in ``tests/test_physics`` and must follow these rules:

* Assign a stable case ID and add it to the inventory above.
* State the equation, exact invariant, or trusted external reference in the test
  docstring. A frozen array without provenance is an implementation regression, not a
  physics reference.
* Assert on the physics-scale observable. For example, compare a peak detuning rather
  than its optical carrier and use width-weighted :math:`L_1` or peak-normalized
  :math:`L_\infty` rather than pointwise relative error in zero-valued tails.
* Derive tolerances from resolution refinement, approximation error, and backend/dtype
  variation. Record that evidence here; do not loosen a threshold without explaining
  which contribution changed.
* Use deterministic inputs and seeds. Include the case ID, observable, measured value,
  and allowed value in failure messages.
* Keep network, MLflow, and plotting side effects out of pull-request tests.
* Mark computationally expensive tests ``@pytest.mark.slow``. Run them with the
  scheduled reference command before changing their reference or tolerance.
* When practical, add a sensitivity control that deliberately violates the invariant.
  The test should demonstrate that its oracle rejects the inconsistent result without
  mutating production code.

Reference artifacts, if unavoidable, must record their generator, source citation,
units, complete input configuration, TSADAR commit, backend/dtype, and checksum.
