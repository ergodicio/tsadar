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
spectrum. These local oracles exclude instrument fitting. The production recovery cases
separately exercise ``ThomsonParams``, ``ThomsonScatteringDiagnostic``, IRFs,
``LossFunction``, parameter activation, and the configured production optimizer.
They use the shipped deck with explicit small reference settings and do not require
experimental files or a tracking server.

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

Every case ID appears in its test docstring. New integrated cases also include the
ID in their assertion diagnostics; focused NumPy checks report the mismatched arrays.

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
     - Scalar density self-consistency
     - A custom scalar MSE on the same forward implementation/grid has its minimum
       at the generating density; this is not production inverse validation
     - Absolute density error below 0.002 in units of :math:`10^{20}\,\mathrm{cm}^{-3}`
     - Fast
   * - ``P-NONCOLL-01``
     - Anisotropic non-collective limit
     - For a shifted Gaussian with covariance :math:`\Sigma`, the analytic Radon
       projection is :math:`N(\hat k\cdot\mu,\hat k^T\Sigma\hat k)`.
       As density decreases, :math:`\epsilon\to1` and the absolute wavelength-space
       spectrum approaches that projection times the physical prefactors
     - Final peak-normalized error below 0.5%; screening below 0.002, scaling with
       density; require at least 10x spectral-error reduction before the grid floor
     - Fast
   * - ``P-INV-02``
     - Production EPW and IAW recovery
     - Recover :math:`(n_e,T_e)` or :math:`(V_a,T_i)` through the diagnostic, IRF,
       loss and SciPy fitting loop, with other parameters fixed
     - 3% parameter error (plus 0.01 flow units), 2% detector relative L2, and
       at least 100x objective reduction from displaced initial conditions
     - Fast
   * - ``P-INV-03``
     - Production ARTS2D recovery
     - Recover :math:`(n_e,T_e)` for a fixed Maxwellian 2-V EDF at three angles;
       use root-aware detector quadrature, the noise-aware angular objective,
       active-leaf sensitivity validation and the partitioned Optax loop
     - 3% parameter error, 2% detector relative L2, and at least 100x objective
       reduction; truth doubles velocity, sinogram, root and integration grids
     - Slow/nightly
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

Selected focused coverage
-------------------------

The following existing tests now carry ``physics`` markers and stable case IDs.
Their scientific assertions are retained; configuration/plumbing-only neighbors
are not added to the selection. These cases run in the fast lane.

.. list-table:: Focused invariant inventory
   :header-rows: 1
   :widths: 20 40 40

   * - IDs / module
     - Reference or invariant
     - Numerical contract
   * - ``P-CHI-01/02``; ``test_arts2d_consistency.py``
     - Maxwellian Faddeeva real/imaginary susceptibility, parity, and the pole tangent
     - 1025 velocity nodes; signed near-zero, exact-grid and adjacent points, and
       tails to |xi|=8.2. Component rtol/atol 5e-4; real tail rtol 1e-3 without an
       absolute floor. Zero-pole tangent rtol 5e-4
   * - ``P-RADON-01``; ``test_arts2d_consistency.py``
     - Shifted anisotropic Gaussian projection, normalization, mean, and variance
     - Analytic Gaussian projection at axial and oblique angles; 129 velocity nodes;
       tolerances retained from the focused test's rotation-discretization checks
   * - ``P-FLOW-01/02``, ``P-ISOTROPIC-01``; same module
     - Parallel/perpendicular flow, charge-weighted frame invariance, and matched
       isotropic 1D/2D spectra
     - Exact frame algebra near roundoff; spectrum tolerance covers the different
       one- and two-dimensional susceptibility discretizations
   * - ``P-SINO-01..08``; ``test_sinogram.py``
     - Exact rotation reference, angular-grid convergence, periodic seam, angle/EDF
       AD gradients, end-to-end 2D agreement, independent finite differences
     - Value error falls from 1e-4 to 5e-6 over 256..1024 angles; EDF-gradient L2
       error below 2%; finite-difference beta-gradient error below 2e-5 with step
       refinement at a point away from interpolation knots
   * - ``P-MARGINAL-01..04``, ``P-MOMENT-01``; ``test_arts2d_marginal.py``
     - Analytic Maxwellian marginal, independent adaptive 3-V integration,
       anisotropic positivity, projection commutation, normalization and moments
     - Shape-specific adaptive-integration rtol 5e-8..6e-6; normalization and mean
       within 2e-15; in-plane thermal second moment within 2e-6
   * - ``P-IRF-01..03``; ``test_irf_area.py``
     - Unresolved-line area/centroid, constant-density interior, real nonuniform
       angular calibration
     - Area within 2e-11, spectral centroid within 2e-11 nm, nonuniform angular
       centroid within 1e-3 degrees; constant interior within 1e-6

The physical-root coverage and phase/refinement checks on spectra and gradients
in ``test_unresolved_arts2d.py`` remain in the slow lane. Its lightweight geometry
layout and aperture-weighting test runs in the fast PR suite.

Recovery design and tolerance evidence
--------------------------------------

``P-INV-02`` generates truth with 512 velocity and 2048 wavelength samples and fits
with 256/1024 samples. Both use 128 detector pixels and nonzero instrumental widths.
The legacy 1D binning gives slightly different detector centers at these resolutions,
so truth is interpolated onto the fit detector centers. This removes exact-grid
self-consistency, but is still a same-model recovery test, not an independent proof
of the forward physics. Independent oracles are supplied by ``P-MAXWELL-01`` and
``P-NONCOLL-01``.

The EPW deck uses the physical central notch; otherwise an under-resolved IAW can
control the legacy spectrum's peak normalization even though the ion feature is
excluded from the EPW loss. Nuisance amplitudes are fixed at 100 detector units.
The recovered physical parameters and residuals are written as JUnit properties
when a report is requested (use ``-o junit_family=legacy --junitxml=physics.xml``).

``P-INV-03`` uses identical detector edges for truth and fit. Its blue-wing window
avoids the central ion feature. Only density and temperature are active; it does not
establish arbitrary-EDF identifiability, withheld-angle prediction or uncertainty
coverage. Those belong to ISS#140 and the wider ISS#151 campaign.

Measured CPU float64 checks (JAX 0.10.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following are measured errors, not the acceptance thresholds above. Doubling
both 1D generation/fit velocity and wavelength resolutions reduces the residuals;
the 3% parameter and 2% spectrum bounds retain margin for backend variation.

.. list-table:: Recovery and resolution evidence
   :header-rows: 1
   :widths: 22 40 38

   * - Case
     - Recovered parameters (truth)
     - Detector relative L2
   * - 1D EPW
     - ne=0.200391 (0.2), Te=0.598048 (0.6)
     - 0.9366%; 0.4577% with both grids doubled
   * - 1D IAW
     - Va=1.500645 (1.5), Ti=0.0800274 (0.08)
     - 0.1956%; 0.0962% with both grids doubled
   * - ARTS2D EPW
     - ne=0.200373 (0.2), Te=0.597356 (0.6)
     - 0.6159%; objective 47.922 -> 0.025698

For ``P-NONCOLL-01``, the peak-normalized spectral error at the final density drops
from 0.1943% on 129 velocity nodes to 0.0614% on 257 nodes. On the latter grid,
decreasing density from 0.02 to 0.0002 to 0.000002 (units of 1e20 cm^-3) reduces
the error from 81.263% to 1.406% to 0.0614%; maximum |epsilon-1| decreases from
7.6298 to 0.07626 to 0.0007626. The test therefore checks the approach to the
limit as well as the final grid error. These CPU checks do not constitute GPU
validation or the full backend/resolution matrix.

Remaining ISS#151 coverage
--------------------------

This PR advances ISS#151; it does not close the entire expanded test plan.

1. Full-spectrum simultaneous rotational/mirror covariance and first-harmonic
   perpendicular-null/parallel-sign-reversal selection rules.
2. Exact aperture and gradient collapse invariants, plus full-spectrum Doppler
   covariance rather than only midpoint and kernel-level checks.
3. A component-level oracle for the production 1D arbitrary-EDF susceptibility,
   and temperature/ionization scans across collective asymptotic validity limits.
4. Explicit zero-width IRF and additional free-form smoothing/moment invariants.
5. A resolution/backend matrix over wavelength, velocity, angular-projection and
   root-scan grids, float32/float64 and real CPU/GPU executions. Compare spectra,
   roots, integrated areas and AD gradients.
6. Noisy multi-seed combined EPW/IAW and anisotropic EDF recovery, including
   calibration perturbations, interval coverage and visible/null-space diagnostics.

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
