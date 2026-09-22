ARTS2D recovery benchmark
========================

ISS140 measures what a fitted EDF recovers under finite angular coverage, in
addition to its training-spectrum error. The executable benchmark uses the
production ``ThomsonScatteringDiagnostic``, root-aware detector quadrature,
angular IRF, EDF parameterizations, and the measured-variance likelihood/global
gain solver introduced in ISS139. It does not require shot data, a calibration
file, an MLflow server, or a GPU for its small smoke preset.

Run the CPU smoke case
----------------------

From an environment containing the package and test dependencies::

    JAX_PLATFORMS=cpu python -m tsadar.benchmarks.arts2d \
        --preset smoke --output results/iss140-smoke

    pytest -q tests/test_inverse/test_recovery_benchmark.py

The output directory must be new. The smoke preset fits all four EDF models for
two gradient evaluations each, using one elliptic truth, one noise seed, three
angles, and eight wavelength bins. Truth uses 32 velocity cells per direction;
fits use 16. The interval is 470--515 nm, avoiding the far-tail interpolation
error encountered with this deliberately coarse grid on a wider interval.
This exercises the complete path, including automatic differentiation and the
detector SVD. Its recovery scores are not a scientific ranking or a convergence
claim. The smoke test belongs to ordinary CPU PR CI.

Run the full GPU experiment
---------------------------

Use a CUDA-capable installation of ``tsadar[gpu]`` and an allocated GPU::

    python -m tsadar.benchmarks.arts2d --preset full \
        --output results/iss140-full

The full preset requires a JAX GPU backend unless ``--allow-cpu`` is explicitly
specified. Device allocation and scheduler submission are external to this
command. No job is automatically submitted. Optional MLflow logging uses the
existing tracking configuration::

    python -m tsadar.benchmarks.arts2d --preset full \
        --output results/iss140-full-tracked \
        --mlflow-experiment arts2d-recovery-iss140

The full preset has four truths, two nominal photon levels (1,000 and 10,000
peak counts before the gain), ten common seeds (0--9), and four fit models: 320
fits. Every fit gets 500 Adam gradient evaluations, learning rate 0.01, global
gradient norm clipping at 100, and no restarts or early stopping. This fixes the
number of forward/gradient evaluations; parameter counts, FLOPs and wall time
still differ between architectures. Both wall time and first-evaluation time
(including compilation) are saved.

JSON overrides select bounded pilots or resolution studies. Unknown field names
are rejected. For example, save the following as ``pilot.json``::

    {
      "seeds": [0, 1, 2],
      "truths": ["elliptic"],
      "peak_counts": [1000.0],
      "steps": 50
    }

Then run::

    python -m tsadar.benchmarks.arts2d --preset full --config pilot.json \
        --output results/iss140-pilot

A refinement override can set ``nvx=256``, ``n_beta=1024``,
``root_scan_panels=4096`` and ``integration_panels=256``. The truth refinement
factor remains two. Compare the saved truth-grid discrepancy and seed-level
scores before interpreting a model ranking. The benchmark reports discretization
mismatch and whether its RMS is below one noise standard deviation; choosing different grids alone does not establish convergence. The full
GPU campaign is an executable experiment specification, not a result implied by
the CPU smoke test.

Full-preset forward preflight (CPU, JAX 0.10.2, float64) produced finite,
nonnegative spectra on both grids for all four truth families. The RMS grid
mismatch in noise standard deviations was:

.. list-table:: Forward discretization check, 2026-09-22
   :header-rows: 1

   * - Truth
     - 1,000 nominal peak counts
     - 10,000 nominal peak counts
   * - Maxwellian
     - 0.074
     - 0.240
   * - Elliptic
     - 0.130
     - 0.421
   * - Super-Gaussian
     - 0.215
     - 0.693
   * - Skew tail
     - 0.058
     - 0.190

These checks evaluate the forward model on the declared two resolutions. They
do not include fitting or establish convergence beyond that comparison. The
full GPU recovery campaign has not been run as part of this implementation.

Controlled inputs and held-out predictions
-----------------------------------------

* Truth families are positive analytic 2-V densities: Maxwellian, rotated
  elliptic Gaussian, isotropic order-three super-Gaussian, and a weak displaced
  warm tail. They are independently sampled and cell-normalized on truth and fit
  grids. The default rotation is 27 degrees.
* Full fits use 128 velocity cells per direction; truth uses 256. Truth also doubles
  sinogram angles, root-scan panels, and integration panels. Both use identical
  physical detector bin edges and IRFs. Full fit settings are 512 sinogram angles,
  2,048 root-scan panels, 128 integration panels, regular order 8, and root order 32.
  This is the same forward algorithm at different resolution, not an independent
  physical oracle.
* The full detector covers 470--515 nm in 64 bins, with nine scattering angles
  from 30 to 110 degrees. Angles at indices 1, 4 and 7 (40, 70 and 100 degrees)
  and detector columns ``[22,26)`` are withheld. The union is the held-out set;
  separate angle and wedge scores are also reported. The wedge-only score excludes
  already-withheld angles, so these two diagnostic subsets do not overlap.
* The clean signal has a global gain of 1.15. Photon noise is Poisson on
  signal plus 10 background counts, with Gaussian read noise of standard deviation
  3 counts. All models share the exact same generated observations and expected
  noise variance for a case/seed. The known synthetic variance is deliberately
  fixed during inference and supplied as ``e_variance``; this is an idealized
  calibrated-noise experiment, not a fitted noise model.
* Electron density, temperature, drift, ion parameters, detector calibration, and
  amplitude parameters are held fixed and recorded. A single unanchored global
  gain is profiled with the production likelihood using training samples only.
  That gain predicts all held-out samples; held-out observations never enter gain
  estimation, gradients, or checkpoint selection. There is no EDF regularization
  in this baseline.
* Every model starts at the same physical Maxwellian. Arbitrary 2-V uses the
  production log-density representation. NN-SH and arbitrary radial-coefficient
  SH use degree two (degree one in smoke); Mora-Yahi is degree one. SH signs are
  zero initially. NN hidden layers have fixed seed 173 and a zero final sign head,
  retaining trainable output derivatives. Coefficient-SH magnitude leaves start
  at -4 to avoid the production default's tiny initial coefficient scale. These
  initialization choices and all instantiated active leaves are recorded.

The best checkpoint is the minimum training objective among the initial state,
each evaluated update, and the final state. It is saved with its exact evaluated
step, weights and metrics. The final state is also scored. This preserves the
loss/weight association required by ISS141 and exposes cases where training loss
improves while EDF error worsens. ``history.json`` reports train, held-out and EDF
metrics at the configured interval (25 updates in the full preset).

Visible and null recovery
-------------------------

The comparison uses a common Cartesian EDF grid, independent of the fitted
parameterization. If the cell area is ``a``, define ``q = sqrt(a) * f``. The
Euclidean error in ``q`` is the discrete physical EDF L2 error. The local
noise-weighted detector Jacobian is computed at the analytic truth sampled on
the inversion grid, with density normalization differentiated and the columns
projected orthogonally onto the zero-mass tangent. Each Jacobian row has zero
mean in the equal-area cell coordinates. This prevents normalization gauge
directions from contaminating the visible EDF basis. The Jacobian contains only
training detector rows and is computed in bounded batches of reverse-mode
cotangents. The fixed plasma parameters are conditioned on.

The fitted global gain is eliminated from the local Fisher information by
projecting the whitened Jacobian orthogonal to the whitened signal. A thin SVD
of that projected matrix supplies common visible EDF directions. The benchmark
keeps singular values greater than ``max(svd_atol, svd_rtol * s_max)``; defaults
are 1.0 and 1e-6. The absolute threshold declares sensitivity above one noise
standard deviation per unit EDF L2 displacement. It is a configurable coordinate
and sensitivity convention, not an assertion that every retained direction is
recoverable for finite displacements.

The saved ``edf_visible_l2`` is the norm of the error projected onto retained
right singular vectors. ``edf_null_l2`` is the norm of the orthogonal remainder,
including exact null directions and weak directions below the declared cutoff.
Their squared norms sum to ``edf_l2**2``. No dense square null-space matrix is
formed. Singular values, Fisher eigenvalues, numerical rank, retained rank,
threshold, Jacobian, and visible basis are saved. These are local measurements
at the common truth and may change under another linearization or SVD threshold.

Physical metrics and uncertainty
--------------------------------

All moments use the production cell-center quadrature. They include normalization,
flow, an electron-current proxy, in-plane kinetic energy and temperature, the
2-by-2 pressure tensor and its eigenvalue anisotropy, and in-plane heat flux.
Both actual moments and signed errors from truth are saved. Velocity units are
``vTe = sqrt(Te/me)``. The current proxy is ``-integral(v*f)``; physical current
requires multiplication by ``e*n_e*vTe``. These are moments of a 2-V marginal.
They do not determine the missing out-of-plane pressure or full 3-V heat flux.

``summary.json`` reports mean metrics with 95% percentile bootstrap intervals over
all configured noise seeds, separately for each truth, photon level and fit model.
It also reports paired differences against arbitrary 2-V, using the same seeds.
There are 2,000 bootstrap resamples with seed zero. A single-seed smoke result
has ``ci95: null``. Intervals describe seed variability for these fixed truth
families and initialization; they do not include physical model uncertainty.

Artifacts and provenance
------------------------

The local result directory contains:

* ``benchmark.json``: complete resolved specification, hash, git SHA/status,
  dependency versions, actual devices and precision.
* ``forward_config.json`` and ``truth_forward_config.json``: resolved production
  settings, including exact physical detector edges and forward resolutions.
* A directory per truth/photon level with ``case.npz``, ``case.json`` and
  ``subspaces.npz``: both truth grids, noise calibration, masks, coarse/refined
  truth spectra, their whitened discrepancy, moments and SVD data.
* A directory per seed with the shared ``observations.npy`` and one directory per
  fitted model containing ``fit_config.json``, ``provenance.json``,
  ``history.json``, ``metrics.json``, ``recovery.npz`` and ``best_weights.eqx``.
  The parameterization is taken from the instantiated model class and radial
  model type, with active leaf paths, shapes and counts. Truth labels are stored
  separately and cannot override the fit identity.
* ``status.json`` records running, failed, or complete state and timestamps.
  ``runs.json`` is updated after each completed fit, and ``summary.json`` after all
  configured runs finish. Nonfinite predictions, gradients, or invalid photon
  spectra fail the run rather than being removed from scores.

MLflow is optional. When enabled, each run name contains the actual fit model,
truth, seed and photon level; histories, best metrics, configurations, observed
data, checkpoints and common SVD artifacts are uploaded. A final summary run
contains the seed intervals and links to the individual run IDs. Local artifacts remain
the complete record even without a tracking service.
