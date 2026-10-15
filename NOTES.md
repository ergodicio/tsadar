# TSADAR Research Notes

Append-only record of numerical-physics investigations and the decisions needed to reproduce them.

## 2026-08-30 — PLANNED: ARTS2D issues #134 and #135

- Provenance: triage performed from `origin/main` at `5b46f06c` (`v0.3.0`). No simulation result has been produced yet.
- Issue #135 and Avi Milder's comment establish that the 2-D Radon projection direction must depend only on `k_hat`, electron drift must enter through `ud dot k_hat`, `atan2` must replace the hand-rolled quadrant calculation, and the 2-D imaginary susceptibility must use the 1-D Fourier/Landau sign convention.
- Repository documentation defines `ud` as electron drift relative to the ion fluid. The current 1-D expression algebraically treats it as an absolute lab-frame electron velocity, while the current 2-D expression treats it as relative to one legacy common ion flow.
- OPEN DECISION (#135): for multiple ion flows, choose the reference ion-fluid velocity used with relative electron drift. Candidate conventions are the charge-weighted bulk velocity `sum(fract_s * Z_s * Va_s) / Zbar` (order-independent) or ion-1 velocity (preserves the current hidden first-species reference).
- Issue #134 has no comments. The current implementation point-samples the narrow dielectric resonance, uses sampled Gaussian PDFs without discrete normalization, peak-restores after convolution, and later bin-averages detector resolution units.
- OPEN DECISION (#134): use a local Lorentzian/residue correction for unresolved roots of `Re(epsilon)` or root-aware direct quadrature of the full integrand. The former is cheaper and fixed-shape/JAX-friendly but is a local approximation that must be checked against a highly refined reference.
- Proposed detector convention awaiting confirmation: return bin-mean spectral density and test conserved area as `sum(bin_mean * bin_width)`; remove only the hidden IRF peak restoration in #134 and leave the explicit post-bin amplitude nuisance normalization for #139.
- Planned dependency: implement #135 first; stack #134 on #135 because both change `calc_in_2D`, and the #134 small-angle reproduction includes nonzero electron drift.

## 2026-08-30 — DECISION: correctness over compatibility for #134/#135

- USER (verbatim): "there are no other users to consider so dont worry about breaking changes to the API and changing the intent fo variables etc. our objective is correctness, Avi is the primary user"
- Consequence for #135: use the documented relative-drift convention, remove stale legacy `general.Va` plumbing, and define the multi-ion reference with an explicit order-independent ion bulk velocity rather than preserving the hidden ion-1 behavior.
- Consequence for #134: prefer root-aware integration of the full spectral integrand where practical; use a local unresolved-mode approximation only if it is quantitatively validated against a highly refined reference and the error is inside the declared detector-space tolerance.
- Retained scope decisions: detector bins represent mean spectral density; the hidden IRF peak restoration is removed; post-bin fitted amplitude handling remains separate; no collision model or fixed imaginary `epsilon` floor is introduced.

## 2026-08-30 — DECISION: compare both unresolved-resonance methods

- USER (verbatim): "hmm, we should consider implementing both for the unresolved resonance. we can see if theres a significant speedup with the direct quadrature + local correction"
- Plan: implement the full root-aware direct-quadrature path and the direct-quadrature-plus-local-correction path behind one detector-bin integration interface.
- Planned comparison: validate detector-space values, conserved area, grid-phase/refinement stability, and parameter gradients against the same highly refined reference; measure first-call compile-plus-execution time separately from warmed steady-state execution.
- No method has been selected as the default yet. Selection will be based on measured accuracy and runtime.

## 2026-08-30 — RESULT: #135 signed susceptibility and flow-frame correction

- The 2-D projection direction is now `atan2(khat_y, khat_x)`, the resonance coordinate is signed, and the imaginary susceptibility uses the same Fourier/Landau sign as the 1-D implementation.
- Each ion retains its own lab-frame `Va`; the electron lab velocity is the charge-weighted ion bulk velocity plus the relative `ud`. The configuration schema and every shipped example deck were migrated from `general.Va` to per-ion `Va`.
- A pole exactly on a sampled velocity node uses the symmetric principal-value limit. Since a general piecewise-linear sampled EDF has no finite knot derivative, its optimizer tangent is explicitly defined as the centered slope across one velocity cell; this convention is stable across float32 and float64.
- Validation: 31 focused susceptibility/sinogram tests passed. These cover analytic Maxwellian values and tangents, projection signs, drift geometry, multi-ion ordering/Galilean invariance, JIT/gradients, per-ion configuration, and 1-D/2-D spectral agreement. A broader run reached 52 passing tests; the two failures were existing MLflow-connected tests unable to reach the configured tracking server, and the run was then stopped.

## 2026-08-30 — DECISION: reject local residue correction for #134

- The corrected 22-degree Maxwellian benchmark has a dielectric root at 474.199678891 nm with HWHM 3.49399e-4 nm; the adaptive detector-area reference is 0.6525638697 after a 0.9 nm FWHM Gaussian response.
- Across five scan-grid phases, full root-aware quadrature had 0.0393–0.0399% detector L1 error and 0.0569–0.1035% loss-gradient L2 error. The first-order local epsilon/numerator residue correction had 0.0138–0.3162% detector error but 15.5–210.5% gradient error.
- Corrected CPU medians (100 interleaved blocking calls): full/local warm values were 3.721/3.722 ms and value-plus-gradient calls were 3.805/2.622 ms. Cold value calls were 0.602/0.726 s; cold value-plus-gradient calls were 2.290/2.467 s. There is no useful forward speedup, and the faster local backward pass is numerically wrong.
- Consequence: implement only the full root-aware quadrature in production. Record the local-correction prototype as a rejected benchmark rather than exposing an inference-unsafe method through the API.

## 2026-08-30 — RESULT: detector-edge and IRF pipeline audit for #134

- ARTS data preparation averages raw wavelength pixels into resolution-unit mean densities. The calibrated wavelength arrays retained by plotting and loss code are bin centers; exact grouped outer edges are now stored separately as `other.detector_specs.electron_wavelength_edges`, including the true width of a ragged final group.
- The legacy spectral and angular Gaussian samples are now normalized to discrete unit sum, and the hidden post-convolution peak restoration was removed. An unresolved contained delta-line test preserves area to `1e-12` in float64, while its peak is broadened rather than restored.
- The ATS IRF can now apply angular blur without a second spectral blur when the root-aware quadrature has already returned spectral detector-bin means. Nineteen focused detector/IRF/vectorization tests passed on CPU.
- Remaining limitation observed during the audit: the calibrated ARTS angle axis is nonuniform (adjacent spacing 0.0771–0.1598 degrees), whereas the legacy angular `jnp.convolve` assumes translation in sample index. Discrete normalization fixes grid-dependent amplitude scaling but does not by itself make angular blur exactly coordinate-aware.

## 2026-08-30 — RESULT: production root-aware detector quadrature for #134

- The production kernel finds sign-changing roots of `Re(epsilon)` on a fixed wavelength scan, differentiates each bisection root implicitly, tan-maps Gauss--Legendre nodes through the local complex-linearized resonance, and integrates exact Gaussian-CDF probability mass into nonuniform detector bins. No collision term or fixed dielectric floor was added.
- Every gradient/scattering-angle geometry is integrated independently before aperture weighting. A memory-bounded `lax.map` avoids materializing the complete angle-by-node-by-detector tensor. Downstream ARTS processing applies only the angular IRF and angular resolution reduction; it does not convolve or bin wavelength a second time.
- Fatal numerical conditions (root-capacity overflow, zero local width, non-finite evaluation, or invalid edges/IRF/source bounds/scan phase) poison detector bins with NaN and are also exposed through fixed-shape diagnostics.
- In the physical 22-degree CPU regression, the 256-panel result varied by at most `2.10e-6` in detector relative L1 and `2.06e-5` in drift-gradient relative L2 across three scan phases. Refining 128 to 256 panels changed values by at most `1.17e-5` and gradients by at most `1.06e-4`, well inside the declared 1% tolerance.
- Synthetic root benchmarks preserve a constant density and a contained unresolved line's area, support nonuniform detector edges, and match transformed reference values and implicit gradients. Focused production-plumbing tests verify geometry ordering, aperture weighting, calibrated centers/edges, and that no second spectral IRF or spectral reduction occurs.

## 2026-08-30 — RESULT: physical-coordinate spectral and angular IRFs for #134

- This entry supersedes the nonuniform-angular-axis limitation recorded above. On the real 1024-pixel ARTS angular calibration, the legacy index-space convolution gave a centered unresolved line physical area of `1.04699` and shifted its centroid by about `0.084` degrees; its effective physical width also varied with the local pixel spacing. Even-length uniform spectral grids had an additional half-sample centering error.
- Spectral and angular blur now use an analytic bin-to-bin Gaussian response matrix. Each element exactly integrates the unit-area Gaussian over a finite source cell and output cell on the calibrated physical coordinate, then divides by the output-cell width to return a bin-mean density. This supports even-length and nonuniform axes without peak restoration or an index-spacing approximation.
- CPU regressions on the real nonuniform angular calibration preserve a centered line's physical area to `2e-11` and centroid to `1e-3` degrees. Uniform 400- and 801-point spectral grids preserve line area to `2e-12` and centroid to `2e-11`; constant densities remain unity away from finite-detector boundaries.

## 2026-08-30 — RESULT: full-range root-scan and refinement validation for #134

- The representative 449–670 nm, 22-degree ARTS2D spectrum has eight sign-changing roots of `Re(epsilon)` at `[474.201875844894, 514.081647830358, 526.276288254028, 526.451996163684, 526.548008175237, 526.723802255372, 539.209067615960, 591.042039725023]` nm. The closest pair is separated by `0.0960120` nm; a root capacity of four was therefore both insufficient and capable of returning a poisoned model over the real detector range.
- Across 39 scan phases from `-0.95` through `0.95`, 2048 root-scan panels missed a close pair in six phases. Both 4096 and 8192 panels found all eight roots at every phase; their maximum solved-root phase spreads were `1.36e-12` and `6.82e-13` nm. Warm single-geometry medians were `23.5`, `32.2`, and `55.8` ms for 2048, 4096, and 8192 panels. The correctness-first default is therefore 4096 root-scan panels, independent of a 256-panel regular integration grid, with static capacity for 16 roots.
- A physical narrow-line regression comparing 64 integration panels against 128 and 256 panels at three phases had maximum detector relative-L1 changes of `4.94e-5` and `3.96e-5`; maximum drift-gradient relative changes were `1.86e-4` and `2.64e-4`. The 256-panel phase spreads were `2.10e-6` for values and `2.06e-5` for gradients, all well below the one-percent acceptance threshold.
- The full-range regression includes phases at which 2048 panels failed, requires exactly the eight roots above, and verifies finite nonzero reverse-mode drift gradients. Its first version exposed NaN cotangents from statically evaluated inactive tan-map candidates; replacing only those inactive evaluations with finite full-panel sentinel mappings fixed the VJP without adding an epsilon floor or changing any active resonance segment.
- Data-fitting mode carries exact calibrated detector edges, including ragged resolution groups. Forward-only decks have no edge calibration, so `forward_epw_start`/`forward_epw_end` retain their documented legacy role as the first and last wavelength centers; finite outer edges are inferred by half-spacing extrapolation rather than silently reinterpreting those inputs as detector boundaries.
- The analytic response is stored densely only for the modest nonuniform angular detector axis. Uniform spectral grids use the identical Toeplitz bin-integrated Gaussian kernel, with FFT convolution above 2048 points; this keeps the legacy 10,240-point 1D path memory-linear instead of materializing an approximately 800 MiB float64 response plus intermediates. Dense-versus-Toeplitz regressions agree to `5e-11`, and the 10,240-point 1D forward snapshot remains green.

## 2026-08-30 — RESULT: final #134 validation

- The local-correction comparison and root-scan timings were measured on Apple arm64 with Python 3.14 and JAX 0.9.1. The production conclusion is unchanged: local correction offered no forward speedup and had unacceptable loss-gradient errors, while the full root-aware method remained below the one-percent value/gradient target.
- The stored ARTS1D golden spectrum was regenerated because it encoded the removed index-space convolution. Replaying the legacy response matched the old snapshot to `5.72e-4` maximum absolute error; the physical-coordinate result differed by `0.104201` at a sharp EPW point. Independent delta-response checks showed that the legacy angular kernel shifted that point by `0.1138` degrees and broadened a requested `0.42466`-degree sigma to `0.5848` degrees, while the analytic response shifted it by `0.0020` degrees and produced `0.4276` degrees after finite-bin integration.
- Final local CPU validation, with the optional HDF4 dependency installed and MLflow redirected to a file store, completed with `186 passed, 5 skipped` in `196.46` seconds. The five skips are GPU-only tests; the only warning is MLflow's filesystem-backend deprecation. `git diff --check` and Python byte-compilation also passed.

## 2026-08-30 — RESULT: #134 GPU gradient-test audit

- The first manually dispatched stacked-branch workflow exposed two independent runner outcomes: the hosted CPU runner was recycled at 64% after 122 passing tests and one skip (exit 143, no assertion or traceback), while the GPU runner completed the suite and reported that the new scalar-drift detector Jacobian regressions attempted 12.5 GB and 16.0 GB allocations.
- The GPU allocations came from using reverse-mode `jacrev` for one scalar input and hundreds of detector-bin outputs. This orientation repeats the reverse pass for every output and does not represent the scalar-loss gradient used by inference.
- The regression now forms the complete scalar-to-detector Jacobian with one forward-mode `jacfwd` tangent evaluation, and independently checks a weighted scalar detector loss with reverse-mode `grad` against the forward-Jacobian contraction. This preserves elementwise tangent coverage and production-style reverse-mode coverage without the artificial quadratic memory cost.
- The focused CPU file remained green (`3 passed`) and fell from `81.72` to `63.87` seconds; the full-range eight-root test fell from `24.46` to `12.87` seconds.

## 2026-08-30 — RESULT: #136 ARTS2D harmonic coordinates and trainability

- Provenance: implementation and local CPU validation started from `origin/main` at `c5360f74` with JAX 0.9.1 in the project `uv` environment.
- The ARTS2D harmonic convention is now explicit: a real projected 3-D spherical-harmonic basis embeds the physical plane as `(X, Y, Z) = (vy, 0, vx)` and calls JAX with `(degree=l, order=m, polar theta, azimuth phi)`. The real `(1, 0)` and `(1, 1)` modes are proportional to `vx / |v|` and `vy / |v|`, respectively; this coordinate repair does not define the physical 3-V-to-2-V marginal tracked by #137.
- Both Mora-Yahi coefficients are scalar JAX array leaves. The distribution filter exposes them and the isotropic shape parameter; the neural-network filter now exposes every weight and bias for every configured `(l, m)` through `Nl`.
- Controlled perturbations of either first-order coefficient produced finite nonzero EDF and two-angle spectrum changes. Reverse-mode gradients of weighted spectra were finite, nonzero, and agreed with centered finite differences to `1e-6` relative tolerance.
- The focused harmonic tests cover Cartesian parity and centroids, active-leaf gradients, `Nl=2` construction, unnormalized-state reporting, and Equinox serialization. The combined harmonic, ARTS2D consistency, spectral-term, and sinogram suite completed with `41 passed` in `67.61` seconds on CPU.

## 2026-08-30 — RESULT: #136 full-suite validation

- With MLflow's file-store compatibility flag enabled and the declared optional HDF4 dependency installed, the complete local CPU suite finished with `192 passed, 7 skipped` in `213.63` seconds. The skips are hardware-dependent tests; the eight warnings are existing SciPy `disp` option warnings.
- `git diff --check` and Python byte-compilation passed before commit preparation.

## 2026-08-31 — CORRECTION: #136 odd-grid harmonic origin

- Review of PR #145 identified that the nominal origin of an odd Cartesian velocity grid can be represented by roundoff-sized nonzero coordinates. The previous `radius > 0` branch therefore assigned an arbitrary direction there; its fallback also left the `(l, m) = (1, 0)` mode nonzero at exact zero.
- Every anisotropic (`l > 0`) projected harmonic is now defined as zero when radius is within 32 machine epsilons of the velocity-coordinate scale. This catches only the coordinate singularity, remaining far below one velocity-cell spacing in both float32 and float64.
- An explicit 49-by-49 regression verifies zero first-order harmonics at the nominal origin and Cartesian parity on the full odd grid. The focused harmonic file completed with `7 passed`; the combined harmonic, ARTS2D consistency, spectral-term, and sinogram suite completed with `42 passed` on CPU.
## 2026-08-30 — RESULT: #137 true ARTS2D Cartesian marginal

- Provenance: implementation was stacked on the #136 prerequisite commit `e7dd720f` and validated locally on CPU with JAX 0.9.1.
- The `sphericalharmonic` path now constructs a positive normalized 3-V hypothesis in coordinates `(X, Y, Z) = (vy, vz, vx)` and integrates `vz` with Gauss-Legendre quadrature before returning the ARTS2D Cartesian marginal. A bounded exponential log-anisotropy replaces the hard clip; density normalization occurs on the 3-V model before marginalization.
- The `arbitrary` path is declared a native Cartesian 2-V marginal. Its isotropic super-Gaussian scale now enforces normalized in-plane thermal second moment 2 for every shape, rather than reusing a 3-V radial scale on a normalized central slice.
- The analytic `m=2` marginal agreed with the 2-D Maxwellian to `6.11e-10` maximum absolute error and `3.86e-9` maximum relative error above `1e-10`. Adaptive infinite-range references for `m=3` and `m=5` agreed at the tested points within `4e-6` and `5e-8` relative tolerance, respectively.
- A representative two-mode anisotropic marginal agreed with a separately refined 384-node rule within `6e-6` relative tolerance. Direct projection of its normalized 3-V hypothesis and projection of the computed 2-V marginal agreed to roundoff for a cardinal view.
- Across `m = 2, 3, 5`, both the 3-V-hypothesis marginal and native 2-V initializer had density 1, zero first moments, and in-plane thermal second moment 2 within `2e-6`. Three in-plane Radon views of the `m=3` hypothesis agreed with direct 3-V integration within `3e-6` relative L1 and preserved projected density within `1e-8`.
- The complete local CPU suite finished with `203 passed, 7 skipped` in `190.21` seconds. The skips are hardware-dependent tests; the eight warnings are existing SciPy `disp` option warnings.

## 2026-08-31 — CORRECTION: #137 endpoint-shape trainability

- Review of PR #147 identified that exact logistic endpoint representations (`normed_m = -inf` for `m=2` and `+inf` for `m=5`) have zero shape gradient and freeze an otherwise active fit parameter.
- The normalized shape is now stored directly on the closed unit interval and projected with `clip` before mapping to `m` in `[2, 5]`. JAX's centered boundary subgradient preserves the exact endpoint value while providing a finite nonzero derivative toward the feasible interior.
- Enabling the `m=5` endpoint derivative exposed reverse-mode underflow in the division used to form relative harmonic coefficients after the super-Gaussian tail rounded to zero. Flooring that denominator at `sqrt(machine tiny)` keeps the division VJP finite; it changes only tails whose isotropic density is already below that threshold.
- Marginal fourth-moment regressions at both accepted endpoints produce finite nonzero AD gradients, agree with centered finite differences within `2e-4` relative tolerance, and verify that an inward parameter step changes the physical shape. The focused harmonic and marginal files completed with `20 passed`; the combined ARTS2D suite completed with `55 passed`; the complete local CPU suite completed with `206 passed, 7 skipped` and only the eight existing SciPy `disp` warnings.
## 2026-08-30 23:41 PDT — RESULT: node-safe principal-value quadrature for #138

- Provenance: implementation and CPU validation on branch `codex/issue-138-ratintn-pv`, based on `origin/main` at `c5360f74`, using Python 3.14, JAX 0.9.1, and interpax 0.3.12.
- This entry supersedes the exact-node symmetric-offset convention recorded in the `#135 signed susceptibility and flow-frame correction` entry above. The quadrature now applies analytic singularity subtraction against a C2 cubic spline of the sampled numerator and evaluates the two removable bracketing divided differences from factored cubic coefficients. It does not displace or floor the pole.
- Direct and precomputed-operator paths now include all `N - 1` intervals. On CPU, every interior node and midpoint of a 17-point grid had finite primal, JVP, pole VJP, and numerator VJP values in both float32 and float64; representable node-minus/node-plus evaluations agreed continuously with the analytic node value and tangent. A GPU was not available locally; the same dtype-parametrized tests will run on the configured CI device.
- Against analytic Maxwellian plasma-dispersion references on 65/129/257 point grids over `[-8, 8]`, maximum errors decreased from `8.13e-5` to `5.98e-6` to `6.73e-7` for `Re Z`, and from `4.08e-4` to `1.18e-5` to `1.58e-6` for `Re Z'`.
- In the physical 22-degree ARTS2D root benchmark, the maximum root error relative to an independently evaluated analytic-Maxwellian susceptibility decreased from `4.48e-2` nm at 129 velocity points to `1.12e-2` nm at 257 and `2.79e-3` nm at 513. The fixed 129-point root golden was refreshed to the corrected values.
- The corrected ARTS1D snapshot changed 64 of 880,640 entries beyond `1e-3`; the maximum absolute change was `0.00320072` and the largest local relative change was `0.0170912`. The refreshed snapshot passed when reloaded in a separate run.
- Final local CPU validation used a file-backed MLflow store and headless Matplotlib: `176 passed, 7 skipped` in `249.92` seconds. The seven skips were GPU-only; the only warning was MLflow's filesystem-backend deprecation. `git diff --check` and Python byte-compilation also passed.

## 2026-08-31 — RESULT: bounded GPU regression and rational-fallback derivative coverage for #138

- The first PR GPU job (`33365443014`, job `99405011867`) ran the new `ratintn` correctness file in about two minutes and the production-grid operator file in 13 seconds. It progressed through 62% of the full suite, then produced no further output after `test_forward/test_sinogram.py` while entering `test_forward/test_unresolved_arts2d.py`; the workflow cancelled at its six-hour limit. The timeout was therefore outside the rational-integral changes.
- The full production-grid operator regression is pinned to CPU again. GPU CI now first asserts that JAX exposes an actual GPU, then runs a bounded 17-point regression covering exact nodes and midpoints, float32 and float64, primal evaluation, pole JVP/VJP, numerator VJP, and direct/operator equivalence. The job has an explicit 20-minute timeout.
- The nonsingular rational fallback now has analytic constant-over-affine tests with the affine root outside the integration domain. Both the exact-log and small-difference Taylor coefficient branches check finite full-denominator VJPs and compare forward- and reverse-mode shift derivatives with the closed-form integral derivative.
- A representative weighted numerator VJP replaced an all-output reverse Jacobian in the every-pole test, retaining coverage while avoiding one reverse pass per output. Focused local CPU validation completed with `27 passed, 4 skipped` in `60.13` seconds; the skips are the new GPU-only cases.

## 2026-08-31 — DECISION: keep expensive physics regressions out of per-PR CI

- The two physical unresolved-ARTS2D convergence/topology regressions are marked `slow` and excluded from normal pull-request CI. The lightweight production-plumbing test in the same file remains in the fast suite.
- Pull requests carry an explicit checklist requiring both `pytest -m "not slow" tests/` and `pytest -m slow tests/` to have been run before submission, with ownership or a linked result recorded for the slow run. For PR #146, the pre-submission full CPU run already covered both slow regressions and completed with `176 passed, 7 skipped`.
- Automated weekly execution on NERSC is deliberately deferred until the NERSC/GitHub Actions integration is designed and configured; no provisional scheduler or runner setup was added here.

## 2026-08-31 — RESULT: hosted `gpu_runner` does not expose a usable GPU

- Two fresh hosted jobs (`99621504504` and retry `99624590900`) installed JAX 0.10.2 with its CUDA 12 plugin, but both failed at the explicit device gate before pytest: `cuInit(0)` returned CUDA error 303 and JAX reported only the CPU backend. This also explains why the earlier nominal GPU suite could run for six hours rather than exercising accelerator behavior.
- The workflow does not treat CPU fallback as GPU success. The bounded GPU regression remains checked in, but its hosted job is opt-in through `GPU_CI_ENABLED` until a working runner (planned NERSC/GitHub Actions integration) exists. GPU-sensitive pull requests carry a checklist item assigning a person or linked run to execute that regression on an actual GPU before submission.

## 2026-10-15 — REFERENCE: ARTS2D recovery benchmark guide (moved from the docs site)

- Provenance: this guide was published on the documentation site as `docs/source/arts2d_recovery_benchmark.rst`, added in `218946d2` (PR #154, 2026-09-24). It was taken off the user-facing docs because it records the ISS140 benchmark design and its preflight results rather than user instructions. The text below is unchanged apart from conversion from reStructuredText to Markdown. The benchmark code it describes is `tsadar/benchmarks/arts2d.py`, tested by `tests/test_inverse/test_recovery_benchmark.py`.

ISS140 measures what a fitted EDF recovers under finite angular coverage, in
addition to its training-spectrum error. The executable benchmark uses the
production `ThomsonScatteringDiagnostic`, root-aware detector quadrature,
angular IRF, EDF parameterizations, and the measured-variance likelihood/global
gain solver introduced in ISS139. It does not require shot data, a calibration
file, an MLflow server, or a GPU for its small smoke preset.

### Run the CPU smoke case

From an environment containing the package and test dependencies:

```
JAX_PLATFORMS=cpu python -m tsadar.benchmarks.arts2d \
    --preset smoke --output results/iss140-smoke

pytest -q tests/test_inverse/test_recovery_benchmark.py
```

The output directory must be new. The smoke preset fits all four EDF models for
two gradient evaluations each, using one elliptic truth, one noise seed, three
angles, and eight wavelength bins. Truth uses 64 velocity cells per direction;
fits use 32. The inversion grid uses 16 sinogram angles, 128 root-scan panels,
and 16 integration panels; truth doubles each resolution. These resolutions
avoid the negative photon signals found on the earlier 16-cell inversion grid.
The interval is 470--515 nm, avoiding the far-tail interpolation error encountered
with this deliberately coarse grid on a wider interval.
This exercises the complete path, including automatic differentiation and the
detector SVD. Its recovery scores are not a scientific ranking or a convergence
claim. The smoke test belongs to ordinary CPU PR CI.

### Run the full GPU experiment

Use a CUDA-capable installation of `tsadar[gpu]` and an allocated GPU:

```
python -m tsadar.benchmarks.arts2d --preset full \
    --output results/iss140-full
```

The full preset requires a JAX GPU backend unless `--allow-cpu` is explicitly
specified. Device allocation and scheduler submission are external to this
command. No job is automatically submitted. Optional MLflow logging uses the
existing tracking configuration:

```
python -m tsadar.benchmarks.arts2d --preset full \
    --output results/iss140-full-tracked \
    --mlflow-experiment arts2d-recovery-iss140
```

The full preset has four truths, two nominal photon levels (1,000 and 10,000
peak counts before the gain), ten common seeds (0--9), and four fit models: 320
fits. Every fit gets 500 Adam gradient evaluations, learning rate 0.01, global
gradient norm clipping at 100, and no restarts or early stopping. This fixes the
number of forward/gradient evaluations; parameter counts, FLOPs and wall time
still differ between architectures. Both wall time and first-evaluation time
(including compilation) are saved.

JSON overrides select bounded pilots or resolution studies. Unknown field names
are rejected. For example, save the following as `pilot.json`:

```json
{
  "seeds": [0, 1, 2],
  "truths": ["elliptic"],
  "peak_counts": [1000.0],
  "steps": 50
}
```

Then run:

```
python -m tsadar.benchmarks.arts2d --preset full --config pilot.json \
    --output results/iss140-pilot
```

A refinement override can set `nvx=256`, `n_beta=1024`,
`root_scan_panels=4096` and `integration_panels=256`. The truth refinement
factor remains two. Compare the saved truth-grid discrepancy and seed-level
scores before interpreting a model ranking. The benchmark reports discretization
mismatch and whether its RMS is below one noise standard deviation; choosing different grids alone does not establish convergence. The full
GPU campaign is an executable experiment specification, not a result implied by
the CPU smoke test.

Full-preset forward preflight (CPU, JAX 0.10.2, float64) produced finite,
nonnegative spectra on both grids for all four truth families. The RMS grid
mismatch in noise standard deviations was:

Forward discretization check, 2026-09-22:

| Truth | 1,000 nominal peak counts | 10,000 nominal peak counts |
| --- | --- | --- |
| Maxwellian | 0.074 | 0.240 |
| Elliptic | 0.130 | 0.421 |
| Super-Gaussian | 0.215 | 0.693 |
| Skew tail | 0.058 | 0.190 |

These checks evaluate the forward model on the declared two resolutions. They
do not include fitting or establish convergence beyond that comparison. The
full GPU recovery campaign has not been run as part of this implementation.

### Controlled inputs and held-out predictions

- Truth families are positive analytic 2-V densities: Maxwellian, rotated
  elliptic Gaussian, isotropic order-three super-Gaussian, and a weak displaced
  warm tail. They are independently sampled and cell-normalized on truth and fit
  grids. The default rotation is 27 degrees.
- Full fits use 128 velocity cells per direction; truth uses 256. Truth also doubles
  sinogram angles, root-scan panels, and integration panels. Both use identical
  physical detector bin edges and IRFs. Full fit settings are 512 sinogram angles,
  2,048 root-scan panels, 128 integration panels, regular order 8, and root order 32.
  This is the same forward algorithm at different resolution, not an independent
  physical oracle.
- The full detector covers 470--515 nm in 64 bins, with nine scattering angles
  from 30 to 110 degrees. Angles at indices 1, 4 and 7 (40, 70 and 100 degrees)
  and detector columns `[22,26)` are withheld. The union is the held-out set;
  separate angle and wedge scores are also reported. The wedge-only score excludes
  already-withheld angles, so these two diagnostic subsets do not overlap.
- The clean signal has a global gain of 1.15. Photon noise is Poisson on
  signal plus 10 background counts, with Gaussian read noise of standard deviation
  3 counts. All models share the exact same generated observations and expected
  noise variance for a case/seed. The known synthetic variance is deliberately
  fixed during inference and supplied as `e_variance`; this is an idealized
  calibrated-noise experiment, not a fitted noise model.
- Electron density, temperature, drift, ion parameters, detector calibration, and
  amplitude parameters are held fixed and recorded. A single unanchored global
  gain is profiled with the production likelihood using training samples only.
  That gain predicts all held-out samples; held-out observations never enter gain
  estimation, gradients, or checkpoint selection. There is no EDF regularization
  in this baseline.
- Every model starts at the same physical Maxwellian. Arbitrary 2-V uses the
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
improves while EDF error worsens. `history.json` reports train, held-out and EDF
metrics at the configured interval (25 updates in the full preset).

### Visible and null recovery

The comparison uses a common Cartesian EDF grid, independent of the fitted
parameterization. If the cell area is `a`, define `q = sqrt(a) * f`. The
Euclidean error in `q` is the discrete physical EDF L2 error. The local
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
keeps singular values greater than `max(svd_atol, svd_rtol * s_max)`; defaults
are 1.0 and 1e-6. The absolute threshold declares sensitivity above one noise
standard deviation per unit EDF L2 displacement. It is a configurable coordinate
and sensitivity convention, not an assertion that every retained direction is
recoverable for finite displacements.

The saved `edf_visible_l2` is the norm of the error projected onto retained
right singular vectors. `edf_null_l2` is the norm of the orthogonal remainder,
including exact null directions and weak directions below the declared cutoff.
Their squared norms sum to `edf_l2**2`. No dense square null-space matrix is
formed. Singular values, Fisher eigenvalues, numerical rank, retained rank,
threshold, Jacobian, and visible basis are saved. These are local measurements
at the common truth and may change under another linearization or SVD threshold.

### Physical metrics and uncertainty

All moments use the production cell-center quadrature. They include normalization,
flow, an electron-current proxy, in-plane kinetic energy and temperature, the
2-by-2 pressure tensor and its eigenvalue anisotropy, and in-plane heat flux.
Both actual moments and signed errors from truth are saved. Velocity units are
`vTe = sqrt(Te/me)`. The current proxy is `-integral(v*f)`; physical current
requires multiplication by `e*n_e*vTe`. These are moments of a 2-V marginal.
They do not determine the missing out-of-plane pressure or full 3-V heat flux.

`summary.json` reports mean metrics with 95% percentile bootstrap intervals over
all configured noise seeds, separately for each truth, photon level and fit model.
It also reports paired differences against arbitrary 2-V, using the same seeds.
There are 2,000 bootstrap resamples with seed zero. A single-seed smoke result
has `ci95: null`. Intervals describe seed variability for these fixed truth
families and initialization; they do not include physical model uncertainty.

### Artifacts and provenance

The local result directory contains:

- `benchmark.json`: complete resolved specification, hash, git SHA/status,
  dependency versions, actual devices and precision.
- `forward_config.json` and `truth_forward_config.json`: resolved production
  settings, including exact physical detector edges and forward resolutions.
- A directory per truth/photon level with `case.npz`, `case.json` and
  `subspaces.npz`: both truth grids, noise calibration, masks, coarse/refined
  truth spectra, their whitened discrepancy, moments and SVD data.
- A directory per seed with the shared `observations.npy` and one directory per
  fitted model containing `fit_config.json`, `provenance.json`,
  `history.json`, `metrics.json`, `recovery.npz` and `best_weights.eqx`.
  The parameterization is taken from the instantiated model class and radial
  model type, with active leaf paths, shapes and counts. Truth labels are stored
  separately and cannot override the fit identity.
- `status.json` records running, failed, or complete state and timestamps.
  `runs.json` is updated after each completed fit, and `summary.json` after all
  configured runs finish. Nonfinite predictions, gradients, or invalid photon
  spectra fail the run rather than being removed from scores. Raw photon signals
  on both truth grids and at every fitted iterate must be finite and nonnegative
  at every detector pixel, including held-out angles and wedges. This check uses
  zero tolerance and no clipping, before background addition or pixel masking.
  Zero signal is allowed; any negative value fails even if the background would
  make the total counts positive.

MLflow is optional. When enabled, each run name contains the actual fit model,
truth, seed and photon level; histories, best metrics, configurations, observed
data, checkpoints and common SVD artifacts are uploaded. A final summary run
contains the seed intervals and links to the individual run IDs. Local artifacts remain
the complete record even without a tracking service.
