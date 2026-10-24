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

## 2026-10-07 — DECISION: default velocity-grid extent raised from 6 to 8 vTe

- USER (verbatim): "Ok based off this test i would like to chagne the defaults used in the generation of edfs and velocity grids to 8vth, do this by changeing the hardcoded defaults not by adding a deck parameter".
- Context: branch `arts-reinstate`, working tree on top of `b77752a8`, uncommitted at the time of this entry. A root-aware (`integrate_detector_bins`) evaluation of the 1D-EDF ARTS fit `s94475-update-5` (MLflow experiment `189128`, run `f61027cd99cb4256a2ac8528e56c40d8`) returned `zero_width` for 10 of 241 scattering angles, 19.0–23.5 degrees. At 19.0 degrees the outer roots of `Re(epsilon)` were at 474.69 and 587.24 nm, phase velocities +6.85 and −7.19 vTe; the fitted velocity grid spanned ±5.95 vTe. The 1D form factor sets `log(fe) = -50` outside the velocity grid (`extrap=[-50, -50]`).
- Change: `vmax = 6.0 -> 8.0` in `tsadar/core/modules/distribution_functions/base.py` (1D grid and 2D Cartesian grid); `6.0 -> 8.0` in `spherical_harmonics.py` for the radial extent `vmax * 1.05 * sqrt(3)` and for the `vz` Gauss-Legendre nodes and weights. The extent stays a hardcoded constant; no deck parameter. The form-factor susceptibility tables (`xi1`, `xi2`) span ±8.2 and were not changed.
- Timing evidence preceding the decision: time-resolved test deck (`tests/configs/time_test_{defaults,inputs}.yaml`, shot 101675, electron spectrum, DLM EDF, batch of 2, L-BFGS-B), local CPU, median of 30 loss+gradient calls after compilation, grid extent varied through a temporary option that was removed afterwards.

  | vmax (vTe) | nvx | dv (vTe) | loss+grad (ms) | full fit (s) | final loss | Te / ne / m |
  |---|---|---|---|---|---|---|
  | 6 | 320 | 0.0375 | 5.56 | 6.8 | 0.05982283 | 0.6309 / 0.2257 / 3.1011 |
  | 8 | 320 | 0.0500 | 5.49 | 6.4 | 0.05982287 | 0.6310 / 0.2257 / 3.1012 |
  | 10 | 320 | 0.0625 | 5.57 | 6.5 | 0.05982346 | 0.6310 / 0.2257 / 3.1001 |
  | 8 | 426 | 0.0376 | 5.51 | 6.5 | 0.05982283 | 0.6309 / 0.2257 / 3.1007 |
  | 10 | 534 | 0.0375 | 5.81 | 6.3 | 0.05982283 | 0.6310 / 0.2257 / 3.1012 |

- Not measured: GPU timing; `arbitrary` 1D EDFs; any 2D or spherical-harmonic EDF, whose grids changed in the same edit. At unchanged `nvx`, the velocity spacing of every grid is larger by 8/6.
- Consequence for existing artifacts: `fitted_weights.eqx` files holding a gridded EDF were fitted on the ±6 vTe grid; loading them after this change assigns the same values to a ±8 vTe grid.
- State at the time of this entry: the grid-dependent test selection (`tests/test_forward`, `tests/test_form_factor`, `tests/test_spherical_harmonics.py`, `tests/test_arts2d_marginal.py`, `tests/test_inverse/test_recovery_benchmark.py`, `tests/test_inverse/test_1d_data.py`, `-m "not slow"`) was LAUNCHED on local CPU and had not reported. No fit has been run on the ±8 vTe grid.
- Open questions: which stored reference spectra and fit values change under the new grid; whether the 2D decks need a larger `nvx` to keep their velocity resolution.
- Shared-vault mirror skipped: no local Obsidian vault named 'Ergodic Research' was resolved on this host.

## 2026-10-07 — RESULT: grid-dependent tests and first root-aware evaluation on the ±8 vTe grid

- Follows the DECISION entry of the same date; same uncommitted working tree on `arts-reinstate`. Local CPU.
- Test selection named in that entry COMPLETED: `7 failed, 150 passed, 5 skipped, 2 deselected` in 516 s. Failures: `tests/test_spherical_harmonics.py::test_l1_real_harmonics_are_independent_cartesian_modes[49]`; five cases in `tests/test_arts2d_marginal.py` (`test_nonmaxwellian_3d_marginal_matches_adaptive_integration[3.0-4e-06]` and `[5.0-5e-08]`, `test_anisotropic_marginal_matches_refined_quadrature_and_stays_smoothly_positive`, `test_reduced_edf_normalization_and_temperature_moment_do_not_drift[_harmonic_model-2.0]`, `test_every_in_plane_projection_matches_direct_3d_integration`); and `tests/test_inverse/test_recovery_benchmark.py::test_production_arts2d_benchmark_smoke`, which raised `FloatingPointError: invalid production benchmark evaluation at step 1` while fitting `mora-yahi` after `arbitrary`, `nn-sh`, and `coefficient-sh` completed. `tests/test_arts2d_marginal.py` constructs its refined reference with a literal `6.0 * nodes`. Every test in `tests/test_forward`, `tests/test_form_factor`, and `tests/test_inverse/test_1d_data.py` in the selection passed. Failure causes were not investigated.
- First evaluation of the 1D root-aware path on the new grid, `configs/arts-1d` decks (shot 94475, background 94477, `fe.type: arbitrary`, `nvx: 128`), initial parameters: grid ±7.938 vTe; `fe` at the grid edge `4.289e-27`; `root_overflow`, `nonfinite`, and `zero_width` raised for 0 of 241 scattering angles; root half-widths from `2.89e-20` to `16.3` nm; initial objective `1600.568`; gradient finite; 3.3 s per loss+gradient call after compilation with `map_batch_size: 8`.
- No fit has been run on the ±8 vTe grid; no GPU timing exists for the 1D root-aware path.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — LAUNCHED: `s94475-update-6`, first 1D root-aware ARTS fit on the ±8 vTe grid

- USER (verbatim): "Start the session and launch the run i will want to compare the run time to the previous run 5".
- Source: `arts-reinstate` working tree on `b77752a83fc8`, dirty, session fingerprint `3a0e2f50d19d`, synced to `$PSCRATCH/tsadar-sessions/20261007T132342Z-10135-534/src`. Command: `python -u run_tsadar.py --cfg configs/arts-1d --mode fit`. Slurm job `59484959`, `shared_interactive`, 1 A100 80 GB, 2 h, account `m4490_g`; log `workdir/arts1d-update-6.log` under the session root.
- Deck: shot 94475, background 94477, `fe.type: arbitrary`, `nvx: 128`, lineouts 150–950. Differences from `s94475-update-5` (run `f61027cd99cb4256a2ac8528e56c40d8`): `other.resonance_quadrature.enabled: auto`, resolved on by the input check, with `map_batch_size: 8`; velocity grid ±8 vTe instead of ±6; MLflow run name. Same GPU type and QOS as run 5.
- State: LAUNCHED. MLflow run id not yet read back.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94475-update-6` COMPLETED; run time against `s94475-update-5`

- `s94475-update-6`: MLflow experiment `189128`, run `1fdcbc949861472b99dcb2643a107d3f`, Slurm job `59484959`, exit code 0, no flagged or non-finite evaluation reported in the log.
- Run time, both on one A100 80 GB under `shared_interactive`: minimization of 1000 steps took 437 s (2.29 steps/s, including about 46 s before the second step) for update-6 and 91 s (10.87 steps/s) for update-5. System-metrics start to stop: 10 min 38 s for update-6, 3 min 7 s for update-5. Shell wall time of the update-6 command: 686 s.
- Final objective reported by the progress bar: `3.40e+02` for update-6 (root-aware, ±8 vTe grid), `3.17e+02` for update-5 (legacy quadrature, ±6 vTe grid). The two runs differ in both the quadrature and the grid, so neither difference is isolated.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — PLANNED: `s94475-update-7` and `s94475-update-8`, 256 velocity points and a wider lineout range

- USER (verbatim): "Rerun at 256 points for run 7. When that completes run at 256 points with the start lowered to 20 and the end raised to 1000".
- Both: same working tree as `s94475-update-6` apart from the deck, same session (Slurm job `59484959`, one A100 80 GB), root-aware quadrature on, velocity grid ±8 vTe, `fe.type: arbitrary`. The smoothing window (`len: 45`, `sampling: 30`) is unchanged in grid points.
- `s94475-update-7`: `fe.nvx: 256` (was 128), lineouts 150–950.
- `s94475-update-8`: `fe.nvx: 256`, lineouts `start: 20`, `end: 1000` (binned rows 2–100 of 103), launched after update-7 exits with code 0.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94475-update-7` and `s94475-update-8` COMPLETED

- Both exit code 0 on Slurm job `59484959` (one A100 80 GB), experiment `189128`. Planned in the entry of the same date.
- `s94475-update-7` (run `927745f78ad04f5cbbdfad1dbda5ff10`; `nvx: 256`, lineouts 150–950, 36 lineouts after batch trimming message): 1000 steps in 447 s (2.24 steps/s); shell wall time 672 s; final objective `2.54e+02`. `s94475-update-6` at `nvx: 128`, otherwise identical, ended at `3.40e+02` in 437 s.
- `s94475-update-8` (run `a76d4c3556bb475f9ed936bd98697f6f`; `nvx: 256`, lineouts 20–1000): 1000 steps in 454 s (2.20 steps/s); shell wall time 654 s; final objective `5.38e+02`. The objective is normalized per valid pixel over a different set of rows than update-7.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — PLANNED: `s94475-update-9`, run 8 repeated with the probe wavelength active

- USER (verbatim): "repeat run 8 with lam on".
- Identical to `s94475-update-8` (`nvx: 256`, lineouts 20–1000, root-aware quadrature, ±8 vTe) except `parameters.general.lam.active: true` (`val: 526.5`, bounds 525–528). Same working tree and session (Slurm job `59484959`).
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94475-update-9` COMPLETED; row-range scoring of runs 7, 8, 9

- `s94475-update-9` (run `605a1ac927264452b8560d14f869b30c`, Slurm job `59484959`): exit code 0; 1000 steps in 457 s (2.19 steps/s); shell wall time 684 s; final objective `470.15` (data term `366.19`, total reduced loss `2.116`); fitted `lam = 525.238` nm (start 526.5, bounds 525–528), `Te = 1.316` keV, `ne = 0.4275e20`.
- Row-range scoring: each fit evaluated on local CPU by `forward_root_aware_arts1d.py --eval_rows 2:100 2:15 15:95 95:100` with all fitted parameters fixed and the detector gains re-profiled by `LossFunction._angular_data_objective`. Self-checks against each fit forward model were below `1e-12`; no angle was flagged. Objective (data term) per binned-row range:

  | run | fit rows | 2:100 | 2:15 | 15:95 | 95:100 |
  |---|---|---|---|---|---|
  | update-7 | 15:95 | 835.6 (731.7) | 4423.1 (4328.6) | 253.8 (171.7) | 409.9 (310.6) |
  | update-8 | 2:100 | 538.3 (436.3) | 1488.1 (1270.6) | 391.7 (310.1) | 382.1 (282.6) |
  | update-9 | 2:100 | 470.2 (366.2) | 1290.9 (1066.3) | 340.0 (258.8) | 358.6 (258.9) |

- Fitted values: update-7 `Te = 1.068`, `ne = 0.4212`; update-8 `Te = 1.355`, `ne = 0.4259`; update-9 as above.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — LAUNCHED: `s94481-update-1`, first fit of shot 94481 with the root-aware 1D path

- USER (verbatim): "I have update the local input deck and added the appropriate files to the data folder, copy them over and run shot 94481 on the gpu".
- Deck as edited by the user, `configs/arts-1d`: shot 94481, background 94483, lineouts 20–1000, `fe.type: arbitrary`, `nvx: 256`, `lam` active, ion `A: 83`, `Z: 25`, MLflow experiment `arts-fits`. Data files `ATS-s94481.hdf` and `ATS-s94483.hdf` synced with the working tree. Root-aware quadrature on, velocity grid ±8 vTe, angular calibration offset 445.39 (set from shot 94475).
- Source: `arts-reinstate` working tree on `b77752a83fc8`, dirty, session fingerprint `4d1eb1f7e32b`. Slurm job `59489306`, `shared_interactive`, one A100, 1 h. Log `workdir/s94481-update-1.log` under `$PSCRATCH/tsadar-sessions/20261007T152502Z-10973-25169`.
- State: LAUNCHED.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94481-update-1` COMPLETED

- MLflow experiment `189128`, run `95114c096e144f60a1e01942b7616d6a`; Slurm job `59489306`; exit code 0; 1000 steps in 463 s (2.16 steps/s); shell wall time 660 s. Launch entry of the same date gives the deck and source state.
- Final objective `1204.19` (data term `843.17`, total reduced loss `5.419`); the objective fell from `6.21e+03` at step 2.
- Fitted values: `Te = 1.4404` keV, `ne = 0.4354e20`, `lam = 525.340` nm, `amp1 = 0.9999`, `amp2 = 1.0240`. Fitted EDF: integral `1.0000`, mean `0.0797` vTe, second moment `0.8923`; `Te` times the second moment is `1.285` keV.
- Lineouts, the fit image, and the notch position for this shot were not inspected.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — LAUNCHED: `s94481-update-2`, shot 94481 with the detector roll-off corners masked

- USER (verbatim): "can you rerun the fit masking the corners? or does that require a bunch of handrolled code?".
- Identical to `s94481-update-1` except `data.bad_pixels`: 74 [binned row, wavelength bin] pixels in rows 2–6, 8–11, 18, 19, 21, 26, 98, 99, all with wavelength between 450 nm and the first bin of that row whose background (shot 94483, 3-row average) reaches 95% of a straight line fitted over 459–475 nm. Wavelength fit ranges unchanged (450–515, 540–650 nm). No code change; the mask is applied by `build_angular_batch`.
- Same session as update-1: Slurm job `59489306`, one A100. State: LAUNCHED.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — PLANNED: `s94475-update-10`, run 9 with the detector roll-off corners masked

- USER (verbatim): "Good now lets repeat this process with 94475 run 9's data ad deck. Setting the appropriate mask which should be nearly identicle. the rerun that as well once the current run finishes".
- Deck: the logged decks of `s94475-update-9` (run `605a1ac927264452b8560d14f869b30c`) plus `data.bad_pixels` and the new run name; stored at `workdir/decks/s94475-update-10` in the session and at `C:\tmp\run_u10` locally. Nothing else differs from update-9.
- Mask: same criterion as for 94481, applied to the 94475 background (shot 94477), kept only in the corner rows 2–11 and 96–99: 63 pixels. The same criterion on 94481 gives 54 pixels in those rows, 50 in common. The criterion also flagged 88 pixels in rows 16–23, 51–53, 74, 87–89 of the 94477 background; these were not masked.
- Queued on Slurm job `59489306` to start when `s94481-update-2` exits.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94481-update-2` COMPLETED

- Run `ace955aa1f4546f3a89f6911ac8cddc2`, Slurm job `59489306`, exit code 0; 1000 steps in 427 s (2.34 steps/s); shell wall time 582 s.
- Final objective `1172.88` (data term `813.43`, total reduced loss `5.278`); `s94481-update-1` without the mask: `1204.19` (`843.17`, `5.419`).
- Lineout 97 (binned row 99): residual sum in the plot title `2.59e+04`, against `4.37e+05` in update-1; no residual is plotted below 456 nm.
- Fitted values: `Te = 1.4650` keV, `ne = 0.4355e20`, `lam = 525.341` nm; EDF second moment `0.8780`; `Te` times the second moment `1.286` keV (update-1: `1.285`).
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94475-update-10` COMPLETED

- Run `9916668411af4316a72cf546c84ee56d`, Slurm job `59489306`, exit code 0; 1000 steps in 426 s (2.35 steps/s); shell wall time 580 s; started 16:09:57Z.
- Final objective `455.06` (data term `352.04`, total reduced loss `2.048`); `s94475-update-9` without the mask: `470.15` (`366.19`, `2.116`).
- Fitted values: `Te = 1.3836` keV, `ne = 0.4282e20`, `lam = 525.232` nm; EDF second moment `0.8087`, mean `0.0851` vTe; `Te` times the second moment `1.119` keV (update-9: `Te = 1.3163`, second moment `0.8485`, product `1.117`).
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — LAUNCHED: `s94475-update-11`, run 10 repeated after the grid-centering commits

- USER (verbatim): "Another agent made a series of changes to this branch to fix a grid issue. Sync this branch to perlmutter and rerun run 10 to see what effect it has".
- Source: `arts-reinstate` at `9cac42c6fb19` (three commits above `b77752a8`: `32bea05f`, `322b623e`, `9cac42c6` "Centre the oversampled model grid on the detector pixels"), same uncommitted working-tree changes as update-10, session fingerprint `e81cc51c8f0a`. Deck identical to `s94475-update-10` apart from the run name; `points_per_pixel: 1`; root-aware quadrature on.
- Slurm job `59496545`, `shared_interactive`, one A100, 1 h. State: LAUNCHED.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-07 — RESULT: `s94475-update-11` and control `s94475-update-12`; effect of the grid-centering commits on the 1D root-aware fit

- `s94475-update-11` (run `aa290a01e8c741ecbeb50039b9d6a764`) and `s94475-update-12` (run `458e1f9332684f1d980eb866e82f3168`): both exit code 0 on Slurm job `59496545`, same source (`9cac42c6fb19` plus the uncommitted working tree) and the deck of `s94475-update-10` apart from the run name. Update-12 is a repeat of update-11 with no change, launched to measure run-to-run variation.
- Fixed-parameter check: the update-10 fitted weights evaluated with the `9cac42c6` code on local CPU give objective `455.0626501915755`; update-10 logged `455.0626501915471` from the earlier code on the GPU.
- Loss histories (`epoch loss`): step 0 is `2739.025166917294` in all three runs. Update-10 and update-11 are identical at step 1 and first differ at step 2 (relative `3.6e-07`); update-11 and update-12 first differ at step 1 (relative `2.6e-09`).

  | pair | max relative loss difference (step) | final relative loss difference | Te | ne | lam | fe where fe > 1e-6 |
  |---|---|---|---|---|---|---|
  | update-10 vs update-11 (different code) | 1.0e-02 (35) | 4.5e-04 | 2.3e-05 | 2.4e-05 | 4.0e-06 | 1.2e-02 |
  | update-11 vs update-12 (same code) | 6.9e-03 (31) | 1.1e-03 | 3.2e-03 | 8.4e-05 | 2.2e-07 | 5.3e-02 |

- Final objectives: update-10 `455.0627`, update-11 `455.2690`, update-12 `454.7831`.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-13 — RESULT: ARTS 1D work moved onto current main; fixed-parameter regression

- Working state committed on `arts-reinstate` as `ac31a4ca`, then replayed onto `origin/main` at `93e00b0c` (PRs #158, #159, #161, #162) as branch `arts-reinstate-on-main`: `879a1c38` (replay) and `728de212` (axis order and tests). One conflict, in `generate_spectra.py`, resolved by using the shared `electron_spectrum_is_detector_binned` helper from #161 and extending it to 1D distribution functions and `enabled: "auto"`.
- `prepare_data` now leaves `other.CCDsize` as [wavelength bins, angle bins] after angular reduction (it had been set to the reduced data shape, [angle, wavelength]); the six angular tests build dummy data as `(CCDsize[1], CCDsize[0])`.
- Fixed-parameter regression, local CPU, `728de212` working tree: the `s94475-update-10` fitted weights with the update-10 inputs (gain key renamed to `LPI_gain.include_LPI_gains`) and the current `configs/arts-1d/defaults.yaml` give objective `455.0626501915755` and data term `352.0409673719676`; update-10 logged `455.0626501915471` and `352.0409673719395`. Self-check against the fit forward model `2.9e-13`; no scattering angle flagged.
- Tests on local CPU: `tests/test_forward/test_irf_area.py`, `test_angular_1v.py`, `test_angular_2v.py`, `test_resonance_notch_filter.py`, `test_1d.py`, `tests/test_inverse/test_validate_inputs.py`, `test_angular_1v.py`, `test_angular_2v.py`: `50 passed, 4 skipped` before the new tests; `test_irf_area.py` and `test_validate_inputs.py` with the new cases: `57 passed`.
- Not done: a GPU fit on the new branch. The NERSC certificate expired on 2026-10-08.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-14 — LAUNCHED: `s94475-update-13`, run 10 deck on `arts-reinstate-on-main`

- USER (verbatim): "rerun the GPU run 10 deck and when it completes compare to run 10 to check for any regression or behaviour change".
- Source: `arts-reinstate-on-main` at `0b3e91853176` (no tracked changes), session fingerprint `69ef03ebc968`. Deck: `s94475-update-10` inputs with `include_gains: false` replaced by `LPI_gain: {include_LPI_gains: false}` and the run name changed; current `configs/arts-1d/defaults.yaml` (detector gain key renamed by #162). Stored at `C:\tmp\run_u13`.
- Slurm job `59777060`, `shared_interactive`, one A100, 1 h. State: LAUNCHED.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-14 — RESULT: `s94475-update-13` and repeat `s94475-update-14` on `arts-reinstate-on-main`

- Both exit code 0 on Slurm job `59777060`; same source (`0b3e91853176`) and deck (`C:\tmp\run_u13`, run name changed for update-14). Runs: update-13 `5a3365206022450dad89caa4f960e070`, update-14 `59ba7f9addbb4b409773fcc23cd0dd57`.
- Final objective (data term): update-10 `455.063` (`352.041`), update-11 `455.269` (`352.354`), update-12 `454.783` (`351.899`), update-13 `459.533` (`356.033`), update-14 `454.734` (`351.804`). Fit time (s): `514.8`, `529.7`, `530.1`, `600.0`, `519.1`.
- `epoch loss` at steps 0 and 1 is identical in update-10, -11, -13 and -14 (`2739.025166917294`, `2658.272188261109`); update-12 differs at step 1 by `2.6e-09`. At step 100: `892.2`, `894.4`, `892.9`, `910.3`, `894.6`; at step 500: `501.7`, `502.3`, `501.2`, `512.9`, `501.2`.
- Update-13 logged config differs from update-10 only in the renamed gain keys. Its saved data image and axes are identical to update-10. Loss and gradient at the update-10 fitted weights, evaluated on local CPU with the `ac31a4ca` code (separate worktree) and with the `arts-reinstate-on-main` code: loss `455.062650191577120` in both, all 261 gradient components identical.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-14 — RESULT: third new-branch repeat `s94475-update-15`

- Run `3a2a4afb2af147c3ae5d43d3e1242031`, Slurm job `59778542`, source `939e21bbe0f2` (code identical to `0b3e9185`), deck as update-13 with the run name changed; exit code 0; fit time `518.5` s.
- Final objective (data term) `455.525` (`352.475`); `epoch loss` at step 100 `893.5`, step 300 `625.6`, step 500 `502.9`; steps 0 and 1 identical to update-10.
- Final objective across repeats: old code (update-10, -11, -12) mean `455.038`, sample standard deviation `0.244`; new code (update-13, -14, -15) mean `456.597`, standard deviation `2.573`, median `455.525`.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-14 — SUMMARY: regression check of `arts-reinstate-on-main` against the old branch

- USER (verbatim): "No need to run more just document our current findings in your research notes."
- Question: does moving the ARTS 1D work onto main (`arts-reinstate-on-main`, code at `0b3e9185`) change the fit of the run 10 deck (shot 94475, lineouts 20–1000, `nvx: 256`, `lam` active, roll-off corners masked, root-aware quadrature, ±8 vTe grid) relative to the old branch (`ac31a4ca` and its predecessors)?
- Deterministic comparison (local CPU): at the `s94475-update-10` fitted weights both codes give loss `455.062650191577120` and identical values for all 261 gradient components; the prepared data image, background, and axes are identical; the logged configs differ only in the gain keys renamed by #162.
- GPU repeats, all on one A100 under `shared_interactive`, identical `epoch loss` at steps 0 and 1 in every run except update-12 (`2.6e-09` at step 1):

  | run | code | final objective | loss at step 100 | loss at step 500 | fit time (s) | Te (keV) | Te × ⟨v²⟩ (keV) | ne (1e20 cm⁻³) | lam (nm) |
  |---|---|---|---|---|---|---|---|---|---|
  | update-10 | old | 455.063 | 892.2 | 501.7 | 514.8 | 1.3836 | 1.1188 | 0.42815 | 525.2322 |
  | update-11 | old | 455.269 | 894.4 | 502.3 | 529.7 | 1.3835 | 1.1189 | 0.42814 | 525.2343 |
  | update-12 | old | 454.783 | 892.9 | 501.2 | 530.1 | 1.3879 | 1.1192 | 0.42818 | 525.2341 |
  | update-13 | new | 459.533 | 910.3 | 512.9 | 600.0 | 1.3489 | 1.1168 | 0.42796 | 525.2382 |
  | update-14 | new | 454.734 | 894.6 | 501.2 | 519.1 | 1.3876 | 1.1189 | 0.42822 | 525.2325 |
  | update-15 | new | 455.525 | 893.5 | 502.9 | 518.5 | 1.3799 | 1.1190 | 0.42808 | 525.2318 |

- Observations: the runs separate from one another at step 1 or 2 by `1e-9`–`4e-7` relative, with identical code included (update-10/-11/-12, update-13/-14/-15). Final objectives: old code mean `455.038` (sample s.d. `0.244`); new code mean `456.597` (s.d. `2.573`), or `455.130` without update-13. Update-13 is the only run whose loss stays above the others from step 100 onward; it also has the lowest fitted Te and the highest second moment, with Te × ⟨v²⟩ within `0.003` keV of the other five runs.
- Status: no further runs requested. The data do not show a difference between the two codes apart from update-13; with three GPU runs per code, a GPU-only effect smaller than the run-to-run spread, or one that occurs in a minority of runs, is not excluded.
## 2026-10-16 — RESULT: provenance of the synthetic ARTS datasets `ATS-s10000`–`ATS-s10007` and `ATS-s10011`

- USER (verbatim): "Inspect those files and determine if they carry data or metadata the tells what they are and what the plasma conditions to generate them are".
- File contents (pyhdf): `ATS-s10000`–`10007` (added in `aa38dd4c`, PR #90, 2025-04-29) each hold `Streak_array` (2, 1024, 1024) float64, frame 1 all zeros, and one file attribute, `title`: `m2_1D_test_dataset`, `m4_1D_test_dataset`, `masym15_tasym15_2D_test_dataset`, then `masym15_tasym15_rot15/30/45/60/90_2D_test_dataset`. No plasma conditions are stored. `ATS-s10011` (untracked) has `title: Maxwellian_test_dataset`, `data_source: s3://public-ergodic-continuum/298/dd544aedf4f945c099e7facc49ccd710`, an `input_deck` attribute holding a full deck, and an `inputs_yaml` dataset whose characters do not decode to that deck.
- Matching MLflow runs (experiment 298 `inverse-thomson-scattering`), all forward runs with artifacts `binary/fit_and_data.nc` (1024 x 1024), `plots`, decks: `angular_1D_maxwellian_testdata` `776c3339549c4436b754b38d36621575` (2024-08-08 20:05), `angular_1D_m4_testdata` `661f7e76ab2e447f9aa5648ddeb47774` (2024-08-08 20:05), `angular_2D_bidlm_masym15_tasym15_testdata_redo` `9f41d2c67da2430dbd30209df68758e5` and `_redo2` `d7f35605f2784454818d24149dded2b6` (identical images), `_rot15` `a6dce9a54a7e4248b86d4ca5fb922a6b`, `_rot30` `9fd79a4a866a4c9cb352242f1c13bc15`, `_rot45` `a9dfe4f1bd2144c5968d06fd7f1c523a`, `_rot60` `5245cc3d624a4c4e85aa8809bd9953db`, `_rot90` `23108affd5094d44854312a2be6aa973` (2024-09-24, commit `591e5b31`); `Maxwellian_testdata_dlm` `dd544aedf4f945c099e7facc49ccd710` (2025-07-16, commit `feaa5001`) for `10011`.
- Match test: each file frame 0 against the run `fit` array under eight transpose/flip orientations with a fitted scale. Best for every file is `fliplr(fit.T)`, correlation 0.978–0.980, scale 91.8–92.3, rms residual 16–17%% of rms signal; other same-name runs (earlier the same day, or `bgscaleE: 1.0`) correlate at 0.43–0.91. Residual variance is 0.76–1.37 times the scaled model level across eight intensity bins; residual mean drifts from −0.4 to −4.9 to +3.4 with level; pixels where the model is zero are zero in the file.
- Generating parameters logged by those runs (old deck format `species1`/`species2`): `Te = 1.0` keV, `ne = 0.4` (1e20 cm⁻³), `lam = 526.5` nm, `ud = 5.0`, `Va = 0`, `amp1 = amp2 = 1`, ion `A = 1`, `Z = 1`, `Ti = 0.03` keV, `fract = 1`; IRF `spect_FWHM_ele = 0.9` nm, `ang_FWHM_ele = 1.0` deg; `CCDsize [1024, 1024]`, `ang_res_unit 10`, `lam_res_unit 5`, `points_per_pixel 1`; `data.shotnum 94475` (calibration source); `bgscaleE 0`. 1D runs: `fe.dim 1`, DLM, `m = 2.0` (`10000`) or `4.0` (`10001`), `v_res 0.01`. 2D runs: `fe.dim 2`, DLM, `m = 2.0`, `m_asym = 1.5`, `temp_asym = 1.5`, `m_theta = 0, 15, 30, 45, 60, 90` deg for `10002`–`10007`, `v_res 0.02`. `10011` run: `fe.dim 1`, DLM `m = 2.0`, `nvx 320`, `Te = 1.0`, `ne = 0.4`, `lam = 526.5`, ion `A = 1`, `Z = 1`, `Ti = 0.03`.
- Not resolved: the processing between the run image and the file (scale ~92, noise, orientation) is not recorded in the files or the runs; the notebook named in the Perlmutter checkout (`create_synthetic_arts_data.ipynb`) could not be read because the NERSC certificate had expired.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-16 — RESULT: Phase 1 drift check of the synthetic ARTS images

- USER (verbatim): "Regenerate a hydrogen case for the fits but first try to reporduce the Kr version for the drift check" and "Dont move from phase 1 to phase 2 without confirmations from me".
- Data provenance found this session: the Perlmutter `ATS-s10011`/`ATS-s10012` were regenerated on 2025-10-28 from krypton runs `548792b2` (m=2) and `c3fbd2e3` (m=4) (ions Kr A=84 Z=18 fract 0.1, H fract 0.85, A=40 Z=16 fract 0.05; Ti 0.1/0.03/0.1); they store `EDF` and `Vx` datasets. The 2025-10-27 Maxwellian fits match the local hydrogen `ATS-s10011` (`dd544aed`): ratio roughness 0.034 against 0.065 for the krypton file and 0.045 for two independent noise draws. The SG4 fits match the Perlmutter krypton `ATS-s10012` at 0.0625, the independent-draw level; the hydrogen SG4 file they used (`f80ab7af`) is no longer present. Copies of the Perlmutter files are in `C:\tmp\synth_pm`.
- Generation recipe (`create_synthetic_arts_data.ipynb`, Perlmutter checkout): `Poisson(100 * fit)`, `np.rot90(., 3)`, multiplied by `spectral_sensitivity.mat` throughput on the axis `0.214116 * px + 449.5272`; frame 1 is the run `data` array (zeros); attributes `title`, `data_source`, `input_deck`; datasets `EDF`, `Vx` from `csv/learned_dist.csv`. The saved `fit` of all four generating runs has every angle row normalized to a peak of 1.
- Calibration entry added for shots 10000–10099 (0.214116 nm/px, 449.5272 nm), commit `8ae020c6`. Forward decks for the four generating runs, translated to the current format (`PhysParams` to `detector_specs`, `general.Va` to per-species `Va`, `extraoptions.fit_*` to `data`), in `configs/arts-2d-recovery/truth_{M_H,M_Kr,SG4_H,SG4_Kr}`. Forward runs on local CPU (`MLFLOW_TRACKING_URI=file:///C:/tmp/mlr`), each with `resonance_quadrature.enabled` false (legacy grid) and `auto` (root-aware); 20 s to 2 min each.
- Wavelength axis: today's `fit_and_data.nc` axis runs 449.7400–668.7800 nm; the 2025 axis 449.8363–668.6837 nm (difference −0.096 nm at the blue end, 0 near 559 nm, +0.096 nm at the red end). Angle axes identical.
- Comparison after normalizing each new row to its peak (rms and max of the difference; correlation) in rows 150–950 (43.2–121.9 deg, the old fits' window) and rms in rows 0–149:

  | truth | variant | rows 150–950 rms | max | corr | rows 0–149 rms |
  |---|---|---|---|---|---|
  | M_H | legacy | 0.0045 | 0.106 | 0.999836 | 0.0094 |
  | M_H | root-aware | 0.0122 | 0.482 | 0.998783 | 0.0181 |
  | M_Kr | legacy | 0.0011 | 0.013 | 0.999991 | 0.0053 |
  | M_Kr | root-aware | 0.0089 | 0.386 | 0.999362 | 0.0102 |
  | SG4_H | legacy | 0.0065 | 0.180 | 0.999653 | 0.0758 |
  | SG4_H | root-aware | 0.0161 | 0.649 | 0.997982 | 0.1350 |
  | SG4_Kr | legacy | 0.0029 | 0.441 | 0.999931 | 0.1182 |
  | SG4_Kr | root-aware | 0.0096 | 0.399 | 0.999232 | 0.1069 |

- Against the `.hdf` frame 0 in the same window (recipe applied to each image, correlation): M_H old 0.98402, legacy 0.98386, root-aware 0.98280; M_Kr old 0.98400, legacy 0.98399, root-aware 0.98334; SG4_Kr old 0.98396, legacy 0.98391, root-aware 0.98318.
- Lineouts at 32.4, 37.4, 58.7 and 95.2 deg: `C:\tmp\drift\drift_lineouts.png`; numbers `C:\tmp\drift\drift_report.json`. At 32.4 deg the SG4 images differ in kind between the 2025 run, the legacy grid and the root-aware quadrature (peak heights and the continuum between 480 and 570 nm).
- State: Phase 1 complete; waiting for user confirmation before Phase 2.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-16 — Phase 2 inputs: un-normalized hydrogen synthetic images and ported fit decks

- USER (verbatim): "Go ahead to phase 2 the cahnges in the spectra all look like the expected results of the newer better physics methods. Recreate the data files without the amplitude normalization so that we can do phase 3 self-consistently, the distribtuion functions can still be compared as the first round of fits will independently normalize the angles" and "When you gen the new data make sure the root aware is on".
- Truth forward runs, MLflow experiment `arts2d-recovery` (189144), commit `9172de17`, `resonance_quadrature.enabled: auto`: `truth_M_H` `ea8cfc02bbc0439fb5ad7edfad458c18` (DLM m=2) and `truth_SG4_H` `785f5b468fe243a4a4c07f85af250468` (DLM m=4); Te 1.0, ne 0.4, H, `nvx` 320. Their `fit` arrays equal the local root-aware forward images exactly and differ from the legacy-grid images by up to 0.23 (M) and 0.16 (SG4) of the image maximum. Runs `7a24277a` and `08f7db28` of the same names were set FAILED (they stopped at a console-encoding error before computing).
- New data files (untracked, `tsadar/external/data`): `ATS-s10021.hdf` (Maxwellian) and `ATS-s10022.hdf` (SG4). Frame 0 = `rot90(Poisson(k * fit), 3) * spectral_sensitivity` with one scale `k` for the image (912.7 and 1094) chosen so the median row peak in rows 150–950 is 100 counts; numpy `default_rng` seeds 21 and 22; frame 1 = zeros. Row-peak counts in rows 150–950: 94.8–229.4 (M), 91.9–181.9 (SG4). Attributes `title`, `data_source` (S3 path of the truth run), `input_deck`, `generation`; datasets `EDF`, `Vx` from the truth run.
- Fit decks: `configs/arts-2d-recovery/{Maxwellian,SG4}_{dlm,arb1D,arb2D,arbSH,nnSH,mySH}`, translated from the 12 `tsadar-tests-2` runs (inputs as logged, with `PhysParams`→`detector_specs`, per-species `Va`, `extraoptions.fit_*`→`data`; defaults = current `configs/arts-1d/defaults.yaml` with `bgscaleE: 0`, the only differing old default). Data changed to `[10021, 10021]` / `[10022, 10022]` with `shot_rot: 90`. All 12 pass `_validate_inputs_` and build `ThomsonParams`; root-aware quadrature is on for all (1D resolved by the input check, 2D by default).
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-21 — RESULT: the fit model at the truth reproduces the synthetic ARTS images; the mismatch was the inverse activation

- USER (verbatim): "I am concerned that you do not reporduce the test data and would like to dig into that first."
- Data-file fix made earlier in this phase: the first `ATS-s10021`/`10022` applied the throughput along the angle axis after `rot90` (a 0.90→1.47 tilt across wavelength, truth data term 67). They were regenerated with `counts * speccal` (speccal shape (1, 1024), along wavelength) before `rot90`, as the notebook does; prepared data / truth ≈ 0.99–1.0 in every wavelength band.
- 2D decks carry `resonance_quadrature.map_batch_size: 1` (commit `5b6afa6e`); a 2D fit step still needed about 59 GB on one A100 (unresolved).
- Symptom: the fit-mode model "at truth" (Te 1, ne 0.4, DLM m 2, shot 10021) differed from the forward truth image binned 5 (wavelength) x 10 (angle) by 4.5% rms, 12.7% max, 15% rms in the lowest-angle row (43.7 deg), with the peak 7% high and weight moved to the outside of both EPW peaks. Ruled out: data files (chi2 0.85 per pixel against the scaled truth with the true Poisson variance), variance model (objective / true variance 0.94), profiled gains (894 against a true scale ~900), nvx 128/320, Ti 0.1/0.03, angle and wavelength axes, IRF widths (calibration gives 0.9 nm / 1.0 deg, the truth deck has the same), scattering angles and weights (both from `get_scattering_angles`).
- Cause: the diagnostic built `ThomsonParams(..., activate=True)`. For active parameters the stored value is `log(1e-2 + x / (1 - x + 1e-2))` of the normalized value (`ts_params.get_act_and_inv_act`, and the same expression in `DLM1V`), which `sigmoid` does not invert exactly. Deck values decode to Te 0.9953, ne 0.4093, m 2.0297 (Maxwellian; m 4 → 3.9837). Forward mode uses `activate=False` (identity), so the truth images are at the exact values.
- At the exact values (`activate=False`), fit model vs truth image, rows 15–94 of the binned image:

  | dataset | rms | max | row 0 rms | median row rms | data term per pixel |
  |---|---|---|---|---|---|
  | Maxwellian 10021, activated | 4.54e-2 | 1.27e-1 | 0.154 | 0.028 | 3.66 |
  | Maxwellian 10021, exact | 1.41e-3 | 3.52e-3 | 0.0021 | 0.0014 | 1.02 |
  | SG4 10022, activated | 5.70e-2 | 2.15e-1 | 0.246 | 0.030 | 5.68 |
  | SG4 10022, exact | 2.22e-3 | 5.48e-3 | 0.0044 | 0.0022 | 0.98 |

  The remaining ~0.2% is within the differences between the two set-ups (nvx 128 against 320, 205 five-pixel detector bins against 1024 one-pixel bins). The earlier GPU "truth" objectives (data term 7.3 M, 11.4 SG4, both shots) were evaluated with the activated values and overstate the truth misfit.
- Consequence for fits: every active parameter starts slightly away from its deck `val` (here +2.3% in ne, −0.5% in Te); this moves only the initial point. Not changed in the code (it touches every fit path). Diagnostics that evaluate a model at given values must use `activate=False`.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-21 — RESULT: the 2D-fe ARTS gradient's memory grows linearly with scattering angle; per-angle rematerialization bounds it

- USER (verbatim): "There have been a large number of commits and PR focused on correcting the memory usage of 2D EDFs additionaly you can see that the previous runs ran which means they did not run out of memory. Try running with the higher memory fraction and while that is running look through the repo history to see if you can identify the problem and a potential fix". The GPU run is pending: Perlmutter was down for maintenance.
- History: the 2025 fits (code at `8ce493b0`) evaluated the 2D susceptibility on the 1024-point wavelength grid at 241 angles, with the per-point EDF rotation wrapped in `jax.checkpoint`. Since #143 (`c5360f74`) 2D fe defaults to the root-aware detector-bin quadrature, `FitModel.detector_integrated_electron_spectrum` in `generate_spectra.py`: a `lax.map` over gradient x angle, each iteration a full `integrate_detector_bins` (4097 root-scan points, 256 panels x 8 regular nodes, 16 roots x 3 panels x 32 tan-mapped nodes, each node a sinogram gather and `ratintn` over `nvx`). Nothing in it is rematerialized, so reverse mode stores every iteration's residuals. `map_batch_size` already defaults to 1, which is why setting it in the decks changed nothing.
- Measurement: CPU, XLA buffer plan (`memory_analysis().temp_size_in_bytes`) of the jitted loss + gradient for `Maxwellian_arb2D` (both multiplexed shots), with the angle set subsampled to N angles (`sa` and the weight columns sliced). Script: scratchpad `mem_arts2d.py`.

  | N angles | as is | `jax.checkpoint` around the per-angle integration |
  |---|---|---|
  | 2 | 1.065 GB | 0.265 GB |
  | 4 | 1.867 GB | 0.265 GB |
  | 8 | 3.470 GB | 0.265 GB |
  | 32 | — | 0.265 GB |

  As is: 0.40 GB per angle, extrapolating to about 97 GB at 241 angles, above an 80 GB A100 at any `XLA_PYTHON_CLIENT_MEM_FRACTION`. With the checkpoint the plan is independent of N.
- Proposed fix (not applied, awaiting the user): `lax.map(jax.checkpoint(integrate_one), ...)` in `detector_integrated_electron_spectrum`. Values are unchanged; the backward pass recomputes each angle's forward integration once, so a step should cost up to roughly one extra forward evaluation. To be timed on the GPU.
- Shared-vault mirror skipped: no vault resolved on this host.
- Gradient check of the proposed checkpoint (CPU, float64, `Maxwellian_arb2D` at its initial parameters, 4 subsampled angles, both shots; scratchpad `remat_grad_check.py`): loss identical bit for bit (2.312641206946842e+03); gradient over all 5 leaves including the 128 x 128 EDF (16384 entries) agrees to max |diff| / max |g| 2.3e-16 overall and 1.8e-15 on the EDF leaf, ||diff|| / ||g|| 4.4e-16, cosine 1 to 15 digits. Rematerialization recomputes the same forward from the same inputs; the only conceivable difference is a recomputed root-panel selection flipping under a non-bitwise-reproducible GPU reduction, which would change the result only at the quadrature-error level.
- USER (verbatim): "Make this change at this location so it impacts both 1D and 2D arts fits. And add to the plan a set of test moving the checkpoint location further into formfactor or ratintn so that we can compare the speed and memory requirements of the different implementations". Applied: `lax.map(checkpoint(integrate_one), ...)` in `FitModel.detector_integrated_electron_spectrum`, for every root-aware ARTS fit (1D and 2D fe). Targeted tests (`tests/test_forward/`: `test_unresolved_arts2d`, `test_resonance_notch_filter`, `test_angular_1v`, `test_angular_2v`, `test_irf_area`; `tests/test_inverse/test_angular_objective`; `tests/test_physics/test_arts_recovery`; the inverse `test_angular_1v`/`2v` fits were not run): 58 passed, 1 skipped, 284 s on CPU. The placement comparison (geometry / quadrature nodes / form factor / `calc_chi_vals` / `ratintn`, and checkpoint policies) is Phase 2b of the plan; GPU memory and step time still to be measured when Perlmutter returns.

## 2026-10-22 — RESULT: GPU memory and step time of the recovery decks with the per-angle checkpoint; map_batch_size 64 for 2D

- USER (verbatim): "Perlmutter should now be available. Resume the plan" and "I believe that the batch size effects the accuracy right?"
- Session `20261022T084315Z-963-15789`, Slurm job `60099517`, one A100 80 GB (`shared_interactive`), source `5b6afa6e` dirty (fingerprint `dd065df7e3de`: the `checkpoint(integrate_one)` change), `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Script `C:\tmp\step_check_arts2d.py`: 10 loss+gradient steps (`lf.vg_loss`, both multiplexed shots) at each deck's initial parameters, then 4 forward-only `lf._loss_`; peak = `memory_stats()["peak_bytes_in_use"]`.
- Six Maxwellian decks, deck settings (`map_batch_size` 1 for 2D, 8 for 1D):

  | deck | fe dim | peak | step 1 | step 2 | median steps 3–10 | forward | 1000 steps |
  |---|---|---|---|---|---|---|---|
  | arb2D | 2 | 1.28 GB | 157.0 s | 30.2 s | 30.18 s | 9.89 s | 505 min |
  | arbSH | 2 | 1.28 GB | 156.2 s | 30.2 s | 30.21 s | 9.84 s | 506 min |
  | nnSH | 2 | 1.28 GB | 155.2 s | 30.3 s | 30.30 s | 9.81 s | 507 min |
  | mySH | 2 | 1.28 GB | 154.3 s | 30.2 s | 30.36 s | 9.84 s | 508 min |
  | dlm | 1 | ≤1.28 GB | 86.1 s | 1.0 s | 0.96 s | 0.27 s | 17.5 min |
  | arb1D | 1 | ≤1.28 GB | 80.2 s | 1.0 s | 0.97 s | 0.27 s | 17.5 min |

  (The 1D peaks are the process peak, already 1.28 GB from the 2D decks run first.) The checkpoint removes the out-of-memory failure; a 2D gradient step costs about 3x a forward evaluation.
- `map_batch_size` sweep, `Maxwellian_arb2D`, one process per value:

  | map_batch_size | peak | step 1 | median steps 3–10 | forward | 1000 steps |
  |---|---|---|---|---|---|
  | 1 | 1.28 GB | 157.0 s | 30.18 s | 9.89 s | 505 min |
  | 8 | 1.52 GB | 250.5 s | 5.78 s | 1.54 s | 101 min |
  | 32 | 6.48 GB | 255.8 s | 3.15 s | 0.58 s | 57 min |
  | 64 | 18.26 GB | 264.1 s | 2.32 s | 0.47 s | 43 min |
  | 128 | 64.15 GB | 311.8 s | 2.65 s | 0.60 s | 49 min |
  | 241 | out of memory (88 GB allocation) | | | | |

  Loss at the initial point 4.119704e+02 for every value. Set `map_batch_size: 64` in the eight 2D recovery decks (`configs/arts-2d-recovery/{Maxwellian,SG4}_{arb2D,arbSH,nnSH,mySH}/defaults.yaml`); the 1D decks keep 8.
- Accuracy: `map_batch_size` only sets how many independent gradient x angle integrations `lax.map` vectorizes per iteration; each keeps its own root search and quadrature nodes. CPU float64 check (scratchpad `mbs_grad_check.py`, `Maxwellian_arb2D`, 8 angles, both shots): loss 1.595985967385055e+03 at 1 and 4, ...054e+03 at 8; gradient max |diff| / max |g| 2.6e-16 (4 vs 1) and 1.9e-16 (8 vs 1). The accuracy controls are `integration_panels`, `root_scan_panels`, `regular_order`, `root_order`, `max_roots`, unchanged.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-22 — LAUNCH: Phase 2 baseline fits of the synthetic ARTS images (12 decks)

- Session `20261022T100730Z-1938-9542`, Slurm job `60100768` (`--kind gpu`, one 4 x A100 80 GB node, 3 h), source `5b6afa6e` dirty (fingerprint `f51bcdf7e8af`: per-angle `checkpoint(integrate_one)`, `map_batch_size: 64` in the eight 2D decks). Command per deck: `python -u run_tsadar.py --cfg configs/arts-2d-recovery/<deck> --mode fit`, through `C:\tmp\run_queue_arts2d.sh` (one queue per GPU, `CUDA_VISIBLE_DEVICES` = queue index; logs `workdir/fits/<deck>.log`, `workdir/fits/queue.log`). MLflow experiment `arts2d-recovery` (189144). A first launch at 03:10 PDT failed for every deck with `python: command not found` (venv not activated in the queue script) before any run was created; those logs are in `workdir/fits_failed_launch/`.
- Queues: GPU 0 `Maxwellian_arb2D`, `Maxwellian_arbSH`, `Maxwellian_dlm`; GPU 1 `Maxwellian_nnSH`, `Maxwellian_mySH`, `Maxwellian_arb1D`; GPU 2 `SG4_arb2D`, `SG4_arbSH`, `SG4_dlm`; GPU 3 `SG4_nnSH`, `SG4_mySH`, `SG4_arb1D`. Expected about 45–50 min per 2D fit and 20 min per 1D fit.
- First runs (03:11 PDT): `Maxwellian_arb2D_baseline` `1af2600f5b424719b64d7ee0b4d89dc3`, `Maxwellian_nnSH_baseline` `e6385ff874e94529a6b437596ecd07fd`, `SG4_arb2D_baseline` `e45d563a379c4cb7a7c3c289ebf8ee7b`, `SG4_nnSH_baseline` `aa16dcb19503479fb857ad20ec479aef`.
- Shared-vault mirror skipped: no vault resolved on this host.
- Gain-prior check (USER, verbatim: "Add two more runs to the queue that try those two other options for the prior. Only do it for one model as it's just a check to see if the prior is effecting the parameters. This prior will be heavily effected by phase 3 so these should just be quick tests"). The default per-row gain prior (`prior_mean` 1, `prior_strength` 0.01) is 93% of the objective on these images (27.6 of 29.75 for `Maxwellian_arb2D`, both shots) because the data/model scale is ~900, and it rewards a brighter model (amp1/amp2 drifted to 1.00–1.04 in all four first fits). Two decks identical to `Maxwellian_dlm` apart from the run name and one gain option: `Maxwellian_dlm_prior0` (`optimizer.angular_objective.gain.prior_strength: 0.0`) and `Maxwellian_dlm_priormean900` (`prior_mean: 900.0`, the known scale: generation k 912.7 after the throughput correction; profiled gains at truth ~894). Queued at 04:39 PDT by `C:\tmp\run_after_arts2d.sh`: `prior0` on GPU 0 after `Maxwellian_dlm` ends, `priormean900` on GPU 2 after `SG4_dlm` ends; same session and source.

## 2026-10-22 — RESULT: Phase 2 baseline fits (current per-row gains) and the gain-prior check

- All 12 baseline fits ended with exit 0 (03:11–05:32 PDT, job `60100768`); artifacts in `C:\tmp\newfits\<M|S>_<model>`, comparison `compare_arts2d_recovery.py --fits old=C:/tmp/oldfits new=C:/tmp/newfits --truth-root C:/tmp/newdata --out C:/tmp/compare_phase2` (`summary.csv`, `summary.md`, `fe_*.png`). Wall time per fit: 2D 44–60 min (arbSH ~60), 1D ~20 min.
- New fits (truth Te 1, ne 0.4; data term summed over the two shots, truth level ≈ 2.05; Te_eff = Te <v^2>; L2 = log10 fe for |v|<3 after standardizing to <v^2> = 1):

  | dataset | model | Te err | ne err | Te_eff err | L2 new (old) | ratio 3–4 | data | gain prior |
  |---|---|---|---|---|---|---|---|---|
  | M | dlm | +0.5% | −0.2% | +0.5% | 0.0085 (0.0085) | 0.86 | 3.06 | 28.0 |
  | M | arb1D | +19.3% | −0.0% | +0.1% | 0.0009 (0.0043) | 1.00 | 2.09 | 27.9 |
  | M | arb2D | +15.8% | −1.3% | +2.5% | 0.060 (0.086) | 0.53 | 2.13 | 27.6 |
  | M | arbSH | −1.7% | −0.3% | −1.7% | 0.0003 (0.012) | 1.00 | 29.8 | 27.6 |
  | M | nnSH | −2.3% | −0.0% | −2.3% | 0.0043 (0.036) | 0.99 | 4.42 | 27.9 |
  | M | mySH | −1.6% | −0.2% | −1.4% | 0.0062 (0.012) | 1.00 | 28.7 | 27.5 |
  | S | dlm | +0.0% | −0.0% | +0.0% | 0.0007 (0.028) | 1.03 | 1.98 | 25.7 |
  | S | arb1D | −16.5% | −0.1% | +0.1% | 0.0014 (0.042) | 1.24 | 2.08 | 25.6 |
  | S | arb2D | −6.2% | +0.3% | −4.3% | 0.054 (0.104) | 12.1 | 2.07 | 24.8 |
  | S | arbSH | −1.4% | −1.3% | −1.4% | 0.013 (0.011) | 0.71 | 70.6 | 24.8 |
  | S | nnSH | −3.6% | +0.6% | −3.4% | 0.0098 (0.016) | 1.09 | 7.07 | 26.2 |
  | S | mySH | −0.8% | −0.1% | −0.6% | 0.070 (0.019) | 1.05 | 56.4 | 24.8 |

- Arbitrary 1D fe: Te and the EDF width are degenerate (Te off by +19% / −16.5% while Te_eff is within 0.1%); only Te_eff is constrained by the spectrum.
- SH models (arbSH, mySH; nnSH partly) did not train: loss flat from step ~99 to 1000 (Maxwellian_arbSH 57.4, SG4_arbSH 95.4, Maxwellian_mySH 56.2) with data terms 14–35x the truth level. CPU gradient at the arbSH initial point (`sh_grad_check.py`, 4 angles): the l=1 coefficients start at exactly zero and `flm_mag` has an exactly zero gradient there (stationary start); Te, ne, m, amp1/2 have finite gradients. Final Maxwellian arbSH m = 2.000 sits on its lower bound. Open question: at the true parameters (activate=False, 4 subsampled angles, `sh_vs_1d_truth.py`) the arbSH and mySH model images differ from the 1D DLM image by ~15% in amplitude and 19–22% rms in shape for both shots; whether that is an SH-specific representation problem or a 2D-path difference is not yet separated (arb2D in that test was at its init_m 3 shape, not a Maxwellian).
- Gain-prior check: `Maxwellian_dlm_priormean900` (`54d4ac8898d44ee188a0c28138fda79b`): Te 1.0043, ne 0.3991, m 2.085, amp1/amp2 1.029/1.030, data 2.79, gain prior 0.003, total 2.80; baseline `Maxwellian_dlm`: Te 1.0046, ne 0.3991, m 2.083, amp 1.040/1.041, data 3.06, prior 28.0. The prior mean does not move Te, ne or m; amp1/amp2 still drift (to 1.03), so the drift is not only the prior's pull. `Maxwellian_dlm_prior0` failed at start (`loss_function.py:338`: unanchored per-row gains with active amp1/amp2 are rejected); proposed fix (amp1/amp2 inactive in that deck) not applied, allocation ended first.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-22 — RESULT: source of the SH (and arb2D) recovery failures: off-center EDF rotation for the multiplexed shot; 2D grid error at nvx 128

- USER (verbatim): "Commit and push what we have now, then run the 2D checks because we need to find the source of the SH issues before running more 2D fits." Pushed `f66cd200` (checkpoint) and `968f7660` (Phase 2) to the new remote branch `origin/arts-reinstate-on-main`.
- The SH EDF representation is correct: at m = 2 (with `init_m` 2; SH starting EDFs are built from `init_m`, not `m`) arbSH and arb2D equal the analytic 2D Maxwellian to 7e-16, mySH to 2e-3, nnSH to 1.7e-2.
- Cause: `tsadar.utils.vector_tools.rotate` rotated about index `n/2`, but `v = 0` of the symmetric cell-centered grid is at `(n-1)/2`. Rotating an isotropic Maxwellian (128 points, +-8 vTe): 90 deg moved the centroid by (+0.137, -0.003) vTe (+1.09 cells), lost 0.7% of the area, max error 10%; 180 deg (+0.14, +0.15) vTe, 17%. The multiplexed second shot (`shot_rot: 90`, `ts_params.py:224`) was therefore modelled with a spurious drift of ~0.14 vTe. At the truth (exact Maxwellian, 4 angles, `sh_vs_1d_truth.py`), the rotated shot differed from the 1D model by 13.6% rms (19.4% max) for every 2D model, the unrotated shot by 1.15%. With a center-corrected rotation both shots agree at 1.15% (identical). The SH models (few free parameters: Te, ne, m, amp1/2; l = 1 terms start where their gradient is exactly zero) cannot absorb the wrong second shot and stall with large data terms; arb2D absorbs it by distorting the EDF (Te +16%, anisotropy).
- Also: indices were clipped to `shape` (out of bounds; JAX clamps silently, NumPy raises), and the integer truncation toward zero disagreed with the `% 1` weights for negative source coordinates.
- Fix (USER: "Go with 1 and remove the unused import"): `rotate` rewritten to rotate about `((n-1)/2)` with floor-based bilinear weights and zero outside the array; same signature, radians, counter-clockwise sense. Also used by `Arbitrary2V.init_bidlm` (rotated bi-DLM starting guess, `flm_type: bidlm`, no current deck), which had the same shift; the 2024 `rot15`–`rot90` synthetic images (`ATS-s10002`–`10007`, no stored EDF) were presumably made with the shifted rotation. Removed the unused `rotate` import from `loss_function.py`. New `tests/test_utils/test_vector_tools.py` (isotropic invariance at 30–270 deg, exact counter-clockwise quarter turn, round trip): 8 passed with the fix, 7 of 8 fail on the old function.
- Remaining 2D-vs-1D gap on the unrotated shot (exact Maxwellian, `conv_2d_vs_1d.py`, 4 angles): 1D nvx 256 vs 128: 3e-9 (converged; tabulated DLM projection). 2D nvx 128 vs 1D: 1.15% rms, 1.85% max, amplitude 1.0074, identical for `n_beta` default, 4096 and 0 (expected: an isotropic sinogram is constant in beta). 2D nvx 256: 0.27% rms, 0.45% max, amplitude 1.0017. Second-order convergence in the 2D velocity grid (dv 0.125 vTe at +-8 vTe / 128; the 2025 fits had dv 0.094 at +-6 / 128, about 0.65%). Decision needed before more 2D fits: nvx 256 for 2D, or a more accurate 2D projection/derivative on the 128 grid.
- Shared-vault mirror skipped: no vault resolved on this host.
- USER (verbatim): "For the 2D go with option 1, use more points. Do we need to regen any test data for rotate change?" Set `nvx: 256` in the eight 2D recovery decks (`configs/arts-2d-recovery/{Maxwellian,SG4}_{arb2D,arbSH,nnSH,mySH}/inputs.yaml`). The deck smoothing window does not act on 2D EDFs (`Arbitrary2V` does not smooth; SH smooths its radial coefficients with `nvr // 4` on the `nvr` grid, unchanged at 128).
- Test data and the rotate fix: nothing to regenerate. The 2024 generator (`591e5b31`, `dist_functional_forms.BiDLM`) used the same `n/2` center but on `jnp.arange(-8, 8, h)`, where v = 0 is exactly index `n/2`, so the `rot15`–`rot90` images were rotated correctly; the shift appeared when the EDF grids became symmetric cell-centered. `ATS-s10021`/`10022` and their truth runs are 1D (no rotation). The repository references (`ThryE-arts2v.npy`, inverse `test_angular_2v`, `test_arts2d_consistency`) use `tests/configs/arts2d_test_inputs.yaml` with a single `shotnum: 10007`, so `dist_rot` stays 0; all passed with the fix.
- Targeted tests with the fix (`tests/test_utils`, forward and inverse `test_angular_2v`, `test_spherical_harmonics`, `test_arts2d_marginal`, `test_sinogram`): 63 passed, 3 skipped, 6 failed. The same 6 fail without the fix (stash). On `origin/main` 2 of them fail (`test_anisotropic_marginal_matches_refined_quadrature_and_stays_smoothly_positive`, `test_reduced_edf_normalization_and_temperature_moment_do_not_drift[_harmonic_model-3.0]`); the other five on this branch (`test_l1_real_harmonics_are_independent_cartesian_modes[49]` 4.5e-13 vs atol 2e-13, `test_nonmaxwellian_3d_marginal_matches_adaptive_integration[3.0-4e-06]` and `[5.0-5e-08]`, `test_every_in_plane_projection_matches_direct_3d_integration` 9.1e-6 vs 3e-6, `..._do_not_drift[_harmonic_model-2.0]`, while the `-3.0` case that fails on main passes here) follow from this branch's +-8 vTe grid (`879a1c38`). Tolerances not changed.

## 2026-10-22 — LAUNCH: 2D recovery reruns with the rotation fix and nvx 256 (`_baseline_2`)

- USER (verbatim): "Once it is in go ahead with the reruns" and "Also if you commit first that would be better so the reruns have a hash that reflects their updates". Committed `46414653` (rotate fix, test, unused import) and `1f8da4cb` (nvx 256, run names `<dataset>_<model>_baseline_2`, notes); not pushed.
- nvx 256 GPU step check (session `20261022T140842Z-5039-8921`, job `60107029`, one A100): `Maxwellian_arb2D` map_batch_size 64: 35.44 GB, step 1 255.6 s, median steps 3–10 3.86 s, forward 0.79 s, ~69 min per 1000 steps, initial loss 385.75 (412.0 at nvx 128 before the rotation fix); map_batch_size 32: 12.32 GB, 4.45 s, ~78 min; `Maxwellian_arbSH` map_batch_size 64: 35.65 GB, 3.92 s, ~70 min, initial loss 332.83. Kept map_batch_size 64.
- Session `20261022T143549Z-5434-20420`, Slurm job `60108231` (`--kind gpu`, 4 x A100, 4 h), source `1f8da4cb` (untracked files only), queues via `C:\tmp\run_queue_arts2d.sh`: GPU 0 `Maxwellian_arb2D`, `Maxwellian_arbSH`; GPU 1 `Maxwellian_nnSH`, `Maxwellian_mySH`; GPU 2 `SG4_arb2D`, `SG4_arbSH`; GPU 3 `SG4_nnSH`, `SG4_mySH`. Started 07:38 PDT: `Maxwellian_arb2D_baseline_2` `ef458e086669475aa3810e9a517253d3`, `Maxwellian_nnSH_baseline_2` `0115abe75e1f4eb69bcefc06d910dbb2`, `SG4_arb2D_baseline_2` `715a8f25252b40e79cf7e217c668a9a9`, `SG4_nnSH_baseline_2` `72cdcfcf0f7a44f2b636842ded0990bf`. The four 1D baseline fits are not rerun (neither fix acts on 1D fe).
- Shared-vault mirror skipped: no vault resolved on this host.
- 07:38–07:49 PDT: all eight `_baseline_2` runs failed at the first gradient step with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 34.85GiB`: `run_tsadar.py` defaults `XLA_PYTHON_CLIENT_MEM_FRACTION` to 0.6 (48 GB) and the queue script did not override it, while the step check that measured 35.4 GB ran at 0.95; the fit loop needs more than the bare loss+gradient. USER chose (verbatim) "Go with 2": keep map_batch_size 64 and set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` in `C:\tmp\run_queue_arts2d.sh` (one fit per GPU). Runs renamed `<dataset>_<model>_baseline_3`.
- The `--kind gpu` node (job `60108231`) has 40 GB A100s (`nvidia-smi`: 40960 MiB), not the 80 GB A100s of `shared_interactive`; so the `_baseline_2` cap was 0.6 x 40 = 24 GB. `_baseline_3` (0.95 x 40 = 38 GB, map_batch_size 64, nvx 256), relaunched 08:15:45 PDT from `19d96cee`, passed the first-step allocation: at 08:35 PDT 42–48 steps done at 4.2–4.3 s/it (including compile). Runs: `Maxwellian_arb2D_baseline_3` `beab835d60664a7c9fd3c4e29d89e4cb`, `Maxwellian_nnSH_baseline_3` `461ed4e1f3cb4bb3b5799ee39c056d6d`, `SG4_arb2D_baseline_3` `437584637c7f496db6d92b2dabe8a8b8`, `SG4_nnSH_baseline_3` `7acb089acb1149a3a7480d00d6c13d0e`. The failed `_baseline_2` logs are in `workdir/fits_baseline2_oom/`.
- 08:58 PDT: `Maxwellian_arb2D_baseline_3` stopped with `NonFiniteOptimizationError: Nonfinite loss at optimizer stage 0, step 371` (loss 30.1 at step 370; the update and the candidate parameters were finite, so the 2D forward model returned a nonfinite loss at a finite parameter point). Nothing was recoverable: the periodic state snapshots are only written when the loop completes. The other three first-round fits were at steps 379–395 and finite. USER (verbatim): "Yes go ahead, just note that previous non-finite loss happening randomly in the middle of a run have all involved GPU math setting a small number to 0 or similar issues that result in divide by zero errors".
- Change: `angular_multiple_optax` (`tsadar/inverse/loops.py`) now logs `nonfinite_last_finite_weights.eqx`, `nonfinite_candidate_weights.eqx` (the tree the nonfinite quantity was evaluated on), `nonfinite_state_weights.txt` and `nonfinite_failure.json` before re-raising; successful fits are unchanged. New test `test_angular_nonfinite_mid_run_logs_reproduction_artifacts`; `tests/test_inverse/test_loops.py` 12 passed. The rerun is `Maxwellian_arb2D_baseline_4` (same deck otherwise), queued on GPU 0 after `Maxwellian_arbSH`; the queued second-round fits on the other GPUs pick up this commit.

## 2026-10-22 — DECISION: DLM m uses the deck bounds; DLM table extended to m = 1.8; m lb 1.8 in the fit decks

- USER (verbatim): "It makes sense to treat m the same as other parameters and actually use the input deck values, but be fore we go ahead i am concerned that if the EDF below m=2 is just the EDF at m=2 then the gradient in m would go to zero and instead of fixing things so it properly recovers m it will instead cause instability near m=2" and "Go ahead but only extend the table to 1.8 as values below 2.0 are not physical and just there to allow 2.0 to be reached naturaly. Also 200 steps is a bit much we should probably use a lb of 1.8 or 1.85 instead".
- Finding: `DLM1V` hard-coded m = 2 + 3 sigmoid(x) (range [2, 5]) and ignored the deck's m bounds (the `lb: 1.8` in the Maxwellian DLM decks had no effect); the SH models hard-coded the same range. A Maxwellian truth (m = 2) therefore sat at x -> -inf: Maxwellian_dlm baseline m 2.51 (step 100), 2.19 (400), 2.13 (600), 2.08 (1000). The DLM table (`DLM_x_-4_-10_10_m_-1_2_5.mat`, 200001 x 31, m = 2..5) only covers m >= 2; simply clamping below 2 would make the loss flat in m there (a trap with no restoring force).
- Table: the stored columns are the 1D projections of the isotropic 3D DLM exp(-(v/a)^m) with unit area and <vx^2> = 1; the closed form Gamma(2/m) Q(2/m, (|x|/a)^m) / (2 a Gamma(3/m)), a = sqrt(3 Gamma(3/m)/Gamma(5/m)), reproduces columns m = 2, 2.5, 3, 4, 5 to 1e-8–1e-10 (`dlm_projection_1d`). `DLM1V` now prepends computed columns m = 1.8 and 1.9 (no new data file); the stored m >= 2 columns and m nodes are kept exactly, so fe for m >= 2 is bit-for-bit unchanged.
- Mapping: `DLM1V` m_shift = lb, m_scale = ub - lb from `fe.params.m` (default [2, 5]); ValueError unless 1.8 <= lb < ub <= 5. SH models likewise (clip activation, `init_m` must lie in [lb, ub]). With lb 1.8, m = 2 sits at sigmoid 0.0625 (logit -2.7, gradient factor 0.19 of the scale) instead of -inf.
- Decks: `fe.params.m.lb: 1.8` in `configs/arts-1d/inputs.yaml` and the 14 recovery fit decks (truth decks unchanged: forward runs use m as given). `configs/1d` (1D temporal) keeps its own `lb: 1.9` (defaults 1.8), which is now honoured. Tests: new `tests/test_physics/test_dlm_m_range.py` (10 passed).
- Not yet synced to the running session: the `_baseline_3` second round and the `Maxwellian_arb2D_baseline_4` reproduction run on the earlier code.
- Targeted tests with the change (`tests/test_spherical_harmonics.py`, `tests/test_arts2d_marginal.py`, `tests/test_forward`, `tests/test_physics`, `tests/test_inverse/test_loops.py`, `tests/test_utils`): 151 passed, 1 skipped, 6 failed — the same 6 pre-existing tight-tolerance failures recorded above.

## 2026-10-22 — INCIDENT: a mid-run source sync made three finished 2D fits fail at the end; fix

- 09:42–09:45 PDT: `SG4_arb2D`, `Maxwellian_nnSH` and `SG4_nnSH` `_baseline_3` completed 1000/1000 steps (final losses 26.5, 29.9, 27.9) and then failed with `OSError: [Errno 116] Stale file handle: 'state_weights.txt'` before saving any results. Cause: the 09:05 `session.sh sync` (to deliver the nonfinite-capture change) runs, on Windows, `rm -rf src && mkdir -p src && tar -xzf -` on the login node; every process started earlier on the compute node kept a working directory pointing at the deleted `src/` (Lustre: reads still worked, creating a file returned ESTALE), and the fit loop wrote `state_weights.txt` into its working directory before saving its artifacts. Fits still running from those queues (`Maxwellian_arbSH`, `SG4_arbSH`, `Maxwellian_mySH`, `SG4_mySH`) were doomed the same way.
- USER (verbatim): "Implement the proposed plan to fix and prevent this kind of crash".
- Fix: (1) `angular_multiple_optax` writes `state_weights.txt` in a private temporary directory before logging it (also removes a latent race: concurrent fits in a shared working directory overwrote each other's file); new test `test_angular_state_snapshots_do_not_touch_the_working_directory`, `tests/test_inverse/test_loops.py` 13 passed. (2) `C:\tmp\run_queue_arts2d.sh` runs every fit in its own directory `workdir/fits/<deck>/` with absolute paths to `run_tsadar.py` and the deck, so a sync never replaces a running fit's working directory. (3) Rule: never sync a session's `src/` while fits started from it are running. (4) All queues, the waiter and the fits on job `60108231` stopped; the 2D decks renamed `_baseline_4`; relaunched from a fresh sync on this commit (includes the m-range change, `m.lb` 1.8 for the SH decks).

## 2026-10-22 — RESULT and CHANGE: arb2D nonfinite loss = root-capacity overflow; width-gated root resolution

- `Maxwellian_arb2D_baseline_4` (`20dff12eb3c4491d9d2a83750f07cd03`) stopped at step 371 again (loss 30.1 at step 370), so the failure is deterministic. Captured artifacts (`nonfinite_*`, last finite step 368, loss 30.115) reproduce it on CPU in float64: shot b1 finite; shot b2 (rotated 90 deg) all 16400 model pixels nonfinite. Rotated EDF, sinogram and spectral terms are finite; one geometry (70 deg) sets `root_overflow`: Re(epsilon) has 18 sign changes there (16 at step 368 = `max_roots`), and the quadrature deliberately returns NaN on overflow; the angular weights and IRF spread it to every detector row. Not a GPU underflow.
- Root census at step 368 (all 241 geometries, rotated shot): 1724 roots; 964 are the ion-acoustic roots inside the notch (4 per geometry, 525.4–527.5 nm, half-width 5e-5–0.19 nm); outside the notch the median half-width is 3.1 nm (node spacing 0.109 nm), only 53 roots are narrower than 0.1 nm, and at most 2 per geometry. Every scanned root was being tan-mapped regardless of width.
- USER (verbatim): "This is a lot of roots to resolve via quadrature, and i dont want to just increase the limit that might fix it but is not durable. Are all roots resolved with quadrature or is there some gating? in theory only roots smaller then the nyquist limit need to be resolved this way"; "For point 2 It would also make sense to ignore the IAW roots but im not sure there is a stable criteria that can be used for this, maybe abs(xie) < some number. The concern is that i want the random peaks at low veloctiy to produce peaks so that it harms the loss and disincentivises spikey shapes."; "Go ahead with the change to the resonance detection"; "use 3".
- Not done: an |xi_e| cutoff cannot separate ion-acoustic roots from low-velocity EDF spikes (both at small xi_e) and would hide the spikes the loss should penalize. A width gate keeps every narrow resonance resolved wherever it lies; the ion-acoustic roots are narrow and stay resolved.
- Change: `integrate_detector_bins(..., resolve_width_nodes=None)`: each scan crossing's half-width is estimated from the scan values (|Im eps| / |dRe eps / dlambda|, Im interpolated to the zero of Re); only crossings narrower than `resolve_width_nodes` x the regular node spacing (source span / (integration_panels x regular_order)) are bisected, tan-mapped and counted against `max_roots`; broader ones stay on the regular panels. New diagnostic `scan_root_count`. `FitModel` passes `other.resonance_quadrature.resolve_width_nodes`, default 3.0 (surfaced in `configs/arts-1d` and `configs/arts-2d` defaults); the low-level default `None` keeps the previous behaviour. `max_roots` stays 16 (Phase 5 will size it).
- Threshold accuracy (production grid, Lorentzian of half-width h, regular panels only vs exact, two grid phases): h = 1 / 1.5 / 2 / 2.5 / 3 / 4 node spacings -> worst max error 2.7e-2 / 4.1e-3 / 6.6e-4 / 1.1e-4 / 2.0e-5 / 6.9e-7; default 3 chosen for 2e-5.
- Checks: SG4 arb1D baseline fit weights, gate 2 vs ungated: model image max diff 6.1e-5, objective 8.2e-7; Maxwellian arb2D step 368, gate 2 vs ungated: image 4.6e-4, objective 5.0e-6; step-371 candidate: ungated 16400 nonfinite pixels, gated finite (objective 30.123). Tests: three new in `tests/test_form_factor/test_resonance_quadrature.py` (13 passed); `test_fit_model_integrates_each_geometry_then_aperture_weights` now sets `resolve_width_nodes: 0` (its 0.02 nm resonance is 2.04 node spacings on that coarse grid and exercises the tan-mapped plumbing) and asserts `scan_root_count`; forward/inverse angular, notch, IRF, objective, recovery and form-factor tests 121 passed + 3 (unresolved_arts2d).
- Shared-vault mirror skipped: no vault resolved on this host.
- GPU check of the gate on the extraction (USER: "repeat the sg4 test with the 1d arb to ensure change the quadrature deos not change the extraction"). Session `20261022T185217Z-8979-5771`, job `60121854`, 2 x A100 40 GB, source `429fea91`, one fit per GPU: `SG4_arb1D_baseline_2` (gate 3, `51f862f4b98c470ab4888f00daec9611`) and `SG4_arb1D_ungated` (`resolve_width_nodes: 0`, `8777bc9ea5bc478b896b212d9ec1ecfb`). Wall time 17:39 vs 17:37 (1.06 s/step both; all `max_roots` slots are evaluated either way). The ungated rerun reproduces the first baseline `SG4_arb1D` to every printed digit (Te 0.8349, Te err -16.5132%, Te_eff +0.1218%, <v^2> 1.1993, fe L2 0.0014, objective 27.662665), so this 1D GPU fit is deterministic. Gated: Te err -16.5111%, Te_eff +0.1220%, <v^2> 1.1992, fe L2 0.0014, band ratios identical to 4 digits, objective 27.662672 (relative change 2.5e-7). The gate does not change the extraction.

## 2026-10-22 — LAUNCH: completing the Phase 2 baseline (5 x 2D, 3 x 1D)

- USER (verbatim): "Go ahead with the next step". Session `20261022T192957Z-9908-31535`, Slurm job `60124085` (`--kind gpu`, 4 x A100 40 GB, 4 h), source `42549c97` (rotation fix, nvx 256, map_batch_size 64, width gate 3, m bounds [1.8, 5], state snapshots off the working directory, nonfinite capture); per-fit working directories (`C:\tmp\run_queue_arts2d.sh`), `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.
- Queues: GPU 0 `Maxwellian_arb2D_baseline_5` -> `Maxwellian_dlm_baseline_2`; GPU 1 `Maxwellian_arbSH_baseline_4` -> `Maxwellian_dlm_prior0_2` (prior_strength 0, amp1/amp2 inactive); GPU 2 `SG4_arbSH_baseline_4` -> `SG4_dlm_baseline_2`; GPU 3 `Maxwellian_mySH_baseline_4` -> `SG4_mySH_baseline_4`. Already done with all fixes: `SG4_arb2D_baseline_4`, `Maxwellian_nnSH_baseline_4`, `SG4_nnSH_baseline_4`, `SG4_arb1D_baseline_2`.
- Rule followed: no sync of this session's `src/` while these run.

## 2026-10-22 — RESULT: Phase 2 baseline complete with all fixes

- All eight fits of job `60124085` ended with exit 0 (16:03–18:23 EDT): `Maxwellian_arb2D_baseline_5` `56bdbe8605f3424e9131226b0d1954d2` (passed step 371; the width gate holds), `Maxwellian_arbSH_baseline_4` `d898e28fc24a4925a401673b6bac3dad`, `SG4_arbSH_baseline_4` `174ae3bc4f0042b1aaa2cd825d3d72fe`, `Maxwellian_mySH_baseline_4` `a24abda66fb944d980f72fc165de9e81`, `SG4_mySH_baseline_4` `690dc04225bb46ad871d5fe7601c8351` (patience stop at step 627), `Maxwellian_dlm_baseline_2` `6d5e0a23c8084f059a19e4f032544c85`, `SG4_dlm_baseline_2` `5dbe12147925449bad37ffe5a64eea66`, `Maxwellian_dlm_prior0_2` `1529c3565c5b4d92a4c22aa3a7cfd3c2`. 2D step 3.9–4.0 s on a 40 GB A100 at nvx 256, map_batch_size 64; ~75–85 min per 2D fit. Artifacts `C:\tmp\newfits_b4`, comparison `C:\tmp\compare_phase2_final` (`summary.csv`, `fe_*.png`).
- Results with all fixes (truth Te 1, ne 0.4; Te err / Te_eff err / fe L2 / data): Maxwellian dlm +0.008% / +0.008% / 0.0005 / 2.05 (m 2.0018); arbSH -0.27% / -0.27% / 0.0007 / 2.05; nnSH -0.27% / -0.27% / 0.0007 / 2.05; mySH -0.30% / -0.30% / 0.0007 / 2.05; arb2D +21.3% / +11.0% / 0.074 / 2.11 (<v^2> 0.915). SG4 dlm +0.015% / +0.015% / 0.0007 / 1.98 (m 3.986, unchanged by the m fix); arb1D (`_baseline_2`, gate) -16.5% / +0.12% / 0.0014; arbSH -0.37% / -0.37% / 0.0021 / 2.05; nnSH -0.38% / -0.38% / 0.0023 / 2.05; mySH -0.29% / -0.29% / 0.0012 / 2.04; arb2D -3.3% / -2.9% / 0.059 / 1.81. Maxwellian arb1D was not rerun (no fix acts on it; first baseline +19.3% / +0.14% / 0.0009).
- Reading: the parametric and SH models now recover the truth (Te_eff within 0.4%, fe L2 <= 0.0023, all at the data level of the truth); the SH models went from not training (data 29–71) to the best 2D results, and the m bounds let their isotropic part reach m = 1.991. The arbitrary 2D EDF (65536 free values, no regularization) is the only model that still fails: wrong width/temperature on the Maxwellian, noise-fitting (data below truth level) on SG4, and the bumps that created the extra roots — the case for EDF regularization in Phase 3. The SH fits share a small Te bias of -0.3%.
- Gain-prior check completed (Maxwellian DLM, m bounds [1.8, 5]): default prior Te 1.00008, ne 0.40003, m 2.0018, amp 1.041/1.040, data 2.046, prior 27.88; `prior_strength: 0` with amp1/amp2 inactive: Te 0.99985, ne 0.40004, m 2.0062, data 1.747, prior 0. The prior does not move the parameters, but it pulls the per-row gains off their best values enough to raise the data term from 1.75 to 2.05 (the 80 extra free gains per shot would only explain ~0.01). The "truth level ~2.05" quoted above includes this prior; without it the noise level is ~1.75. Phase 3's gain mode should drop or recentre the prior; amp1/amp2 can be inactive.
- Shared-vault mirror skipped: no vault resolved on this host.

## 2026-10-22 — DATA: anisotropic synthetic pairs rebuilt (biDLM 10023/10024, MYSH 10025/10026); MYSH dtx lowered to 0.006

- USER (verbatim): "Before we move to phase 3 we still need to do a test against an anisotropic case. The synthetic examples had versions build with the biDLM and the MYSH model, can you rebuild those synthetic data sets as you did with the Maxwellian and SG4 cases and then run these same comparisons against them"; "Check the commit description but i believe that update fixed an issue with the MYSH so the new version is correct".
- Sources (experiment 298): biDLM `1afa3612` (arbitrary 2D, flm_type bidlm, m 2, masym 1.2, tasym 1.2, theta 30, nvx 320); MYSH `8f66a558` (sphericalharmonic, mora-yahi, init_m 3, dtx 0.01, nvr 256, nvx 320). The 2025 second images (`d4e11906` theta 120, `aec3afd9` MYSH_y) were the first EDF rotated; the new pairs are made the way the fits model them: second image = forward run with `ThomsonParams.electron.dist_rot = 90` (`forward_rotated.py`). Rotated truth EDFs are exact (`rot90(f, -1)` in `[vy, vx]`, max diff 1e-32).
- Drift against 2025 (row-normalized rms, rows 150–950): biDLM 5.6% / rotated 6.3%; MYSH 10.1% / 9.1% — expected, the 2025 MYSH images used the pre-#145/#147 harmonic convention (2025 SH fits came out isotropic, Txx/Tyy 1.0000).
- Problem found in the dtx 0.01 rotated MYSH image: a 4-px-wide EPW line in rows 0–269 (24–56 deg), up to 2242x the median row peak (rows 150–269 are fitted). Cause: the corrected MY model, f = f00 exp(8 tanh(dtx coeff(v) Y10 / 8)) with coeff ~ v^7, has a bump-on-tail along the flux (projection slope > 0 at u ~ 3.15 vTe, within +-15 deg of +x); rotated by 90 deg it lies along the low-angle ARTS k (~100–118 deg from the probe) at v_phase ~ 3.1 vTe, so the EPW is near-undamped. Scan with the SH module (reproduces the saved truth to 1e-15): bump onset between dtx 0.009 and 0.01; minimum damping ratio (-f')/(-f0') over u 2–4: 0.88 at 0.006, 0.61 at 0.008, 0.01 at 0.009, -0.78 at 0.01. USER: "there should be a notable increase at low angle but not that much, lower the dtx"; "use 0.006".
- New truth runs (experiment 189144, commit `340b8e74`): `truth_biDLM` `799169f2d0c04259b23fc103fd6de13d`, `truth_biDLM_rot90` `b59bb5b4b1ae4b0a8e0b2c45e0e43f9b`, `truth_MYSH_2` `7b5df9019a894b42be5a620aa1aebc83`, `truth_MYSH_rot90_2` `46d4788a66d74673a6e72dbd12917e99` (dtx 0.006). Rotated MYSH at 0.006: max row peak 6.7x median (narrowest low-angle line 22 px); unrotated 3.4x; biDLM 3.7x. The dtx 0.01 runs (`ec0a6ca6`, `f091fa00`) and their data files are kept in `C:\tmp\aniso\superseded_dtx0.01`.
- Data files `tsadar/external/data/ATS-s10023`–`10026.hdf` (`C:\tmp\aniso\make_aniso_data.py`), same recipe as 10021/10022: counts = Poisson(k fit), k so the median row peak in rows 150–950 is 100 counts (k 889.8, 985.5, 1091.8, 813.9), seeds 23–26, rot90(counts x speccal along wavelength, 3), frame 1 = zeros; EDF dataset is the 320 x 320 truth `[vy, vx]`. The SH images have roundoff negatives (~-1e-28, far wings) clipped to 0 before sampling. Through the fitting code's loader and throughput correction, prepared / (k truth) = 0.98–1.00 in every band, as for 10021/10022.
- Fit decks `configs/arts-2d-recovery/{biDLM,MYSH}_{dlm,arb1D,arb2D,arbSH,nnSH,mySH}` (`63caabf8`): copies of the Maxwellian decks with all current fixes, shotnum [10023, 10024] / [10025, 10026], shot_rot 90, run `<set>_<model>_baseline`. Comparison: `compare_arts2d_aniso.py` (`3fa2055f`) — Te/ne, in-plane moments (flow, Txx, Tyy, Txy, Txx/Tyy, principal axis), 2D log10 L2 and vx/vy marginal L2 against the 2D truth, fits mapped to physical units by v sqrt(Te_fit).

## 2026-10-22 — LAUNCH: anisotropic recovery fits (biDLM, MYSH; 12 fits)

- USER: "Once you reach that point go ahead with the fits"; "Kill the session and start a new one" (idle 1-GPU job `60135034` cancelled). Session `20261023T002017Z-13142-27740`, Slurm job `60136879` (`--kind gpu`, 4 x A100 80 GB, 4 h), source `090f4fb9`; per-fit working directories (`C:\tmp\run_queue_arts2d.sh`, launcher `C:\tmp\launch_aniso_fits.sh`), `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Started 20:22 EDT.
- Queues: GPU 0 `biDLM_arb2D` -> `MYSH_arb2D` -> `biDLM_dlm`; GPU 1 `biDLM_arbSH` -> `MYSH_arbSH` -> `MYSH_dlm`; GPU 2 `biDLM_nnSH` -> `MYSH_nnSH` -> `biDLM_arb1D`; GPU 3 `biDLM_mySH` -> `MYSH_mySH` -> `MYSH_arb1D`. Run names `<set>_<model>_baseline`, experiment `arts2d-recovery`.
- Rule followed: no sync of this session's `src/` while these run.

## 2026-10-22 — CHANGE (Phase 3): gain mode global_wing; Phase 4 decks

- USER: "If the amp1 and amp2 can be solved analytically that is better, it removes them as fitting parameters and which will help the minimizers"; "global_wing is by far the better method go ahead with your plan".
- `global_wing` (`a2407b5f`, `tsadar/inverse/loss_function.py`): per shot, one blue and one red scale for the whole image, g_w = sum_w w m t / sum_w w m^2 over every valid pixel of that wing (w = 1/variance, t = data - background, m = model - background), with the existing bounded solve; exact gradient by the envelope theorem. Replaces fitted amp1/amp2 (must be inactive; the identifiability check rejects them otherwise), keeps relative amplitudes between angle rows (per_row profiles them away). Defaults for the mode: smoothness 0 (non-zero rejected), prior_strength 0. Pixels outside both fit windows take the scale of their side of the probe (split halfway between blue_max and red_min); a wing without fitted pixels shows the other wing's scale. Other modes and the per_row code default unchanged. Tests: 5 new in `tests/test_inverse/test_angular_objective.py` (scale recovery at 1 and 900, relative row amplitudes kept vs per_row_wing absorbing them, jit/grad, validation, side scaling); targeted runs pass (angular objective + validate_inputs 36, IRF + angular forward 40 + 1 skipped).
- Truth check (CPU, `C:\tmp\globalwing\truth_objective.py`, Maxwellian DLM deck at exactly Te 1, ne 0.4, m 2, ATS-s10021, k 912.685): per_row data 1.023 + gain prior 13.94 per shot, 80 gains median 903.2 (888–908); global_wing data 0.877 per shot (1.75 for the two-shot sum, the noise level), gains blue 912.23 +- 0.27, red 911.40 +- 0.35 (-0.05% / -0.14% of k).
- Phase 4 decks (`d81851b8`): `configs/arts-2d-recovery/<set>_<model>_globalwing` for Maxwellian, SG4, biDLM, MYSH x dlm, arb1D, arb2D, arbSH, nnSH, mySH — copies of the baseline decks with mode global_wing, amp1/amp2 inactive, run `<set>_<model>_globalwing`; all 24 validated as merged configs.

## 2026-10-23 — RESULT: anisotropic baselines (biDLM 10023/10024, MYSH 10025/10026; per_row gains)

- All 12 fits of job `60136879` ended with exit 0 (20:22–23:39 EDT; 2D ~3.9–4.0 s/step on 80 GB A100s, ~80 min per 2D fit with compile). Run ids: biDLM dlm `49d42602`, arb1D `147e3a9f`, arb2D `9cb39402`, arbSH `c9849ec5`, nnSH `94a437b4`, mySH `b46a11cf`; MYSH dlm `83e09fe1`, arb1D `c0aba635`, arb2D `a4b77a3f`, arbSH `6ebfdbf3`, nnSH `f20dec39`, mySH `2b913561`. Artifacts `C:\tmp\aniso\newfits\{B,H}_<model>`; comparison `C:\tmp\aniso\compare_baseline_{biDLM,MYSH}` (`compare_arts2d_aniso.py`, against the first-shot truth EDF; 2025 fits against the 2025 truths for reference).
- biDLM (truth Txx/Tyy 1.095, principal axis 30 deg, Txy 0.079): arbSH / nnSH / mySH Te +0.01% / +0.10% / +0.02%, ne -0.3 to -0.45%, but exactly isotropic (Txx/Tyy 1.0000, Txy 0) — with Nl = 1 the SH models have only l = 1 (flow-like) terms and cannot represent a temperature anisotropy (l = 2). arb2D is the only model that captures it: Txx/Tyy 1.080, axis 28.9 deg, but Te +14.2%, ne -3.5%, 2D log10 L2 0.29. dlm Te +0.27%, arb1D +10.7% (width trade-off, as for SG4).
- MYSH (dtx 0.006; truth Txx/Tyy 1.063, ux 0.0209): mySH, the generating family, recovers it — Te -0.14%, ne +0.05%, Txx/Tyy 1.065, ux 0.0219, 2D log10 L2 0.010, marginal L2 0.002. arbSH and nnSH have the same l = 1 freedom but come out isotropic (Txx/Tyy 1.000 / 1.002, ux 0 / 0.008), Te -2.1% / -1.7%, 2D L2 1.21 / 0.98 — their l = 1 coefficient does not train away from its zero start (to investigate after Phase 4; no code change made). arb2D Te +24.9%, Txx/Tyy 1.12. dlm -1.8%, arb1D +3.1%.
- 2025 reference fits (old SH convention) were isotropic for every SH model on both sets.

## 2026-10-23 — RESULT (Phase 4): all 24 recovery fits with global_wing gains

- USER: "Lets do two interactive sessions back to back ... Can you continue to monitor the jobs, start the new jobs and session with absolutely no interaction?" Session A `20261023T034551Z-15223-18091` (job `60142934`, 4 x A100 80 GB; Maxwellian + SG4, 23:47–02:59 EDT) and session B `20261023T070700Z-17039-31997` (job `60147968`; biDLM + MYSH, 03:07–06:24 EDT), source `11b44be3` / `23a0f72a` (NOTES-only difference), launchers `C:\tmp\globalwing\launch_phase4_{A,B}.sh`. All 24 fits exit 0; 2D 3.8–3.9 s/step, the same as per_row. Artifacts `C:\tmp\phase4\{M,S,B,H}_<model>`; comparisons `C:\tmp\compare_phase4_iso` (`compare_arts2d_recovery.py`, 2025 / baseline / globalwing; baseline set `C:\tmp\baseline_final` = `newfits_b4` + first-baseline Maxwellian arb1D) and `C:\tmp\compare_phase4_{biDLM,MYSH}` (`compare_arts2d_aniso.py`).
- Objective: with global_wing every model that can represent the truth ends at the noise level with no prior term — Maxwellian/SG4 data 1.64–1.80 (truth 1.75) against per_row 1.98–2.11 plus a gain prior of 25–28; MYSH mySH 1.81 (per_row 2.12 + 28.1). arb2D goes below noise (M 1.67, S 1.47, biDLM 1.66: fitting noise).
- Isotropic sets, parameters unchanged within run-to-run noise (so the per_row prior was not biasing them): Maxwellian dlm Te -0.010% (m 2.0041), SH -0.28% (m 1.995; the shared SH Te bias is not from the gains), arb1D +19.8% / Te_eff +0.10%; SG4 dlm +0.009% (m 4.0021, was 3.986), SH -0.34 to -0.39%, arb1D -17.5% / Te_eff -0.02%. fe slightly closer: SH log10 L2 M 0.0007 -> 0.0004, S 0.0012–0.0023 -> 0.0006–0.0017; 3–4 vTe band ratios M 1.011–1.013 -> 1.005–1.007, S arbSH 1.062 -> 1.041. arb2D still fails: M Te +19.0% (was +21.3%), Te_eff +7.3% (was +11.0%), ne -2.9% (was -5.0%); S Te -4.0%, Te_eff -1.9%.
- biDLM (Txx/Tyy 1.095, axis 30 deg): only arb2D fits the image (data 1.66; every isotropic or Nl = 1 SH model 36–38): Txx/Tyy 1.087, axis 29.3 deg, Txy 0.072 (truth 0.079), Te +12.6% (was +14.2%), ne -1.9% (was -3.5%), 2D log10 L2 0.33. The SH models remain isotropic (no l = 2 term).
- MYSH (Txx/Tyy 1.063, ux 0.0209): mySH recovers it (Te -0.14%, ne +0.05%, Txx/Tyy 1.065, ux 0.0218, 2D L2 0.0088). arbSH still does not learn its l = 1 term (data 178, Txx/Tyy 1.000); nnSH starts to (data 167 -> 117) but overshoots the flow (ux 0.059) with Txx/Tyy 1.003. arb2D data 4.6, Te +22%. Isotropic 1D models 170–180.
- Reading: global_wing removes the prior bias and the 2 x 80 gain degrees of freedom without changing the parameters of models that fit, and helps the arbitrary 2D EDF somewhat (ne error halved, Te_eff error 11 -> 7% on the Maxwellian). The remaining failures are model-side: (1) arb2D without regularization (fits noise, Te/width trade-off) — the EDF regularization step; (2) arbSH / nnSH not training their l = 1 coefficient on MYSH although mySH, with the same harmonic, does; (3) Nl = 1 SH cannot represent a temperature anisotropy (biDLM) — Nl = 2 would be needed.

## 2026-10-23 — RESULT and CHANGE (Phase 4b): why arbSH / nnSH did not learn l = 1; uncapped VFP-style radial coefficients

- USER: "investigate why arbSH and nnSH don't train next, that is the only surprising result"; on the cap: "Approved, implementing what is used in VFP codes is what we want"; "Once the mode with the new v^4 fix is ready go ahead to running the snythetic data on perlmutter".
- Fitted weights of the MYSH global_wing fits: arbSH's relative l = 1 coefficient f10/f00 reached 5e-6 after 1000 steps (flm_sign moved 0.26, flm_mag 0.17); nnSH's reached its bound 1.0. Truth (MY, dtx 0.006, m 3): f10/f00 = -0.15 at 2 vTe, 1.0 at 3, 5.0 at 3.5, 15.7 at 4.
- Causes: (1) `ArbitraryVr` started at flm_sign = 0 (exact zero) with f = 10^(-10 sigmoid(mag)) tanh(sign) f00, so the magnitude had zero gradient and the sign a 1e-5-scaled one; (2) both arbitrary and nn capped |f_lm / f00| <= 1, a P1-positivity guard from the linear expansion (#83, FLM_NN "from 0 to f00") that the positive log-density form of #147 (f00 exp(8 tanh(sum r_lm Y_lm / 8))) no longer needs; (3) the dist optimizer (rmsprop, 1e-3 -> 1e-4 cosine over 750 steps) moves a parameter only ~0.5 over a fit. #145 (issue #136) fixed the coordinates, MY dt and the nn filter, but not these; the #154 benchmark already worked around (1) in its own initialization.
- EDF-only study (`C:\tmp\shtrain\edf_fit.py`, fit of the SH 2D marginal to the truth EDF, |v| < 4.5, nvx 128, the ARTS fit's optimizers, 1000 steps; MYSH truth ux 0.0209, Txx/Tyy 1.0630): capped arbSH loss 1.12 / ux 0.000 / Txx/Tyy 1.000 (initial l = 1 gradient 2e-8 and 0); capped nnSH 1.02 / 0.029 / 1.005, m pulled to 2.59; mySH 1.9e-4 / 0.0216 / 1.064. Uncapped variants: arbitrary r = p 1.12 (cannot move far enough), r = 10 p 0.57 (unstable, ux -0.125), r = (1 + v^4) p 5.6e-6 / 0.0217 / 1.0630; nn plain 1.7e-3 / 0.0232 / 1.069, nn (1 + v^4) 2.6e-4 / 0.0178 / 1.060.
- Change (`2c9164ac`): f_lm = (1 + v^4) g_lm(v) f00 for arbitrary (g = Hanning-smoothed per-node values, zero start) and nn (g = MLP with zero-initialized final layer, key folded per (l, m)); 1 + v^4 follows the collisional growth of f1/f0 (mean free path ~ v^4) so the trainable values stay O(1) from the core to the 3–4 vTe tail; no cap (the 8 tanh log bound keeps f positive and finite), as in VFP spherical-harmonic codes. Filter spec (`base.py`), the benchmark re-initialization (`tsadar/benchmarks/arts2d.py`) and tests updated; 3 new tests (isotropic start with O(1) l = 1 gradient, coefficients above 1 with a finite positive EDF, distinct nn initializations). `tests/test_spherical_harmonics.py` 10 pass + 1 pre-existing Windows tolerance failure (odd-grid parity, 4.5e-13 vs 2e-13); `test_arts2d_marginal.py` 5 and the benchmark smoke test fail identically on the unmodified code. Library EDF-only check reproduces the variants; biDLM Nl = 2: arbSH Txx/Tyy 1.0953, Txy 0.0799 (truth 1.0951, 0.0787), nnSH 1.0995, 0.0787, both loss 4.1e-3 with m pulled to 2.5–2.7 (the 2D biDLM truth has no defined vz behaviour).
- Caveat: old arbitrary-SH weights (flm_sign, flm_mag: two nvr arrays) deserialise into the new (coeff, scale) without error; do not reload pre-2c9164ac arbSH weights. `compare_arts2d_aniso.py` checks scale = 1 + v^4 and skips such files.
- GPU: Nl = 3 nnSH 4.0 s/step (as Nl = 1), peak 36 GB on an 80 GB A100, ~4 min compile. Session `20261023T115444Z-19106-18786` (job `60155011`), source `6d482519`; session 1 (08:05 EDT): MYSH arbSH/nnSH Nl1 and Nl2, biDLM arbSH/nnSH Nl2 and Nl3; session 2 to follow: biDLM Nl1 and MYSH Nl3.

## 2026-10-23 — OBSERVATION: epoch-loss noise in the SH fits (deferred to Phase 4e)

- Histories of the Phase 4b fits (`C:\tmp\lossnoise`, figure `loss_noise.png`): relative scatter of the loss about a 21-step running median, steps 100–300 / 300–600 / 600–1000 — arbSH Nl1 (MYSH) 4.2% / 1.9% / 0.10%; nnSH Nl1 17% / 9.7% / 0.69%; nnSH Nl2 16% / 13% / 2.3%; arbSH Nl2 14% / 6.5% / 0.44%; biDLM arbSH Nl2 3.7% / 1.3% / 0.08%; biDLM nnSH Nl3 16% / 6.6% / 0.79%; mySH 2.3% / 3.3% / 0.37%; arb2D 0.5% / 0.2% / 0.02%. About half of all steps go uphill in the SH fits (0.25 for arb2D). mySH oscillates with period 2, envelope following the dist learning rate.
- Reading: deterministic full-batch optimization, so oscillation rather than gradient noise; the dist group's rmsprop (no momentum) takes sign-like steps of about the learning rate (mySH dt ~ 0.006 moves ±1e-3 per step at lr 1e-3), at the edge of stability; nn's shared weights and the 1 + v^4 scale make it worse; macro Adam runs at a constant 1e-2. Best-loss checkpoints keep arbSH/arb2D results reliable; nnSH's end scatter (~0.04 in the data term) is comparable to model differences.
- USER: "Lets hold off on any changes to fix the epoch loss noise. Add another phase to the plan to investigate NN methods, during that phase we will investigate this issue and investigate implementing a 1D and 2D NN EDF". Plan: Phase 4e (optimizer study incl. Adam and an L-BFGS polish; 1D and 2D NN EDFs).

## 2026-10-23 — RESULT (Phase 4b complete; first Phase 4c fits): 3-V SH at Nl = 1–3; the <v^2> gauge (regT)

- Sessions `20261023T115444Z-19106-18786` (job `60155011`, 80 GB) and `20261023T150741Z-20928-2493` (job `60158753`, 40 GB). MYSH_arbSH_Nl3 ran out of memory on a 40 GB A100 (one 35.9 GB allocation at map_batch_size 64); rerun as `MYSH_arbSH_Nl3_2` with map_batch_size 32 (`2c3328b2`, exact batching; 5.0 s/step); nnSH Nl3 fit at 64. Artifacts `C:\tmp\phase4b_all` (all Nl fits), `C:\tmp\phase4c`; comparisons `C:\tmp\compare_nl_{biDLM,MYSH}`, `C:\tmp\compare_regT_*`.
- MYSH (truth ux 0.0209, Txx/Tyy 1.063): arbSH Te -0.14 / +2.6 / +5.3% at Nl 1 / 2 / 3, nnSH -0.16 / +2.0 / +7.0%; in-plane moments right at every Nl (ux 0.021–0.023, Txx/Tyy 1.063–1.076); l >= 2 share 0 / 0.26 / 0.56 (arb), 0 / 0.25 / 0.83 (nn); data 1.73–1.89. Te grows with Nl through the out-of-plane null space (Tzz/Tyy 1.07–1.09 at Nl 2).
- biDLM (truth Txx/Tyy 1.095, axis 30 deg, Txy 0.079): Nl 1 isotropic (data 36); Nl 2/3: Txx/Tyy 1.086–1.091, axis 30.5–31.8 deg, Txy 0.078–0.082, 2D L2 0.07–0.11 (arb2D 0.33); Te +1.0 to +5.5% with Tzz 1.05–1.21 and in-plane (Txx+Tyy)/2 0.990–0.994; data 1.91–1.97 (above the ~1.75 noise level: no in-plane m = 4 below Nl = 4).
- Gauge (temperature 10, target 2; arb1D target 1): Te barely moves — Maxwellian arb1D +19.8 -> +17.3% (<v^2> 0.85, penalty 0.21), SG4 arb1D -17.5 -> -16.3% (penalty 0.37), arb2D M +19.0 -> +17.4%, SG4 -4.0 -> -3.7%, biDLM +12.6 -> +11.5%, MYSH +22.3 -> +21.4%. Not converged: total loss still falling over steps 750–1000 at the 1e-4 learning-rate floor (M arb1D 2.097 -> 2.044, SG4 arb1D 2.286 -> 2.203, M arb2D 1.926 -> 1.877; truth level ~1.75 / 1.64). The Te <-> width valley is long and shallow for the sign-like dist steps plus constant-rate macro Adam (the Phase 4e optimizer question). biDLM arb2D with the gauge: data 1.69 (was 1.66).
- USER (on the gauge): "I have tried both the regularization penalty for Te and exact by construction in the past. Both for an unknown reason slowed or paralyzed the minimizer. For now can we run session 4 with the Te regularization turned off?" Session 4 decks without the gauge: `<set>_arb2D_regL` (core log curvature 1e-3, edf weight, floor 1e-4) and `<set>_arb2D_regLF4` (full-grid 1e-4, uniform, floor 1e-30).

## 2026-10-23 — RESULT (Phase 4d, first run): polar-harmonic fits did not converge; F0 tail drift; F0 scale 1 + v^2

- Session `20261023T183645Z-22136-10618` (job `60165989`, 40 GB): 8 polar fits (biDLM, MYSH x arbitrary M = 2, 4, 6, nn M = 4; trained F0 scaled 1 + v^4 for arbitrary, unscaled for nn; <v^2> gauge 10), all exit 0, 4.1–4.3 s/step. Artifacts `C:\tmp\phase4d`; comparison `C:\tmp\compare_polar_{biDLM,MYSH}`.
- biDLM: M4 data 1.90 (3-V SH Nl2 1.97), Te +0.6%, in-plane T 1.002, Txx/Tyy 1.085, axis 31.5 deg, Txy 0.081; 2D L2 0.32 (SH 0.07) from the tail. MYSH: data 1.96–2.31 (SH 1.73–1.81), in-plane T up to +12% (M6), M6 spurious axis 15.6 deg; nn M4 best (2D L2 0.063). Not converged: best loss still falling over steps 750–1000 (MYSH M6 2.45 -> 2.31) with end scatter 1.2–9.9% (arbSH 3-V 0.1%).
- Cause: the arbitrary F0 correction g0, outside the bounded exponent and 1 + v^4 scaled, random-walked in the data-free tail (g0 at 6 / 8 vTe: biDLM M4 +9 / -14, MYSH M4 +23 / -16, M6 +20 / +59 in log f); nn F0 (unscaled) stayed within -2.4. EDF-only (M = 4, init_m 3): unscaled F0 cannot reach biDLM (loss 0.38); 1 + v^2 gives biDLM 2.3e-4 and MYSH 5.6e-5 with tail freedom 6.6 / 0.5 (v^4: 8.5 / 4.2). USER chose 1 + v^2 (`5508f558`; radial classes take scale_power). Reruns `<set>_polar{arb,nn}_M4_2` without the gauge, in session 6.

## 2026-10-24 — RESULT (Phase 4c, session 4): log curvature without the gauge

- Session `20261023T214939Z-23690-7981` (job `60176429`, 40 GB), all 8 fits exit 0 (finished 20:50 EDT; the NERSC certificate expired at 18:02, renewed 08:07 on 10-24). Artifacts `C:\tmp\phase4c_s4`; comparisons `C:\tmp\compare_reg_iso`, `C:\tmp\compare_reg_{biDLM,MYSH}` (none = Phase 4 arb2D; regT from session 2).
- Data term / log-curvature penalty: Maxwellian regL 1.733 / 0.061, regLF4 1.726 / 0.060 (none 1.67, noise ~1.75); SG4 1.550 / 0.031, 1.520 / 0.033 (none 1.47, noise ~1.64); biDLM 1.739 / 0.049, 1.715 / 0.044 (none 1.66); MYSH 4.70 / 0.45, 4.50 / 0.33 (none 4.61).
- Maxwellian arb2D: Te_eff error +7.3% (none) / +11.1% (regT) / +5.5% (regL) / +5.9% (regLF4); fe L2 0.080 / 0.097 / 0.076 / 0.069; 3–4 vTe band ratio 0.50 / 0.43 / 0.53 / 0.51. SG4: Te_eff -1.9 / -2.3 / -1.8 / -1.8%; L2 0.053 / 0.052 / 0.048 / 0.044; 3–4 band 5.2 / 4.8 / 5.0 / 4.7. biDLM: in-plane T 1.056 / 1.074 / 1.045 / 1.044 (truth 1.000), Txx/Tyy 1.087 / 1.085 / 1.092 / 1.089 (1.095), 2D L2 0.327 / 0.335 / 0.326 / 0.305. MYSH: in-plane T 1.110 / 1.135 / 1.095 / 1.107 (1.046), 2D L2 0.44 / 0.45 / 0.43 / 0.42.
- Reading: log curvature stops the noise fitting (data up to the noise level) and helps the EDF a little, the full-grid form slightly more; the large-scale shape errors (Maxwellian tail at half the truth, SG4 tail ~5x high, Te_eff +5–6%) remain at these strengths (penalties 0.03–0.06 of ~1.7). MYSH unchanged.
- Session 5 (`20261024T121413Z-24940-17082`, job `60208319`, 80 GB) started 08:15 EDT: regA + regLF5.

## 2026-10-24 — RESULT: success criteria (P, S, F) and the noise-fitting evidence

- USER: "The criteria for a formalisms success should be does it get plasma conditions rights ne, Te<v^2> (not just Te), does it accurately reporduce the spectrum, and most importantly does it reproduce fe. ... the arb2D model is showing degeneracy, it has a different EDF then the truth but produces the spectrum perfectly. Also add a slide showing your claim that it is fitting the noise".
- Truth level (`C:\tmp\noisefit\truth_level.py`): the generating EDF evaluated through the fit deck (global_wing gains; both shots, the second rotated by shot_rot) gives data terms 1.7536 (Maxwellian), 1.6385 (SG4), 1.7832 (biDLM), 1.7429 (MYSH); fitted gains equal the generation scales.
- Noise absorption β = Σ (F − T)(D − T)/σ² / Σ (D − T)²/σ² over valid pixels, shot 1 (`noise_absorb.py`, figure `noise_fig.py`): arb2D 0.081 (M), 0.102 (SG4), 0.086 (biDLM), 0.081 (MYSH); arb2D with full-grid log curvature (regLF4) 0.061–0.077; every other model |β| < 0.01. The arb2D fit departs from the truth by ~0.3σ per pixel, so it is invisible in a single lineout and shows only in the statistics (mean fit deviation rising linearly with the pixel's noise). This revises the session-4 reading: log curvature reduces but does not stop the noise fitting.
- Criteria (`criteria.py` → `criteria.csv`, goals slide matrix): P = |ne| and |Te·<v²>| (in-plane T for the 2-D truths) within 1%; S = data term within ±0.05 of the truth level and β ≤ 0.02; F = log10 fe L2 ≤ 0.005 (1-D marginal) / ≤ 0.02 (2-D). All three: DLM (M, S), arb1D (M), arbSH Nl1 (M, MYSH), mySH (M). fe right with S off: SG4 arb1D / arbSH / mySH (data +0.11 to +0.13 above the truth level; only DLM matches it, open question) and MYSH mySH (+0.07). Fail F: everything on biDLM, arb2D everywhere, polar M4 first run.
- USER: "add some sort of a combined metric ... even if none of the models work we can judge how bad they are ... if no model meets all of our agreement metrics but for example has beta=0.05 we would probably say that is the best possible". Combined score Q (`criteria.py`): weighted geometric mean of each error over its tolerance, each ratio floored at 1 (so beating one tolerance cannot offset missing another); weights fe 1/2, Te·<v²>, ne, data term, β 1/8 each. Q = 1 meets all; the user's example (only β = 0.05 off) gives Q = 2.5^(1/8) = 1.12. 1-D models on the 2-D truths use their worst 1-D marginal L2 against the 1-D tolerance. Goal 4 column: geometric mean of Q over the four sets. Current: arbSH Nl1 1.52, mySH 1.53, DLM 3.31, arb1D 3.39, arb2D 7.67 (log curvature 6.46); best on biDLM arbSH Nl2 2.19.

## 2026-10-24 — RESULT (Phase 4c, session 5): angular smoothing and weaker log curvature

- Session `20261024T121413Z-24940-17082` (job `60208319`, 80 GB): 8 fits exit 0 (regA 08:16–09:39, regLF5 09:39–11:02 EDT). Artifacts `C:\tmp\phase4c_s5`; comparisons `C:\tmp\compare_reg_{iso,biDLM,MYSH}` via `C:\tmp\noisefit\compare_reg.sh`; criteria `C:\tmp\noisefit\criteria.py` (picks up each regularized fit automatically).
- Q per set M / SG4 / biDLM / MYSH (all four): none 7.4 / 5.1 / 7.3 / 12.6 (7.67); log curvature 1e-4 (regLF4) 5.9 / 4.2 / 5.9 / 11.9 (6.46); 1e-5 (regLF5) 6.9 / 4.7 / 6.6 / 12.1 (7.16); angular 10 (regA) 6.3 / 4.5 / 6.2 / 11.2 (6.66). β: none 0.08–0.10, regLF4 0.06–0.08, regLF5 0.07–0.09, regA 0.05–0.06.
- Maxwellian Te·<v²> error +7.3 (none) / +5.9 (LF4) / +7.0 (LF5) / +5.9% (regA); fe L2 0.080 / 0.069 / 0.079 / 0.080. biDLM 2D L2 0.327 / 0.305 / 0.300 / 0.344. MYSH unfitted throughout (data +2.8 to +3.3 above the truth level).
- Reading: log curvature improves monotonically with strength (1e-5 between none and 1e-4); angular smoothing cuts noise absorption most but leaves the shape. Candidate next test: full-grid log curvature 1e-3. Goals matrix now shows only the best regularized arb2D row (by overall Q).
- Session 6 (`20261024T150538Z-26707-29740`, job `60215169`, 80 GB, source `af4530da`) started 11:08 EDT: regAR (angular 10 + radial 1) on all four sets, then biDLM/MYSH polar{arb,nn}_M4_2.

## 2026-10-24 — RESULT (Phase 4c/4d, session 6): angular + radial smoothing; polar M = 4 reruns

- Session `20261024T150538Z-26707-29740` (job `60215169`, 80 GB, source `af4530da`): 8 fits exit 0 (regAR 11:08–12:32, polar 12:32–13:58 EDT). Artifacts `C:\tmp\phase4c_s6`; comparisons `C:\tmp\compare_reg_*` and `C:\tmp\compare_polar2_{biDLM,MYSH}` (first run vs rerun).
- arb2D regAR (angular 10 + radial 1): Q 5.0 / 3.5 / 5.5 / 10.6 (all four 5.62, the best regularizer so far; regLF4 6.46, regA 6.66, none 7.67); β 0.05–0.06. Maxwellian Te·<v²> +3.7% (none +7.3%), fe L2 0.059 (0.080); SG4 −1.2%, 0.036 (0.053); biDLM 2D L2 0.339, MYSH 0.437 (unchanged).
- Polar M = 4 reruns (F0 1 + v², no gauge): in-plane T biDLM arb 0.999 / nn 1.006 (truth 1.000), MYSH 1.056 / 1.048 (1.046); 2D L2 biDLM 0.50 (first 0.32) / 0.31 (0.52), MYSH 0.24 (0.24) / 0.065 (0.063); data biDLM 1.85 / 1.84 (truth level 1.783), MYSH 2.10 / 1.99 (1.743); MYSH arb axis 16 deg (truth 0). Q: biDLM 5.17 / 3.98, MYSH 4.46 / 2.21. Reading: the tail drift is gone and in-plane T is right without a gauge, but the spectra stay above the truth level (under-fitted) and the shape is 7–25x its tolerance; nn beats arbitrary on both sets; polar is not yet competitive with the 3-V SH on MYSH (arbSH Nl1 Q 1.00).
- USER (on session 5): "log curvature and angular smoothing still show the most promise. We should add a test at 1e-3 and 1e-2 for the log curvature and a combined log curvature + angular smoothing". Decks `db06dd83`: regLF3 (1e-3), regLF2 (1e-2), regLFA (1e-4 + angular 10). Session 7 (`20261024T180357Z-28251-24236`, job `60220205`) started 14:05 EDT: regLF3 then regLFA; session 8: regLF2.

## 2026-10-24 — RESULT (open item 3): had the fits converged?

- USER: "look at the polar model fits and the others too and determine if the fits had converged by the time they finished, models that have not converged have the potential to do better with longer runs". Loss histories of all 90 downloaded fits from MLflow (`C:\tmp\lossnoise\fetch_all.py` → `hist\`); analysis `convergence.py`, verdict `convlib.py` (used by `criteria.py`; † in the goals matrix), figure `convergence_fig.py`.
- Method: medians of the per-step loss in 100-step windows 600–1000 (robust to the step-to-step oscillation); the dist learning rate decays 1e-3 → 1e-4 by step 750 and stays there. Converged = early-stopped (500 steps without a new best: Phase 4 M/S arbSH and mySH, B mySH, H dlm) or a last-100-step drop below max(0.002, half the end scatter); otherwise the remaining drop at the floor is estimated by continuing the geometric slow-down of the last two window drops.
- Not converged, could gain: MYSH arb2D (all regularizations) loss 2.8–3.4 above the truth level, still falling ~0.5 per 100 steps at the end (MYSH arb2D failure is under-training); MYSH polar arb +0.36, ~0.25 still to gain; SG4 arb1D +0.13, ~0.07; Maxwellian arb1D +0.04, ~0.05; MYSH arbSH Nl2 +0.07, ~0.02.
- Converged or plateaued (longer runs at these settings unlikely to help): SG4 arbSH / mySH flat at +0.11 from step ~100 (so the SG4 offset, open item 2, is not convergence for SH; for arb1D partly); biDLM polar arb/nn +0.06 (< 0.01 left); MYSH polar nn +0.25 and nnSH Nl2 +0.12 plateau inside 0.03–0.05 step scatter (optimizer-limited, Phase 4e); DLM, arbSH Nl1, mySH converged.
- arb2D on Maxwellian, SG4, biDLM (all regularizations) still falling but already 0.03–0.17 below the truth level: more steps would fit more noise.
- Suggested test: longer schedule (2000 epochs, cosine decay over 1500, same floor) for MYSH arb2D (none or regAR), MYSH polar arb, Maxwellian and SG4 arb1D.

## 2026-10-24 — PLAN: close the gap to a working model; re-ordered phases

- USER: "make a new plan (for after the current one) with the goal of determining how to close the gap between the current models and a working model ... 3 sets of models the arb2D, spherical harmonics, and polar models ... Each one will need its own path to convergence or elimination ... learn the most possible with the least number of actual runs". Plan `C:\Users\amild\.claude\plans\arts2d-close-the-gap.md`: Stage 0 build the 1D/2D NN EDFs (fourth family) and the tools (truth in each family, evaluation harness, warm start from fitted_weights.eqx, noiseless images); Stage 1 zero-training diagnostics (attainable floor, representability, loss slice fit → truth on noisy and noiseless images → A noise / A degenerate / B local minimum / C slow, information map, gradients incl. the arb2D dead gradient at fval → 0 from f = 10^(−fval²)); Stage 2 warm-started short one-factor runs; Stage 3 cold-start confirmation and elimination. ~4–6 GPU sessions.
- USER (order): "the NN EDFs should get built before the new plan, then the entire new plan should get run and will answer the gauge question as part of why its steps are slow"; "The NN build is now in the new plan"; "Why is the real-shot fit needed? If we dont have a model that we know works on testing data we won't know where any issues with a real shot come from. Also lets rename 4e to 5a and 5 to 5b as both are really refinements". Order: sessions 7–9 → close-the-gap plan (NN build first; gauge question, SG4 offset and polar under-fitting inside it; Phase 4c-2 folded in) → Phase 5a optimizer (was 4e item 1) → Phase 5b scaling (was 5) → real shots (the shot 94475 global_wing check deferred until a model works on all four test problems).

## 2026-10-24 — RESULT (Phase 4c, session 7): log curvature 1e-3; log curvature + angular; sessions 8/9 merged

- Session `20261024T180357Z-28251-24236` (job `60220205`, 80 GB, source `db06dd83`): 8 fits exit 0 (regLF3 14:05–15:29, regLFA 15:29–16:52 EDT). Artifacts `C:\tmp\phase4c_s7`.
- Q M / SG4 / biDLM / MYSH (all four): regLF3 (full-grid log curvature 1e-3) 5.1 / 3.7 / 5.4 / 11.6 (5.90); regLFA (1e-4 + angular 10) 5.3 / 3.9 / 5.4 / 10.7 (5.89); for reference regLF4 6.46, regA 6.66, regAR 5.62 (still the best). β: regLF3 0.044–0.058, regLFA 0.041–0.052 (lowest so far). Data − truth level: regLF3 M +0.046, S −0.056, B −0.006; regLFA M −0.009, S −0.078, B −0.019. fe L2 unchanged: M 0.068 / 0.068, SG4 0.044 / 0.046, biDLM 0.322 / 0.320 (regLF4 0.069, 0.044, 0.305; regAR 0.059, 0.036, 0.339). None converged at step 1000 (0.03–0.13 still to gain on M/S/B).
- Reading: stronger or combined smoothing lowers noise absorption and brings the data term to the truth level, but does not move the large-scale shape error; only the radial term has improved the isotropic shape. β stays 0.04–0.05 even with the data term at the truth level.
- USER approved the session 8/9 review: sessions merged; arbSH Nl = 2 on Maxwellian/SG4 cut to 300 epochs (`e7638c3f`; SH settles on the isotropic sets by step ~100–230); nnSH Nl = 2 on Maxwellian/SG4 and MYSH regLF2 dropped; regLF2 on M/S/B dropped on the regLF3 result (data at or above the truth level on M and biDLM, fe not improving; 1e-2 would over-smooth).
- Session 8 (`20261024T205440Z-30039-14456`, job `60224242`, source `e7638c3f`) started ~17:00 EDT: polar arb/nn M = 4 and arbSH Nl = 2 (300 epochs) on Maxwellian and SG4; ends ~19:30 EDT.
