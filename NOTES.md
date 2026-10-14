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
