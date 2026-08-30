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
