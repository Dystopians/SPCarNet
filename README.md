<h1 align="center">MeshSplatOpt</h1>
<p align="center"><em>Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting</em></p>

<p align="center">
  <strong>English</strong> &nbsp;|&nbsp; <a href="README.zh.md">中文</a>
</p>

<div align="center">
  <a href="docs/NeurIPSRepairPrompts.md">NeurIPS roadmap</a> &nbsp;|&nbsp;
  <a href="docs/car_model/final_stageF12_multiscene_package_report.md">Multi-scene package (F12)</a> &nbsp;|&nbsp;
  <a href="docs/car_model/final_stageF47_F48_csef_family_all_metric_repair_report.md">CSEF-family all-metric (F47–F49)</a> &nbsp;|&nbsp;
  <a href="docs/car_model/final_stageF75_adaptive_policy_reflection_report.md">Adaptive policy (F75)</a>
</div>

<br>

<div align="center">
  <img src="assets/meshsplatopt_method.svg" width="950" alt="MeshSplatOpt method overview">
</div>

> **In one sentence (intent).** Existing Mesh-Splatting / 3DGS pruning methods ask *which primitives can be removed*; MeshSplatOpt asks *which local surface edit best reduces scene-evidence debt while remaining counterfactually certified by held-out rendering and sparse geometry.* The same edit calculus is meant to handle deletion, collapse, snapping, splitting, hole filling, and appearance recovery — every committed edit must clear render, sparse-depth, normal, free-space, and topology certificates, otherwise it rolls back.

> **In one sentence (current evidence).** As of 2026-05-05, the **validation-budget CSEF-family compact-recovery protocol** (clean long → CSEF-family area / redundancy compaction → strict topology-frozen recovery + sparse-depth + small LPIPS where needed) now beats the strongest clean-long baseline on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE on **5 / 5 selected scenes** (`parking_phone_tiny`, `bonsai`, `courtyard`, `room`, `counter`) and on the sparse-normal proxy on 4 / 5 (courtyard ties), with topology reductions of 40 – 70 % per scene; on `parking_phone_tiny` the **adaptive CSEF policy with tiny LPIPS recovery (F75)** is now the strongest single row, supersedes the earlier R53 / F7 rows, and chooses the prune ratio from checkpoint evidence rather than from a hand-set table.

The method scaffold (CSEF + reversible edit calculus + counterfactual certificates) and the recovery recipe are kept honest by separating *what passed every gate* from *what actually improved the headline metrics*. The fixed-CSEF50 audit (F45) intentionally documents that one prune ratio does not work for every scene; the published claim is therefore a validation-selected CSEF-family protocol, not a single universal hyperparameter.

---

## Honest project status

R0 → R56 (scaffold + parking single-scene line) and **F1 → F75 (final cross-scene line)**. Stages with `_FAIL`, `REJECTED`, or `MIXED` decisions are the failure-evidence backbone of the current paper-discipline story.

### Method scaffold (R0–R15)

| stage | scope | decision |
|---|---|---|
| R0 | branch, audit, pivot lock | `PASS` |
| R1–R2 | RFC + related-work / novelty matrix | `PASS` |
| R3 | CSEF data model + diagnostics | `PASS` |
| R4 | defect mining (floater / dent / rough / misalign / hole / giant void) | `PASS` |
| R5 | unified reversible edit abstraction (snapshot · apply · rollback) | `PASS` |
| R6 | strong delete / collapse / merge baselines | `PASS` |
| R7 | snap / deform proposals | `PASS` |
| R8 | giant ground-void & large-hole fill proposals | `PASS` |
| R9 | object-prior vehicle-region repair (gated) | `PASS` |
| R10 | generalized counterfactual validation for arbitrary edits | `PASS` |
| R11 | teacher-guided appearance & geometry recovery | `PASS` |
| R12 | edit portfolio & repair state machine | `PASS` |
| R13 | synthetic repair benchmark | `PASS` (full ≥ delete-only on 5 / 7 categories; unknown void rejected) |
| R14 | real-checkpoint dry-runs, render-backed gates, freeze-densify schedule | `TOPOLOGY_RETENTION_PASS` |
| R15 | three-scene medium-budget freeze validation | `MULTI_SCENE_SCHEDULE_PASS_SNAP_SELECTOR_WEAK` |

### Selector and edit-primitive failure log (R16–R26)

| stage | scope | decision |
|---|---|---|
| R16 | three-scene **full**-budget freeze (2000 → 7000) | `THREE_SCENE_FULL_SCHEDULE_PASS` (schedule, not edit) |
| R17.01–R17.05 | area-seeded / portfolio local snap | `PORTFOLIO_SNAP_GATE_PASS_RECOVERY_QUALITY_FAIL` |
| R17.06 | risk-filtered area snap (boundary excl., uncertainty cap) | `RISK_FILTERED_LOCAL_SNAP_GATE_PASS` (numerical-noise deltas) |
| R18.01–R18.03 | train-residual snap (parking) | `GATE_PASS_RECOVERY_MOSTLY_POSITIVE` (small effect) |
| R19.01–R19.08 | residual snap, cross-scene (courtyard + bonsai) | `CROSS_SCENE_GATE_PASS_RECOVERY_MIXED_POSITIVE` |
| R20 | parking medium residual snap (2000 → 4000) | `MEDIUM_RESIDUAL_SNAP_DEPTH_GAIN_RENDER_QUALITY_FAIL` |
| R21 | residual **patch** snap (k-hop expansion) | `PATCH_SNAP_GATE_PASS_RECOVERY_MIXED` |
| R22 | boundary fan `FILL_PATCH` (parking) | `BOUNDARY_FILL_GATE_PASS_SHORT_PROMISING_MEDIUM_FAIL` |
| R23 | residual-aware boundary-loop selector | `SELECTOR_PASS_GEOMETRY_STILL_WEAK` |
| R24 | nearest-face field initialization on appended fill faces | `PASS` (engineering only) |
| R25 | unfrozen densification post-edit (diagnostic) | `FAIL` — topology blew up to 5.89M tri |
| R26 | plane-grid Delaunay fill (51 v / 106 f) | `FILL_INIT_GRID_ENGINEERING_PASS_MEDIUM_REPAIR_FAIL` |

### Recovery-recipe wins and the clean-baseline correction (R27–R53)

| stage | scope | decision |
|---|---|---|
| R27 | low-λ sparse-COLMAP-depth recovery, λ = 0.005 (medium) | `SPARSE_DEPTH_REPAIR_MEDIUM_PASS`, but matched control shows **sparse-recovery is the dominant contributor** |
| R28 | full-budget grid-fill + sparse vs matched baseline+sparse | **`SPARSE_DEPTH_FULL_PASS_GRID_FILL_REJECTED`** — the edit does not beat baseline+sparse at 7000 |
| R29 | alternate sparse-depth loss spaces (relative / log) | `LOSS_SPACE_DIAGNOSTIC_REJECTED_FOR_PARKING_FULL` |
| R30 | long-horizon continuation up to 20 000 iter | `RENDER_EARLY_STOP_AT_16000` |
| R31 | cross-scene sparse recovery on courtyard + bonsai | `CROSS_SCENE_SPARSE_RECOVERY_PASS` |
| R32–R36 | trusted (low-error) sparse correspondence sampling | `TRUSTED_SAMPLING_GEOMETRY_PASS_RENDER_MIXED` (per-scene fraction) |
| R37 | error-stratified sampler | rejected — controlled negative |
| R38–R39 | λ fine-sweep (0.005 → 0.002) | `NEW_STRONGEST_PARKING_RESULT_AND_LAMBDA_CURVE_PASS` |
| R40–R42 | low-λ regime + cross-scene jump (R40.02 courtyard) | `LOW_LAMBDA_CROSS_SCENE_STRONG_PASS` |
| R43 | long-horizon validation 16k → 30k / 7k → 20k | `LONG_HORIZON_VALIDATION_SPLIT` (parking overtraining; courtyard render-only) |
| R44 | sparse-depth **decay** schedule | `SPARSE_DECAY_LONG_HORIZON_REPAIR_PARTIAL_PASS_CLEAN_LONG_RENDER_FAIL` |
| R45–R46 | clean-render teacher loss from the low-topology R44 checkpoint | `LOW_TOPOLOGY_TEACHER_DISTILLATION_REJECTED` |
| R47–R50 | clean 22k -> 80% area compaction -> topology-frozen recovery | `CLEAN_TO_COMPACT_RECOVERY_PASS_EARLY_STOP_AT_26K` |
| R51–R52 | direct LPIPS-loss recovery from R48 | `DIRECT_LPIPS_LOSS_REJECTED` |
| R53–R56 | clean 22k -> 65/70/75% area compaction + continuation checks | `CLEAN_TO_COMPACT_DOMINATES_CLEAN_LONG_BASELINES` |

The **R44.01 vs clean 22k** comparison is the load-bearing parking failure evidence — see `docs/car_model/parking_best_clean_long_vs_method_long_report.md`. The R48-to-R53 repair that follows it is documented in `docs/car_model/parking_clean_to_compact_repair_report.md`.

### Cross-scene final package and adaptive-policy line (F1–F75)

| stage | scope | decision |
|---|---|---|
| F1–F8 | cross-scene compact recovery pilot on 5 scenes (parking, bonsai, courtyard, room, counter) | `CROSS_SCENE_COMPACT_PILOT_PASS` |
| F10 | counter 4th scene boundary case at fixed CSEF50 | `BORDERLINE_SSIM_FAIL` (later repaired by F46 CSEF20+sparse) |
| F12 | multi-scene package: 5 / 5 compact-recovery PASS at clean-long 22k | `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS` |
| F13 | paper assets package | `PASS` |
| F16 / F19 / F26 | random-same-count compaction control on counter / room / bonsai | rejected — random50 / random40 lose to area / CSEF / QEM |
| F18 / F20 / F22–F25 | post-hoc QEM strong baselines | `PASS_AS_BASELINES` (counter / room QEM50 frozen are PSNR-strong; F25 Open3D QEM did not reach the matched parking topology) |
| F27 / F35 / F36 / F18 / F24 | no-freeze controls (CSEF / area / QEM) on every final-package scene | `NO_FREEZE_FAIL` — strict topology-freeze contract is required |
| F28 / F29 / F30–F32 / F33 | sparse-depth strict recovery on all five scenes | `SPARSE_DEPTH_PASS_PER_SCENE` |
| F33 | parking CSEF70 + sparse-depth strict recovery 26k | `PARKING_PARETO_PROMOTE` (now the F12 parking row) |
| F34 | parking sparse-depth 26k → 30k continuation | `LONG_CONTINUATION_REJECTED` |
| F37 | fast-QEM matched parking topology | `FAST_QEM_REJECTED` — sparse geometry up, render collapses |
| F38 | synthetic no-gate / no-rollback counterfactual | `GATE_BLOCKS_UNSAFE_EDITS_PASS` |
| F39 / F41 / F42 | parking real gate-removed ratio0.04 (500 / 2000 / 7000 iter) | `GATE_RENDER_PASS_GEOMETRY_MIXED` |
| F43 | bonsai 7000-step real gate-removed | `BROAD_STRICT_GATE_NEGATIVE` (no-gate strictly better — kept as discipline evidence) |
| F44 | bonsai calibrated-gate repair | `CALIBRATED_GATE_PASS_CLOSE_TO_NO_GATE` |
| F45 | fixed-CSEF50 audit | `FIXED_PRESET_AUDIT_FAIL` — fixed CSEF50 is **not** a five-scene all-metric win |
| F46 | unified CSEF + sparse-depth + validation-selected budget (room CSEF20, counter CSEF20, parking CSEF50) | `VALIDATION_BUDGET_PASS_WITH_FIXED50_LIMITATION` |
| F47 / F49 | bonsai CSEF50 + sparse-depth + small LPIPS (λ = 0.005) | `CSEF_FAMILY_ALL_METRIC_BONSAI_REPAIR_PASS` |
| F48 | consolidated CSEF-family package, 5 / 5 all-metric clean-long wins, no QEM rescue | `CSEF_FAMILY_ALL_SCENE_ALL_METRIC_PASS` |
| F50 | parking calibrated-gate replication of F44 | `CALIBRATED_GATE_REPLICATION_MIXED` (does not reproduce the bonsai mechanism repair on parking) |
| F57–F67 | adaptive CSEF policy attempts (render-only evidence) | `ADAPTIVE_POLICY_FAIL` (drove wrong fraction / ranking) |
| F68 | adaptive selector with area / redundancy primary, render as risk only | `CORRECTED_ADAPTIVE_SELECTOR_PASS` |
| F69 | adaptive + sparse-depth (no LPIPS) | beats R53; misses F7 LPIPS by 0.000063 |
| F71 / F72 / F73 | adaptive + sparse + heavier LPIPS | `LPIPS_HEAVY_REJECTED` — depth regresses |
| F74 | adaptive + sparse + LPIPS λ = 0.0001 | `CONSERVATIVE_ALL_METRIC_F7_WIN` |
| **F75** | **adaptive + sparse + LPIPS λ = 0.00025 (parking headline)** | **`ACCEPTED_FOR_PARKING_HEADLINE`** — supersedes R53.01 / F7 on every tracked metric |
| F76 (in flight) | F75 fixed-policy multi-scene replication | running |

---

## Where the method actually stands today

### What is validated

- **The validation-budget CSEF-family compact-recovery protocol now passes on 5 / 5 selected scenes.** Each scene has a long-run row that beats the strongest clean-long 22k baseline on PSNR, SSIM, LPIPS, AbsRel, and Depth MAE, with topology reductions of 40 – 70 %. Sparse-normal proxy improves on 4 / 5 (courtyard ties at +0.0085° — explicitly disclosed, not claimed as a win). Per-scene chosen rows: parking CSEF50 + sparse-depth (F46), bonsai CSEF50 + sparse-depth + LPIPS λ = 0.005 (F49), courtyard CSEF50 + sparse-depth (F30), room CSEF20 + sparse-depth (F46), counter CSEF20 + sparse-depth (F46) — the prune ratio is validation-selected per scene from the same CSEF selector family.
- **Adaptive CSEF policy is the strongest single parking row (F75).** It reads checkpoint evidence to choose the prune fraction (parking → 70 %), ranks compaction candidates by area / local redundancy primarily, and uses render-only evidence as a risk / audit signal. Layered with sparse-depth recovery and a tiny LPIPS term (λ = 0.00025), it supersedes R53.01 / F7 on every tracked metric: PSNR 18.7119 / SSIM 0.6479 / LPIPS 0.3375 / AbsRel 0.0789 / Depth MAE 1.8500 / normal 43.95° at the same 2 564 473 triangles. F74 (λ = 0.0001) is the conservative neighbor.
- **Sparse-COLMAP-depth supervision during recovery is the dominant contributor.** Validated regime: λ ∈ [0.001, 0.005] depending on scene, `mixed_low_error` correspondence sampling with a per-scene trusted fraction, decay window after the geometry has anchored.
- **Strict topology-freeze is required.** Use `--freeze_topology_updates --skip_restricted_delaunay` together for fixed-topology continuation. `--skip_restricted_delaunay` alone skips only the Delaunay refresh; it does not disable the standard prune / densify branch. F27 / F35 / F36 / F18 / F24 confirm on every final-package scene that omitting strict freeze collapses or drifts topology and loses render.
- **Counterfactual gating works as designed for unsafe-edit rejection.** F38 (synthetic no-gate / no-rollback) shows the gate exactly restores all unsafe edits; F39 / F41 / F42 (parking real gate-removed at 500 / 2000 / 7000 iter) confirm gate-on rolls back the same no-accept candidate that gate-off commits, and gate-on wins render at 7000 iter. F44 calibrated-gate bonsai repair preserves gating, accepts three recoverable rounds, rejects three later rounds, and lands close to no-gate with a smaller mesh.
- **The full reversible edit pipeline** — proposal JSON → snapshot → apply → render-backed counterfactual gate → automatic rollback — works end-to-end on real Mesh Splatting checkpoints for `SNAP_VERTICES`, `FILL_PATCH` (fan and Delaunay grid), and the synthetic R13 set.
- **The synthetic repair benchmark passes** on `giant_ground_void`, `ground_wall_misalignment`, `local_dent`, `noisy_rough_patch`, and `small_hole`; the unobserved-void case is correctly rejected in normal mode.

### What does *not* work yet

- **Fixed CSEF50 is not a universal preset (F45).** Across the four completed fixed-CSEF50 long rows there is 1 clear pass (courtyard), 2 borderline / mixed (bonsai LPIPS regresses by +0.000257; room depth regresses), and 1 fail (counter). The published claim must be a validation-selected CSEF-family protocol with a per-scene compaction budget, not a single universal hyperparameter.
- **Broad strict-gate dominance is false (F43).** On bonsai 7000-iter, strict gate rolls back all six candidate rounds and ends much worse than no-gate on every tracked metric. F44 calibrated thresholds repair the bonsai mechanism, but F50 calibrated-gate parking replication does not reproduce the F44 mechanism — calibrated thresholds are a scene-aware tradeoff, not a universal gate-superiority claim.
- **Long-continuation past the validated budget hurts.** F34 parking 26k → 30k regresses PSNR / SSIM / LPIPS / normal; R56 R53 26k → 28k loses ~0.35 dB PSNR; R49 / R50 30k continuation also lose. The accepted parking checkpoint stays at 26k.
- **Posthoc QEM is not a render headline.** F37 fast-QEM reaches the matched parking topology and improves sparse geometry but collapses PSNR / SSIM / LPIPS; F25 Open3D QEM did not reach the parking topology target at all (stopped at 8.13M / 8.55M).
- **Random same-count compaction is a clear loser** (F16 counter, F19 room, F26 bonsai) — area, CSEF, and QEM all dominate it. Kept as a discipline-control negative.
- **Edit primitives do not improve headline metrics at full budget on parking.** R28 ablates this directly: matched baseline + sparse-depth (no edit) ties or beats grid-fill + sparse-depth at 7000 iter on parking PSNR. The fill / snap edits are gate-safe and trainable, not quality-improving on their own.
- **The ultra-low-topology R44 path loses on render.** Clean 22k reaches PSNR 18.48 / SSIM 0.635 / LPIPS 0.347; R44.01 reaches 17.17 / 0.549 / 0.442. R44.01 remains useful only as a very-small-topology / normal-proxy point. Teacher distillation from R44 (R45 / R46) does not fix the failure.
- **Heavy LPIPS recovery loss is rejected.** R51 (λ = 0.02) / R52 (λ = 0.05) on top of R48, and F71 / F72 / F73 LPIPS-heavy adaptive runs, all worsen depth metrics; only the tiny λ ∈ {0.0001, 0.00025} regime (F74 / F75) keeps the depth wins.
- **Area-seeded snap selectors fail** (R17 area portfolio, R17.06 risk-filtered) — numerical-noise gate deltas and lose equal-budget continuation. **Residual snap / patch snap are tiny** (R18 / R19 / R20 / R21). **Boundary `FILL_PATCH` fails medium recovery** (R22 fan; R26 grid; R28 full).
- **Unbounded post-edit densification is not a recovery strategy** — R25 grew parking to 5.89M tri and still ended at PSNR 12.03.
- **Alternative sparse-depth loss spaces (`relative`, `log`, `inverse`) are rejected** for parking full budget — the original metric-depth Smooth-L1 form remains the validated variant.

A short summary of the "what does and does not work" carved out of R0–R56 + F1–F75 lives in [`docs/car_model/SPCarNet_research_log.md`](docs/car_model/SPCarNet_research_log.md).

---

## Headline figure — five-scene clean-long vs validation-budget CSEF-family (F40 / F49)

Per-scene rows; columns are GT · clean-long 22k baseline · ours (F49 best CSEF-family row at the validation-budget) · per-pixel error vs. GT for clean-long and for ours.

<div align="center">
  <img src="assets/meshsplatopt_multiscene_clean_vs_method.png" width="900" alt="Five-scene clean-long 22k vs MeshSplatOpt method-best 26k">
</div>

### Five-scene quantitative summary (F12 / F49 best rows vs scene-matched clean-long 22k)

All deltas are `method − clean-long 22k` evaluated independently with `render.py + metrics.py` and a sparse COLMAP geometry proxy. CSEF-family rows are F49 (no QEM rescue).

| scene | clean-long tri | ours tri | reduction | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | ΔAbsRel ↓ | ΔDepth MAE ↓ | ΔNormal ° ↓ | ours row | W&B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `parking_phone_tiny` | 8 548 242 | 4 274 121 | 50.0 % | +0.159 | +0.0102 | −0.0098 | −0.0016 | −0.0055 | −0.79 | CSEF50 + sparse-depth (F46) | `8l96pfjx` |
| `bonsai` | 88 460 | 44 230 | 50.0 % | +0.010 | +0.0020 | −0.0048 | −0.0097 | −0.0854 | −2.15 | CSEF50 + sparse + LPIPS λ = 0.005 (F49) | `cuq7olfd` |
| `courtyard` | 1 677 484 | 838 742 | 50.0 % | +0.449 | +0.0422 | −0.0237 | −0.0330 | −0.2107 | −0.21 | CSEF50 + sparse-depth (F30) | `9aaku1yn` |
| `room` | 84 506 | 67 605 | 20.0 % | +0.710 | +0.0656 | −0.0445 | −0.0027 | −0.0075 | −1.47 | CSEF20 + sparse-depth (F46) | `v7ld1o0x` |
| `counter` | 83 834 | 67 067 | 20.0 % | +0.209 | +0.0234 | −0.0163 | −0.0027 | −0.0050 | −1.18 | CSEF20 + sparse-depth (F46) | `pijpv7ny` |

Five of five scenes improve PSNR, SSIM, LPIPS, AbsRel, and Depth MAE; sparse-normal angle improves on four of five (courtyard ties at +0.0085° — explicitly disclosed). All rows use strict `--freeze_topology_updates --skip_restricted_delaunay` topology freeze and online W&B; recovery is `22000 → 26000`.

### Parking deep-dive (R44 → R53 → F75)

The parking scene also has a separate single-scene line carrying the failure-evidence backbone (R44 vs clean 22k) and the strongest single row (F75). Each row is one held-out test view; columns are GT, clean-long 22k / 30k, R48, and R53.

<div align="center">
  <img src="assets/meshsplatopt_clean_vs_r53_montage.png" width="900" alt="Clean long baseline vs R53, parking_phone_tiny">
</div>

| run | iter | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | Depth MAE ↓ | Normal ° ↓ | triangles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean 7k (historical, weak ref) | 7 000 | 17.20 | 0.535 | 0.451 | 0.076 | 1.75 | 45.56 | 833 775 |
| clean 22k (strongest baseline) | 22 000 | 18.480 | 0.635 | 0.347 | 0.082 | 1.87 | 45.11 | 8 548 242 |
| clean 30k | 30 000 | 18.409 | 0.632 | 0.351 | 0.082 | 1.87 | 44.84 | 8 548 242 |
| ours R44 22k (decay, fail vs clean 22k) | 22 000 | 17.170 | 0.549 | 0.442 | 0.187 | 2.92 | 42.22 | **782 982** |
| ours R48 26k (80 % area prune) | 26 000 | 18.620 | 0.642 | 0.349 | 0.080 | **1.85** | 44.74 | 1 709 648 |
| ours R53.01 26k (70 % area prune) | 26 000 | 18.706 | 0.648 | 0.338 | 0.080 | 1.85 | 44.26 | 2 564 473 |
| ours F7 26k (CSEF70 recovery) | 26 000 | 18.706 | 0.648 | 0.338 | 0.079 | 1.85 | 44.20 | 2 564 473 |
| **ours F75 26k (adaptive + sparse + LPIPS λ = 0.00025)** | **26 000** | **18.712** | **0.648** | **0.338** | **0.079** | **1.85** | **43.95** | **2 564 473** |
| ours F74 26k (adaptive + sparse + LPIPS λ = 0.0001) | 26 000 | 18.711 | 0.648 | 0.338 | 0.079 | 1.85 | 44.07 | 2 564 473 |
| ours R56 28k (R53 continuation, rejected) | 28 000 | 18.36 | 0.624 | 0.367 | n/a | n/a | n/a | 2 564 473 |
| ours R43 30k (no decay, rejected) | 30 000 | 16.25 | 0.511 | 0.477 | 0.194 | 3.02 | 43.71 | 782 982 |

F75 is the accepted parking single-scene headline — it improves over F7 by ΔPSNR +0.0058, ΔLPIPS −0.000773, ΔAbsRel −0.000531, ΔDepth MAE −0.002774, ΔNormal angle −0.2495° at identical topology. F76 multi-scene replication of the F75 fixed policy is in flight.

---

## Counterfactual Surface Evidence Field (CSEF)

Every candidate edit consults a per-face / per-vertex / per-region field with the following channels:

```text
CSEF(x, n, region) = {
  positive_surface_evidence,        # multi-view visibility, COLMAP support, normal agreement, prior support
  negative_free_space_evidence,     # camera rays / sparse points indicating "nothing here"
  explanation_debt,                 # residual pixels, boundary holes, missing depth, unmatched semantics
  prior_support,                    # plane / object / symmetry / smoothness priors
  topology_cost,                    # ∆triangles, ∆memory, ∆render cost
  uncertainty                       # low evidence, posterior variance, poor coverage
}
```

The edit objective is

```text
maximize  evidence_debt_reduction(edit)
        + render_quality_gain(edit)
        + geometry_consistency_gain(edit)
        − free_space_violation(edit)
        − hallucination_risk(edit)
        − topology_cost(edit)
```

subject to render, sparse-depth, normal-proxy, free-space, topology, and changed-pixel certificates all clearing. CSEF is the proposal source; the certificates are the dispositional gate. R17–R26 evidence shows that **the gate works as intended** — every edit type is gate-safe and reversible. The remaining open problem is the **proposal score**: it is not yet predictive enough of post-recovery render gain to outperform a matched baseline+sparse control.

## Reversible edit calculus

Seven first-class operations, all backed by `snapshot → apply → verify → keep | rollback`:

| op | role | current empirical state |
|---|---|---|
| `protect` | preserve supported geometry from later edits | works |
| `delete / prune` | remove unsupported floaters and redundant topology | gate-safe; PRISM line is the named baseline |
| `collapse / merge` | reduce topology while preserving supported surfaces | implemented; not the source of current Pareto win |
| `snap / deform` | correct dents, rough patches, plane / wall misalignment | R17–R21: gate-safe but recovery-quality fail / mixed |
| `split / subdivide` | allocate topology where the mesh under-explains evidence | implemented; not yet load-bearing |
| `fill / patch` | repair small holes and certified giant ground voids | R22 / R26: gate-safe; medium / full **fail** vs baseline+sparse |
| `appearance reset / recovery` | restore radiance after geometry repair | R11 / R44 sparse-decay: load-bearing |

Giant-hole policy distinguishes **observed**, **prior-supported**, and **unknown unobserved** voids: the third is rejected in normal mode and only proposed under an explicit `--allow_prior_only_fill` diagnostic flag — and even then it is labelled `prior_only_flag=true` and excluded from headline metrics.

---

## Validated recovery recipes

Three recipes are validated. **Recipe A — CSEF-family validation-budget (F49)** is the cross-scene headline: 5 / 5 scenes beat clean-long on render and sparse-depth metrics. **Recipe B — adaptive policy with tiny LPIPS (F75)** is the strongest single parking row and chooses the prune fraction from the checkpoint instead of from a hand-set table. **Recipe C — sparse-depth low-λ recovery (R44)** remains the cross-scene base recipe and the very-low-topology Pareto point but loses on render against clean 22k.

### Recipe A — CSEF-family + sparse-depth validation-budget recovery (F49, multi-scene headline)

Three steps: train a strong clean long mesh, run the CSEF-family compaction selector with a per-scene validation-budget prune ratio (parking 50 %, bonsai 50 %, courtyard 50 %, room 20 %, counter 20 %), recover with strict topology freeze + sparse-COLMAP-depth + (only on bonsai) tiny LPIPS λ = 0.005.

```bash
# 1) Clean long checkpoint (re-use whatever long Mesh Splatting checkpoint you have).
python train.py -s <scene> -m outputs/clean_long --eval --iterations 22000

# 2) CSEF-family compaction. The selector reads checkpoint evidence and emits the
#    selected face indices; --prune_fraction is the validation-budget knob. Use
#    0.50 for parking / bonsai / courtyard, 0.20 for room / counter, 0.40 for the
#    counter-fast variant. (F45 audit: do NOT claim a single fixed fraction.)
python scripts/car_model/meshsplatopt_select_compaction_candidates.py \
    --checkpoint outputs/clean_long/point_cloud/iteration_22000/point_cloud_state_dict.pt \
    --policy csef_low_evidence_boundary_protected \
    --prune_fraction 0.50 \
    --output_dir outputs/compact50

python scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py \
    --source_model outputs/clean_long \
    --selected_faces_path outputs/compact50/selected_faces.npy \
    --output_model outputs/compact50/model

# 3) Strict fixed-topology recovery 22000 → 26000 with sparse-COLMAP-depth.
#    For bonsai add --lambda_lpips_loss 0.005 to the train_extra_args.
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
python scripts/car_model/meshsplatopt_run_teacher_recovery.py \
    --model_path  outputs/compact50/model \
    --output_dir  outputs/carnet/meshsplatopt/<run_name> \
    --load_iteration 22000 --iterations 4000 \
    --train_extra_args "--freeze_topology_updates --skip_restricted_delaunay \
       --enable_sparse_colmap_depth_loss \
       --lambda_sparse_colmap_depth 0.005 \
       --sparse_colmap_depth_start_iter 22000 \
       --sparse_colmap_depth_warmup_iters 50 \
       --sparse_colmap_depth_min_matches 16 \
       --sparse_colmap_depth_sample_mode mixed_low_error \
       --sparse_colmap_depth_low_error_fraction 0.50 \
       --sparse_colmap_depth_enable_in_final_finetune"

# 4) Independent paper-facing eval.
python render.py  -m outputs/carnet/meshsplatopt/<run_name>/recovery_model
python metrics.py -m outputs/carnet/meshsplatopt/<run_name>/recovery_model
python evaluate_geometry_colmap.py -s <scene> \
    -m outputs/carnet/meshsplatopt/<run_name>/recovery_model --iteration 26000 --eval \
    --output outputs/carnet/meshsplatopt/<run_name>/recovery_model/geometry_eval_colmap/iter_26000.json
```

### Recipe A variant — area-only compaction (R53, no CSEF)

The same flow as Recipe A but with the area-only selector. It is the area baseline of the F12 / F49 multi-scene table and corresponds exactly to the parking R53.01 / R55 / R48 rows. Replace step 2 in Recipe A with:

```bash
python scripts/car_model/meshprior_apply_topology_control_ablation.py \
    --source_model outputs/clean_long \
    --source_checkpoint outputs/clean_long/point_cloud/iteration_22000/point_cloud_state_dict.pt \
    --output_model outputs/area70/model \
    --prune_fraction 0.70
```

`--prune_fraction` knob (parking): 0.65 → R55 LPIPS-best / 0.70 → R53.01 all-metric / 0.80 → R48 most-compact / 0.90 → R47 prune90 rejected (PSNR drops 2 dB). Continuation past 26 k is rejected (R56 at 28 k loses ~0.35 dB PSNR; R49 / R50 at 30 k also lose).

### Recipe B — adaptive CSEF policy with tiny LPIPS recovery (F75, parking single-scene strongest)

Same shell as Recipe A, but the selector chooses the prune fraction and ranking from checkpoint evidence (no manual `--prune_fraction`), and the recovery adds a tiny LPIPS term layered on sparse-depth.

```bash
# Adaptive selector: chooses fraction (parking → 70 %) and ranks faces by
# area / local-redundancy primary, render evidence as risk-only.
python scripts/car_model/meshsplatopt_select_compaction_candidates.py \
    --checkpoint outputs/clean_long/point_cloud/iteration_22000/point_cloud_state_dict.pt \
    --policy csef_adaptive_policy \
    --output_dir outputs/adaptive_compact

python scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py \
    --source_model outputs/clean_long \
    --selected_faces_path outputs/adaptive_compact/selected_faces.npy \
    --output_model outputs/adaptive_compact/model

python scripts/car_model/meshsplatopt_run_teacher_recovery.py \
    --model_path  outputs/adaptive_compact/model \
    --output_dir  outputs/carnet/meshsplatopt/<run_name> \
    --load_iteration 22000 --iterations 4000 \
    --train_extra_args "--freeze_topology_updates --skip_restricted_delaunay \
       --enable_sparse_colmap_depth_loss \
       --lambda_sparse_colmap_depth 0.001 \
       --sparse_colmap_depth_start_iter 22000 \
       --sparse_colmap_depth_warmup_iters 50 \
       --sparse_colmap_depth_min_matches 16 \
       --sparse_colmap_depth_sample_mode mixed_low_error \
       --sparse_colmap_depth_low_error_fraction 0.50 \
       --sparse_colmap_depth_enable_in_final_finetune \
       --lambda_lpips_loss 0.00025"
```

`--lambda_lpips_loss 0.0001` (F74) is the more conservative robustness neighbour; values above ~0.001 (F71 / R51 / R52) are rejected for hurting depth.

### Recipe C — sparse-depth low-λ + decay (R44, cross-scene base)

Use this when you want the **lowest-topology** parking point (782 982 triangles) or for cross-scene recovery on `courtyard` and `bonsai`. On parking it loses on render vs. clean 22 k and is now the normal-proxy / topology Pareto column, not the headline.

```bash
python scripts/car_model/meshsplatopt_run_teacher_recovery.py \
    --model_path <low_topology_checkpoint_dir> \
    --edit_json   <accepted_edits.json> \
    --output_dir  outputs/carnet/meshsplatopt/<run_name> \
    --load_iteration 16000 --iterations 6000 \
    --train_extra_args " \
       --densify_until_iter 16000 --skip_restricted_delaunay \
       --enable_sparse_colmap_depth_loss \
       --lambda_sparse_colmap_depth 0.001 \
       --sparse_colmap_depth_start_iter 16000 \
       --sparse_colmap_depth_warmup_iters 50 \
       --sparse_colmap_depth_min_matches 16 \
       --sparse_colmap_depth_sample_mode mixed_low_error \
       --sparse_colmap_depth_low_error_fraction 0.50 \
       --sparse_colmap_depth_decay_start_iter 16000 \
       --sparse_colmap_depth_decay_end_iter   20000 \
       --sparse_colmap_depth_decay_final_mult 0.0 \
       --sparse_colmap_depth_enable_in_final_finetune"
```

For courtyard the validated regime is fraction `0.625`, λ `0.002`, `7k → 20k` with decay starting at 7k. For bonsai the validated regime is fraction `0.50`, λ `0.002`, `2k → 7k` (longer continuation has not yet been validated).

### Rejected directions (do not retry without new evidence)

- **Fixed CSEF50 universally (F45)** — borderline / mixed on bonsai / room and a fail on counter; method must be validation-budget per scene.
- **Heavy LPIPS recovery loss** — R51 (λ = 0.02) / R52 (λ = 0.05) on R48; F71 / F72 / F73 on the adaptive policy. Only λ ≤ 0.001 (best F75 at 0.00025, F74 at 0.0001, bonsai F49 at 0.005) keeps the depth wins.
- **Teacher-render distillation from R44** (R45 λ 0.5 / 1.0; R46 counterfactual mask) — all worsen render.
- **Adaptive selector ranked by render-only evidence** (F57–F67) — drives the wrong fraction / ranking. F68 area / redundancy primary with render-as-risk is the corrected form.
- **Long continuation past validated budget** — F34 parking 26k → 30k regresses; R56 R53 26k → 28k loses ~0.35 dB PSNR; R49 / R50 30k continuation also lose.
- **Posthoc QEM on parking** — F37 fast-QEM matched topology collapses render; F25 Open3D QEM did not reach the parking topology target.
- **Random same-count compaction** (F16 counter / F19 room / F26 bonsai) — clearly worse than area / CSEF / QEM at matched topology.
- **No-freeze controls** (F27 / F35 / F36 / F18 / F24 across all five final scenes; R25 on parking) — strict `--freeze_topology_updates --skip_restricted_delaunay` is required.
- **Broad strict-gate dominance (F43)** — bonsai 7000-iter strict gate strictly worse than no-gate. F44 calibrated thresholds repair the bonsai mechanism, but F50 calibrated-gate parking does not reproduce — the gate claim is render-quality + unsafe-edit rejection, not universal geometry dominance.
- **Edit primitives as full-budget winners** — R28 grid-fill rejected; R22 / R26 fan / grid `FILL_PATCH` lose at full budget vs matched baseline+sparse. R17 area / R18 / R19 / R20 / R21 snap variants are too small.
- **Alternative sparse-depth loss spaces** (R29 relative / log / inverse) — original metric-depth Smooth-L1 wins.
- **Unfrozen post-edit densification** (R25) — grew parking to 5.89 M triangles and still lost render.

### Reproducible paper-facing tables

```bash
# Sparse-depth recovery line (R-stage)
python scripts/car_model/meshsplatopt_collect_sparse_recovery_results.py
# → outputs/carnet/meshsplatopt/sparse_recovery_tables/{json,csv,md}

# Adaptive-policy line (F68–F75)
python scripts/car_model/final_collect_stageF68_F73_adaptive_policy.py
# → outputs/carnet/meshsplatopt/final_stageF75_adaptive_policy_evidence/adaptive_policy_results.{json,md}

# Multi-scene F12 / F49 final package (figures + tables under)
ls outputs/carnet/meshsplatopt/final_paper_assets/
ls outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/
```

## Repository layout (MeshSplatOpt additions)

```text
ss3dm_prior/meshsplatopt/        core method
  csef_types.py / csef_builder.py
  defect_types.py / defect_mining.py
  edit_types.py / edit_apply.py / edit_snapshot.py
  topology_baselines.py
  snap_proposals.py
  hole_fill.py / ground_void_fill.py
  object_prior_repair.py
  counterfactual_edit_gate.py
  teacher_recovery.py
  edit_portfolio.py / repair_state_machine.py
  synthetic_damage.py
  checkpoint_adapter.py            # nearest-face init for FILL_PATCH

scripts/car_model/                 CLI entry points
  meshsplatopt_build_csef.py
  meshsplatopt_mine_defects.py
  meshsplatopt_make_snap_proposals.py
  meshsplatopt_make_fill_proposals.py
  meshsplatopt_select_checkpoint_local_snap_edit.py        # R17 area / R17.06 risk-filtered
  meshsplatopt_select_checkpoint_residual_snap_edit.py     # R18 / R19
  meshsplatopt_expand_snap_edit_to_patch.py                # R21
  meshsplatopt_select_checkpoint_boundary_fill_edit.py     # R22 / R23
  meshsplatopt_expand_boundary_fill_to_grid.py             # R26
  meshsplatopt_select_compaction_candidates.py             # F-stage CSEF / area / adaptive selector
  meshsplatopt_apply_compaction_to_checkpoint.py           # F-stage compaction applier (CSEF / adaptive)
  meshprior_apply_topology_control_ablation.py             # area-only prune (R47 / R53 / R55)
  meshsplatopt_validate_edit_counterfactual.py
  meshsplatopt_run_teacher_recovery.py                     # accepts --train_extra_args (sparse-depth / LPIPS)
  meshsplatopt_run_repair_state_machine.py
  meshsplatopt_collect_sparse_recovery_results.py          # R-line paper table
  final_collect_stageF68_F73_adaptive_policy.py            # F-line adaptive-policy table
  final_make_paper_assets.py                               # multi-scene qualitative assets

docs/car_model/                    per-stage design / implementation / smoke / report files
docs/NeurIPSRepairPrompts.md       full R0–R17 stage spec (drafted before R44; F-stage spec embedded in research log)
outputs/carnet/meshsplatopt/                                       per-stage artefacts
outputs/carnet/meshsplatopt/sparse_recovery_tables/                 R-line paper-facing JSON / CSV / Markdown
outputs/carnet/meshsplatopt/best_clean_long_vs_method_long/         R44 clean-baseline correction
outputs/carnet/meshsplatopt/final_stageF40_clean_vs_method_assets/  F-line 5-scene qualitative montage + manifest
outputs/carnet/meshsplatopt/final_paper_assets/                     F-line paper figures (cross-scene montage, triangle-count bar, method diagram)
outputs/carnet/meshsplatopt/final_stageF75_adaptive_policy_evidence/ F75 adaptive policy results
```

The PRISM safety stack (`utils/prism_*`, `ss3dm_prior/meshprior/*`) is preserved and re-used as the rollback / counterfactual primitives — Stage 35 PRISM remains a named baseline rather than the final method.

---

## Operating rules (non-negotiable)

These come from `docs/NeurIPSRepairPrompts.md` §3 and are enforced per-stage:

1. Work one stage at a time; do not proceed after a failed hard gate.
2. Never mix training-time metrics with independent `render.py + metrics.py` metrics.
3. Never use ground truth to choose proposals at inference time.
4. Every edit type must support rollback; the gate must rollback automatically on reject.
5. Every accepted repair must have an audit trail: proposal JSON, before / after snapshots, gate report, W&B link if trained, independent metrics if rendered.
6. Old PRISM stages remain named baselines and are not overwritten.
7. All training runs use W&B online (`WANDB_PROJECT=spcarnet_meshprior`).
8. Every stage writes design, implementation, smoke, and research-log entries.
9. **Negative results are first-class.** Failed gates and `*_FAIL` / `*_REJECTED` decisions stay in the research log and the README; they are the discipline that prevents over-claiming.

---

## Mesh-Splatting foundation

MeshSplatOpt builds on the differentiable opaque-mesh renderer from [MeshSplatting](https://meshsplatting.github.io). The original training, rendering, and evaluation entry points are unchanged on this branch and remain the way to produce input checkpoints.

### Install

```bash
git clone https://github.com/meshsplatting/mesh-splatting --recursive
cd mesh-splatting
micromamba create -n mesh_splatting python=3.11
micromamba activate mesh_splatting
micromamba install nvidia/label/cuda-12.6.0::cuda
pip install torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
bash compile.sh
( cd submodules/simple-knn && pip install . --no-build-isolation )
( cd submodules/effrdel    && pip install -e . )
```

Optional fused-SSIM speed-up:

```bash
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
```

### Train / render / evaluate

```bash
python train.py -s <scene> -m <output_model_path> --eval                      # outdoor
python train.py -s <scene> -m <output_model_path> --indoor --eval             # indoor
python full_eval.py --mipnerf360 <path_to_mipnerf360> --output_path <save>    # MipNeRF-360 sweep
python render.py  -m <model>
python metrics.py -m <model>
python evaluate_geometry_colmap.py -s <scene> -m <model> --iteration <iter> --eval \
    --output <model>/geometry_eval_colmap/iter_<iter>.json                    # COLMAP sparse depth + PCA-normal proxy
```

Optional explicit train/test split (strict out-of-train holdout):

```bash
python create_colmap_outoftrain_split.py -s <scene> -o <scene>/sparse/0/split_outoftrain_v1.json --test_ratio 0.12 --gap_ratio 0.03
python train.py -s <scene> -m <model> --eval --split_strategy file --split_file <split_json>
```

Depth and normal supervision hooks (`extract_normals.py`, `Depth-Anything-V2`, `utils/make_depth_scale.py`) and the SAM-based object-extraction pipeline (`segmentation/*`) are unchanged from the upstream repo.

### Local recommended scenes

```text
/data2/peilincai/mesh_datasets/mipnerf360/{bonsai,flowers}     # COLMAP-compatible
```

---

## Related work (positioning, not contribution)

The full novelty-threat matrix is in `docs/car_model/meshsplatopt_stageR2_related_work_matrix.md`.

- **Mesh / triangle splatting and surface-aligned 3DGS:** MeshSplatting, Triangle Splatting, 2D Triangle Splatting, SuGaR, MeshGS, 2DGS, DN-Splatter.
- **Pruned / compact 3DGS:** LightGaussian, Compact3DGS, EAGLES, Mini-Splatting, EfficientGS, RadSplat, LP-3DGS, MaskGaussian, PUP 3D-GS, GaussianPOP, GaussianSpa, SafeguardGS.
- **Classical mesh processing:** QEM edge collapse, constrained Delaunay triangulation, screened Poisson reconstruction, isotropic / adaptive remeshing, Laplacian / ARAP deformation, hole filling.

These are baselines, not contributions. The intended differentiator was the **unified CSEF + reversible edit calculus + counterfactual certificate** triple. The present empirical state is that the *certification* part is real and load-bearing, but the *edit-quality* part has not yet outperformed a matched sparse-depth recovery without the edit. The honest current contribution is **(i) a counterfactually safe edit / rollback infrastructure for Mesh Splatting checkpoints, and (ii) a low-λ sparse-depth recovery recipe with confidence-weighted COLMAP correspondence sampling and a decay window**, evaluated as a **topology / normal Pareto** point against the strongest clean long-horizon baseline.

---

## Citing

The MeshSplatOpt branch is ongoing work; please cite the MeshSplatting foundation paper.

```bibtex
@article{Held2025MeshSplatting,
  title  = {MeshSplatting: Differentiable Rendering with Opaque Meshes},
  author = {Held, Jan and Son, Sanghyun and Vandeghen, Renaud and Rebain, Daniel and Gadelha, Matheus and Zhou, Yi and Cioppa, Anthony and G Lin, Ming C. and Van Droogenbroeck, Marc and Tagliasacchi, Andrea},
  journal= {arXiv:2512.06818},
  year   = {2025}
}
```

```bibtex
@article{Held2025Triangle,
  title  = {Triangle Splatting for Real-Time Radiance Field Rendering},
  author = {Held, Jan and Vandeghen, Renaud and Deliege, Adrien and Hamdi, Abdullah and Cioppa, Anthony and Giancola, Silvio and Vedaldi, Andrea and Ghanem, Bernard and Tagliasacchi, Andrea and Van Droogenbroeck, Marc},
  journal= {arXiv},
  year   = {2025}
}
```

```bibtex
@InProceedings{held20243d,
  title    = {3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes},
  author   = {Held, Jan and Vandeghen, Renaud and Hamdi, Abdullah and Deliege, Adrien and Cioppa, Anthony and Giancola, Silvio and Vedaldi, Andrea and Ghanem, Bernard and Van Droogenbroeck, Marc},
  booktitle= {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year     = {2025}
}
```

## Acknowledgements

J. Held is funded by the F.R.S.-FNRS. The present research benefited from computational resources made available on Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region under the grant agreement n°1910247. We thank Bernhard Kerbl and George Kopanas for helpful feedback and proofreading on the original MeshSplatting paper.

---

## Documentation maintenance

This README is mirrored in two languages:

- [`README.md`](README.md) — English (canonical)
- [`README.zh.md`](README.zh.md) — 中文

**Whenever this file is edited, the Chinese mirror must be updated in the same change**, and vice versa. Both files share the same section structure so a section-by-section diff is enough. A new R-stage entry must be reflected in: (i) the project-status tables, (ii) the "Where the method actually stands today" lists, (iii) the headline result table if it touches the Pareto frontier, and (iv) the validated recipe blocks if it adds a new flag or recipe.
