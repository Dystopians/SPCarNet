# GEMS — NEGATIVE_RESULTS.md (first-class, §6)

Generated 2026-07-03 for the Stage-2 §8 evidence pack. Every DONE-FAIL cell,
demoted axis, and >1-GPU-hour below-floor diagnostic, each with its mechanism
and durable evidence path. Numbers quoted below are taken verbatim from the
cited LEDGER goals / eval rows (they are re-derivable from the cited
metrics.json via `RESULTS/aggregate/` tables — nothing here is a new number).

Format: **what failed — verdict — mechanism — evidence.**

## 1. E1 budget engine as pre-registered (and E1′ as amended) — FAIL
- Verdict: E1 FAIL under both original and amended criteria (no
  re-thresholding); the *mechanism* (evidence-prune + features-only FT) was
  validated separately on real scenes.
- Mechanism: criterion (b) presumed lossy pruning (measured prune-only is
  near-lossless at B50: garden −0.011 dB); criterion (a) fails on toy only —
  a train-coverage selection effect, not optimization failure. E1′(i) failed
  because safe FT almost fully repaints random-prune damage on a 33-view
  scene (courtyard importance−random +0.13, CI incl. 0).
- Evidence: `KILL_REPORT.md`; LEDGER GOAL#005/#008;
  `analysis/e1_summary.{md,json}`; T1/T2/T6; eval rows `*_e1b`, `*_e1v2`.

## 2. Default and lr×0.1 fine-tuning on converged checkpoints — FAIL (destroyer)
- Verdict: any resumed all-parameter FT damages the model (garden B50 −2.92 dB
  vs no-FT; lr×0.1 WORSE, −4.13 vs clean; systematic attractor).
- Mechanism: near convergence, Adam-normalized POSITION updates drift along a
  value-destroying direction (channel isolation: vertices-only = destroyer,
  features-only = mild repairer, weights-only = innocent); the trainer's own
  end phase sits on the same slope.
- Evidence: LEDGER GOAL#005 root-cause chain; eval rows
  `{garden,toy_parking}_B{50,25}_importance_ft_e1{b,v1}`; T6 FT-channel block;
  `F7_ablations`.

## 3. E2 geometry losses, attempt 1 (free-space hinge + depth consistency) — FAIL
- Verdict: kill condition met on both dev scenes (toy PSNR guard violated,
  g-improvements ≪ 30% bar).
- Mechanism: geometry signal too weak vs mobile-position photometric drift —
  re-enabling positions re-admits the E1 drift mechanism.
- Evidence: LEDGER GOAL#006; eval rows `m3_e2_{toy_parking,courtyard}_B50_v1`;
  T6 geometry-mechanism block.

## 4. E2 variant 2, gradient-routed geometry coupling — FAIL
- Verdict: PSNR guard blown on both scenes (toy −1.21, courtyard −0.75) with
  sub-bar g-movements.
- Mechanism: positions freed from photometric counter-pressure evacuate junk
  AND load-bearing structure; appearance breaks.
- Evidence: LEDGER GOAL#007; eval rows `m3v1_*_B50_v1`; T6.

## 5. E2 variant 3, evidence-based floater deletion (e2v3) — FAIL (high-value falsification)
- Verdict: g3 → −100% (bar met) but PSNR guard violated on BOTH scenes; E2
  axis closed 3/3 and demoted to evaluation-only.
- Mechanism: **selection effect** — support≤1 triangles are precisely what
  test-like viewpoints see (toy view 00035 −8.07 dB from a genuine
  train-coverage gap). Also: floaters do not drive g1 free-space violations
  (g1 moved ≤1.2%).
- Evidence: LEDGER GOAL#008 (pre-registration C + result); eval rows
  `e2v3_*_v1`, `e2v3_*_prunedNoFT_diag`; E9 taxonomy family A.

## 6. E2R opacity-floor release (B6R) — FAIL on the joint pre-registered bar
- Verdict: toy PSNR guard decisively violated (−0.383 CI[−0.586,−0.201]);
  courtyard is a REAL bounded positive (LPIPS better CI excl. 0, g1 −28.5%,
  g3 −34/−41%) but D7 (one rule, all dev scenes) makes the joint bar FAIL.
- Mechanism: fading reproduces the e2v3 selection effect (pinholes on
  low-train-support content) + softened surfaces; also a runtime surprise —
  semi-transparency defeats early ray termination (toy FPS 75.5→39.6).
- Evidence: LEDGER GOAL#R-04; eval rows `e2r_{toy_parking,courtyard}_B50_v1`;
  probe artifacts `analysis/e2r_probe{,_courtyard}/`; T1 B6R row, T3/T5/T6.

## 7. E3 teacher distillation — SUNSET (3 consecutive below-floor)
- Verdict: distill−control positive with CI excl. 0 on garden in ALL three
  variants (+0.039 / +0.051 / +0.125 dB) — a REAL channel, always below the
  +0.15 dB floor; toy unstable; mechanism demoted to diagnostic-only.
- Mechanism: the teacher residual is VIEW-CONDITIONED; view-independent
  carriers can't hold it (recovered fraction ≈5–16% of teacher headroom,
  quantitatively consistent with the v1xx residual-cosine ≈ 0.21 post-mortem).
  Variant 2's SH channel overfits 72 views (toy control LPIPS 0.11→0.21).
- Evidence: LEDGER GOAL#007; eval rows `e3_*`, `e3v1_*`, `e3v2_*`; T6 teacher
  block; `F7_ablations`; >1 GPUh consumed across variants (bake + FT + evals).

## 8. 26k-sourcing probe — REFUTED
- Verdict: sourcing compaction from the 26k checkpoint is not better
  (−0.020 CI[−0.030,−0.010] vs 30k-sourced).
- Mechanism: the features-only FT itself already performs the end-phase-damage
  repair; source choice adds nothing.
- Evidence: LEDGER GOAL#008; eval row `garden_B50_importance_ft_e26src`; T6.

## 9. R3.a TSDF fusion route (ii) — HYPOTHESIS FALSIFIED (citable)
- Verdict: TSDF moves the safety-critical d1 false-free rate the WRONG way on
  every cell (worse 26–41% rel.; all paired CIs on the wrong side).
- Mechanism: grazing-ray bias — projective TSDF at 0.10 m voxel centers under
  oblique viewing votes near-surface voxels "free"; voxelization marks any
  surface-containing cell.
- Evidence: LEDGER GOAL#R-02; `analysis/r3a_occupancy_routes/summary.json`
  (calibration table + verdict); `T5b_r3_trilogy.md`.

## 10. R3.c raw-grid closed-loop consumability — FAIL for BOTH routes
- Verdict: at 0.10 m voxels + 1.0 m inflation, raw occupancy grids are
  unusable for footprint planning — route (i): 93–100% spurious infeasibility
  on courtyard/toy (false-occupied blankets); route (ii): plans on courtyard
  but collides (10.7/100 vs GTREF floor 2.0). Preservation P1 PASSED
  (clean↔B50 planner outcomes identical).
- Mechanism: clean-model junk clutter (route i) / false-free surfaces
  (route ii); a checkpoint-geometry problem, invariant to compaction.
- Evidence: LEDGER GOAL#R-03; `analysis/r3c_planner/summary.json` + panels;
  `T5b_r3_trilogy.md`.

## 11. R3.b certified structural sub-mesh — FALSIFIED 0/4 bar cells (citable)
- Verdict: certification trades spurious infeasibility for REAL collisions
  (courtyard 42/100 found but 16.7 coll/100 ≫ cap 3.0; d1 false-free worse
  +27–31% rel.); calibrate-once/test-elsewhere transferred honestly and
  failed on the real scene. POSITIVE side finding: certified kept-sets are
  EXACTLY identical clean↔B50 on both scenes.
- Mechanism: multi-view-supported junk and load-bearing surface are not
  separable by support+depth-consistency evidence (same selection-effect
  family as e2v3/E2R-toy); on courtyard it sheds real walls (lethal fraction
  falls BELOW GT).
- Evidence: LEDGER GOAL#R-06; `analysis/r3b_submesh/{summary.json,
  calibration_table.json, frozen_params.json}` + panels; `T5b_r3_trilogy.md`.
- Consequence (frozen fix-target for any future geometry fix, quantified):
  courtyard ≥30/100 found at ≤3.0 coll/100 simultaneously.

## 12. Baseline-trainer end-phase decline — negative finding about the BASELINE
- Verdict: clean@26k beats clean@30k (garden +0.32); features-only 26k→30k
  continuation (B0′) beats clean@30k 9/9, yet clean@26k STILL beats B0′ 9/9 —
  the final-phase damage is not fully position-drift.
- Mechanism: optimizer drift on converged models (see §2); partially
  repairable by freezing positions/weights.
- Evidence: LEDGER GOAL#005 (garden 26k diag), GOAL#R-01 (9-scene anchors);
  T2 rows B0-26k / B0′; eval rows `*_clean26k_v1`, `*_cleanfixed30k_v1`.
- Claims handling: NON-CLAIM (trainer property, not a GEMS contribution).

## 13. Per-scene bounding failures inside otherwise-passing cells
- flowers B50: fails the −0.10 floor (B5 −0.147; B4 −0.238) — hard low-texture
  scene; prune damage not FT-recoverable. Evidence: T1/T2; E9 family C.
- ss3dm_town06 B50: −0.447 (least over-parameterized town). Evidence: T1/T2
  (S-GEO), LEDGER GOAL#R-05; E9 family C.
- B25: floor 3/9 vs B0 (2/9 vs B0′); B12.5: 0/9 — graceful degradation only.
  Evidence: T1; LEDGER GOAL#R-00/#010.
- toy_parking B50: −0.52 vs clean persists across every mechanism tried
  (selection effect; adversarial 62× over-parameterized synthetic scene).
  Evidence: T1/T2, LEDGER GOAL#005/#008; E9 family A.

## Cross-reference
The full failure taxonomy with panels and one-paragraph diagnoses (13 cases,
5 mechanism families) is at
`/data/peilincai/gems_stage1/analysis/e9_failure_taxonomy/TAXONOMY.md`
(LEDGER GOAL#R-09). KILL_REPORT.md holds the E1/M2 kill accounting.
