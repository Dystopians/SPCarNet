# GEMS — EXPERIMENT_REPORT.md (Stage 2/3, FINAL EVIDENCE PACK v4)

Started 2026-07-03 (Stage-1R R4); Stage3 closure completed 2026-07-04
(GOAL#C-01/#C-02 plus pack v4 regeneration).
Every table/figure referenced below is script-generated from `metrics.json`
/ durable analysis artifacts (Stage-2 §6: no hand-typed numbers; generators
in `tools/gems/report/`; one-command regeneration + from-scratch T1
verification in `RESULTS/REPRO_PACK/`). Claim boundaries: `CLAIMS.md` v1.3
applies verbatim — the NON-CLAIMS block is reproduced in §6 below.
Claim↔evidence mapping: `RESULTS/CLAIMS_EVIDENCE_MATRIX.md`.

## 1. Method summary

Evidence-guided triangle pruning under explicit budgets + drift-safe
(features-only) fine-tuning on MeshSplatting checkpoints (pipeline diagram:
`RESULTS/figures/F3_pipeline_draft.png` + caption). One train-view render
pass accumulates per-triangle evidence (pixels_total; D4-pure); prune to
budget B under a ±2% fair-budget rule; fine-tune ONLY features/SH (positions
and weights frozen — the sole validated safe channel; all-parameter FT
measurably destroys converged checkpoints). Output = a plain, smaller
checkpoint of the same format; no test-time components. Single-mouth
evaluation (`run_eval.py`, PROTOCOL v1.1.x); all deltas paired per-view
bootstrap, 10k resamples, seed 0, against dual anchors (clean@30k legacy;
clean-fixed@30k primary). The consumption-closure routes (occupancy routes /
planner / certified sub-mesh / three-state R3-FINAL) and the evidence-vs-error
study are ANALYSIS, not method (§7.5 framing).

## 2. Results index (final pointers)

**Tables** (`RESULTS/aggregate/`):
- **T1** `T1_main_pareto.{md,csv}` + `T1_per_scene_detail.csv` — main Pareto
  per suite × budget × {B5,B4,B2,B3,B5-iter,B6R}, dual anchors, win/iso/loss
  + iso-floor counts; R1/H1 context appendix. Headline structure: B50 iso
  8/9 vs B0 (4/9 vs B0′, garden strictly above both); B25 3/9; B12.5 0/9
  graceful; FT value grows as budget shrinks (B5>B4 9/9 at B25/B12.5);
  evidence-vs-random +5.23 dB mean at B12.5 (9/9 CI); B3 QEM −2.7..−3.4 dB
  below B5 at matched budget.
- **T2** `T2_rendering.{md,csv}` — per-scene rendering, dual-anchor CIs,
  26k context rows, B3/B6R rows, D-2 variant scenes (E3-REND).
- **T3** `T3_geometry.{md,csv}` — g1–g4 per scene, GT-CAL calibration row,
  VOID-aware (E2-GEO tables).
- **T4** `T4_efficiency.{md,csv}` + `fps_bench_halfres.json` — tris/disk/
  VRAM/FPS@2 res + measured prune+FT overhead; R1 efficiency context
  appendix (E4-EFF).
- **T5** `T5_downstream.{md,csv}` + `T5b_r3_trilogy.md` — d1/d2 per scene +
  R3.a/c/b plus Stage3 R3-FINAL verdicts script-extracted (E5-DOWN);
  N=500 extension in `analysis/e5_down_ext/` (quoted in F6).
- **T6** `T6_ablations.{md,csv}` — E6 axes: FT channel, schedule,
  evidence-vs-random, importance-DEFINITION family (GOAL#012), sourcing,
  teacher variants, geometry mechanisms incl. B6R-on-SS3DM (E6-ABL).
- **T7** `T7_robustness.md` — Stage3 W1 seed-1 retrain substitute and W2
  50% train-view-drop arm, both completed; residue waived by human ruling.

**Figures** (`RESULTS/figures/`):
- **F1** teaser draft (garden budget–quality–geometry) + caption.
- **F2** Pareto ×6 (PSNR/LPIPS × S-REND/S-GEO/S-DEV incl. D-2 variants;
  B3/B6R series included).
- **F3** pipeline diagram DRAFT + `F3_caption.txt`.
- **F4** qualitative grids ×7 (`figures/qual/`, §5 crop rule script-enforced,
  every invocation logged in `qual/manifest.json`).
- **F5** geometry maps: floater overlays are the third column of every F4
  grid (per-row banked g3_floaters.npz); before/after free-space and
  footprint maps in `analysis/r3b_submesh/panels/` + F6(d).
- **F6** downstream composite (occupancy confusion + N=500 planner table +
  maneuver panels) + `F6_caption.txt`.
- **F7** ablation deltas with CI whiskers (incl. importance-definition and
  B3-vs-B5 blocks).
- **F8** failure board (13 E9 cases, one-line captions; full dossier
  `analysis/e9_failure_taxonomy/TAXONOMY.md`).

**Documents:** `RESULTS/CLAIMS_EVIDENCE_MATRIX.md` ·
`RESULTS/NEGATIVE_RESULTS.md` · `RESULTS/HANDOFF.md` · `RESULTS/REPRO_PACK/`
(env spec, seeds, commit pins, one-command regeneration, from-scratch T1
byte-diff: PASS 2026-07-03, `verify_t1_result.txt`).

**Anchors:** clean@30k (legacy/deployed default), clean-fixed@30k (PRIMARY;
9/9 CI-wins; LEDGER #R-01), clean@26k (context; not compute-matched).

## 3. NEGATIVE_RESULTS (first-class; full file `RESULTS/NEGATIVE_RESULTS.md`)

1. E1/E1′ criteria FAIL (mechanism validated separately; KILL_REPORT.md).
2. Default/low-LR FT on converged checkpoints = destroyer (position drift).
3. E2 geometry losses ×2 FAIL; E2v3 floater deletion FAIL (selection
   effect); E2R joint bar FAIL (courtyard positive bounded); B6R-on-SS3DM
   generalization DONE-FAIL as pre-registered (GOAL#014).
4. E3 teacher distillation SUNSET (real sub-floor channel; view-conditioning
   cap ~5–16%).
5. 26k-sourcing refuted (−0.020 CI).
6. R3.a TSDF fusion falsified (false-free ↑26–41%; grazing-ray bias).
7. R3.c raw grids unusable closed-loop; R3.b certification falsified 0/4;
   Stage3 R3-FINAL three-state carving falsified 0/4 with 0/100 found on toy
   and courtyard clean/B50. N=500: courtyard route-i's only 2/500 plans
   BOTH collide; P-B2/P-B3 numeric letters falsified at better resolution
   (substance survives) — LEDGER #015-B.
8. Trainer end-phase decline (baseline property; 26k > fixed-30k 9/9).
9. Per-scene bounding failures: flowers B50, town06 B50, toy family B50
   (occl variant misses the P1 band by 0.053 dB via a −6.21 dB tail view —
   GOAL#016), B25/B12.5 floors.

## 4. RED-TEAM — "what a skeptical reviewer will say" (final answers)

- **A1 "your clean is decayed — re-anchor":** Done first, pre-emptively.
  PRIMARY anchor = clean-fixed@30k (9/9 CI-wins over clean@30k); ALL
  comparative tables dual-row; headline claims name their anchor (CLAIMS
  v1.1). Residual honesty: clean@26k > fixed@30k 9/9 (26k is not
  compute-matched; reported as a trainer property, NON-CLAIM).
- **A2 "importance ≈ random under FT":** measured and claimed where it
  lives. At B50-with-safe-FT on a 33-view scene the difference is
  inconclusive (courtyard +0.13 CI incl. 0 — NON-CLAIM covers it). At the
  aggressive budget the dominance is now fully evidenced: **B12.5
  evidence-vs-random +3.63..+8.19 dB, mean +5.23, 9/9 scenes CI excl. 0**
  (GOAL#011; T6/F7). FT value itself grows as budget shrinks (B5>B4 9/9 at
  B25 and B12.5). And the importance-DEFINITION axis is flat (±0.05 dB,
  GOAL#012): *having* render evidence is load-bearing, the specific column
  is not — an honest simplification, not a weakness.
- **A3 "modest ratios vs GS-compression literature":** now answered with a
  MEASURED context row instead of positioning prose (GOAL#017,
  `analysis/r1_3dgs_reference/r1_table.md`, quoted in T1/T4): at matched
  artifact storage on identical splits, 3DGS renders garden/bicycle/kitchen
  2.1–3.4 dB above GEMS B5@B50 at 3.1–4.4× FPS, and opacity-pruning 3DGS to
  the GEMS artifact size is essentially free. Reported plainly; exactly the
  trade the frozen NON-CLAIMS anticipated. GEMS's deliverables — triangle
  MESH artifact, g1–g4/downstream consumability, preservation-exactness,
  50%-reduction-at-iso — have no 3DGS-family equivalent artifact. R1 gates
  nothing.
- **A4 "'for parking' without geometry improvement":** answered by the R3
  trilogy + N=500: preservation-exactness in the strongest observable form
  (outcome sets identical clean↔B50, CIs [0,0]); falsified consumption
  routes WITH mechanisms; a frozen quantitative fix-target (courtyard
  ≥30/100 found at ≤3.0 coll/100); SS3DM driving-domain Pareto (B50 3/4) +
  20/20 preservation arms (#R-08); bounded courtyard geometry positive
  (B6R) whose generalization test was run and failed honestly (#014). The
  paper's downstream contribution is the measurement suite + preservation +
  the mechanism-level negatives; NON-CLAIMS state the boundary verbatim.
- **A5 "single seed everywhere; no sensitivity analysis (E7 unrun)":**
  acknowledged as the largest open Tier-2 gap (§7 waive draft W1). Partial
  mitigations on record, honestly labeled as partial: every headline delta
  is a paired per-view bootstrap CI (view-level resampling, not run-level);
  the seeded pipeline's end-to-end repeat noise floor was MEASURED at
  1.6e-5 dB (GOAL#012 degenerate-identical-prune row) — 4 orders of
  magnitude below the D3 floor, so run-to-run jitter cannot manufacture our
  effects; key findings replicate across 15 scenes / 3 suites and, for the
  toy bounds, across 3 generated family instances (D-2). What this does NOT
  cover: seed-sensitivity of the 30k BASE training itself.
- **A6 "the toy scenes are synthetic and adversarial — cherry-picked
  bounds?":** the direction of the concern is inverted — toy is adversarial
  AGAINST our method (62× over-parameterized; selection effect), and we
  kept it anyway as the named bound. D-2 (GOAL#016) shows the profile is a
  family property (d1 false-free 58.5–59.3% on 3/3 instances; B50 residual
  bracketed), not an instance accident.
- **A7 "your F-scores/chamfer on SS3DM are terrible (0.03–0.05 / 1.6–2.7 m)
  — is the mesh usable at all?":** yes, that is C2's point, stated as a
  measurement claim: photometric quality (22–23 dB, LPIPS improving under
  compaction) MASKS geometric unreliability (far-field collapse beyond
  ~40 m; g4 absolutes additionally GT-sampling-density-limited, T3 header).
  GEMS does not claim to fix this (NON-CLAIM); it claims to measure it,
  preserve it exactly under 2× compaction, and quantify why naive
  consumption fails.

## 5. Statistics & caveats (E10)

Paired per-view bootstrap everywhere (10k resamples, seed 0);
**multiple-comparisons caveat:** dozens of CIs are reported across this
program; individual borderline CIs (|effect| near floor) should be read with
Bonferroni-style skepticism — headline claims rest only on effects that are
large, replicated across scenes/suites, or mechanism-backed. Courtyard
rendering CIs are 5-view (underpowered by design; geometry/downstream carry
that scene). Per-scene breakdowns accompany every mean (§6: no
aggregate-only results). Reporting language: "improves/reduces" only where
CI excludes 0 AND the D3 floor is cleared.

## 6. Claim boundaries, NON-CLAIMS (verbatim), and purity audits

**NON-CLAIMS (verbatim from CLAIMS.md v1.2):**

> This is a per-scene optimization setting; the teacher is train-only and
> absent at test time; no claim of state-of-the-art novel-view quality
> versus the 3DGS family; no claim about high-speed driving; downstream
> results are proxies unless the closed-loop stretch item was executed.
> Additionally (Stage-One additions): no claim that GEMS improves geometry
> or downstream metrics vs clean; no claim that evidence-guided importance
> dominates random pruning under safe fine-tuning at moderate budgets on
> small scenes (courtyard B50: +0.13 CI incl. 0); the end-phase-decline
> finding (clean26k > clean30k) is reported as a property of the baseline
> trainer, not a GEMS contribution. (v1.2 addition:) no tested one-time
> occupancy-consumption route currently supports parking-grade closed-loop
> planning from these checkpoints; the blocker is baseline checkpoint
> geometry, not compaction (compaction is outcome-invariant).

**Purity audits (`tools/audit_test_path.py`) — outputs quoted verbatim from
the durable `audit_report.json` files (§7.2: included in the report):**

- `eval/stage2_entry_audit2/audit_report.json` — the **garden B5 compact
  artifact** (`models/garden_B50_importance_ft_e1v2@40000`, the C1′ headline
  artifact):
  `{"ok": true, "static": {"ok": true}, "dynamic": {"ok": true,
  "returncode": 0, "n_modules_loaded": 3443, "n_read_paths": 4165}}`
- `eval/garden_audit_v2/audit_report.json` — garden clean anchor
  (`official_clean30k/garden@30000`):
  `{"ok": true, "static": {"ok": true}, "dynamic": {"ok": true,
  "returncode": 0, "n_modules_loaded": 3443, "n_read_paths": 4166}}`
- `eval/toy_parking_audit/audit_report.json` — **toy clean artifact**
  (`models/toy_parking_clean30k@30000`):
  `{"ok": true, "static": {"ok": true}, "dynamic": {"ok": true,
  "returncode": 0, "n_modules_loaded": 3443, "n_read_paths": 4071}}`
- `eval/b3_audit_garden/audit_report.json` — garden B3 QEM artifact
  (`models/garden_B50_qem_ft_b3@40000`): `{"ok": true, "static": {"ok":
  true}, "dynamic": {"ok": true, "returncode": 0, "n_modules_loaded": 3443,
  "n_read_paths": 4165}}`
- `eval/d2_audit_v2_B5/audit_report.json` — D-2 variant B5 artifact
  (`models/toy_parking_v2_B50_importance_ft_s2@40000`): `{"ok": true,
  "static": {"ok": true}, "dynamic": {"ok": true, "returncode": 0,
  "n_modules_loaded": 3443, "n_read_paths": 4071}}`
- `eval/e2r_audit_courtyard/audit_report.json` — E2R courtyard artifact:
  `{"ok": true, "static": {"ok": true}, "dynamic": {"ok": true,
  "returncode": 0, "n_modules_loaded": 3446, "n_read_paths": 4028}}`

All GREEN (blocklist pattern covers evidence_lumigraph / teacher / ecsr /
phase-J / selector / calibrator / car_model). Honest history: the FIRST
Stage-2 entry audit (`eval/stage2_entry_audit`) was RED — a scope artifact
(train-side `teacher_factory.py` living inside the eval-audited
`tools/gems/`); fixed by relocation to `tools/gems_train/`, re-audited GREEN
(LEDGER, Stage-Two entry). Fresh-environment reproduction: Stage-1
fresh-clone repro PASS (Δ0.0005 dB) + pristine-submodule build check PASS
with bit-exact toy eval (GOAL#R-07).

## 7. Stage-2 §10 completion verification (Stage3 closure, 2026-07-04)

| § | Criterion | Verdict | Evidence / gap |
|---|---|---|---|
| 10.1a | Every Tier-1 cell DONE-PASS / DONE-FAIL / INFEASIBLE-with-note | **PASS** | Stage3 rulings execute the remaining scope decisions: W4/W4a/W5 granted as drafted; W6 superseded by R3-FINAL and closes as IMPOSSIBILITY; town06 INFEASIBLE note stands (#R-08). |
| 10.1b | ≥80% of Tier-2 cells resolved or explicitly waived | **PASS** | W1 substitute run completed and supports waiver; W2 50% train-view-drop arm completed and pose-noise/S-GEN residue waived by Stage3 ruling; W3 withdrawn after videos; W4/W4a/W5 granted. |
| 10.2 | All experiment types present in RESULTS/ + summarized | **PASS** | T7 contains measured seed and view-drop evidence. E8 pose-noise and S-GEN are waived with the W2 datum and explicit limitation language, not silently treated as run. |
| 10.3 | Paired bootstrap on every headline delta; §6 language | **PASS** | Universal CI discipline (E10 DONE); caveat §5; win/iso/loss per scene in T1; language rules enforced in table headers. |
| 10.4 | CLAIMS_EVIDENCE_MATRIX complete & consistent post-shrink | **PASS** | Regenerated at final assembly; no claim lacks evidence; every strong result has a claim-home or stated exclusion; residual scope caveats flagged, not hidden. |
| 10.5 | All §8 deliverables exist; script-reproducible; REPRO_PACK verified by regenerating T1 from scratch | **PASS** | Pack v4 regenerated: 234-row corpus, T1–T7, F1–F8, qual grids, videos, submission handoff, and REPRO_PACK; `verify_t1.sh` PASS 2026-07-04T07:48:36Z. |
| 10.6 | Final purity audit + fresh-env reproduction green on shipped artifacts | **PASS** | Prior green audits remain in force; Stage3 carving tool audited GREEN at `/data/peilincai/gems_stage1/eval/c02_purity_audit_fast/audit_report.json`; fresh-clone repro PASS + pristine-build bit-exact (#R-07). |
| 10.7 | EXPERIMENT_REPORT complete, honest, incl. negatives/limitations/boundaries | **PASS** | This document + NEGATIVE_RESULTS.md + CONSUMPTION_IMPOSSIBILITY.md + §6 verbatim NON-CLAIMS + §8 limitations. |

**Bottom line:** Stage3 removes the final contingencies. W1 seed substitute:
clean seed1-clean seed0 = +0.031 dB CI[-0.018,+0.090], and the B5 residual
delta shift = +0.008 dB CI[+0.000,+0.017], both inside the 0.15 dB support
rule. W2 half-train arm: the B50 residual worsens relative to full-view
garden by -0.030 dB CI[-0.047,-0.014], matching the pre-registered direction.
R3-FINAL fails hard and closes downstream as IMPOSSIBILITY x4 route families.
All §10 rows are PASS; zero PARTIAL rows remain.

### Stage3 human-ruling closure

- **W1:** cheap substitute run completed; full 3-seed x subset/loss-weight
  grid waived with measured support plus the GOAL#012 repeat floor.
- **W2:** garden 50% train-view-drop clean+B5@B50 arm completed; pose-noise
  and S-GEN residue waived with limitation language strengthened by T7.
- **W3:** withdrawn because videos were delivered.
- **W4/W4a/W5:** granted as drafted in Stage3 §1 and recorded in LEDGER.
- **W6:** superseded by R3-FINAL; because R3-FINAL fails, no SS3DM planner
  cells are launched.

## 8. Limitations

Per-scene optimization setting; toy_parking's 62× over-parameterization
makes it adversarial for junk-removal mechanisms (selection effect —
documented thrice, now replicated on 3 family instances); SS3DM g1/g4
absolutes are far-field/GT-sampling-limited (paired deltas are the valid
signal); town06 g4/d1/d2 infeasible under the 20 GB RAM bar; courtyard
z_band approximate; GT-scan unscanned voxels count free (collision lower
bounds); seed robustness is bounded by one garden seed-1 substitute rather
than a full 3-seed grid; robustness/generalization is bounded by one 50%
train-view-drop arm rather than pose-noise and unseen-type S-GEN trainings;
single-GPU-class FPS numbers (laptop bench waived); B0′ primary anchor exists
on S-REND only; no positive closed-loop planning claim survives R3-FINAL.
