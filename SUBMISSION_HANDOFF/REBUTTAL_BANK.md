# GEMS Submission Handoff — Rebuttal Bank

Generated 2026-07-04 for Stage3 closure.

## A1. Is this just tuning MeshSplatting?

No. The evidence pack separates representation edits, safe recovery, geometry
diagnostics, efficiency, and downstream consumers under a single-mouth
evaluation protocol. The strongest method part is not a per-scene parameter
scan; it is the frozen pipeline of evidence-based pruning plus topology-frozen,
features-only recovery. The failure analysis is also part of the contribution:
several plausible geometry/downstream mechanisms were falsified with durable
artifacts rather than hidden.

Pointers: `T1_main_pareto.md`, `T6_ablations.md`, `CLAIMS.md`,
`RESULTS/NEGATIVE_RESULTS.md`.

## A2. Why should reviewers trust the compactness result?

The pack reports both deployed clean@30k and repaired clean-fixed@30k anchors,
paired per-view CIs, per-scene win/iso/loss counts, random and QEM contrasts,
and efficiency metrics. The claims are bounded: B50 is the stable regime;
B25/B12.5 are reported as graceful degradation, not headline superiority.

Pointers: `T1_main_pareto.md`, `T1_per_scene_detail.csv`, `T2_rendering.md`,
`T4_efficiency.md`.

## A3. Why not compare against 3DGS or modern Gaussian compression?

The paper should not claim NVS SOTA. A context-only 3DGS reference exists and
shows 3DGS is much stronger for rendering quality/FPS in the tested storage
setting. That result sharpens the scope: GEMS is a mesh-splat compactness and
reliability-analysis paper, not a general NVS leaderboard paper.

Pointers: `analysis/r1_3dgs_reference/r1_table.md`,
`RESULTS/CLAIMS_EVIDENCE_MATRIX.md`.

## A4. The downstream application failed. Does that invalidate the work?

It invalidates a positive planning claim, not the whole work. The negative is
useful because four consumer families fail for different, quantified reasons:
raw voxelization blocks plans, TSDF creates unsafe false-free regions,
certified sub-mesh sheds load-bearing structure, and three-state carving makes
UNKNOWN too conservative. Across these routes, B50 compaction remains
outcome-invariant; the blocker is baseline checkpoint geometry and train
coverage.

Pointers: `RESULTS/CONSUMPTION_IMPOSSIBILITY.md`,
`T5b_r3_trilogy.md`, `RESULTS/NEGATIVE_RESULTS.md`.

## A5. Why not run more V2/V3 variants after R3-FINAL?

Stage3 pre-registered V2/V3 only for a near-miss. V1 produced 0/100 feasible
plans on toy and courtyard clean/B50; this is not near the bar. Continuing
would be post-hoc threshold search, so the compliant action is to close the
axis with an impossibility addendum.

Pointers: `docs/GEMS_Stage3_Closure_Prompt.md`,
`RESULTS/CONSUMPTION_IMPOSSIBILITY.md`.

## A6. What is the key mechanism behind failures?

Train evidence is predictive but incomplete. It identifies many reliable
regions, but low-coverage or never-train-visible structure can dominate
test-time and downstream errors. Conservative occupancy avoids false-free
hazards but blocks valid free space; aggressive occupancy plans more but
collides. This trade is visible across E2, E2R, R3.b, and R3-FINAL.

Pointers: `T3_geometry.md`, `T5_downstream.md`, `T5b_r3_trilogy.md`,
`analysis/e2geo_evidence_vs_error/summary.json`.

## A7. What about robustness and seeds?

Stage3 executes the human ruling: one full seed-1 garden retrain pair plus one
50% train-view-drop arm. The seed substitute supports the waiver:
clean seed1-seed0 = +0.031 dB CI[-0.018,+0.090], and the B5 residual shift is
+0.008 dB CI[+0.000,+0.017], both inside the 0.15 dB support rule. The
view-drop arm matches the predicted direction: half-train B50 residual worsens
by -0.030 dB CI[-0.047,-0.014] versus full-view garden. The remaining full
grids are explicitly waived, not silently omitted.

Pointers: `T7_robustness.md`, `LEDGER.md` GOAL #C-01.

## A8. Why not SuGaR/2DGS-style meshing or Gaussian-to-mesh pipelines?

Not compared. The current evidence pack is scoped to MeshSplatting-derived
checkpoints and consumers. Related systems should be discussed as future work
or context, not as beaten baselines. If reviewers demand it, the honest answer
is that R1 covers only a 3DGS rendering context row, not meshing or planning.

Pointers: `CLAIMS.md` NON-CLAIMS, `RESULTS/CLAIMS_EVIDENCE_MATRIX.md`.

## A9. Is this ready for a top-tier claim?

Only under a bounded claim. It is not ready as "we solve closed-loop planning"
or "we beat all neural rendering baselines." It is viable as a compactness plus
measurement-suite paper if the submitted story makes the negative evidence
central and avoids decorative overclaiming.

Pointers: `CLAIMS.md`, `EXPERIMENT_REPORT.md`, `RESULTS/HANDOFF.md`.
