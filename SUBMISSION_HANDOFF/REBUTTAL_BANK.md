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

# Stage-4 additions (ECR track, 2026-07-09; prompt §4 RED-TEAM)

## A10. "This is just ULR / Deep Blending 2.0."

Differences, all measured: (1) the base is a mesh-splat checkpoint with a
retained triangle-mesh artifact and frozen geometry/downstream guarantees
(Stage-2/3 pack; preservation-exactness CIs [0,0]); (2) the confidence inputs
are evidence-CERTIFIED — the same machinery whose error-prediction law is
banked (Spearman rho ~= 0.69–0.74 on 3 scenes, GOAL#015A) and whose failure
boundary is quantified (coverage gaps; 5–11x error tails); (3) strict
train-only guarantees are ENFORCED per row by an audit tool
(`tools/audit_test_path.py --ecr`: transport reads ⊆ cache manifest, manifest
disjoint from the recomputed test split, frozen per-view kwargs hash, no GT
accessor reachable in the render path); (4) storage is accounted at TOTAL
artifact honesty (checkpoint + cache raw AND lossless-compressed, in every
row); (5) the design is justified by a falsification corpus proving the baked
alternative is information-theoretically closed on these checkpoints: static
baking captures 4.76% of the Phase-J gap, full-parameter distillation 5–16%,
held-out residual cosine ~= 0.21, and Stage-4's own L1 rung measured the
composition NEGATIVE (full9 −0.109 dB CI[−0.129,−0.090]: distillation consumes
exactly the residual structure the transport exploits). The negatives ARE the
motivation section.

Pointers: `CLAIMS_ECR.md`, `LEDGER.md` GOALs #E-00..#E-04,
`analysis/e0_pj2026/`, `analysis/e2geo_evidence_vs_error/`.

## A11. "Why not 3DGS + an enhancer?"

Answered by measured rows, not prose: the R1 context row already banked 3DGS
at matched artifact storage (2.1–3.4 dB above the GEMS base at 3–4x FPS,
reported plainly); Stage-4 §4 adds both cells, MEASURED (2026-07-10):
(a) 3DGS at matched TOTAL storage (GOAL #E-07): vanilla 3DGS-30k sits under
the ECR TOTAL budget on all 3 scenes (uses 11–55% of it) and stays ahead
+0.32..+1.53 dB — but the Stage-2 gap (2.1–3.4 dB) is mostly closed by the
ECR stack (kitchen +0.32 dB at near-parity LPIPS). (b) The enhancer route
itself (GOAL #E-09): Difix3D+ (nvidia/difix_ref, single-step, given the SAME
train-view evidence rights — nearest support train GT as reference) applied
to our base renders is PSNR-NEGATIVE on all 3 scenes (−0.89..−1.54 dB, CIs
excl. 0) with LPIPS gains (−0.030 on 2 scenes, WORSE on garden) that fall
well short of the transport's; the ECR final stack exceeds the
Difix-enhanced base on BOTH metrics on all 3 scenes. So "base + generative
enhancer" is not a shortcut to this deliverable: evidence transport
dominates enhancement on the same artifact. The mesh artifact,
geometry/downstream measurement suite, and preservation-exactness story
have no 3DGS-family equivalent; rendering-quality comparisons name their
references explicitly.

Pointers: `analysis/r1_3dgs_reference/r1_table.md`,
`analysis/final_stack/e07_matched_total_3dgs.md`,
`analysis/difix_cell/difix_table.md`, MATRIX `ECR` section (L5 + external
cells).

## A12. "Per-scene learned fusion doesn't generalize."

Correct, and claimed as such: the method is per-scene BY DESIGN (NON-CLAIM:
no cross-scene generalization). The fusion net is trained once per scene on
train views only, frozen, applied identically to every test view — the same
legal move as Deep Blending's per-scene training. Cross-scene training is
future work (one line, T3 note only).

Pointers: `CLAIMS_ECR.md` NON-CLAIMS, LEDGER #E-04 pre-registration.

### A10 addendum (2026-07-11, measured): "just ULR/Deep Blending" — now answered head-to-head

Two measured externals sharpen the lineage answer: (a) a PRETRAINED
generalizable IBR baseline (IBRNet, 10 source views — more evidence than our
transport uses, convention verified by a 22.48 dB self-reconstruction gate)
scores 1.0–5.4 dB BELOW even our base anchor on garden/bicycle/kitchen; the
ECR stack exceeds it by +2.6..+5.9 dB (`analysis/ibr_cell/ibr_table.md`).
(b) The tuned per-scene classical point IS our floor: PJ-2026 (frozen
K-nearest warp + confidence fuse + per-scene train-LOO α) — and the ladder
adds +0.361/−0.0169 (full9) and +0.122/−0.0084 (T&T+DB, zero tuning) on top
of it with CIs excl. 0. So the comparison the question implies has been run
in both directions: generic-generalizable (fails here) and tuned-classical
(our floor, exceeded).

## A13. "GaussianEditor-class systems delete/recolor without any of this — why is editing hard here?" (added 2026-07-12)

Because they edit only the REPRESENTATION — which is our C1 control (edited base render), the easy
part. ECR ships photographic evidence NEXT TO the representation for +1.67 dB; the hard problem this
work isolates is keeping that evidence consistent with edits, and it is measurably non-trivial: the
banked evidence matrix shows a full cache rebuild paints deleted objects back (+3.09 dB ghost) and a
stale cache repaints old colors (+1.96), while per-pixel face-provenance invalidation survives both
classes AND beats every simpler masking strategy on TRUE edited ground truth (oracle CIs excl. 0:
+0.16 vs 4-px dilation, +0.26 vs target-side masking, +0.38 vs 2D boxes — which also bleed up to
−0.74 dB on unaffected regions). Positioning matrix (related work, honest one-liners):
- Gaussian editing (GaussianEditor / Gaussian-Grouping class): representation edits, no external
  evidence to invalidate; complementary, not comparable — no cache exists there.
- Neural mesh editing (NeuMesh / SEAL-3D class): edits neural fields anchored to meshes; retraining or
  distillation per edit; no train-photograph reuse at render time.
- Editable IBR / source-view masking (classical IBR with view or region disabling): our TM and
  view-drop baselines ARE this family — both measured and beaten (TM −0.263 oracle in-region;
  view-drop degenerates to no-ECR when the object is widely visible).
- Inpainting/generative removal: excluded by the no-hallucination policy; disocclusions honestly fall
  back to the edited base via the structural gate.
