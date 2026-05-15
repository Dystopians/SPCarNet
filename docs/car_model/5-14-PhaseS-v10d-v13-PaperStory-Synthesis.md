# Phase-S v10d-v13 Paper-Story Synthesis

Date: 2026-05-14

Scope: review-only synthesis of the May 14 Phase-S PatchCert carrier line from
`v10d_scaledcluster_shapefix` through `v13_chartquad`.  This note does not
claim a new paper row.  It records what changed, what the evidence actually
says, and the fixed next milestone.

## Bottom Line

v10d-v13 are real implementation and representation-operator changes, but they
do not close the paper story.  Every completed train-val decision selects the
`phasej_guarded_adaptedge` fallback, so the effective v10d-v13 Phase-S gain over
Phase-J is exactly zero in the collected summaries.

This must not be written as full baseline domination.  The current honest paper
story is still: Phase-J is the stronger baseline-over-clean MeshSplatting row,
while v10d-v13 are a negative or not-yet-successful Phase-S carrier ablation.

## Evidence Index

| version | result paths | W&B group | scenes | train-val outcome | effective outcome |
|---|---|---|---|---|---|
| v10d scaled-cluster shape fix | `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10d_scaledcluster_shapefix_20260514_flowers`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10d_scaledcluster_shapefix_20260514_summary/summary_flowers.md`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v10d_scaledcluster_shapefix_20260514_qualitative/qualitative_summary.md` | `phase_s_patchcert_v10d_scaledcluster_shapefix_20260514` | flowers | rejected, selected Phase-J | mean effective dPSNR/dSSIM/dLPIPS = `0.000000 / 0.000000 / 0.000000` |
| v11 rank-2 carrier | `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v11_rank2carrier_20260514_flowers`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v11_rank2carrier_20260514_bicycle`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v11_rank2carrier_20260514_summary/summary_2scene.md`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v11_rank2carrier_20260514_qualitative/qualitative_summary.md` | `phase_s_patchcert_v11_rank2carrier_20260514` | flowers, bicycle | both rejected, selected Phase-J; bicycle was operator no-op | mean effective dPSNR/dSSIM/dLPIPS = `0.000000 / 0.000000 / 0.000000` |
| v12 chart-linear carrier | `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v12_chartlinear_20260514_flowers`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v12_chartlinear_20260514_bicycle`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v12_chartlinear_20260514_summary/summary_2scene.md`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v12_chartlinear_20260514_qualitative/qualitative_summary.md` | `phase_s_patchcert_v12_chartlinear_20260514` | flowers, bicycle | both rejected, selected Phase-J; bicycle was operator no-op | mean effective dPSNR/dSSIM/dLPIPS = `0.000000 / 0.000000 / 0.000000` |
| v13 chart-quadratic carrier | `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v13_chartquad_20260514_flowers`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v13_chartquad_20260514_bicycle`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v13_chartquad_20260514_summary/summary_2scene.md`; `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v13_chartquad_20260514_qualitative/qualitative_summary.md` | `phase_s_patchcert_v13_chartquad_20260514` | flowers, bicycle | both rejected, selected Phase-J; bicycle was operator no-op | mean effective dPSNR/dSSIM/dLPIPS = `0.000000 / 0.000000 / 0.000000` |

## Real Method Changes

v10d is not just a relabel.  It fixes the v10c scaled-carrier shape bug exposed
by `flowers`: face-local samples carry three local corner ids per pixel, so
the per-face scale has to broadcast over both corner and SH dimensions.  The
fixed rerun uses the scaled shared patch-corner SH carrier with strict PatchCert
carrier replay.

v11 changes the carrier from scaled shared coefficients to a rank-2 mixture
carrier.  On `flowers`, the checkpoint operator becomes non-trivial: 8 accepted
faces, 24 added vertices, one accepted patch, and no topology triangle change.
On `bicycle`, the same policy produces a safe no-op because no patch is accepted.

v12 changes the carrier family again to a chart-linear patch-corner SH carrier.
On `flowers`, it expands the operator-level carrier to 13 accepted faces, 39
added vertices, and two accepted patches.  On `bicycle`, it still no-ops.

v13 adds a chart-quadratic carrier.  On `flowers`, it keeps 13 accepted faces
and two accepted patches while using a rank-6 chart feature basis; the audit
shows the strongest cluster-fit regression numbers among this sequence.  That
is a cleaner carrier fit, but it still does not pass the downstream train-val
selection gate.

## Quantitative Read

The gate-level deltas are too small and in the wrong shape for a paper result:

| version | scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| v10d | flowers | Phase-J fallback | false | `+0.000034332` | `-0.000013351` | `+0.000003874` | `+0.000001907` | `+0.000000060` | `-0.000000417` |
| v11 | flowers | Phase-J fallback | false | `+0.000036240` | `-0.000013351` | `+0.000004023` | `+0.000001907` | `+0.000000000` | `-0.000000060` |
| v11 | bicycle | Phase-J fallback | false | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` |
| v12 | flowers | Phase-J fallback | false | `+0.000040054` | `-0.000013351` | `+0.000003904` | `+0.000000000` | `-0.000000060` | `-0.000000238` |
| v12 | bicycle | Phase-J fallback | false | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` |
| v13 | flowers | Phase-J fallback | false | `+0.000036240` | `-0.000013351` | `+0.000003934` | `+0.000000000` | `-0.000000119` | `-0.000000238` |
| v13 | bicycle | Phase-J fallback | false | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` | `+0.000000000` |

The `flowers` rows fail because the balanced train-val tail is negative and
LPIPS regressions are too frequent.  The `bicycle` rows fail before metric
interpretation because the operator rejects or copies a no-op checkpoint.  The
qualitative panels are useful diagnostics, but they are explicitly selected from
report-only held-out deltas for visualization, not for promotion.

## Why This Is Not Paper-Level Closure

- There is no accepted v10d-v13 scene under the Phase-K train-val gate.
- Effective deltas are zero because rejected scenes fall back to Phase-J.
- `flowers` has a real operator edit, but the tail gate rejects it every time.
- `bicycle` remains a certificate/no-op failure across v11-v13.
- Report-only held-out differences are at 1e-6 to 1e-5 scale and are not visually
  or statistically persuasive.
- The evidence is a two-scene stress slice, not full9 closure.
- The story is relative to Phase-J fallback, not a new direct domination of the
  clean MeshSplatting baseline.
- Carrier fit quality improved, but better fitting the candidate patch does not
  imply better rendered held-out images after the full train-val decision.

## Fixed Next Milestone

The next milestone is **M-v14: train-val selection over Phase-J, not another
carrier-capacity variant**.

Run one frozen carrier policy, with the method name declared before launching,
on the fixed stress set `flowers`, `bicycle`, and one non-flowers/non-bicycle
control scene from the existing Phase-S candidate set.  Use the same train-only
selection rule and keep held-out test metrics report-only.

M-v14 passes only if all of the following hold:

- at least two scenes are selected over `phasej_guarded_adaptedge` by the
  train-val gate;
- no passing scene is an operator no-op;
- the mean effective deltas over Phase-J are strictly positive for PSNR and
  SSIM and strictly negative for LPIPS;
- qualitative panels show non-zero local image changes that correspond to the
  accepted scenes, not only report-only visualization rows.

If M-v14 fails these criteria, this PatchCert carrier branch should be closed as
an integrity/ablation result rather than kept alive as the main paper route.

## Final Weakness List

1. No v10d-v13 train-val acceptance.
2. Zero effective improvement over Phase-J.
3. Negative or fragile tail behavior on `flowers`.
4. Repeated `bicycle` operator no-op.
5. Extremely small held-out report-only deltas.
6. No broad scene coverage or full9 Phase-S closure.
7. No visible paper-grade improvement in the qualitative contact sheets.
8. Improved carrier mechanics do not yet translate into a publishable method
   claim.
