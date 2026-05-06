# Final Stage SCE17 Paper Method Spec and Claim Lock

Date: 2026-05-06

Decision: `SCE17_CLAIM_LOCK_PASS`

## Primary Method Name

Primary name: **MeshSplatOpt-SCE**.

Reason: it preserves continuity with MeshSplatOpt while making the new mechanism explicit: Sparse/Surface Correspondence Evidence, Evidence Conflict Graphs, and certificate-carrying recovery. "Evidence-Sentinel Mesh Surgery" is a good subtitle, but the main paper name should stay close to the existing project.

## Core Contributions

1. **Counterfactual Surface Evidence Field (CSEF)** for proposal and risk estimation. CSEF identifies compact/recovery risks and evidence debts rather than blindly optimizing primitive count.
2. **Evidence Conflict Graph (ECG)** for localizing contradictions between render improvement, sparse geometry evidence, mesh clusters, and certificates.
3. **Certificate-carrying sparse correspondences** as sentinels. These are train/calibration sparse COLMAP correspondences with parent/candidate error state and no test leakage.
4. **One-sided parent-Pareto rollback recovery**. The loss only activates when the current candidate is worse than the parent on a sentinel under AbsRel, MAE, or calibrated combined error.
5. **Optional certificate edit planner** for bidirectional surgery. It can choose rollback-only, appearance-only, snap, split, fill, delete/collapse, or reject, but only when the ECG supplies the required certificates.

## Limited Proposition

Given a fixed sentinel set `S`, parent sparse errors `e_parent(i)`, current sparse errors `e_current(i)`, and margin `m`, the one-sided rollback objective optimizes:

```text
sum_i SmoothL1(ReLU(e_current(i) - e_parent(i) - m), 0)
```

If the optimized loss reaches zero on `S`, then for every active sentinel under the chosen error functional:

```text
e_current(i) <= e_parent(i) + m
```

This is a measured-evidence certificate, not a dense geometry guarantee. It says nothing about unobserved regions, non-sentinel pixels, or arbitrary out-of-trajectory views.

## Not Claimed

- Not a universal single prune ratio.
- Not guaranteed dense geometry correctness.
- Test correspondences are never allowed in training loss or policy selection.
- Prior-only void hallucination is not a valid repair.
- Non-delete edits are not claimed as real-scene headline wins unless SCE15 or later proves them with independent gates.
- Current courtyard SCE7 does not fully beat F82 on all metrics because Depth MAE remains `+0.001787`.

## Main Table Candidates

- F49 CSEF-family 5-scene validation-budget table.
- F82 fixed adaptive policy two-seed table.
- SCE courtyard bottleneck repair table, reported honestly as strong partial unless MAE is closed.
- SCE14 stress-test benchmark table.
- SCE15 local surgery pilot only if a non-rollback action passes independent gates.

## Reviewer Risk Checklist

- **Is it just pruning?** Answer with ECG, sentinel certificates, and SCE14 stress-test repair families.
- **Is it just depth regularization?** Answer with one-sided parent-Pareto formulation and SCE16 ablations.
- **Is it overfit to selected scenes?** Answer with fixed-policy claims separated from validation-budget claims, held-out split discipline, seeds, and failure logs.
- **Is sparse COLMAP weak?** State the proxy limitation and pair it with independent RGB/LPIPS/normal/render metrics.
- **Is validation-selected budget unfair?** Keep F49 validation-budget and F82/SCE fixed-policy claims separate.

## Claim Tiers

### Tier A: Full Top-Conference Claim

Allowed only if SCE14 plus a real SCE15 non-rollback local surgery pilot pass, and if multiscene SCE8 validation is completed with strong fixed-policy evidence.

Claim: MeshSplatOpt-SCE is an evidence-sentinel certified mesh-surgery framework that repairs controlled and real local mesh-splatting failures while preserving sparse geometry certificates.

### Tier B: Strong Method Claim

Allowed if SCE fixes the F95 bottleneck fully or transfers across multiple scenes, while local surgery remains optional infrastructure.

Claim: MeshSplatOpt-SCE introduces certificate-carrying sparse evidence and one-sided parent-Pareto recovery, improving compact mesh splatting under strict train/test separation and explaining failures through ECG.

### Tier C: Honest Narrow Claim

Current safe tier.

Claim: MeshSplatOpt-SCE is a rigorously instrumented evidence-sentinel recovery framework. It substantially improves the strongest rejected courtyard candidate and beats F82 on RGB, LPIPS, AbsRel, and normals, while leaving a tiny Depth MAE gap. ECG/planner/stress-test components provide the next path for broader repair claims.

