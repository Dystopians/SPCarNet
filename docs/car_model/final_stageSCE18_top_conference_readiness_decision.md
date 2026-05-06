# Final Stage SCE18 Top-Conference Readiness Decision

Date: 2026-05-06

Decision: `NO_GO_FULL_TOP_CONFERENCE_YET_CONTINUE_RESEARCH_OR_WORKSHOP`

## Score Table

| criterion | score / 5 | rationale |
|---|---:|---|
| Novel method object | 4 | CSEF + ECG + sentinels + certificate planner are distinct and documented. |
| Load-bearing mechanism | 3 | SCE rollback is clearly useful, but strict all-metric win is not complete and ablation table is still partial. |
| Real-scene evidence | 3 | Courtyard bottleneck is strongly improved, but not fully solved on Depth MAE. |
| Generality | 3 | F82/F49 provide multiscene background; SCE-specific multiscene validation is not complete. |
| Baseline strength | 3 | Strong clean/F82/F95 controls exist; final matched QEM/global-depth/freeze/LPIPS table remains incomplete. |
| Honesty | 5 | Failures, negative controls, no-test-leakage, and limitations are extensively documented. |
| Reproducibility | 4 | Commands, W&B runs, artifacts, smoke tests, and manifests exist; some long multiscene SCE runs remain missing. |
| Qualitative evidence | 3 | Qualitative galleries exist historically, but SCE-specific visual comparison package needs refreshing. |
| Runtime/memory practicality | 3 | Short SCE runs are practical; full long validation cost remains high. |
| Writing clarity | 4 | SCE17 claim lock prevents hidden overclaiming. |

Total: **35 / 50**.

## Exact Missing Blockers

1. Courtyard SCE7 still misses F82 Depth MAE by `+0.001787`.
2. SCE8 fixed-policy validation on bonsai, room, and counter is not complete.
3. SCE15 has no real non-rollback surgery win; planner correctly emits rollback-only for courtyard.
4. SCE16 final reviewer-killer ablation matrix is partial.
5. SCE-specific qualitative gallery should be rebuilt around F82 vs SCE7 best vs failed controls.

## Recommended Title

**MeshSplatOpt-SCE: Evidence-Sentinel Certified Recovery for Compact Mesh Splatting**

## Recommended Main Figures

1. Method overview: CSEF -> ECG -> certificate sentinels -> rollback/planner.
2. Courtyard bottleneck: F82 vs F95 vs SCE7 best qualitative and metric deltas.
3. ECG visualization: localized conflict clusters and certificate pressure.
4. SCE14 stress-test families and synthetic gate.
5. Failure honesty figure: `DSC_0318` residual Depth MAE bottleneck.

## Recommended Main Tables

1. F82 fixed adaptive policy multiscene table.
2. SCE courtyard repair table with all positive metrics and the remaining MAE gap.
3. SCE16 ablation table.
4. SCE14 stress-test table.
5. Reproducibility/no-test-leakage checklist.

## Abstract Draft

Mesh splatting systems can be aggressively compacted, but global pruning and recovery losses often improve rendering while silently damaging sparse geometric evidence. We introduce MeshSplatOpt-SCE, an evidence-sentinel recovery framework that represents sparse correspondences, rendered samples, mesh clusters, and local certificates in an Evidence Conflict Graph. A one-sided parent-Pareto rollback objective penalizes only measured sparse-depth regressions relative to a parent checkpoint, while a certificate-carrying planner decides whether local conflicts justify rollback, appearance repair, or topology edits. On a difficult courtyard bottleneck, MeshSplatOpt-SCE substantially improves RGB, perceptual, AbsRel, and normal metrics over the fixed-policy parent while exposing a small remaining Depth MAE limitation. A synthetic mesh-surgery stress test and reviewer-facing ablations show how certificate-driven repair differs from global depth anchoring or delete-only pruning.

## Final Decision

Current recommendation: **do not submit yet as a full top-conference method paper unless the deadline forces a risk-taking submission**.

Best near-term venue posture: workshop/short paper or continue experiments. To become a strong full top-conference submission, the work needs either:

- close the final courtyard Depth MAE gap and complete SCE8 multiscene fixed-policy validation, or
- produce a real SCE15 non-rollback local surgery win with independent gates, plus finish the SCE16 ablation matrix.

