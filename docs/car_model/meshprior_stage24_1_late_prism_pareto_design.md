# Stage24.1 Late-PRISM Pareto Sweep Design

Date: 2026-05-02

## Goal

Find a stronger integrated training-time topology-control Pareto point than Stage24-v3 on `parking_phone_tiny`.

M24-v3 proved full-budget PRISM commits inside training, but its topology reduction was small: `823651` triangles versus `833775` for current branch. M21.5 posthoc `prune_50` remains the strongest topology/quality diagnostic row at `416888` triangles with similar render metrics. M24.1 tests whether multiple late small PRISM edits can approach that tradeoff without suppressing normal densification.

## Constraints

- Use online W&B for every current-branch training run.
- Use late PRISM only; normal Mesh Splatting densification should finish before topology decisions.
- Keep counterfactual gate enabled.
- Keep final cleanup disabled unless explicitly tested as a separate ablation.
- Report render metrics, COLMAP proxy geometry, topology, commit/reject counts, rollback state, and final-cleanup summary.

## Initial Sweep Plan

Start with the safest high-value point:

```text
run: pareto_ratio0p005_rounds8_7000iter
geometry_acq_until_iter: 6000
stats_collection_iters: 150
candidate_rounds: 8
candidate_prune_ratio_per_round: 0.005
recovery_iters: 80
post_commit_recollect_iters: 10
final_cleanup: disabled
```

Rationale: M24-v2 showed `0.05` is too aggressive and fully rejected. M24-v3 showed `0.01` commits twice but then stalls with limited total topology reduction. The next most informative point is smaller per-round edits with more rounds, which tests whether the counterfactual gate accepts gradual topology reduction.

If this row commits repeatedly but topology reduction remains weak, run `0.01 x 8`. If it is mostly rejected, run `0.005 x 4` or inspect gate diagnostics before spending more GPU.

## Gate

`PASS` if at least one integrated row materially reduces topology relative to current branch while preserving render metrics near current/M21.5.

`SOFT PASS` if safety gates reject the stronger Pareto region but all artifacts are complete.

`FAIL` if W&B, metrics, rollback, final-cleanup accounting, or independent evaluation artifacts are missing.
