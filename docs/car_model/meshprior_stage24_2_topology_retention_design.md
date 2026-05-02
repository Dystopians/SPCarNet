# Stage24.2 Topology-Retention Design

Date: 2026-05-02

## Goal

Test whether accepted M24.1 PRISM topology edits can be retained to the final checkpoint by freezing normal Mesh Splatting densification after the first accepted PRISM candidate commit.

## Rationale

M24.1 found a useful integrated row, but late densification can restore triangles after accepted PRISM edits. This makes final topology worse than the post-PRISM local checkpoint and keeps the integrated method below the M21.5 posthoc topology budget.

The smallest defensible M24.2 change is not a new proposal model. It is a schedule rule:

```text
if PRISM candidate prune commits:
    freeze subsequent standard densification
    keep standard pruning, optimization, counterfactual gate, rollback, and final-cleanup accounting intact
```

## Configuration

Start from the M24.1 best schedule:

```text
run: freeze_after_first_commit_7000iter
geometry_acq_until_iter: 6000
stats_collection_iters: 150
candidate_rounds: 8
candidate_prune_ratio_per_round: 0.005
no_candidate_retry_iters: 10
recovery_iters: 80
post_commit_recollect_iters: 10
final_cleanup: disabled
new flag: prism_freeze_densification_after_first_commit
```

## Gate

`PASS` if final topology is below the M24.1 best (`723438` triangles) while independent PSNR stays within about `0.15` of M24.1 best and normal proxy remains no worse than M24-v3.

`SOFT PASS` if topology improves but quality regresses too much for a headline.

`FAIL` if W&B, checkpoint, rollback, collector, or independent eval artifacts are missing.
