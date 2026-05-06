# Final Stage SCE11 Method Spec

Date: 2026-05-06

Decision: `SCE11_RELEASE_PACKAGE_READY_FOR_WORKSHOP_OR_ARXIV`

## Method

MeshSplatOpt-SCE combines:

1. CSEF proposal/risk signals.
2. Sentinel sparse correspondences built from train/calibration views.
3. Evidence Conflict Graph linking views, sparse points, pixel samples, clusters, certificates, and actions.
4. One-sided parent-Pareto rollback objective.
5. Optional certificate local surgery planner.

## Rollback Objective

```text
L = mean_i SmoothL1(ReLU(e_current(i) - e_parent(i) - margin), 0)
```

The loss never penalizes improvements over the parent.

## Algorithm

1. Audit parent/candidate.
2. Build train/calibration sentinel cache.
3. Run or load candidate recovery.
4. Gate sentinels against parent.
5. If regressed, run SCE rollback recovery.
6. Build ECG and certificate plan.
7. Accept only parent-Pareto-safe checkpoints.

## Failure Modes

- Held-out view with geometry not represented by train sentinels.
- Top-k conflict overfitting.
- Hard/far train proxy not transferring to test.
- Non-delete surgery unsupported by real evidence.

