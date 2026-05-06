# Final Stage SCE15 Real-Scene Local Surgery Pilot Plan

Date: 2026-05-06

Decision: `SCE15_PLAN_AND_SAFE_MATERIALIZER_IMPLEMENTED_NO_TOPOLOGY_PROMOTION`

## Current Evidence

Courtyard is the only real scene with a fully diagnosed SCE7 residual. Both held-out and train/calibration ECGs identify sparse-depth certificate conflicts, but SCE13 emits only `ROLLBACK_ONLY`:

- test top conflict: `cluster 27`
- train top conflict: `cluster 876`
- no certified `SNAP_LOCAL`, `SPLIT_ALLOCATE`, `FILL_PATCH_LOCAL`, or `DELETE_OR_COLLAPSE`

Therefore a real topology edit would be unjustified on courtyard at this point.

## Pilot Protocol

1. Build ECG on train/calibration split.
2. Plan certificate edits with SCE13.
3. Materialize only top-k edits that satisfy certificates.
4. Keep topology frozen unless the action is explicitly topology-changing and `--allow_topology_edits` is set.
5. Run short recovery and independent evaluation.
6. Gate against parent and rollback-only controls.
7. Promote only if a non-delete action improves a local defect metric without violating global parent-Pareto.

## Implemented Wrappers

- `scripts/car_model/meshsplatopt_materialize_certificate_edit_plan.py`
- `scripts/car_model/meshsplatopt_run_certificate_edit_recovery.py`
- `scripts/car_model/meshsplatopt_gate_certificate_edit_result.py`

## Decision

Do not claim real bidirectional surgery from courtyard yet. Current evidence supports SCE certificate/recovery as the empirical core and keeps local surgery as safe infrastructure pending a real scene with certified non-rollback actions.

