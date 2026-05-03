# MeshSplatOpt Stage R7 Snap/Deform Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R7 implements safe snap/deform proposal generation and verifies synthetic dent repair, unsupported floater rejection, misalignment reduction, and exact rollback.

## Files Added

- `ss3dm_prior/meshsplatopt/snap_proposals.py`
- `scripts/car_model/meshsplatopt_make_snap_proposals.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Behavior

The snap proposal generator fits a plane, identifies candidate vertices with large residuals, and emits `SNAP_VERTICES` edits at step sizes `0.1`, `0.25`, and `0.5`. It supports an explicit supported-vertex set so unsupported floaters are rejected instead of attached to a plausible plane.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py
```

Smoke metrics:

- dent plane error: `0.03072 -> 0.019831720797113993`
- misalignment plane error: `0.019200000000000002 -> 0.009984000000000002`
- floater without support: rejected
- rollback: exact

## Artifacts

- `outputs/carnet/meshsplatopt/stageR7_snap_smoke/snap_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR7_snap_smoke/snap_outputs/snap_proposals.json`
- `outputs/carnet/meshsplatopt/stageR7_snap_smoke/snap_outputs/snap_summary.csv`
- `outputs/carnet/meshsplatopt/stageR7_snap_smoke/snap_outputs/snap_debug_before_after.ply`

## Decision

`PASS`. Snap improves synthetic geometry error without attaching unsupported floaters and remains rollback-compatible.
