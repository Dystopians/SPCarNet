# MeshSplatOpt Stage R5 Reversible Edits Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

All required edit types are implemented as reversible numpy mesh operations or reversible metadata operations.

## Files Added

- `ss3dm_prior/meshsplatopt/edit_types.py`
- `ss3dm_prior/meshsplatopt/edit_snapshot.py`
- `ss3dm_prior/meshsplatopt/edit_apply.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR5_reversible_edits.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Supported Operations

| edit type | R5 behavior |
|---|---|
| `PROTECT` | records metadata; no geometry mutation |
| `DELETE_TRIANGLES` | removes selected faces |
| `EDGE_COLLAPSE` | rewrites remove vertex to keep vertex and drops degenerate faces |
| `FACE_MERGE` | conservatively removes redundant affected faces |
| `SNAP_VERTICES` | moves selected vertices to target positions |
| `SPLIT_TRIANGLES` | replaces selected triangles with centroid splits |
| `FILL_PATCH` | appends patch vertices and faces |
| `APPEARANCE_RESET` | records metadata; no geometry mutation |

All operations are reversible through `.npz` snapshots.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR5_reversible_edits.py
```

Smoke artifacts:

- `outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/edit_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/before.ply`
- `outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/after_*.ply`
- `outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/snapshots/*.npz`

## Decision

`PASS`. R6 can build topology baselines using the same mesh state, edit, integrity, and rollback contracts.
