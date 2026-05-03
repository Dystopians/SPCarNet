# MeshSplatOpt Stage R14.1 Checkpoint Adapter Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R14.1 adds a conservative adapter for applying supported MeshSplatOpt edits to Mesh Splatting `point_cloud_state_dict.pt` copies.

## Files Added

- `ss3dm_prior/meshsplatopt/checkpoint_adapter.py`
- `scripts/car_model/meshsplatopt_apply_edit_to_checkpoint.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Supported Edits

- `DELETE_TRIANGLES`: removes selected faces and synchronizes per-face fields.
- `SNAP_VERTICES`: updates `triangles_points`.
- `PROTECT` / `APPEARANCE_RESET`: metadata/no geometry change.

## Deferred Edits

- `FILL_PATCH`
- `SPLIT_TRIANGLES`
- `EDGE_COLLAPSE`
- `FACE_MERGE`

These require robust per-vertex radiance initialization, face/vertex remapping, and optimizer-state handling. R14.1 rejects them rather than writing unsafe checkpoint copies.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py
```

Smoke checks:

- delete updates per-face arrays;
- snap updates vertex position;
- fill is rejected with clear deferred-reason;
- output checkpoint schemas remain valid.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/checkpoint_adapter_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/delete/point_cloud_state_dict.pt`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/snap/point_cloud_state_dict.pt`

## Decision

`PASS`. This partially unblocks R14 for delete/snap checkpoint-copy experiments, but certified fill/split public-scene runs still require radiance initialization support.
