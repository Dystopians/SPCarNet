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
- `FILL_PATCH`: appends new vertices/faces and initializes new vertex radiance from nearest existing vertices.
- `PROTECT` / `APPEARANCE_RESET`: metadata/no geometry change.

## Deferred Edits

- `SPLIT_TRIANGLES`
- `EDGE_COLLAPSE`
- `FACE_MERGE`

These require topology remapping and/or optimizer-state handling.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py
```

Smoke checks:

- delete updates per-face arrays;
- snap updates vertex position;
- fill appends vertices/faces with valid initialized attributes;
- output checkpoint schemas remain valid.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/checkpoint_adapter_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/delete/point_cloud_state_dict.pt`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/snap/point_cloud_state_dict.pt`

## Decision

`PASS`. This unblocks checkpoint-copy delete/snap/fill experiments, but certified public-scene fill still requires render-backed gates and teacher recovery.
