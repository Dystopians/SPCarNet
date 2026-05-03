# MeshSplatOpt Stage R14.4 Constructive Checkpoint Fill Report

Date: 2026-05-02

## Gate

`PASS`.

`FILL_PATCH` is now supported for checkpoint-copy materialization with nearest-neighbor radiance initialization.

## Code Change

Updated:

- `ss3dm_prior/meshsplatopt/checkpoint_adapter.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py`

## Verification

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py
```

Checks:

- delete updates face arrays;
- snap updates vertex position;
- fill appends vertices and faces;
- fill output schema is valid.

## Decision

`PASS`.

This unblocks materializing R8 fill proposals in checkpoint copies. It does not replace teacher recovery; new radiance is only initialized from nearest existing vertices.
