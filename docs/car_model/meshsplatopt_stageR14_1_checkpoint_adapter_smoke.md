# MeshSplatOpt Stage R14.1 Checkpoint Adapter Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Checks

| check | result |
|---|---|
| delete updates face arrays | `PASS` |
| snap updates vertex | `PASS` |
| fill appends vertices/faces | `PASS` |
| fill checkpoint schema valid | `PASS` |
| delete checkpoint schema valid | `PASS` |
| snap checkpoint schema valid | `PASS` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py
```
