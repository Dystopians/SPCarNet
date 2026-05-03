# MeshSplatOpt Stage R4 Defect Mining Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Synthetic Cases

1. A synthetic parking-ground mesh with a large rectangular missing patch.
2. A synthetic out-of-trajectory void hint with no boundary support and no camera coverage.

## Checks

| check | result |
|---|---|
| giant ground void detected | `PASS` |
| unknown unobserved void detected | `PASS` |
| unknown void has no repair edits in normal mode | `PASS` |
| unknown void records a no-repair reason | `PASS` |

## Defect Types Emitted

- `GIANT_GROUND_VOID`
- `UNKNOWN_UNOBSERVED_VOID`

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR4_defect_mining.py
```
