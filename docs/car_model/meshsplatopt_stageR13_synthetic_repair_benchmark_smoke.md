# MeshSplatOpt Stage R13 Synthetic Repair Benchmark Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Gate

| check | result |
|---|---|
| full improves at least 4/7 over delete-only | `PASS` |
| prior-only unknown void rejected | `PASS` |
| topology valid | `PASS` |

Full MeshSplatOpt advantage categories over delete-only:

- `giant_ground_void`
- `ground_wall_misalignment`
- `local_dent`
- `noisy_rough_patch`
- `small_hole`

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_run_synthetic_repair_benchmark.py
```
