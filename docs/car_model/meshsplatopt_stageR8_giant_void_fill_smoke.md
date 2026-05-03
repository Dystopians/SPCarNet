# MeshSplatOpt Stage R8 Giant Void Fill Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Checks

| check | result |
|---|---|
| small hole boundary reduced | `PASS` |
| giant ground void valid patch | `PASS` |
| unknown void rejects in normal mode | `PASS` |
| prior-only diagnostic proposal marked | `PASS` |
| fill rollback exact | `PASS` |
| degenerate boundary rejected | `PASS` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR8_giant_void_fill.py
```

The full machine-readable report is:

```text
outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_smoke_report.json
```
