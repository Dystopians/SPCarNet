# MeshSplatOpt Stage R7 Snap Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Checks

| check | result |
|---|---|
| dent error reduced | `PASS` |
| floater rejected without support | `PASS` |
| misalignment error reduced | `PASS` |
| rollback exact | `PASS` |

## Metrics

| metric | value |
|---|---:|
| dent before error | `0.03072` |
| dent after error | `0.019831720797113993` |
| misalignment before error | `0.019200000000000002` |
| misalignment after error | `0.009984000000000002` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py
```
