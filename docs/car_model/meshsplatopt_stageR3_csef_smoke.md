# MeshSplatOpt Stage R3 CSEF Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Synthetic Scene

The smoke creates a synthetic mesh with:

- a ground plane;
- one missing central cell that creates a boundary/hole region;
- one dented vertex;
- one disconnected floater triangle.

## Metrics

From `outputs/carnet/meshsplatopt/stageR3_csef_smoke/stageR3_csef_smoke_report.json`:

| metric | value |
|---|---:|
| normal debt | `0.18554923879355764` |
| hole boundary debt | `0.34568891727769624` |
| floater uncertainty | `0.9` |
| floater positive surface evidence | `0.10520833333333335` |
| dent debt | `0.31162340519788023` |
| region count | `2` |

## Checks

| check | result |
|---|---|
| boundary/hole region has high explanation debt | `PASS` |
| floater has high uncertainty | `PASS` |
| floater has low positive evidence | `PASS` |
| normal ground has low debt | `PASS` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR3_csef.py
```
