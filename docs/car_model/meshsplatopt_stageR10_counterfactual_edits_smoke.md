# MeshSplatOpt Stage R10 Counterfactual Edits Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Checks

| check | result |
|---|---|
| good fill accepted | `PASS` |
| bad floater rejected | `PASS` |
| snap through free space rejected | `PASS` |
| delete supported surface rejected | `PASS` |
| non-delete edit accepted | `PASS` |
| reject rollback exact | `PASS` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR10_counterfactual_edits.py
```
