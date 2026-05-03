# MeshSplatOpt Stage R5 Reversible Edits Smoke Report

Date: 2026-05-02

## Status

`PASS`.

## Checks

| check | result |
|---|---|
| delete triangles rollback exact | `PASS` |
| snap vertices rollback exact | `PASS` |
| fill patch rollback exact | `PASS` |
| edge collapse preserves valid indices | `PASS` |
| split triangles rollback exact | `PASS` |
| protect metadata rollback exact | `PASS` |
| appearance reset metadata rollback exact | `PASS` |
| invalid index caught | `PASS` |
| degenerate face caught | `PASS` |

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR5_reversible_edits.py
```

The full machine-readable report is:

```text
outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/edit_smoke_report.json
```
