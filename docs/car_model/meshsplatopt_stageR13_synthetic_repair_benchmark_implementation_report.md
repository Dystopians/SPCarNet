# MeshSplatOpt Stage R13 Synthetic Repair Benchmark Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R13 implements a controlled synthetic repair benchmark and result collector.

## Files Added

- `ss3dm_prior/meshsplatopt/synthetic_damage.py`
- `scripts/car_model/meshsplatopt_make_synthetic_repair_benchmark.py`
- `scripts/car_model/meshsplatopt_run_synthetic_repair_benchmark.py`
- `scripts/car_model/meshsplatopt_collect_synthetic_repair_results.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_make_synthetic_repair_benchmark.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_run_synthetic_repair_benchmark.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_collect_synthetic_repair_results.py
```

Gate:

- full MeshSplatOpt improves five categories over delete-only;
- prior-only unknown void is rejected;
- topology remains valid.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/benchmark_spec.json`
- `outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/synthetic_repair_results.json`
- `outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/synthetic_repair_table.csv`
- `outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/synthetic_repair_report.md`

## Decision

`PASS`. Full MeshSplatOpt improves at least four synthetic damage categories over delete-only and rejects the prior-only unknown void.
