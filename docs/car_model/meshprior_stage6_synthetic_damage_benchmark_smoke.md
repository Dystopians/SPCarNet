# MeshPrior Stage 6 Synthetic Damage Benchmark — Smoke Report

| Field | Value |
|---|---|
| Stage | M6 / synthetic damage benchmark smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Command

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
```

## Output Summary

```text
rows: 4
floater_recall: 1.0
```

The smoke covered:

- `local_hole`,
- `floater`,
- `vertex_noise`,
- `density_imbalance`.

It generated `metrics.json`, CSV tables, `failure_cases.md`, and a markdown report.

## Gate Verdict

`PASS`
