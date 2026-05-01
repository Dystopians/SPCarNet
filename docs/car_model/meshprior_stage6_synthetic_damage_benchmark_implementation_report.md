# MeshPrior Stage 6 Synthetic Damage Benchmark — Implementation Report

| Field | Value |
|---|---|
| Stage | M6 / synthetic damage benchmark |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/synthetic_damage.py` | Synthetic box mesh, local hole, floater, vertex noise, density imbalance, and boundary metrics. |
| `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py` | Benchmark runner. |
| `scripts/car_model/meshprior_make_synthetic_damage_report.py` | Markdown report generator. |
| `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py` | Smoke test. |
| `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md` | Stage design. |

## 2. Implementation Summary

M6 currently runs a controlled analytic benchmark on synthetic box meshes. It supports:

- `local_hole`,
- `floater`,
- `vertex_noise`,
- `density_imbalance`.

It evaluates proposal behavior using an analytic box-support field and the M4 protect/prune scorer. This isolates proposal scoring before real scene integration.

Outputs:

```text
metrics.json
metrics.csv
table_by_damage_type.csv
failure_cases.md
```

The report generator writes a markdown summary from `metrics.json`.

## 3. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
```

Smoke output:

```text
rows: 4
floater_recall: 1.0
```

The smoke verified:

- synthetic damage generation,
- metrics JSON/CSV creation,
- markdown report generation,
- floater prune recall reaches 1.0 in the controlled floater case,
- valid surface protect recall is at least 0.9 in the floater case.

## 4. Metric Separation

`metrics.json` separates:

```text
inference_time_metrics
oracle_analysis_metrics
gt_dependent_eval_metrics
```

Synthetic labels are used only for evaluation metrics such as floater precision/recall and valid surface protect recall.

## 5. Known Limitations

- This first benchmark uses analytic box support rather than Stage-3 posterior fields.
- v0.7, v0.8.2, Stage-3, Stage-4, and Stage-5 baselines are not yet integrated into the benchmark matrix.
- Chamfer metrics are intentionally absent in the first pass because M6 focuses on proposal behavior.
- Part-specific damage is deferred until part/symmetry confidence exists.

## 6. Stage Gate

| Gate | Result |
|---|---|
| Synthetic damage generation works | PASS |
| Protect/prune identifies floaters | PASS |
| Protect/prune preserves valid surface triangles in smoke | PASS |
| Report separates inference-time and oracle/eval metrics | PASS |
| Markdown report generation works | PASS |

Decision: `PASS`. The next allowed stage is M7 conservative snap.
