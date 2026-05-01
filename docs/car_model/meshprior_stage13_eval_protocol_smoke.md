# MeshPrior Stage 13 Evaluation Protocol Smoke

Date: 2026-05-01

## Commands

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage13_eval_protocol.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_experiment_matrix.py --dry_run --group all --no_train --eval_only --output_dir outputs/carnet/meshprior/experiment_matrix
micromamba run -n mesh_splatting python scripts/car_model/meshprior_make_neurips_report.py --matrix_results outputs/carnet/meshprior/experiment_matrix/matrix_results.json --output_dir outputs/carnet/meshprior/reports --report_dir docs/car_model/reports
```

## Smoke Result

`smoke_test_meshprior_stage13_eval_protocol.py`: `PASS`

Smoke matrix:

- `total=6`
- `available=2`
- `missing=4`

Full dry-run matrix:

- `total=11`
- `available=7`
- `missing=4`

## Generated Artifacts

- `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`
- `docs/car_model/reports/meshprior_neurips_main_report.md`
- `outputs/carnet/meshprior/reports/object_table.csv`
- `outputs/carnet/meshprior/reports/synthetic_damage_table.csv`
- `outputs/carnet/meshprior/reports/scene_table.csv`
- `outputs/carnet/meshprior/reports/ablation_table.csv`
- `outputs/carnet/meshprior/reports/failure_cases.md`

## Missing Handling Check

The smoke test requires at least one available row and at least one missing row. The full dry-run report lists these missing rows in `failure_cases.md`:

- `v0_7_residual_baseline`
- `spcarnet_stage4_map_refinement`
- `spcarnet_stage5_oracle_k8`
- `protect_prune_proposals`

## Gate

M13 smoke gate: `PASS`.
