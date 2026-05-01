# MeshPrior Pre-M14 Stability Audit

Date: 2026-05-01

## Scope

This audit was run before starting M14 to check whether M0-M13 work had any immediate collapse risk.

## Finding 1: Smoke subprocesses used ambient Python

Status: fixed.

Several MeshPrior smoke tests launched helper scripts with the literal command `python`. In this workspace that can resolve to a Python without project dependencies such as `numpy`, causing the smoke regression suite to fail before testing the actual code.

Fixed files:

- `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py`
- `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py`
- `scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py`
- `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py`

All now use `sys.executable` for subprocess calls.

## Finding 2: Generated outputs are useful but not a source of truth

Status: documented risk, no code change required.

M13 intentionally tolerates missing generated outputs by marking rows as `MISSING`. A fresh checkout without local `outputs/carnet/...` artifacts will not reproduce the exact `7 available / 4 missing` report, but it should still generate tables and preserve missing rows.

The committed registry and report generator are the source of truth. Ignored generated CSV/JSON files are diagnostic artifacts.

## Finding 3: Scene-level MeshPrior evidence is still dry-run

Status: known research limitation, not a stability blocker.

The current work proves conservative proposal generation, gating, rollback, calibration, and report plumbing. It does not yet prove real render-gated MeshPrior insertion improves scene metrics. M14 should treat this as a claim-risk item rather than a completed headline result.

## Verification

Compilation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
```

Result: PASS.

MeshPrior smoke regression:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage2_region_mining.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage7_snap.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage8_fill.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage13_eval_protocol.py
```

Result: PASS for all listed tests.

M13 matrix/report dry-run:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_run_experiment_matrix.py --dry_run --group all --no_train --eval_only --output_dir outputs/carnet/meshprior/experiment_matrix_pre_m14_audit
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_make_neurips_report.py --matrix_results outputs/carnet/meshprior/experiment_matrix_pre_m14_audit/matrix_results.json --output_dir outputs/carnet/meshprior/reports_pre_m14_audit --report_dir /tmp/meshprior_pre_m14_report_docs
```

Result:

- Matrix `total=11`
- `available=7`
- `missing=4`
- Report generation PASS.

## Decision

Pre-M14 stability gate: PASS.

Known remaining risks are research/claim risks, not immediate code-collapse risks:

- real render-gated MeshPrior insertion is not implemented;
- full scene training evidence is still a 200-iteration smoke, not a headline run;
- several historical baseline rows remain `MISSING` until their artifacts are regenerated or linked.
