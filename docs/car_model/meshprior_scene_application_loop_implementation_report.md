# MeshPrior Scene Application Loop Implementation Report

Date: 2026-05-01

## Scope

This bridge closes the gap between dry-run scene gates and a recoverable applied mesh copy. It does not overwrite real scene models and does not launch full training.

## Files Added

- `ss3dm_prior/meshprior/apply_proposals.py`
- `scripts/car_model/meshprior_apply_accepted_proposals.py`
- `scripts/car_model/smoke_test_meshprior_scene_application.py`
- `docs/car_model/meshprior_scene_application_loop_design.md`

## Behavior

The applicator:

- reads proposal rows with `before_npz` and `after_npz`;
- reads a scene gate report;
- applies only accepted proposals;
- saves rollback NPZ before every accepted proposal;
- writes an applied mesh copy;
- records initial/final mesh stats;
- writes an optional recovery/evaluation command plan.

It intentionally treats proposal after-state as an applied mesh copy, not as an in-place edit to a mesh-splatting checkpoint.

## Synthetic Application Result

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_apply_accepted_proposals.py --proposals outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/proposals/proposals.json --gate_report outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/scene_gate/gate_report.json --output_dir outputs/carnet/meshprior/scene_application/m11_synthetic_apply --write_recovery_plan --scene_source outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun --recovery_model outputs/carnet/meshprior/scene_application/m11_synthetic_apply/recovery_model
```

Result:

- status: `PASS`
- accepted proposals: `1`
- applied proposals: `1`
- rejected proposals: `0`
- initial mesh: `8` vertices, `10` faces
- final mesh: `9` vertices, `14` faces
- warnings: none

Artifacts:

- `outputs/carnet/meshprior/scene_application/m11_synthetic_apply/application_manifest.json`
- `outputs/carnet/meshprior/scene_application/m11_synthetic_apply/application_report.md`
- `outputs/carnet/meshprior/scene_application/m11_synthetic_apply/applied_mesh.npz`
- `outputs/carnet/meshprior/scene_application/m11_synthetic_apply/rollback/0000_synthetic_fill_0000_rollback.npz`
- `outputs/carnet/meshprior/scene_application/m11_synthetic_apply/recovery_commands.sh`

## Remaining Blocker

The next step would be applying this bridge to a real scene mesh/checkpoint and running recovery optimization with wandb. That requires choosing the target scene/model copy and GPU. It should not be started silently because it can consume substantial GPU time and produce a new experimental branch of outputs.

## Gate

Scene application bridge gate: `PASS`.

The system can now move from accepted dry-run proposals to an auditable applied mesh copy with rollback.
