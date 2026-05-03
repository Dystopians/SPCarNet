# MeshSplatOpt Stage R10 Generalized Counterfactual Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R10 implements generalized reversible-edit validation for non-delete and delete edits. It does not fake render metrics when no renderable model is provided.

## Files Added

- `ss3dm_prior/meshsplatopt/counterfactual_edit_gate.py`
- `scripts/car_model/meshsplatopt_validate_edit_counterfactual.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR10_counterfactual_edits.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Behavior

The gate snapshots a mesh, applies an edit, evaluates topology integrity and risk/certificate metadata, accepts or rejects, and rolls back rejected edits exactly. The report schema records render/sparse/changed-pixel metric availability as false when those paths are absent.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR10_counterfactual_edits.py
```

Smoke checks:

- good synthetic fill accepted;
- bad floater insertion rejected;
- snap through free space rejected;
- delete supported surface rejected;
- reject rollback exact;
- at least one non-delete edit accepted.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR10_counterfactual_edits_smoke/counterfactual_edits_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR10_counterfactual_edits_smoke/*.npz`

## Decision

`PASS`. R11 can add teacher-guided recovery contracts after accepted edits.
