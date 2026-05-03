# MeshSplatOpt Stage R6 Topology Baselines Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R6 implements strong topology-reduction baselines and smoke-tests delete, boundary-protected delete, and collapse/merge-style reductions on a synthetic mesh.

## Files Added

- `ss3dm_prior/meshsplatopt/topology_baselines.py`
- `scripts/car_model/meshsplatopt_run_topology_baselines.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR6_topology_baselines.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Implemented Baselines

- `prism_score_topk_delete`
- `random_same_count_delete`
- `low_visibility_delete`
- `boundary_protected_delete`
- `qem_style_edge_collapse`
- `planar_face_merge`
- `external_simplification` JSON contract placeholder

The external simplification row is explicitly marked invalid/missing in R6 unless a later stage wires in a concrete trimesh or pymeshlab simplifier.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR6_topology_baselines.py
```

Smoke result:

- input faces: `36`
- `prism_score_topk_delete` hits budgets `27` and `18` faces;
- `boundary_protected_delete` hits budgets `27` and `18` faces;
- `qem_style_edge_collapse` produces valid meshes at both budgets;
- `planar_face_merge` hits budgets `27` and `18` faces.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/topology_baseline_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/baselines/topology_baseline_runs.json`
- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/baselines/topology_baseline_table.csv`
- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/baselines/topology_baseline_report.md`
- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/baselines/meshes/`

## Decision

`PASS`. At least delete, boundary-protected delete, and collapse/merge-style baselines run on synthetic data with valid mesh outputs.
