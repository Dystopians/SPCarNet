# MeshSplatOpt Stage R3 CSEF Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

The implementation builds CSEF diagnostics without modifying geometry and the synthetic smoke separates normal surface, hole/debt region, and floater component.

## Files Added

- `ss3dm_prior/meshsplatopt/__init__.py`
- `ss3dm_prior/meshsplatopt/csef_types.py`
- `ss3dm_prior/meshsplatopt/csef_builder.py`
- `scripts/car_model/meshsplatopt_build_csef.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR3_csef.py`

## Behavior

The builder loads a mesh, samples one CSEF sample per face, groups connected components, computes boundary-edge scores, area/topology cost, placeholder support/free-space fields, explanation debt, and uncertainty.

Current evidence sources are intentionally conservative:

- mesh topology;
- boundary edges;
- connected components;
- triangle area;
- optional external-evidence availability flag.

Sparse/image/render evidence hooks are placeholders in R3. They are recorded as empty refs rather than invented measurements.

## Artifacts

Smoke artifacts:

- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/synthetic_hole_floater_dent.ply`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/csef/csef_samples.npz`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/csef/csef_regions.json`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/csef/csef_summary.csv`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/csef/csef_report.md`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/stageR3_csef_smoke_report.json`

CLI check artifacts:

- `outputs/carnet/meshsplatopt/stageR3_csef_cli_check/`

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR3_csef.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_build_csef.py \
  --mesh_path outputs/carnet/meshsplatopt/stageR3_csef_smoke/synthetic_hole_floater_dent.ply \
  --output_dir outputs/carnet/meshsplatopt/stageR3_csef_cli_check \
  --scene_model synthetic_cli \
  --scene_source smoke
```

Environment note: the default `python` interpreter compiles the repository but does not have `numpy` installed for runtime smoke execution. The project micromamba environment was used for the R3 smoke, matching the historical training/evaluation environment.

## Decision

`PASS`. R4 can consume CSEF regions and sample scalars to mine defect regions.
