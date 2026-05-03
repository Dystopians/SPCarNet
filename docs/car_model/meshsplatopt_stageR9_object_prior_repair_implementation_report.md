# MeshSplatOpt Stage R9 Object-Prior Repair Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R9 adds an optional object-prior proposal generator that cannot bypass scene gates.

## Files Added

- `ss3dm_prior/meshsplatopt/object_prior_repair.py`
- `scripts/car_model/meshsplatopt_make_object_repair_proposals.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR9_object_prior_repair.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Existing Files Read

- `ss3dm_prior/meshprior/region_types.py`
- `ss3dm_prior/meshprior/scene_region_posterior.py`
- `ss3dm_prior/meshprior/protect_prune.py`
- `ss3dm_prior/meshprior/optimizer_adapter.py`
- `docs/car_model/meshprior_stage1_scene_meshprior_RFC.md`

## Behavior

The module emits bounded object-prior proposals:

- vehicle protect mask;
- vehicle surface snap candidate;
- vehicle discontinuity fill candidate when canonicalization confidence is high and posterior uncertainty is low.

Uncertain priors emit only protect metadata. Every proposal records:

- `prior_proposes_evidence_disposes=true`
- `requires_scene_counterfactual_validation=true`

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR9_object_prior_repair.py
```

Smoke checks:

- confident synthetic vehicle package has protect proposal;
- confident synthetic vehicle package has fill proposal;
- uncertain prior has no fill proposal;
- uncertain prior is limited to protect;
- all proposals require scene gates.

## Artifacts

- `outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke/object_prior_repair_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke/object_repair_outputs/object_repair_proposals.json`
- `outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke/object_repair_outputs/object_repair_summary.csv`
- `outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke/object_repair_outputs/object_repair_report.md`

## Decision

`PASS`. Object-prior proposals are bounded and cannot bypass scene validation.
