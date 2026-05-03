# MeshSplatOpt Stage R8 Giant Void Fill Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R8 implements reversible fill proposal generation for boundary loops, ground-plane voids, and diagnostic prior-only unknown voids.

## Files Added

- `ss3dm_prior/meshsplatopt/hole_fill.py`
- `ss3dm_prior/meshsplatopt/ground_void_fill.py`
- `scripts/car_model/meshsplatopt_make_fill_proposals.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR8_giant_void_fill.py`

Updated:

- `ss3dm_prior/meshsplatopt/edit_apply.py` to support global-index fill faces.
- `ss3dm_prior/meshsplatopt/__init__.py`

## Behavior

Implemented fill modes:

- `boundary_loop_fill`: detects boundary loops and fills with a planar centroid fan.
- `ground_plane_void_fill`: creates a ground-plane grid patch and records a certificate.
- `prior_supported_fill`: available only through `allow_prior_only=True` and marked with `prior_only_flag=true`.

Unknown unobserved voids are rejected in normal mode.

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR8_giant_void_fill.py
```

Smoke result:

- small-hole boundary count: `20 -> 4`
- giant ground void patch: valid
- unknown void normal mode: rejected
- diagnostic prior-only proposal: emitted with `prior_only_flag=true`
- rollback: exact
- degenerate boundary: rejected

## Artifacts

- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_smoke_report.json`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_outputs/fill_proposals.json`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_outputs/fill_summary.csv`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_outputs/fill_certificate_report.md`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_outputs/fill_debug_before.ply`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/fill_outputs/fill_debug_after.ply`

## Decision

`PASS`. Giant ground void synthetic repair works, and unknown voids are not silently filled in normal mode.
