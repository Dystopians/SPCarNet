# MeshSplatOpt R23.01 Residual-Aware Boundary Fill Selector Report

Date: 2026-05-03

## Objective

R22 showed that boundary-loop `FILL_PATCH` is gate-safe and trainable, but naive centroid fan fill fails medium recovery. R23 fixes the selector-side weakness by ranking boundary loops using train-view render residual evidence rather than geometry area alone.

## Implementation

Updated:

`scripts/car_model/meshsplatopt_select_checkpoint_boundary_fill_edit.py`

New selector mode:

```bash
--rank residual
```

When enabled, the selector:

- loads train/test residual maps from `model_path/<set>/ours_<iteration>`;
- auto-infers `camera_index_offset`;
- projects each candidate boundary loop into high-residual views;
- samples RGB residual at loop vertices;
- ranks by `mean_loop_residual * sqrt(loop_area)`.

This mirrors the R18/R19 residual-snap selector protocol and avoids using held-out test residuals for inference-time proposal selection.

## Parking Result

Command output:

`outputs/carnet/meshsplatopt/stageR23_01_parking_residual_boundary_fill_selection`

Selection:

- status: `PASS`
- loop count: `48858`
- candidates after filtering: `4545`
- selected loop index: `46134`
- selected loop vertices: `6`
- selected area: `24.723803`
- train residual score: `0.387146`
- rank score: `1.925007`
- camera index offset: `54`
- inserted vertices: `1`
- inserted faces: `6`

The residual-aware selector chose the same loop as R22's largest-area filtered candidate, but now with explicit explanation-debt evidence. This improves auditability but does not by itself fix the medium-run failure. The remaining weakness is fill geometry: centroid fan placement is too crude and needs local fairing/depth-aware placement.

## Decision

`RESIDUAL_BOUNDARY_FILL_SELECTOR_PASS_GEOMETRY_STILL_WEAK`

Selector evidence is now aligned with CSEF explanation debt, but the fill primitive still needs a better geometric construction before more long runs are justified.
