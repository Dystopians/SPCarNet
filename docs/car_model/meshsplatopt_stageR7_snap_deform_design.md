# MeshSplatOpt Stage R7 Snap/Deform Design

Date: 2026-05-02

## Goal

Generate safe snap/deform proposals for dents, rough surfaces, floaters, vehicle discontinuities, and ground/wall misalignment. R7 creates proposals only; counterfactual acceptance comes later.

## Snap Targets

Implemented in R7:

- robust local/global plane target for nearly planar surfaces;
- local fairing target through plane projection;
- step sizes `0.1`, `0.25`, `0.5`;
- maximum displacement cap by scene scale;
- support flag to prevent unsupported floater attachment.

Later stages can add sparse COLMAP point-to-plane, object-prior surfaces, and wall/ground semantic targets.

## Safety Rules

- Boundary vertices are not moved aggressively.
- Unsupported small components are not snapped to the main surface.
- Displacements are capped by scene scale.
- Every proposal records uncertainty and evidence source.
- Proposals are ordinary `SNAP_VERTICES` edits and are rollback-compatible through R5.

## Outputs

- `snap_proposals.json`
- `snap_summary.csv`
- `snap_debug_before_after.ply`

## Gate

`PASS` requires:

1. synthetic dented plane snap reduces surface error;
2. synthetic floater is not incorrectly attached without support;
3. synthetic wall/ground misalignment snap reduces plane residual;
4. rollback restores exact original arrays.
