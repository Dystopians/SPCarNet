# MeshSplatOpt Stage R8 Giant Void Fill Design

Date: 2026-05-02

## Goal

Implement large-hole and giant-ground-void fill proposals as first-class reversible operations. R8 generates proposals and certificates; later stages will counterfactually validate them.

## Fill Modes

Implemented:

- `boundary_loop_fill`: detect boundary loops and triangulate a planar loop with a centroid fan.
- `ground_plane_void_fill`: fit/use local ground plane and generate a grid patch for rectangular voids.
- `prior_supported_fill`: emit diagnostic prior-only proposals only when explicitly allowed.

Contract placeholders:

- `depth_guided_patch_fill`: reserved for render/depth evidence integration.
- vehicle object-prior fill: deferred to R9.

## Certificate Fields

Every fill proposal records:

- boundary loop support;
- neighboring surface support;
- sparse depth support;
- free-space risk;
- semantic/ground/object support;
- camera coverage score;
- prior-only flag;
- expected topology cost;
- expected area repaired.

## Safety Rules

- `UNKNOWN_UNOBSERVED_VOID` is not filled in normal mode.
- Diagnostic prior-only fill must mark `prior_only_flag=true`.
- Degenerate boundary loops are rejected.
- Fill proposals are `FILL_PATCH` edits and rollback-compatible.

## Gate

`PASS` requires:

1. small plane hole closes;
2. giant rectangular parking-ground void produces a valid patch;
3. unknown void rejects in normal mode;
4. diagnostic prior-only mode proposes and marks prior-only;
5. rollback restores original mesh;
6. degenerate boundary loop rejects.
