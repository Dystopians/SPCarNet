# Expression-Layer Geometry Optimization (Planar Ground Merge)

## Goal

This change adds a conservative expression-layer geometry optimizer that merges large near-planar regions (such as ground) into a simpler triangle representation while keeping region boundaries protected.

The target is to reduce triangle/vertex count **inside valid planar regions** without crossing semantic/geometric boundaries.

---

## What Was Changed

### 1) `scene/triangle_model.py`

Added:

- `TriangleModel.optimize_ground_planar_patches(...)`
- `TriangleModel._rebuild_optimizer_after_topology_change()`

Main behavior of `optimize_ground_planar_patches`:

1. Select candidate triangles by up-axis normal constraint (`max_ground_tilt_deg`).
2. Build edge-based adjacency among candidates.
3. Grow connected regions with local normal-consistency gating (`max_neighbor_normal_deg`).
4. For each valid region:
   - Require minimum triangle count and area.
   - Fit a plane via SVD.
   - Reject region if plane residual is too high (`max_plane_residual`).
   - Detect boundary edges/vertices.
   - Keep boundary vertices fixed.
   - Snap **interior** vertices on a planar grid (`snap_cell_size`) to merge topology conservatively.
5. Remap triangles, remove degenerate faces, deduplicate faces, compact vertices.
6. Rebuild optimizer parameter groups for the new topology.

Why this is expression-layer:

- The optimization modifies `TriangleModel` tensors (`vertices`, `_triangle_indices`, SH features, weights).
- Output is saved back into checkpoint format (`point_cloud_state_dict.pt`), not only post-export mesh.

### 2) New script: `optimize_expression_geometry.py`

A runnable entrypoint to apply the optimization on an existing trained iteration:

- Loads scene/triangles from checkpoint
- Runs planar expression optimization
- Saves to target iteration id
- Supports a `parking_lot` preset and near-field filtering

---

## Safety / No-Cross-Boundary Mechanisms

The implementation uses conservative controls:

- Region growing only through shared edges.
- Neighbor merging requires normal consistency and local height continuity.
- Region must pass global plane residual test.
- Boundary vertices are not moved.
- Only interior vertices are snapped.
- Topology is compacted after removing degenerates, no external expansion.

This is designed to avoid out-of-region merging and preserve boundaries.

---

## How To Run

Example:

```bash
python optimize_expression_geometry.py \
  -s /path/to/scene \
  -i images \
  -m /path/to/model \
  --iteration 30000 \
  --save_iteration 30001 \
  --up_axis y \
  --max_ground_tilt_deg 18 \
  --max_neighbor_normal_deg 12 \
  --max_neighbor_height_delta 0.08 \
  --min_region_triangles 300 \
  --min_region_area 0.2 \
  --max_plane_residual 0.015 \
  --snap_cell_size 0.03 \
  --near_field_radius 12
```

Notes:

- `--save_iteration 30001` is recommended to preserve the original checkpoint.
- Increase `snap_cell_size` for stronger simplification; decrease for safer detail retention.
- `--up_axis auto` can avoid wrong world-up assumptions.

---

## Tuning Guide

- `max_ground_tilt_deg`: larger -> more triangles considered "ground-like".
- `max_neighbor_normal_deg`: larger -> bigger connected regions.
- `max_neighbor_height_delta`: larger -> easier to merge over local vertical variations.
- `max_plane_residual`: larger -> allow more non-ideal planar patches.
- `residual_quantile`: robustness of residual gate (e.g. 0.95 ignores worst 5% outliers).
- `snap_cell_size`: larger -> more vertex collapse and stronger simplification.
- `near_field_radius`: restrict optimization to near-ego region (recommended for parking scenes).
- `min_region_triangles` / `min_region_area`: filter tiny patches.

Recommended first pass:

- `max_ground_tilt_deg=15~20`
- `max_neighbor_normal_deg=10~15`
- `max_plane_residual=0.01~0.02`
- `snap_cell_size=0.02~0.05`

---

## Known Limitations

1. Current region descriptor uses geometry-only cues (normals + planarity), not semantic labels.
2. Snapping is conservative but may still oversimplify subtle surface undulations if `snap_cell_size` is too high.
3. Optimizer state is rebuilt (momentum reset) after topology change.
4. This version is targeted at offline/stepwise optimization, not high-frequency in-loop invocation.

---

## Suggested Future Improvements

1. **Residual-aware merging**
   - Add depth/RGB residual EMA per triangle and use it as a merge veto signal.
   - Keep high-error regions dense, simplify only low-error planar regions.

2. **Near-field strict policy**
   - Add distance-to-ego weighting and tighter residual thresholds in near field.
   - Keep far-field more aggressively simplified.

3. **Boundary classifier**
   - Add explicit edge protection from depth discontinuity and color gradient.
   - Prevent accidental merging across object boundaries.

4. **Constrained retriangulation per planar patch**
   - Replace grid snapping with explicit constrained triangulation of patch polygon.
   - Better triangle quality and stronger control over final face count.

5. **In-training staged optimization**
   - Trigger this pass only after densification stabilizes.
   - Follow with short fine-tuning to recover appearance consistency.

6. **Metrics and guardrails**
   - Track face reduction ratio, Chamfer / normal error, boundary leakage rate.
   - Auto-rollback if geometry quality regresses beyond threshold.

---

## Quick Validation Checklist

After optimization:

1. Compare `triangles_before/after` and `vertices_before/after` stats.
2. Render train/test views and verify boundary integrity.
3. Export mesh and compare coverage + detail in near field.
4. Run quantitative reconstruction metrics if available.
