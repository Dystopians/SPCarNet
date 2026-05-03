# MeshSplatOpt Stage R14.1 Checkpoint Adapter Design

Date: 2026-05-02

## Goal

Unblock R14 by adding a conservative adapter between MeshSplatOpt generic mesh edits and Mesh Splatting `point_cloud_state_dict.pt` checkpoints.

## Checkpoint Fields

Observed checkpoint keys:

- `triangles_points`: vertex positions `[V, 3]`;
- `_triangle_indices`: triangle indices `[F, 3]`;
- `vertex_weight`: per-vertex weights `[V, 1]`;
- `features_dc`: per-vertex SH DC features `[V, 1, 3]`;
- `features_rest`: per-vertex SH rest features `[V, 15, 3]`;
- `importance_score`: per-face score `[F]`;
- `image_size`: per-face scalar `[F]`;
- `pixel_count`: per-face scalar `[F]`;
- `sigma`;
- `active_sh_degree`.

## R14.1 Scope

Supported checkpoint edits:

- `DELETE_TRIANGLES`: remove faces and per-face arrays, keep vertex arrays intact.
- `SNAP_VERTICES`: update vertex positions only.
- `PROTECT` and `APPEARANCE_RESET`: metadata-only no-op for checkpoint geometry.

Deferred edits:

- `FILL_PATCH`;
- `SPLIT_TRIANGLES`;
- `EDGE_COLLAPSE`;
- `FACE_MERGE`.

Reason: these require robust initialization or remapping of per-vertex radiance attributes and optimizer state. R14.1 must not fabricate appearance attributes.

## Outputs

- copied `point_cloud_state_dict.pt`;
- `checkpoint_edit_report.json`;
- `checkpoint_edit_report.md`.

## Gate

`PASS` requires synthetic checkpoint smoke to:

1. delete faces and update per-face arrays consistently;
2. snap vertices;
3. reject fill with a clear reason;
4. keep checkpoint load/save schema valid.
