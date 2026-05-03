# MeshSplatOpt Stage R5 Reversible Edits Design

Date: 2026-05-02

## Goal

Implement a unified reversible edit abstraction for MeshSplatOpt surgeries. R5 operates on generic numpy mesh arrays first. Integration with Mesh Splatting checkpoints is deferred to later stages.

## Mesh State

The minimal mesh state is:

- `vertices: float64 [N, 3]`
- `faces: int64 [M, 3]`
- optional `attributes: dict`

## Edit Types

Supported edit types:

- `PROTECT`: no geometry mutation; records protection intent.
- `DELETE_TRIANGLES`: removes affected faces.
- `EDGE_COLLAPSE`: rewrites one vertex id to another and removes degenerate faces.
- `FACE_MERGE`: removes redundant affected faces in a conservative merge placeholder.
- `SNAP_VERTICES`: updates selected vertex positions.
- `SPLIT_TRIANGLES`: inserts centroids and replaces selected faces with three child faces.
- `FILL_PATCH`: appends patch vertices/faces.
- `APPEARANCE_RESET`: no geometry mutation; records appearance reset intent for later checkpoint integration.

All edits are reversible through snapshots. R5 does not claim semantic correctness; it guarantees contract, audit, rollback, and mesh integrity behavior.

## Required Functions

- `create_snapshot(mesh_or_state)`
- `apply_edit(mesh_or_state, edit)`
- `rollback_edit(mesh_or_state, snapshot)`
- `verify_mesh_integrity(mesh_or_state)`
- `summarize_topology_delta(before, after)`

## Snapshot Policy

Snapshots are `.npz` files containing vertices, faces, attributes JSON, and a checksum. Rollback restores exact arrays from the snapshot.

## Gate

`PASS` requires smoke coverage for:

1. delete triangles and exact rollback;
2. snap vertices and exact rollback;
3. fill patch and exact rollback;
4. edge collapse or face merge valid face indices;
5. integrity checker catches degenerate faces and invalid indices.
