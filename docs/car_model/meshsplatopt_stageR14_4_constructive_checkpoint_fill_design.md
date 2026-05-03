# MeshSplatOpt Stage R14.4 Constructive Checkpoint Fill Design

Date: 2026-05-02

## Goal

Add a limited but real `FILL_PATCH` adapter for Mesh Splatting checkpoints so constructive repair proposals can be materialized before teacher recovery.

## Initialization Policy

For each inserted vertex:

- position comes from the MeshSplatOpt fill edit;
- `vertex_weight`, `features_dc`, and `features_rest` are copied from the nearest existing checkpoint vertex;
- face-level fields for new triangles are initialized conservatively to zero.

This is not final appearance optimization. It is a safe initialization for subsequent teacher recovery.

## Still Deferred

- `SPLIT_TRIANGLES`;
- `EDGE_COLLAPSE`;
- `FACE_MERGE`;
- optimizer state preservation for resumed training.

## Gate

`PASS` requires the checkpoint adapter smoke to append fill vertices/faces and preserve checkpoint schema validity.
