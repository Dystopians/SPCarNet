# MeshSplatOpt Stage R14.2 Real Checkpoint Dry-Run Design

Date: 2026-05-02

## Goal

Run a non-training dry-run on an existing real Mesh Splatting checkpoint to verify that R14.1 adapter outputs can be placed into a normal model directory layout and evaluated later by `render.py + metrics.py`.

## Scope

R14.2 applies a low-risk `DELETE_TRIANGLES` edit to a checkpoint copy only. It does not claim method improvement and does not run medium training.

## Outputs

- edited checkpoint copy under `model/point_cloud/iteration_<iter>/point_cloud_state_dict.pt`;
- edit JSON;
- checkpoint edit report;
- render/metrics/geometry command plan.

## Gate

`PASS` requires:

1. real checkpoint input schema valid;
2. edited checkpoint copy schema valid;
3. normal model directory layout created;
4. exact render and metrics commands written.
