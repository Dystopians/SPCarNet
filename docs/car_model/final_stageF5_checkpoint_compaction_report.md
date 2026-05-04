# Final Stage F5 Checkpoint Compaction Report

Date: 2026-05-04

## Decision

`PASS`.

F5 applies selector candidates to real Mesh Splatting checkpoints, preserves the checkpoint schema, writes a normal model directory layout, and validates the compact checkpoint through a low-resolution `render.py` smoke.

## Implementation

Created:

```text
ss3dm_prior/meshsplatopt/checkpoint_compaction.py
scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py
scripts/car_model/smoke_test_final_stageF5_checkpoint_compaction.py
```

The compactor:

1. loads `point_cloud_state_dict.pt`;
2. removes selected face ids from `compaction_candidates.json` or an inline selector run;
3. remaps vertices;
4. synchronizes known per-vertex fields (`triangles_points`, `vertex_weight`, `features_dc`, `features_rest`);
5. synchronizes known per-face fields (`importance_score`, `image_size`, `pixel_count`);
6. copies `cfg_args`, `cameras.json`, and `input.ply`;
7. writes `topology_audit.json` and `topology_audit.md`;
8. validates invalid indices and degenerate faces.

## Smoke Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_final_stageF5_checkpoint_compaction.py
```

Output:

```text
F5 checkpoint compaction smoke PASS: area_triangles=2564473 csef_triangles=2564473
```

## Parking Clean 22k Area70 Reproduction

| selector | source triangles | output triangles | expected R53 pre-recovery triangles | tolerance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| `area_smallest` | 8,548,242 | 2,564,473 | 2,564,473 | 2 | pass |

Artifact:

```text
outputs/carnet/meshsplatopt/final_stageF5_checkpoint_compaction/parking_area70_repro/model/topology_audit.json
```

## Parking Clean 22k CSEF70 Checkpoint

| selector | source triangles | output triangles | source vertices | output vertices | invalid indices | degenerate faces | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `csef_low_evidence_boundary_protected` | 8,548,242 | 2,564,473 | 2,286,499 | 1,661,616 | 0 | 0 | pass |

Artifact:

```text
outputs/carnet/meshsplatopt/final_stageF5_checkpoint_compaction/parking_csef70_smoke/model/topology_audit.json
```

## Render Smoke

Because all GPUs had high memory occupancy, the renderability check used a low-resolution smoke on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view \
  -m outputs/carnet/meshsplatopt/final_stageF5_checkpoint_compaction/parking_csef70_smoke/model \
  --images images --resolution 16 --eval --iteration 22000 --skip_train
```

The model loaded through `render.py`, reported `2,564,473` triangles and `1,661,616` vertices, and rendered all 54 test views.

## Gate

`PASS`.

Parking `area_smallest` 70 percent reproduces the R53 pre-recovery topology count exactly, and the CSEF selector produces a valid renderable checkpoint.
