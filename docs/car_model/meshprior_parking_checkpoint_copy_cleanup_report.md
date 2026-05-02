# MeshPrior Parking Checkpoint-Copy Cleanup Report

Date: 2026-05-01

## Scope

This step applies accepted copied-patch cleanup candidates to a duplicated triangle checkpoint. It verifies checkpoint writeback bookkeeping without overwriting the baseline model.

The source checkpoint is not modified.

## Inputs

- Patch proposal report: `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json`
- Mesh patch summary: `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- Source checkpoint: `outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt`

## Implementation

Added:

- `scripts/car_model/meshprior_apply_parking_patch_cleanup_to_checkpoint_copy.py`
- `scripts/car_model/smoke_test_meshprior_parking_checkpoint_copy_cleanup.py`

The application script:

1. reads accepted `component_cleanup_candidate` rows from the copied-patch proposal test report;
2. maps removed local patch faces back to original global checkpoint face indices;
3. removes those faces from a copied checkpoint state;
4. compacts vertices and remaps triangle indices;
5. applies the same compaction to per-vertex arrays and per-face arrays.

Updated per-vertex arrays:

- `triangles_points`
- `vertex_weight`
- `features_dc`
- `features_rest`

Updated per-face arrays:

- `importance_score`
- `image_size`
- `pixel_count`

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_apply_parking_patch_cleanup_to_checkpoint_copy.py --patch_proposal_report outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json --mesh_patch_summary outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json --triangle_state outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt --output_dir outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_rows.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.md`

Results:

- cleanup applications: `8`
- unique removed faces: `532`
- faces: `64497` -> `63965`
- vertices: `193491` -> `191895`
- source model edited: `false`
- checkpoint copy edited: `true`

Array integrity:

| array | shape |
| --- | --- |
| `triangles_points` | `(191895, 3)` |
| `_triangle_indices` | `(63965, 3)` |
| `vertex_weight` | `(191895, 1)` |
| `features_dc` | `(191895, 1, 3)` |
| `features_rest` | `(191895, 15, 3)` |
| `importance_score` | `(63965,)` |
| `image_size` | `(63965,)` |
| `pixel_count` | `(63965,)` |

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_checkpoint_copy_cleanup.py
```

Result: PASS.

Smoke checks:

- applies two accepted cleanup candidates to a temporary checkpoint copy;
- source model remains unmodified;
- copied face and vertex counts decrease;
- per-vertex feature arrays match the new vertex count;
- per-face arrays match the new face count.

## Gate

Stage gate: SOFT PASS.

Checkpoint-copy writeback is structurally valid. This remains a soft pass because render/geometry metrics have not yet been run on the copied checkpoint. The next step is to create a recovery model directory around the copied checkpoint and test whether the training/evaluation loaders can resume from it.
