# MeshPrior Parking Mesh Patch Extraction Report

Date: 2026-05-01

## Scope

This step extracts local triangle mesh patches for the parking metadata-gate targets. It copies triangles from the trained parking baseline checkpoint and preserves original face and vertex indices for later rollback/gate work.

No source model geometry is modified.

## Inputs

- Action plan: `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json`
- Consolidated clusters: `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json`
- Triangle checkpoint: `outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt`

The checkpoint contains:

- vertices: `193491`
- triangles: `64497`

## Implementation

Added:

- `scripts/car_model/meshprior_extract_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_extraction.py`

The extractor:

1. loads `triangles_points` and `_triangle_indices` from the training checkpoint;
2. computes triangle centroids;
3. expands each gated cluster bbox by `0.5`;
4. selects triangles whose centroids fall inside the expanded bbox;
5. writes a compact patch with local vertices/faces plus `original_face_indices` and `original_vertex_indices`.

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_extract_parking_mesh_patches.py --action_plan outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json --consolidated_regions outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json --triangle_state outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt --output_dir outputs/carnet/meshprior/parking_phone_tiny/mesh_patches
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/patches/*.npz`

Results:

- patches: `8`
- nonempty patches: `8`
- total patch faces: `10826`
- patch face range: `97` - `3902`
- geometry edited: `false`

Patch summary:

| region | proposal types | faces | vertices | views | sparse points |
| --- | --- | ---: | ---: | ---: | ---: |
| `parking_region_0000` | `protect`, `snap_candidate`, `fill_candidate` | 2262 | 6786 | 32 | 3851 |
| `parking_region_0001` | `protect`, `snap_candidate`, `fill_candidate` | 2587 | 7761 | 20 | 2203 |
| `parking_region_0003` | `protect`, `snap_candidate`, `fill_candidate` | 398 | 1194 | 14 | 1028 |
| `parking_region_0006` | `protect`, `snap_candidate`, `fill_candidate` | 229 | 687 | 11 | 548 |
| `parking_region_0004` | `protect`, `snap_candidate`, `fill_candidate` | 671 | 2013 | 8 | 851 |
| `parking_region_0002` | `protect`, `snap_candidate`, `fill_candidate` | 3902 | 11706 | 31 | 1653 |
| `parking_region_0005` | `protect`, `snap_candidate`, `fill_candidate` | 680 | 2040 | 8 | 355 |
| `parking_region_0007` | `protect`, `snap_candidate`, `fill_candidate` | 97 | 291 | 6 | 124 |

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_mesh_patch_extraction.py
```

Result: PASS.

Smoke checks:

- `8` patches are emitted;
- every patch is nonempty;
- patch arrays contain 3D vertices and triangular faces;
- original face indices align with compact faces;
- JSON, CSV, Markdown, and NPZ outputs are written.

## Gate

Stage gate: PASS.

This closes the metadata-to-local-mesh gap for the parking baseline. The next step can be a dry-run patch-level before/after gate using no-op/protect candidates first, then a conservative synthetic edit inside a copied patch only if the no-op/protect gate is stable.
