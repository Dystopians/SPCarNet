# MeshPrior Parking Image Region Mining Report

Date: 2026-05-01

## Scope

This step adds image/COLMAP ROI mining for `parking_phone_tiny`. It is a bridge from scene-level masks to MeshPrior proposal candidates. It does not yet apply object priors or edit scene geometry.

## Files Added

- `scripts/car_model/meshprior_mine_parking_image_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_image_regions.py`

## Inputs

Scene view:

```text
outputs/carnet/meshprior/parking_phone_tiny/dataset_view
```

Inputs used:

- `images`
- `sparse/0/images.bin`
- `sparse/0/points3D.bin`
- `segmentation_dense/*.png`
- `ground_masks/*.png`

The segmentation and ground masks are binary. The miner uses connected components from segmentation masks, filters by size/ground overlap, and attaches COLMAP sparse 3D support from image observations inside the component bbox.

## Commands

Smoke:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_image_regions.py
```

Full mining:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_mine_parking_image_regions.py --scene_root outputs/carnet/meshprior/parking_phone_tiny/dataset_view --output_dir outputs/carnet/meshprior/parking_phone_tiny/image_region_mining --min_area_px 2000 --max_components_per_image 8
```

## Outputs

- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_region_mining_report.md`

## Result

Full run:

- images considered: `425`
- candidate regions: `340`
- eligible candidates: `273`
- median sparse point count: `4`
- median mask area fraction: `0.0030437403549382716`
- max ground overlap among eligible candidates: `0.25251004016064255`

Top candidates have strong sparse support, for example:

| region | image | bbox | mask px | ground overlap | sparse points | score |
|---|---|---|---:|---:|---:|---:|
| `images_00187_roi_000` | `images_00187.jpg` | `[527, 238, 748, 359]` | `24568` | `0.0` | `268` | `1.0` |
| `images_00184_roi_000` | `images_00184.jpg` | `[355, 171, 576, 289]` | `23640` | `0.0` | `252` | `1.0` |
| `images_00937_roi_000` | `images_00937.jpg` | `[622, 507, 908, 751]` | `35862` | `0.0` | `248` | `1.0` |

## Interpretation

This is a candidate generator, not a final acceptance mechanism.

The output is intentionally permissive enough to avoid missing vehicles. It should be followed by:

- multi-view clustering of repeated 2D detections;
- 3D region consolidation;
- object posterior/retrieval scoring;
- scene evidence gates;
- rollback-capable application.

The high eligible count is acceptable at this stage because no geometry is edited from these ROIs directly.

## Gate

Parking image region mining gate: `PASS`.

The scene now has a reproducible ROI candidate source tied to segmentation masks, ground masks, and COLMAP sparse support.
