# MeshPrior Parking Phone Tiny Scene Audit

Date: 2026-05-01

## Dataset

Selected scene source:

```text
/data/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix
```

This is preferred over the earlier VGGT video smoke data because it is a real parking-scale scene with COLMAP, undistorted images, segmentation masks, ground masks, dense stereo artifacts, and an existing out-of-train split.

## Prepared View

Prepared with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_prepare_parking_scene.py --overwrite
```

View:

```text
outputs/carnet/meshprior/parking_phone_tiny/dataset_view
```

View contents are symlinks to the external dataset:

- `images`
- `sparse/0`
- `ground_masks`
- `segmentation_dense`
- `normals`

## Audit Result

- images: `425`
- COLMAP images: `425`
- missing image files: `0`
- extra image files: `0`
- `cameras.bin`: present
- `images.bin`: present
- `points3D.bin`: present
- `points3D.ply`: present
- `split_outoftrain_v1.json`: present
- dense segmentation masks: `425`
- ground masks: `425`

Audit status: `PASS`.

## Role in MeshPrior

This scene should become the first real scene benchmark for:

- baseline scene training;
- vehicle/ground-aware region mining;
- MeshPrior proposal generation;
- scene gate acceptance/rejection;
- recovery optimization;
- baseline-vs-gated-MeshPrior scene metrics.
