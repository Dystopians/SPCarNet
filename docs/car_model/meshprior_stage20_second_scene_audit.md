# MeshPrior Stage 20 Second Scene Audit

Date: 2026-05-01

## Scope

Audit parent-directory datasets for a second real scene suitable for MeshPrior evaluation.

## Directory Scan

Top-level local candidates:

| path | size | assessment |
|---|---:|---|
| `/data/peilincai/parking_phone_tiny_anonymized` | `1.1G` | Valid current parking scene; already used. |
| `/data/peilincai/car_models` | `3.4G` | Object mesh collection, useful for priors, not a scene COLMAP dataset. |
| `/data/peilincai/vggt` | `242M` | Contains examples and a small `data/Video` image/sparse set, but not a supplied parking-lot scene and not appropriate as the second scene for the current paper claim. |
| `/data/peilincai/VideoX-Fun` | `127G` | Video-generation project, no audited parking COLMAP scene found at shallow scan. |
| `/data/peilincai/DriveAGI` | `394G` | Large project directory, no shallow COLMAP parking-scene contract identified in this pass. |

COLMAP-like local hits:

```text
/data/peilincai/parking_phone_tiny_anonymized/colmap/sparse
/data/peilincai/parking_phone_tiny_anonymized/colmap_undistorted/images
/data/peilincai/parking_phone_tiny_anonymized/colmap_undistorted/sparse
/data/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/images
/data/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse
/data/peilincai/vggt/data/Video/v2images
/data/peilincai/vggt/data/Video/sparse
```

The only parking-specific scene is still `parking_phone_tiny_anonymized`.

## Decision

Stage gate: `STOP`.

No second suitable parking-lot COLMAP/image scene is locally available under the parent directory at audit time.

The VGGT example data is intentionally not used as the second MeshPrior scene because it is not the user-provided vehicle/parking target distribution and would weaken the paper story rather than validate it.

## Data Needed From User

Please provide or place one of the following under `/data/peilincai/`:

1. a larger parking-lot COLMAP scene with `images/` and `sparse/0` or equivalent;
2. another vehicle-rich outdoor scene with COLMAP reconstruction;
3. masks or object annotations if available, but they are optional for the first baseline smoke.

Once present, M20 should resume by creating a repo-local dataset view under:

```text
outputs/carnet/meshprior/<scene_name>/dataset_view
```
