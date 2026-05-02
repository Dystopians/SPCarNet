# MeshPrior Parking Region Consolidation Report

Date: 2026-05-01

## Scope

This step consolidates parking image ROI candidates into coarse 3D vehicle-region candidates. It reduces repeated 2D detections and produces stable region targets for later proposal scoring.

## Files Added

- `scripts/car_model/meshprior_cluster_parking_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_region_consolidation.py`

## Inputs

```text
outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json
```

## Method

The clustering uses eligible image ROI candidates with enough COLMAP sparse support:

- filter by `eligible_for_posterior=true`;
- require at least `8` sparse points;
- cluster by 3D centroid within radius `0.35`;
- consolidate member ROI IDs, image names, sparse point counts, 3D bbox, and confidence.

The output remains proposal-candidate metadata. It is not editable mesh geometry.

## Commands

Smoke:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_region_consolidation.py
```

Full consolidation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_cluster_parking_regions.py --image_regions outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json --output_dir outputs/carnet/meshprior/parking_phone_tiny/region_consolidation --cluster_radius 0.35 --min_sparse_points 8
```

## Outputs

- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidation_report.md`

## Result

- input image ROI regions: `340`
- eligible inputs used after sparse-support filtering: `140`
- consolidated 3D clusters: `17`
- eligible clusters: `9`

Top clusters:

| cluster | members | views | sparse points | confidence | bbox extent |
|---|---:|---:|---:|---:|---|
| `parking_region_0000` | `32` | `32` | `3851` | `1.0` | `[0.302, 0.177, 0.385]` |
| `parking_region_0002` | `31` | `31` | `1653` | `1.0` | `[1.848, 0.534, 0.405]` |
| `parking_region_0001` | `20` | `20` | `2203` | `1.0` | `[0.868, 0.378, 0.566]` |
| `parking_region_0003` | `14` | `14` | `1028` | `1.0` | `[0.250, 0.099, 0.131]` |
| `parking_region_0006` | `11` | `11` | `548` | `1.0` | `[0.070, 0.129, 0.151]` |

## Interpretation

This is a meaningful narrowing step:

- single-image ROI candidates: `340`;
- eligible ROI candidates: `273`;
- sparse-supported ROI candidates used for clustering: `140`;
- consolidated proposal targets: `17`;
- high-confidence proposal targets: `9`.

The high-confidence clusters now provide a realistic target list for object-prior/retrieval scoring. Before geometry application, they still need proposal generation and scene gates.

## Gate

Parking region consolidation gate: `PASS`.

The scene now has stable multi-view 3D region candidates for the next proposal-scoring step.
