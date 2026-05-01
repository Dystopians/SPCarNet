# MeshPrior Stage 2 Design — Scene/Object Region Mining

| Field | Value |
|---|---|
| Stage | M2 / region mining |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | `meshprior_stage1_scene_meshprior_RFC.md` |

## 1. Available Scene Representations

This repository exposes several scene-level artifacts and tools that can feed region mining:

| Representation | Existing path / tool | Use in M2 |
|---|---|---|
| Trained mesh-splatting model directory | `models/...`, `train.py`, `render.py` | Source of optimized scene state and exported geometry. |
| PLY mesh exports | `create_ply.py`, `mesh.py`, `train/ours_<iter>/fuse*.ply`, `point_cloud/iteration_*/` | Primary geometry input when available. |
| COLMAP scene | `scene/dataset_readers.py`, `scene/colmap_loader.py`, `utils/read_write_model.py` | Source cameras and sparse points for later gates; M2 records path only. |
| Strict split | `create_colmap_outoftrain_split.py` | Required for fair later scene metrics, not consumed directly by M2. |
| Geometry eval | `evaluate_geometry_colmap.py` | Later scene gate input; M2 does not run it. |
| PRISM stats / pruning | `utils/prism_*`, `scripts/parking_ground/*` | Later consumer of protect/prune scores. |
| Segmentation artifacts | `segmentation/*`, possible mask/json outputs | Optional object-region hints if present. |
| Ground masks / ground association | `utils/ground_*`, `scripts/parking_ground/*` | Negative evidence: avoid applying car priors to ground. |

M2 must work before full PRISM integration and before reliable segmentation is available.

## 2. Minimal Input Contract

The region miner accepts:

```text
--scene_model <path_to_model_or_scene_output>
--scene_source <path_to_colmap_scene>
--output_dir outputs/carnet/meshprior/region_mining/<run_name>
--mode dry_run
```

Minimal usable input is either:

1. a PLY mesh path,
2. a model/output directory containing a discoverable PLY,
3. no mesh at all, in which case dry-run output must be explicit and non-crashing.

Output contract:

```text
regions.json
regions_summary.csv
region_mining_report.md
```

`regions.json` stores a `RegionMiningResult` with region records and run-level notes. It must be sufficient for M3 posterior inference to either process candidate regions or fail clearly with "no regions".

## 3. Region Data Model

M2 introduces these contracts in `ss3dm_prior/meshprior/region_types.py`:

- `SceneMeshRegion`: region id, source mesh path, face indices, bbox, centroid, surface area, component id, and heuristic scores.
- `RegionEvidence`: segmentation, geometry, ground, observed-support, and car-likeness evidence.
- `ObjectCanonicalization`: transform metadata for later canonicalization; M2 may leave this as identity/unknown with confidence.
- `RegionMiningResult`: list of regions plus global input/output metadata and notes.

All contracts are JSON-serializable.

## 4. Avoiding Ground / Wall / Vegetation

M2 is conservative:

- A region is not declared "car" solely because it is a connected component.
- Ground-like regions are downweighted when bbox height is very small relative to horizontal extent.
- Tall thin regions are downweighted as wall/pole/vegetation-like.
- Low confidence regions remain in diagnostics but should not be passed to SP-CarNet unless `car_likeness_score >= threshold`.
- If segmentation artifacts exist, they may raise confidence, but absence of segmentation cannot force a positive car label.

The default output includes all mined components but marks `eligible_for_posterior=false` for low-confidence regions.

## 5. Confidence to Pass to SP-CarNet

Default requirements for `eligible_for_posterior=true`:

- region has faces and finite bbox,
- triangle count is nonzero,
- bbox dimensions are plausible for a compact object after normalization,
- `car_likeness_score >= 0.35` in M2 dry heuristic,
- region is not strongly ground-like.

This threshold is intentionally permissive for later smoke tests but must be treated as provisional. M9 scene gates remain authoritative.

## 6. Fallback Without Segmentation

If masks or object boxes are unavailable:

1. load the scene mesh if possible,
2. split by connected components,
3. compute component diagnostics,
4. assign heuristic car-likeness from bbox aspect ratio, height, area, and compactness,
5. emit a report warning that no semantic segmentation was used.

If no mesh is available:

1. emit `regions.json` with an empty region list,
2. emit `regions_summary.csv` with headers only,
3. emit `region_mining_report.md` explaining dry-run/no-mesh status,
4. exit with code 0 in `--mode dry_run`.

## 7. Stage Gate

M2 passes only if:

- package imports cleanly,
- smoke test creates a synthetic two-component mesh and returns nonempty regions,
- dry-run with missing data exits cleanly,
- output artifacts are written,
- implementation report, smoke doc, and research-log entry are written.

M2 does not run SP-CarNet posterior inference and does not modify scene geometry.
