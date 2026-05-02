# MeshPrior Parking Cluster Proposal Scoring Report

Date: 2026-05-01

## Scope

This step converts consolidated multi-view parking scene clusters into MeshPrior proposal metadata for downstream scene gates. It does not edit scene geometry.

The scored proposals are intentionally metadata-only because the current parking scene bridge has image/COLMAP/cluster evidence but does not yet extract an editable local mesh patch with stable face IDs for each vehicle-region cluster.

## Inputs

- Consolidated clusters: `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json`
- Source dataset view: `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- Prior steps:
  - image/COLMAP ROI mining: `340` image-region candidates
  - 3D consolidation: `17` clusters, `9` eligible for proposal scoring

## Implementation

Added:

- `scripts/car_model/meshprior_score_parking_clusters.py`
- `scripts/car_model/smoke_test_meshprior_parking_cluster_scoring.py`

The scorer emits five proposal types per eligible cluster:

- `protect`
- `prune`
- `snap_candidate`
- `fill_candidate`
- `uncertainty`

Each proposal contains:

- `face_indices: []`
- `metadata.metadata_only: true`
- `metadata.requires_mesh_extraction: true`
- `metadata.requires_scene_gate: true`
- the source cluster payload and decomposed score terms

This keeps the proposal contract compatible with later scene-gate and application stages while preventing accidental geometry edits before mesh extraction exists.

## Full Run

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_score_parking_clusters.py --consolidated_regions outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json --output_dir outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals
```

Outputs:

- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_scores.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_report.md`

Metrics:

- clusters scored: `9`
- proposals emitted: `45`
- proposal types per cluster: `5`
- metadata-only proposals: `45`

Top cluster scores:

| cluster | views | sparse points | support | protect | fill | uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `parking_region_0000` | 32 | 3851 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `parking_region_0002` | 31 | 1653 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `parking_region_0001` | 20 | 2203 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `parking_region_0003` | 14 | 1028 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `parking_region_0006` | 11 | 548 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `parking_region_0005` | 8 | 355 | 0.9233 | 0.9502 | 0.9655 | 0.0767 |
| `parking_region_0008` | 2 | 39 | 0.5200 | 0.6880 | 0.6507 | 0.4800 |

## Verification

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_parking_cluster_scoring.py
```

Result: PASS.

Smoke test checks:

- `3` clusters scored under `--max_clusters 3`
- `15` proposals emitted
- all expected proposal types are present
- all proposals are metadata-only
- JSON, CSV, and Markdown outputs are written

## Gate

Stage gate: PASS.

Reasoning:

- The proposal metadata exists and is deterministic.
- The scoring step is verified by smoke test and compileall.
- It does not claim geometry modification.
- It preserves a hard boundary for the next stage: scene mesh extraction and gate validation are required before any proposal can edit a mesh.

## Next Step

The next high-priority task is a metadata proposal gate that ranks and filters these cluster proposals into an explicit action plan:

- accept `protect` / `fill_candidate` only for high-support multi-view clusters
- keep `prune` disabled or low-priority unless scene evidence becomes weak
- route low-support clusters to `uncertainty`
- emit `requires_mesh_extraction` actions for accepted candidates

After that, the missing bridge is local scene mesh patch extraction with stable face IDs.
