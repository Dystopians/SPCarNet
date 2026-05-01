# MeshPrior Stage 2 Region Mining — Implementation Report

| Field | Value |
|---|---|
| Stage | M2 / region mining |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage2_region_mining_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/__init__.py` | MeshPrior package entrypoint. |
| `ss3dm_prior/meshprior/region_types.py` | JSON-serialisable contracts for mined scene regions, evidence, canonicalization, and result bundles. |
| `scripts/car_model/meshprior_mine_regions.py` | CLI and implementation for dry-run region mining from PLY meshes or empty inputs. |
| `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py` | Synthetic two-component mesh smoke test. |

## 2. Implementation Summary

The region miner now supports:

- direct `.ply` input or model/output directories containing `.ply` files,
- geometry-only connected-component mining,
- optional segmentation artifact discovery,
- robust dry-run behavior when no mesh or segmentation exists,
- per-region diagnostics:
  - triangle count,
  - vertex count,
  - bounding box,
  - centroid,
  - surface area,
  - vertex density,
  - boundary edge count,
  - approximate hole-boundary score,
  - heuristic car-likeness,
  - posterior eligibility flag,
  - provisional canonicalization metadata.

Outputs:

```text
regions.json
regions_summary.csv
region_mining_report.md
```

Safety choices:

- The miner does not invoke SP-CarNet posterior inference.
- The miner does not modify scene geometry.
- Segmentation absence does not crash or create false semantic certainty.
- Very small components with fewer than 4 triangles remain in diagnostics but are not marked `eligible_for_posterior`.

## 3. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage2_region_mining.py
micromamba run -n mesh_splatting python scripts/car_model/meshprior_mine_regions.py \
  --scene_model /tmp/meshprior_missing_model \
  --scene_source /tmp/meshprior_missing_scene \
  --output_dir /tmp/meshprior_stage2_dry_run \
  --mode dry_run
```

Results:

- `compileall`: PASS.
- Synthetic smoke: PASS, `regions=2`, `eligible_for_posterior=1`.
- Missing-data dry-run: PASS, `regions=0`, `eligible_for_posterior=0`, output artifacts written.

No real PLY was found under local `models` or `outputs` during this stage, so the real-scene path remains unexercised until a scene mesh export is available.

## 4. Output Contract

`regions.json` is the authoritative artifact for M3. Each region includes:

- `face_indices`,
- geometry diagnostics,
- `evidence`,
- `canonicalization`,
- `eligible_for_posterior`.

M3 must respect `eligible_for_posterior=false` and either skip the region or report why it is being processed under an override.

## 5. Known Limitations

- Region mining is currently geometry-only unless segmentation artifacts are already present.
- Car-likeness is a coarse bbox/area heuristic, not a trained detector.
- Canonicalization is provisional and records `front_axis_unknown`; M3 must refine orientation or lower posterior confidence.
- PLY face attributes are not preserved.
- Large scene meshes may need streaming or chunking later; M2 prioritizes correctness and contract stability.

## 6. Stage Gate

| Gate | Result |
|---|---|
| Smoke test passes | PASS |
| Dry-run without data exits cleanly | PASS |
| Synthetic mesh produces nonempty regions | PASS |
| No geometry is modified | PASS |
| Outputs match contract | PASS |

Decision: `PASS`. Proceed to M3 only after this report, smoke doc, and research-log entry are present.
