# MeshPrior Stage 3 Scene-Region Posterior — Implementation Report

| Field | Value |
|---|---|
| Stage | M3 / scene-region posterior inference |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage3_scene_region_posterior_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/scene_region_posterior.py` | Sampling, canonicalization, checkpoint loading, posterior inference, field decoding, and uncertainty utilities. |
| `scripts/car_model/meshprior_infer_region_posterior.py` | CLI for processing M2 `regions.json` into per-region posterior artifacts. |
| `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py` | Synthetic region posterior smoke test with missing-checkpoint failure check. |
| `docs/car_model/meshprior_stage3_scene_region_posterior_design.md` | Stage design. |

## 2. Implementation Summary

The M3 wrapper now:

- loads mined regions from `regions.json`,
- skips ineligible regions unless `--include_ineligible` is set,
- loads source PLY meshes,
- samples region points area-proportionally from triangle faces,
- estimates a `bbox_pca` canonical transform,
- canonicalizes sampled points,
- loads the Stage-3 posterior encoder and frozen Stage-2 decoder from checkpoint schema,
- runs deterministic posterior inference with `sample=False`,
- writes per-region latent and diagnostic artifacts,
- computes a coarse occupancy grid,
- attempts Marching-Cubes extraction for diagnostics.

Per-region outputs:

```text
z_mean.npy
z_logvar.npy
canonical_transform.json
posterior_summary.json
sampled_region_points.npy
occupancy_grid_<res>.npy
```

Run-level output:

```text
posterior_index.json
```

## 3. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py
```

Smoke output summary:

```text
ok_regions: 1
region_id: region_0000
field_occupancy_ratio: 0.070068359375
posterior_mu_norm: 2.8356223106384277
posterior_logvar_mean: -3.9360544681549072
uncertainty_score: 0.1464938074350357
latent_sample_l2_mean: 2.3861324787139893
extraction_success: true
vertex_count: 461
face_count: 926
watertight: true
```

The smoke also intentionally ran the CLI with a missing checkpoint and verified that it fails clearly with `posterior_checkpoint not found`.

## 4. Inference-Time Metrics

M3 writes only inference-time diagnostics:

- posterior mean norm,
- posterior logvar mean,
- posterior variance mean,
- uncertainty score,
- latent sample spread,
- field occupancy ratio,
- mesh extraction success.

It does not use clean ground truth to choose or rank a region.

## 5. Known Limitations

- `bbox_pca` canonicalization has sign/front-axis ambiguity.
- Region orientation confidence is diagnostic only; M4 must treat low confidence as a penalty.
- The wrapper loads the full checkpoint per CLI process, not once per long-running service.
- Occupancy grid resolution defaults to 32; smoke used 16 for speed.
- The method currently processes PLY-backed regions only.

## 6. Stage Gate

| Gate | Result |
|---|---|
| Imports / compileall pass | PASS |
| Smoke test passes | PASS |
| Missing checkpoint fails clearly | PASS |
| Local Stage-3 checkpoint produces at least one posterior artifact | PASS |
| Outputs are inference-time only | PASS |

Decision: `PASS`. The next allowed stage is M4 protect/prune proposal generation.
