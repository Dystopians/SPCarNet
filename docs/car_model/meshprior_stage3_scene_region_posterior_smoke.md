# MeshPrior Stage 3 Scene-Region Posterior — Smoke Report

| Field | Value |
|---|---|
| Stage | M3 / scene-region posterior smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Commands

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py
```

The smoke test:

1. creates a synthetic box PLY,
2. runs M2 region mining,
3. verifies missing-checkpoint failure behavior,
4. runs M3 posterior inference on CPU using `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt`,
5. checks that `z_mean.npy`, `canonical_transform.json`, and `posterior_summary.json` are written.

## Output Summary

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

## Gate Verdict

`PASS`
