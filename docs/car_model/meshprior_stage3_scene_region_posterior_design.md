# MeshPrior Stage 3 Design — Scene Region Posterior Inference

| Field | Value |
|---|---|
| Stage | M3 / scene-region posterior inference |
| Date | 2026-05-01 |
| Status | DESIGN |
| Predecessor | M2 region mining |

## 1. Goal

M3 converts a mined scene mesh region into an SP-CarNet object-posterior diagnostic bundle:

```text
scene mesh region -> sampled region points -> canonical frame -> q(z | O) -> field diagnostics
```

It does not generate mesh repair proposals. Proposal generation starts in M4.

## 2. Sampling Region Point Clouds

Input comes from `regions.json`. Each `SceneMeshRegion` contains:

- `source_mesh_path`,
- `face_indices`,
- bbox and evidence diagnostics,
- `eligible_for_posterior`.

The wrapper loads the source PLY and samples points from the selected triangles with area-proportional face sampling and barycentric coordinates. If the mesh has degenerate faces, sampling falls back to region vertices.

Default sample count is `768`, matching Stage-3 `partial_observed_points`.

## 3. Canonicalization

M3 uses a conservative `bbox_pca` canonical transform:

- center = sampled point mean,
- rotation = PCA eigenvectors, right-handed,
- scale = max radius after centering and rotation,
- canonical points = `(points - center) @ rotation / scale`.

The front axis is still uncertain. The transform records:

- mode,
- center,
- scale,
- rotation,
- confidence,
- notes.

Low orientation confidence must reduce later proposal confidence. M3 does not pretend to know the true vehicle front axis.

## 4. Loading Stage-3 Posterior Checkpoints

The expected checkpoint is:

```text
outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt
```

The local M0 audit confirmed this checkpoint exists. Its schema includes:

- `encoder_state_dict`,
- `decoder_state_dict`,
- `model_cfg`,
- `decoder_finetune_enabled`,
- `stage2_latent_table`,
- `stage2_object_id_to_row`.

The decoder config is loaded from the Stage-2 checkpoint referenced by `model_cfg["decoder_checkpoint"]`. If this path is absent, M3 tries a sibling or user-provided fallback only if explicitly added later. Missing checkpoint paths are reported as clear failures.

## 5. Posterior Uncertainty Without Oracle GT

M3 exposes inference-time uncertainty only:

- posterior mean norm,
- posterior logvar mean,
- posterior variance mean,
- latent uncertainty score `sqrt(mean(exp(logvar)))`,
- optional K-sample latent spread from `q(z|O)`,
- coarse field occupancy ratio on a 32^3 grid.

No clean mesh or clean point cloud is used to choose a posterior.

## 6. Shape Field Exposure

For each region M3 writes:

```text
z_mean.npy
z_logvar.npy
canonical_transform.json
posterior_summary.json
sampled_region_points.npy
occupancy_grid_32.npy
```

If Marching Cubes is available, `posterior_summary.json` also includes:

- `extraction_success`,
- `vertex_count`,
- `face_count`,
- `watertight`.

These diagnostics become inputs to M4 protect/prune scoring.

## 7. Failure Behavior

M3 must fail clearly when:

- `regions.json` is missing,
- checkpoint is missing,
- checkpoint schema is unsupported,
- a region references a missing source mesh.

If no eligible regions exist, M3 writes an empty summary and exits successfully.

## 8. Stage Gate

M3 passes if:

- imports and `compileall` pass,
- smoke test passes,
- missing-checkpoint path fails gracefully with a clear message,
- with the local Stage-3 checkpoint, at least one synthetic region produces posterior artifacts.
