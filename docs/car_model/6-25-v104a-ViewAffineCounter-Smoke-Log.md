# v104a View-Affine Counter Smoke Log

Date: 2026-06-25

Status: counter smoke passed. This is not hard-triad validation, not full9 validation, and not final paper completion.

## 0. Verdict

v104a adds a linear view-direction basis on top of the v103 face-local affine barycentric field:

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z]
```

On counter, v104a improves all three metrics over v103 `min_count=1`:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean counter reference | `26.751774` | `0.862055` | `0.252003` |
| v103 affine min_count=1 | `27.208200` | `0.863405` | `0.243176` |
| v104a view-affine min_count=1 | `27.492378` | `0.867344` | `0.239003` |
| v101/v102a endpoint ceiling | `28.442907` | `0.893696` | `0.186557` |

Delta:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104a minus clean | `+0.740604` | `+0.005288` | `-0.013000` |
| v104a minus v103 | `+0.284178` | `+0.003939` | `-0.004173` |
| v104a minus v101/v102a | `-0.950529` | `-0.026352` | `+0.052446` |

Interpretation:

```text
View direction is a useful residual-field feature.
v104a closes a meaningful part of the v103-to-v101 counter gap.
It is still below the v101/v102a ceiling and must pass hard-triad before promotion.
```

## 1. Implementation

Files:

```text
render.py
scripts/car_model/build_v104_view_affine_residual_field.py
```

New field basis:

```text
basis_type = affine_barycentric_viewdir
basis_order = [1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z]
coefficient_layout = triangle,basis,rgb
```

Render-time behavior:

1. Get visible triangle ids from `rend_ids`.
2. Compute barycentric basis from projected triangle vertices.
3. Compute triangle-center view direction from the current camera center.
4. Evaluate `residual_rgb = basis @ triangle_coefficients`.
5. Apply `residual_clip=0.08`, then clamp final RGB.

## 2. Counter Artifacts

Field:

```text
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.pt
```

Manifest:

```text
/dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.manifest.json
```

Render report:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/test/ours_26000_v104a_view_affine_min1_counter/render_py_endpoint_report.json
```

Metrics:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json
```

Field summary:

| item | value |
|---|---:|
| valid triangles | `2,716,436 / 9,644,247` |
| valid triangle fraction | `0.281663865` |
| accumulated pixels | `48,475,697` |
| build elapsed | `447.0349 sec` |
| field size | `433 MiB` |
| solve failures | `0` |
| ridge | `0.001` |
| residual clip | `0.08` |

Render summary:

| item | value |
|---|---:|
| target frames | `30` |
| render elapsed | `50.4928 sec` |
| mean surface valid fraction | `0.999042` |
| mean abs delta | `0.008267` |
| mean changed fraction | `0.998996` |
| intermediate outputs saved | `false` |

## 3. Commands

Build:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v104_view_affine_residual_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/counter/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.pt \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --min_count 1 \
  --ridge 0.001 \
  --residual_clip 0.08 \
  --chunk_pixels 262144
```

Render:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --iteration 26000 \
  --skip_train \
  --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --checkpoint_endpoint_output_method ours_26000_v104a_view_affine_min1_counter \
  --checkpoint_endpoint_surface_field_path /dev/shm/peilincai_spcarnet_v104_view_affine_field_20260625/counter/v104_view_affine_min1_field.pt \
  --checkpoint_endpoint_require_surface_field \
  --checkpoint_endpoint_no_intermediate_outputs \
  --quiet
```

Eval:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --split test \
  --methods ours_26000_v104a_view_affine_min1_counter \
  --merge_model_results
```

## 4. Review Boundary

v104a is deliberately a minimal positive smoke, not the robust final design. A review subagent recommended the following improvements before promotion:

- Use centered per-triangle view features.
- Track `triangle_view_counts` or `min_views`.
- Add condition/rank diagnostics and fallback to v103 affine coefficients.
- Use scale-aware ridge.
- Report fallback/OOD fractions.

Current v104a does not yet implement those stability features. Therefore:

- Do not expand v104a claims beyond counter until hard-triad passes.
- Do not claim unseen-camera generalization.
- Do not claim it is a vanilla MeshSplatting checkpoint.
- Do not claim full9 or paper closure.

## 5. Next Step

Because v104a passes counter smoke, the next action is hard-triad validation on `kitchen` and `bonsai`. If both scenes also improve over v103 and clean, v104a becomes the new best surface-field evidence. If either scene regresses, build v104b with centered view features and fallback.
