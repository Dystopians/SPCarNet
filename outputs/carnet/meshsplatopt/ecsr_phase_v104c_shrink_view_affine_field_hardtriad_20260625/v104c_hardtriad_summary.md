# v104c Shrink View-Affine Hard-Triad Summary

Date: 2026-06-25

Status: complete for `counter`, `kitchen`, and `bonsai`. The policy is fixed across scenes: `min_count=1`, `min_views=1`, `ridge=0.001`, `residual_clip=0.08`, `view_std_floor=1e-4`, `rank_rtol=1e-7`, `condition_max=1e8`, `fallback_mode=shrink`.

## Verdict

v104c is a real representation-field improvement over v104a on the hard triad. It improves every RGB metric on every scene relative to v104a, while preserving the earlier wins over clean MeshSplatting and v103.

It is still below the v101/v102a endpoint ceiling. The correct claim is therefore: fixed-policy shrink view-affine field improves the baked surface-field line, but it does not yet replace the stronger endpoint/delta-bank ceiling.

## Mean Metrics

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 |
| v103 affine min_count=1 | 28.384418 | 0.879855 | 0.226611 |
| v104a raw view-affine | 28.823045 | 0.884927 | 0.219492 |
| v104c shrink view-affine | 28.859798 | 0.885459 | 0.219064 |
| v101/v102a endpoint ceiling | 30.167397 | 0.913355 | 0.163709 |

## Mean Deltas

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c minus clean | +1.037945 | +0.007156 | -0.017830 |
| v104c minus v103 | +0.475380 | +0.005604 | -0.007547 |
| v104c minus v104a | +0.036753 | +0.000532 | -0.000427 |
| v104c minus v101/v102a | -1.307599 | -0.027896 | +0.055355 |

## Per-Scene v104c Metrics

| scene | PSNR | SSIM | LPIPS | dPSNR vs v104a | dSSIM vs v104a | dLPIPS vs v104a |
|---|---:|---:|---:|---:|---:|---:|
| counter | 27.498068 | 0.867420 | 0.238986 | +0.005690 | +0.000076 | -0.000017 |
| kitchen | 28.770449 | 0.881590 | 0.188021 | +0.005157 | +0.000062 | -0.000076 |
| bonsai | 30.310877 | 0.907367 | 0.230186 | +0.099411 | +0.001457 | -0.001189 |

## Field Diagnostics

| scene | valid triangles | accumulated pixels | shrink alpha mean | fallback triangles | build manifest sec | render sec |
|---|---:|---:|---:|---:|---:|---:|
| counter | 2716449 | 48475638 | 0.566197 | 0 | 573.721 | 49.784 |
| kitchen | 3076129 | 56628492 | 0.630734 | 0 | 1271.106 | 54.091 |
| bonsai | 3405888 | 59912816 | 0.583769 | 0 | 1290.648 | 60.663 |

All render reports set `no_test_gt_used_for_policy=true` and consume `v102_surface_residual_field:<field path>` through `render.py`.

## Artifact Paths

| scene | field manifest | render report | results |
|---|---|---|---|
| counter | `/dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/counter/v104c_shrink_view_affine_min1_minviews1_field.manifest.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_counter/render_py_endpoint_report.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json` |
| kitchen | `/dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/kitchen/v104c_shrink_view_affine_min1_minviews1_field.manifest.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/kitchen/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_kitchen/render_py_endpoint_report.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/kitchen/detached_model/results.json` |
| bonsai | `/dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/bonsai/v104c_shrink_view_affine_min1_minviews1_field.manifest.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/bonsai/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_bonsai/render_py_endpoint_report.json` | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/bonsai/detached_model/results.json` |

## Commands

Build command template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 /usr/bin/time -f 'v104c <scene> build wall %e sec' \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v104b_centered_view_affine_residual_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/<scene>/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/<scene>/v104c_shrink_view_affine_min1_minviews1_field.pt \
  --renderer_scaling 4 --residual_dtype float16 --min_count 1 --min_views 1 \
  --ridge 0.001 --residual_clip 0.08 --view_std_floor 1e-4 \
  --rank_rtol 1e-7 --condition_max 1e8 --fallback_mode shrink --chunk_pixels 262144
```

Render and eval use `render.py` with `--checkpoint_endpoint_surface_field_path <field>` and `scripts/car_model/evaluate_render_split_metrics.py --merge_model_results`.

Logs:

```text
/tmp/spcarnet_logs/v104c_counter_build.log
/tmp/spcarnet_logs/v104c_counter_render.log
/tmp/spcarnet_logs/v104c_counter_eval.log
/tmp/spcarnet_logs/v104c_kitchen_build.log
/tmp/spcarnet_logs/v104c_kitchen_render.log
/tmp/spcarnet_logs/v104c_kitchen_eval.log
/tmp/spcarnet_logs/v104c_bonsai_build.log
/tmp/spcarnet_logs/v104c_bonsai_render.log
/tmp/spcarnet_logs/v104c_bonsai_eval.log
```
