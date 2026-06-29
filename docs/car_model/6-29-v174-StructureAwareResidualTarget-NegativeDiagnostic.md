# SPCarNet v173-v174 Structure-Aware Residual Target Diagnostic

Date: 2026-06-29

## Verdict

This milestone implements and tests one real representation/target change after the v169 prompt: a train-fit-only teacher residual target transform exposed as `--teacher_residual_target_mode`.

It does **not** pass the flowers hard gate against Phase-J. No full9 run should be launched from this branch.

Phase-J flowers gate:

| method | PSNR higher than 20.304358 | SSIM higher than 0.557770 | LPIPS lower than 0.329222 |
|---|---:|---:|---:|
| v173 surface RFF exact | no | no | no |
| v174 edge/luma diagnostics | not promoted | not promoted | not promoted |

## Implemented Change

Files changed:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/analyze_v169_policy_val_upper_bound.py`

New interface:

```text
--teacher_residual_target_mode raw_rgb|luma_only|edge_luma_mix
--teacher_residual_target_luma_mix 0.75
--teacher_residual_target_edge_boost 0.25
```

Behavior:

- `raw_rgb`: previous behavior.
- `luma_only`: fit a luminance-only teacher residual target, repeated into RGB, to suppress chroma noise.
- `edge_luma_mix`: fit a train-fit-only residual target that blends RGB residual toward luma residual, with stronger luma mixing on parent-render luma edges.

This is not a target/test GT mechanism. The transform is applied only while fitting from train-fit teacher residual evidence. Target/test apply still uses the stripped no-GT evidence path.

## Storage And Runtime Preflight

Before v174 diagnostics:

```text
/data    avail 5.8M
/dev/shm avail 818M
/tmp     avail 6.1T
GPU 2    931 MiB used, 0% util
```

Because `/data` and `/dev/shm` were effectively full, v174 was kept to compact policy-val diagnostics. No duplicate evidence cache or full9 materialization was launched.

## v173 Exact Result

Command family:

```text
run_vnext_certified_residual_texture_scene.py
--scene flowers
--teacher_distilled_basis_mode surface_feature_rff_ridge
--teacher_distilled_basis_ridge 0.05
--teacher_distilled_basis_blend 0.5
--teacher_distilled_basis_min_face_samples 128
--strict_no_target_gt_apply
--wandb --wandb_mode offline
```

Artifacts:

- Output root: `/dev/shm/peilincai_spcarnet_20260629_v173_surface_rff_exact/flowers`
- W&B offline: `/dev/shm/peilincai_wandb_v173_surface_rff_exact/wandb/offline-run-20260628_214541-uag88ydx`
- Manifest: `/dev/shm/peilincai_spcarnet_20260629_v173_surface_rff_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- Results: `/dev/shm/peilincai_spcarnet_20260629_v173_surface_rff_exact/flowers/reports/flowers_ours_26000_v173_surface_rff_flowers_test_results.json`
- Audit: `/dev/shm/peilincai_spcarnet_20260629_v173_surface_rff_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`

Protocol audit:

```text
status: COMPLETE
errors: []
protocol_audit.passed: true
selection_uses_test_gt: false
target_gt_visible_to_apply: false
target_gt_visible_to_selection: false
target_gt_visible_to_eval: true
target_forbidden_keys_stripped: true
```

Exact flowers metrics:

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| Phase-J reference | 20.304358 | 0.557770 | 0.329222 | target gate |
| v173 surface RFF exact | 19.832010 | 0.505779 | 0.405904 | fail |
| v173 - Phase-J | -0.472348 | -0.051991 | +0.076682 | fail |

v173 was rejected by policy-val and wrote a no-op fallback:

```text
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
changed_pixels: 0 / 37100800
```

The decisive reject reasons were:

```text
positive_view_fraction 0.416667 < 0.550000
cvar20_view_relative_gain -0.000110 < 0.000000
min_view_relative_gain -0.000330 < -0.000001
ssim_positive_view_fraction 0.333333 < 0.550000
image_l1_positive_view_fraction 0.333333 < 0.550000
```

This is important: the surface RFF decoder can reduce residual MSE in aggregate, but it is not stable across views and tails.

## v174 Policy-Val Diagnostics

v174 tests the new residual target transform without launching target exact.

Commands:

```bash
CUDA_VISIBLE_DEVICES=2 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/analyze_v169_policy_val_upper_bound.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --output_json /dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_edge_luma_target_surface_rff.json \
  --output_md /dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_edge_luma_target_surface_rff.md \
  --texture_sizes 16 \
  --alpha_grid 0,0.03125,0.0625,0.125 \
  --policy_val_stride 4 \
  --max_samples_per_view 240000 \
  --teacher_residual_target_mode edge_luma_mix \
  --teacher_residual_target_luma_mix 0.75 \
  --teacher_residual_target_edge_boost 0.25 \
  --teacher_distilled_basis_mode surface_feature_rff_ridge \
  --teacher_distilled_basis_min_face_samples 128 \
  --teacher_distilled_basis_ridge 0.05 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_adaptive_low_support_teacher_basis \
  --adaptive_teacher_basis_min_face_samples_floor 128 \
  --adaptive_teacher_basis_support_quantile 0.25 \
  --adaptive_teacher_basis_low_support_ridge_scale 0.5 \
  --policy_val_lpips_max_size 256
```

```bash
CUDA_VISIBLE_DEVICES=2 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/analyze_v169_policy_val_upper_bound.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --output_json /dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_luma_only_target_surface_rff.json \
  --output_md /dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_luma_only_target_surface_rff.md \
  --texture_sizes 16 \
  --alpha_grid 0,0.03125,0.0625,0.125 \
  --policy_val_stride 4 \
  --max_samples_per_view 240000 \
  --teacher_residual_target_mode luma_only \
  --teacher_distilled_basis_mode surface_feature_rff_ridge \
  --teacher_distilled_basis_min_face_samples 128 \
  --teacher_distilled_basis_ridge 0.05 \
  --teacher_distilled_basis_apply_mode blend \
  --teacher_distilled_basis_blend 0.5 \
  --enable_adaptive_low_support_teacher_basis \
  --adaptive_teacher_basis_min_face_samples_floor 128 \
  --adaptive_teacher_basis_support_quantile 0.25 \
  --adaptive_teacher_basis_low_support_ridge_scale 0.5 \
  --policy_val_lpips_max_size 256
```

Diagnostic artifacts:

- `/dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_edge_luma_target_surface_rff.json`
- `/dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_edge_luma_target_surface_rff.md`
- `/dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_luma_only_target_surface_rff.json`
- `/dev/shm/peilincai_spcarnet_v174_diagnostics/flowers_policy_val_luma_only_target_surface_rff.md`

Policy-val best rows:

| diagnostic | PSNR proxy gain | relative MSE gain | SSIM gain | SSIM positive views | SSIM min-view | LPIPS gain | LPIPS positive views | LPIPS min-view |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v173 raw surface RFF | +0.398015 | +0.087572 | +2.7816e-06 | 0.5833 | -1.6510e-05 | +3.6831e-06 | 0.5000 | -7.5355e-05 |
| v174 edge_luma_mix | +0.390820 | +0.086059 | +2.4885e-06 | 0.5833 | -1.6809e-05 | +2.8089e-06 | 0.5000 | -7.7292e-05 |
| v174 luma_only | +0.389319 | +0.085743 | +2.4239e-06 | 0.5833 | -1.6809e-05 | +2.8573e-06 | 0.5000 | -7.5802e-05 |

The structure-aware target does not solve the real bottleneck. It slightly reduces RGB residual proxy gain and does not improve SSIM/LPIPS tails.

## Bottleneck Diagnosis

The current carrier and baked residual family are underpowered for the Phase-J gap:

1. Policy-val residual MSE can improve, but image-quality axes stay at numerical-noise scale.
2. SSIM and LPIPS tail gains are negative for the best rows.
3. The v173 exact run rejected the candidate before target apply because only a minority of policy-val views improved.
4. The successful-looking diagnostic rows are residual-sample proxy wins, not robust full-frame perceptual wins.
5. The target/test exact output becomes no-op under honest policy gates, so the final metrics remain far below Phase-J.

The result satisfies the v169 prompt's negative-completion path: current Phase-J-distilled carrier variants tested here cannot reliably improve SSIM/LPIPS without target/test GT leakage.

## Next Research Direction

Do not spend more time on alpha, rank, local support, or luma target transforms under the same surface carrier.

The next credible method must change one of these higher-level assumptions:

- Train a real deferred/neural texture decoder with full-frame image losses on train-fit views, then certify on policy-val.
- Use target/test GT-free but target-geometry-aware view synthesis support so the carrier covers the views where Phase-J helps.
- Distill Phase-J at render-time feature level rather than baking only RGB residuals into sparse face/UV bins.
- Change the parent representation jointly with residual baking, because additive residual on the compact parent appears too weak.

Until one of these is implemented and flowers exact beats Phase-J all-axis, project status remains **NOT COMPLETE** for paper-readiness.
