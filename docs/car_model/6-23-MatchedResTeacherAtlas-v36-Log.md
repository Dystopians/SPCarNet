# v36 Matched-Resolution Teacher Atlas Experiment Log

日期：2026-06-23  
场景：Bonsai, Mip-NeRF360, `images_2` full resolution  
结论：`NOT PROMOTED`

## 1. 目的

v35 的 surface residual region texture adapter 已经证明 atlas 接口可以跑通，但 full-res held-out target changed fraction 只有 `0.0000067`。v36 检查一个更具体的假设：

> v35 是否主要受 train evidence 分辨率 / teacher evidence 不匹配影响？如果把 train teacher evidence 换成 `images_2` full-res，并使用 alpha=1 teacher residual，residual atlas 是否能显著影响 held-out test render？

## 2. 主要输入

Compact parent checkpoint:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model
```

Alpha=1 train teacher renders:

```text
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/train/ours_26000_phaseg_v30_triadic_train_teacher
```

Target held-out evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_target_surface_evidence_images2/bonsai
```

Output root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas
```

## 3. Commands

### 3.1 Build Full-Resolution Train Surface Evidence

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_surface_evidence_cache.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/bonsai \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --iteration 26000 --split train --scene_name bonsai \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_surface_evidence_images2_train96_s2 \
  --max_views 96 --view_stride 2 --view_offset 0 \
  --save_view_npz --save_residual_rgb --save_rgb --save_barycentric \
  --images images_2 --quiet
```

Log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/logs/build_bonsai_surface_evidence_images2_train96_s2_gpu4.log
```

Important note: all `96 / 96` view NPZs were written, but the process was interrupted during the final CPU-side summary/top-support aggregation. Because barycentric is currently materialized late in the builder, only `46 / 96` NPZ files contained barycentric coordinates. A barycentric-only subset was created:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_surface_evidence_images2_train46_s2_baryonly/bonsai/views
```

### 3.2 Build Teacher Surface Evidence

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py \
  --base_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_surface_evidence_images2_train46_s2_baryonly/bonsai \
  --teacher_render_dir /data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/bonsai/recovery_model/train/ours_26000_phaseg_v30_triadic_train_teacher/renders \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_surface_evidence_images2_train46_s2_alpha1 \
  --teacher_parent_delta_min 0.01 \
  --teacher_render_error_margin 0.001 \
  --no-rebuild_top_supports \
  --force
```

Summary:

```text
processed_views: 46
skipped_views: 0
mean_active_fraction: 0.1937276085
mean_target_l1: 0.0056360179
mean_raw_parent_delta_l1: 0.0090240108
mean_positive_teacher_gain_l1: 0.0054351548
```

Report:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_surface_evidence_images2_train46_s2_alpha1/teacher_surface_evidence_report.md
```

### 3.3 Build Render-Visible Region Carriers

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_render_visible_region_carriers.py \
  --scene bonsai \
  --evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_surface_evidence_images2_train46_s2_alpha1 \
  --out_json outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.json \
  --out_md outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.md \
  --evidence_dir_out outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_region_evidence_images2_train46_s2_alpha1 \
  --max_views 46 --view_stride 1 --view_offset 0 \
  --residual_l1_key teacher_residual_l1 \
  --residual_rgb_key teacher_residual_rgb \
  --residual_quantile 0.985 \
  --min_residual_l1 0.02 \
  --min_alpha 0.03 \
  --min_pixels 16 \
  --top_regions_per_view 12 \
  --merge_face_jaccard 0.02 \
  --min_merge_shared_faces 2 \
  --store_region_masks
```

Carrier summary:

```text
carrier_count: 64
raw_region_count: 552
evidence_face_count: 2048
```

### 3.4 Apply Matched-Resolution Region Texture Adapter

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_surface_evidence_images2_train46_s2_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/bonsai_target_surface_evidence_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_render_visible_region_carriers_images2_train46_s2_alpha1.json \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_region_texture_adapter_v36_matchedres \
  --target_split test \
  --method_name ours_26000_teacher_region_texture_adapter_v36_matchedres \
  --texture_size 16 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --policy_val_stride 4 \
  --alpha_grid 0,0.125,0.25,0.5,0.75,1.0 \
  --min_l1 0.001 \
  --min_alpha 0.03 \
  --max_samples_per_view 240000 \
  --max_abs_delta_rgb 0.12 \
  --min_policy_val_samples 1024 \
  --min_policy_val_relative_gain 0.001 \
  --force
```

Audit:

```text
accepted: true
atlas_faces: 34
fit_samples: 3564
policy_val_samples: 2166
selected_alpha: 1.0
policy_val_relative_gain: 0.7518149841
target_written_views: 37
target_changed_pixels: 205
target_total_pixels: 59932637
target_changed_fraction: 0.0000034205
```

Audit files:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_region_texture_adapter_v36_matchedres/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_region_texture_adapter_v36_matchedres/surface_residual_region_texture_adapter_audit.md
```

### 3.5 Metrics

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/bonsai_teacher_region_texture_adapter_v36_matchedres
```

Log:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/logs/metrics_bonsai_teacher_region_texture_adapter_v36_matchedres_gpu2.log
```

Result:

```text
ours_26000_teacher_region_texture_adapter_v36_matchedres
PSNR:  28.8648262
SSIM:   0.8960086
LPIPS:  0.2593367
```

## 4. Comparison

| Method | PSNR | SSIM | LPIPS | Notes |
|---|---:|---:|---:|---|
| selected clean MeshSplatting `ours_26000` | 28.895233 | 0.896400 | 0.259493 | local fair clean baseline |
| compact parent | 28.864340 | 0.896012 | 0.259340 | source compact checkpoint |
| v35 atlas stride3 | 28.864641 | 0.896011 | 0.259334 | previous residual atlas |
| v36 matched-res atlas | 28.864826 | 0.896009 | 0.259337 | this experiment |
| Phase-J guarded ELA | 31.862005 | 0.930280 | 0.172555 | current headline endpoint |

Relative to compact parent:

```text
dPSNR:  +0.000486
dSSIM:  -0.000003
dLPIPS: -0.000003
```

Relative to selected clean:

```text
dPSNR:  -0.030407
dSSIM:  -0.000391
dLPIPS: -0.000156
```

Relative to Phase-J:

```text
dPSNR:  -2.997179
dSSIM:  -0.034271
dLPIPS: +0.086782
```

## 5. Interpretation

v36 is a useful negative result, not a promoted method variant.

The train-only gate is strongly positive, and the adapter is technically accepted. However, only `205` pixels out of `59,932,637` held-out test pixels are changed. This is too sparse to affect full-resolution metrics or qualitative output. The matched-resolution train evidence did not solve the core issue.

The current bottleneck is therefore not simply train-evidence resolution. The more likely bottleneck is held-out target surface support:

- the target evidence cache still has missing barycentric information for a substantial subset of test views;
- the adapter can only modify pixels whose target face and barycentric coordinate match learned atlas faces;
- learned atlas faces remain sparse compared with the full Bonsai mesh;
- train policy-val support can look strong while held-out target coverage is nearly zero.

## 6. Actionable Next Step

Before another residual atlas attempt, fix evidence coverage rather than increasing atlas capacity:

1. Make `ecsr_build_surface_evidence_cache.py` write barycentric data per view immediately, not only after late global summary/top-support aggregation.
2. Rebuild full train and target evidence with complete barycentric support.
3. Audit target coverage before fitting: report percentage of held-out pixels whose face id is covered by train atlas candidates.
4. Only run higher-capacity atlas / patch texture if expected target coverage is non-negligible.

For the current mentor PPT, v36 should be described as:

> Matched-resolution residual atlas validates the interface and train-only gate, but held-out support coverage is still too sparse. The accepted paper-facing endpoint remains Phase-J; representation-level baking remains the next-stage research problem.
