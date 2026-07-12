# v39 SSIM-Aware Surface Residual Atlas

日期：2026-06-23  
状态：implementation + Bonsai full-res replay verified  
结论：v39 在 v38 risk-aware atlas 上加入低通 residual texture、bin variance、sign-consistency 统计。Bonsai full-res 上出现第一个 representation-level atlas 对 compact parent 的三指标严格正向 pilot，但幅度极小，仍没有超过 selected clean 的 PSNR/SSIM，也远低于 Phase-J render-time ELA。

## 1. Motivation

v38 已经修复 v37 的大退化，但最好的 Bonsai full-res 行仍差一点 SSIM：

```text
v38 risk-safe bin1 a0.03125:
dPSNR vs compact = +0.001690
dSSIM vs compact = -0.000006
dLPIPS vs compact = -0.000042
```

这说明 residual atlas 的主要剩余问题不是覆盖，而是结构相似度风险。v39 因此加入更偏 SSIM 的约束和表示：

- count-weighted 3x3 low-pass residual texture；
- per-bin residual variance；
- per-bin residual sign consistency；
- optional variance/sign gates in both policy-val and target apply。

## 2. Code Changes

Modified:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New fields in `FaceAtlas`:

```text
variance
sign_consistency
```

New command arguments:

```text
--atlas_lowpass_passes
--atlas_lowpass_neighbor_min_count
--max_atlas_bin_rgb_variance
--min_atlas_bin_sign_consistency
```

Default behavior remains compatible:

- low-pass is disabled by default;
- variance gate is disabled by default with `--max_atlas_bin_rgb_variance -1`;
- sign consistency gate is disabled by default with `--min_atlas_bin_sign_consistency 0`;
- v38 commands remain valid.

Static check:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

Result: passed.

## 3. Commands

### 3.1 v39 Lowpass1, Bin1, Alpha 0.03125

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_render_visible_region_carriers_visible_alpha1.json \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a003125 \
  --target_split test \
  --method_name ours_26000_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a003125 \
  --texture_size 16 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --policy_val_stride 4 \
  --alpha_grid 0,0.03125 \
  --min_l1 0.001 \
  --min_alpha 0.03 \
  --min_atlas_bin_count 1 \
  --min_atlas_face_samples 32 \
  --atlas_lowpass_passes 1 \
  --atlas_lowpass_neighbor_min_count 1 \
  --max_samples_per_view 240000 \
  --max_abs_delta_rgb 0.12 \
  --min_policy_val_samples 1024 \
  --min_policy_val_relative_gain 0.001 \
  --min_policy_val_positive_view_fraction 1.0 \
  --min_policy_val_cvar20_relative_gain 0.0 \
  --min_policy_val_min_view_relative_gain 0.0 \
  --select_alpha_by_risk_gate \
  --min_target_changed_fraction 0.0001 \
  --force
```

Policy-val:

```text
selected_alpha: 0.03125
relative_gain: 0.019216
positive_view_fraction: 1.000000
cvar20_view_relative_gain: 0.007087
min_view_relative_gain: 0.004393
target_changed_fraction: 0.005691
```

### 3.2 v39 Lowpass1, Bin1, Alpha 0.015625

Same as above, but:

```text
--output_model outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625
--method_name ours_26000_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625
--alpha_grid 0,0.015625
```

Policy-val:

```text
selected_alpha: 0.015625
relative_gain: 0.009688
positive_view_fraction: 1.000000
cvar20_view_relative_gain: 0.003693
min_view_relative_gain: 0.002356
target_changed_fraction: 0.005691
```

### 3.3 v39 Lowpass1 + Variance Gate

Variance statistics from the v39 atlas:

```text
supported variance p50/p90/p99:
0.0000005169 / 0.00682081 / 0.01852387
```

Tested variance gate:

```text
--max_atlas_bin_rgb_variance 0.0068
--alpha_grid 0,0.03125
```

Policy-val:

```text
relative_gain: 0.014313
positive_view_fraction: 1.000000
cvar20_view_relative_gain: 0.007968
min_view_relative_gain: 0.006883
target_changed_fraction: 0.004763
```

## 4. Full-Resolution Bonsai Metrics

| method | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM | dLPIPS | strict vs compact |
|---|---:|---:|---:|---:|---:|---:|---|
| selected clean `ours_26000` | 28.895233 | 0.896400 | 0.259493 | +0.030893 | +0.000388 | +0.000153 | no |
| compact parent | 28.864340 | 0.896012 | 0.259340 | +0.000000 | +0.000000 | +0.000000 | baseline |
| v37 visible atlas | 28.801197 | 0.891540 | 0.265000 | -0.063143 | -0.004473 | +0.005660 | no |
| v38 risk-safe bin1 a0.03125 | 28.866030 | 0.896006 | 0.259298 | +0.001690 | -0.000006 | -0.000042 | no |
| v39 lowpass1 bin1 a0.03125 | 28.866009 | 0.896010 | 0.259315 | +0.001669 | -0.000002 | -0.000025 | no |
| v39 lowpass1 bin1 a0.015625 | 28.865229 | 0.896012 | 0.259327 | +0.000889 | +0.00000018 | -0.000013 | yes |
| v39 lowpass1 var0068 bin1 a0.03125 | 28.865723 | 0.896011 | 0.259313 | +0.001383 | -0.000001 | -0.000026 | no |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 | yes |

Exact v39 strict-pilot delta over compact parent:

```text
PSNR: +0.000888824462890625
SSIM: +0.00000017881393432617188
LPIPS: -0.000012576580047607422
```

Delta vs selected clean:

```text
PSNR: -0.030004501342773438
SSIM: -0.0003877878189086914
LPIPS: -0.00016567111015319824
```

## 5. Verdict

v39 is the first Bonsai representation-level atlas pilot in this branch that strictly beats its compact parent on all three RGB metrics, but the effect size is extremely small:

- It should be described as a weak but real strict-positive representation-level pilot.
- It should not replace Phase-J.
- It should not be claimed to beat clean MeshSplatting overall, because it still trails selected clean on PSNR and SSIM.
- It is valuable because it shows that SSIM-aware residual smoothing can cross the compact no-regression line, which v38 could not.

## 6. Next Direction

The next method step should try to make this weak strict win larger without falling back into v37-style over-transfer:

1. Use local SSIM/luminance-contrast proxy in policy-val rather than only MSE residual gain.
2. Learn a per-bin confidence scalar from count, variance, sign consistency, normal/view angle, and support-view coverage.
3. Replace global alpha with confidence-weighted alpha per bin or per carrier.
4. Use carrier-holdout, not only view-holdout, to detect residual memorization.
5. Run the v39 strict-pilot policy on at least `garden`, `room`, and `counter` before considering any promotion beyond Bonsai.

## 7. Evidence Paths

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/logs/apply_bonsai_v39_lowpass1_bin1_face32_a0015625.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/logs/metrics_bonsai_v39_lowpass1_bin1_face32_a0015625_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/logs/metrics_bonsai_v39_lowpass1_bin1_face32_a003125_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/logs/metrics_bonsai_v39_lowpass1_var0068_bin1_face32_a003125_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_var0068_bin1_face32_a003125/results.json
```
