# v38 Risk-Aware Surface Residual Atlas

日期：2026-06-23  
状态：implementation + Bonsai full-res replay verified  
结论：v38 是 v37 之后的真实方法改动。它把 v37 的明显 held-out 退化修复到接近 compact parent，并在 Bonsai 上得到 PSNR/LPIPS 正向、SSIM 近持平的 representation-level pilot；但它仍未全面超过 clean/compact，也不能替代 Phase-J headline。

## 1. Motivation

v37 证明了 target coverage 已经不是主要瓶颈：

```text
v36 target changed pixels: 205
v37 target changed pixels: 578910
v37 actionable pixels: 580404
```

但 v37 full-res Bonsai 指标退化：

```text
v37 visible-train visible-target atlas:
PSNR 28.801197
SSIM 0.891540
LPIPS 0.265000
```

这说明 naive residual texture atlas 的问题已经从“能不能作用到 target”转移到“作用的 residual 是否跨视角泛化安全”。v38 因此加入两个 train-only 风险控制：

1. **atlas support certification**：只有训练中实际观测过的 UV bin / face 才允许写 residual；
2. **risk-safe alpha selection**：不再只选平均 policy-val MSE 最优 alpha，而是要求每个 policy-val view 都非退化，并检查 worst-view / CVaR20。

## 2. Code Changes

Modified:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New arguments:

```text
--min_atlas_bin_count
--min_atlas_face_samples
--min_policy_val_positive_view_fraction
--min_policy_val_cvar20_relative_gain
--min_policy_val_min_view_relative_gain
--select_alpha_by_risk_gate
```

Behavior changes:

- `predict_delta_for_npz` can now suppress target pixels whose face/UV bin does not have enough train atlas support.
- `evaluate_policy_val` now records per-view relative gain for every alpha.
- Each alpha row now includes:
  - `positive_view_fraction`;
  - `min_view_relative_gain`;
  - `p10_view_relative_gain`;
  - `cvar20_view_relative_gain`.
- With `--select_alpha_by_risk_gate`, the script selects the best alpha that satisfies aggregate gain and robust view-level gates, rather than blindly selecting mean-MSE best alpha.
- Audit reports now include `policy_val_risk_gate`.

Static check:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

Result: passed.

## 3. Key Diagnosis From v38 Policy-Val

Using Bonsai full-res v37 visible train/target evidence, the risk statistics expose the v37 failure clearly.

Configuration:

```text
min_atlas_bin_count: 1
min_atlas_face_samples: 32
alpha_grid: 0,0.125,0.25,0.5,0.75,1.0
```

Policy-val rows:

| alpha | rel gain | positive view frac | CVaR20 view gain | min view gain |
|---:|---:|---:|---:|---:|
| 0.125 | 0.074679 | 1.000000 | 0.023605 | 0.011314 |
| 0.250 | 0.135216 | 0.833333 | 0.015788 | -0.007824 |
| 0.500 | 0.213861 | 0.833333 | -0.094115 | -0.174908 |
| 0.750 | 0.235936 | 0.583333 | -0.329707 | -0.520298 |
| 1.000 | 0.201441 | 0.500000 | -0.690989 | -1.037646 |

Interpretation:

> The mean-MSE best alpha is `0.75`, but it is unsafe: only `7 / 12` policy-val views improve and the worst-view loss is large. The risk-safe alpha is `0.125`, because it improves all `12 / 12` policy-val views and keeps CVaR20 positive.

This is the main methodological improvement over v37: the script can now detect and avoid the exact class of false positive that caused v37 to regress.

## 4. Commands

### 4.1 v38 Risk-Safe, Bin1, Alpha Selected by Risk Gate

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/bonsai \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_render_visible_region_carriers_visible_alpha1.json \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32 \
  --target_split test \
  --method_name ours_26000_teacher_region_texture_adapter_v38_risksafe_bin1_face32 \
  --texture_size 16 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --policy_val_stride 4 \
  --alpha_grid 0,0.125,0.25,0.5,0.75,1.0 \
  --min_l1 0.001 \
  --min_alpha 0.03 \
  --min_atlas_bin_count 1 \
  --min_atlas_face_samples 32 \
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

Selected alpha:

```text
selected_alpha: 0.125
positive_view_fraction: 1.0
cvar20_view_relative_gain: 0.023605
target_changed_fraction: 0.005691
```

### 4.2 More Conservative Alpha Sweeps

The same risk-safe gate was also run with fixed small alpha grids:

```text
bin1 face32 alpha 0.0625
bin1 face32 alpha 0.03125
bin2 face32 alpha 0.03125
```

All four variants passed the train-only risk gates and wrote full test renders.

## 5. Full-Resolution Bonsai Metrics

| method | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM | dLPIPS | strict vs compact |
|---|---:|---:|---:|---:|---:|---:|---|
| clean selected `ours_26000` | 28.895233 | 0.896400 | 0.259493 | +0.030893 | +0.000388 | +0.000153 | no |
| compact parent | 28.864340 | 0.896012 | 0.259340 | +0.000000 | +0.000000 | +0.000000 | baseline |
| v36 matched-res atlas | 28.864826 | 0.896009 | 0.259337 | +0.000486 | -0.000004 | -0.000003 | no |
| v37 visible atlas | 28.801197 | 0.891540 | 0.265000 | -0.063143 | -0.004473 | +0.005660 | no |
| v38 risk-safe bin1 a0.125 | 28.869099 | 0.895831 | 0.259456 | +0.004759 | -0.000182 | +0.000117 | no |
| v38 risk-safe bin1 a0.0625 | 28.867365 | 0.895973 | 0.259272 | +0.003025 | -0.000039 | -0.000068 | no |
| v38 risk-safe bin1 a0.03125 | 28.866030 | 0.896006 | 0.259298 | +0.001690 | -0.000006 | -0.000042 | no |
| v38 risk-safe bin2 a0.03125 | 28.865604 | 0.896009 | 0.259308 | +0.001265 | -0.000004 | -0.000032 | no |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 | yes |

Best v38 reading:

- v38 fixes most of the v37 regression.
- v38 can beat compact parent on PSNR and LPIPS simultaneously.
- v38 still misses strict three-metric compact win because SSIM remains slightly below compact by `4e-6` to `1.8e-4`.
- v38 remains far below Phase-J render-time ELA.
- v38 does not beat selected clean on PSNR/SSIM, though LPIPS can be slightly better than clean.

## 6. Current Verdict

v38 is a real engineering and research improvement over v37:

```text
v37 -> v38 best conservative:
PSNR: 28.801197 -> 28.866030
SSIM: 0.891540 -> 0.896006
LPIPS: 0.265000 -> 0.259298
```

But it is not a paper-level endpoint:

- not strict three-metric better than compact parent;
- not better than selected clean on PSNR/SSIM;
- not close to Phase-J ELA;
- still a Bonsai-only representation-level pilot, not a full9 result.

The important lesson is that train-only view-risk statistics are predictive enough to reject unsafe large-alpha residual transfer, but the current residual atlas basis still lacks enough precision to produce a strong, visually obvious, representation-internal improvement.

## 7. Next Technical Direction

The next step should not be another alpha sweep. The remaining SSIM loss suggests the residual texture still changes structural neighborhoods in a way SSIM penalizes. More promising next changes:

1. **SSIM-aware train proxy.** Add a low-cost local-window luminance/contrast consistency proxy to policy-val, not just MSE residual gain.
2. **Signed residual confidence.** Estimate per-bin residual variance and suppress bins where support residual sign is inconsistent across train views.
3. **Carrier-holdout split.** Split within carrier faces/bins, not just views, so a carrier cannot pass by memorizing its own train UV support.
4. **Low-pass residual basis.** Smooth or rank-reduce atlas residuals before target apply to avoid SSIM-damaging high-frequency texture noise.
5. **No-op fallback at representation endpoint.** For paper reporting, v38 should be treated as diagnostic unless it strictly beats compact/clean; Phase-J remains headline.

## 8. Evidence Paths

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/apply_bonsai_v38_risksafe_bin1_face32.log
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/metrics_bonsai_v38_risksafe_bin1_face32_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/metrics_bonsai_v38_risksafe_bin1_face32_a00625_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/metrics_bonsai_v38_risksafe_bin1_face32_a003125_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/metrics_bonsai_v38_risksafe_bin2_face32_a003125_gpu4.log
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32_a00625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32_a003125/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin2_face32_a003125/results.json
```
