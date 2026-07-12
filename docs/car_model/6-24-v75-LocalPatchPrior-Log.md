# v75 Local Patch Prior Log

日期：2026-06-24

状态：已完成诊断，未晋级。

## 动机

v74 已排除 `counter` 的主要瓶颈是 residual amplitude cap 过小，因为 `0.12/0.18/0.24` 在同一 blend 组下完全持平。v75 因此把问题从 scalar cap 转向 residual prior 的结构本身：如果 count-pyramid prior 太粗，是否用 same-face local UV patch prior 可以更精细地补足低支持 bins。

## 方法改动

新增真实 train/eval pipeline 接口：

- `surface_multiscale_prior_mode=local_patch`
- adapter 中新增 same-face local patch prior 统计；
- runner CLI choices 同步允许 `local_patch`；
- 复用已有 blend ladder、target-support pre-rank、target-support candidate selection、policy-val risk gate 和 W&B logging。

代码验证：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

验证通过。

## 实验

场景：`counter`

GPU：`5`

W&B：

```text
run id: j8fhiczt
group: v75_local_patch_prior
url: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/j8fhiczt
```

关键配置：

```text
surface_multiscale_prior_mode = local_patch
surface_multiscale_prior_block_sizes = 1,2,3
surface_multiscale_prior_blend_candidates = 0,0.5,1.0
support_expansion_mode = fit_residual_topk
support_expansion_max_extra_faces = 4096
target_support_prerank_top_k = 1
texture_size = 16
max_abs_delta_rgb = 0.12
```

## 结果

| method | PSNR | SSIM | LPIPS | 结论 |
|---|---:|---:|---:|---|
| v75 local patch prior | `26.753995895` | `0.862119257` | `0.251853049` | 完成，不晋级 |
| v74/v73b/v73/v70/v71a zero-blend 行 | `26.753995895` | `0.862119257` | `0.251853049` | 完全持平 |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | 仍更强 |

policy 选择：

```text
accepted = true
effective_policy = accepted_atlas
selected_alpha = 0.125
selected_surface_multiscale_prior_blend = 0.0
selected_support_mode = fit_residual_topk
selected_support_added_faces = 4096
target_changed_fraction = 0.06563028947904326
target_min_view_changed_fraction = 0.023086759617215888
target_cvar20_changed_fraction = 0.02734173713808936
```

blend 候选：

| blend | blended bins | blended fraction | policy-val best relative gain | selected |
|---:|---:|---:|---:|---|
| `0.0` | `0` | `0.000000` | `0.026849788` | yes |
| `0.5` | `951427` | `0.655469` | `0.026670204` | no |
| `1.0` | `951427` | `0.655469` | `0.026489316` | no |

## 结论

v75 是一个有效的结构性诊断，但不是指标突破。local patch prior 的非零 blend 确实覆盖大量低支持 bins，说明接口和统计是活的；但 policy-val 仍选择 `blend=0.0`，held-out 指标完全回到 v73b/v74 zero-blend 行。

这说明当前 `counter` 的瓶颈不是 count-pyramid prior 太粗，也不是 residual cap 太小，而是更深的 residual representation capacity 与 target-view generalization certificate 问题。下一步不应继续扩大 scalar sweep，应升级可迁移 residual basis 或把 adapter endpoint 明确包装成 certified rendering module。

## 证据路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/counter/apply_metrics_counter.log
```
