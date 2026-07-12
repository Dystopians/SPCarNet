# v27 Soft Local-Trust ELA 实现与 Bonsai Medium 验证日志

日期：2026-06-22

状态：`IMPLEMENTED_MEDIUM_VALIDATED_REJECTED_BY_HONEST_GATE`

## 1. 动机

v26 的 local-trust 机制是 hard binary gate：support count、residual std/agreement、confidence 中任一条件不满足，就把该像素 residual 直接置零。

bonsai medium 已经显示这个设计过保守：

- fallback/candidate ELA 多次出现 `alpha=0`；
- candidate test 约为 `29.4375 PSNR / 0.9046 SSIM / 0.2400 LPIPS`；
- 明显低于当前 Phase-J bonsai headline `31.8620 / 0.9303 / 0.1726`。

因此 v27 不继续做场景参数搜索，而是把 local trust 从“硬拒绝”升级为连续 residual 权重。

## 2. 方法变化

新增 `soft` local-trust 模式：

```text
trust_weight =
  support_score
  * residual_std_score
  * agreement_score
  * confidence_score
```

其中：

- `support_score` 按 support count 连续缩放；
- `residual_std_score` 在 v26 hard threshold 附近软衰减；
- `agreement_score` 用 residual std 的指数一致性；
- `confidence_score` 用 train-view confidence quantile/min threshold 归一化；
- `local_trust_min_weight` 只作为最低激活权重，不再把中等可信 residual 全部删除。

v27 的关键差异：

| 版本 | local trust | 结果预期 |
|---|---|---|
| v26 | hard binary accept/reject | 安全但容易 alpha=0，残差修复被抑制 |
| v27 | soft trust-weighted residual | 保留低/中可信 residual 的衰减贡献，再交给 train-only alpha/region-risk guard |

## 3. 代码接口

核心实现：

- `utils/evidence_lumigraph_adapter.py`
  - 新增 `local_trust_weight_map(...)`；
  - `adapt_frame(...)` 新增 `local_trust_mode` 和 `local_trust_min_weight`；
  - `fit_benefit_calibrator(...)` / `fit_alpha_calibrator(...)` / `calibrate_alpha(...)` 与最终 render 使用同一套 hard/soft local-trust；
  - report 中新增 `local_trust_mode`、`local_trust_mean_weight`、`local_trust_active_fraction`。

CLI 与 pipeline：

- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - 新增 `--local_trust_mode {hard,soft}`；
  - 新增 `--local_trust_min_weight`；
  - W&B 新增 `ela/mean_local_trust_weight` 和 `ela/mean_local_trust_active_fraction`。
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
  - 新增并转发 `--ela_local_trust_mode` / `--ela_local_trust_min_weight`。
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
  - 新增并转发 `--ela_local_trust_mode` / `--ela_local_trust_min_weight`。
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`
  - 新增 fixed profile `field_region_render_risk_strict_v27`；
  - contract id: `field_region_render_risk_strict_v27_soft_local_trust_weighted_residual`；
  - fixed profile override 已验证会拒绝。

## 4. v27 固定 profile

```text
profile: field_region_render_risk_strict_v27
ela_local_trust_gate: true
ela_local_trust_mode: soft
ela_local_trust_min_weight: 0.02
ela_local_trust_min_supports: 2
ela_local_trust_max_residual_std: 0.06
ela_local_trust_min_agreement: 0.20
ela_local_trust_confidence_quantile: 0.10
ela_local_trust_min_confidence: 1.0e-4
```

这不是 per-scene tuning。它是从 v26 失败诊断得出的固定全局 policy。

## 5. 已完成验证

静态编译：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  utils/evidence_lumigraph_adapter.py \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

结果：通过。

ELA smoke：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

结果：

```text
[ELA smoke] passed
```

v27 autovisual dry-run：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v27 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v27_softtrust_recheck \
  --pipeline_label dryrun_field_region_render_risk_strict_v27_softtrust_recheck \
  --wandb_mode online \
  --dry_run \
  --force
```

结果：

```json
{
  "output_root": "/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v27_softtrust_recheck",
  "commands": 8,
  "dry_run": true
}
```

manifest 关键证据：

- fixed profile: `true`
- contract: `field_region_render_risk_strict_v27_soft_local_trust_weighted_residual`
- ELA local-trust mode/min weight: `soft` / `0.02`
- PhaseK 命令包含 `--ela_local_trust_mode soft --ela_local_trust_min_weight 0.02`
- selector 命令包含同样参数。

固定 profile override 拒绝验证：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v27 \
  --ela_local_trust_mode hard \
  --dry_run \
  --force
```

结果：正确报错：

```text
profile field_region_render_risk_strict_v27 is fixed; remove profile-field overrides: ela_local_trust_mode
```

## 6. Bonsai medium 验证

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v27 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4 \
  --pipeline_label field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4 \
  --wandb_mode online \
  --force
```

运行状态：

- exec session id: `63353`
- output root: `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4`
- W&B online 已按 pipeline 配置启用；
- plan stage 已完成 test / trainval / render-region gate / decision；
- candidate-owned refit 已完成并被 honest gate 拒绝；
- selector 已完成，最终选择 `strictfull_s1`，但 report-only held-out test 三指标轻微回退；
- final conclusion: v27 仍不能替代 Phase-J headline。

关键 artifacts：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/plan_generation/bonsai/model/results.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/plan_generation/bonsai/model/test_results.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/plan_generation/bonsai/model/trainval_gate_results.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/plan_generation/decisions/bonsai_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/candidate_owned_refit/decisions/bonsai_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4/selector/bonsai/coupled_selector_decision.json
```

定量结果：

| row | split | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| plan base | test | 28.8649 | 0.8960 | 0.2594 |
| compact soft ELA fallback | test | 29.5067 | 0.9061 | 0.2320 |
| plan soft ELA candidate | test | 29.5038 | 0.9063 | 0.2320 |
| compact soft ELA fallback | trainval | 30.4714 | 0.9144 | 0.2241 |
| plan soft ELA candidate | trainval | 30.4506 | 0.9142 | 0.2247 |

v27 相对 v26 的机制改善是明确的：

| branch | local trust mode | mean trust weight | active fraction | test PSNR / SSIM / LPIPS |
|---|---|---:|---:|---:|
| v26 hard candidate ELA | hard | 0.0000 | 0.0000 | 29.4608 / 0.9048 / 0.2390 |
| v27 soft plan ELA | soft | 0.6132 | 0.9629 | 29.5038 / 0.9063 / 0.2320 |

但是 v27 plan candidate 没有被诚实 gate 接受：

```json
{
  "accepted": false,
  "selected_label": "phasej_guarded_adaptedge",
  "decision_reasons": [
    "psnr_gain_below_0",
    "ssim_regression_exceeds_5e-05",
    "lpips_regression_exceeds_0.00015",
    "balanced_delta_below_0"
  ],
  "test_delta_report_only": {
    "LPIPS": 0.0000365973,
    "PSNR": -0.0028305054,
    "SSIM": 0.0001394153
  },
  "trainval_delta": {
    "LPIPS": 0.0005710721,
    "PSNR": -0.0208473206,
    "SSIM": -0.0002832413
  },
  "trainval_balanced_delta": -0.0379335880
}
```

render-region gate 本身通过：

```json
{
  "accepted": true,
  "tail": {
    "core_balanced_cvar_delta": 0.0,
    "negative_core_balanced_fraction": 0.0,
    "worst_core_balanced_delta": 0.0
  }
}
```

真正短板在 trainval 全局/尾部分布：

```json
{
  "balanced_cvar_delta": -0.1221391925,
  "balanced_negative_fraction": 0.984375,
  "lpips_positive_fraction": 1.0,
  "worst_lpips_regression": 0.0024513900
}
```

candidate-owned refit final decision:

```json
{
  "accepted": false,
  "selected_label": "phasej_guarded_adaptedge",
  "decision_reasons": [
    "psnr_gain_below_0",
    "ssim_regression_exceeds_5e-05",
    "lpips_regression_exceeds_0.00015",
    "balanced_delta_below_0",
    "render_region_tail_cvar_below_-2e-05"
  ],
  "test_delta_report_only": {
    "LPIPS": -0.0005948097,
    "PSNR": 0.0053901672,
    "SSIM": 0.0001439452
  },
  "trainval_delta": {
    "LPIPS": 0.0006463081,
    "PSNR": -0.0196056366,
    "SSIM": -0.0003137589
  },
  "trainval_balanced_delta": -0.0388069749
}
```

selector final decision:

```json
{
  "accepted": true,
  "selected_trial": "strictfull_s1",
  "selected_trainval_balanced_delta": 0.0006342530,
  "effective_report_only_test_delta": {
    "LPIPS": 0.0000103563,
    "PSNR": -0.0000171661,
    "SSIM": -0.0000039935
  }
}
```

Interpretation: selector acceptance is train-val honest, but the accepted trial
is effectively no-op and slightly worse on held-out test. It is therefore a
diagnostic/fairness result, not a promoted method.

## 7. 验收标准

v27 只有在以下条件满足时才能替代 v26 或进入 headline 候选：

1. candidate test 不低于 v26，并尽量恢复 Phase-J 级别 ELA 收益；
2. trainval gate 不因 soft trust 引入明显负尾部风险；
3. selector 不能因 test-only 结果选择 branch；
4. ELA report 中必须显示非零 `mean_local_trust_weight` 和合理 `active_fraction`；
5. 若仍低于 Phase-J headline，则作为 v26 的机制修复但不作为主结果。

当前结论：

```text
v27 is implementation-complete and medium-validated, but rejected by the honest trainval gate.
```

解释：

- v27 不是无效实现：soft trust 的 mean weight 和 active fraction 都明显非零，解决了 v26 hard trust “全零 residual”问题。
- v27 确实略强于 v26 candidate test，但提升太小，而且 plan candidate 相对 fallback 在 trainval 上回退。
- 当前最强 headline 仍应使用 Phase-J compact MeshSplatting + train-only ELA，而不是 v27。
- 下一步如果继续推进，应把目标放在“tail-safe / view-conditional residual repair”或“将 ELA 蒸馏回 representation-level”，而不是继续扩大 soft trust 权重。

## 8. 汇总工具与横向对照

新增只读汇总工具：

```text
scripts/car_model/ecsr_summarize_autovisual_run.py
```

当前 v26/v27 横向汇总：

```text
/data/peilincai/spcarnet_runs/20260622_v26_v27_autovisual_run_summary.json
/data/peilincai/spcarnet_runs/20260622_v26_v27_autovisual_run_summary.md
```

关键横向结论：

| run | stage | decision | test dPSNR | test dSSIM | test dLPIPS | trainval dPSNR | trainval dSSIM | trainval dLPIPS | trainval balanced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v26 hard trust | plan | false | +0.016258 | +0.000576 | +0.000198 | +0.015650 | -0.000066 | -0.000479 | +0.023907 |
| v26 hard trust | candidate-owned | false | +0.039511 | +0.000839 | -0.000856 | +0.006525 | -0.000212 | -0.000514 | +0.012566 |
| v27 soft trust | plan | false | -0.002831 | +0.000139 | +0.000037 | -0.020847 | -0.000283 | +0.000571 | -0.037934 |
| v27 soft trust | candidate-owned | false | +0.005390 | +0.000144 | -0.000595 | -0.019606 | -0.000314 | +0.000646 | -0.038807 |
| v27 soft trust | selector | true | -0.000017 | -0.000004 | +0.000010 | n/a | n/a | n/a | +0.000634 |

解读：v27 解决了 trust 全零，但把 residual 重新放开后出现更强 trainval
负尾部；selector 虽然可以找到 train-val balanced 为正的 strict replay，
但 held-out test 为近似 no-op 且三指标轻微回退。v26 反而在部分 balanced
诊断上更稳，但 SSIM/tail 越界。下一轮应关注 per-view/per-region tail-safe
alpha，而不是单纯 hard/soft trust 二选一。
