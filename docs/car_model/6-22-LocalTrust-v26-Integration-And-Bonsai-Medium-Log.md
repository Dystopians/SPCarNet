# v26 Local-Trust ELA 集成与 Bonsai Medium 验证日志

日期：2026-06-22

状态：`IMPLEMENTED_MEDIUM_DIAGNOSIS_REJECTED`

## 1. 目标

v26 的目标不是继续做场景手调，而是把 Evidence Lumigraph Adapter 的
render-layer 修复加上一层更严格的 local-trust gate：

- 只有当目标像素/局部证据有足够多训练视角支持时才允许 residual transfer；
- 多支持视角的 residual 方差不能太大；
- 预测 residual 与多视角证据需要有方向一致性；
- 低置信区域直接回退到 base render，避免 out-of-trajectory 崩塌。

这个分支当前只作为 Phase-J 之后的候选升级，不能替代已完成 full9
headline。它必须通过中长程、多场景验证，且定量/定性强于 Phase-J 后，
才可以进入主结论。

## 2. 已完成的代码接入

核心文件：

- `utils/evidence_lumigraph_adapter.py`
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py`
- `scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py`

新增或贯通的 profile：

```text
profile: field_region_render_risk_strict_v26
contract: field_region_render_risk_strict_v26_render_layer_local_trust_reversible_residual
fixed_profile: true
```

v26 固定 local-trust 参数：

```text
ela_local_trust_gate: true
ela_local_trust_min_supports: 2
ela_local_trust_max_residual_std: 0.035
ela_local_trust_min_agreement: 0.45
ela_local_trust_agreement_scale: 0.04
ela_local_trust_confidence_quantile: 0.25
ela_local_trust_min_confidence: 0.0001
```

命令路径已经覆盖：

- autovisual profile defaults；
- fixed-profile override rejection；
- PhaseK plan generation；
- candidate-owned refit；
- coupled selector；
- ELA materialization/evaluation；
- smoke synthetic local-trust acceptance/rejection。

## 3. 轻量验证

当前文件状态通过：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  utils/evidence_lumigraph_adapter.py \
  scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

输出：无错误。

ELA smoke：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_stageELA0_evidence_lumigraph_adapter.py
```

输出：

```text
[ELA smoke] passed
```

v26 autovisual dry-run：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v26 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 4 \
  --output_root /data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v26_localtrust_recheck \
  --pipeline_label dryrun_field_region_render_risk_strict_v26_localtrust_recheck \
  --wandb_mode online \
  --dry_run \
  --force
```

输出：

```json
{
  "output_root": "/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v26_localtrust_recheck",
  "commands": 8,
  "dry_run": true
}
```

manifest 证据：

- `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v26_localtrust_recheck/pipeline_command_manifest.json`
- `profile_contract_id = field_region_render_risk_strict_v26_render_layer_local_trust_reversible_residual`
- `commands = 8`
- plan、candidate-owned refit、selector 命令均含 `--ela_local_trust_gate` 与所有数值阈值。

## 4. GPU2 失败记录

第一次 medium 运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v26 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 2 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu2 \
  --pipeline_label field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu2 \
  --wandb_mode online \
  --force
```

结果：资源失败，不是方法失败。

日志：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu2/plan_generation/bonsai/phasek_barycentric_gate.log
```

关键错误：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 694.00 MiB.
```

当时该物理 GPU 上已有约 36GB 常驻进程，剩余显存不足。

## 5. GPU5 medium 验证

运行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --profile field_region_render_risk_strict_v26 \
  --stages plan,filter,selector \
  --scenes bonsai \
  --gpu 5 \
  --output_root /data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5 \
  --pipeline_label field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5 \
  --wandb_mode online \
  --force
```

当前证据：

- surface evidence 已完成；
- plan stage 和 candidate-owned refit 均产生真实 checkpoint update；
- candidate-owned refit 已产生 final decision；
- W&B online 已启用；
- top-level session 仍可能继续执行后续 selector/filter，但已有结果足够判断 v26 hard local-trust 不能作为 headline。

路径：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5
```

主要日志：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/plan_generation/bonsai/phasek_barycentric_gate.log
```

已产生的 evidence artifacts：

```text
surface_evidence/bonsai/surface_evidence_summary.json
surface_evidence/bonsai/surface_evidence_report.md
surface_evidence/bonsai/surface_residual_contact_sheet.png
surface_evidence/bonsai/top_residual_supports.csv
```

### 5.1 当前中间结果

截至 2026-06-22 14:38 PDT，PhaseK plan stage 已完成真实
checkpoint-level face-local residual update，并写出：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/plan_generation/bonsai/model/point_cloud/iteration_26000/point_cloud_state_dict.pt
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/plan_generation/bonsai/model/surface_residual_facelocal_sh1_delta_audit.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/plan_generation/bonsai/model/surface_residual_facelocal_sh1_delta_audit.md
```

delta audit 摘要：

```text
operator: surface_residual_facelocal_shared_field_delta
accepted: true
no-op copy: false
selected faces: 5790
accepted faces: 169
vertices added: 507
fit samples: 85666
policy-val samples: 34894
policy-val relative gain: 0.543541
patch certificate accepted patches: 22
patch certificate accepted faces after growth: 192
topology triangles unchanged: true
degenerate faces: 0
invalid indices: 0
```

这证明 v26 medium 至少已经产生了真实 checkpoint update，而不是 dry-run
或 no-op。

plan decision 摘要：

```json
{
  "accepted": false,
  "selected_label": "phasej_guarded_adaptedge",
  "decision_reasons": [
    "ssim_regression_exceeds_5e-05"
  ],
  "test_delta_report_only": {
    "LPIPS": 0.0001978725,
    "PSNR": 0.0162582397,
    "SSIM": 0.0005757809
  },
  "trainval_delta": {
    "LPIPS": -0.0004788935,
    "PSNR": 0.0156497955,
    "SSIM": -0.0000660419
  },
  "trainval_balanced_delta": 0.0239068270
}
```

这比单纯“完全失败”更细：v26 plan 的 balanced / PSNR / LPIPS 都是正向，
但 SSIM 轻微回退超过 `5e-05` 阈值，因此严格 gate 拒绝。它仍然低于
Phase-J headline，不能作为主结果。

### 5.2 Candidate-owned refit 结果

关键路径：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/candidate_owned_refit/bonsai/model/results.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/candidate_owned_refit/bonsai/model/trainval_gate_results.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/candidate_owned_refit/decisions/bonsai_decision.json
```

定量结果：

| row | split | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| candidate-owned base | test | 28.8651 | 0.8960 | 0.2594 |
| candidate-owned ELA | test | 29.4608 | 0.9048 | 0.2390 |
| candidate-owned trainval gate | trainval | 30.2851 | 0.9111 | 0.2345 |

decision 摘要：

```json
{
  "accepted": false,
  "selected_label": "phasej_guarded_adaptedge",
  "selection_uses_test": false,
  "decision_reasons": [
    "ssim_regression_exceeds_5e-05",
    "render_region_tail_cvar_below_-2e-05"
  ],
  "test_delta_report_only": {
    "LPIPS": -0.0008562505,
    "PSNR": 0.0395107269,
    "SSIM": 0.0008391142
  },
  "trainval_delta": {
    "LPIPS": -0.0005144924,
    "PSNR": 0.0065250397,
    "SSIM": -0.0002124310
  }
}
```

ELA local-trust 统计：

```json
{
  "local_trust_mode": "hard",
  "mean_local_trust_weight": 0.0,
  "mean_local_trust_active_fraction": 0.0,
  "alpha_holdout_safe_zero": true
}
```

解释：

- v26 的 report-only test 有小幅改善，但 honest decision 仍拒绝。
- 主要问题是 hard local-trust 把 residual 修复全部压成零，导致有效修复能力不足。
- 这直接推动了 v27 soft local-trust 设计。

### 5.3 Selector final 结果

v26 selector 已完成，不再是 in-flight 状态。

关键路径：

```text
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/selector/bonsai/coupled_selector_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/selector/coupled_selector_summary.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v26_localtrust_20260622_bonsai_medium_gpu5/selector/trials/strictfull_s1/decisions/bonsai_decision.json
```

最终 selector 摘要：

| scene | candidates | trials | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | best trial trainval balanced |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| bonsai | 20 | 1 | phasej_fallback | false | +0.000000 | +0.000000 | +0.000000 | -0.062735 |

`strictfull_s1` 被拒绝的原因：

```text
psnr_gain_below_0
ssim_regression_exceeds_5e-05
lpips_regression_exceeds_0.00015
balanced_delta_below_0
```

train-val tail 诊断：

```text
balanced_cvar_delta: -0.2238294001
balanced_negative_fraction: 1.0
lpips_positive_fraction: 0.96875
worst_lpips_regression: 0.0028544068
```

这说明 v26 的问题已经不是“没有候选”或“selector 未跑完”，而是
strict replay 在 policy-val view tail 上稳定失败。这个证据直接支持
v28 的 view-tail-safe alpha shrink：alpha safety 必须从 pooled pixel/bin
提升到 view-level tail。

## 6. 判定标准

v26 不能只看 “是否跑完”。需要满足：

1. plan/filter/selector 全部完成；
2. selector 没有因为 train-only gate 拒绝；
3. report-only test 不低于 Phase-J；
4. local-trust 统计能解释它为什么保留/抑制 residual；
5. 至少补跑一个室外和一个室内场景；
6. 定性图中必须能看到局部 artifact 更少或边缘/纹理更稳；
7. 若不能超过 Phase-J，则作为负结果保留，不进入 README headline。

当前结论：

```text
v26 is interface-complete, smoke-tested, and medium-diagnosed, but rejected by the honest gate.
```

下一步不是继续放宽 v26 阈值，而是使用 v27 或更新的 view-conditional /
tail-safe residual repair，把 hard binary trust 替换为更细粒度、可学习或
可校准的 residual confidence。

## 7. 汇总工具

为了避免后续中长程实验结果手工抄错，新增只读汇总工具：

```text
scripts/car_model/ecsr_summarize_autovisual_run.py
```

当前 v26/v27 汇总已经生成：

```text
/data/peilincai/spcarnet_runs/20260622_v26_v27_autovisual_run_summary.json
/data/peilincai/spcarnet_runs/20260622_v26_v27_autovisual_run_summary.md
```

GPU5 运行结束后，需要立即读取：

```text
pipeline_summary.md
selector/coupled_selector_summary.json
selector/*decision*.json
candidate_plans/bonsai/facelocal_visual_candidate_plan.json
filtered_candidate_plans/bonsai/filter_summary.json
```

并把结果追加到本日志。

2026-06-22 后续更新：selector 已完成并已记录在 `5.3 Selector final 结果`。
