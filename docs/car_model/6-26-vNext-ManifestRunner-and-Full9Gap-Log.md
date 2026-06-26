# 6-26 vNext Manifest Runner and Full9 Gap Log

日期：2026-06-26

本日志记录 vNext structure-aware shrink 从三场景 strict 里程碑推进到 full9 所需的工程补齐。本轮新增了 per-scene manifest runner，并把当前 full9 输入缺口从经验判断固化为机器可读 preflight artifact。

## 背景

`scripts/car_model/run_vnext_certified_residual_texture_full9.py` 只能接受统一 `{scene}` 路径模板：

```text
source_model_template
fit_evidence_template
target_evidence_template
region_carrier_template
```

但当前 vNext 已完成/可运行场景的 evidence 命名并不统一：

- `counter/room` 使用 v39 `phasej_trainval_alpha1` evidence；
- `garden` 使用 v39 `phasej_trainval_resize_alpha1` evidence；
- `bonsai` 使用 v37 `visible_alpha1` evidence；
- `bicycle/flowers/kitchen/stump/treehill` 的 v48 full9 evidence 曾经存在于 `/dev/shm/peilincai_spcarnet_v48_full9_20260623`，但当前 snapshot 中已不存在。

因此直接用旧 full9 wrapper 跑 full9 会被单一 template 限制卡住。

## 新增工程

新增脚本：

```text
scripts/car_model/run_vnext_certified_residual_texture_manifest.py
```

功能：

- 读取 per-scene JSON manifest；
- 对每个 scene 独立指定 `source_model / fit_evidence_dir / target_evidence_dir / region_carrier_json`；
- 支持 `--preflight_only`；
- 支持 `--max_parallel` 受控并发；
- 支持 `--skip_existing_complete`；
- 透传 scene runner 的固定策略参数；
- 运行后调用 `assemble_vnext_certified_residual_texture_report.py` 聚合结果。

已验证：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_vnext_certified_residual_texture_manifest.py
```

ready4 dry-run 编排已通过：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_manifest.py \
  --scene_config_json docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json \
  --output_root /dev/shm/peilincai_vnext_manifest_ready4_dryrun_20260626 \
  --method_name ours_26000_vnext_structure_aware_shrink \
  --max_parallel 2 \
  --dry_run \
  --skip_teacher_cache \
  --strict_no_target_gt_apply \
  --texture_size 16 \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_blend_candidates 0.5 \
  --max_abs_delta_rgb_candidates 0.12 \
  --enable_policy_val_structure_aware_shrink \
  --structure_shrink_l1_weight 1.0 \
  --structure_shrink_gradient_weight 1.0 \
  --structure_shrink_edge_weight 0.0 \
  --structure_shrink_risk_tau 0.002 \
  --structure_shrink_max_penalty 1.0
```

Dry-run result：`4 / 4` scene commands constructed and completed as dry-run; assembler returned `0`.

## Scene Configs

新增：

```text
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json
```

`ready4` 表示当前输入四件套已存在的场景：

```text
bonsai
counter
garden
room
```

`full9_gap` 表示 full9 目标配置，其中 4 个 ready，5 个缺输入：

```text
bicycle
flowers
kitchen
stump
treehill
```

## Preflight Results

新增 artifact：

```text
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md
```

结果：

| config | ready | missing input | conclusion |
|---|---:|---:|---|
| ready4 | 4 / 4 | 0 / 4 | 可直接跑 scene runner 或 manifest runner |
| full9 gap | 4 / 9 | 5 / 9 | source model 9/9 存在；5 个场景缺 fit evidence、target evidence、carrier |

Full9 missing-input rows：

| scene | source model | fit evidence | target evidence | carrier |
|---|---:|---:|---:|---:|
| bicycle | present | missing | missing | missing |
| flowers | present | missing | missing | missing |
| kitchen | present | missing | missing | missing |
| stump | present | missing | missing | missing |
| treehill | present | missing | missing | missing |

## Garden Fourth-Scene Run

`garden` 已有完整 v39 input chain，但此前只有 face-softshrink result，没有 structure-aware shrink strict result。因此本轮启动真实 `garden` strict no-target-GT structure-aware shrink run：

```text
run root: /dev/shm/peilincai_spcarnet_vnext_structure_shrink_garden_strict_20260626_071413_garden_structure_strict
gpu: 3
wandb: offline
method: ours_26000_vnext_structure_aware_shrink
```

状态：已完成。W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_garden_strict_20260626_071413_garden_structure_strict/wandb/offline-run-20260626_072412-fb1cy9it
```

结果：

| field | value |
|---|---:|
| protocol audit passed | `true` |
| target GT visible to apply | `false` |
| accepted | `true` |
| selected alpha | `0.125` |
| changed fraction | `0.00205038` |
| PSNR | `24.741142` |
| SSIM | `0.754052` |
| LPIPS | `0.248015` |
| delta vs Phase-F parent | `+0.00013924 / +0.00000316 / -0.00000791` |
| delta vs old garden face-softshrink | `+0.00006294 / +0.00000119 / -0.00000468` |

`garden` 与已有 `counter/bonsai/room` 已合并成 ready4 strict table：

```text
docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md
```

Ready4 summary:

```text
4 / 4 protocol pass
4 / 4 target GT hidden from apply
4 / 4 accepted nonzero
mean delta vs Phase-F compact parent:
  +0.00076151 PSNR / -0.00000302 SSIM / -0.00002038 LPIPS
```

## Missing-Input Rebuild Plan

对 `bicycle/flowers/kitchen/stump/treehill`，下一步必须重建或恢复以下三类输入：

1. target visible-bary evidence；
2. train visible-bary evidence + teacher residual cache；
3. render-visible region carrier + policy-val pruned carrier。

已知基础命令来自 v37/v39 路线：

```bash
scripts/car_model/ecsr_build_surface_evidence_cache.py
scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py
scripts/car_model/ecsr_build_render_visible_region_carriers.py
scripts/car_model/ecsr_prune_region_carriers_by_policy_val.py
```

建议输出到 normalized input tree：

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/{scene}/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/{scene}/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/{scene}/carrier.json
```

当前资源风险：

- `/data` 仅约 `388M` 可用，不能写大 evidence；
- `/dev/shm` 约 `26G` 可用，只适合分批构建；
- GPU 2/3 较空，可用于低并发重建。

## Follow-Up Rebuild Status

After this original full9 gap audit, three missing input chains were rebuilt locally and committed as lightweight evidence packages:

| scene | input state | strict result | artifact |
|---|---|---|---|
| stump | fit evidence + target evidence + policy-val carrier rebuilt | fallback/no-op; certificate rejected tail-risk candidate | `docs/car_model/6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md` |
| treehill | fit evidence + target evidence + policy-val carrier rebuilt | fallback/no-op; certificate rejected lower-tail/SSIM/L1 candidate | `docs/car_model/6-26-vNext-TreehillInputRebuild-Ready6-and-Rejection-Log.md` |
| flowers | fit evidence + target evidence + policy-val carrier rebuilt | fallback/no-op; certificate rejected lower-tail/SSIM/L1 candidate; same-evidence parent equals fallback | `docs/car_model/6-26-vNext-FlowersInputRebuild-Ready7-and-SameEvidenceFallback-Log.md` |

The latest local preflight is now:

```text
ready_scene_count: 7 / 9
missing_input_scene_count: 2 / 9
missing scenes: bicycle,kitchen
```

The latest committed preflight evidence is stored with the flowers artifact:

```text
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.json
```

## Claim Boundary

这次新增的是工程闭环能力和 full9 前置审计，不是 paper-final 性能结论。

可以说：

```text
vNext 已经不再依赖单一模板 full9 wrapper；异构 evidence 的 manifest runner 已实现并通过 ready4 dry-run。
当前 full9 缺口已机器可读定位为 2 个场景的 evidence/carrier 输入缺失，而不是 scene runner 或 strict no-target-GT 协议缺失。
```

不能说：

```text
不能说 vNext full9 已完成。
不能说剩余两个 missing scenes 已被验证。
不能说 vNext 已超过 v106/clean MeshSplatting。
```

## Next Required Work

1. 分批重建 remaining missing 2 scenes 的 normalized input tree。
2. 用 manifest runner 运行 full9 fixed policy。
3. 生成 full9 aggregate table 和同表对比 clean/Phase-F/v104c/v106/Phase-J。
4. 生成 changed-region qualitative panels。
5. 更新 README、vNext artifact index、PPT summary 和 full9 comparison table。
