# SPCarNet / MeshSplatOpt

**基于训练证据的几何安全 Mesh Splatting 压缩与渲染修复。**

[English](README.md) | [Phase-J 结果](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md) | [Surface-lumigraph V8](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md) | [Phase-R 固定策略](docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md) | [Phase-R 室内审计](docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md) | [Phase-J 外部验证](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md) | [当前版本留档](docs/car_model/5-7-Archive-Full9-CompactELA.md) | [5 月 7 日更新](docs/car_model/5-7-Update.md) | [升级路线](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md) | [ECSR 审计](docs/car_model/5-8-ECSR-CurrentStateAudit.md) | [执行日志](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md) | [研究日志](docs/car_model/SPCarNet_research_log.md) | [旧版 README](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)

SPCarNet 是建立在 Mesh Splatting 之上的研究分支。当前 ECSR 版本保留固定的 Phase-F compact checkpoint，再用 train-evidence guarded portfolio 做外观修复：稳定场景走 adaptive-alpha ELA，不稳定场景走 train-selected structural edge fallback。branch、edge gate、alpha、压缩比例都不使用 held-out test 指标选择。

```text
current method: ours_26000_phasej_guarded_adaptedge_ela
report: outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

5 月 7 日 Compact-ELA/SOR 版本仍以 `archive/full9-compact-ela-ssim-peak-20260507`、commit `fae7942` 留档。Phase-J 在当前 full9 RGB 口径上更强，但它仍然是 render-time ELA portfolio，不是完全 baked 到表示里的终局模型。

## 当前结果

**评估口径。** Mip-NeRF360 同协议复现。每个场景的 clean MeshSplatting baseline 从 clean `26000` 与 `30000` checkpoint 中选择，只使用 held-out test 指标：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

训练集指标不用于选择 baseline，也不用于选择最终 test 结果。

**Phase-J 最终报告。**

- 报告：`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`
- 场景：`9 / 9`
- 相对 selected clean MeshSplatting：`9 / 9` 三指标严格胜出
- 相对 Phase-F alpha-grid：`9 / 9` 三指标严格胜出
- 相对 selected clean MeshSplatting baseline 的均值提升：`+1.3311 PSNR`，`+0.0347 SSIM`，`-0.0634 LPIPS`
- 相对 Phase-F alpha-grid 的均值提升：`+0.3971 PSNR`，`+0.0083 SSIM`，`-0.0193 LPIPS`
- 平均三角形减少：`7.6479%`
- 闭环审计：`244 / 246` 个 held-out view 三指标严格胜出；max500 sparse COLMAP 口径下 `9 / 9` 场景 geometry-safe，`6 / 9` 场景严格几何更好。
- 外部 courtyard 验证：ETH3D courtyard clean9000 上 Phase-J 相对 clean MeshSplatting 最高提升 `+0.2642 PSNR`，`+0.0094 SSIM`，`-0.0225 LPIPS`；退化 F82 checkpoint 仅有极小提升，因此作为局限诊断保留。

| 场景 | 分支 | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | 三角形减少 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.0215 | 0.7024 | 0.2661 | +0.7199 | +0.0425 | -0.0660 | 11.81% |
| flowers | adaptive alpha | 20.3044 | 0.5578 | 0.3292 | +0.6221 | +0.0459 | -0.0653 | 11.82% |
| garden | adaptive alpha | 26.3111 | 0.8278 | 0.1358 | +1.2819 | +0.0478 | -0.0655 | 3.47% |
| stump | adaptive alpha | 25.5951 | 0.7241 | 0.2639 | +0.3901 | +0.0189 | -0.0301 | 11.82% |
| treehill | auto edge fallback | 21.2962 | 0.5956 | 0.3363 | +0.3620 | +0.0311 | -0.0697 | 11.81% |
| room | adaptive alpha | 30.3056 | 0.9057 | 0.1960 | +1.5584 | +0.0209 | -0.0539 | 2.10% |
| counter | adaptive alpha | 28.4492 | 0.8937 | 0.1865 | +1.6974 | +0.0317 | -0.0655 | 2.10% |
| kitchen | adaptive alpha | 30.1997 | 0.9161 | 0.1320 | +2.3812 | +0.0396 | -0.0672 | 2.10% |
| bonsai | adaptive alpha | 31.8620 | 0.9303 | 0.1726 | +2.9668 | +0.0339 | -0.0869 | 11.80% |

## ECSR 升级状态

下一条主线是 **ECSR: Evidence-Certified Surface Relocation**。目标是把 SPCarNet 从 image-space residual repair 推进到 representation-level 的 surface compression 与 appearance recovery。

当前执行产物：

- Current-state audit：[`docs/car_model/5-8-ECSR-CurrentStateAudit.md`](docs/car_model/5-8-ECSR-CurrentStateAudit.md)
- Phase-A train-only surface evidence：[`docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md`](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md)
- Phase-B view-support graph：[`docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md`](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md)
- Phase-A/B cached-view policy split：[`docs/car_model/5-8-ECSR-PolicySplit.md`](docs/car_model/5-8-ECSR-PolicySplit.md)
- Full-train fitting/policy-val split：[`docs/car_model/5-8-ECSR-FullTrainPolicySplit.md`](docs/car_model/5-8-ECSR-FullTrainPolicySplit.md)
- Phase-C candidate preflight：[`docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md`](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md)
- Phase-C static topology certificate：[`docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md`](docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md)
- Phase-C materialized checkpoint smoke：[`docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md`](docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md)，[`docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md`](docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md)
- Phase-D attribute-only recovery smoke：[`docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md`](docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md)
- Phase-D constrained attribute recovery：[`docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md`](docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md)
- Phase-D surface residual delta smoke：[`docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md`](docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md)
- Phase-G teacher-bake recovery：[`docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md`](docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md)
- Phase-J guarded adaptive edge policy：[`docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md`](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md)
- Phase-J external courtyard validation：[`docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md`](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md)
- Surface-attached residual lumigraph V8：[`docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md`](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md)
- Phase-R fixed surface-SH1 ladder：[`docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md`](docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md)
- Phase-R 室内多折与 gamma trust 审计：[`docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md`](docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md)
- 执行日志：[`docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md`](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md)
- Phase-A 汇总 contact sheet：`outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`

Phase-A 结果：`9 / 9` 场景通过 surface addressability，但只有 `4 / 9` 通过当前 top-support multiview consistency 检查。这说明 residual 信号是真实且可回投到表面的，但 naive 的单 face residual delta 还不能作为最终方法。

Phase-B 结果：固定 graph policy 在 full9 上找到 `123` 个 train-only local support cluster，其中 `23` 个是 certificate-contraction candidates，`99` 个是 surface-attribute recovery candidates。但 residual-hot cluster 的直接三角形压缩上限很小，因此下一步必须把 compression candidate 和 appearance-recovery candidate 分开，而不能把 residual hotspot 当成压缩目标。

Phase-C preflight 结果：`21 / 123` 个 Phase-B cluster 通过 train-only fitting/policy-val support-mask preflight，其中 `13` 个是 contraction 类型，`8` 个是 attribute-recovery 类型。它们还不是被接受的 ECSR 修改，只是进入 topology smoke test 与 before/after local rendering certificate 的第一批候选。

Phase-C/D 执行更新：full-train split 已覆盖全部 9 个场景。Static topology certification 在 `21` 个 preflight candidate 中通过 `7` 个；其中 `3` 个 contraction candidate 已被 materialize 成真实 checkpoint copy，并且 `3 / 3` 通过 renderer smoke。两个 representation-level recovery MVP 已实现但还不能作为最终方法：attribute-only recovery 的 `2 / 2` 个 smoke run 回退，bounded surface residual DC delta 虽然在 train policy-val mean-L1 上接受 `3 / 4`，但 held-out diagnostic `4 / 4` 回退。这说明 checkpoint 接口已经打通。

Phase-G 尝试把 ELA teacher bake 回 topology-frozen checkpoint，但 official `bicycle` 与 `flowers` pilot 都略低于 clean MeshSplatting，且明显低于 render-time ELA，因此被拒绝为当前主线。Phase-J 是当前接受的方法：一个 no-test-GT guarded portfolio，稳定时用 adaptive alpha，不稳定时用 train-selected structural edge fallback。

Phase-M / V8 目前是最干净的 representation-attached recovery baseline：train residual 存在 surface `face_id` 上，held-out view 只通过 target surface map 查表应用。固定 two-split consensus policy 接受 `flowers` 与 `garden`，其余 `7 / 9` 场景自动 no-op；相对 Phase-F compact base 的 full9 均值为小幅正向变化：`+0.000250` PSNR、`+0.000000868` SSIM、`-0.00000638` LPIPS。这个结果不是当前论文主 RGB endpoint，但它给下一步更高容量的 surface-attached 表示恢复提供了安全基线。

Phase-R 进一步把 residual 写回 checkpoint 中的 surface SH1 属性，并加入 train-only gamma trust-region residual gate。最新 full9 v10 snapshot 在严格 train-heldout gate 下接受 `6 / 9` 个 representation edit，report-only 三指标严格胜出 `6 / 9`，相对 Phase-J 的 report-only 均值为 `+0.002993` PSNR、`+0.000136` SSIM、`-0.000217` LPIPS。新增的 gamma trust-region 把 `room` 从多折拒绝推进到 representation edit 接受，而且没有使用 held-out test 指标做选择。当前剩余瓶颈是 `counter`、`bonsai` 与 `treehill` 这类 edge-fallback 场景；收益已经更稳，但仍属于小幅稳定提升，还不是最终视觉突破。

## 其他评估口径

当前 Phase-J 摘要：

| 评估口径 | 结果 |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` 三指标严格胜出，均值 `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| Phase-F alpha-grid 前序方法 | `9 / 9` 三指标严格胜出，均值 `+0.3971` PSNR，`+0.0083` SSIM，`-0.0193` LPIPS |
| guarded branch decision | `8 / 9` adaptive-alpha branch，`1 / 9` train-selected edge fallback |
| 几何 / 拓扑 | 平均三角形减少 `7.6479%`；Phase-J closure audit 下 `6 / 9` sparse geometry 严格更好，`9 / 9` geometry-safe |
| per-view audit | `244 / 246` 个 held-out view 相对 selected clean baseline 同时提升 PSNR、SSIM、LPIPS |
| 外部验证 | ETH3D courtyard clean9000 三指标严格胜出：最高 `+0.2642` PSNR，`+0.0094` SSIM，`-0.0225` LPIPS；相对旧 ELA7 为 mixed |
| Phase-R v10 representation ladder | 严格 train-heldout 接受 `6 / 9`，report-only 三指标严格胜出 `6 / 9`，相对 Phase-J 均值 `+0.002993` PSNR，`+0.000136` SSIM，`-0.000217` LPIPS；`room` 由 gamma trust-region residual blending 新增通过 |

下面的详细表格保留自 5 月 7 日 Compact-ELA/SOR 留档报告，用于 provenance。LPIPS、AbsRel、DepthMAE、Normal 越低越好。

| 评估口径 | 结果 |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` RGB 胜出，均值 `+0.4979` PSNR，`+0.0158` SSIM，`-0.0234` LPIPS |
| MeshSplatting paper table | `9 / 9` RGB 胜出，均值 `+0.8685` PSNR，`+0.0366` SSIM，`-0.0465` LPIPS |
| clean checkpoint envelope | 9 个场景全部选择 clean `26000` 而非 clean `30000`；平均 score gap `+1.1029` |
| 几何 / 拓扑 | `5 / 9` strict all-axis pass，`9 / 9` RGB + compact + geometry-safe pass，平均三角形减少 `5.7632%` |
| 局部定性 crop | 室外局部 MAE 下降 `12.8%` 到 `32.0%`；混合室内/室外最高局部 MAE 下降 `43.6%` |

**相对 MeshSplatting paper table。**

| 场景 | paper PSNR/SSIM/LPIPS | ours PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|
| bicycle | 23.04 / 0.641 / 0.348 | 23.91 / 0.694 / 0.280 | +0.87 | +0.0527 | -0.0677 |
| flowers | 19.34 / 0.480 / 0.417 | 20.18 / 0.547 / 0.351 | +0.84 | +0.0673 | -0.0660 |
| garden | 24.70 / 0.762 / 0.217 | 26.03 / 0.817 / 0.152 | +1.33 | +0.0551 | -0.0647 |
| stump | 24.78 / 0.678 / 0.316 | 25.36 / 0.713 / 0.282 | +0.58 | +0.0345 | -0.0343 |
| treehill | 20.53 / 0.540 / 0.428 | 21.20 / 0.588 / 0.358 | +0.67 | +0.0482 | -0.0699 |
| room | 28.52 / 0.873 / 0.271 | 29.13 / 0.885 / 0.249 | +0.61 | +0.0119 | -0.0223 |
| counter | 26.51 / 0.846 / 0.279 | 27.24 / 0.864 / 0.250 | +0.73 | +0.0181 | -0.0293 |
| kitchen | 27.42 / 0.858 / 0.227 | 28.00 / 0.877 / 0.199 | +0.58 | +0.0189 | -0.0281 |
| bonsai | 28.19 / 0.876 / 0.294 | 29.78 / 0.898 / 0.257 | +1.59 | +0.0222 | -0.0366 |

**Clean `26000` / `30000` baseline envelope。**

| 场景 | selected | score 26000 | score 30000 | score gap | clean26000 PSNR/SSIM/LPIPS | clean30000 PSNR/SSIM/LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 26000 | 29.857 | 28.894 | +0.963 | 23.30 / 0.660 / 0.332 | 23.02 / 0.641 / 0.347 |
| flowers | 26000 | 22.027 | 21.060 | +0.968 | 19.68 / 0.512 / 0.395 | 19.39 / 0.492 / 0.408 |
| garden | 26000 | 36.604 | 35.623 | +0.981 | 25.03 / 0.780 / 0.201 | 24.71 / 0.762 / 0.216 |
| stump | 26000 | 33.428 | 32.347 | +1.081 | 25.21 / 0.705 / 0.294 | 24.87 / 0.684 / 0.309 |
| treehill | 26000 | 24.104 | 23.124 | +0.980 | 20.93 / 0.565 / 0.406 | 20.65 / 0.545 / 0.421 |
| room | 26000 | 41.446 | 40.575 | +0.871 | 28.75 / 0.885 / 0.250 | 28.48 / 0.873 / 0.268 |
| counter | 26000 | 38.953 | 37.772 | +1.181 | 26.75 / 0.862 / 0.252 | 26.41 / 0.846 / 0.278 |
| kitchen | 26000 | 41.364 | 39.940 | +1.424 | 27.82 / 0.876 / 0.199 | 27.30 / 0.858 / 0.226 |
| bonsai | 26000 | 41.633 | 40.156 | +1.477 | 28.90 / 0.896 / 0.259 | 28.38 / 0.879 / 0.290 |

**几何与拓扑。**

| 场景 | dAbsRel | dDepthMAE | dNormal | 三角形减少 | 顶点减少 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| bicycle | -0.000241 | -0.0204 | -0.0119 | 10.01% | 4.57% | strict all-axis pass |
| flowers | -0.003356 | -0.1250 | -0.0439 | 10.02% | 4.64% | strict all-axis pass |
| garden | -0.000007 | -0.0002 | -0.0010 | 1.50% | 2.69% | geometry-safe |
| stump | -0.005878 | -0.3507 | -0.0260 | 10.02% | 4.57% | strict all-axis pass |
| treehill | -0.001246 | -0.0747 | -0.0122 | 10.01% | 4.86% | strict all-axis pass |
| room | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.03% | geometry-safe |
| counter | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.10% | geometry-safe |
| kitchen | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.29% | geometry-safe |
| bonsai | -0.000368 | -0.0045 | -0.0254 | 10.00% | 3.16% | strict all-axis pass |

## 定性对比

第一组图是公平的全图 held-out render 对比。它的作用是证明比较来自同一 test view 和同一套 selected clean MeshSplatting baseline；但 SPCarNet 当前很多收益属于 residual-level 改善，放在全图尺度上确实不容易被肉眼直接看出来。

<p align="center">
  <img src="assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的全图定性对比">
</p>

更有说服力的定性展示是下面这组局部 held-out error-reduction 图。它由 [`scripts/car_model/generate_spcarnet_advantage_showcase.py`](scripts/car_model/generate_spcarnet_advantage_showcase.py) 自动生成：每个场景先要求该 view 在同一 full9 口径下满足全图 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0`，再在该 view 内寻找纹理区域中 SPCarNet 相对 GT 的局部 RGB 误差下降最大的位置。绿色表示 SPCarNet 比 clean MeshSplatting 更接近 GT，紫红色表示变差。

<p align="center">
  <img src="assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的室外局部 held-out 误差下降对比">
</p>

这组室外 crop 更能体现实际视觉收益：clean MeshSplatting 在花叶、地面纹理、长椅条纹、树皮等位置容易出现局部三角块状平滑或细节丢失；SPCarNet 的 residual repair 会把这些区域拉回到更接近 GT 的状态。另有一组混合室内/室外版本：

<p align="center">
  <img src="assets/spcarnet_m360_where_it_helps_showcase.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的混合局部 held-out 误差下降对比">
</p>

选图清单：`assets/spcarnet_m360_outdoor_detail_selection.json`、`assets/spcarnet_m360_where_it_helps_selection.json`，以及早期全图清单 `assets/spcarnet_m360_full9_gallery_selection.json`。

| 定性 crop | 全图 delta PSNR/SSIM/LPIPS | 局部 dPSNR | 局部 MAE 下降 |
|---|---:|---:|---:|
| flowers / `00014.png` | +0.99 / +0.0616 / -0.0682 | +2.05 | 24.2% |
| garden / `00008.png` | +1.27 / +0.0432 / -0.0551 | +2.70 | 27.6% |
| treehill / `00010.png` | +0.59 / +0.0491 / -0.0881 | +3.03 | 32.0% |
| bicycle / `00021.png` | +1.13 / +0.0385 / -0.0615 | +1.88 | 17.5% |
| stump / `00007.png` | +0.26 / +0.0122 / -0.0208 | +0.81 | 12.8% |
| bonsai / `00001.png` | +2.79 / +0.0063 / -0.0007 | +3.82 | 43.6% |

## 方法概述

当前方法由三个只依赖 train split 的阶段组成。

1. **稀疏遮挡保护的压缩。** CSEF/SOR selector 用训练视角证据给三角形打分。室外场景证据稳定时可以删除约 10% faces；室内几何已经非常稳定时，则启用 micro-budget guard，避免为了压缩数字而破坏指标。

2. **checkpoint-safe 拓扑改写。** 根据 selector 删除面并重写 Mesh Splatting checkpoint，同时保证 face index remap 与 vertex attributes 长度一致。当前版本修复了 room 场景中 trailing unused vertices 导致的渲染 OOM/索引错配问题。

3. **Evidence Lumigraph Adapter。** ELA 用训练渲染得到的 RGB/depth/camera 证据，把局部 residual 信息转移到 held-out view。室内场景使用低分辨率证据，再把 residual 上采样到全分辨率。上采样 alpha 只在 train 视角上选择，并使用 PSNR/SSIM/LPIPS strict filter 加 SSIM-peak guard。

它不是简单的工程补丁，而是一个受约束的决策策略：只有几何证据允许时才压缩，只有训练证据认证 residual 时才修复；否则宁愿 no-op 或 micro-edit，也不提交不安全的“看起来变好”的结果。

### 可选：alpha 选择上的 Frechet-distance 门

alpha selector 提供一个可选的 Frechet 距离信号，作为又一个 train-only non-regression 门，思路移植自 Yang et al., "Representation Frechet Loss for Visual Generation"（[FD-Loss](https://github.com/Jiawei-Yang/FD-Loss)）。对每个 alpha 候选，selector 在 train 校准视角上累计 DINOv2 ViT-B/14 cls 特征（统一在 backbone 的 518x518 输入下打包），估计经验高斯，并计算与 GT batch 的闭式 Frechet 距离。FD 在这里是 calibration 信号，不是训练损失，也从不接触 test GT。

两种使用模式：

- **`--fd_strict`（推荐先用）**：任何 `alpha > 0` 候选只要其 `fd_gain = FD(base, gt) − FD(alpha, gt)` 跌破 `-fd_strict_tol` 就被剔除；`alpha = 0` 永远豁免，作为干净 fallback。这是纯 non-regression 过滤器，**不会**干扰幸存候选之间 PSNR / SSIM / LPIPS 的排序。
- **`--fd_weight w`（高级）**：把 `w * fd_gain` 加进现有 selection score。在 ~32 train views 上 DINOv2 raw FD 的典型量级是 O(5–30)，而其它项是 O(1)，因此 `w` 一旦明显大于 `~0.05` 就会**主导**整个 score。把它当成一个调参旋钮，**不要**当默认值用；做 portfolio 的话首选 `--fd_strict` 单独使用。

默认值与安全网：

- 默认全关（`fd_weight=0`，`fd_strict=False`）；FD 零开销，旧行为按比特一致。
- `alpha=0` 直接复用 base 特征，所以它的 `fd_gain` 严格为 `0`（fallback 行不会因 FD 抖动）。
- 当可用校准视角少于 `--fd_min_views`（默认 8）时整个 FD 路径被跳过，并在 calibration 记录里写 `fd_skipped_reason`。这是为了避开 768-d 经验协方差秩亏、FD 差异主要由噪声而非信号决定的高方差区间。
- 这条路径仅支持单卡（不带 distributed all-gather、不带 streaming queue）。timm 权重下载失败时抛出 `FDBackboneUnavailable` 错误并给出缓存提示，而不是悄悄回退。

代码见 `utils/fd_loss.py` 与 `scripts/car_model/smoke_test_fd_loss.py`（FD 数学 + backbone 前向 + 端到端 `calibrate_alpha` 集成测试，覆盖 alpha=0 豁免与 `fd_min_views` 跳过）。2026-05-11 审计见 `docs/car_model/5-11-FD-Loss-Integration-Audit.md`：`--fd_weight 0.005` 能改善 outdoor mean LPIPS，但会牺牲 PSNR/SSIM，因此不升级为当前 all-axis 主方法。

## 为什么比 MeshSplatting 更好

MeshSplatting 本身已经很强，但 clean checkpoint 仍有视角相关模糊、局部颜色残差、训练迭代过拟合等问题。SPCarNet 在 baseline 外围加了两层控制：

- **几何感知的保守性。** 方法不会假设所有场景都应该同样比例剪枝。garden 和室内场景说明，强行提高压缩率会让论文 claim 变得不公平。
- **train-only 渲染修复。** ELA 不使用 test metric 选择结果，却能恢复局部视觉细节；几何指标仍然由 compact checkpoint 负责，避免用 image-space trick 掩盖拓扑损伤。

因此，当前提升不是“训练更久”或“挑一个更好 checkpoint”：很多 clean `30000` 在 held-out test score 下反而弱于 clean `26000`，而我们仍然超过被选中的 clean baseline。

## 消融总结

| 变体 | 检验内容 | 结果 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | 9 个场景都由 held-out score 选择 clean `26000` |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但不足以带来头条 RGB 提升 |
| Compact + ELA，无 SSIM-peak alpha guard | 单一 scalar score 是否足够 | room 的 PSNR/LPIPS 提升但 held-out SSIM 回退 |
| Compact + ELA，有 SSIM-peak guard | 当前策略 | 修复 room，并在所有室内场景保持同一个 train-only policy |
| 激进剪枝分支 | 是否可以硬推高压缩率 | 被拒绝；敏感场景出现渲染或几何回退 |
| 可选 FD gate（`--fd_weight > 0` 或 `--fd_strict`） | DINOv2 Frechet 距离是否在 LPIPS 之外提供一个额外的 train-only non-regression 门 | 默认关闭；2026-05-11 审计显示它更像 LPIPS-oriented portfolio 信号，会带来 PSNR/SSIM tradeoff，因此不作为主方法 |

更多消融、失败分支和经验教训见文末历史材料链接。

## 复现当前表格

当前留档 run 使用的固定路径：

```bash
OUT_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
POLICY_TAG=sor_adaptive_geo \
METHOD_NAME=ours_26000_sor_adaptive_geo_compact_ela \
CLEAN_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
DATA_ROOT=/data/peilincai/mesh_datasets/mipnerf360 \
SPARSE_OCCLUDER_POLICY=1 \
SPARSE_ADAPTIVE_GEOMETRY_BUDGET=1 \
INDOOR_POLICY_IMAGE_ARG=images_8 \
INDOOR_EVIDENCE_IMAGE_ARG=images_8 \
EVIDENCE_SKIP_FAILED_VIEWS=1 \
WANDB_GROUP=paper_m360_compact_ela_sor_adaptive_geo_26k \
bash scripts/car_model/run_paper_m360_compact_ela_policy_available7.sh
```

收集最终表：

```bash
/home/peilincai/miniconda3/envs/Difix/bin/python \
  scripts/car_model/collect_paper_m360_compact_ela_policy_metrics.py \
  --method_root outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --policy_tag sor_adaptive_geo \
  --method_name ours_26000_sor_adaptive_geo_compact_ela \
  --method_iteration 26000 \
  --out_dir outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --wandb --wandb_project spcarnet_meshprior
```

## 局限与下一步

这个版本值得留档，但还没有真正完成“全面超越 MeshSplatting”的最终目标。

- 平均三角形减少只有 `5.76%`，因为 room、counter、kitchen 被有意限制在 `0.1%` micro-prune。
- strict all-axis pass 是 `5 / 9`，不是 `9 / 9`；剩余场景是 geometry-safe 或 geometry-neutral，而不是严格几何全胜。
- 下一阶段的研究目标是更强的 geometry-preserving compaction，把 indoor/garden 的压缩率拉上去，同时不破坏 RGB、稀疏深度和法向指标。

具体改进规划记录在 [`docs/car_model/5-7-Archive-Full9-CompactELA.md`](docs/car_model/5-7-Archive-Full9-CompactELA.md) 和 representation-level 升级路线 [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)。

## 历史材料

历史阶段日志不再堆在根目录 README 中：

- 旧版英文 README：[`docs/car_model/archive/README_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)
- 旧版中文 README：[`docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)
- 研究日志：[`docs/car_model/SPCarNet_research_log.md`](docs/car_model/SPCarNet_research_log.md)
- 5 月 7 日方法故事线：[`docs/car_model/5-7-Update.md`](docs/car_model/5-7-Update.md)
- Representation-level 升级路线：[`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)
