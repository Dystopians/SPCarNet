# SPCarNet / MeshSplatOpt

**基于训练证据的几何安全 Mesh Splatting 压缩与渲染修复。**

[English](README.md) | [当前版本留档](docs/car_model/5-7-Archive-Full9-CompactELA.md) | [5 月 7 日更新](docs/car_model/5-7-Update.md) | [升级路线](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md) | [ECSR 审计](docs/car_model/5-8-ECSR-CurrentStateAudit.md) | [Phase-A 证据](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md) | [Phase-B graph](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md) | [Policy split](docs/car_model/5-8-ECSR-PolicySplit.md) | [Phase-C preflight](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md) | [执行日志](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md) | [研究日志](docs/car_model/SPCarNet_research_log.md) | [旧版 README](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)

SPCarNet 是建立在 Mesh Splatting 之上的研究分支。当前版本不再依赖手工扫描 prune ratio 来赢指标，而是先用 train split 的证据判断哪些三角形可以安全压缩，再用 train-calibrated Evidence Lumigraph Adapter（ELA）修复 held-out test view 的渲染残差。当前版本已留档：

```text
archive/full9-compact-ela-ssim-peak-20260507
commit fae7942
```

这是一个预期很好的版本，但不是终点：它已经在当前 9 个 Mip-NeRF360 场景上做到 RGB 指标全面优于同口径 clean MeshSplatting，并在当前几何安全准则下保持几何不退化；但平均三角形压缩率仍然偏保守。

## 当前结果

**评估口径。** Mip-NeRF360 同协议复现。每个场景的 clean MeshSplatting baseline 从 clean `26000` 与 `30000` checkpoint 中选择，只使用 held-out test 指标：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

训练集指标不用于选择 baseline，也不用于选择最终 test 结果。

**最终报告。**

- 报告：`outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md`
- W&B collector：`rp0d5gr3`
- 场景：`9 / 9`
- RGB + compact + geometry-safe pass：`9 / 9`
- strict all-axis pass：`5 / 9`
- 相对 selected clean MeshSplatting baseline 的均值提升：`+0.4979 PSNR`，`+0.0158 SSIM`，`-0.0234 LPIPS`
- 平均三角形减少：`5.7632%`

| 场景 | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | 三角形减少 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | 23.9127 | 0.6937 | 0.2803 | +0.6111 | +0.0338 | -0.0518 | 10.01% | strict pass |
| flowers | 20.1828 | 0.5473 | 0.3510 | +0.5005 | +0.0355 | -0.0436 | 10.02% | strict pass |
| garden | 26.0348 | 0.8171 | 0.1523 | +1.0056 | +0.0371 | -0.0490 | 1.50% | geometry-safe |
| stump | 25.3625 | 0.7125 | 0.2817 | +0.1575 | +0.0074 | -0.0123 | 10.02% | strict pass |
| treehill | 21.1984 | 0.5882 | 0.3581 | +0.2642 | +0.0237 | -0.0479 | 10.01% | strict pass |
| room | 29.1310 | 0.8849 | 0.2487 | +0.3837 | +0.0000 | -0.0012 | 0.10% | geometry-safe |
| counter | 27.2404 | 0.8641 | 0.2497 | +0.4886 | +0.0021 | -0.0023 | 0.10% | geometry-safe |
| kitchen | 27.9996 | 0.8769 | 0.1989 | +0.1810 | +0.0005 | -0.0002 | 0.10% | geometry-safe |
| bonsai | 29.7844 | 0.8982 | 0.2574 | +0.8892 | +0.0018 | -0.0021 | 10.00% | strict pass |

## ECSR 升级状态

下一条主线是 **ECSR: Evidence-Certified Surface Relocation**。目标是把 SPCarNet 从 image-space residual repair 推进到 representation-level 的 surface compression 与 appearance recovery。

当前执行产物：

- Current-state audit：[`docs/car_model/5-8-ECSR-CurrentStateAudit.md`](docs/car_model/5-8-ECSR-CurrentStateAudit.md)
- Phase-A train-only surface evidence：[`docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md`](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md)
- Phase-B view-support graph：[`docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md`](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md)
- Phase-A/B cached-view policy split：[`docs/car_model/5-8-ECSR-PolicySplit.md`](docs/car_model/5-8-ECSR-PolicySplit.md)
- Phase-C candidate preflight：[`docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md`](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md)
- 执行日志：[`docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md`](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md)
- Phase-A 汇总 contact sheet：`outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`

Phase-A 结果：`9 / 9` 场景通过 surface addressability，但只有 `4 / 9` 通过当前 top-support multiview consistency 检查。这说明 residual 信号是真实且可回投到表面的，但 naive 的单 face residual delta 还不能作为最终方法。

Phase-B 结果：固定 graph policy 在 full9 上找到 `123` 个 train-only local support cluster，其中 `23` 个是 certificate-contraction candidates，`99` 个是 surface-attribute recovery candidates。但 residual-hot cluster 的直接三角形压缩上限很小，因此下一步必须把 compression candidate 和 appearance-recovery candidate 分开，而不能把 residual hotspot 当成压缩目标。

Phase-C preflight 结果：`21 / 123` 个 Phase-B cluster 通过 train-only fitting/policy-val support-mask preflight，其中 `13` 个是 contraction 类型，`8` 个是 attribute-recovery 类型。它们还不是被接受的 ECSR 修改，只是进入 topology smoke test 与 before/after local rendering certificate 的第一批候选。

## 其他评估口径

下面所有表都来自同一份 full9 报告。LPIPS、AbsRel、DepthMAE、Normal 越低越好。

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
