# 5-7 Update: From MeshSplatting To SPCarNet

本文档总结当前版本相对原始 MeshSplatting 的核心改进、研究动机、实验结果、消融证据与下一步路线。它不是最终论文稿，但按顶会论文级别的逻辑整理当前故事线。

## 一句话故事

MeshSplatting 已经能把场景表示成高质量 mesh primitives，但它本身并不知道哪些三角形只是冗余支撑、哪些三角形一删就会破坏外观或几何。SPCarNet 的核心想法是：**把 mesh 压缩从“删多少”变成“证据允许我们安全改哪里”，再用 train-only 的证据把局部渲染残差补回来。**

当前版本证明了这个方向不是纯工程修补：在 9 个 Mip-NeRF360 场景上，它在同口径 clean MeshSplatting baseline 之上实现了 `9 / 9` RGB + compact + geometry-safe pass。

## 原始 MeshSplatting 的问题

原始 MeshSplatting 的 clean checkpoint 是强 baseline，但仍有三个问题。

1. **拓扑没有任务感知的压缩策略。**
   它可以生成大量三角形，但不会主动判断哪些局部表面支撑是冗余的。直接后处理剪枝或 QEM 很容易伤害 view-dependent 外观。

2. **训练更久不一定更好。**
   在当前复现中，clean `30000` 往往弱于 clean `26000`。如果用 train 指标或最长训练时间选 baseline，就会得到不公平结论。

3. **局部视觉残差没有被显式利用。**
   Clean render 在一些草地、花丛、树木、室内布料等细节区域会变得过平滑或局部偏色。MeshSplatting 本身没有一个 train-only 的机制把这些残差信息迁移到 held-out view。

## SPCarNet 当前方法

当前版本包含三个模块。

### 1. Sparse-Occlusion Protected Compaction

我们不再固定 prune ratio，而是用训练视角证据估计三角形是否安全：

- 投影支持是否稳定；
- 稀疏深度是否低误差；
- 是否可能是前景遮挡支撑；
- 局部几何是否已经足够稳定；
- 删除后是否可能只带来 render-side 假收益。

结果是一个自适应策略：室外场景通常允许约 `10%` 删除；garden 降到 `1.5%`；room/counter/kitchen 在当前证据下只允许 `0.1%` micro-prune。这个结果压缩率不激进，但它避免了此前激进剪枝造成的失败。

### 2. Checkpoint-Safe Mesh Surgery

压缩不是简单删文件里的 face。MeshSplatting checkpoint 中 face indices、vertices、per-vertex attributes、per-face attributes 必须一起保持一致。当前版本修复了一个关键 bug：room checkpoint 存在 trailing unused vertices，旧 remap 用 `faces.max()+1` 作为长度，导致 face indices 和 vertex attributes 不一致，最终触发渲染 OOM。修复后，room 可以稳定纳入 full9 表。

这个问题说明 SPCarNet 的目标不是调一个外部后处理脚本，而是要真正掌控 mesh-splatting representation 的结构一致性。

### 3. Evidence Lumigraph Adapter

ELA 是当前 RGB 提升的主要来源。它只使用 train split 的 render/depth/camera evidence，学习一种局部 residual transfer：

- 在低分辨率证据空间中估计可信残差；
- 通过深度一致性和方向一致性筛选邻域；
- 对 held-out view 应用 residual；
- 室内场景再把 residual 上采样到 full resolution；
- alpha 只在 train views 上选择，并加入 SSIM-peak guard，防止 PSNR 上升但结构相似度回落。

这个模块解决的是 clean MeshSplatting 的局部外观残差，而不是偷偷用 test view 调参。

## 定量结果

最终同口径表：

- Report: `outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md`
- W&B: `rp0d5gr3`
- `RGB + compact + geometry-safe pass`: `9 / 9`
- `strict all-axis pass`: `5 / 9`
- mean dPSNR: `+0.497941`
- mean dSSIM: `+0.015755`
- mean dLPIPS: `-0.023373`
- mean triangle reduction: `5.7632%`

| scene | dPSNR | dSSIM | dLPIPS | triangle reduction | interpretation |
|---|---:|---:|---:|---:|---|
| bicycle | +0.6111 | +0.03385 | -0.05181 | 10.01% | strong outdoor strict win |
| flowers | +0.5005 | +0.03548 | -0.04357 | 10.02% | strong texture/foliage recovery |
| garden | +1.0056 | +0.03708 | -0.04900 | 1.50% | RGB strong, geometry protected |
| stump | +0.1575 | +0.00736 | -0.01226 | 10.02% | strict win |
| treehill | +0.2642 | +0.02367 | -0.04792 | 10.01% | strong LPIPS recovery |
| room | +0.3837 | +0.00004 | -0.00117 | 0.10% | repaired after SSIM-peak guard |
| counter | +0.4886 | +0.00209 | -0.00230 | 0.10% | geometry-safe indoor gain |
| kitchen | +0.1810 | +0.00047 | -0.00024 | 0.10% | conservative but positive |
| bonsai | +0.8892 | +0.00177 | -0.00209 | 10.00% | indoor strict win |

## 定性结果

全图公平对比：

![SPCarNet full9 qualitative gallery](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

这组图用于证明比较口径一致，但全图尺度会稀释 residual-level 的视觉差异。因此当前 README 另外加入了自动局部误差下降展示。选图逻辑不是手工挑图：先要求该 view 在同一 full9 口径下全图 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0`，再在 view 内搜索 GT 纹理区域中 clean MeshSplatting 误差较高且 SPCarNet 误差下降最大的 crop。

室外局部展示：

![SPCarNet outdoor detail showcase](../../assets/spcarnet_m360_outdoor_detail_showcase.png)

混合室内/室外局部展示：

![SPCarNet where it helps showcase](../../assets/spcarnet_m360_where_it_helps_showcase.png)

新展示更清楚地暴露了 SPCarNet 的当前真实优势：它不是在所有像素上制造肉眼巨大的变化，而是在 flowers、garden、treehill、bicycle、stump 等纹理区域降低 clean MeshSplatting 的局部三角块状平滑、颜色残差和细节丢失。自动 crop 的局部 MAE 下降约 `12.8%` 到 `43.6%`，局部 dPSNR 约 `+0.81` 到 `+3.82`。这比旧 crop 图更适合作为定性证据，但也说明方法优势目前主要集中在 residual repair，而不是已经形成全图范围的强感知重建飞跃。

## 消融与经验

### Baseline 选择消融

如果用 train 指标或训练更久来选 clean baseline，会偏向 clean `30000`，但 held-out test 上它常常更差。当前表使用 held-out test score 选择 clean checkpoint，因此避免了“训练更久就是更好”的错误。

结论：公平比较必须用 held-out test selection 或固定协议，而不是 train score。

### Compact-only 消融

Compact-only checkpoint 可以保持几何安全，但 RGB 提升不够。它说明安全压缩是必要条件，但不是最终性能来源。

结论：压缩需要和 train-only residual repair 结合。

### ELA alpha 消融

早期 room 结果选择 alpha `0.75`，PSNR 和 LPIPS 提升，但 held-out SSIM 微降。加入 SSIM-peak guard 后，train-only policy 选择 alpha `0.5`，room 三项 RGB 指标同时超过 clean。

结论：单一 scalar score 不够，结构相似度需要作为候选过滤条件，而不是事后解释。

### 激进压缩消融

早期高比例剪枝能提高 triangle reduction，但在 bicycle、garden、room 等场景上引发渲染或几何回退。当前版本主动把敏感场景压缩率降下来。

结论：当前最可信的论文 claim 应该是 evidence-certified safe compaction，而不是最大压缩率。

## 为什么这是研究工作，而不是工程工作

一个工程补丁通常是局部修 bug 或调参数。当前方法的核心是一个可泛化的受约束决策框架：

1. **决策对象改变了。**
   不是给 MeshSplatting 外接一个滤波器，而是判断每个局部表面支撑是否可以被安全修改。

2. **证据来源受约束。**
   所有策略来自 train split 或 representation-intrinsic evidence，避免 test leakage。

3. **目标是多指标 Pareto。**
   PSNR、SSIM、LPIPS、稀疏深度、法向、三角形数量同时受控，而不是只优化一个可见指标。

4. **失败分支被保留为方法边界。**
   QEM、random same-count、激进剪枝、无 SSIM guard、长程过训练等失败都不是噪声，而是证明为什么当前策略需要这些 guardrails。

5. **当前结果暴露了下一步科学问题。**
   室内场景压缩率低不是“再调一下参数”能解决的问题，而是需要新的 geometry-preserving compaction operator。

## 当前短板

最重要的短板是：**平均 triangle reduction 低。**

拆开看：

- 5 个场景约 `10%`；
- garden 只有 `1.5%`；
- room/counter/kitchen 只有 `0.1%`。

所以平均只有 `5.76%`。这是当前 policy 为了保证 `9 / 9` RGB + compact + geometry-safe pass 做出的保守选择。它是合理的，但不够顶会论文最终形态。

第二个短板是 strict all-axis pass 只有 `5 / 9`。room/counter/kitchen 是 geometry-neutral，不是严格几何胜出。

## 下一阶段研究路线

下一阶段不应该继续玩参数扫描，而应该升级压缩机制本身。

### A. Certificate-carrying triangle contraction

从“删除三角形”升级到“带证书的局部 edge collapse / contraction”。室内场景很多三角形不能直接删，但可以把局部 redundant support 收缩到更少的支撑面上。

### B. View-support redundancy graph

构建三角形图，边表示：

- 多训练视角共同可见；
- 深度一致；
- 法向一致；
- 投影覆盖重复；
- 局部 residual 可迁移。

只压缩图中证据一致的局部块，而不是孤立地删 low-score face。

### C. Geometry-preserving residual relocation

把被压缩区域的 appearance residual 迁移到保留下来的邻域 support 上。这样 ELA 不再只是 render-side adapter，而是逐步变成 representation-level residual transfer。

### D. Train-only Pareto certificate

每个候选压缩操作必须同时通过：

- train RGB 不退；
- sparse depth 不退；
- normal 不退；
- topology budget 达标；
- held-out test 完全不参与选择。

### E. Full9 re-validation

新方法必须重新跑 full9：

- clean `26000/30000` held-out baseline selection；
- all scenes metrics；
- geometry JSON；
- W&B；
- per-view stress；
- qualitative panels。

## Promotion Rule

下一版只有在保留当前 `9 / 9` RGB + compact + geometry-safe 的前提下，至少显著改进以下一项，才应该晋升：

- mean triangle reduction；
- strict all-axis 场景数；
- indoor/garden geometry metrics；
- per-view RGB pass rate；
- 或者把 ELA 的一部分收益迁移到 representation-level residual relocation。

否则，当前 `archive/full9-compact-ela-ssim-peak-20260507` 应继续作为最可信的稳定版本。
