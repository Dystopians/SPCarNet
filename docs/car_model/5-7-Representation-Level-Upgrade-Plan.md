# 5-7 Representation-Level Upgrade Plan For SPCarNet

本文档记录下一阶段 SPCarNet 的主线升级方向：从当前的 **image-space residual repair** 推进到 **representation-level geometry/appearance relocation**。目标不是继续调 prune ratio、ELA alpha 或展示图，而是把当前已经验证有效的局部 residual 证据迁移回 MeshSplatting 的 mesh/splat 表示本身，使方法更像一个可发表的研究贡献。

## 1. 当前判断

当前 full9 版本已经证明 SPCarNet 可以在同口径 selected clean MeshSplatting baseline 之上取得 `9 / 9` RGB + compact + geometry-safe pass。但它仍有三个明显短板：

1. **视觉优势不够强。**
   全图定性图中差异经常很细，只有局部 crop 和 error-reduction map 才容易看出 SPCarNet 的收益。这说明当前优势主要集中在 residual-level correction，而不是全局可见的 representation 改善。

2. **压缩率保守。**
   当前 mean triangle reduction 只有 `5.7632%`。room/counter/kitchen 被限制在 `0.1%`，garden 只有 `1.5%`，说明直接删 face 在几何敏感场景上太粗暴。

3. **方法形态容易被质疑为后处理。**
   ELA 使用 train evidence，不是 test leakage，但如果它长期停留在渲染后的 2D residual repair，论文贡献会被审稿人理解成一个 renderer-side correction，而不是对 MeshSplatting representation 的根本推进。

因此，下一步应该把问题从：

> 如何让 render 结果更像 GT？

升级为：

> 如何在 train evidence 认证下压缩冗余 mesh support，并把被压缩区域的 appearance residual 迁移到保留的 surface support 上？

## 2. 核心研究假设

**Hypothesis.** MeshSplatting 中存在一类 view-support redundant surface primitives：它们对多个训练视角的投影覆盖、深度、法向和颜色 residual 支撑高度重叠。直接删除这些 primitives 会破坏局部外观或几何；但如果将它们收缩/合并到邻近可靠 support，并把其 appearance residual 重定位到保留 support 上，则可以同时获得：

- 更高的 triangle reduction；
- 更稳定的 sparse depth / normal；
- 更强的 full-frame RGB / LPIPS；
- 更清晰的局部视觉细节；
- 更像 representation-level 的论文贡献。

这个方向可命名为：

**Evidence-Guided Surface Residual Relocation**, 简写为 **ESR**。

## 3. 参考工作与启发

### 3.1 MeshSplatting

MeshSplatting 的核心价值在于它不是纯 image-space representation，而是 opaque mesh differentiable rendering，并联合优化 geometry 与 appearance。它面向 AR/VR/game-engine-style mesh pipeline。因此，SPCarNet 若要真正超越 MeshSplatting，最好在 mesh/appearance representation 本身产生贡献，而不是长期停留在 render 后修补。

Reference:
- MeshSplatting: Differentiable Rendering with Opaque Meshes, <https://arxiv.org/abs/2512.06818>

### 3.2 3DGS 压缩与恢复

LightGaussian 说明了 pruning 之后需要 recovery / distillation 才能保持质量；Compact 3DGS 说明压缩不仅是删 primitive，还涉及 view-dependent color、mask、residual codebook 等 appearance capacity 的重新组织。这启发我们：SPCarNet 不应该只做 triangle deletion，而应该在压缩的同时重分配 appearance residual。

References:
- LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS, <https://arxiv.org/abs/2311.17245>
- Compact 3D Gaussian Splatting For Static and Dynamic Radiance Fields, <https://arxiv.org/abs/2408.03822>

### 3.3 冗余 primitive 的结构化判断

Scaffold-GS 与 Mini-Splatting 都说明高质量 splatting representation 中存在大量 spatial / structural redundancy。冗余不是孤立 primitive 的属性，而是局部结构与视角支撑共同决定的。因此下一步应构建 view-support redundancy graph，而不是逐 face 单独打分。

References:
- Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering, <https://arxiv.org/abs/2312.00109>
- Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians, <https://arxiv.org/abs/2403.14166>

### 3.4 Neural texture / neural deferred rendering

Deferred Neural Rendering 与 Neural Deferred Shading 证明了将 appearance information 存回 mesh proxy / surface feature，再由 renderer 解释，是一条有效路线。SPCarNet 的 ELA residual 可以被看成 train-evidence 估计出的局部 appearance signal；下一步应把它从 2D correction 推进到 surface-attached residual。

References:
- Deferred Neural Rendering: Image Synthesis using Neural Textures, <https://arxiv.org/abs/1904.12356>
- Multi-View Mesh Reconstruction with Neural Deferred Shading, <https://arxiv.org/abs/2212.04386>

### 3.5 Mesh simplification

经典 QEM 证明 edge/vertex contraction 比单纯 face deletion 更合理。SPCarNet 不应直接复刻 QEM，而应提出 train-evidence-certified contraction：用投影、深度、法向、颜色 residual、稀疏遮挡等证据判断哪些局部 contraction 是安全的。

Reference:
- Surface Simplification Using Quadric Error Metrics, <https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf>

## 4. Proposed Method: ESR

ESR 包含四个模块。

### 4.1 View-Support Redundancy Graph

为 mesh faces 构建局部图结构。节点是 face 或 face cluster，边表示两个局部 support 在 train views 中是否可被认为冗余：

- 多训练视角共同可见；
- 投影覆盖区域重叠；
- depth residual 一致；
- normal direction 一致；
- photometric residual 可互相解释；
- 删除或收缩其中一个是否会破坏 sparse occluder support。

输出不是单个 prune score，而是一组 candidate contraction groups。

### 4.2 Certificate-Carrying Triangle Contraction

将当前的 face deletion 替换为更温和的 contraction / merge 操作。每个候选操作需要携带 train-only certificate：

- train RGB 不退；
- train SSIM 不退；
- train LPIPS 不退；
- sparse depth 不退；
- normal 不退；
- local topology 不产生 degenerate faces；
- target triangle reduction 实际增加。

只有通过 certificate 的 contraction 才能进入 compact checkpoint。

### 4.3 Surface Residual Relocation

将当前 ELA 学到的 residual 从 image plane 迁移到被保留的 surface support：

- 对每个 contraction group，估计被压缩区域的 train-view residual；
- 将 residual 聚合到保留 face/vertex/local splat attribute；
- rendering 时由保留 support 解释局部 residual；
- 允许保留一个轻量 held-out projection adapter，但主收益应尽量来自 representation。

这一步是从“后处理修图”升级到“representation-level appearance relocation”的关键。

### 4.4 Train-Only Pareto Policy

从 train views 中划分 policy-validation subset，形成内部选择口径：

- fitting train views 用来估计 residual 和候选 contraction；
- policy-validation views 用来选择是否接受候选；
- held-out test views 只用于最终报告，不能参与策略选择。

最终 policy 不允许 per-scene 手工扫描；只允许 scene evidence 自适应。

## 5. Implementation Roadmap

### Phase A: Diagnostic Prototype

目标：证明 residual relocation 有必要且有信号。

任务：

- 从当前 ELA 输出中提取 per-pixel residual；
- 通过 rasterized face id 或 nearest surface projection 回投到 face/vertex；
- 统计 residual 是否集中在可解释的 local support 上；
- 在 flowers/garden/treehill/bicycle 上生成 residual-on-surface heatmap。

晋升标准：

- residual 与局部 texture/geometry support 有明显相关性；
- top residual regions 与 qualitative failure crops 对齐；
- 不依赖 test view 选择。

### Phase B: Redundancy Graph

目标：替代单 face prune score。

任务：

- 构建 face adjacency；
- 为相邻 face pair / cluster 计算 view-support overlap；
- 加入 depth/normal/photometric consistency；
- 输出 contraction candidates；
- 在 compact-only 模式下验证不会复现此前 room/garden 失败。

晋升标准：

- indoor/garden 能提出比 deletion 更安全的候选；
- compact-only geometry 不退；
- 至少不低于当前 triangle reduction。

### Phase C: Certificate-Contraction

目标：把 deletion 换成更安全的 topology operation。

任务：

- 实现 local face/edge contraction prototype；
- 保持 MeshSplatting checkpoint tensor consistency；
- 加入 degenerate face cleanup；
- 加入 train-only Pareto certificate；
- 跑 bicycle/flowers/garden/treehill/room 五场景中程验证。

晋升标准：

- RGB 不低于当前 compact-only；
- sparse depth / normal 不低于当前；
- garden 和室内 triangle reduction 明显超过当前 micro-prune。

### Phase D: Residual Relocation

目标：把 ELA 的主要收益迁移回 representation。

任务：

- 将 residual attribute 附着到保留 face/vertex/local support；
- 设计简单可控的 renderer-side读取逻辑；
- 用 policy-validation views 选择 residual strength；
- 对比 pure ELA、compact-only、ESR。

晋升标准：

- 相比当前 Compact-ELA，至少在 outdoor scenes 上进一步提升 LPIPS 或局部感知指标；
- README qualitative crop 更明显；
- 不增加 test-time leakage；
- 不是只靠后处理 alpha 赢。

### Phase E: Full9 Same-Protocol Validation

目标：形成新的主结果。

必须重新跑：

- selected clean MeshSplatting baseline: clean `26000/30000` held-out score；
- full9 SPCarNet ESR；
- W&B logging；
- geometry JSON；
- per-view metrics；
- qualitative full-frame + local-error panels；
- ablations。

晋升标准：

- 保留当前 `9 / 9` RGB + compact + geometry-safe；
- mean triangle reduction 显著高于 `5.7632%`；
- strict all-axis pass 高于 `5 / 9`，或至少 garden/indoor geometry 明确改善；
- qualitative improvement 比当前 crop 更直观；
- 方法解释从 renderer-side correction 升级为 representation-level compression and appearance relocation。

## 6. Ablation Plan

必须包含：

| variant | purpose |
|---|---|
| Clean MeshSplatting `26000/30000` | same-protocol baseline envelope |
| Current Compact-ELA/SOR | current archived best |
| Redundancy graph + deletion | test graph benefit without contraction |
| Certificate contraction only | isolate topology operator |
| ELA only | isolate image-space repair |
| Surface residual relocation only | test representation-attached residual |
| Full ESR | final combined method |
| No policy-validation split | expose overfitting risk |
| No geometry certificate | show why depth/normal guard is necessary |

## 7. Expected Risks

1. **Renderer integration risk.**
   Surface-attached residual may require touching MeshSplatting renderer internals. If too invasive, start with a minimal per-face/vertex residual attribute and a lightweight lookup path.

2. **Contraction may damage checkpoint consistency.**
   Current room bug showed that topology surgery is fragile. Every contraction operation needs a smoke test for tensor length, face index range, degenerate faces, and renderer memory.

3. **Residual relocation may still look like post-processing.**
   To avoid this, the paper claim must report representation-attached residual ablation separately from final image-space ELA.

4. **Local gains may not become full-frame gains.**
   This would mean ESR is useful for qualitative details but not enough for headline metrics. Then the method must introduce stronger global appearance capacity or accept a narrower claim.

## 8. Why This Moves SPCarNet Toward A Top-Conference Paper

This direction is stronger than continued parameter search because it changes the research object:

- from face deletion to evidence-certified contraction;
- from 2D residual repair to surface residual relocation;
- from per-scene parameter tuning to train-only Pareto policy;
- from "we improve renders" to "we restructure a mesh-splat representation under explicit evidence constraints."

If successful, the paper story becomes:

> We identify train-view-certified redundant surface support in MeshSplatting, compact it through geometry-safe contraction, and relocate the lost appearance residual onto retained surface primitives. This yields a representation-level method that improves quality, compression, and geometry reliability under a fair held-out protocol.

That is materially closer to a top-conference contribution than the current stable version. The current version remains an important baseline and proof of signal, but ESR is the route that could make the work genuinely competitive.

## 9. Stop / Continue Criteria

Continue only if at least one of the following is true after Phase C/D:

- mean triangle reduction improves clearly over `5.7632%` without losing current RGB wins;
- strict all-axis pass increases beyond `5 / 9`;
- local qualitative gains become visibly stronger and are supported by local metrics;
- representation-attached residual matches most of ELA's benefit.

Stop or redesign if:

- improvements require per-scene manual parameter selection;
- full9 falls below current archived RGB pass rate;
- gains only appear in test-selected examples;
- the method cannot be explained as representation-level after ablation.
