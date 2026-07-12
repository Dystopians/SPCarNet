# SPCarNet 当前方法完整技术报告

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可安全主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`

## 0. Executive Summary

SPCarNet 是一个建立在 MeshSplatting 之上的训练证据驱动压缩与修复闭环。它不把 MeshSplatting 当作被替换的 baseline，而是把 MeshSplatting checkpoint 作为强基础表示，然后用训练视角中的 surface/render evidence 判断两件事：

1. 哪些三角形可以安全压缩；
2. 哪些局部外观 residual 可以被可靠转移到新视角。

当前最适合放进主 PPT 的版本是 Phase-J：

```text
clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded adaptive policy / edge fallback
  -> held-out evaluation
```

核心结论：

| 维度 | 当前结论 |
|---|---|
| 主方法 | Phase-J guarded adaptive Evidence Lumigraph Adapter on compact MeshSplatting |
| 公平 baseline | 本地同协议 selected clean MeshSplatting，clean `26000/30000` 只用 held-out test score 选更强者 |
| Mip-NeRF360 full9 | `9 / 9` 场景相对 selected clean baseline 三指标严格胜出 |
| 平均 RGB 提升 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310` |
| 最新 v31 状态 | teacher-surface residual cache 与 face-local SH1 fitting 接口已打通；Bonsai full-res 公平评估接近 compact base，尚未推广为主结果 |
| 重要边界 | 最强 RGB 收益仍主要来自 render-time ELA；representation-level baking 目前尚未形成同等强的稳定收益 |

一句话版本：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% 三角形；但最强外观收益仍来自 render-time ELA，下一步需要把修复进一步内化到 representation-level checkpoint。

最新状态补充：

> v31 已经把 teacher render residual 写进 surface evidence cache，并让 face-local SH1 residual operator 能直接读取 `teacher_residual_rgb / teacher_residual_l1`。这说明“把 ELA teacher 转成 surface-addressed basis”的工程路径已通，但 Bonsai full-res 公平结果仍几乎等于 compact base，说明当前 12-face / +36-vertex 的局部 SH1 edit 容量太小，不能作为 headline。

## 1. 背景与问题定义

MeshSplatting 的优势是输出 triangle mesh，比 Gaussian 或点云更容易接入传统渲染、游戏、AR/VR、数字孪生和下游几何管线。但本地复现和长期审计暴露出三个尚可改进的点：

| 问题 | 现象 | 影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处出现颜色偏差或模糊 | 全图指标和局部视觉质量仍有提升空间 |
| 拓扑冗余 | 部分 faces 对多视角解释贡献低，甚至属于低风险冗余面 | 可以做 rate-distortion 优化 |
| 训练更久不一定更好 | 当前 full9 clean baseline envelope 中 clean `30000` 全部弱于 clean `26000` | 提升不能解释为简单训练更久或挑 checkpoint |

SPCarNet 的研究假设：

> MeshSplatting 已经学到强基础表示，但训练视角里仍然含有可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删掉部分冗余 geometry，并把训练 residual 转移到 held-out view 来修复外观。

## 2. 与原始 MeshSplatting 的本质区别

| 维度 | clean MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 表示 | 原始 opaque triangle mesh checkpoint | compact mesh checkpoint + train-evidence residual adapter |
| 几何处理 | 不做额外删面策略 | sparse-occlusion protected compaction |
| 外观修复 | checkpoint 属性直接渲染 | Evidence Lumigraph Adapter 用训练 residual 修复 held-out render |
| 决策依据 | 默认训练产物 | train-only calibration、policy-val gate、fallback |
| test GT 使用 | 最终评价 | 只做最终评价，不参与方法选择 |
| 失败处理 | 无显式机制 | gate 不通过就 fallback/no-op |

通俗解释：

> 原始 MeshSplatting 是“训练出一个网格然后交付”。SPCarNet 是“训练出网格后，再让网格基于训练视角做体检：哪里能安全删，哪里容易错，哪里有可靠 residual 可以修。证据不足时宁可不动”。

## 3. 方法模块

### 3.1 Train-View Evidence Mining

在训练视角上对 baseline 或 compact checkpoint 渲染，并缓存：

- rendered RGB；
- GT RGB；
- residual `GT - render`；
- per-face visibility；
- per-face hit count / pixel support；
- support-view consistency；
- high-error connected regions；
- depth/surface hit evidence；
- view-dependent residual statistics。

这一阶段只使用训练视角，不使用 held-out test GT 做策略选择。

### 3.2 Sparse-Occlusion Protected Compaction

目标不是追求最大删面比例，而是在 RGB、sparse geometry 和拓扑安全之间做保守 rate-distortion 优化。

三角形是否可以压缩主要由训练证据判断：

- 多视角 visibility 足够稳定；
- face 不是关键 occlusion boundary；
- face 不属于高 residual 解释核心；
- 删除后 policy-val render 没有明显退化；
- sparse geometry audit 没有 AbsRel、DepthMAE、Normal 风险；
- 室内场景启用 micro-budget，避免为了压缩数字破坏已经很强的 geometry。

报告中的 triangle reduction 是删除的三角形占比，不是剩余比例。

### 3.3 Checkpoint-Safe Topology Rewrite

压缩结果会真正写回 MeshSplatting checkpoint：

- 删除 faces；
- remap face indices；
- remap vertices；
- 清理 trailing unused vertices；
- 保证 tensor shape 与 renderer 一致；
- 保证后续 render、metric 和 geometry audit 可运行。

这意味着 SPCarNet 的压缩不是 report-only 后处理，而是 materialized checkpoint edit。

### 3.4 Evidence Lumigraph Adapter

ELA 是当前 Phase-J 视觉收益的主要来源。对训练 support view，先定义 residual：

```text
residual_s(x) = GT_s(x) - Render_s(x)
```

对 target held-out view，系统根据相机、depth、surface hit 和 support confidence，把多个训练 support residual 转移并聚合到 target image：

```text
ResidualEvidence_t(x)
  = aggregate_warped_residuals(
      residual_s,
      camera_geometry,
      surface_hit,
      support_consistency,
      edge/high-frequency evidence
    )
```

最终输出：

```text
Render_final_t(x)
  = Render_base_t(x) + alpha_t(x) * ResidualEvidence_t(x)
```

`alpha` 不是手工为每个场景调出来的固定参数。Phase-J 使用 train-only calibration 和 guarded policy 自动决定：

- adaptive alpha 是否可用；
- structural edge fallback 是否更安全；
- 当前 candidate 是否需要回退。

### 3.5 Guarded Adaptive Policy

Phase-J 的核心不是单一 ELA 公式，而是 guarded portfolio：

- `8 / 9` 场景采用 adaptive-alpha branch；
- `treehill` 使用 train-selected structural edge fallback；
- 所有 branch 只用 train/policy-val evidence 选择；
- gate 不通过时 fallback，而不是强行接受。

主要 gate：

- PSNR non-regression；
- SSIM regression 上限；
- LPIPS regression 上限；
- balanced score 不为负；
- tail / region-risk 不明显恶化；
- sparse geometry 安全；
- no-test-GT branch selection。

这也是当前工作可信的关键：pipeline 会拒绝很多看起来“可能有效”的小改动，而不是把单场景或 test-only 假阳性包装成主结果。

## 4. 实验协议

### 4.1 数据集

主结果使用 Mip-NeRF360 full9：

```text
bicycle, flowers, garden, stump, treehill,
room, counter, kitchen, bonsai
```

### 4.2 Baseline Envelope

每个场景的 clean MeshSplatting baseline 从 clean `26000` 和 clean `30000` checkpoint 中选择。选择只依据 held-out test 指标的统一 score：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

注意：

- baseline 选择使用 held-out test，是为了构造更强 clean envelope；
- 我们方法的 alpha、edge fallback、branch 和压缩策略不使用 held-out test GT；
- held-out test 只用于最终 report-only 评价；
- 当前 full9 中 selected clean baseline 全部选择 clean `26000`，说明 clean `30000` 并没有因为训练更久而更强。

### 4.3 主要证据路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

## 5. 主定量结果

### 5.1 Full9 Scene Table

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | tri red. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.0215 | 0.7024 | 0.2661 | +0.7199 | +0.0425 | -0.0660 | 11.81% |
| flowers | adaptive alpha | 20.3044 | 0.5578 | 0.3292 | +0.6221 | +0.0459 | -0.0653 | 11.82% |
| garden | adaptive alpha | 26.3111 | 0.8278 | 0.1358 | +1.2819 | +0.0478 | -0.0655 | 3.47% |
| stump | adaptive alpha | 25.5951 | 0.7241 | 0.2639 | +0.3901 | +0.0189 | -0.0301 | 11.82% |
| treehill | edge fallback | 21.2962 | 0.5956 | 0.3363 | +0.3620 | +0.0311 | -0.0697 | 11.81% |
| room | adaptive alpha | 30.3056 | 0.9057 | 0.1960 | +1.5584 | +0.0209 | -0.0539 | 2.10% |
| counter | adaptive alpha | 28.4492 | 0.8937 | 0.1865 | +1.6974 | +0.0317 | -0.0655 | 2.10% |
| kitchen | adaptive alpha | 30.1997 | 0.9161 | 0.1320 | +2.3812 | +0.0396 | -0.0672 | 2.10% |
| bonsai | adaptive alpha | 31.8620 | 0.9303 | 0.1726 | +2.9668 | +0.0339 | -0.0869 | 11.80% |

Mean delta versus selected clean MeshSplatting:

```text
dPSNR  = +1.3311
dSSIM  = +0.0347
dLPIPS = -0.0634
mean triangle reduction = 7.6479%
```

### 5.2 Stability and Geometry

| audit | result |
|---|---|
| Scene-level strict RGB wins vs selected clean | `9 / 9` |
| Per-view strict RGB wins vs selected clean | `244 / 246` held-out views |
| Sparse geometry-safe scenes | `9 / 9` |
| Sparse geometry strict wins | `6 / 9` |
| Mean triangle reduction | `7.6479%` |

### 5.3 Relation to MeshSplatting Paper Table

这部分可以作为 sanity check，不应替代本地公平 baseline claim。

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table, Mip-NeRF360 mean | 24.7800 | 0.7280 | 0.3100 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J minus paper table | +1.7028 | +0.0557 | -0.0857 |
| Phase-J minus local selected clean | +1.3311 | +0.0347 | -0.0634 |

推荐表述：

> 更严谨的主 claim 是相对本地同协议 selected clean MeshSplatting。paper table 说明我们的同口径结果没有低于论文数值，但 paper table 不是最强公平 baseline。

## 6. 定性结果

### 6.1 推荐主图

最推荐 PPT 使用：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

这张图基于当前接受 endpoint `ours_26000_phasej_guarded_adaptedge_ela` 自动生成。选择逻辑：

1. 每个候选 view 先要求全图 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0`；
2. 再在纹理区域内寻找 SPCarNet 相对 GT 的局部 RGB 误差下降最大 patch；
3. 绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

```md
![SPCarNet Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)
```

建议讲法：

> 全图上差异通常是 residual-level，肉眼不一定一眼能看出；因此我们用同一 held-out 协议下的局部误差下降图展示哪里确实更接近 GT。它不是手工挑 test 指标，而是从 closure audit 和 per-view delta 自动筛选。

### 6.2 Backup Figures

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png
```

使用建议：

- full-frame gallery 用于证明 baseline 与 ours 的公平同视角对比；
- Phase-J local showcase 用于主讲视觉收益；
- outdoor detail showcase 用于回答“室外是否也有效”；
- Phase-S PatchCert 图用于说明 representation-level 分支是真实 checkpoint edit，但当前不是主 RGB endpoint。

## 7. Ablation and Diagnostics

### 7.1 主方法消融

| 变体 | 检验内容 | 结论 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | full9 均选择 clean `26000`；训练更久不是主要解释 |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但 RGB headline 不足 |
| compact + ELA without SSIM-peak guard | 单 scalar score 是否足够 | 部分场景 PSNR/LPIPS 提升但 SSIM 有风险 |
| compact + guarded adaptive ELA | 当前 Phase-J | full9 `9 / 9` 三指标严格胜出 |
| aggressive pruning | 能否强推压缩率 | 被拒绝；敏感场景出现 geometry/render 风险 |
| optional FD gate | DINOv2 Frechet distance 是否能作为额外 train-only non-regression signal | 默认关闭；更像 LPIPS-oriented portfolio signal，会带来 PSNR/SSIM tradeoff |

### 7.2 近期 v26-v31 探针

| 版本 | 目的 | 状态 | 对当前路线的启发 |
|---|---|---|---|
| v26 hard local-trust | 用二值 trust gate 限制 residual | Bonsai medium 完成，过保守，容易 no-op | hard gate 不够连续 |
| v27 soft local-trust | 连续 trust-weight residual | 接口、smoke、Bonsai medium 完成，但 honest gate 拒绝 | trust 修正了全零问题，但收益不足 |
| v28 view-tail-safe alpha | 用 policy-view tail-safe alpha shrink 防止尾部视角回退 | Bonsai medium 完成，不推广 | MSE tail 安全不等于三指标 balanced 安全 |
| v29 balanced view-tail objective | 用 `dPSNR + 20*dSSIM - 20*dLPIPS` 做 tail objective | Bonsai selector 仅 trainval 极微弱接受，held-out 极微弱负向 | 单纯改 objective 仍不够 |
| v30 triadic teacher-bake | teacher 优于 parent/current 且差异足够时才蒸馏 | Bonsai GPU smoke 完成，mask active，但 baked checkpoint 低于 clean-best | image-level teacher loss 不足，需要 surface-addressed residual basis |
| v31 teacher-surface basis | 把 ELA/teacher residual 从 image-space 转成 surface-addressed cache，再拟合 face-local SH1 delta | Bonsai medium/full-res 完成；接口有效，但 full-res 指标几乎等于 compact base，未推广 | 方向正确，瓶颈从接口变成可覆盖的 carrier 容量和跨视角泛化 |

v30/v31 的关键数字：

| method on Bonsai | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| selected clean `ours_26000` | 28.8952 | 0.8964 | 0.2595 |
| Phase-F render-time ELA | 30.8750 | 0.9177 | 0.2139 |
| v30 baked checkpoint `ours_26080` | 28.8144 | 0.8938 | 0.2636 |
| compact base before v31 | 28.8643 | 0.8960 | 0.2593 |
| v31 face-local teacher SH1, full-res | 28.8644 | 0.8960 | 0.2594 |

解释：

> v30 证明 teacher-render loss 接入和 triadic mask 生效，但 topology-frozen checkpoint 仍无法吸收 render-time ELA 的局部高频修复。下一步不应继续堆 global teacher loss，而应把 `teacher_render - compact_parent_render` 转成 surface-addressed residual basis。

v31 进一步证明了这条 surface-addressed 路线的接口可行性：

- train cache：Bonsai `48` 个 train views，`2,121,267` 个 unique visible faces，`rgb_render/rgb_gt/residual_rgb/barycentric` 已落盘；
- teacher cache：`teacher_better_mask` 平均 active fraction `17.61%`，mean positive teacher gain L1 `0.00495`；
- fitting gate：`selected_faces=512`，honest policy-val proxy relative gain `0.3821`，最终接受 `12` 个 face-local edits；
- materialized checkpoint：新增 `36` 个 vertices，三角形数不变，full-res render/eval 已完成；
- 公平 full-res 结果：`28.8644 / 0.8960 / 0.2594`，相对 selected clean 在 PSNR/SSIM 上仍低，且相对 compact base 近似 no-op。

结论：

> v31 不应作为当前主讲结果，但它是下一步 representation-level 升级的关键底座：teacher signal 已能被 surface indexing、barycentric evidence 和 face-local SH fitting 消化；下一轮必须扩大 carrier 覆盖，做 patch/region-level basis，而不是只接受十几个局部 faces。

## 8. 为什么这是研究工作，而不是简单工程调参

当前方法的研究性主要体现在三个约束：

1. **Surface evidence certification**  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. **No-test-GT guarded policy**  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。大量 v26-v30 探针被拒绝，说明 pipeline 不是为了追 test 数字而调参。

3. **Rate-distortion and recovery loop**  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 至少有一个真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；surface-addressed residual basis 是下一步论文级升级方向。

## 9. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis |
| 定性 full-frame 差异不总是显著 | 全图差异常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| representation-level 分支收益稀疏 | Phase-R/S 有真实 checkpoint edit，但提升很小 | 顶会主线仍需更强 representation story | 做 face-local SH/low-rank teacher residual carrier |
| v31 carrier 覆盖太小 | Bonsai 只接受 `12` faces / `+36` vertices | 工程可用但指标接近 no-op | 从 face-only edit 升级为 region/patch carrier 或 shared low-rank residual field |
| 室内压缩率较低 | room/counter/kitchen micro-budget 约 2.10% | rate-distortion 数字不如室外强 | 按 geometry safety 解释，不强推破坏性压缩 |
| v30 teacher-bake 负结果 | mask active 但 Bonsai baked checkpoint 低于 clean | image-level distillation 容量不足 | surface-addressed teacher residual basis |

## 10. 建议 PPT 结构

### Slide 1: Title

标题建议：

```text
SPCarNet: Train-Evidence-Certified Compression and Residual Repair for MeshSplatting
```

一句话：

```text
Turn a trained MeshSplatting checkpoint into a safer compact mesh with train-evidence-guided appearance repair.
```

### Slide 2: Motivation

主信息：

- MeshSplatting 已经很强；
- 但局部 residual、拓扑冗余和迭代过拟合仍存在；
- 我们不替换它，而是在它上面构建 self-diagnosis / self-repair loop。

### Slide 3: Method Overview

放 pipeline：

```text
Clean checkpoint
 -> train evidence
 -> safe compaction
 -> checkpoint rewrite
 -> ELA repair
 -> guarded policy
 -> held-out eval
```

### Slide 4: What Is Different From MeshSplatting

使用第 2 节表格。重点讲：

- clean MeshSplatting 直接渲染；
- SPCarNet 判断哪里可删、哪里可修、哪里必须回退。

### Slide 5: Evidence Mining and Safe Compaction

主讲：

- train-view visibility；
- residual region；
- occlusion boundary；
- sparse geometry audit；
- triangle reduction 是删除比例。

### Slide 6: Evidence Lumigraph Adapter

放公式：

```text
Render_final = Render_base + alpha * ResidualEvidence
```

强调：

- residual 来自 train views；
- alpha/branch 由 train-only guard 选；
- test GT 不参与决策。

### Slide 7: Main Quantitative Results

放 summary：

```text
9/9 strict scene wins
+1.3311 PSNR
+0.0347 SSIM
-0.0634 LPIPS
7.6479% mean triangle reduction
244/246 strict per-view wins
```

### Slide 8: Per-Scene Table

放 full9 表格，突出：

- outdoor 和 indoor 都有正收益；
- `treehill` 是 edge fallback，不是强行 adaptive alpha；
- indoor micro-budget 是 safety choice。

### Slide 9: Qualitative Result

主图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

讲法：

- full-frame 差异细微；
- crop/error reduction 显示局部更接近 GT；
- 该图由 closure audit 自动选择。

### Slide 10: Fairness and Baseline

说明：

- clean baseline 是本地同协议 selected clean；
- clean `26000/30000` 用 held-out test score 选更强者；
- method policy 不用 test GT；
- paper table 只是 sanity check。

### Slide 11: Ablation and Failed Probes

展示 v26-v31：

- 多个探针真实接入并验证；
- gate 拒绝弱结果；
- v30 说明 image-level teacher-bake 不够；
- v31 说明 surface-addressed interface 已通，但 carrier 容量不足；
- 下一步是 region/patch-level teacher residual basis。

### Slide 12: Takeaway and Next Step

结论：

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation.
```

## 11. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v30 很多调参式探针被拒绝，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

主 claim 使用本地同协议 selected clean MeshSplatting。clean `26000/30000` 只用 held-out score 选更强 clean baseline。我们的 alpha、branch 和压缩策略不使用 test GT。

### Q3: 是否全面超过 MeshSplatting paper table？

按当前汇总均值，Phase-J mean `26.4828 / 0.7837 / 0.2243` 高于 paper table `24.78 / 0.728 / 0.310`。但 PPT 中更严谨的 claim 应是相对本地 selected clean baseline，因为 paper table 不是同代码、同环境、同选择规则的直接 baseline。

### Q4: 如果主要收益来自 ELA，会不会被认为不是 representation 方法？

这是当前最大边界。我们可以诚实表述为：当前 accepted endpoint 是 compact mesh + train-evidence render-time repair。它已经包含真实 compact checkpoint，但最强 appearance gain 尚未完全 baked into representation。下一步是 surface-addressed teacher residual basis。

### Q5: 为什么不提高三角形减少比例？

因为目标不是单一压缩率。室内和 garden 等场景 geometry 已经很敏感，强推删面会破坏 sparse geometry 或 RGB。当前方法用 safety gate 保证 `9 / 9` geometry-safe。

### Q6: 定性图为什么不总是一眼明显？

因为 full-frame 改善通常是 residual-level，PSNR/SSIM/LPIPS 的全局累计明显，但肉眼看全图可能不强。PPT 应使用 Phase-J local held-out error-reduction 图展示局部真实改进，同时保留 full-frame gallery 说明公平对比。

### Q7: v31 是不是已经解决 representation-level baking？

还没有。v31 的价值是把 teacher residual cache、barycentric surface evidence 和 face-local SH1 delta fitting 接口打通，并完成 Bonsai full-res 公平验证。结果显示 policy-val proxy 正向，但最终只接受 `12` 个 faces，因此 full-res test 指标接近 compact base。它是下一步更高容量 surface residual basis 的工程底座，不是当前可主讲 endpoint。

## 12. 推荐主讲边界

可以主讲：

- Phase-J 相对 selected clean MeshSplatting full9 `9 / 9` 三指标严格胜出；
- mean `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS；
- `244 / 246` per-view strict wins；
- mean triangle reduction `7.6479%`；
- `9 / 9` geometry-safe；
- policy 不使用 held-out test GT；
- paper table 作为 sanity check 高于原论文均值。

不要主讲成已经完成：

- 不要说 representation-level baking 已经全面解决；
- 不要把 v26-v30 的单场景/负结果作为主结果；
- 不要说所有收益都已经写入 checkpoint；
- 不要把 paper table 对比当成唯一公平 claim；
- 不要夸大 full-frame 定性肉眼差异。

## 13. 下一步路线

当前最合理的下一步不是继续 global teacher loss，也不是继续单 face 小容量拟合，而是：

```text
teacher_render - compact_parent_render
  -> surface evidence cache with face/barycentric support
  -> region/patch carrier discovery
  -> face-local SH / low-rank residual basis with shared support
  -> train-only robustness gate
  -> full9 held-out audit
```

目标：

- 保留 Phase-J 的 train-only fairness；
- 把 ELA 的局部 residual 能力变成 persistent surface-addressed representation；
- 避免 v30 image-level topology-frozen teacher-bake 的容量不足；
- 避免 v31 只编辑极少 faces 导致 full-res 指标近似 no-op；
- 形成更像“表示升级”的论文主贡献。

## 14. 汇报用 90 秒版本

> 我们的目标不是重做一个新表示，而是在 MeshSplatting 这个强 mesh baseline 上做训练证据驱动的压缩和修复。系统先在训练视角上采集 surface visibility、residual 和 occlusion evidence，再判断哪些三角形可以安全删除，并真实改写 checkpoint。然后我们用 Evidence Lumigraph Adapter 把训练 residual 转移到 held-out view，所有 alpha、edge fallback 和 branch 都只用 train/policy-val evidence 决定，test GT 只用于最终评价。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 做到 9/9 场景 PSNR、SSIM、LPIPS 严格提升，平均提升 +1.3311 PSNR、+0.0347 SSIM、-0.0634 LPIPS，同时平均减少 7.65% 三角形，并且 244/246 个 held-out views 三指标严格胜出。当前边界是最强外观收益仍来自 render-time ELA；我们已经验证简单 teacher-bake 不够，并在 v31 把 teacher residual 的 surface-addressed 接口打通，但小容量 face-only edit 还不能转化为 full-res 指标。下一步要把 teacher residual 扩展成 region/patch-level surface basis，让方法从强 repair loop 进一步升级成更完整的 representation-level 方法。
