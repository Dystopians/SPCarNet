# SPCarNet 当前方法导师汇报技术报告

日期：2026-06-23

用途：mentor 汇报、PPT 制作、当前技术路线交底。

当前仓库状态：基于 commit `a5131a8` 的 dirty worktree。当前可以安全主讲的 endpoint 是 `ours_26000_phasej_guarded_adaptedge_ela`。v26、v27、v28、v29、v30 是后续可信策略探针：v27 已 medium 验证但未通过诚实 gate；v28 Bonsai medium 已结束但不推广；v29 已接入代码并完成 Bonsai candidate-owned refit gate 和 final selector；selector 只得到极微弱 trainval 正向，held-out report-only 三指标极微弱负向；v30 triadic teacher-bake 已证明训练链路和 mask 生效，但 baked checkpoint 仍低于 clean-best。它们当前都不能作为 headline。

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 上的训练证据驱动修复闭环。它不是重新训练一个完全不同的 3D 表示，而是在 MeshSplatting 的 opaque triangle mesh checkpoint 上加入：

- train-view surface/render evidence mining；
- sparse-occlusion protected compaction；
- checkpoint-safe topology rewrite；
- train-only Evidence Lumigraph Adapter；
- guarded alpha / edge / branch selector；
- report-only held-out evaluation。

当前最适合汇报的结论：

| 维度 | 当前状态 |
|---|---|
| 主方法 | Phase-J compact MeshSplatting + guarded adaptive Evidence Lumigraph Adapter |
| 公平 baseline | 本地同协议 selected clean MeshSplatting，clean `26000/30000` 只用 held-out test score 选强者 |
| Mip-NeRF360 full9 RGB | `9 / 9` 场景相对 selected clean baseline 三指标严格胜出 |
| 平均收益 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| 几何和压缩 | 平均 triangle reduction `7.6479%`，`9 / 9` geometry-safe，`6 / 9` sparse geometry 严格更好 |
| per-view 稳定性 | `244 / 246` held-out views 同时提升 PSNR、SSIM、LPIPS |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310` |
| 最重要边界 | 当前最强 RGB 收益仍主要来自 render-time ELA，representation-level baking 仍是下一步 |

建议 mentor 汇报中的一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% 三角形；但最强外观收益仍来自 render-time ELA，下一步需要把修复进一步内化到 representation-level checkpoint。

## 1. 问题背景

MeshSplatting 的优势是清楚的：它输出 triangle mesh，比 Gaussian 或点云更接近传统图形资产管线，也更容易接入游戏、AR/VR、数字孪生和渲染系统。

但本地复现和审计暴露出三个可以继续推进的问题：

| 问题 | 现象 | 对论文目标的影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内高频纹理、细边缘区域存在模糊或颜色偏差 | 全图指标和局部视觉质量仍有提升空间 |
| 拓扑冗余 | 一部分三角形对多视角解释贡献低 | 有 rate-distortion 改善空间 |
| 训练时长不等于更好 | clean `30000` 在当前 full9 envelope 中全部弱于 clean `26000` | 不能把提升解释为简单训练更久 |

SPCarNet 的核心假设是：

> MeshSplatting 已经提供了强基础表示，但训练视角中仍包含可以反推出 surface/re-render 可靠性的证据。只要证据足够可靠，就可以安全地压缩局部几何，并把 train residual 转移到 held-out view 来修复外观。

## 2. 方法总览

当前主方法可以拆成五个模块。

```text
Clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded train-only policy
  -> held-out render-only evaluation
```

### 2.1 与原始 MeshSplatting 的区别

| 维度 | clean MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础表示 | opaque triangle mesh | compact opaque triangle mesh + train-evidence residual adapter |
| 压缩 | 没有额外压缩闭环 | 用 train surface/render evidence 判断安全删面 |
| 外观修复 | checkpoint 属性直接渲染 | 用训练 residual 形成 Evidence Lumigraph Adapter |
| 决策依据 | 训练流程默认 checkpoint | train-only calibration、policy-val gate、edge fallback |
| held-out test 使用 | 最终评价 | 只做最终评价，不参与 alpha、branch、prune 选择 |
| 风险控制 | 主要依赖训练本身 | strict gate、tail audit、fallback、no-op |

通俗解释：

> 原始 MeshSplatting 像是直接交付训练好的网格。SPCarNet 则先让网格做一次训练证据体检：哪里可以删，哪里容易错，哪里有可靠 residual 可以修。如果证据不够，就不动或回退。

## 3. 关键技术模块

### 3.1 Train-View Evidence Mining

系统先在训练视角上渲染 baseline 或 compact checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual `GT - render`；
- face visibility；
- per-face pixel count；
- support view consistency；
- high-error region statistics；
- depth / surface hit evidence。

这个阶段不使用 held-out test GT 做任何选择。

### 3.2 Sparse-Occlusion Protected Compaction

目标是在不破坏渲染和 sparse geometry 的前提下降低三角形数。

三角形是否可以压缩，不是靠固定比例硬删，也不是只按面积排序，而是看它是否满足训练证据中的安全条件：

- 多视角 visibility / hit count 足够稳定；
- 它不是关键 occlusion boundary 或高 residual 解释区域；
- policy-validation 渲染没有明显退化；
- sparse geometry audit 没有 AbsRel、DepthMAE、Normal 的明显风险；
- 室内场景启用 micro-budget，避免为追求压缩数字破坏高质量 geometry。

报告中的 triangle reduction 是删除的三角形占比，不是剩余占比。

### 3.3 Checkpoint-Safe Topology Rewrite

压缩不是只在日志里记录。系统会真正改写 MeshSplatting checkpoint：

- 删除 faces；
- remap vertices 和 face indices；
- 处理 trailing unused vertices；
- 保证 checkpoint tensor shape 与 renderer 一致；
- 保证后续 render、metric、geometry audit 能继续运行。

这一步是当前方法仍然保持 mesh-friendly 的关键。外观修复可以是 render-time，但 compact checkpoint 本身是真实 materialized 的。

### 3.4 Evidence Lumigraph Adapter

ELA 是当前 Phase-J 视觉收益的主要来源。

对每个 target view，ELA 从多个训练 support views 中取 residual：

```text
residual_s(x) = GT_s(x) - Render_s(x)
```

然后通过相机、depth、view consistency 和 support confidence，把训练 residual warp 到 target view。多个 support residual 聚合时，会统计：

- valid support count；
- residual mean；
- residual std；
- support agreement；
- confidence；
- edge / high-frequency evidence。

最终修复形式可以写成：

```text
Render_final = Render_base + alpha(pixel) * ResidualEvidence(pixel)
```

这里 `alpha(pixel)` 不是手动调出来的固定常数，而是由 train calibration 和 policy evidence 自动选择。

### 3.5 Guarded Adaptive Policy

Phase-J 的关键升级是 guarded portfolio：

- 大多数场景使用 adaptive-alpha ELA；
- `treehill` 使用 train-selected structural edge fallback；
- 每个分支必须通过 train-only guard；
- gate 不通过时回退到 Phase-J 或 no-op。

主要 gate 包括：

- PSNR non-regression；
- SSIM regression 上限；
- LPIPS regression 上限；
- balanced score 不为负；
- tail / region-risk 不明显恶化；
- fallback 不使用 test GT 选择。

这也是为什么 v26/v27/v28 中有些局部实验即使 test 上看起来接近，也不能升级为主结果：没有过 honest train-val / tail gate。

## 4. 当前最新探针的状态

这一节不建议放在主结果页，可以作为 backup 和问答。

| 版本 | 机制 | 当前结论 |
|---|---|---|
| v26 hard local-trust | hard binary residual trust gate | Bonsai medium 完成，但过保守，容易 alpha=0 或接近 no-op |
| v27 soft local-trust | continuous trust-weighted residual | 接口、smoke、dry-run、Bonsai medium 完成；candidate-owned 被 honest gate 拒绝，selector 接受行 held-out test 三指标微弱回退 |
| v28 view-tail-safe alpha | policy-view MSE tail safe alpha shrink | Bonsai fix2 medium 完成；plan-stage 被拒，selector 仅 trainval 极微小正向，held-out test report-only 极微小负向，不推广 |
| v29 balanced view-tail objective | 用 train/policy-val balanced score 和 LPIPS 感知 view-tail scale selection | 代码已接入、py_compile、smoke、Bonsai medium selector 已完成；candidate-owned refit 被 trainval/tail gate 拒绝；final selector 仅 trainval 极微弱接受，held-out report-only 极微弱负向，不推广 |
| v30 triadic teacher-bake | 只在 ELA teacher 优于 parent/current 且 teacher-parent 差异足够的像素蒸馏到 topology-frozen checkpoint | 代码、CPU smoke、Bonsai GPU smoke 完成；mask active，但 held-out test 低于 clean-best，不推广 |

v28 的 Bonsai plan-stage 负结果很重要，因为它解释了当前瓶颈：仅用 pixel/bin 或 MSE tail 做安全选择，不一定能保证 view-level PSNR/SSIM/LPIPS balanced 目标同时变好。v29 由此把 view-tail scale replay 的目标从 MSE 扩展到：

```text
dPSNR + 20 * dSSIM - 20 * dLPIPS
```

v29 的 Bonsai candidate-owned refit 结果说明，单纯把 scale-selection 目标改成 balanced 还不够：它在 report-only test 上有小幅正向，但 trainval gate 看到 `dPSNR=-0.0196`、`dSSIM=-0.000314`、`dLPIPS=+0.000646`，并触发 `balanced_delta_below_0` 与 `render_region_tail_cvar_below_-2e-05`。最终 selector 确实找到了一个 trainval-only 接受的 `strictfull_s1` 候选，但幅度极小：trainval `dPSNR=+0.000330`、`dSSIM=+0.0000067`、`dLPIPS=-0.0000085`，held-out report-only 则是 `dPSNR=-0.000017`、`dSSIM=-0.0000040`、`dLPIPS=+0.0000104`。这对汇报很有价值：它证明我们的 pipeline 没有利用 test-only 假阳性，也不会把微弱单场景 selector 结果包装成 headline。

v30 的 Bonsai teacher-bake smoke 进一步证明，问题不是“teacher loss 没接上”。新 triadic mask 在真实训练中 active，W&B 记录 `teacher_render/mask_fraction=0.01725`，topology 也保持不变；但 `ours_26080` 的 held-out test 为 `28.8144 / 0.8938 / 0.2636`，低于 selected clean `28.8952 / 0.8964 / 0.2595`。这说明 image-level topology-frozen distillation 仍不足，需要转向 surface-addressed teacher residual basis。

## 5. 实验协议

### 5.1 数据集和场景

主结果使用 Mip-NeRF360 full9：

```text
bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai
```

### 5.2 Baseline 选择

对每个场景，本地 clean MeshSplatting baseline 从 clean `26000` 和 clean `30000` checkpoint 中选择。选择只依据 held-out test 指标的统一 score：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

注意：

- 训练集指标不用于选择 baseline；
- 我们方法的 alpha、branch、edge fallback、压缩比例不使用 held-out test GT；
- held-out test 只做最终 report-only 评价。

当前 clean baseline envelope 的结果是：9 个场景全部选择 clean `26000`。这反过来证明“训练更久”不是解释我们结果的原因。

### 5.3 主结果路径

主要 evidence 路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

## 6. 主定量结果

### 6.1 Full9 scene table

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

### 6.2 Per-view and geometry audit

| audit | result |
|---|---|
| Scene-level strict RGB wins | `9 / 9` |
| Per-view strict RGB wins | `244 / 246` held-out views |
| Sparse geometry-safe scenes | `9 / 9` |
| Sparse geometry strict wins | `6 / 9` |
| Mean triangle reduction | `7.6479%` |

### 6.3 Geometry table

| scene | dAbsRel | dDepthMAE | dNormal | geom status |
|---|---:|---:|---:|---|
| bicycle | -0.000241 | -0.020421 | -0.011882 | safe, strict better |
| flowers | -0.003356 | -0.124978 | -0.043906 | safe, strict better |
| garden | -0.000007 | -0.000154 | -0.001032 | safe |
| stump | -0.005878 | -0.350702 | -0.026039 | safe, strict better |
| treehill | -0.001246 | -0.074749 | -0.012229 | safe, strict better |
| room | +0.000000 | +0.000000 | +0.000000 | safe |
| counter | +0.000000 | +0.000000 | +0.000000 | safe |
| kitchen | +0.000000 | +0.000000 | +0.000000 | safe |
| bonsai | -0.000368 | -0.004485 | -0.025401 | safe, strict better |

解读：室内场景 geometry 指标为零变化，并不是失败，而是 micro-budget guard 选择了安全不破坏；主收益来自 RGB ELA，压缩率由安全证据决定。

### 6.4 与 MeshSplatting paper table 的关系

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table, Mip-NeRF360 mean | 24.7800 | 0.7280 | 0.3100 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J minus paper table | +1.7028 | +0.0557 | -0.0857 |
| Phase-J minus local clean | +1.3311 | +0.0347 | -0.0634 |

汇报时建议强调：最严谨 claim 是相对本地同协议 selected clean baseline，而 paper table 是外部 sanity check。

## 7. 定性结果

PPT 中建议使用两类图。

### 7.1 全图公平对比

用途：证明同 view、同 baseline、同 held-out 口径。

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

Markdown 预览：

![Full9 qualitative gallery](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

### 7.2 局部 error-reduction 对比

用途：展示人眼可见优势。首选：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_phasej_where_it_helps_selection_20260622.json
```

Markdown 预览：

![Phase-J where it helps](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

这张图由 `scripts/car_model/generate_spcarnet_advantage_showcase.py` 根据 current endpoint 的 closure audit 生成。选择逻辑是：

1. 先筛选 full-view 上 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0` 的 held-out view；
2. 再在这些 view 中寻找 SPCarNet 相对 GT 局部 RGB error 下降最大的 texture crop；
3. 绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

局部 crop 统计：

| crop | full-view dPSNR/dSSIM/dLPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | +6.63 / +0.0452 / -0.0878 | +11.79 | 78.6% |
| kitchen / `00011.png` | +3.43 / +0.0250 / -0.0578 | +10.48 | 71.4% |
| room / `00011.png` | +3.50 / +0.0220 / -0.0656 | +10.36 | 67.7% |
| counter / `00013.png` | +2.17 / +0.0407 / -0.0665 | +6.02 | 54.9% |
| garden / `00006.png` | +1.74 / +0.0479 / -0.0678 | +4.26 | 44.4% |
| flowers / `00014.png` | +1.12 / +0.0754 / -0.1028 | +2.15 | 25.3% |

解释口径：全图缩小后肉眼不一定立刻看到所有差异，因为很多收益是 distributed residual-level improvement。汇报里应该放全图作公平性，放 crop/error map 作直观效果。

## 8. 消融和负结果

| 实验 | 结论 | 汇报价值 |
|---|---|---|
| clean `26000/30000` envelope | 9 个场景都选择 clean `26000` | 排除“只是 baseline 训练不够”的解释 |
| compact-only | geometry-safe，但不足以产生 headline RGB 提升 | 说明 ELA 外观修复是必要模块 |
| adaptive alpha without strict guard | 室内或 tail view 可能产生 SSIM/LPIPS 回退 | 说明 guarded policy 必要 |
| Phase-J guarded adaptedge | full9 当前最强 endpoint | 主结果 |
| Phase-R representation ladder | 接受 `3 / 9`，收益小但更 representation-level | 可作为下一步方向证据 |
| Phase-S region/core-context | full9 effective delta 小，false positive 需要 strictcompact gate 修正 | 说明我们没有包装弱收益 |
| v26/v27 local-trust | 接口闭环，但 Bonsai medium 没有超过 Phase-J | 负结果证明 gate 严格 |
| v28 view-tail-safe | Bonsai medium 已完成但不推广 | 暴露 MSE tail 与 balanced 目标不一致，也说明 near-noop selector 不应作为主结果 |
| v29 balanced view-tail | Bonsai candidate-owned refit 被 trainval/tail gate 拒绝；final selector 仅极微弱 trainval 接受且 held-out report-only 负向 | 证明 balanced objective 已正确接入但不足，下一步需要更强 representation-level 或 teacher-baking 机制 |
| v30 triadic teacher-bake | Bonsai baked checkpoint 低于 selected clean，topology unchanged | 证明 safer image-level distillation 仍不够，下一步应做 surface-addressed teacher residual basis |

这部分可以在 PPT 里作为“我们不是调参游戏”的证据：每个候选都必须过固定 train-only gate，失败就记录并回退。

## 9. 当前局限

必须诚实承认四点：

1. 最强 RGB endpoint 仍包含 render-time ELA，不能声称所有外观修复已经完全 baked into representation。
2. representation-level Phase-R/Phase-S 当前收益仍然小，离论文级终局方法还有距离。
3. 室内场景的 triangle reduction 较低，因为安全策略优先避免 geometry 退化。
4. 定性全图差异不如局部 crop 明显，PPT 展示必须使用 crop/error map 才能看清优势。

## 10. 下一步计划

最合理的推进路线：

| 优先级 | 任务 | 目标 |
|---:|---|---|
| P0 | 基于 v29 已完成的 Bonsai selector 做 view-tail audit 和失败归因 | 解释为什么 balanced view-tail 只能得到微弱 trainval 正向，不能变成稳定 test 改善 |
| P0 | 基于 v28/v29/v30 负结果设计 surface-addressed teacher residual basis | 不再只依赖 image-space alpha replay 或 topology-frozen global teacher loss |
| P1 | 把 ELA teacher 投影到 face/barycentric 支持并写入 face-local SH / low-rank residual basis | 减少 render-time adapter 作为主结果的争议 |
| P1 | 强化局部可视化和 error-map protocol | 让导师和审稿人能直接看到优势 |
| P2 | 扩展外部数据集或 cross-protocol sanity check | 提升泛化可信度 |

## 11. 建议 PPT 结构

| slide | 标题 | 关键内容 |
|---:|---|---|
| 1 | Motivation | MeshSplatting 强，但局部 residual 和拓扑冗余仍可优化 |
| 2 | Fair Baseline | clean `26000/30000` selected envelope，不使用 train metric 选 baseline |
| 3 | Key Idea | train evidence 决定哪里能压缩、哪里能修复、何时回退 |
| 4 | Pipeline | evidence mining -> compaction -> ELA -> guarded selector -> held-out eval |
| 5 | Method Detail | ELA residual warp、alpha calibration、edge fallback |
| 6 | Main Result | full9 `9/9` strict wins，mean gain，triangle reduction |
| 7 | Per-Scene Result | 9 场景表格，重点讲 bonsai/kitchen/counter/room |
| 8 | Geometry | `9/9` geometry-safe，`6/9` strict sparse geometry better |
| 9 | Qualitative | 全图公平对比 +局部 crop/error map |
| 10 | Ablation | clean 30000 不一定更好，compact-only 不够，Phase-J guard 必要 |
| 11 | Honest Risks | render-time ELA、representation-level 内化不足、视觉差异需要局部展示 |
| 12 | Next Step | v28/v29 负结果带来的 representation-level distillation / teacher baking |

## 12. 可能被问到的问题

**Q1：这是不是只是后处理，不是 3D 方法？**

A：不是单纯图像后处理。方法包含真实 compact checkpoint 和 topology rewrite，几何/压缩结果来自 materialized mesh。当前最强 RGB 修复确实使用 render-time ELA，这是边界；下一步是把 ELA teacher bake 回 representation-level checkpoint。

**Q2：有没有用 test set 选参数？**

A：主方法的 alpha、edge fallback、branch、压缩比例都由 train/policy-validation evidence 决定。held-out test 只做 report-only。baseline envelope 的 clean `26000/30000` selection 使用 held-out test 指标来选择更强 clean comparator，这是为了让 baseline 更强，不是给方法调参。

**Q3：为什么三角形只减少 7.65%，室内更低？**

A：因为我们优先保证 geometry-safe。室内场景 geometry 本来稳定，强行删面会破坏指标。当前目标是 Pareto 改善，不是牺牲质量换压缩率。

**Q4：为什么全图看起来差异不大？**

A：很多收益是局部 residual-level improvement，全图缩小时容易被稀释。应展示 full-frame fairness 加局部 crop/error map。closure audit 中 `244/246` held-out views strict win 说明不是只挑几张图。

**Q5：现在能说已经是顶会终局方法吗？**

A：不能这样讲。当前 Phase-J 是强阶段性 endpoint，full9 指标非常漂亮，但最强收益仍在 render-time ELA。要成为更强论文闭环，下一步需要 representation-level baking 或更强泛化验证。

## 13. 证据索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

定性图：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_phasej_where_it_helps_selection_20260622.json
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
```

近期探针日志：

```text
docs/car_model/6-22-SoftLocalTrust-v27-Implementation-And-Bonsai-Medium-Log.md
docs/car_model/6-22-ViewTailSafe-v28-Implementation-And-DryRun-Log.md
docs/car_model/6-23-BalancedViewTail-v29-Implementation-And-Bonsai-Medium-Log.md
docs/car_model/6-23-PhaseG-v30-TriadicTeacherMask-Bonsai-Smoke-Log.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v28_viewtail_fix2_20260622_bonsai_medium_gpu5/viewtail_alpha_audit.md
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/candidate_owned_refit/decisions/bonsai_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/selector/bonsai/coupled_selector_decision.json
/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v29_balanced_viewtail_retry_20260623_bonsai_medium_gpu4/selector/trials/strictfull_s1/decisions/bonsai_decision.json
/data/peilincai/spcarnet_runs/phaseg_v30_triadic_teacher_mask_smoke_20260623_bonsai_gpu5/phaseg_teacher_bake_summary.md
```

实现入口：

```text
utils/evidence_lumigraph_adapter.py
scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py
scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py
scripts/car_model/ecsr_run_facelocal_coupled_selector.py
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
scripts/car_model/ecsr_audit_viewtail_alpha_run.py
```

## 14. 汇报时的最终措辞

推荐最终结论页使用：

> SPCarNet currently establishes a strong train-evidence-certified repair loop on top of MeshSplatting. On Mip-NeRF360 full9, it strictly improves PSNR, SSIM, and LPIPS over a strong local selected clean MeshSplatting baseline in all 9 scenes, with 244/246 per-view strict wins and 7.65% mean triangle reduction. The honest next step is to move the strongest render-time repair into a representation-level checkpoint while preserving the same no-test-GT policy discipline.

中文版本：

> 当前 SPCarNet 已经证明，MeshSplatting 的 triangle mesh 表示可以通过训练证据驱动的安全压缩和残差修复闭环进一步提升。在 Mip-NeRF360 full9 上，我们相对强本地 clean baseline 做到 9/9 场景 PSNR、SSIM、LPIPS 严格胜出，244/246 个 held-out view 三指标同时提升，并平均减少 7.65% 三角形。下一步的核心科学问题，是把当前最强的 render-time 修复能力进一步内化进 representation-level checkpoint。

## 15. PPT 制作备忘

如果明天只汇报 10 到 15 分钟，建议把材料压缩成三层：

### 15.1 必讲主线

1. **Motivation：** MeshSplatting 是强 baseline，但仍有局部 residual 和拓扑冗余。
2. **Core idea：** 用 train evidence 判断哪里能删、哪里能修、哪里必须回退。
3. **Method：** compact checkpoint + ELA residual transfer + guarded selector。
4. **Main result：** full9 相对 selected clean MeshSplatting `9 / 9` 三指标严格胜出，mean `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS，平均减少 `7.6479%` triangles。
5. **Honest boundary：** 最强 RGB endpoint 仍有 render-time ELA；representation-level baking 是下一阶段核心。

### 15.2 最推荐展示的图

| PPT 页 | 图片 | 目的 |
|---|---|---|
| qualitative fairness | `assets/spcarnet_m360_full9_qualitative_gallery.png` | 展示同 view / 同 baseline 的完整 held-out 对比 |
| where it helps | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | 展示局部 error-reduction，最适合让 mentor 直观看到优势 |
| outdoor backup | `assets/spcarnet_m360_outdoor_detail_showcase.png` | 回答“室外是否有效” |
| mixed backup | `assets/spcarnet_m360_where_it_helps_showcase.png` | 回答“是不是只在一两个场景有效” |

不要只放全图。全图是公平性证据，但很多 residual-level 改善在缩略图上会被稀释。主视觉效果页应使用 crop/error map。

### 15.3 可以安全讲的 claim

- 我们的方法相对本地同协议 selected clean MeshSplatting baseline 在 Mip-NeRF360 full9 上 `9 / 9` 场景三指标严格胜出。
- baseline 是 clean `26000/30000` envelope，且只用 held-out test score 选择更强 clean comparator。
- 方法自己的 alpha、branch、edge fallback、压缩策略不使用 held-out test GT。
- 当前 endpoint 是一个 mesh-friendly 的 train-evidence repair loop：compact checkpoint 是真实 materialized 的，ELA 是当前最强外观修复模块。
- 相对 MeshSplatting paper table 的 comparison 可以作为 sanity check，但主 claim 应以本地同协议复现 baseline 为准。

### 15.4 不建议过度声称的内容

- 不要说已经完成“完全 representation-level 内化”。当前最强外观收益仍来自 render-time ELA。
- 不要说 v28/v29/v30 已经全面优于 Phase-J。v28 已证明不应推广；v29 的 candidate-owned refit 已被诚实 gate 拒绝，final selector 也只有极微弱 trainval 正向且 held-out report-only 负向；v30 的 baked checkpoint 仍低于 clean-best，后续最多作为诊断证据。
- 不要把 triangle reduction 讲成非常激进的 compression。当前是 quality-first Pareto improvement，平均约 `7.65%` 删除。
- 不要把 clean `30000` 当作更强 baseline。当前 full9 envelope 中 clean `26000` 按 held-out score 全部更强。

### 15.5 一句话答辩版本

如果 mentor 问“这个工作的科学价值在哪里”，建议这样回答：

> 我们不是简单后处理 MeshSplatting，而是提出了一个 train-evidence-certified 的 mesh repair loop：先基于训练视角证据做安全压缩，再把可验证的 residual 信息通过 surface/view consistency 转移到 held-out 渲染，并用 no-test-GT guarded policy 决定是否接受。当前结果证明 MeshSplatting 的 triangle mesh checkpoint 不是终点，它还能被 evidence-driven 的压缩和修复机制系统性提升；下一步要把 ELA teacher 更彻底地蒸馏进 representation-level checkpoint。
