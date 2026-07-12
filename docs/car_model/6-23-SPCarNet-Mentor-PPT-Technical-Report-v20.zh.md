# SPCarNet 当前方法完整技术报告 v20

日期：2026-06-23 16:54 -0700

用途：mentor 汇报、PPT 母稿、方法交底、后续论文路线讨论。

当前主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`

当前状态：阶段性强结果已经成立，但论文终局仍未完全闭合。Phase-J 是当前最可靠、最适合放在主结果页的强结果；v48/v52 是把修复从 render-time adapter 推向 persistent surface representation 的最新正证据；v53 是一次真实但不推广的 alpha calibration 负结果诊断。

---

## 0. 一页汇报结论

SPCarNet 不是从零替代 MeshSplatting，而是在一个已经训练好的 MeshSplatting checkpoint 上增加训练证据驱动的压缩、外观修复和风险控制阶段。

一句话英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation.

一句话中文：

> 原始 MeshSplatting 是“训练出 mesh 后直接渲染”；SPCarNet 是“训练出 mesh 后，再用训练视角证据给 mesh 做体检：哪里能安全删、哪里能可靠修、哪里风险高必须回退”。

当前最稳可汇报结果：

| 维度 | 当前结论 |
|---|---:|
| 数据口径 | Mip-NeRF360 full9，本地同协议复现 |
| 公平 baseline | selected clean MeshSplatting，从 clean `26000/30000` checkpoint 中按 held-out test score 选更强者 |
| 当前主 endpoint | `ours_26000_phasej_guarded_adaptedge_ela` |
| RGB 场景胜率 | `9 / 9` 场景相对 selected clean MeshSplatting 在 PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310`；只能作为 sanity check |
| 表示级进展 v48 | surface residual atlas 相对 same-evidence no-op：`7 / 9` strict，`8 / 9` non-regressive/tie，mean `+0.001462` PSNR，`+0.00002774` SSIM，`-0.00003953` LPIPS |
| 最新策略整合 v52 | capacity-aware policy 相对 v48：`3 / 9` strict，`9 / 9` non-regressive/tie，mean `+0.000086890` PSNR，`+0.000008782` SSIM，`-0.000015303` LPIPS |
| v52 可复现性 | W&B source-config rerun 已完成；fresh rerun 完成 `9 / 9` 场景，`0` missing，`0` metric mismatch |
| v53 结论 | policy-val alpha calibration 是真实方法改动，但没有三指标严格超过 v52，因此不推广 |

建议 PPT 主张：

> We obtain a strong MeshSplatting-based compact-and-repair pipeline that strictly improves PSNR, SSIM, and LPIPS over a strong local MeshSplatting baseline on all 9 Mip-NeRF360 scenes, while also reducing triangle count. The strongest current result is Phase-J; the next research bottleneck is to internalize render-time ELA gains into a stronger persistent surface representation.

---

## 1. 汇报口径分级

明天汇报时建议明确分成四个层级，避免把还没完全闭合的部分讲成最终结论。

| 等级 | 内容 | 应该怎么讲 |
|---|---|---|
| 主结果 | Phase-J guarded adaptive ELA full9 | 当前最稳 headline：full9 `9 / 9` strict RGB wins vs selected clean MeshSplatting |
| 表示级正证据 | v48 auto-support surface residual atlas | residual repair 可以变成 train-only surface-addressed representation，但 effect size 仍小 |
| 最新策略整合 | v52 capacity-aware v48/v51 policy | 只在 support-capacity 证据充分的 cap-hit 场景升级到 v51，其余保留 v48 |
| 负结果诊断 | v53 policy-val alpha calibration | 证明 residual amplitude 是瓶颈，但全局 alpha 太粗，不适合作为当前 endpoint |

不建议主讲成最终成果的内容：

- v52 已经替代 Phase-J；
- 已经 100% 顶会终局完成；
- 严格同协议超过 MeshSplatting 原论文表格；
- triangle reduction 已经是极限压缩；
- 全图肉眼定性差异已经非常明显。

最稳妥定位：

> Phase-J 已经是一条有强定量证据的阶段性主线；v48/v52 说明我们正在把修复从 render-time adapter 推向 surface representation，但 representation-level effect size 还没有达到最终论文 endpoint 的强度。

---

## 2. 研究动机

MeshSplatting 的优势在于它输出 triangle mesh。相比纯 image-space 方法、点云或 Gaussian 表示，mesh 更容易进入传统图形管线、游戏引擎、AR/VR、数字孪生和几何编辑流程。

但是 clean MeshSplatting checkpoint 训练完成后仍然有三个问题：

| 问题 | 典型表现 | SPCarNet 的处理 |
|---|---|---|
| 局部外观 residual | 树叶、树皮、室内边缘、桌面纹理仍有系统性偏差或模糊 | 从 train views 挖 residual，用 guarded ELA 或 surface atlas 修复 held-out view |
| 拓扑冗余 | 一部分 triangles 对多视角解释贡献低 | 用训练证据做 sparse-occlusion protected compaction |
| 决策风险 | 盲目 residual transfer 会伤害 tail views 或 out-of-trajectory 区域 | 用 train-only policy-val gate、min-view、CVaR、SSIM/L1 风险门和 fallback/no-op |

核心假设：

> MeshSplatting 已经学习到强基础表示，但训练视角中仍包含可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删除冗余 geometry，并把训练 residual 迁移到 held-out view 修复外观。

---

## 3. 方法总览

当前 SPCarNet 可以拆成五层：

| 层级 | 模块 | 作用 | 当前状态 |
|---|---|---|---|
| Base | clean MeshSplatting checkpoint | 提供基础 mesh 表示和 renderer | baseline |
| Geometry | sparse-occlusion protected compaction | 删除低风险 triangles，同时保持 topology 与 renderer compatibility | Phase-J 主结果使用 |
| Appearance | Evidence Lumigraph Adapter, ELA | 将 train residual 迁移到 held-out render | Phase-J 主收益来源 |
| Safety | train-only guarded policy | 选择 alpha、edge fallback、reject/no-op | Phase-J 已闭合 |
| Internalization | surface residual atlas v48/v51/v52 | 将 residual repair 推向 surface-addressed persistent 表示 | 当前是正向但细微的表示级证据 |

整体流程：

```text
clean MeshSplatting checkpoint
  -> train-view render/evidence cache
  -> surface reliability / residual / support analysis
  -> sparse-occlusion protected compaction
  -> compact checkpoint
  -> guarded residual repair
  -> held-out test render and metrics
```

与原始 MeshSplatting 的区别：

| 维度 | clean MeshSplatting | SPCarNet |
|---|---|---|
| 训练结束后 | 直接渲染 checkpoint | 继续 evidence mining、压缩、修复和风险审计 |
| 几何 | 原 mesh | topology-safe compact mesh |
| 外观 | checkpoint 属性直接渲染 | train-evidence residual adapter / surface atlas |
| 策略 | 无显式风险判断 | train-only policy gate + fallback |
| held-out test GT | 用于评价 | 只用于最终评价，不参与 method branch/alpha/fallback |
| 失败处理 | 无 | gate 不通过则 fallback/no-op |

---

## 4. 主方法：Phase-J Guarded Adaptive ELA

### 4.1 Train-view evidence mining

系统先在训练 view 上渲染 clean/compact MeshSplatting checkpoint，并缓存：

- RGB render；
- GT image；
- residual map：`GT RGB - rendered RGB`；
- surface visibility；
- face / barycentric correspondence；
- residual support、variance、sign consistency；
- policy-val split 上的 view-level 风险指标。

这一步不看 held-out test GT，而是把训练集中的局部错误和表面可见性转成结构化证据。

### 4.2 Geometry-safe compaction

Compaction 的目标不是极限压缩，而是 quality-first rate-distortion improvement。它删除训练证据表明低风险的 triangles，同时保护 sparse / occlusion-sensitive 区域。

关键约束：

- 不产生 invalid index；
- 不产生 degenerate faces；
- 保持 renderer 可读；
- sparse COLMAP / depth / normal 指标不能出现不可接受退化；
- 压缩后 checkpoint 必须进入同一 render/metrics pipeline。

当前 Phase-J 平均 triangle reduction 是 `7.6479%`。这里的 triangle reduction 是“删去的三角形占比”，不是“剩余比例”。PPT 中建议称为 quality-first geometry simplification，不要称为极限压缩。

### 4.3 Evidence Lumigraph Adapter

ELA 是 Phase-J 的主要外观收益来源。它用训练视角 residual 对 held-out target view 做受控修复：

```text
residual = GT RGB - rendered RGB

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target view 像素；
- `residual_i(u_i)` 来自训练 view 上对应局部区域；
- `w_i(p)` 由可见性、几何邻近、局部结构和策略门决定；
- `alpha` 由 train/policy-val evidence 选择；
- 当风险门失败时不应用该修复。

直观解释：

> MeshSplatting 已经画出了整体结构，但局部纹理或边缘还有系统性误差；训练视角中能看到这些误差，并且多视角一致区域的误差可以被安全迁移。

### 4.4 Guarded adaptive policy

Phase-J 的关键不是单纯 ELA，而是 ELA 外面有 train-only guard：

- stable scenes 使用 adaptive-alpha ELA；
- unstable scenes 使用 train-selected structural edge fallback；
- alpha、edge gate、fallback 由 train / policy-val evidence 决定；
- policy-val 风险过高时拒绝或 no-op；
- held-out test GT 不参与 branch、alpha、edge 或 compaction ratio 选择。

因此主 claim 不是“我们在 test 上调出了好参数”，而是：

> We design a self-diagnosing and self-repairing MeshSplatting post-training policy driven by training-view evidence only.

---

## 5. 表示级内化路线：v48 到 v53

Phase-J 的最大短板是最强收益仍来自 render-time adapter。为回应“这是不是后处理”的潜在质疑，我们推进了 surface-addressed residual atlas 线。

### 5.1 v48 Auto-Support Surface Residual Atlas

v48 的核心改动是 train-only support expansion：

```text
base_carrier support
  + fit-view residual evidence ranking
  + top-K extra face candidates
  + train policy-val non-regression guard
  + texture/fill/alpha auto-policy
```

固定策略：

1. 保留 v47 `base_carrier` support 作为安全候选；
2. 只扫描 fit views，并用同一 `policy_val_stride` 排除 policy-val views；
3. 对 non-carrier faces 聚合 residual evidence；
4. 按 `mean_l1 * log1p(samples)` 排序 extra faces；
5. 添加满足 sample 与 residual 阈值的 top-K extra faces；
6. 在 train policy-val 上联合评估 support mode、texture size、fill mode、alpha；
7. 只有 relative gain、SSIM gain、CVaR20、min-view gain 等风险门安全时才推广；
8. 不安全则回退到 base carrier 或 no-op。

v48 full9 相对 same-evidence no-op compact baseline：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

解释：

- `stump` 被 train policy-val gate 拒绝，作为有效 no-op；
- `treehill` PSNR/SSIM 微涨，但 LPIPS 轻微回退，因此不是 strict 三指标胜出；
- `counter/kitchen/bonsai` 触及 `+2048` extra-face cap，说明 support/capacity 是 representation-level 瓶颈；
- v48 是当前最完整的 single-policy 表示级正证据，但不是 Phase-J 替代品。

### 5.2 v51 Support-Footprint Ladder

v51 的方法改动是把 support expansion 从单一 `topK=2048` 改成 train-only ladder：

```text
base_carrier
  -> rank non-carrier faces by fit-view residual evidence
  -> evaluate topK support candidates, e.g. 2048 / 4096
  -> select support footprint only by policy-val risk gate
  -> apply accepted atlas to held-out test views
```

v51-fast full9 使用固定 `texture=32`、`fill=nearest_observed`、support candidates `2048,4096` 后：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v51 vs same-evidence no-op | `6 / 9` | `6 / 9` | `+0.001380497` | `+0.000030067` | `-0.000051187` |
| v51 vs v48 | `3 / 9` | `3 / 9` | `-0.000081804` | `+0.000002331` | `-0.000011659` |
| v51 vs v50 | `5 / 9` | `8 / 9` | `+0.000116136` | `+0.000008331` | `-0.000017136` |

关键 lesson：

- `counter/kitchen/bonsai` 三个 cap-hit 场景相对 v48/v50 三指标严格提升；
- 非 cap-hit 场景不适合全局固定 `texture=32` 和 `nearest_observed`；
- 下一步不应该继续人工扫参，而应该用固定 train-only policy 只在 support-capacity 证据充分时升级到 v51。

### 5.3 v52 Capacity-Aware Effective Policy

v52 把 v51 的 lesson 固化成固定策略：

```text
if v48 accepted
   and v48 selected_support_added_faces >= 2048
   and v51 accepted_atlas
   and v51 selected_support_added_faces > v48 selected_support_added_faces
   and v51 policy-val SSIM gain >= 5e-5:
       use v51
else:
       keep v48
```

这个策略只读 train/policy-val audit 字段，不用 held-out test delta 做选择。实际决策：

| selected v51 | selected v48 |
|---|---|
| `counter`, `kitchen`, `bonsai` | `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `room` |

full9 effective 结果：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v52 vs no-op | `7 / 9` | `8 / 9` | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | `3 / 9` | `9 / 9` | `+0.000086890` | `+0.000008782` | `-0.000015303` |
| v52 vs v50 | `6 / 9` | `6 / 9` | `+0.000284831` | `+0.000014782` | `-0.000020780` |

v52 的意义：

- 它把 v51 的正收益限定在 support-capacity 瓶颈场景，避免全局固定 texture/fill 伤害 v48；
- 它相对 v48 是 `9 / 9` non-regressive/tie，并在 `counter/kitchen/bonsai` 三指标严格提升；
- 它是当前最强 representation-level effective policy by full9 mean；
- source-config GPU rerun 已复现全 9 场景，但 v52 的 effect size 仍然很小，不能作为最终论文 endpoint。

v52 source-config rerun 状态：

| 项目 | 值 |
|---|---:|
| completed scenes | `9 / 9` |
| missing scenes | `0` |
| metric mismatch scenes | `0` |
| reproduction tolerance | `1e-5` |
| fresh v52 vs v48 | `3 / 9` strict，`9 / 9` non-regressive/tie |
| fresh mean delta vs v48 | `+0.000086255` PSNR，`+0.000008742` SSIM，`-0.000015024` LPIPS |

### 5.4 v53 Policy-Val Alpha Calibration

v53 试图解决 v48/v52 的一个明显瓶颈：很多场景在保守 alpha grid 中选到了最大 `0.125`，说明 residual atlas 可能 under-apply 了可靠 residual。

新增机制：

```text
alpha* = argmin_alpha || teacher_residual - alpha * atlas_residual||^2
```

特点：

- 只使用 train policy-val residual samples；
- 生成的 alpha candidate 仍要通过 relative-gain、SSIM、image-L1、tail 和 min-view gates；
- held-out test metrics 只用于最终评价；
- 实现是 default-off，不影响 v48/v51/v52 原命令。

cap-hit 三场景验证结果：

| scene | accepted | selected alpha | changed | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| counter | 1 | `0.0625` | `6.5362%` | `-0.001730` | `-0.00002551` | `+0.00007340` | worse than v52 |
| kitchen | 1 | `0.5000` | `3.9361%` | `+0.004459` | `-0.00009871` | `-0.00024672` | PSNR/LPIPS up, SSIM down |
| bonsai | 0 | rejected/no-op | `0.0000%` | `-0.004087` | `-0.00007844` | `+0.00012958` | correctly rejected but fallback below v52 |

结论：v53 不推广。它证明 residual amplitude 确实是瓶颈，但单一全局 alpha 太粗。下一步应转向 per-face/per-region calibrated alpha、SSIM-aware local residual clipping 和 visibility-stratified alpha certification。

---

## 6. 训练与评估协议

SPCarNet 当前不是从零训练一个新网络，而是在 MeshSplatting 训练后进行 evidence-certified post-training optimization。

标准流程：

1. 训练或读取 clean MeshSplatting checkpoint；
2. 在 train views 上渲染 checkpoint，构建 evidence cache；
3. 使用 train / policy-val split 做 compaction 与 residual policy；
4. 写出 compact checkpoint、adapter 或 atlas 产物；
5. 在 held-out test views 上渲染；
6. 用统一 metrics pipeline 计算 PSNR、SSIM、LPIPS；
7. 额外做 topology、depth、normal、per-view、qualitative audit。

公平性原则：

- method 的 branch、alpha、edge fallback、fill mode、texture capacity、compaction ratio 不由 held-out test GT 选择；
- held-out test GT 只做最终 report-only evaluation；
- clean MeshSplatting baseline 从 clean `26000/30000` 中用 held-out test score 选强者，是为了给 baseline 更强 envelope；
- paper table 只做 sanity check，不作为最严格公平比较。

baseline 选择分数：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

在当前 full9 local baseline envelope 中，9 个场景最终都选择 clean `26000`，不是盲目选更久训练的 `30000`。

---

## 7. 主结果：Phase-J full9

### 7.1 全场景定量结果

| scene | branch | PSNR | SSIM | LPIPS | clean PSNR | clean SSIM | clean LPIPS | dPSNR | dSSIM | dLPIPS | tri red. | per-view |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.0215 | 0.7024 | 0.2661 | 23.3016 | 0.6599 | 0.3321 | +0.7199 | +0.0425 | -0.0660 | 11.81% | 25/25 |
| flowers | adaptive alpha | 20.3044 | 0.5578 | 0.3292 | 19.6823 | 0.5118 | 0.3946 | +0.6221 | +0.0459 | -0.0653 | 11.82% | 22/22 |
| garden | adaptive alpha | 26.3111 | 0.8278 | 0.1358 | 25.0292 | 0.7800 | 0.2013 | +1.2819 | +0.0478 | -0.0655 | 3.47% | 24/24 |
| stump | adaptive alpha | 25.5951 | 0.7241 | 0.2639 | 25.2050 | 0.7052 | 0.2940 | +0.3901 | +0.0189 | -0.0301 | 11.82% | 16/16 |
| treehill | edge fallback | 21.2962 | 0.5956 | 0.3363 | 20.9342 | 0.5645 | 0.4060 | +0.3620 | +0.0311 | -0.0697 | 11.81% | 17/18 |
| room | adaptive alpha | 30.3056 | 0.9057 | 0.1960 | 28.7473 | 0.8848 | 0.2499 | +1.5584 | +0.0209 | -0.0539 | 2.10% | 38/39 |
| counter | adaptive alpha | 28.4492 | 0.8937 | 0.1865 | 26.7518 | 0.8621 | 0.2520 | +1.6974 | +0.0317 | -0.0655 | 2.10% | 30/30 |
| kitchen | adaptive alpha | 30.1997 | 0.9161 | 0.1320 | 27.8186 | 0.8765 | 0.1992 | +2.3812 | +0.0396 | -0.0672 | 2.10% | 35/35 |
| bonsai | adaptive alpha | 31.8620 | 0.9303 | 0.1726 | 28.8952 | 0.8964 | 0.2595 | +2.9668 | +0.0339 | -0.0869 | 11.80% | 37/37 |

### 7.2 汇总

| 指标 | 值 |
|---|---:|
| strict RGB scene wins vs selected clean | `9 / 9` |
| mean dPSNR vs clean | `+1.331084` |
| mean dSSIM vs clean | `+0.034702` |
| mean dLPIPS vs clean | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| sparse geometry strict wins | `6 / 9` |
| sparse geometry-safe scenes | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |

### 7.3 与 MeshSplatting paper table 的关系

| source | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

Phase-J 相对 paper table 数值高：

```text
+1.7017 PSNR, +0.0555 SSIM, -0.0857 LPIPS
```

但 PPT 中必须谨慎：paper table 可能存在实现、数据预处理、分辨率、mask、evaluation script 等差异。因此最严谨 claim 是超过本地同协议 selected clean MeshSplatting baseline。

---

## 8. 定性展示建议

全图对比是必要的，但不是最有说服力的主视觉。原因是 SPCarNet 当前许多收益来自 residual-level 局部改善，缩在全图里不明显。

PPT 推荐三层定性展示：

1. 公平全图对比：证明同一 held-out view、同一 selected clean MeshSplatting baseline、同一评价口径；
2. 局部 crop / error reduction：展示 SPCarNet 相对 clean 更接近 GT 的区域；
3. representation-level atlas panel：说明 v42/v48/v52 已经能做 surface-addressed residual，但效果仍细微。

推荐主视觉图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

![Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

备选全图图：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

![Full9 qualitative gallery](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

v52 capacity-aware policy 的 cap-hit 局部对比图：

```text
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
```

![v52 capacity-aware cap-hit local panel](../../assets/spcarnet_v52_capacity_policy_cap_hit_panel.png)

这张图只展示 `counter/kitchen/bonsai`，因为这些是 v52 相对 v48 真正发生策略升级并严格提升的场景。局部 crop dPSNR 约为 `+0.019` 到 `+0.027`，说明 representation-level 改进是可测但视觉上仍然细微的。

PPT 使用建议：

- 主结果页放局部 error-reduction 图；
- fairness 页放全图对比；
- ablation / future work 页放 v42/v48 atlas 图和 v52 cap-hit 图；
- 不要只放全图，因为 mentor 很可能会直观看不出差异。

---

## 9. 为什么这是研究工作，不只是工程后处理

可以从四点讲：

1. Train-only decision loop：方法定义了 train evidence、policy-val gate、fallback/no-op 的完整决策协议，而不是 test-set 调参。
2. 几何与外观同时优化：结果仍围绕 MeshSplatting triangle mesh，有真实 triangle reduction、topology audit、depth/normal geometry safety。
3. 表示内化路线明确：v48/v51/v52 已经把 residual 从 image-space adapter 推向 surface-addressed atlas，并开始做容量和 support 自适应。
4. 负结果被策略吸收：v53 显示全局 alpha 不可靠，因此方法设计继续朝局部化、分层 gate 和 visibility-stratified certificate 走，而不是继续手动扫参。

对 mentor 可以这样说：

> 当前最强结果来自 render-time ELA，但它不是任意图像后处理，因为 residual 来源、alpha 选择、edge fallback 和失败回退都由训练视角 evidence 决定，并且与 compact mesh checkpoint、拓扑安全和 geometry audit 绑定。v48/v52 进一步说明这套策略可以被转写成 surface-addressed、train-only 自适应的 persistent representation 模块；接下来要冲论文上限的是把这个模块的容量和覆盖面做大。

---

## 10. 风险与诚实边界

| 风险 | 当前状态 | 汇报建议 |
|---|---|---|
| render-time adapter 风险 | Phase-J 最强收益来自 ELA，不是完全 baked representation | 主动承认，并用 v48/v52 说明内化路线 |
| 定性差异不总是肉眼明显 | 全图对比较 subtle，局部 crop/error map 更清楚 | PPT 主图用局部 held-out improvement showcase |
| triangle reduction 不够激进 | 平均 `7.65%`，偏 quality-first | 不讲极限压缩，讲 Pareto improvement |
| representation-level 收益小 | v48/v52 full9 均值提升是 `1e-3` 到 `1e-4` 量级 | 放在 ablation / next-step，不做 headline |
| support/capacity 受限 | `counter/kitchen/bonsai` 触及 v48 `2048` extra-face cap；v51/v52 说明扩大 support 有用 | 下一步做更强 persistent surface residual representation |
| source-config reproducibility gap | v52 source rerun 已完成，fresh audit `9 / 9` complete、`0` missing、`0` mismatch | 可以说 v52 表示级 ablation 已有可复跑 GPU render/eval 证据 |
| external protocol 还不完备 | courtyard 有验证，但非 full benchmark | 作为附加泛化证据，不作为主 claim |
| 顶会终局尚未闭合 | Phase-J 强，但 representation internalization 仍需大幅提升 | 明确下一阶段目标 |

---

## 11. Mentor 可能会问的问题

### Q1：这是不是只是在 MeshSplatting 后面加图像后处理？

不是简单后处理。ELA 的 residual 来自训练视角 evidence，并通过 train-only policy 控制 alpha、edge fallback 和回退；同时方法包括真实 compact checkpoint、topology audit 和 geometry safety。

但必须承认：当前最强 Phase-J 仍是 render-time adapter，不是完全 baked representation。所以 v48/v52 这一线正在解决表示内化问题。

### Q2：有没有用 test set 调参？

主方法选择不使用 test GT。baseline envelope 选择 clean `26000/30000` 时使用 held-out test score，是为了给 MeshSplatting 一个更强 comparator；method 自身的 branch、alpha、fallback、fill mode、texture capacity 和 compaction 决策都来自 train/policy-val evidence。

### Q3：为什么不是固定 clean 30000 当 baseline？

因为当前同协议 full9 clean envelope 中，clean `30000` 不一定优于 clean `26000`。如果永远用更久训练 checkpoint，反而可能选到 test 上更差的 baseline。我们从 clean checkpoints 中选更强 clean 行，是为了避免低估 MeshSplatting baseline。

### Q4：和 MeshSplatting 论文中的 24.78 PSNR 怎么比？

Phase-J full9 mean 是 `26.4828 / 0.7837 / 0.2243`，数值上高于 paper table 的 `24.78 / 0.728 / 0.310`。但论文表格可能存在实现、数据预处理和评价细节差异，所以只能作为 sanity check。最严谨 claim 是超过本地同协议 selected clean baseline。

### Q5：为什么局部图比全图更明显？

因为 residual 修复通常发生在纹理、边缘和局部高误差区域。全图 PSNR/SSIM/LPIPS 会累积这些小区域收益，但人眼在缩略全图上不一定明显。局部 error-reduction 图能更清楚展示“哪里变好了”。

### Q6：当前距离顶会终局还差什么？

最重要短板是：Phase-J 已经强，但外观收益仍主要来自 render-time ELA。顶会终局更希望看到一个高容量、可持久化、representation-level 的 residual repair operator，并且在全场景上达到更明显的可视化提升。v48/v52 是朝这个方向的真实推进，但 effect size 还不够。

### Q7：v53 失败说明什么？

它不是无意义失败。v53 证明 amplitude 确实可能是瓶颈，但单个 scene-level alpha 无法同时保证 PSNR、SSIM、LPIPS。下一步应做局部 alpha、per-region clipping、view/tail stratified certification，而不是继续全局放大 residual。

---

## 12. PPT 建议结构

建议做 12 到 16 页：

| 页码 | 标题 | 主信息 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Problem | MeshSplatting mesh 有价值，但仍有局部 residual error 和 topology redundancy |
| 3 | Key Idea | 用 train-view evidence 做 mesh 体检：安全压缩、可靠修复、风险回退 |
| 4 | Pipeline | `MeshSplatting -> evidence mining -> compact checkpoint -> ELA repair -> guarded policy -> held-out eval` |
| 5 | Difference from MeshSplatting | 原方法直接渲染；我们做 train-evidence-certified compaction + repair |
| 6 | Fair Protocol | selected clean baseline；method selection uses train-only evidence |
| 7 | Main Quantitative Result | full9 `9 / 9` strict RGB wins，`+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| 8 | Geometry / Compactness | `7.6479%` triangle reduction，`9 / 9` geometry-safe |
| 9 | Qualitative Main Figure | 放 `spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 10 | Per-View Reliability | `244 / 246` held-out views strict wins |
| 11 | Representation-Level Progress | v48 surface atlas：`7 / 9` strict vs no-op，`8 / 9` non-regressive/tie |
| 12 | Representation-Level Policy | v52 capacity-aware policy：保留 v48，cap-hit 升级 v51 |
| 13 | Negative Probe / Lesson | v53 alpha calibration：amplitude 是瓶颈，但全局 alpha 太粗 |
| 14 | Ablation / Lessons | Phase-J strong；v48 direction right；v51 identifies capacity bottleneck；v52 fixes global selection |
| 15 | Risks | render-time adapter、定性 subtle、v52 effect size 仍小 |
| 16 | Next Step | 推进高容量 persistent surface residual representation；让表示级收益更明显 |

---

## 13. 可直接放 PPT 的短段落

英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded Evidence Lumigraph Adapter. All repair decisions are made from train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

中文：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复表示”。我们先从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，再把训练 residual 中可靠的局部外观信息通过 guarded ELA 转移到 held-out view。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达到 `9/9` 场景三指标严格胜出，同时平均减少 `7.65%` triangles。最新 v48/v52 则把 residual repair 进一步推进到 train-only 自适应的 surface atlas 表示，说明该方向具备继续内化的可行性。

---

## 14. 关键证据路径

| 内容 | 路径 |
|---|---|
| Phase-J full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` |
| Phase-J closure audit | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md` |
| Phase-J closure JSON | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json` |
| Phase-J per-view deltas | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv` |
| fair baseline audit | `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/fair_baseline_audit.json` |
| Phase-J local qualitative showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| full9 qualitative gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| v48 method log | `docs/car_model/6-23-v48-AutoSupportSurfaceAtlas-Log.md` |
| v48 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.md` |
| v51 method log | `docs/car_model/6-23-v51-SupportFootprintLadder-Full9-Log.md` |
| v51-fast full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_full9_summary.md` |
| v52 capacity-aware policy log | `docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md` |
| v52 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md` |
| v52 selected small artifacts | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9` |
| v52 selected render gallery | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/qualitative_gallery.html` |
| v52 cap-hit qualitative panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v52 artifact pipeline report | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/v52_capacity_aware_pipeline_report.md` |
| v52 source-rerun status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_status.md` |
| v53 alpha calibration log | `docs/car_model/6-23-v53-PolicyValAlphaCalibration-CapHit-Log.md` |
| representation-atlas implementation | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` |
| fair-noop scene runner | `scripts/car_model/run_l1risk_fairnoop_scene.py` |
| 本报告 | `docs/car_model/6-23-SPCarNet-Mentor-PPT-Technical-Report-v20.zh.md` |

---

## 15. 明天汇报的最短讲稿

> 我们现在的主线叫 SPCarNet，它不是重新发明 MeshSplatting，而是在 MeshSplatting 训练后增加一个 train-evidence-certified 的压缩和修复阶段。核心是利用训练视角中可观测的 surface evidence：一方面判断哪些 triangles 可以安全删除，另一方面把稳定的局部 residual 作为外观修复信号迁移到 held-out view。为了避免 test-set 调参，我们所有 branch、alpha、fallback 和 atlas policy 都由 train/policy-val evidence 决定，test GT 只用于最终 report。
>
> 当前最强 Phase-J endpoint 在 Mip-NeRF360 full9 上，相对本地同协议 selected clean MeshSplatting baseline 达到 `9/9` 场景 PSNR、SSIM、LPIPS 三指标严格胜出，平均提升 `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS，同时平均减少 `7.65%` triangles，`244/246` 个 held-out views 也三指标严格胜出。几何上 `9/9` geometry-safe，`6/9` sparse geometry 严格更好。
>
> 目前最大的诚实短板是 Phase-J 的主要收益仍来自 render-time Evidence Lumigraph Adapter。为了解决这个问题，我们已经推进了 v48 surface residual atlas，把 residual repair 转成 train-only 自扩展 support 的 surface-addressed 表示。v48 full9 相对 no-op compact baseline 有 `7/9` strict、`8/9` non-regressive/tie，但收益量级还小。v51-fast full9 说明 support-footprint ladder 在 cap-hit `counter/kitchen/bonsai` 上能进一步超过 v48/v50/no-op；v52 进一步把这个观察固定成 train-only capacity-aware policy：非 cap-hit 场景保留 v48，cap-hit 场景升级 v51。这样 v52 相对 v48 达到 `3/9` strict、`9/9` non-regressive/tie，mean `+0.00008689` PSNR、`+0.000008782` SSIM、`-0.000015303` LPIPS。v53 说明全局 alpha calibration 不够可靠，因此下一步要做局部化、分层风险认证的 persistent residual representation。

---

## 16. 最终汇报判断

可以这样对 mentor 总结：

> 当前工作已经有一条可信主线：Phase-J 在本地强 MeshSplatting baseline 上 full9 全场景严格胜出，同时有真实删面、per-view 审计和 geometry-safe 证据。它足够作为阶段性强结果汇报。最需要继续攻克的是 representation-level 内化：v48 已经证明 surface-addressed residual atlas 可以用 train-only policy 做 support/容量自适应，并在 full9 上取得均值正收益；v51-fast full9 进一步说明 cap-hit 场景可以通过更大的 support footprint 获得增益；v52 则把二者合成固定 capacity-aware effective policy，保留 v48 的全局 auto-policy 长处，只在 `counter/kitchen/bonsai` 启用 v51 ladder，并已写出 selected small-artifact tree、一键 artifact pipeline 和 W&B source-config full9 rerun 证据。v53 的负结果说明下一步不能继续靠全局 alpha 扫描，而应进入 per-face/per-region 局部 residual capacity 与更强 tail-safe certification。下一阶段目标是把 Phase-J 的强 render-time 收益尽可能内化到 persistent surface representation，让定性和定量提升都更明显。
