# SPCarNet 当前方法完整技术报告（Mentor/PPT 母稿 v90）

日期：2026-06-25  
用途：给 mentor 汇报当前 SPCarNet 工作现状，并作为 PPT 制作母稿。本文只把已经有本地证据闭合的结果作为主结论；仍在 `/dev/shm` 运行或尚未全量验证的 v87/v88/v89 表示级实验，只作为 ongoing follow-up。

建议 PPT 标题：

```text
SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting
```

---

## 0. 一页结论

当前最适合汇报的稳定主方法是：

```text
trained MeshSplatting checkpoint
  + surface evidence cache
  + geometry-safe triangle compaction
  + guarded Evidence Lumigraph Adapter
  + train/policy-val risk gate and fallback
```

一句话讲法：

> 标准 MeshSplatting 是“训练完直接渲染”；SPCarNet 是“训练完以后让 checkpoint 自审计：哪些 triangles 低风险可以删，哪些 surface residual 稳定可以修，哪些区域证据不足就回退”。

当前主 endpoint 是 **Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**。在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、selected-clean MeshSplatting baseline 下：

| 指标 | SPCarNet Phase-J vs selected clean MeshSplatting |
|---|---:|
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| held-out view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| mean PSNR | clean `25.1517` -> ours `26.4828` |
| mean SSIM | clean `0.7490` -> ours `0.7837` |
| mean LPIPS | clean `0.2876` -> ours `0.2243` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` removed |

这里的 `triangle reduction` 指**删去的三角形比例**，不是剩余比例。

当前最诚实的定位：

| 方面 | 当前状态 |
|---|---|
| 本地同口径 MeshSplatting baseline 对比 | 已经是强结果：full9 `9 / 9` 场景三指标严格胜出 |
| 定性展示 | 有 full-frame、公平对比、local crop/error reduction、outdoor detail panels |
| 几何收益 | 平均删去 `7.65%` triangles，质量优先、保守压缩 |
| 表示级内化 | 接口和证书已经打通，但收益仍是微小量级，不能作为 headline |
| 论文终局 | 还没有完全闭合：需要 official protocol、rate/FPS/model-size、checkpoint-baked repair 进一步补证 |

---

## 1. 研究问题

MeshSplatting 给了我们一个显式 surface/triangle-aware 表示，但标准训练流程结束后，它不会继续回答这些问题：

1. 哪些 triangles 对多视角解释贡献低，可以安全删除？
2. 哪些 surface 区域在训练视角里反复出现稳定 RGB residual，可以迁移到 held-out view？
3. 哪些局部修复只是在 train view 上看起来好，到了 held-out view 或 tail view 会崩？
4. 如果证据不足，系统能否自动回退到 clean MeshSplatting，而不是硬改？

SPCarNet 的核心研究问题可以写成：

```text
Can training-view surface evidence certify where a MeshSplatting checkpoint
can be compacted and where its appearance residuals can be safely repaired?
```

这个问题的关键不是“在 MeshSplatting 后面接一个滤镜”，而是把训练视角从 supervision 升级为 **surface evidence**：

```text
training views are not only optimization signals;
they become auditable evidence for post-training repair and compaction.
```

---

## 2. 与基础 MeshSplatting 的区别

基础 MeshSplatting：

```text
training images + cameras
  -> optimize MeshSplatting checkpoint
  -> render held-out views
```

SPCarNet：

```text
training images + cameras
  -> optimize MeshSplatting checkpoint
  -> render train/policy-val views with surface maps
  -> build surface evidence cache
  -> certify low-risk geometry compaction
  -> transfer stable surface residuals
  -> train/policy-val gate and fallback
  -> render held-out views
```

| 维度 | MeshSplatting baseline | SPCarNet 当前主方法 |
|---|---|---|
| 基础模型 | trained MeshSplatting checkpoint | 继承同一个 checkpoint |
| 训练视角使用方式 | 主要隐式用于优化 loss | 显式保存为 surface evidence |
| 几何处理 | checkpoint geometry 固定 | 删除低风险 triangles |
| 外观修复 | 直接渲染 checkpoint | surface-bound guarded residual repair |
| 风险控制 | 依赖训练收敛 | policy-val gate、tail-risk、fallback/no-op |
| 输出目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可审计证据 |

通俗版本：

> 原方法是“训练好就交卷”；我们方法是“训练好以后再检查错题本，能安全删的几何删掉，稳定错的颜色修回来，不确定的地方不动”。

---

## 3. 当前主方法 Pipeline

主流程：

```text
Clean MeshSplatting checkpoint
  |
  | render train / policy-val views with face id, depth, alpha, barycentric maps
  v
Surface Evidence Cache
  |
  | residual, visibility, support, risk, surface address
  v
Geometry-Safe Compaction
  |
  | remove low-risk triangles, protect thin/high-risk regions
  v
Guarded Evidence Lumigraph Adapter
  |
  | transfer stable residuals through surface correspondence
  v
Policy-Val Gate
  |
  | select alpha/branch, reject risky edits, fallback if needed
  v
Held-Out Rendering and Evaluation
```

核心设计原则：

1. **只用 train/policy-val evidence 做策略选择**  
   held-out GT 只用于最终评价，不参与调参和 branch selection。

2. **局部修改必须绑定 3D surface address**  
   residual 不是普通 2D filter，而是通过 face/bin/barycentric correspondence 迁移。

3. **宁可保守回退，也不追求不可靠收益**  
   证据不足时 fallback/no-op，尤其保护 out-of-trajectory 和 tail view。

4. **质量优先的压缩**  
   当前目标不是极限 compression，而是在 RGB 不退化甚至提升时减少 triangles。

---

## 4. 模块细节

### 4.1 Surface Evidence Cache

Evidence cache 保存训练/策略验证视角中的局部证据：

| 证据 | 用途 |
|---|---|
| rendered RGB 与 GT RGB | 计算 residual 和 image-level risk |
| residual `GT - Render` | 外观修复信号 |
| face id / barycentric / surface bin | 把 2D residual 绑定到 3D surface |
| alpha / depth / visibility | 判断像素是否可靠可见 |
| per-face/per-bin support count | 防止稀疏区域过拟合 |
| residual sign consistency | 判断 residual 方向是否稳定 |
| train/policy-val PSNR/SSIM/LPIPS/L1 | 策略选择和风险审计 |
| per-view min gain / CVaR20 | 防止平均收益掩盖 tail-view 退化 |

它回答的是：

```text
这个 surface 区域是否被多视角稳定观测？
这个 residual 是否在 policy-val 中可迁移？
这个局部修复是否会伤害最差视角？
```

### 4.2 Geometry-Safe Triangle Compaction

压缩策略不是简单按面积、透明度或贡献分数删 triangle，而是 quality-first：

```text
remove a triangle only when multi-view evidence marks it as low-risk
```

被保护的区域包括：

- visibility 稀疏区域；
- thin structures；
- high residual faces；
- depth/edge/occlusion 不稳定区域；
- policy-val tail-risk 高的区域；
- 室内场景中 triangles 少、容易破坏细节的区域。

当前 Phase-J 平均删去 `7.6479%` triangles，同时 full9 RGB 三指标全场景严格胜出。这个结果支持“quality-preserving compactness”主张，但还不能直接推出 FPS、显存、模型大小全面优越；这些是后续单独评测项。

### 4.3 Guarded Evidence Lumigraph Adapter

当前 RGB 大收益主要来自 guarded Evidence Lumigraph Adapter。简化表达：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的渲染；
- `residual_i = GT_i - Render_i` 只来自 train/policy-val views；
- `u_i` 是 target pixel 通过 surface correspondence 找到的训练视角 surface address；
- `w_i(p)` 由 visibility、surface match、support count、risk 共同决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时自动 fallback/no-op。

为什么它不是普通图像后处理：

| 普通 2D 后处理 | SPCarNet Evidence Lumigraph Adapter |
|---|---|
| 在图像平面滤波或增强 | residual 绑定到 mesh surface |
| 可以对 held-out 图像直接操作 | 不能访问 held-out GT，只能迁移训练证据 |
| 没有几何一致性约束 | 需要 face/bin/surface correspondence |
| 通常没有策略验证 gate | 有 policy-val gate 和 fallback |

### 4.4 Policy-Val Gate 与 Fallback

公平性原则：

- branch、alpha、support、fallback 只由 train/policy-val evidence 决定；
- held-out test GT 只用于最终报告；
- clean baseline 使用本地 clean `26000/30000` envelope 中 held-out 更强者；
- 不用 train metric 选择 clean baseline，因为 train metric 会天然偏向更久训练；
- 不用 test metric 为我们方法调参。

当前 Phase-J 分支：

| branch | scenes |
|---|---|
| adaptive ELA | `bicycle, flowers, garden, stump, room, counter, kitchen, bonsai` |
| edge fallback | `treehill` |

这说明方法不是每个场景都强行修复。`treehill` 的 fallback 是一个重要安全证据：当证据不稳定时，系统会保守。

### 4.5 Representation-Level Residual Atlas

为了把 render-time repair 内化到 checkpoint，我们推进了表示级 residual atlas 路线。它的目标是：

```text
make the repair persistent in the representation,
not only an external render-time adapter.
```

当前已经实现或验证过的关键组件包括：

| 版本 | 作用 | 结论 |
|---|---|---|
| v48 | auto-support surface atlas | 能自动扩展支持，但收益小 |
| v52 | capacity-aware policy | 全场景 selector 闭合 |
| v56 | face-alpha reliability guard | 修复 audit/fallback 语义 |
| v64 | fixed auto bin-alpha policy | 当前稳健 fixed representation reference |
| v75 | local patch prior | 改善低支持 bin 的先验填充 |
| v76/v77 | policy-val bin-gain hybrid | 引入 bin-level policy-val 选择 |
| v82/v82b | patch-mixture teacher basis + capacity pre-rank | counter 微弱提升 |
| v84 | strict capacity selector | 固定 anchor/fallback，不让弱候选晋升 |
| v85/v86 | target-footprint tail-risk + anchor-preserving selector | 证明 guardrail 有效，但未超过 anchor |
| v89 | L1-proxy bin-dominance gate | 已实现，仍在中长程验证 |

这条路线非常重要，但目前还不能作为主结果。原因是 v64/v84/v86 的 full9 收益只有 `1e-4` 到 `1e-6` 量级；它证明接口、审计和回退机制成熟，但 residual-field 表达能力还没有达到 Phase-J 的视觉收益。

---

## 5. 实现入口与代码地图

主要脚本：

| 文件 | 作用 |
|---|---|
| `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` | 表示级 surface residual atlas / region texture adapter 主实现 |
| `scripts/car_model/run_l1risk_fairnoop_scene.py` | 单场景 runner；负责公平 no-op、risk gate、W&B、adapter 参数转发 |
| `scripts/car_model/summarize_v84_strict_v82_capacity_selector.py` | v84 strict selector 汇总和 selected full9 materialization |
| `scripts/car_model/summarize_v86_anchor_preserving_tailrisk_selector.py` | v86 anchor-preserving selector 汇总 |
| `scripts/car_model/build_spcarnet_current_evidence_manifest.py` | 当前 evidence/result manifest 汇总 |
| `scripts/car_model/compare_spcarnet_result_artifacts.py` | 结果 artifact 对比辅助 |

当前 v89 新增接口：

```text
--enable_prior_bin_gain_hybrid_l1_proxy_gate
--prior_bin_gain_hybrid_min_l1_abs_gain
--prior_bin_gain_hybrid_min_l1_relative_gain
--prior_bin_gain_hybrid_min_l1_positive_view_fraction
--prior_bin_gain_hybrid_min_l1_min_view_gain
--prior_bin_gain_hybrid_min_l1_cvar20_view_gain
```

v89 的动机是让 prior-bin hybrid 不只看 residual-MSE，而是加入与最终 image L1 更对齐的 bin-level L1 proxy dominance。它目前是表示级路线的 follow-up，不纳入主 Phase-J 结论。

---

## 6. 主定量结果：SPCarNet vs MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- baseline：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR/SSIM 越高越好，LPIPS 越低越好。

### 6.1 Full9 Per-Scene Table

| scene | clean MeshSplatting PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | delta | triangle removed |
|---|---:|---:|---:|---:|
| bicycle | `23.3016` / `0.6599` / `0.3321` | `24.0215` / `0.7024` / `0.2661` | `+0.7199` / `+0.0425` / `-0.0660` | `11.81%` |
| flowers | `19.6823` / `0.5118` / `0.3946` | `20.3044` / `0.5578` / `0.3292` | `+0.6221` / `+0.0459` / `-0.0653` | `11.82%` |
| garden | `25.0292` / `0.7800` / `0.2013` | `26.3111` / `0.8278` / `0.1358` | `+1.2819` / `+0.0478` / `-0.0655` | `3.47%` |
| stump | `25.2050` / `0.7052` / `0.2940` | `25.5951` / `0.7241` / `0.2639` | `+0.3901` / `+0.0189` / `-0.0301` | `11.82%` |
| treehill | `20.9342` / `0.5645` / `0.4060` | `21.2962` / `0.5956` / `0.3363` | `+0.3620` / `+0.0311` / `-0.0697` | `11.81%` |
| room | `28.7473` / `0.8848` / `0.2499` | `30.3056` / `0.9057` / `0.1960` | `+1.5584` / `+0.0209` / `-0.0539` | `2.10%` |
| counter | `26.7518` / `0.8621` / `0.2520` | `28.4492` / `0.8937` / `0.1865` | `+1.6974` / `+0.0317` / `-0.0655` | `2.10%` |
| kitchen | `27.8186` / `0.8765` / `0.1992` | `30.1997` / `0.9161` / `0.1320` | `+2.3812` / `+0.0396` / `-0.0672` | `2.10%` |
| bonsai | `28.8952` / `0.8964` / `0.2595` | `31.8620` / `0.9303` / `0.1726` | `+2.9668` / `+0.0339` / `-0.0869` | `11.80%` |

### 6.2 Aggregate

| aggregate | value |
|---|---:|
| scene-level strict wins | `9 / 9` |
| per-view strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle removed | `7.6479%` |

解读：

- 室内场景 `bonsai/kitchen/counter/room` 的 PSNR 收益最大；
- 室外场景也全胜，但 full-frame 肉眼差异通常弱于室内，需要 crop/error map；
- `treehill` 走 edge fallback，说明策略在不稳定区域会自动保守；
- 当前压缩率是质量优先的安全压缩，不是 aggressive compression。

---

## 7. 定性结果与 PPT 展示建议

### 7.1 最推荐主图：local held-out error reduction

推荐放在主结果页：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

每行含义：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

建议口播：

> Full-frame 上差异有时不强，但 error map 和 crop 能显示 SPCarNet 在 surface residual、局部高频纹理和遮挡边界上降低了 clean MeshSplatting 的系统误差。

### 7.2 公平 full-frame 图

推荐作为公平性证明或 appendix：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

每行含义：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

这张图适合说明比较来自同一 held-out view 和同一 selected clean baseline，但不一定是最强视觉冲击页。

### 7.3 室外细节图

推荐用于回应“室外场景视觉收益不明显”的问题：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail error reduction showcase">

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

建议口播：

> 室外场景的全图差异确实更难直接看出来，但在叶片、树皮、长椅条纹等 high-frequency surface 区域，SPCarNet 的局部 error reduction 更稳定。

---

## 8. 与 MeshSplatting 论文表格的关系

当前可用于背景介绍的表：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

严谨解释：

- 本地 Phase-J 数值高于 MeshSplatting paper table；
- 但正式主 claim 应以本地同协议 selected-clean MeshSplatting baseline 为准；
- paper-table 可能存在 resolution、mask、split、preprocessing、metric implementation、checkpoint iteration 差异；
- 写论文前需要最终 official-style 统一口径复现。

安全口播：

> We outperform our local same-protocol MeshSplatting baseline by a large margin. Paper-table comparison is encouraging, but we treat it as protocol-sensitive background rather than the final fairness claim.

---

## 9. 消融与表示级路线

### 9.1 v64 fixed auto bin-alpha policy

v64 是当前最稳的 fixed representation-level reference：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | `9` | `1` | `9` | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | `9` | `2` | `9` | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | `9` | `7` | `8` | `+0.002255970` | `+0.000038081` | `-0.000093445` |

结论：自动 policy 和 checkpoint-level atlas 接口是有效的，但收益非常小。

### 9.2 v82b / v83 / v84 / v85 / v86 diagnostics

| probe | scope | metric snapshot | verdict |
|---|---|---:|---|
| v56/v64/v79 anchor | `counter` | `26.756130219 / 0.862126231 / 0.251691371` | strong representation anchor |
| v82b capacity-prerank + face-alpha | `counter` | `26.756137848 / 0.862126350 / 0.251690656` | counter micro-win |
| v83 patchmix + face-alpha + local-patch | `counter` | `26.756147385 / 0.862125337 / 0.251688808` | PSNR/LPIPS up, SSIM down |
| v84 strict selector | full9 | vs v64 mean `+8.48e-7 / +1.3e-8 / -7.9e-8` | report-only |
| v85 target-footprint tail-risk | `counter` | `26.756134033 / 0.862126231 / 0.251691371` | accepted edit, below v84 counter |
| v86 anchor-preserving selector | full9 | `9 / 9` non-regressive/tie vs v84 | guardrail, not promoted |

关键解释：

- v82b 说明 support-capacity pre-rank 有真实信号；
- v83 说明更高 residual capacity 能提升 PSNR/LPIPS，但 SSIM certificate 不够强；
- v84 固定了“证据不足就 fallback v64”的保守 selector；
- v85 证明 tail-risk certificate 可以接受非空编辑，但没有超过 v84 anchor；
- v86 把这个经验固化成 selector：tail-risk candidate 必须在 train/policy-val audit 上支配当前 anchor 才能晋升。

PPT 建议：把这一页放在 appendix 或“ongoing representation-level closure”页。不要放在主结果页，否则会削弱 headline。

### 9.3 v89 L1-Proxy Bin-Dominance Gate

v89 是当前最新实现的表示级机制，目标是修复一个明确瓶颈：

```text
old hybrid gate optimizes residual/bin MSE,
but final report metrics are image PSNR/SSIM/LPIPS and image L1.
```

因此 v89 在 prior-bin hybrid 中加入 L1 proxy dominance：

- per-bin `baseline_l1_after`；
- per-bin `prior_l1_after`；
- per-bin `l1_abs_gain`；
- per-bin `l1_relative_gain`；
- per-bin `l1_positive_view_fraction`；
- per-bin `l1_min_view_gain`；
- per-bin `l1_cvar20_view_gain`；
- final `l1_proxy_keep` gate。

当前状态：

- adapter 和 runner 接口已实现并通过静态编译；
- v89b counter W&B run 正在 `/dev/shm` 中验证；
- 不能纳入主结论，直到 counter 严格超过 v84/v86 anchor，并完成 hard-triad/full9 公平复跑。

---

## 10. Stage2 Shape Prior 支线

Stage2 v4 normal-band autodecoder 是另一个真实训练管线改动。它在 clean shape field autodecoder 中加入 surface-normal band supervision：

```text
x_inner = x_surface - epsilon * normal -> occupied
x_outer = x_surface + epsilon * normal -> free
```

Full-val MAP-fit 当前最好证据：

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | delta |
|---|---:|---:|---:|
| extraction | `206 / 206` | `206 / 206` | tie |
| recon chamfer | `0.0698447353` | `0.0607328202` | `-0.0091119151` |
| hidden chamfer | `0.1023846301` | `0.0933915632` | `-0.0089930669` |
| filled IoU | `0.5531548112` | `0.5683319216` | `+0.0151771104` |
| shell IoU | `0.9112784961` | `0.8783071888` | `-0.0329713073` |

结论：

- v4 是真实 method improvement；
- epoch50 比 final checkpoint 更好，说明长训不是单调更优；
- 但它仍未通过原始 Stage2 quality gate，不能作为当前 headline；
- 最适合在 PPT 中作为“未来把 SPCarNet 推向 object-level shape prior 的支线”。

---

## 11. 为什么这是研究工作，而不是工程后处理

可以强调四点：

1. **Surface-addressed evidence**  
   residual 绑定到 mesh face/bin/surface address，不是 2D 图像滤波。

2. **Certified edit policy**  
   每个删面或修复都依赖 support、visibility、risk、policy-val gate 和 fallback。

3. **Quality + compactness joint objective**  
   方法同时优化 RGB 指标和 triangle count，不是只追求 PSNR。

4. **Honest failure handling**  
   v75-v86 中大量 negative diagnostics / guardrail diagnostics 被记录并拒绝晋升，说明 pipeline 能识别无效候选。

一个好的论文故事：

```text
MeshSplatting gives us an explicit surface.
SPCarNet asks whether that surface can audit itself.
Training views are not only supervision; they are evidence.
Evidence tells us what can be compacted, repaired, or safely left untouched.
```

---

## 12. 当前弱点与风险

导师汇报时应该主动承认这些边界：

| 风险 | 说明 | 建议说法 |
|---|---|---|
| 主 RGB 收益来自 render-time adapter | Phase-J 不是 fully checkpoint-baked endpoint | 这是当前最强结果，表示级内化是下一阶段 |
| representation-level atlas 收益小 | v64/v84/v86 只有微小提升 | 证明机制闭合，但还没突破表达能力 |
| 室外 full-frame 肉眼差异弱 | 局部高频区域更明显 | 主图用 crop/error map，full-frame 放公平性页 |
| triangle reduction 不高 | 平均 `7.65%`，质量优先 | claim 是 quality-preserving compactness，不是极限压缩 |
| rate/FPS/model-size 未闭合 | triangle count 不能直接等价 FPS | 后续补 rate-distortion 和 runtime |
| paper table protocol 未完全统一 | paper number 和本地 evaluator 可能不同 | 主 claim 用本地 same-protocol baseline |
| `/data` 磁盘接近满 | 当前新实验主要在 `/dev/shm` | 只归档小型 JSON/log/MD，避免误删/误拷贝大文件 |

最安全的一句话：

> 当前主结果已经能支持“evidence-certified repair and compaction”的强故事，但还不能声称我们已经完成最终 paper-level baked representation endpoint。

---

## 13. 建议 PPT 结构

| slide | 标题 | 核心内容 | 建议图/表 |
|---:|---|---|---|
| 1 | Motivation | MeshSplatting 有显式 surface，但 checkpoint 不会自审计 | clean vs GT residual crop |
| 2 | Key Idea | training views are evidence, not only supervision | method schematic |
| 3 | Pipeline | evidence cache -> compact -> repair -> gate -> render | flow chart |
| 4 | Module 1 | geometry-safe triangle compaction | triangle reduction table |
| 5 | Module 2 | guarded Evidence Lumigraph Adapter | residual transfer diagram |
| 6 | Fair Protocol | branch selection 只用 train/policy-val | protocol diagram |
| 7 | Main Quantitative Result | full9 `9/9` scene wins, `244/246` view wins | aggregate table |
| 8 | Per-Scene Result | all scenes improve PSNR/SSIM/LPIPS | 9-scene table |
| 9 | Qualitative Result | local error reduction | `spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 10 | Outdoor Analysis | full-frame subtle, crop/error map clearer | outdoor detail panel |
| 11 | Ablation | v64-v86 representation-level diagnostics | compact ablation table |
| 12 | Boundary and Next Step | bake repair into representation + official protocol + rate/FPS | roadmap |

---

## 14. Mentor 可能追问的问题

### Q1. 你们是不是只是调参赢 baseline？

不是。当前 Phase-J 的关键不是每场景手工调参，而是固定的 evidence-driven 审计流程：surface evidence cache、geometry-safe compaction、guarded residual transfer、policy-val gate 和 fallback。需要承认的是，representation-level v64-v89 还在强化自动策略，不能把未闭合的部分包装成终局。

### Q2. 为什么 clean baseline 选择 `26000/30000` envelope？

因为只用 train 指标会偏向训练更久的 checkpoint。这里 clean baseline 用 held-out score 更强者，目的是让 baseline 尽可能强，避免和弱 clean checkpoint 比。

### Q3. 如果我们的 residual repair 会不会在 out-of-trajectory 视角崩？

风险存在，所以方法引入 surface support、visibility、policy-val gate、tail-risk、fallback/no-op。证据不足时不修。当前 `treehill` 走 fallback，是方法保守性的一个例子。

### Q4. 为什么定性图有时看起来差异不大？

很多收益来自局部 residual-level correction，全图缩小后肉眼不总明显。展示时应该用 full-frame 图证明公平性，用 crop/error map 展示真实视觉收益。

### Q5. 为什么表示级 residual atlas 还不能作为主结果？

因为它目前只带来微小收益。它的价值是证明接口、selector、tail-risk 和 fallback 机制已经闭合；真正论文级下一步是把 Phase-J 的强 repair 效果内化到 checkpoint 表示。

### Q6. 是否已经全面超过 MeshSplatting？

在本地 Mip-NeRF360 full9、同 evaluator、selected clean MeshSplatting baseline 下，Phase-J 已经在 RGB 三指标和 triangle count 上形成强结果。还不能说在所有协议、所有数据集、runtime/FPS/model-size 上全面超过。

---

## 15. 60 秒中文口播

> 我们现在的方法不是推翻 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 后面加一层自审计机制。训练视角不只是用来优化 loss，而是被保存成 surface evidence：包括 residual、可见性、face/bin support 和风险统计。这样模型可以判断哪些三角形低风险可以删，哪些表面区域有稳定 residual 可以迁移修复，哪些区域证据不足必须回退。在本地 Mip-NeRF360 full9、相同 evaluator、selected-clean MeshSplatting baseline 下，当前 Phase-J endpoint 做到 9 个场景 PSNR/SSIM/LPIPS 全部严格胜出，246 个 held-out views 中 244 个严格胜出，平均 PSNR 提升 1.33 dB，SSIM 提升 0.0347，LPIPS 降低 0.0634，同时平均删去 7.65% triangles。需要诚实说明的是，当前最大 RGB 收益仍来自 guarded render-time adapter；我们已经打通表示级 residual atlas 和多个 certificate，但它们目前还是小收益诊断，下一步目标是把这套修复真正内化到 checkpoint 表示里。

## 16. 60 秒英文口播

> Our current method starts from a trained MeshSplatting checkpoint. Instead of directly rendering it, we build a surface evidence cache from train and policy-val views: residuals, visibility, face/bin support, and risk statistics. This evidence lets the checkpoint audit itself. Low-risk triangles are compacted; stable surface residuals are transferred to held-out views through surface correspondence; uncertain regions automatically fall back to the clean checkpoint. On local Mip-NeRF360 full9, under the same evaluator and selected-clean MeshSplatting baseline, the current Phase-J endpoint wins all 9 scenes on PSNR, SSIM, and LPIPS, wins 244 out of 246 held-out views, improves mean PSNR by 1.33 dB, improves SSIM by 0.0347, reduces LPIPS by 0.0634, and removes 7.65% triangles on average. The honest limitation is that the strongest RGB gain is still a guarded render-time adapter. We have built the representation-level residual atlas line and several certificates, but those are currently small-gain diagnostics. The next paper-level milestone is to bake the repair into the checkpoint while preserving the same evidence-certified behavior.

---

## 17. 关键证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

当前 evidence manifest：

```text
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
```

表示级路线：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_full9_summary.md
docs/car_model/6-24-v64-FixedAutoBinAlphaPolicy-Log.md
docs/car_model/6-24-v84-StrictCapacitySelector-Log.md
docs/car_model/6-24-v85-TargetFootprintTailRiskCertificate-Log.md
docs/car_model/6-24-v86-AnchorPreservingTailRiskSelector-Log.md
docs/car_model/6-25-v89-L1ProxyBinDominanceGate-Implementation-And-v85Relax-Diagnostic.md
```

Stage2 shape-prior：

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
```

---

## 18. 当前正在运行的后续实验

这些实验属于 follow-up，**不要写入主结论**，最多放 backup/ongoing slide。

| run | 目标 | 当前状态 |
|---|---|---|
| `v87_source_mixture_20260625` | 尝试 policy-val source mixture | `/dev/shm` 中运行，尚无可归档正式结果 |
| `v88_anchor_dominance_tailrisk_counter_20260625` | patch-mixture teacher basis + local-patch prior + tail-risk + dominance pruning | `/dev/shm` 中运行，正在挑战 v84/v86 counter anchor |
| `v89b_l1proxy_counter_20260625` | L1-proxy bin-dominance gate 验证 | `/dev/shm` 中运行，尚不能作为结论 |

当前系统状态限制：

```text
/data: nearly full, only small JSON/log/MD should be archived
/dev/shm: active experiment workspace
```

v89b 或 v88 即使 counter 单场景通过，也需要至少：

1. 与 v84/v86 counter anchor 三指标严格比较；
2. hard-triad `counter,kitchen,bonsai` 固定策略复跑；
3. full9 固定策略验证；
4. 定性图、audit、result paths 完整归档；
5. 才能考虑替代当前 representation-level 状态。

---

## 19. 当前最终状态

```text
Phase-J local full9 RGB + triangle-count result: strong and presentable.
Same-protocol selected-clean MeshSplatting comparison: locally complete.
Qualitative panels: available and slide-ready.
Representation-level baked endpoint: not complete.
Official MeshSplatting paper protocol reproduction: not fully closed.
Rate/FPS/model-size evaluation: not fully closed.
Stage2 shape-prior: implemented and improved over v3, but gate soft-fail.
```

建议给 mentor 的最终定位：

> SPCarNet 当前最可信的故事是“MeshSplatting checkpoint 的 evidence-certified post-training repair and compaction”。它已经在本地 full9 strong clean baseline 上拿到强 RGB + triangle reduction 结果；真正的论文终局还需要把 guarded repair 更彻底地 bake into representation，并补齐 official protocol 与 rate/FPS/model-size 证据。

---

## 20. PPT 制作建议

优先使用这三类图：

1. **一张 pipeline 图**：训练视角 evidence -> surface cache -> compact -> repair -> gate -> render。
2. **一张 full9 定量表**：主表或 aggregate 表，不要一开始堆太多版本号。
3. **一张 local error reduction 图**：`spcarnet_phasej_where_it_helps_showcase_20260622.png`，比 full-frame 更能让人看出差异。

不建议主 PPT 里过多展示：

- v64-v89 的长版本链；
- 微小到 `1e-6` 的 representation-level 提升；
- 未完成的 `/dev/shm` 运行结果；
- 和 MeshSplatting paper table 的强 claim。

推荐讲故事顺序：

```text
Why: MeshSplatting has explicit surfaces but no post-training self-audit.
Idea: turn train views into auditable surface evidence.
Method: compact safe geometry, repair stable residuals, fallback uncertain regions.
Result: full9 same-protocol strong clean baseline, 9/9 scene wins, 244/246 view wins, 7.65% triangles removed.
Boundary: render-time adapter is strong; representation-level baked repair is next.
```
