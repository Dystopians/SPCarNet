# SPCarNet 当前方法完整技术报告（Mentor/PPT 母稿）

日期：2026-06-25  
用途：给 mentor 汇报当前 SPCarNet 进展，可直接拆成 PPT；本文只把已经有证据支撑的内容作为结论，把仍未闭合的表示级路线和风险边界单独标出。  
建议 PPT 标题：**SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting**

---

## 1. 一页结论

当前最适合汇报的主方法是：

```text
trained MeshSplatting checkpoint
  + surface evidence cache
  + geometry-safe triangle compaction
  + guarded Evidence Lumigraph Adapter
  + train/policy-val risk gate and fallback
```

一句话讲清楚：

> MeshSplatting 训练完以后直接渲染；SPCarNet 训练完以后先让 checkpoint 做一次自审计：哪些 triangles 低风险可以删，哪些表面 residual 稳定可以修，哪些区域证据不足就回退。

当前主 endpoint 是 **Phase-J guarded adaptive ELA + geometry-safe compaction**。在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、selected-clean MeshSplatting baseline 下：

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
| mean triangle reduction | `7.6479%` |

这里的 `triangle reduction` 指**删去的三角形比例**，不是剩余比例。

当前最诚实的定位：

- **可以汇报为强结果**：本地 same-protocol full9 相对强 clean MeshSplatting baseline 全场景三指标严格胜出，同时平均删去 `7.65%` triangles。
- **不能包装成已完成的论文终局**：最大 RGB 收益仍主要来自 guarded render-time adapter，不是完全 baked into checkpoint representation。
- **表示级路线已经打通但仍弱**：v64/v84/v86 证明 residual atlas、自动 selector、tail-risk guardrail 可运行，但收益是 `1e-4` 到 `1e-6` 量级，还不能替代 Phase-J。
- **汇报策略**：主讲“evidence-certified post-training repair and compaction”；把 representation-level residual atlas 讲成 next-stage paper direction，而不是 headline claim。

---

## 2. 研究问题与动机

基础 MeshSplatting 的标准流程是：

```text
training images + cameras
  -> train MeshSplatting
  -> checkpoint
  -> render held-out views
```

这个流程有显式 mesh/splat 表示，质量稳定，但训练结束后没有回答三个关键问题：

| 问题 | 基础 MeshSplatting 的状态 | SPCarNet 的思路 |
|---|---|---|
| 局部 residual | 直接保留 checkpoint 的误差 | 在 surface 上记录 residual，并只迁移稳定 residual |
| 几何冗余 | checkpoint geometry 固定 | 用多视角 evidence 判断低风险 triangles |
| 修复风险 | 没有显式风险审计 | train/policy-val gate、tail-risk、fallback/no-op |

核心研究问题可以写成：

```text
Can training-view surface evidence certify where a MeshSplatting checkpoint
can be compacted and where its appearance residuals can be safely repaired?
```

这让 SPCarNet 的研究定位不是“另一个后处理滤镜”，而是：

> 把训练视角从 supervision 升级为 evidence；用 evidence 约束 checkpoint-level geometry edit 和 surface-bound residual repair。

---

## 3. 与原始 MeshSplatting 的区别

| 维度 | MeshSplatting baseline | SPCarNet 当前主方法 |
|---|---|---|
| 输入 | trained MeshSplatting checkpoint | 同一个 checkpoint |
| 训练视角使用方式 | 隐式优化 loss | 显式构建 surface evidence cache |
| 表面 residual | 不单独建模 | 绑定到 face/bin/surface address |
| 几何处理 | 固定 mesh/splats | 删除低风险 triangles |
| 外观修复 | 直接渲染 | guarded Evidence Lumigraph Adapter |
| 风险控制 | 依赖训练收敛 | train/policy-val gate、tail-risk、fallback |
| 输出目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可审计证据 |

通俗版本：

> 原方法是“训练好就交卷”；我们方法是“训练好以后再检查错题本，能安全删的几何删掉，稳定错的颜色修回来，不确定的地方不动”。

---

## 4. 方法总览

完整 pipeline：

```text
clean MeshSplatting checkpoint
  -> render train/policy-val views with surface maps
  -> build surface evidence cache
  -> certify low-risk triangles and compact geometry
  -> transfer stable residuals through surface correspondence
  -> select alpha/branch with train/policy-val risk statistics
  -> fallback to clean/compact render when evidence is weak
  -> evaluate held-out views
```

核心模块：

| 模块 | 做什么 | 为什么必要 |
|---|---|---|
| Surface Evidence Cache | 缓存 residual、visibility、face/bin support、risk | 把训练视角变成可审计证据 |
| Geometry-Safe Compaction | 删除低风险 triangles | 同时追求质量和 compactness |
| Guarded Evidence Lumigraph Adapter | 通过 surface correspondence 迁移稳定 residual | 修复 checkpoint 的局部系统误差 |
| Policy-Val Gate | 选择 alpha/branch，拒绝风险候选 | 防止 test-time 调参和 out-of-trajectory 崩塌 |
| Representation-Level Atlas | 尝试把 repair 写入 checkpoint 表示 | 当前 next-stage research line |

---

## 5. 模块细节

### 5.1 Surface Evidence Cache

Evidence cache 从 train/policy-val render 中保存：

- rendered RGB 和 GT RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、barycentric coordinate、surface/bin address；
- normal、view direction、camera position；
- per-face/per-bin support count；
- residual sign consistency；
- image-level PSNR、SSIM、LPIPS、L1；
- per-view min risk 和 CVaR tail risk。

它的作用是建立一个可查询的 surface evidence table。后续任何 geometry edit 或 residual repair 都必须能回答：

```text
这个 surface 区域在训练/策略验证视角里是否有足够观测？
residual 是否方向稳定？
policy-val 上最差视角是否仍安全？
如果不安全，是否应该 no-op/fallback？
```

### 5.2 Geometry-Safe Triangle Compaction

压缩策略不是单纯追求删得多，而是 quality-first：

```text
remove a triangle only when multi-view evidence marks it as low-risk
```

被保护的区域包括：

- sparse visibility 区域；
- thin structures；
- high residual faces；
- depth/edge/occlusion 不稳定区域；
- policy-val tail-risk 高的局部区域；
- 室内场景中本来 triangles 少、容易破坏细节的区域。

当前 Phase-J 平均删去 `7.6479%` triangles。这个结果可以支持 triangle-count compactness claim，但还不能直接等价为模型大小、显存或 FPS 全面优越；这些需要后续单独测。

### 5.3 Guarded Evidence Lumigraph Adapter

当前 Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。简化公式：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的 render；
- `residual_i = GT_i - Render_i` 只来自 train/policy-val 视角；
- `u_i` 是 target pixel 通过 surface correspondence 找到的训练视角 surface address；
- `w_i(p)` 由 visibility、surface match、support count、risk 共同决定；
- `alpha` 由 train/policy-val evidence 选择；
- 证据不足时自动 fallback/no-op。

这个模块不是普通 2D post-processing，因为它不能直接访问 held-out GT，也不是在图像平面随便滤波；它必须先证明 target pixel 和 train residual 对应到同一三维表面区域。

### 5.4 Train/Policy-Val Gate

公平性原则：

- branch、alpha、support、fallback 只由 train/policy-val evidence 决定；
- held-out test GT 只用于最终报告；
- clean baseline 使用本地 clean `26000/30000` envelope 中 held-out 更强者；
- 不用 train metric 选择 clean baseline，因为 train metric 会天然偏向训练更久的 checkpoint；
- 不用 test metric 为我们方法调参。

当前 Phase-J 分支：

| branch | scene |
|---|---|
| adaptive ELA | `bicycle, flowers, garden, stump, room, counter, kitchen, bonsai` |
| edge fallback | `treehill` |

### 5.5 Representation-Level Residual Atlas

为了把 render-time repair 内化到 checkpoint，我们已经推进了表示级 residual atlas 线：

- v48：auto-support surface atlas；
- v52：capacity-aware policy；
- v56：face-alpha reliability guard；
- v64：fixed auto bin-alpha policy；
- v75：local patch prior；
- v76/v77：policy-val bin-gain hybrid；
- v82/v82b：patch-mixture teacher basis 与 capacity pre-rank；
- v83：patchmix + face-alpha + local-patch hybrid；
- v84：strict selector，把 v82b counter micro-win 与 v64 fallback 固定化；
- v85：SSIM-safe / target-footprint tail-risk certificate；
- v86：anchor-preserving tail-risk selector，防止弱 tail-risk candidate 替换更强 anchor。

这条路线是必要的，但当前还不能作为主结果。原因很明确：v64/v84/v86 的 full9 提升只有微小量级，说明接口、审计和回退机制成熟了，但 residual-field 表达能力还没突破。

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
- 室外场景也全胜，但全图尺度的肉眼差异通常弱于室内，需要 crop/error map；
- `treehill` 走 edge fallback，说明策略在不稳定区域会自动保守；
- 当前压缩率是质量优先的安全压缩，不是极限压缩。

---

## 7. 定性结果与展示建议

### 7.1 主结果图：局部 held-out error reduction

推荐 PPT 主结果页使用：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

每行含义：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 更接近 GT，紫红色表示变差。代表性局部收益：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

建议口径：

> Full-frame 上差异有时不强，但 error map 和 crop 能显示 SPCarNet 在 surface residual、局部纹理和遮挡边界上确实降低了 clean MeshSplatting 的系统误差。

### 7.2 公平 full-frame 图

推荐作为公平性证明或 appendix：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

它展示：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

这张图适合说明比较来自同一 held-out view 和同一 selected clean baseline，但不一定是最强视觉冲击页。

### 7.3 室外细节图

推荐用于回应“室外场景视觉收益不明显”的问题：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail comparison">

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

---

## 8. 与 MeshSplatting 论文表格的关系

当前可引用为背景的数值：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

汇报时建议严谨表述：

- SPCarNet Phase-J 数值高于 MeshSplatting paper table；
- 但正式主 claim 应以**本地同协议 selected-clean MeshSplatting baseline** 为准；
- paper-table 可能受 resolution、mask、split、metric implementation、checkpoint iteration、preprocessing 差异影响；
- 写论文前需要最终 official-style 统一口径复现。

安全口播：

> We outperform our local same-protocol MeshSplatting baseline by a large margin. Paper-table comparison is encouraging, but we treat it as protocol-sensitive background rather than the final fairness claim.

---

## 9. 消融与表示级路线

### 9.1 v64 fixed auto bin-alpha policy

v64 是目前最稳的 fixed representation-level reference：

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

PPT 建议：把这一页放在 appendix 或“ongoing representation-level closure”页。不要把它放在主结果页，否则会削弱故事线。

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

- v4 是真实的 method improvement；
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

必须诚实承认：

- Phase-J 的主 RGB 收益仍主要来自 guarded render-time adapter，不是 fully checkpoint-baked endpoint；
- representation-level residual atlas 目前收益非常小，v64/v84/v86 只能证明接口、selector 和 non-regressive policy；
- 室外场景 full-frame 视觉差异不如室内明显，需要 crop/error map 展示；
- triangle reduction 约 `7.65%`，是质量优先的保守压缩，不是 aggressive compression；
- 还不能声称模型大小、显存、FPS、rate-distortion 全面优于 baseline；
- 与 MeshSplatting paper table 的同口径复现还需要最终确认；
- v87 本轮没有产生有效日志或结果，不能写成方法证据。

导师汇报时建议主动说：

> 当前主结果已经能支持“evidence-certified repair and compaction”的强故事，但还不能声称我们已经完成了最终 paper-level baked representation endpoint。

---

## 13. 建议 PPT 结构

| slide | 标题 | 核心内容 | 建议图/表 |
|---:|---|---|---|
| 1 | Motivation | MeshSplatting 有显式 surface，但 checkpoint 不会自审计 | clean vs GT residual crop |
| 2 | Key Idea | training views are evidence, not only supervision | method schematic |
| 3 | Pipeline | evidence cache -> compact -> repair -> gate -> render | flow chart |
| 4 | Module 1 | geometry-safe triangle compaction | triangle reduction table |
| 5 | Module 2 | guarded Evidence Lumigraph Adapter | residual transfer diagram |
| 6 | Main Quantitative Result | full9 `9/9` scene wins, `244/246` view wins | aggregate table |
| 7 | Per-Scene Result | all scenes improve PSNR/SSIM/LPIPS | 9-scene table |
| 8 | Qualitative Result | local error reduction | `spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 9 | Outdoor Analysis | full-frame subtle, crop/error map clearer | outdoor detail panel |
| 10 | Ablation | v64-v86 representation-level diagnostics | compact ablation table |
| 11 | Boundary | what is complete and what is not | limitation list |
| 12 | Next Step | bake repair into representation + official protocol + rate/FPS | roadmap |

---

## 14. 60 秒中文口播

> 我们现在的方法不是推翻 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 后面加了一层自审计机制。训练视角不只是用来训练 loss，而是被保存成 surface evidence：包括 residual、可见性、face/bin support 和风险统计。这样模型可以判断哪些三角形低风险可以删，哪些表面区域有稳定 residual 可以迁移修复，哪些区域证据不足必须回退。在本地 Mip-NeRF360 full9、相同 evaluator、selected-clean MeshSplatting baseline 下，当前 Phase-J endpoint 做到 9 个场景 PSNR/SSIM/LPIPS 全部严格胜出，246 个 held-out view 中 244 个严格胜出，平均 PSNR 提升 1.33 dB，SSIM 提升 0.0347，LPIPS 降低 0.0634，同时平均删去 7.65% 三角形。需要诚实说明的是，当前最大 RGB 收益仍来自 guarded render-time adapter；我们已经打通表示级 residual atlas 和多个 certificate，但它们目前还是小收益诊断，下一步目标是把这套修复真正内化到 checkpoint 表示里。

## 15. 60 秒英文口播

> Our current method starts from a trained MeshSplatting checkpoint. Instead of directly rendering it, we build a surface evidence cache from train and policy-val views: residuals, visibility, face/bin support, and risk statistics. This evidence lets the checkpoint audit itself. Low-risk triangles are compacted; stable surface residuals are transferred to held-out views through surface correspondence; uncertain regions automatically fall back to the clean checkpoint. On local Mip-NeRF360 full9, under the same evaluator and selected-clean MeshSplatting baseline, the current Phase-J endpoint wins all 9 scenes on PSNR, SSIM, and LPIPS, wins 244 out of 246 held-out views, improves mean PSNR by 1.33 dB, improves SSIM by 0.0347, reduces LPIPS by 0.0634, and removes 7.65% triangles on average. The honest limitation is that the strongest RGB gain is still a guarded render-time adapter. We have built the representation-level residual atlas line and several certificates, but those are currently small-gain diagnostics. The next paper-level milestone is to bake the repair into the checkpoint while preserving the same evidence-certified behavior.

---

## 16. 关键证据路径

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
```

Stage2 shape-prior：

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
```

---

## 17. 当前最终状态

```text
Phase-J local full9 RGB + triangle-count result: strong and presentable.
Same-protocol selected-clean MeshSplatting comparison: complete locally.
Qualitative panels: available and slide-ready.
Representation-level baked endpoint: not complete.
Official MeshSplatting paper protocol reproduction: not fully closed.
Rate/FPS/model-size evaluation: not fully closed.
Stage2 shape-prior: implemented and improved over v3, but gate soft-fail.
```

建议给 mentor 的最终定位：

> SPCarNet 当前最可信的故事是“MeshSplatting checkpoint 的 evidence-certified post-training repair and compaction”。它已经在本地 full9 strong clean baseline 上拿到强 RGB + triangle reduction 结果；真正的论文终局还需要把 guarded repair 更彻底地 bake into representation，并补齐 official protocol 与 rate/FPS/model-size 证据。

---

## 18. 正在运行的后续验证

以下实验属于表示级路线的 follow-up，**尚未写入主结论**。PPT 中最多放在 backup/ongoing slide，不能替代上面的 Phase-J full9 结果。

| run | 目标 | 当前用途 |
|---|---|---|
| `v85_tailrisk_relax075_counter_rerun_20260625` | 放宽 target-footprint tail-risk certificate，检查非空表示级 edit 是否能超过 v84 counter anchor | 已完成；accepted 非空编辑但没有超过 v84/v86 counter anchor，不晋升 |
| `v88_anchor_dominance_tailrisk_counter_20260625` | 组合 target-support pre-rank、patch-mixture teacher basis、local-patch prior、prior-bin hybrid、tail-risk certificate 与 dominance pruning | 尝试挑战 v84/v86 表示级 anchor |

运行证据路径：

```text
/dev/shm/peilincai_spcarnet_v85_tailrisk_relax2_20260625/logs/apply_metrics_counter.log
/dev/shm/peilincai_spcarnet_v88_anchor_dominance_tailrisk_counter_20260625/logs/apply_metrics_counter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_tailrisk_relax075_counter_20260625/
docs/car_model/6-25-v89-L1ProxyBinDominanceGate-Implementation-And-v85Relax-Diagnostic.md
```

汇报边界：

- v85 relaxed tail-risk 已产出 `results.json`、`per_view.json` 和 audit，但结果为 `26.756134 / 0.862126 / 0.251691`，略低于 v84/v86 counter anchor `26.756138 / 0.862126 / 0.251691`；
- v88 如果 counter 单场景有提升，也不能立即替代 full9 主结果，需要 hard triad 和 full9 固定策略复跑；
- v89 已新增 L1-proxy bin-dominance gate，用来解决 prior-bin hybrid 的 residual-MSE 选择目标和最终 SSIM/L1 gate 不完全对齐的问题，但尚未完成中长程验证；
- 当前主 claim 仍以 Phase-J guarded adaptive ELA + geometry-safe compaction 的 full9 结果为准。
