# SPCarNet 当前方法完整技术报告（Mentor/PPT 集成版）

日期：2026-06-24  
用途：基于当前仓库状态制作 mentor 汇报 PPT；强调已经闭合的证据、当前方法的研究价值、相对基础 MeshSplatting 的改进，以及仍未闭合的论文级短板。  
建议 PPT 标题：**SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting**

---

## 0. 一页结论

当前最适合对外汇报的主方法是：

```text
MeshSplatting checkpoint
  + geometry-safe triangle compaction
  + train-evidence guarded Evidence Lumigraph Adapter
  + train/policy-val gate and fallback
```

一句话：

> SPCarNet 不重新发明 MeshSplatting，而是让训练好的 MeshSplatting checkpoint 具备自审计能力：用训练视角 surface evidence 判断哪些 triangles 可以低风险压缩，哪些 surface residual 可以迁移修复，证据不足时自动回退。

当前最强已验证 endpoint 是 **Phase-J guarded adaptive ELA + geometry-safe compaction**。在本地 Mip-NeRF360 full9、selected-clean MeshSplatting baseline、相同 split 和 evaluator 下：

| 指标 | SPCarNet Phase-J vs clean MeshSplatting |
|---|---:|
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| held-out view strict wins | `244 / 246` |
| mean PSNR | clean `25.1517` -> ours `26.4828` |
| mean SSIM | clean `0.7490` -> ours `0.7837` |
| mean LPIPS | clean `0.2876` -> ours `0.2243` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |

这里的 `triangle reduction` 指删去的三角形比例，不是剩余比例。

当前状态边界：

- **本地 full9 RGB + triangle reduction 闭环已经比较强**：相对本地 selected-clean MeshSplatting baseline 全场景三指标胜出，同时平均删面约 `7.65%`。
- **论文终局仍未完成**：当前最大 RGB 收益主要来自 render-time guarded ELA，尚未完全 baked into checkpoint representation。
- **表示级路线已有真实接口和消融**：v64/v84/v86 是最稳 fixed representation-level policy/selector 线，但收益只有 `1e-4` 到 `1e-6` 量级；v86 已把 v85 tail-risk 变成 anchor-preserving guardrail，证书路径可运行但没有形成可晋升提升。
- **汇报建议**：主讲 Phase-J 的自审计修复与压缩；把 v64-v86 讲成“把 render-time 修复内化到 representation 的 ongoing research line”，不要把它包装成已经完成的 headline。

---

## 1. 研究问题

基础 MeshSplatting 的流程是：

```text
training images + cameras
  -> train MeshSplatting
  -> checkpoint
  -> directly render held-out views
```

它有显式 surface/mesh 表示，渲染质量稳定，但训练完成后没有显式回答三个问题：

| 问题 | 现象 | SPCarNet 的对应设计 |
|---|---|---|
| 局部外观 residual | 高频纹理、遮挡边界、室内物体表面仍有系统误差 | surface-bound residual repair |
| 几何/拓扑冗余 | 一些 triangles 对多视角解释贡献低 | evidence-certified triangle compaction |
| 修复泛化风险 | 直接把训练 residual 搬到新视角可能伤害 tail view | train/policy-val gate、tail risk、fallback/no-op |

核心问题可以表述为：

```text
Can training-view surface evidence certify where a MeshSplatting checkpoint can be
compacted and where its appearance residuals can be safely repaired?
```

这让 SPCarNet 的定位更像 **post-training self-auditing layer**：不是替代原始 MeshSplatting，而是把一个训练好的 checkpoint 变成可审计、可压缩、可修复的 surface system。

---

## 2. 方法总览

SPCarNet 当前主线由五层组成：

```text
1. Train/policy-val surface evidence cache
2. Geometry-safe triangle compaction
3. Guarded Evidence Lumigraph Adapter
4. Train-only policy gate and fallback
5. Representation-level residual atlas / shape-prior follow-up
```

完整流程：

```text
clean MeshSplatting checkpoint
  -> render train/policy-val views with surface maps
  -> build surface evidence cache
  -> certify low-risk triangles and compact geometry
  -> transfer stable train residuals through surface correspondence
  -> gate with train/policy-val risk statistics
  -> render held-out views
```

与基础 MeshSplatting 的差异：

| 维度 | 基础 MeshSplatting | SPCarNet 当前方法 |
|---|---|---|
| 基础表示 | trained mesh/splat checkpoint | 继承同一 checkpoint |
| train evidence 使用 | 主要隐式用于优化 | 显式缓存 residual/support/risk/visibility |
| 几何处理 | checkpoint 固定 | 低风险 triangles 删除 |
| 外观修复 | 直接渲染 checkpoint | surface-bound guarded residual transfer |
| 风险控制 | 依赖训练收敛 | policy-val gate、tail risk、fallback |
| 当前主输出 | clean render | compact checkpoint + repaired render |

---

## 3. 模块细节

### 3.1 Surface Evidence Cache

Evidence cache 是整个方法的“观测记录”。它从 train/policy-val render 中保存：

- rendered RGB 和 GT RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、barycentric coordinate、surface/bin address；
- normal、view direction、camera position；
- per-face/per-bin support count；
- residual sign consistency；
- image-level PSNR、SSIM、LPIPS、L1；
- per-view min risk 和 CVaR tail risk。

它的作用是把训练视角从“只用于训练 loss”升级为“可审计证据”。后续每一次删三角形或修 residual，都必须能在这个证据缓存里找到多视角 support 和风险统计。

### 3.2 Geometry-Safe Triangle Compaction

压缩策略不是简单删小三角形，也不是为了追求最大压缩率，而是 quality-first：

```text
delete a triangle only when evidence says the edit is low-risk
```

保护对象包括：

- sparse visibility 区域；
- thin structures；
- residual 高或边界不稳定的 faces；
- policy-val 风险高的局部区域；
- 室内场景中本来 triangles 少、容易破坏可见细节的区域。

当前主结果平均删去 `7.6479%` triangles。这个数值能支持 triangle-count compactness claim，但还不能直接等价为完整模型大小、属性存储、显存或 FPS 收益。

### 3.3 Guarded Evidence Lumigraph Adapter

当前 Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。简化形式：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的渲染；
- `residual_i = GT_i - Render_i` 只来自 train/policy-val 视角；
- `u_i` 是 target pixel 通过 surface correspondence 对应到训练视角的 surface address；
- `w_i(p)` 由 visibility、surface match、support count 和 risk 决定；
- `alpha` 由 train/policy-val evidence 选择；
- 证据不足时自动 fallback 或 no-op。

通俗讲：

> 如果同一个 surface 区域在多个训练视角里稳定错同一个方向，我们把它看作可迁移 residual；如果证据少、视角变化大或风险高，就不修。

这不是普通 2D 图像后处理，因为 residual 必须通过 face/bin/surface address 绑定到三维表面，不能读取 held-out GT。

### 3.4 Train/Policy-Val Gate

公平性规则：

- 方法 branch、alpha、support、fallback 只使用 train/policy-val evidence；
- held-out test GT 只用于最终 evaluation；
- clean baseline 用本地 clean `26000/30000` envelope 中 held-out 更强者，避免拿弱 baseline 做比较；
- 不用 train metric 选择 clean baseline，因为 train metric 会天然偏向训练更久；
- 不用 held-out test metric 为我们方法调参。

当前 Phase-J branch：

| branch | scene |
|---|---|
| adaptive ELA | `bicycle, flowers, garden, stump, room, counter, kitchen, bonsai` |
| edge fallback | `treehill` |

### 3.5 Representation-Level Residual Atlas

Phase-J 仍包含 render-time adapter。为了走向论文终局，我们已经搭建了把 residual repair 写入 persistent representation 的路线：

- v48：auto-support surface atlas；
- v52：capacity-aware policy；
- v56：face-alpha reliability guard；
- v64：fixed auto bin-alpha policy；
- v75：local patch prior；
- v76/v77：policy-val bin-gain hybrid / strict hybrid；
- v78/v78b：target-footprint certificate，负结果诊断；
- v79：复现 v56/v64 strong anchor；
- v80：face-alpha + local-patch + bin-gain hybrid near-tie；
- v82/v82b：patch-mixture teacher basis 与 capacity pre-rank；
- v83：patchmix + face-alpha + local-patch hybrid，PSNR/LPIPS 正向但 SSIM 微退；
- v84：strict selector，把 v82b counter micro-win 与 v64 fallback 固定化；
- v85：SSIM-safe / target-footprint tail-risk certificate；前者拒绝不安全候选并 fallback，后者接受非空编辑但 held-out test 仅与 v56/v64/v79 anchor 基本持平且低于 v84 counter row；
- v86：anchor-preserving tail-risk selector，只有当 v85 candidate 在 train/policy-val audit 上支配 v84 anchor 时才晋升；当前 full9 保留 v84，`9 / 9` non-regressive/tie vs v84。

这条线的价值是：它不是单纯调参，而是在把“可审计 residual 修复”逐步内化成 checkpoint-level surface representation。不过当前收益仍显著弱于 Phase-J。

---

## 4. 定量结果：SPCarNet vs 基础 MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- baseline：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR/SSIM 越高越好，LPIPS 越低越好。

### 4.1 Full9 主表

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

### 4.2 Aggregate

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
- 室外场景也全胜，但全图肉眼差异较弱，更适合用 crop + error map 展示；
- `treehill` 使用 edge fallback，说明策略能在不稳定区域自动选择更保守分支；
- 当前 triangle reduction 是质量优先的保守压缩，不是极限压缩。

---

## 5. 定性结果与 PPT 展示建议

### 5.1 主结果图：局部 held-out error reduction

推荐把这张图放在主结果页：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

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

PPT 讲法：

> Full-frame 肉眼差异有时不强，但 error map 和 crop 能显示我们确实在表面细节、遮挡边界和高频纹理上降低了 clean MeshSplatting 的系统误差。

### 5.2 公平 full-frame 图

推荐作为公平性证明：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

它展示：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

建议放在 appendix 或主结果后的 backup 页，说明我们不是只挑局部 crop。

### 5.3 室外细节图

推荐用于回应“室外场景视觉收益不明显”的问题：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

讲法：

> 室外场景的全局结构通常已经很好，因此收益更集中在细节纹理和局部 residual 上；这也是为什么 crop/error map 比纯 full-frame 更能展示方法优势。

---

## 6. 与 MeshSplatting 论文表格的关系

当前可作为背景参考的数值：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

汇报时的严谨说法：

- 本地 SPCarNet Phase-J 高于 MeshSplatting paper table；
- 但主 claim 应以本地同协议 selected-clean MeshSplatting baseline 为准；
- paper table 可能受 resolution、mask、split、metric implementation、checkpoint iteration、preprocessing 差异影响；
- 后续写论文前必须重跑完全统一口径的 official-style comparison。

一句安全讲法：

> We already outperform our local same-protocol MeshSplatting baseline by a large margin; paper-table comparison is encouraging but still treated as protocol-sensitive background, not the final fairness claim.

---

## 7. 表示级路线与消融

### 7.1 v64 fixed auto bin-alpha policy

v64 是目前最稳的 fixed representation-level reference：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | `9` | `1` | `9` | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | `9` | `2` | `9` | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | `9` | `7` | `8` | `+0.002255970` | `+0.000038081` | `-0.000093445` |

结论：接口和自动策略有效，但收益还远不够作为 headline。

### 7.2 v82b / v83 / v84 latest diagnostics

| probe | scope | PSNR | SSIM | LPIPS | verdict |
|---|---|---:|---:|---:|---|
| v56/v64/v79 counter anchor | `counter` | `26.756130219` | `0.862126231` | `0.251691371` | strong anchor |
| v82b capacity-prerank + face-alpha | `counter` | `26.756137848` | `0.862126350` | `0.251690656` | counter micro-win |
| v83 patchmix + facealpha + localpatch | `counter` | `26.756147385` | `0.862125337` | `0.251688808` | PSNR/LPIPS up, SSIM down |
| v84 strict selector | full9 | v64 + `8.48e-7` mean PSNR | v64 + `1.3e-8` mean SSIM | v64 - `7.9e-8` mean LPIPS | report-only |

关键解释：

- v82b 说明 support-capacity pre-rank 有真实信号，但 hard-triad 的 `kitchen/bonsai` 没有稳定超过 v64；
- v83 说明更高 residual capacity 能提升 PSNR/LPIPS，但 SSIM certificate 不够强；
- v84 固定了“证据不足就 fallback v64”的保守 selector，适合做 ablation hygiene，不适合作为论文突破；
- v85 已完成更严格的 SSIM-safe / target-footprint tail-risk certificate 诊断：SSIM-safe fallback 低于 anchor；tail-risk 接受非空编辑但仅达到 v56/v64/v79 anchor 级 micro-tie，且低于 v84 counter row。
- v86 已完成 anchor-preserving selector：当前 v85 counter candidate 因 train/policy-val SSIM/L1 audit 不支配 v84/v82b anchor 被拒绝，full9 保持 `9 / 9` non-regressive/tie vs v84。

### 7.3 Stage2 shape-prior line

Stage2 v4 normal-band autodecoder 是另一条真实训练管线改动，用 surface-normal band supervision 约束 occupancy boundary：

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

结论：shape prior 有真实进步，但 shell IoU 退化，尚未过强几何 gate；可以作为“方法外延/未来方向”，不要当当前主 claim。

---

## 8. 为什么这是研究工作，而不是工程后处理

可强调四点：

1. **surface-addressed evidence**  
   修复不是直接在 2D 图像上滤波，而是把 residual 绑定到 mesh face/bin/surface address。

2. **certified edit policy**  
   每个删面或修复都依赖 support、visibility、risk、policy-val gate 和 fallback，不是手工修图。

3. **quality + compactness joint objective**  
   方法同时优化 RGB 指标和 triangle count，目标不是单纯提升 PSNR。

4. **honest failure handling**  
   v75-v86 中大量 negative diagnostics / guardrail diagnostics 被记录并拒绝晋升，说明 pipeline 能识别无效表示级改动，而不是只保留成功样本。

建议 PPT story：

```text
MeshSplatting gives us an explicit surface.
SPCarNet asks whether that surface can audit itself.
Training views are not only supervision; they are evidence.
Evidence tells us what can be compacted, repaired, or safely left untouched.
```

---

## 9. 当前弱点与风险

必须诚实承认：

- Phase-J 的主 RGB 收益仍主要来自 render-time adapter，不是完全 checkpoint-baked endpoint；
- representation-level residual atlas 目前收益非常小，v64/v84 只能证明接口和 non-regressive policy；
- 室外场景 full-frame 视觉差异不如室内明显，需要 crop/error map 展示；
- triangle reduction 约 `7.65%`，是质量优先的保守压缩，还不是 aggressive compression；
- 与 MeshSplatting paper table 的同口径复现还需要最终确认；
- v85/v86 已跑完但不能作为成功结果：它们证明 safety certificate 和 anchor-preserving selector 可运行，尚未证明 representation-level 指标有实质提升；
- 当前不能声称模型大小、FPS、显存、rate-distortion 全面优于 baseline，除非后续补齐这些指标。

---

## 10. 建议 PPT 结构

### Slide 1: Motivation

标题：**Can MeshSplatting checkpoints audit themselves?**

要点：

- MeshSplatting 已有显式 surface；
- 但 clean checkpoint 仍有 residual 和冗余 geometry；
- 训练视角 evidence 可以决定哪里可删、哪里可修。

### Slide 2: Method Overview

放流程图：

```text
MeshSplatting checkpoint
 -> surface evidence cache
 -> compact
 -> repair
 -> gate/fallback
 -> held-out render
```

### Slide 3: Core Mechanism

讲两个核心：

- geometry-safe triangle compaction；
- guarded Evidence Lumigraph Adapter。

### Slide 4: Main Quantitative Result

放 full9 aggregate：

```text
9/9 scene-level strict wins
244/246 per-view strict wins
+1.331 PSNR, +0.0347 SSIM, -0.0634 LPIPS
7.65% triangle reduction
```

### Slide 5: Per-Scene Table

放 full9 主表或精简成 9 行 delta 表。

### Slide 6: Qualitative Main Figure

放：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

### Slide 7: Outdoor Detail

放：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

说明室外收益更局部，需要 error map/crop。

### Slide 8: Ablation / Ongoing Representation-Level Work

放 v64/v82/v83/v84 表，说明：

- fixed representation-level policy 已经接入；
- 当前还没有 Phase-J 那样强；
- 下一步是让 tail-risk certificate 只作为安全过滤器，并继续把更强 residual atlas bake 进 checkpoint。

### Slide 9: Limitations and Next Steps

三句话：

- main RGB loop is strong;
- representation-level endpoint is not yet final;
- next milestone is baked residual atlas + unified MeshSplatting paper protocol + rate-distortion/FPS metrics.

---

## 11. 关键证据路径

主报告和 README：

```text
README.zh.md
README.md
docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md
docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CleanCurrent.zh.md
docs/car_model/6-24-SPCarNet-vs-MeshSplatting-Complete-Report-With-Visuals.zh.md
```

主定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

当前 evidence manifest：

```text
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
```

Phase-J summary：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

Representation-level logs：

```text
docs/car_model/6-24-v64-FixedAutoBinAlphaPolicy-Log.md
docs/car_model/6-24-v82b-CapacityPrerankFaceAlpha-Log.md
docs/car_model/6-24-v83-PatchMixFaceAlphaLocalPatch-Hybrid-Log.md
docs/car_model/6-24-v84-StrictCapacitySelector-Log.md
docs/car_model/6-24-v85-TargetFootprintTailRiskCertificate-Log.md
docs/car_model/6-24-v86-AnchorPreservingTailRiskSelector-Log.md
```

Current v84 selected summary：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
```

---

## 12. Mentor 汇报时可直接使用的 60 秒口播

> Our current method starts from a trained MeshSplatting checkpoint. Instead of directly rendering it, we build a surface evidence cache from train and policy-val views: residuals, visibility, face/bin support, and risk statistics. This evidence lets the checkpoint audit itself. Low-risk triangles are compacted; stable surface residuals are transferred to held-out views through surface correspondence; uncertain regions automatically fall back to the clean checkpoint. On local Mip-NeRF360 full9, under the same evaluator and selected-clean MeshSplatting baseline, the current Phase-J endpoint wins all 9 scenes on PSNR, SSIM, and LPIPS, wins 244 out of 246 held-out views, improves mean PSNR by 1.33 dB, improves SSIM by 0.0347, reduces LPIPS by 0.0634, and removes 7.65% triangles on average. The honest limitation is that the strongest RGB gain is still a guarded render-time adapter. We have built the representation-level residual atlas line and several certificates, but those are currently small-gain diagnostics. The next paper-level milestone is to bake the repair into the checkpoint while preserving the same evidence-certified behavior.

中文口播：

> 我们现在的方法不是推翻 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 后面加了一层自审计机制。训练视角不只是用来训练 loss，而是被保存成 surface evidence：包括 residual、可见性、face/bin support 和风险统计。这样模型可以判断哪些三角形低风险可以删，哪些表面区域有稳定 residual 可以迁移修复，哪些区域证据不足必须回退。在本地 Mip-NeRF360 full9、相同 evaluator、selected-clean MeshSplatting baseline 下，当前 Phase-J endpoint 做到 9 个场景 PSNR/SSIM/LPIPS 全部严格胜出，246 个 held-out view 中 244 个严格胜出，平均 PSNR 提升 1.33 dB，SSIM 提升 0.0347，LPIPS 降低 0.0634，同时平均删去 7.65% 三角形。需要诚实说明的是，当前最大 RGB 收益仍来自 guarded render-time adapter；我们已经打通表示级 residual atlas 和多个 certificate，但它们目前还是小收益诊断，下一步目标是把这套修复真正内化到 checkpoint 表示里。

---

## 13. 当前最终状态

```text
Phase-J local full9 RGB + triangle-count report: strong and presentable.
Representation-level paper endpoint: not complete.
Unified official MeshSplatting paper protocol: not complete.
Rate/FPS/model-size closure: not complete.
v85 SSIM/tail-risk certificate: completed diagnostic / not promoted.
v86 anchor-preserving tail-risk selector: completed guardrail / not promoted.
```

最终汇报建议：

> 把当前工作定位为“MeshSplatting checkpoint 的 evidence-certified post-training repair and compaction”。这是目前最稳、最有说服力、也最不容易被 fairness challenge 击穿的故事线。
