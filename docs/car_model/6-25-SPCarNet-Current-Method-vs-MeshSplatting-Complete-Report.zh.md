# SPCarNet 当前方法与实验完整报告：对比标准 MeshSplatting

日期：2026-06-25  
用途：mentor/PPT 汇报；完整说明当前方法、与标准 MeshSplatting 的方法差异、定量效果、定性渲染图对比和当前边界。

---

## 0. 汇报结论

当前最适合对外汇报的稳定主方法是：

> **SPCarNet Phase-J：在标准 MeshSplatting checkpoint 之后，用训练视角 surface evidence 做自审计，低风险三角形可压缩，稳定 residual 可修复，不可靠区域自动回退。**

在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、selected-clean MeshSplatting baseline 下，当前主 endpoint 的结果是：

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

这里的 `triangle reduction` 指**删去的三角形占比**，不是剩余比例。

一句话讲法：

> 标准 MeshSplatting 是“训练完直接渲染”；SPCarNet 是“训练完后再检查每个 surface：能安全删的 triangle 就删，能稳定修复的 residual 就修，证据不足就回退”。当前 Phase-J 在本地 full9 强 clean baseline 上 RGB 三指标 9/9 场景严格胜出，同时平均删去 `7.65%` triangles。

---

## 1. 标准 MeshSplatting 与 SPCarNet 的方法差异

### 1.1 标准 MeshSplatting

标准 MeshSplatting 的流程是：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh/splat checkpoint
  -> directly render held-out views
```

它的优势是显式 mesh/surface 表示、渲染质量稳定、可以部署。但原始流程不显式回答这些问题：

- 哪些 triangles 对多视角解释贡献低，可以安全删除？
- 哪些 surface 区域在训练视角中反复出现稳定 RGB residual，可以迁移到 held-out view 修复？
- 哪些局部修复只在训练视角有效，会伤害 tail view 或 out-of-trajectory view？
- 如果证据不足，系统是否会自动回退到 clean baseline？

### 1.2 SPCarNet 当前主线

SPCarNet 不推翻 MeshSplatting，而是在已经训练好的 checkpoint 后面增加一层 evidence-driven audit：

```text
clean MeshSplatting checkpoint
  -> train-view surface evidence cache
  -> geometry-safe triangle compaction
  -> guarded residual repair
  -> train/policy-val gate + fallback
  -> held-out render evaluation
```

| 维度 | 标准 MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础模型 | MeshSplatting checkpoint | 继承 MeshSplatting checkpoint |
| 训练视角证据 | 主要隐式用于优化 | 显式缓存 residual、visibility、face/bin support、risk |
| 几何处理 | checkpoint 固定 | 删除低风险 triangles |
| 外观修复 | 直接渲染 checkpoint | surface-bound guarded residual repair |
| 风险控制 | 依赖训练收敛 | train/policy-val gate、risk gate、fallback/no-op |
| 输出目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可解释审计 |

通俗讲：

> MeshSplatting 是一个训练好的模型；SPCarNet 是让这个模型多一个“自检和修复”的后处理系统。它不会在没有证据时盲目改图，而是只在训练视角证据足够稳定时才修改 surface。

---

## 2. 当前方法模块

### 2.1 Surface Evidence Cache

Evidence cache 把训练视角中的监督信号保存成可审计证据，包括：

- rendered RGB 与 GT RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、surface/bin address；
- normal、view direction、camera position；
- per-face/per-bin support count；
- residual sign consistency；
- train/policy-val risk statistics。

它的作用是判断局部修改是否“多视角稳定、证据足够、迁移风险低”。

### 2.2 Geometry-Safe Compaction

SPCarNet 的压缩不是按面积、透明度或单一贡献分数粗暴删三角形，而是 quality-first：

```text
only remove triangles when evidence says the edit is low-risk
```

当前 Phase-J 平均删去 `7.6479%` triangles，同时保持 full9 `9 / 9` 场景 RGB 三指标严格胜出。

### 2.3 Guarded Evidence Lumigraph Adapter

RGB 提升主要来自 surface-bound residual repair。简化公式：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的渲染；
- `residual_i = GT_i - Render_i` 只来自训练视角；
- `w_i(p)` 由 surface correspondence、visibility、support、risk 决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时 fallback/no-op，避免为了修复局部而破坏整图。

关键点：

> 这不是普通图像后处理。SPCarNet 的 residual 必须绑定到 mesh surface correspondence，不能读取 held-out GT，也不能对最终图像做无约束增强。

### 2.4 Train/Policy-Val Gate

公平性边界：

- 方法选择、alpha、support、fallback 只使用 train/policy-val evidence；
- held-out test GT 只用于最终评价；
- clean baseline 使用本地 clean checkpoint envelope 中 held-out 更强者；
- 不使用 test metrics 为我们方法调参。

---

## 3. 定量对比：SPCarNet vs 标准 MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- 标准方法：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR 越高越好，SSIM 越高越好，LPIPS 越低越好。

### 3.1 Full9 主表

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

### 3.2 结果解读

强结论：

- full9 场景级 RGB 三指标 `9 / 9` strict wins；
- held-out view 级 `244 / 246` strict wins；
- mean PSNR 提升 `+1.331084`，mean SSIM 提升 `+0.034702`，mean LPIPS 下降 `-0.063359`；
- 平均删去 `7.6479%` triangles；
- 室内场景提升尤其明显，`bonsai/kitchen/counter/room` 的 PSNR 收益更大；
- 室外场景也全胜，但更适合用局部 crop 与 error map 展示。

---

## 4. 直观渲染图对比

### 4.1 最推荐 PPT 主图：局部 held-out error reduction

这张图最能直接看出 SPCarNet 相对 clean MeshSplatting 的收益。每行包含：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

代表性局部收益：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

推荐讲法：

> 这张图说明 SPCarNet 的收益不只是表格指标，而是在局部高频纹理、光照残差和 surface detail 上降低了 clean MeshSplatting 的系统误差。

### 4.2 公平 full-frame 渲染对比

这张图适合作为“不是只挑 crop”的公平性证据。每行包含：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

推荐讲法：

> 全图肉眼差异不总显著，因为很多收益来自 residual-level local correction；但 error map 和指标显示 ours error 更低。因此 PPT 里应把 full-frame 图作为公平性证明，把局部图作为视觉收益证明。

### 4.3 室外场景细节对比

这张图专门覆盖 `flowers/garden/treehill/bicycle/stump` 等室外场景。

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail error reduction showcase">

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

推荐讲法：

> 室外场景的全图差异确实更难直接看出来，但在叶片、木纹、树皮、长椅条纹等 high-frequency surface 区域，SPCarNet 的局部 error reduction 更稳定。

---

## 5. 与 MeshSplatting 论文表格的关系

当前可用于汇报背景的表：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| Local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

解释边界：

- 本地 Phase-J 数值高于 MeshSplatting paper table；
- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准；
- paper table 可能存在 resolution、mask、split、preprocessing、evaluator、checkpoint iteration 差异；
- 我们本地 clean baseline 不是故意挑弱，而是在 clean `26000/30000` envelope 中选择 held-out 更强 row。

推荐汇报说法：

> 与原论文表格相比，SPCarNet 当前数值更高；但最严格、最公平的主结论应该写成本地同协议强 clean baseline 对比，因为这样 evaluator、split、checkpoint 和结果路径全部可追溯。

---

## 6. 表示级路线与当前边界

Phase-J 主结果之外，我们也推进了把 residual repair 内化到 persistent surface representation 的路线。它更接近论文终局，但目前不能替代 Phase-J headline。

### 6.1 v64 fixed auto bin-alpha reference

v64 是目前最稳的 fixed representation-level reference：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | `9` | `1` | `9` | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | `9` | `2` | `9` | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | `9` | `7` | `8` | `+0.002255970` | `+0.000038081` | `-0.000093445` |

解释：

> v64 证明 representation-level residual atlas 可以被固定 train/policy-val 策略自动选择和回退，但收益非常小，不能替代 Phase-J 主结果。

### 6.2 v82/v83/v84/v85 边界

| line | scope | result | status |
|---|---|---:|---|
| v82 patch-mixture teacher basis | counter probe | `26.753459930 / 0.862114668 / 0.251868337` | all three metrics below anchor, not promoted |
| v82b capacity pre-rank + face-alpha | counter probe | `26.756137848 / 0.862126350 / 0.251690656` | counter-only strict micro-win, but hard-triad fails |
| v83 patchmix + face-alpha + local-patch hybrid | counter probe | `26.756147385 / 0.862125337 / 0.251688808` | PSNR/LPIPS improve, SSIM regresses, not promoted |
| v84 strict v82 selector + v64 fallback | full9 materialized selector | vs v64 mean `+0.000000848 / +0.000000013 / -0.000000079` | report-only; useful engineering closure, not paper breakthrough |
| v85 target-footprint/tail-risk certificate | counter probe | `26.756134033 / 0.862126231 / 0.251691371` | real accepted edit, anchor-level micro-tie, not promoted |

结论：

> 当前 strongest safe endpoint 仍是 Phase-J。v64/v84/v85 说明表示级接口、自动策略、target footprint 和 tail-risk gating 已经逐步完整，但距离“把 Phase-J 的强视觉收益完全内化到 persistent representation”还需要真正更强的 residual-field 建模突破。

---

## 7. 适合 PPT 的故事线

推荐按 6 页展开：

1. **Problem**：标准 MeshSplatting 渲染强，但不知道哪些 surface 可删、哪些 residual 可修、哪些修改危险。
2. **Idea**：把训练视角 supervision 变成 surface evidence，让模型训练后可以自审计。
3. **Method**：低风险 triangles 删除；稳定 residual 绑定 surface 后迁移修复；证据不足 fallback。
4. **Quantitative**：full9 selected-clean baseline 上 `9 / 9` scene strict wins，mean `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS，mean triangle reduction `7.65%`。
5. **Qualitative**：先放局部 error reduction 主图，再放 full-frame fairness 图，最后放 outdoor detail 图。
6. **Boundary**：当前强收益仍主要来自 guarded render-time adapter；representation-level 内化路线是下一步研究重点。

---

## 8. 证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

图片选择规则：

```text
assets/spcarnet_phasej_where_it_helps_selection_20260622.json
assets/spcarnet_m360_full9_gallery_selection.json
assets/spcarnet_m360_outdoor_detail_selection.json
```

表示级支线证据：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_triad_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
```

---

## 9. 最短汇报版

> SPCarNet 将 MeshSplatting 从“训练完直接渲染”的静态 checkpoint，升级成“训练证据驱动的可压缩、可修复、可回退系统”。当前 Phase-J 在 Mip-NeRF360 selected full9 上相对本地强 clean MeshSplatting baseline 实现 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，`244/246` held-out views 严格胜出，并平均删去 `7.65%` triangles；定性图显示 SPCarNet 在局部高频纹理和 surface residual 区域显著降低 error。最诚实的边界是，强收益目前仍主要来自 guarded render-time adapter，representation-level 内化路线还需要继续突破。
