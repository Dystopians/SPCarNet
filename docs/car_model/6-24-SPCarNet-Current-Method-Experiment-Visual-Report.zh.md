# SPCarNet 当前方法与实验报告（含定性图）

日期：2026-06-24  
用途：给 mentor/PPT 汇报当前方法、与标准 MeshSplatting 的差异、定量收益、定性渲染对比和现阶段短板。  
推荐主线：**SPCarNet = Self-auditing MeshSplatting，用训练视角 surface evidence 做安全压缩与残差修复。**

当前口径说明：本文的主结果固定为已经闭合的 **Phase-J guarded adaptive ELA + geometry-safe compaction**。v64-v78 是表示级内化诊断线，其中 v78 target-footprint certificate 与 v78b fixed-code formal rerun 均已完成但为负结果；v78b counter 指标为 `26.753529 / 0.862111 / 0.251881`，低于 v75 与 v64/v56 reference，未进入本文主结论。

---

## 1. 一页结论

当前最适合作为汇报主结果的是：

> **Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**

它不是重新训练一个完全不同的 renderer，而是在已经训练好的 MeshSplatting checkpoint 上增加一层 surface evidence audit：

```text
clean MeshSplatting checkpoint
  -> train-view surface evidence cache
  -> geometry-safe triangle compaction
  -> guarded residual repair
  -> train-only policy gate / fallback
  -> held-out render evaluation
```

当前主结果，在本地 same-protocol Mip-NeRF360 full9、selected clean MeshSplatting baseline 下：

| 指标 | 结果 |
|---|---:|
| 场景数 | `9 / 9` |
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| per-view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

一句话汇报：

> 与标准 MeshSplatting 相比，SPCarNet 让模型先“审计自己”：哪些 surface 区域有稳定残差就修，哪些 triangles 风险低就删，证据不足就回退。当前 Phase-J 在 full9 selected-clean baseline 上 RGB 三指标全胜，同时平均删去 `7.65%` triangles。

---

## 2. 与标准 MeshSplatting 的方法差异

### 2.1 标准 MeshSplatting

标准流程是：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh/splat checkpoint
  -> directly render held-out views
```

它的核心优点是显式 mesh/surface 表示，高质量、可渲染、可部署。但它本身不显式回答三个问题：

- 哪些三角形对多视角解释贡献很低，可以安全压缩？
- 哪些 surface 区域在训练视角中反复出现稳定 RGB residual，可以迁移到 held-out view 修复？
- 哪些局部修复在 view-tail 或 out-of-trajectory 情况下风险太高，应该自动回退？

### 2.2 SPCarNet

SPCarNet 在 MeshSplatting 之后增加 evidence-driven audit：

| 维度 | 标准 MeshSplatting | SPCarNet 当前 Phase-J |
|---|---|---|
| 基础表示 | trained mesh/splat checkpoint | 继承 MeshSplatting checkpoint |
| 是否审计 train residual | 否 | 是，构建 face/bin/support/risk evidence |
| 几何压缩 | checkpoint 固定 | 删除低风险 triangles |
| 外观修复 | 直接渲染 checkpoint | guarded residual transfer |
| 失败保护 | 依赖训练收敛 | train/policy-val gate 和 fallback |
| 最强当前收益 | baseline render | compact render + evidence-certified repair |

通俗解释：

> MeshSplatting 是“训练完就直接渲染”。SPCarNet 是“训练完后，再用训练视角检查每个 surface 是否可靠：能删就删，能修就修，不确定就不动”。

---

## 3. 方法模块

### 3.1 Surface Evidence Cache

Evidence cache 保存训练视角中的：

- rendered RGB 和 GT RGB；
- residual：`GT - Render`；
- depth、alpha、visibility；
- face id、surface/bin address；
- normal、view direction、camera position；
- per-face/per-bin support；
- residual sign consistency；
- per-view risk、tail risk、policy-val metric。

它的作用是把训练数据从“优化参数的监督信号”变成“审计 surface 是否可修、可删、可回退的证据”。

### 3.2 Geometry-Safe Compaction

压缩原则是 quality-first：

```text
only remove triangles when multi-view evidence says the edit is low-risk
```

当前报告里的 `triangle reduction` 指**删去的 triangles 占比**，不是剩余比例。Phase-J 平均删去 `7.6479%` triangles。

### 3.3 Guarded Evidence Lumigraph Adapter

Phase-J 的 RGB 主要收益来自 residual repair：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后的 checkpoint render；
- `residual_i = GT_i - Render_i` 来自训练视角；
- `w_i(p)` 由 surface correspondence、support、visibility、risk 决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时自动 fallback 或 no-op。

这不是普通图像增强。修复必须绑定到真实 surface evidence，而不是对最终图像做无约束后处理。

### 3.4 Train-Only Policy Gate

公平性规则：

- SPCarNet 的 branch、alpha、support、fallback 只使用 train/policy-val evidence；
- held-out test GT 只用于最终报告；
- selected clean baseline 使用 held-out test metrics 选择更强 clean checkpoint，是为了不低估标准方法；
- 不用 test metrics 来选择我们方法参数。

---

## 4. 定量对比：SPCarNet vs 标准 MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- 标准方法：本地 clean MeshSplatting `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR 越高越好，SSIM 越高越好，LPIPS 越低越好。

### 4.1 Full9 主表

| scene | selected clean PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | delta | triangle removed |
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

Aggregate：

| 指标 | SPCarNet vs selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |

结论：

> 在本地 selected-clean full9 口径上，SPCarNet 当前主 endpoint 明确优于标准 MeshSplatting：RGB 三指标 9/9 场景严格胜出，并且同时减少 triangles。

### 4.2 与 MeshSplatting paper table 的关系

当前最严谨的说法是：

- 主 claim 用本地 same-protocol selected-clean MeshSplatting，因为 evaluator、split、checkpoint、结果路径都在仓库中可追溯；
- 与 MeshSplatting 原论文表格的对比可以作为辅助说明，但最终论文前必须重新确认 split、mask、metric implementation、iteration 和 checkpoint selection 完全一致。

已有早期 paper-table audit 摘要：

| 评估口径 | 结果 |
|---|---:|
| RGB wins vs selected clean MeshSplatting | `9 / 9` |
| mean dPSNR vs selected clean | `+0.497941` |
| mean dSSIM vs selected clean | `+0.015755` |
| mean dLPIPS vs selected clean | `-0.023373` |
| mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| mean triangle reduction | `5.7632%` |

建议汇报口径：

> 本地主表最可靠；paper table 对齐结果只能作为辅助，不作为最终顶会主表，直到完成完全同口径复现。

---

## 5. 定性对比图

### 5.1 最推荐展示图：局部 held-out error reduction

这张图最能直观看出 SPCarNet 相对 clean MeshSplatting 的优势。每行是一个 held-out view/crop：

```text
GT crop / Clean MeshSplatting / SPCarNet / Error reduction
```

绿色表示 SPCarNet 比 clean MeshSplatting 更接近 GT，紫红色表示变差。

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

这张图适合放 PPT 主结果页，因为它不只展示最终 render，还展示了 error reduction。代表性局部结果：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

### 5.2 公平全图对比：同一 held-out view

这张图适合说明我们不是只挑局部裁剪。每行有：

```text
GT / Clean MeshSplatting / Ours / Clean error / Ours error
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

注意：全图对比的肉眼差异通常比局部 error map 更弱，这是当前方法展示上的自然限制。因为很多提升来自 residual-level 细节修复，直接缩到全图时不总显著。因此 PPT 中建议把这张作为“公平性证明”，把上一张局部 error-reduction 作为“视觉收益证明”。

### 5.3 室外场景细节对比

这张图专门回应 outdoor scenes：flowers、garden、treehill、bicycle、stump 的局部 crop 都展示了 clean MeshSplatting 的纹理/细节残差，以及 SPCarNet 的局部误差下降。

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail error reduction showcase">

建议讲法：

> 室外场景全图差异确实不总明显，但在叶片、木纹、树干、金属椅条等 high-frequency surface 区域，SPCarNet 的 error map 显示出更稳定的局部误差下降。

---

## 6. 表示级路线和对象级先验现状

当前主结果是 Phase-J render-time guarded ELA portfolio。我们也推进了两条更接近论文终局方法的路线，但它们现在还不是 headline。

### 6.1 Surface residual atlas / v64-v78

目标：把 Phase-J 的 residual repair 从 render-time adapter 内化到 persistent surface representation 中。

当前最稳固定 representation-level policy 是 v64 fixed auto bin-alpha：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | 9 | 2 | 9 | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | 9 | 7 | 8 | `+0.002255970` | `+0.000038081` | `-0.000093445` |

v78 最新诊断：

| line | counter PSNR | counter SSIM | counter LPIPS | status |
|---|---:|---:|---:|---|
| v64 / v56 reference | `26.756130` | `0.862126` | `0.251691` | best fixed representation-level reference |
| v75 local patch prior, zero-blend selected | `26.753996` | `0.862119` | `0.251853` | not promoted |
| v78 target-footprint certificate | `26.753529` | `0.862111` | `0.251881` | negative diagnostic |

结论：

> v64-v78 证明接口、policy、negative diagnostics 都已经很完整；v78 进一步证明 target-view footprint certificate 可以被公平接入 train/eval pipeline，但仍不能突破 low-capacity residual atlas 的瓶颈。persistent residual atlas 的收益仍远小于 Phase-J。它目前是 ablation / next-step evidence，不是主结果。

### 6.2 Object-level shape prior / Stage 2 v4

Stage 2 v4 normal-band autodecoder 是真实 train/eval pipeline 改动：

```text
x_inner = x_surface - epsilon * normal -> occupied
x_outer = x_surface + epsilon * normal -> free
```

它用 surface-normal band supervision 强化 occupancy boundary，让 Marching Cubes 的 `0.5` crossing 更接近真实表面。

Full-val MAP-fit 当前最好证据：

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | delta |
|---|---:|---:|---:|
| extraction | `206 / 206` | `206 / 206` | tie |
| recon chamfer | `0.0698447353` | `0.0607328202` | `-0.0091119151` |
| hidden chamfer | `0.1023846301` | `0.0933915632` | `-0.0089930669` |
| filled IoU | `0.5531548112` | `0.5683319216` | `+0.0151771104` |
| shell IoU | `0.9112784961` | `0.8783071888` | `-0.0329713073` |

状态：

- v4 full training 已完成：`69300` steps / `300` epochs；
- W&B train run：`dysg8508`；
- W&B epoch50 eval run：`4wu9w305`；
- final checkpoint full206 MAP-fit eval 已完成，W&B run `q1jjwvdm`，结果路径：`outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json`；
- final checkpoint 不如 epoch50：recon chamfer `0.0655826944`，filled IoU `0.5314717742`，shell IoU `0.8563237802`，normal consistency `0.6890807638`；
- 因此当前 Stage2 v4 best documented checkpoint 仍是 epoch50，而 final checkpoint 是后期训练退化的补充诊断；
- Stage 2 仍未过原 gate：`chamfer <= 0.05`、`filled IoU >= 0.92`。

汇报口径：

> 对象级 shape prior 已经有真实改进，但还不是能支撑主论文 claim 的强几何先验。

---

## 7. 当前短板

1. **最强结果仍是 render-time endpoint**  
   Phase-J 很强，但还不是 fully baked representation-level 方法。

2. **全图视觉差异不总显著**  
   必须展示局部 crop、error map 和 local MAE drop，不能只放 full-frame render。

3. **表示级内化收益还小**  
   v64-v78 的接口完整，但 effect size 仍远小于 Phase-J。v78 已经排除了“再加一层 target-footprint/bin gate 就能解决”的简单路径。

4. **几何不是所有指标全胜**  
   可以讲 `9 / 9` geometry-safe 和 `7.6479%` mean triangle reduction；不能讲所有 geometry metric 都 strict win。

5. **paper-table 同口径仍需最终复核**  
   本地主表可信，但与原 MeshSplatting paper table 的最终论文比较必须完全对齐 protocol。

6. **Stage 2 shape prior 未过质量 gate**  
   v4 normal-band 有进步，但 chamfer 和 filled IoU 仍不足。

7. **最新 v78 target-footprint certificate 已完成但未通过 promotion**  
   v78 已经把 target-view footprint certificate 接入 train/eval pipeline，用目标视角的几何/可见性 footprint 约束 policy-val bin-gain hybrid，而不读取 target GT。`counter` 结果为 `26.753529 / 0.862111 / 0.251881`，低于 v75 zero-blend local patch 和 v64/v56 reference，因此只作为负结果诊断。它证明 target-footprint 可以减少弱 hybrid bins，但不能解决当前 low-capacity atlas 的根本瓶颈。本报告的 headline 仍固定为已经闭合的 Phase-J。

---

## 8. 推荐 PPT 结构

1. Title：SPCarNet: Self-Auditing MeshSplatting for Surface Repair and Compaction
2. Problem：MeshSplatting 有局部 residual 和冗余 triangles
3. Key Idea：用 train-view surface evidence 审计 mesh
4. Method：evidence cache、safe compaction、guarded residual repair、policy gate
5. Method Difference：标准 MeshSplatting vs SPCarNet
6. Quantitative Result：full9 selected-clean 9/9 wins
7. Visual Result 1：局部 held-out error reduction 图
8. Visual Result 2：公平 full-frame render 对比图
9. Outdoor Details：室外细节图
10. Ablation：v64-v78 说明 representation-level 内化还没完全闭合
11. Object Prior：Stage 2 v4 normal-band 有进步但未过 gate
12. Limitations and Next Step：stronger residual representation capacity + training objective，target-footprint 已作为负诊断闭合

---

## 9. 证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

当前技术报告：

```text
docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
```

Stage 2 v4 evidence：

```text
configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml
configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
```

---

## 10. 最终汇报口径

最稳妥的总结是：

> SPCarNet 当前已经证明，训练视角 surface evidence 可以把标准 MeshSplatting 变成一个可审计、可修复、可压缩的系统。Phase-J 在本地 full9 selected-clean MeshSplatting baseline 上实现 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，`244/246` 个 held-out views 三指标严格胜出，并平均删去 `7.65%` triangles。定性上，局部 error-reduction 图能清楚展示 SPCarNet 在高频纹理、表面残差区域更接近 GT。当前还不能夸大为 fully baked representation-level endpoint；v64-v78 和 Stage2 v4 说明我们已经推进了表示级和对象先验路线，但它们仍是下一阶段突破点，而不是当前主结果。
