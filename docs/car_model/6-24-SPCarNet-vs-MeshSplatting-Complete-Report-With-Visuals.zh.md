# SPCarNet 当前方法与 MeshSplatting 标准方法对比报告

日期：2026-06-24  
用途：mentor/PPT 汇报。  
当前主结论只使用已经完成闭环的 **Phase-J guarded adaptive ELA + geometry-safe compaction**。v79 v56-seeded anchor 已完成并复现 v56/v64 counter reference，但它只作为表示级诊断证据，不改变本文 headline。v78b fixed-code formal rerun 已完成并持久化小型审计证据；结果为 negative diagnostic，不改变 Phase-J headline。

---

## 1. 摘要结论

SPCarNet 不是替换 MeshSplatting，而是在标准 MeshSplatting checkpoint 之后增加一层 **self-auditing surface evidence pipeline**：

```text
标准 MeshSplatting checkpoint
  -> train-view surface evidence cache
  -> geometry-safe triangle compaction
  -> guarded residual repair
  -> train/policy-val gate and fallback
  -> held-out render evaluation
```

最通俗的说法：

> 标准 MeshSplatting 是“训练完直接渲染”。SPCarNet 是“训练完后，用训练视角证据检查每个 surface：哪些三角形低风险可以删，哪些区域有稳定残差可以修，哪些地方证据不足就回退不动”。

当前本地 full9 Mip-NeRF360、selected-clean MeshSplatting baseline 口径下，SPCarNet Phase-J 达成：

| 指标 | SPCarNet vs clean MeshSplatting |
|---|---:|
| 场景级 PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| held-out view 级 strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

因此，当前最稳妥的 claim 是：

> 在同一批本地 checkpoint 与 evaluator 下，SPCarNet 对标准 MeshSplatting 做到了 RGB 三指标全场景胜出，并平均删去 `7.65%` triangles；它已经是一个“可审计、可修复、可压缩”的 MeshSplatting 后处理/增强系统。

---

## 2. 方法对比

### 2.1 标准 MeshSplatting

标准 MeshSplatting 的工作流是：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh/splat checkpoint
  -> directly render held-out views
```

它的优势是显式 surface/mesh 表示、渲染质量稳定、训练流程直接。但原始方法不显式处理：

- 哪些 triangles 是低贡献冗余几何；
- 哪些 surface 区域在训练视角中有稳定、可迁移的 residual；
- 哪些局部修复对 held-out view 或 out-of-trajectory view 有风险；
- 修复失败时是否应自动回退到 clean baseline。

### 2.2 SPCarNet 当前 Phase-J

SPCarNet 的核心是把训练视角从“只用于优化”的监督信号，升级成“可审计的 surface evidence”。

| 维度 | 标准 MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础模型 | MeshSplatting checkpoint | 继承 MeshSplatting checkpoint |
| evidence 使用 | 训练时隐式使用 | 显式保存 train-view residual、visibility、face/bin support |
| 几何处理 | checkpoint 固定 | 低风险 triangle compaction |
| 外观修复 | 直接渲染 | guarded residual transfer |
| 风险控制 | 依赖训练收敛 | train/policy-val gate + fallback |
| 最终目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可解释审计 |

---

## 3. 当前方法模块

### 3.1 Surface Evidence Cache

缓存训练视角中的 surface 证据，包括：

- rendered RGB 与 GT RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、surface/bin address；
- normal、view direction、camera position；
- 每个 face/bin 的 support count；
- residual sign consistency；
- train/policy-val risk statistics。

作用：让系统知道“这个 residual 是否稳定出现过”“这个 triangle 是否真的支撑了多视角观测”“这个局部修改是否存在 view-tail 风险”。

### 3.2 Geometry-Safe Compaction

压缩只删除 evidence 判定低风险的 triangles。报告中的 `triangle reduction` 指 **删去的三角形占比**，不是剩余比例。

当前主结果平均删去：

```text
7.6479% triangles
```

同时保持 full9 `9 / 9` geometry-safe。

### 3.3 Guarded Evidence Lumigraph Adapter

SPCarNet 的主要 RGB 收益来自绑定在 surface 上的 residual repair：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后的 MeshSplatting render；
- `residual_i = GT_i - Render_i` 来自训练视角；
- `w_i(p)` 由 surface correspondence、visibility、support、risk 决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时自动 fallback 或 no-op。

关键点：这不是普通图像后处理。它必须通过 surface correspondence 把训练视角 residual 迁移到 held-out view，不能读取 held-out GT。

### 3.4 Train/Policy-Val Gate

当前公平性边界：

- policy、alpha、support、fallback 只使用 train/policy-val evidence；
- held-out test GT 只用于最终评估；
- clean MeshSplatting baseline 使用本地 clean checkpoint envelope 中更强者，避免低估标准方法；
- 不使用 test metrics 给我们方法调参。

---

## 4. 定量结果

评估口径：

- 数据集：Mip-NeRF360 full9；
- baseline：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR 越高越好，SSIM 越高越好，LPIPS 越低越好。

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

### 4.2 结果解读

最强的结论：

- RGB 三指标：full9 `9 / 9` 场景严格胜出；
- 视角级：`244 / 246` 个 held-out views strict win；
- 几何压缩：平均删去 `7.6479%` triangles；
- 室内场景提升更显著，bonsai/kitchen/counter/room 的 PSNR 提升尤其大；
- 室外场景也全胜，但肉眼全图差异更弱，更适合用局部 crop 和 error map 展示。

需要避免夸大的结论：

- 不能说当前方法已经是 fully baked representation-level endpoint；
- 不能说所有 geometry metric 全面 strict win；
- 与 MeshSplatting 原论文 table 的比较必须在最终论文前重新检查 split、mask、metric implementation、iteration 和 checkpoint selection。

---

## 5. 定性结果

### 5.1 最推荐主图：局部 held-out error reduction

这张图最适合放 PPT 主结果页，因为每行都包含：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 比 clean MeshSplatting 更接近 GT，紫红色表示变差。

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

代表性局部结果：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

讲法：

> 这张图说明 SPCarNet 的收益不是抽象指标，而是在局部高频纹理、光照残差和 surface detail 上把 clean MeshSplatting 的系统性误差压下去了。

### 5.2 公平 full-frame 渲染对比

这张图适合证明我们不是只挑局部裁剪。每行包含：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

讲法：

> 全图肉眼差异不总显著，这是因为很多收益来自 residual-level local correction；但 error map 和指标显示 ours error 更低。汇报时应把 full-frame 图作为公平性证据，把局部 error-reduction 图作为视觉收益证据。

### 5.3 室外场景细节对比

这张图专门覆盖 outdoor scenes：flowers、garden、treehill、bicycle、stump。

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail error reduction showcase">

讲法：

> 室外场景的全图差异更难直观看出，但在叶片、木纹、树皮、长椅条纹等 high-frequency surface 区域，SPCarNet 的局部 error reduction 更稳定。

---

## 6. 消融与当前边界

### 6.1 表示级 residual atlas 线

v64-v78 的目标是把 Phase-J 的 render-time residual repair 内化到 persistent surface representation。当前最稳固定 representation-level policy 是 v64 fixed auto bin-alpha：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | 9 | 2 | 9 | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | 9 | 7 | 8 | `+0.002255970` | `+0.000038081` | `-0.000093445` |

v78 target-footprint certificate 是有效接入 train/eval pipeline 的负诊断：

| line | counter PSNR | counter SSIM | counter LPIPS | status |
|---|---:|---:|---:|---|
| v64 / v56 reference | `26.756130` | `0.862126` | `0.251691` | best fixed representation-level reference |
| v79 v56-seeded anchor rerun | `26.756130` | `0.862126` | `0.251691` | reproduces v56/v64 anchor |
| v75 local patch prior | `26.753996` | `0.862119` | `0.251853` | not promoted |
| v78 target-footprint certificate | `26.753529` | `0.862111` | `0.251881` | negative diagnostic |

结论：

> representation-level 接口、policy、negative diagnostics 已经比较完整；v79 证明应从 face-alpha + tex32 + support4096 的 v56/v64 强锚点继续，而不是从 v75-v78 的弱配置继续。但当前收益仍远小于 Phase-J。它是下一阶段论文终局的突破方向，不是当前主结果。

### 6.2 Object-level shape prior 线

Stage 2 v4 normal-band autodecoder 是真实训练管线改动，用 surface-normal band supervision 强化 occupancy boundary：

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

> shape prior 有真实进步，但还没过强几何 gate，暂时不作为主 claim。

---

## 7. 当前最诚实的结论

SPCarNet 当前已经明显强于标准 MeshSplatting 的地方：

- full9 selected-clean baseline 上 RGB 三指标 `9 / 9` 场景全胜；
- `244 / 246` held-out views strict win；
- 平均删去 `7.6479%` triangles；
- 局部高频纹理、光照残差、surface detail 的 error map 改善清楚；
- 方法有明确机制：surface evidence、safe compaction、guarded repair、fallback。

SPCarNet 当前还不能过度承诺的地方：

- Phase-J 仍更像 render-time self-auditing endpoint，不是完全内化的 representation-level renderer；
- 全图肉眼差异在部分室外场景不强，需要 error map/crop 解释；
- representation-level v64-v78 的收益还小；
- Stage 2 object prior 仍未达到强几何先验质量；
- 与 MeshSplatting paper table 的最终论文级比较还需要完全同口径复核。

推荐 PPT 主线：

> MeshSplatting 给了高质量显式 surface；SPCarNet 让这个 surface 变得可审计。通过 train-view evidence，我们能判断哪里可以压缩、哪里可以修复、哪里必须回退。当前结果证明，这个审计机制在 full9 selected-clean baseline 上同时带来 RGB 指标提升和 triangle reduction。

---

## 8. 证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/surface_residual_region_texture_adapter_audit.json
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

相关技术报告：

```text
docs/car_model/6-24-SPCarNet-Current-Method-Experiment-Visual-Report.zh.md
docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
```
