# SPCarNet 当前方法完整技术报告（Mentor/PPT Clean 版）

日期：2026-06-24  
用途：给 mentor 汇报当前 SPCarNet 的方法、证据、与 MeshSplatting baseline 的差距、定性结果、局限和下一步路线。  
建议 PPT 标题：**SPCarNet: Self-Auditing MeshSplatting for Evidence-Certified Repair and Compaction**

---

## 0. 汇报结论

当前整体 paper-loop 状态：`NOT COMPLETE`。本报告用于 mentor/PPT 汇报当前最强已验证 endpoint 和清晰短板，不表示论文终局闭环已经完成。

当前最稳、最适合放在 PPT 主结果页的 endpoint 是：

```text
Phase-J guarded adaptive Evidence Lumigraph Adapter
+ geometry-safe triangle compaction
+ train-only policy gate / fallback
```

一句话讲清楚：

> MeshSplatting 训练完后直接渲染；SPCarNet 让它先用训练视角的 surface evidence 审计自己：哪些三角形风险低就删，哪些 surface residual 多视角稳定就修，证据不足就回退。

主结果：

| 指标 | 当前 Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| 数据集/场景 | Mip-NeRF360 full9 |
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| per-view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

必须同时诚实说明：

- 这个 endpoint 在 RGB 质量和 triangle-count reduction 上已经强于本地复现的 selected clean MeshSplatting baseline；完整模型 rate-distortion、FPS 和属性存储收益还没有作为主 claim 闭合。
- 它不是完全从零替代 MeshSplatting，而是建立在 MeshSplatting checkpoint 上的 evidence-certified repair/compaction layer。
- 当前最强 RGB 收益仍主要来自 render-time guarded ELA；完全 baked into checkpoint 的 representation-level endpoint 还没有达到同等强度。
- v64-v83 表示级实验已经提供了系统性诊断和最新表示升级方向；其中 `v82 capacity-prerank + face-alpha` 在 `counter` 单场景上以极小 margin 严格超过 v56/v64/v79/v80 anchor，但 hard-triad 的 `kitchen/bonsai` 结果没有严格超过 v64 anchor，因此 raw v82 不能作为 headline。
- Stage2 v4 shape prior 是真实训练改进；`v4_epoch50` 是 selector score 最优候选，在 Chamfer、filled IoU 和 normal consistency 上相对 v3 有改善，但 shell IoU 退化且仍未过原定 shape-quality gate。

最新补充结果（只作为 backup 或 next-step slide，不作为主结果）：

| probe | scope | PSNR | SSIM | LPIPS | verdict |
|---|---|---:|---:|---:|---|
| v82 capacity-prerank + face-alpha | `counter` only | `26.756137848` | `0.862126350` | `0.251690656` | beats v56/v64/v79/v80 counter anchor by tiny margin |
| v82 capacity-prerank + face-alpha | hard triad `kitchen/bonsai` | kitchen `27.823442459`, bonsai `28.866504669` | kitchen `0.876437545`, bonsai `0.896055281` | kitchen `0.198780462`, bonsai `0.259272754` | raw policy fails strict promotion vs v64 anchor |
| v83 patchmix + face-alpha + local-patch hybrid | `counter` only | `26.756147385` | `0.862125337` | `0.251688808` | improves PSNR/LPIPS over anchors, but tiny SSIM regression; not promoted |
| v84 strict selector | full9 materialized selector | mean dPSNR vs v64 `+0.000000848` | mean dSSIM vs v64 `+0.000000013` | mean dLPIPS vs v64 `-0.000000079` | non-regressive vs v64 with counter v82 and v64 fallback; report-only |

对比 anchor：

```text
v56/v64/v79 counter anchor: 26.756130219 / 0.862126231 / 0.251691371
v80 near-tie:               26.756135941 / 0.862126231 / 0.251691461
v82 capacity-prerank:       26.756137848 / 0.862126350 / 0.251690656
v83 patchmix hybrid:        26.756147385 / 0.862125337 / 0.251688808
v82 hard triad verdict:     counter tiny strict win, kitchen mixed, bonsai strict fail
v84 selector verdict:       full9 non-regressive vs v64, but only +8.48e-7 mean PSNR
```

---

## 1. 30 秒口播版

中文：

> 我们的出发点不是推翻 MeshSplatting，而是让一个已经训练好的 MeshSplatting checkpoint 具备自诊断和自修复能力。训练视角里可见且被 surface map 覆盖的像素可以关联到 mesh face/bin，因此我们可以统计某个 surface 区域是否有稳定 residual、是否有足够多视角支持、是否处于风险边界。SPCarNet 只在 evidence 充足的位置做 residual repair，只删除 evidence 证明低风险的 triangles，证据不足时自动回退。当前 Phase-J 在本地 full9 selected-clean MeshSplatting baseline 上 9/9 场景三指标严格胜出，同时平均删去 7.65% triangles。

英文：

> SPCarNet turns a trained MeshSplatting checkpoint into a self-auditing surface system: it compacts low-risk triangles, transfers only evidence-certified residuals, and falls back when the train-view certificate is weak.

PPT 第一页可用 claim：

```text
From MeshSplatting to Self-Auditing Surface Repair and Compaction
```

---

## 2. 研究问题与动机

MeshSplatting 的优点是显式 mesh/surface 表示、高质量 novel-view rendering 和较好的部署潜力。但 clean checkpoint 仍有三个问题：

| 问题 | 现象 | 为什么重要 |
|---|---|---|
| 局部外观 residual | 细纹理、遮挡边界、桌面/树叶/室内物体上仍有稳定误差 | 全图指标和人眼局部观感都会受影响 |
| 几何/拓扑冗余 | 部分 triangles 对多视角解释贡献低 | 影响 triangle count，并可能影响模型大小、渲染成本和可编辑性 |
| 修复泛化风险 | 直接把 train residual 搬到 test view 可能伤害 tail view | 必须避免 out-of-trajectory collapse |

核心研究问题：

```text
Given a trained MeshSplatting checkpoint,
can training-view surface evidence certify where the mesh can be compacted
and where appearance residuals can be safely repaired?
```

这不是单纯图像后处理。SPCarNet 的每个修复都绑定到 surface address、face/bin support、visibility、policy-val risk 和 fallback 机制。

---

## 3. 与基础 MeshSplatting 的区别

基础 MeshSplatting：

```text
images + cameras
  -> train MeshSplatting
  -> checkpoint
  -> render target views
```

SPCarNet：

```text
images + cameras
  -> train/load MeshSplatting checkpoint
  -> render train/policy-val views with surface maps
  -> build surface evidence: residual, support, risk, visibility
  -> geometry-safe compaction
  -> evidence-certified residual repair
  -> train-only policy gate and fallback
  -> held-out evaluation
```

| 维度 | MeshSplatting baseline | SPCarNet 当前方法 |
|---|---|---|
| 基础表示 | trained mesh/splat checkpoint | 继承同一 checkpoint |
| 是否显式使用 train residual | 否 | 是，构建 surface evidence cache |
| 几何压缩 | checkpoint 固定 | 低风险 triangles 删除 |
| 外观修复 | 直接渲染 checkpoint | guarded residual transfer |
| 泛化保护 | 依赖训练收敛 | train/policy-val gate、tail risk、fallback |
| 当前最强 endpoint | clean selected checkpoint | compact checkpoint + guarded ELA |

对 mentor 的安全说法：

> 我们不是声称完全替代 MeshSplatting，而是证明 MeshSplatting checkpoint 可以被训练视角证据系统性增强：质量更高、triangles 更少、且有明确的失败回退机制。

---

## 4. 方法总览

SPCarNet 当前由五个层次组成。

| 层次 | 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| Evidence | Surface evidence cache | train/policy-val renders + GT + surface maps | residual/support/risk/bin evidence | 判断哪里可信 |
| Geometry | Geometry-safe compaction | checkpoint + evidence | compact checkpoint | 删除低风险 triangles |
| Appearance | Guarded ELA | compact render + residual evidence | repaired render | 修复稳定 residual |
| Policy | Train-only gate | policy-val risk metrics | accept/fallback/no-op | 避免 test leakage 和 tail collapse |
| Future | Surface residual atlas / Stage2 shape prior | residual atlas / shape auto-decoder | representation-level candidate | 把修复能力内化到表示层 |

关键代码入口：

| 功能 | 路径 |
|---|---|
| Region texture residual adapter | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` |
| Single-scene L1-risk fair no-op runner | `scripts/car_model/run_l1risk_fairnoop_scene.py` |
| v64 fixed auto-policy runner | `scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py` |
| Stage2 shape-field eval | `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py` |
| Stage2 v4 training wrapper | `scripts/car_model/train_spcarnet_shape_field_autodecoder_v4_band.sh` |
| Evidence manifest builder | `scripts/car_model/build_spcarnet_current_evidence_manifest.py` |

---

## 5. 核心模块细节

### 5.1 Surface Evidence Cache

Evidence cache 是 SPCarNet 的数据底座。它从 train/policy-val views 中保存：

- rendered RGB；
- ground-truth RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、barycentric coordinate、UV/bin address；
- normal、view direction、camera center；
- per-face/per-bin support count；
- residual sign consistency；
- image L1、PSNR、SSIM、LPIPS；
- min-view 和 CVaR tail risk。

它的意义是把训练视角从“仅用于优化 checkpoint”升级成“用于审计 surface 是否可修、可删、可回退”的证据。

### 5.2 Geometry-Safe Compaction

压缩目标是 quality-first，而不是最大化删面：

```text
delete a triangle only when multi-view evidence says the edit is low-risk
```

保护重点：

- sparse visibility 区域；
- thin structures；
- high residual or boundary faces；
- policy-val 风险高的局部区域；
- indoor scenes 中本来 triangles 少、风险高的区域。

报告中的 `triangle reduction` 指**删去的三角形比例**，不是剩余比例。当前 Phase-J 平均删面 `7.6479%`。这只是 triangle-count compactness claim，不等价于完整模型大小、属性存储或 FPS 的 rate-distortion claim。

### 5.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。简化表达：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是 compact checkpoint 在 target view 上的 render；
- `residual_i = GT_i - Render_i` 来自 train/policy-val view；
- `u_i` 是与 target pixel `p` 对应的 surface address；
- `w_i` 由 visibility、surface correspondence、support 和风险决定；
- `alpha` 由 train-only evidence 自动选择；
- 证据不足时 fallback 或 no-op。

通俗讲：

> 如果同一个 surface 区域在多个训练视角里反复错同一个方向，我们认为它是稳定 residual，可以被迁移修复；如果证据稀疏或视角尾部风险高，就不修。

### 5.4 Train-Only Policy Gate

公平性规则：

- 我们方法的 branch、alpha、support、fallback 都由 train/policy-val evidence 决定；
- held-out test GT 只用于最终评估；
- baseline clean checkpoint envelope 用 held-out metrics 选更强 clean checkpoint，是为了给 baseline 一个强 comparator；
- 不用 train metric 选 baseline，因为这会偏向训练更久的 checkpoint；
- 不用 test metric 选择我们方法参数。

当前 Phase-J branch：

| branch | scene |
|---|---|
| adaptive ELA | `bicycle, flowers, garden, stump, room, counter, kitchen, bonsai` |
| edge fallback | `treehill` |

### 5.5 Representation-Level Residual Atlas

v48-v83 是把 render-time ELA 的思想进一步写进 surface/face/bin residual 表示的尝试：

- v48：auto-support surface atlas；
- v52：capacity-aware support policy；
- v56：face-alpha reliability guard；
- v64：fixed auto bin-alpha policy；
- v75：local patch surface prior；
- v76：policy-val bin-gain hybrid；
- v77：strict multi-view bin-gain hybrid；
- v78/v78b：target-support/target-footprint certificate audit；修复后已补齐 target-view coverage 统计，但指标仍低于 v64/v56 reference，因此仅保留为负结果诊断。
- v79：v56-seeded face-alpha anchor；复现 v56/v64 counter strong anchor，说明后续 representation-level 尝试应以这个强 anchor 为比较基线。
- v80：face-alpha + local-patch + bin-gain hybrid；在 `counter` 上恢复到 near-tie，超过 v75 和 v78b，但 LPIPS 仍略差于 v56/v64/v79 anchor，因此不提升为主结果。
- v81：normal-camera linear view-conditioned residual basis；机制真实生效且 policy-val 接受，但 held-out 三指标均低于 v56/v64/v79 anchor，因此是负诊断。
- v82 patch-mixture：teacher-distilled residual basis；目标是把 teacher residual 从单一 face-smooth fit 升级为带局部 UV patch 容量的 per-face basis，并继续由 train/policy-val certificate 审计。counter probe 已完成，guard 将 patch basis 回退到 legacy teacher basis，held-out 三指标低于 v56/v64/v79/v80 anchor，因此是负诊断。
- v82 capacity-prerank + face-alpha：将 target-support capacity pre-rank 与 face-alpha calibration 组合；`counter` 单场景 strict 超过 v56/v64/v79/v80 anchor，但 margin 约为 `+0.00000763` PSNR / `+0.00000012` SSIM / `-0.00000072` LPIPS。随后 hard-triad 发现 `kitchen` SSIM 低于 v64、`bonsai` 三指标低于 v64，因此 raw v82 不 promoted。
- v83：patchmix + face-alpha + local-patch hybrid；counter probe 已完成，PSNR/LPIPS 超过 v56/v64/v79/v80/v82 anchors，但 SSIM 有 `-0.000000894` 量级回退，因此是 mixed diagnostic，不 promoted。

当前结论：

> 这些分支证明 pipeline 已经能做真实 representation-level 改动和审计，但收益仍非常小。瓶颈不是继续调 alpha/blend/cap，而是 residual representation capacity 与 target-view generalization certificate。

---

## 6. 评估协议

### 6.1 主实验

| 项 | 设置 |
|---|---|
| 数据集 | Mip-NeRF360 full9 |
| baseline | 本地 clean MeshSplatting `26000/30000` checkpoint envelope |
| baseline 选择 | 每个 scene 在 clean `26000` 与 `30000` 中选择 held-out test 更强者 |
| ours | Phase-J guarded adaptive ELA + geometry-safe compaction |
| 指标 | PSNR ↑, SSIM ↑, LPIPS ↓, triangle-count reduction ↑ |
| fair selection | 我们方法的策略选择不使用 held-out test GT |

为什么 baseline 用 selected-clean envelope：

> 这是为了避免“拿弱 clean checkpoint 当 baseline”的公平性问题。baseline 在两个 clean checkpoints 中选更强者；我们方法参数仍由 train/policy-val evidence 决定。

### 6.2 论文 paper-table 口径

当前最严谨的主 claim 应使用本地 same-protocol baseline，因为 split、checkpoint、evaluator 和结果路径都可追溯。与 MeshSplatting 论文表格的对齐可以作为辅助，但最终投论文前需要再次确认：

- train/test split；
- mask/crop；
- metric implementation；
- checkpoint iteration；
- clean checkpoint selection；
- image resolution 和 color-space convention。

---

## 7. 主定量结果

### 7.1 Full9 per-scene result

| scene | SPCarNet PSNR | SPCarNet SSIM | SPCarNet LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | tri red. |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | `24.021544` | `0.702357` | `0.266088` | `+0.719931` | `+0.042489` | `-0.065989` | `11.81%` |
| flowers | `20.304358` | `0.557770` | `0.329222` | `+0.622101` | `+0.045948` | `-0.065341` | `11.82%` |
| garden | `26.311111` | `0.827843` | `0.135843` | `+1.281900` | `+0.047808` | `-0.065472` | `3.47%` |
| stump | `25.595104` | `0.724074` | `0.263909` | `+0.390062` | `+0.018909` | `-0.030095` | `11.82%` |
| treehill | `21.296227` | `0.595606` | `0.336319` | `+0.362045` | `+0.031083` | `-0.069725` | `11.81%` |
| room | `30.305639` | `0.905730` | `0.195989` | `+1.558363` | `+0.020887` | `-0.053913` | `2.10%` |
| counter | `28.449171` | `0.893731` | `0.186472` | `+1.697397` | `+0.031675` | `-0.065531` | `2.10%` |
| kitchen | `30.199732` | `0.916087` | `0.131955` | `+2.381180` | `+0.039635` | `-0.067231` | `2.10%` |
| bonsai | `31.862005` | `0.930280` | `0.172555` | `+2.966772` | `+0.033879` | `-0.086937` | `11.80%` |

Aggregate：

| aggregate | value |
|---|---:|
| mean dPSNR vs clean | `+1.331084` |
| mean dSSIM vs clean | `+0.034702` |
| mean dLPIPS vs clean | `-0.063359` |
| mean dPSNR vs source ELA | `+0.833143` |
| mean dSSIM vs source ELA | `+0.018946` |
| mean dLPIPS vs source ELA | `-0.039986` |
| mean total triangle reduction | `7.6479%` |

主结果证据：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json
```

### 7.2 Closure audit

| 审计项 | 结果 |
|---|---:|
| strict RGB scene wins vs selected clean | `9 / 9` |
| strict RGB scene wins vs source Compact-ELA/SOR | `9 / 9` |
| per-view strict RGB wins | `244 / 246` (`99.19%`) |
| sparse geometry strict wins | `6 / 9` |
| sparse geometry-safe scenes | `9 / 9` |

解释：

> RGB 主 claim 已闭合；几何可以讲 safe 和 triangle-count reduction，但不能夸成所有 sparse geometry 指标 9/9 strict win。`244 / 246` 的两个非 strict per-view 来自 `treehill` 的 `17 / 18` 和 `room` 的 `38 / 39`。

证据：

```text
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv
```

---

## 8. 定性结果与 PPT 推荐图

### 8.1 最推荐主图：where-it-helps local crop panel

这张图最适合讲“人眼为什么应该相信它有改善”，因为它不是只放 full render，而是放局部 crop 和 error reduction。

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

Markdown 预览：

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J where-it-helps local crop panel">

建议 PPT 说法：

> Full-frame 缩小后差异会被掩盖，因此我们用 held-out local crop 和 error-reduction map 展示改进。绿色区域表示 SPCarNet 更接近 GT。

选图 provenance 建议写进 caption 或 speaker note：

> Selected held-out showcase; selection criterion is full-view positive metrics plus local error reduction. It is a visual explanation of where the method helps, not an unbiased random sample of all views.

### 8.2 全场景 qualitative gallery

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_full9_crop_gallery.png
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet Mip-NeRF360 full9 qualitative gallery">

用途：

- 展示 full9 都有结果；
- 作为 backup slide；
- 不建议单独作为最主要视觉证据，因为 full-frame 缩小后改进不够显眼。

### 8.3 Outdoor detail showcase

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail showcase">

用途：

- 回答“室外场景是否有收益”；
- 强调局部 texture/edge residual repair；
- 同时承认室外 full-frame 视觉提升没有室内那么直观。

### 8.4 Representation-level diagnostic panels

```text
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
assets/spcarnet_v42_atlas_qualitative_panel.png
```

这些图适合放 backup，不适合做主结果页。原因是 v52/v56/v64 的 representation-level 改进幅度很小，更多说明策略演进和失败诊断，而不是 headline 视觉突破。

---

## 9. 表示级路线与消融结果

### 9.1 v52 capacity-aware policy

v52 把 v48 auto-support 和 v51 larger-support 结合成 train-only fixed policy：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v52 vs no-op | 9 | 7 | 8 | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | 9 | 3 | 9 | `+0.000086890` | `+0.000008782` | `-0.000015303` |
| v52 vs v50 | 9 | 6 | 6 | `+0.000284831` | `+0.000014782` | `-0.000020780` |

意义：

> v52 证明 capacity-aware support policy 可以避免 cap-hit 场景错失收益，但效果仍是小增益。

### 9.2 v56 face-alpha reliability guard

v56 只在 local alpha 证据足够可靠时启用 face-alpha branch，否则回退 v52。

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v56 vs v52 | 9 | 1 | 9 | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 vs no-op | 9 | 7 | 8 | `+0.001845890` | `+0.000037803` | `-0.000074494` |
| v56 vs v48 | 9 | 3 | 9 | `+0.000383589` | `+0.000010067` | `-0.000034966` |

意义：

> v56 把 raw local-alpha 的风险挡住了，但仍不是大幅提升。

### 9.3 v64 fixed auto bin-alpha policy

v64 是当前最稳的固定 representation-level candidate。它用 train/policy-val guard 只在 `kitchen` 接受 v63b bin-alpha，其余回退 v56。

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | 9 | 2 | 9 | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | 9 | 7 | 8 | `+0.002255970` | `+0.000038081` | `-0.000093445` |

讲法：

> v64 证明了固定自动策略可行，但增益太小，不能作为论文主贡献。

### 9.4 v75-v83 诊断

| branch | 改动 | 结果 | 结论 |
|---|---|---|---|
| v75 local patch prior | 用 same-face local UV patch prior 填 low-support bins | counter 选择 `blend=0.0`，与 zero-blend row 持平 | coarse count-pyramid 不是唯一瓶颈 |
| v76 policy-val bin-gain hybrid | policy-val bin 级别挑选 nonzero prior bins | held-out 略低于 v75/v64 | weak bin certificate 不足以泛化 |
| v77 strict bin-gain hybrid | 加 multi-view/absolute-gain stricter gate | 阻断 weak hybrid，回到 zero-blend | 安全性更强，但不是指标提升 |
| v78/v78b target-footprint certificate | 修正 target-view footprint/bin 统计与候选选择审计 | held-out 仍低于 v75 和 v64/v56 | audit 修复必要，但不等于表示能力提升 |
| v79 v56-seeded anchor | 复现 v56/v64 counter strong anchor | `26.756130 / 0.862126 / 0.251691` | 建立后续 counter probes 的强比较基线 |
| v80 face-alpha + local-patch + hybrid | 组合 face-alpha、local-patch prior、bin-gain hybrid | `26.756135941 / 0.862126231 / 0.251691461` | 超过 v75/v78b 并接近 anchor，但 LPIPS 略差，不 promoted |
| v81 view-conditioned basis | per-bin normal-camera linear residual basis | `26.753919601 / 0.862121582 / 0.251836061` | policy-val 接受但 held-out 回退，说明简单线性视角基不够 |
| v82 patch-mixture teacher basis | 新增 `face_uv_patch_mixture_ridge` teacher-distilled basis | `26.753459930 / 0.862114668 / 0.251868337` | guard 回退 patch basis；低于 v56/v64/v79/v80 anchor，不 promoted |
| v82 capacity-prerank + face-alpha | target-support pre-rank + face-alpha calibration | counter `26.756137848 / 0.862126350 / 0.251690656`; kitchen `27.823442459 / 0.876437545 / 0.198780462`; bonsai `28.866504669 / 0.896055281 / 0.259272754` | counter-only 极小 strict win；hard-triad raw policy 对 v64 不 strict non-regressive，因此不 promoted |
| v83 patchmix + face-alpha + local-patch hybrid | 把 patch-mixture teacher basis 与 face-alpha/local-patch/hybrid 组合 | counter `26.756147385 / 0.862125337 / 0.251688808` | PSNR/LPIPS 超过 v56/v64/v79/v80/v82 anchors，但 SSIM 微退；mixed diagnostic，不 promoted |
| v84 strict capacity selector | v82 capacity-prerank 通过严格 train/policy-val guard 才采用，否则 v64 fallback | full9 mean vs v64 `+0.000000848 / +0.000000013 / -0.000000079` | 工程上完成自动 selector 与物化闭环；但 rule 来自 hard-triad 后诊断，仍是 report-only |

关键 takeaway：

> 继续在 alpha、blend、support cap 上调参已经难以突破。下一步需要更强 residual representation 和 target-view generalization certificate。

### 9.5 最新表示升级：Patch-Mixture Teacher Basis

v82/v83 的动机是：v81 的 simple normal-camera linear basis 太弱，local patch prior 又只在 fit-time 改 atlas texture，没有形成更强的 per-face residual function。因此最新代码把 teacher-distilled basis 从：

```text
[1, camera, normal, normal_dot_camera, u, v, u^2, v^2, u*v]
```

扩展为：

```text
base face/UV/view features
+ 3x3 local UV RBF patch mixture
+ patch weights * normal_dot_camera
```

实现位置：

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

当前原则：

- 它是一个真实 representation-level 方法改动，不是单纯换 alpha；
- 只允许由 train/policy-val guard 接受；
- 完成前只作为 counter probe；
- 必须严格超过 v56/v64/v79 anchor 和 v80 near-tie，才允许进入 hard triad 或 full9；
- 如果 policy-val 接受但 held-out 回退，它会和 v81 一样被记录为负诊断。

v82 patch-mixture 已完成 counter probe。它实现了上述新 basis，但 `policy_val_nonregressive` guard 认为 patch basis 的 policy-val SSIM gain 低于 legacy teacher branch：

```text
patch basis ssim_gain: 0.00015930
legacy teacher ssim_gain: 0.00016533
guard decision: fallback_to_legacy
```

最终 held-out counter 指标为：

```text
26.753459930 PSNR / 0.862114668 SSIM / 0.251868337 LPIPS
```

这低于 v56/v64/v79 anchor `26.756130219 / 0.862126231 / 0.251691371`，也低于 v80 near-tie，因此 v82 不应进入 hard-triad/full9。它的价值是证明新表示接口可运行且 guard 能挡住更弱 basis，而不是证明当前 patch mixture 已经突破瓶颈。

v83 把 patch-mixture teacher basis 与 face-alpha calibration、local-patch prior 和
policy-val bin-gain hybrid 组合后，counter probe 已完成：

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64/v79 anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v82 capacity-prerank | `26.756137848` | `0.862126350` | `0.251690656` |
| v83 patchmix hybrid | `26.756147385` | `0.862125337` | `0.251688808` |

结论是：v83 在 PSNR 和 LPIPS 上进一步突破 counter plateau，但 SSIM 相对
v56/v64/v79 anchor 低 `0.000000894`，相对 v82 低 `0.000001013`。因此它证明
“更强 residual capacity 有效果”，但也证明当前 certificate 还不够 SSIM-safe，不能
作为 strict three-metric promoted endpoint。

### 9.6 最新单场景里程碑：Capacity-Prerank + Face-Alpha

另一个 v82 分支把 target-support pre-rank、support-capacity candidate 和 face-alpha calibration 组合起来。它不使用 held-out test metric 选择 branch，而是由 policy-val gate 接受：

| audit field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| changed fraction | `0.064177659` |
| policy-val SSIM gain | `0.000294710` |
| policy-val SSIM positive fraction | `1.0` |
| policy-val image-L1 gain | `0.000026924` |
| policy-val image-L1 positive fraction | `0.916666667` |

Held-out counter result：

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64/v79 anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v80 near-tie | `26.756135941` | `0.862126231` | `0.251691461` |
| v82 capacity-prerank | `26.756137848` | `0.862126350` | `0.251690656` |

Hard-triad 追加验证：

| scene | v82 capacity-prerank | v64/v56/v79 anchor | delta vs anchor | verdict |
|---|---|---|---|---|
| counter | `26.756137848 / 0.862126350 / 0.251690656` | `26.756130219 / 0.862126231 / 0.251691371` | `+0.000007629 / +0.000000119 / -0.000000715` | tiny strict win |
| kitchen | `27.823442459 / 0.876437545 / 0.198780462` | `27.822626114 / 0.876537859 / 0.198848858` | `+0.000816345 / -0.000100315 / -0.000068396` | mixed; strict fail due SSIM |
| bonsai | `28.866504669 / 0.896055281 / 0.259272754` | `28.868467331 / 0.896088481 / 0.259204030` | `-0.001962662 / -0.000033200 / +0.000068724` | strict fail |

结论：

> 这是近期第一个在 counter 上同时严格超过 v56/v64/v79/v80 anchor 的表示级 probe，但 hard-triad 证明 raw fixed policy 还不能泛化到强 anchor。它应当作为“policy-val evidence 与 held-out 泛化仍有 gap”的诊断结果，而不是 paper headline。下一步需要更强的 automatic selector 或更稳定的 residual representation，而不是继续手工宣传这个 raw v82。

### 9.7 v84 strict selector：把 v82 信号纳入固定策略

为了避免继续做人工 per-scene 参数选择，v84 新增了一个固定、可审计 selector：

```text
if v82 train/policy-val evidence is strict and alpha is moderate:
    use v82 capacity-prerank
else:
    fallback to v64 selected policy
```

实现：

```text
scripts/car_model/summarize_v84_strict_v82_capacity_selector.py
```

结果：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v84 vs v64 | `9` | `1` | `9` | `+0.000000848` | `+0.000000013` | `-0.000000079` |
| v84 vs v56 | `9` | `2` | `9` | `+0.000410928` | `+0.000000291` | `-0.000019030` |
| v84 vs no-op | `9` | `7` | `8` | `+0.002256817` | `+0.000038094` | `-0.000093525` |

选择结果：`counter` 采用 v82；`kitchen/bonsai` 因 guard 不通过回退 v64；其余六个未 materialize v82 candidate 的场景也回退 v64。

结论：

> v84 完成了表示级分支的自动 selector 与 selected tree 物化，工程口径比“counter 单场景亮点”干净。但它的收益极小，且规则是在 v82 hard-triad 诊断后形成，不能包装为 paper-clean breakthrough。它适合作为 backup ablation 和下一轮 blind validation 的固定候选。

公平性边界：v84 selector 不读取 held-out PSNR/SSIM/LPIPS 来决定 branch；guard
使用 train/policy-val audit 字段，并额外检查 `target_apply.changed_fraction` 作为目标侧
effect-size 约束。这个 changed-fraction 不使用 target GT 指标。由于 `/data` 空间不足，
v84 selected tree 的 render/GT 采用软链接，部分链接指向 `/dev/shm`，所以它是当前机器
状态下可用的物化树，不是完全耐久的归档件。

---

## 10. Stage2 shape prior 现状

Stage2 的目标是学习对象级 shape-field auto-decoder，为后续几何先验和修复提供更强 representation support。

### 10.1 v3 held-out MAP-fit 修复

旧问题：

> auto-decoder 只有 train-object latent table，val/test object 没有 latent，导致 held-out eval extraction 失败。

当前修复：

- `--fit_missing_latents`；
- decoder frozen；
- held-out object 做 z-only MAP fitting；
- per-object 标记 `latent_source = heldout_map_fit`；
- `206 / 206` val objects extracted。

### 10.2 v4 normal-band objective

v4 加入 surface-normal band supervision：

```text
x_inner = x_surface - epsilon * normal -> occupied
x_outer = x_surface + epsilon * normal -> free
```

目标是让 occupancy field 在 surface 附近形成更清晰的 crossing，改善 Marching-Cubes mesh。

Full-val comparison：

| metric | v3 MAP-fit | v4 epoch50 | v4 final | best |
|---|---:|---:|---:|---|
| `recon_chamfer_l1_mean` | `0.0698447353` | `0.0607328202` | `0.0655826944` | v4 epoch50 |
| `hidden_chamfer_l1_mean` | `0.1023846301` | `0.0933915632` | `0.0963624408` | v4 epoch50 |
| `mesh_iou_at_0.5_mean` | `0.5531548112` | `0.5683319216` | `0.5314717742` | v4 epoch50 |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` | `0.8783071888` | `0.8563237802` | v3 |
| `surface_normal_consistency_mean` | `0.7182239138` | `0.7195177524` | `0.6890807638` | v4 epoch50 |

Selector decision：

```text
status: BEST_AVAILABLE_GATE_FAIL_WITH_LATE_DEGRADATION
best candidate: v4_epoch50
gate pass: False
```

W&B：

| run | purpose |
|---|---|
| `dysg8508` | v4 full training |
| `4wu9w305` | v4 epoch50 full-val MAP-fit eval |
| `q1jjwvdm` | v4 final full-val MAP-fit eval |

讲法：

> Stage2 v4 是真实训练方法改动；`v4_epoch50` 是 selector score 最优候选，并相对 v3 改善 Chamfer、filled IoU 和 normal consistency。但 shell IoU 退化，且未达到原定 gate，因此不能放在主结果页，只能作为几何先验路线的 next-step evidence。

证据：

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
```

---

## 11. 为什么这是研究工作，不只是工程 patch

可以这样组织故事：

1. **问题不是单纯指标调参，而是如何判断显式 surface 表示哪里可靠。**  
   MeshSplatting 的 checkpoint 有真实 surface，因此可以把 residual、visibility 和 risk 都绑定到 surface 上。

2. **SPCarNet 提出 evidence-certified repair/compaction。**  
   不是所有区域都修，而是由训练视角证据决定可修、可删、可回退。

3. **安全性是方法的一部分。**  
   train-only policy gate、tail risk 和 fallback 防止 out-of-trajectory 视角崩塌。

4. **负结果给出清晰机制诊断。**  
   v65-v77 说明继续做更复杂 bin prior 仍不够，需要新的 residual representation 和 target-view certificate。

5. **主结果证明 MeshSplatting 可以被系统性增强。**  
   full9 9/9 RGB wins + 7.65% triangle-count reduction 是强证据。

PPT 中建议避免说：

```text
We solved MeshSplatting.
```

建议说：

```text
We show that a trained MeshSplatting model can be converted into a self-auditing
surface system with certified repair, fallback, and compaction.
```

---

## 12. 当前短板与风险

| 短板 | 当前事实 | PPT/讨论时的安全说法 |
|---|---|---|
| 最强 RGB 仍是 render-time adapter | Phase-J 不是 fully baked checkpoint | 作为当前 endpoint，可讲强结果；representation 内化是下一步 |
| 表示级收益微小 | v64 相对 v56 只有 `+0.000410` PSNR；v84 相对 v64 只有 `+0.000000848` mean PSNR | 已系统性诊断瓶颈；v82/v83/v84 是新的表示级 probe，不包装成 headline |
| 几何不是所有指标 strict win | sparse geometry strict wins `6 / 9` | 讲 geometry-safe `9 / 9` 和 triangle-count reduction |
| 定性 full-frame 差异不总是显眼 | 局部修复在 full image 缩小后不明显 | 用 crop + error reduction map |
| paper-table 对齐仍需最终复核 | 本地主表最可靠 | paper table 只作辅助，不作最终主 claim |
| Stage2 shape prior 未过 gate | v4 epoch50 改善但 gate fail | next-step evidence，不作为主结果 |

---

## 13. 建议 PPT 结构

### Slide 1: Title

**SPCarNet: Self-Auditing MeshSplatting for Evidence-Certified Repair and Compaction**

一句话：训练好的 MeshSplatting checkpoint 可以用 train-view surface evidence 做自诊断、自修复和 geometry-safe triangle-count compaction。

### Slide 2: Motivation

MeshSplatting 很强，但仍有：

- local residual；
- redundant triangles；
- unsafe naive repair risk。

### Slide 3: Key Idea

Pipeline：

```text
MeshSplatting checkpoint
  -> surface evidence
  -> geometry-safe triangle-count compaction
  -> guarded residual repair
  -> train-only policy gate
```

### Slide 4: Evidence Cache

展示 residual / face id / support / risk 表格。

### Slide 5: Geometry-Safe Compaction

重点：

- mean triangle-count reduction `7.6479%`；
- geometry-safe `9 / 9`；
- quality-first compaction。

### Slide 6: Guarded Residual Repair

放公式：

```text
I_ours = I_compact + alpha * evidence-weighted residual
```

强调证据不足就 fallback。

### Slide 7: Main Quantitative Result

放 aggregate 表：

- `9 / 9` strict scene wins；
- `244 / 246` strict per-view wins；
- `+1.3311 PSNR`；
- `+0.0347 SSIM`；
- `-0.0634 LPIPS`；
- `7.6479%` triangle-count reduction。

### Slide 8: Per-Scene Table

放 full9 per-scene compact table，不必放所有小数；保留 dPSNR/dLPIPS/tri red。

### Slide 9: Qualitative Result

放：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

重点讲 local crop 和 error reduction。

### Slide 10: Ablation / Representation-Level Lessons

放 v52/v56/v64/v75-v77 summary：

- support/capacity helps but small；
- stricter gates improve safety；
- residual representation capacity 是主要瓶颈。

### Slide 11: Stage2 Shape Prior

放 v3 vs v4 epoch50 表：

- chamfer improves；
- filled IoU improves；
- gate not yet pass；
- not headline。

### Slide 12: Limitations and Next Step

三点：

1. bake guarded ELA into persistent surface representation；
2. target-view support certificate；
3. strict paper-protocol rerun and broader cross-scene validation。

---

## 14. Mentor 可能会问的问题

### Q1: 这是不是只是图像后处理？

回答：

> 不是无约束 image filter。修复信号来自训练视角 residual，但只有能通过 surface correspondence、face/bin support、visibility 和 policy-val risk 的区域才会被迁移。它绑定到 MeshSplatting 的显式 surface，而不是对最终图像直接做通用增强。

### Q2: 有没有 test leakage？

回答：

> 我们方法的 branch、alpha、support、fallback 都来自 train/policy-val evidence；held-out test GT 只用于最终报告。baseline selected-clean 用 test envelope 是为了给 clean MeshSplatting 更强 comparator，不是用来选我们方法。

### Q3: 为什么不直接说全面超过 MeshSplatting paper？

回答：

> 当前最严格、最可复现的主 claim 是本地 same-protocol selected-clean baseline。paper table 对齐可以作为辅助，但最终投稿前必须重新确认 split、metric、checkpoint 和 mask/crop 完全一致。

### Q4: 为什么室外视觉提升不明显？

回答：

> 室外 full-frame 信息量大，局部 residual 修复缩小后不显眼；所以定性展示应该用 crop 和 error map。指标上 outdoor scenes 仍然 strict win，但视觉展示要选择局部纹理、边缘和残差集中的区域。

### Q5: representation-level 方法为什么收益小？

回答：

> 当前 v52-v83 说明简单 face/bin atlas、local prior、bin-gain certificate 的容量和泛化证据仍不够。v82/v83 已经开始把方向转到 patch-mixture teacher basis、capacity-prerank 这种更强 residual function / support selector，但 hard-triad 也说明 train/policy-val positive signal 仍可能和 held-out SSIM/LPIPS 泛化不一致。它们有价值，因为系统性排除了“继续调 blend/alpha/cap 就能突破”的假设。下一步需要更强 residual basis 和更可靠的 target-view support-certified policy。

### Q6: 当前工作距离顶会还有什么缺口？

回答：

> 主结果已经说明 MeshSplatting 可以被 evidence-certified repair/compaction 明显增强。顶会级主稿还需要把 render-time adapter 风险降下来：要么把 ELA bake 到 persistent representation，要么给 adapter 提供更强理论/实验解释；同时需要完成 paper-protocol final rerun 和更宽数据集验证。

---

## 15. 可引用证据清单

### 主结果

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv
```

### 当前证据 manifest

```text
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
outputs/carnet/spcarnet/current_evidence_manifest_20260624.json
```

Manifest 当前状态：

| item | value |
|---|---:|
| total items | `18` |
| existing | `18` |
| missing required | `0` |
| missing optional | `0` |

注意：manifest 只验证文件存在、大小和 hash，不等价于 paper-loop complete。

### 定性图

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_full9_crop_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
```

### 表示级诊断

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_patchmix_teacher_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_triad_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
```

### Stage2 shape prior

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
```

---

## 16. 当前状态判断

### 可以汇报的成熟结论

- Phase-J 当前是最强可审计 endpoint。
- 相对本地 selected clean MeshSplatting，full9 `9 / 9` scene-level strict RGB wins。
- 平均提升 `+1.331084` PSNR、`+0.034702` SSIM、`-0.063359` LPIPS。
- 同时平均删去 `7.6479%` triangles；该数字只代表 triangle-count reduction。
- per-view strict wins `244 / 246`，两个非 strict per-view 来自 `treehill` 和 `room`。
- geometry-safe `9 / 9`。
- 表示级最新 backup：v82 capacity-prerank 在 `counter` 上以极小 margin 严格超过 v56/v64/v79/v80 anchor，但 hard-triad raw policy 没有严格超过 v64，不能 promoted；v83 counter 进一步改善 PSNR/LPIPS，但 SSIM 微退，同样不能 promoted。

### 不应夸大的部分

- 不应说 fully baked representation 已经完成。
- 不应说所有几何指标都全面 strict win。
- 不应把 v64-v77 包装成主结果。
- 不应把 v82 capacity-prerank 的 counter-only tiny win 包装成 full9 或顶会级闭环；hard-triad 已明确显示 raw policy 仍有泛化短板。
- 不应把 v83 的 PSNR/LPIPS counter win 包装成 strict win；它有 SSIM 微退。
- 不应把 Stage2 v4 包装成已通过 shape gate。
- 不应把 paper-table 对齐当作已最终闭合。

### 最后一句总结

> 当前 SPCarNet 已经有一个强而可信的 evidence-certified MeshSplatting enhancement 结果；它足够支撑 mentor 汇报中的主线和实验亮点。若目标是顶会终稿，下一步必须把 render-time ELA 的收益进一步内化到 persistent representation，或者给 adapter 级 endpoint 做更强的理论和跨协议实验闭环。

---

## 17. 可直接拆成 PPT 的逐页文案

这一节是给实际做 slides 用的短版，不替代前面的技术报告。

### Slide 1: Title

标题：

```text
SPCarNet: Self-Auditing MeshSplatting for Evidence-Certified Repair and Compaction
```

页面主句：

```text
We turn a trained MeshSplatting checkpoint into a self-auditing surface system.
```

讲稿重点：

> MeshSplatting 本身已经很强，所以我们的目标不是从零替代它，而是在它训练完成后，用训练视角中可追溯到 surface 的证据判断哪里可以安全压缩、哪里可以安全修复、哪里必须回退。

### Slide 2: Problem

页面 bullet：

- clean MeshSplatting still has stable local residuals；
- some triangles are low-risk redundant geometry；
- naive residual transfer may fail on out-of-trajectory views。

讲稿重点：

> 显式 mesh/surface 表示的优势是可定位、可审计。问题是原始 checkpoint 没有利用训练视角 residual 来做后验自检。

### Slide 3: Core Idea

页面 pipeline：

```text
MeshSplatting checkpoint
  -> surface evidence cache
  -> safe triangle compaction
  -> guarded residual repair
  -> train-only policy gate
  -> held-out evaluation
```

讲稿重点：

> 所有决策先绑定到 surface address，再由 support、visibility、risk 和 policy-val gate 判断是否生效。

### Slide 4: Surface Evidence

页面 bullet：

- face/bin address；
- train residual；
- multi-view support；
- sign consistency；
- policy-val L1/SSIM/tail risk。

讲稿重点：

> Evidence cache 是这项工作的核心数据结构。它把训练视角从优化样本变成 surface-level audit signal。

### Slide 5: Geometry Compaction

页面数字：

```text
mean triangle-count reduction: 7.6479%
geometry-safe scenes: 9 / 9
```

讲稿重点：

> 这里的目标不是极限压缩，而是在质量不崩的前提下删掉低风险 triangles。报告时只说 triangle-count reduction，不说完整 rate-distortion 已闭合。

### Slide 6: Guarded ELA

页面公式：

```text
I_ours = I_compact + alpha * evidence_weighted_residual
```

页面 bullet：

- residual comes from train/policy-val views；
- transfer is surface-addressed；
- weak evidence triggers fallback。

讲稿重点：

> 这不是通用图像滤波。修复只发生在 surface correspondence 和 evidence certificate 都支持的位置。

### Slide 7: Main Quantitative Result

页面主表：

| metric | result |
|---|---:|
| scene strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |

讲稿重点：

> 主 claim 是相对本地同协议 selected clean MeshSplatting baseline。这个口径最公平、最可追溯。

### Slide 8: Per-Scene Result

页面建议：

- 放第 7 节 per-scene 表的简化版；
- 每行保留 scene、dPSNR、dLPIPS、triangle reduction；
- 用颜色标出 `9 / 9` scenes 全部 RGB strict win。

讲稿重点：

> 室内外都有效，但视觉显著性不同。室外更需要 crop 和 error map 才容易看清。

### Slide 9: Qualitative Evidence

页面图片：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

讲稿重点：

> Full-frame 缩小后差异不明显，因此主图用 held-out crop 和 error reduction map。Caption 要说明这是 showcase，不是随机抽样。

### Slide 10: Ablation Lessons

页面 bullet：

- v52: capacity-aware policy gives safe small gains；
- v56: face-alpha reliability guard blocks risky local edits；
- v64: fixed auto bin-alpha is stable but tiny；
- v75-v77: stronger local priors/certificates did not break through。

讲稿重点：

> 这些实验的价值在于定位瓶颈：问题不再是简单调 alpha 或 support cap，而是 residual representation capacity 和 target-view generalization certificate。

### Slide 11: Stage2 Shape Prior

页面表：

| metric | v3 | v4 epoch50 | interpretation |
|---|---:|---:|---|
| Chamfer | `0.0698447` | `0.0607328` | better |
| filled IoU | `0.5531548` | `0.5683319` | better |
| shell IoU | `0.9112785` | `0.8783072` | worse |
| gate | pass | fail | not headline |

讲稿重点：

> Stage2 是真实训练路线，不是调参。但它目前只能作为下一阶段几何先验证据，不能包装成主结果。

### Slide 12: Honest Status and Next Step

页面 bullet：

- current endpoint: strong, auditable Phase-J；
- not complete: strongest RGB gains still render-time；
- next: bake repair into persistent surface representation；
- next: paper-protocol rerun and broader validation。

结尾句：

> SPCarNet currently proves that MeshSplatting can be systematically enhanced by surface evidence. The next paper-level milestone is to internalize the repair into the representation while keeping the same auditability.
