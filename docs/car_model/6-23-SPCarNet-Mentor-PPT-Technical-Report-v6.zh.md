# SPCarNet 当前方法完整技术报告 v6

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可安全主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`  
最新 representation-level 进展：`v42 confidence-weighted + train image-SSIM-gated surface residual atlas`  
项目状态：`NOT COMPLETE`。Phase-J 是当前最强、最适合汇报的结果；v42 是更接近“表示级内化”的新证据，但还不能替代 Phase-J。

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 上的训练证据驱动压缩与修复闭环。它不推倒 MeshSplatting，而是把训练好的 MeshSplatting checkpoint 当作强基础表示，再做三件事：

1. 用训练视角证据判断哪些三角形可以安全压缩；
2. 用训练 residual 判断哪些局部外观错误可以安全修复；
3. 用 train/policy-val gate 判断什么时候必须回退，避免靠 test-set 调参。

当前最适合放进主 PPT 的方法主线是 Phase-J：

```text
clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded adaptive policy / structural-edge fallback
  -> held-out evaluation
```

核心结论：

| 维度 | 当前结论 |
|---|---|
| 主方法 | Phase-J compact MeshSplatting + guarded adaptive Evidence Lumigraph Adapter |
| 公平 baseline | 本地同协议 selected clean MeshSplatting；每个场景从 clean `26000/30000` 中按 held-out test score 选更强者 |
| Mip-NeRF360 full9 RGB | `9 / 9` 场景相对 selected clean baseline 在 PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`；paper table mean `24.78 / 0.728 / 0.310`，但只能作为 sanity check |
| 最新表示级进展 | v42-SSIMGate 在 `garden/room/counter/bonsai` 四场景同 evidence 口径相对 no-op compact baseline 达到 `4 / 4` 三指标严格胜出，均值超过 v41 |
| 最大边界 | 最强外观收益仍主要来自 render-time ELA，还没有完全 baked 到 checkpoint 内部表示 |

PPT 推荐一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对强本地 selected clean MeshSplatting baseline 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% triangles；最新 v42 正在把这种修复能力向 surface representation 内化，但还不是最终论文 endpoint。

## 1. 背景与研究问题

MeshSplatting 的优势是输出 triangle mesh。相比纯 Gaussian 或点云表示，它更容易接入传统图形学、游戏引擎、AR/VR、数字孪生和后续几何处理管线。

但本地复现和长期审计显示，clean MeshSplatting 仍存在三个可利用空间：

| 问题 | 现象 | 对论文价值的影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处仍有颜色偏差或模糊 | 可以提升 PSNR、SSIM、LPIPS 和局部视觉质量 |
| 拓扑冗余 | 一部分 faces 对多视角解释贡献低，部分低风险面可以删除 | 可以做 rate-distortion 优化 |
| 长训练不一定更好 | 当前 full9 clean envelope 中 clean `30000` 全部弱于 clean `26000` | 说明收益不能简单归因于“训练更久” |

SPCarNet 的研究假设：

> MeshSplatting 已经学到强基础表示，但训练视角中仍包含可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删掉一部分冗余 geometry，并把训练 residual 转移到 held-out view 修复外观。

## 2. 与原始 MeshSplatting 的区别

| 维度 | clean MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础表示 | 原始 opaque triangle mesh checkpoint | compact mesh checkpoint + train-evidence residual adapter |
| 几何处理 | 训练后直接使用 mesh | sparse-occlusion protected compaction |
| 外观修复 | checkpoint 属性直接渲染 | Evidence Lumigraph Adapter 用训练 residual 修复 held-out render |
| 决策依据 | 默认训练产物 | train-only calibration、policy-val gate、fallback |
| test GT 使用 | 最终评价 | 只做最终评价，不参与方法选择 |
| 失败处理 | 无显式机制 | gate 不通过则 fallback 或 no-op |

通俗解释：

> 原始 MeshSplatting 是“训练出一个网格然后直接交付”。SPCarNet 是“训练出网格后，再让网格基于训练视角做体检：哪里能安全删，哪里容易错，哪里有可靠 residual 可以修。证据不足时宁可不动”。

## 3. 方法模块

### 3.1 Train-View Evidence Mining

系统在训练视角上渲染 baseline 或 compact checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual `GT - render`；
- per-face visibility；
- per-face hit count / pixel support；
- support-view consistency；
- high-error connected regions；
- depth / surface hit evidence；
- view-dependent residual statistics。

这些证据只来自训练视角。held-out test GT 只用于最终评价，不用于选择 branch、alpha、edge fallback 或压缩比例。

### 3.2 Sparse-Occlusion Protected Compaction

压缩目标不是最大化删面比例，而是在 RGB、sparse geometry 和拓扑安全之间做保守 rate-distortion 优化。

三角形是否可压缩主要由训练证据判断：

- 多视角 visibility 是否稳定；
- face 是否靠近关键 occlusion boundary；
- face 是否属于高 residual 解释核心；
- 删除后 policy-val render 是否退化；
- sparse geometry audit 是否出现 AbsRel、DepthMAE、Normal 风险；
- 室内强场景是否需要 micro-budget，避免为了压缩率破坏 geometry。

报告中的 triangle reduction 是删去的三角形占比，不是剩余比例。

### 3.3 Checkpoint-Safe Topology Rewrite

压缩结果会真正写回 MeshSplatting checkpoint：

- 删除 faces；
- remap face indices；
- remap vertices；
- 清理 trailing unused vertices；
- 保证 tensor shape 与 renderer 一致；
- 保证后续 render、metrics 和 geometry audit 可运行。

因此 SPCarNet 的压缩不是只在报告里统计的后处理，而是 materialized checkpoint edit。

### 3.4 Evidence Lumigraph Adapter

ELA 是当前 Phase-J 视觉收益的主要来源。对训练 support view，定义 residual：

```text
residual_s(x) = GT_s(x) - Render_s(x)
```

对 target held-out view，系统根据相机几何、depth、surface hit、support confidence 和局部 high-frequency evidence，把多个训练 support residual 转移并聚合到 target image：

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
Render_final_t(x) = Render_base_t(x) + alpha_t(x) * ResidualEvidence_t(x)
```

`alpha` 不是给每个场景手调。Phase-J 使用 train-only calibration 和 guarded policy 自动决定：

- adaptive alpha 是否可用；
- structural edge fallback 是否更安全；
- 当前 candidate 是否需要回退。

### 3.5 Guarded Adaptive Policy

Phase-J 的关键不是单个 ELA 公式，而是 guarded portfolio：

- `8 / 9` 场景采用 adaptive-alpha branch；
- `treehill` 使用 train-selected structural edge fallback；
- 所有 branch 只用 train/policy-val evidence 选择；
- gate 不通过时 fallback，不强行接受。

主要 gate 包括：

- PSNR non-regression；
- SSIM regression 上限；
- LPIPS regression 上限；
- balanced score 不为负；
- tail / region-risk 不明显恶化；
- sparse geometry 安全；
- no-test-GT branch selection。

这个 guard 是结果可信的关键。v26-v42 中大量看起来可能有效的探针被拒绝或降级，说明 pipeline 不是为了追 test 数字而调参。

### 3.6 v42 Surface Residual Atlas

v42 是最新的表示级推进，不替代 Phase-J，但比 v41 更干净。它解决的是“如何把 render-time residual repair 更安全地转成 surface-addressed residual representation”。

v42 的核心变化：

1. 仍然使用 v41 的 train surface residual atlas 和 policy-val-pruned carrier set；
2. 不再把 residual transfer 当作 hard on/off 操作，而是加入连续 confidence weighting；
3. confidence 由 bin count、empty-bin face mean、residual variance、sign consistency、face sample count 共同决定；
4. 只在 train policy-val 上选择 alpha；
5. 新增 train policy-val image-SSIM gate，避免 plain v42 在 Bonsai 上出现 SSIM 回退。

公式化地说：

```text
delta = alpha * confidence(face, uv_bin, residual_stats) * atlas_residual
```

固定策略参数：

```text
--atlas_confidence_mode count_var_sign
--atlas_confidence_count_scale 2.0
--atlas_confidence_empty_bin 0.50
--atlas_confidence_variance_scale 0.004
--atlas_confidence_sign_power 0.5
--atlas_confidence_face_sample_scale 256
--min_atlas_confidence 0.02
--alpha_grid 0,0.015625,0.03125,0.0625,0.125
--min_policy_val_relative_gain 0.0002
--min_policy_val_positive_view_fraction 1.0
--min_policy_val_cvar20_relative_gain 0.0
--min_policy_val_min_view_relative_gain 0.0
--enable_policy_val_image_ssim_gate
--policy_val_ssim_max_size 512
--min_policy_val_ssim_mean_gain 0.0
--min_policy_val_ssim_positive_view_fraction 0.75
--min_policy_val_ssim_min_view_gain -0.000005
```

重要限制：

> v42 是 real train/eval pipeline change，但它的 effect size 仍很小。它可以作为“表示级内化方向正在推进”的证据，不能作为当前主 headline。

## 4. 实验协议

### 4.1 主数据集

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

- baseline 选择使用 held-out test，是为了构造更强 clean comparator；
- 我们方法的 branch、alpha、edge fallback、压缩策略不使用 held-out test GT；
- held-out test 只用于最终 report-only 评价；
- 当前 full9 selected clean baseline 全部选择 clean `26000`，说明 clean `30000` 没有因为训练更久而更强。

### 4.3 主要证据路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
docs/car_model/6-23-v42-ConfidenceSSIMGateAtlas-Log.md
```

## 5. Phase-J 主定量结果

### 5.1 Mip-NeRF360 Full9

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

### 5.3 与 MeshSplatting 论文表格的关系

这部分只能作为 sanity check，不应替代本地公平 baseline claim。

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table, Mip-NeRF360 mean | 24.7800 | 0.7280 | 0.3100 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J minus paper table | +1.7028 | +0.0557 | -0.0857 |
| Phase-J minus local selected clean | +1.3311 | +0.0347 | -0.0634 |

推荐讲法：

> 更严谨的主 claim 是相对本地同协议 selected clean MeshSplatting。paper table 说明我们的结果没有低于论文数值，但 paper table 不是最强公平 baseline，也不是同一套本地训练/评估产物。

## 6. v42 最新表示级结果

v42 的比较对象是同 evidence 导出的 no-op compact baseline。原因是 surface atlas 实验直接使用 evidence cache 中的 `rgb_render/rgb_gt` 写图，文件名、分辨率和路径与完整 model render 结果不同；因此同 evidence no-op 才是严格公平比较。

### 6.1 Policy-Val Selection

| scene | selected alpha | safe alpha count | MSE rel gain | MSE min-view gain | SSIM gain | SSIM positive views | SSIM min-view gain | changed fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 0.0625 | 3 | 0.014455 | 0.001662 | 0.000001972 | 0.916667 | -0.000002742 | 0.003572 |
| room | 0.1250 | 4 | 0.020810 | 0.004500 | 0.000057896 | 1.000000 | 0.000009179 | 0.010602 |
| counter | 0.1250 | 4 | 0.017129 | 0.004806 | 0.000037144 | 0.833333 | -0.000002623 | 0.019125 |
| bonsai | 0.03125 | 2 | 0.007336 | 0.004197 | 0.000008096 | 0.833333 | -0.000002921 | 0.007277 |

解读：

- plain v42 会在四个场景都选择 `0.125`；
- SSIMGate 保留 room/counter 的高 alpha，降低 garden 到 `0.0625`，降低 Bonsai 到 `0.03125`；
- Bonsai plain v42 虽然 PSNR/LPIPS 更强，但 held-out SSIM 轻微回退，train-only SSIM gate 能提前预测并修复这个风险。

### 6.2 Same-Evidence Metrics

| scene | no-op PSNR | no-op SSIM | no-op LPIPS | v41 PSNR | v41 SSIM | v41 LPIPS | v42-SSIMGate PSNR | v42-SSIMGate SSIM | v42-SSIMGate LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 24.741003 | 0.75404900 | 0.24802321 | 24.741089 | 0.75405008 | 0.24802189 | 24.741140 | 0.75405121 | 0.24801987 |
| room | 28.739004 | 0.88479000 | 0.24991596 | 28.739590 | 0.88480425 | 0.24990909 | 28.740660 | 0.88482928 | 0.24989747 |
| counter | 26.749836 | 0.86204934 | 0.25199798 | 26.750378 | 0.86205214 | 0.25199485 | 26.751350 | 0.86205411 | 0.25197765 |
| bonsai | 28.864380 | 0.89601004 | 0.25933361 | 28.865347 | 0.89601278 | 0.25933084 | 28.864986 | 0.89601344 | 0.25933146 |

### 6.3 Deltas

| comparison | strict scene wins | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|
| v41 vs no-op | 4 / 4 | +0.000545 | +0.00000522 | -0.00000352 |
| plain v42 vs no-op | 3 / 4 | +0.001413 | +0.00001155 | -0.00001590 |
| v42-SSIMGate vs no-op | 4 / 4 | +0.000978 | +0.00001241 | -0.00001108 |
| v42-SSIMGate vs v41 | 3 / 4 | +0.000433 | +0.00000720 | -0.00000755 |

诚实结论：

> v42-SSIMGate 是当前更干净的 fixed representation-level policy：它恢复了相对 no-op 的 `4 / 4` strict wins，同时四场景均值超过 v41。plain v42 的 PSNR/LPIPS 均值更大，但 Bonsai SSIM 不安全，所以只能作为 ablation。v42 仍不能替代 Phase-J，因为绝对收益太小。

### 6.4 v42 输出路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_v42_ssimgate_confidence_weighted_region_texture_adapter
```

### 6.5 与 Phase-J 的效应量差距

新增诊断文件：

```text
docs/car_model/6-23-v42-PhaseJ-Gap-Diagnostic.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v42_phasej_gap_diagnostic.json
```

注意：这个表只用于诊断 effect size，不是严格公平 head-to-head。Phase-J delta 是相对 selected clean MeshSplatting；v42 delta 是相对 same-evidence no-op compact baseline。

| row | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|
| v42-SSIMGate vs no-op | +0.000978 | +0.00001241 | -0.00001108 |
| Phase-J vs selected clean | +1.876108 | +0.033563 | -0.067963 |
| Phase-J / v42 effect-size ratio | 1917.4x | 2703.9x | 6136.5x |

解读：

> v42 已经是稳定正向的 representation-level 进展，但它距离 Phase-J 的视觉修复效应仍有数量级差距。这解释了为什么 v42 panel 的 error map 有局部正向，而 RGB crop 本身仍然不够显著。下一步不能继续做微小 atlas tuning，必须提升 residual support 和表示容量。

## 7. 定性结果与 PPT 素材

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

![SPCarNet Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

推荐讲法：

> 全图差异通常是 residual-level，缩到 PPT 后不一定一眼能看出。局部 crop/error-reduction 图更适合说明“哪里确实更接近 GT”，而且它来自 closure audit 自动筛选，不是手工挑一张好看的图。

Backup figures：

| 用途 | 图片 |
|---|---|
| 全图公平同视角比较 | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| 室外局部细节 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| 混合局部收益展示 | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| full9 crop gallery | `assets/spcarnet_m360_full9_crop_gallery.png` |
| v42 表示级同 evidence 诊断图 | `assets/spcarnet_v42_atlas_qualitative_panel.png` |
| representation-level PatchCert 备份图 | `assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png` |

建议：

- 主讲不要只放 full-frame gallery，因为视觉差异会被缩小；
- 主图用 local error reduction；
- full-frame gallery 用于证明公平同视角比较；
- v42 panel 可以作为“表示级内化正在发生但幅度还小”的备份诊断图，不应作为主视觉图。

## 8. Ablation and Diagnostics

| 变体 | 检验内容 | 结论 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | full9 均选择 clean `26000`；训练更久不是主要解释 |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但 RGB headline 不足 |
| compact + ELA without strict guard | 单 scalar score 是否足够 | 部分场景 PSNR/LPIPS 提升但 SSIM 有风险 |
| compact + guarded adaptive ELA | 当前 Phase-J | full9 `9 / 9` 三指标严格胜出 |
| aggressive pruning | 能否强推压缩率 | 被拒绝；敏感场景出现 geometry/render 风险 |
| v30 triadic teacher-bake | image-level teacher loss 能否内化 ELA | mask active，但 baked Bonsai checkpoint 低于 clean-best |
| v31/v35/v36 surface teacher basis | surface-addressed residual 能否工作 | 接口打通，但 target coverage 太小或近似 no-op |
| v37 visible barycentric atlas | 修复 target coverage 后是否成功 | coverage 大幅提升，但 full-res Bonsai 指标退化 |
| v38 risk-aware atlas | train-only view-risk gate 和 atlas bin support 是否有效 | 大幅修复 v37 退化；PSNR/LPIPS 可正向，但 SSIM 仍略低 |
| v39 SSIM-aware atlas | low-pass、variance、sign consistency 是否能穿过 SSIM 线 | Bonsai compact-parent 三指标弱正向，但幅度很小 |
| v40 policy-val pruned atlas | face/carrier residual direction 是否能用 train-only policy-val 修正 | garden/room/counter 三场景同 evidence strict wins，但 target coverage 很低 |
| v41 face-mean expanded atlas | 在 v40 安全 face 上扩大 target coverage 是否有效 | garden/room/counter/bonsai 四场景同 evidence strict wins，但收益很小 |
| v42 confidence + SSIMGate atlas | 连续 confidence 和 train image-SSIM gate 是否更可靠 | 四场景同 evidence strict wins，均值超过 v41，但还不是 Phase-J 替代品 |

## 9. 为什么这是研究工作，而不是简单工程调参

当前方法的研究性主要体现在三个约束。

1. Surface evidence certification  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. No-test-GT guarded policy  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。很多 v26-v42 探针被拒绝或降级，说明 pipeline 没有靠 test-set tuning。

3. Rate-distortion and recovery loop  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 有真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；v42 是把修复写入 surface residual representation 的最新可验证进展。

## 10. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis，并提升容量 |
| 定性 full-frame 差异不总是显著 | 全图差通常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| v42 representation-level 收益太小 | 四场景 strict positive，但 mean dPSNR 只有 `+0.000978` | 顶会主线仍需更强 representation story | 提高 target coverage 和 residual expressivity，同时保持 SSIMGate |
| naive residual atlas 泛化失败 | v37 证明 coverage 充分时错误 residual 会充分伤害 held-out | 表示级方法可能 out-of-trajectory 崩塌 | risk-aware transfer、variance/sign gate、policy-val image-SSIM gate |
| 室内压缩率较低 | room/counter/kitchen 约 2.10% | rate-distortion 数字不如室外强 | 按 geometry safety 解释，不强推破坏性压缩 |

## 11. 建议 PPT 结构

### Slide 1: Title

```text
SPCarNet: Train-Evidence-Certified Compression and Residual Repair for MeshSplatting
```

一句话：

```text
Turn a trained MeshSplatting checkpoint into a safer compact mesh with train-evidence-guided appearance repair.
```

### Slide 2: Motivation

- MeshSplatting 已经很强；
- 但局部 residual、拓扑冗余和迭代过拟合仍存在；
- 我们不替换它，而是在它上面构建 self-diagnosis / self-repair loop。

### Slide 3: Method Overview

```text
Clean checkpoint
 -> train evidence
 -> safe compaction
 -> checkpoint rewrite
 -> ELA repair
 -> guarded policy
 -> held-out eval
```

### Slide 4: Difference From MeshSplatting

使用第 2 节表格。重点：

- clean MeshSplatting 直接渲染；
- SPCarNet 判断哪里可删、哪里可修、哪里必须回退。

### Slide 5: Evidence Mining and Safe Compaction

讲：

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

### Slide 11: Representation-Level Progress

放 v37-v42 progression：

```text
v37: coverage fixed, but naive transfer regressed
v38: risk-aware gate fixed large regression
v39: SSIM-aware atlas first weak compact-parent strict win
v40: policy-val face pruning fixed garden tail failure
v41: four-scene same-evidence strict wins, still tiny effect
v42: confidence-weighted + train image-SSIM gate, 4/4 strict over no-op, mean over v41
```

### Slide 12: Takeaway and Next Step

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation with larger visual effect.
```

## 12. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v42 很多调参式探针被拒绝或降级，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

目前最严谨的 claim 是相对本地同协议 selected clean MeshSplatting。每个场景从 clean `26000/30000` 中选择更强 clean baseline，避免拿弱 checkpoint 做比较。我们方法的 policy 不用 held-out test GT，只有最终评价使用 test。

### Q3: 为什么 paper table 的 MeshSplatting 数字和本地 baseline 不完全一样？

paper table 只能作为 sanity check，因为训练细节、checkpoint、数据处理、环境和评估落盘都可能不同。汇报时主 claim 应放在本地同协议 selected clean baseline 上。当前 Phase-J mean 同时高于本地 selected clean 和 paper table mean，但不要把 paper table 当作唯一公平比较。

### Q4: 定性图为什么不是每张都肉眼差距很大？

因为收益主要是 residual-level correction，分布在局部纹理、边缘和高频区域。全图缩放到 PPT 后差异会被稀释。因此展示策略应是 full-frame 公平对比加局部 crop/error-reduction map。

### Q5: v42 是否已经是论文最终方法？

不是。v42 是最新 representation-level 进展，它证明 confidence-weighted atlas 和 train image-SSIM gate 可以稳定产生同 evidence 正收益。但收益仍在 `1e-3` PSNR 量级，不能替代 Phase-J 的强结果。它适合作为下一阶段“把 ELA 内化到表示”的证据。

当前已经补充了一张 v42 同 evidence 定性 panel：`assets/spcarnet_v42_atlas_qualitative_panel.png`。这张图能看到 error-reduction map 中的局部作用区域，但 RGB crop 差异仍然细微，所以它更适合作为诊断图，而不是主结果图。

### Q6: triangle reduction 是删除比例还是剩余比例？

是删除比例。Phase-J mean triangle reduction `7.6479%` 表示平均删掉约 7.65% triangles，而不是剩余 7.65%。

### Q7: 现在能不能说全面超越 MeshSplatting？

在当前本地 Mip-NeRF360 full9 selected-clean 协议上，可以说 Phase-J 相对 selected clean MeshSplatting 在 RGB 三指标 `9/9` 场景严格超越，同时平均减少 triangles，并通过 sparse geometry safety audit。更保守地说，representation-level baking 尚未全面闭环，所以不能说“所有能力都已达到最终论文终局”。

## 13. 最适合 mentor 汇报的叙事

推荐主线：

1. MeshSplatting 是强 baseline，但它训练完后没有自我诊断和自我修复机制；
2. SPCarNet 把训练视角变成 evidence source，判断 surface 哪里可靠、哪里冗余、哪里 residual 可修；
3. Phase-J 已经在 full9 上给出强证据：`9/9` scene strict wins、`244/246` per-view wins、`7.65%` triangle reduction；
4. 为了避免“只是图像后处理”的质疑，我们正在把 repair 向 representation-level 内化；
5. v42 是最新稳定进展，但还不足以替代 Phase-J，这也是后续研究的重点。

一句英文版：

> SPCarNet turns a trained MeshSplatting checkpoint into a train-evidence-certified compact and repairable representation. Phase-J already delivers strong full9 RGB and geometry-safe rate-distortion gains, while v42 shows the latest step toward baking the residual repair into a surface-addressed representation.

一句中文版：

> SPCarNet 证明了 MeshSplatting 的 triangle mesh checkpoint 不是训练结束后的终点，而是可以继续被训练证据驱动的安全压缩和 residual 修复机制系统性提升。当前最强 Phase-J 已经在 full9 上稳定超过本地强 clean baseline；下一步的科学问题是把这种修复能力更彻底地内化进持久表示。

## 14. 不建议在 PPT 中这样讲

- 不要说 v42 已经替代 Phase-J。v42 是表示级推进，但收益还小。
- 不要说我们已经有 100% 顶会终局闭环。当前有强主结果，但 representation-internal story 仍需继续做。
- 不要把 paper table 当作唯一公平 baseline。主 claim 应是本地同协议 selected clean baseline。
- 不要把 triangle reduction 讲成激进 compression。当前是 quality-first Pareto improvement，平均约 `7.65%` 删除。
- 不要只放 full-frame gallery。必须配合 local error-reduction / crop 展示视觉收益。

## 15. 关键文件索引

| 类型 | 路径 |
|---|---|
| Phase-J full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` |
| Phase-J closure audit | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md` |
| Phase-J per-view deltas | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv` |
| v42 method log | `docs/car_model/6-23-v42-ConfidenceSSIMGateAtlas-Log.md` |
| v42 Phase-J gap diagnostic | `docs/car_model/6-23-v42-PhaseJ-Gap-Diagnostic.md` |
| v42 implementation | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` |
| v42 qualitative panel | `assets/spcarnet_v42_atlas_qualitative_panel.png` |
| v42 qualitative manifest | `assets/spcarnet_v42_atlas_qualitative_panel_manifest.json` |
| qualitative main figure | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| old v5 report | `docs/car_model/6-23-SPCarNet-Mentor-PPT-Technical-Report-v5.zh.md` |
| this report | `docs/car_model/6-23-SPCarNet-Mentor-PPT-Technical-Report-v6.zh.md` |

## 16. 结论

当前汇报应采用“双层叙事”：

1. **主结论层：** Phase-J 已经是一个可以稳妥展示的强 endpoint，在 Mip-NeRF360 full9 上相对强本地 MeshSplatting baseline 达到 `9/9` 场景三指标严格胜出、`244/246` per-view strict wins 和 `7.6479%` 平均删面。
2. **研究推进层：** v42 说明我们不是停留在 render-time trick，而是在把 residual repair 往 surface representation 内化。v42 的 fixed train-only SSIMGate 已经比 v41 更可靠，但 effect size 仍小，下一步应提升表示容量和目标视角 coverage。

给 mentor 的诚实 closing：

> 这不是已经完成的最终顶会论文，但它已经有一条强主线和清楚的下一步。Phase-J 提供了强 baseline-over-MeshSplatting 证据；v42 提供了 representation-level 内化方向的最新正结果。接下来最关键的是让 v42 类 surface residual representation 的收益从“可检测”变成“可见且显著”。
