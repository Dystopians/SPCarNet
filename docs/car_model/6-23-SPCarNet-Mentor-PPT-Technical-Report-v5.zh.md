# SPCarNet 当前方法完整技术报告 v5

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可安全主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`  
当前最新 representation-level 进展：`v41 face-mean expanded policy-val-pruned surface residual atlas`  
项目状态：`NOT COMPLETE`，Phase-J 是当前强结果，v41 是更干净的表示级推进但还不是论文终局

更新提示：v5 报告写成后又完成了 v42 confidence-weighted + train image-SSIM-gated atlas。最新 representation-level 结果见 `docs/car_model/6-23-v42-ConfidenceSSIMGateAtlas-Log.md`；v42-SSIMGate 在 `garden/room/counter/bonsai` 四场景同 evidence 口径下相对 no-op compact baseline 达到 `4 / 4` 三指标严格胜出，并在均值上超过 v41，但仍不是 Phase-J 替代品。

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 上的训练证据驱动压缩与修复闭环。它不把 MeshSplatting 推倒重来，而是把训练好的 MeshSplatting checkpoint 当作基础表示，再做三件事：

1. 用训练视角证据判断哪些三角形可以安全压缩；
2. 用训练 residual 判断哪些局部外观错误可以安全修复；
3. 用 train/policy-val gate 判断什么时候必须回退，避免靠 test-set 调参。

当前最适合汇报的主线是 Phase-J：

```text
clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded adaptive policy / edge fallback
  -> held-out evaluation
```

核心结论：

| 维度 | 当前结论 |
|---|---|
| 主方法 | Phase-J compact MeshSplatting + guarded adaptive Evidence Lumigraph Adapter |
| 公平 baseline | 本地同协议 selected clean MeshSplatting；每个场景从 clean `26000/30000` 中只按 held-out test score 选择更强者 |
| Mip-NeRF360 full9 | `9 / 9` 场景相对 selected clean baseline 在 PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`；paper table mean `24.78 / 0.728 / 0.310`，但这只能作为 sanity check |
| 最新表示级进展 | v41 在 `garden/room/counter/bonsai` 四场景同 evidence 口径下均严格超过 no-op compact baseline，但收益仍很小 |
| 最大边界 | 最强外观收益仍主要来自 render-time ELA，还没有完全 baked 到 checkpoint 内部表示 |

PPT 推荐一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% triangles；但最强外观收益仍来自 render-time ELA，最新 v41 正在把修复内化到 surface representation。

## 1. 背景与研究问题

MeshSplatting 的优势是把神经重建结果表达成 triangle mesh。相比纯 Gaussian 或点云表示，它更容易进入传统图形学、游戏引擎、AR/VR、数字孪生和后续几何处理管线。

但本地复现和长期审计显示，clean MeshSplatting 仍存在三个可利用空间：

| 问题 | 现象 | 对论文价值的影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处仍有颜色偏差或模糊 | 可以提升 PSNR、SSIM、LPIPS 和局部视觉质量 |
| 拓扑冗余 | 一部分 faces 对多视角解释贡献低，部分低风险面可以删除 | 可以做 rate-distortion 优化 |
| 长训练不一定更好 | 当前 full9 clean envelope 中 clean `30000` 全部弱于 clean `26000` | 说明收益不能简单归因于“训练更久” |

SPCarNet 的核心研究假设：

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

报告里的 triangle reduction 是删去的三角形占比，不是剩余比例。

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

这个 guard 是结果可信的关键。v26-v41 中大量看起来可能有效的探针被拒绝或降级，说明 pipeline 不是为了追 test 数字而调参。

### 3.6 v41 Surface Residual Atlas

v41 是最新的表示级推进，不替代 Phase-J，但解决了 v37-v40 中几个具体瓶颈：

1. v37 修复 target barycentric coverage 后，naive atlas 会把错误 residual 充分贴到 held-out view，导致指标退化；
2. v38 加入 risk-aware alpha gate，修复大退化，但 SSIM 仍不稳定；
3. v39 加入 low-pass、variance、sign consistency，第一次在 Bonsai 上弱正向超过 compact parent；
4. v40 加入 train-only policy-val face/carrier pruning，修复 garden robust-gate 失败；
5. v41 保留 policy-val-pruned carriers，并允许 retained face 的 face-mean residual 扩展到未观测 UV bin，提高 target changed fraction。

v41 的核心机制：

```text
train evidence residual atlas
  -> policy-val face contribution audit
  -> remove faces with unsafe transfer direction
  -> robust view-tail gate
  -> retained-face face-mean coverage expansion
  -> same-evidence held-out evaluation
```

关键点：

- pruning 和 alpha selection 只用 train/policy-val evidence；
- held-out test GT 只用于最终 same-evidence metrics；
- 它是 representation-level method change，是真实写出新的 render result；
- 目前收益太小，不能替代 Phase-J headline。

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
docs/car_model/6-23-v40-PolicyValPrunedAtlas-Garden-Log.md
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

## 6. v41 最新表示级结果

v41 的比较对象是同 evidence 导出的 no-op compact baseline。原因是 surface atlas 实验直接使用 evidence cache 中的 `rgb_render/rgb_gt` 写图，文件名、分辨率和路径与完整 model render 结果不同；因此需要同 evidence no-op 才是严格公平比较。

### 6.1 v41 Policy-Val Audit

| scene | atlas faces | fit samples | policy-val samples | selected alpha | policy-val gain | positive views | CVaR20 gain | min-view gain | changed fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 319 | 95,747 | 32,994 | 0.015625 | 0.010628 | 1.000 | 0.004551 | 0.000269 | 0.003852 |
| room | 1,160 | 462,164 | 161,316 | 0.015625 | 0.006623 | 1.000 | 0.003421 | 0.001024 | 0.010996 |
| counter | 1,574 | 679,257 | 255,007 | 0.015625 | 0.006271 | 1.000 | 0.001645 | 0.000154 | 0.019906 |
| bonsai | 1,110 | 159,447 | 61,159 | 0.015625 | 0.014207 | 1.000 | 0.005240 | 0.004015 | 0.007670 |

解读：

- 四个场景 policy-val positive-view fraction 都是 `1.0`；
- min-view gain 均非负，说明没有被某个 policy-val view 明显拖垮；
- v41 比 v40 的 target changed fraction 更高，说明 coverage expansion 起效；
- 但 changed fraction 仍低，视觉效果不会像 Phase-J 那样明显。

### 6.2 v41 Same-Evidence Metrics

| scene | no-op PSNR | no-op SSIM | no-op LPIPS | v41 PSNR | v41 SSIM | v41 LPIPS | dPSNR | dSSIM | dLPIPS | strict win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | 24.741003 | 0.75404900 | 0.24802321 | 24.741089 | 0.75405008 | 0.24802189 | +0.0000858 | +0.00000107 | -0.00000133 | yes |
| room | 28.739004 | 0.88479000 | 0.24991596 | 28.739590 | 0.88480425 | 0.24990909 | +0.0005856 | +0.00001425 | -0.00000687 | yes |
| counter | 26.749836 | 0.86204934 | 0.25199798 | 26.750378 | 0.86205214 | 0.25199485 | +0.0005417 | +0.00000280 | -0.00000313 | yes |
| bonsai | 28.864380 | 0.89601004 | 0.25933361 | 28.865347 | 0.89601278 | 0.25933084 | +0.0009670 | +0.00000274 | -0.00000277 | yes |

四场景平均 v41 delta：

```text
mean dPSNR  = +0.0005450
mean dSSIM  = +0.00000522
mean dLPIPS = -0.00000352
```

诚实结论：

> v41 是目前最干净的 representation-level atlas 版本：它在四个场景上同 evidence 口径三指标严格正向，并且 policy-val tail gate 全部通过。但它的 effect size 很小，不能作为最终论文主结果；它更适合作为“我们正在把 render-time repair 内化为 surface representation”的证据。

### 6.3 v41 与 Phase-J 的关系

| 分支 | 优点 | 缺点 | 汇报定位 |
|---|---|---|---|
| Phase-J | full9 强结果，9/9 scene strict wins，244/246 per-view wins，视觉收益最明显 | 外观修复主要是 render-time ELA | 当前主讲结果 |
| v41 atlas | 真实 representation-level 改动，四场景同 evidence strict wins，train-only policy-val pruning 更干净 | 收益很小，视觉冲击弱，尚未超过 Phase-J | 下一阶段方法闭环证据 |

## 7. 定性结果与 PPT 素材

### 7.1 推荐主图

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

讲法：

> 全图差异通常是 residual-level，缩到 PPT 后不一定一眼能看出。局部 crop/error-reduction 图更适合说明“哪里确实更接近 GT”，而且它来自 closure audit 自动筛选，不是手工挑一张好看的图。

### 7.2 Backup Figures

| 用途 | 图片 |
|---|---|
| 全图公平同视角比较 | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| 室外局部细节 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| 混合局部收益展示 | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| full9 crop gallery | `assets/spcarnet_m360_full9_crop_gallery.png` |
| representation-level PatchCert 备份图 | `assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png` |

建议：

- 主讲不要只放 full-frame gallery，因为视觉差异会被缩小；
- 主图用 local error reduction；
- full-frame gallery 用于证明公平同视角比较；
- v41 暂时不要作为视觉主图，因为它的 changed fraction 和 metric delta 都很小。

## 8. Ablation and Diagnostics

| 变体 | 检验内容 | 结论 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | full9 均选择 clean `26000`；训练更久不是主要解释 |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但 RGB headline 不足 |
| compact + ELA without SSIM-peak guard | 单 scalar score 是否足够 | 部分场景 PSNR/LPIPS 提升但 SSIM 有风险 |
| compact + guarded adaptive ELA | 当前 Phase-J | full9 `9 / 9` 三指标严格胜出 |
| aggressive pruning | 能否强推压缩率 | 被拒绝；敏感场景出现 geometry/render 风险 |
| v30 triadic teacher-bake | image-level teacher loss 能否内化 ELA | mask active，但 baked Bonsai checkpoint 低于 clean-best |
| v31/v35/v36 surface teacher basis | surface-addressed residual 能否工作 | 接口打通，但 target coverage 太小或近似 no-op |
| v37 visible barycentric atlas | 修复 target coverage 后是否成功 | coverage 大幅提升，但 full-res Bonsai 指标退化 |
| v38 risk-aware atlas | train-only view-risk gate 和 atlas bin support 是否有效 | 大幅修复 v37 退化；PSNR/LPIPS 可正向，但 SSIM 仍略低 |
| v39 SSIM-aware atlas | low-pass、variance、sign consistency 是否能穿过 SSIM 线 | Bonsai compact-parent 三指标弱正向，但幅度很小 |
| v40 policy-val pruned atlas | face/carrier residual direction 是否能用 train-only policy-val 修正 | garden/room/counter 三场景同 evidence strict wins，但 target coverage 很低 |
| v41 face-mean expanded atlas | 在 v40 安全 face 上扩大 target coverage 是否有效 | garden/room/counter/bonsai 四场景同 evidence strict wins，但仍不是 Phase-J 替代品 |

## 9. 为什么这是研究工作，而不是简单工程调参

当前方法的研究性主要体现在三个约束。

1. Surface evidence certification  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. No-test-GT guarded policy  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。很多 v26-v41 探针被拒绝或降级，说明 pipeline 没有靠 test-set tuning。

3. Rate-distortion and recovery loop  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 有真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；v41 是把修复写入 surface residual representation 的最新可验证进展。

## 10. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis |
| 定性 full-frame 差异不总是显著 | 全图差通常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| v41 representation-level 收益太小 | 四场景 strict positive，但 mean dPSNR 只有 `+0.000545` | 顶会主线仍需更强 representation story | 提高 target coverage 和 residual expressivity，同时保持 policy-val gate |
| v37 residual atlas 泛化失败 | target coverage 已修，但 naive atlas 退化 | 错误 residual 被充分作用到 held-out | risk-aware transfer、variance/sign gate、target-risk proxy |
| 室内压缩率较低 | room/counter/kitchen micro-budget 约 2.10% | rate-distortion 数字不如室外强 | 按 geometry safety 解释，不强推破坏性压缩 |

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

放 v37-v41 progression：

```text
v37: coverage fixed, but naive transfer regressed
v38: risk-aware gate fixed large regression
v39: SSIM-aware atlas first weak compact-parent strict win
v40: policy-val face pruning fixed garden tail failure
v41: four-scene same-evidence strict wins, still tiny effect
```

### Slide 12: Takeaway and Next Step

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation with larger visual effect.
```

## 12. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v41 很多调参式探针被拒绝或降级，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

主 claim 是相对本地同协议 selected clean MeshSplatting。baseline 从 clean `26000/30000` 中只按 held-out test score 选更强者；我们方法的 branch、alpha、edge fallback、压缩策略不使用 test GT。

### Q3: 为什么 paper table 不是主 claim？

paper table 可以说明我们的结果没有低于论文数值，但不同实现、checkpoint、数据处理和评价脚本可能有差异。最严谨 claim 仍是本地同协议 selected clean baseline。

### Q4: 如果 Phase-J 主要是 render-time ELA，会不会不够像 representation method？

这是当前最重要边界。我们已经有真实 compact checkpoint 和 checkpoint-safe topology rewrite，但最强 RGB 修复仍来自 render-time adapter。v37-v41 正是在推进 representation-level baking；目前它们给出了清晰诊断和四场景弱正向结果，但还不能替代 Phase-J。

### Q5: 为什么 v37 coverage 修好了，指标反而变差？

因为 v37 解决的是“能否作用到目标像素”的问题，而不是“作用的 residual 是否一定泛化正确”。当 target changed pixels 大幅提升后，错误 residual 也会更充分地表现出来，所以指标退化。这说明下一步需要 risk-aware transfer，而不是继续盲目扩大 atlas。

### Q6: v41 已经四场景 strict positive，能不能说 representation-level 闭环完成？

不能。v41 的四场景同 evidence strict wins 很干净，但幅度只有 `+0.000545` mean PSNR、`+0.00000522` mean SSIM、`-0.00000352` mean LPIPS。它证明方向正确，但 effect size 和视觉显著性还不够。

## 13. 汇报时的诚实版本

建议主讲：

> 当前 Phase-J 在我们选定的 Mip-NeRF360 full9 口径下已经全面超过本地 clean MeshSplatting baseline，同时保留平均 7.65% triangle reduction 和 geometry safety。这是目前可以安全汇报的强结果。

主动说明：

> 但这还不是论文终局。最强外观收益仍是 render-time ELA；representation-level 分支已经从 v37 的 coverage 修复、v38 的 risk-aware gate、v39 的 SSIM-aware atlas、v40 的 policy-val pruning，推进到 v41 的四场景同 evidence strict positive。下一步要把 v41 的弱正向扩大为可见、可投稿的 representation-level gain。

## 14. 文件和结果索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
```

v41：

```text
docs/car_model/6-23-v40-PolicyValPrunedAtlas-Garden-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v41_facemean_expanded_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_v41_facemean_expanded_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_v41_facemean_expanded_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_v41_facemean_expanded_region_texture_adapter/results.json
```

same-evidence no-op baselines：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_evidence_noop_compact_baseline/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_evidence_noop_compact_baseline/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_evidence_noop_compact_baseline/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_evidence_noop_compact_baseline/results.json
```

实现脚本：

```text
scripts/car_model/ecsr_prune_region_carriers_by_policy_val.py
scripts/car_model/ecsr_export_evidence_rgb_baseline.py
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

## 15. 下一步建议

最有价值的下一步不是继续手动 alpha sweep，而是围绕 v41 的弱正向结果做真正 representation-level 加强：

1. 增加 target coverage，但必须由 policy-val safety gate 约束，避免 v37 式退化；
2. 做 carrier-aware / region-aware pruning，让空间连贯区域一起被保留或拒绝；
3. 引入 per-bin confidence，由 count、variance、sign consistency、normal/view angle、support-view coverage 决定；
4. 用 confidence-weighted residual amplitude 替代单一全局 alpha；
5. 把 policy-val objective 从 MSE 扩展到局部 SSIM/luminance-contrast proxy；
6. promotion gate 必须同时要求 same-evidence strict wins、full-protocol positive、可见定性收益，以及不牺牲 geometry/triangle reduction。

当前对 mentor 的定位：

> Phase-J 证明“训练证据驱动修复闭环”在本地公平 baseline 上有强结果；v41 证明“把修复内化到 surface representation”已经有跨场景弱正向路径。下一阶段的核心任务是把 v41 的可靠但微小收益扩大成多场景、可见、可投稿的 representation-level gain。
