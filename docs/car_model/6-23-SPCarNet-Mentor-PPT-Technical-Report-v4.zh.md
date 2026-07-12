# SPCarNet 当前方法完整技术报告 v4

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可安全主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`  
当前项目状态：`NOT COMPLETE`，Phase-J 可以作为阶段性强结果，representation-level 终局仍未闭环

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 上的训练证据驱动压缩与修复闭环。它不是替换 MeshSplatting，而是把训练好的 MeshSplatting checkpoint 当作基础表示，再用训练视角证据做三件事：

1. 判断哪些三角形可以安全压缩；
2. 判断哪些局部外观 residual 可以可靠转移到 held-out view；
3. 当证据不可靠时自动回退，而不是强行修改。

当前最适合 PPT 主讲的是 Phase-J：

```text
clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded adaptive policy / edge fallback
  -> held-out evaluation
```

核心结果：

| 维度 | 当前结论 |
|---|---|
| 主方法 | Phase-J compact MeshSplatting + guarded adaptive Evidence Lumigraph Adapter |
| 公平 baseline | 本地同协议 selected clean MeshSplatting；每个场景从 clean `26000/30000` 中只按 held-out test score 选择更强者 |
| Mip-NeRF360 full9 | `9 / 9` 场景相对 selected clean baseline PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 论文表格 sanity check | Phase-J mean `26.4828 / 0.7837 / 0.2243`，MeshSplatting paper table mean `24.78 / 0.728 / 0.310` |
| 最新 representation-level 进展 | v39 SSIM-aware atlas 第一次在 Bonsai 上对 compact parent 三指标严格微弱正向，但幅度极小，仍低于 selected clean 和 Phase-J |
| 最重要边界 | 最强外观收益仍主要来自 render-time ELA，还没有完全 baked 到 checkpoint 内部表示 |

PPT 推荐一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% triangles；但最强外观收益仍来自 render-time ELA，下一步要把修复进一步内化到 representation-level checkpoint。

## 1. 背景与问题定义

MeshSplatting 的优势是把神经重建结果表达为 triangle mesh。相比纯 Gaussian 或点云表示，它更容易进入传统图形学、游戏引擎、AR/VR、数字孪生和后续几何处理管线。

本地复现和长期审计显示，clean MeshSplatting 仍有三个可以改进的空间：

| 问题 | 现象 | 对论文价值的影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处仍有颜色偏差或模糊 | 可以提升 PSNR、SSIM、LPIPS 和局部视觉质量 |
| 拓扑冗余 | 一部分 faces 对多视角解释贡献低，部分低风险面可以删 | 可以做 rate-distortion 优化 |
| 长训练不一定更好 | 当前 full9 clean envelope 中 clean `30000` 全部弱于 clean `26000` | 说明收益不能简单归因于“训练更久” |

SPCarNet 的核心研究假设是：

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

这一阶段只使用训练视角，不使用 held-out test GT 选择方法参数。

### 3.2 Sparse-Occlusion Protected Compaction

压缩目标不是最大化删面比例，而是在 RGB、sparse geometry 和拓扑安全之间做保守 rate-distortion 优化。

三角形是否可压缩主要由训练证据判断：

- 多视角 visibility 足够稳定；
- face 不是关键 occlusion boundary；
- face 不属于高 residual 解释核心；
- 删除后 policy-val render 没有明显退化；
- sparse geometry audit 没有 AbsRel、DepthMAE、Normal 风险；
- 对室内强场景启用 micro-budget，避免为了压缩率破坏已很强的 geometry。

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

`alpha` 不是给每个场景手调的参数。Phase-J 使用 train-only calibration 和 guarded policy 自动决定：

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

这个 guard 也是当前结果可信的关键。v26-v39 中大量看起来可能有效的探针被拒绝或降级，说明 pipeline 不是为了追 test 数字而调参。

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
docs/car_model/6-23-SSIMAwareAtlas-v39-Implementation-Log.md
```

## 5. 主定量结果

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

> 更严谨的主 claim 是相对本地同协议 selected clean MeshSplatting。paper table 说明我们的结果没有低于论文表格，但 paper table 不是最强公平 baseline。

## 6. 定性结果与 PPT 素材

### 6.1 推荐主图

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

### 6.2 Backup Figures

| 用途 | 图片 |
|---|---|
| 全图公平同视角比较 | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| 室外局部细节 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| 混合局部收益展示 | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| representation-level PatchCert 备份图 | `assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png` |

建议：

- 主讲不要只放 full-frame gallery，因为视觉差异会被缩小；
- 主图用 local error reduction；
- full-frame gallery 用于证明公平同视角比较；
- Phase-S / v39 图只作为“正在推进 representation-level 内化”的 backup，不作为主结果。

## 7. 最新 representation-level 进展

当前最重要的新探索是 v37/v38/v39 surface residual atlas。它们不是 Phase-J 主结果，但对下一步论文级升级非常关键。

### 7.1 v37: coverage bottleneck 被修掉，但泛化失败

v36 的问题是 target 几乎没被改动：

```text
target changed pixels: 205 / 59,932,637
target changed fraction: 0.00000342
```

v37 做了三个实质改动：

1. `ecsr_build_surface_evidence_cache.py` 新增 `--barycentric_scope visible`，对所有 visible pixels 写入 `barycentric / barycentric_valid`；
2. 新增 `ecsr_audit_surface_residual_atlas_coverage.py`，在 atlas apply 前审计 target candidate coverage；
3. `ecsr_apply_surface_residual_region_texture_adapter.py` 新增 `--min_target_changed_fraction`，拒绝“policy-val 接受但 target 几乎 no-op”的假阳性。

full-res Bonsai target evidence 改善明显：

```text
old actionable pixels: 675
v37 actionable pixels: 580,404
barycentric valid fraction: 0.93035663
candidate/actionable fraction: 0.00968427
```

但指标退化：

| method on Bonsai | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| selected clean `ours_26000` | 28.8952 | 0.8964 | 0.2595 |
| compact parent | 28.8643 | 0.8960 | 0.2593 |
| v37 visible-train visible-target atlas | 28.8012 | 0.8915 | 0.2650 |
| Phase-J render-time ELA | 31.8620 | 0.9303 | 0.1726 |

结论：

> v37 证明 coverage 和 materialization 已经不再是主要问题；真正短板变成“train residual atlas 虽然能解释 policy-val residual，但直接贴到 held-out surface 上会产生跨视角、遮挡和材质泛化误差”。

### 7.2 v38: risk-aware atlas 修复大退化

v38 把 v37 的失败诊断转成真实方法改动：

- `--min_atlas_bin_count`：target apply 只允许训练中实际观测过足够次数的 UV-bin 生效；
- `--min_atlas_face_samples`：过滤训练样本过少的 face；
- per-view policy-val risk statistics：记录 positive-view fraction、min-view gain、CVaR20 view gain；
- `--select_alpha_by_risk_gate`：选择满足 train-only 风险门控的 alpha，而不是只选平均 MSE 最优 alpha。

关键诊断：

| alpha | mean MSE rel gain | positive view frac | CVaR20 view gain | min view gain |
|---:|---:|---:|---:|---:|
| 0.125 | 0.074679 | 1.000000 | 0.023605 | 0.011314 |
| 0.750 | 0.235936 | 0.583333 | -0.329707 | -0.520298 |

解释：v37 选择的大 alpha 虽然平均 MSE 更高，但尾部视角很危险。v38 会自动选择更小但所有 policy-val view 都正向的 alpha。

v38 best conservative Bonsai：

| method | PSNR | SSIM | LPIPS | strict vs compact |
|---|---:|---:|---:|---|
| compact parent | 28.864340 | 0.896012 | 0.259340 | baseline |
| v37 visible atlas | 28.801197 | 0.891540 | 0.265000 | no |
| v38 risk-safe bin1 a0.03125 | 28.866030 | 0.896006 | 0.259298 | no |

v38 修复了大退化，并让 PSNR/LPIPS 正向，但 SSIM 仍略低于 compact parent。

### 7.3 v39: SSIM-aware atlas 的弱正向 pilot

v39 在 v38 上加入：

- count-weighted 3x3 low-pass residual texture；
- per-bin residual variance；
- per-bin residual sign consistency；
- optional variance/sign gates in policy-val and target apply。

v39 best strict-pilot Bonsai：

| method | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM | dLPIPS | strict vs compact |
|---|---:|---:|---:|---:|---:|---:|---|
| compact parent | 28.864340 | 0.896012306 | 0.259339690 | +0.000000 | +0.000000 | +0.000000 | baseline |
| v38 risk-safe bin1 a0.03125 | 28.866030 | 0.896006 | 0.259298 | +0.001690 | -0.000006 | -0.000042 | no |
| v39 lowpass1 bin1 a0.015625 | 28.865229 | 0.896012485 | 0.259327114 | +0.000889 | +0.00000018 | -0.000013 | yes |
| selected clean `ours_26000` | 28.895233 | 0.896400273 | 0.259492785 | +0.030893 | +0.000388 | +0.000153 | no |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 | yes |

精确 v39 delta over compact parent：

```text
PSNR: +0.000888824462890625
SSIM: +0.00000017881393432617188
LPIPS: -0.000012576580047607422
```

诚实结论：

> v39 是本分支第一个对 compact parent 三指标严格正向的 representation-level atlas pilot，但收益极小，尚未超过 selected clean 的 PSNR/SSIM，更远低于 Phase-J render-time ELA。它的意义是证明 SSIM-aware smoothing 可以穿过 compact no-regression 线，但还不能作为 headline。

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
| v39 SSIM-aware atlas | low-pass、variance、sign consistency 是否能穿过 SSIM 线 | Bonsai compact-parent 三指标弱正向，但幅度太小，不能替代 Phase-J |

## 9. 为什么这是研究工作，而不是简单工程调参

当前方法的研究性主要体现在三个约束。

1. Surface evidence certification  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. No-test-GT guarded policy  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。很多 v26-v39 探针被拒绝或降级，说明 pipeline 不是为了追 test-set tuning。

3. Rate-distortion and recovery loop  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 有真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；SSIM-aware、risk-aware 的 surface residual atlas 是下一步论文级升级方向。

## 10. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis |
| 定性 full-frame 差异不总是显著 | 全图差通常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| representation-level 分支收益稀疏 | v39 已有弱正向 Bonsai pilot，但幅度极小 | 顶会主线仍需更强 representation story | 做多场景 carrier-holdout、confidence-weighted residual basis |
| v37 residual atlas 泛化失败 | target coverage 已修，但 naive atlas 退化 | 错误 residual 被充分作用到 held-out | risk-aware alpha、variance gate、target-risk proxy |
| v39 未超过 clean/Phase-J | 只相对 compact parent 微弱正向 | 不能包装成主结果 | 扩大到 garden/room/counter，多场景验证并提高 effect size |
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

### Slide 11: Ablations and Failed Probes

展示 v26-v39：

- 多个探针真实接入并验证；
- gate 拒绝弱结果；
- v37 修复 coverage 后仍退化，说明泛化才是瓶颈；
- v38 用 risk-aware gate 修复大退化；
- v39 第一次穿过 compact-parent 三指标线，但 effect size 很小。

### Slide 12: Takeaway and Next Step

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation.
```

## 12. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v39 很多调参式探针被拒绝或降级，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

主 claim 是相对本地同协议 selected clean MeshSplatting。baseline 从 clean `26000/30000` 中只按 held-out test score 选更强者；我们方法的 branch、alpha、edge fallback、压缩策略不使用 test GT。

### Q3: 为什么 paper table 不是主 claim？

paper table 可以说明我们的复现和结果没有低于论文数值，但不同实现、checkpoint、数据处理和评价脚本可能有差异。最严谨 claim 仍是本地同协议 selected clean baseline。

### Q4: 如果 Phase-J 主要是 render-time ELA，会不会不够像 representation method？

这是当前最重要边界。我们已经有真实 compact checkpoint 和 checkpoint-safe topology rewrite，但最强 RGB 修复仍来自 render-time adapter。v37-v39 正是在推进 representation-level baking；目前它们给出了清晰诊断和一个弱正向 pilot，但还不能替代 Phase-J。

### Q5: 为什么 v37 coverage 修好了，指标反而变差？

因为 v37 解决的是“能否作用到目标像素”的问题，而不是“作用的 residual 是否一定泛化正确”。当 target changed pixels 从 205 提升到 578,910 后，错误 residual 也会更充分地表现出来，所以指标退化。这说明下一步需要 risk-aware transfer，而不是继续盲目扩大 atlas。

### Q6: v39 已经三指标超过 compact parent，能不能说 representation-level 闭环完成？

不能。v39 的 Bonsai strict win 只有 `+0.000889` PSNR、`+0.00000018` SSIM、`-0.000013` LPIPS，且还低于 selected clean 的 PSNR/SSIM，也没有多场景验证。它是有价值的方向证据，不是最终结论。

## 13. 汇报时的诚实版本

建议主讲：

> 当前 Phase-J 在我们选定的 Mip-NeRF360 full9 口径下已经全面超过本地 clean MeshSplatting baseline，同时保留平均 7.65% triangle reduction 和 geometry safety。这是目前可以安全汇报的强结果。

主动说明：

> 但这还不是论文终局。最强外观收益仍是 render-time ELA；representation-level 分支已经从 v37 的 coverage 修复、v38 的 risk-aware gate，推进到 v39 的 SSIM-aware atlas 弱正向 pilot，但目前 effect size 和多场景证据仍不足。下一步要把 ELA teacher 的收益内化为 SSIM-aware、variance-aware、view-holdout-certified 的 surface residual representation。

## 14. 文件和结果索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
```

v37：

```text
docs/car_model/6-23-VisibleBarycentricCoverage-v37-Implementation-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/v37_target_coverage_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_region_texture_adapter_v37_visible_train_target/results.json
```

v38：

```text
docs/car_model/6-23-RiskAwareAtlas-v38-Implementation-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32_a003125/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin2_face32_a003125/results.json
```

v39：

```text
docs/car_model/6-23-SSIMAwareAtlas-v39-Implementation-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/bonsai_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_ssimaware_atlas/logs/metrics_bonsai_v39_lowpass1_bin1_face32_a0015625_gpu4.log
```

## 15. 下一步建议

最有价值的下一步不是继续手动 alpha sweep，而是围绕 v39 的弱正向结果做真正 representation-level 加强：

1. 多场景复验 v39 strict-pilot policy：至少 `garden`、`room`、`counter`、`bonsai`；
2. 做 carrier-holdout，不只做 view-holdout，防止 atlas 记忆自己的 train UV support；
3. 引入 per-bin confidence：由 count、variance、sign consistency、normal/view angle、support-view coverage 决定；
4. 用 confidence-weighted alpha 替代全局 alpha；
5. 把 policy-val objective 从 MSE 扩展到局部 SSIM/luminance-contrast proxy；
6. 严格规定 promotion gate：只有同时超过 compact parent 和 selected clean，并完成多场景验证，才能替代 Phase-J。

当前报告建议对 mentor 的定位：

> Phase-J 证明“训练证据驱动修复闭环”在本地公平 baseline 上有强结果；v39 证明“把修复内化到 surface representation”已经出现第一条弱正向路径。下一阶段的核心任务是把 v39 的弱正向 pilot 扩大成多场景、可见、可投稿的 representation-level gain。
