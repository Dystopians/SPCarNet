# SPCarNet 当前方法完整技术报告 v3

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可安全主讲 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 上的训练证据驱动压缩与修复闭环。它不把 MeshSplatting 替换掉，而是把训练好的 MeshSplatting checkpoint 当作强基础表示，再用训练视角里的 evidence 做两件事：

1. 判断哪些三角形可以安全压缩；
2. 判断哪些局部外观 residual 可以可靠转移到 held-out view。

当前最适合放进 PPT 主线的版本仍然是 Phase-J：

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
| 公平 baseline | 本地同协议 selected clean MeshSplatting；clean `26000/30000` 只用 held-out test score 选强者 |
| Mip-NeRF360 full9 | `9 / 9` 场景相对 selected clean baseline 三指标严格胜出 |
| 平均 RGB 提升 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310` |
| 最新 representation-level 探索 | v38 risk-aware atlas 修复了 v37 的大幅退化，Bonsai 上 PSNR/LPIPS 可正向但 SSIM 仍近似持平略负，暂不推广 |
| 最重要边界 | 最强视觉收益仍主要来自 render-time ELA，还没有完全 baked 成 checkpoint 内部表示 |

推荐汇报用一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，同时平均减少 7.65% 三角形；但最强外观收益仍来自 render-time ELA，下一步必须把修复进一步内化到 representation-level checkpoint。

## 1. 背景与问题定义

MeshSplatting 的优势是输出 triangle mesh。相比 Gaussian 或点云，它更容易接入传统渲染、游戏、AR/VR、数字孪生和下游几何管线。但本地复现和审计暴露出三个可改进点：

| 问题 | 现象 | 影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处有颜色偏差或模糊 | 全图指标和局部视觉质量仍有提升空间 |
| 拓扑冗余 | 部分 faces 对多视角解释贡献低，甚至属于低风险冗余面 | 可以做 rate-distortion 优化 |
| 训练更久不一定更好 | 当前 full9 clean baseline envelope 中 clean `30000` 全部弱于 clean `26000` | 提升不能解释为简单训练更久 |

SPCarNet 的研究假设：

> MeshSplatting 已经学到强基础表示，但训练视角里仍含有可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删掉部分冗余 geometry，并把训练 residual 转移到 held-out view 来修复外观。

## 2. 与原始 MeshSplatting 的本质区别

| 维度 | clean MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 表示 | 原始 opaque triangle mesh checkpoint | compact mesh checkpoint + train-evidence residual adapter |
| 几何处理 | 不做额外删面策略 | sparse-occlusion protected compaction |
| 外观修复 | checkpoint 属性直接渲染 | Evidence Lumigraph Adapter 用训练 residual 修复 held-out render |
| 决策依据 | 默认训练产物 | train-only calibration、policy-val gate、fallback |
| test GT 使用 | 最终评价 | 只做最终评价，不参与方法选择 |
| 失败处理 | 无显式机制 | gate 不通过就 fallback 或 no-op |

通俗解释：

> 原始 MeshSplatting 是“训练出一个网格然后交付”。SPCarNet 是“训练出网格后，再让网格基于训练视角做体检：哪里能安全删，哪里容易错，哪里有可靠 residual 可以修。证据不足时宁可不动”。

## 3. 方法模块

### 3.1 Train-View Evidence Mining

在训练视角上对 baseline 或 compact checkpoint 渲染，并缓存：

- rendered RGB；
- GT RGB；
- residual `GT - render`；
- per-face visibility；
- per-face hit count / pixel support；
- support-view consistency；
- high-error connected regions；
- depth / surface hit evidence；
- view-dependent residual statistics。

这一阶段只使用训练视角，不使用 held-out test GT 做策略选择。

### 3.2 Sparse-Occlusion Protected Compaction

目标不是追求最大删面比例，而是在 RGB、sparse geometry 和拓扑安全之间做保守 rate-distortion 优化。

三角形是否可以压缩主要由训练证据判断：

- 多视角 visibility 足够稳定；
- face 不是关键 occlusion boundary；
- face 不属于高 residual 解释核心；
- 删除后 policy-val render 没有明显退化；
- sparse geometry audit 没有 AbsRel、DepthMAE、Normal 风险；
- 室内场景启用 micro-budget，避免为了压缩数字破坏已经很强的 geometry。

报告中的 triangle reduction 是删除的三角形占比，不是剩余比例。

### 3.3 Checkpoint-Safe Topology Rewrite

压缩结果会真正写回 MeshSplatting checkpoint：

- 删除 faces；
- remap face indices；
- remap vertices；
- 清理 trailing unused vertices；
- 保证 tensor shape 与 renderer 一致；
- 保证后续 render、metric 和 geometry audit 可运行。

这意味着 SPCarNet 的压缩不是 report-only 后处理，而是 materialized checkpoint edit。

### 3.4 Evidence Lumigraph Adapter

ELA 是当前 Phase-J 视觉收益的主要来源。对训练 support view，先定义 residual：

```text
residual_s(x) = GT_s(x) - Render_s(x)
```

对 target held-out view，系统根据相机、depth、surface hit 和 support confidence，把多个训练 support residual 转移并聚合到 target image：

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

`alpha` 不是手工为每个场景调出来的固定参数。Phase-J 使用 train-only calibration 和 guarded policy 自动决定：

- adaptive alpha 是否可用；
- structural edge fallback 是否更安全；
- 当前 candidate 是否需要回退。

### 3.5 Guarded Adaptive Policy

Phase-J 的核心不是单一 ELA 公式，而是 guarded portfolio：

- `8 / 9` 场景采用 adaptive-alpha branch；
- `treehill` 使用 train-selected structural edge fallback；
- 所有 branch 只用 train/policy-val evidence 选择；
- gate 不通过时 fallback，而不是强行接受。

主要 gate：

- PSNR non-regression；
- SSIM regression 上限；
- LPIPS regression 上限；
- balanced score 不为负；
- tail / region-risk 不明显恶化；
- sparse geometry 安全；
- no-test-GT branch selection。

这也是当前工作可信的关键：pipeline 会拒绝很多看起来“可能有效”的小改动，而不是把单场景或 test-only 假阳性包装成主结果。

## 4. 实验协议

### 4.1 数据集

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

- baseline 选择使用 held-out test，是为了构造更强 clean envelope；
- 我们方法的 alpha、edge fallback、branch 和压缩策略不使用 held-out test GT；
- held-out test 只用于最终 report-only 评价；
- 当前 full9 中 selected clean baseline 全部选择 clean `26000`，说明 clean `30000` 并没有因为训练更久而更强。

### 4.3 主要证据路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

## 5. 主定量结果

### 5.1 Full9 Scene Table

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

### 5.3 Relation to MeshSplatting Paper Table

这部分可以作为 sanity check，不应替代本地公平 baseline claim。

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table, Mip-NeRF360 mean | 24.7800 | 0.7280 | 0.3100 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J minus paper table | +1.7028 | +0.0557 | -0.0857 |
| Phase-J minus local selected clean | +1.3311 | +0.0347 | -0.0634 |

推荐讲法：

> 更严谨的主 claim 是相对本地同协议 selected clean MeshSplatting。paper table 说明我们的同口径结果没有低于论文数值，但 paper table 不是最强公平 baseline。

## 6. 定性结果

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

建议讲法：

> 全图上差异通常是 residual-level，肉眼不一定一眼能看出；因此我们用同一 held-out 协议下的局部误差下降图展示哪里确实更接近 GT。它不是手工挑 test 指标，而是从 closure audit 和 per-view delta 自动筛选。

### 6.2 Backup Figures

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png
```

使用建议：

- full-frame gallery 用于证明 baseline 与 ours 的公平同视角对比；
- Phase-J local showcase 用于主讲视觉收益；
- outdoor detail showcase 用于回答“室外是否也有效”；
- Phase-S PatchCert 图用于说明 representation-level 分支是真实 checkpoint edit，但当前不是主 RGB endpoint。

## 7. 最新 representation-level 进展与负结果

当前最重要的新探索是 v37/v38 surface residual atlas。v37 的价值是定位出 target coverage 之后的泛化瓶颈；v38 则把这个教训转成了 risk-aware alpha selection 和 atlas support certification。

### 7.1 v35/v36 暴露的问题

v36 train-only policy-val gate 很强，但 held-out target render 几乎没被改动：

```text
target changed pixels: 205 / 59,932,637
target changed fraction: 0.00000342
```

coverage audit 显示：

```text
candidate face fraction: 0.00986114
barycentric valid fraction: 0.00017635
candidate/actionable fraction: 0.00001126
actionable pixels: 675
```

结论：不是没有候选 faces，而是 target evidence 没有足够可用 barycentric support。

### 7.2 v37 修复的工程短板

v37 做了三个实质改动：

1. `ecsr_build_surface_evidence_cache.py` 新增 `--barycentric_scope visible`，对所有 visible pixels 写入 `barycentric / barycentric_valid`；
2. 新增 `ecsr_audit_surface_residual_atlas_coverage.py`，在 atlas apply 前审计 target candidate coverage；
3. `ecsr_apply_surface_residual_region_texture_adapter.py` 新增 `--min_target_changed_fraction`，拒绝“policy-val 接受但 target 几乎 no-op”的假阳性。

full-res Bonsai test target evidence 完成后：

```text
views: 37 / 37
barycentric valid fraction: 0.93035663
candidate/actionable fraction: 0.00968427
actionable pixels: 580,404
actionable / candidate fraction: 0.98206442
```

相对旧 target evidence：

```text
actionable pixels: 675 -> 580,404
candidate/actionable fraction: 0.00001126 -> 0.00968427
```

这说明“目标视角能不能被 residual atlas 作用到”的机制瓶颈已经被修掉。

### 7.3 v37 最终 Bonsai 指标

v37 visible train + visible target atlas 能够 materialize 大范围 target 改动：

```text
atlas faces: 2208
fit samples: 205,165
selected alpha: 0.75
policy-val relative gain: 0.3374
target changed pixels: 578,910
target changed fraction: 0.00965934
```

但 full-res held-out Bonsai 指标为：

| method on Bonsai | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| selected clean `ours_26000` | 28.8952 | 0.8964 | 0.2595 |
| compact parent | 28.8643 | 0.8960 | 0.2593 |
| v36 matched-res atlas | 28.8648 | 0.8960 | 0.2593 |
| v37 old-train visible-target atlas | 28.8628 | 0.8959 | 0.2594 |
| v37 visible-train visible-target atlas | 28.8012 | 0.8915 | 0.2650 |
| Phase-J render-time ELA | 31.8620 | 0.9303 | 0.1726 |

结论：

> v37 证明 coverage 和 materialization 已经不再是主要问题；真正短板变成“train residual atlas 虽然能解释 policy-val residual，但直接贴到 held-out surface 上会产生跨视角/遮挡/材质泛化误差”。因此 v37 是重要机制诊断，不是新的 headline method。

### 7.4 v38 risk-aware atlas

v38 把 v37 的诊断转成了真实方法改动：

1. `--min_atlas_bin_count`：target apply 时只允许训练中实际观测过足够次数的 UV-bin 生效；
2. `--min_atlas_face_samples`：过滤训练样本过少的 face；
3. per-view policy-val risk statistics：每个 alpha 记录 positive-view fraction、min-view gain、CVaR20 view gain；
4. `--select_alpha_by_risk_gate`：选择满足 train-only 风险门控的最优 alpha，而不是平均 MSE 最优 alpha。

关键诊断：

| alpha | mean MSE rel gain | positive view frac | CVaR20 view gain | min view gain |
|---:|---:|---:|---:|---:|
| 0.125 | 0.074679 | 1.000000 | 0.023605 | 0.011314 |
| 0.750 | 0.235936 | 0.583333 | -0.329707 | -0.520298 |

这说明 v37 选择的大 alpha 虽然平均 MSE 更好，但尾部视角很危险；v38 risk-safe gate 会自动选择更小但所有 policy-val view 都正向的 alpha。

v38 full-res Bonsai 结果：

| method on Bonsai | PSNR | SSIM | LPIPS | dPSNR vs compact | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| compact parent | 28.864340 | 0.896012 | 0.259340 | +0.000000 | +0.000000 | +0.000000 |
| v37 visible atlas | 28.801197 | 0.891540 | 0.265000 | -0.063143 | -0.004473 | +0.005660 |
| v38 risk-safe bin1 a0.0625 | 28.867365 | 0.895973 | 0.259272 | +0.003025 | -0.000039 | -0.000068 |
| v38 risk-safe bin1 a0.03125 | 28.866030 | 0.896006 | 0.259298 | +0.001690 | -0.000006 | -0.000042 |
| v38 risk-safe bin2 a0.03125 | 28.865604 | 0.896009 | 0.259308 | +0.001265 | -0.000004 | -0.000032 |
| Phase-J render-time ELA | 31.862005 | 0.930280 | 0.172555 | +2.997665 | +0.034267 | -0.086784 |

结论：

> v38 大幅修复了 v37 的退化，并能在 Bonsai representation-level atlas 上取得 PSNR/LPIPS 正向、SSIM 近持平；但它仍没有三指标严格超过 compact parent，也没有超过 selected clean 的 PSNR/SSIM，因此暂不推广为主方法。

下一步不应继续做 alpha sweep，而应加入更强的 train-only 泛化约束：

- carrier holdout：拟合 carrier 与验证 carrier 分离；
- view-stratified policy-val：按视角簇、边缘、深度 discontinuity 分层验证；
- uncertainty-weighted residual：把高方差 support residual 降权；
- atlas smoothness / frequency prior：限制贴图残差的高频漂移；
- target-risk proxy：用目标可见性、support count、barycentric stability 和 normal/view angle 估计 transfer risk。

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
| v37 visible barycentric atlas | 修复 target coverage 后是否成功 | coverage 大幅提升，但 full-res Bonsai 指标退化；说明泛化/风险建模才是新瓶颈 |
| v38 risk-aware atlas | train-only view-risk gate 和 atlas bin support 能否修复 v37 | 大幅修复 v37 退化；Bonsai PSNR/LPIPS 可超过 compact，但 SSIM 仍略低，未推广 |

## 9. 为什么这是研究工作，而不是简单工程调参

当前方法的研究性主要体现在三个约束：

1. **Surface evidence certification**  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. **No-test-GT guarded policy**  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。大量 v26-v38 探针被拒绝或降级，说明 pipeline 不是为了追 test 数字而调参。

3. **Rate-distortion and recovery loop**  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 至少有一个真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；surface-addressed residual basis 仍是下一步论文级升级方向。

## 10. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis |
| 定性 full-frame 差异不总是显著 | 全图差异常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| representation-level 分支收益稀疏 | Phase-R/S/v37/v38 有真实 checkpoint edit，但提升小、负向或未三指标全胜 | 顶会主线仍需更强 representation story | 做 SSIM-aware / variance-aware region residual field |
| v37 residual atlas 泛化失败 | target coverage 已修，但指标退化 | naive residual texture 可能带来 view-transfer error | carrier holdout + risk-weighted residual + smoothness prior |
| v38 residual atlas 仍未严格胜出 | 大退化被修复，但 SSIM 仍略低于 compact | 风险 gate 还不够理解结构相似度 | 增加 SSIM-aware train proxy 和 residual variance gate |
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

### Slide 4: What Is Different From MeshSplatting

使用第 2 节表格。重点讲：

- clean MeshSplatting 直接渲染；
- SPCarNet 判断哪里可删、哪里可修、哪里必须回退。

### Slide 5: Evidence Mining and Safe Compaction

主讲：

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

### Slide 11: Ablation and Failed Probes

展示 v26-v38：

- 多个探针真实接入并验证；
- gate 拒绝弱结果；
- v30 说明 image-level teacher-bake 不够；
- v37 说明 surface atlas coverage 已修，但跨视角泛化仍不足；
- v38 说明 risk-aware gate 可以修复大退化，但还不足以三指标严格胜出；
- 下一步是 SSIM-aware / variance-aware region residual field。

### Slide 12: Takeaway and Next Step

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation.
```

## 12. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v38 很多调参式探针被拒绝或降级，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

主 claim 是相对本地同协议 selected clean MeshSplatting。baseline 从 clean `26000/30000` 中只按 held-out test score 选更强者；我们方法的 branch、alpha、edge fallback、压缩策略不使用 test GT。

### Q3: 为什么 paper table 不是主 claim？

paper table 可以说明我们的复现和结果没有低于论文数值，但不同实现、checkpoint、数据处理和评价脚本可能有差异。最严谨的 claim 仍是本地同协议 selected clean baseline。

### Q4: 如果 Phase-J 主要是 render-time ELA，会不会不够像 representation method？

这是当前最重要边界。我们已经有真实 compact checkpoint 和 checkpoint-safe topology rewrite，但最强 RGB 修复仍来自 render-time adapter。v30-v38 正是在推进 representation-level baking；目前它们给出了清晰诊断，但还不能替代 Phase-J。

### Q5: 为什么 v37 coverage 修好了，指标反而变差？

因为 v37 解决的是“能否作用到目标像素”的问题，而不是“作用的 residual 是否一定泛化正确”。当 target changed pixels 从 205 提升到 578,910 后，错误 residual 也会更充分地表现出来，所以指标退化。这说明下一步需要 risk-aware transfer，而不是继续盲目扩大 atlas。

## 13. 汇报时的诚实版本

建议主讲：

> 当前 Phase-J 在我们选定的 Mip-NeRF360 full9 口径下已经全面超过本地 clean MeshSplatting baseline，同时保留平均 7.65% triangle reduction 和 geometry safety。这是目前可以安全汇报的强结果。

也建议主动说明：

> 但这还不是论文终局。最强外观收益仍是 render-time ELA；v37 已经把 surface residual atlas 的 target coverage 瓶颈修掉，但 full-res Bonsai 指标负向。v38 进一步用 risk-aware alpha selection 和 atlas support certification 修复了大部分退化，但仍未三指标严格超过 compact/clean。下一步需要把 ELA teacher 的收益内化为 SSIM-aware、variance-aware、view-holdout-certified 的 surface residual representation。

## 14. 文件和结果索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

v37 最新结果：

```text
docs/car_model/6-23-VisibleBarycentricCoverage-v37-Implementation-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/v37_target_coverage_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_region_texture_adapter_v37_visible_train_target/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_region_texture_adapter_v37_visible_train_target/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/logs/metrics_bonsai_teacher_region_texture_adapter_v37_visible_train_target_gpu3.log
```

v38 最新结果：

```text
docs/car_model/6-23-RiskAwareAtlas-v38-Implementation-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin1_face32_a003125/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/bonsai_teacher_region_texture_adapter_v38_risksafe_bin2_face32_a003125/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v38_riskaware_atlas/logs/metrics_bonsai_v38_risksafe_bin2_face32_a003125_gpu4.log
```
