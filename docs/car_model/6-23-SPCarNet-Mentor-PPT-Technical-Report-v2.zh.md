# SPCarNet 当前方法完整技术报告（Mentor PPT 版）

日期：2026-06-23  
用途：mentor 汇报、PPT 制作、当前方法交底  
当前可主讲主结果：`ours_26000_phasej_guarded_adaptedge_ela`  
当前状态判断：Phase-J 已经是强的 train-evidence repair loop；v31-v36 已打通 representation-level teacher residual、region carrier、residual atlas 和 matched-resolution evidence 的关键接口并完成 Bonsai full-res 诊断，但还不能作为 headline。v37 已针对 v35/v36 的 held-out target coverage 瓶颈实现 visible-scope barycentric evidence、target coverage audit 和 target changed-fraction gate，并完成 Bonsai `images_4` 单视角 smoke；full-res `images_2` 多视角重跑仍待可用 GPU，因此 Phase-J 仍是当前可主讲主结果。

## 0. 一页结论

SPCarNet 是建立在 MeshSplatting 之上的训练证据驱动压缩与修复框架。它不把 MeshSplatting 替换掉，而是把 MeshSplatting checkpoint 当成强基础表示，然后做两件事：

1. 用训练视角 evidence 判断哪些三角形可以安全压缩；
2. 用训练 residual 和 guarded policy 修复 held-out 渲染中的局部外观错误。

当前最适合作为 PPT 主线的版本是 Phase-J：

```text
clean MeshSplatting checkpoint
  -> train-view surface/render evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> Evidence Lumigraph Adapter
  -> guarded adaptive / edge fallback policy
  -> held-out evaluation
```

核心结果：

| 维度 | 当前结果 |
|---|---|
| 主方法 | Phase-J guarded adaptive Evidence Lumigraph Adapter on compact MeshSplatting |
| 公平 baseline | 本地同协议 selected clean MeshSplatting，clean `26000/30000` 只按 held-out score 选择更强 clean checkpoint |
| Mip-NeRF360 full9 | `9 / 9` 场景相对 selected clean baseline 三指标严格胜出 |
| 平均 RGB 收益 | `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| 逐视角稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310` |
| 重要边界 | 最强 appearance gain 仍主要来自 render-time ELA；v31-v37 已证明 surface-addressed representation 路线接口可行并定位/修复了一部分 held-out support coverage 机制，但 full-res atlas rerun 尚未完成，不能替代 Phase-J |

汇报时最稳的一句话：

> 我们把 MeshSplatting 从“训练完直接渲染”升级成“训练证据驱动的安全压缩与残差修复闭环”。当前 Phase-J 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 做到 9/9 场景 PSNR、SSIM、LPIPS 严格提升，平均提升 +1.3311 PSNR、+0.0347 SSIM、-0.0634 LPIPS，同时平均减少 7.65% 三角形；但最强外观收益仍来自 render-time ELA，下一阶段要把修复进一步内化到 representation-level checkpoint。

## 1. 背景与问题定义

MeshSplatting 的优势是输出 triangle mesh，比纯 Gaussian 或点云更容易接入传统图形资产、实时渲染、AR/VR、数字孪生和下游几何管线。但本地复现和长期审计显示，它还有三个可改进空间：

| 问题 | 现象 | 对论文目标的影响 |
|---|---|---|
| 局部 residual 错误 | foliage、树皮、室内纹理、细边缘处仍有颜色偏差或模糊 | 全图指标和局部视觉质量有提升空间 |
| 拓扑冗余 | 一些 faces 在多视角解释中贡献较低 | 可以做 rate-distortion 优化 |
| 训练更久不必然更好 | 当前 full9 clean envelope 中 clean `30000` 全部弱于 clean `26000` | 不能把提升解释为单纯更久训练 |

SPCarNet 的研究假设：

> MeshSplatting 已经学到强基础 mesh 表示，但训练视角中仍包含 surface reliability、occlusion risk 和 appearance residual 的可用证据。只要证据足够可靠，就可以安全删掉冗余几何，并把训练 residual 转移到 held-out view 修复外观。

## 2. 与原始 MeshSplatting 的本质区别

| 维度 | clean MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础表示 | 原始 opaque triangle mesh checkpoint | compact mesh checkpoint + train-evidence residual adapter |
| 几何处理 | 默认训练产物，不做额外删面策略 | sparse-occlusion protected compaction |
| 外观修复 | checkpoint 属性直接渲染 | Evidence Lumigraph Adapter 修复局部 residual |
| 决策依据 | 训练流程默认结果 | train-only calibration、policy-val gate、fallback |
| test GT 使用 | 最终评价 | 只做最终评价，不参与 policy/branch/alpha 选择 |
| 失败处理 | 无显式回退机制 | gate 不通过就 fallback/no-op |

通俗解释：

> 原始 MeshSplatting 是“训练出一个网格然后直接交付”。SPCarNet 是“训练出网格后，再让网格基于训练视角做体检：哪里能安全删，哪里容易错，哪里有可靠 residual 可以修。证据不足时宁可不动”。

## 3. 当前方法模块

### 3.1 Train-View Evidence Mining

系统先在训练视角上渲染 baseline 或 compact checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual `GT - render`；
- per-face visibility；
- per-face hit count / pixel support；
- support-view consistency；
- high-error connected regions；
- depth / surface hit evidence；
- view-dependent residual statistics。

这个阶段只使用训练视角，不使用 held-out test GT 做策略选择。

### 3.2 Sparse-Occlusion Protected Compaction

目标不是最大化删面比例，而是在 RGB、sparse geometry 和拓扑安全之间做保守 rate-distortion 优化。

三角形是否可以压缩主要由训练证据判断：

- 多视角 visibility 足够稳定；
- face 不是关键 occlusion boundary；
- face 不属于高 residual 解释核心；
- 删除后 policy-val render 没有明显退化；
- sparse geometry audit 没有 AbsRel、DepthMAE、Normal 风险；
- 室内场景启用 micro-budget，避免为了压缩数字破坏已经很强的 geometry。

报告中的 triangle reduction 是删除的三角形占比，不是剩余比例。

### 3.3 Checkpoint-Safe Topology Rewrite

压缩会真正写回 MeshSplatting checkpoint：

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

对 target held-out view，系统根据相机、surface hit、view support、depth consistency 和 residual confidence，把多个训练 residual 转移并聚合到 target image：

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
Render_final_t(x)
  = Render_base_t(x) + alpha_t(x) * ResidualEvidence_t(x)
```

`alpha` 不是每个场景手动调出来的固定参数。Phase-J 使用 train-only calibration 和 guarded policy 自动决定：

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

### 3.6 Representation-Level Teacher Residual Branch

v30-v36 是当前正在推进的 representation-level 升级路线，目标是把 Phase-J 的 render-time 修复能力逐步烘焙进 checkpoint。

目前已完成的关键接口：

- `teacher_render - compact_parent_render` 被转成 surface evidence cache；
- evidence cache 中新增 `teacher_residual_rgb / teacher_residual_l1`；
- face-local residual operator 支持通过 `--residual_rgb_key / --residual_l1_key` 读取 teacher target；
- candidate-owned render region builder 和 global render-visible region carrier builder 已支持 teacher residual key；
- candidate expansion fallback stats 已修复，expanded faces 不再因为缺失 top-support CSV 行而被静默丢弃；
- Bonsai full-resolution v31/v32/v33/v35/v36 pilot 已完成 render 和 metric；
- v35/v36 新增 surface residual region texture/atlas adapter，并验证 matched-resolution teacher evidence 后仍存在 held-out target support 稀疏瓶颈。

当前结论：

> v31-v36 证明了 teacher residual 可以被 surface indexing、barycentric evidence、region carrier、face-local SH fitting、shared residual field 和 residual texture atlas 消化；但当前 full-res 收益仍接近 compact base，说明 bottleneck 已经从“接口没通”转移到“当前 representation carrier 在 held-out view 上覆盖极稀疏，尚不足以吸收 ELA 级 image-space residual”。

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

v31-v36 representation-level 诊断路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v31_teacher_surface/
outputs/carnet/meshsplatopt/ecsr_phase_v32_teacher_patch_carrier/
outputs/carnet/meshsplatopt/ecsr_phase_v33_teacher_region_capacity/
outputs/carnet/meshsplatopt/ecsr_phase_v35_teacher_region_atlas/
outputs/carnet/meshsplatopt/ecsr_phase_v36_matchedres_teacher_atlas/
outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_smoke/
docs/car_model/6-23-TeacherSurfaceBasis-v31-Implementation-Log.md
docs/car_model/6-23-TeacherRegionCapacity-v33-Experiment-Log.md
docs/car_model/6-23-TeacherBakeLong-v34-And-RegionAtlas-v35-Log.md
docs/car_model/6-23-MatchedResTeacherAtlas-v36-Log.md
docs/car_model/6-23-VisibleBarycentricCoverage-v37-Implementation-Log.md
```

## 5. 主定量结果

### 5.1 Full9 Scene Table

| scene | selected branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | tri red. | per-view wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.0215 | 0.7024 | 0.2661 | +0.7199 | +0.0425 | -0.0660 | 11.81% | 25/25 |
| flowers | adaptive alpha | 20.3044 | 0.5578 | 0.3292 | +0.6221 | +0.0459 | -0.0653 | 11.82% | 22/22 |
| garden | adaptive alpha | 26.3111 | 0.8278 | 0.1358 | +1.2819 | +0.0478 | -0.0655 | 3.47% | 24/24 |
| stump | adaptive alpha | 25.5951 | 0.7241 | 0.2639 | +0.3901 | +0.0189 | -0.0301 | 11.82% | 16/16 |
| treehill | edge fallback | 21.2962 | 0.5956 | 0.3363 | +0.3620 | +0.0311 | -0.0697 | 11.81% | 17/18 |
| room | adaptive alpha | 30.3056 | 0.9057 | 0.1960 | +1.5584 | +0.0209 | -0.0539 | 2.10% | 38/39 |
| counter | adaptive alpha | 28.4492 | 0.8937 | 0.1865 | +1.6974 | +0.0317 | -0.0655 | 2.10% | 30/30 |
| kitchen | adaptive alpha | 30.1997 | 0.9161 | 0.1320 | +2.3812 | +0.0396 | -0.0672 | 2.10% | 35/35 |
| bonsai | adaptive alpha | 31.8620 | 0.9303 | 0.1726 | +2.9668 | +0.0339 | -0.0869 | 11.80% | 37/37 |

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
| Local reproduced clean `ours_30000` | 24.8002 | 0.7310 | 0.3072 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J minus paper table | +1.7028 | +0.0557 | -0.0857 |
| Phase-J minus local selected clean | +1.3311 | +0.0347 | -0.0634 |

推荐表述：

> 更严谨的主 claim 是相对本地同协议 selected clean MeshSplatting。paper table 说明我们的复现实验没有低于论文数值，但 paper table 不是最强公平 baseline。

## 6. 定性结果

### 6.1 推荐主图

最推荐 PPT 使用：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

Markdown 引用：

```md
![SPCarNet Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)
```

这张图基于当前接受 endpoint `ours_26000_phasej_guarded_adaptedge_ela` 自动生成。选择逻辑：

1. 每个候选 view 先要求全图 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0`；
2. 再在纹理区域内寻找 SPCarNet 相对 GT 的局部 RGB 误差下降最大 patch；
3. 绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

建议讲法：

> 全图上差异通常是 residual-level，肉眼不一定一眼看出；因此我们用同一 held-out 协议下的局部误差下降图展示哪里确实更接近 GT。它不是手工挑 test 指标，而是从 closure audit 和 per-view delta 自动筛选。

### 6.2 Backup Figures

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_full9_crop_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png
```

使用建议：

- full-frame gallery 用于证明 baseline 与 ours 的公平同视角对比；
- local held-out error-reduction showcase 用于主讲视觉收益；
- outdoor detail showcase 用于回答“室外是否也有效”；
- Phase-S PatchCert 图用于说明 representation-level 分支是真实 checkpoint edit，但当前不是主 RGB endpoint。

## 7. Ablation and Diagnostics

### 7.1 主方法消融

| 变体 | 检验内容 | 结论 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | full9 均选择 clean `26000`；训练更久不是主要解释 |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但 RGB headline 不足 |
| compact + ELA without strong guard | 单 scalar score 是否足够 | 部分场景 PSNR/LPIPS 提升，但 SSIM 和 tail 有风险 |
| compact + guarded adaptive ELA | 当前 Phase-J | full9 `9 / 9` 三指标严格胜出 |
| aggressive pruning | 能否强推压缩率 | 被拒绝；敏感场景出现 geometry/render 风险 |
| optional FD gate | DINOv2 Frechet distance 是否能作为额外 train-only non-regression signal | 默认关闭；更像 LPIPS-oriented portfolio signal，会带来 PSNR/SSIM tradeoff |

### 7.2 近期 v26-v37 探针

| 版本 | 目的 | 状态 | 启发 |
|---|---|---|---|
| v26 hard local-trust | 用二值 trust gate 限制 residual | Bonsai medium 完成，过保守，容易 no-op | hard gate 不够连续 |
| v27 soft local-trust | 连续 trust-weight residual | 接口、smoke、Bonsai medium 完成，但 honest gate 拒绝 | trust 修正全零问题，但收益不足 |
| v28 view-tail-safe alpha | policy-view tail-safe alpha shrink | Bonsai medium 完成，不推广 | MSE tail 安全不等于三指标 balanced 安全 |
| v29 balanced view-tail objective | 用 `dPSNR + 20*dSSIM - 20*dLPIPS` 做 tail objective | Bonsai selector 仅 trainval 极微弱接受，held-out 极微弱负向 | 单纯改 objective 仍不够 |
| v30 triadic teacher-bake | teacher 优于 parent/current 且差异足够时才蒸馏 | Bonsai GPU smoke 完成，mask active，但 baked checkpoint 低于 clean-best | image-level teacher loss 不足 |
| v31 teacher-surface basis | 把 ELA teacher residual 从 image-space 转成 surface-addressed cache，再拟合 face-local SH1 delta | Bonsai full-res 完成；接口有效，但指标几乎等于 compact base | 方向正确，瓶颈是 carrier 覆盖和容量 |
| v32 teacher-region carrier | 从 teacher residual 建 render-visible region carriers，并用 region/core/context 权重拟合 | Bonsai full-res region pilot 有非零 PSNR 正向，但 SSIM/LPIPS 仍不够 | region evidence 能扩大覆盖，但 face-local SH1 容量仍弱 |
| v33 region capacity | 提高到 SH3、加入 validation global-gain，并测试 shared RBF residual field | Bonsai full-res 完成；shared-field 仅 `+0.00096` PSNR vs compact，SSIM/LPIPS 微退 | 仅提高 face-local/field capacity 不是突破口 |
| v34 teacher-bake long | 用 Phase-G teacher-render loss 和 parent rollback 做更长 Bonsai recovery | Bonsai 26000->27000 完成，W&B online；最终 `28.2562 / 0.8744 / 0.2999`，明显低于 clean；teacher 源审计显示 `alpha=0.0` | no-op teacher 源会让长训练失去意义，必须先审计 teacher policy |
| v35 surface residual atlas | 新增 teacher-region residual texture/atlas adapter，尝试比 sparse face-local delta 更大的 support | v3 stride3 gate 接受，full-res `28.8646 / 0.8960 / 0.2593`；相对 compact 仅 `+0.00030` PSNR、SSIM 微退，target changed fraction `0.000007` | atlas 接口可行，但 held-out support 覆盖太小，不能推广 |
| v36 matched-res teacher atlas | 用 full-res images_2 train evidence、alpha=1 teacher residual 和 region atlas 再试一次 | gate 接受，policy-val relative gain `0.7518`，full-res `28.8648 / 0.8960 / 0.2593`；target changed fraction `0.0000034` | 分辨率 mismatch 不是主瓶颈，真正瓶颈是 held-out barycentric/support 覆盖极稀疏 |
| v37 visible barycentric coverage | 修复 evidence cache：新增 visible-scope barycentric per-view immediate write；新增 target coverage audit；adapter 新增 target changed-fraction gate | Bonsai `images_4` 单视角 smoke：barycentric valid fraction `0.8102`、covered faces `112882`；existing v36 target audit actionable fraction `0.00001126`；coverage gate replay 正确拒绝 v36 | 已修机制短板，但 `images_2` full-res rerun 因 GPU2 OOM 尚未完成；下一步需要重建 full target evidence 后再跑 atlas |

## 8. v31-v37 Representation-Level 进展

### 8.1 v31: Teacher Surface Evidence

v31 完成的是从 render-time teacher 到 surface-addressed evidence 的接口：

- train cache：Bonsai `48` 个 train views，`2,121,267` 个 unique visible faces；
- per-view NPZ 包含 `face_id`、`residual_l1`、`residual_rgb`、`rgb_render`、`rgb_gt`、`barycentric`；
- teacher cache 新增 `teacher_residual_rgb`、`teacher_residual_l1`、`teacher_better_mask`；
- teacher-better mask 平均 active fraction `17.61%`；
- mean positive teacher gain L1 `0.00495`；
- face-local SH1 operator 能直接读取 teacher residual key。

Bonsai full-res 结果：

| method | PSNR | SSIM | LPIPS | 说明 |
|---|---:|---:|---:|---|
| selected clean MeshSplatting `ours_26000` | 28.8952 | 0.8964 | 0.2595 | local fair clean baseline |
| compact base before v31 | 28.8643 | 0.8960 | 0.2593 | v31 parent checkpoint |
| v31 face-local teacher SH1 | 28.8644 | 0.8960 | 0.2594 | materialized checkpoint，未推广 |

解释：

> v31 不是失败的接口，而是成功暴露了真实瓶颈：teacher target 有意义、policy-val proxy 正向，但最终只接受很少 face-local edits，因此 full-res test 指标接近 compact base。

### 8.2 v32: Teacher Patch / Region Carrier

v32 进一步解决 v31 覆盖太小的问题：

- candidate-owned teacher patch carriers：`12` plan carriers、`9` carriers with regions、`40` total regions、`255` output carrier faces；
- global teacher render-visible carriers：扫描 `48` views，得到 `576` raw regions、`96` merged carriers、`3470` evidence faces；
- region-minpilot 使用 `2048` selected faces、`40` accepted faces、`+120` vertices；
- full-res Bonsai render/eval 已完成。

v32 Bonsai full-res 结果：

| method | PSNR | SSIM | LPIPS | 相对 compact base | 结论 |
|---|---:|---:|---:|---|---|
| v32 seed-expansion minpilot | 28.8645 | 0.8960 | 0.2594 | `+0.00014` PSNR，SSIM/LPIPS 微退 | 不推广 |
| v32 relaxed PatchCert grow | 28.8645 | 0.8960 | 0.2594 | `+0.00014` PSNR，SSIM/LPIPS 微退 | 不推广 |
| v32 global teacher-region minpilot | 28.8653 | 0.8960 | 0.2593 | `+0.00091` PSNR，SSIM/LPIPS 微退 | 当前最好 representation-level pilot，但仍不足 |

重要诊断：

- region evidence 能扩大覆盖并产生非零 held-out movement；
- train-only policy-val proxy 明显正向，region-minpilot final relative gain `0.6678`；
- topology safe：triangles `9,555,533`，degenerate `0`，invalid `0`；
- 但 full-res held-out 指标仍只是噪声级提升，没有接近 Phase-J render-time ELA。

结论：

> v32 把 representation-level 路线从 face-only 推到了 region evidence，但 face-local SH1 的容量和跨视角表达仍然不够。下一步应尝试更高容量的 surface residual basis，例如 patch-shared low-rank field、higher-order SH、view-conditioned residual carriers 或更严格的 train/test disjoint patch certificate。

### 8.3 v33: Higher-Capacity Region Basis

v33 检查一个直接假设：如果瓶颈只是 face-local 表达容量不够，那么提高 SH 阶数、加入 train-only validation gain，或使用 shared RBF residual field 应该能产生超过噪声级的 full-res 改善。

本轮完成两个 Bonsai full-res pilot：

- independent face-local `SH3 + global_gain`；
- shared RBF residual field `SH3 + global_gain`。

v33 Bonsai full-res 结果：

| method | PSNR | SSIM | LPIPS | 相对 compact base | 结论 |
|---|---:|---:|---:|---|---|
| v33 SH3 global-gain | 28.8642 | 0.8960 | 0.2593 | `-0.00014` PSNR，SSIM/LPIPS 微退 | 不推广 |
| v33 shared-field SH3 | 28.8653 | 0.8960 | 0.2594 | `+0.00096` PSNR，SSIM/LPIPS 微退 | 当前高容量诊断最好，但仍是噪声级 |

关键诊断：

- shared-field 的 train-only final policy-val relative gain 达到 `0.6601`，高于 independent SH3 的 `0.5365`；
- full-res test 只得到 `+0.00096` PSNR，且 SSIM/LPIPS 仍变差；
- 这说明 bottleneck 不只是 SH 阶数或局部 field 容量，而是当前 sparse face-local materialization 对 ELA image-space residual 的可吸收性太弱。

结论：

> v33 不应作为方法结果；它是一个重要负结果，说明下一步不能继续小幅堆 face-local SH/field 容量，而应转向更大 support 的 surface residual texture/atlas、patch-level residual representation，或把 Phase-J 明确定位为 train-evidence render-time repair 主贡献。

### 8.4 v34/v35/v36: Teacher-Bake And Residual Atlas Follow-Up

v34/v35/v36 的目标是沿着 v33 诊断继续推进，但当前 Bonsai 结果仍不能进入 headline。

- v34 是 teacher-bake long recovery。它接入了 Phase-G teacher-render loss、parent rollback loss 和 W&B online logging，尝试让 checkpoint 通过继续训练吸收 teacher render signal。结果是 `28.256157 / 0.874371 / 0.299884`，相对 selected clean 为 `-0.6391 PSNR / -0.0220 SSIM / +0.0404 LPIPS`。关键原因是 teacher-render report 显示 `alpha=0.0`，teacher renders 实际退化为 compact parent/no-op teacher；因此 v34 是有效的负结果和 runner 审计，不是有效 teacher-bake 证据。
- v35 是 surface residual region texture/atlas adapter。它新增脚本 `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`，用 teacher-region residual、face id、barycentric coordinate 和 train-only policy-val gate 拟合更大 support 的 residual texture atlas。v3 stride3 通过 gate：policy-val samples `1090`，relative gain `0.521415`，selected alpha `0.75`。但 full-res target changed fraction 只有 `0.000007`，最终 `28.864641 / 0.896011 / 0.259334`，只比 compact 多 `+0.00030` PSNR、SSIM 微退，不能推广。
- v36 是 matched-resolution teacher atlas。它用 Bonsai images_2 full-res train evidence、alpha=1 teacher residual cache 和 render-visible region carriers 重跑 atlas fitting。gate 接受：fit samples `3564`，policy-val samples `2166`，selected alpha `1.0`，relative gain `0.751815`。但 test target changed pixels 只有 `205 / 59,932,637`，changed fraction `0.0000034`。最终 full-res metric 为 `28.864826 / 0.896009 / 0.259337`，比 compact 仅 `+0.00049` PSNR，SSIM 几乎不变且仍低于 clean PSNR/SSIM，更远低于 Phase-J `31.862005 / 0.930280 / 0.172555`。

汇报建议：

> v34/v35/v36 可以作为“为什么 representation-level baking 还没有完成”的 backup slide，不建议放入主定量结果。主 PPT 应明确说：当前已验证的强结果是 Phase-J；representation-level baking 的接口和瓶颈已经摸清，但最终突破仍在推进。v36 进一步说明，单纯把 train evidence 分辨率对齐并不足以解决问题，held-out surface support / barycentric coverage 才是 residual atlas 当前无法替代 ELA 的关键瓶颈。

### 8.5 v37: Visible Barycentric Coverage Fix

v37 是针对 v35/v36 直接瓶颈的工程与方法可靠性修复。它不是新的指标 endpoint，但它补上了 representation-level atlas 下一步必须具备的证据闭环：

- `ecsr_build_surface_evidence_cache.py` 新增 `--barycentric_scope visible`，在每个 view NPZ 写入时立刻保存所有可用 visible pixels 的 barycentric，而不是最后只回填 top residual supports；
- `barycentric_valid` 现在要求坐标有限且在 atlas 可用范围内，避免 float16 overflow / invalid coordinate 污染；
- 新增 `ecsr_audit_surface_residual_atlas_coverage.py`，在跑 atlas 前审计 target candidate/actionable coverage；
- `ecsr_apply_surface_residual_region_texture_adapter.py` 新增 `--min_target_changed_fraction`，防止 policy-val 通过但 held-out target 几乎 no-op 的 atlas 被误标成 accepted。

关键验证：

| check | result |
|---|---|
| synthetic barycentric smoke | barycentric sum `1.0 -> 1.0`，valid pixels `5` |
| Bonsai `images_4` one-view renderer smoke | `barycentric_valid_pixel_fraction=0.8102`，covered faces `112882`，finite all true |
| existing v36 target coverage audit | `views missing barycentric=15/37`，candidate face fraction `0.009861`，actionable fraction only `0.00001126` |
| v36 coverage-gate replay | policy-val relative gain `0.7518` but final `accepted=false` because target changed fraction `0.00000342 < 0.0001` |
| Bonsai `images_2` one-view smoke | blocked by GPU2 OOM; log shows only `650.50 MiB` free |

结论：

> v37 把 v36 的“为什么 policy-val 强但 test 不动”从猜测变成可审计事实，并修掉了 target barycentric 只覆盖 top supports 的机制问题。下一步只有在重建 full-res target evidence 后，才应该继续 residual atlas full evaluation。

## 9. 为什么这是研究工作，不是简单工程调参

当前方法的研究性主要体现在三个约束：

1. **Surface evidence certification**  
   三角形压缩不是固定比例剪枝，而是基于多视角 visibility、occlusion risk、residual region 和 sparse geometry audit 的证据认证。

2. **No-test-GT guarded policy**  
   方法选择只依赖 train/policy-val evidence。held-out test 只用于最终评价。大量 v26-v36 探针被拒绝，说明 pipeline 不是为了追 test 数字而调参。

3. **Rate-distortion and recovery loop**  
   方法同时追求 RGB、geometry safety 和 triangle reduction。相比只做图像后处理，SPCarNet 至少有一个真实 compact checkpoint；相比只删面，它又通过 ELA 补偿局部外观 residual。

更准确的定位：

> SPCarNet 当前已经是一个强的 train-evidence-certified repair loop，但还不是完全 representation-internal 的最终形态。Phase-J 是可汇报 endpoint；surface-addressed teacher residual basis 是下一步论文级升级方向。

## 10. 当前短板

| 短板 | 现状 | 风险 | 下一步 |
|---|---|---|---|
| 最强 RGB 收益仍来自 render-time ELA | Phase-J 很强，但 ELA 不是完全 baked checkpoint | 容易被质疑为渲染阶段 adapter | 将 teacher residual 写入 surface-addressed basis |
| 定性 full-frame 差异不总是显著 | 全图差异常是 residual-level | PPT 中肉眼冲击力不足 | 主图使用 local error-reduction showcase + crop evidence |
| representation-level 分支收益稀疏 | v31-v37 有真实 checkpoint edit / adapter / long training / coverage repair，但 Bonsai full-res atlas rerun 尚未完成 | 顶会主线仍需更强 representation story | 用 v37 visible barycentric 重建 full-res target evidence 后，再做大 support patch/region texture；或重新定位为 hybrid repair |
| residual atlas target 覆盖过低 | v35 changed fraction `0.000007`，v36 changed fraction `0.0000034`；v37 已实装 coverage audit 和 target changed-fraction gate | train-only gate 正向但实际 test 渲染几乎 no-op | 已有机制修复；下一步需在有足够 GPU 时重建 `images_2` target evidence 并重跑 atlas |
| v31 carrier 覆盖太小 | Bonsai 只接受 `12` faces / `+36` vertices | 工程可用但指标接近 no-op | 使用 render-visible region carrier |
| v33 高容量仍弱 | shared-field 接受 `49` faces / `+147` vertices，仅 `+0.00096` PSNR 且 SSIM/LPIPS 微退 | 还不能替代 ELA | 更大 support 的 residual texture/atlas 或 render-time repair 主线 |
| 室内压缩率较低 | room/counter/kitchen micro-budget 约 `2.10%` | rate-distortion 数字不如室外强 | 按 geometry safety 解释，不强推破坏性压缩 |

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

主信息：

- MeshSplatting 已经很强；
- 但局部 residual、拓扑冗余和迭代过拟合仍存在；
- 我们不替换它，而是在它上面构建 self-diagnosis / self-repair loop。

### Slide 3: Method Overview

放 pipeline：

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

展示 v26-v37：

- 多个探针真实接入并验证；
- gate 拒绝弱结果；
- v30 说明 image-level teacher-bake 不够；
- v31/v32/v33 说明 surface-addressed interface 已通，但 sparse face-local/field materialization 不足；
- v35/v36 说明 residual atlas 接口有效，但 held-out target support 覆盖太稀疏；
- v37 说明该 coverage 瓶颈已被审计并部分修复，但还没完成 full-res 指标闭环；
- 下一步是先修 evidence coverage，再做 region/patch-level teacher residual basis。

### Slide 12: Takeaway and Next Step

结论：

```text
Current: strong, fair, full9 baseline-beating repair loop.
Next: bake the strongest render-time repair into a persistent surface-addressed representation.
```

## 12. Mentor Q&A 备答

### Q1: 这是不是只是在调参数？

不是。当前主线包含真实 checkpoint-safe topology rewrite、train-view evidence mining、no-test-GT guarded policy 和 held-out audit。v26-v37 很多探针被拒绝，反而证明 pipeline 没有靠 test-set tuning。

### Q2: 和 MeshSplatting baseline 是否真正公平？

主 claim 使用本地同协议 selected clean MeshSplatting。clean `26000/30000` 只用 held-out score 选更强 clean baseline。我们的 alpha、branch 和压缩策略不使用 test GT。

### Q3: 是否全面超过 MeshSplatting paper table？

按当前汇总均值，Phase-J mean `26.4828 / 0.7837 / 0.2243` 高于 paper table `24.78 / 0.728 / 0.310`。但 PPT 中更严谨的 claim 应是相对本地 selected clean baseline，因为 paper table 不是同代码、同环境、同选择规则的直接 baseline。

### Q4: 如果主要收益来自 ELA，会不会被认为不是 representation 方法？

这是当前最大边界。可以诚实表述为：当前 accepted endpoint 是 compact mesh + train-evidence render-time repair。它已经包含真实 compact checkpoint，但最强 appearance gain 尚未完全 baked into representation。下一步是 surface-addressed teacher residual basis。

### Q5: 为什么不提高三角形减少比例？

因为目标不是单一压缩率。室内和 garden 等场景 geometry 已经很敏感，强推删面会破坏 sparse geometry 或 RGB。当前方法用 safety gate 保证 `9 / 9` geometry-safe。

### Q6: 定性图为什么不总是一眼明显？

因为 full-frame 改善通常是 residual-level，PSNR/SSIM/LPIPS 的全局累计明显，但肉眼看全图可能不强。PPT 应使用 Phase-J local held-out error-reduction 图展示局部真实改进，同时保留 full-frame gallery 说明公平对比。

### Q7: v31-v36 是不是已经解决 representation-level baking？

还没有。v31-v37 的价值是把 teacher residual cache、barycentric surface evidence、render-visible region carrier、face-local fitting、shared residual field、residual texture atlas 和 target coverage audit 接口打通。结果显示 policy-val proxy 正向，但 v35/v36 的 target changed fraction 低到 `1e-6` 量级；v37 已修复 evidence coverage 机制并能拒绝 near-noop atlas，但 full-res rerun 还没完成。它是下一步更大 support residual representation 的工程底座，不是当前可主讲 endpoint。

## 13. 推荐主讲边界

可以主讲：

- Phase-J 相对 selected clean MeshSplatting full9 `9 / 9` 三指标严格胜出；
- mean `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS；
- `244 / 246` per-view strict wins；
- mean triangle reduction `7.6479%`；
- `9 / 9` geometry-safe；
- policy 不使用 held-out test GT；
- paper table 作为 sanity check 高于原论文均值。

不要主讲成已经完成：

- 不要说 representation-level baking 已经全面解决；
- 不要把 v31-v37 的 Bonsai pilot 当作 headline；
- 不要说所有收益都已经写入 checkpoint；
- 不要把 paper table 对比当成唯一公平 claim；
- 不要夸大 full-frame 定性肉眼差异。

## 14. 下一步路线

当前最合理的下一步不是继续 global teacher loss，也不是继续单 face 小容量拟合，而是：

```text
teacher_render - compact_parent_render
  -> surface evidence cache with face/barycentric support
  -> render-visible region / patch carrier discovery
  -> full target barycentric / support coverage with v37 visible mode
  -> higher-capacity SH, atlas, or low-rank residual basis
  -> train-only robustness gate
  -> multi-scene held-out audit
```

目标：

- 保留 Phase-J 的 train-only fairness；
- 把 ELA 的局部 residual 能力变成 persistent surface-addressed representation；
- 避免 v30 image-level topology-frozen teacher-bake 的容量不足；
- 避免 v31/v32/v33 sparse face-local edit 吸收不了 ELA image-space residual 的瓶颈；
- 用 v37 coverage audit / target changed-fraction gate 避免 v35/v36 residual atlas 在 held-out target 上几乎 no-op 的假接受；
- 形成更像“表示升级”的论文主贡献。

## 15. 汇报用 90 秒版本

> 我们的目标不是重做一个新表示，而是在 MeshSplatting 这个强 mesh baseline 上做训练证据驱动的压缩和修复。系统先在训练视角上采集 surface visibility、residual 和 occlusion evidence，再判断哪些三角形可以安全删除，并真实改写 checkpoint。然后我们用 Evidence Lumigraph Adapter 把训练 residual 转移到 held-out view，所有 alpha、edge fallback 和 branch 都只用 train/policy-val evidence 决定，test GT 只用于最终评价。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地 selected clean MeshSplatting 做到 9/9 场景 PSNR、SSIM、LPIPS 严格提升，平均提升 +1.3311 PSNR、+0.0347 SSIM、-0.0634 LPIPS，同时平均减少 7.65% 三角形，并且 244/246 个 held-out views 三指标严格胜出。当前边界是最强外观收益仍来自 render-time ELA；我们已经验证简单 teacher-bake 不够，并在 v31-v37 把 teacher residual 的 surface-addressed 接口、region carrier、shared-field capacity、residual atlas 和 target coverage audit 打通。v37 已经把 held-out coverage 近似为零的问题定位并修复到 evidence 生成机制层，但 full-res `images_2` atlas rerun 还受 GPU 显存条件限制。下一步要用 visible barycentric 重建 full target evidence，再做更大 support 的 residual texture/atlas，或把 Phase-J 明确定位为 train-evidence render-time repair 主贡献。
