# SPCarNet 当前方法完整技术报告（Mentor PPT 版）

日期：2026-06-24

用途：mentor 汇报、PPT 拆页、方法讨论、后续实验交接

当前最适合汇报的主方法：`Phase-J guarded adaptive Evidence Lumigraph Adapter`

当前最新固定策略分支：`v64 fixed auto bin-alpha policy`

---

## 0. 一页摘要

SPCarNet 的目标不是重新发明 MeshSplatting，而是在 MeshSplatting 已训练好的强 mesh checkpoint 上，增加一套训练证据驱动的压缩、修复和回退机制。

最通俗的一句话：

> MeshSplatting 是训练完直接渲染；SPCarNet 是训练完以后先做自检：哪些三角形冗余可以删，哪些局部区域总是画错可以修，哪些区域证据不足必须保持原样。

当前可汇报结论分两层：

| 层级 | 当前状态 | PPT 讲法 |
|---|---|---|
| Phase-J endpoint | 已经在本地 full9 同协议下全面击败 clean MeshSplatting，并减少 triangles | 这是当前主结果和最稳 headline |
| v48-v64 representation track | 已实现 surface residual atlas、view-conditioned basis、bin-alpha calibration 和 v64 fixed auto policy，但效果仍小 | 这是把 render-time repair 内化成 persistent surface representation 的研究路线 |

主结果数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| Mip-NeRF360 selected full9 scene wins | `9 / 9` |
| Mean dPSNR | `+1.331084` |
| Mean dSSIM | `+0.034702` |
| Mean dLPIPS | `-0.063359` |
| Per-view strict RGB wins | `244 / 246` |
| Mean triangle reduction | `7.6479%` |
| Geometry-safe scenes | `9 / 9` |

最新 v64 数字：

| 对比 | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |

一句话结论：

> 当前主线已经能可靠证明 SPCarNet 相比基础 MeshSplatting 有明确收益；最新 v64 证明表示级 residual repair 可以被固定自动策略安全吸收，但还没有强到替代 Phase-J 主结果。

---

## 1. 研究背景和问题定义

MeshSplatting 的核心优势是把神经渲染结果放进显式 mesh 表示中。它天然适合编辑、压缩和部署，但基础 checkpoint 仍有三个问题：

| 问题 | 表现 | SPCarNet 要解决什么 |
|---|---|---|
| 局部外观误差 | 叶片、桌面、边缘、细纹理等区域仍有稳定残差 | 从训练视角 residual 中挖掘可迁移修复信号 |
| 几何冗余 | 很多 triangles 对多视角解释贡献低 | 在不破坏质量的前提下删去冗余面 |
| 泛化风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 区域 | 用 train-only policy gate 决定修复、回退或 no-op |

研究问题可以表述为：

> Given a trained MeshSplatting checkpoint, can we use training-view surface evidence to certify where the mesh can be compacted and where appearance residuals can be safely repaired?

这不是简单的图像后处理问题，因为修复信号必须绑定到 mesh surface、visibility、face id、barycentric/UV bin 和多视角一致性上。

---

## 2. 方法总览

基础 MeshSplatting pipeline：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render held-out views
```

SPCarNet pipeline：

```text
training images + cameras
  -> train/load MeshSplatting checkpoint
  -> build train/policy-val evidence cache
  -> estimate surface support, residual, risk, visibility
  -> geometry-safe compaction
  -> guarded residual repair
  -> train-only policy selection and fallback
  -> audited held-out evaluation
```

核心模块：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Evidence Cache | clean/compact renders, GT, cameras, mesh visibility | residual、face id、normal、barycentric、support、risk | 把训练视角转成可审计证据 |
| Geometry-Safe Compaction | mesh + evidence | compact mesh | 删除低风险冗余 triangles |
| Guarded ELA | compact render + train residual evidence | repaired render | 用训练残差修复 held-out view |
| Train-Only Policy | policy-val metrics and risk | accept/fallback/no-op | 防止 test leakage 和 risky edit |
| Surface Residual Atlas | face/UV/bin evidence | persistent residual field | 将 render-time repair 内化到 surface 表示 |
| v64 Fixed Auto Policy | v63b audits + v56 fallback | selected full9 tree | 固定自动选择 v63b 或回退 v56 |

---

## 3. 模块细节

### 3.1 Evidence Cache

Evidence cache 是整个方法的核心数据层。它缓存训练视角和 policy-val 视角中的：

- rendered RGB；
- ground-truth RGB；
- residual map：`GT - Render`；
- alpha / visibility；
- face id；
- barycentric coordinate 或 UV bin；
- normal、depth、camera center；
- per-face 和 per-bin sample count；
- residual sign consistency；
- PSNR、SSIM、LPIPS、image L1；
- per-view mean、min-view、CVaR tail risk。

它的作用是把训练视角从“只用来优化 checkpoint 参数”升级成“用来判断 surface 是否可靠、是否可修、是否必须回退”的证据来源。

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩目标不是最大化删面比例，而是 quality-first rate-distortion：

```text
remove triangles only if evidence says the edit is low-risk
```

关键原则：

- 低可见性、低贡献、低风险 faces 优先；
- 边界、遮挡敏感区域和 sparse geometry 风险区受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- topology audit 和 sparse geometry audit 独立记录；
- 修复失败时允许回退，不强行修改每个场景。

这里的 `triangle reduction` 指删去的 triangles 占比。当前 Phase-J 平均删面 `7.6479%`。

### 3.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。可以把它理解成 mesh surface 上的“证据约束残差迁移”。

简化公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 的像素；
- `residual_i(u_i)` 来自训练视角的局部 residual；
- `w_i(p)` 由可见性、几何对应、local support 和风险门控决定；
- `alpha` 由 train/policy-val evidence 选择；
- 如果 evidence 不足或 tail risk 高，则回退或 no-op。

通俗讲法：

> 我们不是凭空增强图片，而是看训练视角里同一个 surface 区域是否反复出现稳定错误。如果错误稳定且多视角支持充足，就把这部分 residual 迁移到目标视角；否则不动。

### 3.4 Train-Only Policy 和 Fallback

SPCarNet 强调 fairness：

- branch selection、alpha、support、fallback 都来自 train 或 policy-val evidence；
- held-out test GT 只用于最终汇报；
- 不用 test metric 选择参数；
- 不用 train metric 选择更久训练的 baseline；
- 风险高时回退稳定版本。

Phase-J 中，绝大多数场景使用 adaptive alpha，`treehill` 使用 auto edge fallback。这一分支选择不是看 held-out test 得出的。

### 3.5 Surface Residual Atlas

为了把 Phase-J 的 render-time repair 推向可部署的 persistent representation，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face/bin selection
  -> residual atlas fitting
  -> policy-val risk gate
  -> target view surface lookup
  -> persistent surface-addressed repair
```

这个分支不是重新训练一个大模型，而是从训练证据中估计一个 mesh-attached residual field。它回答的是：

> 能否把训练视角 residual 变成绑定在 mesh 表面的、可审计的、可回退的修复表示？

目前结论是：接口和安全策略已经成型，但效果量级还很小，仍不适合作为主 headline。

### 3.6 v64 Fixed Auto Bin-Alpha Policy

v61/v62 的诊断表明，只靠“这个 bin 能不能修”的二值门控不够。错误的 residual magnitude 即使作用区域很小，也可能伤害 held-out view。

v63/v63b 因此加入 bin-level residual magnitude calibration：

```text
alpha_bin = argmin_alpha || residual_gt - alpha * residual_pred ||^2
```

v64 再把 v63b 变成固定自动策略：

```text
if v63b has strong train/policy-val bin-alpha evidence:
    use v63b bin-alpha residual atlas
else:
    fallback to v56 selected policy
```

固定 guard：

| Condition | Threshold |
|---|---:|
| accepted atlas | true |
| local alpha mode | `policy_val_bin_alpha` |
| bin-alpha count | `[32, 256]` |
| selected alpha | `[0.5, 1.0]` |
| policy-val relative gain | `>= 0.05` |
| policy-val SSIM gain | `>= 0.0003` |
| policy-val SSIM positive fraction | `>= 1.0` |
| policy-val image-L1 gain | `>= 0.00004` |
| policy-val image-L1 positive fraction | `>= 1.0` |
| policy-val image-L1 min-view gain | `>= 0.00001` |

这条规则不写场景名，也不读取 held-out test metric 做选择。它自动接受 kitchen 的 v63b，并回避 counter 等不稳定候选。

---

## 4. 与基础 MeshSplatting 的区别

| 维度 | 基础 MeshSplatting | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | 继续 evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | compact mesh + topology audit |
| 外观 | checkpoint 属性直接渲染 | guarded residual adapter / surface residual atlas |
| 风险控制 | 无显式 policy gate | train-only gate、CVaR/min-view、fallback |
| Test GT 用途 | 评价 | 只评价，不参与策略选择 |
| 失败处理 | 没有显式回退 | gate 不通过则 no-op 或回退稳定版本 |

mentor 面前建议这样讲：

> MeshSplatting 给了我们一个强 starting point；SPCarNet 不推翻它，而是把它变成可自检、可压缩、可局部修复的系统。

---

## 5. 主定量结果：Phase-J vs Clean MeshSplatting

### 5.1 Full9 per-scene result

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | tri red. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | 11.81% |
| flowers | adaptive alpha | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | 11.82% |
| garden | adaptive alpha | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | 3.47% |
| stump | adaptive alpha | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | 11.82% |
| treehill | auto edge fallback | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | 11.81% |
| room | adaptive alpha | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | 2.10% |
| counter | adaptive alpha | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | 2.10% |
| kitchen | adaptive alpha | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | 2.10% |
| bonsai | adaptive alpha | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | 11.80% |

### 5.2 Aggregate result

| Metric | Value |
|---|---:|
| Scenes | `9` |
| Strict RGB wins vs selected clean | `9 / 9` |
| Mean dPSNR vs clean | `+1.331084` |
| Mean dSSIM vs clean | `+0.034702` |
| Mean dLPIPS vs clean | `-0.063359` |
| Mean dPSNR vs source ELA | `+0.833143` |
| Mean dSSIM vs source ELA | `+0.018946` |
| Mean dLPIPS vs source ELA | `-0.039986` |
| Mean triangle reduction | `7.6479%` |

PPT headline：

> On Mip-NeRF360 full9, SPCarNet Phase-J strictly improves PSNR, SSIM, and LPIPS on all 9 selected scenes while removing 7.65% triangles on average.

---

## 6. 与 MeshSplatting 论文口径的关系

现有可引用对比：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

严谨说法：

- 可以说本地 Phase-J 数值高于 MeshSplatting paper table。
- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准。
- 论文表格可能存在 resolution、mask、split、preprocessing、evaluator 差异，不能把 paper table 当成唯一公平 baseline。
- 我们本地 clean baseline 不是故意挑弱，而是从 clean envelope 中选择 held-out 更强 row。

PPT 里建议写：

> We report the main claim against our locally reproduced clean MeshSplatting baseline under the same data, rendering, and evaluator protocol. Paper-table numbers are shown only as external context.

---

## 7. 最新 v64 结果：固定自动策略

### 7.1 Full9 candidate status

v64 已经补齐所有 v63b full9 candidate，并保留 W&B online logging：

| scene | W&B run | v64 decision | reason |
|---|---|---|---|
| bicycle | `jvfx3s6s` | v56 fallback | v63b policy-val rejected |
| flowers | `8e692zyt` | v56 fallback | v63b policy-val rejected |
| garden | `xtsz2bry` | v56 fallback | v63b policy-val rejected |
| stump | `lifyawln` | v56 fallback | v63b policy-val rejected |
| treehill | `byyuaduj` | v56 fallback | v63b policy-val rejected |
| room | `xaho7gyq` | v56 fallback | weak alpha/gain evidence |
| counter | `rlctknlk` | v56 fallback | weak alpha/gain evidence and too many bins |
| kitchen | `tyqm9u38` | v63b bin-alpha | strong policy-val evidence |
| bonsai | `olt8riwt` | v56 fallback | only 2 reliable bins |

### 7.2 v64 selected rows

| scene | selected source | PSNR | SSIM | LPIPS | dPSNR vs v56 | dSSIM vs v56 | dLPIPS vs v56 |
|---|---|---:|---:|---:|---:|---:|---:|
| bicycle | v56 fallback | 23.294018 | 0.659658 | 0.332266 | +0.000000 | +0.000000 | +0.000000 |
| flowers | v56 fallback | 19.668833 | 0.511683 | 0.394785 | +0.000000 | +0.000000 | +0.000000 |
| garden | v56 fallback | 24.742157 | 0.754090 | 0.247972 | +0.000000 | +0.000000 | +0.000000 |
| stump | v56 fallback | 25.180920 | 0.704420 | 0.294214 | +0.000000 | +0.000000 | +0.000000 |
| treehill | v56 fallback | 20.923422 | 0.564226 | 0.406127 | +0.000000 | +0.000000 | +0.000000 |
| room | v56 fallback | 28.740660 | 0.884829 | 0.249897 | +0.000000 | +0.000000 | +0.000000 |
| counter | v56 fallback | 26.756130 | 0.862126 | 0.251691 | +0.000000 | +0.000000 | +0.000000 |
| kitchen | v63b bin-alpha | 27.822626 | 0.876538 | 0.198849 | +0.003691 | +0.000003 | -0.000171 |
| bonsai | v56 fallback | 28.868467 | 0.896088 | 0.259204 | +0.000000 | +0.000000 | +0.000000 |

### 7.3 v64 aggregate

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |
| v64 vs v50 | 9 | 6 | 6 | +0.000991609 | +0.000016345 | -0.000059394 |

正确解读：

- v64 是一个真实 fixed-policy milestone；
- 它把 v63b 从手工 probe 变成了 train-policy-val 自动选择策略；
- 它能自动选择 kitchen 的 v63b 并回退其他场景；
- 它相对 v56 保持 `9 / 9` non-regressive/tie；
- 但它的收益太小，不能作为论文终局 headline。

---

## 8. 消融和负面诊断

这一阶段最有价值的不是每次都变强，而是逐步定位了 residual field 的真正短板。

| 版本 | 改动 | 结果 | 教训 |
|---|---|---|---|
| v48 | auto-support surface residual atlas | vs no-op full9 `7 / 9` strict，mean dPSNR `+0.001462` | surface atlas 可行，但量级小 |
| v52 | capacity-aware fixed policy | vs no-op `7 / 9` strict，mean dPSNR `+0.001549` | 固定 train-only policy 更稳 |
| v56 | face-alpha reliability guard | vs v52 `9 / 9` non-regressive/tie | 可以安全吸收 counter 小收益 |
| v59 | view-conditioned residual basis | counter/kitchen mixed | 表达力增强会带来 OOD 风险 |
| v60 | view-basis OOD guard | counter 小幅正，kitchen mixed | OOD fallback 必要但不足 |
| v61 | face-level gain guard | counter/kitchen 弱于关键 reference | face-level gain 不能保证 held-out 安全 |
| v62 | bin uncertainty guard | changed area 很小仍可能退化 | 只缩 apply mask 不能根治 residual magnitude |
| v63/v63b | bin-level alpha calibration | kitchen 严格超过 v52/v60，counter 仍失败 | magnitude calibration 有效但需要自动策略 |
| v64 | fixed auto bin-alpha policy | full9 `9/9` nonreg/tie vs v56 | 自动策略闭环，但收益仍小 |

关键反思：

> 表示级 residual atlas 的核心瓶颈不是“有没有 gate”，而是 residual magnitude、view-conditioned generalization、occlusion boundary 和 low-support bins 的联合不确定性。

---

## 9. 定性展示建议

全图直接放进 PPT 时，residual-level improvement 可能不够显眼。建议使用三层定性证据：

1. 全图对比：证明同一场景、同一视角、同一 evaluator。
2. 局部 crop：突出叶片、桌面边缘、细纹理、遮挡边界。
3. Error map：展示 residual 降低位置，这是最能证明局部修复价值的图。

推荐直接用于 PPT 的图片：

| 用途 | 路径 |
|---|---|
| Phase-J where-it-helps showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| Mip-NeRF360 full9 全图 gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| Mip-NeRF360 full9 crop gallery | `assets/spcarnet_m360_full9_crop_gallery.png` |
| where-it-helps backup | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| outdoor detail backup | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| v52 capacity/cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v56 counter face-alpha panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |

PPT 讲法：

> 全图负责公平性，crop 和 error map 负责解释改进位置。我们不夸大全图肉眼差异，而是强调 residual-level repair 的局部稳定收益。

---

## 10. 为什么这是研究工作，不是工程调参

| 贡献 | 研究含义 | 为什么不是普通调参 |
|---|---|---|
| Training evidence mining | 从训练视角恢复 surface reliability 和 residual risk | 决策来源是可复用 evidence interface |
| Geometry-safe compaction | 质量优先的 mesh rate-distortion | 不只是删面比例，而是有 topology/geometry audit |
| Guarded residual repair | 多视角 residual transfer 绑定 surface support | 不是任意图像滤镜 |
| Train-only policy/fallback | 证据不足自动 no-op | 有部署安全逻辑，避免 test leakage |
| Surface residual atlas | 把 repair 内化到 face/UV-addressed 表示 | 是 representation-level internalization |
| Bin-alpha calibration | 从二值 allowlist 变成局部 residual magnitude calibration | 针对 failure mode 的模型结构改造 |
| Fixed auto policy | 自动选择和回退，不写场景名 | 减少手工场景参数游戏 |

论文故事可以这样组织：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. It uses training-view surface evidence to decide where to remove geometry, where to transfer residual corrections, and where to safely fall back.

---

## 11. Fairness 和可复现性

需要主动说明的公平性边界：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean envelope 中选择 held-out 更强 row；
- train metric 不用于选择“训练更久”的 baseline；
- held-out test GT 只用于最终评价；
- branch、alpha、support、fallback 不读取 held-out test metric；
- W&B online logging 覆盖 v63b/v64 full9 candidate；
- v64 结果是 report-only fixed policy，尚不包装成 paper endpoint；
- 表示级分支没有 promotion 时，不混入 Phase-J 主结果。

可复现主路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html
```

核心代码路径：

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
scripts/car_model/summarize_v64_bin_alpha_auto_policy.py
scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
```

---

## 12. 当前局限

| 局限 | 影响 | PPT 中如何诚实处理 |
|---|---|---|
| Phase-J 最强收益来自 render-time ELA | 可能被质疑不是 fully persistent representation | 主动承认，并把 v48-v64 作为内化路线 |
| v64 表示级收益很小 | 不能替代 Phase-J headline | 说成 fixed-policy milestone，不说成终局 |
| 全图定性差异不总明显 | mentor 可能觉得视觉不够强 | 用 crop 和 error map 展示局部 residual repair |
| counter 等场景仍暴露 residual magnitude 风险 | 说明 persistent residual field 尚未根治 | 强调下一步是 uncertainty-certified residual field |
| 与 paper table 口径可能不同 | 不能过度宣称 external SOTA | 主 claim 以本地同协议 baseline 为准 |

最重要的诚实结论：

> 当前我们已经解决了“能否击败基础 MeshSplatting baseline”的核心证明问题，但还没有完全解决“把强收益全部内化为 persistent surface representation”的终局问题。

---

## 13. Mentor 可能问的问题和建议回答

| 问题 | 建议回答 |
|---|---|
| 你们相比原始 MeshSplatting 到底强在哪里？ | 我们在本地同协议 full9 上 9/9 场景严格提升 PSNR/SSIM/LPIPS，同时平均减少 7.65% triangles。 |
| 这个是不是只是后处理？ | Phase-J 有 render-time adapter 成分，但它不是任意图像后处理，而是由训练视角 surface evidence、visibility、face support 和 risk gate 约束的 residual transfer。我们也在推进 v48-v64 的 persistent atlas 内化路线。 |
| 最新 v64 是不是最终方法？ | 不是。v64 是最新 fixed auto policy milestone，证明 bin-alpha residual atlas 可以自动安全选择，但收益还太小，不能替代 Phase-J 主结果。 |
| 会不会用了 test set 调参数？ | 不会。alpha、fallback、branch selection 使用 train/policy-val evidence；held-out test 只用于最终 reporting。 |
| clean baseline 是否公平？ | 本地 clean baseline 从 clean envelope 中选择 held-out 更强 row。我们不使用 train metric 选择更久训练的 checkpoint。 |
| 视觉上为什么有些图差异不明显？ | 方法主要修复局部 residual，全图缩放后不一定显眼。应看局部 crop 和 error map。 |
| 论文卖点是什么？ | Evidence-certified compact-and-repair MeshSplatting：用训练证据决定删面、修复和回退，让 mesh representation 更紧凑、更准确、更安全。 |

---

## 14. PPT 拆页建议

| Slide | 标题 | 内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Problem | MeshSplatting strong but has residual error, geometry redundancy, tail-view risk |
| 3 | Key Idea | 用训练视角证据判断 where to compact, repair, fallback |
| 4 | Pipeline | MeshSplatting -> evidence cache -> compaction -> guarded repair -> eval |
| 5 | Evidence Cache | residual、face id、visibility、barycentric、support、risk |
| 6 | Geometry Compaction | quality-first triangle reduction with topology audit |
| 7 | Guarded ELA | residual transfer formula and fallback gate |
| 8 | Fair Protocol | local clean baseline, train-only policy, test-only reporting |
| 9 | Main Quant Result | Phase-J full9 `9/9`, mean dPSNR `+1.3311` |
| 10 | Per-Scene Table | 9 scenes, RGB deltas, triangle reduction |
| 11 | Qualitative | full image + crop + error map |
| 12 | Representation Track | v48-v64 surface atlas and bin-alpha policy |
| 13 | Latest v64 | full9 fixed auto policy, kitchen selected, others fallback |
| 14 | Why Research | evidence-certified compact-and-repair representation |
| 15 | Limitations | render-time component, small persistent gains, visual subtlety |
| 16 | Next Step | uncertainty-certified persistent residual field |

---

## 15. 可直接放进 PPT 的中文摘要

SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态 mesh checkpoint，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些 triangles 可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。在 Mip-NeRF360 selected full9 上，当前 Phase-J endpoint 相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。最新 v64 fixed auto policy 进一步把表示级 bin-alpha residual atlas 从手工 probe 推进到自动选择和回退策略，在 full9 上相对 v56 保持 `9/9` non-regressive/tie，但效果量级仍小，说明下一步需要更强的 uncertainty-certified persistent residual field。

---

## 16. 可直接放进 PPT 的英文摘要

SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, it mines training-view surface evidence, removes low-risk redundant triangles, and transfers reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 selected full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction. The latest v64 branch completes a fixed auto policy over full9 v63b bin-alpha candidates, selecting kitchen automatically while falling back elsewhere; it is non-regressive over v56 but still exposes the need for stronger uncertainty-certified persistent residual fields.

---

## 17. 最终汇报口径

可以大胆说：

- 我们已经在本地同协议 full9 上全面超过基础 MeshSplatting baseline；
- 我们不是只提高 RGB，还同时减少 triangles；
- 方法核心是训练证据驱动的 compact-and-repair，而不是 test-set 参数游戏；
- v64 已经把最新 residual atlas probe 固化成自动策略。

不要过度说：

- 不要说 v64 已经是论文终局；
- 不要说 persistent residual atlas 已经全面解决；
- 不要只拿 paper table 做唯一公平对比；
- 不要把全图视觉差异夸大，应使用 crop 和 error map。

Bottom line：

> Phase-J is the current strong and presentation-safe endpoint. v64 is the latest fixed-policy representation milestone. The next paper-level leap is to make the persistent residual field strong enough that it visibly and quantitatively inherits the Phase-J gains.

