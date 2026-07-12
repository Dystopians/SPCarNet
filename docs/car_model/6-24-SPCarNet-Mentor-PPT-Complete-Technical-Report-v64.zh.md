# SPCarNet 当前完整技术报告（Mentor/PPT 版，v64 更新）

日期：2026-06-24
用途：mentor 汇报、PPT 拆页、方法讨论、实验交接

当前可汇报主方法：`Phase-J guarded adaptive Evidence Lumigraph Adapter`
当前表示级研究线：`v48 -> v52 -> v56 -> v59 -> v60 -> v61 -> v62 -> v63/v63b -> v64`

---

## 0. Executive Summary

SPCarNet 的核心目标是把 MeshSplatting 从“训练好以后直接渲染的静态 mesh checkpoint”，升级成“训练证据驱动的可压缩、可修复、可回退 mesh 表示”。

一句话讲法：

> MeshSplatting 给出一个强 mesh starting point；SPCarNet 再用训练视角 evidence 判断哪些 triangles 可以安全删、哪些局部 residual 可以可靠修、哪些区域证据不足必须保持 no-op/fallback。

当前最适合给 mentor 汇报的主结论仍然是已经闭合的 Phase-J endpoint：

| 项目 | 结果 |
|---|---:|
| Dataset / protocol | Mip-NeRF360 full9，本地同协议 held-out eval |
| Baseline | 本地 selected clean MeshSplatting baseline |
| Strict RGB scene wins | `9 / 9` |
| Mean dPSNR vs clean | `+1.331084` |
| Mean dSSIM vs clean | `+0.034702` |
| Mean dLPIPS vs clean | `-0.063359` |
| Per-view strict RGB wins | `244 / 246` |
| Mean triangle reduction | `7.6479%` |
| Sparse geometry strict wins | `6 / 9` |
| Geometry-safe scenes | `9 / 9` |

需要主动说明的边界：

| 分支 | 是否作为主结果 | 当前定位 |
|---|---|---|
| Phase-J guarded adaptive ELA | 是 | 当前 presentation-safe 主线，RGB+compactness 已闭合 |
| v48/v52/v56 surface residual atlas | ablation / 表示级路线 | 证明 residual repair 可以内化到 face/UV 表示，但收益很小 |
| v59/v60 view-conditioned atlas | 诊断 | 引入视角条件表达，暴露 OOD 泛化风险 |
| v61/v62 region/bin guard | 负面诊断 | 证明只缩 apply mask 不能根治 residual field 问题 |
| v63/v63b bin alpha calibration | 真实方法改造 | kitchen 上首次严格超过 v52/v60，counter 仍未过关键 reference；不能包装为全局主结果 |
| v64 fixed auto bin-alpha policy | 最新固定策略 | full9 候选补齐，train-policy-val 自动选择 kitchen v63b，其余回退 v56；相对 v56 `9/9` non-regressive/tie、`1/9` strict，但效果仍太小 |

推荐开场：

> 我们已经有一个能公平击败本地 clean MeshSplatting baseline 的 Phase-J endpoint，并且同时减少 triangles。它足够作为当前主线汇报。与此同时，论文终局仍要把 Phase-J 的 render-time repair 进一步内化为 persistent surface residual representation。v48-v64 是这条内化路线的连续实验、接口建设和风险诊断。最新 v64 已经把 v63b 的场景 probe 固化为 train-policy-val 自动策略，但提升仍很小，所以不能替代 Phase-J headline。

---

## 1. 背景和研究问题

MeshSplatting 的优势是显式 mesh 表示：可渲染、可编辑、可压缩，也更接近图形管线和部署需求。但基础 MeshSplatting checkpoint 有三个缺口：

| 缺口 | 在结果中的表现 | SPCarNet 的目标 |
|---|---|---|
| 外观 residual | 叶片、桌面、边缘、细纹理等区域仍有稳定误差 | 从训练视角 residual 中挖掘可迁移修复信号 |
| 几何冗余 | 部分 triangles 对多视角解释贡献低 | 在 topology/sparse geometry audit 下安全删面 |
| 泛化风险 | 盲目修复会伤害 tail view 或 out-of-trajectory 区域 | train-only policy gate + fallback/no-op |

研究问题：

> Given a trained MeshSplatting checkpoint, can we certify which parts of the mesh can be compacted and which surface residuals can be safely transferred, using training-view evidence only?

---

## 2. 与基础 MeshSplatting 的区别

基础 MeshSplatting：

```text
images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render held-out views
```

SPCarNet：

```text
images + cameras
  -> train/load MeshSplatting checkpoint
  -> build train/policy-val evidence cache
  -> estimate surface reliability, residual, support, risk
  -> geometry-safe compaction
  -> guarded residual repair
  -> train-only policy selection and fallback
  -> audited held-out evaluation
```

核心区别：

| 维度 | MeshSplatting baseline | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | 继续 evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | compact mesh，带 topology/sparse geometry audit |
| 外观 | checkpoint 属性直接渲染 | guarded residual adapter / surface residual atlas |
| 风险控制 | 无显式 policy gate | train-only risk gate、CVaR/min-view、fallback/no-op |
| Test GT | 只评价 | 只评价，不参与策略分支、alpha 或 fallback |
| 失败处理 | 没有显式回退 | gate 不通过则 no-op 或回退稳定版本 |

通俗版：

> MeshSplatting 是“训练完就交卷”。SPCarNet 是“训练完以后再根据训练证据做自检：哪些面冗余、哪些地方总是画错、哪些修复不可靠；可靠就改，不可靠就不碰”。

---

## 3. 方法总览

SPCarNet 当前由六层组成：

```text
Clean MeshSplatting checkpoint
  -> Evidence Cache
  -> Geometry-Safe Compaction
  -> Guarded Residual Repair
  -> Train-Only Policy / Fallback
  -> Surface Residual Atlas Internalization
  -> Audited Evaluation
```

### 3.1 Evidence Cache

对训练视角和 policy-val 视角渲染 clean/compact checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual map：`GT - Render`；
- visibility / alpha；
- face id；
- barycentric coordinate / UV bin；
- normal、depth、camera center；
- residual support、variance、sign consistency；
- per-view PSNR/SSIM/LPIPS/L1 风险指标。

这一步把训练视角从“只用于训练参数”升级为“用于判断 surface 哪些区域可信、哪里可修、哪里该回退”的证据来源。

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩不是盲目删面，而是 quality-first rate-distortion：

- 删除多视角贡献低、风险低的 triangles；
- 保护 sparse / occlusion-sensitive 区域；
- 检查 renderer 可加载性；
- 记录 topology audit 和 sparse geometry audit；
- 与 residual repair 共同形成最终 endpoint。

这里的 `triangle reduction` 是“删去的 triangles 占比”，不是剩余占比。Phase-J 平均删面 `7.6479%`，同时保持 `9 / 9` geometry-safe。

### 3.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要外观收益来自 guarded Evidence Lumigraph Adapter。直观公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 像素；
- `residual_i(u_i)` 来自训练 view 的局部 residual；
- `w_i(p)` 由可见性、几何对应、local support 和 risk gate 决定；
- `alpha` 由 train/policy-val evidence 选择；
- 风险门不通过时不应用修复。

通俗解释：

> MeshSplatting 先画出整体结果；SPCarNet 再看训练视角里哪些地方反复画错、而且错得稳定一致，然后只把这些稳定 residual 迁移到目标视角。证据不足的区域不碰。

### 3.4 Train-Only Adaptive Policy

Phase-J 外层有一个安全策略：

- 大多数 stable scenes 使用 adaptive alpha；
- `treehill` 这类 adaptive alpha 风险较高的场景走 auto edge fallback；
- alpha、edge fallback、branch selection 来自 train/policy-val evidence；
- held-out test GT 只用于最终 reporting；
- policy-val 风险高时自动 fallback。

这部分是公平性边界的核心：不能看 test 再挑参数，不能用 train metric 选择“更久训练”的 baseline。

### 3.5 Surface Residual Atlas

为了把 Phase-J 的 render-time repair 推向更强的 persistent representation，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face / UV bin selection
  -> residual atlas fitting
  -> train policy-val risk gate
  -> target surface map lookup
  -> persistent surface-addressed repair
```

它不是重新训练一个大型神经网络，而是在训练后从 residual evidence 中估计一个 mesh-attached residual field / policy。它的研究价值在于把“哪里该修、修多少、何时回退”显式绑定到 surface evidence 上。

### 3.6 v63 Bin-Level Residual Magnitude Calibration

v61/v62 证明只靠更细的 allowlist mask 不能解决问题：即使 changed area 很小，错误 residual magnitude 仍会伤害 held-out view。v63 的新思路是校准每个 `(face_id, uv_bin)` 的 residual 幅值。

核心估计：

```text
alpha_bin = argmin_alpha || residual_gt - alpha * residual_pred ||^2
```

实际实现中加入：

- policy-val only 的局部 least-squares alpha；
- alpha 上限和下限；
- per-bin sample 数量门；
- positive-view fraction 门；
- fallback global alpha；
- count / denominator shrink，防止低样本 bin 过拟合；
- 最大 profile bin 数量限制，控制复杂度；
- W&B logging 和 audit 字段。

新增接口位于：

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

关键 flags：

```text
--enable_policy_val_bin_alpha_calibration
--bin_alpha_calibration_max_alpha
--bin_alpha_calibration_min_alpha
--bin_alpha_calibration_multipliers
--bin_alpha_calibration_min_bin_samples
--bin_alpha_calibration_min_positive_view_fraction
--bin_alpha_calibration_shrink_count_tau
--bin_alpha_calibration_shrink_denominator_tau
--bin_alpha_calibration_shrink_prior
--bin_alpha_calibration_max_profile_bins
```

---

## 4. 主结果：Phase-J vs Clean MeshSplatting

### 4.1 Per-scene result

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

### 4.2 Aggregate result

| Metric | Value |
|---|---:|
| Strict RGB scene wins vs selected clean | `9 / 9` |
| Mean dPSNR vs clean | `+1.331084` |
| Mean dSSIM vs clean | `+0.034702` |
| Mean dLPIPS vs clean | `-0.063359` |
| Per-view strict RGB wins | `244 / 246` (`99.19%`) |
| Mean triangle reduction | `7.6479%` |
| Sparse geometry strict wins | `6 / 9` |
| Geometry-safe scenes | `9 / 9` |

### 4.3 与 MeshSplatting paper table 的关系

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

严谨说法：

- 可以说本地 Phase-J 数值高于 MeshSplatting paper table。
- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准，因为 paper table 可能存在 resolution、mask、split、preprocessing、evaluator 差异。
- selected clean baseline 来自本地 clean `26000/30000` envelope 中更强 row，不是故意挑弱 baseline。

---

## 5. 表示级内化路线和最新 v64 结果

### 5.1 v48-v56：安全但小效果的 surface atlas

| 版本 | 方法改动 | 关键结果 | 结论 |
|---|---|---|---|
| v48 | auto-support surface residual atlas | vs no-op full9: `7 / 9` strict，mean dPSNR `+0.001462` | surface atlas 可行，但效果量级小 |
| v52 | capacity-aware fixed policy | vs no-op: `7 / 9` strict，mean dPSNR `+0.001549` | 固定 train-only policy 更稳 |
| v56 | face-alpha reliability guard | vs v52: `9 / 9` non-regressive/tie，mean dLPIPS `-0.0000197` | 当前较稳表示级候选，但主要收益来自 counter |

### 5.2 v59-v62：表达力和 guard 的负面诊断

| 版本 | 方法改动 | 结果 | 教训 |
|---|---|---|---|
| v59 | view-conditioned residual basis | counter/kitchen mixed | 表达力增强会带来 OOD 风险 |
| v60 | view-basis OOD guard | counter 小幅正，kitchen mixed | OOD fallback 有必要但不足 |
| v61 | face-level gain guard | counter/kitchen 均弱于关键 reference | policy-val face 正收益不保证 held-out 安全 |
| v62 | bin-level uncertainty guard | counter changed area 极小仍退化，kitchen no-op | 只缩 apply mask 不是根治，需要校准 residual magnitude |

### 5.3 v63/v63b：Bin-Level Alpha Calibration

最新 v63/v63b 是真实方法改造：不再只判断“这个 bin 能不能修”，而是估计“这个 bin 应该修多少”。它是对 v61/v62 failure mode 的直接响应。

| scene | version | W&B | PSNR | SSIM | LPIPS | changed frac. | local alpha bins | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| counter | v63 max0.5 | `g4ub52pr` | 26.752016 | 0.862093 | 0.251934 | 0.065630 | 664 | better than v61/v62, worse than v52/v56/v60 |
| kitchen | v63 max0.5 | `hu5k9lyu` | 27.823883 | 0.876437 | 0.198897 | 0.039585 | 75 | PSNR/LPIPS better than v52/v60, SSIM slightly worse |
| counter | v63b max0.35 | `rlctknlk` | 26.751209 | 0.862078 | 0.251961 | 0.065630 | 667 | better than v61/v62, still worse than v52/v56/v60 |
| kitchen | v63b max0.35 | `tyqm9u38` | 27.822626 | 0.876538 | 0.198849 | 0.039585 | 77 | strict three-metric improvement over v52 and v60 |

v63b kitchen 对关键 reference 的差值：

| Reference | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| vs v52 | +0.003691 | +0.0000025 | -0.000171 |
| vs v60 | +0.003469 | +0.0000046 | -0.000182 |
| vs v61 | +0.005329 | +0.0000523 | -0.000290 |
| vs v62 | +0.006237 | +0.0000950 | -0.000352 |

v63b counter 对关键 reference 的差值：

| Reference | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| vs v52 | -0.002251 | -0.0000370 | +0.000093 |
| vs v56/raw v55d | -0.004921 | -0.0000485 | +0.000270 |
| vs v60 | -0.002787 | -0.0000415 | +0.000108 |
| vs v62 | +0.001326 | +0.0000279 | -0.000035 |

当前判断：

> v63b 是一个有价值的新里程碑，因为它首次在 kitchen 上把表示级 residual atlas 推到严格三指标超过 v52/v60；但它没有解决 counter 上的关键 reference 差距，所以不能宣称全局 promotion。它直接推动了 v64：把 v63b 的 policy-val 信号固化成 fixed auto-select/auto-reject policy。

### 5.4 v64 Fixed Auto Bin-Alpha Policy

v64 的目标是停止场景级手工判断，把 v63b 变成自动策略：

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

full9 v63b candidates 已全部跑完，并用 W&B online logging：

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

v64 aggregate：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |
| v64 vs v50 | 9 | 6 | 6 | +0.000991609 | +0.000016345 | -0.000059394 |

v64 的意义：

- v63b candidate 已补齐 full9；
- 选择规则只使用 train/policy-val audit，不使用 held-out metric 做选择；
- selected tree `9 / 9` render/GT links 有效；
- qualitative gallery 已生成；
- 相对 v56 达成 `9 / 9` non-regressive/tie，自动吸收 kitchen 的正收益并回避 counter 失败。

诚实边界：

> v64 是 fixed-policy 工程闭环，不是论文终局。它证明 bin-level magnitude calibration 可以作为自动策略安全吸收局部收益，但效果量级仍只有 `1e-4` 到 `1e-6`，不足以替代 Phase-J headline，也不足以说明 persistent residual representation 已经根本解决。

---

## 6. 为什么这是研究工作，不只是工程调参

| 贡献 | 研究意义 | 为什么不是简单调参 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源是可复用 evidence interface，不是看 test 后挑参数 |
| Geometry-safe compaction | 在 topology/sparse geometry audit 下删面 | 目标是 rate-distortion-safe mesh representation |
| Guarded residual repair | 多视角 residual transfer 被 visibility/support/risk 约束 | 不是任意图像滤镜 |
| Train-only policy/fallback | 证据不足自动 no-op | 有部署安全逻辑，不强行每场景修改 |
| Surface residual atlas | 把 render-time ELA 推向 face/UV-addressed 表示 | 是 representation-level internalization |
| v63 bin alpha calibration | 从二值 allowlist 升级到局部 residual magnitude calibration | 针对 failure mode 的模型结构改动，不是单纯扫阈值 |
| v64 fixed auto policy | 把 v63b 从场景 probe 固化成自动选择/回退策略 | 不写场景名、不看 held-out metric，选择依据是 train-policy-val audit |

论文故事：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. The current closed endpoint is Phase-J, and the ongoing frontier is uncertainty-certified persistent surface residual fields.

---

## 7. 定性展示建议

全图缩到 PPT 后，SPCarNet 的 residual-level 改进不一定肉眼明显。建议三层展示：

1. 公平全图：证明同一场景、同一 held-out view、同一 evaluator。
2. 局部 crop：突出叶片、边缘、纹理、桌面等细节区域。
3. Error map：展示残差下降的位置，这是最能说服 mentor 的图。

可直接放进 PPT 的图片：

| 用途 | 图片路径 |
|---|---|
| Phase-J 主推 local improvement | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| full9 全图 gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| full9 crop gallery | `assets/spcarnet_m360_full9_crop_gallery.png` |
| where-it-helps backup | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| outdoor detail backup | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| v52 capacity/cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v56 counter face-alpha panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |

PPT 讲法：

> 全图用于证明公平协议，crop 和 error map 用于证明改进发生在哪里。我们不夸大全图肉眼差异，而是用局部误差下降解释 residual-level repair 的价值。

---

## 8. Fairness 和可复现口径

必须主动说明：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean `26000/30000` envelope 中选择 held-out 更强 row；
- train metrics 不用于选择最终 clean baseline；
- held-out test GT 只用于最终评价，不参与 method branch、alpha、support、texture/fill、fallback 选择；
- Phase-J 的 branch selection 来自 train/policy-val evidence；
- v52 source-config rerun 已完成 full9 reproduction；
- v59-v64 probe 使用 W&B online logging，保留 command/config/result path/error；
- 表示级分支没有 promotion 时，不混入主 claim。

容易被问到的问题：

| 问题 | 推荐回答 |
|---|---|
| 为什么不用 clean30000 固定对比？ | 更久训练不一定更好；我们从 clean envelope 中选择 held-out 更强 clean row，是更严格的 baseline。 |
| Phase-J 是不是后处理？ | 它包含 render-time ELA，所以不能包装成 fully baked representation；但 residual、alpha、fallback 都由 train-view surface evidence 和 risk gate 决定，不是任意图像滤镜。 |
| 最新 v64 是不是最好？ | 在表示级路线里，v64 是当前最稳的 fixed auto policy：相对 v56 `9/9` non-regressive/tie，但只新增 kitchen 的小收益，不能替代 Phase-J 主结果。 |
| 如果全图看不明显怎么办？ | 展示 crop/error map。定量上 full9 9/9 strict、244/246 per-view strict；定性上需要看局部残差修复。 |
| out-of-trajectory 会不会崩？ | Phase-J 有 risk gate 和 fallback/no-op；v60-v64 进一步证明 persistent residual field 需要更强 uncertainty / calibration。v64 会在证据不足时回退 v56。 |

---

## 9. 当前短板和下一步

| 短板 | 影响 | 当前结论 |
|---|---|---|
| Phase-J 最强收益来自 render-time ELA | 可能被质疑为后处理 | 汇报时承认；把 v48-v64 作为内化路线 |
| 表示级 atlas 收益太小 | 很难替代 Phase-J headline | v52/v56/v64 可做 ablation，不做主 claim |
| counter reference 仍难超过 | 说明局部 residual magnitude/policy 仍不够智能 | 需要固定 auto-select/auto-reject，而不是每场景挑参数 |
| 全图视觉差异不总明显 | PPT 说服力不足 | 用 crop/error map 讲局部 residual repair |
| Paper table 口径不同 | 不能过度 claim | 主 claim 以本地同协议 selected clean baseline 为准 |

下一步最有价值方向：

1. 用 fresh blind/long-run 验证 v64，确认 fixed policy 没有过拟合 counter/kitchen probe。
2. 继续推进 uncertainty-certified residual field，显式建模 residual magnitude、multi-view consistency、view-feature support 和 occlusion boundary risk。
3. 用 Phase-J 的强 render-time residual 作为 teacher，蒸馏到 persistent face/UV 或 mesh-attached field。
4. 研究比 sparse bin alpha 更强的 residual field learner，使表示级收益从 `1e-4` 提升到肉眼和指标都明显的量级。

---

## 10. PPT 拆页建议

| Slide | Title | 内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Motivation | MeshSplatting 输出 mesh，但仍有 residual error、冗余 triangles、tail-view risk |
| 3 | Core Idea | 用训练视角 evidence 判断哪里能删、哪里能修、哪里必须回退 |
| 4 | Pipeline | MeshSplatting -> evidence cache -> compaction -> guarded repair -> evaluation |
| 5 | Difference | clean MeshSplatting vs SPCarNet 表格 |
| 6 | Evidence Cache | face id、barycentric、residual、visibility、support、risk |
| 7 | Guarded Repair | ELA 公式和 train-only alpha/fallback |
| 8 | Fair Protocol | selected clean baseline、train-only selection、test-only reporting |
| 9 | Main Quant Result | Phase-J full9 `9/9` strict，mean `+1.3311` PSNR |
| 10 | Per-Scene Table | 9 个场景结果和 triangle reduction |
| 11 | Geometry/Compactness | `7.6479%` mean triangle reduction，`9/9` geometry-safe |
| 12 | Main Qualitative | crop + error map |
| 13 | Representation Track | v48/v52/v56 surface residual atlas |
| 14 | Latest Method Probe | v64 fixed auto bin-alpha policy：full9 完成，kitchen 自动吸收，其余回退 |
| 15 | Why Research | evidence-certified compact-and-repair representation |
| 16 | Limitations | render-time ELA、表示级收益小、全图视觉差异细 |
| 17 | Next Step | fixed auto policy + uncertainty-certified persistent surface residual field |

---

## 11. 可直接放进 PPT 的文案

中文摘要：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。最新 v64 将表示级 residual atlas 从场景 probe 推进到 fixed auto policy：full9 v63b candidates 全部完成，策略自动选择 kitchen v63b、其余回退 v56，相对 v56 保持 `9/9` non-regressive/tie 并取得小幅正收益。

英文摘要：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction. The latest v64 branch completes a fixed auto policy over full9 v63b bin-alpha candidates, selecting kitchen automatically while falling back elsewhere, yielding non-regressive gains over v56 but still exposing that persistent residual fields need stronger modeling.

Bottom line：

> 主结果已经能讲，终局论文方法还没完全闭合。下一步不是继续手动调参，而是把 policy-val evidence 固化成自动策略，并推进 uncertainty-certified persistent surface residual representation。

---

## 12. Artifact 和证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
```

当前报告：

```text
docs/car_model/6-24-SPCarNet-Mentor-PPT-Complete-Technical-Report-v64.zh.md
```

表示级日志：

```text
docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md
docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md
docs/car_model/6-24-v60-ViewBasisOODGuard-Log.md
docs/car_model/6-24-v61-RegionFaceGainGuard-Log.md
docs/car_model/6-24-v62-BinUncertaintyGuard-Log.md
docs/car_model/6-24-v63-BinAlphaCalibration-Log.md
docs/car_model/6-24-v64-FixedAutoBinAlphaPolicy-Log.md
```

v63/v63b/v64 result paths：

```text
/dev/shm/peilincai_spcarnet_v63_bin_alpha_counter_20260624/
/dev/shm/peilincai_spcarnet_v63_bin_alpha_kitchen_20260624/
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_counter_20260624/
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_kitchen_20260624/
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_full9_20260624/
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/
```

W&B runs：

```text
v63 counter: g4ub52pr
v63 kitchen: hu5k9lyu
v63b counter: rlctknlk
v63b kitchen: tyqm9u38
v64 full9 additional v63b: jvfx3s6s, 8e692zyt, xtsz2bry, lifyawln, byyuaduj, xaho7gyq, olt8riwt
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_full9_crop_gallery.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html
```

---

## 13. 最终汇报立场

推荐立场：

> 我们已经有一个清楚击败本地 clean MeshSplatting baseline 的 Phase-J endpoint，并且它同时带来 RGB 指标提升和 triangle reduction。这个结果足够作为当前方法主线向 mentor 汇报。与此同时，我们必须承认 Phase-J 的最强外观收益仍来自 render-time ELA，所以论文终局还需要把它进一步内化成 persistent surface representation。v48-v64 是这条路线的连续实验，目前证明了 evidence interface、审计、W&B 验证、安全 guard、view-conditioned basis、bin-level magnitude calibration 和 fixed auto policy 都可行；但 v64 的增益仍非常小，说明下一步必须做 uncertainty-certified residual field / teacher distillation，而不是继续手工挑场景或扫阈值。

一句话 bottom line：

> 当前可汇报主结果强，论文终局还未 100% 闭合；最关键下一步是把 Phase-J 的强 residual repair 从 render-time adapter 蒸馏/内化为稳定、可泛化、可审计的 persistent surface field。
