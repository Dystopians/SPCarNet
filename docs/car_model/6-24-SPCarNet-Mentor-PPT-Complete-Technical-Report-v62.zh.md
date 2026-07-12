# SPCarNet 当前完整技术报告（Mentor/PPT 版，v62 更新）

日期：2026-06-24
用途：mentor 汇报、PPT 拆页、方法讨论、实验交接
当前可汇报主方法：`Phase-J guarded adaptive Evidence Lumigraph Adapter`
当前表示级研究线：`v48 -> v52 -> v56 -> v59 -> v60 -> v61 -> v62`

---

## 0. 一句话结论

SPCarNet 的当前最强可汇报结果不是最新 v62 probe，而是已经闭合的 Phase-J endpoint：

> SPCarNet 将 MeshSplatting 从“训练完直接渲染的静态 mesh checkpoint”，升级为“训练证据驱动的可压缩、可修复、可回退 mesh 表示”。它利用训练视角 evidence 判断哪些 triangles 可以安全删除，哪些区域存在稳定 residual 可以修复，哪些区域证据不足必须 no-op/fallback。

当前本地同协议 Mip-NeRF360 full9 上，Phase-J 相对本地 selected clean MeshSplatting baseline 已经闭合：

| 项目 | 结果 |
|---|---:|
| 场景数 | `9` |
| RGB strict scene wins vs selected clean | `9 / 9` |
| Mean dPSNR vs clean | `+1.331084` |
| Mean dSSIM vs clean | `+0.034702` |
| Mean dLPIPS vs clean | `-0.063359` |
| Per-view strict RGB wins | `244 / 246` |
| Mean triangle reduction | `7.6479%` |
| Sparse geometry strict wins | `6 / 9` |
| Geometry-safe scenes | `9 / 9` |

需要对 mentor 诚实区分：

| 分支 | 是否作为主结果 | 汇报定位 |
|---|---|---|
| Phase-J guarded adaptive ELA | 是 | 当前 presentation-safe 主线，RGB+compactness 已闭合 |
| v48/v52/v56 surface residual atlas | 可作为 ablation/表示级路线 | 证明 residual repair 可内化到 face/UV 表示，但收益很小 |
| v59/v60 view-conditioned atlas | 不作为主结果 | 真实方法改动，暴露跨视角泛化风险 |
| v61 face-level gain guard | 不作为主结果 | 已完成 W&B probe，负面诊断 |
| v62 bin-level uncertainty guard | 不作为主结果 | 已完成 W&B probe，负面诊断，说明“只缩 mask”不够 |

推荐开场：

> 我们已经有一个强的 Phase-J endpoint，可以在本地强 clean MeshSplatting baseline 上实现 9/9 场景 RGB 严格胜出，同时减少 triangles。论文终局仍需进一步把 Phase-J 的 render-time repair 内化为更强的 persistent surface residual representation；v48-v62 是这条内化路线的完整实验链和风险分析。

---

## 1. 问题背景

MeshSplatting 的优势是显式 mesh 表示：可渲染、可编辑、可压缩，也更接近图形管线和部署需求。但基础 MeshSplatting checkpoint 存在三个关键缺口：

| 缺口 | 表现 | SPCarNet 的目标 |
|---|---|---|
| 外观 residual | 细纹理、边缘、叶片、桌面等局部区域仍有稳定误差 | 从训练视角 residual 中挖掘可迁移修复信号 |
| 几何冗余 | 部分 triangles 对多视角解释贡献低 | 在 topology/sparse geometry audit 下安全删面 |
| 泛化风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 区域 | train-only policy gate + no-op/fallback |

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
  -> build train-view evidence cache
  -> estimate surface reliability / residual / support
  -> geometry-safe compaction
  -> guarded residual repair
  -> train-only policy selection + fallback
  -> audited held-out evaluation
```

核心差异：

| 维度 | MeshSplatting baseline | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | 继续 evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | compact mesh，带 topology/sparse geometry audit |
| 外观 | checkpoint 属性直接渲染 | guarded residual adapter / surface residual atlas |
| 风险控制 | 无显式 policy gate | train-only risk gate、CVaR/min-view、fallback/no-op |
| Test GT | 只评价 | 只评价，不参与策略分支或参数选择 |
| 失败处理 | 没有显式回退 | gate 不通过则 no-op 或回退稳定版本 |

PPT 表述：

> MeshSplatting 给了我们一个强 mesh starting point；SPCarNet 让这个 mesh 具备“自诊断、自压缩、自修复”的训练后闭环。

---

## 3. 方法总览

SPCarNet 当前由五层组成：

```text
Clean MeshSplatting checkpoint
  -> Evidence Cache
  -> Geometry-Safe Compaction
  -> Guarded Residual Repair
  -> Train-Only Policy / Fallback
  -> Audited Evaluation
```

### 3.1 Evidence Cache

对训练视角和 policy-val 视角渲染 clean/compact checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual map：`GT - Render`；
- visibility / alpha；
- face id；
- barycentric coordinate；
- normal、depth、camera center；
- residual support、variance、sign consistency；
- per-view PSNR/SSIM/LPIPS/L1 风险指标。

这一步的作用是把训练视角从“只用于训练参数”升级为“用于判断 surface 哪些区域可信、哪里可修、哪里该回退”的证据来源。

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩不是盲目极限删面，而是 quality-first rate-distortion：

- 删除多视角贡献低、风险低的 triangles；
- 保护 sparse / occlusion-sensitive 区域；
- 检查 renderer 可加载性；
- 记录 topology audit 和 sparse geometry audit；
- 与 residual repair 共同形成最终 endpoint。

当前 Phase-J 平均 triangle reduction 为 `7.6479%`。这里的 reduction 是“删去的 triangles 占比”，不是剩余占比。

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

### 3.5 Surface Residual Atlas

Phase-J 的局限是最强外观收益仍来自 render-time adapter。为推进更“表示级”的论文故事，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face / UV bin selection
  -> residual atlas fitting
  -> train policy-val risk gate
  -> target surface map lookup
  -> persistent surface-addressed repair
```

这条线对应 v48/v52/v56/v59/v60/v61/v62。

---

## 4. 主结果：Phase-J

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

推荐严谨说法：

- 可以说本地 Phase-J 数值高于 MeshSplatting paper table。
- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准，因为 paper table 可能存在 resolution、mask、split、preprocessing、evaluator 差异。
- selected clean baseline 来自本地 clean `26000/30000` envelope 中更强 row，不是故意挑弱 baseline。

---

## 5. 表示级内化路线

### 5.1 v48 Auto-Support Surface Residual Atlas

目标：证明 residual repair 可以从 render-time adapter 转成 face/UV-addressed surface representation。

| Comparison | Strict scene wins | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs same-evidence no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

结论：正向但效果量级小，不能替代 Phase-J。

### 5.2 v52 Capacity-Aware Policy

固定 train-only policy：

```text
if v48 accepted
   and v48 support hits cap
   and v51 accepted
   and v51 adds larger support
   and policy-val SSIM is non-regressive:
       use v51
else:
       keep v48
```

| Comparison | Strict | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v52 vs no-op | `7 / 9` | `8 / 9` | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | `3 / 9` | `9 / 9` | `+0.000086890` | `+0.000008782` | `-0.000015303` |

结论：v52 是固定策略，不是 per-scene test-set 参数游戏；但收益仍很小。

### 5.3 v56 Face-Alpha Reliability Guard

v55d 发现 `counter` 有 per-face alpha 正信号，但 `kitchen/bonsai` 有风险。v56 固化成 guard：

```text
use v55d only if
  accepted_atlas
  and local_alpha_profile.enabled
  and face_alpha_count >= 128
  and selected_alpha <= 0.5
  and selected_image_l1_positive_view_fraction >= 0.9
  and selected_ssim_min_view_gain >= 5e-5
else
  fallback to v52
```

| Comparison | Strict | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v56 vs v52 | `1 / 9` | `9 / 9` | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 vs no-op | `7 / 9` | `8 / 9` | `+0.001845890` | `+0.000037803` | `-0.000074494` |

结论：v56 是当前最稳的表示级安全候选，但不能作为 headline，因为净收益几乎只来自 `counter`。

### 5.4 v59/v60 View-Conditioned Residual Basis

目标：让 residual atlas 随视角和表面法向变化，而不是每个 face/UV bin 只有一个 mean residual。

feature：

```text
[1,
 normalized_camera_center_x,
 normalized_camera_center_y,
 normalized_camera_center_z,
 normal_x,
 normal_y,
 normal_z,
 dot(normal, normalized_camera_center)]
```

v60 在 v59 基础上加入 OOD fallback：

```text
--view_conditioned_basis_ood_mode diag_z
--view_conditioned_basis_ood_max_z 2.5
--view_conditioned_basis_ood_min_std 0.05
```

v60 clean probe：

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| counter | `d9tozw7s` | 26.7539958954 | 0.8621192575 | 0.2518530488 | true | 0.065630 | strict small positive vs v52, still worse than v56 |
| kitchen | `924sxfsd` | 27.8191566467 | 0.8765332103 | 0.1990308464 | true | 0.039585 | mixed vs v52 |

结论：真实方法改动，但未达到 promotion criteria。

### 5.5 v61 Region-Level Face Gain Guard

目标：解决 v60 的 failure mode：全局 policy-val 非退化不代表每个 surface face 都安全。

规则：

```text
keep face if
  samples >= face_gain_guard_min_face_samples
  and relative_gain >= face_gain_guard_min_relative_gain
  and positive_view_fraction >= face_gain_guard_min_positive_view_fraction
```

结果：

| scene | W&B run | PSNR | SSIM | LPIPS | changed fraction | allowed / candidate faces | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| counter | `a9bf3hbb` | 26.7510070801 | 0.8620729446 | 0.2519522905 | 0.017889 | 577 / 5628 | worse than v52/v56/v60 |
| kitchen | `nhyjuth5` | 27.8172969818 | 0.8764855266 | 0.1991390735 | 0.012064 | 817 / 4333 | worse than v52/v60 |

结论：face-level allowlist 确实缩小了 target changed area，但没有保护 held-out RGB；负面诊断有效，不能 promotion。

### 5.6 v62 Bin-Level Uncertainty Guard

目标：比 v61 更细粒度，不按 face，而按 `(face_id, uv_bin)` 做 policy-val uncertainty/no-regression guard。

规则：

```text
keep bin if
  samples >= bin_uncertainty_guard_min_bin_samples
  and relative_gain >= bin_uncertainty_guard_min_relative_gain
  and positive_view_fraction >= bin_uncertainty_guard_min_positive_view_fraction
  and optional mean_variance / sign_consistency checks pass
```

新增接口：

```text
--enable_policy_val_bin_uncertainty_guard
--bin_uncertainty_guard_min_bin_samples
--bin_uncertainty_guard_min_relative_gain
--bin_uncertainty_guard_min_positive_view_fraction
--bin_uncertainty_guard_max_mean_variance
--bin_uncertainty_guard_min_mean_sign_consistency
```

验证：

- adapter/runner `py_compile` passed；
- adapter/runner `--help` flags exposed；
- synthetic bin allowlist smoke passed；
- counter/kitchen W&B probe completed。

结果：

| scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | allowed / candidate bins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| counter | `tdu3x70o` | 26.7498836517 | 0.8620498180 | 0.2519963086 | true | 0.000196 | 74 / 250224 | worse than v52/v56/v60/v61 |
| kitchen | `wnse9uz8` | 27.8163890839 | 0.8764428496 | 0.1992005110 | false | 0.0 | n/a | no-op fallback, still worse than v52/v60/v61 |

结论：

> v62 证明“只在 apply 阶段把 mask 缩得更保守”不能解决表示级 residual 的核心问题。真正瓶颈不是 guard 粒度不够，而是 residual field 本身的 magnitude calibration、multi-view consistency 和 uncertainty modeling 不够强。

---

## 6. Ablation 和经验教训

| 版本 | 做了什么 | 结果 | 教训 |
|---|---|---|---|
| Phase-J | guarded adaptive ELA + compaction | 9/9 strict vs clean，7.65% triangles reduction | 当前最强主结果 |
| v48 | auto-support residual atlas | 7/9 strict vs no-op，小幅正 | surface atlas 可行，但容量不足 |
| v52 | capacity-aware fixed policy | 9/9 non-regressive/tie vs v48 | 固定策略能防回退，但收益小 |
| v56 | face-alpha reliability guard | 9/9 non-regressive/tie vs v52，小幅正 | 局部 alpha 有价值，但泛化有限 |
| v59 | view-conditioned basis | counter/kitchen mixed | 表达力增强会带来 OOD 风险 |
| v60 | view-basis OOD guard | counter 小幅正，kitchen mixed | OOD fallback 有必要但不足 |
| v61 | face-level gain guard | 两场景均弱于关键 reference | policy-val face 正收益不保证 held-out 安全 |
| v62 | bin-level uncertainty guard | counter 极小 changed area 仍退化，kitchen no-op | 只缩 apply mask 不是根治，需要建模 residual uncertainty |

核心反思：

1. Phase-J 之所以强，是因为它直接用多视角 evidence 做 guarded residual transfer，覆盖面和效果量级足够大。
2. 表示级 atlas 当前收益小，不是因为没有 guard，而是因为 persistent face/UV residual field 对视角相关性、残差幅值、遮挡边界和 support uncertainty 的表达仍不够。
3. 继续扫阈值很难越过 v52/v56/F82 这类固定 policy 上限；下一步需要建模层面的升级。

---

## 7. 为什么这是研究工作，不只是工程调参

| 贡献 | 研究意义 | 为什么不是简单调参 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源是可复用 evidence interface，不是看 test 后挑参数 |
| Geometry-safe compaction | 在 topology/sparse geometry audit 下删面 | 目标是 rate-distortion-safe mesh representation |
| Guarded residual repair | 多视角 residual transfer 被 visibility/support/risk 约束 | 不是任意图像滤镜 |
| Train-only policy/fallback | 证据不足自动 no-op | 有部署安全逻辑，不强行每场景修改 |
| Surface residual atlas | 把 render-time ELA 推向 face/UV-addressed 表示 | 是 representation-level internalization |
| View-conditioned/OOD/region/bin guards | 系统诊断 residual 是否跨视角、跨区域可泛化 | 暴露并定位泛化边界，而不是只追单点数字 |

论文故事：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. The current closed endpoint is Phase-J, and the ongoing frontier is uncertainty-certified persistent surface residual fields.

---

## 8. 定性展示建议

全图缩到 PPT 后，SPCarNet 的 residual-level 改进不一定肉眼明显。因此建议三层展示：

1. 公平全图：证明同一场景、同一 held-out view、同一 evaluator。
2. 局部 crop：突出叶片、边缘、纹理、桌面等细节区域。
3. Error map：展示残差下降的位置，这是最能说服 mentor 的图。

推荐图片：

| 用途 | 图片 |
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

## 9. Fairness 和可复现口径

必须主动说明：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean `26000/30000` envelope 中选择更强者；
- train metrics 不用于选择最终 clean baseline；
- held-out test GT 只用于最终评价，不参与 method branch、alpha、support、texture/fill、fallback 选择；
- Phase-J 的 branch selection 来自 train/policy-val evidence；
- v52 source-config rerun 已完成 full9 reproduction；
- v59-v62 probe 使用 W&B online logging，保留 command/config/result path/error；
- 表示级分支没有 promotion 时，不混入主 claim。

容易被问到的问题：

| 问题 | 推荐回答 |
|---|---|
| 为什么不用 clean30000 固定对比？ | 更久训练不一定更好；我们从 clean envelope 中选择 held-out 更强 clean row，是更严格的 baseline。 |
| Phase-J 是不是后处理？ | 它包含 render-time ELA，所以不能包装成 fully baked representation；但 residual、alpha、fallback 都由 train-view surface evidence 和 risk gate 决定，不是任意图像滤镜。 |
| 最新 v62 是不是最好？ | 不是。v62 是表示级内化路线的负面诊断，当前主结果仍是 Phase-J。 |
| 如果全图看不明显怎么办？ | 展示 crop/error map。定量上 full9 9/9 strict、244/246 per-view strict；定性上需要看局部残差修复。 |
| out-of-trajectory 会不会崩？ | Phase-J 有 risk gate 和 fallback/no-op；v60-v62 进一步证明仅靠局部 guard 不够，下一步需要更强 uncertainty-certified residual field。 |

---

## 10. 当前短板

| 短板 | 影响 | 当前结论 |
|---|---|---|
| Phase-J 最强收益来自 render-time ELA | 可能被质疑为后处理 | 汇报时承认；把 v48-v62 作为内化路线 |
| 表示级 atlas 收益太小 | 很难替代 Phase-J headline | v52/v56 可做安全 ablation，不做主 claim |
| v59-v62 未 promotion | persistent residual field 泛化仍不稳 | 需要建模升级，而不是继续扫 guard 阈值 |
| 全图视觉差异不总明显 | PPT 说服力不足 | 用 crop/error map 讲局部 residual repair |
| Paper table 口径不同 | 不能过度 claim | 主 claim 以本地同协议 selected clean baseline 为准 |

下一步最有价值方向：

1. 设计 uncertainty-certified residual field，而不是继续“mask 越缩越小”的 guard。
2. 显式建模 residual magnitude calibration、multi-view consistency、view-feature support 和 occlusion boundary risk。
3. 用 Phase-J 的强 render-time residual 作为 teacher，训练/蒸馏到 persistent face/UV 或 mesh-attached field。
4. 保留 v52/v56 作为可靠 ablation 和安全 fallback。

---

## 11. PPT 拆页建议

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
| 14 | Latest Diagnostics | v59/v60/v61/v62 mixed/negative，为什么不能 promotion |
| 15 | Why Research | evidence-certified compact-and-repair representation |
| 16 | Limitations | render-time ELA、表示级收益小、全图视觉差异细 |
| 17 | Next Step | uncertainty-certified persistent surface residual field |

---

## 12. 可直接放进 PPT 的文案

中文摘要：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。后续 v48-v62 则把这一路线从 render-time repair 推向 persistent surface residual representation，并暴露出 support coverage、view-conditioned generalization 和 uncertainty calibration 是下一步核心挑战。

英文摘要：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

Bottom line：

> 主结果已经能讲，终局论文方法还没完全闭合。下一步不是继续调参数，而是推进 uncertainty-certified persistent surface residual representation，并证明它能稳定跨场景带来可见收益。

---

## 13. Artifact 和证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
```

当前报告：

```text
docs/car_model/6-24-SPCarNet-Mentor-PPT-Complete-Technical-Report-v62.zh.md
```

表示级日志：

```text
docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md
docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md
docs/car_model/6-24-v60-ViewBasisOODGuard-Log.md
docs/car_model/6-24-v61-RegionFaceGainGuard-Log.md
docs/car_model/6-24-v62-BinUncertaintyGuard-Log.md
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
```

v62 probe：

```text
counter W&B: tdu3x70o
kitchen W&B: wnse9uz8
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_counter_20260624/
/dev/shm/peilincai_spcarnet_v62_bin_uncertainty_kitchen_20260624/
```

---

## 14. 最终汇报立场

推荐立场：

> 我们已经有一个清楚击败本地 clean MeshSplatting baseline 的 Phase-J endpoint，并且它同时带来 RGB 指标提升和 triangle reduction。这个结果足够作为当前方法主线向 mentor 汇报。与此同时，我们必须承认 Phase-J 的最强外观收益仍来自 render-time ELA，所以论文终局还需要把它进一步内化成 persistent surface representation。v48-v62 是这条内化路线的连续实验，目前证明了接口、审计、W&B 验证和安全 guard 都可行，但 v61/v62 的负面结果说明仅靠局部 allowlist 不能越过当前瓶颈。下一步必须做 uncertainty-certified residual field 级别的建模升级。

一句话 bottom line：

> 当前可汇报主结果强，论文终局还未 100% 闭合；最关键下一步是把 Phase-J 的强 residual repair 从 render-time adapter 蒸馏/内化为稳定、可泛化、可审计的 persistent surface field。
