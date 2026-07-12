# SPCarNet 当前方法完整技术报告（Mentor/PPT 版）

日期：2026-06-24
用途：mentor 汇报、PPT 拆页、方法讨论、后续实验交接
当前 presentation-safe endpoint：`ours_26000_phasej_guarded_adaptedge_ela`
当前 representation-level 研究线：`v48 -> v52 -> v56 -> v59 -> v60 -> v61`

---

## 0. 先给 mentor 的核心结论

SPCarNet 的当前最强、最适合对外汇报版本是 Phase-J：

> SPCarNet 把 MeshSplatting 从“训练完直接渲染的静态 mesh checkpoint”，升级成“训练证据驱动的可压缩、可修复、可回退 mesh 表示”。它用训练视角 evidence 判断哪里可以安全删三角形，哪里可以迁移 residual 修复外观，哪里风险过高必须 no-op/fallback。

在当前本地同协议 Mip-NeRF360 full9 上，Phase-J 相对 selected clean MeshSplatting baseline 已闭合：

| 项目 | 当前结果 |
|---|---:|
| 场景 | `9` |
| Baseline | 本地 selected clean MeshSplatting，从 clean `26000/30000` envelope 选择更强 clean row |
| RGB scene strict wins | `9 / 9` |
| Mean RGB delta vs clean | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| Per-view strict RGB wins | `244 / 246` held-out views |
| Mean triangle reduction | `7.6479%`，表示删去的三角形占比 |
| Sparse geometry strict wins | `6 / 9` |
| Geometry-safe scenes | `9 / 9` |

必须诚实区分：

| 分支 | 能否当主结果 | 说明 |
|---|---|---|
| Phase-J guarded adaptive ELA | 可以 | full9 已闭合，是当前 PPT 主线 |
| v48/v52/v56 surface residual atlas | 可以作为表示级路线 | 证明 residual repair 能往 face/UV 表示内化，但收益远小于 Phase-J |
| v59 view-conditioned basis | 不能 | 真实方法改动，但 counter/kitchen mixed |
| v60 View-Basis OOD Guard | 不能 | counter 相对 v52 小幅严格正，kitchen mixed，未 promotion |
| v61 Region-Level Face Gain Guard | 不能 | 已实现并完成 counter/kitchen W&B probe，但两场景均弱于关键 reference，不能 promotion |

一句话安全汇报：

> 当前真正可以主讲的是 Phase-J：它在本地强 clean MeshSplatting baseline 上实现 full9 RGB 全场景严格胜出，并同时减少 mesh triangles；最新 v48-v61 是把这个强 render-time repair 进一步内化到 persistent surface representation 的研究路线，目前还未替代 Phase-J。

---

## 1. 背景和问题定义

MeshSplatting 的优点是输出 mesh，天然适合传统图形管线、编辑、压缩、AR/VR 和数字孪生部署。但训练好的 MeshSplatting checkpoint 仍有几个可改进空间：

| 问题 | 具体表现 | SPCarNet 的切入点 |
|---|---|---|
| 外观 residual | 叶片、桌面、边缘、纹理细节仍有系统误差 | 从训练视角 residual 中挖掘稳定修复信号 |
| 几何冗余 | 部分 triangles 对多视角解释贡献低 | 做 topology-safe / geometry-safe compaction |
| 少数视角风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 区域 | train-only risk gate + fallback/no-op |
| 表示内化不足 | 强外观收益目前主要来自 render-time adapter | 继续推进 face/UV-addressed residual atlas 和 view-conditioned residual field |

研究问题可以表述为：

> Given a trained MeshSplatting checkpoint, can we certify where the mesh can be compacted and where residual appearance evidence can be safely transferred, using training views only?

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

核心差别：

| 维度 | clean MeshSplatting | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | 继续做 evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | compact mesh，带 topology / sparse geometry audit |
| 外观 | checkpoint 属性直接渲染 | guarded residual adapter / surface residual atlas |
| 策略 | 无显式风险判断 | train-only policy gate、CVaR/min-view guard、fallback |
| Test GT | 只评价 | 只评价，不参与方法分支和参数选择 |
| 失败处理 | 没有显式 no-op 机制 | gate 不通过则 no-op 或回退稳定版本 |

适合 PPT 的表达：

> MeshSplatting 给了我们一个强 mesh starting point；SPCarNet 不是推翻它，而是让这个 mesh 有“自诊断、自压缩、自修复”的后训练闭环。

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

对训练视角和 policy-val 视角渲染 MeshSplatting checkpoint，并缓存：

- rendered RGB；
- GT RGB；
- residual map：`GT - Render`；
- visibility / alpha；
- face id；
- barycentric coordinate；
- normal、depth、camera center；
- residual support、variance、sign consistency；
- per-view PSNR/SSIM/LPIPS/L1 风险指标。

作用：

> 把训练视角从“只用于训练”变成“用于判断 surface 哪些区域可信、哪里可修、哪里该回退”的证据来源。

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩目标不是极限删面，而是 quality-first rate-distortion：

- 删除多视角贡献低、风险低的 triangles；
- 保护 sparse / occlusion-sensitive 区域；
- 检查 renderer 可加载性；
- 做 topology audit 和 sparse geometry audit；
- 与 residual repair 共同形成最终 endpoint。

当前 Phase-J 平均 triangle reduction 是 `7.6479%`。这里的 reduction 是“删去的占比”，不是剩余占比。

### 3.3 Evidence Lumigraph Adapter

Phase-J 的主要外观收益来自 guarded Evidence Lumigraph Adapter。直观公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 像素；
- `residual_i(u_i)` 来自训练 view 的局部残差；
- `w_i(p)` 由可见性、几何对应、局部 support 和 risk gate 决定；
- `alpha` 由 train/policy-val evidence 选择；
- 风险门不通过时不应用修复。

通俗解释：

> MeshSplatting 先画出整体结果；SPCarNet 再看训练视角里哪些地方反复画错、而且错得很一致，然后只把这些稳定 residual 迁移到目标视角。证据不足的区域不碰。

### 3.4 Guarded Adaptive Policy

Phase-J 外层有一个 train-only 安全策略：

- 大多数 stable scenes 使用 adaptive alpha；
- `treehill` 这类 adaptive alpha 风险较高的场景走 auto edge fallback；
- alpha、edge fallback、branch selection 来自 train/policy-val evidence；
- held-out test GT 只用于最终 reporting；
- policy-val 风险高时自动 fallback。

### 3.5 Surface Residual Atlas

Phase-J 的局限是最强外观收益仍是 render-time adapter。为把贡献推向更“表示级”的论文故事，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face / UV bin selection
  -> residual atlas fitting
  -> train policy-val risk gate
  -> target surface map lookup
  -> persistent surface-addressed repair
```

这条线对应 v48/v52/v56/v59/v60/v61。

---

## 4. Phase-J 主结果

### 4.1 Per-scene result

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.0215 | 0.7024 | 0.2661 | +0.7199 | +0.0425 | -0.0660 | 11.81% |
| flowers | adaptive alpha | 20.3044 | 0.5578 | 0.3292 | +0.6221 | +0.0459 | -0.0653 | 11.82% |
| garden | adaptive alpha | 26.3111 | 0.8278 | 0.1358 | +1.2819 | +0.0478 | -0.0655 | 3.47% |
| stump | adaptive alpha | 25.5951 | 0.7241 | 0.2639 | +0.3901 | +0.0189 | -0.0301 | 11.82% |
| treehill | auto edge fallback | 21.2962 | 0.5956 | 0.3363 | +0.3620 | +0.0311 | -0.0697 | 11.81% |
| room | adaptive alpha | 30.3056 | 0.9057 | 0.1960 | +1.5584 | +0.0209 | -0.0539 | 2.10% |
| counter | adaptive alpha | 28.4492 | 0.8937 | 0.1865 | +1.6974 | +0.0317 | -0.0655 | 2.10% |
| kitchen | adaptive alpha | 30.1997 | 0.9161 | 0.1320 | +2.3812 | +0.0396 | -0.0672 | 2.10% |
| bonsai | adaptive alpha | 31.8620 | 0.9303 | 0.1726 | +2.9668 | +0.0339 | -0.0869 | 11.80% |

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
| Sparse geometry-safe scenes | `9 / 9` |

### 4.3 与 MeshSplatting paper table 的口径关系

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

推荐讲法：

- 可以说：本地 Phase-J 数值高于 MeshSplatting paper table。
- 更严谨的主 claim：SPCarNet 超过我们本地同协议复现并选择出的 stronger clean MeshSplatting baseline。
- 不建议把 paper table 当唯一公平 baseline，因为 resolution、mask、split、preprocessing、evaluator 可能不完全一致。

---

## 5. 表示级内化路线和版本脉络

### 5.1 v48 Auto-Support Surface Residual Atlas

目标：证明 residual repair 可以从 render-time adapter 转成 face/UV-addressed surface representation。

结果：

| Comparison | Strict scene wins | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs same-evidence no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

结论：正向但效果量级很小，不能替代 Phase-J。

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

结果：

| Comparison | Strict | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v52 vs no-op | `7 / 9` | `8 / 9` | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | `3 / 9` | `9 / 9` | `+0.000086890` | `+0.000008782` | `-0.000015303` |

结论：这是固定策略，不是每场景调参；但收益仍小。

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

结果：

| Comparison | Strict | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v56 vs v52 | `1 / 9` | `9 / 9` | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 vs no-op | `7 / 9` | `8 / 9` | `+0.001845890` | `+0.000037803` | `-0.000074494` |

结论：v56 是当前最稳的表示级安全候选，但不能作为 headline，因为净收益基本来自 `counter`。

### 5.4 v59 Surface-Aware View-Conditioned Basis

目标：让 residual atlas 随视角和表面法向变化：

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

probe 结果：

| Scene | W&B run | PSNR | SSIM | LPIPS | Guard decision |
|---|---|---:|---:|---:|---|
| counter | `oiwl6r88` | 26.7536087036 | 0.8621008992 | 0.2518228889 | keep_view_basis |
| kitchen | `08bwukw3` | 27.8192043304 | 0.8765304089 | 0.1990223229 | keep_view_basis |

结论：真实方法改动，但结果 mixed，不 promotion。

### 5.5 v60 View-Basis OOD Guard

目标：修复 v59 的局部视角泛化风险。对每个 face/UV bin 统计 fit-view feature mean/std；target feature 若超出局部分布，则回退 mean residual atlas。

新增接口：

```text
--view_conditioned_basis_ood_mode {none,diag_z}
--view_conditioned_basis_ood_max_z 2.5
--view_conditioned_basis_ood_min_std 0.05
```

clean probe：

| Scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | effective basis |
|---|---|---:|---:|---:|---:|---:|---|
| counter | `d9tozw7s` | 26.7539958954 | 0.8621192575 | 0.2518530488 | true | 0.065630 | normal_camera_linear |
| kitchen | `924sxfsd` | 27.8191566467 | 0.8765332103 | 0.1990308464 | true | 0.039585 | normal_camera_linear |

相对 reference：

| Scene | Reference | dPSNR | dSSIM | dLPIPS | Verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | +0.0005359650 | +0.0000045896 | -0.0000152886 | strict small positive vs v52 |
| counter | v56/raw v55d | -0.0021343231 | -0.0000069737 | +0.0001616776 | worse than stronger counter reference |
| kitchen | v52 | +0.0002212524 | -0.0000021457 | +0.0000114292 | mixed |

结论：v60 修复了接口和 correctness 问题，并带来诊断进展，但未满足 promotion criteria。

### 5.6 v61 Region-Level Face Gain Guard

目标：解决 v60 的下一层失败模式：全局 policy-val 非退化，不代表每个 surface face 都非退化。v61 增加 face-level no-regression allowlist：

> target 上只对那些在 train policy-val 上有足够样本、足够正 view 支持、且 residual prediction 确实降低误差的 faces 应用 residual。

核心规则：

```text
keep face if
  samples >= face_gain_guard_min_face_samples
  and relative_gain >= face_gain_guard_min_relative_gain
  and positive_view_fraction >= face_gain_guard_min_positive_view_fraction
```

已实现接口：

```text
--enable_policy_val_face_gain_guard
--face_gain_guard_min_face_samples
--face_gain_guard_min_relative_gain
--face_gain_guard_min_positive_view_fraction
```

probe 结果：

| Scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | allowed / candidate faces |
|---|---|---:|---:|---:|---:|---:|---:|
| counter | `a9bf3hbb` | 26.7510070801 | 0.8620729446 | 0.2519522905 | true | 0.017889 | 577 / 5628 |
| kitchen | `nhyjuth5` | 27.8172969818 | 0.8764855266 | 0.1991390735 | true | 0.012064 | 817 / 4333 |

相对关键 reference：

| Scene | Reference | dPSNR | dSSIM | dLPIPS | Verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | -0.0024528503 | -0.0000417233 | +0.0000839531 | worse |
| counter | v56/raw v55d | -0.0051231384 | -0.0000532866 | +0.0002609193 | worse |
| counter | v60 | -0.0029888153 | -0.0000463129 | +0.0000992417 | worse |
| kitchen | v52 | -0.0016384125 | -0.0000498294 | +0.0001196563 | worse |
| kitchen | v60 | -0.0018596649 | -0.0000476837 | +0.0001082271 | worse |

验证状态：

- `py_compile` passed；
- adapter/runner `--help` flags exposed；
- synthetic face allowlist smoke passed；
- counter/kitchen clean probe completed with W&B online logging；
- 结论是 negative diagnostic：face-level allowlist 确实降低了 changed fraction，但没有保护 held-out PSNR/SSIM/LPIPS，不能作为当前方法主结果。

---

## 6. 为什么这是研究工作，不只是工程调参

| 贡献 | 研究意义 | 为什么不是简单调参 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源是可复用 evidence interface，不是看 test 后挑参数 |
| Geometry-safe compaction | 在 topology/sparse geometry audit 下删面 | 目标是 rate-distortion-safe mesh representation |
| Guarded residual repair | 多视角 residual transfer 被 visibility/support/risk 约束 | 不是任意图像滤镜 |
| Train-only policy/fallback | 证据不足自动 no-op | 有部署安全逻辑，不强行每场景修改 |
| Surface residual atlas | 把 render-time ELA 推向 face/UV-addressed 表示 | 是 representation-level internalization |
| View-conditioned/OOD/region guard | 检查 residual 是否跨视角、跨区域可泛化 | 暴露并解决泛化边界，而不是只追单点数字 |

论文故事：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. The current closed endpoint is Phase-J, and the ongoing frontier is uncertainty-certified persistent surface residual fields.

---

## 7. 定性展示建议

全图缩到 PPT 后，SPCarNet 的 residual-level 优势不一定肉眼明显。因此建议三层展示：

1. 公平全图：证明同一场景、同一 held-out view、同一 evaluator。
2. 局部 crop：突出叶片、边缘、纹理、桌面等细节区域。
3. Error map：展示残差下降的位置，这是最能说服 mentor 的图。

推荐图片：

| 用途 | 图片 |
|---|---|
| 主推 local improvement | `../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| full9 全图 gallery | `../../assets/spcarnet_m360_full9_qualitative_gallery.png` |
| full9 crop gallery | `../../assets/spcarnet_m360_full9_crop_gallery.png` |
| where-it-helps backup | `../../assets/spcarnet_m360_where_it_helps_showcase.png` |
| v52 support/capacity panel | `../../assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v56 counter face-alpha panel | `../../assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |

PPT 讲法：

> 全图用于证明公平协议，crop 和 error map 用于证明改进发生在哪里。我们不夸大全图肉眼差异，而是用局部误差下降解释 residual-level repair 的价值。

---

## 8. Fairness 和可复现口径

必须主动说明：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean `26000/30000` envelope 中选择更强者；
- train metrics 不用于选择最终 clean baseline；
- held-out test GT 只用于最终评价，不参与 method branch、alpha、support、texture/fill、fallback 选择；
- Phase-J 的 branch selection 来自 train/policy-val evidence；
- v59/v60/v61 probe 使用 W&B online logging，保留 command/config/result path/error；
- 表示级分支没有 promotion 时，不混入主 claim。

容易被问到的问题：

| 问题 | 推荐回答 |
|---|---|
| 为什么不用 clean30000 固定对比？ | 更久训练不一定更好；我们从 clean envelope 中选择 held-out 更强 clean row，是为了更严格。 |
| Phase-J 是不是后处理？ | 它包含 render-time ELA，所以不能包装成 fully baked representation；但 residual、alpha、fallback 都由 train-view surface evidence 和 risk gate 决定，不是任意后处理。 |
| v60/v61 是不是最新最好？ | 不是。它们是表示级内化路线的最新机制，v60 未 promotion，v61 仍在 probe。 |
| 如果全图看不明显怎么办？ | 展示 crop/error map。定量上 full9 9/9 strict、244/246 per-view strict；定性上需要看局部残差修复。 |
| out-of-trajectory 会不会崩？ | 风险门和 fallback/no-op 正是为此设计；v60/v61 进一步增加 OOD 和 face-level no-regression guard。 |

---

## 9. 当前短板和下一步

| 短板 | 影响 | 当前应对 |
|---|---|---|
| Phase-J 最强收益来自 render-time ELA | 论文中可能被质疑为后处理 | v48-v61 推进 persistent surface residual representation |
| 表示级 atlas 收益太小 | 很难替代 Phase-J headline | 扩大 support coverage，引入局部 alpha、view-conditioned basis、region gain guard |
| v56 净收益几乎只来自 counter | 多场景说服力不足 | 保持为安全候选，不作为主结果 |
| v59/v60/v61 mixed | view-conditioned residual 泛化仍不稳 | v61 证明 face allowlist 还不够，需要更强 uncertainty-certified residual field |
| 全图视觉差异不总明显 | 汇报说服力不足 | 主展示 crop/error map，而不是只放 full-frame |
| Paper table 口径不同 | 不能过度 claim | 主 claim 以本地同协议 selected clean baseline 为准 |

下一步优先级：

1. 不 promote v61；它在 counter/kitchen 上均弱于关键 reference。
2. 下一步不应继续扫 face guard 阈值，而应转向更强 uncertainty-certified surface residual field。
3. 若继续表示级路线，应引入显式 uncertainty、multi-view consistency 和 residual magnitude calibration，而不是只做局部 allowlist。
4. 为 PPT 优先补齐最清晰的 crop/error-map 对比图，弱化 full-frame 肉眼不可见的问题。

---

## 10. PPT 拆页建议

| Slide | Title | 内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Motivation | MeshSplatting 输出 mesh，但仍有 residual error、冗余 triangles、tail-view risk |
| 3 | One-Sentence Idea | 用训练视角 evidence 判断哪里能删、哪里能修、哪里必须回退 |
| 4 | Pipeline | MeshSplatting -> evidence cache -> compaction -> guarded repair -> evaluation |
| 5 | Difference | clean MeshSplatting vs SPCarNet 表格 |
| 6 | Method Details | Evidence cache、compaction、ELA、risk gate |
| 7 | Fair Protocol | selected clean baseline、train-only selection、test-only reporting |
| 8 | Main Quant Result | Phase-J full9 `9/9` strict，mean `+1.3311` PSNR |
| 9 | Per-Scene Table | 9 个场景结果和 triangle reduction |
| 10 | Geometry/Compactness | `7.6479%` mean triangle reduction，`9/9` geometry-safe |
| 11 | Main Qualitative | crop + error map |
| 12 | Representation Track | v48/v52/v56 的 surface residual atlas |
| 13 | Latest Diagnostics | v59/v60 mixed，v61 face-level guard 已验证为 negative diagnostic |
| 14 | Why Research | evidence-certified compact-and-repair representation，不是调参 |
| 15 | Limitations | render-time ELA、表示级收益小、视觉全图差异细 |
| 16 | Next Steps | uncertainty-certified persistent surface residual field |

---

## 11. 可直接放进 PPT 的文案

中文摘要：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。后续 v48-v61 则把这一路线从 render-time repair 推向 persistent surface residual representation，并暴露出 support coverage、view-conditioned generalization 和 OOD safety 是下一步核心挑战。

英文摘要：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

---

## 12. Artifact 和证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
```

当前 mentor 报告：

```text
docs/car_model/6-24-SPCarNet-Mentor-PPT-Current-Technical-Report-v61.zh.md
```

上一版 v60 报告：

```text
docs/car_model/6-24-SPCarNet-Mentor-PPT-Complete-Technical-Report-v60.zh.md
```

v61 log：

```text
docs/car_model/6-24-v61-RegionFaceGainGuard-Log.md
```

定性图：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_full9_crop_gallery.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
```

v60 W&B：

```text
counter: d9tozw7s
kitchen: 924sxfsd
```

v61 W&B：

```text
counter: a9bf3hbb
kitchen: nhyjuth5
```

---

## 13. 最终汇报立场

推荐立场：

> 我们已经有一个可以清楚击败本地 clean MeshSplatting baseline 的 Phase-J endpoint，并且它同时带来 RGB 指标提升和 triangle reduction。这个结果足够作为当前方法主线向 mentor 汇报。与此同时，我们也必须承认 Phase-J 的最强外观收益仍来自 render-time ELA，所以论文终局还需要把它进一步内化成 persistent surface representation。v48-v61 是这条内化路线的连续实验，目前证明了接口和诊断链路可行，但 v61 再次说明只靠局部 allowlist 不足以替代 Phase-J，需要更强 uncertainty-certified surface residual field。

一句话 bottom line：

> 主结果已经能讲，终局论文方法还没完全闭合；下一步不是继续“调参游戏”，而是继续推进 uncertainty-certified surface residual representation，并证明它能稳定跨场景带来可见收益。
