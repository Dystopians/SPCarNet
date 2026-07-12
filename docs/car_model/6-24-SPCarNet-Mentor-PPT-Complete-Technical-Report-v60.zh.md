# SPCarNet 当前方法完整技术报告（Mentor/PPT 母稿）

日期：2026-06-24

用途：mentor 汇报、PPT 拆页、论文路线讨论、后续实验交接。

当前最稳可汇报 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`

当前最新研究分支：`v60 View-Basis OOD Guard`，已实现并完成 counter/kitchen clean probe。结果是有效诊断和局部小幅改善，但未满足 promotion criteria，因此不能作为 headline 结果。

---

## 0. 汇报前先给结论

最适合对 mentor 讲的核心结论：

> SPCarNet 不是从零替代 MeshSplatting，而是把 MeshSplatting 训练出的 mesh checkpoint 升级成一个带自检能力的 compact-and-repair 表示：用训练视角 evidence 判断哪里可以安全删三角形、哪里可以迁移 residual 修复外观、哪里风险过高必须回退。

当前证据最强的版本是 Phase-J：

| 项目 | 结果 |
|---|---:|
| Benchmark | Mip-NeRF360 full9，本地同协议 |
| Baseline | selected clean MeshSplatting，从 clean `26000/30000` envelope 中选择更强 clean baseline |
| RGB scene strict wins | `9 / 9` |
| Mean RGB delta vs selected clean | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| Per-view strict RGB wins | `244 / 246` held-out views |
| Mean triangle reduction | `7.6479%`，这里是删掉的三角形占比 |
| Geometry safety | `9 / 9` geometry-safe，`6 / 9` sparse geometry strict wins |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310`，但严格 claim 仍以本地同协议 baseline 为主 |

必须诚实说明：

- Phase-J 是当前最强结果，也是现在做 PPT 最应该主讲的版本。
- v48/v52/v56/v59/v60 是把 Phase-J 的 render-time repair 内化成 persistent surface representation 的研究路线。
- v56 是目前最稳的表示级安全候选，但收益非常小。
- v59/v60 是真实接口和真实方法改动，但 v59 结果 mixed，v60 虽完成 probe 也未满足 promotion criteria，不能包装成已解决问题。

---

## 1. 一句话版本

中文：

> SPCarNet 将 MeshSplatting 从“训练完直接渲染的静态 mesh”，升级为“训练证据驱动的可压缩、可修复、可回退的 mesh 表示”。

英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation.

给非本方向听众的解释：

> MeshSplatting 已经能产出可渲染 mesh，但训练完后局部纹理、细节边缘和冗余三角形仍有问题。我们不盲目修改整个模型，而是让模型看自己的训练视角证据：哪里多视角一致就修，哪里贡献低就删，哪里证据不足就保持原样。

---

## 2. 背景：为什么从 MeshSplatting 出发

MeshSplatting 的优势：

- 它输出 mesh，更容易进入传统图形管线、AR/VR、游戏引擎和数字孪生流程。
- 相比纯 image-space 或纯 point/Gaussian 表示，mesh 更适合后续几何编辑、压缩和部署。
- 它已经是一个强 baseline，适合作为改进对象。

它的实际短板：

| 问题 | 现象 | 对论文故事的意义 |
|---|---|---|
| 局部 residual error | 树叶、桌面、室内边界、纹理细节仍存在系统性偏差 | 训练视角 residual 中有可挖掘的修复信号 |
| Geometry redundancy | 部分 triangles 对多视角解释贡献低 | mesh 表示天然适合做 topology-safe compaction |
| Tail-view risk | 盲目修复会破坏少数视角或 out-of-trajectory 区域 | 必须有 train-only risk gate 和 fallback |
| 表示内化不足 | 最强收益来自 render-time adapter | 后续研究重点是 persistent surface residual representation |

SPCarNet 的核心假设：

> 一个训练好的 mesh checkpoint 自身已经包含很多局部可靠性线索。只要从训练视角构造 surface evidence，就能判断哪些局部修改是安全的，哪些必须回退。

---

## 3. 与基础 MeshSplatting 的区别

基础 MeshSplatting pipeline：

```text
images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render test views
```

SPCarNet pipeline：

```text
images + cameras
  -> train/load MeshSplatting checkpoint
  -> train-view evidence cache
  -> surface reliability / residual / support analysis
  -> geometry-safe compaction
  -> guarded residual repair
  -> train-only policy selection + fallback
  -> held-out render and metrics
```

关键区别：

| 维度 | clean MeshSplatting | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | 继续进行 evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | topology-safe compact mesh |
| 外观 | checkpoint 属性直接渲染 | train-evidence residual adapter / surface residual atlas |
| 策略 | 无显式风险判断 | train-only policy gate、CVaR/min-view guard、fallback |
| Test GT | 用于评价 | 只用于最终评价，不参与方法分支和参数选择 |
| 失败处理 | 没有显式安全机制 | gate 不通过就 no-op 或回退上一稳定版本 |

PPT 里建议强调：

> 我们不是“在 test 上挑一张好看的图”，而是把训练视角证据变成一个可审计的决策系统。

---

## 4. 方法总览

SPCarNet 由五个层次组成：

```text
Clean MeshSplatting checkpoint
  -> Evidence Cache
  -> Geometry-Safe Compaction
  -> Guarded Residual Repair
  -> Train-Only Policy / Fallback
  -> Audited Evaluation
```

| 模块 | 作用 | 当前状态 |
|---|---|---|
| Clean baseline envelope | 构造强 clean MeshSplatting 对照 | clean `26000/30000` envelope，本地 selected clean baseline |
| Evidence cache | 缓存 train views 的 render、GT、residual、face/barycentric/normal/camera evidence | 支撑 Phase-J 和 v48 到 v60 |
| Geometry-safe compaction | 删除低风险 triangles，同时做 topology/sparse geometry audit | Phase-J mean triangle reduction `7.6479%` |
| Guarded ELA | 用训练 residual 在 held-out view 上做受控修复 | Phase-J 主收益来源 |
| Surface residual atlas | 把 residual repair 推向 face/UV-addressed persistent 表示 | v48/v52/v56/v59/v60 |
| Risk guard / fallback | 证据不足时自动 no-op 或回退 | 防止局部收益换全局崩坏 |

---

## 5. 主方法：Phase-J Guarded Adaptive ELA

### 5.1 训练视角 evidence mining

对训练视角运行 clean/compact MeshSplatting render，并缓存：

- rendered RGB；
- GT RGB；
- residual map：`GT - Render`；
- visibility；
- face id 和 barycentric coordinate；
- normal、depth、camera center；
- residual support、variance、sign consistency；
- policy-val split 上的 view-level 风险指标。

这一步的意义：

> 训练视角不仅用来训练 baseline，也用来估计哪些 surface 区域的错误是稳定、可迁移、可修复的。

### 5.2 Geometry-safe compaction

SPCarNet 的压缩不是追求极限删面，而是 quality-first rate-distortion：

- 删除多视角证据显示低风险的 triangles；
- 保护 sparse / occlusion-sensitive 区域；
- 保证 checkpoint 能被 renderer 正常加载；
- 保证 geometry audit 不出现明显崩坏；
- 与 residual repair 组成共同 endpoint。

当前 Phase-J 平均 triangle reduction 是 `7.6479%`，指删去的三角形占比。

### 5.3 Evidence Lumigraph Adapter

ELA 的直观公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 的像素；
- `residual_i(u_i)` 来自训练 view 的对应局部证据；
- `w_i(p)` 由可见性、局部结构、几何对应和 policy gate 决定；
- `alpha` 由 train/policy-val evidence 选择；
- 风险门不通过时，不应用修复。

通俗说：

> clean MeshSplatting 画出了整体，但局部细节还有系统误差。训练视角能告诉我们这些误差在哪里稳定出现。SPCarNet 只把稳定、可见性一致、风险可控的误差迁移到目标视角。

### 5.4 Guarded adaptive policy

Phase-J 不只是 ELA，而是 ELA 外面套了一个安全策略：

- stable scenes 使用 adaptive alpha；
- `treehill` 这类 adaptive alpha 不稳定场景使用 auto edge fallback；
- alpha、edge fallback 和 branch selection 都来自 train/policy-val evidence；
- policy-val 风险高则 fallback；
- held-out test GT 不参与方法选择。

论文式表述：

> We design a self-diagnosing MeshSplatting post-training policy that uses training-view evidence to decide whether to compact, repair, or fallback.

---

## 6. Phase-J 主结果

### 6.1 Per-scene 结果

| scene | selected branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
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

### 6.2 汇总结果

| Audit item | Value |
|---|---:|
| Scenes | `9` |
| Strict RGB scene wins vs selected clean | `9 / 9` |
| Mean delta vs selected clean | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| Per-view strict RGB wins | `244 / 246` |
| Mean triangle reduction | `7.6479%` |
| Sparse geometry strict wins | `6 / 9` |
| Geometry-safe scenes | `9 / 9` |

### 6.3 与 MeshSplatting paper table 的口径关系

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

推荐表述：

- 可以说 Phase-J 的本地结果数值高于 MeshSplatting paper table。
- 更严谨的主 claim 是超过本地同协议 selected clean MeshSplatting baseline。
- 不应只拿 paper table 当公平对照，因为 resolution、mask、数据预处理、split 和 evaluator 可能不同。

---

## 7. 表示级内化路线

Phase-J 的短板是最强收益来自 render-time ELA。为了把方法推向更像“表示本身”的论文贡献，我们推进了 surface-addressed residual representation。

### 7.1 v48 Auto-Support Surface Residual Atlas

核心思想：

```text
fit-view residual evidence
  -> select reliable face/UV bins
  -> expand support by residual hotness
  -> fit residual atlas
  -> train policy-val guard
  -> apply to target through target surface maps
```

结果：

| Comparison | Strict scene wins | Non-regressive/tie | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs same-evidence no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

意义：

> 证明 residual repair 可以从 image/render-time adapter 走向 face/UV-addressed surface representation，但效果量级仍远小于 Phase-J。

### 7.2 v52 Capacity-Aware Policy

v52 把 support cap-hit 观察变成固定 train-only policy：

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

意义：

> 这不是手工给每个场景调参数，而是把“support 不够时才升级容量”的规则固化进 train-only policy。

### 7.3 v56 Face-Alpha Reliability Guard

v55d 发现 per-face alpha 在 `counter` 上有明显正信号，但 `kitchen/bonsai` 有风险。v56 把它改成安全 guard：

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

已完成的可靠性补充：

- selected `counter` source rerun 复现：`26.756130 / 0.862126 / 0.251691`；
- v56 effective source status：`9 / 9` completed，`0` missing，`0` mismatch；
- `flowers/treehill/bicycle/garden/stump/room` fresh probes 均被 fixed guard 正确拒绝或 fallback；
- `min_target_changed_fraction=0.0` 边界消融完成，关键决策不变。

诚实定位：

> v56 是当前表示级安全策略的重要候选，但净收益只来自 `counter`，不应作为主 headline。

### 7.4 v59 Surface-Aware View-Conditioned Basis

v59 试图让 residual atlas 能随视角和表面法向变化：

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

新增接口：

- `--view_conditioned_basis_mode normal_camera_linear`
- `--view_conditioned_basis_guard_mode policy_val_nonregressive`
- `--view_conditioned_basis_min_bin_samples`
- `--view_conditioned_basis_ridge`

v59 probe：

| Scene | W&B run | PSNR | SSIM | LPIPS | Effective basis | Guard decision |
|---|---|---:|---:|---:|---|---|
| counter | `oiwl6r88` | `26.7536087036` | `0.8621008992` | `0.2518228889` | `normal_camera_linear` | `keep_view_basis` |
| kitchen | `08bwukw3` | `27.8192043304` | `0.8765304089` | `0.1990223229` | `normal_camera_linear` | `keep_view_basis` |

相对 reference：

| Scene | Reference | dPSNR | dSSIM | dLPIPS | Verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `+0.0001487732` | `-0.0000137687` | `-0.0000454485` | mixed |
| counter | v56/raw v55d | `-0.0025215149` | `-0.0000253320` | `+0.0001315177` | worse than guarded face-alpha reference |
| kitchen | v52 | `+0.0002689361` | `-0.0000049471` | `+0.0000029057` | not strict |
| kitchen | v57a | `-0.0037288666` | `+0.0000942350` | `+0.0001632422` | SSIM up, PSNR/LPIPS worse |

结论：

> v59 是真实 train/eval pipeline 改动，但不能 promote。它说明 linear normal/camera basis 不够，需要局部 uncertainty 和 OOD fallback。

### 7.5 v60 View-Basis OOD Guard

v60 针对 v59 的失败模式加入 OOD fallback：

> 对每个 face/UV bin，统计 fit-view feature mean/std。target sample 的 normal/camera feature 如果超出 fit-view 局部分布，就不用 view-conditioned basis，回退到 mean residual atlas。

新增接口：

- `--view_conditioned_basis_ood_mode {none,diag_z}`
- `--view_conditioned_basis_ood_max_z 2.5`
- `--view_conditioned_basis_ood_min_std 0.05`

关键实现：

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_l1risk_fairnoop_scene.py`

已修复的正确性问题：

- basis support 和 mean/std 统计现在基于 valid view-feature samples，而不是 residual `count_grid`；
- `changed_fraction` 现在统计实际非零 delta 像素，而不是所有几何有效像素；
- OOD 参数验证只在启用 OOD 模式时生效；
- synthetic OOD fallback test 和 missing-feature support test 已通过。

v60b clean probe 已完成：

| Scene | W&B run | PSNR | SSIM | LPIPS | accepted | changed fraction | effective basis | guard decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| counter | `d9tozw7s` | `26.7539958954` | `0.8621192575` | `0.2518530488` | `true` | `0.065630` | `normal_camera_linear` | `keep_view_basis` |
| kitchen | `924sxfsd` | `27.8191566467` | `0.8765332103` | `0.1990308464` | `true` | `0.039585` | `normal_camera_linear` | `keep_view_basis` |

相对关键 reference：

| Scene | Reference | dPSNR | dSSIM | dLPIPS | Verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `+0.0005359650` | `+0.0000045896` | `-0.0000152886` | strict small positive vs v52 |
| counter | v56/raw v55d | `-0.0021343231` | `-0.0000069737` | `+0.0001616776` | still worse than stronger counter reference |
| counter | v59 | `+0.0003871918` | `+0.0000183583` | `+0.0000301599` | PSNR/SSIM up, LPIPS worse |
| kitchen | v52 | `+0.0002212524` | `-0.0000021457` | `+0.0000114292` | not strict, SSIM/LPIPS regress |
| kitchen | v59 | `-0.0000476837` | `+0.0000028014` | `+0.0000085235` | mixed vs v59 |

Promotion criteria：

- strict improve `counter` against v56/raw v55d without SSIM/LPIPS regression；
- or strict improve `kitchen` against v52；
- or show train-policy-val fallback protects held-out metrics better than v59。

Promotion verdict：

> v60 不 promotion。它证明 OOD fallback 接口和正确性修复是必要的，并让 counter 相对 v52 变成三指标小幅严格正，但没有超过 v56/raw v55d；kitchen 仍不能严格超过 v52。

当前不能说：

> v60 已经全面解决 v59。

当前可以说：

> v60 是针对 v59 泛化风险的正确工程化和科学化下一步，已完成 counter/kitchen W&B clean probe，但当前证据只支持“诊断进展”，不支持“新 best endpoint”。

---

## 8. 定性展示建议

PPT 不建议只放 full-frame，因为很多收益是 residual-level，缩到一页后不够明显。建议三层展示：

1. 全图公平对比：证明同一 test view、同一 selected clean baseline、同一 evaluator。
2. 局部 crop 和 error map：展示残差减少的位置。
3. 表示级面板：说明 v48/v52/v56 在推进 surface representation，而不是只做 image-space 修图。

推荐图：

| 用途 | 路径 | 讲法 |
|---|---|---|
| 主推局部收益图 | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | 当前最适合放主 slide 的图，展示 Phase-J local held-out error reduction |
| full9 全图 gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` | 用于证明公平协议，不能单独承担视觉说服力 |
| 室外细节 | `assets/spcarnet_m360_outdoor_detail_showcase.png` | 展示树叶、边缘、纹理区域 |
| mixed where-it-helps | `assets/spcarnet_m360_where_it_helps_showcase.png` | 作为 backup qualitative |
| v52 cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` | 表示级 support/capacity 正证据 |
| v56 counter panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` | per-face alpha 与 guard 正证据 |

推荐 PPT 讲法：

> 全图用于证明公平，局部 crop/error map 用于证明改进发生在哪里。我们不夸大全图肉眼差异，而是用误差图说明 residual-level 修复。

---

## 9. 消融和版本脉络

| Version | Main idea | Result | PPT positioning |
|---|---|---|---|
| Phase-J | guarded adaptive ELA + compact mesh | `9 / 9` strict vs selected clean；mean `+1.331084` PSNR | 当前 headline |
| v42 | confidence/SSIM-gated atlas | 多场景 positive，但 coverage 小 | 表示级可行性初证 |
| v48 | train-only support expansion | full9 `7 / 9` strict vs no-op | surface atlas 正证据 |
| v51 | support-footprint ladder | cap-hit 场景优于 v48 | 大 support 对局部场景有效 |
| v52 | capacity-aware v48/v51 policy | vs v48 `3 / 9` strict，`9 / 9` non-reg/tie | 固定 train-only policy |
| v55d | per-face alpha calibration | `counter` strict win，但 `kitchen/bonsai` 不闭合 | 找到局部 alpha 信号 |
| v56 | reliability-guarded face alpha | vs v52 `1 / 9` strict，`9 / 9` non-reg/tie | 安全候选，非主 endpoint |
| v57a | face-alpha reliability shrink | `counter` 正向但弱于 raw v55d | 负结果/诊断 |
| v58 | camera-center linear view basis | support 提升仍无法保护 SSIM | 证明 camera-center-only 不够 |
| v59 | normal/camera surface-aware basis | counter/kitchen mixed | 真实接口，不能 promote |
| v60 | local OOD fallback for view basis | counter vs v52 小幅 strict positive；kitchen 仍 mixed；不 promotion | 最新诊断分支 |

---

## 10. 为什么这是研究工作

核心研究问题：

> Can a trained mesh representation certify where it can be safely compacted and where training residuals can be transferred without hurting held-out views?

对应研究贡献：

| 贡献 | 研究意义 | 为什么不是简单调参 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源是可复用 evidence interface，不是 test-set tuning |
| Geometry-safe compaction | 在 topology 和 sparse geometry audit 下删面 | 目标是 rate-distortion-safe mesh representation |
| Guarded residual repair | 多视角 evidence 约束 residual transfer | 不是任意滤镜，有 visibility/support/risk gate |
| Train-only policy/fallback | 证据不足自动 no-op | 安全部署逻辑，不追求每个场景强行改 |
| Surface residual atlas | 把 render-time residual repair 推向 face/UV-addressed 表示 | 是 representation-level internalization |
| View-conditioned diagnosis | 检验 residual 是否需要随视角变化 | v59/v60 暴露了线性 view basis 和 OOD 泛化边界 |

一句话论文故事：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation, and the current research frontier is to internalize its strong render-time repair into uncertainty-certified surface residual fields.

---

## 11. 公平性说明

必须讲清楚的点：

- Method branch、alpha、support、texture/fill、fallback 都来自 train/policy-val evidence。
- Held-out test GT 只用于最终报告和审计。
- Clean baseline 不是随便挑弱的，而是从 clean `26000/30000` envelope 中选更强 clean baseline。
- Train metrics 不用于挑最终 clean baseline 或最终 method result。
- v59/v60 probe 用 W&B online logging，便于追踪和复现。

容易被质疑的地方和回答：

| 质疑 | 回答 |
|---|---|
| 为什么不固定 clean30000？ | 更久训练不一定更好。我们选 stronger clean baseline 是为了更严格，而不是为了降低对照。 |
| Phase-J 是不是后处理？ | 它包含 render-time adapter，所以不包装成 fully baked representation；但它由 train-view surface evidence 和 risk gate 驱动，不是任意图像滤波。 |
| v56 是不是看了 held-out 后设计的？ | v56 是 report-only candidate，不能作为最终主 endpoint。它的价值是把局部 alpha 风险固化为 guard，并完成 source/fresh audit。 |
| 为什么 v59 不 promote？ | 因为 held-out 结果 mixed，不能严格超过当前 reference。 |
| 为什么 v60 还不能写成成功？ | 因为 clean probe 已完成但未满足 promotion criteria：counter 不及 v56，kitchen 不严格超过 v52。 |

---

## 12. 当前短板

| 短板 | 影响 | 下一步 |
|---|---|---|
| Phase-J 最强收益仍来自 render-time ELA | 论文中可能被质疑为后处理 | 把收益内化到 persistent surface residual representation |
| 表示级 v48/v52/v56 收益小 | 难作为主 headline | 提高 support coverage、local alpha 泛化和 view consistency |
| v56 净收益只来自 `counter` | 多场景说服力不足 | 新 split/新场景验证 fixed guard |
| v59 mixed | normal/camera linear basis 不足 | v60 OOD fallback，加更强 uncertainty/region guard |
| 定性全图差异不总明显 | PPT 说服力不够 | 主图使用 crop/error map，而不是只放 full-frame |
| Paper table 口径不完全一致 | 不能过度 claim | 主 claim 以本地同协议 selected clean baseline 为准 |

---

## 13. Mentor 可能会问的问题

### Q1：我们是否真正超过 MeshSplatting baseline？

回答：

> 是，在当前最强且最可汇报的 Phase-J endpoint 上，我们相对本地同协议 selected clean MeshSplatting baseline 达成 full9 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，并有 `7.6479%` 平均删面。需要注意的是，最新表示级 v59/v60 还不是主结果。

### Q2：这个方法是后处理还是表示学习？

回答：

> 当前最强 Phase-J 包含 render-time ELA，所以我会诚实说它还不是 fully baked representation。但它不是普通后处理，因为 residual 来源、alpha、fallback 和风险门全部由训练视角 surface evidence 决定。v48/v52/v56/v59/v60 正是在把它内化为 surface residual atlas 和 view-conditioned residual representation。

### Q3：为什么报告主结果不用 v59/v60？

回答：

> 因为 v59 在 counter/kitchen 上是 mixed，v60 虽完成 counter/kitchen clean probe，但没有满足 promotion criteria。为了科学可信，PPT 主结果应该用已闭合、全量 full9 的 Phase-J；v59/v60 作为最新研究路线和下一步内部推进。

### Q4：如果 full-frame 看不出区别，怎么证明有效？

回答：

> full-frame 缩放后很多 residual-level 收益不显著，所以我们同时展示全图、局部 crop 和 error map。定量上 full9 是 `9/9` strict，per-view 是 `244/246` strict；定性上用局部 error reduction 展示修复发生的位置。

### Q5：out-of-trajectory 或少数视角会不会崩？

回答：

> 这是 risk gate 的目的。证据不足或 tail risk 高时 fallback/no-op。Phase-J 有 `244/246` per-view strict RGB wins 和 `9/9` geometry-safe；v60 还进一步引入局部 view-feature OOD fallback。

### Q6：下一步最值得做什么？

回答：

> 不是继续扫 alpha 或 texture size，而是把 Phase-J 的强 residual repair 内化成 uncertainty-certified surface residual representation。v60 OOD guard 是这条路线的第一步，但当前还不够，后面需要 region-level no-regression、support coverage expansion 和更强局部 view model。

---

## 14. PPT 拆页建议

| Slide | Title | 内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Motivation | MeshSplatting 输出 mesh，但有 residual error、redundancy、tail-view risk |
| 3 | Core Idea | 用 train-view evidence 判断哪里能删、哪里能修、哪里回退 |
| 4 | Pipeline | MeshSplatting -> evidence cache -> compaction -> guarded repair -> eval |
| 5 | Difference from MeshSplatting | 对照 clean pipeline 和 SPCarNet pipeline |
| 6 | Fair Protocol | selected clean baseline，train-only method selection |
| 7 | Main Quant Result | Phase-J full9 `9/9` strict，mean `+1.3311` PSNR |
| 8 | Per-Scene Table | 9 个场景指标和 triangle reduction |
| 9 | Geometry/Compactness | mean reduction `7.6479%`，`9/9` geometry-safe |
| 10 | Main Qualitative | Phase-J where-it-helps crop/error map |
| 11 | Why It Works | residual transfer + visibility/support + guard |
| 12 | Representation Track | v48/v52/v56 的 surface residual atlas 路线 |
| 13 | Latest Diagnostic | v59 mixed，v60 OOD guard completed but not promoted |
| 14 | Limitations | render-time adapter、表示级收益小、视觉全图差异细 |
| 15 | Next Step | uncertainty-certified persistent surface residual field |
| 16 | Backup | paper table sanity check、W&B IDs、artifact paths |

---

## 15. 可直接放进 PPT 的短文案

中文摘要：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。后续 v48/v52/v56/v59/v60 则把这一路线从 render-time repair 推向 persistent surface residual representation，并暴露出 support coverage、view-conditioned generalization 和 OOD safety 是下一步核心挑战。

英文摘要：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

---

## 16. Artifact 和证据索引

| 内容 | 路径 |
|---|---|
| Phase-J full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` |
| Phase-J closure audit | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md` |
| Phase-J closure JSON | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json` |
| Phase-J per-view deltas | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv` |
| Phase-J qualitative showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| full9 qualitative gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| outdoor detail showcase | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| v48 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.md` |
| v52 policy log | `docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md` |
| v52 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md` |
| v56 guard log | `docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md` |
| v56 source/fresh log | `docs/car_model/6-24-v56-SourceRerun-And-FreshProbe-Log.md` |
| v57a shrink probe log | `docs/car_model/6-24-v57a-FaceAlphaReliabilityShrink-Probe-Log.md` |
| v58 view basis probe log | `docs/car_model/6-24-v58a-ViewConditionedSurfaceResidualBasis-Probe-Log.md` |
| v59 basis log | `docs/car_model/6-24-v59-SurfaceAwareViewConditionedBasis-Log.md` |
| v60 OOD guard log | `docs/car_model/6-24-v60-ViewBasisOODGuard-Log.md` |
| v60 counter output root | `/dev/shm/peilincai_spcarnet_v60b_basis_ood_counter_20260624` |
| v60 kitchen output root | `/dev/shm/peilincai_spcarnet_v60b_basis_ood_kitchen_20260624` |

W&B IDs：

```text
v59 counter normal-camera basis: oiwl6r88
v59 kitchen normal-camera basis: 08bwukw3
v60b counter OOD guard clean probe: d9tozw7s
v60b kitchen OOD guard clean probe: 924sxfsd
```

---

## 17. 汇报收束页

建议最后一页这样收束：

> Current status: SPCarNet has a strong, auditable Phase-J endpoint that beats a strong local MeshSplatting baseline on all 9 Mip-NeRF360 scenes while reducing triangles. The method is not merely checkpoint selection: it mines training-view surface evidence, performs geometry-safe compaction, transfers residuals under risk gates, and falls back when evidence is weak. The remaining paper-level challenge is to internalize the strongest render-time repair into a persistent, uncertainty-certified surface residual representation. The v48/v52/v56/v59/v60 line is exactly this transition path.

中文收束：

> 现在最稳的讲法是：我们已经证明 MeshSplatting checkpoint 之后还有一个很强的 evidence-certified compact-and-repair 空间，Phase-J 给出了 full9 强结果；接下来要把这个强修复能力从 render-time adapter 进一步内化成可持久化的 surface residual representation。v60 的 OOD guard 已经完成 counter/kitchen probe，但仍只是诊断进展，不能提前写成新 best。
