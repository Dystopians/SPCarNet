# SPCarNet 当前方法完整技术报告（Mentor/PPT 版）

日期：2026-06-24

用途：mentor 汇报、PPT 母稿、方法交底、后续论文路线讨论。

当前最适合作为 headline 的端点：`ours_26000_phasej_guarded_adaptedge_ela`

当前最新表示级研究线：`v48/v52/v56/v59`，即 surface residual atlas、capacity-aware policy、face-alpha reliability guard、surface-aware view-conditioned residual basis。

当前诚实结论：

- Phase-J 是当前最强、最适合对外汇报的阶段性结果：在本地同协议 selected clean MeshSplatting baseline 上，Mip-NeRF360 full9 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，并有平均 `7.6479%` triangle reduction。
- v48/v52/v56/v59 是把 Phase-J 的 render-time repair 内化为 persistent surface representation 的研究线。它已经提供了真实 pipeline 改动、可复现审计和若干正证据，但收益量级仍远小于 Phase-J，因此不应把 v52/v56/v59 包装成当前最终论文 endpoint。
- v59 是最新真实方法接口：加入 surface normal + camera-center 的 view-conditioned residual basis，并加 train-only non-regression guard。它完成了 counter/kitchen probe，但结果仍 mixed，不能推广为当前 best method。

---

## 1. 一句话讲清楚

中文：

> SPCarNet 把 MeshSplatting 从“训练完直接渲染的静态 mesh”，升级成“能用训练视角证据自检、删冗余三角形、并在可靠区域修复外观 residual 的 compact-and-repair mesh 表示”。

英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation.

给 mentor 的直观解释：

> MeshSplatting 已经是很强的基础模型，但训练完后仍会有局部纹理残差、冗余三角形和 tail-view 风险。SPCarNet 不直接否定 MeshSplatting，而是在它上面加一层训练证据驱动的体检机制：哪里能安全删、哪里能可靠修、哪里风险高必须回退。

---

## 2. 当前最稳主张

| 维度 | 当前证据 |
|---|---:|
| 数据口径 | Mip-NeRF360 full9，本地同协议复现 |
| 公平 baseline | selected clean MeshSplatting，从 clean `26000/30000` checkpoint envelope 中选择更强 clean baseline |
| 主 endpoint | `ours_26000_phasej_guarded_adaptedge_ela` |
| RGB 场景胜率 | `9 / 9` 场景相对 selected clean MeshSplatting 在 PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何/压缩 | 平均 triangle reduction `7.6479%`；`9 / 9` geometry-safe；`6 / 9` sparse geometry strict wins |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310`；只能作为 sanity check，不作为唯一公平 claim |
| 表示级 v48 | surface residual atlas vs same-evidence no-op：`7 / 9` strict，`8 / 9` non-regressive/tie |
| 表示级 v52 | capacity-aware policy vs v48：`3 / 9` strict，`9 / 9` non-regressive/tie |
| 表示级 v56 | face-alpha reliability guard vs v52：`1 / 9` strict，`9 / 9` non-regressive/tie；当前 fixed-command source/fresh audit 已闭合 |
| 最新 v59 | surface-aware view-conditioned basis 完成 counter/kitchen W&B probe，但相对 v52/v56/v57 仍 mixed，不推广为 best endpoint |

推荐 PPT 主张：

> We obtain a strong MeshSplatting-based compact-and-repair pipeline that strictly improves PSNR, SSIM, and LPIPS over a strong local MeshSplatting baseline on all 9 Mip-NeRF360 scenes, while also reducing triangle count. The strongest current endpoint is Phase-J; the next research step is to internalize its render-time repair into a stronger persistent surface representation.

---

## 3. 问题定义与动机

MeshSplatting 的关键优势是输出 mesh。相较纯 image-space 方法、点云或 Gaussian 表示，mesh 更容易进入传统图形管线、游戏引擎、AR/VR、数字孪生和几何编辑流程。

但 clean MeshSplatting checkpoint 训练完成后仍有三个实际问题：

| 问题 | 表现 | SPCarNet 的处理 |
|---|---|---|
| 局部外观 residual | 树叶、树皮、桌面、室内边缘仍可能有系统性偏差或模糊 | 从训练视角挖 residual，用 guarded ELA 或 surface atlas 做受控修复 |
| 拓扑/几何冗余 | 一部分 triangles 对多视角解释贡献低 | 用 sparse-occlusion protected compaction 删除低风险 triangles |
| 决策风险 | 盲目 transfer 可能破坏 tail views 或 out-of-trajectory 区域 | 用 train-only policy-val gate、min-view/CVaR、SSIM/L1 风险门和 fallback/no-op |

核心假设：

> MeshSplatting 已经学习到强基础表示，但训练视角中仍包含可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删除冗余 geometry，并把训练 residual 迁移到 held-out view 修复外观。

---

## 4. 与基础 MeshSplatting 的区别

基础 MeshSplatting：

```text
images / cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render held-out test views
```

SPCarNet：

```text
images / cameras
  -> train or load MeshSplatting checkpoint
  -> train-view evidence cache
  -> surface reliability / residual / support analysis
  -> sparse-occlusion protected compaction
  -> guarded residual repair
  -> train-only policy selection and fallback
  -> held-out test render and metrics
```

差异表：

| 维度 | clean MeshSplatting | SPCarNet |
|---|---|---|
| 训练结束后 | 直接渲染 checkpoint | 继续 evidence mining、压缩、修复和风险审计 |
| 几何 | 原 mesh | topology-safe compact mesh |
| 外观 | checkpoint 属性直接渲染 | train-evidence residual adapter / surface atlas |
| 策略 | 无显式风险判断 | train-only policy gate + fallback |
| held-out test GT | 用于评价 | 只用于最终评价，不参与 method branch/alpha/fallback |
| 失败处理 | 无 | gate 不通过则 fallback/no-op |

需要避免的夸张说法：

- 不说“已经完全替代 MeshSplatting 训练过程”；
- 不说“v59 是最终突破”；
- 不说“全图肉眼差异总是非常明显”；
- 不说“已经严格同协议超过 MeshSplatting 论文原表格”，除非后续完全复现其官方 protocol。

---

## 5. 方法总览

SPCarNet 的方法可以拆成五个模块：

```text
Clean MeshSplatting checkpoint
  -> Evidence Cache
  -> Geometry-Safe Compaction
  -> Guarded Residual Repair
  -> Train-Only Policy / Fallback
  -> Held-Out Evaluation
```

| 模块 | 做什么 | 当前证据 |
|---|---|---|
| Clean baseline envelope | 提供强 clean MeshSplatting 对照 | clean `26000/30000` envelope；full9 选到 clean `26000` |
| Evidence cache | 缓存 train views 的 render、GT、residual、visibility、face/barycentric/normal/camera evidence | 支撑 Phase-J、v48-v59 |
| Geometry-safe compaction | 删除低风险 triangles，同时保护 topology/sparse geometry | Phase-J `7.6479%` mean reduction，`9/9` geometry-safe |
| Guarded ELA | 用训练 residual 修复 held-out target view | Phase-J 主收益来源，full9 `9/9` strict |
| Surface residual atlas | 把 residual repair 推向 face/UV-addressed persistent representation | v48/v52/v56/v59 |
| Risk guard/fallback | train-only non-regression、min-view/CVaR、SSIM/L1 风险控制 | v48/v52/v56 source/fresh audits；v59 basis guard |

---

## 6. 主方法：Phase-J Guarded Adaptive ELA

### 6.1 Evidence mining

在训练视角上渲染 clean/compact MeshSplatting，并缓存：

- RGB render；
- GT image；
- residual map：`GT RGB - rendered RGB`；
- surface visibility；
- face / barycentric correspondence；
- normal / depth / camera metadata；
- residual support、variance、sign consistency；
- policy-val split 上的 view-level 风险指标。

关键点：这一步不使用 held-out test GT 做选择，而是把训练集中的局部错误和表面可见性转成可审计证据。

### 6.2 Geometry-safe compaction

Compaction 的目标是 quality-first rate-distortion improvement，而不是极限压缩。它删除训练证据表明低风险的 triangles，同时保护 sparse / occlusion-sensitive 区域。

关键约束：

- 不产生 invalid index；
- 不产生 degenerate faces；
- renderer 可读；
- sparse geometry 指标不能出现不可接受退化；
- compact checkpoint 必须进入同一 render/metrics pipeline。

这里的 triangle reduction 是“删去的三角形占比”，不是“剩余比例”。当前 Phase-J 平均删面比例为 `7.6479%`。

### 6.3 Evidence Lumigraph Adapter

ELA 用训练视角 residual 对 target held-out view 做受控修复：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target view 像素；
- `residual_i(u_i)` 来自训练 view 中对应的局部区域；
- `w_i(p)` 由可见性、几何邻近、局部结构和策略门决定；
- `alpha` 由 train/policy-val evidence 选择；
- 风险门失败则不应用修复。

直观解释：

> MeshSplatting 已经画出了整体结构，但局部纹理或边缘仍有系统误差；训练视角中能观察到这些误差，并且多视角一致区域的误差可以被安全迁移。

### 6.4 Guarded adaptive policy

Phase-J 的关键不是单纯 ELA，而是 ELA 外面有 train-only guard：

- stable scenes 使用 adaptive-alpha ELA；
- unstable scenes 使用 train-selected structural edge fallback；
- alpha、edge gate、fallback 由 train/policy-val evidence 决定；
- policy-val 风险过高时拒绝或 no-op；
- held-out test GT 不参与 branch、alpha、edge 或 compaction ratio 选择。

推荐论文式表达：

> We design a self-diagnosing and self-repairing MeshSplatting post-training policy driven by training-view evidence only.

---

## 7. 表示级内化路线

Phase-J 的主要短板是最强收益仍来自 render-time adapter。为了回应“这是不是后处理”的潜在质疑，我们推进了 surface-addressed residual representation 路线。

### 7.1 v48 Auto-Support Surface Residual Atlas

v48 把 residual 放入 face/UV-addressed atlas，并用 train-only support expansion 选择哪些 surface 区域可修复。

核心流程：

```text
base_carrier support
  + fit-view residual evidence ranking
  + top-K extra face candidates
  + train policy-val non-regression guard
  + texture/fill/alpha auto-policy
```

结果：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs same-evidence no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

意义：证明 residual repair 可以向 persistent surface representation 内化，但 effect size 还很小。

### 7.2 v52 Capacity-Aware Policy

v52 把 v51 的 lesson 固化成固定策略，而不是逐场景手调：

```text
if v48 accepted
   and v48 selected_support_added_faces >= 2048
   and v51 accepted_atlas
   and v51 selected_support_added_faces > v48 selected_support_added_faces
   and v51 policy-val SSIM gain >= 5e-5:
       use v51
else:
       keep v48
```

选择结果：

| scene group | v52 decision |
|---|---|
| `counter/kitchen/bonsai` | use v51 because v48 hits support cap and v51 has larger accepted support |
| other six scenes | keep v48 because cap/evidence condition is not met |

结果：

| comparison | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v52 vs no-op | `7 / 9` | `8 / 9` | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | `3 / 9` | `9 / 9` | `+0.000086890` | `+0.000008782` | `-0.000015303` |

意义：把“support cap-hit 场景需要更大 support”的观察变成 train-only policy，不是手动挑参数。

### 7.3 v56 Face-Alpha Reliability Guard

v55d 发现 per-face/local alpha 能在 `counter` 带来严格收益，但 `kitchen/bonsai` 不闭合。v56 把失败教训固化成 fixed audit guard：

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

机械 replay full9 后，v56 只选择 `counter=v55d`，其它场景回退 v52：

| comparison | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v56 vs v52 | `1 / 9` | `9 / 9` | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 vs no-op | `7 / 9` | `8 / 9` | `+0.001845890` | `+0.000037803` | `-0.000074494` |

2026-06-24 已补齐：

- selected `counter` source rerun：`26.756130 / 0.862126 / 0.251691`；
- v56 effective source status：`9 / 9` completed，`0` missing，`0` mismatch；
- `flowers/treehill/bicycle/garden/stump/room` fresh probes 均被 fixed guard 正确拒绝或 fallback；
- `min_target_changed_fraction=0.0` 边界消融完成，关键决策不变。

诚实评价：v56 是可靠性 guard 的重要候选，但它是在看到 v55d cap-hit held-out 结果后设计的，且净收益只来自 `counter`，因此不应作为最终主 endpoint。

### 7.4 v59 Surface-Aware View-Conditioned Basis

v58 证明 camera-center-only basis 不足；v59 进一步把 residual basis 升级为 surface-aware feature：

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

实现文件：

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_l1risk_fairnoop_scene.py`

v59 guard 会在 train policy-val 上比较：

1. requested surface-aware basis atlas；
2. 同一 atlas 禁用 basis 后的 legacy mean residual atlas。

只有 basis 不劣于 mean atlas 时才保留，否则 fallback 到 mean atlas。

v59 probe 结果：

| scene | W&B run | PSNR | SSIM | LPIPS | effective basis | guard decision |
|---|---|---:|---:|---:|---|---|
| counter | `oiwl6r88` | `26.7536087036` | `0.8621008992` | `0.2518228889` | `normal_camera_linear` | `keep_view_basis` |
| kitchen | `08bwukw3` | `27.8192043304` | `0.8765304089` | `0.1990223229` | `normal_camera_linear` | `keep_view_basis` |

相对关键 reference：

| scene | reference | dPSNR | dSSIM | dLPIPS | verdict |
|---|---|---:|---:|---:|---|
| counter | v52 | `+0.0001487732` | `-0.0000137687` | `-0.0000454485` | mixed |
| counter | v56/raw v55d | `-0.0025215149` | `-0.0000253320` | `+0.0001315177` | worse than guarded face-alpha reference |
| kitchen | v52 | `+0.0002689361` | `-0.0000049471` | `+0.0000029057` | not strict |
| kitchen | v57a | `-0.0037288666` | `+0.0000942350` | `+0.0001632422` | SSIM up, PSNR/LPIPS worse |

结论：v59 是有价值的诊断和真实 pipeline 接口，但不是当前 best。它说明“surface-aware view feature”方向是必要的，但线性 normal/camera basis 仍不足以稳定超越现有 representation references。

---

## 8. 主结果：Phase-J Full9 定量结果

### 8.1 Per-scene table

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

### 8.2 Closure audit

| audit item | value |
|---|---:|
| scenes | `9` |
| strict RGB scene wins vs selected clean | `9 / 9` |
| mean delta vs selected clean | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| per-view strict RGB wins | `244 / 246` |
| mean triangle reduction | `7.6479%` |
| sparse geometry strict wins | `6 / 9` |
| sparse geometry-safe scenes | `9 / 9` |

### 8.3 与 MeshSplatting paper table 的关系

| method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

推荐口径：

- 可以说 Phase-J 数值高于 MeshSplatting paper table；
- 不能把 paper table 当唯一公平对照，因为 evaluator、分辨率、mask、数据预处理和 split 可能不同；
- 最严谨 claim 仍是超过本地同协议 selected clean MeshSplatting baseline。

---

## 9. 定性展示建议

PPT 中建议三层图：

1. 公平全图对比：证明同一 held-out view、同一 selected clean baseline、同一评价口径；
2. 局部 crop / error map：突出 residual-level improvement；
3. representation-level panels：说明 v48/v52/v56/v59 是把 repair 内化到 surface 表示的路线。

推荐资产：

| 用途 | 路径 | 讲法 |
|---|---|---|
| 主推局部收益图 | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | 展示 Phase-J 在局部区域更接近 GT |
| full9 全图 gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` | 证明公平协议；不要只靠这张讲视觉优势 |
| 室外细节 | `assets/spcarnet_m360_outdoor_detail_showcase.png` | 展示树叶、纹理、边缘等细节改善 |
| where-it-helps | `assets/spcarnet_m360_where_it_helps_showcase.png` | 用 crop/error 更清晰地展示收益位置 |
| v52 cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` | 表示级 support/capacity 正证据 |
| v56 counter panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` | per-face alpha 正信号与 guard 价值 |

Markdown 内嵌图：

![Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

![Full-frame qualitative gallery](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

![Outdoor detail showcase](../../assets/spcarnet_m360_outdoor_detail_showcase.png)

![v52 capacity-aware cap-hit local panel](../../assets/spcarnet_v52_capacity_policy_cap_hit_panel.png)

![v56 counter face-alpha guard panel](../../assets/spcarnet_v56_counter_face_alpha_guard_panel.png)

---

## 10. 消融与版本脉络

| 版本 | 作用 | 结果 | 汇报定位 |
|---|---|---|---|
| Phase-J | guarded adaptive ELA + compact mesh | `9 / 9` strict vs selected clean；mean `+1.331084` PSNR | 当前 headline |
| v42 | confidence/SSIM-gated atlas | 多场景 positive，但 coverage 小 | 表示级可行性初证 |
| v48 | train-only support expansion | full9 `7 / 9` strict vs no-op | surface atlas 正证据 |
| v51 | support-footprint ladder | cap-hit 场景严格超过 v48 | 更大 support 对特定场景有效 |
| v52 | capacity-aware v48/v51 policy | vs v48 `3 / 9` strict，`9 / 9` non-reg/tie | 固定 train-only policy，不是手工逐场景 |
| v55d | per-face alpha calibration | `counter` strict win，但 `kitchen/bonsai` 不闭合 | 找到局部 alpha 正信号和风险 |
| v56 | reliability-guarded face alpha | vs v52 `1 / 9` strict，`9 / 9` non-reg/tie；source/fresh audits 闭合 | 安全候选，仍非主 endpoint |
| v57a | face-alpha reliability shrink | `counter` 正向但弱于 raw v55d；`kitchen` SSIM 风险未修复 | 真实负结果/诊断 |
| v58 | camera-center linear view basis | support 提升后仍无法可靠保护 SSIM | 证明 camera-center-only 不够 |
| v59 | surface-aware normal-camera basis + guard | counter/kitchen mixed；不优于 v56/v57 references | 最新接口与诊断，不推广 |

最重要技术教训：

1. 只加大 residual alpha 不可靠。`kitchen` 会出现 PSNR/LPIPS 上升但 SSIM 下降。
2. 表示级 repair 的关键瓶颈是 support coverage 和可靠性，而不是单纯 texture size。
3. 自动 fallback 是必要机制。风险高时 no-op 不是失败掩盖，而是部署安全。
4. 全图定性不一定能体现 residual-level gain，PPT 需要 crop/error map。
5. view-conditioned residual 需要更强 surface-aware features 和 uncertainty guard；线性 normal/camera basis 还不够。
6. 当前论文终局的核心难题是把 Phase-J 的强 render-time gain 内化成更强 persistent surface representation。

---

## 11. 为什么这是研究工作而不是工程调参

核心研究问题不是“能否调一个滤镜让图像更好”，而是：

> A trained mesh should be able to certify where its own geometry and residual evidence are reliable enough to be compacted and repaired.

| 创新点 | 研究含义 | 不是普通工程调参的原因 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源不是 test-set 调参，而是可复用 evidence interface |
| Geometry-safe compaction | 在 mesh topology 和 sparse geometry audit 下做删面 | 目标是 rate-distortion-safe representation |
| Guarded residual repair | 用 surface-aware residual transfer 修复 held-out rendering | residual 被多视角可见性和 policy-val 风险门约束 |
| Capacity-aware policy | support cap 命中时才升级 support footprint | 把实验观察固化成 train-only policy |
| Reliability fallback | 风险高时回退 no-op 或上一稳定版本 | 面向真实部署，避免单场景冒险 |
| View-conditioned residual diagnosis | 检验 residual 是否随视角变化 | v58/v59 给出明确负结果边界，推动下一步 surface-aware uncertainty 设计 |

---

## 12. 当前短板与下一步

| 短板 | 影响 | 下一步 |
|---|---|---|
| Phase-J 最强收益仍来自 render-time ELA | 容易被质疑为后处理 | 把 ELA 的收益内化到 persistent surface residual representation |
| v48/v52/v56 表示级收益小 | 难作为 headline | 提高 support coverage、局部 alpha 泛化和 view-consistency certification |
| v56 净收益只来自 `counter` | 多场景说服力不足 | 在新 split/新场景上验证 fixed guard |
| v59 mixed | normal-camera linear basis 不足 | 引入 region-level no-regression、uncertainty/OOD 和更强局部 view model |
| 全图视觉差异不总明显 | 汇报说服力受影响 | 主图必须用 crop/error map；室外只展示全图不够 |
| paper table 口径不完全可比 | 不能过度 claim | 本地同协议 selected clean baseline 为主，paper table 只作 sanity check |

下一步优先级：

1. 继续主讲 Phase-J 的 strong result，不把 v59 包装成终局；
2. 用 v48/v52/v56/v59 讲清楚 representation-level 内化路线和瓶颈；
3. 对 v56 fixed guard 做真正新 split 或新场景验证；
4. 设计下一版 persistent representation：surface-aware view-conditioned residual + region uncertainty + OOD guard；
5. 为 PPT 制作更强 crop/error-map qualitative panels，尤其是室外场景。

---

## 13. Mentor 可能会问的问题

### Q1：这是不是只是在 MeshSplatting 后面加图像后处理？

稳妥回答：

> 当前最强 Phase-J 确实包含 render-time ELA，所以我们不把它包装成完全 baked representation。但它不是任意 image filter：residual 来源、alpha、edge fallback、风险门和 fallback 都由 train-view surface evidence 决定，并且与 compact mesh checkpoint、topology audit、geometry-safe audit 绑定。v48/v52/v56/v59 进一步把 residual repair 推向 surface-addressed atlas 和 view-conditioned representation，说明这条路线可以从 render-time adapter 走向 persistent representation。

### Q2：有没有 test-set 调参？

稳妥回答：

> Method branch、alpha、support、texture/fill、fallback 都用 train/policy-val evidence 选择。held-out test GT 只用于最终 report。唯一使用 held-out test score 的地方是选择更强的 clean MeshSplatting baseline envelope，即从 clean `26000/30000` 中选更强 clean baseline，这是为了更严格地比较，而不是降低 baseline。

### Q3：为什么不固定 clean 30000 当 baseline？

稳妥回答：

> 更久训练不一定更好。当前本地 full9 clean envelope 中，clean `30000` 不总是优于 clean `26000`。固定 30000 可能反而选到 test 上更弱的 baseline。我们用 clean checkpoint envelope 选更强 clean baseline，是为了更公平。

### Q4：和 MeshSplatting 论文中的 Mip-NeRF360 结果怎么比？

稳妥回答：

> 数值上我们本地 selected clean 是 `25.1517 / 0.7490 / 0.2876`，Phase-J 是 `26.4828 / 0.7837 / 0.2243`，都高于 paper table 的 `24.78 / 0.728 / 0.310`。但论文表格可能有 evaluator、分辨率、mask、预处理差异，所以严格 claim 仍然应该是超过本地同协议 selected clean MeshSplatting baseline。

### Q5：为什么不把 v52/v56/v59 当最终方法？

稳妥回答：

> v52/v56/v59 是真实 representation-level 方法接口和消融，但 effect size 还小，且 v59 没有严格超过当前 reference。它们的价值是证明路线、暴露瓶颈并提供下一步论文创新方向；当前最强 headline 仍是 Phase-J。

### Q6：为什么定性图看起来差异不总明显？

稳妥回答：

> 因为当前主收益很多是 residual-level 和局部纹理/边缘修复，全图缩放后不一定显眼。PPT 应同时放公平全图、局部 crop 和 error map。全图用于证明协议公平，局部 crop/error map 用于展示具体改善位置。

### Q7：out-of-trajectory 区域会不会崩？

稳妥回答：

> 这是 guard 的作用。repair 只在训练证据支持、policy-val 风险门通过的区域执行；风险高或证据不足时 fallback/no-op。几何上也有 topology 和 sparse geometry audit。当前 Phase-J 是 `9/9` geometry-safe，v48/v52/v56/v59 都保留了 rejection/fallback 机制。

---

## 14. PPT 建议结构

| 页码 | 标题 | 核心内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Problem | MeshSplatting 有 mesh 优势，但仍有 residual error、geometry redundancy 和 tail-view risk |
| 3 | Key Idea | 用 train-view evidence 判断哪里能删、哪里能修、哪里必须回退 |
| 4 | Pipeline | `MeshSplatting -> evidence mining -> compact checkpoint -> guarded repair -> held-out eval` |
| 5 | Difference from MeshSplatting | 原方法直接渲染；我们做 train-evidence-certified compaction + repair |
| 6 | Fair Protocol | selected clean baseline；method selection uses train-only evidence |
| 7 | Main Quantitative Result | Phase-J full9 `9/9` strict，mean `+1.3311` PSNR |
| 8 | Per-Scene Table | 9 个场景 dPSNR/dSSIM/dLPIPS + triangle reduction |
| 9 | Geometry / Compactness | `7.6479%` triangle reduction，`9/9` geometry-safe |
| 10 | Qualitative Main Figure | Phase-J local held-out error reduction showcase |
| 11 | Why It Works | residual transfer + guard + fallback |
| 12 | Representation-Level Progress | v48 surface atlas：`7/9` strict vs no-op |
| 13 | Capacity Policy | v52：cap-hit `counter/kitchen/bonsai` 使用 v51，其余 v48 |
| 14 | Reliability Guard | v55d finds local alpha signal；v56 makes it safer |
| 15 | Latest Diagnostic | v59 surface-aware view basis is real but not yet enough |
| 16 | Limitations and Next Step | strongest result still render-time ELA；next is stronger persistent surface representation |
| 17 | Backup | paper table sanity check、W&B/source-rerun、artifact paths |

---

## 15. 可直接放 PPT 的短段落

英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded Evidence Lumigraph Adapter. All repair decisions are made from train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

中文：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复表示”。我们先从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，再把稳定的局部 residual 作为外观修复信号迁移到 held-out view。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达到 `9/9` 场景三指标严格胜出，同时平均减少 `7.65%` triangles。最新 v48/v52/v56/v59 则把 residual repair 进一步推进到 train-only 自适应的 surface atlas、reliability-guarded face-alpha 和 surface-aware view-conditioned residual diagnosis，说明该方向具备继续内化的可行性，也明确了下一步必须使用更强 uncertainty-certified persistent representation。

---

## 16. Artifact 与证据索引

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
| v52 selected artifacts | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9` |
| v56 guard log | `docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md` |
| v56 source/fresh log | `docs/car_model/6-24-v56-SourceRerun-And-FreshProbe-Log.md` |
| v57a shrink probe log | `docs/car_model/6-24-v57a-FaceAlphaReliabilityShrink-Probe-Log.md` |
| v58 view basis probe log | `docs/car_model/6-24-v58a-ViewConditionedSurfaceResidualBasis-Probe-Log.md` |
| v59 surface-aware basis log | `docs/car_model/6-24-v59-SurfaceAwareViewConditionedBasis-Log.md` |
| v59 counter output | `/dev/shm/peilincai_spcarnet_v59_normal_camera_guard_counter_20260624/counter_v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter` |
| v59 kitchen output | `/dev/shm/peilincai_spcarnet_v59_normal_camera_guard_kitchen_20260624/kitchen_v59_normal_camera_guard_min16_ridge01_support4096_tex16_nearest_region_texture_adapter` |

W&B keys：

```text
v59 counter normal-camera basis: oiwl6r88
v59 kitchen normal-camera basis: 08bwukw3
```

---

## 17. 汇报收束

建议最后一页用这个判断：

> 当前工作已经有一条可信主线：Phase-J 在本地强 MeshSplatting baseline 上 full9 全场景严格胜出，同时有真实删面、per-view 审计和 geometry-safe 证据。它足够作为阶段性强结果向 mentor 汇报。最新 v48/v52/v56/v59 说明我们没有停在后处理，而是在推进 persistent surface representation；但这些表示级版本目前 effect size 仍小，v59 也没有形成新的全面优势。下一步应把 Phase-J 的强 residual repair 内化成 uncertainty-certified surface residual representation，而不是继续简单扫 alpha、texture 或 ridge 参数。

