# SPCarNet 当前方法完整技术报告

日期：2026-06-24

用途：mentor 汇报、PPT 母稿、方法交底、后续论文路线讨论。

当前最适合作为主结果汇报的 endpoint：`ours_26000_phasej_guarded_adaptedge_ela`

当前最新研究线：`v48/v52/v56/v58` surface residual atlas / capacity-aware policy / face-alpha reliability guard / view-conditioned residual basis。

当前诚实状态：Phase-J 已经是强阶段性结果；v48/v52/v56/v58 是把 Phase-J 的 render-time repair 内化成 persistent surface representation 的推进路线，但还不能说已经完成论文终局。v58 是真实 pipeline 改动和有效负结果：它证明 camera-center linear basis 不足以解决 held-out SSIM 风险。

---

## 0. 汇报总览

SPCarNet 的定位不是从零替代 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 上增加一套训练证据驱动的压缩、修复和风险控制机制。

一句话中文：

> 原始 MeshSplatting 是“训练出 mesh 后直接渲染”；SPCarNet 是“训练出 mesh 后，再用训练视角证据给 mesh 做体检：哪里能安全删、哪里能可靠修、哪里风险高必须回退”。

一句话英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation.

当前最稳主结果：

| 维度 | 结论 |
|---|---:|
| 数据口径 | Mip-NeRF360 full9，本地同协议复现 |
| 公平 baseline | selected clean MeshSplatting，从 clean `26000/30000` checkpoint 中按 held-out test score 选更强者 |
| 主 endpoint | `ours_26000_phasej_guarded_adaptedge_ela` |
| RGB 场景胜率 | `9 / 9` 场景相对 selected clean MeshSplatting 在 PSNR、SSIM、LPIPS 三指标严格胜出 |
| 平均 RGB 提升 | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| per-view 稳定性 | `244 / 246` held-out views 三指标严格胜出 |
| 几何 / 压缩 | 平均 triangle reduction `7.6479%`，`9 / 9` geometry-safe，`6 / 9` sparse geometry 严格更好 |
| 与 MeshSplatting paper table | Phase-J mean `26.4828 / 0.7837 / 0.2243`，paper table mean `24.78 / 0.728 / 0.310`；只能作为 sanity check |
| 表示级 v48 | surface residual atlas vs same-evidence no-op：`7 / 9` strict，`8 / 9` non-regressive/tie |
| 表示级 v52 | capacity-aware policy vs v48：`3 / 9` strict，`9 / 9` non-regressive/tie |
| 最新候选 v56 | face-alpha reliability guard vs v52：`1 / 9` strict，`9 / 9` non-regressive/tie；selected effective rows 已 source-rerun reproduced，六个原 missing-audit 场景已补 fresh probes，完整 `min_target_changed_fraction=0.0` 审计已完成且决策不变，但仍不是 paper endpoint |
| 最新探针 v57a | face-alpha reliability shrink 是真实 train/eval pipeline 改动；`counter` 仍正于 v52 但弱于 raw v55d；`kitchen` 保留 PSNR/LPIPS 正向但未修复 SSIM 回退，因此当前不推广 |
| 最新探针 v58 | view-conditioned surface residual basis 已接入 train/eval pipeline；texture16 把 basis 覆盖率提高到 `11.5%/15.3%`，但 `counter` PSNR/SSIM 退化、`kitchen` SSIM 仍退化，因此当前不推广 |

PPT 主张建议：

> We obtain a strong MeshSplatting-based compact-and-repair pipeline that strictly improves PSNR, SSIM, and LPIPS over a strong local MeshSplatting baseline on all 9 Mip-NeRF360 scenes, while also reducing triangle count. The strongest current result is Phase-J; the next research bottleneck is to internalize render-time ELA gains into a stronger persistent surface representation.

---

## 1. 汇报口径分层

建议把汇报分成三层，避免把尚未闭合的候选方法讲成最终论文 endpoint。

| 层级 | 内容 | 汇报说法 |
|---|---|---|
| 主结果 | Phase-J guarded adaptive ELA full9 | 当前最稳 headline：full9 `9 / 9` strict RGB wins vs selected clean MeshSplatting，同时有删面与几何安全证据 |
| 表示级正证据 | v48/v52 surface residual atlas | 证明 residual repair 可以向 surface-addressed representation 内化，但收益量级仍小 |
| 最新诊断/候选 | v55d/v56 face-alpha calibration / guard | v55d 找到局部 alpha 正信号，v56 把它变成更安全的 fixed guard；`counter` 已 source rerun，`flowers/treehill/bicycle/garden/stump/room` fresh probes 被正确拒绝 |
| 最新表示探索 | v57a/v58 reliability shrink / view-conditioned basis | 两者都是真实 pipeline 改动，但都未修复 `kitchen` SSIM 风险；适合作为诚实诊断和下一步动机，不适合作为主结果 |

不建议主讲的说法：

- v52 或 v56 已经替代 Phase-J；
- 当前已经 100% 顶会终局闭合；
- 已经严格同协议超过 MeshSplatting 原论文表格；
- 全图肉眼定性差异总是非常明显；
- triangle reduction 已经是极限压缩。

更稳妥的定位：

> Phase-J 是当前可主讲的强结果；v48/v52/v56/v58 是为了解决“render-time adapter 如何内化成持久表示”的下一阶段研究线，其中 v58 给出了明确的负结果边界：仅用 camera-center linear residual basis 不能可靠提升 held-out SSIM。

---

## 2. 研究动机

MeshSplatting 的优势在于它输出 triangle mesh。相比纯 image-space 方法、点云或 Gaussian 表示，mesh 更容易进入传统图形管线、游戏引擎、AR/VR、数字孪生和几何编辑流程。

但是 clean MeshSplatting checkpoint 训练完成后仍有三个问题：

| 问题 | 典型表现 | SPCarNet 的处理 |
|---|---|---|
| 局部外观 residual | 树叶、树皮、桌面纹理、室内边缘仍有系统性偏差或模糊 | 从 train views 挖 residual，用 guarded ELA 或 surface atlas 修复 held-out view |
| 拓扑冗余 | 一部分 triangles 对多视角解释贡献低 | 用训练证据做 sparse-occlusion protected compaction |
| 决策风险 | 盲目 residual transfer 会伤害 tail views 或 out-of-trajectory 区域 | 用 train-only policy-val gate、min-view、CVaR、SSIM/L1 风险门和 fallback/no-op |

核心假设：

> MeshSplatting 已经学习到强基础表示，但训练视角中仍包含可反推出 surface reliability、occlusion risk 和 appearance residual 的证据。只要证据足够可靠，就可以安全删除冗余 geometry，并把训练 residual 迁移到 held-out view 修复外观。

---

## 3. 与原始 MeshSplatting 的区别

原始 MeshSplatting：

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

给 mentor 的通俗解释：

> MeshSplatting 是一个强 base model；SPCarNet 不否定它，而是把它变成一个可自检、可压缩、可修复的 mesh 表示。我们不是只看 test 图像调滤镜，而是从训练视角中估计哪些 surface 区域可信、哪些 residual 可迁移，并用 train-only 风险门决定是否执行。

---

## 4. 主方法：Phase-J Guarded Adaptive ELA

### 4.1 Train-view evidence mining

系统先在训练 view 上渲染 clean/compact MeshSplatting checkpoint，并缓存：

- RGB render；
- GT image；
- residual map：`GT RGB - rendered RGB`；
- surface visibility；
- face / barycentric correspondence；
- residual support、variance、sign consistency；
- policy-val split 上的 view-level 风险指标。

这一步不看 held-out test GT，而是把训练集中的局部错误和表面可见性转成结构化证据。

### 4.2 Geometry-safe compaction

Compaction 的目标不是极限压缩，而是 quality-first rate-distortion improvement。它删除训练证据表明低风险的 triangles，同时保护 sparse / occlusion-sensitive 区域。

关键约束：

- 不产生 invalid index；
- 不产生 degenerate faces；
- 保持 renderer 可读；
- sparse COLMAP / depth / normal 指标不能出现不可接受退化；
- 压缩后 checkpoint 必须进入同一 render/metrics pipeline。

当前 Phase-J 平均 triangle reduction 是 `7.6479%`。这里的 triangle reduction 是“删去的三角形占比”，不是“剩余比例”。PPT 中建议称为 quality-first geometry simplification，不要称为极限压缩。

### 4.3 Evidence Lumigraph Adapter

ELA 是 Phase-J 的主要外观收益来源。它用训练视角 residual 对 held-out target view 做受控修复：

```text
residual = GT RGB - rendered RGB

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target view 像素；
- `residual_i(u_i)` 来自训练 view 上对应局部区域；
- `w_i(p)` 由可见性、几何邻近、局部结构和策略门决定；
- `alpha` 由 train/policy-val evidence 选择；
- 当风险门失败时不应用该修复。

直观解释：

> MeshSplatting 已经画出了整体结构，但局部纹理或边缘还有系统性误差；训练视角中能看到这些误差，并且多视角一致区域的误差可以被安全迁移。

### 4.4 Guarded adaptive policy

Phase-J 的关键不是单纯 ELA，而是 ELA 外面有 train-only guard：

- stable scenes 使用 adaptive-alpha ELA；
- unstable scenes 使用 train-selected structural edge fallback；
- alpha、edge gate、fallback 由 train / policy-val evidence 决定；
- policy-val 风险过高时拒绝或 no-op；
- held-out test GT 不参与 branch、alpha、edge 或 compaction ratio 选择。

因此主 claim 是：

> We design a self-diagnosing and self-repairing MeshSplatting post-training policy driven by training-view evidence only.

### 4.5 当前可讲的模块拆解

PPT 中建议不要按版本号堆叠，而是按功能模块讲。版本号放 backup 或消融页。

| 模块 | 做什么 | 关键实现/证据 | 汇报价值 |
|---|---|---|---|
| MeshSplatting base | 提供 clean mesh checkpoint 和基础渲染能力 | clean `26000/30000` local baseline envelope | 说明我们是在强 baseline 上改进，而不是弱 baseline trick |
| Evidence cache | 在训练视角收集 render、GT、residual、visibility、face/barycentric 对应 | `ecsr_build_surface_evidence_cache.py`、teacher surface evidence roots | 把问题从 test-time 调图变成 train-evidence decision |
| Geometry-safe compaction | 删除低风险 triangles，同时保护 topology 和 sparse geometry | Phase-J closure audit、topology/sparse geometry audit | 同时提升/保持渲染质量并减少 mesh complexity |
| Guarded ELA | 把训练视角 residual 迁移到 held-out view 的可靠区域 | Phase-J guarded adaptive ELA，crop/error-map qualitative panels | 当前主收益来源，支撑 `9/9` strict full9 结果 |
| Surface residual atlas | 把 residual 从 render-time adapter 推向 face/UV-addressed persistent representation | v48/v52 atlas logs and selected artifacts | 回答“是不是纯后处理”的质疑 |
| Capacity-aware policy | 只在 support cap-hit 且 train policy-val 安全时扩大 support | v52 policy and source rerun | 说明不是逐场景手工调参，而是固定 train-only policy |
| Reliability guard | 对局部 alpha / risky residual transfer 做拒绝和 fallback | v56 source-rerun + fresh-probe rejection audits | 体现真实部署所需的 no-regression 思路 |
| View-conditioned basis | 测试 residual 是否应随 target view 条件变化 | v58 camera-center linear basis implementation and probes | 提供下一步创新边界：需要 surface-aware view features |

---

## 5. 表示级内化路线：v48 到 v58

Phase-J 的最大短板是最强收益仍来自 render-time adapter。为回应“这是不是后处理”的潜在质疑，我们推进了 surface-addressed residual atlas 线。

### 5.1 v48 Auto-Support Surface Residual Atlas

v48 的核心改动是 train-only support expansion：

```text
base_carrier support
  + fit-view residual evidence ranking
  + top-K extra face candidates
  + train policy-val non-regression guard
  + texture/fill/alpha auto-policy
```

固定策略：

1. 保留 v47 `base_carrier` support 作为安全候选；
2. 只扫描 fit views，并用同一 `policy_val_stride` 排除 policy-val views；
3. 对 non-carrier faces 聚合 residual evidence；
4. 按 `mean_l1 * log1p(samples)` 排序 extra faces；
5. 添加满足 sample 与 residual 阈值的 top-K extra faces；
6. 在 train policy-val 上联合评估 support mode、texture size、fill mode、alpha；
7. 只有 relative gain、SSIM gain、CVaR20、min-view gain 等风险门安全时才推广；
8. 不安全则回退到 base carrier 或 no-op。

v48 full9 相对 same-evidence no-op compact baseline：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs no-op full9 | `7 / 9` | `8 / 9` | `+0.001462` | `+0.00002774` | `-0.00003953` |

解释：

- `stump` 被 train policy-val gate 拒绝，作为有效 no-op；
- `treehill` PSNR/SSIM 微涨，但 LPIPS 轻微回退，因此不是 strict 三指标胜出；
- `counter/kitchen/bonsai` 触及 `+2048` extra-face cap，说明 support/capacity 是 representation-level 瓶颈；
- v48 是当前最完整的 single-policy 表示级正证据，但不是 Phase-J 替代品。

### 5.2 v51 Support-Footprint Ladder

v51 把 support expansion 从单一 `topK=2048` 改成 train-only ladder：

```text
base_carrier
  -> rank non-carrier faces by fit-view residual evidence
  -> evaluate topK support candidates, e.g. 2048 / 4096
  -> select support footprint only by policy-val risk gate
  -> apply accepted atlas to held-out test views
```

v51-fast full9 使用固定 `texture=32`、`fill=nearest_observed`、support candidates `2048,4096` 后：

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v51 vs same-evidence no-op | `6 / 9` | `6 / 9` | `+0.001380497` | `+0.000030067` | `-0.000051187` |
| v51 vs v48 | `3 / 9` | `3 / 9` | `-0.000081804` | `+0.000002331` | `-0.000011659` |
| v51 vs v50 | `5 / 9` | `8 / 9` | `+0.000116136` | `+0.000008331` | `-0.000017136` |

关键 lesson：

- `counter/kitchen/bonsai` 三个 cap-hit 场景相对 v48/v50 三指标严格提升；
- 非 cap-hit 场景不适合全局固定 `texture=32` 和 `nearest_observed`；
- 下一步不应该继续人工扫参，而应该用固定 train-only policy 只在 support-capacity 证据充分时升级到 v51。

### 5.3 v52 Capacity-Aware Effective Policy

v52 把 v51 的 lesson 固化成固定策略，而不是人工挑场景：

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

v52 结果：

| comparison | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v52 vs no-op | `7 / 9` | `8 / 9` | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | `3 / 9` | `9 / 9` | `+0.000086890` | `+0.000008782` | `-0.000015303` |
| v52 vs v50 | `6 / 9` | `6 / 9` | `+0.000284831` | `+0.000014782` | `-0.000020780` |

v52 的意义：

- 它把“cap-hit 场景需要更大 support”的发现变成固定 train-only policy；
- 它不是 per-scene 手动调参；
- 它有 W&B source-config rerun，完成 `9 / 9` 场景复现，`0` missing，`0` metric mismatch；
- 它仍然是小收益 representation-level milestone，不是当前 headline。

### 5.4 v53/v55d/v56 Alpha Calibration 线

v53 尝试用 policy-val least-squares 估计全局 residual alpha。结果证明 residual amplitude 确实是瓶颈，但单一全局 alpha 太粗，不能保证 SSIM 和 tail-view 安全。

v55d 进一步估计 per-face/local alpha，并加入 effective alpha cap。cap-hit 三场景结果：

| scene | selected alpha | face-alpha count | dPSNR vs v52 | dSSIM vs v52 | dLPIPS vs v52 | verdict |
|---|---:|---:|---:|---:|---:|---|
| counter | `0.5000` | `394` | `+0.002670` | `+0.00001156` | `-0.00017697` | strict win |
| kitchen | `1.0000` | `240` | `+0.004507` | `-0.00009782` | `-0.00023896` | PSNR/LPIPS up, SSIM down |
| bonsai | `0.1250` | `26` | `-0.001932` | `-0.00003034` | `+0.00006741` | worse than v52 |

v56 把 v55d 的失败教训固化成固定 audit guard：

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
| v56 vs v48 | `3 / 9` | `9 / 9` | `+0.000383589` | `+0.000010067` | `-0.000034966` |

注意：v56 的选择规则只使用 train/policy-val audit fields，但它是在看到 v55d cap-hit held-out 结果后设计的。因此它是下一版固定策略候选，不是已经可以作为论文 endpoint 的最终结论。

### 5.5 v56 Source-Rerun / Fresh-Probe 更新

2026-06-24 已经补上 v56 当前 fixed-command 下的 selected-row reproducibility 和 missing-audit fresh-probe 缺口：

| item | result | meaning |
|---|---|---|
| selected `counter` source rerun | `26.756130 / 0.862126 / 0.251691`，guard pass，selected alpha `0.5`，face-alpha count `394` | 复现了 v56 唯一启用 v55d 的有效行 |
| v56 effective source status | `9 / 9` completed，`0` missing，`0` mismatch，status `COMPLETE_REPRODUCED` | fresh `counter` + 已闭合 v52 fallback roots 可以复现 selected effective policy |
| `flowers` fresh v55d probe | candidate 被 fixed guard 拒绝：face-alpha count `5`，L1 positive fraction `0.5`，SSIM min-view gain 为负 | 把一个原先的 `missing_v55d_audit` fallback 变成显式 train/policy-val rejection |
| `treehill` fresh v55d probe | candidate 被 fixed guard 拒绝：face-alpha count `10`，L1 positive fraction `0.6667`，SSIM min-view gain `-0.000304` | 在 v48 曾有 LPIPS 风险的场景上，guard 没有冒险推广 |
| `bicycle` fresh v55d probe | candidate 被 fixed guard 拒绝：face-alpha count `15`，L1 positive fraction `0.5833`，SSIM min-view gain 为负 | 又一个稀疏局部 alpha + policy-val tail 风险负例 |
| `garden` fresh v55d probe | candidate 内部 accepted，但 fixed guard 拒绝：face-alpha count `92<128`，SSIM min-view gain `3.22e-06<5e-05` | 边界案例：有 test render 改动，但可靠性证据不足，不推广 |
| `stump` fresh v55d probe | candidate 被 fixed guard 拒绝：face-alpha count `5`，L1 positive fraction `0.7778`，SSIM min-view gain 为负 | fallback/no-op 负例：支持极稀疏且 tail 证据不足 |
| `room` fresh v55d probe | candidate 内部 accepted，但 fixed guard 拒绝：face-alpha count `142` 通过，SSIM min-view gain `2.74e-06<5e-05` | 边界案例：有足够 face-alpha support，但 worst-view SSIM margin 不够 |
| `garden/room` mtc0 boundary ablation | `--min_target_changed_fraction 0.0` 后 metrics 与 fixed guard decisions 不变 | 说明关键边界拒绝来自 support / worst-view SSIM 证据，而不是 `0.001` changed-fraction floor |
| `flowers/treehill/bicycle/stump` mtc0 audit | 完整 `0.0` 重跑完成，四者仍被 fixed guard 拒绝；full source status `COMPLETE_REPRODUCED` | 把完整阈值公平审计闭合到 6/6 原 missing-audit 场景 |

W&B 记录：

```text
v56 counter parent run: knt0skxs
v56 counter per-scene run: bbiugsyu
flowers fresh-probe parent run: qd1mrg2i
flowers fresh-probe per-scene run: tdypmap4
treehill fresh-probe parent run: 6gvhmy8o
treehill fresh-probe per-scene run: mhnjkb5z
bicycle fresh-probe parent run: vocupy0a
garden fresh-probe parent run: s3zieof1
stump fresh-probe parent run: nz6ft79h
room fresh-probe parent run: 5rttakj6
garden mtc0 parent run: 485snual
garden mtc0 per-scene run: 2027uxg1
room mtc0 parent run: uhqsu0wo
room mtc0 per-scene run: z8k86kzz
flowers mtc0 parent run: g6dib53z
treehill mtc0 parent run: aau1supu
bicycle mtc0 parent run: e9z6lhls
stump mtc0 parent run: ng46x0x6
```

这使 v56 更适合被讲成“可靠性 guard 的候选策略已经闭合当前 fixed-command 的 source-level/fresh-probe 审计”，但仍不应该讲成最终主方法。原因有三点：

- fixed guard 是在原始 v55d cap-hit held-out 结果之后设计的；
- 当前所有原 missing-audit fallback 已补 fresh v55d candidate audits，且完整 `0.0` 阈值消融已完成，但还没有在真正新场景/新协议上证明泛化；
- v56 相对 v52 的增益只来自 `counter`，effect size 仍小。

### 5.6 v57a/v58：从 alpha shrink 到 view-conditioned basis 的负结果边界

v57a 和 v58 的共同目的，是解决 v55d 暴露出来的核心问题：

```text
residual amplitude can improve PSNR/LPIPS,
but the same transfer can still damage SSIM or tail-view structure.
```

v57a 测试 per-face alpha 的 reliability shrink。它把局部 alpha 往 fallback/zero prior 收缩，希望减少高风险 face 的过拟合。结果是：

| scene | comparison | dPSNR | dSSIM | dLPIPS | conclusion |
|---|---|---:|---:|---:|---|
| counter | v57a vs v52 | `+0.001549` | `+0.00000966` | `-0.00011748` | 仍正向，但弱于 raw v55d |
| kitchen | v57a vs v52 | `+0.003998` | `-0.00009918` | `-0.00016034` | PSNR/LPIPS 正向，SSIM 风险未解决 |

v58 进一步把表示本身从“每个 face/UV bin 存一个 mean residual”升级成可选 view-conditioned basis：

```text
residual(face, uv, view)
  = beta0(face, uv)
  + beta1(face, uv) * cx
  + beta2(face, uv) * cy
  + beta3(face, uv) * cz
```

其中 `[cx, cy, cz]` 来自 evidence `.npz` 中归一化后的 `camera_center`。它已经完整接入：

- adapter CLI：`--view_conditioned_basis_mode camera_center_linear`；
- runner CLI：同名参数透传；
- train-fit ridge least squares；
- target/test-time predicted residual；
- audit / W&B logging；
- unsupported bins 自动回退 legacy mean atlas。

v58 probe 结论：

| version | scene | texture | min bin samples | supported-bin fraction | PSNR | SSIM | LPIPS | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v58a | counter | 32 | 64 | `0.000000` | `26.7534599304` | `0.8621146679` | `0.2518683374` | threshold too high; degenerates to v52 |
| v58a | kitchen | 32 | 64 | `0.000000` | `27.8189353943` | `0.8765353560` | `0.1990194172` | threshold too high; degenerates to v52 |
| v58b | counter | 32 | 4 | `0.020009` | `26.7529773712` | `0.8620327711` | `0.2517679930` | LPIPS improves vs v52, PSNR/SSIM regress |
| v58b | kitchen | 32 | 4 | `0.025695` | `27.8192138672` | `0.8765175939` | `0.1989833564` | PSNR/LPIPS improve vs v52, SSIM still regresses |
| v58c | counter | 16 | 4 | `0.115170` | `26.7527809143` | `0.8619601130` | `0.2517301738` | coverage higher, PSNR/SSIM worse |
| v58c | kitchen | 16 | 4 | `0.152937` | `27.8194313049` | `0.8764430285` | `0.1990009248` | PSNR/LPIPS slightly positive vs v52, SSIM still below v52 |

v58 的技术意义不是“又一次失败”，而是收窄了下一步创新空间：

- `texture16` 明显提高了 active basis support，但没有解决 held-out SSIM，说明瓶颈不只是 support density；
- 只使用 camera-center 线性项无法表达 surface normal/view-angle、遮挡边界和局部材质变化；
- train policy-val 对 SSIM/L1 的正信号仍可能外推失败，说明需要更强 uncertainty guard；
- 下一步应该用 surface-aware view features，例如 normal-view dot product、ray footprint、face visibility variance、per-region uncertainty，而不是继续围绕 alpha/shrink 参数扫描。

PPT 中建议把 v57/v58 放在“我们认真诊断过表示级瓶颈”的页，而不是放在主结果页。

---

## 6. 训练与验证流程

SPCarNet 当前不是从零训练一个新神经网络，而是在 MeshSplatting 训练后进行 evidence-certified post-training optimization。

实际流程：

1. 训练或读取 clean MeshSplatting checkpoint；
2. 渲染 train/policy-val/test views；
3. 在 train/policy-val 上构建 evidence cache；
4. 根据 evidence 执行 compaction 与 residual repair；
5. 用 train/policy-val gate 选择 branch、alpha、support、texture/fill 和 fallback；
6. 固定策略后在 held-out test 上报告 PSNR/SSIM/LPIPS 和几何指标；
7. 对关键版本用 W&B source-config rerun 复现。

Fairness 要点：

- clean MeshSplatting baseline 从 clean `26000/30000` 中用 held-out test score 选强者，是为了给 baseline 更强 envelope；
- method 自身的 branch、alpha、fallback、fill mode、texture capacity 和 compaction 决策都来自 train/policy-val evidence；
- train metrics 不用于选择最终 test baseline；
- held-out test GT 只用于报告，不用于 method selection；
- v52 source-config rerun 已补齐 W&B 可复现性证据；
- v56 selected effective rows 现在也有 source-rerun status：`9 / 9` completed，`0` missing，`0` mismatch；
- v56 仍不能升级成 paper endpoint，因为 guard 是看过 v55d held-out 后形成的；当前 fixed-command fresh candidate audit 已覆盖所有原缺失场景，关键 `garden/room` 边界消融也已完成，但还没有在真正新场景/新协议上证明泛化。

baseline 选择分数：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

在当前 full9 local baseline envelope 中，9 个场景最终都选择 clean `26000`，不是盲目选更久训练的 `30000`。

---

## 7. 主结果：Phase-J Full9

### 7.1 Per-scene table

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

### 7.2 闭环审计

| audit item | value |
|---|---:|
| scenes | `9` |
| strict RGB scene wins vs selected clean | `9 / 9` |
| mean delta vs selected clean | `+1.331084` PSNR，`+0.034702` SSIM，`-0.063359` LPIPS |
| per-view strict RGB wins | `244 / 246` |
| mean triangle reduction | `7.6479%` |
| sparse geometry strict wins | `6 / 9` |
| sparse geometry-safe scenes | `9 / 9` |

### 7.3 与 MeshSplatting paper table 的关系

| method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

Phase-J 相对 paper table 数值高，但 PPT 中必须谨慎：

- paper table 可能存在实现、数据预处理、分辨率、mask、evaluation script 等差异；
- 我们最严谨的 claim 是超过本地同协议 selected clean MeshSplatting baseline；
- paper table 可以作为 sanity check 或背景对照，不能作为唯一公平比较。

---

## 8. 定性展示建议

PPT 推荐三层定性展示：

1. 公平全图对比：证明同一 held-out view、同一 selected clean MeshSplatting baseline、同一评价口径；
2. 局部 crop / error map：更清楚展示 residual-level improvement；
3. representation-level atlas panel：说明 v48/v52/v56 已经能做 surface-addressed residual，但效果仍细微。

### 8.1 主推图：Phase-J local held-out error reduction

路径：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

![Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

这张图适合主结果页，因为它不是只放全图，而是突出 SPCarNet 相对 clean MeshSplatting 更接近 GT 的局部区域。

### 8.2 全图公平对比 gallery

路径：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/qualitative_gallery.html
```

![Full-frame qualitative gallery](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

这张图用于证明比较协议公平，但全图肉眼差异可能不够明显。不要只依赖这张图讲视觉优势。

### 8.3 室外局部细节展示

路径：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
```

![Outdoor detail showcase](../../assets/spcarnet_m360_outdoor_detail_showcase.png)

![Outdoor local improvement showcase](../../assets/spcarnet_m360_where_it_helps_showcase.png)

建议把室外场景用局部 crop 或 error reduction 展示，而不是只放整图。原因是室外全图内容复杂，SPCarNet 的收益常集中在树叶、边缘、纹理和局部 residual 区域。

### 8.4 v52/v56 representation-level panels

路径：

```text
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
```

![v52 capacity-aware cap-hit local panel](../../assets/spcarnet_v52_capacity_policy_cap_hit_panel.png)

![v56 counter face-alpha guard panel](../../assets/spcarnet_v56_counter_face_alpha_guard_panel.png)

v52 图只展示 `counter/kitchen/bonsai`，因为这些是 v52 相对 v48 真正发生策略升级并严格提升的场景。v56 图展示 `counter` 的 per-face alpha 正信号。二者适合放在 ablation 或 next-step 页，不建议放在主结果页替代 Phase-J 图。

---

## 9. 消融与经验教训

| 版本 | 作用 | 结果 | 结论 |
|---|---|---|---|
| Phase-J | guarded adaptive ELA + compact mesh | `9 / 9` strict vs selected clean | 当前 headline |
| v42 | confidence/SSIM-gated atlas | 四场景 positive，但 coverage 小 | 表示级可行性初证 |
| v48 | train-only support expansion | full9 `7 / 9` strict vs no-op | support bottleneck 被缓解 |
| v51 | support-footprint ladder | cap-hit 场景严格超过 v48 | 更大 support 对特定场景有用 |
| v52 | capacity-aware v48/v51 policy | vs v48 `3 / 9` strict, `9 / 9` non-reg/tie | 固定 train-only policy 比全局替换更合理 |
| v53 | global alpha calibration | 不推广 | 全局 residual amplitude 太粗 |
| v55d | per-face alpha calibration | counter 严格优于 v52，但 kitchen/bonsai 不闭合 | 不推广 raw v55d；需要 reliability guard |
| v56 | reliability-guarded face alpha | vs v52 `1 / 9` strict, `9 / 9` non-reg/tie；selected rows source-reproduced，六个原 missing-audit 场景已补 fresh probes | 安全候选；当前 fixed-command 缺失审计已闭合，但仍需新协议泛化 |
| v57a | face-alpha reliability shrink | counter 相对 v52 仍正向但弱于 raw v55d；kitchen 相对 v52 为 `+0.003998` PSNR、`-0.000099` SSIM、`-0.000160` LPIPS | 真实 pipeline 接口，但简单 shrink 未修复 SSIM 风险；不推广，应转向 view-conditioned residual basis |
| v58 | camera-center linear view-conditioned basis | texture16 把 active basis support 提到 `11.5%/15.3%`，但 counter PSNR/SSIM 退化、kitchen SSIM 仍退化 | 真实 representation 改动，但 camera-center linear feature 不足；不推广 |

最重要技术教训：

1. 只加大 residual alpha 不可靠。`kitchen` 会 PSNR/LPIPS 上升但 SSIM 下降，说明 SSIM 与 local structure 需要更细粒度控制。
2. 表示级 repair 的主要瓶颈是 support coverage。v48/v51/v52 的进展说明“让更多 target rays 命中有 residual support 的 faces”比单纯调 texture 更关键。
3. 自动 fallback 是必要机制。`stump` 被 v48 拒绝成 no-op 是正确行为，而不是失败掩盖。
4. 全图定性不一定能体现 residual-level gain。PPT 需要配 crop 和 error map。
5. view-conditioned residual 需要 surface-aware features。v58 证明只用 camera-center 线性项，即便覆盖率提高，也不能可靠保护 SSIM。
6. 当前最强 Phase-J 仍有 render-time adapter 风险。论文终局需要更强 persistent surface representation。

---

## 10. 为什么这是研究工作而不是工程调参

这条线的研究价值在于把 MeshSplatting checkpoint 后处理变成了可审计的 train-evidence decision problem。

核心创新点可以这样组织：

| 创新点 | 研究含义 | 不是普通工程调参的原因 |
|---|---|---|
| Train-view evidence mining | 从训练视角恢复 surface reliability、residual support 和风险结构 | 决策来源不是 test-set 调参，而是可复用的 evidence interface |
| Geometry-safe compaction | 在 mesh topology 与 sparse geometry audit 下做删面 | 目标是 rate-distortion-safe representation，不是任意简化 |
| Guarded residual repair | 用 surface-aware residual transfer 修复 held-out rendering | residual 被约束在多视角可见性和 policy-val 风险门内 |
| Capacity-aware policy | support cap 命中时才升级更大 support footprint | 把实验观察固化成 train-only policy，而非人工挑场景 |
| Reliability fallback | 风险高时回退 no-op 或上一稳定版本 | 非常适合真实部署，因为它避免单场景冒险 |
| View-conditioned residual diagnosis | 检验 residual 是否需要随视角条件变化 | v58 证明 camera-center linear basis 不足，推动下一步 surface-aware view feature 设计 |

可以放进 PPT 的一句话：

> The scientific question is not whether a filter can improve an image, but whether a trained mesh can certify where its own geometry and residual evidence are reliable enough to be compacted and repaired.

---

## 11. 当前短板与后续路线

当前短板：

| 短板 | 影响 | 下一步 |
|---|---|---|
| Phase-J 最强收益仍来自 render-time ELA | 论文中容易被质疑为后处理 | 提升 persistent surface residual representation |
| v48/v52 表示级收益小 | 难作为 headline | 增大 support coverage、局部 alpha、view-consistency certification |
| 全图定性差异不总明显 | PPT 说服力受影响 | 用 crop/error maps 和 outdoor detail panel |
| raw v55d 未多场景闭合 | 不能主张 per-face alpha 已解决问题 | 当前 `flowers/treehill/bicycle/garden/stump/room` fresh probes 已被正确拒绝；完整 `min_target_changed_fraction=0.0` 审计已闭合；下一步需要真正新协议验证 |
| v57a reliability shrink 不够强 | counter 正信号被削弱；kitchen 虽保留 PSNR/LPIPS 增益，但 SSIM 仍低于 v52 | 放弃继续围绕简单 shrink 扫参数，转向 view-conditioned residual basis |
| v58 camera-center basis 不够强 | support 覆盖率提高后仍未闭合 SSIM，counter 还出现 PSNR/SSIM 退化 | 下一步要加入 normal/view-angle、occlusion uncertainty、region-level no-regression guard |
| 与 paper table 口径可能不同 | 不能直接 claim 严格超 paper protocol | 保留本地同协议 baseline 为主 claim，paper table 做 sanity check |

下一步优先级：

1. 继续用 fresh split 或更多场景重新验证 v56 reliability guard，避免把看过 cap-hit held-out 后形成的规则直接当论文结论；
2. 对 v56 做更强外推验证：当前 `garden/room` 的 `--min_target_changed_fraction 0.001` vs `0.0` 边界消融已完成，下一步应在新 split / 新场景上验证固定 guard；
3. 在 v58 之后不要继续只扫 camera-center/texture 参数，应升级到 surface-aware view-conditioned representation；
4. 继续扩大 representation-level support coverage，让收益不只来自 `counter` 单场景；
5. 继续提升 representation-level effect size，而不是继续做纯参数扫描；
6. 对室外场景制作更强的 local crop / error-map qualitative panel，用视觉证据解释 Phase-J 的收益位置。

---

## 12. Mentor 可能会问的问题

### Q1：这是不是只是在 MeshSplatting 后面加图像后处理？

最稳回答：

> 当前最强 Phase-J 确实包含 render-time ELA，所以我们不把它包装成完全 baked representation。但它不是任意 image filter：residual 来源、alpha、edge fallback、风险门和 fallback 都由 train-view surface evidence 决定，并且与 compact mesh checkpoint、topology audit、geometry-safe audit 绑定。v48/v52 进一步把 residual repair 推向 surface-addressed atlas，说明这条路线可以从 render-time adapter 走向 persistent representation。

### Q2：有没有 test-set 调参？

最稳回答：

> Method branch、alpha、support、texture/fill、fallback 都用 train/policy-val evidence 选择。held-out test GT 只用于最终 report。唯一用 held-out test score 的地方是选择更强的 clean MeshSplatting baseline envelope，也就是从 clean `26000/30000` 中选更强 baseline，以避免低估 baseline。

### Q3：为什么不固定 clean 30000 当 baseline？

最稳回答：

> 更久训练不一定更好。当前本地 full9 clean envelope 中，clean `30000` 不总是优于 clean `26000`。固定 30000 可能反而选到 test 上更弱的 baseline。我们用 clean checkpoint envelope 选更强 clean baseline，是为了更严格地比较。

### Q4：和 MeshSplatting 论文中的 24.78 PSNR 怎么比？

最稳回答：

> 数值上我们本地 selected clean 是 `25.1517 / 0.7490 / 0.2876`，Phase-J 是 `26.4828 / 0.7837 / 0.2243`，都高于 paper table 的 `24.78 / 0.728 / 0.310`。但论文表格可能有 evaluator、分辨率、mask、预处理差异，所以严格 claim 仍然应该是超过本地同协议 selected clean MeshSplatting baseline。

### Q5：v52 既然是 representation-level，为什么不作为主结果？

最稳回答：

> v52 是真实的 representation-level 正证据，但 effect size 还小，主要作用是证明路线和 policy 是可行的。主结果仍是 Phase-J，因为它在 full9 selected clean baseline 上有 `9/9` strict、`244/246` per-view strict 和几何压缩证据。论文终局需要把 Phase-J 的收益进一步内化到更强的 persistent surface representation。

### Q6：为什么定性图看起来差异不总明显？

最稳回答：

> 因为当前主收益很多是 residual-level 和局部纹理/边缘修复，全图缩放后不一定显眼。PPT 应该同时放公平全图、局部 crop 和 error map。全图用于证明协议公平，局部 crop/error map 用于展示具体改善位置。

### Q7：如果 out-of-trajectory 区域没有训练证据，会不会崩？

最稳回答：

> 这是我们设计 guard 的原因。repair 只在训练证据支持、policy-val 风险门通过的区域执行；风险高或证据不足时 fallback/no-op。几何上也有 topology 和 sparse geometry audit。当前 Phase-J 是 `9/9` geometry-safe，v48/v52 也保持 rejected fallback 机制。

### Q8：v56 已经 `COMPLETE_REPRODUCED`，为什么还不是最终方法？

最稳回答：

> `COMPLETE_REPRODUCED` 只说明当前 v56 selected effective rows 可以从 source configs 复现，不说明 v56 的 policy 已经完成 paper-level validation。v56 的 guard 是在观察过 v55d cap-hit held-out 结果后形成的，因此仍需要在真正新场景/新协议上验证。现在 `counter` 已复现为正例，`flowers/treehill/bicycle/garden/stump/room` fresh probes 已被正确拒绝，这是好信号，但 v56 的净收益仍只来自 `counter`，还不足以把它作为最终主方法。

---

## 13. PPT 建议结构

| 页码 | 标题 | 核心内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Problem | MeshSplatting mesh 有价值，但仍有局部 residual error 和 topology redundancy |
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
| 14 | Alpha Calibration Lessons | v53 negative，v55d finds counter signal，v56 source-rerun positive + six fresh rejections |
| 15 | View-Conditioned Basis Probe | v58 is a real representation change, but camera-center linear basis does not close SSIM risk |
| 16 | Limitations and Next Step | strongest result still render-time ELA；next is stronger surface-aware persistent representation |
| 17 | Backup | paper table sanity check、W&B/source-rerun、artifact paths |

---

## 14. 可直接放 PPT 的短段落

英文：

> SPCarNet upgrades MeshSplatting from a static trained mesh into a train-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, we mine training-view surface evidence, remove low-risk redundant triangles, and transfer reliable residual appearance cues through a guarded Evidence Lumigraph Adapter. All repair decisions are made from train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction.

中文：

> SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态网格，升级成“训练证据驱动的可压缩、可修复表示”。我们先从训练视角挖掘 surface evidence，判断哪些三角形可以安全删除，再把稳定的局部 residual 作为外观修复信号迁移到 held-out view。当前 Phase-J endpoint 在 Mip-NeRF360 full9 上相对本地强 clean MeshSplatting baseline 达到 `9/9` 场景三指标严格胜出，同时平均减少 `7.65%` triangles。最新 v48/v52/v56/v58 则把 residual repair 进一步推进到 train-only 自适应的 surface atlas、reliability-guarded face-alpha 和 view-conditioned residual diagnosis，说明该方向具备继续内化的可行性，也明确了下一步必须使用更强 surface-aware view features。

最终收束句：

> 当前工作已经有一条可信主线：Phase-J 在本地强 MeshSplatting baseline 上 full9 全场景严格胜出，同时有真实删面、per-view 审计和 geometry-safe 证据。它足够作为阶段性强结果汇报。最需要继续攻克的是 representation-level 内化：v48/v52 已经证明 surface-addressed residual atlas 可以用 train-only policy 做 support/容量自适应；v56 进一步开始验证 reliability-guarded face-alpha，当前已有 `counter` 正例和 `flowers/treehill/bicycle/garden/stump/room` fresh rejection/boundary audit，但 effect size 仍小。v57a 的 counter/kitchen probe 说明简单 reliability shrink 不是答案；v58 进一步证明 camera-center linear view-conditioned basis 也不足以解决 SSIM 风险。下一步不是继续扫参数，而是把局部 alpha、support coverage、surface normal/view-angle features 和 uncertainty certification 结合成更强的 persistent surface residual representation。

---

## 15. 证据索引

| 内容 | 路径 |
|---|---|
| Phase-J full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` |
| Phase-J closure audit | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md` |
| Phase-J closure JSON | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json` |
| Phase-J per-view deltas | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv` |
| fair baseline audit | `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/fair_baseline_audit.json` |
| Phase-J qualitative showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| full9 qualitative gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| outdoor detail showcase | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| v48 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.md` |
| v52 policy log | `docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md` |
| v52 full9 summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md` |
| v52 selected artifacts | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9` |
| v52 gallery | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/qualitative_gallery.html` |
| v52 cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v52 source-rerun status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_status.md` |
| v53 diagnostic log | `docs/car_model/6-23-v53-PolicyValAlphaCalibration-CapHit-Log.md` |
| v55d diagnostic log | `docs/car_model/6-23-v55d-FaceAlphaCalibration-CapHit-Log.md` |
| v55d cap-hit results | `/dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623` |
| v56 guard log | `docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md` |
| v56 guard summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_full9_summary.md` |
| v56 selected artifact tree | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_selected_full9` |
| v56 artifact pipeline report | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_selected_full9/v56_face_alpha_guard_pipeline_report.md` |
| v56 selected gallery | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_selected_full9/qualitative_gallery.html` |
| v56 counter qualitative panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |
| v56 source-rerun/fresh-probe log | `docs/car_model/6-24-v56-SourceRerun-And-FreshProbe-Log.md` |
| v56 source-rerun status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_source_rerun_status.md` |
| v56 freshcheck summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_freshcheck_summary.md` |
| v56 freshcheck source status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_freshcheck_source_status.md` |
| v56 mtc0 garden/room summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_garden_room_freshcheck_summary.md` |
| v56 mtc0 garden/room source status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_garden_room_source_status.md` |
| v56 mtc0 full summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_freshcheck_summary.md` |
| v56 mtc0 full source status | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_source_status.md` |
| v57a shrink probe log | `docs/car_model/6-24-v57a-FaceAlphaReliabilityShrink-Probe-Log.md` |
| v57a counter output | `/dev/shm/peilincai_spcarnet_v57a_face_alpha_shrink_counter_20260624/counter_v57a_face_alpha_shrink_count512_den01_zero_support4096_tex32_nearest_region_texture_adapter` |
| v57a kitchen output | `/dev/shm/peilincai_spcarnet_v57a_face_alpha_shrink_kitchen_20260624/kitchen_v57a_face_alpha_shrink_count512_den01_zero_support4096_tex32_nearest_region_texture_adapter` |
| v58 view-conditioned basis probe log | `docs/car_model/6-24-v58a-ViewConditionedSurfaceResidualBasis-Probe-Log.md` |
| v58a counter output | `/dev/shm/peilincai_spcarnet_v58a_viewbasis_counter_20260624/counter_v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter` |
| v58a kitchen output | `/dev/shm/peilincai_spcarnet_v58a_viewbasis_kitchen_20260624/kitchen_v58a_viewbasis_camcenter_min64_ridge001_support4096_tex32_nearest_region_texture_adapter` |
| v58b counter output | `/dev/shm/peilincai_spcarnet_v58b_viewbasis_counter_20260624/counter_v58b_viewbasis_camcenter_min4_ridge001_support4096_tex32_nearest_region_texture_adapter` |
| v58b kitchen output | `/dev/shm/peilincai_spcarnet_v58b_viewbasis_kitchen_20260624/kitchen_v58b_viewbasis_camcenter_min4_ridge001_support4096_tex32_nearest_region_texture_adapter` |
| v58c counter output | `/dev/shm/peilincai_spcarnet_v58c_viewbasis_tex16_counter_20260624/counter_v58c_viewbasis_camcenter_min4_ridge001_support4096_tex16_nearest_region_texture_adapter` |
| v58c kitchen output | `/dev/shm/peilincai_spcarnet_v58c_viewbasis_tex16_kitchen_20260624/kitchen_v58c_viewbasis_camcenter_min4_ridge001_support4096_tex16_nearest_region_texture_adapter` |
| 本报告 | `docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report.zh.md` |

v52 source-rerun W&B：

```text
main: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/2j5osvgg
counter: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/9uwsu9m1
kitchen: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/b9w1zonu
bonsai: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/uh4a9hvu
room: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/o785gymj
stump no-op fix: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/n4ucitf6
```

v55d W&B：

```text
counter: wwp6tn65
kitchen: 8znw2xhy
bonsai: 6k94f7mm
```

v56 W&B：

```text
counter source-rerun parent: knt0skxs
counter source-rerun per-scene: bbiugsyu
flowers fresh-probe parent: qd1mrg2i
flowers fresh-probe per-scene: tdypmap4
treehill fresh-probe parent: 6gvhmy8o
treehill fresh-probe per-scene: mhnjkb5z
bicycle fresh-probe parent: vocupy0a
garden fresh-probe parent: s3zieof1
stump fresh-probe parent: nz6ft79h
room fresh-probe parent: 5rttakj6
garden mtc0 parent: 485snual
garden mtc0 per-scene: 2027uxg1
room mtc0 parent: uhqsu0wo
room mtc0 per-scene: z8k86kzz
flowers mtc0 parent: g6dib53z
treehill mtc0 parent: aau1supu
bicycle mtc0 parent: e9z6lhls
stump mtc0 parent: ng46x0x6
v57a counter shrink: fptifheb
v57a kitchen shrink: 4zevmx9g
v58a counter min64 tex32: 8ng4dnih
v58a kitchen min64 tex32: ezsrdzbx
v58b counter min4 tex32: 1wvclw9g
v58b kitchen min4 tex32: 7puzt1qa
v58c counter min4 tex16: e76hlmtb
v58c kitchen min4 tex16: ig7k3vtp
```

---

## 16. 最终汇报策略

建议 PPT 的口径是：

1. 主讲 Phase-J：这是当前有强证据的 MeshSplatting-over-baseline 结果；
2. 用 v48/v52 说明我们不是停留在 image-space adapter，而是在推进 surface-addressed representation；
3. 用 v53/v55d/v56 说明我们认真分析了 residual amplitude 和 reliability，不把负结果藏起来；v56 当前 fixed-command 的 missing-audit fresh probes 已闭合；
4. 用 v57a/v58 说明简单 face-alpha shrink 和 camera-center linear view basis 都已被验证但不是突破点，下一步要转向 surface-aware view-conditioned persistent surface residual representation；
5. 明确下一步：把 Phase-J 的大收益尽可能内化进 persistent surface representation，让定性差异更明显、论文故事更完整。

一句话底线：

> 现在已经有一个很强的阶段性结果可以向 mentor 汇报，但还不能把它包装成“顶会终局已完成”；最好的策略是用 Phase-J 做主结果，用 v48/v52/v56 展示我们已经找到下一阶段 representation-level 升级的方向和证据，并用 v57/v58 解释为什么下一步必须从 surface-aware view-conditioned representation 做真正创新。需要诚实说明 v56 已闭合当前 fixed-command 的 fresh-probe 审计和 `garden/room` 边界消融、但 effect size 仍小；v58 是有效负结果，不是当前 endpoint。
