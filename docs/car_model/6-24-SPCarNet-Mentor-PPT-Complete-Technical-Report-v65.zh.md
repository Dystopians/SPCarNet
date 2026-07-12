# SPCarNet 当前完整技术报告（Mentor PPT 版，v65）

日期：2026-06-24

用途：mentor 汇报、PPT 拆页、方法讨论、结果复盘、后续实验规划

当前建议汇报主线：

- 主结果 endpoint：`Phase-J guarded adaptive Evidence Lumigraph Adapter`
- 当前最佳固定表示级策略：`v64 fixed auto bin-alpha policy`
- 最新未提升诊断：`v65 teacher-distilled shared residual basis`

---

## 0. 执行摘要

SPCarNet 的目标不是替换 MeshSplatting，而是把一个已经训练好的 MeshSplatting checkpoint 变成一个能自检、能安全压缩、能局部修复、能自动回退的表示系统。

一句最通俗的话：

> MeshSplatting 是训练完直接渲染；SPCarNet 是训练完以后先检查哪些三角形可以安全删除，哪些 surface 区域在训练视角中反复画错可以修，哪些区域证据不足必须保持原样。

当前状态必须分成两层讲：

| 层级 | 当前结论 | 汇报口径 |
|---|---|---|
| Phase-J endpoint | 在本地同协议 full9 上全面超过 selected clean MeshSplatting，同时减少 triangles | 这是当前 presentation-safe 主结果 |
| v48-v65 representation track | 已实现 surface atlas、capacity policy、face/bin alpha calibration、view basis、teacher basis，但收益仍小 | 这是把 render-time 修复内化到 persistent surface representation 的研究路线 |

最适合 PPT 第一页放的数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

最新表示级策略 v64 的数字：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |

最新 v65 结论：

| scene | v65 result | guard decision | verdict |
|---|---:|---|---|
| kitchen | same as v64 within float noise | fallback to legacy atlas | not promoted |
| room | worse than v56/v64 | fallback to legacy atlas | not promoted |

当前最诚实的总判断：

> 我们已经能证明 SPCarNet 相比基础 MeshSplatting 在当前本地 full9 口径下有明确、全面的 RGB 指标收益和三角形压缩收益；但最强收益仍来自 Phase-J 的 guarded render-time ELA。v64/v65 证明表示级内化路线可行且安全接口逐步闭合，但 persistent residual representation 的效果还没有强到替代 Phase-J 主结果。

---

## 1. 背景：MeshSplatting 还缺什么

MeshSplatting 已经是很强的显式 mesh 渲染方法。它把 scene 表示落在 triangle mesh 上，因此天然适合压缩、编辑和部署。但基础 checkpoint 仍有三个问题：

| 问题 | 表现 | SPCarNet 的目标 |
|---|---|---|
| 局部外观 residual | 叶片、桌面、边缘、遮挡边界、细纹理区域仍有稳定误差 | 从训练视角 residual 中挖掘可迁移修复信号 |
| 几何冗余 | 一些 triangles 对多视角解释贡献低 | 在质量不退化的前提下删除冗余 triangles |
| 泛化风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 视角 | 用 train-only evidence gate 决定修复、回退或 no-op |

研究问题可以写成：

> Given a trained MeshSplatting checkpoint, can we use training-view surface evidence to certify where the mesh can be compacted and where appearance residuals can be safely repaired?

这不是普通图像后处理。SPCarNet 的修复信号必须绑定到真实 mesh surface、face id、可见性、barycentric/UV bin、多视角一致性和 policy-validation 风险上。

---

## 2. 方法总览

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
  -> build train/policy-val evidence cache
  -> estimate surface support, residual, visibility, risk
  -> geometry-safe compaction
  -> guarded residual repair
  -> train-only policy selection and fallback
  -> held-out evaluation
```

核心模块：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Evidence Cache | train renders、GT、camera、surface maps | residual、face id、normal、UV/bin、support、risk | 把训练视角变成可审计证据 |
| Geometry-Safe Compaction | mesh + evidence | compact mesh | 删除低风险冗余 triangles |
| Guarded ELA | compact render + train residual evidence | repaired render | 用多视角一致 residual 修复 held-out view |
| Train-Only Policy | policy-val metrics and risk | accept/fallback/no-op | 防止 test leakage 和 risky edit |
| Surface Residual Atlas | face/UV/bin evidence | persistent residual field | 把修复信号绑定到 surface 表示 |
| v64 Fixed Auto Policy | v63b audit + v56 fallback | selected full9 tree | 自动选择 bin-alpha residual atlas 或回退 |
| v65 Teacher Basis | Phase-J teacher residual + face/view features | per-face ridge residual model | 尝试更强 teacher distillation，当前负结果 |

---

## 3. 方法细节

### 3.1 Evidence Cache：把训练视角变成决策证据

Evidence cache 保存每个训练或 policy-val 视角中的：

- rendered RGB；
- ground-truth RGB；
- residual map：`GT - Render`；
- alpha / visibility；
- face id；
- barycentric coordinate 或 UV bin；
- normal、depth、camera center；
- per-face 和 per-bin sample count；
- residual sign consistency；
- image L1、PSNR、SSIM、LPIPS；
- per-view mean、min-view、CVaR tail risk。

这个模块的核心作用是：训练视角不再只是优化参数的数据，而是后续判断 surface 是否可靠、是否可修、是否必须回退的证据。

### 3.2 Geometry-Safe Compaction：质量优先的删面

SPCarNet 的压缩不是追求最大删面，而是 quality-first rate-distortion。

压缩原则：

- 低可见性、低贡献、低风险 faces 优先；
- 遮挡边界、thin structure、sparse geometry 风险区受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- topology audit、sparse COLMAP geometry audit 独立记录；
- 修复失败时允许 fallback，不强行修改所有场景。

当前报告中的 `triangle reduction` 指删去的 triangles 占原始 triangles 的比例。Phase-J 平均删面为 `7.6479%`。

### 3.3 Guarded Evidence Lumigraph Adapter：当前主收益来源

Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。它可以理解成在 mesh surface 上做训练证据约束的 residual transfer。

简化公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 的像素；
- `residual_i(u_i)` 来自训练视角中相同或相近 surface 区域的 residual；
- `w_i(p)` 由可见性、surface correspondence、local support 和风险门控决定；
- `alpha` 由 train/policy-val evidence 选择；
- 如果证据不足或 tail risk 高，则回退或 no-op。

通俗讲法：

> 我们不是凭空增强图片，而是看训练视角里同一个 surface 区域是否反复出现稳定错误。如果错误稳定且多视角支持充足，就把这部分 residual 迁移到目标视角；否则不动。

### 3.4 Train-Only Policy：避免 test-set 参数游戏

SPCarNet 的 policy 设计强调 fairness：

- branch selection、alpha、support、fallback 都来自 train 或 policy-val evidence；
- held-out test GT 只用于最终报告；
- 不用 test metric 选择参数；
- 不用 train metric 选择训练更久的 baseline；
- 风险高时回退稳定版本。

Phase-J 中多数场景使用 adaptive alpha，`treehill` 使用 auto edge fallback。这一分支选择不是看 held-out test 得出的。

### 3.5 Surface Residual Atlas：表示级内化路线

为降低 render-time adapter 风险，我们进一步实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face/bin selection
  -> residual atlas fitting
  -> policy-val risk gate
  -> target view surface lookup
  -> persistent surface-addressed repair
```

这个分支不是训练一个大型新网络，而是从训练证据中估计 mesh-attached residual field。它回答的问题是：

> 能否把训练视角 residual 变成绑定在 mesh 表面的、可审计、可回退、可复用的修复表示？

当前结论：接口、安全策略和 full9 自动 policy 已经成型，但 RGB 效果量级仍小，还不能替代 Phase-J。

### 3.6 v64 Fixed Auto Bin-Alpha Policy：最新可汇报的固定策略

v61/v62 诊断说明，仅判断某个 bin 能不能修是不够的。Residual 的 magnitude 错了，即使作用区域很小，也会伤害 held-out view。

v63/v63b 加入 bin-level residual magnitude calibration：

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

v64 guard 不写场景名，也不读取 held-out test metric：

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

它自动接受 `kitchen` 的 v63b，并回避 `counter` 等不稳定候选。

### 3.7 v65 Teacher-Distilled Shared Basis：最新负面诊断

v65 尝试从 Phase-J teacher residual 中蒸馏一个更强的 per-face shared residual basis：

```text
fit-view Phase-J teacher residuals
  -> per-face ridge residual model
  -> view/normal/UV-conditioned prediction
  -> policy-val non-regression guard
  -> target application or fallback
```

新增接口：

```text
--teacher_distilled_basis_mode {none,face_uv_normal_camera_ridge}
--teacher_distilled_basis_guard_mode {none,policy_val_nonregressive}
--teacher_distilled_basis_min_face_samples
--teacher_distilled_basis_ridge
--teacher_distilled_basis_ood_max_z
--teacher_distilled_basis_ood_min_std
--teacher_distilled_basis_apply_mode {replace_supported,blend,fill_empty_only}
--teacher_distilled_basis_blend
```

特征基：

```text
[1,
 camera_center_x, camera_center_y, camera_center_z,
 normal_x, normal_y, normal_z,
 dot(normal, camera_center),
 u, v, u^2, v^2, u*v]
```

每个 face 解一个 ridge residual model：

```text
residual_rgb ~= X @ W_face
```

v65 的重要价值不是提升指标，而是证明当前线性 per-face shared basis 不够强：

- kitchen 有 `782` 个 supported teacher-basis faces，但 policy-val guard 认为它弱于 legacy atlas；
- room 有 `227` 个 supported teacher-basis faces，也被 guard fallback；
- 说明瓶颈不是简单 sample 数量，而是模型类太刚性，无法表达 UV-local/high-frequency/occlusion-boundary residual。

---

## 4. 与基础 MeshSplatting 的区别

| 维度 | 基础 MeshSplatting | SPCarNet |
|---|---|---|
| 训练后行为 | 直接渲染 checkpoint | evidence mining、压缩、修复、审计 |
| 几何 | 原始 mesh | compact mesh + topology/geometry audit |
| 外观 | checkpoint 属性直接渲染 | guarded residual adapter / surface residual atlas |
| 风险控制 | 无显式 policy gate | train-only gate、CVaR/min-view、fallback |
| Test GT 用途 | 评价 | 只评价，不参与策略选择 |
| 失败处理 | 没有显式回退 | gate 不通过则 no-op 或回退稳定版本 |

mentor 面前建议这样讲：

> MeshSplatting 给了我们一个强 starting point；SPCarNet 不推翻它，而是把它升级成可自检、可压缩、可局部修复、可审计回退的系统。

---

## 5. 主结果：Phase-J vs Clean MeshSplatting

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
| Sparse geometry-safe scenes | `9 / 9` |
| Sparse geometry strict wins | `6 / 9` |
| Per-view strict RGB wins | `244 / 246` |

PPT headline：

> On Mip-NeRF360 full9, SPCarNet Phase-J strictly improves PSNR, SSIM, and LPIPS on all 9 selected scenes while removing 7.65% triangles on average.

---

## 6. 与 MeshSplatting 论文表格的关系

当前可引用的外部口径对比：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| Local selected clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |

严谨说法：

- 可以说本地 Phase-J 数值高于 MeshSplatting paper table；
- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准；
- 论文表格可能存在 resolution、mask、split、preprocessing、evaluator 差异；
- 我们本地 clean baseline 不是故意挑弱，而是从 clean envelope 中选择 held-out 更强 row；
- train metric 不参与选择 baseline 或最终方法结果。

PPT 建议写法：

> We report the main claim against our locally reproduced clean MeshSplatting baseline under the same data, rendering, and evaluator protocol. Paper-table numbers are shown only as external context.

---

## 7. 最新表示级结果：v64 Fixed Auto Policy

### 7.1 Full9 candidate status

v64 已补齐所有 v63b full9 candidate，并保留 W&B online logging：

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

- v64 是真实 fixed-policy milestone；
- 它把 v63b 从手工 probe 变成 train-policy-val 自动选择策略；
- 它能自动选择 `kitchen` 的 v63b 并回退其他场景；
- 它相对 v56 保持 `9 / 9` non-regressive/tie；
- 但它的平均收益太小，不能作为论文终局 headline。

---

## 8. 最新 v65 结果：Teacher Basis 未提升

v65 是一次更“方法级”的尝试：不是继续调阈值，而是试图从 Phase-J teacher residual 中蒸馏更强的 per-face residual basis。

### 8.1 Probe setup

输出路径：

```text
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624
```

W&B runs：

| scene | GPU | W&B run | status |
|---|---:|---|---|
| kitchen | 5 | `zrqz5kzw` | completed |
| room | 4 | `bfnmewgo` | completed |

### 8.2 Kitchen

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v52/v56 reference | 27.818935 | 0.876535 | 0.199019 |
| v64 selected | 27.822626 | 0.876538 | 0.198849 |
| v65 probe | 27.822626 | 0.876538 | 0.198849 |

Teacher basis audit：

| field | value |
|---|---:|
| requested mode | `face_uv_normal_camera_ridge` |
| effective mode | `none` |
| guard decision | `fallback_to_legacy` |
| supported faces | `782` |
| candidate faces | `4343` |
| supported-face fraction | `0.180060` |
| selected alpha | `1.0` |
| changed fraction | `0.039585` |
| bin-alpha count | `67` |

Fallback reasons：

```text
ssim_gain 0.00033229 < legacy 0.00036997
image_l1_gain 0.00004583 < legacy 0.00004735
image_l1_cvar20_view_gain 0.00001873 < legacy 0.00002198
image_l1_min_view_gain 0.00001412 < legacy 0.00001479
```

### 8.3 Room

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v52/v56/v64 reference | 28.740660 | 0.884829 | 0.249897 |
| v65 probe | 28.739618 | 0.884807 | 0.249906 |

Teacher basis audit：

| field | value |
|---|---:|
| requested mode | `face_uv_normal_camera_ridge` |
| effective mode | `none` |
| guard decision | `fallback_to_legacy` |
| supported faces | `227` |
| candidate faces | `1160` |
| supported-face fraction | `0.195690` |
| selected alpha | `0.125` |
| changed fraction | `0.010602` |
| bin-alpha count | `177` |

Fallback reasons：

```text
ssim_gain 0.00002179 < legacy 0.00002352
image_l1_cvar20_view_gain -0.00000027 < legacy 0.00000010
image_l1_min_view_gain -0.00000144 < legacy -0.00000121
```

### 8.4 v65 takeaway

v65 的结论：

- teacher basis 接口、CLI、W&B logging 已补齐；
- policy-val guard 能正确阻止弱 teacher basis 被提升；
- kitchen 没有超过 v64；
- room 相比 v56/v64 反而更差；
- 因此 v65 不应作为当前汇报 endpoint。

最重要的技术教训：

> 线性 per-face shared basis 太刚性。Phase-J 的 residual 很可能包含 UV-local、高频、遮挡边界、view-dependent 的复杂成分；只靠 camera/normal/UV 二次项无法稳定蒸馏。

---

## 9. 消融和版本演进

这一阶段的价值不只是最终结果，也包括逐步定位 residual repair 的真实瓶颈。

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
| v65 | teacher-distilled shared basis | kitchen fallback，room 负结果 | 线性 shared basis 太弱，需要局部 mixture 或 uncertainty model |

关键反思：

> 表示级 residual atlas 的核心瓶颈不是“有没有 gate”，而是 residual magnitude、view-conditioned generalization、occlusion boundary 和 low-support bins 的联合不确定性。

---

## 10. 定性展示策略

全图直接放进 PPT 时，residual-level improvement 可能不够显眼。因此建议使用三层定性证据：

1. 全图对比：证明同一 scene、同一 view、同一 evaluator；
2. 局部 crop：突出叶片、桌面边缘、细纹理、遮挡边界；
3. Error map：展示 residual 降低位置，这是最能说明局部修复价值的图。

推荐用于 PPT 的素材：

| 用途 | 路径 |
|---|---|
| Phase-J where-it-helps showcase | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| Mip-NeRF360 full9 全图 gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| Mip-NeRF360 full9 crop gallery | `assets/spcarnet_m360_full9_crop_gallery.png` |
| where-it-helps backup | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| outdoor detail backup | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| v52 capacity/cap-hit panel | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |
| v56 counter face-alpha panel | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |
| v64 selected full9 gallery | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html` |

PPT 讲法：

> 全图负责公平性，crop 和 error map 负责解释改进位置。我们不夸大全图肉眼差异，而是强调 residual-level repair 的局部稳定收益。

示例图片：

![Phase-J where-it-helps showcase](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

![v56 counter face-alpha guard panel](../../assets/spcarnet_v56_counter_face_alpha_guard_panel.png)

---

## 11. 为什么这是研究工作，不是工程调参

| 贡献 | 研究含义 | 为什么不是普通调参 |
|---|---|---|
| Training evidence mining | 从训练视角恢复 surface reliability 和 residual risk | 决策来源是可复用 evidence interface |
| Geometry-safe compaction | 质量优先的 mesh rate-distortion | 不只是删面比例，而是有 topology/geometry audit |
| Guarded residual repair | 多视角 residual transfer 绑定 surface support | 不是任意图像滤镜 |
| Train-only policy/fallback | 证据不足自动 no-op | 有部署安全逻辑，避免 test leakage |
| Surface residual atlas | 把 repair 内化到 face/UV-addressed 表示 | 是 representation-level internalization |
| Bin-alpha calibration | 从二值 allowlist 变成局部 residual magnitude calibration | 针对 failure mode 的模型结构改造 |
| Fixed auto policy | 自动选择和回退，不写场景名 | 减少手工场景参数游戏 |
| Teacher-distilled basis | 尝试从强 teacher 内化到 persistent surface model | 负结果也说明模型表达力边界 |

论文故事可以这样组织：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. It uses training-view surface evidence to decide where to remove geometry, where to transfer residual corrections, and where to safely fall back.

---

## 12. Fairness 和可复现性

需要主动说明的公平性边界：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean `26000` 和 `30000` envelope 中按 held-out score 选择更强 row；
- train metrics 不用于选择更久训练的 baseline；
- held-out test GT 只用于最终评价；
- branch、alpha、support、fallback 不读取 held-out test metric；
- W&B online logging 覆盖 v63b/v64 full9 candidates 和 v65 probes；
- v64 是 report-only fixed policy，还不包装成 paper endpoint；
- v65 是负结果，不混入主结果。

关键 evidence 路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624
```

核心代码路径：

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
scripts/car_model/summarize_v64_bin_alpha_auto_policy.py
scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
```

验证命令示例：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  scripts/car_model/summarize_v64_bin_alpha_auto_policy.py \
  scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
```

---

## 13. 当前局限

| 局限 | 影响 | 汇报中如何诚实处理 |
|---|---|---|
| Phase-J 最强收益来自 render-time ELA | 可能被质疑不是 fully persistent representation | 主动承认，并把 v48-v65 作为内化路线 |
| v64 表示级收益很小 | 不能替代 Phase-J headline | 说成 fixed-policy milestone，不说成终局 |
| v65 teacher basis 未提升 | 说明简单线性蒸馏不够 | 作为负面诊断，指向更强 residual field |
| 全图定性差异不总明显 | mentor 可能觉得视觉不够强 | 用 crop 和 error map 展示局部 residual repair |
| sparse geometry strict win 不是 9/9 | 几何指标还没有全面严格胜出 | 说 geometry-safe 是当前闭合，strict geometry 仍是后续方向 |
| paper table 口径可能不同 | 不能过度宣称 external SOTA | 主 claim 以本地同协议 baseline 为准 |

最重要的诚实结论：

> 当前我们已经解决了“能否击败基础 MeshSplatting baseline”的核心证明问题，但还没有完全解决“把强收益全部内化为 persistent surface representation”的终局问题。

---

## 14. Mentor 可能问的问题和建议回答

| 问题 | 建议回答 |
|---|---|
| 你们相比原始 MeshSplatting 到底强在哪里？ | 在本地同协议 full9 上，Phase-J 对 selected clean MeshSplatting 达成 9/9 场景 PSNR/SSIM/LPIPS 严格提升，同时平均减少 7.65% triangles。 |
| 这是不是只是后处理？ | Phase-J 有 render-time adapter 成分，但它不是任意图像后处理，而是由训练视角 surface evidence、visibility、face support 和 risk gate 约束的 residual transfer。我们也在推进 v48-v65 的 persistent atlas 内化路线。 |
| 最新 v64 是不是最终方法？ | 不是。v64 是最新 fixed auto policy milestone，证明 bin-alpha residual atlas 可以自动安全选择，但收益还太小，不能替代 Phase-J 主结果。 |
| v65 为什么没有提升？ | 它使用线性 per-face shared basis，policy-val 发现其不如 legacy atlas。说明 Phase-J teacher residual 包含更强的局部和 view-dependent 结构，需要 local mixture 或 uncertainty-aware residual field。 |
| 会不会用了 test set 调参数？ | 不会。alpha、fallback、branch selection 使用 train/policy-val evidence；held-out test 只用于最终 reporting。 |
| clean baseline 是否公平？ | 本地 clean baseline 从 clean envelope 中选择 held-out 更强 row。我们不使用 train metric 选择更久训练 checkpoint。 |
| 视觉上为什么有些图差异不明显？ | 方法主要修复局部 residual，全图缩放后不一定显眼。应看局部 crop 和 error map。 |
| 下一步最值得做什么？ | 不是继续阈值扫描，而是做 uncertainty-certified persistent residual field，让表示级模型继承 Phase-J 的大收益。 |

---

## 15. PPT 拆页建议

| Slide | 标题 | 内容 |
|---:|---|---|
| 1 | Title | `SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting` |
| 2 | Problem | MeshSplatting strong but has residual error, geometry redundancy, tail-view risk |
| 3 | Key Idea | 用训练视角证据判断 where to compact, repair, fallback |
| 4 | Pipeline | MeshSplatting -> evidence cache -> compaction -> guarded repair -> eval |
| 5 | Evidence Cache | residual、face id、visibility、barycentric/UV、support、risk |
| 6 | Geometry Compaction | quality-first triangle reduction with topology/geometry audit |
| 7 | Guarded ELA | residual transfer formula and train-only fallback gate |
| 8 | Fair Protocol | local clean baseline, train-only policy, test-only reporting |
| 9 | Main Quant Result | Phase-J full9 `9/9`, mean dPSNR `+1.3311` |
| 10 | Per-Scene Table | 9 scenes, RGB deltas, triangle reduction |
| 11 | Qualitative | full image + crop + error map |
| 12 | Representation Track | v48-v65 surface atlas and residual field internalization |
| 13 | Latest v64 | fixed auto bin-alpha policy, kitchen selected, others fallback |
| 14 | Latest v65 | teacher-basis negative result and lesson |
| 15 | Why Research | evidence-certified compact-and-repair representation |
| 16 | Limitations | render-time component, small persistent gains, visual subtlety |
| 17 | Next Step | uncertainty-certified persistent residual field |

---

## 16. 可直接放进 PPT 的中文摘要

SPCarNet 将 MeshSplatting 从“训练完成后直接渲染”的静态 mesh checkpoint，升级成“训练证据驱动的可压缩、可修复、可回退表示”。我们从训练视角挖掘 surface evidence，判断哪些 triangles 可以安全删除，哪些局部 residual 可以迁移修复，哪些区域因为证据不足必须保持原样。在 Mip-NeRF360 selected full9 上，当前 Phase-J endpoint 相对本地强 clean MeshSplatting baseline 达成 `9/9` 场景 PSNR/SSIM/LPIPS 严格胜出，同时平均减少 `7.65%` triangles。最新 v64 fixed auto policy 进一步把表示级 bin-alpha residual atlas 从手工 probe 推进到自动选择和回退策略，在 full9 上相对 v56 保持 `9/9` non-regressive/tie。v65 teacher-distilled shared basis 是一次更高容量 residual field 尝试，但 policy-val guard 发现其不如 legacy atlas，因此不提升为 endpoint。当前最清晰的下一步是构建 uncertainty-certified persistent residual field，让表示级模型继承 Phase-J 的大幅收益。

---

## 17. 可直接放进 PPT 的英文摘要

SPCarNet upgrades MeshSplatting from a static trained mesh into a training-evidence-certified compact and repairable representation. Given a clean MeshSplatting checkpoint, it mines training-view surface evidence, removes low-risk redundant triangles, and transfers reliable residual appearance cues through a guarded residual adapter. All repair decisions are driven by train/policy-validation evidence, while held-out test views are used only for reporting. On Mip-NeRF360 selected full9, the current Phase-J endpoint strictly improves PSNR, SSIM, and LPIPS over a strong locally selected clean MeshSplatting baseline on all 9 scenes, with 244/246 per-view strict wins and 7.65% average triangle reduction. The latest v64 branch completes a fixed auto policy over full9 v63b bin-alpha candidates, selecting kitchen automatically while falling back elsewhere; it is non-regressive over v56 but remains small-effect. The newest v65 teacher-distilled basis is a negative diagnostic: a linear per-face shared basis is too rigid to inherit the Phase-J teacher signal, motivating uncertainty-certified persistent residual fields.

---

## 18. 最终汇报口径

可以大胆说：

- 当前 Phase-J 在本地同协议 full9 上全面超过基础 MeshSplatting baseline；
- 不只是 RGB 指标提升，还同时有平均 `7.65%` triangle reduction；
- 决策来自 train/policy-val evidence，不是 test-set 参数游戏；
- v64 已经把最新 residual atlas probe 固化成自动策略；
- v65 说明简单 teacher basis 不够强，下一步需要真正的 uncertainty-aware persistent residual field。

不要过度说：

- 不要说 v64/v65 已经是论文终局；
- 不要说 persistent residual atlas 已经全面解决；
- 不要只拿 paper table 做唯一公平对比；
- 不要把全图视觉差异夸大，应使用 crop 和 error map。

Bottom line：

> Phase-J is the current strong and presentation-safe endpoint. v64 is the latest fixed-policy representation milestone. v65 is a useful negative diagnostic. The next paper-level leap is to make the persistent residual field strong enough that it visibly and quantitatively inherits the Phase-J gains.

