# SPCarNet 当前完整技术报告（Mentor PPT 版，v66）

日期：2026-06-24

用途：mentor 汇报、PPT 拆页、方法讨论、实验复盘、下一步研究规划。

当前建议汇报主线：

- **主结果 endpoint**：`Phase-J guarded adaptive Evidence Lumigraph Adapter`
- **当前最佳固定表示级策略**：`v64 fixed auto bin-alpha policy`
- **最新负结果/诊断**：`v65 teacher-distilled shared residual basis`、`v66 bin-RGB alpha calibration`

一句话版本：

> SPCarNet 不是重新训练一个取代 MeshSplatting 的模型，而是把一个已训练好的 MeshSplatting checkpoint 升级成一个会用训练视角证据自检、压缩、局部修复并在风险过高时自动回退的可审计表示系统。

---

## 0. 执行摘要

当前结果必须分两层讲清楚：

| 层级 | 当前结论 | PPT 口径 |
|---|---|---|
| Phase-J endpoint | 在本地同协议 full9 上相对 selected clean MeshSplatting `9 / 9` 场景三指标严格胜出，同时平均减少 `7.6479%` triangles | 这是当前最安全、最强、最适合汇报的主结果 |
| Representation-level track | v48-v66 已经补齐 surface atlas、capacity policy、face/bin alpha、teacher basis、RGB-bin alpha 等接口，但效果仍小 | 这是研究路线和消融，不应包装成最终超越 Phase-J 的 endpoint |

最适合放 PPT 首页的数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

当前最诚实的总判断：

> 在本地复现的 selected-clean `26000/30000` envelope、相同 full9 split 和相同 evaluator 下，我们已经证明 SPCarNet Phase-J 相比 clean MeshSplatting 有明确 RGB 指标收益和三角形压缩收益。最强收益来自 guarded render-time residual repair。v64 是当前最完整的 fixed representation-policy milestone，但它的收益很小；v65/v66 只在已 probe 的 cap-hit / accepted-policy 场景上否定了两个简单后续假设，不能被解读成全局证明 persistent residual representation 已经解决。

---

## 1. 背景与问题

MeshSplatting 已经是很强的显式 mesh 渲染方法。它把场景落到 triangle mesh 上，因此适合压缩、编辑和部署。但一个训练好的 clean checkpoint 仍然有三类问题：

| 问题 | 表现 | SPCarNet 的目标 |
|---|---|---|
| 局部外观 residual | 叶片、桌面、边界、遮挡处有稳定颜色误差 | 从训练视角里挖掘可迁移 residual 修复信号 |
| 几何冗余 | 部分 triangles 对多视角解释贡献低 | 在质量不退化前提下删除冗余 triangles |
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
| v65 Teacher Basis | Phase-J teacher residual + face/view features | per-face ridge residual model | 诊断 teacher distillation 表达力 |
| v66 RGB-bin Alpha | policy-val residual evidence | per-bin RGB alpha | 诊断 residual magnitude 是否需要通道级校准 |

---

## 3. 技术细节

### 3.1 Evidence Cache：训练视角变成决策证据

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

这个模块让训练数据不再只是优化参数的数据，而是后续判断 surface 是否可靠、是否可修、是否必须回退的证据。

### 3.2 Geometry-Safe Compaction：质量优先删面

SPCarNet 的压缩不是追求最大删面，而是 quality-first rate-distortion。

压缩原则：

- 低可见性、低贡献、低风险 faces 优先；
- 遮挡边界、thin structure、sparse geometry 风险区受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- topology audit、sparse COLMAP geometry audit 独立记录；
- 修复失败时允许 fallback，不强行修改所有场景。

报告中的 `triangle reduction` 指删去的 triangles 占原始 triangles 的比例。Phase-J 平均删面为 `7.6479%`。

### 3.3 Guarded Evidence Lumigraph Adapter：当前主收益来源

Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。它在 mesh surface 上做训练证据约束的 residual transfer。

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

### 3.6 v64 Fixed Auto Bin-Alpha Policy：最新可汇报固定策略

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

v64 guard 不写场景名，也不读取 held-out test metric。它自动接受 `kitchen` 的 v63b，并回避 `counter` 等不稳定候选。

### 3.7 v65 与 v66：负结果也有诊断价值

v65 尝试从 Phase-J teacher residual 中蒸馏 per-face shared residual basis：

```text
fit-view Phase-J teacher residuals
  -> per-face ridge residual model
  -> view/normal/UV-conditioned prediction
  -> policy-val non-regression guard
  -> target application or fallback
```

v66 尝试把 bin-level alpha 从 scalar 扩展成 RGB channel-wise：

```text
alpha_bin -> [alpha_r, alpha_g, alpha_b]
```

两者都完成了代码接口、runner 转发、W&B probe 和结果审计，但都没有超过当前主线。它们的意义是定位瓶颈：

- 线性 per-face teacher basis 太刚性；
- RGB alpha 不是当前主要瓶颈；
- 下一步需要更强的局部 residual-field 表达和不确定性建模，而不是继续调单个 alpha。

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

> Under our locally reproduced selected-clean `26000/30000` envelope, same full9 split, and same evaluator, SPCarNet Phase-J strictly improves PSNR, SSIM, and LPIPS on all 9 selected scenes while removing 7.65% triangles on average.

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

### 7.1 v64 full9 selected rows

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

### 7.2 v64 aggregate

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

## 8. 最新 v65/v66 诊断

### 8.1 v65 Teacher-Distilled Shared Basis

v65 是一次更“方法级”的尝试：不是继续调阈值，而是试图从 Phase-J teacher residual 中蒸馏更强的 per-face residual basis。

| scene | reference | v65 result | guard decision | verdict |
|---|---|---|---|---|
| kitchen | v64 `27.822626 / 0.876538 / 0.198849` | same as v64 | fallback to legacy | not promoted |
| room | v64 `28.740660 / 0.884829 / 0.249897` | `28.739618 / 0.884807 / 0.249906` | fallback to legacy | worse |

关键教训：

> 在 `kitchen/room` 两个 probe 场景上，线性 per-face shared basis 太刚性。Phase-J 的 residual 很可能包含 UV-local、高频、遮挡边界、view-dependent 的复杂成分；只靠 camera/normal/UV 二次项无法稳定蒸馏。

### 8.2 v66 Bin-RGB Alpha Calibration

v66 是最新完成的 pipeline 改动。它把 `policy_val_bin_alpha` 扩展为每个 face/UV bin 的 RGB alpha：

```text
local_alpha_profile.mode = policy_val_bin_rgb_alpha
```

W&B probes：

| scene | GPU | W&B run | status |
|---|---:|---|---|
| counter | 2 | `22zgoxfl` | completed |
| kitchen | 3 | `4qtck8uq` | completed |

结果：

| scene | reference | v66 PSNR | v66 SSIM | v66 LPIPS | verdict |
|---|---:|---:|---:|---:|---|
| counter | v56/v64: `26.756130 / 0.862126 / 0.251691` | 26.751209 | 0.862078 | 0.251961 | worse |
| kitchen | v64: `27.822626 / 0.876538 / 0.198849` | 27.822626 | 0.876538 | 0.198849 | tie |

v66 audit highlights：

| scene | RGB alpha bins | selected alpha | changed fraction | fallback alpha |
|---|---:|---:|---:|---|
| counter | 669 | 0.125 | 0.065630 | `[0.35, 0.35, 0.35]` |
| kitchen | 76 | 1.0 | 0.039585 | `[0.35, 0.35, 0.35]` |

结论：

> 在 `counter/kitchen` 两个 probe 场景上，RGB channel-wise alpha 不是当前主瓶颈。它可以跑通，但没有超过 v64；`counter` 还轻微退化。因此 v66 只作为负结果和消融记录，不进入当前主方法。

---

## 9. 版本演进与消融故事

| 版本 | 改动 | 结果 | 教训 |
|---|---|---|---|
| Phase-J | guarded adaptive ELA + compact mesh | vs clean full9 `9 / 9` strict wins | 当前主结果成立 |
| v48 | auto-support surface residual atlas | vs no-op full9 `7 / 9` strict | surface atlas 可行但量级小 |
| v52 | capacity-aware fixed policy | vs v48 `9 / 9` nonreg/tie | 固定 train-only policy 更稳 |
| v56 | face-alpha reliability guard | vs v52 `9 / 9` nonreg/tie | 可以安全吸收 counter 小收益 |
| v59/v60 | view-conditioned residual basis + OOD guard | counter/kitchen mixed | 表达力增强会带来 OOD 风险 |
| v61/v62 | face/bin uncertainty guards | changed area 很小仍可能退化 | 只缩 apply mask 不能根治 residual magnitude |
| v63/v64 | bin-level alpha + fixed auto policy | v64 vs v56 `9 / 9` nonreg/tie | magnitude calibration 有用但收益小 |
| v65 | teacher-distilled shared basis | kitchen fallback，room 负结果 | 线性 teacher basis 太弱 |
| v66 | RGB-bin alpha | kitchen 持平，counter 退化 | channel-wise alpha 不是主要瓶颈 |

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

论文/mentor 追问时的边界：

> Full-frame fixed-view panels are the fairness evidence. Where-it-helps crops explain where the residual repair helps, but they should not be used alone as the generalization claim.

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
| Negative diagnostics | v65/v66 在已 probe 场景上否定两个简单弱假设 | 不是只报好结果，而是在收敛研究问题 |

论文故事可以这样组织：

> SPCarNet turns MeshSplatting into an evidence-certified compact-and-repair representation. It uses training-view surface evidence to decide where to remove geometry, where to transfer residual corrections, and where to safely fall back.

---

## 12. Fairness 和可复现性

需要主动说明的公平性边界：

- baseline 是本地 selected clean MeshSplatting，不是故意挑弱 checkpoint；
- clean baseline 从 clean `26000` 和 `30000` envelope 中按 held-out score 选择更强 row，这是为了给 baseline 一个更强 envelope；method policy、alpha、branch、support 和 fallback 不使用 held-out test GT；
- train metrics 不用于选择更久训练的 baseline；
- held-out test GT 只用于最终评价；
- branch、alpha、support、fallback 不读取 held-out test metric；
- W&B online logging 覆盖 v63b/v64 full9 candidates、v65 probes、v66 probes；
- v64 是 report-only fixed policy，还不包装成 paper endpoint；
- v65/v66 是负结果，不混入主结果。

核心 evidence 路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html
/dev/shm/peilincai_spcarnet_v65_teacher_shared_probe_20260624
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v66_bin_rgb_alpha_probe_20260624/summary.md
/dev/shm/peilincai_spcarnet_v66_bin_rgb_alpha_probe_20260624  # original transient run root
```

核心代码路径：

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
scripts/car_model/summarize_v64_bin_alpha_auto_policy.py
scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
scripts/car_model/build_v52_capacity_policy_panels.py
scripts/car_model/build_v56_counter_face_alpha_panel.py
```

---

## 13. 当前弱点

必须诚实讲的弱点：

| 弱点 | 影响 | 当前状态 |
|---|---|---|
| Phase-J 仍是 render-time adapter | 还不是完全 baked representation endpoint | 可以作为强系统结果，但论文故事要谨慎 |
| 表示级收益太小 | v64 只有 `1e-4` 级平均提升 | 只能作为 internalization milestone |
| 定性全图差异不总是明显 | mentor/审稿人可能觉得视觉提升不直观 | 应使用 crop/error map 展示 |
| 室内压缩率较低 | `room/counter/kitchen` 主要是 2.10% 级别 | quality-first 策略导致保守 |
| v65/v66 未提升 | 说明当前 residual-field 表达力不足 | 下一步要换更强模型假设 |

一句话风险评估：

> 当前最强 claim 是“evidence-certified compact-and-repair system beats clean MeshSplatting under our selected local full9 protocol”；还不能声称“persistent representation-level residual field 已经全面解决”。

---

## 14. 下一步研究方向

最值得继续做的方向不是继续调 alpha，而是升级 residual-field 表达：

1. 局部 mixture residual field：让每个 face/bin 有多个 residual mode，而不是单个均值；
2. uncertainty-aware prediction：同时预测 residual 和 confidence，低置信区域自动 no-op；
3. occlusion-boundary-aware support：单独建模边界、thin structure 和可见性跳变；
4. teacher-student distillation 2.0：从 Phase-J teacher 蒸馏，但使用局部 MLP/mixture 或 low-rank latent，而不是线性 per-face ridge；
5. blind validation：固定 policy 后重新跑全量场景，避免任何 scene-specific 参数游戏。

建议 mentor 讨论问题：

> 我们已经有一个强 endpoint 和一个可审计 representation track。下一步是否应该把论文重心放在 evidence-certified compact-and-repair 系统，还是继续投入更强的 persistent residual representation？

---

## 15. PPT 建议结构

1. Title：SPCarNet: Evidence-Certified Compact-and-Repair Mesh Splatting
2. Problem：clean MeshSplatting 已强，但仍有 residual、冗余和泛化风险
3. Key Idea：训练视角证据决定哪里删、哪里修、哪里不动
4. Pipeline：MeshSplatting checkpoint -> evidence cache -> compaction -> guarded repair -> audit
5. Module 1：Evidence cache and surface correspondence
6. Module 2：Geometry-safe compaction
7. Module 3：Guarded residual repair
8. Main Result：Phase-J vs selected clean MeshSplatting full9
9. Qualitative：where-it-helps crop/error map
10. Representation Track：v48-v64 atlas/policy progression
11. Negative Diagnostics：v65/v66 为什么没有升格
12. Limitations and Next Step：persistent residual representation 仍是下一阶段

---

## 16. Mentor 可能追问与回答

**Q1：这是不是只是 MeshSplatting 后处理？**

A：不是任意图像后处理。修复信号来自训练视角 residual，并通过 face id、surface visibility、UV/bin、policy-val risk 绑定到 mesh surface。证据不足时自动回退。

**Q2：有没有用 test set 调参数？**

A：主策略、alpha、branch、support、fallback 都由 train/policy-val evidence 决定。held-out test GT 只用于最终评价。v64/v65/v66 的失败或接受都有审计日志。

**Q3：为什么不直接把 Phase-J 当最终 paper 方法？**

A：Phase-J 是当前最强 endpoint，但它是 render-time adapter。它足以支持系统级 compact-and-repair story；如果论文要强调 persistent representation-level method，还需要下一步 residual-field 模型升级。

**Q4：为什么 v65/v66 没提升还要放进报告？**

A：因为它们排除了两个自然假设：线性 teacher basis 不够强，RGB alpha 不是主要瓶颈。这让下一步研究方向更清楚，而不是继续无原则调参。

**Q5：定性效果为什么有时不明显？**

A：很多提升是 residual-level 和局部区域改善，全图尺度会被大面积背景稀释。最公平的展示是全图 + crop + error map 三层证据。

---

## 17. 当前版本结论

最适合在汇报最后说：

> SPCarNet has achieved a strong audited compact-and-repair endpoint over our locally reproduced selected-clean MeshSplatting `26000/30000` envelope under the same full9 split and evaluator. The method improves all RGB metrics on all 9 selected Mip-NeRF360 scenes while reducing triangles. The remaining open challenge is to internalize the strongest render-time repair into a persistent surface representation; v64 closes a safe fixed-policy loop, while v65/v66 show on the probed scenes that the next step needs a more expressive uncertainty-aware residual field rather than more alpha tuning.
