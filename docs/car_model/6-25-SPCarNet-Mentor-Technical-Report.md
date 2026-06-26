# SPCarNet / POD-MoE Mentor 技术报告

Date: 2026-06-25

用途：2026-06-26 给 mentor 做 PPT 的技术报告。本文以当前已完成的 **v106 POD-MoE base-preserve** full9 结果为主，补充 SPCarNet 相对基础 MeshSplatting 的方法差异、模块级实现、证据路径、局限与下一步。所有结果按“已完成 / 未验证”分开标注。

## 0. 当前汇报入口与最新状态

从 fresh clone 准备 PPT 时，建议先看这些入口文件：

| 用途 | 文件 |
|---|---|
| 根目录入口 | `SPCARNET_REPORT_INDEX.md` |
| 可克隆汇报包 manifest | `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md` |
| 当前第一阅读技术报告 | `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md` |
| 当前可克隆索引 | `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md` |
| v106 full9 对比表 | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |
| v113b safety summary | `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md` |
| v113c/v114 continuation summary | `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md` |
| strict branch mechanical package | `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md` |

当前可以汇报的最强事实：

- `v106 POD-MoE base-preserve` 是当前已验证质量主线；在本地 selected full9 表上相对 clean MeshSplatting baseline 的均值提升为 `+0.679598 PSNR`, `+0.011812 SSIM`, `-0.019185 LPIPS`。
- v106 相对 v104c anchor 的增量很小但方向一致：mean `+0.002181 PSNR`, `+0.000103 SSIM`, `-0.000112 LPIPS`。
- `v110/v110b` strict split 证明了一个关键短板：只靠 train/odd gate 仍可能在 held-out test 上低于 v106 parent，garden 是代表性失败。
- `v113b` 新增 lower-tail metric certificate 和 target-GT-free OOT support certificate；它能把 flowers/garden 的 strict-gate 风险修回 parent-safe，但不是质量突破。
- `v113c` 把 OOT fallback 细化到 frame level，garden 从 v110b 的 `25.430321 / 0.783703 / 0.186970` 提升到 `25.499817 / 0.786888 / 0.184260`，但仍低于 v106 parent 的 `25.790945 / 0.799382 / 0.174480`，因此不晋升。
- `v114 OOF-refit POD-MoE` 是当前 candidate-side 长程尝试，目标是把改进从“只会回退保安全的 gate”推进到“候选本身更强”。当前 garden field build 仍在运行，还没有 render/eval 结果。

当前本地长程任务状态：

| 任务 | 本地根目录 | 当前状态 |
|---|---|---|
| v110 counter strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter` | field build 约 `100/105` train/even views；field artifact 尚未产出 |
| v110 bonsai strict candidate | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/bonsai` | field build 约 `105/128` train/even views；field artifact 尚未产出 |
| v111 flowers end-to-end strict | `/dev/shm/peilincai_spcarnet_v111_end_to_end_strict_parent_gate_20260625/flowers` | parent field build 约 `121/151` train/all views；field artifact 尚未产出 |
| v114 garden OOF-refit POD-MoE | `/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden` | field build 约 `26/161` train/all views；field artifact 尚未产出 |

因此停止口径必须诚实：当前项目已经有完整可克隆汇报包、v106 全九场景正向质量线、定性 contact sheets、strict-split 接口和安全 gate 诊断；但 paper-final strict branch 尚未闭环，不能宣称 v113/v114 已全面超越 v106 或完成最终论文终局。

## 1. 当前可汇报结论

### 结论强度分级

| claim | 当前强度 | 证据 | 汇报口径 |
|---|---|---|---|
| v106 相对本地 clean MeshSplatting baseline 提升 | strong local evidence | full9 `9 / 9` scenes，PSNR/SSIM/LPIPS 三项均值和逐场景均正向 | 可以作为当前主结果，但必须说明这是本地 selected clean baseline 与本地 evaluator |
| v106 相对 v104c anchor 继续提升 | stable but tiny | full9 mean `+0.002181 PSNR / +0.000103 SSIM / -0.000112 LPIPS`，`9 / 9` scenes 三项同向 | 可以强调“稳定、保守、方向一致”，不要说“大幅” |
| v106 视觉效果显著优于 baseline | weak / subtle | contact sheets 有局部 error-map 变淡，但肉眼 RGB 差异不明显 | PPT 中应配 error map/crop，不要只放整图 RGB |
| v106 解决几何压缩问题 | not supported by v106 evidence | v106 是 residual-field / POD-MoE，不是 triangle-pruning 方法 | 不应把 earlier compaction 的删面率归因给 v106 |
| v107b 优于 v106 | refuted by current 4-scene probe | `counter`, `flowers`, `garden`, `bonsai` 四个有效 probe 均通过且相对 clean 仍正向，但相对 v106 mean 为 `-0.003646 PSNR / -0.000134 SSIM / +0.000104 LPIPS` | 写成 completed negative reliability stress test，不能晋升 |

### 已完成结论

- 当前 representation-level 候选：**v106 POD-MoE base-preserve**。
- 完成范围：本地 Mip-NeRF360 selected full9，同一 evaluator，`9 / 9` 场景均有结果。
- 主要结果：v106 相对本地 v104c representation-field anchor 在 `9 / 9` 场景上三项图像指标同向小幅改善。
- 平均增益相对 v104c：`+0.002181 PSNR`，`+0.000103 SSIM`，`-0.000112 LPIPS`。
- 平均增益相对 clean MeshSplatting：`+0.679598 PSNR`，`+0.011812 SSIM`，`-0.019185 LPIPS`。这主要继承自 v104c 的稳定 residual-field 改善，v106 本身是在 v104c 之上再加小幅专家增益。
- 逐场景相对 clean baseline 的 PSNR 增益范围大约是 `+0.255415` 到 `+1.420856`；最明显的 scenes 是 `bonsai`, `kitchen`, `room`, `garden`，较弱的是 `stump`, `treehill`, `flowers`。

安全 headline：

> SPCarNet v106 将单一 surface-attached residual field 扩展为保守的 POD-MoE 专家混合：保留 v104c 稳定基底，再叠加 detail expert 与 occlusion-boundary expert。它在 full9 上稳定超过 v104c，但当前收益幅度很小，还不能包装成大幅 paper-level breakthrough。

### 未完成 / 未验证结论

- **v107b cross-fitted POD-MoE reliability 是 completed negative probe / not promoted 状态**。`counter`, `flowers`, `garden`, `bonsai` 四个场景已完成且通过，同场景相对 clean baseline 仍明显正向，但相对 v106 三项均值均为负向。因此不能宣称 v107b 优于 v106。
- v107b 的意义是机制级验证：把 expert fitting 和 expert reliability scoring 分到 even/odd target-view split 上，降低同一证据自评估带来的 optimistic reliability。
- 本报告不把 v107b 写成升级结果，只写成 reliability stress test：更严格的 held-out expert reliability 单独引入后，在四个 probe scenes 上没有带来 endpoint quality upgrade。

## 2. 与基础 MeshSplatting 的区别

基础 MeshSplatting 在这里是 local clean checkpoint render：给定训练好的 mesh-splatting checkpoint，直接渲染目标视角；它本身没有显式的三角形级残差修复专家，也没有把训练 residual 重新投回 surface face 的机制。

SPCarNet 当前主线是在 MeshSplatting checkpoint 之上做 evidence-guided post-training repair：

| 层级 | 基础 MeshSplatting | SPCarNet / v106 |
|---|---|---|
| 表示 | 原始 mesh splats / checkpoint 属性 | 在三角形 surface 上附加 residual field |
| 修复信号 | 无显式 residual expert | 从训练 / diagnostic evidence 拟合 residual correction |
| 当前稳定 anchor | clean checkpoint | v104c shrink view-affine field |
| v106 新增 | 无 | base-preserve POD-MoE：detail + boundary experts |
| 选择口径 | clean `26000/30000` checkpoint 按 held-out test score 选 baseline | v106 与 v104c 在同一 full9 evaluator 下比较 |
| 几何压缩 | 由基础 checkpoint 决定 | v106 不声称新增三角形减少；几何压缩仍属于早期 Phase-J/compaction 线 |

需要对 mentor 说清楚的一点：v106 是 representation-level residual-field 改进，不是当前最强 RGB endpoint 的全部故事。README 中更大的 Phase-J RGB endpoint 增益和平均删面 `7.6479%` 属于 earlier render-time ELA / compaction portfolio 线；v106 不应借用那条线的几何压缩收益。

## 3. 方法概述

可以用下面的三层图讲：

```text
clean MeshSplatting render
        |
        v
v104c shrink view-affine residual field
        |
        v
v106 base-preserve POD-MoE
  = stable base residual
  + detail expert delta
  + occlusion-boundary expert delta
```

v104c 的 residual field 使用低阶 triangle-local 特征：

```text
[1, barycentric_u, barycentric_v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

v106 的核心不是替换 v104c，而是 **base-preserve**：

```text
adapted residual =
  base residual
  + weighted detail expert delta
  + weighted occlusion-boundary expert delta
```

这避免了早期 POD variants 可能压制稳定 base、损伤 PSNR 的问题。直观讲，v104c 先给一个稳的低频 / view-affine 修复，v106 只允许两个小专家在证据支持的位置补充残差。

## 4. 模块级实现

### 4.1 Base Field

角色：保留 v104c shrink view-affine residual field，作为稳定基底。

要点：

- 以 triangle-local barycentric 坐标和 view direction 为输入。
- shrink 逻辑抑制支持不足的 view-affine fit，避免过拟合。
- v106 的专家输出不是替代 base，而是在 base 上叠加。

### 4.2 Detail Expert

角色：处理单一低阶 field 容易平滑掉的高频 residual detail。

证据口径：

- 每个场景都有 detail-supported triangles 统计。
- full9 compare 中 v106 的 detail triangles 大约在 `142k-393k` 范围，具体见 `full9_compare.md`。

### 4.3 Occlusion-Boundary Expert

角色：处理 visibility / depth / triangle boundary 附近的结构化误差。边界错误和纹理细节错误不是同一种 residual，因此拆成独立 expert 更合理。

证据口径：

- 每个场景都有 boundary-supported triangles 统计。
- full9 compare 中 boundary triangles 大约在 `862k-1.80M` 范围。

### 4.4 Reliability / MSE Scale

角色：让 expert 只在证据支持的位置起作用，并对不可靠 delta 做 damp。

当前 v106 使用的 expert certificate：

```text
weighted_normal_equation_lambda_star
```

局限：v106 仍在同一 normal-equation evidence 上拟合 expert 并评估 reliability，因此 reliability 可能偏乐观。这正是 v107b 要修的点。

### 4.5 Renderer / Artifact Interface

v106 渲染侧关键字段：

```text
basis_type: affine_barycentric_viewdir_pod_mixture
field_variant: pod_moe
pod_base_keep_mode: base_preserving_boundary
```

报告中可以把它概括成：v106 没有改训练主循环去重训 MeshSplatting，而是加载一个 triangle-attached residual sidecar，在渲染时对 base checkpoint 做保守残差适配。

## 5. 关键实验结果

### 5.1 Full9 Mean

LPIPS 越低越好。

| method | scenes | PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | 0.000000 | 0.000000 | 0.000000 |
| v104c shrink view-affine | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 | +0.011709 | -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 | +0.011812 | -0.019185 |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +1.329628 | +0.034657 | -0.063316 |

解释：

- v106 是当前已完成、可汇报的 POD-MoE anchor。
- v101/v102 endpoint/reference 明显更强，但它是 endpoint/reference row，不应被写成 v106 已达到的结果。
- v106 的 paper value 主要在“稳定专家混合机制”而不是大数值提升。

### 5.2 v106 vs v104c Per Scene

| scene | v106 PSNR | dPSNR | v106 SSIM | dSSIM | v106 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.719175 | +0.001526 | 0.675086 | +0.000115 | 0.313405 | -0.000098 |
| flowers | 20.077723 | +0.001879 | 0.531240 | +0.000163 | 0.374393 | -0.000080 |
| garden | 25.790945 | +0.002851 | 0.799382 | +0.000119 | 0.174480 | -0.000104 |
| stump | 25.460457 | +0.001146 | 0.714661 | +0.000061 | 0.282135 | -0.000078 |
| treehill | 21.245092 | +0.001329 | 0.578518 | +0.000099 | 0.384177 | -0.000121 |
| room | 29.600351 | +0.002516 | 0.891889 | +0.000051 | 0.230616 | -0.000048 |
| counter | 27.499645 | +0.001577 | 0.867521 | +0.000102 | 0.238847 | -0.000139 |
| kitchen | 28.772043 | +0.001595 | 0.881652 | +0.000062 | 0.187815 | -0.000206 |
| bonsai | 30.316090 | +0.005213 | 0.907520 | +0.000154 | 0.230050 | -0.000136 |
| mean | 25.831280 | +0.002181 | 0.760830 | +0.000103 | 0.268435 | -0.000112 |

推荐 PPT 话术：

- “不是肉眼一眼能看出的巨大提升，而是所有场景方向一致的小幅改进。”
- “最清楚的意义是 v104c 的单一 field 可以被保守专家混合扩展，而不破坏稳定 base。”
- “相对 clean MeshSplatting 的优势是清楚的；相对我们更强的 v104c anchor，v106 是机制上更合理但数值非常克制的一步。”

### 5.3 v106 vs Local Clean MeshSplatting Per Scene

这张表只对应本地 selected full9 / same evaluator / selected clean baseline。它适合证明当前工作在本地协议下相对基础 clean MeshSplatting 有稳定正向结果，不应写成已经同口径超过 MeshSplatting paper 的所有官方 setting。

| scene | clean PSNR | v106 PSNR | dPSNR | clean SSIM | v106 SSIM | dSSIM | clean LPIPS | v106 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 23.301613 | 23.719175 | +0.417562 | 0.659867 | 0.675086 | +0.015219 | 0.332077 | 0.313405 | -0.018672 |
| bonsai | 28.895233 | 30.316090 | +1.420856 | 0.896400 | 0.907520 | +0.011120 | 0.259493 | 0.230050 | -0.029443 |
| counter | 26.751774 | 27.499645 | +0.747871 | 0.862055 | 0.867521 | +0.005466 | 0.252003 | 0.238847 | -0.013156 |
| flowers | 19.682257 | 20.077723 | +0.395466 | 0.511822 | 0.531240 | +0.019418 | 0.394563 | 0.374393 | -0.020170 |
| garden | 25.029211 | 25.790945 | +0.761734 | 0.780035 | 0.799382 | +0.019347 | 0.201314 | 0.174480 | -0.026834 |
| kitchen | 27.818552 | 28.772043 | +0.953491 | 0.876452 | 0.881652 | +0.005199 | 0.199186 | 0.187815 | -0.011371 |
| room | 28.747276 | 29.600351 | +0.853075 | 0.884843 | 0.891889 | +0.007046 | 0.249903 | 0.230616 | -0.019286 |
| stump | 25.205042 | 25.460457 | +0.255415 | 0.705166 | 0.714661 | +0.009495 | 0.294004 | 0.282135 | -0.011869 |
| treehill | 20.934181 | 21.245092 | +0.310911 | 0.564523 | 0.578518 | +0.013995 | 0.406044 | 0.384177 | -0.021867 |
| mean | 25.151682 | 25.831280 | +0.679598 | 0.749018 | 0.760830 | +0.011812 | 0.287621 | 0.268435 | -0.019185 |

### 5.4 MSE-Direction Diagnostic

v106 相对 v104c 的 residual delta 在多数 held-out test views 上降低 MSE：

| scene group | improved views | worse views | mean delta MSE |
|---|---:|---:|---:|
| bicycle | 25 / 25 | 0 | -0.00000149 |
| flowers | 22 / 22 | 0 | -0.00000408 |
| garden | 23 / 24 | 1 | -0.00000157 |
| stump | 16 / 16 | 0 | -0.00000080 |
| treehill | 16 / 18 | 2 | -0.00000212 |
| room | 38 / 39 | 1 | -0.00000059 |
| counter | 23 / 30 | 7 | -0.00000026 |
| kitchen | 30 / 35 | 5 | -0.00000050 |
| bonsai | 36 / 37 | 1 | -0.00000017 |

解释：MSE 方向是健康的，但 delta 很小，支持“保守稳定”而不是“视觉突破”的结论。

## 6. 定性证据路径

生成的 contact sheets 对比：

```text
GT | v104c baseline | v106 candidate | |v104c-GT| error | |v106-GT| error
```

建议用于 PPT 的图：

- `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png`
- `docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png`

辅助 manifest / JSON：

- `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.json`
- `docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.json`
- `docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.json`
- `docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.json`

使用建议：

- 先展示 full9 table，再展示 crop panel。
- 不要让 mentor 期待大面积 RGB 可视差异；v106 是 base-preserving，所以视觉变化本来就应该小。
- 重点看 error map 是否局部变淡，以及边界 / 纹理区域是否有轻微收敛。
- 如果只展示整图 RGB，很容易看不出区别；PPT 应优先使用 crop + error-map contact sheet，并在标题中标明该图展示的是 subtle repair rather than global style change。

## 7. 当前证据文件

v106 已完成证据：

- `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.csv`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.json`

selected source report roots recorded in `full9_assembled.md`:

- `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/...`
- `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/...`
- `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/...`

v107b completed negative probe 记录：

- `docs/car_model/6-25-v107-CrossFit-PODMoE-Probe-Log.md`

该日志已经记录 `counter`, `flowers`, `garden`, `bonsai` 四场景 completed negative probe、统一 v106 对比表和 field diagnostics。

v108 MSE-descent-locked POD-MoE 记录：

- `docs/car_model/6-25-v108-MSE-Descent-Locked-PODMoE-Log.md`
- crossfit-risk flowers probe 已完成：相对 v106 为 `-0.001858 PSNR / -0.000162 SSIM / +0.000081 LPIPS`，因此不能晋升。
- render-space audit 显示 v108 crossfit flowers 相对 v106 是 `0 / 22` views MSE 改善、`22 / 22` views MSE 变差，mean delta MSE `+0.00000403`。
- normal-equation gate-source ablation 已完成：相对 v106 为 `-0.001305 PSNR / -0.000114 SSIM / +0.000034 LPIPS`，render-space audit 同样是 `0 / 22` views 改善、`22 / 22` views 变差，因此失败不只是 crossfit 过度压制。

v109 Render-Realized Parent Gate 记录：

- `docs/car_model/6-25-v109-RenderRealizedParentGate-Log.md`
- v109 flowers feasibility 已完成：用 train render+GT 选择 policy，test 应用不读 test GT，记录 `no_target_gt_used_for_policy=True`。
- train calibration 选择 fallback-to-parent，`target_mean_mask=0.0`。
- final test metrics 与 v106 完全一致：`20.077723 PSNR / 0.531240 SSIM / 0.374393 LPIPS`。
- render-space audit 显示 v109 相对 v106 mean abs delta 为 `0.0`，即完全保留 parent。它是安全闭环，不是质量突破。

## 8. 口径边界与局限

可以安全说：

- v106 是真实 pipeline / render artifact 变化，不是文档改名。
- v106 full9 已完成，`9 / 9` selected scenes 对 v104c 三指标同向提升。
- v106 相对 clean MeshSplatting 的提升主要来自 v104c anchor，v106 在此基础上提供额外小幅专家收益。
- v106 的 MSE-direction diagnostic 大体健康，说明 residual correction 方向不是随机噪声。

不能说 / 不应暗示：

- 不能说 v106 是大幅 paper-level breakthrough。
- 不能说 v106 已追上 v101/v102 endpoint/reference。
- 不能把 Phase-J 的平均删面 `7.6479%` 或大幅 RGB endpoint 增益写成 v106 POD-MoE 自己带来的结果。
- 不能说 v107b 已经优于 v106；四场景 probe 已经完成且相对 v106 回退。
- 不能把当前 diagnostic pipeline 描述成纯 train-only unseen-camera generalization；v106 报告中明确提示当前仍依赖 target-camera sidecar / distilled evidence，汇报时要保守表述。

当前主要局限：

1. 数值收益小：平均 `+0.002181 PSNR` 量级，视觉变化也很 subtle。
2. Reliability 自评估偏乐观风险：v106 fitting 与 reliability scoring 使用同一 weighted normal-equation evidence。
3. 几何故事未由 v106 推进：POD-MoE 是 residual-field 改进，不是 compression / topology 结果。
4. Full9 是 local selected protocol：适合作为 mentor discussion 和方法迭代证据，还不是最终 paper table。
5. Evidence sidecar 口径要讲清楚：v104c/v106/v107 使用 target-camera delta distillation / sidecar evidence 来构建 residual field，不是 vanilla MeshSplatting checkpoint，也不是严格 train-only unseen-camera generalization。

## 9. 实现审查结论

多 subagent 只读审查后的结论：

- `run_v105_evidence_gated_mixture_scene.py` 的 v107b scene runner 已经把 field build、`render.py` surface-field endpoint、`evaluate_render_split_metrics.py` evaluator 串起来；方法不是只存在于文档里。
- `render.py` 的 `temperature_controlled` POD view gate 逻辑符合当前设计：当 `view_gate_temperature=0.0` 时不会乘 POD view gate；旧 v106 缺 `pod_view_gate_mode` 的 artifact 默认走 `implicit_unit_temperature` 兼容路径。
- v107b identity check 比 v106 更严格，会检查 `method_version=v107_crossfit_pod_moe_expert_reliability`、`gate_source=crossfit_risk`、`view_gate_temperature=0.0`、`pod_view_gate_mode=temperature_controlled` 和 field sha，因此能挡住明显的 v106 / pre-patch v107 混用。
- 仍有一个工程风险：identity 还不是完整 build fingerprint，没有记录 builder code hash；同参数、同 method version 但来自旧代码的 artifact 理论上仍可能复用。因此 v107b 晋升前必须检查 fresh root、`--force_field --force_render --force_eval` 日志、manifest 与 render report。
- 已新增轻量 CPU smoke：`scripts/car_model/smoke_test_pod_view_gate_modes.py`。验证结果为 `temperature_controlled_vgt0_gain=0.05000000`, `implicit_gain=0.00000000`, `legacy_missing_gain=0.00000000`，覆盖了 v107 `view_gate_temperature=0.0` 与旧 v106 缺字段兼容行为。
- 已增强 render report 审计字段：`render.py` 会把 `method_version`, `expert_reliability_variant`, `pod_expert_reliability_variant`, `expert_reliability_combine`, `expert_mse_certificate`, `pod_crossfit_split` 写入 `surface_residual_field` summary；未来 runner 会对 v107 crossfit 的 render-side `method_version` 做严格 identity check。

v107b 完成后必须检查：

| item | required value / check |
|---|---|
| field files | 每个 scene 有 fresh `.pt` 与 `.manifest.json` |
| report files | 每个 scene 有 `<scene>_v107_crossfit_pod_moe_report.json/md` |
| return codes | `v102`, `field`, `render`, `eval` 全为 `0` |
| field identity | `field_identity.checks` 全 true |
| render identity | `render_stats.identity_checks` 全 true，field sha 与 manifest 一致 |
| v107 semantics | `pod_view_gate_mode=temperature_controlled`, `pod_crossfit_split=target_view_even_odd`, reliability 是 held-out crossfit variant |
| fair comparison | 同场景显式比较 clean, v104c, v106, v107；不能只和 v104c 比 |
| stale bank risk | 若可行，手动核对 manifest 中 `source_delta_bank_sha256` 与当前 v102 delta bank |

## 10. 下一步：v107b Cross-Fitted Reliability

v107b 要回答的问题：

```text
POD-MoE experts 是否真的可靠，
还是只是对同一份 fitting evidence 的自评估过于乐观？
```

预期机制：

```text
1. 将 target views 拆成 even / odd split；
2. 在 split A 上拟合 detail / boundary experts；
3. 在 split B 上评分 reliability 与 MSE scale；
4. 反向再做一次；
5. 保持 render-time tensor format 尽量不变，用于公平 ablation。
```

v107b 当前状态必须这样写：

- status：`probe / pending / not promoted`
- initial scenes：`counter`, `flowers`, `garden`, `bonsai`
- gate source：`crossfit_risk`
- expected method version：`v107_crossfit_pod_moe_expert_reliability`
- report root：`/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports`
- field root：`/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_field`
- 当前四场景 probe 已经可填，mean 相对 v106 为 `-0.003646 PSNR`, `-0.000134 SSIM`, `+0.000104 LPIPS`，所以 v107b 不能晋升。

完成后需要收集：

- 每个 scene 的 report JSON / Markdown。
- field `.pt` 与 `.manifest.json`。
- `return_codes`：`v102`, `field`, `render`, `eval`。
- `field_identity.checks` 与 `render_stats.identity_checks`。
- `metrics`, `clean_metrics`, `v104c_metrics`, `deltas.vs_v104c`。
- 与 v106 同场景显式相减的 v107-v106 deltas。
- patch A/B 的验证：统计去重是否生效；`pod_view_gate_mode=temperature_controlled` 是否真正跳过 POD view gate。

晋升规则建议：

- 任一 scene 缺结果、失败或 identity check false：v107b 仍标为 incomplete。
- 如果 v107b 一项提升但另一项回退：直接写 mixed，不晋升。
- 如果 v107b 像当前四场景结果一样相对 v106 三项都回退：直接写 completed negative，不晋升。
- 即使四个 probe scenes 全部优于 v106，也只能写成 probe positive；还需要 remaining full9 场景后才能称为 full9 upgrade。

### 10.1 v107b 负结果后的方法诊断

v107b 的负结果不是简单的 GPU / evaluator / artifact 问题。`counter`, `flowers`, `garden`, `bonsai` 都通过并明显优于 clean MeshSplatting，但它们相对 v106 回退。这说明问题更可能来自机制本身：

- cross-fit reliability 要求 even->odd 和 odd->even 两个方向都支持，并使用保守 combine，导致 expert capacity 被压低。
- boundary expert 被压得尤其明显：`boundary_crossfit_gain_mean` 在 `flowers` / `garden` / `counter` / `bonsai` 约为 `0.000102` / `0.000084` / `0.000070` / `0.000078`，远低于 detail expert。
- v107b 评分的是 split-fit expert，但最终渲染用 full-data expert delta；证书没有直接约束“渲染时实际叠加的联合 expert residual”是否是 MSE descent。

因此下一步不能继续做场景参数扫描，而应做机制升级：**MSE-descent-locked POD-MoE**。核心想法是给 detail / boundary experts 加一个 joint descent certificate，让它们只有在对 teacher residual 的联合方向不增加 MSE 时才被保留或缩放。它的研究价值在于把“专家是否可靠”从单独 gate 判断，改成对最终 runtime correction 的联合下降约束。

### 10.2 v108 首个结果与当前判断

v108 crossfit-risk flowers probe 已经完成，identity checks 全部通过，说明这是有效实现结果而不是 artifact 混用。但它仍低于 v106：

| scene | PSNR | dPSNR vs v106 | dPSNR vs clean | SSIM | dSSIM vs v106 | LPIPS | dLPIPS vs v106 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flowers | 20.075865 | -0.001858 | +0.393608 | 0.531078 | -0.000162 | 0.374474 | +0.000081 |

更关键的是 render-space MSE audit：相对 v106，v108 crossfit 在 `22 / 22` held-out views 上 MSE 都更差，mean delta MSE `+0.00000403`。这说明当前 weighted-normal-equation descent lock 还没有闭合到最终 render-realized parent improvement。

当前最重要的并行实验是 v108 normal-equation ablation。若它优于 v106，则说明主要瓶颈是 crossfit gate 过度压制；若它仍低于 v106，则下一步应转为 **Parent-Preserving Render-Realized Descent Certificate**：以 v106 为 parent，在 train/calib render-space 证明候选 correction 不退化，再应用到 test，而不能使用 test GT 做 oracle selection。

### 10.3 v109 Parent Gate 安全闭环

v108 normal-equation flowers 也低于 v106，因此已经实现 v109 快速 feasibility：以 v106 为 parent、v108 normal-equation 为 candidate，在 train split 上用真实 render-space 指标校准 policy，在 test split 上只使用 `candidate-parent` 可观测差异生成 mask，不读 test GT。

flowers 结果：

| method | PSNR | SSIM | LPIPS | 说明 |
|---|---:|---:|---:|---|
| v106 parent | 20.077723 | 0.531240 | 0.374393 | 当前 flowers parent |
| v108 normal-equation | 20.076418 | 0.531125 | 0.374427 | proxy descent 但 render 退化 |
| v109 parent gate | 20.077723 | 0.531240 | 0.374393 | train-calibrated fallback，test 不读 GT |

v109 的关键价值是把坏候选挡住：它没有超过 v106，但把 v108 的负迁移变成“自动 no-op to parent”。这比继续报告 proxy descent 更诚实，也更接近论文级方法闭环：任何未来 candidate 必须在 train/calib render-space 通过 parent-preserving certificate，不能只依赖 coefficient-space proxy。

当前 v109 仍是 feasibility 级别。论文级版本需要：

1. builder 支持 train-even/train-odd candidate fitting；
2. calibration view 不参与 candidate fitting；
3. 多场景验证 fallback / accept 行为；
4. 找到在 v109 certificate 下真正非零 mask 且超过 v106 的 candidate。

## 11. 建议 PPT 结构

1. **Motivation**：基础 MeshSplatting 没有 triangle-level residual repair expert；SPCarNet 目标是在 checkpoint 后做 evidence-guided repair。
2. **Anchor**：v104c shrink view-affine field 是当前稳定 representation-field anchor，相对 clean 有明显 full9 提升。
3. **v106 Method**：base-preserve POD-MoE，保留 v104c base，再加 detail / occlusion-boundary experts。
4. **Implementation**：`field_variant=pod_moe`，`pod_base_keep_mode=base_preserving_boundary`，expert reliability 做 conservative weighting。
5. **Result**：v106 在 `9 / 9` scenes 相对 v104c 三指标小幅提升，full9 mean `25.831280 / 0.760830 / 0.268435`。
6. **Qualitative**：展示 garden 或 flowers contact sheet，强调 error map 细微改善。
7. **Honest Boundary**：不是大突破，不借用 Phase-J compression gain，不声称 v107b 已完成或已超越 v106。
8. **Next Step**：v107b cross-fitted reliability 当前是 stricter audit；四场景 negative 表明仅靠 held-out expert reliability 还不够。v108 MSE-descent-locked POD-MoE 已实现，但 crossfit-risk 与 normal-equation flowers 都相对 v106 负向。v109 parent gate 已完成 flowers feasibility，能无 test-GT policy 地挡住负迁移并保留 v106；下一步必须找到能通过 v109 证书且非零接受的真正正向 candidate。

## 12. Mentor 可能会问的问题

**Q1: 既然 v106 相比 v104c 只提升 `+0.002181 PSNR`，为什么还有研究价值？**

A: 价值不在大数值，而在机制验证：把单一 surface residual field 扩展为 base-preserving expert mixture 后，没有破坏原来的强 anchor，并在 full9 所有场景三指标同向改善。这证明 expert mixture 可以作为后续更强可靠性机制的接口。当前不能包装成最终突破，只能作为 representation-level stepping stone。

**Q2: 为什么定性图不明显？**

A: 因为 v106 被设计成保守 base-preserve，expert 只做小残差，不会产生大面积 RGB 改变。应该用 crop + error map 展示局部误差减少；如果想要更强视觉差异，需要后续 v107/v108 类可靠性或更强 expert capacity，而不是把 v106 说成视觉 breakthrough。

**Q3: 与基础 MeshSplatting baseline 是否已经全面超过？**

A: 在当前本地 selected full9 protocol 下，v106 相对 clean MeshSplatting baseline 的 PSNR/SSIM/LPIPS 三项在 `9 / 9` scenes 都是正向；但这不是论文官方表格同口径复现，也不应被说成已经超过所有 MeshSplatting paper settings。

**Q4: 这是不是调参而不是方法？**

A: v106 有明确表示层变化：`affine_barycentric_viewdir_pod_mixture` field、base-preserve residual composition、detail/boundary experts、weighted reliability certificate。问题是当前收益还小，所以它是有效的方法接口，但还不是足够强的最终论文主方法。

## 13. 一句话版本

> v106 的价值是把 v104c 单一 surface residual field 稳定扩展成 base-preserving expert mixture；full9 上所有场景相对 v104c 都有三指标小幅正向，但效果仍是 conservative diagnostic。v107b 用 cross-fitting 验证 expert reliability，四场景 probe 已显示相对 v106 回退，因此只能作为 reliability stress test / negative result 汇报。v108 的两条 flowers probe 都低于 v106，证明 proxy descent 不足以闭合到最终 render 指标。v109 已把方向转到 parent-preserving render-space certificate：当前能无 test-GT policy 地挡住坏候选并保留 v106，但还没有产生新的质量提升。
