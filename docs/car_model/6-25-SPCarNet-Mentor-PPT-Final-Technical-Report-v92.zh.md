# SPCarNet 当前方法完整技术报告（Mentor/PPT 母稿 v92）

日期：2026-06-25
用途：给 mentor 汇报当前 SPCarNet 工作现状，并作为 PPT 制作母稿。
推荐标题：

```text
SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting
```

本文只把已经有本地 evidence 闭合的结果作为主结论。仍未完成或未通过 promotion gate 的实验会放在“边界与下一步”，不包装成 headline。

---

## 1. 一页结论

SPCarNet 当前最稳定、最适合汇报的主方法是：

```text
trained MeshSplatting checkpoint
  + surface evidence cache
  + geometry-safe triangle compaction
  + guarded Evidence Lumigraph Adapter
  + train/policy-val risk gate and fallback
```

通俗讲法：

> MeshSplatting 是“训练完直接渲染”；SPCarNet 是“训练完以后让 checkpoint 自审计：低风险三角形可以删，稳定的表面颜色误差可以修，不确定区域自动回退”。

当前主 endpoint 是 **Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**。在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、强 clean MeshSplatting baseline 下：

| 指标 | 结果 |
|---|---:|
| scene-level PSNR/SSIM/LPIPS strict wins vs clean MeshSplatting | `9 / 9` |
| held-out view PSNR/SSIM/LPIPS strict wins vs clean MeshSplatting | `244 / 246` |
| mean PSNR | clean `25.1517` -> ours `26.4828` |
| mean SSIM | clean `0.7490` -> ours `0.7837` |
| mean LPIPS | clean `0.2876` -> ours `0.2243` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` removed |

这里的 `triangle reduction` 指**删去的三角形比例**，不是剩余比例。

当前最诚实的定位：

| 方面 | 状态 |
|---|---|
| 与本地 MeshSplatting baseline 对比 | 强：full9 `9 / 9` 场景 RGB 三指标严格胜出 |
| 与 MeshSplatting 论文表格口径 | 本地 clean30k 复现 `24.8002 / 0.7310 / 0.3072`，接近论文 `24.78 / 0.728 / 0.310` |
| 定性展示 | 有 full-frame 公平对比、local crop/error reduction、outdoor detail panels |
| 定性追溯 | 已有 manifest 绑定 scene/view/crop/metric/source-image paths |
| 几何收益 | 平均删除 `7.65%` triangles，属于质量优先的安全压缩 |
| Runtime | full9 render-only、isolated adapter postprocess、integrated render+adapter no-I/O 表已补：省显存/模型大小/triangles，但当前不提速，adapter 是主要瓶颈 |
| 表示级内化 | 接口和审计已经打通；v87/v88/v89b 都未通过 strict promotion gate |
| 论文闭环 | 还没 100% 完成：还需要真正 checkpoint-baked endpoint，或把 adapter 大幅加速到可部署口径；如果 claim 部署速度，还需 PNG/I/O/metrics 口径 |

---

## 2. 研究动机

MeshSplatting 的重要优势是显式 surface/triangle-aware 表示，但标准训练流程结束后，它不会主动回答这些问题：

1. 哪些 triangles 对多视角解释贡献低，可以安全删除？
2. 哪些 surface 区域在训练视角中反复出现稳定 residual，可以迁移到 held-out view？
3. 哪些修复只是在 train/policy-val view 上看起来好，到了 held-out view 会崩？
4. 如果证据不足，系统能否自动回退到 clean MeshSplatting，而不是硬改？

SPCarNet 的核心研究问题是：

```text
Can training-view surface evidence certify where a MeshSplatting checkpoint
can be compacted and where its appearance residuals can be safely repaired?
```

关键思想：

```text
training views are not only optimization signals;
they become auditable surface evidence for post-training repair and compaction.
```

---

## 3. 与基础 MeshSplatting 的区别

基础 MeshSplatting：

```text
training images + cameras
  -> optimize MeshSplatting checkpoint
  -> render held-out views
```

SPCarNet：

```text
training images + cameras
  -> optimize MeshSplatting checkpoint
  -> render train / policy-val views with surface maps
  -> build surface evidence cache
  -> certify low-risk geometry compaction
  -> transfer stable surface residuals
  -> train/policy-val gate and fallback
  -> render held-out views
```

| 维度 | MeshSplatting baseline | SPCarNet 当前主方法 |
|---|---|---|
| 基础模型 | trained MeshSplatting checkpoint | 继承同一个 checkpoint |
| 训练视角使用方式 | 主要隐式用于优化 loss | 显式保存为 surface evidence |
| 几何处理 | checkpoint geometry 固定 | 删除低风险 triangles |
| 外观修复 | 直接渲染 checkpoint | surface-bound guarded residual repair |
| 风险控制 | 依赖训练收敛 | policy-val gate、tail-risk、fallback/no-op |
| 输出目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可审计证据 |

一句适合 PPT 的话：

> 原方法是“训练好就交卷”；我们方法是“训练好以后再检查错题本，能安全删的几何删掉，稳定错的颜色修回来，不确定的地方不动”。

---

## 4. 方法总览

主流程：

```text
Clean MeshSplatting checkpoint
  |
  | render train / policy-val views with face id, depth, alpha, barycentric maps
  v
Surface Evidence Cache
  |
  | residual, visibility, support, risk, surface address
  v
Geometry-Safe Compaction
  |
  | remove low-risk triangles, protect thin/high-risk regions
  v
Guarded Evidence Lumigraph Adapter
  |
  | transfer stable residuals through surface correspondence
  v
Policy-Val Gate
  |
  | select branch/alpha, reject risky edits, fallback if needed
  v
Held-Out Rendering and Evaluation
```

核心原则：

| 原则 | 具体含义 |
|---|---|
| held-out GT 不参与策略选择 | branch、alpha、support、fallback 只由 train/policy-val evidence 决定 |
| residual 必须绑定 3D surface | 不是普通 2D filter，而是 face/bin/barycentric surface transfer |
| 证据不足自动回退 | 不强行修复 out-of-trajectory 或 support 稀疏区域 |
| 质量优先压缩 | 不追求极限 compression，而是在 RGB 不退化甚至提升时减少 triangles |

---

## 5. 模块细节

### 5.1 Surface Evidence Cache

Evidence cache 保存训练/策略验证视角中的局部证据：

| 证据 | 作用 |
|---|---|
| rendered RGB 与 GT RGB | 计算 residual 和 image-level risk |
| residual `GT - Render` | 外观修复信号 |
| face id / barycentric / surface bin | 把 2D residual 绑定到 3D surface |
| alpha / depth / visibility | 判断像素是否可靠可见 |
| per-face/per-bin support count | 防止稀疏区域过拟合 |
| residual sign consistency | 判断 residual 方向是否稳定 |
| train/policy-val PSNR/SSIM/LPIPS/L1 | 策略选择和风险审计 |
| per-view min gain / CVaR20 | 防止平均收益掩盖 tail-view 退化 |

它回答的问题是：

```text
这个 surface 区域是否被多视角稳定观测？
这个 residual 是否能迁移到 policy-val view？
这个修复是否会伤害最差视角？
```

### 5.2 Geometry-Safe Triangle Compaction

压缩不是按面积或透明度粗暴删 triangle，而是：

```text
remove a triangle only when multi-view evidence marks it as low-risk
```

被保护的区域包括：

- visibility 稀疏区域；
- thin structures；
- high residual faces；
- depth/edge/occlusion 不稳定区域；
- policy-val tail-risk 高的区域；
- 室内场景中 triangles 少、容易破坏细节的区域。

当前 Phase-J 平均删去 `7.6479%` triangles，同时 full9 RGB 三指标全场景严格胜出。这个结果支持 **quality-preserving compactness**，但不是极限压缩。

### 5.3 Guarded Evidence Lumigraph Adapter

当前大幅 RGB 收益主要来自 guarded Evidence Lumigraph Adapter。简化表达：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的渲染；
- `residual_i = GT_i - Render_i` 只来自 train/policy-val views；
- `u_i` 是 target pixel 通过 surface correspondence 找到的训练视角 surface address；
- `w_i(p)` 由 visibility、surface match、support count、risk 共同决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时自动 fallback/no-op。

为什么它不是普通图像后处理：

| 普通 2D 后处理 | SPCarNet Evidence Lumigraph Adapter |
|---|---|
| 在图像平面滤波或增强 | residual 绑定到 mesh surface |
| 可以对 held-out 图像直接操作 | 不能访问 held-out GT，只能迁移训练证据 |
| 没有几何一致性约束 | 需要 face/bin/surface correspondence |
| 通常没有策略验证 gate | 有 policy-val gate 和 fallback |

### 5.4 Policy-Val Gate 与 Fallback

当前 policy 的公平性边界：

- branch、alpha、support、fallback 只由 train/policy-val evidence 决定；
- held-out test GT 只用于最终评价；
- clean baseline 使用本地 clean `26000/30000` envelope 中 held-out 更强者；
- 不用 train metric 选择 clean baseline，因为 train metric 会天然偏向更久训练；
- 不用 test metric 为我们方法调参。

当前 Phase-J 分支：

| branch | scenes |
|---|---|
| adaptive ELA | `bicycle, flowers, garden, stump, room, counter, kitchen, bonsai` |
| edge fallback | `treehill` |

`treehill` 的 fallback 是一个安全性证据：当证据不稳定时，系统会保守，而不是强行修。

### 5.5 Representation-Level Residual Atlas

这条路线的目标是把 render-time repair 内化到 checkpoint：

```text
make the repair persistent in the representation,
not only an external render-time adapter.
```

当前已经实现并验证了多个接口：auto-support、capacity-aware policy、face-alpha reliability、bin-alpha policy、local patch prior、bin-gain hybrid、tail-risk certificate、L1-proxy bin-dominance gate。结论是：

- 接口、审计、fallback 机制成熟；
- 但是收益仍是 `1e-4` 到 `1e-6` 量级；
- v87/v88/v89b 都没有通过 strict promotion gate；
- 这条线可以作为 paper 后续突破方向，但不能替代当前 Phase-J headline。

---

## 6. 主定量结果：SPCarNet vs MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- baseline：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- 我们方法：Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR/SSIM 越高越好，LPIPS 越低越好。

### 6.1 Full9 Per-Scene Table

| scene | clean MeshSplatting PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | delta | triangle removed |
|---|---:|---:|---:|---:|
| bicycle | `23.3016` / `0.6599` / `0.3321` | `24.0215` / `0.7024` / `0.2661` | `+0.7199` / `+0.0425` / `-0.0660` | `11.81%` |
| flowers | `19.6823` / `0.5118` / `0.3946` | `20.3044` / `0.5578` / `0.3292` | `+0.6221` / `+0.0459` / `-0.0653` | `11.82%` |
| garden | `25.0292` / `0.7800` / `0.2013` | `26.3111` / `0.8278` / `0.1358` | `+1.2819` / `+0.0478` / `-0.0655` | `3.47%` |
| stump | `25.2050` / `0.7052` / `0.2940` | `25.5951` / `0.7241` / `0.2639` | `+0.3901` / `+0.0189` / `-0.0301` | `11.82%` |
| treehill | `20.9342` / `0.5645` / `0.4060` | `21.2962` / `0.5956` / `0.3363` | `+0.3620` / `+0.0311` / `-0.0697` | `11.81%` |
| room | `28.7473` / `0.8848` / `0.2499` | `30.3056` / `0.9057` / `0.1960` | `+1.5584` / `+0.0209` / `-0.0539` | `2.10%` |
| counter | `26.7518` / `0.8621` / `0.2520` | `28.4492` / `0.8937` / `0.1865` | `+1.6974` / `+0.0317` / `-0.0655` | `2.10%` |
| kitchen | `27.8186` / `0.8765` / `0.1992` | `30.1997` / `0.9161` / `0.1320` | `+2.3812` / `+0.0396` / `-0.0672` | `2.10%` |
| bonsai | `28.8952` / `0.8964` / `0.2595` | `31.8620` / `0.9303` / `0.1726` | `+2.9668` / `+0.0339` / `-0.0869` | `11.80%` |

### 6.2 Aggregate

| aggregate | value |
|---|---:|
| scene-level strict wins | `9 / 9` |
| per-view strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle removed | `7.6479%` |

解读：

- 室内场景 `bonsai/kitchen/counter/room` 的 PSNR 收益最大；
- 室外场景也全胜，但 full-frame 肉眼差异通常弱于室内，需要 crop/error map；
- `treehill` 走 edge fallback，说明策略在不稳定区域会自动保守；
- 当前压缩率是质量优先的安全压缩，不是 aggressive compression。

---

## 7. 与 MeshSplatting 论文表格的关系

我们刷新了 official clean30k 复现：

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| local official clean30k reproduction | `24.8002` | `0.7310` | `0.3072` |
| local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

严谨解释：

- 本地 clean30k 复现非常接近 MeshSplatting paper table，说明本地 evaluator/protocol 有可信度；
- Phase-J 数值高于 MeshSplatting paper table；
- 正式主 claim 仍应以本地同协议 selected-clean MeshSplatting baseline 为准，因为这是更公平也更强的 baseline；
- paper-table 可能存在 resolution、mask、split、preprocessing、metric implementation、checkpoint iteration 差异；
- 写论文前需要最终 official-style 统一口径复现和补充 rate/FPS/model-size。

安全口播：

> We outperform our local same-protocol MeshSplatting baseline by a large margin. Paper-table comparison is encouraging, and our clean30k reproduction is close to the paper table, but the final fairness claim should use same-protocol local baselines.

---

## 8. Official-Style Compact-ELA 支撑表

除了最强 Phase-J headline，我们还整理了一个更接近 paper protocol 的 Compact-ELA 支撑表：

| item | value |
|---|---:|
| available scenes | `9 / 9` |
| strict all-axis pass | `5 / 9` |
| RGB + compact + geometry-safe pass | `9 / 9` |
| RGB + compact pass | `9 / 9` |
| mean dPSNR vs selected clean | `+0.497941` |
| mean dSSIM vs selected clean | `+0.015755` |
| mean dLPIPS vs selected clean | `-0.023373` |
| mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| mean triangle reduction | `5.7632%` removed |

作用：

- 证明不是只有一个 aggressive Phase-J 表能赢；
- 证明更保守的 Compact-ELA 在 official-style full9 下也做到 `9 / 9` RGB + compact pass；
- 但它不是最强 headline，因为 `5 / 9` strict all-axis pass 表示部分 geometry metrics 只是 safe/tie，不是三轴严格提升。

---

## 9. Runtime / Rate / Geometry 结果

### 9.1 Static Rate Profile

| item | value |
|---|---:|
| scenes | `9` |
| mean Compact-ELA triangle reduction | `5.7632%` |
| mean Phase-J triangle reduction | `7.6479%` |
| mean Compact-ELA dPSNR / dSSIM / dLPIPS | `+0.497941 / +0.015755 / -0.023373` |
| mean Phase-J dPSNR / dSSIM / dLPIPS | `+1.331084 / +0.034702 / -0.063359` |

### 9.2 Full9 Render-Only Runtime

测试范围：renderer forward only，不保存 PNG，不跑 `metrics.py`，不包含 LPIPS、disk I/O，也不包含 Phase-J render-time ELA post-process。

| item | value |
|---|---:|
| scenes | `9` |
| mean clean FPS | `31.739752` |
| mean compact FPS | `30.075865` |
| mean FPS ratio | `0.946023` |
| FPS win scenes | `0 / 9` |
| mean peak allocated reduction | `2.5733%` |
| peak allocated reduction scenes | `9 / 9` |
| mean checkpoint-byte reduction | `4.6753%` |
| checkpoint-byte reduction scenes | `9 / 9` |
| mean triangle reduction | `7.6479%` |

诚实结论：

- compact checkpoints 在 `9 / 9` 场景减少 triangles、checkpoint bytes、CUDA peak allocated memory；
- 但当前 render-only FPS 在 `9 / 9` 场景都比 clean 慢；
- 所以目前可以 claim memory/size/triangle reduction，不能 claim speedup；
- integrated render+adapter no-I/O runtime 已补，但仍证明当前 adapter 口径太慢。

### 9.3 Phase-J Adapter Postprocess Runtime

新增 isolated adapter postprocess profiler：

```text
scripts/car_model/benchmark_ela_postprocess_runtime.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/per_scene.csv
```

测试范围：只计 `utils.evidence_lumigraph_adapter.adapt_frame`，不写 PNG，不跑 renderer，不跑 metrics/LPIPS，不做 policy calibration。

| item | value |
|---|---:|
| scenes | `9` |
| target views | `246` |
| repeats per scene | `2` |
| weighted adapter ms/view | `1061.298183` |
| weighted adapter FPS | `0.942242` |
| max CUDA peak allocated | `4437.766 MiB` |
| render-only compact ms/view | `34.092124` |
| approximate render + adapter ms/view | `1095.390307` |
| approximate render + adapter FPS | `0.912917` |
| adapter/render time ratio | `31.130304x` |

诚实结论：

- Phase-J 当前 RGB 收益很强，但 render-time adapter postprocess 非常慢；
- 这个结果关闭了“adapter postprocess 未测”的证据缺口；
- 但它也明确说明当前方法不能 claim speedup；
- 如果要走部署或顶会系统效率故事，下一步必须把 repair bake into checkpoint，或者重写 adapter/kernel 以减少 per-view warping 成本。

### 9.3b Phase-J Integrated Render + Adapter Runtime

新增 integrated runtime profiler：

```text
scripts/car_model/benchmark_phasej_integrated_runtime.py
scripts/car_model/summarize_phasej_runtime_profiles.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/per_scene.csv
```

测试范围：同一个进程里测 canonical renderer forward + Phase-J `adapt_frame`；不写 PNG，不跑 `metrics.py`，不跑 LPIPS，不做 policy calibration。v2 runner 额外检查 Scene/evidence frame name 对齐，并复用 support `FrameLoader` cache，避免 profiler 自己把 adapter 时间夸大。

| item | value |
|---|---:|
| scenes | `9` |
| target views | `246` |
| repeats per scene | `2` |
| weighted integrated ms/view | `951.410896` |
| weighted integrated FPS | `1.051071` |
| weighted render ms/view | `37.090434` |
| weighted adapter ms/view | `913.855245` |
| adapter/render time ratio | `24.638570x` |
| max CUDA peak allocated | `17703.596 MiB` |
| integrated/render-only compact ms ratio | `27.044247x` |

新的结论：

- “没有 integrated runtime 证据”这个缺口已经关闭；
- 但 integrated 结果确认当前 Phase-J 不是速度方法：no-I/O 口径下仍约比 compact render-only 慢 `27x`；
- 这会影响论文定位：当前强项是 evidence-certified quality repair + compaction，而不是 real-time/deployment acceleration。

---

## 10. 定性结果与 PPT 展示建议

### 10.1 最推荐主图：local held-out error reduction

推荐放在主结果页：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

<img src="../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J local held-out error reduction">

每行含义：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

建议口播：

> Full-frame 上差异有时不强，但 error map 和 crop 能显示 SPCarNet 在 surface residual、局部高频纹理和遮挡边界上降低了 clean MeshSplatting 的系统误差。

### 10.2 公平 full-frame 图

推荐作为公平性证明或 appendix：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

<img src="../../assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame held-out comparison against clean MeshSplatting">

每行含义：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

### 10.3 室外细节图

推荐用于回应“室外场景视觉收益不明显”的问题：

```text
assets/spcarnet_m360_outdoor_detail_showcase.png
```

<img src="../../assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor detail error reduction showcase">

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

### 10.4 定性追溯

```text
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.json
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.csv
```

| item | value |
|---|---:|
| panels traced | `3` |
| examples traced | `16` |
| figures existing | `3 / 3` |
| source image path check | `all true` |

---

## 11. 消融与负结果：为什么不是简单调参

### 11.1 v64 fixed auto bin-alpha policy

v64 是当前最稳的 fixed representation-level reference：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | `9` | `1` | `9` | `+0.000410080` | `+0.000000278` | `-0.000018951` |
| v64 vs v52 | `9` | `2` | `9` | `+0.000706779` | `+0.000001563` | `-0.000038614` |
| v64 vs no-op | `9` | `7` | `8` | `+0.002255970` | `+0.000038081` | `-0.000093445` |

结论：自动 policy 和 checkpoint-level atlas 接口有效，但收益很小。

### 11.2 v82b/v83/v84/v85/v86/v87/v88/v89b diagnostics

| probe | scope | metric snapshot | verdict |
|---|---|---:|---|
| v56/v64/v79 anchor | `counter` | `26.756130219 / 0.862126231 / 0.251691371` | strong representation anchor |
| v82b capacity-prerank + face-alpha | `counter` | `26.756137848 / 0.862126350 / 0.251690656` | counter micro-win |
| v83 patchmix + face-alpha + local-patch | `counter` | `26.756147385 / 0.862125337 / 0.251688808` | PSNR/LPIPS up, SSIM down |
| v84 strict selector | full9 | vs v64 mean `+8.48e-7 / +1.3e-8 / -7.9e-8` | report-only |
| v85 target-footprint tail-risk | `counter` | `26.756134033 / 0.862126231 / 0.251691371` | accepted edit, below v84 counter |
| v86 anchor-preserving selector | full9 | `9 / 9` non-regressive/tie vs v84 | guardrail, not promoted |
| v87 source mixture | `counter` | `26.756130219 / 0.862126231 / 0.251691371` | below v84/v86 anchor |
| v88 anchor-dominance tail-risk | `counter` | `26.756156921 / 0.862125456 / 0.251688033` | PSNR/LPIPS up, SSIM down; not promoted |
| v89b L1-proxy bin-dominance | `counter` | `26.756139755 / 0.862126350 / 0.251690716` | tiny PSNR signal, LPIPS fails gate |

解释：

- 这些不是主结果，而是 representation-baked 方向的严格筛选；
- policy-val 上通过的候选，如果 held-out 三指标不同时超过 anchor，就不晋级；
- v88 尤其重要：它证明有 PSNR/LPIPS 信号，但 SSIM 仍卡住，说明当前 certificate 还不够强。

这对 PPT 的价值：

> 我们不是在玩参数游戏；相反，我们用 strict promotion gate 拒绝了大量看起来有局部收益但不够稳的候选。

---

## 12. Stage2 Shape Prior 支线

Stage2 v4 normal-band autodecoder 是另一个真实训练管线改动。它在 clean shape field autodecoder 中加入 surface-normal band supervision：

```text
x_inner = x_surface - epsilon * normal -> occupied
x_outer = x_surface + epsilon * normal -> free
```

Full-val MAP-fit 当前最好证据：

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | delta |
|---|---:|---:|---:|
| extraction | `206 / 206` | `206 / 206` | tie |
| recon chamfer | `0.0698447353` | `0.0607328202` | `-0.0091119151` |
| hidden chamfer | `0.1023846301` | `0.0933915632` | `-0.0089930669` |
| filled IoU | `0.5531548112` | `0.5683319216` | `+0.0151771104` |
| shell IoU | `0.9112784961` | `0.8783071888` | `-0.0329713073` |

结论：

- v4 是真实 method improvement；
- epoch50 比 final checkpoint 更好，说明长训不是单调更优；
- 但它仍未通过原始 Stage2 quality gate，不能作为当前 headline；
- 最适合在 PPT 中作为“未来把 SPCarNet 推向 object-level shape prior 的支线”。

---

## 13. 为什么这是研究工作，而不是工程后处理

可以强调四点：

| 点 | 说明 |
|---|---|
| Surface-addressed evidence | residual 绑定到 mesh face/bin/surface address，不是 2D 图像滤波 |
| Certified edit policy | 每个删面或修复都依赖 support、visibility、risk、policy-val gate 和 fallback |
| Quality + compactness joint objective | 同时优化 RGB 指标和 triangle count，不是只追求 PSNR |
| Honest failure handling | v75-v89 中大量 negative diagnostics 被记录并拒绝晋升 |

一个好的论文故事：

```text
MeshSplatting gives us an explicit surface.
SPCarNet asks whether that surface can audit itself.
Training views are not only supervision; they are evidence.
Evidence tells us what can be compacted, repaired, or safely left untouched.
```

---

## 14. 当前弱点与风险

| 风险 | 说明 | 建议说法 |
|---|---|---|
| 主 RGB 收益来自 render-time adapter | Phase-J 不是 fully checkpoint-baked endpoint | 这是当前最强结果，表示级内化是下一阶段 |
| representation-level atlas 收益小 | v64-v89b 只有微小提升或未过 gate | 证明机制闭合，但还没突破表达能力 |
| 室外 full-frame 肉眼差异弱 | 局部高频区域更明显 | 主图用 crop/error map，full-frame 放公平性页 |
| triangle reduction 不高 | 平均 `7.65%`，质量优先 | claim 是 quality-preserving compactness，不是极限压缩 |
| runtime 不能 claim speedup | compact 省显存/大小，但 render-only FPS 更低；integrated render+adapter no-I/O 为 `951.411 ms/view`、`1.051 FPS` | 后续做 checkpoint-baked repair 或 kernel-level adapter 加速 |
| paper table protocol 未完全统一 | paper number 和本地 evaluator 可能不同 | 主 claim 用本地 same-protocol baseline |
| `/data` 磁盘接近满 | 当前新实验主要在 `/dev/shm` | 只归档小型 JSON/log/MD，避免误拷大文件 |

最安全的一句话：

> 当前主结果已经能支持“evidence-certified repair and compaction”的强故事，但还不能声称我们已经完成最终 paper-level checkpoint-baked representation endpoint。

---

## 15. 建议 PPT 结构

| slide | 标题 | 核心内容 | 建议图/表 |
|---:|---|---|---|
| 1 | Motivation | MeshSplatting 有显式 surface，但 checkpoint 不会自审计 | clean vs GT residual crop |
| 2 | Key Idea | training views are evidence, not only supervision | one-sentence idea |
| 3 | Pipeline | evidence cache -> compact -> repair -> gate -> render | method schematic |
| 4 | Surface Evidence | residual / support / visibility / risk | evidence table |
| 5 | Geometry Compaction | low-risk triangle removal | triangle reduction table |
| 6 | Guarded Repair | surface-bound residual transfer | ELA equation/diagram |
| 7 | Fair Protocol | branch selection 只用 train/policy-val | protocol diagram |
| 8 | Main Quantitative Result | full9 `9/9` scene wins, `244/246` view wins | aggregate table |
| 9 | Per-Scene Result | all scenes improve RGB metrics | 9-scene table |
| 10 | Qualitative Result | local error reduction | `spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 11 | Outdoor Analysis | full-frame subtle, crop/error clearer | outdoor detail panel |
| 12 | Ablation | v64-v89 representation-level diagnostics | compact ablation table |
| 13 | Runtime/Rate | smaller memory/bytes/triangles, integrated adapter bottleneck | runtime + RD frontier table |
| 14 | Boundary and Next Step | bake repair into representation or accelerate adapter | roadmap |

---

## 16. Mentor 可能追问的问题

### Q1. 你们是不是只是调参赢 baseline？

不是。当前 Phase-J 的关键不是每场景手工调参，而是固定的 evidence-driven 审计流程：surface evidence cache、geometry-safe compaction、guarded residual transfer、policy-val gate 和 fallback。需要承认的是，representation-level v64-v89 还在强化自动策略，不能把未闭合的部分包装成终局。

### Q2. 为什么 clean baseline 选择 `26000/30000` envelope？

因为只用 train 指标会偏向训练更久的 checkpoint。这里 clean baseline 用 held-out score 更强者，目的是让 baseline 尽可能强，避免和弱 clean checkpoint 比。

### Q3. residual repair 会不会在 out-of-trajectory 视角崩？

风险存在，所以方法引入 surface support、visibility、policy-val gate、tail-risk、fallback/no-op。证据不足时不修。当前 `treehill` 走 fallback，是方法保守性的一个例子。

### Q4. 为什么定性图有时看起来差异不大？

很多收益来自局部 residual-level correction，全图缩小后肉眼不总明显。展示时应该用 full-frame 图证明公平性，用 crop/error map 展示真实视觉收益。

### Q5. 为什么表示级 residual atlas 还不能作为主结果？

因为它目前只带来微小收益。它的价值是证明接口、selector、tail-risk 和 fallback 机制已经闭合；真正论文级下一步是把 Phase-J 的强 repair 效果内化到 checkpoint 表示里。

### Q6. 是否已经全面超过 MeshSplatting？

在本地 Mip-NeRF360 full9、同 evaluator、selected-clean MeshSplatting baseline 下，Phase-J 已经在 RGB 三指标和 triangle count 上形成强结果。还不能说在所有协议、所有数据集、runtime/FPS/model-size 上全面超过。

### Q7. 当前最需要 mentor 指导什么？

最值得讨论的是下一步取舍：是把工作定位为 evidence-certified post-training repair/compaction，快速补齐 appendix 和部署边界；还是继续投入 representation-baked atlas / adapter acceleration，追求更干净但风险更高的终局方法。

---

## 17. 60 秒中文口播

> 我们现在的方法不是推翻 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 后面加一层自审计机制。训练视角不只是用来优化 loss，而是被保存成 surface evidence：包括 residual、可见性、face/bin support 和风险统计。这样模型可以判断哪些三角形低风险可以删，哪些表面区域有稳定 residual 可以迁移修复，哪些区域证据不足必须回退。在本地 Mip-NeRF360 full9、相同 evaluator、selected-clean MeshSplatting baseline 下，当前 Phase-J endpoint 做到 9 个场景 PSNR/SSIM/LPIPS 全部严格胜出，246 个 held-out views 中 244 个严格胜出，平均 PSNR 提升 1.33 dB，SSIM 提升 0.0347，LPIPS 降低 0.0634，同时平均删去 7.65% triangles。需要诚实说明的是，当前最大 RGB 收益仍来自 guarded render-time adapter；我们已经打通表示级 residual atlas 和多个 certificate，但它们目前还是小收益诊断，下一步目标是把这套修复真正内化到 checkpoint 表示里。

## 18. 60 秒英文口播

> Our current method starts from a trained MeshSplatting checkpoint. Instead of directly rendering it, we build a surface evidence cache from train and policy-val views: residuals, visibility, face/bin support, and risk statistics. This evidence lets the checkpoint audit itself. Low-risk triangles are compacted; stable surface residuals are transferred to held-out views through surface correspondence; uncertain regions automatically fall back to the clean checkpoint. On local Mip-NeRF360 full9, under the same evaluator and selected-clean MeshSplatting baseline, the current Phase-J endpoint wins all 9 scenes on PSNR, SSIM, and LPIPS, wins 244 out of 246 held-out views, improves mean PSNR by 1.33 dB, improves SSIM by 0.0347, reduces LPIPS by 0.0634, and removes 7.65% triangles on average. The honest limitation is that the strongest RGB gain is still a guarded render-time adapter. We have built the representation-level residual atlas line and several certificates, but those are currently small-gain diagnostics. The next paper-level milestone is to bake the repair into the checkpoint while preserving the same evidence-certified behavior.

---

## 19. 关键证据路径

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv
```

官方口径桥接：

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.json
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.csv
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean_report.md
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean.json
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean.csv
```

静态 rate/profile：

```text
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.md
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.json
outputs/carnet/spcarnet/static_rate_profile_20260625/per_scene.csv
```

Render-only runtime profiling：

```text
scripts/car_model/benchmark_render_runtime_profile.py
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/per_scene.csv
```

Adapter postprocess runtime profiling：

```text
scripts/car_model/benchmark_ela_postprocess_runtime.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/per_scene.csv
```

Integrated render+adapter no-I/O runtime profiling：

```text
scripts/car_model/benchmark_phasej_integrated_runtime.py
scripts/car_model/summarize_phasej_runtime_profiles.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/per_scene.csv
```

Rate/frontier closure：

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/evidence_manifest_delta_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/v90_v91_process_result_audit.md
```

定性图与追溯：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.json
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.csv
```

表示级路线：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v87_source_mixture_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v89b_l1proxy_counter_20260625/results.json
docs/car_model/6-25-v87-v88-RepresentationCounterDiagnostics.md
docs/car_model/6-25-v89-L1ProxyBinDominanceGate-Implementation-And-v85Relax-Diagnostic.md
```

Stage2 shape-prior：

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
```

---

## 20. 当前最终状态

```text
Phase-J local full9 RGB + triangle-count result: strong and presentable.
Same-protocol selected-clean MeshSplatting comparison: locally complete.
Official clean30k reproduction: refreshed and close to MeshSplatting paper table.
Official-style Compact-ELA support table: refreshed and complete.
Qualitative panels: available and slide-ready with manifest traceability.
Runtime: render-only, isolated adapter, and integrated render+adapter no-I/O full9 tables exist; memory/size positive, FPS negative, adapter is the main speed bottleneck.
Representation-level baked endpoint: not complete; v87/v88/v89b not promoted.
Stage2 shape-prior: implemented and improved over v3, but gate soft-fail.
```

建议给 mentor 的最终定位：

> SPCarNet 当前最可信的故事是“MeshSplatting checkpoint 的 evidence-certified post-training repair and compaction”。它已经在本地 full9 strong clean baseline 上拿到强 RGB + triangle reduction 结果，并补上了 render-only memory/size profiling、isolated adapter postprocess profiling 和 integrated render+adapter no-I/O profiling；真正的论文终局还需要把 guarded repair 更彻底地 bake into representation，或者把 adapter 工程化到可部署速度。

---

## 21. 下一步建议

短期最应该做：

1. 做 checkpoint-baked representation candidate 或 adapter kernel acceleration；checkpoint-baked candidate 必须先在 counter 上三指标严格超过 v84/v86 anchor，否则不扩展。
2. 扩展 rate-distortion curve：当前已有 selected-clean/Compact-ELA/Phase-J frontier；下一步需要更多 compression targets。
3. 如果要 claim deployment speed，再补包含 PNG/I/O/metrics 的部署口径 runtime；否则只把当前 integrated no-I/O 作为瓶颈证据。
4. 为 PPT 准备一张更干净的 method schematic：不要放版本号，只画 evidence -> compaction -> repair -> gate。
5. 把 full-frame、公平 crop、error map 三类定性图分开展示，避免 mentor 只看全图觉得差异不明显。

不建议继续做：

- 继续用 per-scene hand-tuned 参数追逐微小收益；
- 把 v87/v88/v89b 的 `1e-6` 级别 representation 结果放进主结论；
- claim FPS speedup；
- claim 已经完全 paper-closed。
