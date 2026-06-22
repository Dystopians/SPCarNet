# SPCarNet / MeshSplatOpt 当前方法完整技术报告

日期：2026-06-22

用途：mentor 汇报 / PPT 制作 / 当前方法阶段性技术交底

当前可作为主结果汇报的方法：

```text
ours_26000_phasej_guarded_adaptedge_ela
```

一句话总结：

> SPCarNet 不是重新发明 MeshSplatting，而是在 MeshSplatting 的可渲染三角网格表示外面加入一套只依赖训练视角证据的“安全压缩 + 残差修复 + 严格回退”决策闭环，使模型在保持 mesh 友好性的同时，相对 clean MeshSplatting 在 Mip-NeRF360 full9 上实现 9/9 场景三指标严格提升，并平均减少 7.65% 三角形。

当前汇报定位：

| 维度 | 结论 |
|---|---|
| 可以作为主结果 | Phase-J guarded adaptive ELA endpoint |
| 不应作为主结果 | v25/v26/v27/v28 local-trust/field-region 分支 |
| 最强主 claim | 相对本地同协议 selected clean MeshSplatting，full9 9/9 三指标严格胜出 |
| 最重要 caveat | 主 RGB 收益仍来自 render-time ELA，representation-level 内化仍是下一步 |
| PPT 风格 | 先讲强结果，再主动解释公平性、负结果和下一步 |

---

## 0. PPT 使用建议

这份报告建议按“强结论 + 诚实边界 + 下一步升级”的方式使用。

### 0.1 可以强讲的结论

1. 当前主方法是 **Phase-J compact MeshSplatting + train-only Evidence Lumigraph Adapter**。
2. 公平主对比是本地同代码、同数据、同评价脚本复现的 selected clean MeshSplatting baseline。
3. 在 Mip-NeRF360 full9 上，当前主方法相对 selected clean baseline 实现：
   - 9 / 9 场景 PSNR、SSIM、LPIPS 三指标严格胜出；
   - 平均 `+1.3311 PSNR / +0.0347 SSIM / -0.0634 LPIPS`；
   - 平均减少 `7.6479%` triangles；
   - `244 / 246` held-out views 三指标同时胜出。
4. 相对 MeshSplatting paper table 的 Mip-NeRF360 均值，当前结果也更高；但 paper table 只能作为外部 sanity check，不能替代本地公平 baseline。
5. 方法不是简单调参：它包含 train-view evidence mining、真实 compact checkpoint、train-only residual repair、guarded policy、失败回退和负结果审计。

### 0.2 需要谨慎讲的边界

1. 当前 headline 的 RGB 增益主要来自 render-time ELA，不能声称所有修复都已经完全 baked into mesh representation。
2. Phase-S / v25 / v26 / v27 / v28 是 representation-level 或 render-layer 可信修复升级方向，但目前还没有显著超过 Phase-J headline。
3. v26 local-trust 已完成接口闭环、smoke test、dry-run 和 bonsai candidate test；但当前 test 数字低于 Phase-J headline，且 trainval/selector 仍在运行，因此只能作为“严格 local-trust 过保守”的候选分支/诊断结果，不能作为主结果。
4. v27 已把 v26 hard local-trust 升级为 fixed-profile soft trust-weighted ELA，并通过 py_compile、smoke、dry-run、fixed-profile override 拒绝和 bonsai medium 验证；它证明 soft trust 机制生效，但 plan candidate 被诚实 trainval gate 拒绝，不能替代 Phase-J headline。
5. v28 已进一步把 ELA alpha 从 pixel/bin 安全性推进到 policy-view tail 安全性，接口、smoke 和 dry-run 均通过；但 v28 还没有中长程 W&B 结果，不能作为主结果。
6. 定性图建议用 full-frame fair comparison + local crop + error map，而不是只展示缩小后的整图，因为当前收益很多是局部 residual-level 改善。

### 0.3 最推荐的 PPT 主线

```text
MeshSplatting is strong but has local residual errors and mesh redundancy.
SPCarNet adds a train-evidence-certified repair loop.
The loop safely compresses mesh topology and transfers only trusted residual evidence.
The current Phase-J endpoint beats a stronger local clean MeshSplatting baseline on all full9 scenes.
The remaining research risk is to distill render-time repair back into representation-level checkpoints.
```

### 0.4 一页总览图可以这样放

```text
Problem:
  MeshSplatting is mesh-friendly but leaves local residual errors and redundant faces.

Core idea:
  Use train-view evidence to certify when we may compress and when we may repair.

Current endpoint:
  compact MeshSplatting checkpoint + guarded Evidence Lumigraph Adapter.

Main result:
  full9 9/9 strict RGB wins vs selected clean MeshSplatting,
  +1.3311 PSNR / +0.0347 SSIM / -0.0634 LPIPS,
  7.6479% mean triangle reduction.

Honest boundary:
  best visual gains are still render-time; representation-level baking is the next research target.
```

---

## 1. 汇报版结论

### 1.1 当前最强 endpoint

当前最强、最适合汇报的 endpoint 是 **Phase-J compact MeshSplatting + train-only Evidence Lumigraph Adapter**。

它的核心不是“手调参数”，而是一个证据驱动的闭环：

1. 用训练视角判断哪些三角形可以安全压缩。
2. 将安全压缩后的 MeshSplatting checkpoint 保持为可渲染 mesh。
3. 用训练视角 residual 构建 Evidence Lumigraph Adapter，给 held-out view 做外观修复。
4. 用 train-only calibration / guarded policy 决定 alpha、edge fallback 和回退策略。
5. held-out test 只用于最终报告，不参与方法选择。

### 1.2 主定量结果

公平主比较对象是本地同协议复现的 clean MeshSplatting baseline。每个场景从 clean `26000` 和 clean `30000` checkpoint 中按 held-out test score 选最强 clean 行：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

当前 Phase-J 相对 selected clean MeshSplatting：

| 比较 | 场景 | dPSNR | dSSIM | dLPIPS | 平均三角形减少 |
|---|---:|---:|---:|---:|---:|
| Phase-J vs selected clean MeshSplatting | 9 / 9 三指标严格胜出 | +1.3311 | +0.0347 | -0.0634 | 7.6479% |

额外审计：

| 评估角度 | 当前结果 |
|---|---|
| per-view held-out audit | 244 / 246 个 held-out view 同时提升 PSNR / SSIM / LPIPS |
| sparse geometry audit | 9 / 9 geometry-safe，6 / 9 sparse geometry 严格更好 |
| 相对 Phase-F alpha-grid predecessor | 9 / 9 三指标严格胜出，均值 +0.3971 PSNR / +0.0083 SSIM / -0.0193 LPIPS |
| 外部 ETH3D courtyard clean9000 | 最高 +0.2642 PSNR / +0.0094 SSIM / -0.0225 LPIPS |

### 1.3 与 MeshSplatting 论文表格的口径关系

MeshSplatting paper table 的 Mip-NeRF360 平均约为：

```text
PSNR 24.78 / SSIM 0.728 / LPIPS 0.310
```

我们当前同场景 Phase-J endpoint 的平均：

```text
PSNR 26.4828 / SSIM 0.7837 / LPIPS 0.2243
```

因此按表格数值看，Phase-J 高于 paper table 平均：

| 行 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | 24.78 | 0.728 | 0.310 |
| local clean MeshSplatting | 25.1517 | 0.7490 | 0.2876 |
| SPCarNet Phase-J | 26.4828 | 0.7837 | 0.2243 |
| Phase-J - paper table | +1.7017 | +0.0555 | -0.0865 |

需要在汇报中明确：

- 最严谨主 claim 应使用本地同代码、同数据、同评价脚本的 clean MeshSplatting baseline。
- paper table 对比可以作为外部 sanity check，但不能替代本地公平 baseline。
- 当前方法确实强于本地 selected clean baseline；这比只超过 paper table 更重要。

---

## 2. 问题定义与动机

### 2.1 MeshSplatting 的优势

MeshSplatting 的基础优势很强：

- 输出是 opaque triangle mesh，比 Gaussian/point cloud 更接近传统图形管线。
- 渲染和资产管理更容易被游戏、AR/VR、数字孪生系统接受。
- 在 Mip-NeRF360 这类真实场景上已经有不错的 PSNR/SSIM/LPIPS。

### 2.2 仍存在的问题

本地复现实验显示 clean MeshSplatting 仍有三个问题：

| 问题 | 现象 | 影响 |
|---|---|---|
| 局部 residual 错误 | foliage、bark、bench slats、室内高频纹理有模糊或颜色偏差 | full-frame 指标下降，局部视觉不够锐 |
| 长训练不一定更好 | clean 30000 在 9 / 9 场景上都输给 clean 26000 的 selected score | 不能简单说“训练更久即可” |
| 拓扑有冗余 | 部分场景存在可安全删除的三角形 | 有 rate-distortion 提升空间 |

### 2.3 我们的方法哲学

核心原则：

> 不盲目重写 mesh，不盲目用 test 结果选参数；只有当训练视角证据同时支持“安全压缩”和“安全修复”时，才执行修改。否则回退。

这使 SPCarNet 与普通后处理或参数搜索有本质区别。

---

## 3. 方法总览

### 3.1 Pipeline

```text
Clean MeshSplatting checkpoint
  -> train-view evidence mining
  -> sparse-occlusion protected compaction
  -> checkpoint-safe topology rewrite
  -> train-only Evidence Lumigraph Adapter
  -> guarded alpha / edge policy selection
  -> held-out render and evaluation
```

### 3.2 可以放进 PPT 的方法图

```text
                 train views only
                       |
                       v
+------------------------------+
| Surface / render evidence    |
+------------------------------+
       |                 |
       v                 v
+-------------+   +-------------------------+
| Safe prune  |   | Residual support graph  |
+-------------+   +-------------------------+
       |                 |
       v                 v
+----------------+  +----------------------+
| Compact mesh   |  | Evidence Lumigraph   |
| checkpoint     |  | Adapter              |
+----------------+  +----------------------+
       |                 |
       +--------+--------+
                v
      Guarded train-only policy
                |
                v
      Held-out render / metrics
```

### 3.3 与原始 MeshSplatting 的区别

通俗说：

- 原始 MeshSplatting：训练一个 mesh 表示，然后直接渲染。
- SPCarNet：先让这个 mesh 自己“体检”，找出哪些地方多余、哪些地方容易出错；只在训练视角证据可靠时做压缩和修复；如果证据不可靠，就保持原样或走安全 fallback。

更技术地说：

| 维度 | clean MeshSplatting | SPCarNet 当前方法 |
|---|---|---|
| 表示 | opaque triangle mesh | compact opaque triangle mesh + train-evidence residual adapter |
| 压缩 | 无额外证据闭环 | train-view sparse-occlusion protected compaction |
| 外观修复 | checkpoint 自身颜色/属性 | Evidence Lumigraph Adapter 从 train residual 转移稳定修正 |
| 参数选择 | checkpoint iteration 或默认训练流程 | train-only alpha / edge / branch guard |
| test set 使用 | 只评估 | 只评估，不用于选 branch/alpha/edge/prune |
| 风险控制 | 依赖训练本身 | strict RGB/geometry guard + no-op/fallback |

---

## 4. 模块细节

### 4.1 模块 A：Sparse-Occlusion Protected Compaction

目标：

> 减少无用或低风险三角形，同时不破坏 held-out view 渲染和 sparse geometry。

做法：

1. 从训练视角统计 surface/render evidence。
2. 判断三角形是否对多个视角稳定可见，是否承担关键颜色/深度解释。
3. 对户外场景允许约 10% 到 12% 的安全压缩。
4. 对室内或敏感场景启用 micro-budget guard，避免为了压缩数字强行删面。

为什么重要：

- 方法不仅提升 RGB，也提供真实 mesh compactness。
- 三角形减少来自实际 checkpoint，不是报告层面的虚假压缩。
- 室内场景压缩比例较低是有意设计：当前 evidence 表明这些场景更容易因为删除产生 geometry/render 风险。

### 4.2 模块 B：Checkpoint-Safe Topology Rewrite

目标：

> 将压缩决策落到真实 MeshSplatting checkpoint 中，并保证 renderer / evaluator 仍能正常运行。

它需要处理：

- face 删除；
- vertex / face index remapping；
- trailing unused vertices；
- checkpoint tensor shape 一致性；
- render 和 metric 脚本兼容。

这一步是工程上很重要的部分，因为它保证最后产物仍是 mesh checkpoint，而不是只在图片上做修补。

### 4.3 模块 C：Evidence Lumigraph Adapter

目标：

> 用训练视角中的真实 residual 证据修复 held-out view 的局部外观错误。

核心步骤：

1. 渲染训练视角。
2. 计算 train render 与 train GT 的 RGB residual。
3. 对每个 target view，选择相邻 support views。
4. 通过相机、深度和视角一致性将 support residual warp 到 target view。
5. 聚合多视角一致的 residual signal。
6. 用 train calibration 选择 alpha / benefit / edge gate。
7. 对 held-out render 应用修复。

关键实现接口：

| 文件 | 角色 |
|---|---|
| `utils/evidence_lumigraph_adapter.py` | ELA 核心：support selection、residual warp、signal aggregation、alpha calibration、adapt_frame |
| `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py` | ELA CLI：批量渲染修复、报告、W&B log |
| `scripts/car_model/ecsr_run_phasef_ela_adapter_eval.py` | Phase-F/Phase-J ELA 评估入口 |
| `scripts/car_model/ecsr_materialize_phaseh_guarded_adapter.py` | guarded portfolio materialization |

### 4.4 模块 D：Guarded Adaptive Policy

Phase-J 的关键升级是：不是所有场景都强行用同一种 ELA。

策略：

- 8 / 9 场景使用 adaptive alpha ELA。
- `treehill` 使用 train-selected structural edge fallback。

`treehill` 为什么特殊：

- Phase-H adaptive alpha 提升 PSNR，但伤害 SSIM/LPIPS。
- Phase-J 用训练集 edge-gate search 选择 q=0.5、alpha=0.75。
- 最终 `treehill` 相对 Phase-F 也实现三指标严格提升。

这解决了 Phase-H 最后的非严格胜出场景。

---

## 5. 主实验结果

### 5.1 Full9 per-scene table

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | tri red. |
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

均值：

```text
dPSNR  = +1.3311
dSSIM  = +0.0347
dLPIPS = -0.0634
mean triangle reduction = 7.6479%
```

### 5.2 与 MeshSplatting paper table 的 per-scene 对比

| scene | paper PSNR/SSIM/LPIPS | Phase-J PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|
| bicycle | 23.04 / 0.641 / 0.348 | 24.0215 / 0.7024 / 0.2661 | +0.9815 | +0.0614 | -0.0819 |
| flowers | 19.34 / 0.480 / 0.417 | 20.3044 / 0.5578 / 0.3292 | +0.9644 | +0.0778 | -0.0878 |
| garden | 24.70 / 0.762 / 0.217 | 26.3111 / 0.8278 / 0.1358 | +1.6111 | +0.0658 | -0.0812 |
| stump | 24.78 / 0.678 / 0.316 | 25.5951 / 0.7241 / 0.2639 | +0.8151 | +0.0461 | -0.0521 |
| treehill | 20.53 / 0.540 / 0.428 | 21.2962 / 0.5956 / 0.3363 | +0.7662 | +0.0556 | -0.0917 |
| room | 28.52 / 0.873 / 0.271 | 30.3056 / 0.9057 / 0.1960 | +1.7856 | +0.0327 | -0.0750 |
| counter | 26.51 / 0.846 / 0.279 | 28.4492 / 0.8937 / 0.1865 | +1.9392 | +0.0477 | -0.0925 |
| kitchen | 27.42 / 0.858 / 0.227 | 30.1997 / 0.9161 / 0.1320 | +2.7797 | +0.0581 | -0.0950 |
| bonsai | 28.19 / 0.876 / 0.294 | 31.8620 / 0.9303 / 0.1726 | +3.6720 | +0.0543 | -0.1214 |

### 5.3 Geometry / topology

| scene | dAbsRel | dDepthMAE | dNormal | triangle red. | vertex red. | 状态 |
|---|---:|---:|---:|---:|---:|---|
| bicycle | -0.000241 | -0.0204 | -0.0119 | 10.01% | 4.57% | strict all-axis pass |
| flowers | -0.003356 | -0.1250 | -0.0439 | 10.02% | 4.64% | strict all-axis pass |
| garden | -0.000007 | -0.0002 | -0.0010 | 1.50% | 2.69% | geometry-safe |
| stump | -0.005878 | -0.3507 | -0.0260 | 10.02% | 4.57% | strict all-axis pass |
| treehill | -0.001246 | -0.0747 | -0.0122 | 10.01% | 4.86% | strict all-axis pass |
| room | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.03% | geometry-safe |
| counter | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.10% | geometry-safe |
| kitchen | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.29% | geometry-safe |
| bonsai | -0.000368 | -0.0045 | -0.0254 | 10.00% | 3.16% | strict all-axis pass |

解读：

- RGB 主指标是 9 / 9 严格胜出。
- geometry 是 9 / 9 safe，其中 6 / 9 严格更好。
- 室内场景三角形减少较小，是因为安全策略避免破坏高质量室内 geometry。

---

## 6. 定性结果该如何展示

### 6.1 全图对比

全图图像用于证明比较公平：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
```

推荐 PPT 说法：

> 这一页不是为了让观众在投影上立刻看出所有差异，而是证明我们使用同一 held-out view、同一 clean baseline、同一评价口径做比较。

### 6.2 局部 error-reduction 对比

局部图更适合展示方法优势：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
```

首选图是 `spcarnet_phasej_where_it_helps_showcase_20260622.png`，它直接从 Phase-J closure audit CSV 和 per-view delta CSV 生成，路径指向当前接受的 `ours_26000_phasej_guarded_adaptedge_ela` endpoint，而不是旧版 Compact-ELA/SOR layout。

这些图由 `scripts/car_model/generate_spcarnet_advantage_showcase.py` 生成：

1. 先筛选 full-view 上满足 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0` 的 held-out view。
2. 再在该 view 中寻找 texture crop，其中 SPCarNet 相对 GT 的 RGB error 下降最大。
3. 绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

可汇报局部 crop 结果：

| crop | full-view dPSNR/dSSIM/dLPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / 00001.png | +6.63 / +0.0452 / -0.0878 | +11.79 | 78.6% |
| kitchen / 00011.png | +3.43 / +0.0250 / -0.0578 | +10.48 | 71.4% |
| room / 00011.png | +3.50 / +0.0220 / -0.0656 | +10.36 | 67.7% |
| counter / 00013.png | +2.17 / +0.0407 / -0.0665 | +6.02 | 54.9% |
| garden / 00006.png | +1.74 / +0.0479 / -0.0678 | +4.26 | 44.4% |
| flowers / 00014.png | +1.12 / +0.0754 / -0.1028 | +2.15 | 25.3% |

### 6.3 为什么肉眼有时不明显

需要提前解释：

- 当前方法的很多收益是 residual-level distributed improvement，不一定是全局结构大变化。
- full-frame image 在 PPT 上缩小后会掩盖局部 texture/edge 修复。
- 因此定性展示应采用“全图公平性 + 局部放大 + error map”三联图。

---

## 7. 消融与负结果

### 7.1 clean 26000 vs clean 30000

重要结论：

> 训练更久不等于更好。

当前 clean baseline selection 中，9 / 9 场景均选择 clean `26000` 而不是 clean `30000`。这说明我们的提升不能被解释为“只是 baseline 没训练够”。

### 7.2 compaction alone 不足以解释 RGB 增益

压缩提供 compactness 和部分 geometry 侧收益，但主 RGB 提升来自 ELA residual repair。

推荐汇报表达：

```text
safe compaction gives us a better mesh carrier;
ELA gives us evidence-certified appearance recovery.
```

### 7.3 为什么需要 guarded policy

ELA 很强，但也可能过度修复。Phase-J 的核心就是把 ELA 放进 guard 中：

- train-only alpha selection；
- PSNR / SSIM / LPIPS non-regression；
- SSIM-peak guard；
- adaptive alpha 不稳定时切到 structural edge fallback。

### 7.4 Phase-S / v25 / v26 / v27 / v28 的真实状态

当前不能把 Phase-S/v25/v26/v27/v28 作为 headline。

| 分支 | 状态 | 结论 |
|---|---|---|
| Phase-S face-local residual repair | 已实现真实 representation-level checkpoint edit | 有可靠正收益，但相对 Phase-J 增益很小，不是论文级突破 |
| v25 witness-group CVaR | 已实现 train-objective change，bonsai medium W&B 验证 | 被 selector 拒绝，是有价值负结果 |
| v26 local-trust ELA | ELA / PhaseK / selector / autovisual profile 接口已接入，CLI help、py_compile、dry-run、local-trust smoke 均通过；bonsai medium 已完成 candidate-owned final decision | hard local-trust 过保守，candidate report-only test 小幅改善但 honest trainval/render-region gate 拒绝，不能作为 headline |
| v27 soft local-trust ELA | 新增 continuous trust-weighted residual，fixed profile `field_region_render_risk_strict_v27`；py_compile、smoke、dry-run、fixed-profile override 拒绝和 bonsai medium plan decision 均已完成 | soft trust 机制有效，明显好于 v26 hard trust 的“全零 trust”问题；但 trainval gate 拒绝 plan candidate，仍不能替代 Phase-J headline |
| v28 view-tail-safe alpha shrink | 已新增 policy-view tail 安全 alpha shrink，fixed profile `field_region_render_risk_strict_v28`；py_compile、ELA smoke、CLI visibility 和 dry-run command manifest 均通过 | 这是对 v27 tail 负迁移的直接修复，但尚未完成中长程 W&B 验证；当前只能作为下一轮候选机制，不可替代 Phase-J headline |

v25 bonsai medium 关键结果：

| stage | accepted | train-val balanced | report-only LPIPS | report-only PSNR | report-only SSIM | rejection |
|---|---:|---:|---:|---:|---:|---|
| plan | false | -0.000281 | +0.0000108 | +0.000547 | -0.0000075 | balanced below 0 |
| candidate-owned refit | false | -0.000420 | +0.0001369 | -0.002649 | -0.0000739 | balanced below 0; tail CVaR below floor |
| selector strictfull_s1 | false | -0.000123 | +0.0000087 | +0.0000687 | -0.0000025 | balanced below 0 |
| final selected | fallback | 0 | 0 | 0 | 0 | phasej fallback |

这页适合用来说明项目的科学严谨性：我们不是只挑好看的结果，也会保留和解释失败。

v26 bonsai medium 当前已落盘的关键 snapshot：

| branch | split | PSNR | SSIM | LPIPS | 解释 |
|---|---|---:|---:|---:|---|
| Phase-J local-trust fallback | test | 29.4213 | 0.9040 | 0.2398 | alpha 被 safe-zero 到 0，明显弱于 Phase-J headline |
| v26 candidate base | test | 28.8649 | 0.8960 | 0.2594 | face-local checkpoint update 后的 base render，尚未通过主线 |
| v26 candidate + ELA | test | 29.4608 | 0.9048 | 0.2390 | report-only test 相对 fallback 小幅改善 |
| v26 candidate trainval gate | trainval | 30.2851 | 0.9111 | 0.2345 | trainval 仍未达到 honest gate 要求 |

解读：

- v26 证明了接口和真实 checkpoint update 可以跑通，但当前 hard local-trust 规则过于保守。
- v26 candidate-owned decision 最终 `accepted=false`，原因包括 `ssim_regression_exceeds_5e-05` 和 render-region tail CVaR 风险。
- ELA report 中 hard local-trust 的 mean trust weight / active fraction 为 `0`，说明安全门把 residual 修复几乎完全抑制了。
- 它的价值在于给下一步 v27 提供明确诊断：local trust 不应该是纯二值硬拒绝，更合理的是 continuous trust-weighted residual 或两级 trust gate。
- 因此明天汇报时可以把 v26 放在“当前失败诊断与下一步”页，而不是主结果页。

v27 的当前状态：

| item | evidence |
|---|---|
| method | soft local-trust: support / residual std / agreement / confidence 共同生成连续 residual weight |
| fixed profile | `field_region_render_risk_strict_v27` |
| contract | `field_region_render_risk_strict_v27_soft_local_trust_weighted_residual` |
| smoke | `[ELA smoke] passed` |
| dry-run | `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v27_softtrust_recheck` |
| medium run | `/data/peilincai/spcarnet_runs/field_region_render_risk_strict_v27_softtrust_20260622_bonsai_medium_gpu4` |
| base plan test | 28.8649 PSNR / 0.8960 SSIM / 0.2594 LPIPS |
| v27 compact soft ELA test | 29.5067 PSNR / 0.9061 SSIM / 0.2320 LPIPS |
| v27 plan soft ELA test | 29.5038 PSNR / 0.9063 SSIM / 0.2320 LPIPS |
| v27 plan trainval | 30.4506 PSNR / 0.9142 SSIM / 0.2247 LPIPS |
| soft trust stats | test mean trust weight 0.6132, active fraction 0.9629; train mean trust weight 0.5873, active fraction 0.9574 |
| plan decision | `accepted=false`; selected fallback `phasej_guarded_adaptedge`; reasons: PSNR/SSIM/LPIPS/trainval balanced regression |
| current conclusion | v27 修复了 v26 “trust 全零”问题，但没有超过当前 fallback/Phase-J 主线，不能作为 headline |

### 7.5 v26/v27 横向诊断表

下面这张表适合放在 backup slide 或和 mentor 讨论下一步时使用。它说明最近两轮不是“白做”，而是把失败原因定位得更清楚：v26 过硬，v27 过软，下一步应该做 view/region conditional 的安全 alpha 或局部 rollback，而不是继续全局调阈值。

证据来自：

```text
/data/peilincai/spcarnet_runs/20260622_v26_v27_autovisual_run_summary.md
```

| branch | candidate | decision | test dPSNR | test dSSIM | test dLPIPS | trainval balanced | tail CVaR | 诊断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v26 hard local-trust | plan | false | +0.016258 | +0.000576 | +0.000198 | +0.023907 | -0.018957 | PSNR/SSIM 有增益，但 LPIPS 和 tail 不安全；hard trust 太保守且不连续 |
| v26 hard local-trust | candidate-owned refit | false | +0.039511 | +0.000839 | -0.000856 | +0.012566 | -0.062490 | test 三指标好看，但 trainval SSIM/tail 风险导致 honest gate 拒绝 |
| v27 soft local-trust | plan | false | -0.002831 | +0.000139 | +0.000037 | -0.037934 | -0.122139 | soft trust 解决了全零问题，但全局引入了更大的 trainval/tail 负迁移 |

对导师建议说法：

> v26/v27 不是当前主结果，但它们让下一步方向更明确：local trust 不应该是全局二值开关，也不应该是全局连续权重。它需要变成按 target view、局部区域和 tail-risk 条件变化的安全策略，并在局部失败时只回退失败区域，而不是整张图或整个候选全拒绝。

### 7.6 v28：view-tail-safe alpha shrink

v28 是当前刚实现的最新机制，动机来自 v27 的失败诊断：v27 已经让 residual transfer 变成连续权重，但 alpha 的安全性仍主要停留在 pooled pixel/bin 层面。一个 bin 的平均 MSE gain 可以为正，却仍可能在少数 policy-validation views 上造成 balanced score 或 LPIPS tail regression。

v28 的改动是：在原来的 per-bin adaptive alpha 拟合之后，再用 train/policy-view evidence 选择一个全局 view-tail scale：

```text
alpha_final(pixel) = view_tail_scale * alpha_bin(pixel)
```

每个候选 scale 都在 policy views 上计算：

- mean view gain；
- worst-tail CVaR view gain；
- negative-view fraction。

只有满足 tail 安全约束的 scale 才能被选中；grid 中包含 `0.0`，所以当所有候选都有风险时，系统可以自动退回 no-op，而不是强行修复。

当前实现状态：

| item | evidence |
|---|---|
| fixed profile | `field_region_render_risk_strict_v28` |
| contract | `field_region_render_risk_strict_v28_view_tail_safe_alpha_shrink` |
| default scale grid | `1.0,0.75,0.5,0.25,0.0` |
| core implementation | `utils/evidence_lumigraph_adapter.py` |
| ELA CLI | `--alpha_view_tail_scale_grid`, `--alpha_view_tail_cvar_fraction`, `--alpha_view_tail_min_gain`, `--alpha_view_tail_max_negative_fraction` |
| PhaseK / selector / AutoVisual | 均已转发 `ela_alpha_view_tail_*` |
| verification | py_compile passed, `git diff --check` passed, `[ELA smoke] passed`, CLI help 可见，dry-run manifest 确认 plan/candidate/selector 都带 v28 参数 |
| dry-run root | `/data/peilincai/spcarnet_runs/dryrun_field_region_render_risk_strict_v28_viewtail_20260622` |
| current conclusion | 真实方法改动已进入 train/eval pipeline，但还未完成中长程验证，不能宣传为已胜出 |

建议在 PPT 中这样定位：

> v28 是从失败诊断中长出来的下一轮机制：它不再只问“某类像素的 alpha 平均是否安全”，而是要求“这个 alpha 在 policy-validation view tail 上也安全”。这使 ELA 从局部像素统计走向 view-level risk control。

不建议在主结果页放 v28 数字，因为它还没有真实 medium/full W&B 结果。它更适合放在“ongoing upgrade / next experiment”页，说明我们知道 Phase-J 的 remaining risk 并且已经有明确修复机制。

---

## 8. 为什么这是研究工作而不只是工程 patch

可以从三个层次回答。

### 8.1 方法层

SPCarNet 不是简单后处理，而是一个 constrained decision policy：

- 压缩必须由 train-view geometry evidence 支持。
- 外观修复必须由 train-view residual evidence 支持。
- branch / alpha / edge fallback 不能由 held-out test 选择。
- 不通过 guard 的场景回退，不强行报告虚假增益。

### 8.2 表示层

结果仍围绕 MeshSplatting 的核心资产：opaque triangle mesh。

- 压缩落在真实 checkpoint 上。
- renderer/evaluator 可以直接消费 compact checkpoint。
- ECSR 后续路线尝试把 residual repair 从 render-time adapter 蒸馏回 representation-level checkpoint edit。

### 8.3 实验层

当前证据不是单场景 cherry-pick：

- Mip-NeRF360 full9；
- 9 / 9 selected clean baseline strict RGB win；
- 244 / 246 per-view strict RGB win；
- 9 / 9 geometry-safe；
- 与 local clean 和 paper table 都有对照；
- 对失败分支保留日志和拒绝原因。

---

## 9. 当前短板与风险

需要在 mentor 面前诚实讲清楚：

1. 最强 endpoint 仍包含 render-time ELA adapter，不是完全 baked representation。
2. 定性全图差异有时不够强，必须用局部 crop / error map 展示。
3. 平均三角形减少 7.65%，室内场景压缩比例低，rate-distortion 故事仍可加强。
4. Phase-S/v25/v26/v27/v28 等后续分支还没有显著超越 Phase-J；v26/v27/v28 都是有价值的失败诊断与机制验证，不能替代当前 Phase-J 主结果。
5. paper table 对比只能作为辅助；主 claim 要放在本地 fair baseline。

推荐表述：

> 当前方法已经证明 MeshSplatting 可以被 train-only evidence loop 稳定增强；下一步的论文级升级应把 ELA 的修复能力进一步内化到 representation-level，从而减少“render-time adapter”的 reviewer 风险。

---

## 10. PPT 建议结构

### Slide 1：标题

SPCarNet: Evidence-Certified Compact Residual Repair for MeshSplatting

### Slide 2：Motivation

MeshSplatting 很强，但仍有局部 residual error、训练过拟合敏感、拓扑冗余。

### Slide 3：Baseline protocol

展示 clean `26000/30000` baseline selection：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

强调 train metrics 不参与 baseline/method selection。

### Slide 4：Method overview

放 pipeline：

```text
MeshSplatting -> evidence mining -> safe compaction -> ELA repair -> guarded policy
```

### Slide 5：Safe compaction

讲 triangle reduction 和 geometry-safe 审计。

### Slide 6：Evidence Lumigraph Adapter

讲 support-view residual warp、alpha calibration、train-only guard。

### Slide 7：Main quantitative result

放 full9 mean：

```text
9 / 9 strict RGB wins
+1.3311 PSNR
+0.0347 SSIM
-0.0634 LPIPS
7.6479% triangle reduction
```

### Slide 8：Per-scene table

放 9 场景结果表，突出 `bonsai/kitchen/counter/room` 等强收益。

### Slide 9：Geometry / topology audit

展示 9 / 9 geometry-safe，6 / 9 strict sparse geometry win。

### Slide 10：Qualitative

左：full-frame fair comparison。

右：local crop / error map。

建议图：

- `assets/spcarnet_m360_full9_qualitative_gallery.png`
- `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`
- `assets/spcarnet_phasej_where_it_helps_selection_20260622.json`
- `assets/spcarnet_m360_outdoor_detail_showcase.png`
- `assets/spcarnet_m360_where_it_helps_showcase.png`

### Slide 11：Ablation and failure diagnosis

讲 clean 30000 不如 26000、ELA guard 必要、v25 被拒绝。

### Slide 12：Limitations and next direction

讲 render-time adapter 风险，以及下一步 representation-level local-trust / distillation；v26 local-trust、v27 soft local-trust 和 v28 view-tail-safe alpha shrink 都可以作为“已完成的失败诊断与机制验证”，不要放进已完成主结论。

---

## 11. Mentor 可能会问的问题

### Q1：这是不是只是在 MeshSplatting 后面加了图像后处理？

不是简单图像后处理。当前结果有两层：

- compact checkpoint 是真实 mesh topology 修改，有 triangle reduction 和 geometry audit；
- ELA 是 train-view geometry/depth/camera-aware residual transfer，不使用 held-out GT 调参。

但需要诚实承认：当前最强 RGB endpoint 仍有 render-time adapter 成分，后续要把它 distill / bake 回 representation-level。

### Q2：有没有 test leakage？

当前汇报主结果的设计目标是避免 test leakage：

- clean baseline 从 clean 26000/30000 中按 held-out score 选择，这是 baseline envelope，不用 train metrics 偏置 longer checkpoint。
- 方法 branch / alpha / edge fallback 使用 train-only calibration。
- held-out test 只用于最终报告。

PPT 中建议强调“method selection does not use test GT”。

### Q3：为什么室内场景 triangle reduction 低？

因为当前策略是 evidence-conservative。

室内场景 geometry 本身更稳定，盲目删除面容易破坏 sparse depth / normal，因此系统采用 micro-budget guard。这样牺牲了一部分压缩率，但保护了 9 / 9 geometry-safe claim。

### Q4：和 MeshSplatting paper 24.78 PSNR 怎么比？

可以说：

- paper table 是外部参考；
- 我们本地 clean reproduction 平均 25.1517 PSNR，已经高于 paper table；
- 当前 Phase-J 是 26.4828 PSNR，继续高于本地 clean；
- 因此最可信 claim 是“超过我们复现的更强 local clean MeshSplatting baseline”。

### Q5：当前是否已经是 top-conference ready？

作为阶段性结果很强，但完整顶会闭环仍有风险：

- RGB 和 geometry 证据强；
- 方法故事清晰；
- 但 headline 仍依赖 render-time ELA；
- representation-level 分支收益仍小。

建议的诚实回答：

> 当前结果足以支撑一个强实验发现：MeshSplatting 可以通过 train-only evidence-certified compact repair 明显增强。若目标是顶会主稿，还需要把修复能力更进一步内化到表示层，降低 adapter-like 风险。

---

## 12. 一分钟汇报稿

> 我们的工作基于 MeshSplatting，但目标不是换掉它，而是让它拥有一个训练视角证据驱动的自诊断和自修复闭环。首先，我们用训练视角判断哪些三角形可以安全删除，从而得到 compact mesh checkpoint；然后，我们从训练渲染和 GT 的 residual 中构建 Evidence Lumigraph Adapter，把多视角一致的 residual 信息转移到 held-out view；最后，我们用 train-only calibration 决定 alpha、edge fallback 和回退策略，held-out test 只做最终评估。在 Mip-NeRF360 full9 上，当前 Phase-J endpoint 相对 selected clean MeshSplatting 实现 9/9 场景 PSNR、SSIM、LPIPS 严格提升，均值提升 +1.3311 PSNR、+0.0347 SSIM、-0.0634 LPIPS，同时平均减少 7.6479% 三角形，并且 244/246 个 held-out view 都是三指标严格胜出。它也高于 MeshSplatting paper table 平均。当前局限是最强外观收益仍来自 render-time ELA，后续需要将这一证据修复能力进一步蒸馏进 representation-level checkpoint。

---

## 13. 证据索引

主结果：

- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/ours_26000_phasej_guarded_adaptedge_ela_guarded_decisions.json`

方法日志：

- `docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md`
- `docs/car_model/5-30-WitnessGroupCVaR-v25-Log.md`
- `docs/car_model/6-22-LocalTrust-v26-Integration-And-Bonsai-Medium-Log.md`
- `docs/car_model/6-22-SoftLocalTrust-v27-Implementation-And-Bonsai-Medium-Log.md`
- `docs/car_model/6-22-ViewTailSafe-v28-Implementation-And-DryRun-Log.md`
- `docs/car_model/SPCarNet_research_log.md`

README 结果块：

- `README.md`
- `README.zh.md`

定性图：

- `assets/spcarnet_m360_full9_qualitative_gallery.png`
- `assets/spcarnet_m360_outdoor_detail_showcase.png`
- `assets/spcarnet_m360_where_it_helps_showcase.png`
- `assets/meshsplatopt_clean_vs_ours_montage.png`

核心实现：

- `utils/evidence_lumigraph_adapter.py`
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
- `scripts/car_model/ecsr_run_phasef_ela_adapter_eval.py`
- `scripts/car_model/ecsr_materialize_phaseh_guarded_adapter.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
