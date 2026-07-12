# SPCarNet 当前完整技术报告（Mentor PPT 版，v68）

日期：2026-06-24

用途：mentor 汇报、PPT 拆页、方法讲解、实验复盘、后续研究计划。

---

## 0. 一页结论

SPCarNet 当前必须分两条线汇报：

| 层级 | 当前结论 | PPT 建议口径 |
|---|---|---|
| 主结果 endpoint | `Phase-J guarded adaptive Evidence Lumigraph Adapter` 在本地同协议 full9 selected-clean baseline 上 `9 / 9` 场景三指标严格胜出，平均减少 `7.6479%` triangles | 当前最安全、最能支撑汇报的结果 |
| 表示级内化路线 | v48-v68 已实现 surface atlas、capacity policy、face/bin alpha、teacher basis、RGB alpha、uncertainty shrink 等接口，但收益仍小 | 作为研究路线、消融和反思，不要包装成已完成论文终局 |

一句话讲法：

> SPCarNet 不是从零训练一个新模型，而是把一个已训练好的 MeshSplatting checkpoint 升级成一个会用训练视角证据做几何压缩、局部修复、风险自检和自动回退的可审计表示系统。

最适合 PPT 首页的数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

当前诚实结论：

> 对 mentor 可以说：我们已经有一个强的、可审计的 MeshSplatting 后处理/修复 endpoint，在本地 selected-clean full9 口径上全面超过 clean baseline，并同时压缩 triangles。不能说的是：表示级 residual field 已经完全替代 render-time adapter。v64 是当前最稳 representation-level policy，但效果量级太小；v65-v68 都是重要负诊断。

---

## 1. 背景问题

MeshSplatting 的优势是把场景落到显式 triangle mesh 上，方便渲染、压缩、编辑和几何审计。但一个训练好的 clean checkpoint 仍然存在三个缺口：

| 缺口 | 表现 | SPCarNet 的目标 |
|---|---|---|
| 局部外观 residual | 叶片、边界、桌面、遮挡处有稳定 RGB 残差 | 从训练视角证据中恢复可迁移 residual |
| 几何冗余 | 一部分 triangles 对多视角解释贡献低 | 在质量不退化前提下删除冗余 triangles |
| 泛化风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 视角 | 用 train-only evidence gate 决定修复、回退或 no-op |

研究问题：

```text
Given a trained MeshSplatting checkpoint,
can we use training-view surface evidence to certify
where the mesh can be compacted and where appearance residuals can be safely repaired?
```

这不是普通图像增强。SPCarNet 的修复信号必须绑定到真实 mesh surface、face id、可见性、barycentric/UV bin、多视角一致性和 policy-validation 风险上。

---

## 2. 方法总览

基础 MeshSplatting:

```text
images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render test views
```

SPCarNet:

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
| Guarded ELA | compact render + train residual evidence | repaired render | 迁移多视角稳定 residual |
| Train-Only Policy | policy-val metrics and risk | accept/fallback/no-op | 避免 test leakage 和参数游戏 |
| Surface Residual Atlas | face/UV/bin evidence | persistent residual field | 把 residual 内化到 surface 表示 |
| v64 Fixed Auto Policy | v63b audit + v56 fallback | selected full9 tree | 固定自动选择 bin-alpha 或回退 |
| v68 Keep/Downweight Shrink | policy-val bin gain/variance/sign evidence | local downweight profile | 诊断 uncertainty-aware residual transfer |

---

## 3. 方法细节

### 3.1 Evidence Cache

Evidence cache 保存训练或 policy-val 视角中的：

- rendered RGB；
- ground-truth RGB；
- residual map：`GT - Render`；
- alpha / visibility；
- face id；
- barycentric coordinate 或 UV/bin；
- normal、depth、camera center；
- per-face 和 per-bin sample count；
- residual sign consistency；
- image L1、PSNR、SSIM、LPIPS；
- per-view mean、min-view、CVaR tail risk。

它把训练数据从“只用于优化参数的数据”变成“后续判断 surface 是否可靠、是否可修、是否必须回退的证据”。

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩是 quality-first rate-distortion，不是最大删面：

- 低可见性、低贡献、低风险 faces 优先删除；
- 遮挡边界、thin structure、sparse geometry 风险区受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- topology audit 和 sparse COLMAP geometry audit 独立记录；
- 修复失败时允许 fallback，不强行修改所有场景。

报告中的 `triangle reduction` 指删去的 triangles 占原始 triangles 的比例。Phase-J 平均删面为 `7.6479%`。

### 3.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 收益来自 guarded ELA。它在 mesh surface 上做训练证据约束的 residual transfer：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 的像素；
- `residual_i(u_i)` 来自训练视角中相同或相近 surface 区域；
- `w_i(p)` 由可见性、surface correspondence、local support 和风险门控决定；
- `alpha` 由 train/policy-val evidence 选择；
- 如果证据不足或 tail risk 高，则回退或 no-op。

通俗讲法：

> 我们不是凭空增强图片，而是看训练视角里同一个 surface 区域是否反复出现稳定错误。如果错误稳定且多视角支持充足，就把这部分 residual 迁移到目标视角；否则不动。

### 3.4 Train-Only Policy

公平性规则：

- branch selection、alpha、support、fallback 都来自 train 或 policy-val evidence；
- held-out test GT 只用于最终报告；
- 不用 test metric 选择参数；
- 不用 train metric 选择训练更久的 baseline；
- 风险高时回退稳定版本。

Phase-J 中多数场景使用 adaptive alpha，`treehill` 使用 auto edge fallback。这个选择不是看 held-out test 得出的。

### 3.5 Surface Residual Atlas

为了把 render-time adapter 的收益内化到表示里，我们实现了 face/UV-addressed residual atlas：

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

### 3.6 v64 Fixed Auto Bin-Alpha Policy

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

### 3.7 v68 Keep-With-Downweight Uncertainty Shrink

v67 的问题是 `sparse_positive` 语义过于稀疏：只保存强正证据 bins，未知 bins 很容易没有 residual mass。v68 新增 `keep_with_downweight`：

```text
unknown bins keep fallback residual strength
observed risky bins get explicit local downweight
```

风险信号包括：

- policy-val positive-view deficit；
- negative-gain confidence；
- atlas variance penalty；
- residual sign inconsistency；
- sample-count confidence。

v68 证明语义修复有效：`counter/kitchen` 都比 v67 好。但它仍低于 v56/v64 selected reference，所以不晋级。

---

## 4. 评估口径

主评估使用本地 Mip-NeRF360 same-protocol full9 reproduction：

- baseline 是 clean MeshSplatting `26000/30000` checkpoint envelope；
- 每个场景只用 held-out test 指标选 clean baseline；
- scoring 只用于 baseline envelope：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

关键边界：

| 口径 | 能否作为主 claim | 说明 |
|---|---|---|
| 本地 selected clean MeshSplatting | 是 | 同 split、同 evaluator、同本地 artifacts |
| Phase-F compact baseline | 是，但用于内部 ablation | 证明 ELA/repair 相对 compact parent 的增益 |
| MeshSplatting paper table | 只能作附录参考 | 论文训练细节、checkpoint、评价配置不一定完全同口径 |
| train metrics | 否 | 不用于 baseline 或最终方法选择 |

---

## 5. 主结果：Phase-J vs 本地 selected clean MeshSplatting

Aggregate:

| metric | value |
|---|---:|
| scenes | `9 / 9` |
| strict scene RGB wins | `9 / 9` |
| strict per-view RGB wins | `244 / 246` |
| mean dPSNR vs clean | `+1.331084` |
| mean dSSIM vs clean | `+0.034702` |
| mean dLPIPS vs clean | `-0.063359` |
| mean dPSNR vs source ELA | `+0.833143` |
| mean dSSIM vs source ELA | `+0.018946` |
| mean dLPIPS vs source ELA | `-0.039986` |
| mean triangle reduction | `7.6479%` |

Per-scene:

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

Evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

---

## 6. 表示级路线：v48 到 v68

这条线的目标是把 Phase-J 的 render-time 修复收益内化到 persistent mesh representation。当前结果仍然小，但它补齐了论文所需的很多方法与消融接口。

| version | real method change | outcome |
|---|---|---|
| v48 | auto-support surface residual atlas | full9 positive vs no-op, but small effect |
| v52 | capacity-aware fixed policy over v48/v51 | `9 / 9` non-regressive/tie vs v48, small gain |
| v55d | per-face/local alpha calibration | strong `counter`, unsafe globally |
| v56 | fixed face-alpha reliability guard | selects only `counter`, safe report-only candidate |
| v63/v64 | bin-level alpha calibration and fixed auto policy | v64 promotes only `kitchen`, best fixed representation policy |
| v65 | teacher-distilled shared residual basis | negative diagnostic |
| v66 | per-bin RGB alpha | negative diagnostic, ties kitchen but hurts counter |
| v67 | sparse-positive uncertainty shrink | negative diagnostic, too conservative |
| v68 | keep-with-downweight uncertainty shrink | improves v67, still below v64/v56 |

Best fixed representation-level result, v64:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |

Evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
```

v68 latest diagnostic:

| scene | reference | v67 | v68 | decision |
|---|---:|---:|---:|---|
| counter | `26.756130 / 0.862126 / 0.251691` | `26.749853 / 0.862050 / 0.251998` | `26.753967 / 0.862119 / 0.251854` | improves v67, below reference |
| kitchen | `27.822626 / 0.876538 / 0.198849` | `27.816389 / 0.876443 / 0.199201` | `27.819143 / 0.876533 / 0.199032` | improves v67, below reference |

Evidence:

```text
docs/car_model/6-24-v68-KeepDownweightUncertainty-Probe-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/summary.md
```

---

## 7. 相对 MeshSplatting 论文表格的附录口径

README 中保留了一组相对 MeshSplatting paper table 的历史对比。它能说明我们方法具备和论文表格同级别的可比性，但不建议作为主 claim，因为本地复现实验和论文表格可能存在训练长度、checkpoint、evaluation script、image resolution、mask/crop 等口径差异。

历史附录均值：

| comparison | result |
|---|---:|
| MeshSplatting paper table | `9 / 9` RGB wins |
| mean dPSNR | `+0.8685` |
| mean dSSIM | `+0.0366` |
| mean dLPIPS | `-0.0465` |

建议 PPT 说法：

> 在本地同协议复现上，SPCarNet Phase-J 相对 selected clean MeshSplatting 是严格的 full9 胜出；相对 paper table 的对比也呈正向，但我们把它作为附录，不作为最强公平主 claim。

---

## 8. 定性展示建议

已有可直接放 PPT 的视觉资产：

| asset | 用途 |
|---|---|
| `assets/spcarnet_m360_full9_qualitative_gallery.png` | full9 总览 |
| `assets/spcarnet_v42_atlas_qualitative_panel.png` | surface atlas 局部修复 |
| `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` | capacity policy / cap-hit 场景 |
| `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` | counter face-alpha guard 局部效果 |

推荐展示顺序：

1. 先放 full9 table，不先放小差异 crop。
2. 再放 `counter/kitchen/bonsai` 这类数值提升强的 scene。
3. 局部图尽量展示 error map 或 zoom-in，不要只放原图对比，因为人眼很难看出微小 residual 改善。
4. 表示级 v64/v68 的图只作为 ablation，不作为主视觉效果。

---

## 9. Ablation 讲法

PPT 中建议把 ablation 分三层：

### 9.1 主方法贡献

| ablation | 结论 |
|---|---|
| clean MeshSplatting -> Phase-F compact | 有压缩，但需要 repair 保质量 |
| Phase-F compact -> Phase-J guarded ELA | 主要 RGB 提升来源 |
| fixed alpha -> adaptive alpha / edge fallback | 修复 tail-view 风险 |
| no policy gate -> train-only policy | 避免 test leakage 和不安全编辑 |

### 9.2 表示级探索

| ablation | 结论 |
|---|---|
| v48/v52 support/capacity | surface atlas 可安全小幅修复 |
| v55d/v56 face alpha | local alpha 能解决局部 amplitude，但需要 guard |
| v63/v64 bin alpha | bin-level magnitude calibration 是当前最稳 fixed representation policy |
| v65 teacher basis | 简单线性 shared basis 不够 |
| v66 RGB alpha | channel-wise alpha 没有解决核心瓶颈 |
| v67/v68 uncertainty shrink | 风险降权有用，但 capacity/support 仍不足 |

### 9.3 当前最重要的负结论

> 反复调 alpha、RGB alpha、uncertainty shrink 已经不能带来大幅收益。真正瓶颈是 residual 表示容量、surface support 覆盖和 out-of-trajectory 泛化保护。

---

## 10. 当前短板

| 短板 | 证据 | 影响 |
|---|---|---|
| Phase-J 仍是 render-time adapter | 主收益来自 ELA，不完全 baked into checkpoint | 顶会论文需要更强 representation-level story |
| 表示级收益太小 | v64 mean dPSNR 只有 `+0.000410080` vs v56 | 不足以单独作为 paper endpoint |
| v65-v68 均未晋级 | teacher basis、RGB alpha、uncertainty shrink 都低于 selected reference | 说明单纯校准不够 |
| 定性提升有时不明显 | 局部 residual 改善难被人眼直接看出 | PPT 需要 error map/crop 支撑 |
| paper-table 口径不完全同源 | 本地 artifacts 与论文表格可能配置不同 | 主 claim 应坚持本地 selected-clean full9 |

---

## 11. 下一步建议

最值得继续推进的方向：

1. **多尺度 surface-conditioned residual basis**  
   从单 face/bin residual 升级到 multi-scale surface patch basis，提升容量。

2. **target-view support-aware residual selection**  
   不是只看训练 bin 是否可靠，还要看 target view 是否落在同分布 surface/normal/camera support 中。

3. **tail-view certificate**  
   把 per-view worst-case、CVaR、SSIM/L1/LPIPS 多指标证书作为 accept 条件。

4. **local qualitative objective**  
   针对 PPT 和论文图，补充 error-map、crop-level PSNR/SSIM/LPIPS，而不是只看全图均值。

5. **fresh blind policy validation**  
   对 v64-style fixed policy 在未参与规则设计的场景或 split 上重跑，降低后验调参嫌疑。

---

## 12. PPT 拆页大纲

建议 14 页：

| 页码 | 标题 | 核心信息 |
|---:|---|---|
| 1 | Problem | MeshSplatting strong but leaves residual and redundancy |
| 2 | Key Idea | Use train-view surface evidence to compact and repair safely |
| 3 | Pipeline | Evidence cache -> compaction -> repair -> policy gate |
| 4 | Evidence Cache | residual/face/bin/normal/view-risk are auditable |
| 5 | Geometry-Safe Compaction | quality-first triangle reduction |
| 6 | Guarded ELA | main residual-transfer mechanism |
| 7 | Fairness Protocol | train-only policy, held-out test only for report |
| 8 | Main Results | Phase-J 9/9 wins and 7.6479% triangle reduction |
| 9 | Per-Scene Table | full9 scene table |
| 10 | Qualitative | full9 gallery and zoom-in/error-map panels |
| 11 | Representation Track | v48-v68 method evolution |
| 12 | v64/v68 Ablation | best fixed policy and latest negative diagnostic |
| 13 | Limitations | render-time endpoint, small representation gains |
| 14 | Next Step | multi-scale surface-conditioned residual field |

---

## 13. 可能被 mentor 问到的问题

### Q1: 这是不是只是后处理？

答：Phase-J 的主收益是 render-time residual repair，所以它确实更接近 guarded adapter endpoint；但它不是无约束图像后处理，因为 residual 被 face/surface/evidence/policy gate 约束。表示级路线 v48-v68 正是在把它内化到 mesh-attached representation。

### Q2: 有没有 test leakage？

答：主方法的 branch、alpha、edge fallback、compaction ratio 都由 train 或 policy-val evidence 决定。held-out test GT 只用于最终评估。需要注意的是 v64 仍标为 report-only，因为规则是在看过早期 probe 后设计的，后续需要 fresh blind validation。

### Q3: 为什么 representation-level 还没有大幅超过？

答：当前 surface atlas 主要是 local residual lookup/calibration，容量和 coverage 都低于 Phase-J 的 render-time multi-view adapter。v65-v68 的负结果说明，仅调 alpha 或 shrink 不能解决容量瓶颈。

### Q4: 能不能声称全面超过 MeshSplatting？

答：在本地 selected-clean full9 口径，Phase-J 是 `9 / 9` 三指标严格胜出并且减少 triangles。对 MeshSplatting paper table 可以附录展示正向对比，但主 claim 应该限定为本地同协议复现。

### Q5: 当前是不是足够投顶会？

答：当前主结果足够支撑一次强汇报，但还不建议包装成顶会终局。顶会级闭环还需要把主要收益从 render-time adapter 更强地迁移到 representation-level method，或者把 adapter 本身包装成有清晰理论和应用价值的 certified rendering module。

---

## 14. 结论

当前最稳的故事是：

> SPCarNet demonstrates that a trained MeshSplatting representation can be upgraded by train-only surface evidence into a geometry-safe, residual-aware, self-auditing rendering system. On local full9 Mip-NeRF360 reproduction, the Phase-J endpoint strictly improves all scenes over selected clean MeshSplatting while reducing triangles. The ongoing representation-level track has implemented the right interfaces and exposed the core bottleneck: the next breakthrough must increase surface residual capacity and target-view support certification, not continue scalar parameter tuning.

停止线：

```text
Final status for paper loop: NOT COMPLETE
Reason: Phase-J is strong and presentation-ready, but the representation-level endpoint has not yet achieved paper-level effect size.
```

