# SPCarNet 当前方法完整技术报告（Mentor/PPT 汇报版）

日期：2026-06-24  
用途：给 mentor 汇报、制作 PPT、解释当前方法、展示已有定量/定性证据、明确当前弱点和下一步路线。  
当前建议汇报主线：**Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**。  
当前最佳表示级固定策略：**v64 fixed auto bin-alpha policy**。  
此前已完成但未晋级增强：**v73b target-support cheap pre-ranking**，已在 `counter` 上完成验证；它把 target-support proxy 前置到 expensive policy-val/refit 之前，成功把 support candidates 从 `2` 剪到 `1` 并保留 `fit_residual_topk`，但最终仍选择 `blend=0.0`，指标与 v73/v70/v71a 持平并低于 v64/v56 reference，因此不晋级。v73 target-support candidate selection、v72 local prior allowlist 仍作为负诊断保留。

随后已完成但未晋级增强：**v74 residual delta-cap ladder**。v74 把 residual RGB 幅度上限从固定常数升级为 train/policy-val 选择的候选 ladder，并修复 policy-val 与最终 target apply 之间的 delta clipping 一致性。它已在 `counter` 上完成 W&B online 验证，最终仍选择原始保守 cap `0.12` 和 `blend=0.0`，指标与 v73b/v73/v70/v71a 持平并低于 v64/v56 reference，因此不晋级。

随后已完成但未晋级增强：**v75 local patch surface prior**。v75 新增 `surface_multiscale_prior_mode=local_patch`，用 same-face local UV patch residual 估计替代更粗的 count-pyramid prior，并用 train-policy blend ladder 在 `0/0.5/1.0` 中选择。它已在 `counter` 上完成 W&B online 验证，nonzero local patch prior 覆盖了大量 low-support bins，但最终 policy 仍选择 `blend=0.0`，指标与 v74/v73b/v73/v70/v71a 持平并低于 v64/v56 reference，因此不晋级。

当前最新已完成但未晋级增强：**v76 policy-val bin-gain hybrid prior**。v76 把 v75 的 local patch prior 改成更局部的证书机制：先保留 zero-blend atlas，再只复制 train-policy-val 上逐 bin 证明有收益的 nonzero prior bins。它已在 `counter` 上完成 W&B online 验证并真实选中 hybrid atlas，但 held-out test 为 `26.753532 / 0.862111 / 0.251881`，低于 v75 zero-blend 行和 v64/v56 reference，因此不晋级。

---

## 1. 一页结论

SPCarNet 当前最稳妥的讲法是：

> MeshSplatting 训练出一个显式 mesh 表示以后，SPCarNet 继续利用训练视角中的 surface evidence 做自检、压缩和局部 residual 修复：哪些三角形可以安全减少，哪些 surface 区域有稳定错误可以修，哪些区域证据不足必须保持原样或回退。

当前结果要分两层讲：

| 层级 | 当前状态 | PPT 口径 |
|---|---|---|
| 主结果 endpoint | Phase-J 在本地 same-protocol Mip-NeRF360 full9 selected-clean MeshSplatting baseline 上 `9 / 9` 场景三指标严格胜出，平均删除 `7.6479%` triangles | 主结果，适合放 headline |
| 表示级内化路线 | v48-v76 已实现 surface atlas、capacity policy、face/bin alpha、uncertainty shrink、multi-scale prior、evidence-consistent gate、local prior allowlist、target-support candidate selection、target-support pre-rank、delta-cap ladder、local patch prior 和 policy-val bin-gain hybrid 等模块；v64 是目前最佳固定策略，但收益量级仍小 | 作为研究路线、消融和下一阶段突破方向 |

最适合第一页 PPT 的数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| per-view strict RGB wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

当前最诚实的结论：

> 我们已经有一个强的、可审计的 MeshSplatting 后处理/修复 endpoint，在本地 selected-clean full9 口径上全面超过 clean MeshSplatting，并同时压缩 triangles。还不能过度声称的是：表示级 residual field 已经完全替代 render-time adapter。v64 是当前最稳 representation-level policy，但效果量级仍小；v65-v76 的价值是定位了真正瓶颈，即 sparse surface support、residual capacity 和 target-view 泛化证书，而不是继续手动调参。

---

## 1.1 PPT 快速版摘要

如果只给 mentor 讲 3 分钟，建议按下面顺序：

| 讲述点 | 一句话版本 | 支撑证据 |
|---|---|---|
| 问题 | MeshSplatting 已经很强，但仍有局部 residual 和几何冗余 | clean baseline 与 held-out GT 的 residual/error map |
| 核心想法 | 不重新发明表示，而是让训练视角 evidence 告诉我们哪里能删、哪里能修、哪里必须不动 | face/bin/residual/support/risk evidence cache |
| 方法 | SPCarNet = geometry-safe compaction + surface-evidence residual repair + train-only policy gate | Phase-J guarded adaptive ELA pipeline |
| 主结果 | 本地 Mip-NeRF360 full9 selected-clean 口径下，9/9 场景三指标全胜 | `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS |
| 几何收益 | RGB 提升不是靠增加复杂度换来的，同时平均删去 `7.6479%` triangles | Phase-J closure audit |
| 当前边界 | 表示级 atlas 已经有完整接口，但收益仍小，不能包装成终局 | v64 小幅正向，v65-v76 多个负诊断 |
| 下一步 | 从“调参”转向“更强 surface residual 表示 + target-support certificate” | v72/v73 证明 policy-val positivity 和 target footprint 还不足以保证 target 有效 |

PPT 中最建议突出的一句话：

> SPCarNet turns a trained MeshSplatting model into a self-auditing surface system: it repairs only where training-view surface evidence certifies stable residuals, compacts only low-risk triangles, and falls back when the certificate is weak.

中文口径：

> SPCarNet 不是盲目增强渲染图，而是把训练视角中的残差信息投回 mesh surface，只在多视角证据稳定的位置修复，只在低风险位置删面，不可靠时自动回退。

---

## 2. 研究问题和动机

MeshSplatting 的优势是把 3D Gaussian Splatting 类的高质量渲染落到显式 mesh/surface 表示上。显式 mesh 便于部署、压缩、编辑和几何审计，但 clean MeshSplatting checkpoint 仍有三类问题：

| 问题 | 现象 | SPCarNet 目标 |
|---|---|---|
| 局部外观 residual | 桌面、叶片、遮挡边界、细纹理区域仍存在稳定 RGB 残差 | 用训练视角 residual 找到可迁移修复信号 |
| 几何冗余 | 一部分 triangles 对多视角解释贡献很低 | 在质量不退化前提下删除低风险冗余 faces |
| 泛化风险 | 盲目修复会伤害 tail views 或 out-of-trajectory 视角 | 用 train-only policy gate 决定修复、回退或 no-op |

研究问题可以写成：

```text
Given a trained MeshSplatting checkpoint,
can training-view surface evidence certify
where the mesh can be compacted and where appearance residuals can be safely repaired?
```

这不是普通图像增强。SPCarNet 的修复必须被真实 surface、face id、barycentric/UV bin、可见性、多视角一致性和 policy-validation 风险共同约束。

---

## 3. 方法总览

基础 MeshSplatting：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render held-out views
```

SPCarNet：

```text
training images + cameras
  -> train/load MeshSplatting checkpoint
  -> build train/policy-val evidence cache
  -> estimate surface support, residual, visibility, and risk
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
| v64 Fixed Auto Policy | v63b audit + v56 fallback | selected full9 policy tree | 自动选择 bin-alpha 或回退 |
| v70/v71a Multi-Scale Prior | face/bin count + coarse surface residual | low-support bin prior | 解决 sparse bin support 和 residual coverage 问题 |

---

## 4. 关键技术模块

### 4.1 Evidence Cache

Evidence cache 是整个方法的数据底座。它缓存 train/policy-val 视角中的：

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

这一步把训练数据从“只用于优化 checkpoint 参数”升级成“用于判断 surface 是否可靠、是否可修、是否必须回退”的证据。

### 4.2 Geometry-Safe Compaction

SPCarNet 的压缩原则是 quality-first rate-distortion，而不是最大化删面：

```text
remove triangles only when surface evidence says the edit is low-risk
```

主要策略：

- 低可见性、低贡献、低风险 faces 优先删除；
- 遮挡边界、thin structure、sparse geometry 风险区受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- topology audit、sparse COLMAP geometry audit 和 RGB audit 分开记录；
- 修复失败时允许 fallback，不强行修改所有场景。

报告中的 `triangle reduction` 指删去的 triangles 占原始 triangles 的比例。当前 Phase-J 平均删面为 `7.6479%`。

### 4.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 收益来自 guarded Evidence Lumigraph Adapter。它可以理解成 mesh surface 上的“证据约束 residual transfer”。

简化形式：

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

### 4.4 Train-Only Policy Gate

公平性规则：

- branch selection、alpha、support、fallback 都来自 train 或 policy-val evidence；
- held-out test GT 只用于最终报告；
- 不用 test metric 选择参数；
- 不用 train metric 选择训练更久的 baseline；
- 风险高时回退稳定版本。

Phase-J 中多数场景使用 adaptive alpha，`treehill` 使用 auto edge fallback。这个选择不是看 held-out test 得出的。

### 4.5 Surface Residual Atlas

为了把 render-time adapter 的收益内化到表示里，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face/bin selection
  -> residual atlas fitting
  -> policy-val risk gate
  -> target view surface lookup
  -> persistent surface-addressed repair
```

它回答的问题是：

> 能否把训练视角 residual 变成绑定在 mesh 表面的、可审计、可回退、可复用的修复表示？

当前结论：接口、安全策略和 full9 自动 policy 已经成型，但 RGB 效果量级仍小，还不能替代 Phase-J。

### 4.6 v64 Fixed Auto Bin-Alpha Policy

v64 是当前最稳的表示级固定策略。它把 v63b 的 bin-level residual magnitude calibration 变成自动规则：

```text
if v63b has strong train/policy-val bin-alpha evidence:
    use v63b bin-alpha residual atlas
else:
    fallback to v56 selected policy
```

v64 guard 不写场景名，也不读取 held-out test metric。它自动接受 `kitchen` 的 v63b，并回避 `counter` 等不稳定候选。

### 4.7 v69/v70/v71a Multi-Scale Prior 路线

v65-v70 暴露出更底层的问题：只调 residual alpha、uncertainty shrink 或 RGB channel alpha 不能解决 sparse support 区域 residual 覆盖不足。很多 face/UV bin 的直接样本数太少，导致 residual atlas 要么太保守，要么迁移不稳。

v69 新增 count-pyramid prior：

```text
direct face/bin residual
  -> if bin has enough samples: keep direct estimate
  -> if bin is low-support: blend with same-face coarse residual block prior
```

v70 把固定 blend 改成 train-only policy-val blend ladder：

```text
blend candidates = 0, 0.125, 0.25, 0.5
  -> each candidate refits the atlas from train evidence
  -> policy-val gates select safe prior strength
  -> nonzero blend must be non-regressive vs zero-blend anchor
```

v70 在 `counter/kitchen` 上都自动选择 `blend=0.0`，说明当前 coarse prior 还没有可靠到能提升 held-out test。它的价值是把“不要固定启用强 prior”变成可审计的自动回退。

v71a 当前进一步加入 evidence-consistent gate：

```text
only use coarse prior when the local bin and coarse prior agree in evidence:
  - enough direct samples
  - enough prior weight
  - residual sign consistency passes
  - local variance is not too high
  - direct/prior cosine is non-negative
```

v71a 是真实 train/eval pipeline 改动，已接入 adapter、scene runner 和 W&B：

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_l1risk_fairnoop_scene.py`
- W&B group：`v71a_evidence_consistent_prior`
- 已完成场景：`counter`、`kitchen`
- 当前状态：完成但未晋级；两场景均由 policy-val 选择 `blend=0.0`，说明当前 coarse prior 即使加入 evidence-consistent gate 也没有带来安全的非零 prior 提升。

---

## 5. 评估口径

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

## 6. 主结果：Phase-J vs 本地 selected clean MeshSplatting

Aggregate：

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

Per-scene：

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

证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv
outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_per_view_deltas.csv
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

---

## 7. 表示级路线：v48 到 v76

这条线的目标是把 Phase-J 的 render-time 修复收益内化到 persistent mesh representation。当前接口与审计机制已经成熟，但效果量级仍然偏小。

| version | real method change | outcome |
|---|---|---|
| v48 | auto-support surface residual atlas | full9 positive vs no-op, but small effect |
| v52 | capacity-aware fixed policy over v48/v51 | `9 / 9` non-regressive/tie vs v48, small gain |
| v55d | per-face/local alpha calibration | strong `counter`, unsafe globally |
| v56 | fixed face-alpha reliability guard | selects only `counter`, safe report-only candidate |
| v63/v64 | bin-level alpha calibration and fixed auto policy | v64 promotes only `kitchen`, best fixed representation policy |
| v65 | teacher-distilled shared residual basis | negative diagnostic |
| v66 | per-bin RGB alpha | negative diagnostic |
| v67 | sparse-positive uncertainty shrink | negative diagnostic, too conservative |
| v68 | keep-with-downweight uncertainty shrink | improves v67, still below v64/v56 |
| v69 | count-pyramid multi-scale prior for low-support bins | real pipeline change, but fixed strong blend regressed |
| v70 | policy-val blend ladder for count-pyramid prior | selects zero blend automatically; safe but below v64 |
| v71a | evidence-consistent gate for count-pyramid prior | completed counter/kitchen; selected `blend=0.0`, exactly matches v70, not promoted |
| v72 | local prior allowlist with policy-val bin uncertainty guard | completed counter; nonzero prior and allowlist active, but held-out regresses, not promoted |
| v73 | target-support candidate selection | completed counter; target-support ranking active and target footprint larger, but selected `blend=0.0`, ties v70/v71a, not promoted |
| v73b | cheap target-support support-set pre-ranking | completed counter; prunes support candidates from `2` to `1`, keeps expanded support, but metrics still tie v73/v70/v71a, not promoted |
| v74 | residual delta-cap ladder and cap-consistent policy-val | completed counter; selected cap `0.12` and `blend=0.0`, ties v73b/v73/v70/v71a, not promoted |
| v75 | same-face local patch prior for low-support bins | completed counter; nonzero prior covers `951427` bins but policy selects `blend=0.0`, ties v74/v73b, not promoted |
| v76 | policy-val bin-gain hybrid local prior | completed counter; selected hybrid prior with `13708` certified bins, but held-out metrics slightly regress vs v75 and v64/v56, not promoted |

Best fixed representation-level result, v64：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |

v64 per-scene decision summary：

| scene | selected | reason |
|---|---|---|
| bicycle | v56 fallback | bin-alpha evidence too sparse/weak |
| flowers | v56 fallback | policy-val SSIM/L1 evidence insufficient |
| garden | v56 fallback | evidence insufficient |
| stump | v56 fallback | evidence insufficient |
| treehill | v56 fallback | weak policy-val robustness |
| room | v56 fallback | selected alpha too low and gain too small |
| counter | v56 fallback | bin-alpha count too high/unstable and alpha too low |
| kitchen | v63b bin-alpha | pass; `+0.003690720` PSNR vs v56 |
| bonsai | v56 fallback | bin-alpha support too sparse |

v68/v69/v70 focused diagnostics：

| scene | reference | v68 | v69 | v70 | conclusion |
|---|---:|---:|---:|---:|---|
| counter | `26.756130 / 0.862126 / 0.251691` | `26.753967 / 0.862119 / 0.251854` | `26.751703 / 0.862084 / 0.251951` | `26.753996 / 0.862119 / 0.251853` | v70 safe fallback, still below reference |
| kitchen | `27.822626 / 0.876538 / 0.198849` | `27.819143 / 0.876533 / 0.199032` | `27.819000 / 0.876532 / 0.199036` | `27.819157 / 0.876533 / 0.199031` | v70 safe fallback, still below reference |

v70 audit takeaway：

> policy-val blend ladder rejected nonzero count-pyramid prior on both tested scenes and selected `blend=0.0` with `alpha=0.125`。这避免了 v69 的退化，但没有超过 v64。说明粗尺度 prior 的核心问题不是 blend 数值，而是 prior 本身需要更强的 evidence consistency 和 target-view support certificate。

v71a completed diagnostic：

| item | status |
|---|---|
| code implementation | completed |
| static validation | `py_compile` passed |
| CLI exposure | adapter and runner help passed |
| W&B logging | enabled |
| scenes completed | `counter`, `kitchen` |
| selected blend | `0.0` on both scenes |
| current conclusion | safe zero-blend fallback; not promoted |

| scene | PSNR | SSIM | LPIPS | selected blend | vs v70 | vs selected reference |
|---|---:|---:|---:|---:|---|---|
| counter | `26.753996` | `0.862119` | `0.251853` | `0.0` | exact tie | `-0.002134 / -0.000006743 / +0.000162049` |
| kitchen | `27.819157` | `0.876533` | `0.199031` | `0.0` | exact tie | `-0.003469 / -0.000004790 / +0.000181846` |

v71a audit takeaway：

> evidence-consistent gate 是必要的安全接口，但当前 same-face count-pyramid prior 没有产生 policy-val-safe nonzero improvement。下一步不应继续只扫 blend/gate 阈值，而应提升 residual 表示容量和 target-view support certificate。

v72 local prior allowlist diagnostic：

| item | value |
|---|---:|
| scene | `counter` |
| W&B run | `fnc0ktxk` |
| selected blend | `1.0` |
| selected alpha | `0.125` |
| prior blended bins | `201339` |
| bin-guard allowed bins | `4791 / 250224` |
| bin-guard allowed faces | `741` |
| target changed fraction | `0.003807394` |

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v72 local prior allowlist | `26.750389` | `0.862056` | `0.251968` | active but regressive |
| v70/v71a counter | `26.753996` | `0.862119` | `0.251853` | safer zero-blend row |
| selected v64/v56 reference | `26.756130` | `0.862126` | `0.251691` | current reference |

v72 audit takeaway：

> v72 证明了局部 prior allowlist 能真实生效，但它没有把 count-pyramid prior 变成有效收益。policy-val gain 为正、target 也确实被改动，但 target changed fraction 只有 `0.3807%`，held-out 三指标仍退化。因此下一步应把 target-visible support 放进候选选择证书，而不是仅在最后做 `min_target_changed_fraction` sanity check。

v73 target-support candidate selection diagnostic：

| item | value |
|---|---:|
| scene | `counter` |
| W&B run | `kgfav7cf` |
| selected support mode | `fit_residual_topk` |
| selected support added faces | `4096` |
| selected texture size | `16` |
| selected fill mode | `nearest_observed` |
| selected blend | `0.0` |
| selected alpha | `0.125` |
| target changed fraction | `0.065630289` |
| target min-view changed fraction | `0.023086760` |
| target CVaR20 changed fraction | `0.027341737` |

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v73 target-support selection | `26.753996` | `0.862119` | `0.251853` | active, larger target support, but not promoted |
| v70/v71a counter | `26.753996` | `0.862119` | `0.251853` | same zero-blend safe row |
| selected v64/v56 reference | `26.756130` | `0.862126` | `0.251691` | current reference |

v73 audit takeaway：

> v73 完成了 target-support candidate selection 的真实接口闭环：候选排序能看到 target changed fraction、min-view target support 和 CVaR target support，并因此选择 expanded-support candidate。它修复了 v72 的“target footprint 太小”诊断，但没有解决指标瓶颈，因为最终仍选择 `blend=0.0` 并低于 v64/v56。下一步不应继续扫 scalar blend，而应提高 residual 表示容量，并把 target-support profiling 前置到 cheap pre-ranking 以降低运行开销。

v73b target-support pre-rank diagnostic：

| item | value |
|---|---:|
| scene | `counter` |
| W&B run | `qn1ntfyy` |
| input support candidates | `2` |
| retained support candidates | `1` |
| retained support mode | `fit_residual_topk` |
| selected support added faces | `4096` |
| selected blend | `0.0` |
| selected alpha | `0.125` |
| target changed fraction | `0.065630289` |

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v73b target-support pre-rank | `26.753996` | `0.862119` | `0.251853` | active, prunes candidate search, but not promoted |
| v73 target-support selection | `26.753996` | `0.862119` | `0.251853` | same selected zero-blend row |
| selected v64/v56 reference | `26.756130` | `0.862126` | `0.251691` | current reference |

v73b audit takeaway：

> v73b 把 target-support 证书前置，完成了候选搜索层面的工程闭环：`fit_residual_topk` 的 target coverage 明显高于 base carrier，因此被保留，base support 被剪掉。它减少了无效候选进入昂贵 refit 的机会，但没有带来新 RGB 收益，因为最终 safe policy 仍回到 `blend=0.0`。这进一步证明：target-support gating 是必要接口，不是充分条件；下一阶段必须提升 residual atlas 表示容量。

证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v68_keepdown_probe_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v69_multiscale_prior_probe_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v70_blendladder_probe_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v71a_evidence_consistent_prior_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v72_local_prior_allowlist_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73_target_support_selection_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73b_target_support_prerank_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md
```

v74 delta-cap ladder diagnostic:

| item | value |
|---|---|
| scene | `counter` |
| W&B run | `q9g7b7o9` |
| W&B URL | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/q9g7b7o9` |
| cap candidates | `0.12, 0.18, 0.24` |
| blend candidates | `0, 1.0` |
| purpose | test whether representation-level atlas is bottlenecked by over-tight residual amplitude clipping |
| code status | `py_compile` passed for adapter and runner |
| result status | completed; not promoted |

v74 method change:

```text
fit residual atlas candidate
  -> clip residual delta by candidate max_abs_delta_rgb
  -> evaluate policy-val with the same clipped delta
  -> select cap/blend/alpha only from train-policy evidence
  -> apply the same selected cap to target views
```

The important correction is not just a new hyperparameter. Before v74, the residual cap was effectively a fixed safety ceiling. v74 exposes it to the same policy-val candidate mechanism as blend/alpha, and the audit now records `selected_max_abs_delta_rgb`, candidate list, and per-candidate score ordering. This closes a policy consistency gap: policy-val and target application now use the same clipped residual amplitude.

Current command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v74_delta_cap_ladder_20260624 \
  --tag v74_deltacap_ladder_targetsupport_prerank_top1_countpyramid_blendladder_support4096_tex16_nearest_region_texture_adapter \
  --texture_size_candidates 16 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 4096 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --surface_multiscale_prior_mode count_pyramid \
  --surface_multiscale_prior_block_sizes 2,4,6 \
  --surface_multiscale_prior_min_bin_samples 8 \
  --surface_multiscale_prior_count_tau 32.0 \
  --surface_multiscale_prior_blend 1.0 \
  --surface_multiscale_prior_blend_candidates 0,1.0 \
  --surface_multiscale_prior_gate_mode evidence_consistent \
  --view_conditioned_basis_mode normal_camera_linear \
  --view_conditioned_basis_guard_mode policy_val_nonregressive \
  --max_abs_delta_rgb 0.12 \
  --max_abs_delta_rgb_candidates 0.12,0.18,0.24 \
  --enable_target_support_candidate_selection \
  --target_support_prerank_top_k 1 \
  --target_support_prerank_max_views 8 \
  --min_target_changed_fraction 0.0 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --wandb_project SPCarNet \
  --wandb_group v74_delta_cap_ladder \
  --wandb_run_name v74_deltacap_ladder_counter_20260624 \
  --wandb_mode online \
  --force
```

Result:

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v74 delta-cap ladder | `26.753996` | `0.862119` | `0.251853` | completed, not promoted |
| v73b/v73/v70/v71a zero-blend row | `26.753996` | `0.862119` | `0.251853` | exact tie |
| selected v64/v56 counter reference | `26.756130` | `0.862126` | `0.251691` | still stronger |

v74 selected:

```text
max_abs_delta_rgb = 0.12
surface_multiscale_prior blend = 0.0
alpha = 0.125
target changed fraction = 0.065630289
```

The cap ladder candidate audit shows that caps `0.12`, `0.18`, and `0.24` have identical policy-val scores within each blend group. This means the current `counter` residual predictions are not materially clipped by the original `0.12` cap. The active bottleneck is therefore not residual amplitude clipping; it is still residual representation capacity / support quality.

Presentation boundary:

> v74 可以在汇报中讲成“已完成的容量接口诊断”：它闭合了 cap-consistent policy 接口，但负结果说明 `counter` 的主要瓶颈不是 residual amplitude cap，而是更深的 residual representation capacity / support certificate。

v75 local patch prior diagnostic:

| item | value |
|---|---|
| scene | `counter` |
| W&B run | `j8fhiczt` |
| W&B URL | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/j8fhiczt` |
| prior mode | `local_patch` |
| patch radii | `1,2,3` |
| blend candidates | `0,0.5,1.0` |
| selected blend | `0.0` |
| selected alpha | `0.125` |
| target changed fraction | `0.065630289` |
| result status | completed; not promoted |

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v75 local patch prior | `26.753996` | `0.862119` | `0.251853` | completed, not promoted |
| v74/v73b/v73/v70/v71a zero-blend row | `26.753996` | `0.862119` | `0.251853` | exact tie |
| selected v64/v56 counter reference | `26.756130` | `0.862126` | `0.251691` | current reference |

v75 audit takeaway：

> v75 证明 local patch prior 真实接入并能覆盖 low-support bins：`blend=0.5/1.0` 都会混合 `951427` 个 bins，约 `65.5469%`。但 train-policy 仍选择 `blend=0.0`，held-out 指标完全回到 v74/v73b zero-blend 行。这进一步排除了“count-pyramid prior 太粗”这一单因子解释；当前瓶颈仍是 residual representation capacity 与 target-view generalization certificate 的联合问题。

v76 policy-val bin-gain hybrid diagnostic:

| item | value |
|---|---|
| scene | `counter` |
| W&B run | `8qetk7tj` |
| W&B URL | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/8qetk7tj` |
| method change | zero-blend atlas plus policy-val-certified local-patch prior bins |
| selected blend | `1.0` |
| selected alpha | `0.125` |
| selected hybrid | `true` |
| candidate bins | `233306` |
| allowed bins | `13708` |
| allowed-bin fraction | `0.058755454` |
| target changed fraction | `0.065630289` |
| result status | completed; not promoted |

| method | PSNR | SSIM | LPIPS | conclusion |
|---|---:|---:|---:|---|
| v76 policy-val bin-gain hybrid | `26.753532` | `0.862111` | `0.251881` | completed, not promoted |
| v75 local patch / zero-blend row | `26.753996` | `0.862119` | `0.251853` | stronger |
| selected v64/v56 counter reference | `26.756130` | `0.862126` | `0.251691` | current reference |

v76 audit takeaway：

> v76 证明了更细粒度的 local prior certificate 可以真实跑通：hybrid atlas 被构建、被 policy-val 选中，并把 `13708` 个 prior bins 写入最终 target apply。但 policy-val 相对 zero-blend 的优势只有极小 margin，held-out test 反而略低于 v75 zero-blend 和 v64/v56 reference。这个结果说明“逐 bin policy-val 正收益”仍不足以构成 target-view 泛化证书，尤其当很多 bin 只有少量样本和单 policy-view 支持时。下一步如果继续该方向，应提高 multi-view/support 门槛，而主路线仍应转向更高容量 surface-conditioned residual representation。

---

## 8. 定性展示建议

已有可直接放 PPT 的视觉资产：

| asset | 用途 |
|---|---|
| `assets/spcarnet_m360_full9_qualitative_gallery.png` | full9 总览 |
| `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | Phase-J 局部 held-out error reduction |
| `assets/spcarnet_m360_outdoor_detail_showcase.png` | 室外局部细节展示 |
| `assets/spcarnet_m360_where_it_helps_showcase.png` | 混合室内/室外局部展示 |
| `assets/spcarnet_v42_atlas_qualitative_panel.png` | surface atlas 局部修复 |
| `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` | capacity policy / cap-hit 场景 |
| `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` | counter face-alpha guard 局部效果 |

展示策略：

1. 先放 full9 定量表，建立方法不是只挑图。
2. 再放 `counter/kitchen/bonsai` 这类数值提升强的场景。
3. 定性图尽量用 zoom-in 和 error map，不只放原图对比，因为全图 residual 改善在人眼上可能很细。
4. 表示级 v64/v68/v70 的图作为 ablation，不作为主视觉 headline。

推荐 PPT 视觉页标题：

```text
Surface-evidence repair improves residual errors while preserving geometry.
```

---

## 9. Ablation 讲法

主方法贡献：

| ablation | 结论 |
|---|---|
| clean MeshSplatting -> Phase-F compact | 有压缩，但需要 repair 保质量 |
| Phase-F compact -> Phase-J guarded ELA | 主要 RGB 提升来源 |
| fixed alpha -> adaptive alpha / edge fallback | 修复 tail-view 风险 |
| no policy gate -> train-only policy | 避免 test leakage 和不安全编辑 |

表示级探索：

| ablation | 结论 |
|---|---|
| v48/v52 support/capacity | surface atlas 可安全小幅修复 |
| v55d/v56 face alpha | local alpha 能解决局部 amplitude，但需要 guard |
| v63/v64 bin alpha | bin-level magnitude calibration 是当前最稳 fixed representation policy |
| v65 teacher basis | 简单线性 shared basis 不够 |
| v66 RGB alpha | channel-wise alpha 没有解决核心瓶颈 |
| v67/v68 uncertainty shrink | 风险降权有用，但 capacity/support 仍不足 |
| v69 multi-scale prior | 针对 support/coverage 的真实改动；首轮过度 blended |
| v70 blend ladder | 自动拒绝 unsafe nonzero prior blend，避免 v69 退化 |
| v71a evidence-consistent prior gate | 已验证；policy 仍选择 zero-blend，说明 coarse prior 当前不能安全晋级 |
| v72 local prior allowlist | 已验证；local allowlist 生效但 held-out 退化，说明 policy-val positivity 还需要 target-support certificate |
| v73 target-support candidate selection | 已验证；target-support ranking 生效并扩大 target footprint，但仍回到 zero-blend 行，说明 footprint 本身还不等于有效 residual capacity |
| v73b target-support pre-rank | 已验证；cheap support pre-rank 生效并减少候选，但指标仍未超过 v64/v56，说明 search/pruning 不能代替表示容量提升 |
| v74 delta-cap ladder | 已验证；cap candidate 接入 policy-val，但 `0.12/0.18/0.24` 同组持平，说明 amplitude cap 不是主要瓶颈 |
| v75 local patch prior | 已验证；local patch prior 大量覆盖 low-support bins，但 policy 仍选 `blend=0.0`，说明粗 prior 不是唯一瓶颈 |

当前最重要的负结论：

> 反复调 alpha、RGB alpha、uncertainty shrink 或固定 multi-scale blend 已经不能带来大幅收益。真正瓶颈是 residual 表示容量、surface support 覆盖和 out-of-trajectory 泛化保护。

---

## 10. 相对 MeshSplatting paper table 的附录口径

README 中保留了一组相对 MeshSplatting paper table 的历史对比。它能说明我们方法具备和论文表格同级别的可比性，但不建议作为主 claim，因为本地复现实验和论文表格可能存在训练长度、checkpoint、评价配置、image resolution、mask/crop 等口径差异。

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

## 11. 当前短板和风险

| 短板 | 证据 | 影响 |
|---|---|---|
| Phase-J 仍是 render-time adapter | 主收益来自 ELA，不完全 baked into checkpoint | 顶会论文需要更强 representation-level story，或把 adapter 明确包装成 certified rendering module |
| 表示级收益太小 | v64 mean dPSNR 只有 `+0.000410080` vs v56 | 不足以单独作为 paper endpoint |
| v65-v70 均未晋级 | teacher basis、RGB alpha、uncertainty shrink、multi-scale prior 和 blend ladder 都低于 selected reference | 说明单纯校准或粗尺度 prior 不够 |
| 定性提升有时不明显 | 局部 residual 改善在全图尺度上不显眼 | PPT 需要 error map/crop 支撑 |
| paper-table 口径不完全同源 | 本地 artifacts 与论文表格可能配置不同 | 主 claim 应坚持本地 selected-clean full9 |
| v71a 未晋级 | `counter/kitchen` 完成后均选择 `blend=0.0`，与 v70 持平 | 不能声称 evidence-consistent prior 带来指标突破 |
| v72 未晋级 | `counter` 上非零 prior 和 bin allowlist 都生效，但 held-out 退化 | 不能声称 local prior allowlist 解决 count-pyramid prior 瓶颈 |
| v73 未晋级 | `counter` 上 target-support selector 生效，target changed fraction 提升到 `6.5630%`，但指标与 v70/v71a 持平并低于 v64/v56 | 不能声称 target-support ranking 已解决表示级收益瓶颈 |
| v73b 未晋级 | `counter` 上 target-support pre-rank 成功剪掉 base support，但最终指标仍与 v73 持平 | 不能把 candidate pruning 讲成指标突破 |
| v74 未晋级 | `counter` 上 residual delta-cap ladder 选择原始 `0.12` cap，所有 cap 候选同组持平 | 不能声称放宽 residual 幅度解决了瓶颈 |
| v75 未晋级 | `counter` 上 local patch prior 的非零 blend 覆盖 `951427` bins，但 policy 仍选择 `blend=0.0` | 不能声称 local patch prior 已解决 low-support residual capacity |
| v76 未晋级 | `counter` 上 policy-val bin-gain hybrid 被选中，但 held-out 低于 v75 zero-blend 和 v64/v56 | 不能声称局部 bin-gain certificate 已解决 target-view 泛化 |

---

## 12. 下一步研究路线

最值得继续推进的方向：

1. **多尺度 surface-conditioned residual basis**  
   从单 face/bin residual 升级到 multi-scale surface patch basis，提升低支持区域的容量和覆盖。

2. **target-view support-aware residual selection**  
   不只看训练 bin 是否可靠，还要看 target view 是否落在同分布 surface/normal/camera support 中。

3. **tail-view certificate**  
   把 per-view worst-case、CVaR、SSIM/L1/LPIPS 多指标证书作为 accept 条件。

4. **local qualitative objective**  
   针对 PPT 和论文图，补充 error-map、crop-level PSNR/SSIM/LPIPS，而不是只看全图均值。

5. **fresh blind policy validation**  
   对 v64/v70/v71a-style fixed policy 在未参与规则设计的场景或 split 上重跑，降低后验调参嫌疑。

当前代码侧已经完成 `target-support candidate selection`、`target-support pre-ranking`、`delta-cap ladder`、`local_patch` surface prior 和 `policy-val bin-gain hybrid`。原则是，在 train-policy-safe 的候选中，不只按 policy-val gain 排序，还要优先选择在 target geometry/evidence 上有足够可见作用范围、per-view changed fraction 和 CVaR target support 的候选。v73 证明 target-support 排序真实生效，v73b 进一步证明 cheap pre-rank 能在昂贵 refit 前剪掉弱 support candidate；v74 证明 residual amplitude cap 不是当前 `counter` 主瓶颈；v75 证明 local patch prior 覆盖 low-support bins 仍不足以晋级；v76 进一步证明逐 bin policy-val gain 证书如果缺少强 multi-view support，仍无法保证 held-out 泛化。下一步应保留 target-support/局部证书接口，但把主要创新转向更高容量、更可泛化的 surface-conditioned residual representation。

---

## 13. PPT 拆页大纲

建议 16 页：

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
| 11 | Representation Track | v48-v76 method evolution |
| 12 | v64/v70-v76 Ablation | best fixed policy, completed negative diagnostics, non-promoted prior gates |
| 13 | Why Not Just Tuning | v65-v76 show the bottleneck is capacity/support, not scalar thresholds |
| 14 | Limitations | render-time endpoint, small representation gains |
| 15 | Next Step | target-support-certified candidate selection |
| 16 | Takeaway | evidence-certified mesh upgrade, strong endpoint, honest remaining gap |

PPT 图表放置建议：

| 页码 | 建议图/表 | 路径 |
|---:|---|---|
| 8 | full9 aggregate table | 本报告第 6 节 |
| 9 | per-scene result table | 本报告第 6 节 |
| 10 | full9 qualitative gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| 10 | where-it-helps crop/error map | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 11 | surface-atlas/face-alpha ablation panels | `assets/spcarnet_v42_atlas_qualitative_panel.png`, `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |
| 12 | v64/v70-v76 diagnostic table | 本报告第 7 节 |

### 13.1 可直接拆成 PPT 的逐页内容

下面是可以直接复制到 PPT 的 slide plan。建议不要把每页塞满；mentor 汇报时更重要的是清楚地区分“已经能强 claim 的 Phase-J endpoint”和“仍在推进的 representation-level 终局路线”。

| 页 | 标题 | 页面主句 | 推荐素材 |
|---:|---|---|---|
| 1 | SPCarNet: Surface-Evidence Upgrade for MeshSplatting | 用训练视角 surface evidence 把 MeshSplatting 升级成可自检、可压缩、可修复、可回退的系统。 | 一句话方法图 |
| 2 | Why MeshSplatting Still Has Room | clean MeshSplatting 很强，但仍有局部 residual、几何冗余和 tail-view 风险。 | clean vs GT residual crop |
| 3 | Core Insight | residual 只有在同一个 surface 区域被多视角稳定观察到时才值得迁移。 | face/bin/evidence 示意 |
| 4 | Pipeline | evidence cache -> geometry-safe compaction -> guarded residual repair -> train-only policy gate。 | 本报告第 3 节流程 |
| 5 | Evidence Cache | 每个训练视角被转成 face id、barycentric/bin、alpha、normal、residual、risk 证据。 | 模块表 |
| 6 | Geometry-Safe Compaction | 删除的是低风险冗余 triangles，不是用质量换压缩。 | triangle reduction 表 |
| 7 | Guarded ELA | 主要 RGB 收益来自 surface-constrained residual transfer，而不是无约束图像增强。 | Phase-J 公式和 crop |
| 8 | Fairness Protocol | policy 只看 train/policy-val evidence；held-out test 只用于最终报告。 | 口径边界表 |
| 9 | Main Quantitative Result | 本地 selected-clean full9 下 `9/9` 场景三指标严格胜出，`244/246` views 严格胜出。 | 第 6 节 aggregate |
| 10 | Per-Scene Result | 室内/室外 full9 都正向，且平均删面 `7.6479%`。 | 第 6 节 per-scene table |
| 11 | Qualitative Evidence | 用 zoom-in 和 error map 展示 residual 下降；全图差异可能不显眼。 | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 12 | Outdoor Detail | 室外细节收益需要 crop/error-map 表达，不建议只放原图并排。 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| 13 | Representation-Level Track | v48-v76 已把 surface atlas、alpha calibration、multi-scale prior、target support、pre-rank、delta-cap ladder、local patch prior 和 bin-gain hybrid 都接入 pipeline。 | 第 7 节 version table |
| 14 | Best Fixed Representation Policy | v64 是当前最稳 fixed policy，但 effect size 小，不能包装成终局。 | v64 comparison table |
| 15 | What Failed and Why It Matters | v65-v76 负结果证明瓶颈不是继续调 scalar alpha/blend/cap。 | v70-v76 diagnostic table |
| 16 | Current Bottleneck | target footprint 变大仍不能保证指标提升，说明 residual capacity 和 target support certificate 要一起解决。 | v72/v73 对比 |
| 17 | Research Value | 贡献不是“调参”，而是可审计的 surface-evidence decision system。 | method contribution bullets |
| 18 | Limitations | 主收益仍是 render-time endpoint；representation-level effect size 还小。 | risk table |
| 19 | Next Research Step | 升级 multi-scale surface-conditioned residual basis，并前置 target-support cheap pre-ranking。 | next-step diagram |
| 20 | Takeaway | SPCarNet 已经是强汇报 endpoint，但 paper-level 终局仍需 representation-level 更大收益。 | 一页结论 |

### 13.2 讲给 mentor 的 claim 边界

建议明确说：

```text
我们已经在本地同协议 full9 clean MeshSplatting baseline 上得到一个强 endpoint：
9/9 场景三指标严格胜出，同时平均删面 7.6479%。
```

不建议说：

```text
我们已经彻底解决 MeshSplatting，或者 representation-level 方法已经全面替代 adapter。
```

更稳妥的表述是：

```text
Phase-J 证明 train-view surface evidence 可以可靠提升 MeshSplatting；
v48-v76 证明我们已经把这条路线往 persistent surface representation 内化，
但当前 representation-level effect size 还不够，需要进一步提升 residual capacity。
```

### 13.3 代码和证据索引

| 证据类型 | 路径 | PPT 用法 |
|---|---|---|
| 当前完整技术报告 | `docs/car_model/6-24-SPCarNet-Current-Method-Full-Technical-Report-ForMentor.zh.md` | 主讲稿 |
| Phase-J full9 主表 | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` | 主结果 |
| Phase-J closure audit | `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv` | 公平性和几何审计 |
| v64 fixed policy summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md` | 表示级最佳固定策略 |
| v73 target-support summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73_target_support_selection_20260624/summary.md` | 最新已完成诊断 |
| v73b pre-rank summary | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73b_target_support_prerank_20260624/summary.md` | 最新完成诊断 |
| v74 completed diagnostic | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v74_delta_cap_ladder_20260624/summary.md` | 最新容量接口负诊断 |
| v75 completed diagnostic | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md` | 最新 local patch prior 负诊断 |
| v76 completed diagnostic | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md` | 最新 policy-val bin-gain hybrid 负诊断 |
| adapter 主实现 | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` | 方法实现证据 |
| scene runner / W&B | `scripts/car_model/run_l1risk_fairnoop_scene.py` | 训练评估和 logging 入口 |
| v64 policy materialization | `scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py` | fixed-policy 重现实验 |
| qualitative gallery | `assets/spcarnet_m360_full9_qualitative_gallery.png` | 总览图 |
| where-it-helps panel | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | 局部效果页 |
| outdoor detail panel | `assets/spcarnet_m360_outdoor_detail_showcase.png` | 室外细节页 |

### 13.4 最新完成实验：v73b

v73b 的目的不是换一个更好看的表，而是修复 v73 的工程短板：v73 已经能把 target-support profile 接入候选排序，但 profile 是在昂贵的 policy-val/refit 后才附加的，不能减少运行成本。v73b 把 target-support proxy 前置到 support-set 层：

```text
support candidates
  -> cheap target face-support proxy on target geometry
  -> keep top-K support sets
  -> expensive policy-val/refit only on retained candidates
```

v73b 已接入并完成 `counter` 验证：

- `--target_support_prerank_top_k`
- `--target_support_prerank_max_views`
- W&B policy metrics；
- adapter audit 中的 `target_support_prerank`；
- runner 到 adapter 的 CLI 转发。

结果：

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v73b target-support pre-rank | `26.753996` | `0.862119` | `0.251853` |
| v73 target-support selection | `26.753996` | `0.862119` | `0.251853` |
| selected v64/v56 counter reference | `26.756130` | `0.862126` | `0.251691` |

结论：v73b 是 runtime/policy-search closure，不是 paper headline。它说明 target-support pre-ranking 必须保留，但下一阶段突破点仍然是 residual representation capacity，而不是更复杂的候选排序。

### 13.5 最新完成实验：v74

v74 是在 v73b 基础上新增的 residual amplitude capacity probe。此前很多表示级方法的改动都没有显著提升，可能原因之一是 atlas 输出被统一 `max_abs_delta_rgb=0.12` 安全上限压得太保守；但直接放大上限又可能导致局部过修复。因此 v74 不是手动改大 cap，而是把 cap 变成 policy-val 候选：

```text
max_abs_delta_rgb candidates = 0.12, 0.18, 0.24
```

每个 candidate 都必须经过同一套 train/policy-val image L1、SSIM、relative gain、positive-view fraction、target-support sanity check 和 no-op fallback。tie-break 仍偏向更保守的 cap，避免把容量提升变成无约束放大 residual。

最终状态：

| item | value |
|---|---|
| code path | adapter + scene runner |
| W&B | online, run `q9g7b7o9` |
| GPU | `5` |
| output root | `/dev/shm/peilincai_spcarnet_v74_delta_cap_ladder_20260624` |
| log | `/dev/shm/peilincai_spcarnet_v74_delta_cap_ladder_20260624/logs/apply_metrics_counter.log` |
| status | completed; not promoted |

v74 的实际结果支持下面这个负结论：

> 单纯放宽 residual amplitude cap 仍不能突破 v64/v56 reference，下一步必须从 residual basis/patch-level representation 结构本身升级。

### 13.6 最新完成实验：v75

v75 在 v74 之后测试一个更结构性的假设：如果 count-pyramid prior 太粗，是否 same-face local UV patch prior 能给低支持 bins 更细粒度的 residual 估计。它新增 `surface_multiscale_prior_mode=local_patch`，并在 `counter` 上跑 `blend=0/0.5/1.0` 的 train-policy ladder。

最终状态：

| item | value |
|---|---|
| W&B | `j8fhiczt` |
| prior mode | `local_patch` |
| patch radii | `1,2,3` |
| selected blend | `0.0` |
| selected alpha | `0.125` |
| target changed fraction | `0.065630289` |
| result | `26.753996 / 0.862119 / 0.251853` |
| status | completed; not promoted |

v75 的实际结果支持下面这个负结论：

> local patch prior 的非零 blend 可以覆盖大量 low-support bins，但 train-policy 仍选择 zero-blend，说明单纯把 prior 从 coarse block 换成 local patch 还不足以解决 residual 表示容量和 target-view 泛化问题。

### 13.7 最新完成实验：v76

v76 在 v75 之后测试更局部的证书假设：如果整张 local patch prior atlas 太激进，是否可以只把 policy-val 中逐 bin 有正收益的 prior bins 混入 zero-blend atlas。它新增 `policy_val_prior_bin_gain_hybrid` 路径，并把 hybrid candidate 放入同一套 risk gate 中排序。

最终状态：

| item | value |
|---|---|
| W&B | `8qetk7tj` |
| hybrid selected | `true` |
| selected blend | `1.0` |
| selected alpha | `0.125` |
| candidate bins | `233306` |
| allowed bins | `13708` |
| allowed-bin fraction | `0.058755454` |
| target changed fraction | `0.065630289` |
| result | `26.753532 / 0.862111 / 0.251881` |
| status | completed; not promoted |

v76 的实际结果支持下面这个负结论：

> 局部 bin-gain certificate 真实生效，但当前阈值下的 policy-val 正收益不等于 held-out 泛化收益。很多被允许 bins 的样本和视角支持仍然过少，因此 v76 比 v75 zero-blend 更差。下一轮如果继续这条线，应要求更高样本数、更强多视角一致性和更大 gain margin；但论文主线仍应强调 residual representation capacity 和 target-view certificate，而不是把 v76 包装成突破。

---

## 14. 可能被 mentor 问到的问题

### Q1: 这是不是只是后处理？

答：Phase-J 的主收益是 render-time residual repair，所以它确实更接近 guarded adapter endpoint；但它不是无约束图像后处理，因为 residual 被 face/surface/evidence/policy gate 约束。表示级路线 v48-v76 正是在把它内化到 mesh-attached representation。

### Q2: 有没有 test leakage？

答：主方法的 branch、alpha、edge fallback、compaction ratio 都由 train 或 policy-val evidence 决定。held-out test GT 只用于最终评估。需要注意的是，v64 仍标为 report-only，因为规则是在看过早期 probe 后设计的，后续需要 fresh blind validation。

### Q3: 为什么 representation-level 还没有大幅超过？

答：当前 surface atlas 主要是 local residual lookup/calibration，容量和 coverage 都低于 Phase-J 的 render-time multi-view adapter。v65-v70 的负结果说明，仅调 alpha、shrink 或 blend ladder 不能解决容量瓶颈。v71a 进一步证明，即使把 prior 启用条件做成 evidence-consistent gate，policy-val 仍会选择 zero-blend。v72 又证明，即使强制局部非零 prior 并用 bin allowlist 筛选，target footprint 过小且 held-out 退化。v73 把 target-support 排序接入后，footprint 变大了，但指标仍只回到 zero-blend 行。v73b 再把 target-support 前置到 cheap pre-rank，成功剪掉弱 support candidate，但指标仍未提升。v74 说明 residual cap 放宽也不是当前 `counter` 的瓶颈，因为 `0.12/0.18/0.24` 在同一 blend 下 policy-val 分数完全相同。v75 进一步说明 local patch prior 虽然能覆盖大量 low-support bins，但 policy 仍拒绝非零 blend。v76 又证明，逐 bin policy-val 正收益如果缺少足够多视角/样本证书，仍可能在 held-out 上退化。所以瓶颈不是一个标量 gate 阈值，也不只是 footprint 或 coarse prior，而是 residual 表示容量和 target-view support certificate 的联合问题。

### Q4: 能不能声称全面超过 MeshSplatting？

答：在本地 selected-clean full9 口径，Phase-J 是 `9 / 9` 三指标严格胜出并且减少 triangles。对 MeshSplatting paper table 可以附录展示正向对比，但主 claim 应该限定为本地同协议复现。

### Q5: 当前是不是足够投顶会？

答：当前主结果足够支撑一次强汇报，但还不建议包装成顶会终局。顶会级闭环还需要把主要收益从 render-time adapter 更强地迁移到 representation-level method，或者把 adapter 本身包装成有清晰理论和应用价值的 certified rendering module。

---

## 15. 推荐汇报话术

开场：

> 我们不是要替代 MeshSplatting，而是把一个已经训练好的 MeshSplatting checkpoint 升级成可自检、可压缩、可修复、可回退的表示系统。

方法：

> 关键是把训练视角 residual 投回 mesh surface，形成 face/bin/view-risk 证据。只有当多视角证据显示某个 surface 区域稳定可修时，我们才迁移 residual；如果风险高，就回退。

结果：

> 在本地 full9 same-protocol selected clean MeshSplatting baseline 上，Phase-J 做到 9/9 场景三指标胜出，per-view 244/246 胜出，同时平均删除 7.6479% triangles。

限制：

> 目前最强结果仍来自 guarded adapter endpoint。表示级 residual atlas 已经实现了完整接口和安全 policy，但收益还小。下一步要做的是提升 residual representation capacity，而不是继续手动调参数。

---

## 16. 结论

当前最稳的 story 是：

> SPCarNet demonstrates that a trained MeshSplatting representation can be upgraded by train-only surface evidence into a geometry-safe, residual-aware, self-auditing rendering system. On local full9 Mip-NeRF360 reproduction, the Phase-J endpoint strictly improves all scenes over selected clean MeshSplatting while reducing triangles. The ongoing representation-level track has implemented the right interfaces and exposed the core bottleneck: the next breakthrough must increase surface residual capacity and target-view support certification, not continue scalar parameter tuning.

当前工程/论文闭环状态：

```text
Final status for paper loop: NOT COMPLETE

Reason:
Phase-J is strong and presentation-ready, but the representation-level endpoint
has not yet achieved paper-level effect size. v64 is the best fixed representation
policy; v65-v76 are completed diagnostics; v71a selected zero blend on
counter/kitchen, v72 forced a local nonzero prior allowlist and still
regressed on held-out metrics, v73 added target-support ranking but tied
the zero-blend v70/v71a row, and v73b added cheap target-support pre-rank but
again tied v73. v74 added a residual delta-cap ladder but selected the original
0.12 cap and tied v73b. v75 added a local patch prior but selected zero blend
and tied v74/v73b. v76 selected a policy-val bin-gain hybrid atlas but regressed
against v75 zero-blend and stayed below v64/v56. None of them is promoted.
```
