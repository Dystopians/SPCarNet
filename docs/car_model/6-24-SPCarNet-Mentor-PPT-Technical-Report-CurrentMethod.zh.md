# SPCarNet 当前方法技术报告（Mentor/PPT 汇报版）

日期：2026-06-24  
用途：给 mentor 汇报、制作 PPT、说明当前方法的技术贡献、实验收益、未解决短板和下一步计划。  
当前可汇报主线：**Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**。  
当前最佳表示级候选：**v64 fixed auto bin-alpha policy**，但只作为 report-only representation-level 候选。  
当前最新诊断：**v76 policy-val bin-gain hybrid prior**，已完成 W&B 验证，未晋级。

---

## 1. 一页结论

SPCarNet 的核心定位是：

> 在已经训练好的 MeshSplatting 模型之上，利用训练视角的 surface evidence 自动判断哪里可以安全压缩、哪里存在稳定 residual 可以修复、哪里证据不足必须回退。

与最基础的 MeshSplatting 相比，当前最强的 Phase-J endpoint 在本地同协议 Mip-NeRF360 full9 selected-clean baseline 上实现：

| 指标 | 当前结果 |
|---|---:|
| 场景数 | `9 / 9` |
| scene-level PSNR/SSIM/LPIPS strict wins vs selected clean MeshSplatting | `9 / 9` |
| per-view PSNR/SSIM/LPIPS strict wins vs selected clean MeshSplatting | `244 / 246` |
| mean dPSNR vs selected clean | `+1.331084` |
| mean dSSIM vs selected clean | `+0.034702` |
| mean dLPIPS vs selected clean | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

最重要的汇报边界：

- 可以强 claim：当前 Phase-J 是一个强的、可审计的 MeshSplatting 后处理/修复 endpoint，在本地 selected-clean full9 口径下同时提升 RGB 指标并减少 triangles。
- 不能过度 claim：当前主要收益仍来自 render-time evidence-lumigraph adapter，不是已经完全 baked into checkpoint 的终局表示。
- 表示级路线已经完成大量真实接口和诊断，v64 是当前最稳固定 policy，但效果量级很小，不能单独作为顶会级最终方法。
- v65-v76 的负结果很有价值：它们证明瓶颈不是继续扫 alpha、blend、cap 这些标量，而是 residual representation capacity 和 target-view generalization certificate。

---

## 2. 给 PPT 的 30 秒版本

建议开场这样讲：

> MeshSplatting 已经能产生强的显式 mesh 表示，但它的 clean checkpoint 仍存在局部纹理残差和几何冗余。SPCarNet 不重新训练一个完全不同的模型，而是让训练视角中的 surface evidence 反过来审计这个 mesh：多视角证据稳定的位置可以修复，低风险面片可以压缩，证据不足的位置自动回退。当前 Phase-J 在本地 full9 Mip-NeRF360 selected-clean baseline 上 9/9 场景三指标胜出，同时平均删除 7.65% triangles。

一句英文版：

> SPCarNet turns a trained MeshSplatting model into a self-auditing surface system: it repairs only where training-view surface evidence certifies stable residuals, compacts only low-risk triangles, and falls back when the certificate is weak.

---

## 3. 与 MeshSplatting 的区别

基础 MeshSplatting：

```text
images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render test views
```

SPCarNet：

```text
images + cameras
  -> train/load MeshSplatting checkpoint
  -> build train/policy-val surface evidence
  -> choose safe triangle compaction
  -> estimate stable residual repair from training views
  -> select adapter/alpha/fallback by train-only policy gate
  -> render held-out views
```

关键差异：

| 维度 | MeshSplatting | SPCarNet |
|---|---|---|
| 表示基础 | 显式 mesh/splat representation | 继承 MeshSplatting checkpoint |
| 是否利用训练残差做审计 | 不显式做 surface-level residual audit | 构建 face/bin/residual/support/risk evidence |
| 几何压缩 | baseline checkpoint 固定 | train-only quality-first compaction |
| 外观修复 | 由原 checkpoint 直接渲染 | stable residual transfer + policy gate |
| 安全机制 | 训练收敛和 checkpoint 选择 | train/policy-val gate、tail risk、fallback |
| 当前收益来源 | 原始 MeshSplatting 表示 | compact checkpoint + guarded residual repair |

---

## 4. 方法总览

SPCarNet 当前由五个主要模块组成。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Evidence Cache | train renders、GT、surface maps、camera | residual、face id、UV/bin、visibility、risk | 把训练视角变成可审计证据 |
| Geometry-Safe Compaction | mesh checkpoint + evidence | compact checkpoint | 删除低风险冗余 triangles |
| Evidence Lumigraph Adapter | compact render + train residual evidence | repaired render | 把稳定 surface residual 转移到 target view |
| Train-Only Policy Gate | policy-val metrics、view-tail risk | accept / fallback / no-op | 避免 test leakage 和不安全修复 |
| Surface Residual Atlas | face/UV/bin residual evidence | persistent residual field | 当前表示级内化路线 |

### 4.1 Evidence Cache

Evidence cache 是方法的数据底座。它记录训练和 policy-val 视角中的：

- rendered RGB；
- ground-truth RGB；
- residual map：`GT - Render`；
- alpha、visibility、depth；
- face id、barycentric 或 UV/bin；
- normal、view direction、camera center；
- per-face / per-bin support count；
- residual sign consistency；
- per-view PSNR、SSIM、LPIPS、image L1；
- min-view 和 CVaR tail risk。

这一步的意义是把训练数据从“只用于优化网络/参数”升级为“用于审计 surface 是否可靠”的证据。

### 4.2 Geometry-Safe Compaction

压缩原则是 quality-first，而不是最大化删面：

```text
only remove triangles when multi-view evidence says the edit is low-risk
```

具体做法：

- 低可见性、低贡献、低风险 faces 优先删除；
- 遮挡边界、thin structure、sparse geometry 区域受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- RGB、几何、topology 分开审计；
- 一旦 policy-val 发现风险，回退更保守版本。

当前 Phase-J 平均删除 triangles 的占比是 `7.6479%`。这里的 triangle reduction 指“删去的比例”，不是剩余比例。

### 4.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 提升来自 guarded Evidence Lumigraph Adapter。直观理解是：

```text
target render = compact render + train-evidence residual correction
```

简化公式：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 像素；
- `residual_i = GT_i - Render_i` 来自训练视角；
- `u_i` 是与 target pixel 对应或相近的 surface location；
- `w_i(p)` 由 visibility、surface correspondence、support 和风险门控决定；
- `alpha` 由 train/policy-val evidence 选择；
- 证据不足时自动 fallback 或 no-op。

这不是普通 image enhancement，因为 residual 被 surface、face id、visibility 和 train-only policy gate 约束。

### 4.4 Train-Only Policy Gate

公平性规则：

- branch、alpha、edge fallback 和 compaction ratio 不用 held-out test GT 选择；
- held-out test 只用于最终报告；
- baseline envelope 用 held-out test 选择最强 clean checkpoint，是为了不低估 baseline，不参与我们方法调参；
- W&B 记录中程/长程实验，便于追溯命令、GPU、配置和结果。

当前 Phase-J 中：

- `8 / 9` 场景走 adaptive-alpha ELA；
- `treehill` 走 train-selected edge fallback；
- fallback 的存在是方法安全性的一部分，不是失败后手动挑图。

---

## 5. 主结果：Phase-J vs selected clean MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- clean baseline candidates：local clean `26000` 和 `30000`；
- baseline selection score：`PSNR + 20 * SSIM - 20 * LPIPS`；
- 选择 baseline 时使用 held-out test metrics，是为了构造 strongest clean baseline envelope；
- 我们方法的 branch/policy 不使用 held-out test metrics。

Phase-J full9 主表：

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
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

Aggregate：

| 指标 | 值 |
|---|---:|
| mean dPSNR vs selected clean | `+1.331084` |
| mean dSSIM vs selected clean | `+0.034702` |
| mean dLPIPS vs selected clean | `-0.063359` |
| mean dPSNR vs source ELA | `+0.833143` |
| mean dSSIM vs source ELA | `+0.018946` |
| mean dLPIPS vs source ELA | `-0.039986` |
| mean triangle reduction | `7.6479%` |

证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

---

## 6. 与 MeshSplatting paper table 的对比

本地还有一个同协议 paper-table audit，使用的是早期 Compact-ELA/SOR 版本，不是最新 Phase-J endpoint。它相对 MeshSplatting paper table 同样为正：

| 评估口径 | 结果 |
|---|---:|
| RGB wins vs selected clean MeshSplatting | `9 / 9` |
| mean dPSNR vs selected clean | `+0.497941` |
| mean dSSIM vs selected clean | `+0.015755` |
| mean dLPIPS vs selected clean | `-0.023373` |
| mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| mean triangle reduction | `5.7632%` |

需要这样向 mentor 解释：

- 主 claim 应该放在本地 selected-clean full9 同协议复现上，因为这是我们完整可追溯的 baseline。
- paper table 可以作为辅助对比，证明数值量级并不弱；但如果做论文最终表，需要严格确认与 paper 的 split、mask、metric implementation、checkpoint 迭代完全一致。
- 当前 Phase-J 主结果比早期 Compact-ELA/SOR 更强，但 paper-table 对齐报告暂时不是 Phase-J 版本。

证据路径：

```text
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md
```

---

## 7. 表示级路线：v48 到 v76

这条线的目标是把 Phase-J 的 render-time residual repair 尽量内化到 persistent mesh/surface representation。当前结论是：接口和审计已经成熟，但 effect size 仍远小于 Phase-J。

| 版本 | 关键改动 | 结论 |
|---|---|---|
| v48 | auto-support surface residual atlas | full9 正向，但 effect size 小 |
| v51 | support-footprint ladder | cap-hit scenes 有改善，均值不如 v48 |
| v52 | capacity-aware policy | train-only 选择 v48/v51，安全但小收益 |
| v55d | per-face/local alpha calibration | `counter` 正向，但 `kitchen/bonsai` 不稳 |
| v56 | face-alpha reliability guard | 只选择 `counter`，`9/9` non-regressive/tie |
| v57a | reliability shrink | 接口有效，但弱于 raw v55d |
| v58-v64 | view-conditioned basis、OOD guard、face/bin alpha | v64 成为当前最稳 fixed policy |
| v65 | teacher-distilled shared basis | 没有超过 v64 |
| v67-v68 | uncertainty shrink / keep-downweight | 诊断有用，未晋级 |
| v69-v71a | count-pyramid multi-scale prior + blend ladder + evidence gate | policy 自动选择 `blend=0.0` |
| v72 | local prior allowlist | target footprint 太小，未晋级 |
| v73 | target-support candidate selection | support 选择生效，但指标仍低于 v64/v56 |
| v73b | target-support pre-rank | cheap pre-rank 生效，减少昂贵 refit，但不提升指标 |
| v74 | residual delta-cap ladder | 证明 `0.12` cap 不是主要瓶颈 |
| v75 | same-face local patch prior | nonzero prior 覆盖大量 bins，但 policy 仍选 `blend=0.0` |
| v76 | policy-val bin-gain hybrid prior | hybrid atlas 被选中，但 held-out 低于 v75 zero-blend 和 v64/v56 |

当前最佳固定表示级结果：v64 fixed auto bin-alpha policy。

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |

v64 的诚实状态：

- fixed rule 不读取 held-out metrics；
- materialized selected full9 tree 已完成；
- selection uses held-out metrics = `False`；
- 但因为 guard 是在早期 probe 后设计的，仍标为 report-only candidate，需要 fresh blind/long-run validation 才能作为 paper endpoint。

证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/v64_bin_alpha_auto_policy_pipeline_report.md
```

---

## 8. 最新诊断：v75/v76 local prior certificates

v75 的问题假设：

> v70-v71a 的 count-pyramid prior 可能太粗，导致低支持 UV/bin 的 residual 估计不准。能否改成 same-face local UV patch prior？

新增接口：

- `surface_multiscale_prior_mode=local_patch`
- same-face local patch prior statistics；
- blend candidates：`0,0.5,1.0`；
- 复用 target-support pre-rank、policy-val gate、delta cap 和 W&B logging。

`counter` 实验结果：

| method | PSNR | SSIM | LPIPS | 结论 |
|---|---:|---:|---:|---|
| v75 local patch prior | `26.753995895` | `0.862119257` | `0.251853049` | 完成，不晋级 |
| v74/v73b/v73/v70/v71a zero-blend 行 | `26.753995895` | `0.862119257` | `0.251853049` | 完全持平 |
| selected v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | 仍更强 |

候选诊断：

| blend | blended bins | blended fraction | policy-val best relative gain | selected |
|---:|---:|---:|---:|---|
| `0.0` | `0` | `0.000000` | `0.026849788` | yes |
| `0.5` | `951427` | `0.655469` | `0.026670204` | no |
| `1.0` | `951427` | `0.655469` | `0.026489316` | no |

结论：

- local patch prior 确实生效，非零 blend 覆盖约 `65.5469%` low-support bins；
- 但 train-policy 仍选择 `blend=0.0`，held-out 指标回到 zero-blend；
- 这排除了“count-pyramid 太粗”这个单因子解释；
- 下一步应该构建更高容量、更可泛化的 surface-conditioned residual representation，而不是继续扩大 scalar blend/cap sweep。

证据路径：

```text
docs/car_model/6-24-v75-LocalPatchPrior-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md
```

v76 在 v75 之后进一步测试更局部的证书：

> 如果整张 local patch prior atlas 太激进，是否可以只把 policy-val 中逐 bin 有正收益的 prior bins 混入 zero-blend atlas？

新增接口：

- `enable_policy_val_prior_bin_gain_hybrid`；
- 逐 bin 比较 zero-blend atlas 与 nonzero local-patch prior atlas；
- `prior_bin_gain_hybrid_min_bin_samples`；
- `prior_bin_gain_hybrid_min_relative_gain`；
- `prior_bin_gain_hybrid_min_positive_view_fraction`；
- hybrid candidate 进入同一套 train-policy risk gate 和 W&B logging。

`counter` 实验结果：

| method | PSNR | SSIM | LPIPS | 结论 |
|---|---:|---:|---:|---|
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | 完成，不晋级 |
| v75 local patch / zero-blend row | `26.753995895` | `0.862119257` | `0.251853049` | 更强 |
| selected v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | 仍更强 |

hybrid 诊断：

| item | value |
|---|---:|
| candidate bins | `233306` |
| allowed bins | `13708` |
| allowed-bin fraction | `0.058755454` |
| selected blend | `1.0` |
| selected alpha | `0.125` |
| target changed fraction | `0.065630289` |

结论：

- v76 真实跑通并选中了 hybrid atlas；
- 但 policy-val 的微小局部收益没有转化成 held-out test 收益；
- 当前 bin certificate 对少样本/单视角 bins 仍太弱；
- 这进一步说明下一步必须提高 residual representation capacity 和 target-view generalization certificate，而不是继续做标量 sweep。

证据路径：

```text
docs/car_model/6-24-v76-PolicyValBinGainHybrid-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md
```

---

## 9. 定性材料建议

PPT 中不建议只放全图对比，因为当前很多收益是局部 residual-level 改善，全图肉眼差异会偏弱。建议按下面顺序放图：

| 图 | 用途 | 路径 |
|---|---|---|
| 全图 fair comparison | 证明同一 held-out view / same baseline | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| Phase-J where-it-helps | 展示局部 error reduction | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| outdoor detail showcase | 针对室外弱项展示局部收益 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| mixed local showcase | 展示室内/室外局部修复 | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| v56 counter panel | 表示级路线的局部例子 | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |
| v52 capacity panel | capacity-aware policy 的 cap-hit 例子 | `assets/spcarnet_v52_capacity_policy_cap_hit_panel.png` |

建议讲法：

- 全图图像用于证明公平对比，不用于强调视觉冲击。
- 局部 crop / error map 用于解释“方法在哪里起作用”。
- 如果 mentor 问为什么视觉差异不明显，应直接承认：当前 Phase-J 指标强，但很多是 residual-level correction；表示级路线还需要更大容量才能形成肉眼更强的 qualitative story。

---

## 10. 为什么这不是简单调参

当前工作从 v48 到 v76 已经把多个真实 train/eval pipeline 接口补齐：

- surface-addressed residual atlas；
- support expansion；
- capacity-aware policy；
- per-face local alpha；
- face/bin reliability guard；
- view-conditioned basis；
- OOD guard；
- bin-level alpha calibration；
- multi-scale prior；
- evidence-consistent gate；
- target-support candidate selection；
- target-support cheap pre-rank；
- residual delta-cap ladder；
- local patch surface prior；
- policy-val bin-gain hybrid prior。

这些模块的价值不是每一个都提升指标，而是逐步定位了瓶颈：

| 失败假设 | 被哪个实验排除 | 结论 |
|---|---|---|
| 只要调高 alpha 就能提升 | v53-v57a | 全局或局部 alpha 会引入 view-tail risk |
| 只要用 view-conditioned basis 就能提升 | v58-v60 | OOD/target 泛化不足 |
| 只要放宽 cap 就能提升 | v74 | `counter` 上 cap `0.12/0.18/0.24` 基本同分 |
| 只要加入 coarse prior 就能提升 | v69-v71a | policy 自动拒绝非零 blend |
| count-pyramid prior 太粗是主因 | v75 | local patch prior 仍未晋级 |
| target footprint 太小是唯一主因 | v73/v73b | footprint 增大但指标未超过 v64/v56 |
| 逐 bin policy-val gain 足够安全 | v76 | hybrid 被选中但 held-out 退化 |

所以当前判断是：

> 下一阶段的关键不应是继续手动找参数，而应升级 residual representation capacity，并为 target-view generalization 建立更强证书。

---

## 11. 当前短板

| 短板 | 影响 | 应对口径 |
|---|---|---|
| 主收益仍来自 render-time adapter | 论文贡献容易被质疑为后处理 | 把 Phase-J 讲成 certified rendering endpoint，同时继续推进 representation-level 内化 |
| 表示级收益太小 | v64 mean gain 只有 `+0.000410080` PSNR vs v56 | 作为 ablation/diagnostic，不作为 headline |
| 定性视觉差异有时不明显 | mentor 可能觉得图上看不出强提升 | 用 error-reduction crop 和 train-defined local support 展示 |
| v64 仍是 report-only | guard 在 probe 后设计，有后验风险 | 需要 fresh blind/long-run validation |
| paper-table 对齐不完全闭合 | 最新 Phase-J 没有完全转成 paper-table audit | 主 claim 放 local same-protocol，paper-table 作为附录辅助 |
| target-view 泛化证书仍弱 | v73/v75/v76 说明 footprint/prior/bin-gain 都不是充分条件 | 需要更强 surface-conditioned residual basis 和多视角 support certificate |

---

## 12. 推荐 PPT 结构

建议 18 页左右。

| 页 | 标题 | 核心内容 | 推荐素材 |
|---:|---|---|---|
| 1 | Title | SPCarNet: evidence-certified MeshSplatting repair and compaction | 方法名 + 一句话 |
| 2 | Problem | clean MeshSplatting 仍有 residual 和冗余 triangles | baseline render/error map |
| 3 | Key Idea | train-view evidence tells us where to repair/compact/fallback | pipeline diagram |
| 4 | Difference from MeshSplatting | baseline vs SPCarNet | 本报告第 3 节 |
| 5 | Evidence Cache | residual、face、UV/bin、support、risk | evidence visualization |
| 6 | Geometry-Safe Compaction | quality-first triangle deletion | triangle reduction table |
| 7 | Guarded Residual Repair | Evidence Lumigraph Adapter | residual transfer equation |
| 8 | Policy Gate | train-only branch/alpha/fallback | policy flowchart |
| 9 | Main Result | Phase-J `9/9` wins and `7.6479%` triangle reduction | main table |
| 10 | Per-Scene Results | full9 per-scene metrics | scene table |
| 11 | Qualitative | full-frame plus local error reduction | qualitative assets |
| 12 | Paper Table Context | same-protocol vs paper table | auxiliary table |
| 13 | Representation Track | v48-v76 evolution | version table |
| 14 | Best Fixed Policy | v64 summary | v64 table |
| 15 | Negative Diagnostics | v70-v76 show bottleneck | diagnostic table |
| 16 | Why Research, Not Engineering | evidence certificates, risk gates, ablations | bullets |
| 17 | Limitations | render-time endpoint, small representation gains | honest boundary |
| 18 | Next Step | high-capacity surface residual representation | future plan |

---

## 13. Mentor 可能追问与建议回答

### Q1: 这是不是只是图像后处理？

答：Phase-J 的主要收益确实是 render-time adapter，但不是无约束图像增强。它的 residual 来自训练视角，必须通过 face/surface correspondence、visibility、support、policy-val risk gate 和 fallback。更准确的说法是 certified surface-evidence rendering repair。

### Q2: 有没有 test leakage？

答：方法选择 branch、alpha、fallback、compaction ratio 时不用 held-out test GT。held-out test 只用于最终评估。selected-clean baseline 用 held-out test 选择更强 baseline envelope，是为了公平地不低估 MeshSplatting。

### Q3: 为什么定性图不够震撼？

答：当前指标提升来自许多局部 residual-level 改善，全图尺度上不一定肉眼明显。应该展示局部 error map 和 train-defined support crop。这个现象也说明下一阶段需要更高容量的 representation-level residual basis，才能产生更强视觉差异。

### Q4: 当前是否全面超越 MeshSplatting？

答：在本地 same-protocol full9 selected-clean baseline 上，Phase-J 是 `9 / 9` 场景三指标胜出并减少 triangles。更宽泛的 paper-level claim 还需要把最新 Phase-J 完全转成 paper-table 同口径审计，并补 fresh/blind validation。

### Q5: v64/v75/v76 为什么不能作为最终方法？

答：v64 是当前最稳 fixed representation policy，但 effect size 太小，而且仍是 report-only。v75 是结构性诊断：local patch prior 生效但未提升。v76 进一步证明逐 bin policy-val gain certificate 真实接入但还不能保证 held-out 泛化。它们支持论文中的分析故事，但不能替代 Phase-J 作为当前 headline。

### Q6: 下一步最值得做什么？

答：不要继续标量调参。应从 residual representation capacity 入手，例如 policy-val bin-gain hybrid、surface-conditioned residual basis、view-aware but OOD-guarded local field，目标是把 Phase-J 的收益更强地迁移到 persistent representation。

---

## 14. 汇报时建议使用的三句话

1. SPCarNet 的核心不是“训练更久”，而是把训练视角 residual 投回 surface，形成可审计的修复和压缩证书。
2. 当前 Phase-J 在本地 full9 selected-clean MeshSplatting baseline 上 `9/9` 场景三指标胜出，同时平均删除 `7.6479%` triangles。
3. 表示级路线已经证明接口和安全 policy 可行，但 effect size 还小；v65-v76 的主要价值是证明真正瓶颈在 residual capacity 和 target-view 泛化证书，而不是继续调 alpha、blend 或 cap。

---

## 15. 证据索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

paper-table 同协议辅助对比：

```text
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md
```

当前完整长报告：

```text
docs/car_model/6-24-SPCarNet-Current-Method-Full-Technical-Report-ForMentor.zh.md
```

v64 表示级最佳固定 policy：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/v64_bin_alpha_auto_policy_pipeline_report.md
```

v75/v76 最新诊断：

```text
docs/car_model/6-24-v75-LocalPatchPrior-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v75_local_patch_prior_20260624/summary.md
docs/car_model/6-24-v76-PolicyValBinGainHybrid-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md
```

定性素材：

```text
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
```

---

## 16. 当前最终状态

```text
Final status for paper loop: NOT COMPLETE
```

原因：

- Phase-J 已经足够作为当前强汇报 endpoint；
- 但论文终局还没有完全闭合，因为主要收益仍是 render-time adapter；
- representation-level endpoint 目前最佳 v64 的收益量级太小；
- v76 最新 policy-val bin-gain hybrid 已证明局部 bin-gain certificate 仍不能单独解决 bottleneck；
- 下一阶段必须进行 residual representation capacity 和 target-view certificate 的实质升级。

当前适合对 mentor 的表述：

> 我们已经有一个很强、可审计、结果完整的 MeshSplatting repair/compaction endpoint，可以作为当前进展主线汇报。但如果目标是顶会最终方法，还需要把表示级方法继续做强，或者把 certified rendering adapter 本身包装成更完整、更有理论和应用价值的贡献。
