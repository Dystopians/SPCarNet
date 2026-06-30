# SPCarNet 完整评估报告：Metrics / 工程闭环 / 论文级可用性

Date: 2026-06-28

## 2026-06-29 v249-v252 v169 Representation Gate Update

新增日志：`docs/car_model/6-29-v249-v252-v169-RepresentationGate-Log.md`。

新增机器可读汇总：`docs/car_model/results/v249_v252_v169_representation_gate_summary.json`。

这轮严格参考 `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` 执行：先做 flowers policy-val、teacher residual projection audit 和表示层改动；未通过 Phase-J all-axis gate 前不启动 full9。

代码层面新增了真实方法与协议修复：

- `scripts/car_model/train_surface_conditioned_residual_unet.py` 新增 train-fit-only `teacher_benefit_mask_mode`，只在 Phase-J teacher 相对 parent 真正有收益的区域学习 teacher residual，其余区域训练成 parent/no-op。
- 默认把 `alpha=0` 从 policy best 选择中排除，避免失败方法被 no-op 伪装成“最优策略”；alpha-0 仍保留在诊断 rows 中。
- checkpoint 现在显式保存 `surface_evidence_stats`，独立 checkpoint apply 可以重建 surface-evidence model。
- 报告中明确标注 Phase-J flowers reference 只是 numeric reference，不能替代 official flowers exact。

关键结果是负向但清晰：

| run | change | alpha | PSNR gain | SSIM gain | LPIPS gain | changed | energy retention | cosine | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v249a | LPIPS no-harm GT-assisted U-Net | 0.25 | +0.027357 | +0.000589 | +0.000250 | 0.246711 | 0.020147 | 0.127558 | tails fail |
| v250a | edge/confidence memory texture | 0.125 | +0.007847 | -0.000152 | -0.000019 | n/a | active 0.048002 | active 0.284919 | SSIM/LPIPS fail |
| v250b | raw-RGB memory texture | 0.125 | +0.007915 | -0.000107 | -0.000004 | n/a | active 0.031182 | active 0.295237 | SSIM/LPIPS fail |
| v251a | low-rank K=4 surface texture | 0.0 | +0.000000 | +0.000000 | +0.000000 | 0.000000 | n/a | n/a | strict policy selects no-op |
| v251b | surface texture U-Net evidence | 0.0 | +0.000000 | +0.000000 | +0.000000 | 0.000000 | n/a | n/a | strict policy selects no-op |
| v252a | low-rank + teacher-benefit mask | 0.0625 | +0.000094 | +0.000002 | +0.000002 | 0.000369 | 0.000019 | 0.021462 | near no-op |
| v252b | surface U-Net + teacher-benefit mask | 0.0625 | +0.000382 | +0.000011 | +0.000004 | 0.003078 | 0.000158 | 0.026398 | near no-op |

结论：Phase-J teacher signal 很强（v251/v252 policy-val teacher headroom 约 `+0.913279 PSNR / +0.065512 SSIM / +0.017600 LPIPS`），但当前 baked surface RGB residual carrier 无法可靠承载这个信号。v252 的 teacher-benefit mask 确实降低了尾部破坏，但把 residual magnitude 压到接近 no-op；projection audit 证明 energy retention 低到 `0.000019` 和 `0.000158`，不是 full9 没跑导致的证据缺口。

当前状态仍是 `NOT COMPLETE for paper-level all-axis win`。但对于 v169 prompt 的诊断标准，已经满足 B 类结论：当前 carrier family 无法在无 target/test GT 泄漏的前提下稳定改善 SSIM/LPIPS。下一步不应继续 alpha、face gate、support threshold 或 full9 promotion，而应换更强的 view-dependent source-feature/deferred surface renderer 表示。

## 2026-06-29 v246-v247 Source Evidence Bank / Projection Loss Update

新增日志：`docs/car_model/6-29-v246-v247-SourceEvidenceBank-ProjectionLoss-Log.md`。

新增机器可读汇总：`docs/car_model/results/v246_v247_sourcebank_projection_loss_summary.json`。

这轮严格按 `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` 的早停逻辑推进：先做 flowers policy-val 和 residual projection audit，未过 all-axis gate 就不启动 full9。

代码层面已经实现了真实方法改动：

- `scripts/car_model/train_surface_conditioned_residual_unet.py` 新增 source-evidence-bank conditioning，以及 teacher residual cosine / energy projection losses，并把对应指标写入 W&B。
- `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py` 补齐 `surface_texture_unet` evidence stats 加载和 no-GT apply 支持。

但质量结论是负向的：

| run | policy alpha | PSNR gain | SSIM gain | LPIPS gain | min SSIM | min LPIPS | projection energy | projection cosine | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v246a source-bank no-prior | 0.5 | +0.052707 | +0.002253 | +0.001792 | +0.000603 | -0.000030 | 0.074658 | 0.124085 | weak; strict all-axis not certified |
| v247a projection-loss GT-assisted | 0.5 | +0.037900 | +0.000707 | +0.001625 | -0.000346 | -0.001074 | 0.085702 | 0.121128 | failed tail/all-axis |
| v247b teacher-only projection-loss | 0.0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.083270 | 0.115199 | selected no-op |

Projection audit 显示 v247a 只保留约 `8.57%` Phase-J teacher residual energy，cosine 只有 `0.121`，且相对 teacher 的 image metrics 变差：`-0.054637 PSNR / -0.001954 SSIM / -0.000028 LPIPS gain`。v247b teacher-only ablation 同样失败。target no-GT precheck 通过，target/test GT 没有进入 apply；target apply 是因为 policy-val all-axis gate 失败而跳过。

结论：v246-v247 证明了新的工程接口和 projection loss 可以运行，但没有解决表示载体弱的问题。当前状态仍是 `NOT COMPLETE`。不要从该分支启动 full9；下一步必须换更强的 surface-attached decoder / patch-structure-aware representation，而不是继续 source-bank top-k、alpha 或 scalar residual sweep。

## 2026-06-29 v169 Teacher-Signal / Carrier Upper-Bound Update

新增诊断日志：`docs/car_model/6-29-v169-TeacherSignal-CarrierUpperBound-Diagnostics.md`。

这轮按 `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` 补齐了两个关键缺口：

- 新增 `scripts/car_model/analyze_v169_teacher_signal_audit.py`，正式审计 teacher-parent residual 是否非零、mask/clip 稀释、active coverage、policy-val PSNR/L1 gain，并显式记录 `target_or_test_gt_usage = none`。
- 修复 `scripts/car_model/train_surface_conditioned_residual_unet.py` 中 `_sample_patch` no-crop fallback 少传 `face_ids` 的接口 bug。

核心结果：

| scene | teacher signal | carrier upper-bound verdict |
|---|---|---|
| flowers | policy-val masked teacher PSNR gain `+0.841581`，mask L1 retention `0.574638` | full-image PSNR rescan 只有噪声级 all-axis 正数：`+0.000168 PSNR / +0.000000402 SSIM / +0.00000244 LPIPS`，robust gate 失败 |
| counter | policy-val masked teacher PSNR gain `+2.651850`，mask L1 retention `0.632333` | policy-val proxy robust 正：`+0.076027 PSNR / +0.00005296 SSIM / +0.00005631 LPIPS`，但这不能推翻 v192 target exact 仍未在 PSNR/SSIM 超 Phase-J |

结论：Phase-J teacher residual 确实存在，问题不是 teacher 没信号；真正短板是表示载体和泛化证书。flowers 当前 carrier 只在极小 alpha 下给出近似噪声级正数，不能据此启动 full9 promotion；counter policy-val 有信号，但 held-out target/test 对 Phase-J 的 all-axis 超越仍未闭环。因此状态仍是 `NOT COMPLETE`，下一步必须做 surface-attached feature texture / view-dependent low-rank basis 级别的表示升级，而不是继续 alpha、footprint 或 scalar atlas 微调。

## 2026-06-29 v169 更新

新增日志：`docs/car_model/6-29-v169-SurfaceUNet-Progress-And-Bottleneck-Log.md`。

更新后的结论是：v191 surface-conditioned residual U-Net 已经在 flowers exact 上通过 v169 固定 Phase-J all-axis gate（`20.606058 / 0.578882 / 0.323687` vs `20.304358 / 0.557770 / 0.329222`），但 counter 仍未全面超过 Phase-J。当前最好的 baked U-Net counter 是 v192：`28.097420 / 0.891432 / 0.184687`，它 LPIPS 优于 Phase-J counter `0.186472`，但 PSNR/SSIM 仍低于 Phase-J `28.449171 / 0.893731`。新增 v194 teacher-only ablation 为 `19.903099 / 0.510229 / 0.404076`，说明 v191 成功明显依赖 train-fit GT loss，不能包装成纯 teacher-only distillation。因此总体状态仍是 `NOT COMPLETE`，但 6-29 已经从“flowers 未过 Phase-J”推进到“flowers 过硬门槛，counter 与 teacher-only 消融成为主要瓶颈”。

## 直接回答：比 Phase-J 更弱，最新思路未成功达标

截至 2026-06-28，**当前 vNext / 新 prompt 方法比 Phase-J 更弱**。最新改进思路不是“完全无效”，但它只完成了工程机制推进，没有完成论文主结果意义上的质量突破，因此应判定为：

> **engineering-progress / quality-fail / NOT COMPLETE**

具体证据如下：

- **Phase-J full9**：`26.482766 / 0.783720 / 0.224261`，在 selected-clean MeshSplatting 口径下 `9 / 9` 场景三指标严格胜出，并有 `7.6479%` 平均三角形减少。
- **v166 flowers exact**：`20.452814 / 0.549059 / 0.355544`。
- **v167 flowers exact**：`20.452776 / 0.549059 / 0.355544`，但这是 fallback no-op 后的结果，真实 affine candidate 被 policy-val 拒绝。
- **v168 Phase-J distillation profile**：已完成 runner protocol、dry-run 和负向 parser guard；它不是 exact metric win，作用是把下一步 Phase-J-to-baked representation 的 teacher/parent/no-GT 约束固定下来。
- **Phase-J flowers**：`20.304358 / 0.557770 / 0.329222`。
- 因此 v166 flowers 只在 PSNR 上比 Phase-J 高 `+0.148457`，但 SSIM 低 `-0.008711`，LPIPS 差 `+0.026322`。这不是 all-axis win，不能说比 Phase-J 更强。
- v166 也没有超过 v165：v165 flowers 是 `20.452848 / 0.549059 / 0.355544`，v166 PSNR 还低 `0.000034`，SSIM/LPIPS 基本不变。
- v167 是更强 train-only affine/patch residual capacity 的首次完整闭环：它填充了 `313 / 393` eligible target-impact bins，但 policy-val 认为 SSIM/L1/tail-risk 变差并拒绝，最终 `changed_pixels=0` fallback no-op。因此 v167 是一次有价值的负结果，不是质量成功。

所以，最新思路的真实结论是：

1. **成功的部分**：strict no-target-GT verifier、target-impact footprint、train-only multisample/affine residual fill、manifest/W&B/audit 都跑通了；v166/v167 都是可复核的完整实验，v168 则把 Phase-J distillation 的公平接口固定成 runner profile。
2. **失败的部分**：v165-v167 没有把扩大后的 footprint 转化为可见质量提升，尤其没有改善 SSIM/LPIPS，也没有超过 Phase-J；v167 进一步说明单纯 per-face ridge/patch residual field 也会被 policy-val 识别为风险。
3. **下一步**：不能继续把主线放在扩大 footprint、调 alpha、局部均值 residual fill 或简单 face-local ridge field；必须用 v168 profile 转向更接近 Phase-J teacher distillation / stronger baked representation 的方案，并先在 flowers 上 all-axis 超过 Phase-J 后再进入 full9 promotion。

## Claim Readiness Matrix

自动版报告已生成：

- `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`
- 生成脚本：`scripts/car_model/build_spcarnet_claim_readiness_report.py`

| claim | 当前状态 | 可用证据 | 缺口 |
|---|---|---|---|
| Phase-J 是当前最强本地 RGB endpoint | 本地成立 | full9 `26.482766 / 0.783720 / 0.224261`，`9 / 9` 场景胜 clean | 必须说明它不是 baked representation |
| v106 是当前最强 verified baked representation | 部分成立 | full9 `25.831280 / 0.760830 / 0.268435`，胜 selected clean | 视觉优势弱，且低于 Phase-J |
| vNext/new prompt 可作为论文主方法 | 不成立 | 有 no-GT、manifest、audit、fallback 工程闭环 | v165-v167 未超 Phase-J，v168 还只是 dry-run |
| v168 是质量成功 | 不成立 | profile dry-run 和负向 parser guard 通过 | 缺 exact metrics、定性图和 ablation |
| 项目已 paper-final | 不成立 | 工程与文档进展显著 | 缺 all-axis win、固定 full9 promotion、强定性证据 |

## 结论先行

当前结论是：**还没有达到“论文终局闭环”**。
如果看本地 full9 RGB endpoint，当前最强闭环仍是 **Phase-J guarded adaptive edge policy**：它在 selected-clean MeshSplatting 口径上 `9 / 9` 场景严格三指标胜出，mean 为 `26.482766 / 0.783720 / 0.224261`，平均三角形减少 `7.6479%`。
如果只看“是否存在一个本地 full9 上超过 clean MeshSplatting 的表示级版本”，答案是 **有**：当前最强、可验证的 baked representation 结果仍然是 **v106 POD-MoE base-preserve**，它在本地 selected full9 口径上超过 clean MeshSplatting，但低于 Phase-J endpoint。
如果看“新一代 vNext certified residual surface texture / 新 prompt 方法是否已经成为可推广、可写成论文主结果的方法”，答案是 **还没有**：vNext 的工程协议很强，但已完成 full9 metrics 低于 clean MeshSplatting、v106 和 Phase-J。v165 flowers exact run 把 target changed pixels 从 v164 的 `860` 扩大到 `8324`，约 `9.68x`，但 PSNR 只提升 `+0.000051`；v166 加入 train-only target-impact multisample residual fill 后仍为 `20.452814 / 0.549059 / 0.355544`；v167 加入 train-only affine/patch residual field 后被 policy-val 拒绝并 fallback no-op，为 `20.452776 / 0.549059 / 0.355544`。v168 目前只是 Phase-J distillation protocol dry-run，不是质量结果。这说明当前瓶颈已经从“完全改不到”转为“能安全改动，但 residual 表示强度不足，改动不能转化为视觉/感知收益”。

一句话评价：

- **Metrics 层面**：Phase-J 是当前最强本地 RGB endpoint；v106 达到本地 baked-representation baseline 超越；vNext/new prompt 尚未达标。
- **工程层面**：vNext 的审计、manifest、strict no-target-GT apply、target-evidence verifier、fallback/no-op 和 W&B 记录已经接近论文级工程框架，但 runtime、存储稳定性和全场景 promotion 仍是明显短板。
- **论文层面**：目前可以讲一个“逐步走向可审计修复/压缩”的研究故事，但不能诚实宣称 vNext 已经全面胜出；paper-final 状态仍是 `NOT COMPLETE`。

## Phase-J 对比判定：当前更弱，不应包装成成功

直接回答当前最关键的问题：

- **和 Phase-J 相比，当前新 prompt / vNext 路线还更弱。**
- **最新改进思路在工程机制上是有效推进，但按论文主结果标准还没有成功。**
- **不能把 vNext/new-prompt 结果写成已经超过 Phase-J；截至 2026-06-28，最诚实的表述是 `NOT COMPLETE / v167 exact failed to beat Phase-J; v168 protocol dry-run only`。**

原因很明确：

1. Phase-J full9 是当前本地最强 RGB endpoint：`26.482766 / 0.783720 / 0.224261`，相对 clean MeshSplatting 为 `+1.331084 / +0.034702 / -0.063360`，并且 `9 / 9` 场景严格三指标胜出。
2. 当前最强 baked representation v106 是 `25.831280 / 0.760830 / 0.268435`，虽然超过 clean MeshSplatting，但仍比 Phase-J 低 `0.651486` PSNR、`0.022890` SSIM，LPIPS 高 `0.044174`。
3. 已完成的 vNext full9 结果更弱：structure-aware shrink cleanup 为 `25.067699 / 0.741260 / 0.306689`，effective-margin gate 为 `25.067410 / 0.741259 / 0.306695`，二者都低于 clean MeshSplatting、v106 和 Phase-J。
4. flowers 单场景 v165 相对 Phase-J 是混合结果，不是胜利：v165 为 `20.452848 / 0.549059 / 0.355544`，Phase-J flowers 为 `20.304358 / 0.557770 / 0.329222`。v165 只有 PSNR 更高，SSIM 与 LPIPS 都明显更差，因此不能称为 all-axis 超过 Phase-J。
5. v166 `target-impact multisample residual fill` exact flowers run 已完整结束，manifest `COMPLETE` 且 errors `[]`。它通过了 strict no-target-GT verifier，执行了 target-impact multisample fill，并在内部 policy-val gate 上接受候选；但最终 test 指标为 `20.452814 / 0.549059 / 0.355544`，相对 Phase-J flowers `20.304358 / 0.557770 / 0.329222` 仍然只赢 PSNR、输 SSIM 和 LPIPS。因此 v166 不能替代 Phase-J 对比结论，也不能进入 full9 promotion。
6. v167 `target-impact affine/patch residual field` exact flowers run 也已完成，但 policy-val 拒绝候选并 fallback no-op，最终 `20.452776 / 0.549059 / 0.355544`，仍不是 all-axis win。
7. v168 只是把 Phase-J distillation profile 接入 runner，并通过 dry-run/负向 parser guard；它还没有 exact metrics，不能包装成质量成功。

因此，当前结论不是“最新方法已经击败 Phase-J”，而是：

> 最新方法把瓶颈从“无法安全扩大 target footprint”推进到“能 no-GT 地扩大和审计 footprint，但 residual 表示还没有把改动转化为足够强的 SSIM/LPIPS/视觉收益”。这是一个有价值的诊断和工程进展，但不是论文终局成功。

后续必须做的修复实验：

- v166 已完成并验证失败：不是运行没结束，而是结果没有达标。
- 下一步不能继续把重点放在只扩大 footprint 或微调 alpha；必须换更强的 train-only residual representation，使 SSIM/LPIPS 和视觉质量实际变好。
- 新方法必须先在 flowers 上 all-axis 超过 Phase-J，再进入 full9 promotion；否则继续全场景会浪费大量 GPU/CPU 时间。

## 当前最重要的数字

### Full9 汇总

| method | scenes | PSNR | SSIM | LPIPS | 相对 clean MeshSplatting | 当前角色 |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | 本地公平基线 |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | 稳定表示级 anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | 当前最强已验证 baked representation |
| Phase-J guarded adaptive edge policy | 9 | 26.482766 | 0.783720 | 0.224261 | +1.331084 / +0.034702 / -0.063360 | 当前最强本地 RGB endpoint/reference，不能直接混作 baked representation |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | 协议完整，但未推广 |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | 更安全，但更接近 no-op |

解释：

- v106 相对本地 clean MeshSplatting 是明确正向：PSNR/SSIM 更高，LPIPS 更低。
- v106 相对 v104c 只有小幅增益：`+0.002181` PSNR、`+0.000103` SSIM、`-0.000112` LPIPS；它是稳定提升，但不是“大幅颠覆”。
- Phase-J 相对 v106 仍高 `+0.651486` PSNR、`+0.022890` SSIM、`-0.044174` LPIPS；所以如果论文故事声称新 prompt 方法是主线，必须正面解释为什么它还没超过 Phase-J endpoint。
- vNext full9 低于 clean MeshSplatting，也低于 v106 和 Phase-J；它目前应被视为“工程协议和瓶颈诊断路线”，不能作为最终质量主线。

## Phase-J：必须纳入的新 prompt 对照

Phase-J 是当前本地 full9 RGB 闭环中最强的 endpoint 参照。它不是最终 desired baked representation，但它是我们承诺要击败的强基线之一，因为它在相同 selected-clean full9 split 下已经形成了完整 RGB、压缩和几何安全审计。

Phase-J closure audit:

- strict RGB scene wins vs selected clean MeshSplatting: `9 / 9`
- per-view strict RGB wins: `244 / 246`
- mean delta vs clean: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS
- mean total triangle reduction: `7.6479%`
- sparse geometry strict wins: `6 / 9`
- geometry-safe scenes: `9 / 9`
- evidence: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`

| scene | Phase-J PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM | dLPIPS | tri red. | per-view strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | 11.81% | 25 / 25 |
| flowers | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | 11.82% | 22 / 22 |
| garden | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | 3.47% | 24 / 24 |
| stump | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | 11.82% | 16 / 16 |
| treehill | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | 11.81% | 17 / 18 |
| room | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | 2.10% | 38 / 39 |
| counter | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | 2.10% | 30 / 30 |
| kitchen | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | 2.10% | 35 / 35 |
| bonsai | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | 11.80% | 37 / 37 |

与新 prompt / vNext 的关系：

- Phase-J 是 render-time ELA endpoint，不是完全 baked representation；因此它不能直接回答“表示级论文方法是否成功”，但它必须作为 RGB endpoint 上限参照。
- v106 是当前最强 baked representation，但仍明显弱于 Phase-J。
- vNext/new prompt 的目标是把 Phase-J 类似的修复能力推进到可审计、no-target-GT、surface/texture representation 路线；目前 full9 结果还没达到。
- flowers 单场景上，v165 `20.452848 / 0.549059 / 0.355544` 相对 Phase-J flowers `20.304358 / 0.557770 / 0.329222` 是混合结果：PSNR 更高，但 SSIM 和 LPIPS 明显更差，所以不能宣称 all-axis 超过 Phase-J。
- v166 `target-impact multisample residual fill` 已完整结束。它把 target-impact candidate bins 扩到 `457 / 4` bins/faces，并用 train-only multisample residual 填充 `105 / 130` eligible bins；但最终 `20.452814 / 0.549059 / 0.355544` 仍未超过 Phase-J flowers，且低于 v165 PSNR，因此判定为机制性进展、质量失败。

## v106：当前最强可汇报表示级版本

v106 的核心方法是保留 v104c 风格的稳定 shrink view-affine residual field，再叠加两个保守的三角形残差专家：`detail` 与 `occlusion_boundary`。它不是简单参数扫描，而是把局部细节与遮挡边界作为两个不同专家，以 base-preserve 方式避免破坏原始 MeshSplatting 表示。

### v106 per-scene 结果

| scene | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.719175 | 0.675086 | 0.313405 | +0.001526 | +0.000115 | -0.000098 |
| flowers | 20.077723 | 0.531240 | 0.374393 | +0.001879 | +0.000163 | -0.000080 |
| garden | 25.790945 | 0.799382 | 0.174480 | +0.002851 | +0.000119 | -0.000104 |
| stump | 25.460457 | 0.714661 | 0.282135 | +0.001146 | +0.000061 | -0.000078 |
| treehill | 21.245092 | 0.578518 | 0.384177 | +0.001329 | +0.000099 | -0.000121 |
| room | 29.600351 | 0.891889 | 0.230616 | +0.002516 | +0.000051 | -0.000048 |
| counter | 27.499645 | 0.867521 | 0.238847 | +0.001577 | +0.000102 | -0.000139 |
| kitchen | 28.772043 | 0.881652 | 0.187815 | +0.001595 | +0.000062 | -0.000206 |
| bonsai | 30.316090 | 0.907520 | 0.230050 | +0.005213 | +0.000154 | -0.000136 |
| mean | 25.831280 | 0.760830 | 0.268435 | +0.002181 | +0.000103 | -0.000112 |

证据文件：

- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md`

评估：

- 这是当前最适合用于汇报的 **baked representation** 结果。
- 论文风险在于：增益方向稳定，但幅度偏小，视觉差异不容易在全图直接看出来。
- 它可以作为“当前最好版本”，但还不足以支撑“远超 MeshSplatting / 颠覆式提升”的强 claim。

## vNext：工程闭环很强，但质量未达标

vNext certified residual surface texture 的目标是把修复限制在可认证的三角形/UV/bin footprint 上，并用 policy-val、image gate、bin uncertainty guard、fallback/no-op 来避免 out-of-trajectory 崩塌。它的研究价值在于协议和可审计性，而不是当前指标。

### 已完成 full9 证据

| run | scenes | protocol pass | accepted nonzero | fallback/no-op | mean changed fraction | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| structure-aware shrink cleanup | 9 | 9 | 6 | 3 | 0.002756271 | 25.067699 | 0.741260 | 0.306689 |
| effective-margin gate | 9 | 9 | 1 | 8 | 0.001371507 | 25.067410 | 0.741259 | 0.306695 |

证据文件：

- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`

评估：

- `structure-aware shrink cleanup` 说明 vNext 能在 full9 上完整跑通，并且有 6 个场景产生非零改动。
- `effective-margin gate` 说明更严格的 safety gate 可以抑制低效候选，但代价是 8/9 场景 fallback/no-op。
- 两者都低于 clean MeshSplatting 和 v106，所以不能推广为主结果。

## v162 / v163 / v164 / v165 / v166 / v167 flowers 诊断

flowers 是当前 vNext 短板诊断最清楚的场景。v162-v164 的核心发现是：不是 alpha 不够好，而是被认证允许修改的 target footprint 太小，导致全图指标和人眼视觉几乎不变。v165 证明：仅把 target-visible footprint 放大仍不够，必须让写入的 train-only residual representation 本身更有表达力。v166 进一步证明：即便用 train-only multisample residual fill 补充无 policy row 的 target-impact bins，当前局部残差仍不能转化为 SSIM/LPIPS 或视觉收益。

| version | 状态 | 核心机制 | accepted | alpha | changed pixels | allowed bins / faces | PSNR | SSIM | LPIPS | 诊断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v162 | complete | sparse-selective bridge 语义修复 | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | 真实修复，但 footprint 极小 |
| v163 | complete | target-footprint residual-debt support expansion | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | support expansion 只找到 1 个 eligible face，未改善 |
| v164 | complete | target-visible connected region growth | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | connected growth 无 eligible bins，未扩大 footprint |
| v165 | complete | train-only target-impact residual basis | true | 0.1875 | 8324 | 1145 / 26 | 20.452848 | 0.549059 | 0.355544 | footprint 明显扩大，但指标增益只有噪声级 |
| v166 | complete | train-only target-impact multisample residual fill | true | 0.1875 | 3859 | 457 / 4; filled 105 / 130 bins | 20.452814 | 0.549059 | 0.355544 | no-GT 与 multisample 机制成立，但质量低于 v165，未超过 Phase-J |
| v167 | complete | train-only target-impact affine/patch residual field | false | 0.0 | 0 | 1182 final bins; affine filled 313 / 393 bins | 20.452776 | 0.549059 | 0.355544 | stronger capacity 已执行但被 policy-val 拒绝，最终 fallback no-op；负证据 |

v162-v166 证据：

- v162 root: `/dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective`
- v163 root: `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion`
- v163 report: `docs/car_model/6-28-SPCarNet-Metrics-Engineering-Paper-Evaluation-v163.md`
- v164 root: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers`
- v164 manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- v164 audit: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v164 metrics: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_ours_26000_v164_target_connected_exact_flowers_test_results.json`
- v165 root: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers`
- v165 manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v165 audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v165 metrics: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_ours_26000_v165_target_impact_exact_flowers_test_results.json`
- v165 render dir: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/renders`
- v165 GT dir: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/gt`
- v166 root: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers`
- v166 manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v166 audit: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v166 metrics: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- v166 no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_target_apply_no_gt_verify.json`
- v166 render dir: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`
- v166 W&B offline run: `/dev/shm/peilincai_wandb_v166_target_impact_multisample_exact/wandb/offline-run-20260628_165449-r68qgrb6`
- v167 root: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers`
- v167 manifest: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v167 audit: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v167 metrics: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_results.json`
- v167 no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_target_apply_no_gt_verify.json`
- v167 render dir: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/test/ours_26000_v167_affine_flowers/renders`
- v167 W&B offline run: `/dev/shm/peilincai_wandb_v167_affine_exact/wandb/offline-run-20260628_173303-a59lvtxg`

v167 exact run 完成状态：

- manifest status: `COMPLETE`
- manifest errors: `[]`
- commands: strip `74.887s`，verify `0.116s`，apply `817.160s`，populate eval GT `11.575s`，evaluate `44.405s`
- no-GT verifier: `passed=true`，`target_gt_visible_to_apply=false`，`target_residual_visible_to_apply=false`
- affine fill audit: `enabled=true`，`uses_policy_val_gt=false`，`uses_train_fit_gt=true`，`uses_target_or_test_gt=false`
- affine fill: `eligible_bin_count=393`，`filled_bin_count=313`，`train_fit_views_used=34`，`sample_event_count=7774`，`fit_face_count=24`
- sparse materialization: `allowed_bin_count=1183`，target-impact `final_allowed_bin_count=1182`
- policy-val result: both candidates rejected; final `accepted=false`，`effective_policy=fallback_noop`，`selected_alpha=0.0`
- reject reason includes negative tail and image gates: `cvar20_view_relative_gain=-0.134897`，`min_view_relative_gain=-0.341250`，`ssim_gain=-0.000002156`，`image_l1_gain=-0.000000127`
- target apply: `changed_pixels=0`，`changed_fraction=0.0`，`fallback_noop=true`
- metrics after fallback no-op: PSNR `20.452775955200195`，SSIM `0.5490592122077942`，LPIPS `0.35554420948028564`
- interpretation: v167 proves that a simple train-only face-local affine/patch field has enough interface capacity to fill many bins, but its predicted corrections are not policy-val safe. This moves the diagnosis from “capacity interface missing” to “learned correction direction is not aligned with held-out SSIM/L1 risk”.

v164 已实现内容：

- 在 `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` 中加入 `target_connected_region_growth`。
- 在 `scripts/car_model/run_vnext_certified_residual_texture_scene.py` 中加入 runner/parser 参数转发。
- 新增接口包括 `--enable_sparse_materialization_target_connected_region_growth`、radius、min pixels/views、policy samples、positive view fraction、允许的负增益上限和 max extra bins。
- dry-run 通过，真实 exact run 完整跑通，输出 root 为 `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers`。

v164 exact run 完成状态：

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- manifest status: `COMPLETE`
- manifest errors: `[]`
- log: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/logs/02_certified_texture.log`
- apply elapsed: `23702.957s`，populate eval GT elapsed: `41.668s`，evaluate elapsed: `43.504s`
- W&B offline run: `/dev/shm/peilincai_wandb_v164_target_connected_exact/wandb/offline-run-20260628_134505-6569eb6r`
- connected growth: `enabled=true`，`reason=no_eligible_connected_bins`，`seed_allowed_bin_count=40`，`candidate_bin_count=0`，`added_bin_count=0`，`added_target_pixels=0`，`final_allowed_bin_count=121`
- target-visible expansion: `original_allowed_bin_count=40`，`candidate_bin_count=81`，`added_bin_count=81`，`final_allowed_bin_count=121`，`added_target_pixels=479`
- target apply: `changed_pixels=860`，`png_quantized_changed_pixels=849`，`changed_fraction=2.3180093151630152e-05`
- 当前解释：v164 完整验证了 connected growth 这条补丁路线，但结论是负面的。它没有找到可安全加入的 connected bins，因此没有扩大 footprint，metrics 与 v162/v163 完全一致。这说明当前瓶颈不是“少一个邻域扩张开关”，而是 certified sparse bin 候选集本身过窄。

v165 exact run 完成状态：

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- manifest status: `COMPLETE`
- manifest errors: `[]`
- protocol audit: `passed=true`，`target_apply_leak=false`，`target_gt_visible_to_apply=false`，`target_gt_visible_to_selection=false`，`target_gt_visible_to_eval=true`，`target_forbidden_keys_stripped=true`
- commands: `strip_target_evidence_no_gt` elapsed `152.590s`，`apply_certified_residual_texture` elapsed `5415.726s`，`populate_eval_gt_from_target_evidence` elapsed `11.721s`，`evaluate_vnext_target` elapsed `43.088s`
- W&B offline run: `/dev/shm/peilincai_wandb_v165_target_impact_exact/wandb/offline-run-20260628_153357-ezjo72h3`
- target-impact residual basis: `candidate_bin_count=2600`，`added_bin_count=1024`，`added_policy_row_bin_count=732`，`added_without_policy_row_bin_count=292`，`added_target_pixels=9275`，`added_target_view_hits=2240`
- final certified footprint: `original_allowed_bin_count=121`，`original_allowed_face_count=13`，`final_allowed_bin_count=1145`，`final_allowed_face_count=26`
- target apply: `changed_pixels=8324`，`png_quantized_changed_pixels=7896`，`changed_fraction=0.00022436173883042952`
- metrics: PSNR `20.452848434448242`，SSIM `0.5490590929985046`，LPIPS `0.3555436134338379`
- delta vs v164: PSNR `+0.0000514984`，SSIM `-0.0000000596`，LPIPS `-0.0000004172`，changed pixels `+7464`
- 当前解释：v165 是重要的工程和诊断进展，因为它首次把 flowers 的 target footprint 从几百像素级扩大到数千像素级，并保持 no-target-GT apply 审计通过；但它不是质量突破，因为三指标提升几乎不可见，定性图也很难形成强展示。

v166 exact run 完成状态：

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- manifest status: `COMPLETE`
- manifest errors: `[]`
- commands: `strip_target_evidence_no_gt` elapsed `76.797s`，`verify_stripped_target_evidence_no_gt` elapsed `0.109s`，`apply_certified_residual_texture` elapsed `3473.020s`，`populate_eval_gt_from_target_evidence` elapsed `11.644s`，`evaluate_vnext_target` elapsed `41.919s`
- no-GT verifier: `passed=true`，`target_gt_visible_to_apply=false`，`target_residual_visible_to_apply=false`
- candidate gate: candidate 1 accepted with `relative_gain=0.047246531`；candidate 2 accepted with `relative_gain=0.047240704`，slightly weaker
- target-impact residual basis: `final_allowed_bin_count=457`，`final_allowed_face_count=4`，`candidate_bin_count=456`，`added_bin_count=456`，`added_policy_row_bin_count=326`，`added_without_policy_row_bin_count=130`，`added_target_pixels=4792`
- target-impact multisample fill: `eligible_bin_count=130`，`filled_bin_count=105`，`train_fit_views_used=34`，`sample_event_count=3127`，`uses_target_or_test_gt=false`
- target apply: `changed_pixels=3859`，`png_quantized_changed_pixels=3807`，`changed_fraction=0.00010401392961876832`
- metrics: PSNR `20.45281410217285`，SSIM `0.5490593314170837`，LPIPS `0.3555438816547394`
- delta vs v165: PSNR `-0.0000343323`，SSIM `+0.0000002384`，LPIPS `+0.0000002682`，changed pixels `-4465`
- delta vs Phase-J flowers: PSNR `+0.1484565735`，SSIM `-0.0087108016`，LPIPS `+0.0263217386`
- 当前解释：v166 是一次干净、可审计、no-target-GT 的完整负结果。它证明 multisample fill 可以补足部分 target-impact bins，但补充后的残差仍过弱或方向不对，不能改善 SSIM/LPIPS，也不能形成对 Phase-J 的 all-axis 胜利。

## 工程级评估

### 已经达到论文级工程雏形的部分

- strict no-target-GT apply：target apply 阶段不直接读取 test GT。
- 独立 eval-GT population：最终评估阶段再补 GT，用于公平评价。
- command manifest：保存命令、路径、状态、return code、错误和输出位置。
- adapter audit：记录 accepted/fallback、alpha、target changed pixels、bin guard、sparse materialization、topology 等。
- fallback/no-op：候选不安全时显式回退，避免把坏结果硬写成方法输出。
- W&B offline：长程/中程实验已经按要求接入 offline logging，且在 `/data` 满盘时转移到 `/dev/shm`。
- 接口闭合：v164 的 connected growth、v165 的 train-only target-impact residual basis、v166 的 target-impact multisample residual fill 都已经在 adapter 与 runner 两侧都有 CLI 参数和校验。
- target evidence verifier：新增 `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`，可以独立扫描 stripped target evidence 中是否仍残留 `rgb_gt`、`residual_rgb`、teacher residual 等 forbidden keys。
- strict guard：runner 现在强制所有 target-footprint apply path 必须启用 `--strict_no_target_gt_apply`，否则直接 parser error，避免把 target footprint 机制误跑成 target-GT 可见流程。
- audit 修复：v165 后已修正 target-impact / connected-growth footprint cache 共享隐患，并补上 target-impact added-sample 统计；v166 patched exact run 已生成包含 no-GT verifier、target-impact multisample fill 和完整 command manifest 的 audit。

### 仍然不足的工程问题

- runtime 太慢：v162 flowers adapter 约 `5771.652s`，v163 flowers adapter 约 `8684.925s`；v164 exact apply 约 `23702.957s`；v165 exact apply 约 `5415.726s`；v166 exact apply 约 `3473.020s`。v166 比 v165 更快，但仍远不适合作为高吞吐论文实验系统。
- GPU 利用率低：大量耗时在 CPU/IO/NumPy/Python evidence traversal，不是典型 GPU 训练瓶颈。
- `/data` 已满，`/dev/shm` 也接近满载；W&B、manifest 和长程实验存在失败风险。
- v164 虽完整跑通，但以 6.58 小时成本得到零 footprint 增量；v165 虽扩大 footprint，却没有实质质量增益；v166 虽加入 multisample fill，但质量仍低于 v165 且没有超过 Phase-J。三者共同说明当前 verification cost / improvement ratio 不适合论文主系统。
- vNext 仍缺一个“固定策略 full9 promotion run”能同时击败 clean MeshSplatting 和 v106。
- v165 exact run 是在新增 verifier 集成前启动的，所以 manifest 里还没有 `verify_stripped_target_evidence_no_gt` 这一步；v166 patched exact run 已包含并通过该 verify command。

## 论文级评估

### 可以诚实写进 PPT/讨论的 claim

1. 我们已经建立了本地 same-protocol clean MeshSplatting baseline，并能进行 full9 比较。
2. v106 POD-MoE base-preserve 是当前最强已验证 baked representation，full9 平均三指标超过 clean MeshSplatting。
3. vNext 是一个更严谨的 no-target-GT、可审计 residual surface texture 框架，能明确记录何时改、改哪里、为什么拒绝。
4. v162/v163/v164 的负面结果很有价值：它说明仅靠 support expansion、connected growth 或更严格 gate 不能解决 footprint 太小的问题。
5. v164 的失败把瓶颈定位得更明确：安全候选集不足，而不是 alpha、单个 face support 或邻域半径设置不足。
6. v165 进一步把瓶颈推进了一步：target-impact 机制能扩大 certified footprint，但现有 train-only residual basis 的表达力不足，无法转化为明显指标或视觉收益。
7. v166 给出了关键负证据：target-impact multisample residual fill 能 no-GT 地填充 `105 / 130` eligible bins，但最终仍输 Phase-J 的 SSIM/LPIPS，说明短板不是单纯“无 policy row bins 没有被填”，而是 residual representation 本身需要升级。

### 当前不能写成最终论文主张的 claim

1. 不能说 vNext 已全面超越 MeshSplatting。
2. 不能说当前方法有“人眼明显可见”的稳定视觉提升。
3. 不能说当前方法在几何、压缩、PSNR/SSIM/LPIPS、LPIPS 感知质量上全部全面胜出。
4. 不能把 v101/v102 强 RGB endpoint 与 baked representation 结果混成一个口径。
5. 不能把 v164 当成成功改进；它已经完成验证，但没有带来 footprint 或 metrics 增益。
6. 不能把 v165 当成成功质量改进；它是成功的 footprint/工程实验，但不是成功的 paper-quality result。
7. 不能把 v166 当成成功质量改进；它是成功的 no-GT/multisample interface 实验，但实测没有超过 Phase-J，也没有超过 v165。

### 公平性与论文口径缺口

- 当前 clean MeshSplatting baseline 是本地 selected full9 口径，不等同于官方论文中完整 Mip-NeRF360 表格的绝对口径；它可以用于本地同口径比较，但不能直接宣称超过官方 paper number。
- v106 vs clean 的本地比较较公平，因为二者都用 selected full9 汇总；但仍需要补齐 per-scene clean checkpoint 选择规则、checkpoint snapshot、eval script hash 和失败场景说明。
- 当前报告的主要指标仍集中在 PSNR/SSIM/LPIPS；论文级“全面胜出”还需要三角形数量、mesh/texture/model size、渲染速度、训练/后处理时间、显存/存储占用、几何一致性和定性局部放大图。
- vNext 的 full9 promotion 结果还没有超过 clean/v106，因此不能把 vNext 与 v106 混成一个“统一已经全面胜出”的最终方法。

## 当前达标度判断

| 维度 | 达标度 | 说明 |
|---|---:|---|
| baseline 公平性 | 80% | 本地 clean selected full9 已有，但仍需和论文官方 Mip-NeRF360 口径继续对齐 |
| metrics 超越 | 55% | v106 超 clean，但 vNext 不超；提升幅度仍小 |
| 工程闭环 | 84% | manifest/audit/fallback/W&B/strict verifier 基本完整，v164/v165/v166 exact run 已闭环，但 runtime 和存储仍弱 |
| 论文故事 | 62% | 有方法线、反思线和明确瓶颈推进，但缺强主结果 |
| 定性展示 | 45% | full-frame 视觉差异偏弱；v165 扩大 footprint 后仍缺可视化强收益 |
| 最终 paper-ready | 58% | 可做阶段性汇报，不宜宣称终局完成 |

综合判断：**当前约 58%-62% paper-loop 完成度**。
它比最初盲目调参阶段强很多，已经有 baseline、full9、工程审计、W&B 长程记录、strict no-target-GT 防线和明确瓶颈；但离“顶会主结果闭环”仍有明显距离。

## 下一步优先级

1. 停止把 vNext 的主要希望放在同一套 sparse bin allowlist 的小半径扩张、alpha 微调或 multisample fill 上；v164/v165/v166 已经证明这条线最多解决 footprint，不能自然带来视觉质量突破。
2. 下一步应转向更强的 train-only representation：例如 face-local residual basis 的容量升级、target-visible residual field 的低秩/多专家表示、或以 policy-val certificate 约束的局部纹理优化。关键要求是继续保持 no-target-GT apply。
3. 新策略必须先在 flowers 做 footprint/visual diagnostic，再固定策略跑 full9，与 clean MeshSplatting、Phase-J、v106、vNext effective-margin gate、v165 和 v166 做同口径比较。
4. 工程上优先缓存 policy-val reusable evidence、减少重复 atlas traversal，并把 patched verifier 纳入所有 exact run manifest，否则 vNext 难以作为可复现实验系统。
5. README/PPT 中必须明确区分三条线：Phase-J/endpoint reference、v106 baked representation、vNext certified representation route。当前最适合汇报的正结果是 v106；v165/v166 是瓶颈诊断和工程可信度证据。

## Evidence Index

- v106 full9 assembled: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- v106 full9 compare: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- Phase-J closure audit: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`
- Phase-J closure audit CSV: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- Phase-J paper same-protocol refresh: `/dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- Phase-J qualitative showcase: `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`
- vNext structure-aware full9 summary: `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- vNext effective-margin full9 summary: `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`
- v163 detailed evaluation: `docs/car_model/6-28-SPCarNet-Metrics-Engineering-Paper-Evaluation-v163.md`
- v164 exact manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- v164 exact audit: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v164 exact metrics: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_ours_26000_v164_target_connected_exact_flowers_test_results.json`
- v165 exact manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v165 exact audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v165 exact metrics: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_ours_26000_v165_target_impact_exact_flowers_test_results.json`
- v165 no-GT verifier audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/manual_target_apply_no_gt_verify_after_patch.json`
- v165 exact renders: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/renders`
- v166 target-impact multisample exact run, complete: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact`
- v166 exact manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v166 exact audit: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v166 exact metrics: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- v166 exact renders: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`
- v166 W&B offline run: `/dev/shm/peilincai_wandb_v166_target_impact_multisample_exact/wandb/offline-run-20260628_165449-r68qgrb6`
- v166 dry-run manifest with strict no-GT verifier and multisample CLI: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_dryrun/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v164 adapter implementation: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- v164 runner implementation: `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- no-GT target evidence verifier: `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`

## Final Status

Final status: NOT COMPLETE.

## 2026-06-29 Update: v195-v199 Surface-Texture / Low-Rank Attempt

The new `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` route has now been implemented and tested on flowers exact, but it did **not** pass the required Phase-J flowers gate.

New code paths:

- `surface_texture_mlp`: trainable per-face/per-UV surface feature texture plus tiny decoder.
- `lowrank_surface_texture`: support-aware rank-K residual basis with inactive-support no-op guarantee.
- `--surface_target_visible_evidence_dir`: no-GT target-visible face priority for capacity allocation.

Official flowers exact results:

| Run | Method | PSNR | SSIM | LPIPS | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| Phase-J gate | reference | 20.304358 | 0.557770 | 0.329222 | target |
| v195 | surface texture MLP, teacher-only | 19.878033 | 0.509020 | 0.402998 | fail all axes |
| v196 | surface texture MLP, GT-assisted diagnostic | 20.084991 | 0.523929 | 0.385202 | fail all axes |
| v197 | support-aware low-rank, teacher-only | 19.834993 | 0.505835 | 0.405083 | fail all axes |
| v198 | support-aware low-rank, GT-assisted diagnostic | 19.833418 | 0.505749 | 0.404551 | fail all axes |
| v199 | low-rank + no-GT target-visible capacity | 19.835337 | 0.505801 | 0.404194 | fail all axes |

Important lesson: v199 increased target known-face support from about `0.0501` to `0.1677` and active support from about `0.0294` to `0.1059`, with inactive-support changed fraction staying `0.0`. That confirms the support allocator and safety gate work mechanically. The official metric failure means the remaining bottleneck is cross-view residual generalization: the train/support residual field still does not transfer well enough to target views.

Detailed log:

```text
docs/car_model/6-29-v195-v199-SurfaceTexture-LowRank-Diagnostics.md
docs/car_model/results/v195_v199_surface_texture_lowrank_summary.json
```

No v195-v199 result should be promoted to full9 or paper-ready status.

### 2026-06-29 Residual Projection Audit

A new audit tool was added:

```text
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

It compares checkpoint-predicted residuals with `teacher_residual_rgb` on
policy-val views, and compares final target residuals with target GT residuals
after no-GT apply. The compact result is:

| Run | Policy retention | Policy cosine | Target retention | Target cosine |
| --- | ---: | ---: | ---: | ---: |
| v191 image-space U-Net calibration | 9.916031 | 0.279888 | 0.253365 | 0.393485 |
| v195 surface texture MLP | 0.068206 | 0.112638 | 0.002863 | 0.133734 |
| v196 GT-assisted surface MLP diagnostic | 1.427611 | 0.138419 | 0.029127 | 0.199612 |
| v199 support-aware low-rank | 0.015229 | 0.039391 | 0.000847 | 0.028702 |

Conclusion: the surface/low-rank family fails at residual projection and
alignment before target promotion. Future candidates need an explicit
source-view projection gate before any exact target/full9 run.

## 2026-06-28 Update: v168 Low-Copy Direct-Teacher Patch

`feedback.md` is the current handoff file in the repository root:

```text
/data/peilincai/mesh-splatting/feedback.md
```

After the first v168 exact flowers attempt failed before metrics with `OSError: [Errno 122] Disk quota exceeded`, a storage-unblock patch was implemented:

- `scripts/car_model/ecsr_reparent_surface_evidence_cache.py`: added `--copy_mode {copy,hardlink,symlink,auto_link}`.
- `scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py`: added `--copy_mode` and `--rewrite_rgb_render_to_parent`.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`: added `--reparent_copy_mode`, `--teacher_cache_copy_mode`, `--teacher_cache_rewrite_rgb_render_to_parent`, and `--skip_reparent_fit_evidence_for_teacher_cache`.

This lets the v168 Phase-J distillation route skip a separate full `fit_evidence_reparented` cache and fuse fit reparenting into teacher-cache construction. It is an engineering unblock, not a quality claim.

Validated:

- py_compile passed for the three modified scripts.
- `git diff --check` passed for the three modified scripts.
- low-copy reparent smoke with one view passed.
- low-copy teacher-cache smoke with one view passed.
- parser guard for an unsafe skip configuration failed as expected.
- direct-teacher low-copy dry-run passed and produced a command chain without `reparent_fit_evidence`.

Current exact run in progress:

```text
/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers
```

W&B offline root:

```text
/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact
```

Current status at this update: the run had reached `02_certified_texture.log` policy-candidate evaluation. There were no completed final metrics yet. Do not promote this as a success until the run writes final results and is compared against Phase-J flowers:

- PSNR > `20.304358`
- SSIM > `0.557770`
- LPIPS < `0.329222`

未完成项：

- vNext/new prompt 尚未在 full9 上超过 clean MeshSplatting、Phase-J 或 v106。
- Phase-J 对比已补入本文档；下一步必须让 vNext 的 flowers exact 和固定 full9 promotion 都显式报告 vs Phase-J，而不是只报 vs clean/v106。
- 定性优势仍不明显，当前 footprint 太小。
- 工程 runtime 和存储稳定性仍需修复。
- v164 target-connected growth 已完成但无增益；v165 target-impact footprint 扩大但指标几乎不动；v166 multisample fill 仍未改善质量；v167 affine/patch fill 被 policy-val 拒绝；v168 只完成 Phase-J distillation protocol dry-run。下一步必须真正运行并验证更强的 Phase-J-distilled train-only residual representation，而不是继续小修 policy 参数或只扩大 footprint。

下一条最精确的继续方向是：

```bash
WANDB_DIR=/dev/shm/peilincai_wandb_v168_phasej_distill_flowers \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<low_or_mid_load_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --teacher_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasej_trainval_gate/renders \
  --parent_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders \
  --reparent_target_parent_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_phasef_extra_compact_base/renders \
  --distillation_profile teacher_to_reparented_parent \
  --output_root /dev/shm/peilincai_spcarnet_20260628_v168_phasej_distill_flowers \
  --method_name ours_26000_v168_phasej_distill_flowers \
  --enable_train_only_target_impact_residual_basis \
  --target_impact_max_extra_bins 1024 \
  --wandb --wandb_mode offline \
  --wandb_group v168_phasej_distill_flowers \
  --wandb_name v168-phasej-distill-flowers
```

这条命令只能作为下一轮 patched-run 基础；真正需要新增的是更强的 train-only residual capacity，而不只是重复 v165 的 footprint expansion 或 v166 的 multisample fill。

## 2026-06-29 Update: v253-v254 Deferred Source Renderer

最新按 `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md` 推进了一次真实表示层升级：

- 新增 `scripts/car_model/train_surface_deferred_source_residual_renderer.py`。
- v253 不再是静态 RGB atlas 或 alpha scan；它为每个 face/UV bin 存多个 train-fit Phase-J teacher residual source，并按目标视角方向、法线一致性、parent RGB 相似度、support count、teacher gain 做 deferred aggregation。
- 目标 apply 使用 stripped no-GT evidence；target GT 只在 apply 后用于 evaluation。
- 支持 `--bank_checkpoint`，因此后续 policy/eval ablation 可以固定表示、避免重建 bank。
- v254 额外测试了 residual channel shaping：`luma_only` 和 `chroma_shrink`。

关键结论：**v253 是有效的表示层里程碑，但不是论文闭环成功**。它首次在 policy-val 上产生非零 all-axis 小幅正增益，但固定策略 target exact 仍被 LPIPS 卡住，不能跑 full9。

| run | selected alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v253b raw RGB | 0.031250 | +0.001240 | +0.000015 | +0.000004 | +0.001063 | +0.000028 | -0.000002 | fail |
| v253d conservative | 0.015625 | +0.000628 | +0.000008 | +0.000006 | +0.000537 | +0.000014 | -0.000001 | fail |
| v254a luma only | 0.031250 | +0.001141 | +0.000012 | +0.000002 | +0.000985 | +0.000025 | -0.000005 | fail |
| v254b chroma shrink | 0.031250 | +0.001166 | +0.000013 | +0.000003 | +0.001005 | +0.000025 | -0.000004 | fail |

Artifacts:

```text
docs/car_model/6-29-v253-v254-DeferredSourceRenderer-Log.md
docs/car_model/results/v253_v254_deferred_source_renderer_summary.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/v253_deferred_source_renderer_audit.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v253b_source_feature_deferred_targetexact/target_exact_fixed_policy
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

The next step should not be another alpha grid. The residual bank needs a target-blind perceptual confidence/reliability predictor that suppresses source/bin residuals with weak multi-source agreement, high residual variance, or poor edge/teacher-gain consistency before target apply.

## 2026-06-29 Update: v255 Source-Agreement Confidence

v255 tested the simplest version of that confidence idea: a target-blind soft
agreement gate based on top-k source residual variance in the frozen v253b bank.

Implementation update:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New options:

```text
--source_agreement_mode {off,soft,hard}
--source_agreement_beta
--source_agreement_min_confidence
```

Result:

| stage | alpha | PSNR gain | SSIM gain | LPIPS gain | mean confidence | all-axis |
|---|---:|---:|---:|---:|---:|---|
| policy-val | 0.046875 | +0.001655 | +0.000018 | +0.000001 | 0.655315 | pass |
| target exact | 0.046875 | +0.001395 | +0.000036 | -0.000008 | 0.651719 | fail |

Artifacts:

```text
docs/car_model/6-29-v255-SourceAgreementConfidence-Log.md
docs/car_model/results/v255_source_agreement_confidence_summary.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/v253_deferred_source_renderer_audit.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v255a_loadedbank_soft_agreement_targetexact/wandb/offline-run-20260629_200707-e3wmgpr9
```

Verdict: source residual variance alone is not a sufficient perceptual
reliability signal. It improves target PSNR/SSIM but makes target LPIPS more
negative than v253b/v253d. The next model should use a learned/calibrated
perceptual reliability predictor, not just hand-designed agreement confidence.

## 2026-06-29 Update: v256 Policy-Val L1 Reliability

v256 implements the first learned/calibrated target-blind reliability policy in
the v253 family:

- build/load the v253 deferred source-feature bank;
- use policy-val GT only to estimate per-face/per-UV-bin local L1 improvement;
- convert that into a frozen reliability map;
- apply to stripped target no-GT evidence;
- load target GT only after apply for exact evaluation.

Implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

New controls:

```text
--policy_reliability_mode local_l1
--policy_reliability_alpha
--policy_reliability_min_count
--policy_reliability_min_positive_fraction
--policy_reliability_min_mean_gain
--policy_reliability_gain_scale
--policy_reliability_floor
```

Result:

| run | min positive fraction | alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v256a | 0.52 | 0.125 | +0.002737 | +0.000087 | +0.000035 | +0.000830 | +0.000026 | +0.000013 | pass |
| v256b | 0.50 | 0.250 | +0.005508 | +0.000175 | +0.000070 | +0.001659 | +0.000050 | +0.000026 | pass |
| v256c | 0.48 | 0.500 | +0.010844 | +0.000343 | +0.000144 | +0.003185 | +0.000091 | +0.000050 | pass |

Current best is v256c:

```text
target exact: 19.835239 PSNR / 0.620001 SSIM / 0.180285 LPIPS
gains vs parent: +0.003185 / +0.000091 / +0.000050
```

This is a real improvement over v253-v255 because the target exact LPIPS mean is
now positive instead of negative. It is still **not** enough for full9 or paper
readiness:

- Phase-J flowers PSNR gate is still `20.304358`, so v256c is still `-0.469119`
  PSNR below it under this flowers exact evidence path.
- target SSIM and LPIPS tails remain slightly negative;
- changed fraction is only `0.007788`, so qualitative changes are still subtle.

Artifacts:

```text
docs/car_model/6-29-v256-PolicyL1Reliability-Log.md
docs/car_model/results/v256_policy_l1_reliability_summary.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/v253_deferred_source_renderer_audit.json
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/target_exact_fixed_policy
/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v256c_policy_l1_reliability_minpos048_targetexact/wandb/offline-run-20260629_201901-7rm7opzk
```

Current verdict remains:

```text
Final status: NOT COMPLETE.
```
