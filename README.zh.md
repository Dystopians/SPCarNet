# SPCarNet / MeshSplatOpt

**基于训练证据、协议审计安全的 Mesh Splatting 压缩与渲染修复。**

## 当前报告包（2026-06-26）

本地 clone 后做 mentor/PPT 分析请从这里开始：

- [根目录报告入口](SPCARNET_REPORT_INDEX.md)
- [克隆/PPT 技术总览](docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md)
- [2026-06-26 最新状态附录](docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md)
- [vNext 实现日志](docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md)
- [vNext 可行性与执行计划](docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md)
- [vNext soft-shrink garden 里程碑](docs/car_model/6-26-SPCarNet-vNext-SoftShrink-Garden-Milestone-Log.md)
- [vNext 技术报告与 artifact 索引](docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md)
- [vNext structure-aware shrink strict 多场景日志](docs/car_model/6-26-vNext-StructureAwareShrink-Strict-Multiscene-Log.md)
- [vNext manifest runner 与 full9 缺口日志](docs/car_model/6-26-vNext-ManifestRunner-and-Full9Gap-Log.md)
- [vNext structure-aware shrink ready4 artifact 聚合表](docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md)
- [vNext stump 重建 / ready5 拒绝日志](docs/car_model/6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md)
- [vNext ready4 preflight](docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md)
- [vNext full9 gap preflight](docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md)
- [可克隆报告包 manifest](docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md)
- [当前 mentor/PPT 技术报告](docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md)
- [中文长版导师技术报告](docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md)
- [当前可克隆报告索引](docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md)

简短状态：`v106 POD-MoE base-preserve` 是当前已验证的质量主线，在 assembled selected full9 表上相对本地 clean MeshSplatting baseline 三个指标均值都更好。`v113b/v113c` 是严格 gate 的安全修复，改善安全性并部分修复 garden v110b，但没有超过 v106。`v114_oof_refit_pod_moe` 是当前正在跑的 candidate-side 长程实验，还不是已完成结果。最新状态附录：v110 counter 在 field build 阶段以 return code `-9` 失败，大概率是内存/共享盘压力导致，因此 strict branch 仍需低内存 field-builder 修复后重跑。

vNext 状态：certified residual surface texture 方向可以推进，但不能承诺已经产生论文级结果。最新 structure-aware shrink 里程碑新增 train-policy-val 局部 L1/gradient 结构风险 shrink，并修复了 parent-edge apply/profile 接口转发。在 ready 场景 `counter,bonsai,room,garden` strict no-target-GT apply 下，固定 structure-aware policy 为 `4 / 4` accepted，相对 Phase-F compact parent 的平均变化是 `+0.00076151` PSNR、`-0.00000302` SSIM、`-0.00002038` LPIPS。最重要修复是 `room` 从旧 strict face-softshrink fallback/no-op 变成 accepted nonzero output，`garden` 也相对 Phase-F parent 和旧 garden face-softshrink pilot 三指标小幅正向。下一步本地重建补齐了 `stump` 输入链，把 preflight 推进到 `5 / 9` ready；但 strict stump run 被 tail-risk certificate 正确拒绝为 fallback/no-op。剩余缺输入场景是 `bicycle,flowers,kitchen,treehill`。

## 当前 v106 POD-MoE 状态（2026-06-25）

当前 representation-level 候选是 **v106 POD-MoE base-preserve**。它保留 v104c 风格的 shrink view-affine residual field 作为稳定基底，再叠加两个保守的三角形残差专家：`detail` 与 `occlusion_boundary`。

必须明确口径：这是本地 Mip-NeRF360 full9 协议下的固定策略诊断。它在 9 个选定场景上都相对本地 v104c representation-field anchor 正向，但收益幅度很小，还不能包装成“大幅论文级突破”。

| 方法 | 场景数 | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | - | - | - |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | 0.000000 | 0.000000 | 0.000000 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.002181 | +0.000103 | -0.000112 |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +0.652211 | +0.022949 | -0.044243 |

完整证据：

- Mentor 技术报告：[v106 POD-MoE mentor 技术报告](docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md)
- Full9 表：[full9 assembled result](docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md)
- 对比报告：[v106 vs v104c report](docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md)
- 方法日志：[v106 POD-MoE base-preserve log](docs/car_model/6-25-v106-PODMoE-BasePreserve-HardTriad-Log.md)
- 定性示例：![v106 garden qualitative panel](docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png)

下一步方法主线是 **v114 OOF-refit POD-MoE candidate-side validation**，因为 v110/v110b 已经证明只改 safety gate 大多只能保住 v106，而不能产生更强 candidate。v114 使用 train/all 系数，但用 out-of-fold gain 限制 expert reliability，目标是在不进行 per-scene 参数扫描的情况下恢复有效容量。

**Mentor/PPT 集成技术报告：** [SPCarNet 当前方法完整技术报告（集成版）](docs/car_model/6-24-SPCarNet-Mentor-PPT-Integrated-Technical-Report.zh.md)

**当前 PPT 汇报报告：** [当前方法/实验/渲染对比汇报报告](docs/car_model/6-25-SPCarNet-Current-Method-Experiment-Report-With-Visuals-ForMentor.zh.md)；[当前方法与 MeshSplatting 完整对比报告](docs/car_model/6-25-SPCarNet-Current-Method-vs-MeshSplatting-Complete-Report.zh.md)；[完整方法/实验/渲染对比报告](docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md)；[SPCarNet mentor/PPT clean 技术报告](docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CleanCurrent.zh.md)；[完整长版技术报告](docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md)；[方法/实验/定性图可视化报告](docs/car_model/6-24-SPCarNet-Current-Method-Experiment-Visual-Report.zh.md)；[claim boundary 与论文缺口](docs/car_model/6-24-SPCarNet-Claim-Boundary-And-Paper-Gap.zh.md)；[v83 patchmix hybrid 日志](docs/car_model/6-24-v83-PatchMixFaceAlphaLocalPatch-Hybrid-Log.md)；[v84 strict selector 日志](docs/car_model/6-24-v84-StrictCapacitySelector-Log.md)；[v86 anchor-preserving tail-risk selector 日志](docs/car_model/6-24-v86-AnchorPreservingTailRiskSelector-Log.md)；[v83/subagent 续跑日志](docs/car_model/6-24-SPCarNet-PaperLoop-Continuation-v83-And-Subagents.md)

**当前证据 manifest：** `outputs/carnet/spcarnet/current_evidence_manifest_20260624.md`（`23 / 23` 个证据文件存在，必需项缺失 `0`；仅表示存在性与哈希已记录，不等于正确性证明）。

**当前 representation-level 诊断：** v79 已复现 v56/v64 counter anchor（`26.756130219 / 0.862126231 / 0.251691371`），记录见 [v79 v56-seeded anchor 日志](docs/car_model/6-24-v79-V56SeededFaceAlphaAnchor-Log.md)。v80（`face-alpha + local-patch + bin-gain hybrid`）已完成为 near-tie 诊断，W&B run `izuzuhy0`，记录见 [v80 face-alpha hybrid local-patch 日志](docs/car_model/6-24-v80-FaceAlphaHybridLocalPatch-Log.md)：它在 `counter` 上达到 `26.756135941 / 0.862126231 / 0.251691461`，从 v76/v78b 回退中恢复且超过 v75，但 LPIPS 仍略差于 v56/v64/v79 anchor，因此不提升为主结果。v81（`view-conditioned residual basis`）也已完成，记录见 [v81 view-conditioned basis 日志](docs/car_model/6-24-v81-ViewConditionedBasis-Log.md)：结果为 `26.753919601 / 0.862121582 / 0.251836061`，三项均弱于 anchor，因此不提升。v82（`patch-mixture teacher basis`）已完成，W&B run `6subv75i`，记录见 [v82 patch-mixture teacher-basis 日志](docs/car_model/6-24-v82-PatchMixtureTeacherBasis-Log.md)：新 basis 接口可运行，但 guard 回退到 legacy teacher basis，结果 `26.753459930 / 0.862114668 / 0.251868337` 仍低于 anchor，因此不提升。v82b（`capacity pre-rank + face-alpha`）在 `counter` 上达到 strict micro-win `26.756137848 / 0.862126350 / 0.251690656`，但 hard-triad 验证显示 `kitchen/bonsai` 相对 v64 不能严格晋升。v83（`patchmix + face-alpha + local-patch hybrid`）在 `counter` 上达到 `26.756147385 / 0.862125337 / 0.251688808`，PSNR/LPIPS 超过 v56/v64/v79/v80/v82，但 SSIM 微退，因此是 mixed 诊断，不提升。v84 已将 v82b 纳入严格 train/policy-val selector 并回退 v64：full9 相对 v64 为 `9 / 9` non-regressive/tie，mean `+0.000000848` PSNR、`+0.000000013` SSIM、`-0.000000079` LPIPS；但规则是在 v82 hard-triad 诊断后形成，仍是 report-only，不是论文主结果。v85 已完成两条 safety 诊断：SSIM-safe pre-rank patchmix 正确拒绝所有候选但 fallback 低于 anchor；target-footprint tail-risk 接受非空编辑并达到 `26.756134033 / 0.862126231 / 0.251691371`，本质上与 anchor 持平，因此仍不提升。见 [v82b capacity pre-rank face-alpha 日志](docs/car_model/6-24-v82b-CapacityPrerankFaceAlpha-Log.md)、[v82 hard-triad summary](outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_triad_20260624/summary.md)、[v83 patchmix hybrid 日志](docs/car_model/6-24-v83-PatchMixFaceAlphaLocalPatch-Hybrid-Log.md)、[v84 strict selector 日志](docs/car_model/6-24-v84-StrictCapacitySelector-Log.md)、[v85 SSIM-safe 日志](docs/car_model/6-24-v85-SSIMSafePreRankPatchMix-Log.md) 和 [v85 target-footprint tail-risk 日志](docs/car_model/6-24-v85-TargetFootprintTailRiskCertificate-Log.md)。

**v86 anchor-preserving selector 更新：** v86 把当前 v84 selected endpoint 作为 anchor，只有当 v85 target-footprint tail-risk 在 train/policy-val audit 上支配该 anchor 时才晋升。当前 v85 `counter` 候选因 SSIM/L1 audit 弱于 v84/v82b 被拒绝，所以 v86 相对 v84 为 `9 / 9` non-regressive/tie，并保留更强的 counter row。证据：[v86 anchor-preserving tail-risk selector 日志](docs/car_model/6-24-v86-AnchorPreservingTailRiskSelector-Log.md) 和 `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_full9_summary.md`。

**近期完成的负结果诊断：** [target-footprint certificate 日志](docs/car_model/6-24-v78-TargetFootprintCertificate-Running-Log.md) 与 [strict bin-gain hybrid 日志](docs/car_model/6-24-v77-StrictBinGainHybrid-Log.md)。pre-fix v78 与修复代码后的 v78b formal rerun 均已完成为负结果诊断，未提升为主结果；W&B runs 为 `7pz9pulx` 与 `fvfj1s4q`，summary 位于 `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v78b_*_formal_20260624/`。

[English](README.md) | [mentor/PPT 技术报告 2026-06-24](docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CleanCurrent.zh.md) | [v75 local patch prior 日志](docs/car_model/6-24-v75-LocalPatchPrior-Log.md) | [v74 delta-cap ladder 日志](docs/car_model/6-24-v74-DeltaCapLadder-Log.md) | [v70 policy-val blend-ladder prior 探针](docs/car_model/6-24-v70-PolicyValBlendLadder-MultiscalePrior-Log.md) | [v69 多尺度 surface prior 探针](docs/car_model/6-24-v69-MultiscaleSurfacePrior-Probe-Log.md) | [v68 keep-downweight shrink 探针](docs/car_model/6-24-v68-KeepDownweightUncertainty-Probe-Log.md) | [v67 uncertainty-shrink 探针](docs/car_model/6-24-v67-UncertaintyShrink-Probe-Log.md) | [v66 bin-RGB alpha 探针](docs/car_model/6-24-v66-BinRGBAlphaCalibration-Probe-Log.md) | [v57a face-alpha shrink 探针](docs/car_model/6-24-v57a-FaceAlphaReliabilityShrink-Probe-Log.md) | [v56 source-rerun/fresh-probe 日志](docs/car_model/6-24-v56-SourceRerun-And-FreshProbe-Log.md) | [v56 face-alpha guard 日志](docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md) | [v55d face-alpha calibration 日志](docs/car_model/6-23-v55d-FaceAlphaCalibration-CapHit-Log.md) | [v52 capacity-aware policy 日志](docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md) | [v51 support ladder full9 日志](docs/car_model/6-23-v51-SupportFootprintLadder-Full9-Log.md) | [v48 auto-support atlas 日志](docs/car_model/6-23-v48-AutoSupportSurfaceAtlas-Log.md) | [v47 auto-capacity atlas 日志](docs/car_model/6-23-v47-AutoCapacitySurfaceAtlas-Log.md) | [v42 confidence/SSIM-gated atlas 日志](docs/car_model/6-23-v42-ConfidenceSSIMGateAtlas-Log.md) | [v39 SSIM-aware atlas 日志](docs/car_model/6-23-SSIMAwareAtlas-v39-Implementation-Log.md) | [v38 risk-aware atlas 日志](docs/car_model/6-23-RiskAwareAtlas-v38-Implementation-Log.md) | [v37 visible barycentric 日志](docs/car_model/6-23-VisibleBarycentricCoverage-v37-Implementation-Log.md) | [v36 matched-res atlas 日志](docs/car_model/6-23-MatchedResTeacherAtlas-v36-Log.md) | [当前方法技术报告](docs/car_model/6-23-SPCarNet-Current-Method-Mentor-Technical-Report.zh.md) | [当前方法/证据日志](docs/car_model/5-14-SPCarNet-Method-Modules-And-Evidence-Log.md) | [Phase-S region core/context portfolio](docs/car_model/5-17-PhaseS-RegionCoreContext-Portfolio-Log.md) | [Phase-S shared residual-field](docs/car_model/5-16-PhaseS-SharedResidualField-Operator.md) | [Phase-S effect-aware / rank2 / auto-visual](docs/car_model/5-15-PhaseS-EffectAware-Portfolio-Rank2-AutoVisual.md) | [Phase-S v20 auto-prefix / portfolio](docs/car_model/5-14-PhaseS-v20-AutoPrefix-Portfolio-Policy.md) | [Phase-S fold-aware PatchCert 续跑日志](docs/car_model/5-14-PhaseS-V6Multifold-V7V8-FoldAware-PatchCert-Log.md) | [Phase-S Compact-Stratified PatchCert 日志](docs/car_model/5-14-PhaseS-CompactStratified-Gate-Log.md) | [Phase-S Direct PatchCert 日志](docs/car_model/5-14-PhaseS-DirectPatchCert-Carrier-Pilot.md) | [Phase-S PatchRisk 日志](docs/car_model/5-14-PhaseS-PatchRisk-Carrier-Pilot.md) | [Phase-S GeoRisk/CVaR 日志](docs/car_model/5-14-PhaseS-GeoRiskCVaR-Selector-Log.md) | [Phase-S risk-tail/alpha 日志](docs/car_model/5-14-PhaseS-RiskTail-Alpha-ModuleLog.md) | [Phase-S coupled selector](docs/car_model/5-13-Coupled-Selector-Pilot.md) | [Phase-J 结果](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md) | [Surface-lumigraph V8](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md) | [Phase-R 全折审计](docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md) | [Phase-S gaincert 审计](docs/car_model/5-12-PhaseS-GainCertV1-Audit.md) | [SPCarNet selector 审计](docs/car_model/5-12-SPCarNet-RagSym-Rerank-Audit.md) | [Full9 状态](docs/car_model/5-12-Full9-PaperLoop-Evidence-Status.md) | [闭环状态](docs/car_model/5-12-PaperLoop-ClosedLoop-Status.md) | [续跑报告](docs/car_model/5-12-Subagent-PaperLoop-Continuation-Report.md) | [Phase-J 外部验证](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md) | [当前版本留档](docs/car_model/5-7-Archive-Full9-CompactELA.md) | [执行日志](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md) | [研究日志](docs/car_model/SPCarNet_research_log.md) | [旧版 README](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)

较早 Phase-K 日志：[candidate-aware portfolio closure](docs/car_model/5-22-PhaseK-PolicyPortfolio-Closure.md)、[multiscene validation log](docs/car_model/5-22-CandidateAwareELA-Multiscene-Validation-Log.md)、[rank-K PatchCert validation](docs/car_model/5-22-RankK-PatchBasis-Validation-Log.md)。

最新完成诊断：[v77 strict bin-gain hybrid 探针](docs/car_model/6-24-v77-StrictBinGainHybrid-Log.md)，W&B run `3ho2y4s1`。前一版 v76 记录：[v76 policy-val bin-gain hybrid prior 探针](docs/car_model/6-24-v76-PolicyValBinGainHybrid-Log.md)。

SPCarNet 是建立在 Mesh Splatting 之上的研究分支。当前 ECSR 版本保留固定的 Phase-F compact checkpoint，再用 train-evidence guarded portfolio 做外观修复：稳定场景走 adaptive-alpha ELA，不稳定场景走 train-selected structural edge fallback。branch、edge gate、alpha、压缩比例都不使用 held-out test 指标选择。

```text
current method: ours_26000_phasej_guarded_adaptedge_ela
report: outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
mentor_report: docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CleanCurrent.zh.md
evidence_manifest: outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
```

5 月 7 日 Compact-ELA/SOR 版本仍以 `archive/full9-compact-ela-ssim-peak-20260507`、commit `fae7942` 留档。Phase-J 在当前 full9 RGB 口径上更强，但它仍然是 render-time ELA portfolio，不是完全 baked 到表示里的终局模型。当前安全的论文表述是 **针对 MeshSplatting checkpoint 的 evidence-certified post-training repair and compaction**，不是“已经完全替代或解决 MeshSplatting”。

**当前闭环状态，2026-06-24：** Phase-J local full9 RGB 闭环是 `COMPLETE`；paper-level representation / paper-table 同口径闭环仍是 `NOT COMPLETE`。当前最强可审计 endpoint 仍是 Phase-J：在本地复现的 selected-clean `26000/30000` envelope、相同 full9 split 和相同 evaluator 下，scene-level PSNR/SSIM/LPIPS 为 `9 / 9` strict，per-view RGB 为 `244 / 246` strict，平均删面 `7.6479%`。当前固定 representation-level 路线仍只是 v64/v84 量级：v64 相对 v56 为 `9 / 9` non-regressive/tie，只提升 `kitchen`，mean gain 为 `+0.000410080` PSNR、`+0.000000278` SSIM、`-0.000018951` LPIPS；v84 加入严格 v82b counter selector 后，相对 v64 仍为 `9 / 9` non-regressive/tie，但新增均值只有 `+0.000000848` PSNR、`+0.000000013` SSIM、`-0.000000079` LPIPS。v65 teacher basis、v66 bin-RGB alpha、v67 uncertainty shrink、v68 keep-with-downweight shrink、v69 count-pyramid multi-scale prior、v70 policy-val blend-ladder prior、v71a evidence-consistent prior gate、v72 local prior allowlist、v73 target-support candidate selection、v73b target-support pre-rank、v74 residual delta-cap ladder、v75 local patch prior、v76 policy-val bin-gain hybrid prior、v77 strict bin-gain hybrid、v78/v78b target-footprint certificate、v80 face-alpha hybrid local-patch、v81 view-conditioned basis、v82 patch-mixture teacher basis、v82b raw capacity-prerank、v83 patchmix hybrid、v84 strict selector 和 v85 SSIM/tail-risk certificate 都是诊断 / report-only 候选，不是 promoted endpoint。当前完整报告：[`方法/实验/渲染对比报告`](docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md)；持久证据：`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v81_viewbasis_20260624/summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_patchmix_teacher_20260624/summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_ssimsafe_prerank_patchmix_counter_20260624/summary.md` 和 `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_target_tailrisk_counter_20260625/summary.md`。

**Paper-loop 状态，2026-05-17：** `NOT COMPLETE`。Phase-J 仍是当前最强 endpoint：clean-best 与 Phase-J RGB 行在 `9 / 9` 场景完整，且 Phase-J 相对 selected clean MeshSplatting 在 `9 / 9` 场景三指标严格胜出。Phase-S 现在是一个真实的 representation-level face-local repair 分支，但可靠收益仍然稀疏。最新的 region core/context weighted fitting 把 train-only render-visible region membership 接入 residual fitting objective，再用固定的 effect-aware portfolio 拒绝不安全行。直接 core/context 方法在 `flowers` 上有明显 report-only 正收益，但也会误接受 `kitchen`、`bonsai` 和 `counter`；最终固定 train-val-only portfolio 因此只保留 `bicycle=patchcert_v6`、`flowers=rvregion_corectx_A`、`garden=rvregion_garden`、`counter=riskpilot`、`kitchen=rvregion_indoor`，其它场景回退 Phase-J。Full9 effective report-only delta 为 `+0.000947740` PSNR、`+0.000062552` SSIM、`-0.000098634` LPIPS。它强于 2026-05-15 effect-aware portfolio 和 2026-05-16 robust region prior，但收益幅度仍小，不是论文级突破。证据：`outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v1.md`。最新模块/证据日志：[`Phase-S region core/context portfolio`](docs/car_model/5-17-PhaseS-RegionCoreContext-Portfolio-Log.md)、[`Phase-S shared residual-field`](docs/car_model/5-16-PhaseS-SharedResidualField-Operator.md)、[`Phase-S effect-aware / rank2 / auto-visual`](docs/car_model/5-15-PhaseS-EffectAware-Portfolio-Rank2-AutoVisual.md)、[`Phase-S v20 auto-prefix / portfolio`](docs/car_model/5-14-PhaseS-v20-AutoPrefix-Portfolio-Policy.md)、[`Compact-Stratified PatchCert`](docs/car_model/5-14-PhaseS-CompactStratified-Gate-Log.md)、[`Direct PatchCert`](docs/car_model/5-14-PhaseS-DirectPatchCert-Carrier-Pilot.md)、[`PatchRisk`](docs/car_model/5-14-PhaseS-PatchRisk-Carrier-Pilot.md)、[`GeoRisk/CVaR`](docs/car_model/5-14-PhaseS-GeoRiskCVaR-Selector-Log.md)、[`risk-tail/alpha`](docs/car_model/5-14-PhaseS-RiskTail-Alpha-ModuleLog.md)。

**Paper-loop 更新，2026-05-22：** `NOT COMPLETE`。Candidate-aware ELA 现在对 Phase-J fallback 和 Phase-S candidate 使用同一套 train-only per-model auto-policy，再只用 train-val 证据选择 edge/plain 变体。在 `counter,bonsai,room,flowers` 上，固定 portfolio 在 `4 / 4` 场景选择 plain candidate，但 mean report-only delta 只有 `+0.000053883` PSNR、`-0.000000268` SSIM、`+0.000000469` LPIPS。定性 panel 同样很细微，因此这是 fairness/policy 改进，而不是论文级 representation breakthrough。

**Rank-K 更新，2026-05-22：** `NOT COMPLETE`。我们实现了真实的 representation-level rank-4 PatchCert carrier-basis operator，并在 `flowers,counter,bonsai,room` 上用 W&B online 完整验证。它接受 `3 / 4` 个场景，effective held-out delta 为 `+0.000042439` PSNR、`-0.000000015` SSIM、`-0.000000183` LPIPS。`bonsai` 是最清楚的正向结果，但 `counter` 仍没过 compact PSNR gate，`room` 也暴露了 held-out balanced 轻微回退。证据见 [`rank-K PatchCert validation`](docs/car_model/5-22-RankK-PatchBasis-Validation-Log.md)。

**Paper-loop 更新，2026-06-22：** `NOT COMPLETE`。Phase-J 仍是当前可以安全汇报的 endpoint，已整理进 mentor/PPT 技术报告。v26 hard local-trust、v27 soft local-trust、v28 view-tail-safe alpha shrink 是最新的修复策略探针。v26/v27 Bonsai medium 已完整落盘，二者都没有通过 honest train-val/tail gate；v27 修复了 trust 全零问题，但最终 selector 接受行在 held-out test 上接近 no-op 且三指标微弱回退。v28 是真实进入 train/eval pipeline 的方法改动，新增 policy-view tail-safe alpha scale；首个 real run 暴露了 region-risk 参数转发 bug，已修复并用 W&B online 重新启动。v28 仍在 medium 验证中，因此不能作为 headline 结果。

**Paper-loop 更新，2026-06-23：** `NOT COMPLETE`。mentor/PPT 技术报告已按当前 Phase-J 主结果和 v48 representation-level atlas 证据刷新。v48 现在补齐了 full9 effective summary，相对 same-evidence no-op compact baseline 为 `7 / 9` 三指标严格胜出、`8 / 9` non-regressive/tie，均值 `+0.001462` PSNR、`+0.00002774` SSIM、`-0.00003953` LPIPS。这是一个真实的 train-evidence surface-atlas policy，并带自动回退；但它仍然是 ablation / next-step 结果：`stump` 被 gate 拒绝回退 no-op，`treehill` 有轻微 LPIPS 回退。证据：[`v48 auto-support atlas 日志`](docs/car_model/6-23-v48-AutoSupportSurfaceAtlas-Log.md)、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_full9_missing_scene_small_artifacts_20260623`。

**Support-capacity 更新，2026-06-23：** `NOT COMPLETE`。v51-fast support-footprint ladder 已按固定 train-only policy 跑完 full9（support candidates `2048,4096`，texture `32`，fill `nearest_observed`）。它相对 same-evidence no-op 为 `6 / 9` strict，相对 v50 为 `5 / 9` strict / `8 / 9` non-regressive；但相对 v48 只在 cap-hit `counter/kitchen/bonsai` 三个场景严格更好，full9 mean PSNR 略低于 v48，因为固定 texture/fill 放弃了 v48 auto-policy 的优势。证据：[`v51 support ladder full9 日志`](docs/car_model/6-23-v51-SupportFootprintLadder-Full9-Log.md)、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_full9_summary.md`。

**Capacity-policy 更新，2026-06-23：** `NOT COMPLETE`。v52 已把 v48/v51 选择固化成 train-only capacity-aware effective policy：非 cap-hit 场景保留 v48，只有 v48 触及 `2048` support cap 且 v51 用更大 support 通过 gate 时才升级。该策略选择 `counter/kitchen/bonsai` 走 v51，其余走 v48。相对 v48 的 full9 effective delta 为 `+0.000086890` PSNR、`+0.000008782` SSIM、`-0.000015303` LPIPS，`9 / 9` non-regressive/tie。selected small-artifact tree 已包含 9 个场景的 render/GT symlink，并补了 selected-render HTML gallery 与 cap-hit 局部对比图；一键 artifact pipeline 可刷新这些小产物、gallery、panel 和 manifest。W&B-logged source-config rerun 现已完成：`9 / 9` 场景复现，missing 为 `0`，在 `1e-5` reproducibility tolerance 下 metric mismatch 为 `0`。这闭合了 v52 的可复现性缺口，但 v52 仍是小收益 representation-level policy，不是最终论文 endpoint。证据：[`v52 capacity-aware policy 日志`](docs/car_model/6-23-v52-CapacityAwarePolicy-Log.md)、`scripts/car_model/run_v52_capacity_aware_pipeline.py`、`scripts/car_model/plan_v52_capacity_aware_source_rerun.py`、`scripts/car_model/summarize_v52_source_rerun.py`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/v52_capacity_aware_pipeline_report.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_plan.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_status.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/qualitative_gallery.html`、`assets/spcarnet_v52_capacity_policy_cap_hit_panel.png`。

**Alpha-calibration 探针，2026-06-23：** `NOT PROMOTED`。v53 给 surface residual atlas 加入 policy-val least-squares alpha calibration，并在 v52 三个 cap-hit 场景上用 W&B 验证。它证明 residual amplitude 确实是瓶颈，但全局 alpha 过于粗糙：`kitchen` 的 PSNR/LPIPS 提升但 SSIM 回退，`counter` 低于 v52，`bonsai` 被安全 gate 拒绝。因此 v53 只作为负结果/诊断记录，不作为新 endpoint。证据：[`v53 alpha calibration 日志`](docs/car_model/6-23-v53-PolicyValAlphaCalibration-CapHit-Log.md)。

**Face-alpha calibration 探针，2026-06-23：** `NOT PROMOTED_AS_GLOBAL_REPLACEMENT`。v55d 加入 policy-val per-face/local alpha calibration 和 effective alpha cap。它在 `counter` 上严格超过 v52（`+0.002670` PSNR、`+0.00001156` SSIM、`-0.00017697` LPIPS），但没有闭合 cap-hit 三场景：`kitchen` 的 PSNR/LPIPS 提升但 SSIM 回退，`bonsai` 三指标低于 v52。下一步候选是固定 reliability guard：只有 local-alpha 证据足够密、且不需要高全局 multiplier 时才启用 v55d。证据：[`v55d face-alpha calibration 日志`](docs/car_model/6-23-v55d-FaceAlphaCalibration-CapHit-Log.md)。

**Face-alpha guard 候选，2026-06-23：** `REPORT_ONLY_EFFECTIVE_POLICY_CANDIDATE`。v56 对 v52/v55d 应用固定 train/policy-val audit guard：只有 local-alpha 证据足够密且 selected alpha 不高于 `0.5` 时启用 v55d，否则回退 v52。它只选择 `counter`，full9 相对 v52 为 `+0.000296699` PSNR、`+0.000001285` SSIM、`-0.000019663` LPIPS，`9 / 9` non-regressive/tie。artifact pipeline 现在已物化 selected full9 tree，`9 / 9` render/GT 均已链接，并生成 selected gallery 与 counter crop/error-map panel。它比 raw v55d 安全，但由于 guard 是看过 v55d held-out 后设计的，仍不是论文 endpoint。证据：[`v56 face-alpha guard 日志`](docs/car_model/6-23-v56-FaceAlphaReliabilityGuard-Log.md)、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_full9_summary.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_selected_full9/v56_face_alpha_guard_pipeline_report.md`、`assets/spcarnet_v56_counter_face_alpha_guard_panel.png`。

**v56 source-rerun / fresh-probe 更新，2026-06-24：** `CURRENT_MISSING_AUDIT_FRESH_PROBES_CLOSED`。新增 v56 source-rerun 脚本已从 source config 复现 selected `counter` v55d 行并接入 W&B（`26.756130 / 0.862126 / 0.251691`，guard pass，face-alpha count `394`），再结合已闭合的 v52 source-rerun roots 得到 v56 effective-policy status：`9 / 9` completed，`0` missing，`0` mismatch。新的 `flowers/treehill/bicycle/garden/stump/room` v55d candidate fresh probes 都被固定 guard 拒绝：`flowers/treehill/bicycle/stump` 暴露 sparse local-alpha support 和 policy-val view robustness 不足；`garden/room` 虽然内部 accepted，但 fixed v56 因 support 或 worst-view SSIM margin 不够而拒绝。更严格的完整 `--min_target_changed_fraction 0.0` 审计也已完成，metrics 与 guard decisions 均不变。v56 因此是当前 fixed-command 审计覆盖已闭合的安全 report-only 候选，不是 paper endpoint。证据：[`v56 source-rerun/fresh-probe 日志`](docs/car_model/6-24-v56-SourceRerun-And-FreshProbe-Log.md)、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_source_status.md`、`outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_freshcheck_summary.md`。

**v57a reliability-shrink 探针，2026-06-24：** `REAL_PIPELINE_CHANGE_PROBED_NOT_PROMOTED`。v57a 在 atlas adapter 和 runner 中加入 train-policy-val face-alpha reliability shrink。`counter` W&B probe 相对 v52 仍正向（`+0.001548767` PSNR、`+0.000009656` SSIM、`-0.000117481` LPIPS），但弱于 raw v55d（`-0.001121521` PSNR、`-0.000001907` SSIM、`+0.000059486` LPIPS）。`kitchen` risk probe 保留了相对 v52 的 PSNR/LPIPS 收益，但 SSIM 仍回退（`-0.000099182`），因此 v57a 是有用接口但不能推广为当前 endpoint。证据：[`v57a face-alpha reliability shrink 探针`](docs/car_model/6-24-v57a-FaceAlphaReliabilityShrink-Probe-Log.md)。

## 当前结果

**评估口径。** Mip-NeRF360 同协议复现。每个场景的 clean MeshSplatting baseline 从 clean `26000` 与 `30000` checkpoint 中选择，只使用 held-out test 指标：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

训练集指标不用于选择 baseline，也不用于选择最终 test 结果。

**当前 Phase-J RGB endpoint。**

- 报告：`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`
- 场景：`9 / 9`
- 相对 selected clean MeshSplatting：`9 / 9` 三指标严格胜出
- 相对 Phase-F alpha-grid：`9 / 9` 三指标严格胜出
- 相对 selected clean MeshSplatting baseline 的均值提升：`+1.3311 PSNR`，`+0.0347 SSIM`，`-0.0634 LPIPS`
- 相对 Phase-F alpha-grid 的均值提升：`+0.3971 PSNR`，`+0.0083 SSIM`，`-0.0193 LPIPS`
- 平均三角形减少：`7.6479%`
- 闭环审计：`244 / 246` 个 held-out view 三指标严格胜出；max500 sparse COLMAP 口径下 `9 / 9` 场景 geometry-safe，`6 / 9` 场景严格几何更好。
- 外部 courtyard 验证：ETH3D courtyard clean9000 上 Phase-J 相对 clean MeshSplatting 最高提升 `+0.2642 PSNR`，`+0.0094 SSIM`，`-0.0225 LPIPS`；退化 F82 checkpoint 仅有极小提升，因此作为局限诊断保留。

| 场景 | 分支 | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | 三角形减少 |
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

## ECSR 升级状态

下一条主线是 **ECSR: Evidence-Certified Surface Relocation**。目标是把 SPCarNet 从 image-space residual repair 推进到 representation-level 的 surface compression 与 appearance recovery。

当前执行产物：

- Current-state audit：[`docs/car_model/5-8-ECSR-CurrentStateAudit.md`](docs/car_model/5-8-ECSR-CurrentStateAudit.md)
- Phase-A train-only surface evidence：[`docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md`](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md)
- Phase-B view-support graph：[`docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md`](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md)
- Phase-A/B cached-view policy split：[`docs/car_model/5-8-ECSR-PolicySplit.md`](docs/car_model/5-8-ECSR-PolicySplit.md)
- Full-train fitting/policy-val split：[`docs/car_model/5-8-ECSR-FullTrainPolicySplit.md`](docs/car_model/5-8-ECSR-FullTrainPolicySplit.md)
- Phase-C candidate preflight：[`docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md`](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md)
- Phase-C static topology certificate：[`docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md`](docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md)
- Phase-C materialized checkpoint smoke：[`docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md`](docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md)，[`docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md`](docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md)
- Phase-D attribute-only recovery smoke：[`docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md`](docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md)
- Phase-D constrained attribute recovery：[`docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md`](docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md)
- Phase-D surface residual delta smoke：[`docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md`](docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md)
- Phase-G teacher-bake recovery：[`docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md`](docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md)
- Phase-J guarded adaptive edge policy：[`docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md`](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md)
- Phase-J external courtyard validation：[`docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md`](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md)
- Surface-attached residual lumigraph V8：[`docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md`](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md)
- Phase-R fixed surface-SH1 ladder：[`docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md`](docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md)
- Phase-R 室内多折与 gamma trust 审计：[`docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md`](docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md)
- Phase-R 户外全折稳健审计：[`docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md`](docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md)
- 执行日志：[`docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md`](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md)
- Phase-A 汇总 contact sheet：`outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`

Phase-A 结果：`9 / 9` 场景通过 surface addressability，但只有 `4 / 9` 通过当前 top-support multiview consistency 检查。这说明 residual 信号是真实且可回投到表面的，但 naive 的单 face residual delta 还不能作为最终方法。

Phase-B 结果：固定 graph policy 在 full9 上找到 `123` 个 train-only local support cluster，其中 `23` 个是 certificate-contraction candidates，`99` 个是 surface-attribute recovery candidates。但 residual-hot cluster 的直接三角形压缩上限很小，因此下一步必须把 compression candidate 和 appearance-recovery candidate 分开，而不能把 residual hotspot 当成压缩目标。

Phase-C preflight 结果：`21 / 123` 个 Phase-B cluster 通过 train-only fitting/policy-val support-mask preflight，其中 `13` 个是 contraction 类型，`8` 个是 attribute-recovery 类型。它们还不是被接受的 ECSR 修改，只是进入 topology smoke test 与 before/after local rendering certificate 的第一批候选。

Phase-C/D 执行更新：full-train split 已覆盖全部 9 个场景。Static topology certification 在 `21` 个 preflight candidate 中通过 `7` 个；其中 `3` 个 contraction candidate 已被 materialize 成真实 checkpoint copy，并且 `3 / 3` 通过 renderer smoke。两个 representation-level recovery MVP 已实现但还不能作为最终方法：attribute-only recovery 的 `2 / 2` 个 smoke run 回退，bounded surface residual DC delta 虽然在 train policy-val mean-L1 上接受 `3 / 4`，但 held-out diagnostic `4 / 4` 回退。这说明 checkpoint 接口已经打通。

Phase-G 尝试把 ELA teacher bake 回 topology-frozen checkpoint，但 official `bicycle` 与 `flowers` pilot 都略低于 clean MeshSplatting，且明显低于 render-time ELA，因此被拒绝为当前主线。后续 v30 triadic teacher-mask 在 `bonsai` 上证明 image-level bake 的 mask 已经更安全且真实生效，但 baked checkpoint 仍低于 selected clean（`dPSNR -0.0808`、`dSSIM -0.0026`、`dLPIPS +0.0041`），所以下一步 representation 改进应转向 surface-addressed teacher residual basis，而不是继续调 global teacher loss。Phase-J 是当前接受的方法：一个 no-test-GT guarded portfolio，稳定时用 adaptive alpha，不稳定时用 train-selected structural edge fallback。

Phase-M / V8 目前是最干净的 representation-attached recovery baseline：train residual 存在 surface `face_id` 上，held-out view 只通过 target surface map 查表应用。固定 two-split consensus policy 接受 `flowers` 与 `garden`，其余 `7 / 9` 场景自动 no-op；相对 Phase-F compact base 的 full9 均值为小幅正向变化：`+0.000250` PSNR、`+0.000000868` SSIM、`-0.00000638` LPIPS。这个结果不是当前论文主 RGB endpoint，但它给下一步更高容量的 surface-attached 表示恢复提供了安全基线。

Phase-R 进一步把 residual 写回 checkpoint 中的 surface SH1 属性，并加入 train-only gamma trust-region residual gate。新的 v11 审计把户外候选也纳入与室内一致的四折 train-only gate，修正了 v10 对完成度的乐观估计：v11 只接受 `3 / 9` 个 representation edit（`stump`、`room`、`kitchen`），report-only 三指标严格胜出 `3 / 9`，相对 Phase-J no-op fallback 的均值为 `+0.002531` PSNR、`+0.000080` SSIM、`-0.000120` LPIPS。它更可靠，但完成度更低：`bicycle`、`flowers`、`garden`、`counter`、`bonsai`、`treehill` 都仍是 fallback，因此 Phase-R 目前是严格的 representation-level baseline，而不是最终视觉 endpoint。

Phase-S 是当前 representation-level repair 主线。它使用 face-local SH1
residual carrier、train-only face/view consensus，并且在真正 materialize
checkpoint edit 前加入 per-face gain certificate。risk-tail selector 在全部
8 个有候选场景上固定测试 `top1x2,risk4x1,risk8x0.5`，所有 render gate 都启用
W&B 记录。它接受 `flowers`、`counter`、`treehill`，拒绝
`garden/bicycle/room/kitchen/bonsai`，被拒绝场景回退到 Phase-J。full8 mean
effective report-only delta 为 `+0.000684500` PSNR、`+0.000058956` SSIM、
`-0.000073545` LPIPS。per-face alpha refit 已经接入 materializer，但第一轮
`counter/garden/bicycle` pilot 没有优于 uniform risk-tail，因此作为负结果保留。
GeoRisk/CVaR 新增几何邻域惩罚、per-face train-certificate tail risk、
局部 residual concentration，以及 train-val render CVaR 诊断。本轮要求的
7 场景 replay 只接受 `flowers` 与 `counter`；它是审计/policy 改进，而不是新的性能突破。
PatchRisk 与 direct PatchCert 进一步加入显式局部 patch carrier，其中 direct
PatchCert 更强：v5 固定 5 场景 replay 接受 `bicycle`；新的 v6 compact-stratified
gate 要求小容量 carrier，同时约束 train-val aggregate、tail 与分层 view-group
risk，因此公平接受 `bicycle` 和 `flowers`。`garden`、`counter`、`bonsai` 仍回退到
Phase-J。
v19b/v20 carrier-holdout 线进一步收紧审计：policy-val tuning samples 与
carrier holdout samples 分离，strict replay 会检查 cluster-basis 完整性，v20
还使用 deterministic auto-prefix carrier policy，避免手动 top-k carrier 扫描。
代价也很明确：full9 v20 continuation 虽然在公平 train-val gate 下接受
`garden` 与 `room`，但这两行在 held-out test 上接近 no-op。因此固定 portfolio v2
保留已经为正的 GeoRisk `flowers/counter`，再加入很小的 v20 `garden/room` 行，其它
场景回退到 Phase-J。

在 object-prior 侧，nested K=8 SPCarNet selector 新增了基于 observed-visible
preservation 的 `visible_only` 策略。在 206 个验证物体上，它相对 contained
K=1/first candidate 同时改善四个 inference-time 指标：recon
`0.06786 -> 0.06259`，hidden `0.10013 -> 0.09425`，free-space
`0.03643 -> 0.03217`，visible preservation `0.06246 -> 0.05592`。这是明确的
selector 升级，但 oracle 行仍然更强，因此 shape completion 故事还没有闭环。

## 其他评估口径

当前 Phase-J 摘要：

| 评估口径 | 结果 |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` 三指标严格胜出，均值 `+1.3311` PSNR，`+0.0347` SSIM，`-0.0634` LPIPS |
| Phase-F alpha-grid 前序方法 | `9 / 9` 三指标严格胜出，均值 `+0.3971` PSNR，`+0.0083` SSIM，`-0.0193` LPIPS |
| guarded branch decision | `8 / 9` adaptive-alpha branch，`1 / 9` train-selected edge fallback |
| 几何 / 拓扑 | 平均三角形减少 `7.6479%`；Phase-J closure audit 下 `6 / 9` sparse geometry 严格更好，`9 / 9` geometry-safe |
| per-view audit | `244 / 246` 个 held-out view 相对 selected clean baseline 同时提升 PSNR、SSIM、LPIPS |
| 外部验证 | ETH3D courtyard clean9000 三指标严格胜出：最高 `+0.2642` PSNR，`+0.0094` SSIM，`-0.0225` LPIPS；相对旧 ELA7 为 mixed |
| Phase-R v11 full-robust representation ladder | 多折 train-only 接受 `3 / 9`，report-only 三指标严格胜出 `3 / 9`，相对 Phase-J no-op fallback 均值 `+0.002531` PSNR，`+0.000080` SSIM，`-0.000120` LPIPS；该结果取代更乐观的 v10 单折/多折混合快照 |
| Phase-S risk-tail full8 | 8 个有候选场景接受 `3 / 8`，相对 Phase-J fallback 的 mean effective report-only delta 为 `+0.000684500` PSNR，`+0.000058956` SSIM，`-0.000073545` LPIPS；定性图在 `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative` |
| Phase-S GeoRisk/CVaR 7-scene replay | 本轮要求的 7 个 hard/control 场景接受 `2 / 7`（`flowers`、`counter`），相对 Phase-J fallback 的 mean effective report-only delta 为 `+0.000782013` PSNR，`+0.000067328` SSIM，`-0.000083983` LPIPS；新增几何/CVaR 可审计诊断，但没有超越旧 risk-tail 覆盖范围；定性图在 `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative` |
| Phase-S PatchRisk / direct PatchCert carrier | PatchRisk strict 5 场景 replay 接受 `1 / 5`（`counter`），均值 `+0.000014877` PSNR，`+0.000000072` SSIM，`-0.000000089` LPIPS；direct PatchCert v5 接受 `1 / 5`（`bicycle`），mean effective `+0.000077` PSNR，`+0.000007` SSIM，`-0.000023` LPIPS；direct PatchCert v6 compact-stratified gate 接受 `2 / 5`（`bicycle`、`flowers`），相对 Phase-J fallback 的 mean effective 为 `+0.001163` PSNR，`+0.000101` SSIM，`-0.000141` LPIPS；定性图在 `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v6_compactstrat_gate_20260514_qualitative` |
| Phase-S v20 auto-prefix PatchCert | deterministic disjoint carrier-holdout auto-prefix policy；full9 continuation 已补齐 `9 / 9` decisions；train-val gate 接受 `2 / 9`（`garden`、`room`），`bicycle`、`flowers`、`counter`、`bonsai`、`kitchen` 被 balanced/tail gate 拒绝，`stump/treehill` 为 no-op；接受行 report-only delta 基本是零量级，因此这是审计覆盖提升，不是可视化突破 |
| Phase-S fixed portfolio v2 | 在 GeoRisk/PatchRisk/v19b/v20 candidates 上只用 train-val 选择；接受 `4 / 9`（`flowers=georisk`、`counter=georisk`、`garden=v20`、`room=v20`），其它场景回退；mean effective report-only delta 为 `+0.000608232` PSNR，`+0.000052366` SSIM，`-0.000065320` LPIPS；summary 在 `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_portfolio_policy_v2_20260515/portfolio_summary.md` |
| Phase-S effect-aware portfolio v1 | 只用 train-val 的 portfolio，同时加入 non-noop、operator-pass 与 effect-size gate；接受 `3 / 9`（`bicycle=patchcert_v6`、`flowers=gaincert_v2`、`counter=riskpilot`），拒绝 v20 near-noop 行；mean effective report-only delta 为 `+0.000652101` PSNR，`+0.000056287` SSIM，`-0.000078238` LPIPS；summary 在 `outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v1_20260515/portfolio_summary.md` |
| Phase-S render-visible region-prior robust | 用 train-only 图像残差连通区域反投影成 face carriers，再拟合 shared face-local residual field；只有默认 gate 通过且 LPIPS/tail/stratified train-val robustness 同时通过时才提升，否则回退 Phase-J；full9 默认 gate 接受 `4 / 9`，robust promotion 接受 `2 / 9`（`garden`、`kitchen`）；相对 Phase-J fallback 的 robust effective report-only delta 为 `+0.000298606` PSNR、`+0.000006563` SSIM、`-0.000020499` LPIPS；summary 在 `outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/phase_s_regionprior_full9_robust_summary.md` |
| Phase-S region core/context weighted portfolio | 把 render-visible region 的 core/context/outside membership 接入 Phase-S fitting 权重，再用固定 train-val-only portfolio 选择；5 月 20 日 strictcompact re-decision 已把 compact/tail/stratified gate 失败变成真正拒绝，因此 raw corectx 在 `kitchen/bonsai/counter` 上的 false positive 不再 eligible；最终 full9 仍接受 `5 / 9`（`bicycle=patchcert_v6`、`flowers=rvregion_corectx_strictcompact`、`garden=rvregion_garden`、`counter=riskpilot`、`kitchen=rvregion_indoor`），相对 Phase-J fallback 的 mean effective report-only delta 为 `+0.000947740` PSNR，`+0.000062552` SSIM，`-0.000098634` LPIPS；summary 在 `outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v2_strictcompact.md` |
| Phase-S v20 定性诊断 | contact sheets 已复制到 `assets/spcarnet_phase_s_v20_remainingA_contact_sheet.png`、`assets/spcarnet_phase_s_v20_remainingB_contact_sheet.png`、`assets/spcarnet_phase_s_v20_remainingC_contact_sheet.png`；这些是放大差分诊断图，不是强 full-frame visual win |
| Phase-S gaincert v1 | 四折 gate 接受 `garden`、`flowers`、`bonsai`、`kitchen`、`room` 与近似 no-op 的 `stump`；拒绝 `bicycle`；`counter/treehill` 在 single-gate 阶段被阻断 |
| full9 paper-loop collector | clean-best `9 / 9`，Phase-J `9 / 9`，Phase-J 相对 clean-best 三指标严格胜出 `9 / 9`；Phase-S closure 为 `False`，因为严格 gate 只有 `7 / 9`，接受 `6 / 9`，且只有 `3 / 7` 是 train-val all-axis 胜出 |
| Stage ELA12 clean-best audit | selected-clean 子集仍是 `5 / 5` strict full-pass，per-view RGB pass 为 `164 / 165`，envelope pass 为 `163 / 165`；这不是 Mip-NeRF360 全 9 场景 benchmark |
| SPCarNet visible selector | `visible_only` 相对 contained K=1/first 改善 nested K=8 recon/hidden/free/visible 指标；oracle gap 仍存在 |

**最新 Phase-S region core/context 定性图。** A 组展示 raw `flowers/garden`
正向行；5 月 20 日 strictcompact policy 只从这组 core/context 候选中提升
`flowers`，因为 raw `garden` edit 超出 compact patch budget，而旧的
garden-specific region prior 更干净。B 组是诊断图：这些 false-positive 场景
现在会被 required compact/tail/stratified train-val gate 拒绝。

![Phase-S region core/context A](assets/spcarnet_phase_s_region_corectx_A_contact_sheet.png)

![Phase-S region core/context B](assets/spcarnet_phase_s_region_corectx_B_contact_sheet.png)

strictcompact `flowers` 行也补充了 train-defined surface-support 局部评估。
mask/crop 位置来自训练残差支持，并投影到 held-out test 渲染后才计算指标。在前
12 个 eligible held-out views 上，crop PSNR/SSIM 为 `12 / 12` 胜，crop LPIPS
为 `11 / 12` 胜；均值 delta 为 `+0.010150` crop PSNR、`+0.00038835` crop
SSIM、`-0.00060000` crop LPIPS。

![Phase-S v2 strictcompact flowers local support](assets/spcarnet_phase_s_v2_strictcompact_flowers_local_support.png)

下面的详细表格保留自 5 月 7 日 Compact-ELA/SOR 留档报告，用于 provenance。LPIPS、AbsRel、DepthMAE、Normal 越低越好。

| 评估口径 | 结果 |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` RGB 胜出，均值 `+0.4979` PSNR，`+0.0158` SSIM，`-0.0234` LPIPS |
| MeshSplatting paper table | `9 / 9` RGB 胜出，均值 `+0.8685` PSNR，`+0.0366` SSIM，`-0.0465` LPIPS |
| clean checkpoint envelope | 9 个场景全部选择 clean `26000` 而非 clean `30000`；平均 score gap `+1.1029` |
| 几何 / 拓扑 | `5 / 9` strict all-axis pass，`9 / 9` RGB + compact + geometry-safe pass，平均三角形减少 `5.7632%` |
| 局部定性 crop | 室外局部 MAE 下降 `12.8%` 到 `32.0%`；混合室内/室外最高局部 MAE 下降 `43.6%` |

**相对 MeshSplatting paper table，留档/临时口径。**

这张表保留自 5 月 7 日 Compact-ELA/SOR 留档报告，用于 provenance。除非重新核对
paper-table 的 split、mask、metric implementation、checkpoint selection 和 rendering
setting，否则不能把它作为当前主论文 claim。当前安全 claim 是上文的本地
same-protocol selected-clean MeshSplatting 对比。

| 场景 | paper PSNR/SSIM/LPIPS | ours PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|
| bicycle | 23.04 / 0.641 / 0.348 | 23.91 / 0.694 / 0.280 | +0.87 | +0.0527 | -0.0677 |
| flowers | 19.34 / 0.480 / 0.417 | 20.18 / 0.547 / 0.351 | +0.84 | +0.0673 | -0.0660 |
| garden | 24.70 / 0.762 / 0.217 | 26.03 / 0.817 / 0.152 | +1.33 | +0.0551 | -0.0647 |
| stump | 24.78 / 0.678 / 0.316 | 25.36 / 0.713 / 0.282 | +0.58 | +0.0345 | -0.0343 |
| treehill | 20.53 / 0.540 / 0.428 | 21.20 / 0.588 / 0.358 | +0.67 | +0.0482 | -0.0699 |
| room | 28.52 / 0.873 / 0.271 | 29.13 / 0.885 / 0.249 | +0.61 | +0.0119 | -0.0223 |
| counter | 26.51 / 0.846 / 0.279 | 27.24 / 0.864 / 0.250 | +0.73 | +0.0181 | -0.0293 |
| kitchen | 27.42 / 0.858 / 0.227 | 28.00 / 0.877 / 0.199 | +0.58 | +0.0189 | -0.0281 |
| bonsai | 28.19 / 0.876 / 0.294 | 29.78 / 0.898 / 0.257 | +1.59 | +0.0222 | -0.0366 |

**Clean `26000` / `30000` baseline envelope。**

| 场景 | selected | score 26000 | score 30000 | score gap | clean26000 PSNR/SSIM/LPIPS | clean30000 PSNR/SSIM/LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 26000 | 29.857 | 28.894 | +0.963 | 23.30 / 0.660 / 0.332 | 23.02 / 0.641 / 0.347 |
| flowers | 26000 | 22.027 | 21.060 | +0.968 | 19.68 / 0.512 / 0.395 | 19.39 / 0.492 / 0.408 |
| garden | 26000 | 36.604 | 35.623 | +0.981 | 25.03 / 0.780 / 0.201 | 24.71 / 0.762 / 0.216 |
| stump | 26000 | 33.428 | 32.347 | +1.081 | 25.21 / 0.705 / 0.294 | 24.87 / 0.684 / 0.309 |
| treehill | 26000 | 24.104 | 23.124 | +0.980 | 20.93 / 0.565 / 0.406 | 20.65 / 0.545 / 0.421 |
| room | 26000 | 41.446 | 40.575 | +0.871 | 28.75 / 0.885 / 0.250 | 28.48 / 0.873 / 0.268 |
| counter | 26000 | 38.953 | 37.772 | +1.181 | 26.75 / 0.862 / 0.252 | 26.41 / 0.846 / 0.278 |
| kitchen | 26000 | 41.364 | 39.940 | +1.424 | 27.82 / 0.876 / 0.199 | 27.30 / 0.858 / 0.226 |
| bonsai | 26000 | 41.633 | 40.156 | +1.477 | 28.90 / 0.896 / 0.259 | 28.38 / 0.879 / 0.290 |

**几何与拓扑。**

| 场景 | dAbsRel | dDepthMAE | dNormal | 三角形减少 | 顶点减少 | 状态 |
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

## 定性对比

第一组图是公平的全图 held-out render 对比。它的作用是证明比较来自同一 test view 和同一套 selected clean MeshSplatting baseline；但 SPCarNet 当前很多收益属于 residual-level 改善，放在全图尺度上确实不容易被肉眼直接看出来。

<p align="center">
  <img src="assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的全图定性对比">
</p>

最有说服力的定性展示是下面这组 Phase-J closure-audit 局部 held-out
error-reduction 图。它由
[`scripts/car_model/generate_spcarnet_advantage_showcase.py`](scripts/car_model/generate_spcarnet_advantage_showcase.py)
根据 `phasej_closure_audit.csv` 和 `phasej_per_view_deltas.csv` 自动生成：每个场景先要求该 view 在同一 selected-clean full9 口径下满足全图 `dPSNR > 0`、`dSSIM > 0`、`dLPIPS < 0`，再在该 view 内寻找纹理区域中 SPCarNet 相对 GT 的局部 RGB 误差下降最大的位置。绿色表示 SPCarNet 比 clean MeshSplatting 更接近 GT，紫红色表示变差。

<p align="center">
  <img src="assets/spcarnet_phasej_where_it_helps_showcase_20260622.png" width="980" alt="SPCarNet Phase-J 与 clean MeshSplatting 的局部 held-out 误差下降对比">
</p>

这张新的 Phase-J 专用图更适合放进汇报，因为它的 manifest 直接指向当前接受的 endpoint
`ours_26000_phasej_guarded_adaptedge_ela`。旧的室外图和混合室内/室外图仍保留为 provenance/backup：

<p align="center">
  <img src="assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的室外局部 held-out 误差下降对比">
</p>

<p align="center">
  <img src="assets/spcarnet_m360_where_it_helps_showcase.png" width="980" alt="SPCarNet 与 clean MeshSplatting 的混合局部 held-out 误差下降对比">
</p>

最新的 representation-level Phase-S PatchCert 证据幅度更小，但适合用于方法研发汇报：
它展示的是真正 checkpoint edit，而不仅是 render-time ELA。v6 compact-stratified
gate 接受 `bicycle` 和 `flowers`；被拒绝的行也放在同一张图里，用来说明安全回退逻辑。

<p align="center">
  <img src="assets/spcarnet_phase_s_patchcert_v6_compactstrat_contact_sheet.png" width="980" alt="Phase-S PatchCert v6 compact-stratified 定性对比">
</p>

当前固定 Phase-S portfolio v2 更完整但仍然保守：full9 上接受
`flowers/counter` 的 GeoRisk 行，以及 `garden/room` 的 v20 行。真正有可见价值的
正向例子仍主要是 GeoRisk `flowers/counter`；v20 contact sheet 放在下面作为诊断证据，
因为它们虽然被 train-val gate 接受，但视觉上接近 no-op。

<p align="center">
  <img src="assets/spcarnet_phase_s_portfolio_flowers_georisk_panel.png" width="980" alt="Phase-S portfolio GeoRisk flowers 定性对比">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_portfolio_counter_georisk_panel.png" width="980" alt="Phase-S portfolio GeoRisk counter 定性对比">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_v20_remainingA_contact_sheet.png" width="980" alt="Phase-S v20 remainingA 诊断 contact sheet">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_v20_remainingB_contact_sheet.png" width="980" alt="Phase-S v20 remainingB 诊断 contact sheet">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_v20_remainingC_contact_sheet.png" width="980" alt="Phase-S v20 remainingC 诊断 contact sheet">
</p>

较早的 render-visible region-prior/core-context 分支是一个有用的表示层里程碑。
它比纯 face-score 选择更贴近可见残差区域，但 full9 收益仍然很小。后续
v48-v73 surface-atlas 诊断已经取代它，成为当前 representation-level 研究主线；
核心科学判断仍然是：完全 baked 到表示中的收益还比较细微。

<p align="center">
  <img src="assets/spcarnet_phase_s_regionprior_garden_contact_sheet.png" width="980" alt="Phase-S render-visible region-prior garden 诊断 contact sheet">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_regionprior_indoor_contact_sheet.png" width="980" alt="Phase-S render-visible region-prior indoor 诊断 contact sheet">
</p>

<p align="center">
  <img src="assets/spcarnet_phase_s_regionprior_outdoor_contact_sheet.png" width="980" alt="Phase-S render-visible region-prior outdoor 诊断 contact sheet">
</p>

选图清单：`assets/spcarnet_phasej_where_it_helps_selection_20260622.json`、
`assets/spcarnet_m360_outdoor_detail_selection.json`、
`assets/spcarnet_m360_where_it_helps_selection.json`，以及早期全图清单
`assets/spcarnet_m360_full9_gallery_selection.json`。

| 定性 crop | 全图 delta PSNR/SSIM/LPIPS | 局部 dPSNR | 局部 MAE 下降 |
|---|---:|---:|---:|
| bonsai / `00001.png` | +6.63 / +0.0452 / -0.0878 | +11.79 | 78.6% |
| kitchen / `00011.png` | +3.43 / +0.0250 / -0.0578 | +10.48 | 71.4% |
| room / `00011.png` | +3.50 / +0.0220 / -0.0656 | +10.36 | 67.7% |
| counter / `00013.png` | +2.17 / +0.0407 / -0.0665 | +6.02 | 54.9% |
| garden / `00006.png` | +1.74 / +0.0479 / -0.0678 | +4.26 | 44.4% |
| flowers / `00014.png` | +1.12 / +0.0754 / -0.1028 | +2.15 | 25.3% |

## 方法概述

当前方法由三个只依赖 train split 的阶段组成。

1. **稀疏遮挡保护的压缩。** CSEF/SOR selector 用训练视角证据给三角形打分。室外场景证据稳定时可以删除约 10% faces；室内几何已经非常稳定时，则启用 micro-budget guard，避免为了压缩数字而破坏指标。

2. **checkpoint-safe 拓扑改写。** 根据 selector 删除面并重写 Mesh Splatting checkpoint，同时保证 face index remap 与 vertex attributes 长度一致。当前版本修复了 room 场景中 trailing unused vertices 导致的渲染 OOM/索引错配问题。

3. **Evidence Lumigraph Adapter。** ELA 用训练渲染得到的 RGB/depth/camera 证据，把局部 residual 信息转移到 held-out view。室内场景使用低分辨率证据，再把 residual 上采样到全分辨率。上采样 alpha 只在 train 视角上选择，并使用 PSNR/SSIM/LPIPS strict filter 加 SSIM-peak guard。

它不是简单的工程补丁，而是一个受约束的决策策略：只有几何证据允许时才压缩，只有训练证据认证 residual 时才修复；否则宁愿 no-op 或 micro-edit，也不提交不安全的“看起来变好”的结果。

### 可选：alpha 选择上的 Frechet-distance 门

alpha selector 提供一个可选的 Frechet 距离信号，作为又一个 train-only non-regression 门，思路移植自 Yang et al., "Representation Frechet Loss for Visual Generation"（[FD-Loss](https://github.com/Jiawei-Yang/FD-Loss)）。对每个 alpha 候选，selector 在 train 校准视角上累计 DINOv2 ViT-B/14 cls 特征（统一在 backbone 的 518x518 输入下打包），估计经验高斯，并计算与 GT batch 的闭式 Frechet 距离。FD 在这里是 calibration 信号，不是训练损失，也从不接触 test GT。

两种使用模式：

- **`--fd_strict`（推荐先用）**：任何 `alpha > 0` 候选只要其 `fd_gain = FD(base, gt) − FD(alpha, gt)` 跌破 `-fd_strict_tol` 就被剔除；`alpha = 0` 永远豁免，作为干净 fallback。这是纯 non-regression 过滤器，**不会**干扰幸存候选之间 PSNR / SSIM / LPIPS 的排序。
- **`--fd_weight w`（高级）**：把 `w * fd_gain` 加进现有 selection score。在 ~32 train views 上 DINOv2 raw FD 的典型量级是 O(5–30)，而其它项是 O(1)，因此 `w` 一旦明显大于 `~0.05` 就会**主导**整个 score。把它当成一个调参旋钮，**不要**当默认值用；做 portfolio 的话首选 `--fd_strict` 单独使用。

默认值与安全网：

- 默认全关（`fd_weight=0`，`fd_strict=False`）；FD 零开销，旧行为按比特一致。
- `alpha=0` 直接复用 base 特征，所以它的 `fd_gain` 严格为 `0`（fallback 行不会因 FD 抖动）。
- 当可用校准视角少于 `--fd_min_views`（默认 8）时整个 FD 路径被跳过，并在 calibration 记录里写 `fd_skipped_reason`。这是为了避开 768-d 经验协方差秩亏、FD 差异主要由噪声而非信号决定的高方差区间。
- 这条路径仅支持单卡（不带 distributed all-gather、不带 streaming queue）。timm 权重下载失败时抛出 `FDBackboneUnavailable` 错误并给出缓存提示，而不是悄悄回退。

代码见 `utils/fd_loss.py` 与 `scripts/car_model/smoke_test_fd_loss.py`（FD 数学 + backbone 前向 + 端到端 `calibrate_alpha` 集成测试，覆盖 alpha=0 豁免与 `fd_min_views` 跳过）。2026-05-11 审计见 `docs/car_model/5-11-FD-Loss-Integration-Audit.md`：`--fd_weight 0.005` 能改善 outdoor mean LPIPS，但会牺牲 PSNR/SSIM，因此不升级为当前 all-axis 主方法。

## 为什么比 MeshSplatting 更好

MeshSplatting 本身已经很强，但 clean checkpoint 仍有视角相关模糊、局部颜色残差、训练迭代过拟合等问题。SPCarNet 在 baseline 外围加了两层控制：

- **几何感知的保守性。** 方法不会假设所有场景都应该同样比例剪枝。garden 和室内场景说明，强行提高压缩率会让论文 claim 变得不公平。
- **train-only 渲染修复。** ELA 不使用 test metric 选择结果，却能恢复局部视觉细节；几何指标仍然由 compact checkpoint 负责，避免用 image-space trick 掩盖拓扑损伤。

因此，当前提升不是“训练更久”或“挑一个更好 checkpoint”：很多 clean `30000` 在 held-out test score 下反而弱于 clean `26000`，而我们仍然超过被选中的 clean baseline。

## 消融总结

| 变体 | 检验内容 | 结果 |
|---|---|---|
| clean MeshSplatting `26000/30000` | 公平 baseline envelope | 9 个场景都由 held-out score 选择 clean `26000` |
| compact-only checkpoint | 只删面是否足够 | 几何安全，但不足以带来头条 RGB 提升 |
| Compact + ELA，无 SSIM-peak alpha guard | 单一 scalar score 是否足够 | room 的 PSNR/LPIPS 提升但 held-out SSIM 回退 |
| Compact + ELA，有 SSIM-peak guard | 当前策略 | 修复 room，并在所有室内场景保持同一个 train-only policy |
| 激进剪枝分支 | 是否可以硬推高压缩率 | 被拒绝；敏感场景出现渲染或几何回退 |
| 可选 FD gate（`--fd_weight > 0` 或 `--fd_strict`） | DINOv2 Frechet 距离是否在 LPIPS 之外提供一个额外的 train-only non-regression 门 | 默认关闭；2026-05-11 审计显示它更像 LPIPS-oriented portfolio 信号，会带来 PSNR/SSIM tradeoff，因此不作为主方法 |

更多消融、失败分支和经验教训见文末历史材料链接。

## 复现当前表格

当前留档 run 使用的固定路径：

```bash
OUT_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
POLICY_TAG=sor_adaptive_geo \
METHOD_NAME=ours_26000_sor_adaptive_geo_compact_ela \
CLEAN_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
DATA_ROOT=/data/peilincai/mesh_datasets/mipnerf360 \
SPARSE_OCCLUDER_POLICY=1 \
SPARSE_ADAPTIVE_GEOMETRY_BUDGET=1 \
INDOOR_POLICY_IMAGE_ARG=images_8 \
INDOOR_EVIDENCE_IMAGE_ARG=images_8 \
EVIDENCE_SKIP_FAILED_VIEWS=1 \
WANDB_GROUP=paper_m360_compact_ela_sor_adaptive_geo_26k \
bash scripts/car_model/run_paper_m360_compact_ela_policy_available7.sh
```

收集最终表：

```bash
/home/peilincai/miniconda3/envs/Difix/bin/python \
  scripts/car_model/collect_paper_m360_compact_ela_policy_metrics.py \
  --method_root outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --policy_tag sor_adaptive_geo \
  --method_name ours_26000_sor_adaptive_geo_compact_ela \
  --method_iteration 26000 \
  --out_dir outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --wandb --wandb_project spcarnet_meshprior
```

## 局限与下一步

这个版本值得留档，但还没有真正完成“全面超越 MeshSplatting”的最终目标。

- 当前 Phase-J endpoint 的平均三角形减少是 `7.6479%`；但室内 micro-prune 场景仍然限制 rate-distortion 故事。
- strict all-axis pass 是 `5 / 9`，不是 `9 / 9`；剩余场景是 geometry-safe 或 geometry-neutral，而不是严格几何全胜。
- 当前最强的宽口径 RGB endpoint 仍是 Phase-J，也就是 render-time guarded ELA portfolio。固定 representation-level 路线目前仍停在 v64/v84 量级；v64/v79 是强锚点，v75-v78 是负诊断，v80 只做到 near-tie 且 LPIPS 略差，v81/v82 三项均回退，v82b 有 counter-only strict micro-win 但 raw hard-triad 失败，v83 因 SSIM 回退而是 mixed 诊断，v84 只是相对 v64 均值极小提升的 report-only selector，仍未形成可替代 Phase-J 的论文主结果。
- rate-distortion 报告必须包含顶点数与属性数，不能只报三角形数，因为 face-local SH1 在接受 face 上可能复制顶点。
- 下一阶段的研究目标是更强的 geometry-preserving compaction 与 representation repair operator，把 indoor/garden 压缩率拉上去，并解决被拒绝的室外场景，同时不破坏 RGB、稀疏深度和法向指标。

具体改进规划记录在 [`docs/car_model/5-7-Archive-Full9-CompactELA.md`](docs/car_model/5-7-Archive-Full9-CompactELA.md) 和 representation-level 升级路线 [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)。

## 历史材料

历史阶段日志不再堆在根目录 README 中：

- 旧版英文 README：[`docs/car_model/archive/README_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)
- 旧版中文 README：[`docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)
- 研究日志：[`docs/car_model/SPCarNet_research_log.md`](docs/car_model/SPCarNet_research_log.md)
- 5 月 7 日方法故事线：[`docs/car_model/5-7-Update.md`](docs/car_model/5-7-Update.md)
- Representation-level 升级路线：[`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)
