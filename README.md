# SPCarNet / MeshSplatOpt

**Train-only evidence-guided compact Mesh Splatting with geometry-safe reconstruction repair.**

[中文](README.zh.md) | [Current method/evidence log](docs/car_model/5-14-SPCarNet-Method-Modules-And-Evidence-Log.md) | [Phase-S GeoRisk/CVaR log](docs/car_model/5-14-PhaseS-GeoRiskCVaR-Selector-Log.md) | [Phase-S risk-tail/alpha log](docs/car_model/5-14-PhaseS-RiskTail-Alpha-ModuleLog.md) | [Phase-S coupled selector](docs/car_model/5-13-Coupled-Selector-Pilot.md) | [Phase-J result](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md) | [Surface-lumigraph V8](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md) | [Phase-R full-robust audit](docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md) | [Phase-S gaincert audit](docs/car_model/5-12-PhaseS-GainCertV1-Audit.md) | [SPCarNet selector audit](docs/car_model/5-12-SPCarNet-RagSym-Rerank-Audit.md) | [Full9 status](docs/car_model/5-12-Full9-PaperLoop-Evidence-Status.md) | [Closed-loop status](docs/car_model/5-12-PaperLoop-ClosedLoop-Status.md) | [Continuation report](docs/car_model/5-12-Subagent-PaperLoop-Continuation-Report.md) | [Phase-J external validation](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md) | [Current archive](docs/car_model/5-7-Archive-Full9-CompactELA.md) | [Execution log](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md) | [Research log](docs/car_model/SPCarNet_research_log.md) | [Legacy README](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)

SPCarNet is a research branch built on Mesh Splatting. The current ECSR version keeps the fixed Phase-F compact checkpoints, then uses a train-evidence guarded portfolio for appearance recovery: stable scenes use adaptive-alpha ELA, and unstable scenes use a train-selected structural edge fallback. No held-out test metric is used to select the branch, edge gate, alpha, or compaction ratio.

```text
current method: ours_26000_phasej_guarded_adaptedge_ela
report: outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

The May 7 Compact-ELA/SOR checkpoint remains archived as `archive/full9-compact-ela-ssim-peak-20260507` at commit `fae7942`. Phase-J is stronger on the current selected full9 RGB protocol, but it is still a render-time ELA portfolio rather than a fully baked representation-level endpoint.

**Paper-loop status, 2026-05-14:** `NOT COMPLETE`. Phase-J is the current strong endpoint: clean-best rows and Phase-J RGB rows are complete on `9 / 9` scenes, and Phase-J strictly beats the selected clean MeshSplatting row on `9 / 9`. Phase-S is now a real representation-level face-local repair branch with risk-tail selection: the full eight candidate-scene replay accepts `3 / 8` scenes and gives mean effective report-only deltas of `+0.000684500` PSNR, `+0.000058956` SSIM, and `-0.000073545` LPIPS over Phase-J fallback. The new GeoRisk/CVaR selector adds geometry-neighborhood ranking and train-val CVaR diagnostics, but on the requested 7-scene hard/control replay it accepts `2 / 7` scenes and does not expand coverage beyond the previous risk-tail positives. This is useful audit progress, not a final paper endpoint. Latest module/evidence logs: [`GeoRisk/CVaR`](docs/car_model/5-14-PhaseS-GeoRiskCVaR-Selector-Log.md), [`risk-tail/alpha`](docs/car_model/5-14-PhaseS-RiskTail-Alpha-ModuleLog.md).

## Current Result

**Protocol.** Mip-NeRF360 same-protocol reproduction. For every scene, the clean MeshSplatting baseline is selected from clean `26000` and `30000` checkpoints using held-out test metrics only:

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

Train metrics are not used to pick the baseline or the final method result.

**Current Phase-J RGB endpoint.**

- Report: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`
- Scenes: `9 / 9`
- Strict RGB wins vs selected clean MeshSplatting: `9 / 9`
- Strict RGB wins vs Phase-F alpha-grid: `9 / 9`
- Mean delta vs selected clean MeshSplatting: `+1.3311 PSNR`, `+0.0347 SSIM`, `-0.0634 LPIPS`
- Mean delta vs Phase-F alpha-grid: `+0.3971 PSNR`, `+0.0083 SSIM`, `-0.0193 LPIPS`
- Mean triangle reduction: `7.6479%`
- Closure audit: `244 / 246` held-out views are strict RGB wins; sparse COLMAP geometry is safe on `9 / 9` scenes and strictly better on `6 / 9` under the max500 audit.
- External courtyard validation: on ETH3D courtyard clean9000, Phase-J improves clean MeshSplatting by up to `+0.2642 PSNR`, `+0.0094 SSIM`, `-0.0225 LPIPS`; the degraded F82 checkpoint only shows tiny improvements, so it is kept as a limitation diagnostic.

| scene | selected branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
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

## ECSR Upgrade Status

The next method track is **ECSR: Evidence-Certified Surface Relocation**. Its goal is to move SPCarNet from image-space residual repair toward representation-level surface compression and appearance recovery.

Current execution artifacts:

- Current-state audit: [`docs/car_model/5-8-ECSR-CurrentStateAudit.md`](docs/car_model/5-8-ECSR-CurrentStateAudit.md)
- Phase-A train-only surface evidence: [`docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md`](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md)
- Phase-B view-support graph: [`docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md`](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md)
- Phase-A/B cached-view policy split: [`docs/car_model/5-8-ECSR-PolicySplit.md`](docs/car_model/5-8-ECSR-PolicySplit.md)
- Full-train fitting/policy-val split: [`docs/car_model/5-8-ECSR-FullTrainPolicySplit.md`](docs/car_model/5-8-ECSR-FullTrainPolicySplit.md)
- Phase-C candidate preflight: [`docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md`](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md)
- Phase-C static topology certificate: [`docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md`](docs/car_model/5-8-ECSR-PhaseC-StaticTopologyCertificate.md)
- Phase-C materialized checkpoint smoke: [`docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md`](docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md), [`docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md`](docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md)
- Phase-D attribute-only recovery smoke: [`docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md`](docs/car_model/5-8-ECSR-PhaseD-AttributeOnlySmoke.md)
- Phase-D constrained attribute recovery: [`docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md`](docs/car_model/5-8-ECSR-PhaseD-ConstrainedAttributeRecovery.md)
- Phase-D surface residual delta smoke: [`docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md`](docs/car_model/5-8-ECSR-PhaseD-SurfaceResidualDeltaSmoke.md)
- Phase-G teacher-bake recovery: [`docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md`](docs/car_model/5-8-ECSR-PhaseG-TeacherBakeRecovery.md)
- Phase-J guarded adaptive edge policy: [`docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md`](docs/car_model/5-8-ECSR-PhaseJ-GuardedAdaptiveEdgePolicy.md)
- Phase-J external courtyard validation: [`docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md`](docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md)
- Surface-attached residual lumigraph V8: [`docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md`](docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md)
- Phase-R fixed surface-SH1 ladder: [`docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md`](docs/car_model/5-10-ECSR-PhaseR-FixedCandidateLadder.md)
- Phase-R indoor multi-fold and gamma trust audit: [`docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md`](docs/car_model/5-11-PhaseR-Indoor-Multifold-Gate-Audit.md)
- Phase-R full-robust outdoor multi-fold audit: [`docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md`](docs/car_model/5-12-PhaseR-FullRobust-Outdoor-Multifold-Audit.md)
- Execution log: [`docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md`](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md)
- Combined Phase-A contact sheet: `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`

Phase-A result: `9 / 9` scenes pass surface addressability, but only `4 / 9` pass the current top-support multiview consistency check. This means the residual signal is real and surface-addressable, but a naive single-face residual delta is not yet a safe final method.

Phase-B result: the fixed graph policy finds `123` train-only local support clusters across full9, including `23` certificate-contraction candidates and `99` surface-attribute recovery candidates. The direct triangle-reduction upper bound of residual-hot clusters is tiny, so the next method step must separate compression candidates from appearance-recovery candidates instead of treating residual hotspots as the compression target.

Phase-C preflight result: `21 / 123` Phase-B clusters pass the train-only fitting/policy-val support-mask preflight (`13` contraction-type, `8` attribute-recovery-type). These are not accepted ECSR edits yet; they are the first eligible set for topology smoke tests and before/after local rendering certificates.

Phase-C/D execution update: the full-train split is complete for all 9 scenes. Static topology certification passes `7 / 21` preflight candidates; `3` contraction candidates were materialized as real checkpoint copies and all `3 / 3` pass renderer smoke. Two representation-level recovery MVPs are implemented but rejected as final methods: attribute-only recovery regresses `2 / 2` smoke runs, and bounded surface residual DC delta regresses `4 / 4` held-out diagnostics despite `3 / 4` train policy-val mean-L1 accepts. This established the checkpoint interface.

Phase-G tested teacher-baking ELA back into a topology-frozen checkpoint and was rejected: official `bicycle` and `flowers` pilots both remained slightly below clean MeshSplatting and far below render-time ELA. Phase-J is therefore the accepted current method: a no-test-GT guarded portfolio that uses adaptive alpha where stable and a train-selected structural edge fallback where adaptive alpha is unstable.

Phase-M / V8 adds the cleanest representation-attached recovery baseline so far: train residuals are stored on surface `face_id`s and applied to held-out views through target surface maps only. A fixed two-split consensus policy accepts `flowers` and `garden`, rejects the other `7 / 9` scenes as no-op, and gives a tiny positive full9 mean delta of `+0.000250` PSNR, `+0.000000868` SSIM, and `-0.00000638` LPIPS versus the Phase-F compact base. This is not the paper-facing RGB endpoint; it is the safe surface-attached baseline for the next higher-capacity representation work.

Phase-R upgrades this to checkpoint-baked surface SH1 residuals with a fixed candidate ladder plus a train-only gamma trust-region residual gate. A stricter v11 audit now runs the outdoor candidates through the same four-offset train-only gate used indoors. This corrected an optimistic v10 snapshot: v11 accepts only `3 / 9` representation edits (`stump`, `room`, `kitchen`), gives `3 / 9` report-only strict RGB wins, and has mean report-only deltas of `+0.002531` PSNR, `+0.000080` SSIM, and `-0.000120` LPIPS versus Phase-J with no-op fallback. The result is more reliable but less complete: `bicycle`, `flowers`, `garden`, `counter`, `bonsai`, and `treehill` remain fallback under the full-robust gate, so Phase-R is a rigorous representation-level baseline rather than the final visual endpoint.

Phase-S is the current representation-level repair branch. It uses face-local
SH1 residual carriers, train-only face/view consensus, and per-face gain
certificates before a checkpoint edit is materialized. The risk-tail selector
tests `top1x2,risk4x1,risk8x0.5` on all 8 candidate-bearing scenes with
W&B-logged render gates. It accepts `flowers`, `counter`, and `treehill`,
rejects `garden/bicycle/room/kitchen/bonsai`, and falls back to Phase-J on
rejection. The full8 mean effective report-only delta is `+0.000684500` PSNR,
`+0.000058956` SSIM, and `-0.000073545` LPIPS. Per-face alpha refit is wired
through the materializer, but the first `counter/garden/bicycle` pilot does not
improve over uniform risk-tail and is kept as a measured negative result.
GeoRisk/CVaR adds geometry-neighborhood penalties, per-face train-certificate
tail risk, local residual concentration, and train-val render CVaR diagnostics.
The requested 7-scene replay accepts `flowers` and `counter` only; it is an
audit/policy improvement, not a new performance breakthrough.

On the object-prior side, the nested K=8 SPCarNet selector now includes a
`visible_only` observed-visible preservation policy.  On 206 validation objects
it improves all four reported inference-time metrics versus the contained
K=1/first candidate: recon `0.06786 -> 0.06259`, hidden `0.10013 -> 0.09425`,
free-space `0.03643 -> 0.03217`, and visible preservation `0.06246 -> 0.05592`.
This is a real selector upgrade, while the oracle row remains better and keeps
the completion story open.

## Additional Evaluation Views

Current Phase-J summary:

| evaluation view | result |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` strict RGB wins, mean `+1.3311` PSNR, `+0.0347` SSIM, `-0.0634` LPIPS |
| Phase-F alpha-grid predecessor | `9 / 9` strict RGB wins, mean `+0.3971` PSNR, `+0.0083` SSIM, `-0.0193` LPIPS |
| guarded branch decision | `8 / 9` adaptive-alpha branch, `1 / 9` train-selected edge fallback |
| geometry / topology | mean triangle reduction `7.6479%`; `6 / 9` strict sparse-geometry wins, `9 / 9` geometry-safe scenes under the Phase-J closure audit |
| per-view audit | `244 / 246` held-out views strictly improve PSNR, SSIM, and LPIPS over the selected clean baseline |
| external validation | ETH3D courtyard clean9000 strict RGB win: up to `+0.2642` PSNR, `+0.0094` SSIM, `-0.0225` LPIPS; mixed vs older ELA7 |
| Phase-R v11 full-robust representation ladder | `3 / 9` multi-offset train-only accepted selections, `3 / 9` report-only strict RGB wins, mean `+0.002531` PSNR, `+0.000080` SSIM, `-0.000120` LPIPS vs Phase-J with no-op fallback; this supersedes the more optimistic v10 mixed single/multi-fold snapshot |
| Phase-S risk-tail full8 | `3 / 8` candidate-bearing scenes accepted, mean effective report-only delta `+0.000684500` PSNR, `+0.000058956` SSIM, `-0.000073545` LPIPS vs Phase-J fallback; qualitative panels in `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative` |
| Phase-S GeoRisk/CVaR 7-scene replay | `2 / 7` requested hard/control scenes accepted (`flowers`, `counter`), mean effective report-only delta `+0.000782013` PSNR, `+0.000067328` SSIM, `-0.000083983` LPIPS vs Phase-J fallback; this adds auditable geometry/CVaR diagnostics but does not beat the prior risk-tail coverage; qualitative panels in `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative` |
| Phase-S gaincert v1 | strict four-offset gate accepts `garden`, `flowers`, `bonsai`, `kitchen`, `room`, and near-no-op `stump`; rejects `bicycle`; `counter/treehill` are blocked by single-gate rejection |
| full9 paper-loop collector | clean-best `9 / 9`, Phase-J `9 / 9`, Phase-J strict RGB wins vs clean-best `9 / 9`; Phase-S closure is `False` because strict gates are `7 / 9` with `6 / 9` accepts and only `3 / 7` all-axis train-val wins |
| Stage ELA12 clean-best audit | selected-clean subset remains `5 / 5` strict full-pass with `164 / 165` per-view RGB pass and `163 / 165` envelope pass; this is not the full nine-scene Mip-NeRF360 benchmark |
| SPCarNet visible selector | `visible_only` improves nested K=8 recon/hidden/free/visible metrics versus contained K=1/first; oracle gap remains |

The detailed tables below are retained from the May 7 archived Compact-ELA/SOR report for provenance. Lower is better for LPIPS, AbsRel, DepthMAE, and Normal.

| evaluation view | result |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` RGB wins, mean `+0.4979` PSNR, `+0.0158` SSIM, `-0.0234` LPIPS |
| MeshSplatting paper table | `9 / 9` RGB wins, mean `+0.8685` PSNR, `+0.0366` SSIM, `-0.0465` LPIPS |
| clean checkpoint envelope | clean `26000` is selected over clean `30000` on all `9 / 9` scenes; mean score gap `+1.1029` |
| geometry / topology | `5 / 9` strict all-axis pass, `9 / 9` RGB + compact + geometry-safe pass, mean triangle reduction `5.7632%` |
| local qualitative crops | outdoor local MAE drop `12.8%` to `32.0%`; mixed indoor/outdoor local MAE drop up to `43.6%` |

**Against the MeshSplatting paper table.**

| scene | paper PSNR/SSIM/LPIPS | ours PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
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

**Clean `26000` / `30000` baseline envelope.**

| scene | selected | score 26000 | score 30000 | score gap | clean26000 PSNR/SSIM/LPIPS | clean30000 PSNR/SSIM/LPIPS |
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

**Geometry and topology.**

| scene | dAbsRel | dDepthMAE | dNormal | triangle red. | vertex red. | status |
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

## Qualitative Comparison

The first panel is the fair full-frame view comparison from real held-out renders. It is useful for checking that the comparison uses the same test views and the selected clean MeshSplatting baseline, but the improvement is often residual-level and therefore visually subtle at full-frame scale.

<p align="center">
  <img src="assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame qualitative comparison against clean MeshSplatting">
</p>

The stronger qualitative evidence is the local held-out error-reduction view below. It is generated by [`scripts/car_model/generate_spcarnet_advantage_showcase.py`](scripts/car_model/generate_spcarnet_advantage_showcase.py): for each scene, the script first requires full-view `dPSNR > 0`, `dSSIM > 0`, and `dLPIPS < 0` under the same full9 protocol, then searches that view for textured crops where SPCarNet reduces RGB error against GT. Green means SPCarNet is closer to GT than clean MeshSplatting; magenta marks pixels where it is worse.

<p align="center">
  <img src="assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor local held-out error reduction against clean MeshSplatting">
</p>

The outdoor crops make the practical visual gain clearer: clean MeshSplatting often shows local triangular/blocky smoothing on foliage, ground texture, bench slats, and bark, while SPCarNet recovers sharper residual detail. A mixed indoor/outdoor version is also provided:

<p align="center">
  <img src="assets/spcarnet_m360_where_it_helps_showcase.png" width="980" alt="SPCarNet mixed local held-out error reduction against clean MeshSplatting">
</p>

Selection manifests: `assets/spcarnet_m360_outdoor_detail_selection.json`, `assets/spcarnet_m360_where_it_helps_selection.json`, and the earlier full-frame manifest `assets/spcarnet_m360_full9_gallery_selection.json`.

| qualitative crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | +0.99 / +0.0616 / -0.0682 | +2.05 | 24.2% |
| garden / `00008.png` | +1.27 / +0.0432 / -0.0551 | +2.70 | 27.6% |
| treehill / `00010.png` | +0.59 / +0.0491 / -0.0881 | +3.03 | 32.0% |
| bicycle / `00021.png` | +1.13 / +0.0385 / -0.0615 | +1.88 | 17.5% |
| stump / `00007.png` | +0.26 / +0.0122 / -0.0208 | +0.81 | 12.8% |
| bonsai / `00001.png` | +2.79 / +0.0063 / -0.0007 | +3.82 | 43.6% |

## Method

The current method has three train-only stages.

1. **Sparse-occlusion protected compaction.** A CSEF/SOR selector scores triangles using train-view evidence. Outdoor scenes can remove around 10% of faces when evidence is stable. Indoor scenes with very low geometry error are protected by a micro-budget guard instead of being forced into destructive pruning.

2. **Checkpoint-safe topology rewrite.** The selected faces are removed from the Mesh Splatting checkpoint while keeping tensor shapes and face-index remapping consistent. The current version fixes a real room failure caused by trailing unused vertices in the checkpoint.

3. **Evidence Lumigraph Adapter.** ELA uses train-rendered RGB/depth/camera evidence to transfer local residual information to held-out views. Indoor scenes use low-resolution evidence and then upsample the residual to full resolution. The upsample alpha is selected only on train views with a strict PSNR/SSIM/LPIPS filter plus an SSIM-peak guard.

This is a research method rather than a post-hoc engineering patch because the main claim is a constrained decision policy: compact only when geometry evidence permits it, repair only when train evidence certifies the residual, and otherwise prefer a no-op or micro-edit over an unsafe apparent improvement.

### Optional: Frechet-distance gate on alpha selection

The alpha selector exposes an optional Frechet-distance signal as one more train-only non-regression gate, ported in spirit from Yang et al., "Representation Frechet Loss for Visual Generation" ([FD-Loss](https://github.com/Jiawei-Yang/FD-Loss)). For each candidate alpha the selector accumulates DINOv2 ViT-B/14 cls features over the train calibration views (batched at the backbone's 518x518 input size), estimates an empirical Gaussian, and computes the closed-form Frechet distance against the GT batch. The gate is a calibration signal, not a training loss, and never sees test GT.

Two modes:

- **`--fd_strict` (recommended first)**: any `alpha > 0` whose expected `fd_gain = FD(base, gt) - FD(alpha, gt)` drops below `-fd_strict_tol` is removed from the candidate set; `alpha = 0` is exempt and acts as a clean fallback. This is a pure non-regression filter and does not perturb the existing PSNR / SSIM / LPIPS ranking among the survivors.
- **`--fd_weight w` (advanced)**: adds `w * fd_gain` to the existing selection score. Raw DINOv2 FD on ~32 train views is typically O(5-30) while the other terms are O(1), so values much above `~0.05` will dominate the score. Treat this as a tunable knob, not a recommended default; for portfolio use, prefer `--fd_strict` alone.

Defaults and safety rails:

- Default is off (`fd_weight=0`, `fd_strict=False`); FD has zero overhead and the legacy behavior is bitwise identical.
- `alpha=0` reuses the base features so its `fd_gain` is exactly `0` (no numerical drift in the alpha=0 fallback row).
- If fewer than `--fd_min_views` (default 8) calibration views are available, FD is skipped and reported with `fd_skipped_reason` in the calibration record. This guards against the high-variance regime where the 768-d empirical covariance is rank-deficient and FD differences are dominated by noise rather than signal.
- The backbone runs single-GPU only; no distributed all-gather, no streaming queue. If the timm weights cannot be downloaded the gate raises a `FDBackboneUnavailable` error with cache hints rather than silently failing.

See `utils/fd_loss.py` and `scripts/car_model/smoke_test_fd_loss.py` (math + backbone forward + an end-to-end `calibrate_alpha` integration test that confirms the alpha=0 carve-out and the `fd_min_views` skip). The 2026-05-11 audit in `docs/car_model/5-11-FD-Loss-Integration-Audit.md` keeps FD optional: `--fd_weight 0.005` improved outdoor mean LPIPS but reduced PSNR/SSIM, so it is not promoted to the current all-axis main method.

## Why It Improves MeshSplatting

MeshSplatting already produces strong meshes, but its clean checkpoints still show view-dependent texture blur, local residual color errors, and overfitting sensitivity across iterations. SPCarNet adds two controls around the baseline:

- **Geometry-aware conservatism.** It does not assume every scene should be pruned equally. Garden and indoor scenes demonstrate that aggressive deletion can look attractive as a compression number but harm the fair claim.
- **Train-only view repair.** ELA improves RGB quality without selecting from held-out test metrics. It recovers residual visual detail while the compact checkpoint keeps the geometry accounting honest.

The result is not simply "train longer" or "pick a nicer checkpoint": clean `30000` is often worse than clean `26000` under held-out scoring, and the method still improves over the selected clean baseline.

## Ablation Summary

| variant | what it tests | outcome |
|---|---|---|
| Clean MeshSplatting `26000/30000` | fair baseline envelope | clean `26000` is selected on all 9 scenes by held-out score |
| Compact-only checkpoint | whether deletion alone is enough | safe but not enough for headline RGB gains |
| Compact + ELA without SSIM-peak alpha guard | whether scalar score alone is enough | room improves PSNR/LPIPS but loses held-out SSIM |
| Compact + ELA with SSIM-peak guard | current policy | restores room and keeps all indoor scenes fair under one train-only policy |
| Aggressive pruning branches | whether high compression can be forced | rejected; caused render/geometry regressions on sensitive scenes |
| Optional FD gate (`--fd_weight > 0` or `--fd_strict`) | whether DINOv2 Frechet distance adds a train-only non-regression gate beyond LPIPS | off by default; 2026-05-11 audit found LPIPS-oriented gains but PSNR/SSIM tradeoffs, so this remains an optional portfolio signal rather than the main method |

More detailed ablations and failed branches are archived in the research log and historical reports linked below.

## Reproduce Current Table

The archived run used the fixed method root:

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

Collect the final table:

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

## Limitations And Next Work

This version is promising, but it is not yet a complete "fully dominates MeshSplatting" endpoint.

- Average triangle reduction for the current Phase-J endpoint is `7.6479%`; indoor micro-pruned scenes still limit the rate-distortion story.
- Strict all-axis pass is `5 / 9`, not `9 / 9`; the remaining scenes are geometry-safe or geometry-neutral rather than strict geometry wins.
- The strongest broad RGB endpoint is still Phase-J, a render-time guarded ELA portfolio. Phase-S risk-tail is a real representation-level module, but it promotes only `3 / 8` candidate-bearing scenes and its mean is dominated by `flowers`.
- Rate-distortion reporting must include vertices and attributes, not only triangle count, because face-local SH1 can duplicate vertices on accepted faces.
- The next research target is a stronger geometry-preserving compaction and representation repair operator that can raise indoor/garden compression and solve the rejected outdoor scenes without breaking RGB, sparse depth, or normal metrics.

The concrete improvement plan is recorded in [`docs/car_model/5-7-Archive-Full9-CompactELA.md`](docs/car_model/5-7-Archive-Full9-CompactELA.md) and the representation-level upgrade roadmap [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md).

## Historical Material

Historical development logs are intentionally kept out of the top-level README:

- Legacy English README: [`docs/car_model/archive/README_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)
- Legacy Chinese README: [`docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)
- Research log: [`docs/car_model/SPCarNet_research_log.md`](docs/car_model/SPCarNet_research_log.md)
- May 7 method story: [`docs/car_model/5-7-Update.md`](docs/car_model/5-7-Update.md)
- Representation-level upgrade plan: [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)
