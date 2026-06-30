# SPCarNet / MeshSplatting Feedback for Next-Stage AI Model

Date: 2026-06-28

Latest update on 2026-06-30, v279-v280:

The newest branch is **train-fit baked surface feature texture + neural residual decoder** in `scripts/car_model/train_perceptual_surface_residual_decoder.py`. New files:

- `docs/car_model/6-30-v279-v280-SurfaceFeatureTexture-v169-Gate-Log.md`
- `docs/car_model/results/v279_v280_surface_feature_texture_summary.json`

Important new facts:

- v279 added a calibration face reliability gate and confirmed that reliability gating alone only makes the method conservative; it does not solve target perceptual transfer.
- v280 added `--surface_texture_mode v1`, a real representation-level upgrade. It bakes train-fit teacher residual statistics into a `face x UV-bin` surface feature texture and appends those features to the neural residual decoder during train, policy-val, and stripped no-GT target apply.
- v280a texture coverage: `65536` candidate faces, `1048576` bins, `0.998383` covered faces, `0.406538` covered bins, `3788244` train-fit samples.
- target/test apply no longer opens eval GT before adapted output is generated; eval GT is loaded only after no-GT apply for metrics.
- v280a greatly improves policy-val: gains `+0.031057 PSNR / +0.000530 SSIM / +0.000980 LPIPS`.
- v280a still fails target exact: `19.840939 / 0.618396 / 0.181078`, gains `+0.008885 / -0.001514 / -0.000743`.
- v280a is still `0.463419` PSNR below the Phase-J flowers gate. Do not run full9.
- Offline alpha-rescale diagnostic from v280a target renders shows that alphas `0.025` through `0.300` never achieve all-axis target gains; LPIPS remains negative even when SSIM becomes slightly positive.

Current direct verdict:

> v280 is the right kind of representation-level attempt, and it proves that a train-fit surface feature texture can carry more policy-val teacher signal. It still fails held-out target exact because the residual direction is perceptually unsafe. The next model should not continue alpha/face-gate scans. It should change the supervision target, for example patch/perceptual teacher residual distillation, or add a target-free residual-direction uncertainty model learned from source-view disagreement.

Latest update on 2026-06-30, v274:

The newest branch is **structure-safe texture low-rank residual carrier** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v274-StructureSafeTexture-Gate-Log.md`
- `docs/car_model/results/v274_structure_safe_texture_summary.json`

Important new facts:

- v274 adds a real decoder mode: `--residual_decoder_mode structure_safe_texture_lowrank`.
- The source bank now saves `residual_edge`, `residual_luma_abs`, and `teacher_better_fraction`.
- The texture low-rank branch now gates residual injection by source/target edge agreement, residual-edge support, teacher-better support, and unique-view support.
- A structure prefilter was required before covariance/eigendecomposition; otherwise full-resolution texture-lowrank evaluation became too slow.
- All effective v274 runs used W&B offline and completed flowers policy-val + target exact with target/test GT stripped before apply.

Effective results:

- v266c reference target: `19.845698 / 0.620201 / 0.179915`, gains `+0.013644 / +0.000290 / +0.000419`.
- v270d reference target: `19.844320 / 0.620226 / 0.179934`, gains `+0.012266 / +0.000315 / +0.000401`.
- v274d loaded-v266 bank target: `19.845704 / 0.620200 / 0.179917`, gains `+0.013650 / +0.000290 / +0.000418`.
- v274e fresh-fit structure stats target: `19.844540 / 0.620225 / 0.180015`, gains `+0.012486 / +0.000314 / +0.000320`.
- v274f loaded-v270 bank target: `19.844289 / 0.620224 / 0.179933`, gains `+0.012235 / +0.000314 / +0.000402`.

Direct verdict:

> v274 is a valid representation-level implementation and exact flowers validation, but not a paper-level win. It barely changes the v266c/v270d target frontier and remains about `0.459` PSNR below the Phase-J flowers gate. The fresh structure statistics improve policy-val but do not transfer to target exact. Do not run full9 from v274. The next model should move beyond local source-slot texture blending toward a learned view-dependent surface feature decoder or patch-level teacher residual carrier.

Latest update on 2026-06-30, v273:

The newest branch is **source-consensus residual denoise** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v273-ConsensusDenoise-Gate-Log.md`
- `docs/car_model/results/v273_consensus_denoise_summary.json`

Important new facts:

- v273 is not another scalar confidence head. It modifies the residual carrier by rewriting train-fit source residual slots toward leave-one-out source-view consensus residuals.
- New CLI:
  - `--source_consistency_mode denoise`
  - `--source_consistency_denoise_blend`
- v273a blend 0.50 denoised `626926` source slots, `69.81%` of valid slots, and reduced residual energy to `89.73%`.
- v273b blend 0.15 preserved more residual energy, `96.70%`, but still did not beat v266c target exact.
- v273 target exact:
  - v266c reference: `19.845698 / 0.620201 / 0.179915`
  - v270d reference: `19.844320 / 0.620226 / 0.179934`
  - v273a: `19.844213 / 0.620207 / 0.179945`
  - v273b: `19.844259 / 0.620205 / 0.179934`

Current direct verdict:

> v273 is a valid residual-bank/carrier modification but a quality failure. Source-consensus denoise improves policy-val but does not transfer to target exact. The current bottleneck is probably missing coherent view-dependent/high-frequency capacity, not source-slot residual noise. Do not continue denoise strength scans.

Latest update on 2026-06-30, v272:

The newest branch is **learned source-consistency feature head** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v272-LearnedConsistencyHead-Gate-Log.md`
- `docs/car_model/results/v272_learned_consistency_head_summary.json`

Important new facts:

- v272 added `--source_consistency_mode feature_only`, so source-view consistency can be used as a feature rather than another hard residual/weight multiplier.
- Checkpoints now save/load `source_consistency_apply_weight`, `source_consistency_apply_amplitude`, and `learned_ood_head_ceiling`.
- The learned OOD/gain head now sees source consistency reliability/amplitude/gap, base confidence, and raw residual magnitude.
- W&B offline logs and audit Markdown now record learned-head floor/ceiling.
- The implementation passed `py_compile`, `git diff --check`, CLI help, one smoke run, and four full flowers target exact runs.
- All v272 full runs improved policy-val, but none improved target exact over the v266/v270 frontier:
  - v266c reference target: `19.845698 / 0.620201 / 0.179915`.
  - v270d reference target: `19.844320 / 0.620226 / 0.179934`.
  - v272b target: `19.843843 / 0.620191 / 0.179945`.
  - v272c target: `19.844036 / 0.620193 / 0.179934`.
  - v272d target: `19.843998 / 0.620177 / 0.179918`.
  - v272e target: `19.844132 / 0.620207 / 0.179923`.

Current direct verdict:

> v272 is a valid engineering/method interface upgrade but a quality failure. Learned scalar confidence, even with target-free source-consistency features and boost-only variants, overfits policy-val and does not transfer to flowers target exact. Do not promote v272 to full9. The next model should change the residual carrier or supervision target instead of stacking another scalar policy head.

Latest update on 2026-06-30, v264-v266:

The newest branch is **edge-aware / low-rank hybrid deferred source residual rendering** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v264-v266-EdgeLowrankHybrid-Log.md`
- `docs/car_model/results/v264_v266_edge_lowrank_hybrid_summary.json`

Important new facts:

- v264 added `edge_local_linear`: parent-edge features are included in local face/UV ridge residual decoding. This is a real decoder change, not an alpha scan.
- v265 added `lowrank_source_basis`: source-slot low-rank teacher residual bases are fit from train-fit evidence only. Checkpoints now save `source_view_id` so source-view diversity can be audited.
- v266 added `hybrid_edge_lowrank`: edge-local-linear is used as the stable base, and low-rank detail is injected through a disagreement-aware blend. This directly responds to the v265 negative result where pure low-rank replacement hurt PSNR/SSIM.
- v266c is the best deferred-source target PSNR so far: `19.845698 / 0.620201 / 0.179915`, gains `+0.013644 / +0.000290 / +0.000419`, changed fraction `0.054285`, PSNR tail CVaR `-0.002039`.
- However, v266c is not all-axis best. v264a still has the best target SSIM `0.620226`; v264b still has the best target LPIPS `0.179872`.
- Most importantly, v266c still fails the Phase-J flowers PSNR gate by `0.458660` PSNR. Full9 remains blocked.

Current direct verdict:

> v266 is a meaningful mechanism improvement over v263-v264 on PSNR and PSNR-tail, but it is still **NOT COMPLETE** for paper-level success. The source-slot low-rank representation is too local and not coherent enough across UV bins. The next model should not repeat low-rank slot blending, edge-gain nudges, or alpha scans. It should move to coherent face/patch texture features across UV bins, explicit patch/gradient residual supervision, and a target-free uncertainty/visibility model.

Latest update on 2026-06-29, v260-v263:

The newest branch is **local-linear deferred source residual decoding with no-GT target-visible face expansion** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-29-v260-v263-LocalLinearTargetVisible-Log.md`
- `docs/car_model/results/v260_v263_local_linear_target_visible_summary.json`

Important new facts:

- v260 added `--ood_gain_mode learned_linear`, a policy-val-supervised OOD/gain head over target-free support features. It learned a nontrivial signal, but it is only an auxiliary guard and did not improve target exact over v259/v258.
- v261 added the real representation change: `--residual_decoder_mode local_linear`, replacing convex source residual averaging with a per-face/UV local ridge decoder from source camera/parent RGB to residual, evaluated at target camera/parent RGB.
- v262 rebuilt a 32k-face source bank; v263 added `--target_visible_face_quota`, using only stripped target geometry/alpha/face visibility to expand the carrier. No target/test RGB GT or target residual keys are read during apply.
- v263a is the best result in this line: `19.844512 / 0.620224 / 0.179968`, gains `+0.012458 / +0.000314 / +0.000367`, changed fraction `0.040890`, target active fraction `0.199257`.
- v263a still fails the Phase-J flowers gate because PSNR is `0.459846` below Phase-J flowers `20.304358`. Full9 remains blocked.
- v263b extended alpha to `3.0`; policy-val selected `alpha=1.5`, but target exact degraded to `19.839942 / 0.619739 / 0.179855`, with SSIM gain `-0.000172` and PSNR tail CVaR `-0.013618`. Therefore simple alpha amplification is not the solution.

Current direct verdict:

> v263a is meaningful representation-level progress and the best v260-v263 flowers target exact result, but it is still **NOT COMPLETE**. The bottleneck is now target-visible useful changed fraction and cross-view/OOD generalization, not just source residual energy or scalar confidence. The next model should not repeat learned OOD-head-only, alpha amplification, or fixed beta scans; it should build a stronger patch/edge-aware or learned source-feature surface decoder that can safely affect more target-visible pixels.

Latest update on 2026-06-29, v259:

The newest branch is **target-support / OOD-aware gain** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.  New files:

- `docs/car_model/6-29-v259-TargetSupportOODGain-Log.md`
- `docs/car_model/results/v259_ood_gain_summary.json`

Important new facts:

- v259 adds `policy_tail_risk`, learned from policy-val positive fraction, negative gain magnitude, and gain variance per face/UV bin.
- v259 adds `--ood_gain_mode boosted_soft`, which automatically shrinks only boosted residuals using target-free OOD/source-support features: source camera view gap, residual variance ratio, parent RGB mismatch, effective source count concentration, and policy-val tail risk.
- v259a OOD beta 1 gives the best target SSIM mean in the deferred-source line: `19.838006 / 0.620050 / 0.180238`, gains `+0.005952 / +0.000139 / +0.000097`.
- v259b OOD beta 2 makes target PSNR tail CVaR positive for the first time in this v253-v259 line: tail `+0.000040 / -0.000116 / -0.000148`, but mean drops to `19.837280 / 0.620046 / 0.180256`.
- v258a remains best mean PSNR/LPIPS in this local line, but its tails are much riskier: target tail `-0.002007 / -0.000258 / -0.000380`.
- No v259 run passes the Phase-J flowers gate because PSNR is still about `0.466` below Phase-J flowers `20.304358`.  Full9 remains blocked.

Current direct verdict:

> v259 is meaningful method progress for OOD/tail safety, but it is still **NOT COMPLETE**.  It confirms that hand-crafted OOD shrink can trade residual energy for safer tails, but a fixed beta is not enough.  The next stage should train a policy-val-supervised OOD/gain head or use a stronger residual carrier, not continue manual beta/gain scanning.

Previous update on 2026-06-29, v257-v258:

The current newest branch is **policy-calibrated deferred residual gain** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.  New files:

- `docs/car_model/6-29-v257-v258-PolicyCalibratedGain-Log.md`
- `docs/car_model/results/v257_v258_policy_calibrated_gain_summary.json`

Important new facts:

- v257 added `patch_perceptual_v1` reliability: policy-val reliability now uses local RGB L1, luma patch, and luma-gradient gains, not only scalar L1.
- v258 added `positive_soft` policy gain: the bank now learns a per-face/UV-bin `policy_gain` in addition to `policy_reliability`, so trusted bins can retain more teacher residual energy.
- This is a real train/eval pipeline change.  Checkpoints save/load `policy_gain`; target no-GT apply uses the same prediction path; target/test GT is loaded only after apply for exact evaluation.
- v258a increased active teacher residual energy retention from v257a `0.035923` to `0.467043` and improved target exact mean gains to `+0.006250 PSNR / +0.000108 SSIM / +0.000139 LPIPS`.
- v258b with lower max gain improved target SSIM slightly more: target exact `19.838286 / 0.620047 / 0.180217`, gains `+0.006232 / +0.000137 / +0.000118`.
- v258c added source-agreement confidence; it reduced target tail damage relative to v258a/b, but also reduced mean gains.
- No v257-v258 run passes the v169 Phase-J flowers gate.  Best PSNR remains about `0.466` below Phase-J flowers `20.304358`, so full9 remains blocked.
- The new bottleneck is sharper: stronger residual energy helps means but creates target-tail/OOD risk.  The next model should build a train/policy-val target-support/OOD-aware gain predictor rather than manual gain caps, alpha scans, or full9 promotion.

Current direct verdict:

> v258 is meaningful method progress and the best deferred-source flowers target mean result in this line, but it is still **NOT COMPLETE** for paper-level all-axis success because Phase-J PSNR is not beaten and target tails are not safe enough.

Previous update on 2026-06-29:

The v169 prompt has now been executed through v249-v252 representation-gate experiments.  Treat the current status as **NOT COMPLETE for paper-level all-axis win**, but with a much clearer bottleneck diagnosis.  New files:

- `docs/car_model/6-29-v249-v252-v169-RepresentationGate-Log.md`
- `docs/car_model/results/v249_v252_v169_representation_gate_summary.json`

Important new facts:

- Phase-J teacher signal is strong on flowers policy-val: about `+0.913279 PSNR / +0.065512 SSIM / +0.017600 LPIPS` teacher headroom in v251/v252 reports.
- v249a LPIPS no-harm GT-assisted U-Net has positive mean gains but fails tails: `+0.027357 PSNR / +0.000589 SSIM / +0.000250 LPIPS`, with min SSIM `-0.000152` and min LPIPS `-0.001432`; projection energy retention is only `0.020147`, cosine `0.127558`.
- v250 memory textures improve active local projection but fail GT policy-val SSIM/LPIPS: v250a `+0.007847 PSNR / -0.000152 SSIM / -0.000019 LPIPS`, v250b `+0.007915 PSNR / -0.000107 SSIM / -0.000004 LPIPS`.
- v251 low-rank/surface-feature carriers select `alpha=0` under strict tail guard, meaning the safest policy is no-op.
- v252 added a real train-fit-only `teacher_benefit_mask_mode` method and excluded `alpha=0` from policy best by default.  It reduced damage but collapsed useful residual magnitude:
  - v252a: `+0.000094 PSNR / +0.000002 SSIM / +0.000002 LPIPS`, changed fraction `0.000369`, projection energy `0.000019`, cosine `0.021462`.
  - v252b: `+0.000382 PSNR / +0.000011 SSIM / +0.000004 LPIPS`, changed fraction `0.003078`, projection energy `0.000158`, cosine `0.026398`.
- No full9 was launched because the v169 flowers gate was not passed.  Target/test apply was skipped when policy-val all-axis failed, so no target/test RGB GT leakage occurred in v252.

Lesson for the next model:

> Do not continue alpha scans, face gates, support thresholds, footprint expansion, or simple baked RGB residual carriers.  The measured blocker is residual representation capacity/alignment: current carriers retain almost none of the teacher residual once no-harm/tail constraints are enforced.  The next viable direction should be a stronger view-dependent source-feature/deferred surface renderer or another genuinely new representation class, then flowers policy-val and exact must be re-certified before any full9.

Latest update: the first v168 exact flowers attempt failed before evaluation because the old pipeline copied a full reparented evidence cache. A low-copy/direct-teacher unblock patch is now implemented and validated by smoke tests plus dry-run. A new v168 direct-teacher low-copy exact flowers run is currently running, so treat v168 as **protocol-ready, exact-metric-in-progress, not yet a metric win**.

This file is a handoff report for a stronger AI model. It records current facts, experiment data, failures, and lessons. The goal is to prevent repeating the same loops: small parameter tuning, unfair comparisons, and footprint expansion without real visual/metric gains.

## 0. Direct Status

The project is **not paper-final yet**.

The strongest local RGB endpoint is currently **Phase-J guarded adaptive edge policy**, not the latest vNext certified residual surface texture route.

The newest complete vNext idea tested so far, **v167 target-impact affine/patch residual fill**, is:

> **engineering-progress / quality-fail / NOT COMPLETE**

It completed the strict no-target-GT pipeline and produced valid manifests, W&B logs, target-evidence stripping, verifier outputs, metrics, and renders. However, it did **not** beat Phase-J on all metrics and did **not** improve meaningfully over v165/v166. In fact, the new affine candidate was rejected by policy-val and fell back to no-op.

Direct answer for future agents:

> Compared with Phase-J, the current vNext/new-prompt route is weaker. The latest improvement idea succeeded as an engineering mechanism but failed as a paper-level quality result.

Newest post-v167 progress:

> `v168` adds a runner-level Phase-J distillation profile, `--distillation_profile teacher_to_reparented_parent`, to make the next Phase-J-to-baked-representation experiment explicit and harder to misconfigure. It has passed py_compile, dry-run, negative parser guard, low-copy smoke tests, and a direct-teacher low-copy dry-run. The first exact flowers attempt failed during the fit-evidence reparent copy with `OSError: [Errno 122] Disk quota exceeded`; after the low-copy/direct-teacher patch, a new exact flowers run was launched and is still in progress. Therefore v168 has **not** produced completed exact metrics and is **not** a metric win yet.

Newest reporting/tooling progress:

> `scripts/car_model/build_spcarnet_claim_readiness_report.py` now generates a conservative claim-readiness report from current local artifacts. The current generated report is `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`; it marks Phase-J local endpoint as `PASS_LOCAL`, v106 baked representation as `PARTIAL_PASS`, v166/v167 flowers gates as `FAIL`, v168 exact metric win as `NOT_RUN`, and vNext paper-main readiness as `FAIL`. This auto report predates the failed v168 exact attempt classification; the more precise current v168 status in this handoff is `BLOCKED_PARTIAL_NO_METRICS`.

## 0.1 Claim Readiness Matrix

| possible claim | current status | evidence that supports it | missing blocker |
|---|---|---|---|
| Phase-J is the strongest local RGB endpoint | supported locally | full9 Phase-J mean `26.482766 / 0.783720 / 0.224261`, `9 / 9` scene wins vs selected clean | must state it is render-time endpoint, not baked representation |
| v106 is the strongest verified baked representation over clean MeshSplatting | partially supported | full9 v106 mean `25.831280 / 0.760830 / 0.268435`, better than selected clean | visually subtle; still weaker than Phase-J |
| vNext certified residual texture is paper-main quality method | not supported | vNext has no-GT verifier, manifests, audits, fallback, v165-v167 negative evidence | does not beat Phase-J/v106/clean; needs Phase-J-distilled exact win |
| v168 Phase-J distillation profile is a quality improvement | not supported yet | py_compile, dry-run, negative parser guard; low-copy smoke tests; direct-teacher low-copy exact currently running | no completed exact metrics yet; `/dev/shm` remains critically tight |
| Current project is paper-final | no | engineering/reporting progress is significant | no all-axis vNext win, no fixed full9 promotion, weak qualitative evidence |

## 0.2 Latest Hard Blocker: v168 Exact Run Failed Before Metrics

This is a critical handoff fact. The most recent exact validation attempt is not a negative-quality result and not a success. It is a **storage/quota-blocked partial run**.

Attempted run root:

- `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers`

Partial artifacts:

- manifest: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- report: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- first log: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/logs/00_reparent_fit_evidence.log`
- partial copied evidence: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/fit_evidence_reparented`

Observed failure:

```text
reparent_fit_evidence returncode=1
shutil.copytree(...)
OSError / shutil.Error: [Errno 122] Disk quota exceeded
```

Current storage snapshot at the time of this update:

```text
/data:     28T total, 27T used, 9.6M available, 100% used
/dev/shm: 252G total, 246G used, 6.5G available, 98% used
/tmp (/): 14T total, 7.1T used, 6.1T available, but user quota exceeded
quota:    /dev/nvme0n1p4 user space 100G*, limit 100G
```

The partial v168 exact output itself uses about:

```text
391M  /tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact
5.3M  /tmp/peilincai_pycache_v168_exact
```

Recommendation for the next model:

1. Do not report this partial v168 run as completed evidence.
2. Either free durable/user-quota space first or implement a no-copy / symlink / overlay reparent mode before rerunning exact validation.
3. If cleaning space, it is reasonable to remove the failed partial v168 `/tmp` output after documenting it, because it is not a valid completed experiment.
4. Re-run v168 exact only after ensuring the output location can hold copied/reparented evidence, target evidence, rendered outputs, reports, and W&B offline logs.

Implemented follow-up after this blocker:

- Added `--copy_mode {copy,hardlink,symlink,auto_link}` to `scripts/car_model/ecsr_reparent_surface_evidence_cache.py`.
- Added `--copy_mode` and `--rewrite_rgb_render_to_parent` to `scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py`.
- Added runner flags `--reparent_copy_mode`, `--teacher_cache_copy_mode`, `--teacher_cache_rewrite_rgb_render_to_parent`, and `--skip_reparent_fit_evidence_for_teacher_cache`.
- The direct-teacher low-copy path skips the separate `fit_evidence_reparented` cache and lets teacher-cache construction rewrite output `rgb_render`/parent residual fields against the parent render.
- Static checks, smoke tests with `--max_views 1`, negative parser guard, and direct-teacher dry-run passed.
- A new exact flowers run was started at `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers` with W&B offline logging under `/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact`. At the time of this handoff update, it had reached `02_certified_texture.log` policy-candidate evaluation and had not yet completed final metrics.

## 1. Best Known Metrics

### 1.1 Full9 Summary

All numbers below are from local selected-clean MeshSplatting full9 evaluation unless otherwise noted.

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean MeshSplatting | role |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | local fair baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | stable representation anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | strongest verified baked representation |
| Phase-J guarded adaptive edge policy | 9 | 26.482766 | 0.783720 | 0.224261 | +1.331084 / +0.034702 / -0.063360 | strongest RGB endpoint/reference |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | protocol complete but weaker |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | safer but mostly no-op |

Interpretation:

- v106 is a real baked-representation improvement over clean MeshSplatting on full9.
- Phase-J is much stronger than v106 in RGB metrics, but Phase-J is a render-time endpoint rather than the same kind of baked representation.
- vNext certified residual surface texture currently has strong engineering/audit value but is not yet a quality winner.

### 1.2 Phase-J Per-Scene Facts

Phase-J closure audit:

- strict RGB scene wins vs selected clean MeshSplatting: `9 / 9`
- per-view strict RGB wins: `244 / 246`
- mean delta vs clean: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS
- mean total triangle reduction: `7.6479%`
- sparse geometry strict wins: `6 / 9`
- geometry-safe scenes: `9 / 9`

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

Evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- `/dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`

## 2. vNext Flowers Diagnostic History

Flowers is the clearest vNext diagnostic scene.

Core discovery: **footprint expansion alone does not produce quality gain**. v165 expanded target changed pixels by about `9.68x` over v164, but metrics barely moved. v166 then added train-only multisample residual fill, but still failed to improve SSIM/LPIPS and did not beat Phase-J.

| version | status | mechanism | accepted | alpha | changed pixels | allowed bins / faces | PSNR | SSIM | LPIPS | diagnosis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v162 | complete | sparse-selective bridge semantic repair | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | real repair but footprint too small |
| v163 | complete | target-footprint residual-debt support expansion | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | only 1 eligible face, no improvement |
| v164 | complete | target-visible connected region growth | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | no eligible connected bins |
| v165 | complete | train-only target-impact residual basis | true | 0.1875 | 8324 | 1145 / 26 | 20.452848 | 0.549059 | 0.355544 | footprint expanded, metrics unchanged |
| v166 | complete | train-only target-impact multisample residual fill | true | 0.1875 | 3859 | 457 / 4; filled 105 / 130 bins | 20.452814 | 0.549059 | 0.355544 | no-GT fill works, quality still fails |
| v167 | complete | train-only target-impact affine/patch residual field | false | 0.0 | 0 | 1182 final bins; affine filled 313 / 393 bins | 20.452776 | 0.549059 | 0.355544 | stronger capacity ran but was rejected; fallback no-op |

### 2.1 v166 vs Phase-J Flowers

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 | reference to beat |
| v165 flowers exact | 20.452848 | 0.549059 | 0.355544 | PSNR higher, SSIM/LPIPS worse |
| v166 flowers exact | 20.452814 | 0.549059 | 0.355544 | PSNR higher, SSIM/LPIPS worse |
| v167 flowers exact | 20.452776 | 0.549059 | 0.355544 | fallback no-op after policy-val rejection |

v166 delta vs Phase-J flowers:

- PSNR: `+0.148457`
- SSIM: `-0.008711`
- LPIPS: `+0.026322`

This is not an all-axis win. It is a failure under the current project standard.

v167 confirms the same conclusion with a stronger representation attempt. It filled `313 / 393` target-impact affine bins using train-fit evidence only, but policy-val rejected both candidates because SSIM/L1/tail-risk were negative. Final target apply had `changed_pixels=0`.

### 2.2 v166 Artifacts

Root:

- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers`

Key outputs:

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- audit: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- metrics: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_target_apply_no_gt_verify.json`
- renders: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`
- W&B offline run: `/dev/shm/peilincai_wandb_v166_target_impact_multisample_exact/wandb/offline-run-20260628_165449-r68qgrb6`

v166 verified facts:

- manifest status: `COMPLETE`
- manifest errors: `[]`
- no-GT verifier passed: `true`
- `target_gt_visible_to_apply=false`
- `target_residual_visible_to_apply=false`
- adapter accepted: `true`
- effective policy: `accepted_atlas`
- selected alpha: `0.1875`
- target changed pixels: `3859`
- PNG-quantized changed pixels: `3807`
- changed fraction: `0.0001040139`
- target-impact final allowed bins/faces: `457 / 4`
- target-impact added bins: `456`
- target-impact added policy-row bins: `326`
- target-impact added no-policy-row bins: `130`
- multisample eligible bins: `130`
- multisample filled bins: `105`
- train-fit views used: `34`
- sample events: `3127`
- uses policy-val GT: `false`
- uses train-fit GT/residual evidence: `true`
- uses target/test GT: `false`

### 2.3 v167 Artifacts

Root:

- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers`

Key outputs:

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- audit: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- metrics: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_results.json`
- no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_target_apply_no_gt_verify.json`
- renders: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/test/ours_26000_v167_affine_flowers/renders`
- W&B offline run: `/dev/shm/peilincai_wandb_v167_affine_exact/wandb/offline-run-20260628_173303-a59lvtxg`

v167 verified facts:

- manifest status: `COMPLETE`
- manifest errors: `[]`
- no-GT verifier passed: `true`
- `target_gt_visible_to_apply=false`
- `target_residual_visible_to_apply=false`
- adapter accepted: `false`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- affine eligible bins: `393`
- affine filled bins: `313`
- train-fit views used by affine: `34`
- affine sample events: `7774`
- affine fit faces: `24`
- uses policy-val GT: `false`
- uses train-fit GT/residual evidence: `true`
- uses target/test GT: `false`
- rejection included `cvar20_view_relative_gain=-0.134897`, `min_view_relative_gain=-0.341250`, `ssim_gain=-0.000002156`, and `image_l1_gain=-0.000000127`

Interpretation: v167 moved beyond local neighbor averaging and implemented a stronger face-local ridge/patch field, but the learned correction direction was still not safe on held-out policy-val views. Future work should not simply add another small per-face regression layer; it should distill a stronger teacher or change the representation target.

## 3. Engineering State

### 3.1 vNext Pipeline Strengths

The vNext certified residual surface texture pipeline has strong engineering infrastructure:

- scene runner and manifest runner exist;
- W&B offline logging is used in medium/long runs;
- target evidence is stripped before apply;
- strict no-target-GT verifier exists;
- audit JSON records settings, selection, sparse profile, target apply stats, and fallback/no-op behavior;
- policy-val gates include SSIM/L1/effective-margin checks;
- fallback/no-op is available on rejected candidates;
- result paths, commands, and errors are recorded in manifests.

Important code paths:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_manifest.py`
- `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`
- `scripts/car_model/summarize_vnext_accounting.py`

### 3.2 Current Worktree Warning

The repository is dirty. Do not blindly revert files.

Known current local additions/edits relevant to this handoff:

- `docs/Latest.md` exists and contains the latest honest status report.
- `feedback.md` is this handoff file.
- `scripts/car_model/build_spcarnet_claim_readiness_report.py` exists and builds `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`.
- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` contains the completed v167 target-impact affine/patch residual fill implementation, including `_teacher_distilled_basis_features_from_uv_camera_normal(...)`, `apply_target_impact_affine_residual_fill(...)`, CLI parsing, validation, audit fields, and candidate-loop integration.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py` forwards the v167 affine-fill flags through the certified scene runner.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py` now also contains the v168 distillation profile `teacher_to_reparented_parent`, which forces strict no-target-GT apply and requires split-matched parent renders for Phase-J-to-baked residual experiments.
- `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md` records the v168 dry-run and negative parser guard.
- v167 was exact-tested on flowers and is a **completed negative result**, not an unfinished draft. It filled many bins without target/test GT leakage, but policy-val rejected the candidate and the final result fell back to no-op.
- v168 exact flowers was attempted after the dry-run, but it is a **partial storage-failed run**: `reparent_fit_evidence` returned code `1` due to `/tmp` user quota, before teacher cache, target stripping, adapter apply, eval, or W&B metric logging could complete.
- v168 direct-teacher low-copy was then implemented to remove the separate fit reparent copy from the exact path. This is an engineering unblock, not a quality claim.

There are many other modified/untracked files from previous work. A next model should inspect `git status --short` and avoid reverting unrelated changes.

## 4. Lessons Learned

### 4.1 Fair Comparison Lessons

Do not compare weak or short baselines against long/improved methods. Earlier confusion came from comparing mismatched training lengths or choosing checkpoints by train metrics. The fair baseline must be the best local clean MeshSplatting checkpoint/eval under the same scene, split, and evaluation protocol.

Do not select checkpoints using train metrics. That mostly rewards longer training and overfit behavior. Use held-out/test protocol, or a fixed official validation split if available.

Do not tune a separate hand-picked parameter set per scene and call it a general method. A paper-level method needs a fixed adaptive policy that reads allowed scene statistics and makes decisions automatically.

### 4.2 Method Lessons

Footprint expansion is not quality improvement. v165 expanded flowers target changed pixels from `860` to `8324`, but PSNR moved only about `+0.000051` and SSIM/LPIPS were unchanged. This is the most important negative result.

Local neighbor residual fill is too weak. v166 filled `105 / 130` eligible no-policy target-impact bins from train-fit multisample residuals, but metrics did not improve. The residual representation itself is not expressive enough.

Simple face-local affine/patch residual fields are also insufficient. v167 filled `313 / 393` eligible bins, but policy-val rejected it for negative tail risk and image metrics. The issue is not only interface capacity; the residual target/prediction direction is misaligned with robust held-out quality.

Alpha tuning is not the bottleneck anymore. v162-v166 show that changing alpha or allowing more target bins does not automatically improve SSIM/LPIPS.

Policy-val gates are necessary but not sufficient. They prevent catastrophic regressions but can select near-no-op candidates or certify changes that do not matter visually.

Target/test GT leakage must remain forbidden. Any new method can use target visibility, geometry, camera, face IDs, barycentric footprints, and rendered target evidence with GT stripped; it must not read target/test RGB GT or residual GT during apply/selection.

### 4.3 Paper-Story Lessons

Phase-J is currently the strongest empirical RGB endpoint, but it is not a baked representation. The paper story must clearly distinguish:

- clean MeshSplatting baseline;
- Phase-J render-time endpoint/reference;
- v106 baked representation;
- vNext certified representation route.

v106 is the most honest current positive representation result, but its gain is modest and may be visually subtle. It is not enough for a strong top-conference claim without a deeper representation-level advance.

vNext has a stronger research story around certified/no-GT/safe repair, but current metrics are not good enough. It needs a real representation upgrade, not more interface additions.

### 4.4 Qualitative Lessons

Full-image qualitative panels often fail to reveal small improvements. Future qualitative evidence should include:

- targeted crops at high residual/error regions;
- difference/error maps;
- before/after zoomed comparison;
- outdoor scenes where current weak spots are visible;
- geometry/triangle reduction overlays if claiming compression/geometry gains.

But visual storytelling cannot compensate for losing SSIM/LPIPS or failing the all-axis win requirement.

### 4.5 Runtime and Storage Lessons

The current vNext exact runs are expensive:

- v162 flowers adapter: about `5771.652s`
- v163 flowers adapter: about `8684.925s`
- v164 exact apply: about `23702.957s`
- v165 exact apply: about `5415.726s`
- v166 exact apply: about `3473.020s`

Storage is fragile:

- `/data` is currently full: about `9.6M` available at the latest check.
- `/dev/shm` is also near full: about `6.5G` available at the latest check.
- `/tmp` has filesystem space but the user quota on `/dev/nvme0n1p4` is exceeded: `100G* / 100G`.
- v168 exact failed before metrics because `ecsr_reparent_surface_evidence_cache.py` uses `shutil.copytree(...)`, so it tries to materialize another evidence copy under the output root.
- The low-copy/direct-teacher patch avoids the extra fit reparent cache, but target reparenting, GT-stripping, teacher cache, model output, W&B, and reports still require several GB. `/dev/shm` remains the main live risk.
- W&B offline logs and long-run manifests can fail if storage is not checked.
- Many latest artifacts are under `/dev/shm`, which is temporary storage. If the machine reboots or `/dev/shm` is cleaned, these paths may disappear. Persist the important JSON reports and qualitative renders to a durable location once `/data` has space.

Next experiments should use a staged gate:

1. syntax/static check;
2. dry-run manifest;
3. storage/quota preflight;
4. no-copy or low-copy evidence reparent if storage remains constrained;
5. flowers exact;
6. only if flowers beats Phase-J all-axis, fixed full9 promotion.

## 5. What Not To Do Next

Do not continue with only:

- more alpha-grid tuning;
- more per-scene handpicked parameters;
- more target footprint expansion without stronger residual content;
- train-metric checkpoint selection;
- clean short-run vs our long-run comparisons;
- qualitative-only claims without quantitative all-axis support;
- full9 expensive promotion before flowers passes the Phase-J gate.

These have already consumed many iterations and did not solve the core bottleneck.

## 6. Suggested Next Research Direction

The next model should focus on a stronger **train-only residual representation** that converts target-visible footprint into real RGB/SSIM/LPIPS improvement while preserving no-target-GT apply.

Promising directions:

1. **Phase-J-distilled baked representation**
   - Treat Phase-J as the strong teacher endpoint on train/policy-val views.
   - Distill its correction into a surface/texture representation.
   - Apply only the distilled representation at test time.
   - This directly targets the observed gap: Phase-J is strong, vNext baked representation is weak.
   - Use the v168 runner profile `--distillation_profile teacher_to_reparented_parent` so teacher renders, fit parent renders, target parent renders, and no-target-GT apply are recorded and checked in one manifest.

2. **Face-local residual field upgrade, but only with a stronger target**
   - Learn a per-face or per-region residual field over `(u, v, normal, train-view camera, support confidence)`.
   - Use train-fit residual evidence only, or preferably Phase-J teacher corrections on train/policy-val views.
   - Predict target-visible UV bins without target/test RGB GT.
   - Use policy-val to gate the learned field before test apply.

3. **Adaptive fixed policy, not scanned parameters**
   - Build a scene-adaptive policy from scene statistics: residual distribution, coverage, face support, view consistency, bin uncertainty, camera spread, and geometry risk.
   - Freeze the policy before full9 promotion.
   - Do not choose custom parameters per scene after seeing test results.

4. **Multi-objective gate**
   - Require RGB improvement and geometric/compression safety together.
   - Metrics should include PSNR, SSIM, LPIPS, changed fraction, triangle reduction, geometry safety, fallback rate, and per-view strict wins.

5. **Target-aware but GT-free**
   - It is acceptable to use target/test camera, face visibility, barycentric footprint, alpha/coverage, and GT-stripped target evidence.
   - It is not acceptable to use target/test RGB GT, target residual GT, or any metric computed against target GT during selection/apply.

## 7. Minimum Success Criteria for the Next Prompt

A future prompt/model should not stop until it has evidence for all of the following:

1. A real method change is implemented in the train/eval pipeline.
2. The method uses a fixed or genuinely adaptive policy, not per-scene hand tuning.
3. Flowers exact run beats Phase-J all-axis:
   - PSNR higher than `20.304358`;
   - SSIM higher than `0.557770`;
   - LPIPS lower than `0.329222`.
4. If flowers passes, full9 is run under the same fixed policy.
5. Full9 reports clean MeshSplatting, Phase-J, v106, vNext previous best, improved method, and ablations.
6. Metrics and qualitative outputs are saved.
7. Commands, configs, result paths, and errors are documented.
8. The final report honestly marks weaknesses and failed experiments.

## 8. Suggested Prompt for the Next Stronger Model

```text
You are working in /data/peilincai/mesh-splatting.

Read feedback.md and docs/Latest.md first. Treat current evidence as authoritative.

The current bottleneck is not footprint expansion; v165/v166 proved that larger target-impact footprints and local multisample residual fills do not improve SSIM/LPIPS. Build a stronger train-only residual representation that can be baked into the surface/texture pipeline without target/test RGB GT leakage.

Requirements:
- Preserve strict no-target-GT apply.
- Do not tune per-scene parameters manually.
- Do not use train metrics for checkpoint/model selection.
- First validate on flowers exact against Phase-J flowers: 20.304358 PSNR, 0.557770 SSIM, 0.329222 LPIPS.
- Only promote to full9 if flowers is an all-axis win.
- Compare against clean MeshSplatting, Phase-J, v106, vNext previous best, and ablations.
- Save metrics, qualitative renders/crops, commands, configs, errors, and W&B offline logs.

Suggested route:
1. Implement a train-only face-local residual field or Phase-J-distilled baked representation.
2. Gate it with policy-val/nonregression checks.
3. Run dry-run, then flowers exact.
4. If failed, diagnose whether the failure is representation capacity, target coverage, gate selection, or render/apply mismatch.
5. Iterate until the evidence supports a paper-level claim or clearly proves the direction is blocked.
```

## 9. Important Index

Latest status:

- `docs/Latest.md`
- `feedback.md`
- `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`
- `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`

Core code:

- `scripts/car_model/build_spcarnet_claim_readiness_report.py`
- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_manifest.py`
- `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`
- `scripts/car_model/summarize_vnext_accounting.py`

Phase-J evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- `/dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`

vNext full9 evidence:

- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`

v166 evidence:

- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_target_apply_no_gt_verify.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`

v167 evidence:

- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_results.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_target_apply_no_gt_verify.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/test/ours_26000_v167_affine_flowers/renders`
- `/dev/shm/peilincai_wandb_v167_affine_exact/wandb/offline-run-20260628_173303-a59lvtxg`

v168 protocol evidence:

- dry-run root: `/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers`
- dry-run manifest: `/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- durable log: `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`
- failed exact attempt root: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers`
- failed exact attempt first log: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/logs/00_reparent_fit_evidence.log`
- failed exact attempt manifest/report: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`, `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- direct-teacher low-copy exact in-progress root: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers`
- direct-teacher low-copy W&B offline root: `/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact`
- direct-teacher low-copy command markers: `--reparent_copy_mode auto_link --teacher_cache_copy_mode auto_link --teacher_cache_rewrite_rgb_render_to_parent --skip_reparent_fit_evidence_for_teacher_cache --reparent_allow_resize`

Exact replay/inspection commands are recorded in the `commands` arrays inside the v166, v167, and v168 manifest JSON files. The v168 exact manifest is a partial failed manifest, so use it only for command/error reconstruction. The current environment does not have `jq`; use Python JSON parsing or another JSON viewer if needed.

## 10. Concrete v167 Implementation and Failure Notes

These notes came from the completed v167 implementation and flowers exact run. They should help a future model avoid wasting time repeating the same patch.

Current implementation points:

- Bin-selection contract: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`, function area around `target_impact_residual_basis`. Consume `added_bins_by_face`, `added_policy_bins_by_face`, and `added_no_policy_bins_by_face`.
- Existing fill functions: `apply_target_impact_carrier_fill(...)`, `apply_target_impact_multisample_residual_fill(...)`, and the implemented `apply_target_impact_affine_residual_fill(...)`.
- Candidate-loop insertion: affine fill is called after carrier/multisample atlas mutation and before the post-materialization `evaluate_policy_val(...)`. This preserves policy-val recertification before target apply.
- Do not patch target rendering directly. `apply_to_target(...)` should only render the already-certified atlas.
- Runner forwarding: flags are added in `scripts/car_model/run_vnext_certified_residual_texture_scene.py` beside existing target-impact fill flags, inside the `enable_train_only_target_impact_residual_basis` block.

Actual v167 CLI shape:

- `--target_impact_affine_fill_mode {off,no_policy_rows,all_added}`
- `--target_impact_affine_fill_feature_mode {face_uv_normal_camera_ridge,face_uv_patch_mixture_ridge}`
- `--target_impact_affine_fill_min_samples`
- `--target_impact_affine_fill_max_samples_per_face`
- `--target_impact_affine_fill_max_views`
- `--target_impact_affine_fill_blend`
- `--target_impact_affine_fill_ridge`
- `--target_impact_affine_fill_max_condition`
- `--target_impact_affine_fill_min_norm`
- `--target_impact_affine_fill_synthetic_count`

Validation requirements:

- nonnegative max counts/views/min norm/synthetic count;
- positive ridge and max condition;
- blend in `[0, 1]`;
- enough min samples for affine/ridge fit;
- non-`off` mode requires `--enable_train_only_target_impact_residual_basis`;
- preferably also require sparse materialization to be enabled.

Audit requirements:

- Write summary into `sparse_materialization_profile["target_impact_affine_fill"]`.
- Also write summary into `cand_fit_summary["target_impact_affine_fill"]`.
- Include `uses_train_fit_gt=true`, `uses_policy_val_gt=false`, `uses_target_or_test_gt=false`.
- Include eligible/filled/skipped counts, train views used, sample events, fit condition/ridge stats, filled bins by face, top filled bins, old/new residual norms.

No-target-GT invariants:

- Target footprint can read only target/test geometry/visibility style keys: `face_id`, `barycentric`, optional `barycentric_valid`, and `alpha`.
- New residual fitting must read residual values only from train-fit evidence / `cand_fit_views`, not target/test evidence.
- Keep target evidence stripping and verification before apply.
- Eval GT should only be populated after texture apply.
- Preserve clipping through `clip_delta_rgb(...)`.
- Update `FaceAtlas.texture`, `counts`, `variance`, and `sign_consistency` consistently.
- Keep post-fill policy-val gate between atlas mutation and target apply.

Likely v167 failure modes:

- Per-face affine/ridge fit can be underdetermined or ill-conditioned from collinear/same-bin samples.
- Synthetic filled bins may be masked out if `counts` stays below atlas/bin thresholds.
- Uncapped target-impact bins can make the run too slow.
- Carrier, multisample, and affine fills can overwrite the same bins; define order or make modes mutually exclusive.
- Direct adapter invocation can bypass runner-level target evidence stripping; certified experiments should go through the runner/manifest path.
- Most importantly, v167 already proved that simple face-local affine/patch residual fields can fill target-impact bins but still point in the wrong perceptual direction. The next model should not merely retune these v167 flags; it should change the residual target or representation, preferably by distilling a stronger teacher such as Phase-J into a baked representation.

## 11. Final Honest Evaluation

Current progress is significant but not enough:

- Engineering closure: high, roughly `80%+`.
- Evidence/reporting closure: moderate-high, roughly `75%`.
- Paper-quality method closure: not enough, roughly `45-55%`.
- Main quality blocker: no current vNext representation beats Phase-J all-axis; v106 beats clean but is not visually or conceptually strong enough yet.
- Main execution blocker: the latest v168 exact run cannot complete under current storage/quota because evidence reparenting performs a full copy.

The next breakthrough likely requires distilling or replacing Phase-J-like render-time correction with a truly baked, train-only, policy-val-certified representation, rather than continuing to adjust sparse texture footprints.

## 12. Recommended Execution Priority for the Next Model

The next model should not start by launching another full exact run into the same quota problem. Recommended order:

1. Read this file, `docs/Latest.md`, `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`, and `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`.
2. Inspect `scripts/car_model/ecsr_reparent_surface_evidence_cache.py` and decide whether to add a no-copy / symlink / overlay reparent mode. This is the fastest route to unblock v168 under constrained storage.
3. If storage is freed externally, rerun v168 exact without changing method code. If not, implement the low-copy reparent path and validate it with a tiny dry-run.
4. Run v168 exact flowers with W&B offline logging and strict no-target-GT verifier.
5. Compare only against the Phase-J flowers gate: `20.304358 / 0.557770 / 0.329222`. Passing means all three: higher PSNR, higher SSIM, lower LPIPS.
6. If v168 fails quality after exact completion, diagnose whether the teacher residual is being diluted, clipped, masked by sparse materialization thresholds, or rejected by policy-val. Do not retune by test metrics.
7. If flowers passes, freeze the policy and promote to fixed full9. Include clean MeshSplatting, v106, Phase-J, previous vNext, improved vNext, and ablations.
8. Save metrics, per-view results, qualitative crops, error maps, commands, logs, W&B offline paths, manifests, and failure reasons.

Minimum first engineering patch likely needed:

```text
Add a low-copy mode to evidence reparenting so v168 exact does not materialize a full duplicate evidence cache under /tmp or /dev/shm before any metric can be computed.
```

Minimum first experiment after that patch:

```text
Run v168 Phase-J-distilled flowers exact, not full9. Full9 is justified only after the flowers all-axis gate beats Phase-J.
```
# 2026-06-29 v195-v199 Surface-Texture / Low-Rank Feedback Addendum

This addendum records the latest attempt based on
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Main files:

- Implementation:
  `scripts/car_model/train_surface_conditioned_residual_unet.py`
- Standalone apply:
  `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py`
- Detailed log:
  `docs/car_model/6-29-v195-v199-SurfaceTexture-LowRank-Diagnostics.md`
- Machine-readable summary:
  `docs/car_model/results/v195_v199_surface_texture_lowrank_summary.json`

New method pieces implemented:

- `surface_texture_mlp`: trainable per-face/per-UV-bin surface feature texture
  with a compact decoder.
- `lowrank_surface_texture`: support-aware rank-K residual basis with a hard
  inactive-support no-op gate.
- `--surface_target_visible_evidence_dir`: no-GT target-visible face priority,
  used only for capacity allocation.
- `--artifact_prefix`: future checkpoint/report artifacts can avoid stale
  `v184_*` filenames.

Hard gate:

| Reference | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 |

Official flowers exact results:

| Run | Method | Train GT | PSNR | SSIM | LPIPS | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v195 | surface texture MLP | no | 19.878033 | 0.509020 | 0.402998 | fail all axes |
| v196 | surface texture MLP | yes | 20.084991 | 0.523929 | 0.385202 | fail all axes; diagnostic only |
| v197 | support-aware low-rank | no | 19.834993 | 0.505835 | 0.405083 | fail all axes |
| v198 | support-aware low-rank | yes | 19.833418 | 0.505749 | 0.404551 | fail all axes; diagnostic only |
| v199 | low-rank + target-visible capacity | no | 19.835337 | 0.505801 | 0.404194 | fail all axes |

Key lessons for the next model:

1. The surface texture MLP can pass policy-val, but official target transfer
   collapses. This is likely policy-val/source-view overfitting.
2. The support-aware low-rank gate prevents unsafe writes: inactive-support
   changed fraction is exactly `0.0` in v197-v199.
3. The same gate is too conservative unless target-visible capacity is added.
   v199 raises known target face fraction to `0.167715` and active support to
   `0.105916`, but official metrics still stay near v197/v198.
4. Therefore the current blocker is not just face capacity or memory size. The
   blocker is cross-view residual generalization under the current surface
   representation.
5. The next serious route should keep the no-GT target-visible allocator and
   inactive-support no-op guarantee, but replace static per-row residual storage
   with a stronger view-conditioned residual field validated on held-out source
   views before any target apply.
6. Before building another carrier, run a teacher-residual projection audit:
   compare raw `Phase-J - parent` residual, projected carrier residual, and final
   applied residual per view/per region. If projection loses energy or structure,
   the carrier is the bottleneck. If projection is healthy but target apply
   fails, debug masking/confidence/clipping/target-transfer dilution.
7. Important fairness nuance: `--surface_target_visible_evidence_dir` uses
   target-view geometry/visibility for capacity allocation. It does not use
   target RGB GT or residuals, but it is transductive and must be disclosed.
8. Policy-val all-axis pass only means improvement over the parent render on
   held-out fit/policy-val views. It is not the Phase-J gate and did not predict
   official target success in v195-v199.
9. The script defaults still include nonzero train-fit GT loss. Teacher-only
   claims require explicitly setting all GT weights to zero.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v278 Structure/Perceptual Target Feedback Addendum

Detailed log and machine-readable summary:

```text
docs/car_model/6-30-v278-StructurePerceptualTarget-Negative-Log.md
docs/car_model/results/v278_structure_perceptual_target_summary.json
```

Main implementation:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

What changed:

- Added train-time residual target transforms:
  `raw`, `gain_soft`, `structure_safe`, and `structure_gain`.
- v278a trained against a `structure_gain` target using train-fit
  `teacher_gain_l1`, parent/residual luma-gradient support, and chroma shrink.
- This changed the actual supervised residual target used by pixel and image
  proxy losses; it was not an apply-only threshold.

Result:

| run | policy PSNR/SSIM/LPIPS gain | target PSNR/SSIM/LPIPS gain | lesson |
| --- | --- | --- | --- |
| v278a | +0.016578 / +0.000043 / +0.000299 | +0.008341 / -0.001218 / -0.000904 | stronger policy-val, worse target exact |

Lesson:

The simple scalar structure/gain target transform is not the missing ingredient.
It makes policy-val look more convincing but worsens target SSIM and LPIPS. The
next route should learn target-safety from multi-view agreement or a calibration
split with held-out-view structure/perceptual gains, then certify on a separate
policy-val split before target exact.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v275-v277 Learned Surface Decoder Feedback Addendum

This addendum records the latest work based on:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

Detailed log and machine-readable summary:

```text
docs/car_model/6-30-v275-v277-LearnedSurfaceDecoder-v169-Gate-Log.md
docs/car_model/results/v275_v277_learned_surface_decoder_summary.json
```

Main implementation:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

What changed:

- Added a learned surface-attached residual decoder path that trains on
  Phase-J teacher-parent residuals and evaluates with target no-GT evidence.
- Added strict target exact evaluation and Phase-J flowers gate reporting.
- Added a parent-luma-gradient structure gate that uses only target-blind parent
  render and predicted residual structure.
- Added gain-soft confidence labels from train-fit `teacher_gain_l1`, replacing
  the ineffective all-one confidence target observed in v275b.
- Added deploy-time confidence thresholding selected only on policy-val.

Important results:

| run | target PSNR gain | target SSIM gain | target LPIPS gain | changed fraction | lesson |
| --- | ---: | ---: | ---: | ---: | --- |
| v275b | +0.009091 | -0.000808 | -0.000724 | 0.139362 | learned decoder improves PSNR only |
| v276a | +0.009069 | -0.001037 | -0.000304 | 0.139342 | structure gate reduces LPIPS damage but worsens SSIM |
| v277a | +0.010690 | -0.001008 | -0.000456 | 0.139102 | gain-soft confidence improves PSNR but not structure |
| v277c | +0.009657 | -0.000896 | -0.000488 | 0.132016 | confidence threshold modestly reduces changed area |
| v277d | +0.000945 | -0.000138 | -0.000284 | 0.004060 | conservative threshold almost no-ops and still fails |

Key lesson:

Policy-val all-axis success is not enough. The learned decoder can pass
policy-val, but target exact still has negative SSIM and LPIPS. Confidence
thresholding can make the failure smaller, but it does not turn the residual
direction into a reliable positive correction. This is evidence that the current
surface carrier and raw RGB teacher residual target remain underpowered for
Phase-J distillation.

Next recommended prompt for a stronger model:

```text
Continue from v275-v277. Do not tune alpha or thresholds first. Replace the raw
RGB teacher-parent residual target with a structure/perceptual teacher target:
for example, train a view-dependent surface representation that predicts a
low-rank residual basis plus a learned reliability score supervised by
held-out-view SSIM/LPIPS gains. The model must use train-fit evidence only,
certify on policy-val, apply to stripped target no-GT evidence, and require
flowers exact PSNR/SSIM/LPIPS all-axis vs parent and Phase-J before full9.
Measure whether the new target improves the target SSIM/LPIPS sign, not only
whether it preserves PSNR.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v271 Source-View Consistency Feedback Addendum

New implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and summary:

```text
docs/car_model/6-30-v271-SourceViewConsistency-Gate-Log.md
docs/car_model/results/v271_source_view_consistency_summary.json
```

What changed:

- Added source-view leave-one-out residual consistency calibration.
- Each source residual slot is predicted from other source-view slots in the
  same face/UV bin.
- LOO cosine and relative error are converted into source-slot reliability.
- The reliability map is frozen before policy-val and target no-GT apply.

Key result:

| run | exact PSNR | exact SSIM | exact LPIPS | lesson |
|---|---:|---:|---:|---|
| v266c | 19.845698 | 0.620201 | 0.179915 | previous best |
| v271c | 19.845337 | 0.620191 | 0.179887 | LPIPS improves, PSNR/SSIM drop |
| v271d | 19.845648 | 0.620200 | 0.179919 | almost recovers PSNR/SSIM, loses LPIPS |

Lesson:

LOO source consistency is a meaningful uncertainty signal, but directly
multiplying source weights by it is too blunt. It removes some teacher residuals
that are inconsistent across source views but still useful on the target
trajectory. The result is an LPIPS/PSNR tradeoff, not an all-axis win.

Next recommended direction:

```text
Continue from v271. Keep source_consistency_reliability, LOO cosine/error,
policy reliability, tail risk, parent mismatch, view gap, source count, and
residual variance as features. Train a compact policy-val confidence/amplitude
head that predicts whether to apply, shrink, preserve, or slightly boost each
surface residual. Do not use source consistency as a hard source-weight
multiplier. The head must be frozen on policy-val and evaluated on stripped
target no-GT evidence. Flowers exact must beat Phase-J PSNR 20.304358, SSIM
0.557770, and LPIPS 0.329222 before any full9.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v269-v270 Face-Texture Low-Rank Feedback Addendum

This addendum records the direct follow-up to:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

New implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and JSON summary:

```text
docs/car_model/6-30-v269-v270-FaceTextureLowrank-v169-Gate-Log.md
docs/car_model/results/v269_v270_face_texture_lowrank_summary.json
```

What was implemented:

- `patch_coherent_hybrid`: same-face neighboring UV/bin residual carrier.
- `face_texture_lowrank`: coherent same-face UV low-rank Phase-J teacher
  residual texture. It predicts target coefficients from view, parent RGB,
  edge, and relative UV offset features.
- `hybrid_edge_texture_lowrank`: stable edge-local-linear base plus the new
  coherent face-texture low-rank carrier.

Key result:

| run | mode | alpha | flowers exact PSNR | SSIM | LPIPS | Phase-J PSNR gap | result |
|---|---|---:|---:|---:|---:|---:|---|
| v266c | hybrid_edge_lowrank | 1.000 | 19.845698 | 0.620201 | 0.179915 | -0.458660 | previous best |
| v269c | face_texture_lowrank | 0.125 | 19.834773 | 0.620011 | 0.180294 | -0.469585 | too diluted |
| v270d | hybrid_edge_texture_lowrank | 1.000 | 19.844320 | 0.620226 | 0.179934 | -0.460038 | not better overall |

Important lesson:

The v169 prompt was directionally correct: a surface-attached teacher residual
texture carrier is more principled than another alpha/footprint tweak. However,
this implementation proves that same-face UV low-rank capacity alone is not
enough. It gives strong policy-val all-axis gains, but target exact PSNR remains
below both Phase-J and the previous v266c best.

Concrete bottleneck:

- v270d policy-val is strong: `+0.066941 PSNR / +0.002718 SSIM /
  +0.001205 LPIPS`.
- v270d target exact is only `+0.012266 PSNR / +0.000315 SSIM /
  +0.000401 LPIPS` vs parent.
- v270d target exact is slightly better than v266c in SSIM, but worse in PSNR
  and LPIPS.
- v270d remains `-0.460038` PSNR below the Phase-J flowers reference.

No-GT protocol:

The completed exact runs used stripped target no-GT evidence and loaded target
GT only after apply for evaluation. The no-GT verifier passed.

Next recommendation for a stronger model:

```text
Continue from v270d, but do not tune alpha or UV radius first. The same-face
low-rank texture carrier is not enough. Replace the residual-only projection
with a teacher-student objective that jointly predicts residual amplitude and
confidence under held-out source-view validation. The method should explicitly
penalize teacher residual directions that pass train-policy-val but fail target
trajectory transfer. Candidate directions: view-held-out residual sign
consistency, residual covariance/uncertainty calibration, compact learned
surface decoder with confidence head, or pseudo-target/source-view leave-one-out
distillation. Keep the v169 gate: flowers exact must exceed Phase-J PSNR
20.304358, SSIM 0.557770, and LPIPS 0.329222 before full9.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-29 v255 Source-Agreement Confidence Addendum

v255 tested whether the v253 LPIPS failure can be fixed by a simple target-blind
source agreement confidence:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
--source_agreement_mode soft
--source_agreement_beta 0.25
```

Summary artifact:

```text
docs/car_model/results/v255_source_agreement_confidence_summary.json
docs/car_model/6-29-v255-SourceAgreementConfidence-Log.md
```

Result:

| stage | alpha | PSNR gain | SSIM gain | LPIPS gain | mean confidence | all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| policy-val | 0.046875 | +0.001655 | +0.000018 | +0.000001 | 0.655315 | pass |
| target exact | 0.046875 | +0.001395 | +0.000036 | -0.000008 | 0.651719 | fail |

Lesson:

The source residual variance gate attenuates residuals, but it does not solve
perceptual transfer. It even makes target LPIPS more negative than v253b/v253d
while preserving PSNR/SSIM gains. Therefore the next model should not repeat a
hand-designed agreement scalar. It should learn or calibrate perceptual
reliability from held-out policy-val evidence, using richer features:
multi-source agreement, source view diversity, residual variance, edge support,
teacher-gain stability, normal/view consistency, and parent-color consistency.

# 2026-06-29 v256 Policy-Val L1 Reliability Addendum

v256 implements a learned/calibrated target-blind reliability map:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
--policy_reliability_mode local_l1
```

The policy uses policy-val GT only to learn whether each face/UV bin locally
reduces L1 error. The learned reliability map is frozen before target apply.
Target evidence remains stripped no-GT; target GT is loaded only after apply for
evaluation.

Artifacts:

```text
docs/car_model/6-29-v256-PolicyL1Reliability-Log.md
docs/car_model/results/v256_policy_l1_reliability_summary.json
```

Results:

| run | min positive fraction | alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v256a | 0.52 | 0.125 | +0.002737 | +0.000087 | +0.000035 | +0.000830 | +0.000026 | +0.000013 | pass |
| v256b | 0.50 | 0.250 | +0.005508 | +0.000175 | +0.000070 | +0.001659 | +0.000050 | +0.000026 | pass |
| v256c | 0.48 | 0.500 | +0.010844 | +0.000343 | +0.000144 | +0.003185 | +0.000091 | +0.000050 | pass |

Current best:

```text
v256c target exact = 19.835239 / 0.620001 / 0.180285
v256c target gains vs parent = +0.003185 / +0.000091 / +0.000050
```

This is the first v253-family result that fixes the target mean LPIPS failure.
It should replace v253/v255 as the current best method state.

Remaining limitations:

- It still does not pass the Phase-J flowers PSNR gate (`20.304358`).
- Target SSIM and LPIPS tails are still slightly negative.
- The visual changed fraction remains small (`0.007788` in v256c), so
  qualitative improvements may be subtle.
- Full9 is still blocked by the v169 rule.

Next recommended prompt:

```text
Continue from v256c. Preserve the target-blind policy-val reliability principle,
but replace local L1 reliability with a richer patch/perceptual reliability
model. Use policy-val only to learn reliability from patch L1, luma-gradient
error, SSIM proxy, LPIPS-sensitive edge statistics, source view diversity,
teacher-gain stability, and residual variance. Require policy-val and target
exact mean metrics and tails to improve before any full9. Do not use target/test
GT for policy selection.
```

# 2026-06-29 Residual Projection Audit Addendum

New tool:

```text
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

Compact artifacts:

```text
docs/car_model/6-29-v191-v199-ResidualProjectionAudit-Summary.md
docs/car_model/results/v191_v199_residual_projection_summary.json
```

Key result:

| Run | Policy retention | Policy cosine | Target retention | Target cosine |
| --- | ---: | ---: | ---: | ---: |
| v191 image-space U-Net calibration | 9.916031 | 0.279888 | 0.253365 | 0.393485 |
| v195 surface texture MLP | 0.068206 | 0.112638 | 0.002863 | 0.133734 |
| v196 GT-assisted surface MLP diagnostic | 1.427611 | 0.138419 | 0.029127 | 0.199612 |
| v199 support-aware low-rank | 0.015229 | 0.039391 | 0.000847 | 0.028702 |

Main lesson for the next model: the surface carrier is not simply losing at
official target evaluation; it fails to project/aligned the teacher residual
already on held-out policy-val evidence. A future method should require a
source-view projection gate before full target runs:

```text
policy residual cosine >= 0.25
target-free policy residual energy retention in [0.25, 4.0]
policy PSNR/SSIM vs teacher does not degrade materially
```

# 2026-06-29 v253-v254 Deferred Source Renderer Feedback Addendum

This addendum records the latest attempt based on
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Main new implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and machine-readable summary:

```text
docs/car_model/6-29-v253-v254-DeferredSourceRenderer-Log.md
docs/car_model/results/v253_v254_deferred_source_renderer_summary.json
```

What changed:

- v253 is a real representation change, not another alpha scan.
- It builds a train-fit Phase-J teacher residual source bank over face/UV bins.
- Each target pixel gathers source residuals by view direction, normal
  agreement, parent-RGB similarity, support count, and teacher gain.
- Target apply uses stripped no-GT evidence; target GT is loaded only after
  apply for evaluation.
- `--bank_checkpoint` allows fixed-bank policy/eval ablations without rebuilding
  the representation.
- v254 tested residual channel shaping (`luma_only`, `chroma_shrink`) as a
  perceptual-transfer diagnostic.

Key result:

| run | selected alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v253b raw RGB | 0.031250 | +0.001240 | +0.000015 | +0.000004 | +0.001063 | +0.000028 | -0.000002 | fail |
| v253c fine alpha | 0.046875 | +0.001837 | +0.000020 | +0.000001 | +0.001579 | +0.000040 | -0.000007 | fail |
| v253d conservative alpha | 0.015625 | +0.000628 | +0.000008 | +0.000006 | +0.000537 | +0.000014 | -0.000001 | fail |
| v254a luma only | 0.031250 | +0.001141 | +0.000012 | +0.000002 | +0.000985 | +0.000025 | -0.000005 | fail |
| v254b chroma shrink | 0.031250 | +0.001166 | +0.000013 | +0.000003 | +0.001005 | +0.000025 | -0.000004 | fail |

Important lesson:

v253 is the strongest representation-level step after v249-v252 because it
produces consistent PSNR/SSIM target gains and a policy-val all-axis pass.
However, it is still not enough. The fixed-policy target exact LPIPS gain is
slightly negative in every variant. Conservative alpha reduces the damage but
nearly collapses the visual change. Luma/chroma shaping does not fix it.

Current bottleneck:

The source bank can transfer a small MSE/SSIM-improving correction, but it cannot
yet certify that the residual direction is perceptually correct out of source
trajectory. Active projection cosine is around `0.279`, but selected-alpha
energy retention is only about `0.00119`, so the method is still too weak to make
visible or paper-level improvements.

Next recommended prompt for a stronger model:

```text
Continue from v253/v254. Do not tune alpha first. Add a target-blind perceptual
confidence/reliability predictor for source-bank residuals. It should use only
train-fit and policy-val evidence to estimate whether a face/bin/source residual
is safe: multi-source agreement, residual variance, source view diversity,
normal/view consistency, edge support, teacher-gain stability, and parent-color
consistency. Freeze the policy on policy-val, apply to stripped target no-GT
evidence, and require target exact PSNR/SSIM/LPIPS all-axis vs parent before any
full9. Compare against v253b/v253d and report no-GT audit, W&B offline path,
commands, and qualitative target render triplets.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```
