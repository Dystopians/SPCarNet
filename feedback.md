# SPCarNet / MeshSplatting Feedback for Next-Stage AI Model

Date: 2026-06-28

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
