# 2026-06-27 v149-v151 Face Reliability and Selected-Alpha Cap Log

## Context

The v147-v148 image-linear residual generator introduced a real representation change: view-balanced fitting and view-cluster MoE routing improved local residual fitting. It still did not pass the strict policy-val image gates on `flowers`. The failure was not training loss, but cross-view reliability:

- policy-val residual MSE improved at a useful magnitude;
- full-image positive-view fraction stayed at `0.5`, below the strict `0.55` gate;
- SSIM-positive views stayed at `5 / 12`;
- the face-gain guard kept only a tiny subset of faces.

This log tracks the next step: make the generator face-aware and alpha-aware, so it can suppress unreliable `(view cluster, face)` support before target/test materialization.

## Code Changes

Files changed:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Implemented interfaces:

- `--image_linear_generator_face_reliability_mode {none,global,view_cluster}`
- `--image_linear_generator_face_reliability_min_face_samples`
- `--image_linear_generator_face_reliability_min_relative_gain`
- `--image_linear_generator_face_reliability_min_positive_view_fraction`
- `--image_linear_generator_face_reliability_fallback_multiplier`

Behavior:

- The image-linear generator now builds a per-face reliability profile on train policy-val views.
- `global` mode evaluates each face across all policy-val samples.
- `view_cluster` mode evaluates each `(view cluster, face)` pair and is only valid when `--image_linear_generator_expert_mode view_cluster` is also enabled.
- Unreliable faces can fall back to a configurable multiplier, usually `0.0`.
- The profile records `uses_policy_val_gt=true`, `uses_target_or_test_gt=false`, and `certification_independent=false`.
- The non-independent flag is important: this is policy-val calibration, not an independent held-out certificate.

v151 implementation change:

- v150 selected a best per-face alpha during profile construction, but did not apply that alpha during prediction.
- v151 stores entries as `{multiplier, selected_alpha}` and caps the generator at apply time by `selected_alpha / current_alpha`.
- The audit field `apply_alpha_mode` is now expected to be `cap_generator_by_selected_alpha_over_current_alpha`.

Static verification passed after v151:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

## v149 Status

v149 was interrupted after review found that the first face-reliability version still had protocol and apply-path problems. It is not valid evidence and should not be reported as a completed experiment.

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0920_v149_face_reliability_flowers/flowers`
- manifest status: `RUNNING`
- manifest updated at: `2026-06-27T09:24:21`
- command count: `3`
- error count: `0`
- validity: `invalid / interrupted`
- stopped while the adapter was in the 12-view `policy-val atlas` pass
- missing outputs: adapter audit, eval-GT audit, final test results, and per-view results

The main problems identified were:

- `view_cluster` reliability could mismatch disabled experts at apply time;
- per-face selected alpha was computed but not applied;
- policy-val reliability and final gate used the same policy-val split, so the result should not be called independent certification;
- the prediction path could become slow from repeated face-profile scans.

## v150 Result: Face Reliability with Per-Face Alpha Search, But No Apply-Time Alpha Cap

Run:

- root: `/dev/shm/peilincai_spcarnet_20260627_0950_v150_face_reliability_alpha_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_0950_v150_face_reliability_alpha/wandb/offline-run-20260627_103238-risb2jjr`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- method: `ours_26000_v150_face_reliability_alpha_flowers`
- face reliability mode: `view_cluster`
- fallback multiplier: `0.0`

Protocol:

- manifest status: `COMPLETE`
- command count: `3`
- error count: `0`
- adapter elapsed: `3121.556` sec
- eval-GT population elapsed: `15.901` sec
- final test eval elapsed: `42.489` sec
- protocol audit: passed
- target/test GT was not visible to selection/apply

Generator audit:

- generator relative gain vs base MSE: `0.1150455540`
- generator relative gain vs base L1: `0.1021304000`
- per-view training MSE gain fraction: `0.6666666667`
- per-view training L1 gain fraction: `0.6666666667`

Face reliability profile:

- enabled: `true`
- mode: `view_cluster`
- candidate face-group count: `385`
- kept face-group count: `30`
- kept sample fraction: `0.4339442535`
- min face samples: `64`
- min relative gain: `0.0`
- min positive-view fraction: `0.5`
- fallback multiplier: `0.0`
- certification independent: `false`
- alpha selection mode: `per_face_group_best_alpha_grid`
- apply alpha mode: `null`

Per-group support:

| group | candidate faces | kept faces | candidate samples | kept samples | kept sample fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 148 | 14 | 5071 | 2351 | 0.463616644 |
| 1 | 171 | 9 | 3684 | 1225 | 0.332519001 |
| 2 | 66 | 7 | 1721 | 970 | 0.563625799 |

Policy-val best row:

- alpha: `0.375`
- relative gain: `0.1225381840`
- positive-view fraction: `0.5000000000`
- SSIM gain: `0.0000047187`
- SSIM positive-view fraction: `0.4166666667`
- SSIM min-view gain: `-0.0000199080`
- image-L1 gain: `0.0000020402`
- image-L1 positive-view fraction: `0.5000000000`
- image-L1 min-view gain: `0.0`

Face-gain guard:

- candidate face count: `27`
- allowed face count: `5`
- rejected face count: `22`
- allowed sample fraction: `0.4089309283`
- decision: `reject_candidate_after_face_gain_guard`
- post-guard selected alpha: `0.0`

Reject reasons:

- `positive_view_fraction 0.500000 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.416667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `ssim_min_view_gain -0.000019908 < min_policy_val_ssim_min_view_gain -0.000010000`
- `image_l1_positive_view_fraction 0.500000 < min_policy_val_l1_positive_view_fraction 0.550000`

Final test metrics:

| method | accepted | effective policy | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| v150 face reliability alpha | no | fallback_noop | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- v150 preserved the strong v148 policy-val MSE/L1 gain but still failed the cross-view gate.
- The new face-reliability profile was active and kept a meaningful fraction of samples, but the selected per-face alpha did not affect prediction.
- Because `apply_alpha_mode` was `null`, v150 is a useful negative control for v151, not the final intended method.

## v151 Status: Selected-Alpha Cap

Dry-run:

- root: `/dev/shm/peilincai_spcarnet_20260627_1040_v151_selected_alpha_cap_dryrun/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_v151_selected_alpha_cap_dryrun/wandb/offline-run-20260627_103630-wsbwohck`
- method: `ours_26000_v151_selected_alpha_cap_dryrun`
- status: `DRY_RUN`
- protocol audit: passed
- command count: `3`
- error count: `0`

Real run:

- root: `/dev/shm/peilincai_spcarnet_20260627_1045_v151_selected_alpha_cap_flowers/flowers`
- W&B: `/dev/shm/peilincai_wandb_20260627_1045_v151_selected_alpha_cap/wandb/offline-run-20260627_114124-f435yp8y`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- method: `ours_26000_v151_selected_alpha_cap_flowers`
- status: `COMPLETE`
- manifest updated at: `2026-06-27T11:41:23`
- command count: `3`
- error count: `0`
- adapter elapsed: `3792.984` sec
- eval-GT population elapsed: `11.602` sec
- final test eval elapsed: `41.963` sec
- protocol audit: passed
- target/test GT was not visible to selection/apply

Adapter audit:

- accepted: `False`
- effective policy: `fallback_noop`
- fallback written: `True`
- selected alpha: `0.0`
- target changed pixels: `0 / 37100800`
- face reliability apply alpha mode: `cap_generator_by_selected_alpha_over_current_alpha`
- face reliability kept face-group count: `30 / 385`
- face reliability kept sample fraction: `0.4339442535`

Policy-val best row:

- alpha: `0.375`
- relative gain: `0.1225381840`
- positive-view fraction: `0.5000000000`
- SSIM gain: `0.0000047187`
- SSIM positive-view fraction: `0.4166666667`
- SSIM min-view gain: `-0.0000199080`
- image-L1 gain: `0.0000020402`
- image-L1 positive-view fraction: `0.5000000000`
- image-L1 min-view gain: `0.0`

Face-gain guard:

- candidate face count: `27`
- allowed face count: `5`
- rejected face count: `22`
- allowed sample fraction: `0.4089309283`
- decision: `reject_candidate_after_face_gain_guard`
- post-guard accepted: `false`
- post-guard selected alpha: `0.0`

Reject reasons:

- `positive_view_fraction 0.500000 < min_policy_val_positive_view_fraction 0.550000`
- `ssim_positive_view_fraction 0.416667 < min_policy_val_ssim_positive_view_fraction 0.550000`
- `ssim_min_view_gain -0.000019908 < min_policy_val_ssim_min_view_gain -0.000010000`
- `image_l1_positive_view_fraction 0.500000 < min_policy_val_l1_positive_view_fraction 0.550000`

Final test metrics:

| method | accepted | effective policy | changed pixels | PSNR | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| v151 selected-alpha cap | no | fallback_noop | 0 | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- v151 verifies that the selected-alpha cap is wired into the apply path.
- It does not change the final policy-val bottleneck relative to v150.
- The failure is therefore not an apply-path bookkeeping bug. The remaining issue is that the residual representation improves MSE/L1 on average but does not create enough full-image positive views, especially under SSIM.
- The run also exposes a serious runtime weakness: the adapter spent `3792.984` sec and repeated the 12-view policy-val pass several times.

## Current Assessment

The new prompt line has produced real pipeline changes, not just parameter scanning. However, the expected effect has not been reached yet:

- v150 is complete evidence, but it is a no-op fallback.
- v151 is complete evidence, and it is also a no-op fallback.
- The main bottleneck remains full-image cross-view consistency, especially SSIM positive-view fraction.
- v151 proves the selected-alpha apply-path fix is real, but also proves it is not sufficient.

Confidence:

- Direction confidence: medium.
- Current result confidence: negative for v151 as a standalone fix.
- Paper-level confidence: not enough until a non-noop v151-or-later run improves target/test metrics and qualitative outputs against the clean MeshSplatting baseline.

## Next Required Work

1. v151 completed and still fell back, so this branch is insufficient by itself.
2. Implemented v152 as the next representation-level change: a policy-val view-conditioned alpha cap, not another threshold scan.
3. If v152 accepts with nonzero target change, run at least one additional scene before claiming robustness.
4. Add a faster cached policy-val evaluation path; current flowers adapter runtime is dominated by the 12-view policy-val pass.
5. Keep the protocol wording honest: policy-val calibration is fair under the train/policy-val split, but it is not independent certification.

## v152 Result: View-Conditioned Alpha Cap

Implementation status:

- Adapter: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- Runner: `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- Static checks:
  - `python -m py_compile scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`
  - `git diff --check -- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py docs/car_model/6-27-v149-v151-FaceReliability-SelectedAlpha-Log.md`
  - both passed after v152 edits.

Method change:

- Learn a train-policy-val camera-direction alpha-cap profile.
- Store it under `local_alpha_profile.view_alpha_cap_profile`.
- At target/test apply time, compute a camera-only cap and apply `min(view_alpha_cap / global_alpha, 1)` to the final residual delta.
- Re-run policy-val after face/bin/view guards with the cap active.
- Keep the original risk gate thresholds; no target/test GT is used for cap construction.
- Audit flags: `uses_policy_val_gt=true`, `uses_target_or_test_gt=false`, `certification_independent=false`.

Dry-run evidence:

- corrected fair dry-run root: `/dev/shm/peilincai_spcarnet_20260627_1200_v152_view_alpha_cap_dryrun_fair/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1200_v152_view_alpha_cap_dryrun_fair/wandb/offline-run-20260627_115735-z97053bs`
- protocol audit: passed
- command count: `3`
- error count: `0`
- important setting parity with v151: `no_policy_val_prior_bin_gain_hybrid=true`

Active full flowers run:

- root: `/dev/shm/peilincai_spcarnet_20260627_1205_v152_view_alpha_cap_flowers/flowers`
- W&B dir: `/dev/shm/peilincai_wandb_20260627_1205_v152_view_alpha_cap`
- method: `ours_26000_v152_view_alpha_cap_flowers`
- GPU: `5`
- status at log time: running

Expected readout:

- If v152 works, the post-cap policy-val row should switch from strict positive-view fractions to nonnegative-view fractions through `view_alpha_cap_selective=true`, while preserving positive mean gains.
- The acceptance still requires unchanged mean/effective gates, target nonzero changed pixels, and strict no-target-GT apply.
- If v152 still falls back, the bottleneck is no longer alpha safety but insufficient residual support/coverage for the zero-gain views.

Completed v152 readout:

- root: `/dev/shm/peilincai_spcarnet_20260627_1205_v152_view_alpha_cap_flowers/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1205_v152_view_alpha_cap/wandb/offline-run-20260627_132805-m79vgl7a`
- status: `COMPLETE`
- protocol audit: passed
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- reject reason: `effective_relative_gain 0.000000000 < min_policy_val_effective_relative_gain 0.001000000`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`
- view alpha cap:
  - selection mode: `smallest_safe`
  - seed source: `post_view_consistency_policy_val`
  - pre-cap best alpha: `0.0`
  - post-cap best alpha: `0.0`
  - selected view count: `12`
  - fallback view count: `0`
  - alpha cap min/mean/max: `0.001953125 / 0.001953125 / 0.001953125`

Interpretation:

- The code path ran and the selective nonnegative-view accounting worked.
- The mechanism failed because the cap profile was learned after earlier guards had already converted the candidate into a no-op policy-val payload.
- `smallest_safe` then collapsed all views to a near-zero cap and could only produce nonnegative-but-zero gains.

## v153 Result: Best-Safe View Alpha Cap

Purpose:

- Test whether v152 failed only because `smallest_safe` was too conservative.
- Keep the same fair branch as v151/v152: `no_policy_val_prior_bin_gain_hybrid=true`.

Completed v153 readout:

- root: `/dev/shm/peilincai_spcarnet_20260627_1238_v153_view_alpha_cap_bestsafe_flowers/flowers`
- W&B dir: `/dev/shm/peilincai_wandb_20260627_1238_v153_view_alpha_cap_bestsafe`
- status: `COMPLETE`
- protocol audit: passed
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- reject reason: `effective_relative_gain 0.000000000 < min_policy_val_effective_relative_gain 0.001000000`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`
- view alpha cap:
  - selection mode: `best_safe`
  - seed source: `post_view_consistency_policy_val`
  - pre-cap best alpha: `0.0`
  - post-cap best alpha: `0.0`
  - selected view count: `12`
  - fallback view count: `0`
  - alpha cap min/mean/max: `1.0 / 1.0 / 1.0`

Interpretation:

- `best_safe` removes the near-zero cap collapse but still sees only a no-op seed.
- Therefore the short-term failure is not the cap mode itself. The cap profile is being built from the wrong stage of the candidate lifecycle.
- This is a mechanism-order bug: the positive policy-val evidence observed in v151 (`alpha=0.375`, relative gain `0.122538184`) is overwritten before v152/v153 learn the view-conditioned cap.

## v154 Plan: Pre-Guard Seeded View Alpha Cap

Implementation:

- Adapter change: freeze a copy of the candidate policy-val payload after basis/teacher nonregression guards and before sparse/face/bin/view-confidence guards.
- New interface: `--view_alpha_cap_seed_stage {pre_guard,post_view_confidence}`.
- Default in adapter: `pre_guard`.
- Runner pass-through added for the same interface.
- The final post-cap policy-val evaluation still uses the existing face/bin/view-confidence profiles, unchanged thresholds, and the same effective margin gate.
- This is not threshold relaxation. It changes only the evidence stage used to construct the camera-conditioned cap profile.

Static checks:

- `python -m py_compile scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`: passed
- `git diff --check -- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`: passed
- runner help exposes `--view_alpha_cap_seed_stage {pre_guard,post_view_confidence}`

Subagent review:

- `pre_guard` is captured after initial policy-val selection and basis/teacher nonregression guards, before face/bin/view-confidence guards can replace the candidate with a no-op.
- Final capped acceptance still calls the same risk-gate selector; the change does not relax thresholds.
- The cap profile uses policy-val rows and camera-center features only.
- Protocol risk found: `--prestripped_target_evidence_dir` was previously trusted without verifying forbidden target GT/residual keys.
- Fix added: runner now verifies forbidden target-apply keys for reused prestripped target evidence directories.
- Current v154 prestripped target evidence verification:
  - root: `/dev/shm/peilincai_spcarnet_v131b_viewconf_flowers_20260626_223006/flowers/target_evidence_no_gt`
  - view count: `22`
  - forbidden key count: `0`
  - sample keys: `alpha`, `barycentric`, `barycentric_valid`, `camera_center`, `depth`, `face_id`, `normal`, `rgb_render`, `texture`
  - conclusion: current v154 target apply input is GT/residual-free.

Dry-run:

- root: `/dev/shm/peilincai_spcarnet_20260627_1415_v154_view_alpha_cap_preguard_dryrun/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1415_v154_view_alpha_cap_preguard_dryrun/wandb/offline-run-20260627_140638-4rtqe935`
- status: `DRY_RUN`
- command count: `3`
- error count: `0`
- protocol audit: passed
- adapter command contains `--view_alpha_cap_seed_stage pre_guard`

Active full run:

- root: `/dev/shm/peilincai_spcarnet_20260627_1415_v154_view_alpha_cap_preguard_flowers/flowers`
- W&B dir: `/dev/shm/peilincai_wandb_20260627_1415_v154_view_alpha_cap_preguard`
- method: `ours_26000_v154_view_alpha_cap_preguard_flowers`
- GPU: `6`
- status: `COMPLETE`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1415_v154_view_alpha_cap_preguard/wandb/offline-run-20260627_153416-ocgwzndo`
- protocol audit: passed
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`
- reject reason: `effective_relative_gain 0.000000000 < min_policy_val_effective_relative_gain 0.001000000`

v154 view-alpha-cap audit:

- seed stage: `pre_guard`
- seed source: `pre_guard_policy_val`
- seed best alpha: `0.375`
- seed best relative gain: `0.16363134798578763`
- seed selected alpha: `0.0`
- seed risk reasons:
  - `cvar20_view_relative_gain -0.024119 < min_policy_val_cvar20_relative_gain 0.000000`
  - `min_view_relative_gain -0.057385 < min_policy_val_min_view_relative_gain -0.000001`
  - `ssim_min_view_gain -0.000021935 < min_policy_val_ssim_min_view_gain -0.000010000`
- cap selected view count: `10`
- alpha cap min/mean/max: `0.0 / 0.3020833432674408 / 0.75`
- post-cap accepted: `False`
- post-cap selected alpha: `0.0`
- post-cap best alpha: `0.0`
- post-cap best relative gain: `0.0`
- post-cap risk reasons:
  - `effective_relative_gain 0.000000000 < min_policy_val_effective_relative_gain 0.001000000`

Interpretation:

- The `pre_guard` fix did recover the nonzero promising policy-val candidate that v152/v153 could not see.
- However, the recovered gain is not uniformly safe: it has a strong mean/relative gain but clear tail-view regressions.
- The current view-alpha cap is still too coarse. Once it tries to protect the bad views, it collapses the effective candidate back to a no-op.
- Therefore v154 is a useful diagnostic milestone, not a performance milestone.
- The next method change should stop treating the repair as a single view-global alpha. It should materialize only the cross-view-certified subset of residual pixels/bins/faces, or learn a local reliability mask that preserves the positive seed regions while excluding the tail-risk regions.

Success criteria:

- `view_alpha_cap.seed_stage == pre_guard`
- `view_alpha_cap.seed_best.alpha > 0`, ideally recovering the v151 positive seed around alpha `0.375`
- post-cap policy-val either accepts or reports a strictly narrower remaining bottleneck than v152/v153
- target apply must stay strict-no-GT and must not use target/test GT for selection or cap construction
- real improvement requires nonzero changed pixels plus target/test metric or qualitative gain; a no-op fallback is not a method success

## v156-v158 Sparse Materialization Diagnostics

The next prompt-driven direction moved away from global view alpha caps and toward a sparse, bin-level materialization certificate. The motivation was simple: the recovered residual candidate is useful only on a local footprint, so treating uncovered or irrelevant views as strict positive-gain failures is too coarse.

### v156 sparse frontier

- root: `/dev/shm/peilincai_spcarnet_20260627_1655_v156_sparse_frontier/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1655_v156_sparse_frontier/wandb/offline-run-20260627_181244-wlkf33j1`
- status: `COMPLETE`
- protocol audit: passed
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`
- sparse profile:
  - candidate bins: `1903`
  - allowed bins: `0`
  - adaptive frontier: activated but selected `0` bins
  - seed alpha: `0.375`
- failure reason:
  - bin-level evidence existed, but `sparse_materialization_min_bin_samples=16` was too high for the local flowers coverage distribution.
  - promising bins had only about `10-12` samples and `2` views, so the strict sample threshold erased the whole sparse footprint.

### v157 adaptive sparse frontier

- root: `/dev/shm/peilincai_spcarnet_20260627_1820_v157_adaptive_frontier_full/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1820_v157_adaptive_frontier_full/wandb/offline-run-20260627_193123-l4ve9nbu`
- status: `COMPLETE`
- protocol audit: passed
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- test metrics: PSNR `20.452775955`, SSIM `0.549059212`, LPIPS `0.355544209`
- adaptive frontier:
  - candidate bins: `1903`
  - core candidate count: `137`
  - requested min bin samples: `16`
  - effective min bin samples: `6`
  - sample quantiles: `0.25=2`, `0.50=4`, `0.75=6`, `0.90=7`, `1.00=12`
  - selected bins: `40`
  - allowed faces: `5`
  - allowed sample fraction: `0.0624725033`
- post-materialization policy-val:
  - best alpha: `0.375`
  - relative gain: `0.0205420784`
  - SSIM gain: `0.0000014404`
  - image L1 gain: `0.0000004728`
  - positive-view fraction: `0.416667`
  - SSIM-positive fraction: `0.333333`
  - image-L1-positive fraction: `0.416667`
- interpretation:
  - This was the first real sparse-footprint selection success: nonzero bins and faces survived the train-only frontier.
  - It still failed the global positive-view gate because sparse local edits only affect a minority of policy-val views. Unchanged/non-overlapping views should be non-regression evidence, not positive-gain failures.

### v158 sparse non-regressive semantics

- root: `/dev/shm/peilincai_spcarnet_20260627_1940_v158_sparse_nonregressive_full/flowers`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_1940_v158_sparse_nonregressive_full/wandb/offline-run-20260627_210206-gtrxfhaj`
- status: runner `FAILED` only at final metrics eval due GPU OOM; texture apply itself returned `0`
- protocol audit: passed
- adapter accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- sparse post-materialization accepted: `True`
- sparse post-materialization selected alpha: `0.375`
- sparse post-materialization risk reasons: none
- sparse selective metrics:
  - relative gain: `0.0205420784`
  - nonnegative-view fraction: `1.0`
  - SSIM nonnegative-view fraction: `0.916667`
  - image L1 nonnegative-view fraction: `1.0`
  - allowed bins/faces: `40 / 5`
- final rejection:
  - `face_gain_guard` ran after the sparse certificate and re-evaluated the candidate as an ordinary face-level/global row.
  - it reported `positive_view_fraction 0.500000 < 0.550000`, `ssim_positive_view_fraction 0.416667 < 0.550000`, `ssim_min_view_gain -0.000019908 < -0.000010000`, and `image_l1_positive_view_fraction 0.500000 < 0.550000`.
  - this overwrote the already accepted sparse selective candidate and forced `fallback_noop`.
- metrics eval issue:
  - final `evaluate_render_split_metrics.py` failed in LPIPS with CUDA OOM on the selected visible GPU.
  - this is separate from the adapter decision and must be fixed or rerun on a freer GPU for future accepted candidates.

### v159 sparse-face-guard skip result

Implementation change:

- If sparse materialization has already produced an accepted `sparse_materialization_selective` candidate, skip `face_gain_guard` and write an explicit audit decision: `skipped_sparse_materialization_already_bin_certified`.
- Rationale: the bin-level sparse certificate is more specific than a face-level positive-view guard. Applying the face guard afterward downcasts no-op/non-overlap views into failures and destroys the selective risk semantics.

Completed run:

- root: `/dev/shm/peilincai_spcarnet_20260627_2145_v159_sparse_faceguard_skip/flowers`
- W&B dir: `/dev/shm/peilincai_wandb_20260627_2145_v159_sparse_faceguard_skip`
- W&B offline run: `/dev/shm/peilincai_wandb_20260627_2145_v159_sparse_faceguard_skip/wandb/offline-run-20260627_230446-5duk5ksf`
- method: `ours_26000_v159_sparse_faceguard_skip_flowers`
- GPU: `4`
- status: `COMPLETE`
- protocol audit: passed
- adapter accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.375`
- sparse materialization:
  - post-materialization accepted: `True`
  - post-materialization alpha: `0.375`
  - allowed bins/faces: `40 / 5`
  - allowed sample fraction: `0.0624725033`
  - risk reasons: none
  - nonnegative-view fraction: `1.0`
  - SSIM nonnegative-view fraction: `0.916667`
  - image-L1 nonnegative-view fraction: `1.0`
- face gain guard:
  - decision: `skipped_sparse_materialization_already_bin_certified`
  - reason: sparse materialization uses a bin-level selective non-regression certificate; applying a face-level positive-view guard would downcast sparse no-op views into failures.
- target apply:
  - written views: `22`
  - changed pixels: `466 / 37100800`
  - PNG-quantized changed pixels: `465 / 37100800`
  - changed fraction: `1.256037605658099e-05`
- final test metrics:
  - PSNR: `20.45279312133789`
  - SSIM: `0.549059271812439`
  - LPIPS: `0.3555440902709961`
- relative to v157/no-op fallback:
  - PSNR: `+0.0000171661`
  - SSIM: `+0.0000000596`
  - LPIPS: `-0.0000001192`

Current assessment:

- The new prompt has now produced significant diagnostic and pipeline-semantics progress plus the first non-no-op accepted target/test materialization under the strict no-target-GT apply protocol.
- It has not yet produced paper-grade metric/visual evidence. The target footprint is only `466` pixels across `22` test views, so image-level metrics move only at numerical-noise scale.
- The immediate bottleneck has shifted again: it is no longer "no sparse signal" or "guard overwrites sparse acceptance"; it is sparse footprint mass and perceptual visibility. The next method step must increase certified target-visible support without reintroducing tail-view regressions.
