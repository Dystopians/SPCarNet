# 2026-06-27 v134b/v135 View-Consistency Status

## v134b Positive-Consensus Residual Policy

Run root: `/dev/shm/peilincai_spcarnet_v134b_positive_consensus_flowers_20260627_0125`

W&B offline run: `/dev/shm/peilincai_wandb_v134b_positive_consensus_flowers_20260627_0125/wandb/offline-run-20260627_011556-tylij349`

Result: complete but no effective method change.

- `accepted=false`
- `effective_policy=fallback_noop`
- `changed_pixels=0 / 37100800`
- `local_alpha_profile.uncertainty_shrink_policy_mode=positive_consensus`
- `candidate_bin_count=20349`
- `bin_uncertainty_shrink_count=0`
- reject reason: positive-view, SSIM, L1, LPIPS positive fractions were all `0.0`, and effective relative gain was `0.0 < 0.001`
- test metrics remained fallback/no-op: PSNR `20.452776`, SSIM `0.549059`, LPIPS `0.355544`

Interpretation: v134b fixed the previous interface crash and proved that the strict positive-consensus policy is safe, but it also removed all bins on `flowers`. This is not a performance improvement. The bottleneck is cross-view residual inconsistency, not a missing threshold.

## v135 View-Cluster Expert Residual

Implemented in:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New mechanism:

- Fit target-safe view clusters using normalized `camera_center` from fit evidence only.
- Accumulate separate per-cluster face/UV residual atlases in addition to the global atlas.
- At prediction/apply time, route each view to the nearest camera-center expert.
- Unsupported expert bins fall back to the global residual, avoiding out-of-support hallucination.
- Save expert tensors in the atlas `.npz` for audit/repro.

Key CLI:

- `--view_cluster_expert_count`
- `--view_cluster_feature_mode camera_center`
- `--view_cluster_min_views`
- `--view_cluster_min_bin_samples`
- `--view_cluster_fallback_mode global`

Validation status:

- `py_compile` passed for both changed scripts.
- `git diff --check` passed for both changed scripts.
- v135 flowers experiment finished under strict no-target-GT apply:
  `/dev/shm/peilincai_spcarnet_v135_viewcluster_experts_flowers_20260627_0218`
- W&B directory:
  `/dev/shm/peilincai_wandb_v135_viewcluster_experts_flowers_20260627_0218`
- final status: complete but no effective method change
- `accepted=false`, `effective_policy=fallback_noop`, `changed_pixels=0`
- fit cluster counts: `[12, 14, 8]`
- supported expert bins: `9779 / 262656 = 0.03723`
- `local_alpha_profile.candidate_bin_count=20343`
- `bin_uncertainty_shrink_count=0`
- test metrics remained fallback/no-op: PSNR `20.452776`, SSIM `0.549059`, LPIPS `0.355544`

Interpretation: the view-cluster atlas itself was built and audited correctly, but the downstream positive-consensus shrink still selected zero bins. The representation upgrade alone did not overcome the policy-val rejection bottleneck.

## v136/v136b Cluster-Local Shrink

Implemented in:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New mechanism:

- Calibrate bin uncertainty shrink per `(view_cluster, face, bin)` instead of only per `(face, bin)`.
- During target apply, route each view to its closest expert and read the cluster-local shrink profile.
- Keep an optional global fallback flag, but v136b used no fallback so that the audit directly measures cluster-local support.
- Add CLI alias `--enable_policy_val_cluster_local_shrink` for prompt compatibility.

v136 first run:

- run root: `/dev/shm/peilincai_spcarnet_v136_cluster_local_shrink_flowers_20260627_020252`
- W&B offline run: `/dev/shm/peilincai_wandb_v136_cluster_local_shrink_flowers_20260627_020252/wandb/offline-run-20260627_020425-7qok5nw1`
- failed with `ValueError: too many values to unpack (expected 2)` in the shrink audit loop after moving from `(face, bin)` keys to `(cluster, face, bin)` keys
- fixed by restoring the global guard loop to `face, bin_id = key` and handling cluster keys only in the cluster-local shrink row loop

v136b rerun:

- run root: `/dev/shm/peilincai_spcarnet_v136b_cluster_local_shrink_flowers_20260627_020617`
- W&B offline run: `/dev/shm/peilincai_wandb_v136b_cluster_local_shrink_flowers_20260627_020617/wandb/offline-run-20260627_022233-98bz4wm0`
- `accepted=false`, `effective_policy=fallback_noop`, `changed_pixels=0`
- `view_cluster_local_shrink=true`
- policy-val cluster view counts: `{'0': 5, '1': 4, '2': 3}`
- `candidate_bin_count=23680`
- `bin_uncertainty_shrink_count=0`
- `view_cluster_selected_cluster_count=0`
- test metrics remained fallback/no-op: PSNR `20.452776`, SSIM `0.549059`, LPIPS `0.355544`

Interpretation: cluster-local shrink is now a real train/eval pipeline feature and the implementation is valid, but the data still contains no bins that pass the positive-consensus rule. This means the no-op issue is not merely caused by mixing incompatible camera clusters.

## v137 Train-Only Adaptive Residual-Activity Threshold

Implemented in:

- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New mechanism:

- Scan fit/train evidence only.
- Estimate the active residual distribution from `teacher_residual_l1`.
- Replace a hand-picked `--min_l1` with an automatic quantile threshold.
- Record the full threshold audit in the run manifest, including `uses_target_or_test_gt=false`.

Run:

- run root: `/dev/shm/peilincai_spcarnet_v137_activity_cluster_shrink_flowers_20260627_022925`
- W&B offline run: `/dev/shm/peilincai_wandb_v137_activity_cluster_shrink_flowers_20260627_022925/wandb/offline-run-20260627_024917-2e66qmx0`
- protocol audit passed, no target GT visible to apply/selection
- selected train-only threshold: `min_l1=0.01174163818359375`
- residual distribution: `zero_fraction=0.8837033451397165`, q50 `0.0`, q75 `0.0`, q90 `0.01174163818359375`, q95 `0.0196533203125`, q99 `0.0430908203125`
- `accepted=false`, `effective_policy=fallback_noop`, `changed_pixels=0`
- reject reason: positive-view, SSIM, image-L1, and LPIPS positive-view fractions all remained `0.0`; effective relative gain remained `0.0 < 0.001`
- `view_cluster_local_shrink=true`
- `candidate_bin_count=6626`
- `bin_uncertainty_shrink_count=0`
- `view_cluster_selected_cluster_count=0`
- test metrics remained fallback/no-op: PSNR `20.452776`, SSIM `0.549059`, LPIPS `0.355544`

Interpretation: the adaptive residual-activity threshold correctly filters out the 88.37% zero-residual mass and reduces candidates from about 23k to 6.6k, but it still does not create image-level positive evidence. This is an important negative result: the current residual-texture/shrink family is safety-certified but not strong enough to improve the outdoor `flowers` target.

## Current Read

v134b is a negative result but useful: it demonstrates that "only apply bins with strict multi-view positive consensus" collapses to no-op on the difficult outdoor `flowers` scene.

v135-v137 are real engineering progress, but not yet result progress. The new prompt direction has improved the method's auditability and target-safe adaptivity:

- view-conditioned residual experts are implemented
- cluster-local shrink is implemented
- train-only adaptive residual-activity thresholding is implemented
- W&B and manifest evidence are saved
- strict leakage checks pass

However, the expected performance effect has not appeared. The method still falls back to no-op on `flowers`, so this line has not reached the paper-level target. The next method change should stop optimizing the residual-MSE shrink proxy alone and directly certify image-space improvement at the bin/patch level, because all current failures show `0.0` positive-view fraction across image PSNR/SSIM/L1/LPIPS gates.
