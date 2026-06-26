# 6-26 vNext Treehill Input Rebuild, Ready6, and Rejection Log

Date: 2026-06-26

This log records the second missing-scene rebuild after the ready4 milestone. It extends the local vNext full9 input coverage from `5 / 9` to `6 / 9` by rebuilding `treehill` fit/target evidence plus a policy-val-pruned carrier under `/dev/shm`.

## Result

`treehill` is now locally input-ready for the manifest runner:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/carrier.json
```

The full9 preflight moved from `5 / 9` ready to `6 / 9` ready. Remaining missing-input scenes:

```text
bicycle
flowers
kitchen
```

Preflight artifact:

```text
docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.json
```

## Rebuild Notes

Train visible-bary base evidence was rebuilt for `46` train views with the same `images_2` convention used by the current ready/stump vNext chain. Teacher fit evidence was built with `--allow_resize`, matching the stump lesson that teacher render resolution can differ from evidence resolution.

Teacher cache summary:

```text
processed_views: 46
mean_active_fraction: 0.243095
mean_target_l1: 0.009339
mean_raw_parent_delta_l1: 0.020204
mean_positive_teacher_gain_l1: 0.008936
top_support_rows: 8192
```

Target visible-bary evidence was rebuilt for `18` test views and symlinked to the manifest path:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/treehill/target_evidence
```

Unpruned carrier build:

```text
carriers: 64
regions: 552
evidence_faces: 1061
```

Policy-val pruning succeeded:

```text
output carriers: 45
retained faces: 226
removed faces: 824
greedy removed faces: 0
```

## Strict vNext Treehill Run

W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_treehill_strict_20260626_0832/wandb/offline-run-20260626_083741-nrcul71g
```

Treehill strict run status:

```text
status: COMPLETE
protocol_audit_passed: true
selection_uses_test_gt: false
target_gt_visible_to_apply: false
target_gt_visible_to_selection: false
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
target_changed_fraction: 0.0
```

Held-out test metrics are the no-op/fallback parent metrics:

```text
PSNR: 20.838715
SSIM: 0.558089
LPIPS: 0.445541
```

Main rejection reasons:

```text
cvar20_view_relative_gain -0.053640 < 0.000000
min_view_relative_gain -0.077837 < -0.000001
ssim_gain -0.000009413 < -0.000000100
ssim_min_view_gain -0.000124395 < -0.000010000
image_l1_min_view_gain -0.000009734 < -0.000001000
```

This is a useful negative result. The rebuilt input chain works, the strict no-target-GT protocol is intact, and the fixed structure-aware certificate correctly rejects a treehill candidate that improves mean MSE but is unsafe for lower-tail views and SSIM.

## Artifacts

Treehill strict run:

```text
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/treehill_vnext_certified_residual_texture_report.md
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/treehill_vnext_certified_residual_texture_manifest.json
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/surface_residual_region_texture_adapter_audit.md
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/treehill_ours_26000_vnext_structure_aware_shrink_test_results.json
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/treehill_ours_26000_vnext_structure_aware_shrink_test_per_view.json
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/policyval_pruned_carrier.md
docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/teacher_surface_evidence_summary.json
```

Ready6 preflight:

```text
docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.json
```

## Current Boundary

This milestone improves engineering completeness, not quality metrics. It changes the current vNext state from:

```text
ready5 input coverage, with stump correctly rejected to fallback/no-op
```

to:

```text
ready6 input coverage, with stump and treehill correctly rejected to fallback/no-op
```

The remaining hard blocker for full9 is rebuilding inputs for `bicycle,flowers,kitchen`. The method-quality blocker remains unchanged: the fixed vNext policy has four accepted/nonzero ready scenes with tiny gains, while outdoor tail-risk scenes can still be unsafe and must fall back.
