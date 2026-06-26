# 6-26 vNext Flowers Input Rebuild, Ready7, and Same-Evidence Fallback Log

Date: 2026-06-26

This log records the third missing-scene rebuild after the ready4 milestone. It extends the local vNext full9 input coverage from `6 / 9` to `7 / 9` by rebuilding `flowers` fit/target evidence plus a policy-val-pruned carrier under `/dev/shm`.

## Result

`flowers` is now locally input-ready for the manifest runner:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json
```

The full9 preflight moved from `6 / 9` ready to `7 / 9` ready. Remaining missing-input scenes:

```text
bicycle
kitchen
```

Preflight artifact:

```text
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.json
```

## Rebuild Notes

Train visible-bary base evidence was rebuilt for `46` train views with the same `images_2` convention used by the current stump/treehill vNext input chain. Teacher fit evidence was built with `--allow_resize`.

Teacher cache summary:

```text
processed_views: 46
mean_active_fraction: 0.305503
mean_target_l1: 0.011414
mean_raw_parent_delta_l1: 0.020640
mean_positive_teacher_gain_l1: 0.010627
top_support_rows: 8192
```

Target visible-bary evidence was rebuilt for `22` test views and symlinked to the manifest path:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence
```

Unpruned carrier build:

```text
carriers: 64
regions: 552
evidence_faces: 932
```

Policy-val pruning succeeded:

```text
output carriers: 57
retained faces: 342
removed faces: 588
greedy removed faces: 0
```

## Strict vNext Flowers Run

W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_flowers_strict_20260626_0935/wandb/offline-run-20260626_094324-x92wgfik
```

Flowers strict run status:

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

Same-evidence held-out metrics:

```text
same-evidence parent PSNR: 19.519194
same-evidence parent SSIM: 0.490780
same-evidence parent LPIPS: 0.424170
vNext fallback PSNR: 19.519194
vNext fallback SSIM: 0.490780
vNext fallback LPIPS: 0.424170
```

This same-evidence row is important. The rebuilt `flowers` target evidence is `images_2` at `1600 x 1054`, while the historical Phase-F parent metrics in the source model are lower-resolution `1256 x 828`. Directly comparing the new vNext output to the old source-model `test_results.json` is therefore invalid. The fair comparison is `rgb_render` versus `rgb_gt` from the same target evidence cache, and under that comparison the fallback is exact no-op.

Main rejection reasons:

```text
cvar20_view_relative_gain -0.224441 < 0.000000
min_view_relative_gain -0.278408 < -0.000001
ssim_gain -0.000082279 < -0.000000100
ssim_positive_view_fraction 0.333333 < 0.550000
ssim_min_view_gain -0.000208378 < -0.000010000
image_l1_gain -0.000000106 < 0.000000000
image_l1_positive_view_fraction 0.500000 < 0.550000
image_l1_min_view_gain -0.000023652 < -0.000001000
```

This is a useful negative result. The rebuilt input chain works, the strict no-target-GT protocol is intact, and the fixed structure-aware certificate correctly rejects a flowers candidate that is unsafe for lower-tail views, SSIM, and image L1.

## Runtime Note

The strict target stripping step is a real engineering bottleneck for large `images_2` target caches:

```text
strip_target_evidence_no_gt elapsed_sec: 2017.35
apply_certified_residual_texture elapsed_sec: 244.52
populate_eval_gt elapsed_sec: 11.87
evaluate_vnext_target elapsed_sec: 41.75
```

The long strip time is dominated by rewriting large NPZ files while removing target GT/residual fields. It does not affect metric validity, but it should be optimized before large full9 reruns.

## Artifacts

Flowers strict run:

```text
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_vnext_certified_residual_texture_report.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_vnext_certified_residual_texture_manifest.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/surface_residual_region_texture_adapter_audit.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_ours_26000_vnext_structure_aware_shrink_test_results.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_ours_26000_vnext_structure_aware_shrink_test_per_view.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_same_evidence_parent_vs_vnext_test_results.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_same_evidence_parent_vs_vnext_test_per_view.json
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/policyval_pruned_carrier.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/teacher_surface_evidence_summary.json
```

Ready7 preflight:

```text
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.json
```

## Current Boundary

This milestone improves engineering completeness and resolves a resolution-comparison trap. It changes the current vNext state from:

```text
ready6 input coverage, with stump and treehill correctly rejected to fallback/no-op
```

to:

```text
ready7 input coverage, with stump/treehill/flowers correctly rejected to same-evidence fallback/no-op
```

The remaining hard blocker for full9 is rebuilding inputs for `bicycle,kitchen`. The method-quality blocker remains unchanged: the fixed vNext policy has four accepted/nonzero ready scenes with tiny gains, while several outdoor/tail-risk scenes still require exact fallback.
