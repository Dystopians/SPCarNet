# 6-26 vNext Kitchen Input Rebuild, Ready8, and Accepted Milestone Log

Date: 2026-06-26

This log records the fourth missing-scene rebuild after the ready4 milestone. It extends local vNext full9 input coverage from `7 / 9` to `8 / 9` by rebuilding `kitchen` fit/target evidence plus a policy-val-pruned carrier under `/dev/shm`.

## Result

`kitchen` is now locally input-ready for the manifest runner:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/carrier.json
```

The full9 preflight moved from `7 / 9` ready to `8 / 9` ready. Remaining missing-input scene:

```text
bicycle
```

Preflight artifact:

```text
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.json
```

## Rebuild Notes

Train visible-bary base evidence was rebuilt for `46` train views with the current `images_2` convention. Teacher fit evidence was built with `--allow_resize`.

Teacher cache summary:

```text
processed_views: 46
mean_active_fraction: 0.343699
mean_target_l1: 0.008766
mean_raw_parent_delta_l1: 0.014311
mean_positive_teacher_gain_l1: 0.008613
top_support_rows: 8192
nonzero_faces: 1706920
```

Target visible-bary evidence was rebuilt for `35` test views and symlinked to the manifest path:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/kitchen/target_evidence
```

Unpruned carrier build:

```text
carriers: 64
regions: 544
evidence_faces: 2048
```

Policy-val pruning succeeded:

```text
output carriers: 57
retained faces: 1315
removed faces: 1384
greedy removed faces: 0
```

The carrier pruning uses train evidence only. The held-out test split is not used for carrier pruning, threshold selection, alpha selection, or fallback decisions.

## Strict vNext Kitchen Run

W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_kitchen_strict_20260626_1023/wandb/offline-run-20260626_102859-n6220f69
```

Kitchen strict run status:

```text
status: COMPLETE
protocol_audit_passed: true
selection_uses_test_gt: false
target_gt_visible_to_apply: false
target_gt_visible_to_selection: false
accepted: true
effective_policy: accepted_atlas
selected_alpha: 0.125
target_changed_fraction: 0.003549714
```

Same-evidence held-out metrics:

```text
same-evidence parent PSNR: 27.816387
same-evidence parent SSIM: 0.876443
same-evidence parent LPIPS: 0.199201
vNext structure-aware PSNR: 27.817173
vNext structure-aware SSIM: 0.876445
vNext structure-aware LPIPS: 0.199172
delta, better direction: +0.000786 PSNR / +0.00000256 SSIM / -0.00002818 LPIPS
```

Per-view better/tie/worse versus same-evidence parent:

| metric | better | tie | worse | mean better-direction delta |
|---|---:|---:|---:|---:|
| PSNR | `25` | `0` | `10` | `+0.000785828` |
| SSIM | `16` | `0` | `19` | `+0.000002580` |
| LPIPS | `30` | `0` | `5` | `+0.000028158` |

This is the first rebuilt missing-scene vNext run in this sequence that produces an accepted nonzero output rather than fallback/no-op. The quality gain is real under the same-evidence comparison, but it is still extremely small and should not be presented as a paper-level visual breakthrough.

## Runtime Note

The kitchen strict run is much faster than flowers because its target stripping step did not hit the same compressed-NPZ bottleneck:

```text
strip_target_evidence_no_gt elapsed_sec: 87.86
apply_certified_residual_texture elapsed_sec: 418.70
populate_eval_gt elapsed_sec: 24.17
evaluate_vnext_target elapsed_sec: 63.30
```

## Artifacts

Kitchen strict run:

```text
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_vnext_certified_residual_texture_report.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_vnext_certified_residual_texture_manifest.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/surface_residual_region_texture_adapter_audit.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_ours_26000_vnext_structure_aware_shrink_test_results.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_ours_26000_vnext_structure_aware_shrink_test_per_view.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_same_evidence_parent_vs_vnext_test_results.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_same_evidence_parent_vs_vnext_test_per_view.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/policyval_pruned_carrier.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/carrier_unpruned.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/teacher_surface_evidence_summary.json
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/teacher_surface_evidence_report.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/topology_audit.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/topology_audit.json
```

Ready8 preflight:

```text
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.json
```

## Current Boundary

This milestone improves both engineering completeness and strict-protocol evidence:

```text
ready8 input coverage, with kitchen accepted as a nonzero same-evidence three-metric micro-gain
```

It does not close the paper loop. The remaining full9 input blocker is `bicycle`, and the method-quality blocker remains effect size: the current fixed vNext policy has tiny accepted gains on several ready scenes, safe fallback on several outdoor/tail-risk scenes, and no proof yet that it beats v106 or clean MeshSplatting under a full9 fixed protocol.
