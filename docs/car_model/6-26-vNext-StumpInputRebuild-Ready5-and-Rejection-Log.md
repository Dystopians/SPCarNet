# 6-26 vNext Stump Input Rebuild, Ready5, and Rejection Log

Date: 2026-06-26

This log records the first missing-scene rebuild after the ready4 milestone. The goal was to test whether the vNext full9 gap can be closed by rebuilding the lost `/dev/shm` evidence chain one scene at a time.

## Result

`stump` is now locally input-ready for the manifest runner:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/carrier.json
```

The full9 preflight moved from `4 / 9` ready to `5 / 9` ready. Remaining missing-input scenes:

```text
bicycle
flowers
kitchen
treehill
```

Preflight artifact:

```text
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.json
```

## Rebuild Notes

Train visible-bary base evidence was rebuilt for `46` train views. The teacher fit-evidence step initially failed because Phase-J teacher renders are `1245x825` while the evidence cache is `1600x1060`; rerun succeeded with `--allow_resize`.

Teacher cache summary:

```text
processed_views: 46
mean_active_fraction: 0.154848
mean_target_l1: 0.003734
top_support_rows: 8192
```

Target visible-bary evidence was rebuilt for `16` test views and symlinked to the manifest path:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/stump/target_evidence
```

Unpruned carrier build:

```text
carriers: 64
regions: 458
evidence_faces: 533
```

## Interface Fix

The first carrier-prune attempt failed because `scripts/car_model/ecsr_prune_region_carriers_by_policy_val.py` called `fit_atlas()` without the newer multiscale/view-basis/teacher-basis parameters:

```text
TypeError: fit_atlas() missing 24 required positional arguments
```

The runner was fixed to preserve legacy pruning behavior by passing explicit disabled/default values for the newer arguments:

```text
surface_multiscale_prior_mode=none
view_conditioned_basis_mode=none
teacher_distilled_basis_mode=none
```

After the fix, policy-val pruning succeeded:

```text
input carriers: 64
output carriers: 30
candidate faces: 533
atlas faces: 519
retained faces: 85
removed faces: 434
greedy removals: 3
```

Prune artifact:

```text
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/policyval_pruned_carrier.md
```

## Strict vNext Stump Run

W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_stump_strict_20260626/wandb/offline-run-20260626_080257-h19iwtlv
```

Stump strict run status:

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
PSNR: 25.043329
SSIM: 0.689480
LPIPS: 0.349850
```

Main rejection reason:

```text
cvar20_view_relative_gain -0.172454 < 0.000000
min_view_relative_gain -0.344907 < -0.000001
```

This is a useful negative result. The rebuilt input chain works, the no-test-GT protocol is intact, and the certificate correctly rejects a tail-risky `stump` candidate instead of forcing a low-confidence edit.

## Artifacts

Stump strict run:

```text
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_vnext_certified_residual_texture_report.md
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_vnext_certified_residual_texture_manifest.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/surface_residual_region_texture_adapter_audit.md
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_ours_26000_vnext_structure_aware_shrink_test_results.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_ours_26000_vnext_structure_aware_shrink_test_per_view.json
```

Ready5 preflight:

```text
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.json
```

## Current Boundary

This milestone improves engineering completeness, not quality metrics. It changes the current vNext state from:

```text
ready4 with four accepted/nonzero scenes
```

to:

```text
ready5 input coverage, with stump correctly rejected to fallback/no-op
```

The remaining hard blocker for full9 is rebuilding inputs for `bicycle,flowers,kitchen,treehill`. The method-quality blocker remains the small-effect / tail-risk issue: `stump` shows policy-val mean MSE can improve while lower-tail views still make the candidate unsafe.
