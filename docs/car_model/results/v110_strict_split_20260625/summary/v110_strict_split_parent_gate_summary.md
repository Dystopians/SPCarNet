# v110 Strict Split Parent Gate Summary

- output_root: `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625`
- detached_root: `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625`
- clean_baseline_root: `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k`
- scenes: `flowers, garden, counter, bonsai`
- all_reports_present: `False`
- all_results_present: `True`
- all_clean_results_present: `True`
- all_required_metrics_present: `False`
- missing_count: `5`

Metric triplets are `PSNR / SSIM / LPIPS`. Deltas are v110 minus the reference triplet.

| scene | report | results | clean method | clean | v106 parent method | v106 parent | v110 gated method | v110 gated | delta vs clean | delta vs v106 | missing |
|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|---|
| flowers | yes | yes | `ours_26000` | 19.682257 / 0.511822 / 0.394563 | `ours_26000_v106_podmoe_basepreserve_flowers` | 20.077723 / 0.531240 / 0.374393 | `ours_26000_v110_strict_train_even_odd_parent_gate_flowers` | 19.966076 / 0.522843 / 0.380387 | +0.283819 / +0.011021 / -0.014176 | -0.111647 / -0.008397 / +0.005994 | none |
| garden | no | yes | `ours_26000` | 25.029211 / 0.780035 / 0.201314 | `ours_26000_v106_podmoe_basepreserve_garden` | 25.790945 / 0.799382 / 0.174480 | `NA` | NA / NA / NA | NA / NA / NA | NA / NA / NA | missing_report_json; missing_method:v110_gated |
| counter | no | yes | `ours_26000` | 26.751774 / 0.862055 / 0.252003 | `ours_26000_v106_podmoe_basepreserve_counter` | 27.499645 / 0.867521 / 0.238847 | `NA` | NA / NA / NA | NA / NA / NA | NA / NA / NA | missing_report_json; missing_method:v110_gated |
| bonsai | yes | yes | `ours_26000` | 28.895233 / 0.896400 / 0.259493 | `ours_26000_v106_podmoe_basepreserve_bonsai` | 30.316090 / 0.907520 / 0.230050 | `NA` | NA / NA / NA | NA / NA / NA | NA / NA / NA | missing_method:v110_gated |

## Means

| label | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean_baseline | 25.089619 | 0.762578 | 0.276843 |
| v106_parent | 25.921101 | 0.776416 | 0.254443 |
| v110_gated | 19.966076 | 0.522843 | 0.380387 |
| v110_gated_minus_clean_baseline | 0.283819 | 0.011021 | -0.014176 |
| v110_gated_minus_v106_parent | -0.111647 | -0.008397 | 0.005994 |
