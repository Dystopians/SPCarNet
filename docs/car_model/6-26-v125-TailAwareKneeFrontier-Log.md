# v125 Tail-Aware Knee Frontier Log

Date: 2026-06-26

## Scope

This log documents the existing v125 TailAwareKneeFrontier result artifacts only.
No code or existing documentation was changed for this documentation pass.

v125 follows the v124 `knee` frontier next-fix note by enabling the tail-aware
frontier mode:

```text
--policy_val_alpha_frontier_mode tail_knee
--policy_val_alpha_frontier_tail_knee_min_score_fraction 0.7
--policy_val_alpha_frontier_tail_knee_min_regression_count 3
--policy_val_alpha_frontier_tail_knee_eps 1e-10
```

The replay keeps the v122/v124 support and certification context: coview
support transfer enabled, skip existing atlas bins, `max_faces=512`,
`neighbor_stride=4`, `min_source_mean_l1=0.005`, `residual_scale=0.125`,
train policy-val thresholds, stripped target evidence for apply, and test GT
restored only for final metric evaluation.

## Source Artifacts

Baseline context:

- v122/v118-v119 log: `docs/car_model/6-26-v118-v119-FaceGraphResidualTransfer-Milestone-And-Bottleneck-Log.md`
- v124 log: `docs/car_model/6-26-v124-AdaptiveKneeFrontier-Log.md`
- v122 qualitative panels:
  - `docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_panel.png`
  - `docs/car_model/vnext_artifacts/counter_v122_alpha1875_panel_20260626/counter_v122_vs_v106_all_positive_panel.png`

v125 counter:

```text
/dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter
/dev/shm/peilincai_wandb_v125_counter_20260626_202759/wandb/offline-run-20260626_204605-pczjbz74
```

Key result files:

- manifest: `reports/counter_vnext_certified_residual_texture_manifest.json`
- run report: `reports/counter_vnext_certified_residual_texture_report.md`
- results: `reports/counter_ours_26000_v125_tail_knee_frontier_counter_test_results.json`
- per-view: `reports/counter_ours_26000_v125_tail_knee_frontier_counter_test_per_view.json`
- eval-GT population audit: `reports/counter_ours_26000_v125_tail_knee_frontier_counter_test_eval_gt_population_audit.json`
- adapter audit: `model/surface_residual_region_texture_adapter_audit.json`
- topology audit: `model/topology_audit.json`
- stripped-target audit: `target_evidence_no_gt/target_evidence_no_gt_audit.json`

v125 flowers:

```text
/dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers
/dev/shm/peilincai_wandb_v125_flowers_20260626_202830/wandb/offline-run-20260626_203730-mjxsas0m
```

Key result files:

- manifest: `reports/flowers_vnext_certified_residual_texture_manifest.json`
- run report: `reports/flowers_vnext_certified_residual_texture_report.md`
- results: `reports/flowers_ours_26000_v125_tail_knee_frontier_flowers_test_results.json`
- per-view: `reports/flowers_ours_26000_v125_tail_knee_frontier_flowers_test_per_view.json`
- eval-GT population audit: `reports/flowers_ours_26000_v125_tail_knee_frontier_flowers_test_eval_gt_population_audit.json`
- adapter audit: `model/surface_residual_region_texture_adapter_audit.json`
- topology audit: `model/topology_audit.json`
- stripped-target audit: `target_evidence_no_gt/target_evidence_no_gt_audit.json`

## Command Audit

The exact `cmd_string` entries are recorded in each v125 manifest under
`commands[]` and repeated in each run report under `## Commands`. All audited
commands returned `0`.

Counter commands:

| stage | returncode | elapsed sec | log |
|---|---:|---:|---|
| `strip_target_evidence_no_gt` | 0 | 89.666 | `/dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/logs/01b_strip_target_evidence_no_gt.log` |
| `apply_certified_residual_texture` | 0 | 914.944 | `/dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/logs/02_certified_texture.log` |
| `populate_eval_gt_from_target_evidence` | 0 | 23.844 | `/dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/logs/02b_populate_eval_gt.log` |
| `evaluate_vnext_target` | 0 | 56.905 | `/dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/logs/03_eval.log` |

Flowers commands:

| stage | returncode | elapsed sec | log |
|---|---:|---:|---|
| `strip_target_evidence_no_gt` | 0 | 71.441 | `/dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/logs/01b_strip_target_evidence_no_gt.log` |
| `apply_certified_residual_texture` | 0 | 409.822 | `/dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/logs/02_certified_texture.log` |
| `populate_eval_gt_from_target_evidence` | 0 | 11.718 | `/dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/logs/02b_populate_eval_gt.log` |
| `evaluate_vnext_target` | 0 | 45.999 | `/dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/logs/03_eval.log` |

Representative exact evaluation commands:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py --model_path /dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/model --split test --methods ours_26000_v125_tail_knee_frontier_counter --output /dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/reports/counter_ours_26000_v125_tail_knee_frontier_counter_test_results.json --per_view_output /dev/shm/peilincai_spcarnet_v125_counter_tail_knee_frontier_20260626_202759/counter/reports/counter_ours_26000_v125_tail_knee_frontier_counter_test_per_view.json --merge_model_results
```

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py --model_path /dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/model --split test --methods ours_26000_v125_tail_knee_frontier_flowers --output /dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/reports/flowers_ours_26000_v125_tail_knee_frontier_flowers_test_results.json --per_view_output /dev/shm/peilincai_spcarnet_v125_flowers_tail_knee_frontier_20260626_202830/flowers/reports/flowers_ours_26000_v125_tail_knee_frontier_flowers_test_per_view.json --merge_model_results
```

The exact apply commands are long and are intentionally not duplicated here;
they are available verbatim in the two manifest `commands[1].cmd_string` fields
and in the two run report command sections listed above. The key v125 apply
delta from v124 is the tail-knee frontier mode and tail regression guard fields
listed in the Scope section.

## Protocol Audit

| scene | status | protocol passed | selection uses test GT | GT visible to apply | GT visible to eval | forbidden target keys stripped | command errors |
|---|---|---|---|---|---|---|---|
| counter | COMPLETE | True | False | False | True | True | none |
| flowers | COMPLETE | True | False | False | True | True | none |

Additional audit details:

| scene | stripped target audit | eval-GT population | capacity selected on | thresholds selected on |
|---|---|---|---|---|
| counter | removed `rgb_gt` for 30 views; `target_gt_visible_to_apply=False` | wrote 30 / 30 GT images, missing renders 0 | `train_policy_val_and_gt_free_target_footprint` | `train_policy_val` |
| flowers | removed `rgb_gt` for 22 views; `target_gt_visible_to_apply=False` | wrote 22 / 22 GT images, missing renders 0 | `train_policy_val_and_gt_free_target_footprint` | `train_policy_val` |

Topology audits stayed in the expected 2 percent compact source family:

| scene | pre triangles | post triangles | removed fraction | invalid indices | degenerate faces |
|---|---:|---:|---:|---:|---:|
| counter | 9,841,068 | 9,644,247 | 0.019999963 | 0 | 0 |
| flowers | 8,683,018 | 8,509,358 | 0.019999959 | 0 | 0 |

## Counter Result

Audit state:

- accepted: `True`
- selected alpha: `0.25`
- frontier reason: `selected_tail_knee_before_robust_tail_regression`
- best relative-gain alpha in frontier audit: `0.375`
- selected support mode: `coview_face_residual_transfer`
- support added faces: `275`
- support candidate faces: `1849`
- target changed fraction: `0.020591568328217506`
- PNG-quantized changed fraction: `0.010340460036787773`
- min-view changed fraction: `0.002148770346845543`
- CVaR20 changed fraction: `0.007080017940428872`

Held-out test metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v115/v106 anchor | 27.499700546 | 0.867478549 | 0.238779992 |
| v122 fixed alpha1875 | 27.500209808 | 0.867498755 | 0.238754511 |
| v124 adaptive knee frontier | 27.500129700 | 0.867495179 | 0.238751680 |
| v125 tail-aware knee frontier | 27.499988556 | 0.867463112 | 0.238734171 |

Comparison to v122:

- PSNR: `-0.000221252`
- SSIM: `-0.000035644`
- LPIPS: `-0.000020340` lower is better

Interpretation: v125 activates the tail-aware frontier and improves LPIPS
relative to v122/v124, but it selects `alpha=0.25`, which reopens the
PSNR/SSIM regression that v122 avoided with fixed `alpha=0.1875`. The result is
validated and useful, but it is not a strict replacement for v122.

## Flowers Result

Audit state:

- accepted: `False`
- selected alpha: `0.0`
- frontier reason: `no_safe_rows`
- selected support mode: `base_carrier`
- support added faces: `0`
- target changed fraction: `0.0`
- PNG-quantized changed fraction: `0.0`

Held-out test metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v122 fixed alpha1875 / fallback | 20.452775955 | 0.549059212 | 0.355544209 |
| v124 adaptive knee frontier / fallback | 20.452775955 | 0.549059212 | 0.355544209 |
| v125 tail-aware knee frontier / fallback | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation: flowers remains a safe no-op. This preserves the no-regression
behavior seen in v122/v124, but it does not add cross-scene improvement evidence.

## Status And Conclusion

v125 TailAwareKneeFrontier is validated as an ablation:

- both scenes completed;
- all audited commands returned `0`;
- W&B offline logging exists for both scenes;
- protocol audit passed with target/test GT hidden from selection and apply;
- counter selected the intended `tail_knee` frontier path;
- flowers safely rejected and fell back to no-op.

v125 should not be promoted over v122. On the only improving scene, `counter`,
v125 does not beat the v122 fixed alpha1875 row on PSNR or SSIM, even though it
does improve LPIPS. On `flowers`, v125 ties v122/v124 through no-op fallback.
The honest status is therefore: validated but not-promoted ablation. The current
best reported fixed-policy counter row remains v122, and v125 mainly shows that
the tail-aware knee rule still permits too much alpha under the current
thresholds.
