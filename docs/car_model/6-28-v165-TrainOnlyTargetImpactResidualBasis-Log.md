# 2026-06-28 v165 Train-Only Target-Impact Residual Basis Log

## Purpose

v162-v164 exposed a bottleneck: target-visible/connected sparse growth can stay stuck when the target-visible bins have no eligible policy-val row. That keeps the method local and makes qualitative gains hard to see even when the base-preserve policy is safe.

v165 adds a default-off train-only target-impact residual basis path. When enabled, sparse materialization may add high-footprint target-visible bins even if they do not have a policy-val row, as long as the residual basis itself is learned from train/policy-val evidence and target/test GT is not used for selection.

## Implementation

- Adapter: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- Runner: `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- New main flag: `--enable_train_only_target_impact_residual_basis`
- New controls:
  - `--target_impact_min_pixels`
  - `--target_impact_min_views`
  - `--target_impact_min_policy_samples`
  - `--target_impact_max_extra_bins`
  - `--target_impact_max_views`

The new branch is audited under `target_impact_residual_basis` and records candidate bins, added bins, bins added without policy-val rows, and added target-footprint pixels. It is also embedded into the generated method report lines.

## Safety Boundary

- Uses train/policy-val GT: yes, through the existing residual generator and policy-val evidence.
- Uses target/test GT for selection or apply: no.
- Uses target footprint: yes, after strict target evidence stripping, for visibility and impact ranking only.
- Default behavior changes: none. The branch is off unless explicitly enabled.

## Verification So Far

Static checks passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Runner help exposes the new interface:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py --help | \
  rg -n "target_impact|train_only_target_impact"
```

Strict flowers dry-run passed:

- Dry-run root: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact_dryrun`
- Protocol audit: `passed=True`
- Adapter command includes `--enable_train_only_target_impact_residual_basis`
- Adapter command includes `--target_impact_max_extra_bins 1024`

## Exact Experiment Result

Exact run:

- Scene: `flowers`
- Output root: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact`
- Method: `ours_26000_v165_target_impact_exact_flowers`
- GPU request: `--gpu 6`
- W&B offline dir: `/dev/shm/peilincai_wandb_v165_target_impact_exact`
- Protocol: `--strict_no_target_gt_apply`
- Config basis: v164 exact plus train-only target-impact residual basis.

The exact run completed.

Manifest:

- `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- status: `COMPLETE`
- errors: `[]`
- protocol audit: `passed=true`
- target apply leak: `false`
- target GT visible to apply: `false`
- target GT visible to selection: `false`
- target GT visible to eval: `true`
- target forbidden keys stripped: `true`

Command runtimes:

| command | return code | elapsed |
|---|---:|---:|
| `strip_target_evidence_no_gt` | 0 | 152.590s |
| `apply_certified_residual_texture` | 0 | 5415.726s |
| `populate_eval_gt_from_target_evidence` | 0 | 11.721s |
| `evaluate_vnext_target` | 0 | 43.088s |

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v164 target-connected exact flowers | 20.452796936 | 0.549059153 | 0.355544031 |
| v165 target-impact exact flowers | 20.452848434 | 0.549059093 | 0.355543613 |
| delta | +0.000051498 | -0.000000060 | -0.000000417 |

Footprint:

| version | allowed bins / faces | changed pixels | png changed pixels | changed fraction |
|---|---:|---:|---:|---:|
| v164 | 121 / 13 | 860 | 849 | 0.0000231801 |
| v165 | 1145 / 26 | 8324 | 7896 | 0.0002243617 |

Target-impact residual basis audit:

- candidate bins: `2600`
- added bins: `1024`
- added policy-row bins: `732`
- added no-policy-row bins: `292`
- added target pixels: `9275`
- added target view hits: `2240`
- original allowed bins/faces: `121 / 13`
- final allowed bins/faces: `1145 / 26`
- target footprint covered bins: `2721`
- target footprint candidate faces: `27`
- target views used: `22`

Qualitative outputs:

- renders: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/renders`
- gt: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/gt`

W&B:

- `/dev/shm/peilincai_wandb_v165_target_impact_exact/wandb/offline-run-20260628_153357-ezjo72h3`

## Post-Run Engineering Fixes

The exact run above started before the final verifier integration patch, so its manifest has four commands and does not contain the new explicit `verify_stripped_target_evidence_no_gt` command. After the run, the following engineering fixes were added:

- Added `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`.
- Runner now inserts `verify_stripped_target_evidence_no_gt` after target evidence stripping for future strict runs.
- Runner now rejects target-footprint apply paths unless `--strict_no_target_gt_apply` is set.
- Target-impact footprint stats now use an independent cache controlled by `--target_impact_max_views`.
- Connected-growth footprint stats now use an independent cache controlled by `--target_connected_max_views`.
- Target-impact audit now counts added policy-row samples as well as no-policy-row additions.

Manual verifier audit on the current v165 stripped target evidence passed:

- `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/manual_target_apply_no_gt_verify_after_patch.json`
- `passed=true`
- `bad_view_count=0`
- `view_count=22`
- sample keys: `alpha,barycentric,barycentric_valid,camera_center,depth,face_id,normal,rgb_render,texture`

## Decision Criteria

This branch should only be considered useful if it improves the v164/v106 evidence without introducing hidden target-GT leakage:

- Better or at least non-regressive PSNR/SSIM/LPIPS on flowers against v164 and local clean baseline.
- Audit shows nonzero `target_impact_residual_basis.added_bin_count`.
- Added bins are interpretable: high target footprint, bounded count, and no target/test GT usage.
- Qualitative output shows visible repair area growth, not just metric noise.

If metrics regress, the branch should remain experimental and disabled by default.

## Decision

v165 should remain experimental and disabled by default.

It is a real engineering milestone because it proves target-impact expansion can enlarge the certified footprint by roughly `9.68x` changed pixels while keeping strict no-target-GT apply. It is not a paper-quality quality milestone because the metrics are effectively unchanged:

- PSNR gain vs v164: `+0.000051`
- SSIM delta vs v164: `-0.00000006`
- LPIPS gain vs v164: `-0.00000042`

Interpretation: the current bottleneck is no longer only target footprint. The residual representation written into the enlarged footprint is too weak to create visible or metric-level improvement. The next method change must increase train-only residual capacity under the same verifier/certificate boundary.
