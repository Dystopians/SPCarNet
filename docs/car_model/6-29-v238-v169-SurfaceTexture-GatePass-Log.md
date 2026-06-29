# v169 -> v238 Surface Texture Gate-Pass Log

Date: 2026-06-29

## Verdict

v238 is the first run in this branch that satisfies the v169 flowers exact all-axis gate against the local Phase-J reference and the prompt hard thresholds.

This is a stage milestone, not final paper readiness. The fixed policy still needs full9, cross-scene ablations, qualitative panels, and a stricter paper-claim review before claiming the whole project is closed.

## What Changed

The previous v237 route used native-1256 Phase-J-anchored evidence, but its geometry fields were deterministic constants. It was a useful no-GT image-space residual adapter, not a convincing baked surface representation.

v238 changes the carrier back to a real surface-attached representation:

- `ecsr_rebase_evidence_rgb_render_from_renders.py` now resizes real dense geometry when rebasing to native render resolution:
  - continuous fields use bilinear resize: `alpha`, `depth`, `normal`, `barycentric`, `texture`;
  - discrete fields use nearest resize: `face_id`, `barycentric_valid`;
  - audit records `geometry_resized_view_count`.
- `train_surface_conditioned_residual_unet.py` now lets surface face selection/support stats compute residual L1 from `teacher_residual_rgb` when `teacher_residual_l1` is absent.
- v238 uses `model_type=surface_texture_unet` with `--enable_surface_support_gate`, so target changes are restricted to supported selected surface bins instead of being pure image postprocessing.

## Storage And Compute Preflight

- `/data`: about `203M` free during the milestone; too low for large artifacts.
- `/tmp`: about `6.0T` free; used for new v238/v239 evidence and outputs.
- `/dev/shm`: about `1.7G` free; not used for new large artifacts.
- GPU choice: GPU2/GPU3 were low occupancy; v238 used GPU2, v239 used GPU3.
- W&B mode: offline.

## Evidence Inputs

Real-geometry native-1256 evidence was generated under:

- train/fit evidence: `/tmp/peilincai_spcarnet_v238_surface_native1256_inputs/flowers/teacher_surface_evidence_phasej_native1256_surfacegeom`
- target no-GT evidence: `/tmp/peilincai_spcarnet_v238_surface_native1256_inputs/flowers/target_evidence_no_gt_phasej_native1256_surfacegeom`
- train audit: `/tmp/peilincai_spcarnet_v238_surface_native1256_inputs/flowers/teacher_surface_evidence_phasej_native1256_surfacegeom_audit.json`
- target audit: `/tmp/peilincai_spcarnet_v238_surface_native1256_inputs/flowers/target_evidence_no_gt_phasej_native1256_surfacegeom_audit.json`
- no-GT verifier: `/tmp/peilincai_spcarnet_v238_surface_native1256_inputs/flowers/target_phasej_native1256_surfacegeom_no_gt_verify.json`

Audit facts:

- train views rewritten: `46`
- target views rewritten: `22`
- geometry resized views: `46` train, `22` target
- target GT visible to apply: `false`
- target residual visible to apply: `false`
- sample target keys only include geometry and parent render: `alpha`, `barycentric`, `barycentric_valid`, `camera_center`, `depth`, `face_id`, `normal`, `rgb_render`, `texture`

## v238 Configuration

Output:

- `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256`

Key settings:

- `model_type=surface_texture_unet`
- `surface_texture_size=8`
- `surface_feature_dim=8`
- `surface_face_max_unique=8192`
- `enable_surface_support_gate=true`
- `lowrank_min_bin_support=8`
- `alpha_grid=0,0.25,0.5,0.75,1`
- selected alpha: `0.25`
- W&B offline run: `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256/wandb/offline-run-20260629_131642-b6eyujm6`

Surface capacity:

- selected faces: `8192`
- surface texture rows: `524289`
- estimated surface texture parameters: `4194312`
- active support rows: `95524`
- active row fraction: `0.182198`
- target known face fraction: `0.112010`
- target active support fraction: `0.062922`
- target changed fraction: `0.052635`
- inactive support changed fraction: `0.0`

## Policy-Val Certificate

Report:

- `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256/v238_surface_texture_unet_native1256_report.json`

Best all-axis row:

| alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR min gain | SSIM min gain | LPIPS min gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.007123 | +0.000738 | +0.000521 | +0.001065 | +0.000153 | -0.000460 |

LPIPS positive-view fraction is only `0.6667`, so the LPIPS tail remains a weakness even though the mean gate passed.

## Flowers Exact Result

Exact result path:

- `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256/v238_flowers_native1256_exact_results.json`
- per-view: `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256/v238_flowers_native1256_exact_per_view.json`
- renders: `/tmp/peilincai_spcarnet_v238_surface_texture_unet_native1256/flowers_exact_target_apply/test/ours_26000_v238_surface_texture_unet_native1256_flowers/renders`
- qualitative panel: `assets/spcarnet_v238_phasej_flowers_native1256_panel.jpg`

Same local evaluator, same GT symlink, same split:

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| Phase-J reference native1256 | 20.300608 | 0.557458 | 0.329505 | baseline |
| v238 surface texture U-Net | 20.306461 | 0.558319 | 0.328470 | PASS |

Prompt hard gate:

| threshold | required | v238 | pass |
|---|---:|---:|---|
| PSNR | > 20.304358 | 20.306461 | yes |
| SSIM | > 0.557770 | 0.558319 | yes |
| LPIPS | < 0.329222 | 0.328470 | yes |

## Ablation / Negative Evidence

v237 image-space native1256 adapter:

- exact: `20.297960 / 0.557542 / 0.326311`
- better LPIPS, but PSNR below Phase-J and below the prompt threshold.
- not a real surface-baked representation because geometry was constant.

v239 strict low-rank surface texture:

- output: `/tmp/peilincai_spcarnet_v239_lowrank_surface_texture_native1256`
- policy-val selected no-op, `policy_val_all_axis_pass=false`
- target apply skipped.

Interpretation: a strict low-rank per-surface-bin residual basis is currently underpowered. v238 succeeds because it combines a surface neural texture with local U-Net context while hard-gating changes to supported surface bins.

## Current Claim Boundary

Defensible now:

- v238 is a real method change relative to v237: it uses nonconstant native-resolution face/UV geometry and a learned surface texture carrier.
- v238 passes the flowers exact all-axis gate against Phase-J under the native-1256 local evaluator.
- target/test GT was not visible during apply.

Not defensible yet:

- full paper readiness;
- full9 superiority;
- robust cross-scene superiority;
- large perceptual margin;
- compression/triangle-count superiority.

## Next Required Step

Freeze v238 as the current flowers-gate-passing policy, then run fixed-policy full9 and ablations:

1. parent only / Phase-J reference;
2. v237 image-space adapter;
3. v238 surface texture U-Net;
4. v239 strict low-rank surface texture negative ablation;
5. qualitative panels on views where v238 improves LPIPS/SSIM visibly.

If full9 exposes a failure, the next method target should be LPIPS-tail stabilization, not another alpha scan.
