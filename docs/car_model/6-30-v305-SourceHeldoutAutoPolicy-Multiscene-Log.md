# v305 Source-Heldout Auto Policy Multiscene Log

Date: 2026-06-30

## Direct Reflection Answer

The reflection became useful only after it changed the method. Earlier
iterations overfit to parameter scans and single-scene evidence. The effective
reflection was:

1. keep the target/test GT out of policy decisions;
2. preserve the strong online support-transport signal instead of baking it into
   a weak surface carrier;
3. train a small support-transport calibrator only on train source-heldout
   supervision;
4. add a train-only source-heldout output guard so fixed, learned, and hybrid
   branches are selected by evidence rather than by manual scene tuning.

This is not yet a paper-complete claim, but it is a real method step beyond
parameter search.

## Implemented Method Change

Main files:

```text
scripts/car_model/train_source_heldout_support_transport_calibrator.py
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
docs/car_model/results/v304_frozen_hybrid_policy_multiscene_summary.json
docs/car_model/results/v305_sourceheldout_auto_policy_multiscene_summary.json
```

The v302 calibrator learns a constrained residual transport correction from
source-heldout train views. For each target view, the support evidence path
warps residuals from train-source views and exposes image-space features:
raw signal, absolute signal, base RGB, confidence, validity, support count,
residual variance, base edges, and coordinates. The calibrator predicts a
bounded residual correction.

The v305 update adds `--output_variant source_heldout_auto`. It evaluates the
pre-registered candidates on train source-heldout validation only:

- fixed raw ELA: `0.25 * signal`;
- learned: `0.5 * calibrator(signal, features)`;
- hybrid: `0.5 * fixed + 0.5 * learned`.

The selector chooses learned or hybrid only if it beats fixed on source-heldout
PSNR and SSIM. Otherwise it falls back to fixed. Test GT is read only after the
selected renders are written for reporting.

## Key Commands

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile scripts/car_model/apply_source_heldout_support_transport_calibrator.py
git diff --check -- scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Representative v305 command:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/garden \
  --target_split test \
  --support_source_mode source_split \
  --heldout_stride 4 \
  --heldout_offset 0 \
  --device cuda \
  --k 4 \
  --anchor_alpha 0.25 \
  --learned_scale 0.5 \
  --blend 0.5 \
  --output_variant source_heldout_auto \
  --selector_val_stride 3 \
  --selector_val_offset 0 \
  --evidence_max_side 256 \
  --compute_ssim \
  --ssim_max_side 256 \
  --save_example_views 1 \
  --copy_gt \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics
```

## v304 Frozen Hybrid Evidence

v304 freezes the v302 hybrid policy on all nine ratio-0200 scenes. It is useful
as the failure that motivated v305.

| scene | hybrid PSNR gain | hybrid SSIM gain | hybrid-fixed PSNR | hybrid-fixed SSIM | all-axis vs fixed |
|---|---:|---:|---:|---:|:---:|
| bicycle | +0.112088 | +0.002953 | +0.013743 | +0.000133 | yes |
| bonsai | +0.531387 | +0.005616 | +0.044990 | +0.000317 | yes |
| counter | +0.386257 | +0.006588 | +0.046626 | +0.000471 | yes |
| flowers | +0.088861 | +0.004048 | +0.010637 | +0.000253 | yes |
| garden | +0.140931 | +0.001909 | +0.007750 | +0.000078 | yes |
| kitchen | +0.446521 | +0.003702 | +0.053460 | +0.000298 | yes |
| room | +0.421838 | +0.004990 | +0.028147 | +0.000321 | yes |
| stump | +0.062020 | +0.001134 | +0.004990 | -0.000074 | no |
| treehill | +0.107090 | +0.001691 | +0.016333 | +0.000098 | yes |

Macro: `+0.255222 PSNR`, `+0.003626 SSIM`, positive versus base on `9/9`
scenes, all-axis versus fixed on `8/9` scenes.

The stump failure is the important lesson: fixed hybrid improves PSNR but can
slightly hurt SSIM versus fixed raw ELA.

## v305 Auto Policy Evidence

v305 keeps one fixed meta-policy and lets train source-heldout evidence select
the output branch.

| scene | selected | selected PSNR gain | selected SSIM gain | selected-fixed PSNR | selected-fixed SSIM | safe vs fixed |
|---|---|---:|---:|---:|---:|:---:|
| bicycle | hybrid | +0.112088 | +0.002953 | +0.013743 | +0.000133 | yes |
| bonsai | learned | +0.567712 | +0.005785 | +0.081315 | +0.000486 | yes |
| counter | learned | +0.426360 | +0.006908 | +0.086728 | +0.000792 | yes |
| flowers | hybrid | +0.088861 | +0.004048 | +0.010637 | +0.000253 | yes |
| garden | hybrid | +0.140931 | +0.001909 | +0.007750 | +0.000078 | yes |
| kitchen | learned | +0.493623 | +0.003911 | +0.100562 | +0.000508 | yes |
| room | hybrid | +0.421838 | +0.004990 | +0.028147 | +0.000321 | yes |
| stump | fixed | +0.057030 | +0.001208 | +0.000000 | +0.000000 | yes |
| treehill | fixed | +0.090757 | +0.001593 | +0.000000 | +0.000000 | yes |

Macro over 9 scenes / 246 test views:

```text
selected PSNR gain:              +0.266578
selected SSIM gain:              +0.003701
selected minus fixed PSNR gain:  +0.036542
selected minus fixed SSIM gain:  +0.000286
safe vs fixed scene rate:        9/9
positive vs base scene rate:     9/9
mean positive-view fraction:     0.954228
mean covered fraction:           0.948203
```

This means v305 fixes the v304 stump all-axis failure and improves macro PSNR
over frozen hybrid by about `+0.011356`.

## Artifacts

```text
outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630
outputs/carnet/spcarnet_v303_v302_policy_flowers_test_apply_20260630
outputs/carnet/spcarnet_v304_v302_policy_multiscene_test_apply_20260630
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630
docs/car_model/results/v304_frozen_hybrid_policy_multiscene_summary.json
docs/car_model/results/v305_sourceheldout_auto_policy_multiscene_summary.json
```

Representative W&B offline runs:

```text
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/bicycle/wandb/offline-run-20260630_181246-sdk01kpj
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/bonsai/wandb/offline-run-20260630_181401-vs44urq1
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/counter/wandb/offline-run-20260630_181345-cznxesyt
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/garden/wandb/offline-run-20260630_181520-zpejkqtp
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/kitchen/wandb/offline-run-20260630_181610-r722879e
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/stump/wandb/offline-run-20260630_181128-1s4icxll
outputs/carnet/spcarnet_v305_sourceheldout_auto_policy_test_apply_20260630/treehill/wandb/offline-run-20260630_181710-83pz4f3d
```

## Remaining Weaknesses

v305 is a strong engineering and method milestone, but not complete:

- it is still evaluated on the ratio-0200 compact-model branch, not a fresh
  clean long MeshSplatting training rerun;
- it improves scene-level averages, but some individual test views still have
  negative PSNR gain: bicycle, stump, and treehill have negative tail views;
- selected output can fall back to fixed, so the learned module is not active on
  every scene;
- no LPIPS/DISTS/geometric triangle-accounting table is included in this v305
  pass;
- qualitative panels need to be rebuilt from v305 selected renders.

The next method step should be a target-GT-free per-view risk gate. It should
learn from source-heldout train views when to apply the selected correction and
when to no-op a single target view, with tail PSNR/SSIM as the first criterion.

Current verdict:

```text
Final status: NOT COMPLETE.
```
