# SPCarNet v169 to v244 Dense Target-Visible Upgrade Report

Date: 2026-06-29

This report records the work performed after reading
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`. The prompt's
hard rule is used here without relaxation:

- do not promote to full9 before flowers exact beats Phase-J all-axis;
- compare against the Phase-J flowers gate, not only against Phase-F parent;
- target/test RGB GT and target residual GT must be absent during apply;
- if flowers fails, write a diagnostic report and continue from the bottleneck.

## Executive Verdict

Final status for this milestone: **NOT COMPLETE**.

The method change is real: the run family moved from the older Phase-J-plus
adapter path toward Phase-J-to-Phase-F residual baking, added target-visible
surface capacity, soft support fallback, raw teacher residual selection, and a
GT-assisted train-fit ablation. The best current candidate, v244, improves the
Phase-F baked parent strongly on flowers exact:

- PSNR: `+0.134535`
- SSIM: `+0.015082`
- LPIPS: `-0.033077`

However, v244 still does **not** beat Phase-J:

- PSNR: `-0.497377` behind Phase-J;
- SSIM: `-0.030697` behind Phase-J;
- LPIPS: `+0.032205` worse than Phase-J.

Therefore the v169 hard flowers gate is still failed, and full9 remains
blocked by the prompt itself.

## Prompt Gate

The hard prompt threshold is:

| metric | required |
|---|---:|
| PSNR | `> 20.304358` |
| SSIM | `> 0.557770` |
| LPIPS | `< 0.329222` |

The local native1256 reference used in this run is:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-F parent | 19.668695 | 0.511678 | 0.394788 |
| Phase-J reference | 20.300608 | 0.557458 | 0.329505 |

The local Phase-J reference is slightly below the prompt's stricter numbers,
so the fair conclusion is the same under either comparison: current v244 is
not through the gate.

## What Changed In Code

Main implementation file:

```text
scripts/car_model/train_surface_conditioned_residual_unet.py
```

Implemented changes:

- residual-debt train mask support:
  - `--residual_debt_mask`
  - `--residual_debt_quantile`
  - `--residual_debt_min_l1`
  - `--residual_debt_dilate`
  - `--residual_debt_noop_weight`
- target no-GT precheck before target apply, so skipped target apply still
  records the verifier state;
- `--residual_l1_key` override so raw teacher residual selection can be tested
  independently from the residual RGB tensor;
- soft support fallback for `SurfaceTextureConditionedUNet`:
  - `--surface_support_gate_floor`
  - `--surface_support_unknown_gate_floor`
- report fields for the new gates, support floors, residual-debt settings, and
  GT-usage audit.

Runner touched:

```text
scripts/car_model/run_v238_surface_texture_full9_scene.py
```

The runner now contains v241 as a target-visible dense Phase-J-to-Phase-F
distillation variant. The direct v242-v244 experiments below were launched from
the train/eval scripts because they needed faster flowers-only iteration before
any full9 promotion.

## Why This Is A Real Method Change

The earlier v238 milestone was a useful Phase-J-plus adapter, but it did not
strictly answer the v169 research question because it used Phase-J renders as
the parent and learned a residual on top of that endpoint.

The v241-v244 line instead uses:

- parent: Phase-F baked compact render;
- teacher: Phase-J render;
- residual target: Phase-J minus Phase-F parent, or the raw teacher-parent
  residual variant;
- target apply input: Phase-F parent plus target geometry only;
- exact evaluation: applied output vs Phase-J and Phase-F on held-out flowers.

This directly tests whether Phase-J's render-time gain can be baked back into
a surface-conditioned representation.

## Flowers Exact Results

| method | train-fit GT? | face/bin rows | support floors | policy-val gain PSNR/SSIM/LPIPS | target changed | exact PSNR | exact SSIM | exact LPIPS | vs Phase-F parent | vs Phase-J |
|---|---:|---:|---|---|---:|---:|---:|---:|---|---|
| v169 native1256 teacher-only | no | 524,289 | hard gate | `+0.002417 / +0.000117 / +0.000101` | 0.0239 | 19.670961 | 0.511814 | 0.394431 | `+0.002266 / +0.000136 / -0.000357` | `-0.629646 / -0.045644 / +0.064925` |
| v241 dense teacher-only | no | 4,194,305 | `0.18 / 0.06` | `+0.027757 / +0.001739 / +0.000763` | 0.1503 | 19.694145 | 0.513428 | 0.391717 | `+0.025450 / +0.001750 / -0.003071` | `-0.606462 / -0.044030 / +0.062212` |
| v242 raw-dense teacher-only | no | 4,194,305 | `0.18 / 0.06` | `+0.028424 / +0.001064 / +0.000587` | 0.1893 | 19.692448 | 0.512409 | 0.391072 | `+0.023752 / +0.000731 / -0.003716` | `-0.608160 / -0.045049 / +0.061566` |
| v243 raw-dense GT-assisted | yes | 4,194,305 | `0.18 / 0.06` | `+0.067333 / +0.006435 / +0.003905` | 0.3052 | 19.724134 | 0.518789 | 0.377212 | `+0.055439 / +0.007111 / -0.017576` | `-0.576473 / -0.038669 / +0.047706` |
| v244 dense-high GT-assisted | yes | 8,388,609 | `0.25 / 0.08` | `+0.147609 / +0.012008 / +0.008732` | 0.4821 | 19.803230 | 0.526760 | 0.361711 | `+0.134535 / +0.015082 / -0.033077` | `-0.497377 / -0.030697 / +0.032205` |

LPIPS deltas use lower-is-better semantics in the table: negative vs parent is
an improvement; positive vs Phase-J means worse than Phase-J.

## Gap Recovery

Phase-J minus Phase-F parent gap on flowers exact:

| metric | Phase-J gap over parent | v244 recovered | recovery |
|---|---:|---:|---:|
| PSNR | 0.631912 | 0.134535 | 21.29% |
| SSIM | 0.045780 | 0.015082 | 32.94% |
| LPIPS reduction | 0.065282 | 0.033077 | 50.66% |

This is the strongest useful signal from the new line: v244 recovers about
half of the Phase-J perceptual LPIPS gain, but only about one fifth of the
PSNR gap and one third of the SSIM gap. The surface-conditioned residual carrier
is no longer a no-op, but it is still not a Phase-J replacement.

## No-GT Apply Audit

All v241-v244 target/test apply runs passed the no-target-GT check:

| method | target no-GT verifier | target GT visible | target residual visible | train-fit GT usage |
|---|---|---|---|---|
| v241 | pass | false | false | false |
| v242 | pass | false | false | false |
| v243 | pass | false | false | true, train-fit only |
| v244 | pass | false | false | true, train-fit only |

The GT-assisted variants must be claimed as train-fit GT-assisted ablations,
not as pure teacher-only distillation.

## Artifacts

Key reports:

```text
docs/car_model/6-29-v169-Native1256-TeacherResidual-Diagnostic.md
docs/car_model/6-29-v169-native1256-summary.json
docs/car_model/6-29-v169-to-v244-DenseTargetVisible-Upgrade-Report.md
docs/car_model/6-29-v169-to-v244-dense-target-visible-summary.json
```

Experiment roots:

```text
/tmp/peilincai_spcarnet_v241_dense_flowers_20260629/v241_dense_native1256
/tmp/peilincai_spcarnet_v242_raw_dense_flowers_20260629/v242_raw_dense_native1256
/tmp/peilincai_spcarnet_v243_gtassist_dense_flowers_20260629/v243_gtassist_dense_native1256
/tmp/peilincai_spcarnet_v244_gtassist_densehi_flowers_20260629/v244_gtassist_densehi_native1256
```

Exact result JSONs:

```text
/tmp/peilincai_spcarnet_v241_dense_flowers_20260629/v241_dense_native1256_flowers_exact_results.json
/tmp/peilincai_spcarnet_v242_raw_dense_flowers_20260629/v242_raw_dense_native1256_flowers_exact_results.json
/tmp/peilincai_spcarnet_v243_gtassist_dense_flowers_20260629/v243_gtassist_dense_native1256_flowers_exact_results.json
/tmp/peilincai_spcarnet_v244_gtassist_densehi_flowers_20260629/v244_gtassist_densehi_native1256_flowers_exact_results.json
```

W&B offline run roots:

```text
/tmp/peilincai_spcarnet_v241_dense_flowers_20260629/v241_dense_native1256/wandb/offline-run-20260629_161812-dhi6j3pn
/tmp/peilincai_spcarnet_v242_raw_dense_flowers_20260629/v242_raw_dense_native1256/wandb/offline-run-20260629_162545-c08yje3u
/tmp/peilincai_spcarnet_v243_gtassist_dense_flowers_20260629/v243_gtassist_dense_native1256/wandb/offline-run-20260629_163219-hqh7c7to
/tmp/peilincai_spcarnet_v244_gtassist_densehi_flowers_20260629/v244_gtassist_densehi_native1256/wandb/offline-run-20260629_163918-hos9nd60
```

## Commands And Configs

Representative launch environment:

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline TMPDIR=/tmp \
PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_surface_conditioned_residual_unet.py \
  --scene flowers \
  --model_type surface_texture_unet \
  --teacher_evidence_dir <phasej_to_phasef_teacher_evidence> \
  --target_evidence_dir <phasef_target_evidence_no_gt> \
  --target_eval_evidence_dir <phasef_target_evidence_no_gt> \
  --residual_rgb_key <teacher_residual_rgb or teacher_residual_rgb_raw> \
  --residual_l1_key <empty or teacher_parent_delta_l1> \
  --enable_surface_support_gate \
  --surface_target_visible_evidence_dir <target_visible_no_gt> \
  --policy_select_mode tail_guard \
  --wandb_project spcarnet-v169
```

v244 key settings:

```text
steps=2400
train_max_side=768
patch_size=384
learning_rate=0.00018
surface_max_faces=131072
surface_texture_size=8
surface_feature_dim=12
base_channels=32
max_delta=0.12
confidence_bias=-0.8
surface_support_gate_floor=0.25
surface_support_unknown_gate_floor=0.08
teacher_l1/ssim/lpips/grad/highfreq=0.70/0.24/0.15/0.08/0.10
gt_l1/ssim/lpips/grad/highfreq=0.24/0.28/0.18/0.08/0.10
```

Exact evaluation used:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/evaluate_render_split_metrics.py \
  -m <exact_eval_root> \
  --split test \
  --methods <candidate_method> phasej_reference_native1256 phasef_parent_native1256 \
  --output <flowers_exact_results.json> \
  --per_view_output <flowers_exact_per_view.json> \
  --merge_model_results
```

## Lessons Learned

1. The v169 native1256 correction was essential. The older raw-resolution path
   mixed `1600x1054` evidence with a `1256x828` gate, which made earlier
   conclusions unreliable.

2. Dense target-visible capacity helps. Moving from the small v169 carrier to
   v241 increased target changed fraction from about `0.024` to `0.150` and
   converted a near-no-op into a measurable exact improvement.

3. Raw teacher residual selection alone is not enough. v242 slightly improved
   LPIPS over v241, but it hurt PSNR/SSIM and did not materially change the
   Phase-J gap.

4. Train-fit GT assistance is the first change that produced a large exact
   LPIPS gain. This improves practical quality, but it changes the claim from
   teacher-only distillation to train-fit GT-assisted recovery.

5. v244 is still capacity-limited and/or representation-limited. It changes
   almost half the target pixels but still cannot reproduce Phase-J's structure
   and detail. The bottleneck is no longer only coverage; it is the missing
   view-dependent multi-source signal and stronger geometry/structure coupling.

6. Full9 is not justified yet. The prompt explicitly requires flowers exact
   all-axis success before full9. Running full9 now would generate expensive
   evidence for a method that has not passed the first required gate.

## Next Required Step

The next useful change should not be another support floor or alpha variant.
Implement v245 as a **view-conditioned multi-source residual predictor**:

- aggregate nearest train-view teacher residual evidence for target-visible
  faces without target/test RGB GT;
- feed surface texture, face/UV support, view direction, normal/depth, and
  source-view agreement into the decoder;
- train with teacher residual plus train-fit GT-assisted structural losses;
- add an exact-gap recovery score to policy-val so the run measures how much
  of the Phase-J gap is actually recovered;
- rerun flowers native1256 exact against Phase-J before any full9 launch.

Exact continuation prompt:

```text
Continue from docs/car_model/6-29-v169-to-v244-DenseTargetVisible-Upgrade-Report.md. Implement v245 with a real view-conditioned multi-source residual predictor: combine surface texture with nearest train-view evidence aggregation over target-visible faces, train with teacher+train GT, add policy-val exact-gap recovery metric, then rerun flowers native1256 exact against Phase-J. Do not run full9 until flowers exact beats Phase-J all-axis.
```
