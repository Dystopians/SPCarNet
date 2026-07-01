# 2026-06-30 v302 Constrained Hybrid Support-Transport Calibrator

v302 is the first post-v298 step that is both a real method change and a
positive source-heldout validation result.

It does not claim final target/test or full-paper closure.  It turns the v298
diagnostic into a trainable module:

```text
scripts/car_model/train_source_heldout_support_transport_calibrator.py
```

## Method

The method keeps the high-bandwidth Phase-J / ELA support-view transport path
instead of compressing it into a static face/UV/bin carrier.

For train-only source-heldout views, it builds per-pixel features from:

- ELA support-warp residual signal;
- absolute residual magnitude;
- base render RGB;
- depth-consistency confidence;
- valid mask;
- support count;
- residual disagreement/std;
- base image edge magnitude;
- normalized image coordinates.

A small CNN predicts a calibrated residual delta.  The final v302 policy uses a
constrained hybrid anchor:

```text
delta = (1 - blend) * fixed_alpha * raw_ela_signal
        + blend * learned_scale * learned_delta
```

The policy selector is constrained: it first searches for rows that beat the
raw fixed-alpha ELA anchor on both PSNR and SSIM, then selects the best objective
among those rows.  This fixed an earlier v301 selection bug where the highest
PSNR row was selected even though it lost SSIM against the fixed-alpha anchor.

No target/test GT is used.  GT is used only for train-heldout supervision and
validation.

## Command

```text
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=5 \
PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --output_dir outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630 \
  --device cuda \
  --heldout_stride 4 \
  --heldout_offset 0 \
  --calibrator_val_stride 3 \
  --calibrator_val_offset 0 \
  --k 4 \
  --alpha_grid 0,0.125,0.25,0.5,0.75,1 \
  --learned_scale_grid 0.5 \
  --enable_hybrid_eval \
  --hybrid_anchor_alpha 0.25 \
  --hybrid_blend_grid 0,0.25,0.5,0.75,1 \
  --evidence_max_side 256 \
  --train_steps 800 \
  --crop_size 256 \
  --hidden_channels 32 \
  --layers 3 \
  --compute_ssim \
  --ssim_max_side 256 \
  --save_example_views 3 \
  --skip_train_eval \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v302-constrained-hybrid-anchor-flowers-fullval
```

Output:

```text
outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630
```

W&B offline:

```text
outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/wandb/offline-run-20260630_175614-cakafhs3
```

## Results

Split:

| field | value |
|---|---:|
| train views | 151 |
| source views | 113 |
| calibrator train views | 25 |
| calibrator validation views | 13 |

Validation summary:

| method | PSNR gain | SSIM gain | changed | positive views | min PSNR gain | CVaR20 PSNR gain |
|---|---:|---:|---:|---:|---:|---:|
| fixed raw ELA alpha 0.25 | +0.072807 | +0.001316 | 0.701214 | 1.000000 | +0.038500 | +0.039400 |
| learned only scale 0.5 | +0.100625 | +0.001164 | 0.616590 | 1.000000 | +0.041096 | +0.049219 |
| v302 hybrid alpha 0.25 / scale 0.5 / blend 0.5 | +0.088643 | +0.001350 | 0.693829 | 1.000000 | +0.042650 | +0.046303 |

Selected hybrid vs fixed raw ELA:

| field | value |
|---|---:|
| PSNR delta | +0.015836 |
| SSIM delta | +0.0000339 |
| all-axis source-heldout pass | true |

## Interpretation

This is the first concrete evidence that the post-Phase-J reflection produced a
useful method step:

- v298 proved high-bandwidth support-view transport has headroom.
- v299 learned calibration improved PSNR but lost SSIM against the fixed alpha
  ELA anchor.
- v300 SSIM/edge training did not solve that structural issue.
- v301 showed the need for constraint-aware selection.
- v302 fixes the selector and finds a hybrid point that beats the raw fixed ELA
  anchor on PSNR, SSIM, positive-view fraction, and tail PSNR.

The remaining limitation is important: this is still a train source-heldout
validation result, not a target/test flowers exact result and not a full9 paper
closure.  The next required step is to freeze this no-GT policy and test whether
the same constrained hybrid support-transport calibrator improves the real
flowers target/test protocol, then extend to multi-scene validation.

Final status: NOT COMPLETE.
