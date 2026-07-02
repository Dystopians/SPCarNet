# v346-v348 Representation Transport Log

Date: 2026-07-02

Status: **NOT COMPLETE**. Phase-J remains stronger than the v346-v348 line.

## Context

The previous selector-style line produced small local gains but did not close
the older Phase-J representation gap. v346-v348 therefore moved back into the
train/eval pipeline and tested whether stronger source-heldout supervision,
source-baked texture anchors, and locally continuous residual transport could
create a real learned representation upgrade.

Authoritative summary files:

```text
docs/car_model/results/v348_directional_smooth_transport_flowers_summary.json
docs/car_model/results/v348_directional_smooth_transport_flowers_summary.md
```

## Implemented Interfaces

File:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

New default-off train/eval interfaces:

- `--source_heldout_image_loss_weight`
- `--source_heldout_image_loss_every`
- `--source_heldout_image_loss_stride`
- `--texture_anchor_scale`
- `--texture_anchor_reliability_power`
- `--texture_anchor_floor`
- `--texture_anchor_use_holdout_confidence`
- `--apply_delta_smooth_radius`
- `--apply_delta_smooth_iterations`

New diagnostics are reported during policy validation:

- `mean_abs_raw_pred`
- `mean_abs_view_gated_pred`
- `mean_abs_applied_delta`
- `mean_pred_confidence`
- `mean_view_support_gate`
- smoothing support and magnitude summaries

These interfaces are default-off, so old commands keep their previous behavior.

## v346: Source-Heldout Image Proxy

Goal: make the learned decoder respect heldout-source image structure rather
than fitting only sparse residual samples.

Result on flowers policy-val:

| run | steps | all-axis | Phase-J gate | PSNR gain | SSIM gain | LPIPS gain |
|---|---:|---|---|---:|---:|---:|
| v346a no patch | 24 | false | false | +0.000008111843 | -0.000000576178 | +0.000000209237 |
| v346b patch proxy | 24 | false | false | +0.000001931315 | -0.000000144045 | +0.000000253941 |

Conclusion: the source-heldout image proxy executes and logs correctly, but it
does not fix the learned residual direction.

## v347: Texture Anchor and Smooth Transport

Goal: inject the source-baked face/UV residual mean as a calibrated anchor, then
spread sparse residual predictions through support-normalized smoothing before
the usual policy validation gate.

Representative results:

| run | steps | faces | smooth | all-axis | PSNR gain | SSIM gain | LPIPS gain |
|---|---:|---:|---|---|---:|---:|---:|
| v347e 128 no anchor | 48 | 128 | off | false | +0.000060685368 | -0.000001177192 | -0.000000903383 |
| v347f 128 anchor | 48 | 128 | off | false | +0.000039510675 | -0.000000908971 | +0.000000117347 |
| v347h no anchor smooth r1 | 0 | 128 | r1 | false | +0.000075002801 | -0.000001231829 | -0.000000332172 |
| v347j anchor smooth r2 | 0 | 128 | r2 | false | +0.000051259577 | -0.000001142422 | +0.000001520539 |

Conclusion: larger support and smoothing can increase coverage and sometimes
LPIPS, but SSIM remains systematically negative. The anchor raises raw residual
amplitude but can make the selected safe policy collapse to alpha zero.

## v348: 600-Step Directional Smooth Transport

Goal: test whether longer training and stronger direction-oriented supervision
solve the v347 bottleneck.

| run | steps | faces | smooth | all-axis | PSNR gain | SSIM gain | LPIPS gain | raw pred | applied delta |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| v348a no anchor | 600 | 128 | r2 | false | +0.000052119297 | -0.000000819564 | +0.000000728294 | 0.017696081101 | 0.000002566493 |
| v348b anchor | 600 | 128 | r2 | false | +0.000057815692 | -0.000000899037 | +0.000001224379 | 0.017998826943 | 0.000002612664 |

Conclusion: longer training increases raw prediction magnitude, but it still
does not pass all-axis policy validation or Phase-J gating. This is evidence
against the hypothesis that the failure is merely undertraining.

## Phase-J Boundary

Current Phase-J reference:

```text
method: ours_26000_phasej_guarded_adaptedge_ela
strict RGB scene wins vs clean: 9/9
strict RGB per-view wins vs clean: 244/246
mean gain vs clean: +1.331084 PSNR / +0.034702 SSIM / -0.063359 LPIPS
mean triangle reduction: 7.6479%
```

The v346-v348 line is far below this boundary. It should not be presented as a
successor to Phase-J.

## Teacher-Residual Oracle

The important positive result is the oracle test on the same 128 candidate
faces:

| oracle setting | PSNR gain | SSIM gain | LPIPS gain | changed fraction | positive view fractions |
|---|---:|---:|---:|---:|---|
| radius2 alpha0.125 | +0.001035985790 | +0.000022078554 | +0.000013890366 | 0.001296894766 | 1.0/1.0/0.6667 |
| radius2 alpha1.0 | +0.006106957967 | +0.000113919377 | +0.000171336035 | 0.001671262255 | 1.0/1.0/0.75 |
| radius3 alpha1.0 | +0.006463811090 | +0.000087459882 | +0.000240119795 | 0.002275853890 | 1.0/1.0/0.8333 |

Interpretation: the support set and apply path have real positive headroom. The
current learned decoder cannot reproduce the teacher residual direction and
structure reliably enough.

## Current Bottleneck

The failure mode is now sharper:

1. Sparse per-face/UV residual means are too low-bandwidth for structure-safe
   recovery.
2. L1-style source-heldout training learns conservative average colors rather
   than the view-conditioned residual direction that improves SSIM.
3. Smoothing helps coverage but also spreads direction errors, so it does not
   turn weak residual predictions into a safe structural correction.
4. Anchoring to source-baked means increases amplitude but does not solve
   target-view compatibility.

## Next Required Method Change

The next milestone should not be another alpha/grid tweak. It should replace
the residual generator with a higher-bandwidth transport representation:

- retain support-view residual patches or local feature descriptors instead of
  only per-bin mean residuals;
- condition prediction on source/target view geometry and neighbor evidence;
- train with an explicit structural direction objective that optimizes local
  gradient/SSIM-compatible residuals;
- keep Phase-J or v345e-style strict gates only as the final safety layer, not
  as the main source of gains.

Final status: **NOT COMPLETE**.
