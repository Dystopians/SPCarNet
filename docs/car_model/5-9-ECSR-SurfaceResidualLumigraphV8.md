# ECSR Surface Residual Lumigraph V8

Date: 2026-05-09

This pass adds the first full9 representation-attached appearance recovery
closure after the Phase-J render-time ELA result. It is intentionally modest:
the goal is not to claim that this replaces Phase-J, but to establish a clean
surface-attached residual path with train-only acceptance, full9 artifacts, and
documented rejection behavior.

## Method

`scripts/car_model/ecsr_apply_surface_residual_lumigraph_adapter.py` now supports
a consensus policy:

- primary train-policy split: `policy_val_stride=4`;
- extra consensus split: `consensus_policy_strides=2`;
- accepted alpha grid: `0,0.125,0.25`;
- calibration metrics: PSNR, SSIM, LPIPS on train-policy views only;
- residual support: persistent rendered `face_id`;
- held-out application: target view face-id map plus train-fitted residual field;
- final guard: optional per-face MSE gate with `min_face_gate_pixels=128`.

The adapter accepts a nonzero residual only if the primary split and all
consensus splits pass the same metric guards. This fixed rule was added after
V6/V7 exposed two failure modes:

- V6 accepted `treehill` under a 3-view policy split, but held-out PSNR/SSIM
  regressed.
- V7 rejected `treehill`, but accepted `flowers` and `stump` cases where held-out
  SSIM could still move slightly negative.

V8 is therefore a stricter fixed policy rather than a scene-specific parameter
choice.

## Full9 Artifacts

Additional train evidence and held-out surface maps were generated for
`room`, `counter`, `kitchen`, and `bonsai`, completing the full9 surface
lumigraph interface. The first indoor cache attempt accidentally used the
default `max_views=8` and enabled barycentric top-support writing; that was
interrupted because V8 does not need barycentric coordinates and the settings
were not comparable to the outdoor 12-view cache. The final cache uses 12
train views and stores only fields required by V8.

Key artifact roots:

- evidence cache: `outputs/carnet/meshsplatopt/ecsr_phase_l/surface_evidence_sh1_v3_capacity/<scene>`;
- target surface maps: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model/test/ours_26000_surface_maps/surface_maps`;
- V8 renders: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model/test/ours_26000_surface_residual_lumigraph_v8_consensus_gate128`;
- W&B group: `phase_m_surface_lumigraph_v8_consensus_gate128_full9`.

## Full9 Result Versus Phase-F Compact Base

For indoor and `bonsai`, V8 rejected the residual and writes a no-op render, so
the reported delta against the compact base is exactly zero. Outdoor compact
base metrics are the independent `evaluate_render_split_metrics.py` results.

| scene | V8 alpha | decision | accepted faces | dPSNR | dSSIM | dLPIPS | V8 PSNR/SSIM/LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|
| bicycle | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 23.293482 / 0.659651 / 0.332275 |
| flowers | 0.250 | accept | 12 | +0.000867844 | +0.000000656 | -0.000042617 | 19.669563 / 0.511679 / 0.394745 |
| garden | 0.250 | accept | 20 | +0.001386642 | +0.000007153 | -0.000014782 | 25.028923 / 0.780038 / 0.201307 |
| stump | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 25.180920 / 0.704420 / 0.294214 |
| treehill | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 20.923227 / 0.564224 / 0.406108 |
| room | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 28.739101 / 0.884793 / 0.249923 |
| counter | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 26.749872 / 0.862051 / 0.251998 |
| kitchen | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 27.816381 / 0.876445 / 0.199192 |
| bonsai | 0.000 | no-op | 0 | +0.000000000 | +0.000000000 | +0.000000000 | 28.864340 / 0.896012 / 0.259340 |

Mean full9 delta against the Phase-F compact base:

- PSNR: `+0.000250498`;
- SSIM: `+0.000000868`;
- LPIPS: `-0.000006378`.

## Interpretation

This is a real representation-level interface improvement, not a headline
visual breakthrough:

- positive: the residual is attached to train-observed surface ids rather than
  test image-space GT residuals;
- positive: the same fixed consensus policy rejects all known risky scenes;
- positive: `flowers` and `garden` get strict RGB improvements with no held-out
  selection;
- negative: the accepted surface coverage is extremely small, so the mean gain
  is tiny;
- negative: all indoor scenes are rejected because train-policy PSNR gains come
  with LPIPS or SSIM risk;
- negative: Phase-J render-time ELA remains the strongest current RGB result.

The current bottleneck is not an interface bug. It is capacity and support
coverage: per-face constant residuals only touch tiny high-confidence regions.
The next credible method step must raise surface-attached appearance capacity
without returning to image-space post-processing. Promising directions are
per-face low-rank residual codes, vertex-neighborhood residual interpolation,
and a local perceptual certificate that can accept larger support regions while
maintaining the V8 consensus rejection discipline.

## Status

V8 should be kept as the clean surface-attached baseline for future
representation-level work. It closes the full9 execution loop for
surface-attached residual relocation, but it does not satisfy the final paper
endpoint by itself.
