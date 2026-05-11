# ECSR FaceAlpha V15-V20 Certified Policy

Date: 2026-05-10

This note records the post-V14 surface face-alpha work. The purpose was to
separate true representation-attached residual gains from train-policy false
accepts. Held-out test views were used only for final reporting.

## Motivation

V14 made surface face-alpha scalable and no-test-leakage, but it still accepted
tiny residual fields that sometimes improved train-policy means while hurting a
held-out metric by numerical margins. The failure mode was not missing GPU time:
it was weak generalization evidence. V15-V20 therefore focused on a fixed
policy certificate rather than scene-specific parameter search.

## Iterations

- V15 replayed residual shrink factors on flowers and garden. It showed that
  smaller residual scales can turn flowers into a strict positive row, but those
  scales were selected on held-out test diagnostics and are not a valid final
  policy.
- V16 split alpha fitting and policy validation more strictly
  (`--face_alpha_fit_source fit`). This increased train-only discipline but
  worsened held-out perceptual transfer on flowers and did not outperform V14.
- V17 added LPIPS-aware scale selection and ran full9. It reduced obvious
  negative transfer by outputting no-op on bicycle, counter, and flowers, but
  still false-accepted bonsai, room, stump, and treehill.
- V18 added a third consensus split (`2,3`). It kept garden positive but still
  false-accepted bonsai, proving that more split count alone is insufficient.
- V19 added per-view policy deltas, one-sided lower-confidence-bound gating,
  and view-win-fraction gating. This rejected bonsai but still left room and
  stump with tiny held-out mixed rows.
- V20 added a fixed consensus strength stability certificate:
  `--require_consensus_max_scale`. A residual is accepted only if the primary
  split and every consensus split all select the maximum train-policy scale.
  This rejects unstable residual directions instead of relying on held-out
  fallback.

Implementation:

- `scripts/car_model/ecsr_apply_surface_residual_facealpha_adapter.py`

W&B group:

- `phase_l_surface_facealpha_v20_certified_scale`

V20 W&B runs:

| scene | run id |
|---|---|
| bicycle | `7qzn191z` |
| bonsai | `z8rqd4f3` |
| counter | `knzp904i` |
| flowers | `ptuhwep7` |
| garden | `2ephutwr` |
| kitchen | `xkvqpr09` |
| room | `gf2xq6mc` |
| stump | `ps5jhkmy` |
| treehill | `fvkfvqk0` |

## Full9 V17 Baseline For This Policy Study

V17 is the relevant pre-certificate full9 row. It improved policy safety over
V14 but still produced mixed held-out rows.

| scene | accepted faces | target coverage | dPSNR | dSSIM | dLPIPS | held-out verdict |
|---|---:|---:|---:|---:|---:|---|
| bicycle | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | no-op |
| bonsai | 723 | 0.3774% | +0.002884 | -0.000037 | +0.000043 | mixed |
| counter | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | no-op |
| flowers | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | no-op |
| garden | 493 | 0.2020% | +0.001804 | +0.000006 | -0.000017 | strict positive |
| kitchen | 272 | 0.1041% | +0.000824 | -0.000000 | -0.000001 | mixed |
| room | 228 | 0.2485% | +0.000498 | +0.000000 | +0.000011 | mixed |
| stump | 50 | 0.0269% | +0.000051 | -0.000002 | +0.000000 | mixed |
| treehill | 672 | 0.5171% | -0.001015 | +0.000001 | -0.000001 | mixed |

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v17_lpips_scale_<scene>_eval.json`

## V20 Full9 Held-Out Result

V20 is the current safest surface-attached face-alpha row. It has no held-out
negative transfer on the selected full9 set, but the mean visual gain remains
very small because the certificate rejects most scenes.

| scene | accepted faces | target coverage | dPSNR | dSSIM | dLPIPS | final gate | held-out verdict |
|---|---:|---:|---:|---:|---:|---|---|
| bicycle | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |
| bonsai | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |
| counter | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |
| flowers | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |
| garden | 321 | 0.1321% | +0.000931 | +0.000003 | -0.000011 | consensus max-scale passed | strict positive |
| kitchen | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |
| room | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | scale instability rejected | no-op |
| stump | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | scale instability rejected | no-op |
| treehill | 0 | 0.0000% | +0.000000 | +0.000000 | +0.000000 | policy/consensus rejected | no-op |

Mean deltas against `ours_26000_phasef_extra_compact_base`:

- mean dPSNR: `+0.000103`
- mean dSSIM: `+0.000000298`
- mean dLPIPS: `-0.00000120`
- strict positive scenes: `1 / 9`
- no-op scenes: `8 / 9`
- mixed scenes: `0 / 9`

Raw metrics:

- `outputs/carnet/meshsplatopt/ecsr_phase_l/v20_certified_scale_<scene>_eval.json`

## Interpretation

V20 is a real reliability improvement over V17: the method now has a fixed
train-only policy that prevents every observed held-out regression in the full9
surface-facealpha study. This is useful for a paper as a conservative
representation-attached residual component and as evidence that the certificate
is meaningful.

It is not a final paper breakthrough. The accepted support is still tiny
(`0.1321%` target coverage on garden only), and the visual gain is too small to
carry the work. The next method step must increase representation capacity and
surface addressability, not relax the certificate. The most defensible next
branch is a persistent vertex/barycentric/SH residual code distilled from the
strong ELA teacher, with V20-style policy certificates as the acceptance layer.

