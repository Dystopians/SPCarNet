# v73 Target-Support Candidate Selection Log

Date: 2026-06-24  
Status: completed diagnostic, not promoted.

## Purpose

v72 showed that a candidate can have positive policy-val evidence while barely affecting target held-out views. v73 adds a target-support profile to candidate selection so accepted candidates can be ranked not only by train-policy gain but also by target-view footprint.

This is still a train/eval pipeline method change. It does not use target GT for selection; it uses target geometry/evidence coverage and predicted delta support.

## Code Changes

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
  - added target-support profile evaluation for candidates;
  - added `--enable_target_support_candidate_selection`;
  - attached target changed fraction, min-view changed fraction, CVaR20 changed fraction, and valid fraction to candidate score ordering;
  - added a `target_support_lexicographic` guard suffix for auditable selection.
- `scripts/car_model/run_l1risk_fairnoop_scene.py`
  - exposed `--enable_target_support_candidate_selection`;
  - logged target-support fields to W&B.

Static validation passed before the run:

```text
py_compile adapter and runner
adapter help exposes --enable_target_support_candidate_selection
runner help exposes --enable_target_support_candidate_selection
```

## Experiment

Scene: `counter`  
W&B run: `kgfav7cf`  
W&B URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/kgfav7cf`

Persistent root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73_target_support_selection_20260624
```

Main audit:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73_target_support_selection_20260624/counter_v73_targetsupport_countpyramid_blendladder_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```

## Metrics

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v73 target-support selection | `26.753995895` | `0.862119257` | `0.251853049` | ties v70/v71a, below v64/v56 |
| v70/v71a zero-blend reference | `26.753995895` | `0.862119257` | `0.251853049` | safe fallback |
| selected v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | current selected reference |

## Candidate Selection Audit

Selected candidate:

- support mode: `fit_residual_topk`
- support added faces: `4096`
- texture size: `16`
- fill mode: `nearest_observed`
- selected alpha: `0.125`
- count-pyramid prior selected blend: `0.0`
- accepted candidate count: `2`
- guard: `zero_blend_or_base_face_mean_nonregressive_relative_ssim_l1_cvar_min_view_target_support_lexicographic`

Target-support profile:

| field | value |
|---|---:|
| target changed fraction | `0.065630289` |
| target valid fraction | `0.065630289` |
| min-view changed fraction | `0.023086760` |
| CVaR20 changed fraction | `0.027341737` |
| mean absolute delta | `0.000075108` |
| active mean absolute delta | `0.001144406` |

Score order:

| support mode | blend | accepted | policy-val relative gain | image-L1 positive views | target changed fraction |
|---|---:|---|---:|---:|---:|
| `fit_residual_topk` | `0.0` | true | `0.026849788` | `0.916667` | `0.065630289` |
| `fit_residual_topk` | `1.0` | true | `0.025892815` | `1.000000` | `0.065630289` |
| `base_carrier` | `0.0` | false | `0.024443546` | `0.833333` | `0.0` |
| `base_carrier` | `1.0` | false | `0.023780780` | `0.833333` | `0.0` |

## Interpretation

v73 fixes one real weakness in v72: candidate selection now sees whether an accepted residual atlas actually covers target-view pixels. It therefore chooses the expanded-support candidate instead of a base candidate with no target-support profile.

However, this does not solve the main metric bottleneck. The safe candidate still selects `blend=0.0`, which exactly recovers v70/v71a and remains below the v64/v56 selected reference on `counter`.

The conclusion is not "target support is useless." The conclusion is narrower:

```text
target support is necessary for candidate selection,
but the current residual atlas / count-pyramid prior capacity is still too weak
to turn larger target footprint into better held-out RGB metrics.
```

## Runtime Note

The current implementation attaches target-support profiles after building expensive policy candidates. This validates the ranking interface but does not reduce runtime. The next engineering step should add a two-stage selector:

1. cheap target-support pre-profile and support-set pruning;
2. expensive policy-val/refit only for top-K candidates.

## Promotion Decision

Do not promote v73. Keep Phase-J as the presentation-safe endpoint and v64 as the best fixed representation-level policy. v73 should be cited as an honest diagnostic that clarifies the next bottleneck: representation capacity and support-aware candidate design, not another scalar blend or alpha sweep.
