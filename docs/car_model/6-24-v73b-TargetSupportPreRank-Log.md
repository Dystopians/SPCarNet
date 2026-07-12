# v73b Target-Support Pre-Rank Log

Date: 2026-06-24  
Status: completed diagnostic, not promoted.

## Purpose

v73 attached target-support profiles after expensive candidate fitting. That proved the ranking signal, but it did not reduce runtime. v73b moves a cheap target geometry support proxy before policy-val/refit so weak support sets can be pruned early.

## Code Changes

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
  - added `evaluate_target_face_support_proxy`;
  - added `--target_support_prerank_top_k`;
  - added `--target_support_prerank_max_views`;
  - records `target_support_prerank` in adapter audit.
- `scripts/car_model/run_l1risk_fairnoop_scene.py`
  - exposes and forwards both pre-rank CLI flags;
  - logs pre-rank fields to W&B.

Static validation passed:

```text
py_compile adapter and runner
adapter help exposes target-support pre-rank flags
runner help exposes target-support pre-rank flags
```

## Experiment

Scene: `counter`  
W&B run: `qn1ntfyy`  
W&B URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/qn1ntfyy`

Persistent root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73b_target_support_prerank_20260624
```

Main audit:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v73b_target_support_prerank_20260624/counter_v73b_targetsupport_prerank_top1_countpyramid_blendladder_support4096_tex16_nearest_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```

## Metrics

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| v73b target-support pre-rank | `26.753995895` | `0.862119257` | `0.251853049` | ties v73/v70/v71a, below v64/v56 |
| selected v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | current selected reference |

## Candidate Pre-Rank Audit

| support mode | added faces | retained | coverage fraction | CVaR20 coverage | min-view coverage |
|---|---:|---|---:|---:|---:|
| `fit_residual_topk` | `4096` | yes | `0.070668033` | `0.030667436` | `0.029021694` |
| `base_carrier` | `0` | no | `0.022178402` | `0.003666513` | `0.000802620` |

Final selected candidate:

- support mode: `fit_residual_topk`
- support added faces: `4096`
- texture size: `16`
- fill mode: `nearest_observed`
- selected alpha: `0.125`
- selected count-pyramid blend: `0.0`
- target changed fraction: `0.065630289`

## Interpretation

v73b is a real train/eval pipeline change and it successfully prunes support candidates before expensive fitting. It should be kept as an engineering closure for target-support-aware candidate search.

It is not a promoted method endpoint because it does not improve metrics over v73/v70/v71a and remains below the selected v64/v56 counter reference. This reinforces the current diagnosis: target support and candidate pruning are necessary, but the residual atlas representation still lacks enough capacity to turn larger target footprint into better held-out RGB metrics.

## W&B Note

The run was launched with `WANDB_MODE=online`. Final sync hit API timeout / per-run file-rate-limit retries, then completed successfully. Run URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/qn1ntfyy`.
