# v83 PatchMix + FaceAlpha + LocalPatch Hybrid Log

Date: `2026-06-24`

Status: `COUNTER_PROBE_COMPLETED_NOT_PROMOTED`

## Purpose

v82 patch-mixture teacher basis proved that a richer residual basis can be wired
into the train/eval pipeline, but that branch fell back to legacy teacher basis
and underperformed the strong counter anchor. v83 tested a more aggressive
combination:

```text
patch-mixture teacher basis
+ face-alpha calibration
+ local-patch multiscale prior
+ policy-val bin-gain hybrid
+ support-capacity expansion
```

The goal was to check whether higher residual capacity can break the
v56/v64/v79 counter plateau without manually choosing held-out parameters.

## Command Evidence

The run was executed with W&B logging under:

```text
WANDB group: v83_teacher_patchmix
run name:    v83_teacher_patchmix_counter_20260624
```

The main applied method name is:

```text
ours_26000_counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter
```

The full command is preserved in:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/logs/apply_metrics_counter.log
```

## Held-Out Counter Result

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v56/v64/v79 counter anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v80 near-tie hybrid | `26.756135941` | `0.862126231` | `0.251691461` |
| v82 capacity-prerank | `26.756137848` | `0.862126350` | `0.251690656` |
| v83 patchmix+facealpha+localpatch | `26.756147385` | `0.862125337` | `0.251688808` |

Delta vs v56/v64/v79 anchor:

| dPSNR | dSSIM | dLPIPS |
|---:|---:|---:|
| `+0.000017166` | `-0.000000894` | `-0.000002563` |

Delta vs v82 capacity-prerank:

| dPSNR | dSSIM | dLPIPS |
|---:|---:|---:|
| `+0.000009537` | `-0.000001013` | `-0.000001848` |

## Train/Policy-Val Audit

| audit field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| target changed fraction | `0.0639413869` |
| policy-val SSIM gain | `0.0002925346` |
| policy-val SSIM positive view fraction | `1.0` |
| policy-val image-L1 gain | `0.0000265453` |
| policy-val image-L1 positive view fraction | `0.9166666667` |
| selected support mode | `fit_residual_topk` |
| selected added faces | `4096` |
| selected texture size | `32` |

## Interpretation

v83 is a meaningful diagnostic because it beats the counter anchor on PSNR and
LPIPS and also improves those two metrics over v82 capacity-prerank. It does not
strictly promote because SSIM is slightly lower than both anchors.

Safe reporting language:

```text
v83 shows that richer patch-mixture residual capacity can improve PSNR/LPIPS,
but the current certificate is still not SSIM-safe enough for strict promotion.
```

Do not present v83 as a full9 result or a paper headline. It should only be used
as evidence that the next representation-level step should focus on stronger
SSIM-preserving target-view certificates, not manual alpha/support tuning.

## Persistent Evidence

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
```
