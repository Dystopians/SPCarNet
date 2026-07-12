# v87/v88 Representation-Level Counter Diagnostics

Date: 2026-06-25

This log tracks the v87/v88 representation-level counter probes that were launched after the v84/v86 anchor. The goal is to decide whether either probe deserves hard-triad/full9 expansion. The answer is strict: a counter probe must beat the v84/v86 anchor on PSNR, SSIM, and LPIPS before expansion.

Anchor:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor | `26.7561378479` | `0.8621263504` | `0.2516906559` |

Promotion rule:

```text
PSNR > 26.7561378479
SSIM > 0.8621263504
LPIPS < 0.2516906559
accepted_atlas
target changed fraction >= 0.001
policy-val SSIM/L1 audit no weaker than the anchor line
```

---

## v87 Source-Mixture Counter

Archived evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v87_source_mixture_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v87_source_mixture_counter_20260625/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v87_source_mixture_counter_20260625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v87_source_mixture_counter_20260625/apply_metrics_counter.log
```

Counter result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor | `26.7561378479` | `0.8621263504` | `0.2516906559` |
| v87 source-mixture counter | `26.7561302185` | `0.8621262312` | `0.2516913712` |
| delta vs anchor | `-0.0000076294` | `-0.0000001192` | `+0.0000007153` |

Policy audit:

| field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| target changed fraction | `0.0639013177` |
| selected hybrid | `true` |
| selected source mixture | `none` |
| policy-val SSIM gain | `0.0002937565` |
| policy-val image-L1 gain | `0.0000267522` |
| policy-val image-L1 min-view gain | `-0.0000008121` |
| policy-val image-L1 CVaR20 gain | `0.0000026754` |

Verdict:

```text
Do not promote v87 to hard-triad or full9.
```

Interpretation:

- The source-mixture machinery did not become the selected mechanism; the final selected branch is still a prior-bin hybrid.
- The run accepts a non-empty edit, but all three held-out metrics are worse than the v84/v86 counter anchor.
- This reinforces the current bottleneck: policy-val accepted edits can remain below the already-strong representation anchor, so new candidates need a stricter anchor-dominance certificate before expensive expansion.

---

## v88 Anchor-Dominance Tail-Risk Counter

Archived evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/topology_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v88_anchor_dominance_tailrisk_counter_20260625/apply_metrics_counter.log
```

Counter result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor | `26.7561378479` | `0.8621263504` | `0.2516906559` |
| v88 anchor-dominance tail-risk counter | `26.7561569214` | `0.8621254563` | `0.2516880333` |
| delta vs anchor | `+0.0000190735` | `-0.0000008941` | `-0.0000026226` |

Policy audit:

| field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| target changed fraction | `0.0642177280` |
| selected hybrid | `true` |
| selected source mixture | `hybrid baseline=1/6, source=3/6` |
| policy-val SSIM gain | `0.0002933294` |
| policy-val image-L1 gain | `0.0000266956` |
| policy-val image-L1 min-view gain | `-0.0000008643` |
| policy-val image-L1 CVaR20 gain | `0.0000025630` |

Verdict:

```text
Do not promote v88 to hard-triad or full9.
```

Interpretation:

- v88 is a useful diagnostic because it improves held-out PSNR and LPIPS over the v84/v86 counter anchor.
- It still fails the strict promotion gate because SSIM is lower by `8.94e-7`.
- The policy-val tail-risk audit passed, but the held-out SSIM delta shows that policy-val dominance is still not sufficient to certify representation-level promotion.
- This run should be reported as a bottleneck/lesson, not as a headline method.
