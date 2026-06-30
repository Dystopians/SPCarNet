# v259 Target-Support / OOD-Aware Gain Log

Date: 2026-06-29

This log continues the v169 rule from `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`: flowers exact must beat Phase-J all-axis before full9 promotion.

Phase-J flowers reference:

- PSNR `20.304358`
- SSIM `0.557770`
- LPIPS `0.329222`

## Status

`NOT COMPLETE`.

v259 is a real method upgrade over v258: it adds target-support / OOD-aware shrink for boosted surface residuals.  It improves target tail behavior relative to the most aggressive v258a and gives the best target SSIM mean in this deferred-source line, but it still does not beat Phase-J PSNR and does not fully repair SSIM/LPIPS tail risk.

No full9 run was launched.

## Method Change

Implemented in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.

v258 introduced `policy_gain`, which recovers teacher residual energy in policy-val reliable bins.  v259 adds:

- `policy_tail_risk`: learned from policy-val positive fraction, negative gain magnitude, and gain variance per face/UV bin;
- `--ood_gain_mode boosted_soft`: a target-time confidence term applied only to boosted residuals;
- OOD features computed without target/test GT:
  - source camera view gap;
  - source residual variance ratio;
  - parent RGB mismatch;
  - effective source count concentration;
  - policy-val tail risk.

The shrink is:

```text
confidence *= exp(-beta * max(policy_gain - 1, 0) * (0.5 + policy_tail_risk) * ood_score)
```

This keeps the v258 energy-recovery mechanism but prevents high-gain residuals from being applied as strongly when source support looks out-of-distribution.

## Commands / Artifacts

Full command lines are embedded in each audit file.

| run | audit JSON | audit Markdown | target renders |
|---|---|---|---|
| v259a OOD beta 1 | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259a_ood_gain2_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259a_ood_gain2_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259a_ood_gain2_targetexact/target_exact_fixed_policy` |
| v259b OOD beta 2 | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259b_ood_gain2_beta2_targetexact/v253_deferred_source_renderer_audit.json` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259b_ood_gain2_beta2_targetexact/v253_deferred_source_renderer_audit.md` | `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v259b_ood_gain2_beta2_targetexact/target_exact_fixed_policy` |

Machine summary:

- `docs/car_model/results/v259_ood_gain_summary.json`

W&B was run offline under each output directory's `wandb/` subdirectory.

## Policy-Val Results

| run | alpha | PSNR gain | SSIM gain | LPIPS gain | PSNR tail | SSIM tail | LPIPS tail | active teacher energy | mean OOD conf. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v258a gain max 2.0 | 1.0 | +0.030404 | +0.000942 | +0.000450 | +0.022339 | +0.000562 | +0.000089 | 0.467043 | n/a |
| v258b gain max 1.5 | 1.0 | +0.026003 | +0.000812 | +0.000392 | +0.019119 | +0.000525 | +0.000127 | 0.281217 | n/a |
| v258c gain max 1.5 + source agreement | 1.0 | +0.022564 | +0.000696 | +0.000363 | +0.016288 | +0.000455 | +0.000100 | 0.241122 | n/a |
| v259a OOD beta 1 | 1.0 | +0.023506 | +0.000739 | +0.000368 | +0.016505 | +0.000492 | +0.000099 | 0.267199 | 0.883841 |
| v259b OOD beta 2 | 1.0 | +0.018684 | +0.000589 | +0.000307 | +0.012555 | +0.000418 | +0.000096 | 0.173487 | 0.815009 |

Policy-val stayed all-axis positive after OOD shrink.  Stronger beta reduces residual energy and mean gain, as expected.

## Target Exact Results

| run | PSNR | SSIM | LPIPS | PSNR gain | SSIM gain | LPIPS gain | PSNR tail | SSIM tail | LPIPS tail | pos. PSNR / SSIM / LPIPS | Phase-J PSNR gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| v258a gain max 2.0 | 19.838304 | 0.620019 | 0.180196 | +0.006250 | +0.000108 | +0.000139 | -0.002007 | -0.000258 | -0.000380 | 0.864 / 0.636 / 0.636 | -0.466054 |
| v258b gain max 1.5 | 19.838286 | 0.620047 | 0.180217 | +0.006232 | +0.000137 | +0.000118 | -0.000816 | -0.000151 | -0.000245 | 0.864 / 0.682 / 0.727 | -0.466072 |
| v258c gain max 1.5 + source agreement | 19.837588 | 0.620037 | 0.180235 | +0.005534 | +0.000126 | +0.000100 | -0.000674 | -0.000137 | -0.000203 | 0.909 / 0.682 / 0.773 | -0.466770 |
| v259a OOD beta 1 | 19.838006 | 0.620050 | 0.180238 | +0.005952 | +0.000139 | +0.000097 | -0.000483 | -0.000164 | -0.000233 | 0.909 / 0.773 / 0.682 | -0.466352 |
| v259b OOD beta 2 | 19.837280 | 0.620046 | 0.180256 | +0.005226 | +0.000135 | +0.000079 | +0.000040 | -0.000116 | -0.000148 | 0.955 / 0.818 / 0.682 | -0.467078 |

## Interpretation

v259 confirms the diagnosis from v258:

- v258a is the best mean PSNR/LPIPS in this line, but its target tails are risky.
- v259a gives the best target SSIM mean and improves the PSNR tail vs v258a.
- v259b makes the PSNR tail CVaR positive for the first time in this line, but the stronger shrink lowers mean PSNR/LPIPS and still leaves SSIM/LPIPS tail CVaR negative.

The result is useful but not a paper-final breakthrough.  The bottleneck has moved from "residual energy too low" to "energy vs tail-risk tradeoff still not learned well enough."  A hand-crafted OOD score helps but is not sufficient to close the Phase-J PSNR gap.

## No-GT Apply Audit

All v259 target apply used:

- no-GT apply evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_no_gt`
- eval-only GT evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/target_evidence_reparented`

The script's forbidden-key verifier ran before preview/exact apply.  Target GT was used only after apply for metrics.

## Next Step

Do not run full9 yet.

The next useful jump is no longer another fixed OOD beta.  The method needs a learned OOD/gain predictor or a stronger residual carrier:

1. train a tiny feature decoder or logistic gain head on policy-val OOD features and target-free source support features;
2. optimize the head for all-axis policy-val mean and tails;
3. freeze it before target apply;
4. only then re-run flowers exact.

Full9 remains blocked until flowers exact beats Phase-J all-axis.
