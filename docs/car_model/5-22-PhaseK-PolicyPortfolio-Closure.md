# Phase-K Candidate-Aware ELA Policy Portfolio Closure

Date: 2026-05-21 PDT / 2026-05-22 UTC

## Executive Status

Final status for this milestone: `NOT COMPLETE` as a paper-level closed loop.

The engineering checklist is mostly satisfied: the train/eval pipeline has a
real method change, edge/plain ablations were run on four scenes with W&B
online, quantitative summaries and qualitative panels were saved, and the
selection policy now uses train-val evidence only. The scientific target is not
satisfied: the best fixed policy gives only noise-scale test gains and no
visually obvious improvement.

## What Changed In The Method

The earlier problem was asymmetric adaptation. The candidate checkpoint could
look different depending on which Phase-J ELA policy it inherited, so a
candidate-only recalibration would be unfair. The new pipeline tests a cleaner
question:

> If both the Phase-J fallback and the Phase-S representation edit receive the
> same train-only per-model ELA auto-policy, does the materialized Phase-S edit
> still help?

The implementation adds three relevant pieces:

1. Symmetric per-model ELA:
   - runner flag: `--ela_policy_source per_model_auto`;
   - both baseline and candidate use the same alpha grid, objective, holdout
     split, LPIPS calibration, and train-only support policy.
2. Safer compact carrier selection:
   - runner forwards
     `--delta_patch_cert_carrier_holdout_auto_prefix_positive_tail_safe`;
   - the operator can stop prefix growth when the next carrier is not
     individually positive/tail-safe.
3. Fixed policy portfolio:
   - script: `scripts/car_model/ecsr_select_phasek_policy_portfolio.py`;
   - chooses among edge/plain variants using train-val absolute score, gains,
     compact geometry, and tail risk;
   - held-out test metrics are report-only.

## Compared Rows

| row | purpose | output root |
|---|---|---|
| Phase-J per-model-auto baseline | fair fallback baseline | policy-root compact model plus `/home/peilincai/spcarnet_runs/baseline_render_cache/candidate_aware_ela_multiscene_plain_20260522` |
| Phase-S plain candidate | best absolute-quality candidate-aware variant | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522` |
| Phase-S edge candidate | conservative safety ablation | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522` |
| Train-val portfolio | fixed policy over edge/plain | `/home/peilincai/spcarnet_runs/phasek_policy_portfolio_20260522/portfolio_counter_bonsai_room_flowers.md` |

The four-scene validation set is `counter,bonsai,room,flowers`, all at
iteration `26000`, using online W&B logging.

## Quantitative Result

| scene | selected variant | report dPSNR | report dSSIM | report dLPIPS | interpretation |
|---|---|---:|---:|---:|---|
| counter | plain | +0.000022888 | +0.000000000 | +0.000000492 | positive PSNR, LPIPS slightly worse |
| bonsai | plain | +0.000185013 | -0.000000715 | +0.000000432 | best PSNR gain, but SSIM/LPIPS mixed |
| room | plain | +0.000005722 | +0.000000000 | +0.000000149 | near no-op |
| flowers | plain | +0.000001907 | -0.000000358 | +0.000000805 | report-only balanced regression |

Portfolio mean report-only effective delta:

```text
dPSNR  = +0.000053883
dSSIM  = -0.000000268
dLPIPS = +0.000000469
```

Strict compact per-scene gate:

- edge accepts `4 / 4`, but includes `flowers` report-only regression and lower
  absolute quality;
- plain accepts `2 / 4` under strict per-scene compact gate, but the portfolio
  selects plain candidate on `4 / 4` because fixed train-val portfolio scoring
  prioritizes absolute quality and tail-limited validity.

## Qualitative Result

Generated artifacts:

- `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/qualitative_summary.md`
- `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/qualitative_manifest.json`
- `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/patchcert_qualitative_contact_sheet.png`

The panels are useful for audit because they show GT, Phase-J, candidate,
absolute error, and amplified delta. They are not strong paper figures. The
best single-view `bonsai` rows reach roughly `+0.0037` PSNR, but the changes are
hard to see without amplified residual maps. This matches the aggregate metrics.

## Review

What is solid:

- No held-out test metric is used for policy selection.
- Baseline and candidate now receive symmetric train-only ELA recalibration.
- The edge/plain distinction is represented as a fixed train-val portfolio, not
  manual per-scene tuning.
- The candidate edit is non-noop and the operator audit is present on all four
  scenes.
- Quantitative and qualitative artifacts are saved with exact paths.

What is weak:

- The effect size is far below what would support a top-conference main claim.
- Mean LPIPS and SSIM are slightly worse under the selected portfolio.
- Edge gating can pass stricter gates but at the cost of lower absolute quality.
- Plain candidate produces better absolute quality but still has very small
  deltas and report-only `flowers` regression.
- The qualitative difference is not compelling in full-frame panels.

## Stop/Go Recommendation

Do not claim that this branch has solved the representation-level method gap.
The correct claim is narrower:

> Candidate-aware ELA plus train-val portfolio selection fixes an evaluation and
> policy fairness issue, but it exposes that the current Phase-S face-local
> residual edit is too weak to produce visible held-out improvements.

The next command should pursue a larger representation move, not another
parameter scan. If continuing this exact branch, use this continuation prompt:

```text
Continue from docs/car_model/5-22-PhaseK-PolicyPortfolio-Closure.md.
Do not tune edge/plain parameters. Implement a stronger representation-level
candidate that changes coherent multi-face support while preserving train-val
only selection, then run the same four-scene fair Phase-K protocol plus
qualitative panels. Stop only if the new method gives non-noise gains and no
mean SSIM/LPIPS regression versus the symmetric Phase-J baseline.
```
