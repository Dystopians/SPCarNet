# Candidate-Aware ELA Multiscene Validation Log

Date: 2026-05-21 PDT / 2026-05-22 UTC

This log records the completed fair multiscene validation pass for symmetric
candidate-aware ELA. It is deliberately conservative: selection uses train-val
metrics only, held-out test metrics are report-only, and the current result is
not promoted to a paper-level endpoint because the measured gains remain
near-noise and visually subtle.

## Protocol

Both arms receive the same train-only per-model ELA auto-policy:

- Phase-J compact baseline: selected `ratio_0200/compact_model`.
- Phase-S candidate: face-local SH1 residual edit with train-only carrier
  selection and compact topology budget.
- ELA policy source: `per_model_auto`.
- Policy objective: `balanced`.
- Policy holdout: `policy_holdout_fraction=0.25`, `policy_holdout_offset=0`.
- Calibration: `--calib_lpips`, `--calib_sampler uniform`,
  `--calib_max_views 16`, `--calib_stride 6`.
- Alpha grid: `0,0.0625,0.125,0.25,0.5`.
- Selection: train-val gate only. Test metrics remain report-only.

Two variants were run:

- `edge`: enables ELA edge gating and acts as a conservative safety ablation.
- `plain`: disables ELA edge gating and preserves stronger absolute quality.

## Code Interfaces Landed

- `caec279 Add symmetric candidate-aware ELA ablation`
  - Adds `--ela_policy_source per_model_auto` for symmetric baseline/candidate
    train-only ELA auto-policy.
- `f091673 Fix per-model auto ELA support split flag`
  - Adds the missing top-level `--support_policy_fit_only` runner flag.
- `4377fac Allow Phase-K summaries outside repo output root`
  - Allows large `/home/...` output roots while still writing valid summaries.
- `1b15645 Skip completed Phase-K scenes on resume`
  - Prevents reruns from recomputing completed scene summaries by default.
- `7dbeb5f Add train-val Phase-K policy portfolio selector`
  - Adds fixed train-val-only portfolio selection across edge/plain variants.
- `e07ba5f Add Phase-K train artifact cleanup option`
  - Adds `--cleanup_train_artifacts_after_scene` for future long runs.

## Output Roots

| variant | GPU | output root | W&B group |
|---|---:|---|---|
| edge | 7 | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522` | `candidate_aware_ela_multiscene_edge_20260522` |
| plain | 1 | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522` | `candidate_aware_ela_multiscene_plain_20260522` |
| baseline render rebuild for qualitative only | 7 | `/home/peilincai/spcarnet_runs/baseline_render_cache/candidate_aware_ela_multiscene_plain_20260522` | `candidate_aware_ela_multiscene_plain_20260522_rebuild_baseline_test` |

Generated summaries:

- Edge summary:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/edge_phasek_summary.md`
- Plain summary:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522/plain_phasek_summary.md`
- Edge/plain portfolio:
  `/home/peilincai/spcarnet_runs/phasek_policy_portfolio_20260522/portfolio_counter_bonsai_room_flowers.md`
- Plain qualitative:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/patchcert_qualitative_contact_sheet.png`

## Completed Results

### Edge Variant

The edge variant passes the strict compact gate on all four scenes, but its
absolute test quality is lower than the plain variant and the gains are tiny.

| scene | accepted | faces | train-val balanced | test balanced | test dPSNR | test dSSIM | test dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | true | 16 | +0.000070632 | +0.000039995 | +0.000043869 | +0.000000000 | +0.000000194 | passes; tiny gain |
| bonsai | true | 45 | +0.000172615 | +0.000320673 | +0.000352859 | +0.000001192 | +0.000002801 | passes; PSNR/SSIM up, LPIPS worse |
| room | true | 16 | +0.000025511 | +0.000005901 | +0.000003815 | +0.000000119 | +0.000000015 | passes; near no-op |
| flowers | true | 16 | +0.000021458 | -0.000019431 | +0.000003815 | -0.000000656 | +0.000000507 | train-val pass but report-only test balanced regresses |

Mean effective deltas because all scenes are accepted:
`+0.000101089` PSNR, `+0.000000164` SSIM, `+0.000000879` LPIPS.

### Plain Variant

The plain variant preserves stronger absolute quality and is selected by the
portfolio, but the strict per-scene compact gate accepts only `room` and
`flowers`.

| scene | accepted | faces | train-val balanced | test balanced | test dPSNR | test dSSIM | test dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | false | 16 | +0.000021696 | +0.000013053 | +0.000022888 | +0.000000000 | +0.000000492 | standard gain positive; strict compact stratified PSNR fails |
| bonsai | false | 45 | +0.000068009 | +0.000162065 | +0.000185013 | -0.000000715 | +0.000000432 | positive PSNR; rejected by strict LPIPS tail |
| room | true | 16 | +0.000033438 | +0.000002742 | +0.000005722 | +0.000000000 | +0.000000149 | accepted; near no-op |
| flowers | true | 16 | +0.000039816 | -0.000021338 | +0.000001907 | -0.000000358 | +0.000000805 | train-val pass but report-only test balanced regresses |

Strict gate effective deltas:
`+0.000001907` PSNR, `-0.000000089` SSIM, `+0.000000238` LPIPS.

### Train-Val Portfolio

The fixed portfolio consumes edge/plain decision roots and selects using only
train-val absolute score, train-val gains, compact geometry limits, and
train-val tail risk. Test metrics are report-only. It selects plain candidate
for all four scenes:

| scene | selected variant | selected kind | report dPSNR | report dSSIM | report dLPIPS | train-val score |
|---|---|---|---:|---:|---:|---:|
| counter | plain | candidate | +0.000022888 | +0.000000000 | +0.000000492 | 43.166845240 |
| bonsai | plain | candidate | +0.000185013 | -0.000000715 | +0.000000432 | 46.177148328 |
| room | plain | candidate | +0.000005722 | +0.000000000 | +0.000000149 | 46.935930737 |
| flowers | plain | candidate | +0.000001907 | -0.000000358 | +0.000000805 | 27.586228162 |

Portfolio mean report-only effective deltas:
`+0.000053883` PSNR, `-0.000000268` SSIM, `+0.000000469` LPIPS.

## Qualitative Evidence

Plain qualitative panels were generated after rebuilding only the missing
baseline test render cache:

- Summary:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/qualitative_summary.md`
- Contact sheet:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/patchcert_qualitative_contact_sheet.png`
- Manifest:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522_qualitative/qualitative_manifest.json`

The per-view panels choose the best report-only views for visualization only.
The strongest rows are still modest: `bonsai` has individual views up to about
`+0.0037` PSNR, while most accepted `room/flowers` views are around `1e-4` PSNR
or smaller. This supports the current conclusion that the branch is useful for
fairness/automation diagnosis, not yet for visually obvious paper figures.

## Exact Summary Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_collect_phasek_barycentric_gate_summary.py \
  --decision_root /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/decisions \
  --scenes counter,bonsai,room,flowers \
  --output_json /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/edge_phasek_summary.json \
  --output_md /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/edge_phasek_summary.md

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_collect_phasek_barycentric_gate_summary.py \
  --decision_root /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522/decisions \
  --scenes counter,bonsai,room,flowers \
  --output_json /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522/plain_phasek_summary.json \
  --output_md /home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522/plain_phasek_summary.md

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_select_phasek_policy_portfolio.py \
  --variant edge=/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522 \
  --variant plain=/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522 \
  --scenes counter,bonsai,room,flowers \
  --output_json /home/peilincai/spcarnet_runs/phasek_policy_portfolio_20260522/portfolio_counter_bonsai_room_flowers.json \
  --output_md /home/peilincai/spcarnet_runs/phasek_policy_portfolio_20260522/portfolio_counter_bonsai_room_flowers.md \
  --require_tail
```

## Interpretation

This run fixed a fairness flaw: candidate-aware ELA is now symmetric instead of
candidate-only. It also converts the edge/plain choice into a deterministic
train-val portfolio instead of a manual parameter pick. That is a real
methodological improvement.

It does not solve the main scientific weakness. The selected portfolio is only
`+5.4e-5` PSNR on average and slightly worse on mean SSIM/LPIPS. The qualitative
contact sheet is correspondingly subtle. The branch should remain logged as a
fairness and policy automation improvement, not as a top-conference-strength
representation-level breakthrough.

## Next Gate

The next method step should not be another plain/edge parameter sweep. The hard
blocker is effect size: the current face-local residual edit can pass train-only
safety checks, but the materialized representation change is too weak to create
visible held-out improvement. A credible next step needs a larger
representation change while keeping the same train-val-only promotion rule, for
example a multi-face coherent residual field with local support evaluation or a
geometry-aware carrier that changes more than tiny SH coefficients.
