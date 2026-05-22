# Candidate-Aware ELA Multiscene Validation Log

Date: 2026-05-21 PDT / 2026-05-22 UTC

This log records the current fair multiscene validation pass for symmetric
candidate-aware ELA. It is intentionally conservative: completed evidence is
reported as evidence, live jobs are marked live, and the method is not promoted
to the main paper claim unless it survives multiple scenes with clear margins.

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

Two variants are running:

- `edge`: enables ELA edge gating and currently acts as a conservative
  acceptance ablation.
- `plain`: disables ELA edge gating and tests whether the edge gate is hurting
  absolute quality.

## Code Fixes Landed

- `f091673 Fix per-model auto ELA support split flag`
  - Added the missing top-level `--support_policy_fit_only` runner flag.
  - Without this, the fair per-model-auto path crashed when entering train
    support-only ELA.
- `4377fac Allow Phase-K summaries outside repo output root`
  - Added robust summary path handling for `/home/...` output roots.
  - This allows large render/evaluation artifacts to live outside the repo while
    keeping summary JSON generation valid.

Both commits were pushed to `https://github.com/Dystopians/SPCarNet.git` main.

## Running Commands

Common runner:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence \
  --scenes counter,bonsai,room,flowers \
  --iteration 26000 \
  --ela_policy_source per_model_auto \
  --ela_policy_objective balanced \
  --ela_calib_lpips \
  --ela_alpha_grid 0,0.0625,0.125,0.25,0.5 \
  --policy_holdout_fraction 0.25 \
  --policy_holdout_offset 0 \
  --calib_sampler uniform \
  --calib_max_views 16 \
  --calib_stride 6 \
  --gate_compact_enable \
  --gate_compact_require
```

Variant-specific output roots and W&B groups:

| variant | GPU | output root | W&B group |
|---|---:|---|---|
| edge | 7 | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522` | `candidate_aware_ela_multiscene_edge_20260522` |
| plain | 1 | `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_plain_20260522` | `candidate_aware_ela_multiscene_plain_20260522` |

To avoid filling `/data`, Phase-J render directories for these method names are
symlinked into `/home/peilincai/spcarnet_runs/baseline_render_cache/...`.

## Completed Results

### Edge Variant

| scene | accepted | faces | train-val balanced | test balanced | test dPSNR | test dSSIM | test dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | true | 16 | +0.000070632 | +0.000039995 | +0.000043869 | +0.000000000 | +0.000000194 | strict compact gate passes, but gain is tiny |
| bonsai | true | 45 | +0.000172615 | +0.000320673 | +0.000352859 | +0.000001192 | +0.000002801 | passes; PSNR/SSIM up, LPIPS slightly worse |

Evidence:

- Counter summary:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/counter/phasek_scene_summary.json`
- Bonsai summary:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522/bonsai/phasek_scene_summary.json`
- Qualitative contact sheet:
  `/home/peilincai/spcarnet_runs/candidate_aware_ela_multiscene_edge_20260522_qualitative/patchcert_qualitative_contact_sheet.png`

### Plain Variant

| scene | accepted | faces | train-val balanced | test balanced | test dPSNR | test dSSIM | test dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| counter | false | 16 | +0.000021696 | +0.000013053 | +0.000022888 | +0.000000000 | +0.000000492 | higher absolute quality than edge, but compact stratified PSNR gate fails |

Counter absolute quality comparison:

| variant | baseline PSNR | candidate PSNR | baseline SSIM | candidate SSIM | baseline LPIPS | candidate LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| edge | 27.846403 | 27.846447 | 0.884926 | 0.884926 | 0.212002 | 0.212002 |
| plain | 27.973318 | 27.973341 | 0.888341 | 0.888341 | 0.206064 | 0.206064 |

This is the key current lesson: edge gating can make the strict compact gate
pass, but it also lowers the absolute ELA baseline/candidate quality. Plain ELA
has stronger absolute image metrics but did not pass the current compact
stratified gate on `counter`.

## Live Status

As of this log:

- `edge`: `counter` and `bonsai` complete; `room` is running; `flowers` pending.
- `plain`: `counter` complete; `bonsai` is running; `room` and `flowers`
  pending.

## Interim Interpretation

This is real method evidence but not a paper-level closed loop yet. The
candidate edit is non-noop and can pass strict compact gates on at least two
scenes under the conservative edge variant. However, all observed gains are
small, and the better absolute-quality plain ELA row currently fails the strict
compact gate on `counter`. The next decision should be based on the multiscene
edge/plain split:

- If edge keeps accepting but plain consistently gives higher absolute quality,
  edge should remain a safety ablation, not the default method.
- If plain fails only by small compact-tail margins while improving absolute
  metrics, the next implementation target should be a train-only tail-robust
  compact gate that does not reward lower-quality ELA.
- If neither variant produces visible or metric-substantial gains on `room` and
  `flowers`, this branch should be marked diagnostic rather than paper-facing.

