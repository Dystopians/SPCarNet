# Phase-S Risk-Tail / Alpha Module Log

Date: 2026-05-14

This log records the latest Phase-S module state after the full eight
candidate-scene replay. It is intentionally evidence-first: Phase-J remains the
main strong endpoint, while Phase-S risk-tail is a real but still sparse
representation-level repair branch.

## Current Status

```text
Clean MeshSplatting
  -> Phase-F compact ladder
  -> Phase-J guarded adaptive-edge ELA replay
  -> optional Phase-S face-local risk-tail repair
```

Phase-S now has a fixed train/eval path with no held-out test selection:

- full eight candidate-scene risk-tail replay completed;
- `3 / 8` candidate-bearing scenes are promoted;
- mean effective report-only delta over the eight candidate scenes is
  `+0.000684500 PSNR`, `+0.000058956 SSIM`, `-0.000073545 LPIPS`;
- the mean is dominated by `flowers`, where the selected risk-tail repair gives
  `+0.005418777 PSNR`, `+0.000470877 SSIM`, `-0.000586182 LPIPS`;
- `garden`, `bicycle`, `room`, `kitchen`, and `bonsai` fall back to Phase-J;
- per-face alpha refit is implemented in the pipeline, but the first pilot does
  not improve beyond the risk-tail selector and is not promoted as the current
  main method.

## Module Details

| module | role | implementation | current evidence |
|---|---|---|---|
| Phase-F compact ladder | Provides fixed compact checkpoints and compression ratios. | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix` | Source for Phase-J and Phase-S replay. |
| Phase-J guarded ELA | Current strong baseline-over-MeshSplatting endpoint. | `scripts/car_model/ecsr_run_phasej_guarded_adaptedge...` plus Phase-F summaries | `9 / 9` strict RGB wins vs selected clean MeshSplatting; mean triangle reduction `7.6479%`. |
| Surface evidence cache | Train-only residual/evidence maps used by Phase-S. | `outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16` | Provides face residuals, view support, and certificates; test data is not used for selection. |
| Face-local SH residual planner | Writes bounded face-local residual candidates without applying them. | `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py` | Produces candidates for 8/9 scenes under the fixed plan. |
| Coupled selector | Runs real render-gated trials for fixed face sets and falls back to Phase-J if unsafe. | `scripts/car_model/ecsr_run_facelocal_coupled_selector.py` | Full8 risk-tail replay accepts `3 / 8` scenes. |
| Risk-greedy set builder | Penalizes redundant view support and similar residual directions. | `riskN` mode in `ecsr_run_facelocal_coupled_selector.py` | Promotes `flowers` and `counter`; avoids several risky low-amplitude edits. |
| Tail-stable promotion | Allows lower mean train-val score only when per-view tails are stable. | `--selector_enable_tail_stable_promotion`; default `--selector_tail_min_trainval_balanced_delta 1.8e-5` | Keeps `flowers/treehill`, rejects the exposed `garden` false-positive. |
| Per-face alpha refit | Fits train-only scalar alpha per selected face and passes JSON to materializer. | `scripts/car_model/ecsr_fit_facelocal_plan_alphas.py`; materializer arg `--materialize_plan_alpha_json` | Interface complete, but first pilot alphas are nearly `1.0`; not promoted. |
| Stump relaxed discovery | Loosens candidate discovery for the former zero-candidate scene. | `facelocal_zero_candidate_relaxed_v1_*` outputs | Finds 2 candidates, but selector rejects all trials; fallback remains Phase-J. |

## Risk-Tail Selection Rule

Each scene reads the same train-only candidate plan:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json
```

The fixed full8 trial set is:

```text
top1x2,risk4x1,risk8x0.5
```

`riskN` starts from the train certificate score and greedily adds faces with a
pairwise redundancy penalty:

```text
pair_risk(i, j) =
  view_support_overlap(i, j)
  * (0.5 + 0.5 * abs(cosine(delta_coeff_i, delta_coeff_j)))

adjusted_score_i =
  train_certificate_score_i
  * max(0.05, 1 - lambda_pair * max_j pair_risk(i, j))
  * (1 + 0.05 * new_supported_view_count_i)
```

The inner Phase-K gate still controls materialization. The outer selector then
promotes only train-val-safe rows. Held-out test deltas are copied only after
selection as report-only evidence.

## New Quantitative Evidence

### Risk-Tail Full8

Collected summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_summary/summary_8candidate_tailstable.md`

| scene | candidates | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | note |
|---|---:|---|---:|---:|---:|---:|---|
| bicycle | 7 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val too small or negative |
| flowers | 35 | risk4/s1 | true | +0.005418777 | +0.000470877 | -0.000586182 | strong positive Phase-S case |
| garden | 110 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | tail threshold rejects exposed false-positive |
| treehill | 84 | top1/s2 | true | +0.000001907 | +0.000000358 | -0.000000477 | tiny but safe positive |
| room | 76 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best trial below threshold |
| counter | 127 | risk4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | all-metric fix over top1 |
| kitchen | 145 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best trial below threshold |
| bonsai | 1266 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val negative |
| **mean** | - | - | **3/8** | **+0.000684500** | **+0.000058956** | **-0.000073545** | sparse but real improvement |

### Alpha Refit Pilot

Collected summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_alpha_refit_v1_pilot_20260514_summary/summary_3scene.md`

| scene | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | interpretation |
|---|---|---:|---:|---:|---:|---|
| counter | risk4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | same as uniform risk4; alpha did not add gain |
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | still rejected after stricter tail threshold |
| bicycle | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | still too small |
| **mean** | - | **1/3** | **+0.000018438** | **+0.000000139** | **-0.000000566** | not promoted |

Alpha diagnostics show the current proxy usually chooses almost no shrink:

| scene | trial | alpha count | mean alpha | reading |
|---|---|---:|---:|---|
| garden | top1/s2 | 1 | ~1.000000 | residual proxy believes full strength is best |
| garden | risk4/s1 | 4 | ~0.999994 | no useful shrink |
| bicycle | risk8/s0.5 | 7 | ~0.999997 | no useful shrink |

The alpha interface is kept because it is now a real materialization control,
but the first result says the current evidence proxy cannot solve the remaining
train/test mismatch by scalar shrink alone.

### Stump Relaxed Discovery

Collected summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_zero_candidate_relaxed_v1_selector_20260514_summary/summary.md`

The relaxed planner finds `2` stump candidates, but all trials remain below the
selector threshold. This converts the old "zero candidates" problem into a
measured negative result: current face-local residual evidence still lacks
enough reliable stump signal.

## Qualitative Evidence

New risk-tail qualitative panels:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/qualitative_panel_manifest.json`

Each panel shows:

```text
GT | Phase-J baseline | Phase-S risk-tail | |Phase-J-GT| x5 | |Phase-S-GT| x5 | |S-J| x20
```

The views were selected automatically by largest report-only test PSNR gain
within the accepted scene/trial, not by manual image picking.

![flowers risk4 panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/flowers_risk4_s1_00019_phasej_vs_risktail_panel.png)

![counter risk4 panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/counter_risk4_s1_00002_phasej_vs_risktail_panel.png)

![treehill top1 panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/treehill_top1_s2_00004_phasej_vs_risktail_panel.png)

Per-view panel deltas:

| scene | trial | view | dPSNR | dSSIM | dLPIPS |
|---|---|---|---:|---:|---:|
| flowers | risk4/s1 | 00019 | +0.016326904 | +0.001679182 | -0.001603484 |
| counter | risk4/s1 | 00002 | +0.000520706 | +0.000003517 | +0.000004441 |
| treehill | top1/s2 | 00004 | +0.000267029 | +0.000001311 | -0.000001192 |

The qualitative story is therefore honest:

- `flowers` is visually and numerically the clearest Phase-S win;
- `counter` is a local all-metric average win, but some individual LPIPS views
  can still be mixed;
- `treehill` is safe but tiny;
- rejected scenes should be shown as fallbacks, not as visual wins.

## Commands and Artifacts

Risk-tail full8 command pattern:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes <scene> \
  --gpu <gpu> \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_<scene> \
  --trial_specs top1x2,risk4x1,risk8x0.5 \
  --wandb_group phase_s_facelocal_coupled_selector_v1_risk_full8_20260514 \
  --candidate_prefix facelocal_coupled_v1_riskpilot \
  --skip_failed_views \
  --selector_min_trainval_balanced_delta 0.00005 \
  --selector_enable_tail_stable_promotion
```

Alpha pilot command pattern:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes <scene> \
  --gpu <gpu> \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_alpha_refit_v1_pilot_20260514_<scene> \
  --trial_specs top1x2,risk4x1,risk8x0.5 \
  --wandb_group phase_s_facelocal_alpha_refit_v1_pilot_20260514 \
  --candidate_prefix facelocal_alpha_refit_v1 \
  --selector_fit_plan_alphas \
  --selector_enable_tail_stable_promotion
```

Environment note:

- `/data` reached 100% during this run; old superseded generated outputs under
  `ecsr_phase_d` and several obsolete Phase-S exploratory directories were
  cleaned to restore experiment write space.
- Current Phase-J, Phase-R evidence, risk-tail, alpha, stump-relaxed, summary,
  and qualitative outputs were retained.

## Honest Weaknesses

- This is not a complete paper endpoint yet. Phase-J is still the robust
  baseline-over-MeshSplatting claim.
- Phase-S risk-tail improves the mean mainly because `flowers` is strong; the
  method does not broadly promote all scenes.
- Full-frame visual differences remain subtle outside `flowers`; amplified
  difference maps are needed.
- `garden` exposes a train/test mismatch: too-loose tail promotion accepted a
  negative row, so the default tail threshold was tightened to `1.8e-5`.
- Per-face alpha refit is an implemented interface but currently a negative
  result; it does not yet give an adaptive strength policy.
- `stump` candidate discovery is no longer zero, but the candidates are too
  weak to promote.

## Recommendation

Use Phase-J as the main current method and Phase-S risk-tail as the newest
optional repair module. For slides, show the full8 risk-tail table and the
`flowers` panel as the strongest new evidence, then explicitly state that
alpha-refit and stump-relaxed are measured negative results rather than hidden
successes.
