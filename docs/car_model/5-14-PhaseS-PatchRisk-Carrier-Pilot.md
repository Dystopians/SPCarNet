# Phase-S PatchRisk Carrier Pilot

Date: 2026-05-14

This log records the follow-up attempt after GeoRisk/CVaR. The goal was to
move beyond single-face repair without introducing scene-specific parameters:
select train-certified seed faces, expand each seed into a small local patch of
candidate-plan faces, materialize the expanded carrier, and promote only through
a strict train-val render gate.

## Method Change

Implementation:

- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_fit_facelocal_plan_alphas.py`

New trial grammar:

```text
patchriskNxS
```

The mode is global and train-only:

1. Pick `N` seed faces with the existing GeoRisk score.
2. Expand each seed to nearby candidate-plan faces using source-mesh topology
   and centroid neighbors.
3. Admit only neighbors that already exist in the train-only candidate plan and
   pass fixed policy-gain, sample-count, residual-direction, and per-face
   tail-risk filters.
4. Fit conservative per-face alpha shrink factors on train evidence.
5. Run the existing Phase-K render gate; held-out test remains report-only.

The important distinction from GeoRisk/CVaR is that `patchrisk` changes the
materialized carrier set: it can write a local patch of residual carriers rather
than only a few isolated faces. It still does not use held-out test metrics for
selection.

## Commands

W&B group:

```text
phase_s_patchrisk_carrier_v1_20260514
```

Per-scene command template:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes <scene> \
  --gpu <gpu> \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_<scene> \
  --trial_specs patchrisk2x0.75,patchrisk4x0.5 \
  --candidate_prefix patchrisk_carrier_v1 \
  --wandb_group phase_s_patchrisk_carrier_v1_20260514 \
  --selector_fit_plan_alphas \
  --selector_alpha_steps 300 \
  --selector_alpha_max_total_samples 180000 \
  --selector_min_trainval_balanced_delta 0.00005 \
  --selector_tail_max_balanced_cvar_loss 0.00018 \
  --selector_tail_min_mean_to_cvar_ratio 0.25 \
  --selector_tail_max_lpips_positive_fraction 0.55 \
  --skip_failed_views
```

An initial run also tested `--selector_enable_tail_stable_promotion`. That was
rejected as unsafe because it accepted `garden` despite negative report-only
test deltas. The scene decisions were then reselected without rerendering using
the strict command above.

Collector:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py \
  --scenes garden,bicycle,bonsai,flowers,counter \
  --decision_path_template 'outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_{scene}/{scene}/coupled_selector_decision.json' \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_summary/summary_5scene_strict.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_summary/summary_5scene_strict.md \
  --output_csv outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_summary/summary_5scene_strict.csv
```

## Quantitative Evidence

Summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_summary/summary_5scene_strict.md`

| scene | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---|---:|---:|---:|---:|---|
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | strict gate correctly rejects the false positive |
| bicycle | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | still carrier/evidence starved |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val strongly negative despite many candidates |
| flowers | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | report-only positive but train-val below threshold, so not promoted |
| counter | patchrisk2/s0.75 | true | +0.000074387 | +0.000000358 | -0.000000447 | reliable control-scene positive |
| **mean** | - | **1/5** | **+0.000014877** | **+0.000000072** | **-0.000000089** | not a hard-scene breakthrough |

Important trial diagnostics:

| scene | trial | faces | train-val balanced | report-only dPSNR/dSSIM/dLPIPS | decision |
|---|---|---:|---:|---:|---|
| garden | patchrisk4/s0.5 | 16 | +0.000033438 | -0.000005722 / -0.000000179 / +0.000000328 | rejected after strict reselection |
| bicycle | patchrisk4/s0.5 | 7 | -0.000001192 | +0.000001907 / +0.000000000 / +0.000000149 | inner gate rejected |
| bonsai | patchrisk2/s0.75 | 12 | -0.000105500 | +0.000024796 / +0.000000119 / -0.000000447 | train-val rejected |
| flowers | patchrisk4/s0.5 | 21 | +0.000012159 | +0.005414963 / +0.000470459 / -0.000587225 | missed positive under fair gate |
| counter | patchrisk2/s0.75 | 10 | +0.000100374 | +0.000074387 / +0.000000358 / -0.000000447 | accepted |

## Honest Assessment

This is a real carrier-level pipeline improvement, but the result is not a
paper-level breakthrough.

What improved:

- `patchrisk` is now a first-class fixed trial mode.
- The trial manifest records seed faces, expanded patch faces, patch sizes,
  neighbor rejection reasons, and role annotations.
- It can materialize larger local carrier sets than GeoRisk/CVaR.
- On `counter`, it improves over the previous GeoRisk/CVaR control result.
- The strict promotion rule prevents the `garden` false positive from becoming
  an effective method result.

What did not improve:

- No hard scene was newly accepted.
- `bicycle` remains too weak/candidate-starved.
- `bonsai` has many candidates but the patch carrier is train-val negative.
- `flowers` shows a strong report-only positive that the train-val selector is
  not allowed to use.
- The accepted mean is still tiny and comes only from `counter`.

Conclusion: plan-replay patch expansion is not enough. The next attempt should
use direct patch-certified carrier discovery/fitting, rather than expanding the
existing candidate plan after the fact.
