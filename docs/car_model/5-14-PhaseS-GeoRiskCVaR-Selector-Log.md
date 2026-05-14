# Phase-S GeoRisk/CVaR Selector Log

Date: 2026-05-14

This log records the next Phase-S iteration after the risk-tail/alpha package:
a geometry-neighborhood and CVaR tail-risk selector was implemented and run on
the hard/control scene set requested for the next milestone. The result is
scientifically useful but not a new headline win: it strengthens the audit and
false-positive rejection story, while the accepted-scene coverage does not
improve over the previous risk-tail selector.

## Method Change

The selector now supports a new trial grammar:

```text
georiskNxS
```

Implementation:

- `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`
- `scripts/car_model/ecsr_fit_facelocal_plan_alphas.py`

The new `georisk` mode keeps selection train-only and adds three signals on top
of the previous risk-greedy face-set builder:

1. Geometry-neighborhood redundancy from source checkpoint triangle adjacency.
2. Per-face train-certificate lower-tail/CVaR risk over certified views.
3. Train-only local residual-error concentration as a small positive bonus.

The adjusted greedy score is audited per selected face in each trial manifest:

```text
adjusted =
  train_certificate_score
  * pair_redundancy_factor
  * geometry_adjacency_factor
  * per_face_tail_factor
  * (1 + coverage_bonus + local_error_bonus)
```

The selector also now records trial-level train-val per-view CVaR diagnostics:

- `balanced_cvar_delta`
- `balanced_cvar_loss`
- `mean_to_cvar_ratio`
- `lpips_positive_fraction`
- `psnr_cvar_delta`
- `lpips_worst_cvar_regression`

These fields are computed from train-val render metrics only. Held-out test
deltas remain report-only and are not used for selecting the promoted trial.

## Commands

All real trial runs used W&B logging:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  --scenes <scene> \
  --gpu <gpu> \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_<scene> \
  --trial_specs georisk4x1,georisk8x0.5 \
  --wandb_group phase_s_georisk_cvar_v1_20260514 \
  --candidate_prefix facelocal_georisk_cvar_v1 \
  --selector_min_trainval_balanced_delta 0.00005 \
  --selector_enable_tail_stable_promotion \
  --skip_failed_views
```

After the implementation was extended with CVaR tail diagnostics, the same
trial outputs were reselected without rerendering. This wrote the new CVaR
fields into `coupled_selector_decision.json`.

Collection command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py \
  --scenes garden,bicycle,room,kitchen,bonsai,flowers,counter \
  --decision_path_template 'outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_{scene}/{scene}/coupled_selector_decision.json' \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_summary/summary_7scene.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_summary/summary_7scene.md \
  --output_csv outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_summary/summary_7scene.csv
```

## Quantitative Evidence

Summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_summary/summary_7scene.md`

| scene | candidates | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | note |
|---|---:|---|---:|---:|---:|---:|---|
| garden | 110 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | false-positive candidates rejected |
| bicycle | 7 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | georisk trials fail inner/outer gate |
| room | 76 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best georisk trial too small |
| kitchen | 145 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | PSNR positive but LPIPS tail not compelling |
| bonsai | 1266 | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val negative |
| flowers | 35 | georisk4/s1 | true | +0.005418777 | +0.000470877 | -0.000586182 | same strong positive as risk-tail risk4 |
| counter | 127 | georisk4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | same all-metric positive as risk-tail risk4 |
| **mean** | - | - | **2/7** | **+0.000782013** | **+0.000067328** | **-0.000083983** | positive but dominated by flowers |

The important comparison is not the mean alone. GeoRisk/CVaR does not improve
accepted-scene coverage over the previous risk-tail selector:

- risk-tail full8 accepted `flowers`, `counter`, and tiny `treehill`;
- GeoRisk/CVaR on the requested 7-scene set accepts `flowers` and `counter`;
- on overlapping scenes, the accepted trials are effectively the same as the
  previous `risk4_s1` positives;
- hard scenes `bicycle`, `garden`, `room`, `kitchen`, and `bonsai` still fall
  back to Phase-J.

## Tail Diagnostics

Representative train-val tail rows:

| scene | trial | selector pass | train-val balanced | balanced neg frac | balanced CVaR loss | mean/CVaR ratio | report-only dPSNR/dSSIM/dLPIPS |
|---|---|---:|---:|---:|---:|---:|---|
| flowers | georisk4/s1 | true | +0.000019789 | 0.3684 | 0.000077352 | 0.3345 | +0.005418777 / +0.000470877 / -0.000586182 |
| counter | georisk4/s1 | true | +0.000101507 | 0.4151 | 0.000192677 | 0.8251 | +0.000055313 / +0.000000417 / -0.000001699 |
| garden | georisk8/s0.5 | false | +0.000018060 | 0.4390 | 0.000054942 | 0.4699 | -0.000001907 / -0.000000179 / +0.000000447 |
| bicycle | georisk8/s0.5 | false | -0.000001192 | 0.2558 | 0.000062575 | 0.0826 | +0.000001907 / +0.000000000 / +0.000000149 |
| bonsai | georisk8/s0.5 | false | -0.000044823 | 0.5156 | 0.000423482 | 0.0000 | -0.000024796 / -0.000000417 / +0.000000477 |

This confirms the main weakness: per-face train certificates and geometry
adjacency are not sufficient to predict render-level generalization in the hard
scenes. Trial-level render CVaR is useful for auditing and rejecting
false-positive candidates, but it has not created new strong positives.

## Qualitative Evidence

New qualitative assets:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/qualitative_summary.md`

Each panel shows full frame and an automatically selected local crop:

```text
GT | Phase-J | GeoRisk/CVaR | green-better/magenta-worse error change | abs diff x80
```

Panels:

![flowers georisk panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/flowers_georisk4_s1_00019_georisk_cvar_panel.png)

![counter georisk panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/counter_georisk4_s1_00002_georisk_cvar_panel.png)

![garden rejected panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/garden_georisk8_s0p5_00006_georisk_cvar_panel.png)

Per-view panel deltas:

| scene | trial | mode | view | dPSNR | dSSIM | dLPIPS |
|---|---|---|---|---:|---:|---:|
| flowers | georisk4/s1 | accepted positive | 00019 | +0.016326904 | +0.001679182 | -0.001603484 |
| counter | georisk4/s1 | accepted positive | 00002 | +0.000520706 | +0.000003517 | +0.000004441 |
| garden | georisk8/s0.5 | rejected false positive | 00006 | -0.000154495 | -0.000001371 | +0.000004120 |

The qualitative result is mixed. `flowers` is clear. `counter` has a positive
scene mean and local PSNR/SSIM improvement, but the displayed per-view LPIPS is
slightly worse. `garden` is intentionally shown as a rejected failure case.

## Validation

Validation commands completed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_fit_facelocal_plan_alphas.py \
  scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py

git diff --check -- \
  scripts/car_model/ecsr_run_facelocal_coupled_selector.py \
  scripts/car_model/ecsr_fit_facelocal_plan_alphas.py \
  scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py
```

Input validation smoke tests also reject invalid negative risk/CVaR arguments.

## Honest Assessment

This is a real method and pipeline improvement, but not a paper-level
breakthrough yet.

What improved:

- `georisk` is now a first-class, auditable train-only selector mode;
- source checkpoint geometry is loaded and summarized in the manifest;
- per-face tail/CVaR and local error terms are recorded;
- train-val render CVaR is now available for promotion diagnostics;
- qualitative panels now include local crops and error-change maps;
- false positives such as `garden` remain rejected.

What did not improve:

- accepted-scene coverage did not expand beyond the previous risk-tail positives;
- the selected positive trials for `flowers` and `counter` match the old
  risk-tail `risk4_s1` behavior;
- `bicycle`, `room`, `kitchen`, `bonsai`, and `garden` still lack a strong
  representation-level repair;
- mean metrics remain dominated by `flowers`;
- this does not yet support a claim that Phase-S broadly improves over Phase-J
  or clean MeshSplatting by itself.

Next step:

The bottleneck appears to be carrier capacity and residual evidence quality,
not only selector ranking. The next major attempt should change the
representation carrier or evidence objective, for example a patch-level
surface residual field with render-consistency supervision, instead of adding
more K/scale variants to the same face-local SH1 carrier.
