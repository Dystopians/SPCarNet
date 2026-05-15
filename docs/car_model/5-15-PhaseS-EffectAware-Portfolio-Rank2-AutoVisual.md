# Phase-S Effect-Aware Portfolio, Rank2 Carrier, And Auto-Visual Pipeline

Date: 2026-05-15

Status: `NOT COMPLETE`. This note records the next repair loop after the v20
full9 continuation. The loop did implement and validate real train/eval method
changes, but the strict evidence still says the Phase-S representation repair is
too small and too sparse to claim a paper-level closure.

## Motivation

The v20 auto-prefix carrier policy improved audit cleanliness: carrier holdout
is disjoint from policy tuning, replay is strict, and carrier count is selected
by a deterministic train-only prefix. The result did not solve the scientific
bottleneck. The accepted v20 `garden` and `room` rows are effectively metric
noise, so the portfolio accepted count improved without a matching visual or
quantitative breakthrough.

This loop therefore makes two changes:

1. Add an effect-aware portfolio mode that rejects train-val accepted rows when
   their train-val effect size is too small or their operator audit is no-op.
2. Test a `rank2` PatchCert carrier basis as a representation-level method
   ablation, changing carrier capacity while keeping v20 strict/disjoint
   evidence rules.

## Implemented Changes

### Effect-Aware Portfolio Selector

File:

```text
scripts/car_model/ecsr_select_phase_s_policy_portfolio.py
```

New optional gates:

```text
--min_trainval_psnr_delta
--max_trainval_ssim_regression
--max_trainval_lpips_regression
--min_trainval_effect_score
--require_operator_audit
--require_operator_policy_pass
--reject_no_op_operator
```

Default behavior remains backward-compatible. With no new flags, the selector
still reproduces portfolio v2 accepted count `4 / 9`. The stricter mode ranks
eligible candidates by train-val effect score:

```text
dPSNR + 20 * dSSIM - 20 * dLPIPS
```

Held-out test metrics remain report-only after selection.

### Auto-Visual Face-Local Pipeline

File:

```text
scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py
```

This is a scene-agnostic coordinator around existing ECSR tools. It:

1. generates strict train-only face-local SH residual candidate plans;
2. optionally refits per-face materialization alphas;
3. runs render-calibrated train-val selector trials over small face subsets;
4. saves command manifests, logs, decisions, and report-only test deltas.

It is intended as a reproducible pipeline wrapper, not as a new metric claim by
itself.

Smoke validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  scripts/car_model/ecsr_select_phase_s_policy_portfolio.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py \
  --scenes bicycle \
  --profile smoke \
  --dry_run \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/autovisual_facelocal_v1_smoke_20260515 \
  --gpu -1 \
  --wandb_mode offline
```

Smoke outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/autovisual_facelocal_v1_smoke_20260515/pipeline_command_manifest.json
outputs/carnet/meshsplatopt/ecsr_phase_s/autovisual_facelocal_v1_smoke_20260515/pipeline_command_manifest.md
outputs/carnet/meshsplatopt/ecsr_phase_s/autovisual_facelocal_v1_smoke_20260515/pipeline_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_s/autovisual_facelocal_v1_smoke_20260515/pipeline_summary.md
```

## Effect-Aware Portfolio v1

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_select_phase_s_policy_portfolio.py \
  --scenes garden,bicycle,room,kitchen,bonsai,flowers,counter,stump,treehill \
  --candidate georisk=outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_{scene}/{scene}/coupled_selector_decision.json \
  --candidate riskpilot=outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_{scene}/{scene}/coupled_selector_decision.json \
  --candidate patchrisk=outputs/carnet/meshsplatopt/ecsr_phase_s/patchrisk_carrier_v1_20260514_{scene}/{scene}/coupled_selector_decision.json \
  --candidate gaincert_v2=outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v2_faceshrink_fairreplay_20260513_{scene}/decisions/{scene}_decision.json \
  --candidate patchcert_v6=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v6_compactstrat_gate_20260514_redecision/decisions/{scene}_decision.json \
  --candidate v20_auto=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_disjoint_sampleholdout_chartquad_key_20260514/decisions/{scene}_decision.json \
  --candidate v20_remainingA=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_remainingA_20260515/decisions/{scene}_decision.json \
  --candidate v20_remainingB=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_remainingB_20260515/decisions/{scene}_decision.json \
  --candidate v20_remainingC=outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v20_autoprefix_remainingC_20260515/decisions/{scene}_decision.json \
  --min_trainval_balanced_delta 0 \
  --min_trainval_psnr_delta 0.00002 \
  --max_trainval_ssim_regression 0.00001 \
  --max_trainval_lpips_regression 0.00001 \
  --min_trainval_effect_score 0.00005 \
  --require_operator_policy_pass \
  --reject_no_op_operator \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v1_20260515/portfolio_summary.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v1_20260515/portfolio_summary.md \
  --output_csv outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v1_20260515/portfolio_summary.csv
```

Result:

| scene | selected | accepted | train-val dPSNR | report-only dPSNR | report-only dSSIM | report-only dLPIPS |
|---|---|---:|---:|---:|---:|---:|
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| bicycle | patchcert_v6 | true | +0.000020981 | +0.000387192 | +0.000035524 | -0.000115275 |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| kitchen | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| flowers | gaincert_v2 | true | +0.000043869 | +0.005426407 | +0.000470638 | -0.000587165 |
| counter | riskpilot | true | +0.000102997 | +0.000055313 | +0.000000417 | -0.000001699 |
| stump | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| treehill | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| **mean** | - | **3 / 9** | - | **+0.000652101** | **+0.000056287** | **-0.000078238** |

Reading: this is not a new representation win. It is a stricter fixed policy
that removes v20 `garden/room` near-noop rows and promotes only rows with
nontrivial train-val effect-size evidence. The mean improves slightly versus
portfolio v2 because it drops noise rows, but the scientific bottleneck remains
coverage: only `bicycle`, `flowers`, and `counter` are selected.

## Rank2 PatchCert Carrier v21 And Coverage-Aware v22

Hypothesis: v20 chart-quadratic carriers may be too restrictive after strict
holdout and auto-prefix selection. A rank-2 carrier can express two residual
modes inside the same certified patch while preserving the no-test strict
selection boundary.

Only the carrier basis changes:

```text
--delta_patch_cert_cluster_basis_mode rank2
```

The first rank2 groups ran with W&B group
`phase_s_patchcert_v21_rank2_autoprefix_20260515`:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v21_rank2_autoprefix_groupA_20260515
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v21_rank2_autoprefix_groupB_20260515
```

Group A scenes: `bicycle,flowers,counter` on GPU 1.

Group B scenes: `garden,bonsai` on GPU 4.

The follow-up retry rows exposed a second weakness: the original auto-prefix
rule could select a very small certified prefix with excellent proxy gain but
negligible render impact. v22 therefore adds coverage-aware auto-prefix
selection:

```text
--delta_patch_cert_carrier_holdout_auto_prefix_min_faces 16
--delta_patch_cert_carrier_holdout_auto_prefix_face_bonus 0.02
```

Implementation details:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
  now supports a train-only auto-prefix face-count floor and face-count bonus.
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py` forwards the
  new coverage-aware flags.
- `scripts/car_model/ecsr_run_autovisual_facelocal_pipeline.py` exposes the
  same flags to the scene-agnostic coordinator.
- `--no_op_on_fail` is now a real BooleanOptionalAction, so `--no-no_op_on_fail`
  works during failure diagnostics.
- `scripts/car_model/ecsr_select_phase_s_policy_portfolio.py` now rejects
  missing operator audit when any operator-audit-dependent portfolio gate is
  enabled.
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py` hardlinks
  copied GT images when possible before falling back to `copy2`, reducing disk
  pressure during long ELA/eval sweeps.

Promotion criteria for both v21 and v22:

- accepted rows must have `selection_uses_test=false`;
- operator audit must show a real checkpoint edit, not no-op copy;
- accepted rows must exceed effect-aware train-val thresholds;
- held-out test metrics remain report-only;
- a result that only changes accepted count through near-noop deltas is not a
  method success.

### v21 Retry Results

Evidence paths:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v21_rank2_autoprefix_retry_counter_20260515/decisions/counter_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v21_rank2_autoprefix_retry_kitchen_20260515/decisions/kitchen_decision.json
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v2_rank2_strictaudit_20260515/portfolio_summary.md
```

| scene | accepted by inner gate | train-val dPSNR | train-val dSSIM | train-val dLPIPS | strict reading |
|---|---:|---:|---:|---:|---|
| counter retry | false | -0.000003815 | -0.000000119 | +0.000000015 | real operator, rejected by balanced/tail gates |
| kitchen retry | true | +0.000005722 | +0.000000000 | -0.000000015 | real operator, but effect-size gate rejects it as metric noise |
| rank2 strict portfolio | 3 / 9 | +0.000652101 mean effective report-only PSNR | +0.000056287 mean effective SSIM | -0.000078238 mean effective LPIPS | rank2 rows do not improve the strict portfolio |

This falsifies the simple "rank2 capacity alone fixes Phase-S" hypothesis.

### Coverage-Aware v22 Pilot

Command family:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --scenes garden,bonsai,room \
  --gpu 7 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v22_coverageaware_retry_garden_bonsai_room_20260515 \
  --candidate_label phase_s_patchcert_v22_coverageaware_retry_20260515 \
  --delta_patch_cert_cluster_basis_mode rank2 \
  --delta_patch_cert_carrier_holdout_auto_prefix \
  --delta_patch_cert_carrier_holdout_auto_prefix_min_faces 16 \
  --delta_patch_cert_carrier_holdout_auto_prefix_face_bonus 0.02 \
  --wandb_group phase_s_patchcert_v22_coverageaware_retry_20260515
```

The initial `room` run failed because an earlier disk cleanup removed source
Phase-J depth maps. This was an experiment-hygiene error, not a method failure.
The source depth maps were restored with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_render_evidence_maps.py \
  -s /data/peilincai/mesh_datasets/mipnerf360/room \
  -m outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model \
  -i images_2 --resolution -1 --eval --iteration 26000 \
  --method_name ours_26000_phasef_extra_compact_base --quiet
```

Then `room` was rerun under W&B name
`phase_s_patchcert_v22_coverageaware_retry_room_retry_after_depthrestore_20260515`.

Decision and audit results:

| scene | operator edit | selected carriers/faces | gate accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS | strict reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | 84 vertices, 28 accepted faces | 6 / 28 | true | +0.000026703 | -0.000000060 | -0.000000209 | +0.000001907 | -0.000000179 | -0.000000015 | real non-noop, but still below effect-size threshold |
| bonsai | 150 vertices, 50 accepted faces | 50 / 50 | true | +0.000061035 | -0.000000119 | +0.000000954 | +0.000019073 | +0.000000119 | +0.000001267 | PSNR positive, LPIPS regresses; strict effect row rejects it |
| room | 48 vertices, 16 accepted faces | 3 / 16 | false | +0.000011444 | +0.000000000 | -0.000000075 | +0.000009537 | +0.000000060 | +0.000000089 | tail negative fraction rejects it |

Strict v22 pilot portfolio:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_effectaware_portfolio_v22_covaware_pilot_strictaudit_20260515/portfolio_summary.md
```

With `--min_trainval_effect_score 0.00005`, `--require_operator_audit`,
`--require_operator_policy_pass`, and `--reject_no_op_operator`, the strict
pilot accepts `0 / 3` new v22 rows and falls back to Phase-J on
`garden,bonsai,room`. The rejection is intentional: v22 improves operator
coverage, but the render-space effect is still too small for a strong claim.

### Qualitative Evidence

The v22 qualitative panels were generated with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_build_phase_s_patchcert_qualitative.py \
  --scenes garden,bonsai,room \
  --root_template outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v22_coverageaware_retry_garden_bonsai_room_20260515 \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v22_coverageaware_retry_garden_bonsai_room_20260515_qualitative \
  --views_per_scene 2 --image_width 300 --diff_boost 100
```

Outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v22_coverageaware_retry_garden_bonsai_room_20260515_qualitative/patchcert_qualitative_contact_sheet.png
outputs/carnet/meshsplatopt/ecsr_phase_s/phase_s_patchcert_v22_coverageaware_retry_garden_bonsai_room_20260515_qualitative/qualitative_summary.md
```

The selected panels are useful diagnostics, not strong paper figures. Per-view
PSNR gains are visible in amplified differences, but full-frame changes remain
hard to see. This agrees with the numeric evidence: Phase-S v22 edits are real
but still low amplitude.

## Current Assessment

This loop completes the engineering checklist for a real follow-up method
iteration: interface gaps are fixed, W&B long rows were run, rank2 and
coverage-aware ablations were evaluated, strict portfolios were produced, and
qualitative panels were saved. It does not complete the scientific checklist.

What improved:

- near-noop accepted rows are no longer counted as method progress;
- the carrier auto-prefix policy can prefer broader train-certified coverage;
- `garden`, `bonsai`, and `room` now produce real non-noop v22 operators;
- all promotion remains train-val-only and held-out test is report-only.

What remains weak:

- strict effect-aware selection still falls back on all three v22 pilot scenes;
- report-only test deltas are mostly `1e-6` to `1e-5`, not a visible result;
- `room` still fails tail safety even with a real operator edit;
- the best current Phase-S portfolio remains a sparse `3 / 9` row dominated by
  older `flowers` and `counter` wins.

Final status for this loop: `NOT COMPLETE`. The next method step should stop
expanding certified patch size alone and instead change the learned
representation update itself: a train-only, view-consistent residual field that
shares parameters across many certified faces, regularized by cross-view
photometric agreement and promoted by the same strict train-val gate.
