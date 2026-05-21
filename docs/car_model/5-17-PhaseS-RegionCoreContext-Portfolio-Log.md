# Phase-S Region Core/Context Weighted Fitting and Portfolio Log

Date: 2026-05-17

Status: `NOT COMPLETE`, but this is a real train/eval pipeline upgrade over
the 2026-05-16 render-visible region prior.

## Motivation

The 2026-05-16 render-visible region prior changed which faces are proposed:
train-only high-residual image regions are projected back to surface carriers.
That improved proposal locality, but the actual residual fitting objective still
treated all sampled pixels on selected faces almost the same, weighted mainly by
residual magnitude and face consistency. In practice this created a
proxy-to-render mismatch: local proxy gains could look strong while full-render
held-out metrics stayed tiny or regressed.

This update sends the render-visible region information into the fitting
objective itself. It is not a scene-specific parameter scan. The policy is fixed
across the launched scenes:

```text
region core weight: 4.0
region context weight: 0.5
region outside weight: 0.15
region boundary margin: 2 px
shared residual field anchors: 16
delta strength: 0.035
max abs RGB delta: 0.05
Phase-K train-val gate: unchanged
held-out test: report-only
```

## Implementation

Changed files:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_select_phase_s_policy_portfolio.py`

New fitting interface:

```text
--region_carrier_json
--region_core_weight
--region_context_weight
--region_outside_weight
--region_boundary_px
```

New Phase-K runner forwarding interface:

```text
--delta_facelocal_region_carrier_json
--delta_region_core_weight
--delta_region_context_weight
--delta_region_outside_weight
--delta_region_boundary_px
```

The fitter loads a train-only render-visible carrier JSON and assigns each
sample to one of three bins:

- `core`: sample lands inside a carrier region box for the same view and face.
- `context`: sample is on a carrier-supported face/view but outside the region
  box.
- `outside`: sample is on a selected face but has no matching region support.

The original sample weight is multiplied by the bin weight before SH/shared-field
optimization. The audit now records `fit_region_bins` and
`policy_val_region_bins` so the selected edits can be inspected.

The portfolio script also now reports
`mean_effective_test_balanced_delta_report_only`. A later fix computes this
balanced report-only score from dPSNR/dSSIM/dLPIPS when older candidate decision
files do not store an explicit balanced field.

## Experiment Commands

Group A, outdoor successful probe:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --force \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_A \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence \
  --scenes garden,flowers \
  --iteration 26000 \
  --gpu 4 \
  --delta_facelocal_region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/{scene}/render_visible_region_carriers.json \
  --delta_region_core_weight 4.0 \
  --delta_region_context_weight 0.5 \
  --delta_region_outside_weight 0.15 \
  --delta_region_boundary_px 2 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_region_corectx_20260517 \
  --wandb_name phase_s_region_corectx_v1_A
```

Group B, indoor/control probe:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --force \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_B \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence \
  --scenes kitchen,bonsai,counter \
  --iteration 26000 \
  --gpu 0 \
  --delta_facelocal_region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/{scene}/render_visible_region_carriers.json \
  --delta_region_core_weight 4.0 \
  --delta_region_context_weight 0.5 \
  --delta_region_outside_weight 0.15 \
  --delta_region_boundary_px 2 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_region_corectx_20260517 \
  --wandb_name phase_s_region_corectx_v1_B
```

Final fixed portfolio summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v1.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v1.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v1.csv
```

Qualitative outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_A_qualitative/patchcert_qualitative_contact_sheet.png
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_B_qualitative/patchcert_qualitative_contact_sheet.png
assets/spcarnet_phase_s_region_corectx_A_contact_sheet.png
assets/spcarnet_phase_s_region_corectx_B_contact_sheet.png
```

## Direct Core/Context Results

These rows are direct Phase-K train-val decisions for the new core/context
weighted method. Held-out test deltas are report-only.

| scene | gate accepted | train-val balanced | report-only balanced | dPSNR test | dSSIM test | dLPIPS test | accepted faces | vertices added | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | true | +0.000037074 | +0.000015736 | +0.000013351 | -0.000000119 | -0.000000238 | 183 | 549 | safe but weaker than the 2026-05-16 region-prior garden row |
| flowers | true | +0.000135303 | +0.026483655 | +0.005399704 | +0.000467956 | -0.000586241 | 66 | 198 | the new positive result; visible local improvements |
| kitchen | true | +0.000104040 | -0.026346326 | -0.022924423 | -0.000011384 | +0.000159711 | 214 | 642 | false positive, must be blocked by portfolio |
| bonsai | true | +0.000068069 | -0.009002686 | -0.006856918 | +0.000661612 | +0.000768900 | 226 | 678 | false positive with LPIPS regression |
| counter | true | +0.000010073 | -0.013494253 | -0.004993439 | -0.000232875 | +0.000192165 | 142 | 426 | false positive, older riskpilot remains better |

The important scientific conclusion is mixed. The new fitting objective can
convert the render-visible prior into a stronger `flowers` representation edit,
but the unchanged mean train-val gate still admits bad indoor/control rows. The
method is therefore useful only through a stricter fixed portfolio.

## Final Fixed Portfolio

The final portfolio uses only train-val decisions and fixed thresholds:

```text
min train-val balanced delta: 0
min train-val PSNR delta: 0.00002
max train-val SSIM regression: 0.00001
max train-val LPIPS regression: 0.000001
min train-val effect score: 0.00005
require operator policy pass: true
reject no-op operator: true
```

Accepted scenes: `5 / 9`.

Mean effective report-only delta over Phase-J fallback:

```text
dPSNR:  +0.000947740
dSSIM:  +0.000062552
dLPIPS: -0.000098634
balanced: +0.004171458
```

| scene | selected policy | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | report-only balanced |
|---|---|---:|---:|---:|---:|---:|
| bicycle | patchcert_v6 | true | +0.000387192 | +0.000035524 | -0.000115275 | +0.003403187 |
| flowers | rvregion_corectx_A | true | +0.005399704 | +0.000467956 | -0.000586241 | +0.026483655 |
| garden | rvregion_garden | true | +0.000043869 | -0.000000417 | -0.000000089 | +0.000037313 |
| stump | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| treehill | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| counter | riskpilot | true | +0.000055313 | +0.000000417 | -0.000001699 | +0.000097632 |
| kitchen | rvregion_indoor | true | +0.002643585 | +0.000059485 | -0.000184402 | +0.007521331 |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |

Compared with the earlier effect-aware portfolio v1 from 2026-05-15
(`3 / 9`, `+0.000652101` PSNR, `+0.000056287` SSIM, `-0.000078238` LPIPS),
this is a real improvement in coverage and mean metrics. Compared with the
2026-05-16 robust render-visible region prior (`2 / 9`,
`+0.000298606` PSNR, `+0.000006563` SSIM, `-0.000020499` LPIPS), it is also
better. The effect size is still very small.

## Qualitative Reading

The useful qualitative panel is Group A:

```text
assets/spcarnet_phase_s_region_corectx_A_contact_sheet.png
```

It includes two `flowers` held-out views with report-only local improvements:

- `flowers/00019.png`: `+0.016281` PSNR, `+0.001674` SSIM, `-0.001596` LPIPS.
- `flowers/00009.png`: `+0.016106` PSNR, `+0.001383` SSIM, `-0.001848` LPIPS.

Group B is diagnostic rather than promotional:

```text
assets/spcarnet_phase_s_region_corectx_B_contact_sheet.png
```

It shows why the portfolio must reject or fall back on false-positive rows:
some views improve locally, but scene-level report-only means regress on
`kitchen`, `bonsai`, and `counter`.

## Current Weaknesses

- The direct core/context method is not safe as a standalone method. It accepts
  all five launched scenes, but three have negative report-only balanced deltas.
- The full9 effective gain is positive but still small. It is not a
  paper-level breakthrough by itself.
- Gains remain concentrated in `flowers`; `stump`, `treehill`, `room`, and
  `bonsai` still fall back to Phase-J under the final portfolio.
- Mean train-val render gates are insufficient for this operator family. Tail,
  stratified, and effect-size gates are necessary.
- The best paper-facing endpoint remains Phase-J against clean MeshSplatting.
  Phase-S is a representation-level repair study with sparse accepted wins.

## Next Gate

The next credible method upgrade should target the false-positive failure mode,
not simply increase weights or scan more constants:

1. Add tail/stratified gate awareness directly to the Phase-K direct decision
   for region-weighted fitting, so `kitchen/bonsai/counter` are rejected before
   portfolio aggregation.
2. Replace coarse bbox membership with a true per-pixel residual mask or
   differentiable masked render-space objective only after the current
   sample-weighted objective has been fully analyzed.
3. Require any next claim to improve the final fixed portfolio by a non-noise
   amount and preserve `selection_uses_test=false`.

## 2026-05-20 Strictcompact Re-Decision

Status: `NOT COMPLETE`, but this closes a concrete fairness/safety gap in the
May 17 portfolio.

The previous `compact_gate_enable` implementation recorded compact/tail risk
diagnostics, but compact gate failure did not reject a candidate that had
already passed the ordinary mean train-val gate. That made the gate useful for
auditing but insufficient as a fixed policy. The May 20 patch adds:

```text
ecsr_decide_phasek_trainval_gate.py: --compact_gate_require
ecsr_run_phasek_barycentric_gate_scene.py: --gate_compact_require
```

When required, the candidate must pass the ordinary train-val gate, operator
audit, and the compact/tail/stratified gate. Compact success no longer rescues a
candidate that fails the ordinary gate in this mode.

Re-decision outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strict_compact_decisions/
```

Strictcompact direct results:

| scene | accepted | compact accepted | report-only balanced | compact reason |
|---|---:|---:|---:|---|
| flowers | true | true | +0.026483655 | pass |
| garden | false | false | +0.000015736 | faces/vertices/ratio exceed compact budget |
| kitchen | false | false | -0.026346326 | faces/vertices/ratio exceed compact budget |
| bonsai | false | false | -0.009002686 | faces/vertices/ratio exceed compact budget |
| counter | false | false | -0.013494253 | stratified PSNR tail below threshold |

The important change is not a larger mean score; it is cleaner provenance. Raw
core/context false positives are now rejected before portfolio aggregation
using train-val-only evidence.

Fixed v2 portfolio outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v2_strictcompact.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v2_strictcompact.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v2_strictcompact.csv
```

The v2 portfolio keeps the same effective full9 metrics as v1, but replaces raw
`rvregion_corectx_A/B` eligibility with the strictcompact re-decision:

```text
accepted: 5 / 9
dPSNR:  +0.000947740
dSSIM:  +0.000062552
dLPIPS: -0.000098634
balanced: +0.004171458
```

Selected policies:

| scene | selected policy | effective dPSNR | effective dSSIM | effective dLPIPS | report-only balanced |
|---|---|---:|---:|---:|---:|
| bicycle | patchcert_v6 | +0.000387192 | +0.000035524 | -0.000115275 | +0.003403187 |
| flowers | rvregion_corectx_strictcompact | +0.005399704 | +0.000467956 | -0.000586241 | +0.026483655 |
| garden | rvregion_garden | +0.000043869 | -0.000000417 | -0.000000089 | +0.000037313 |
| stump | Phase-J fallback | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| treehill | Phase-J fallback | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| room | Phase-J fallback | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |
| counter | riskpilot | +0.000055313 | +0.000000417 | -0.000001699 | +0.000097632 |
| kitchen | rvregion_indoor | +0.002643585 | +0.000059485 | -0.000184402 | +0.007521331 |
| bonsai | Phase-J fallback | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 |

Honest reading: this is a required policy fix, not a new visual breakthrough.
It strengthens the paper-story auditability because it removes a known gate
loophole, but the Phase-S scientific bottleneck remains non-trivial accepted
coverage beyond `flowers`, `bicycle`, and `kitchen` via older region prior.

## 2026-05-20 Local Visual Evidence

To make the positive `flowers` result easier to inspect visually, I also ran the
existing train-defined surface-support local metric protocol. The support masks
come from train residual evidence, then are projected onto held-out test renders;
metrics are computed only after those masks/crops are fixed.

Command:

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_eval_surface_support_local_metrics.py \
  --scene flowers \
  --evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260516/evidence/flowers \
  --surface_maps_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_surface_maps/surface_maps \
  --baseline_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_phasej_guarded_adaptedge_ela_replay_corectx_v1 \
  --candidate_dir outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_A/flowers/model/test/ours_26000_phase_s_rvregion_corectx_v1_phasej_ela \
  --baseline_label Phase-J \
  --candidate_label SPCarNet-v2-corectx \
  --output_dir outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_v2_strictcompact_local_metrics/flowers \
  --top_faces 256 \
  --min_mask_pixels 768 \
  --alpha_min 0.05 \
  --dilate 8 \
  --crop_pad 24 \
  --max_views 12 \
  --save_panels \
  --device cuda
```

Outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_v2_strictcompact_local_metrics/flowers/surface_support_local_metrics.md
assets/spcarnet_phase_s_v2_strictcompact_flowers_local_support.png
```

Summary:

| metric | value |
|---|---:|
| evaluated held-out views | 12 |
| mean delta mask PSNR | +0.003223 |
| mean delta mask MAE | +0.00002632 |
| mean delta crop PSNR | +0.010150 |
| mean delta crop SSIM | +0.00038835 |
| mean delta crop LPIPS | -0.00060000 |
| wins mask PSNR / mask MAE | 9 / 12, 4 / 12 |
| wins crop PSNR / crop SSIM / crop LPIPS | 12 / 12, 12 / 12, 11 / 12 |

This improves the qualitative evidence for `flowers`: the strongest story is a
local crop/crop-LPIPS improvement on train-defined support regions, not a
large full-frame perceptual jump.
