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

## 2026-05-20 Runner-Level Smoke

After adding `--gate_compact_require`, I ran one end-to-end Phase-K pipeline
smoke on `flowers` to verify the new gate path works through the orchestrator,
not only through manual re-decision. W&B logging was online and the run was
placed on physical GPU 7 via `CUDA_VISIBLE_DEVICES=7 --gpu 0`.

Output root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_smoke_20260520
```

Decision:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_smoke_20260520/decisions/flowers_decision.json
```

Result:

| field | value |
|---|---:|
| accepted | true |
| compact gate accepted | true |
| train-val balanced | +0.000135303 |
| train-val dPSNR / dSSIM / dLPIPS | +0.000114441 / -0.000001013 / -0.000002056 |
| report-only test balanced | +0.026483655 |
| report-only test dPSNR / dSSIM / dLPIPS | +0.005399704 / +0.000467956 / -0.000586241 |
| tail negative fraction / CVaR | 0.473684 / -0.000163801 |
| LPIPS positive fraction / worst regression | 0.447368 / +0.000015557 |

W&B runs:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/40oag7se
https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/0rjogm71
https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/zlm9q5x1
https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/bboy0r5c
```

This confirms the new runner forwarding and required compact gate semantics in
the real train/eval path. It is still a single-scene smoke; it does not replace
the existing full9 fixed-portfolio evidence.

## 2026-05-21 End-to-End Strictcompact Multi-Scene Replay

Status: `NOT_COMPLETE_SMALL_POSITIVE`. This run turns the May 20 manual
strictcompact re-decision into an end-to-end replay across the remaining
core/context scenes. It is a fairness and reliability milestone, not a new
large effect-size breakthrough.

Fixed method configuration:

```text
operator: facelocal_sh1
delta strength: 0.035
max abs RGB delta: 0.05
shared residual field: enabled, 16 anchors
region core/context/outside weights: 4.0 / 0.5 / 0.15
region boundary: 2 px
policy-val minimum relative gain: 0.02
strict gate: --gate_tail_require_available, --gate_compact_enable, --gate_compact_require
compact budget: <=160 faces, <=512 vertices, <=1.5e-05 face ratio
W&B group: phase_s_strictcompact_pipeline_20260521
selection: train-val only; held-out test remains report-only
```

Output roots:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_smoke_20260520
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_garden_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_indoorB_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_counter_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_merged_20260521
```

Execution notes:

- `garden` was replayed as a standalone W&B online run.
- `kitchen,bonsai` were replayed in the `indoorB` runner.
- `counter` was split into a standalone runner for parallel completion. After
  `bonsai` and standalone `counter` both wrote decisions, the duplicate
  `indoorB` counter continuation was terminated to avoid unnecessary GPU use.

End-to-end strictcompact decisions:

| scene | accepted | train-val balanced | report-only balanced | dPSNR test | dSSIM test | dLPIPS test | compact accepted | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| flowers | true | +0.000135303 | +0.026483655 | +0.005399704 | +0.000467956 | -0.000586241 | true | pass |
| garden | false | +0.000037074 | +0.000015736 | +0.000013351 | -0.000000119 | -0.000000238 | false | faces/vertices/face-ratio exceed compact budget |
| kitchen | false | +0.000104040 | -0.026346326 | -0.022924423 | -0.000011384 | +0.000159711 | false | faces/vertices/face-ratio exceed compact budget |
| bonsai | false | +0.000068069 | -0.009002686 | -0.006856918 | +0.000661612 | +0.000768900 | false | faces/vertices/face-ratio exceed compact budget |
| counter | false | +0.000010073 | -0.013494253 | -0.004993439 | -0.000232875 | +0.000192165 | false | stratified PSNR tail below threshold |

W&B run ids:

```text
garden: nyp1bofx, yexkhndb, ai4l9wnn, igzwyl46
kitchen: ayeriwna, eg9847k8, tlv5eg2p, nfptbnlk
bonsai: 7q9lf1mf, iu2wjloq, renmthd7, zy7fjqac
counter: zdn5gfs1, 87s357yi, hs74w1jn, aggx5p5k
```

Merged strictpipeline candidate directory:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_regionmasked_corectx_strictcompact_pipeline_merged_20260521/decisions/{scene}_decision.json
```

Fixed v3 portfolio:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v3_strictpipeline.md
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v3_strictpipeline.json
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phase_s_effectaware_region_portfolio_v3_strictpipeline.csv
```

It accepts `5 / 9` scenes and preserves the May 20 effective mean:

```text
dPSNR:  +0.000947740
dSSIM:  +0.000062552
dLPIPS: -0.000098634
balanced: +0.004171458
```

Selected policies:

| scene | selected policy | accepted | effective dPSNR | effective dSSIM | effective dLPIPS |
|---|---|---:|---:|---:|---:|
| bicycle | patchcert_v6 | true | +0.000387192 | +0.000035524 | -0.000115275 |
| flowers | rvregion_corectx_strictpipeline | true | +0.005399704 | +0.000467956 | -0.000586241 |
| garden | rvregion_garden | true | +0.000043869 | -0.000000417 | -0.000000089 |
| stump | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| treehill | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| counter | riskpilot | true | +0.000055313 | +0.000000417 | -0.000001699 |
| kitchen | rvregion_indoor | true | +0.002643585 | +0.000059485 | -0.000184402 |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |

Interpretation:

- The end-to-end strict pipeline reproduces the manual strictcompact
  conclusion, so the May 20 safety fix is no longer only a post-hoc
  re-decision.
- `flowers` is now selected from the strictpipeline end-to-end candidate rather
  than the manually copied strictcompact row.
- The strict compact/tail gate correctly blocks the three bad direct
  core/context rows (`kitchen`, `bonsai`, `counter`) and the over-budget
  `garden` row.
- The final portfolio does not improve beyond the May 20 mean; the bottleneck
  remains effect size and accepted coverage, especially `room`, `stump`,
  `treehill`, and `bonsai`.

Next gate:

1. Do not claim Phase-S as a paper-level closed loop from this evidence alone.
2. Keep v3 as the current audited representation-layer policy because it is
   safer and fairer than the raw core/context gate.
3. The next method work must improve the operator itself or the train-only risk
   predictor enough to add non-noise accepted coverage without test leakage.

## 2026-05-21 Mask-Aware Region Core Ablation

Status: `NOT_COMPLETE_FAILED_ABLATION`. This was a focused follow-up to the
strictpipeline v3 bottleneck. The hypothesis was that bbox-level
render-visible regions were too coarse, so the fitter should see the actual
train residual connected-component mask and weight only those pixels as the
high-confidence region core.

Implemented interfaces:

```text
scripts/car_model/ecsr_build_render_visible_region_carriers.py
  --store_region_masks
  writes mask_shape_hw and mask_rle_counts for each train-only connected component

scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py
  decodes mask RLEs
  applies --region_boundary_px as true mask dilation
  assigns precise mask core / bbox context / outside bins for masked carriers
  rejects malformed RLEs whose run lengths do not exactly match the mask area
```

Subagent review found one real issue before the final run: masked supports were
initially too broad because any face/view with a carrier support could mark
far-away samples as context. That is now fixed: masked carriers keep far-away
samples in the outside bin, use bbox/dilated support as context, and use only
the RLE mask hit as core. The same review found no row-major RLE, bbox-local
shape, or dilation-indexing error after the fix.

Carrier and experiment roots:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/masked_region_carriers_v1_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_maskcore_v1_flowers_counter_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_maskcore_dilated_v1_flowers_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_maskcore_tribin_v1_flowers_20260521
outputs/carnet/meshsplatopt/ecsr_phase_s/render_visible_region_carriers_20260517/phasek_maskcore_tribin_scale050_flowers_20260521
```

Fixed validation scene: `flowers`, iteration `26000`, W&B online, same Phase-K
gate as the strictpipeline replay, held-out test remains report-only. The
`scale050` row is explicitly an uncertified render-trust pilot; it is not a
strict accepted replay because the train-val gate did not accept it.

| variant | accepted | train-val balanced | report-only balanced | train dPSNR | train dSSIM | train dLPIPS | test dPSNR | test dSSIM | test dLPIPS | gate reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| maskcore_v1_exact | false | +0.000045657 | +0.000032067 | +0.000034332 | +0.000000477 | -0.000000089 | -0.000001907 | +0.000000358 | -0.000001341 | compact stratified PSNR tail below threshold |
| maskcore_dilated | false | +0.001345992 | -0.000008464 | -0.000026703 | -0.000038564 | -0.000107199 | +0.000007629 | -0.000001848 | -0.000001043 | PSNR/SSIM train-val regression and compact stratified failures |
| maskcore_tribin | false | +0.001346588 | -0.000012636 | -0.000026703 | -0.000038564 | -0.000107229 | +0.000007629 | -0.000001967 | -0.000000954 | PSNR/SSIM train-val regression and compact stratified failures |
| maskcore_tribin_scale050 | false | +0.001362205 | +0.000014186 | -0.000053406 | -0.000036895 | -0.000107676 | +0.000007629 | -0.000000536 | -0.000000864 | PSNR/SSIM train-val regression and compact stratified failures |

Audit summaries:

```text
maskcore_v1_exact: selected 196 faces, accepted 114 faces, +342 vertices
  fit bins outside/context/core: 3868 / 377 / 1790
  policy-val bins outside/context/core: 154 / 440 / 2493

maskcore_dilated: selected 196 faces, accepted 114 faces, +342 vertices
  fit bins outside/context/core: 3868 / 28 / 2139
  policy-val bins outside/context/core: 154 / 73 / 2860

maskcore_tribin: selected 196 faces, accepted 114 faces, +342 vertices
  fit bins outside/context/core: 3873 / 23 / 2139
  policy-val bins outside/context/core: 174 / 53 / 2860

maskcore_tribin_scale050: materialized 114 faces, +342 vertices, scale 0.5
```

Interpretation:

- The mask-aware interface is useful and now implemented cleanly, but it is not
  a winning operator by itself.
- Exact masks are safer on PSNR/SSIM but shrink the effect to numerical noise
  and still fail the stratified compact gate.
- Mask dilation produces a larger LPIPS improvement, but the same train-val
  render check shows PSNR/SSIM regressions. This is not acceptable for the
  current balanced objective.
- The `scale=0.5` render-trust pilot does not solve the problem; it keeps the
  LPIPS-driven balanced gain but worsens train-val PSNR.
- None of these rows should replace the v3 portfolio. The best current audited
  policy remains `phase_s_effectaware_region_portfolio_v3_strictpipeline`.

Next gate:

1. Stop expanding the mask-core parameter sweep unless a new operator objective
   changes the PSNR/SSIM behavior directly.
2. The next credible method step should be metric-aware during training or
   selection, not only mask-aware sampling. Candidate directions are
   train-val render-trust line search with strict certification, per-carrier
   PSNR/SSIM risk prediction, or a lower-frequency residual basis that cannot
   create small full-frame SSIM drops.
3. Keep the mask RLE support as infrastructure because it may be needed by a
   future operator, but mark the current mask-core ablation as failed evidence.
