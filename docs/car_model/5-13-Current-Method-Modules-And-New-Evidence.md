# Current Method Modules and New Evidence Log

Date: 2026-05-13

This log records the current SPCarNet / MeshSplatOpt method state, what each
module does, how the modules are implemented in the repo, and the latest
quantitative and qualitative comparison evidence.

## Executive Status

The current method should be described as a two-stage system.

1. **Phase-J / MeshSplatOpt is the current strong endpoint against clean
   MeshSplatting.** On the selected full9 evidence set, it wins strict RGB
   metrics on all 9 scenes and also reduces triangle count.
2. **Phase-S / face-local render-calibrated repair is a real new method change,
   but its current gain over Phase-J is still extremely small.** It now has a
   fixed train-only candidate-plan and train-val gate path, but the rendered
   changes are mostly invisible in full-frame images.

This means the honest paper story is currently:

> MeshSplatOpt Phase-J gives the reliable baseline-over-MeshSplatting result.
> Phase-S adds a representation-level repair mechanism with strong safeguards,
> but its current top1/scale2 policy is a low-amplitude stabilization module,
> not yet a visually decisive improvement.

## Module-by-Module Method Detail

| module | role | implementation | current behavior |
|---|---|---|---|
| Clean MeshSplatting baseline | Reference method. Trains and renders the original MeshSplatting representation. | Existing train/render/metrics pipeline and clean checkpoints in `outputs/carnet/meshsplatopt`. | Used only as the baseline row. No repair, no candidate gate, no compact-policy selection. |
| Phase-F compact candidate ladder | Produces compact checkpoints at fixed ratios and evaluates them. | Phase-F outputs under `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix`. | Gives the compact model family that Phase-J and Phase-S build on. |
| Phase-J guarded adaptive-edge + ELA | Main current endpoint. Applies guarded compaction and evidence-lumigraph adaptation, then replays render-calibrated selected models. | Evidence summary: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`. | Strongly beats clean MeshSplatting on the selected full9 set and reduces triangles. |
| Surface evidence cache | Converts train-view render residuals into per-face evidence. It records which faces are hit, how many pixels support each face, and whether the residual direction is consistent across views. | Built by the Phase-K/S runner from evidence roots such as `outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16/{scene}`. | Supplies train-only evidence; no held-out test image is used for selection. |
| Phase-S face-local SH residual planner | Fits a bounded local SH residual for candidate faces, then writes a candidate plan without applying it. | `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py` with `--candidate_plan_out` and `--max_faces_to_apply 0`. | Produces a ranked face plan with train-only certificates. Stump currently has zero candidates under the fixed policy. |
| Candidate certificates | Filters faces by policy-val relative gain, per-face relative gain, view consensus, view-gain certificate, and validation shrink. | Implemented inside `ecsr_apply_surface_residual_facelocal_sh1_delta.py`. Key flags include `--min_face_policy_val_relative_gain`, `--min_face_view_consensus`, `--min_face_gain_certificate_fraction`, and `--validation_shrink_mode face`. | Prevents large unstable residual patches, but also makes the final edit very conservative. |
| Plan materialization | Reads the candidate plan and materializes only a bounded subset. The current fixed policy uses top1 face and scale 2.0. | `ecsr_apply_surface_residual_facelocal_sh1_delta.py` with `--materialize_plan_in`, `--materialize_plan_limit 1`, and `--materialize_plan_scale 2.0`. | Adds one local face residual carrier: triangle count is preserved, while three local vertices are added for the selected face. |
| Train-val render gate | Renders held-out train-val views and accepts the candidate only when the balanced PSNR/SSIM/LPIPS gate passes. | `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py` and `scripts/car_model/ecsr_decide_phasek_trainval_gate.py`. | Selection is train-val only. Test metrics are report-only. Rejected scenes fall back to Phase-J. |
| Phase-S summary collector | Aggregates decision JSON files into a scene table with effective report-only test deltas. | `scripts/car_model/ecsr_collect_facelocal_rendercalib_phase_s_summary.py`. | Current snapshot: 8 present scenes, 6 accepted, 2 rejected, and stump blocked by zero candidates. |
| Coupled render-risk selector | Tests multiple train-only face sets through the real train-val render gate and promotes only meaningful improvements. | `scripts/car_model/ecsr_run_facelocal_coupled_selector.py` and `scripts/car_model/ecsr_collect_facelocal_coupled_selector_summary.py`. | New pilot: 8 candidate scenes, 1 accepted (`counter`), mean effective report-only `+0.000006914 PSNR`, `+0.000000052 SSIM`, `-0.000000212 LPIPS`. |

## Implementation Details

The core new Phase-S path is implemented by adding two capabilities to
`ecsr_apply_surface_residual_facelocal_sh1_delta.py`.

First, plan mode writes a train-only candidate list:

```bash
--candidate_plan_out outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json
--max_faces_to_apply 0
```

Second, materialize mode replays a fixed subset of that plan:

```bash
--materialize_plan_in outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json
--materialize_plan_limit 1
--materialize_plan_scale 2.0
```

The runner forwards these arguments through
`ecsr_run_phasek_barycentric_gate_scene.py`, so the train-val render gate sees
exactly the same fixed policy across scenes. The important design point is that
we are no longer manually selecting scene-specific face counts or scales in the
decision path; the current fair replay policy is top1 / scale2 for every scene.

## Phase-J vs Clean MeshSplatting

This is the clearest positive result in the current repository.

Evidence file:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`

| scene | Phase-J PSNR | Phase-J SSIM | Phase-J LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | triangle reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | 11.81% |
| flowers | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | 11.82% |
| garden | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | 3.47% |
| stump | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | 11.82% |
| treehill | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | 11.81% |
| room | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | 2.10% |
| counter | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | 2.10% |
| kitchen | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | 2.10% |
| bonsai | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | 11.80% |
| **mean** | - | - | - | **+1.331084** | **+0.034702** | **-0.063359** | **7.6479%** |

Conclusion: Phase-J is the current method version that clearly beats the clean
MeshSplatting baseline under the selected full9 protocol.

## Phase-S New Round: Fixed Top1 Scale2 Policy

The new Phase-S round tests a stricter representation-level update on top of
Phase-J. The fixed policy is:

- plan from train-only surface residual evidence;
- select rank-0 candidate face only;
- scale its face-local SH residual by 2.0;
- materialize the face-local carrier;
- accept or reject only with the train-val render gate;
- report held-out test deltas without using them for selection.

Current summary path:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_current/summary.md`

Snapshot status:

- requested scenes: 9
- present decision scenes: 8
- accepted present scenes: 6
- rejected present scenes: 2
- missing decision scenes: 1
- mean effective report-only dPSNR on present scenes: -0.000005245
- mean effective report-only dSSIM on present scenes: -0.000000022
- mean effective report-only dLPIPS on present scenes: +0.000000015

| scene | decision | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | report-only test dSSIM | report-only test dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | accepted | +0.000005722 | +0.000000238 | +0.000000238 | +0.000000000 | +0.000000000 | -0.000000030 | safe but visually negligible |
| flowers | accepted | +0.000009537 | +0.000000119 | +0.000000060 | +0.000003815 | +0.000000060 | +0.000000000 | tiny positive PSNR/SSIM |
| garden | accepted | +0.000017166 | -0.000000060 | +0.000000045 | -0.000020981 | -0.000000477 | +0.000000626 | train gate passes, report-only test regresses slightly |
| treehill | accepted | +0.000047684 | +0.000000298 | +0.000001758 | +0.000001907 | +0.000000358 | -0.000000477 | tiny mixed positive |
| stump | no decision | n/a | n/a | n/a | n/a | n/a | n/a | fixed policy found zero candidate faces |
| room | accepted | +0.000000000 | +0.000000000 | -0.000000060 | +0.000000000 | +0.000000000 | -0.000000060 | exact-tie pass, no visible change |
| counter | accepted | +0.000001907 | +0.000000060 | -0.000000075 | -0.000026703 | -0.000000119 | +0.000000060 | train gate passes, report-only test regresses slightly |
| kitchen | rejected | +0.000000000 | +0.000000000 | +0.000000015 | +0.000009537 | +0.000000000 | +0.000000075 | falls back because train balanced delta is below zero |
| bonsai | rejected | -0.000003815 | +0.000000060 | +0.000000000 | -0.000003815 | +0.000000000 | +0.000000075 | falls back because train PSNR and balanced delta fail |

Conclusion: the fixed Phase-S policy is stable enough to pass several train-val
gates, but the effect size is currently near numerical noise. It should not be
presented as the main performance gain.

## Phase-S Coupled Selector Update

A new coupled render-risk selector was added after the top1/scale2 round:

`docs/car_model/5-13-Coupled-Selector-Pilot.md`

The selector reads the same train-only candidate plan, builds fixed face sets
(`topN` and train-certificate `scoreN`), runs the existing train-val render gate
for each trial, and promotes only trials whose train-val balanced delta reaches
`0.00005`. This prevents numerically tiny edits from becoming a method claim.

Eight candidate-bearing scenes were evaluated. `stump` remains a zero-candidate
fallback.

| scene | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS |
|---|---|---:|---:|---:|---:|
| bicycle | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| flowers | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| treehill | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| counter | score4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 |
| kitchen | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 |
| **mean** | - | 1/8 | **+0.000006914** | **+0.000000052** | **-0.000000212** |

This is a genuine method improvement over top1/s2 on `counter`: top1/s2 had
negative report-only test deltas there, while the coupled score4/s1 set improves
all three metrics. It is still too sparse to be the final paper result.

## Candidate Plan Coverage

| scene | candidate count | status |
|---|---:|---|
| bicycle | 7 | replay accepted |
| flowers | 35 | replay accepted |
| garden | 110 | replay accepted, report-only test slightly negative |
| stump | 0 | blocked by fixed certificates |
| treehill | 84 | replay accepted |
| room | 76 | replay accepted, no visible change |
| counter | 127 | replay accepted, report-only test slightly negative |
| kitchen | 145 | replay rejected |
| bonsai | 1266 | replay rejected |

This table is important because it shows the current bottleneck. The certificates
can identify plausible residual faces in most scenes, but materializing only one
face is too conservative to create visible improvement. Materializing multiple
faces previously caused LPIPS-balanced regressions on bicycle, so the next real
research step should improve the multi-face carrier or coupled selector rather
than only changing thresholds.

## Qualitative Comparison

Qualitative assets were generated under:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative`

Each image contains:

1. Phase-J baseline render;
2. Phase-S top1/scale2 render;
3. GT when available;
4. absolute Phase-S minus Phase-J render difference amplified by 80x.

The x80 diff panel is necessary because the full-frame Phase-S edit is usually
too small for human inspection.

### Bicycle

![bicycle Phase-J vs Phase-S](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/bicycle_standard_00000_phasej_vs_phases_x80diff.png)

- common test views: 25
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000073
- max absolute channel difference on 0-255 scale: 10
- selected view PSNR delta: +0.000022821

### Flowers

![flowers Phase-J vs Phase-S](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/flowers_standard_00000_phasej_vs_phases_x80diff.png)

- common test views: 22
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000108
- max absolute channel difference on 0-255 scale: 10
- selected view PSNR delta: +0.000067204

### Garden

![garden Phase-J vs Phase-S standard](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/garden_standard_00000_phasej_vs_phases_x80diff.png)

![garden Phase-J vs Phase-S maxdiff](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/garden_maxdiff_00013_phasej_vs_phases_x80diff.png)

- common test views: 24
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000521
- max absolute channel difference on 0-255 scale: 19
- max-diff selected view PSNR delta: -0.000245720

### Treehill

![treehill Phase-J vs Phase-S standard](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/treehill_standard_00000_phasej_vs_phases_x80diff.png)

![treehill Phase-J vs Phase-S maxdiff](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/treehill_maxdiff_00010_phasej_vs_phases_x80diff.png)

- common test views: 18
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000957
- max absolute channel difference on 0-255 scale: 38
- max-diff selected view PSNR delta: -0.000189849

### Room

![room Phase-J vs Phase-S](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/room_standard_00000_phasej_vs_phases_x80diff.png)

- accepted by train-val gate: true
- common test views: 39
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000021
- max absolute channel difference on 0-255 scale: 2
- selected view PSNR delta: +0.000010203

### Counter

![counter Phase-J vs Phase-S standard](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/counter_standard_00000_phasej_vs_phases_x80diff.png)

![counter Phase-J vs Phase-S maxdiff](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/counter_maxdiff_00026_phasej_vs_phases_x80diff.png)

- accepted by train-val gate: true
- common test views: 30
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000036
- max absolute channel difference on 0-255 scale: 7
- max-diff selected view PSNR delta: -0.000577692

### Kitchen

![kitchen Phase-J vs Phase-S](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/kitchen_standard_00000_phasej_vs_phases_x80diff.png)

- accepted by train-val gate: false
- selected output: fallback to Phase-J
- common test views: 35
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000052
- max absolute channel difference on 0-255 scale: 9
- selected view PSNR delta: +0.000135679

### Bonsai

![bonsai Phase-J vs Phase-S standard](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/bonsai_standard_00000_phasej_vs_phases_x80diff.png)

![bonsai Phase-J vs Phase-S maxdiff](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/bonsai_maxdiff_00034_phasej_vs_phases_x80diff.png)

- accepted by train-val gate: false
- selected output: fallback to Phase-J
- common test views: 37
- max Phase-S vs Phase-J model MAE on 0-255 scale: 0.000038
- max absolute channel difference on 0-255 scale: 6
- max-diff selected view PSNR delta: -0.000085153

Qualitative conclusion: Phase-J vs clean MeshSplatting is the comparison that
can support a visible method story. Phase-S vs Phase-J currently requires
amplified difference maps to be seen, so it is not yet a strong qualitative
claim.

## Commands and Artifacts

Representative Phase-S plan command:

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/kitchen/ratio_0200/compact_model \
  --evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16/kitchen \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/kitchen/plan_model \
  --iteration 26000 \
  --max_faces_to_apply 0 \
  --candidate_plan_out outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/kitchen/facelocal_sh3_candidate_plan.json \
  --device cuda
```

Representative Phase-S replay command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --scenes kitchen \
  --gpu 7 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_kitchen \
  --delta_operator facelocal_sh1 \
  --delta_facelocal_materialize_plan_in 'outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json' \
  --delta_facelocal_materialize_plan_limit 1 \
  --delta_facelocal_materialize_plan_scale 2.0 \
  --gate_min_balanced_delta 0.0 \
  --wandb_group phase_s_facelocal_rendercalib_v1_top1_s2_full9_20260513 \
  --wandb_name facelocal_rendercalib_v1_top1_s2_kitchen
```

Generated artifacts:

- Phase-J full9 baseline comparison:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`
- Phase-S current summary:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_current/summary.md`
- Phase-S qualitative summary:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_qualitative/qualitative_summary.md`
- Decision JSON examples:
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_bicycle/decisions/bicycle_decision.json`
  and
  `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_treehill/decisions/treehill_decision.json`

## Honest Weaknesses

- Phase-S top1/scale2 is currently too weak to be visually compelling.
- The fixed policy passes train-val gates by making very small edits.
- Garden shows a train-val pass but a report-only test regression, which means
  the current selector is not yet a reliable quality booster.
- Stump has zero candidate faces under the current certificates.
- Multi-face materialization remains the central unresolved problem: more faces
  are needed for visible improvement, but naive top-N materialization has already
  shown LPIPS-balanced regressions.

## Next Research Direction

The next method upgrade should target coupled multi-face selection instead of
more single-face threshold tuning. A credible next step is:

1. keep the train-only candidate plan;
2. evaluate small coupled face sets with the actual render gate;
3. penalize LPIPS-sensitive changes explicitly;
4. require a visible or at least measurable minimum render delta before claiming
   a new Phase-S improvement;
5. keep Phase-J as the fallback endpoint whenever Phase-S fails.

Until that upgrade produces a larger cross-scene report-only margin, Phase-S
should be presented as a principled representation-level repair framework under
active development, not as the final source of the paper's main metric gains.
