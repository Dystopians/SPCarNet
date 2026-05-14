# SPCarNet / MeshSplatOpt Method Modules and Evidence Log

Date: 2026-05-14

This log is the current compact handoff for the method itself: what each module
does, where it is implemented, what the latest quantitative evidence says, and
what qualitative comparisons can be shown without exaggerating the result.

## Current Bottom Line

The current paper-facing method is best described as:

```text
Clean MeshSplatting
  -> Phase-F compact candidate ladder
  -> Phase-J guarded adaptive-edge + ELA replay
  -> optional Phase-S face-local coupled repair with train-val-only promotion
```

The strong result is still Phase-J versus clean MeshSplatting. Phase-S is now a
real train/eval pipeline module with a fixed candidate policy, a coupled
face-set selector, a risk-greedy selector, and a tail-stable promotion rule. The
latest full8 risk-tail replay accepts `3 / 8` candidate-bearing scenes. The
follow-up GeoRisk/CVaR replay adds geometry-neighborhood ranking and train-val
CVaR diagnostics, but accepts only `2 / 7` requested hard/control scenes and
does not improve coverage over risk-tail. Logs:
`docs/car_model/5-14-PhaseS-RiskTail-Alpha-ModuleLog.md` and
`docs/car_model/5-14-PhaseS-GeoRiskCVaR-Selector-Log.md`.

## Module Map

| module | purpose | implementation | evidence status |
|---|---|---|---|
| Clean MeshSplatting baseline | Fair reference checkpoint/render/eval row. | Existing training, rendering, and metrics pipeline. | Baseline for all deltas. |
| Phase-F compact ladder | Creates candidate compact representations at fixed policy points. | Outputs under `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix`. | Supplies compact models for Phase-J. |
| Phase-J guarded adaptive-edge + ELA | Main reliable improvement over clean MeshSplatting. | Full9 summary: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`. | Wins RGB metrics on all selected full9 scenes and reduces triangles. |
| Surface evidence cache | Converts train-view residuals into per-face support, consistency, and view certificates. | Evidence root such as `outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16`. | Train-only evidence; no held-out test selection. |
| Phase-S face-local residual planner | Fits bounded SH residuals to candidate faces and writes a plan without applying it. | `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`. | Finds candidates in 8/9 strict scenes; relaxed stump discovery finds 2 candidates but no accepted repair. |
| Phase-S top1/s2 replay | Fixed minimal repair policy: take rank-0 face and scale residual by 2.0. | `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py` with `--delta_facelocal_materialize_plan_limit 1` and `--delta_facelocal_materialize_plan_scale 2.0`. | Safe but nearly inert; mean effective report-only deltas are around numerical noise. |
| Coupled score selector | Tests train-only multi-face sets through the real train-val render gate. | `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`, `scoreN` mode. | Fixes `counter`, but accepts only 1/8 candidate-bearing scenes. |
| Risk-greedy selector | Greedily selects multi-face sets while penalizing redundant view support and similar residual directions. | `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`, `riskN` mode. | Full8 risk-tail replay accepts `flowers`, `counter`, and `treehill`; rejected scenes fall back to Phase-J. |
| GeoRisk/CVaR selector | Adds geometry-neighborhood redundancy, per-face train-certificate tail risk, local residual concentration, and train-val render CVaR diagnostics. | `scripts/car_model/ecsr_run_facelocal_coupled_selector.py`, `georiskN` mode. | Requested 7-scene replay accepts `flowers` and `counter`; useful audit upgrade but no coverage gain over risk-tail. |
| Per-face alpha refit | Fits train-only scalar multipliers for selected face-local residuals and passes them to materialization. | `scripts/car_model/ecsr_fit_facelocal_plan_alphas.py`; materializer arg `--materialize_plan_alpha_json`. | Interface complete, but first 3-scene pilot does not improve over uniform risk-tail. |
| Train-val gate and fallback | Accepts repairs only with train-val metrics; test remains report-only. Rejected scenes fall back to Phase-J. | `scripts/car_model/ecsr_decide_phasek_trainval_gate.py` and coupled selector decision JSONs. | Keeps the effective method from being harmed by risky Phase-S edits. |

## How the New Phase-S Selectors Work

All Phase-S selectors start from the same train-only candidate plan:

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json
```

`topN` preserves plan rank. `scoreN` ranks by a train-only certificate:

```text
relative_gain
* validation_shrink
* view_consensus
* beneficial_view_fraction
* min_view_gain_term
* consistency
* log1p(policy_val_samples)
* log1p(face_pixels)
* sqrt(view_hits)
```

`riskN` keeps that score but adds a pairwise set-construction penalty:

```text
pair_risk(i, j) =
  view_support_overlap(i, j)
  * (0.5 + 0.5 * abs(cosine(delta_coeff_i, delta_coeff_j)))
```

`georiskN` keeps the same no-test selection boundary and adds:

```text
geometry adjacency penalty from source checkpoint triangle indices
per-face lower-tail/CVaR risk from train certificates
local residual concentration bonus from train evidence
trial-level train-val render CVaR diagnostics for outer promotion auditing
```

The selector then materializes each fixed face set, runs the existing render
gate, and promotes only if:

```text
inner train-val gate passes
and train-val balanced delta >= 0.00005
```

Held-out test metrics are saved only after selection, as report-only evidence.

## Quantitative Evidence

### Phase-J vs Clean MeshSplatting

Evidence:

`outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`

| protocol | scenes | strict RGB wins | mean dPSNR | mean dSSIM | mean dLPIPS | mean triangle reduction |
|---|---:|---:|---:|---:|---:|---:|
| Phase-J vs clean MeshSplatting | 9 | 9/9 | +1.331084 | +0.034702 | -0.063359 | 7.6479% |

This is the strongest current result. It is the row that clearly beats the
original MeshSplatting baseline on the selected full9 evidence set.

### Phase-S Top1/s2 on Top of Phase-J

Evidence:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_rendercalib_v1_top1_s2_fairreplay_20260513_current/summary.md`

| protocol | present scenes | accepted | mean effective dPSNR | mean effective dSSIM | mean effective dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---|
| top1/s2 fixed replay | 8 | 6 | -0.000005245 | -0.000000022 | +0.000000015 | too small; can regress `garden` and `counter` report-only test |

Top1/s2 is a useful safety baseline, not a visually compelling repair method.

### Coupled Score Selector

Evidence:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_summary/summary_8candidate_scenes.md`

| protocol | candidate scenes | accepted | mean effective dPSNR | mean effective dSSIM | mean effective dLPIPS | reading |
|---|---:|---:|---:|---:|---:|---|
| score/top coupled selector | 8 | 1 | +0.000006914 | +0.000000052 | -0.000000212 | fixes `counter`, rejects weak/risky edits elsewhere |

The important counter contrast:

| counter method | report-only dPSNR | report-only dSSIM | report-only dLPIPS |
|---|---:|---:|---:|
| top1/s2 | -0.000026703 | -0.000000119 | +0.000000060 |
| coupled score4/s1 | +0.000055313 | +0.000000417 | -0.000001699 |

### Risk-Tail Selector Full8

Evidence:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_summary/summary_8candidate_tailstable.md`

| scene | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---|---:|---:|---:|---:|---|
| bicycle | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | too small or train-val negative |
| flowers | risk4/s1 | true | +0.005418777 | +0.000470877 | -0.000586182 | strongest Phase-S win |
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | stricter tail threshold rejects exposed false-positive |
| treehill | top1/s2 | true | +0.000001907 | +0.000000358 | -0.000000477 | tiny but safe |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best trial below threshold |
| counter | risk4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | accepted; all three metrics improve over Phase-J |
| kitchen | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best trial below threshold |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val negative |
| **mean** | - | 3/8 | **+0.000684500** | **+0.000058956** | **-0.000073545** | sparse; dominated by flowers |

This improves the earlier 1/3 pilot, but it is still not a complete Phase-S
paper endpoint because most scenes fall back and the mean is dominated by one
strong outdoor scene.

### GeoRisk/CVaR Selector Replay

Evidence:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_summary/summary_7scene.md`

| scene | selected | accepted | effective dPSNR | effective dSSIM | effective dLPIPS | reading |
|---|---|---:|---:|---:|---:|---|
| garden | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | false-positive trials rejected |
| bicycle | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | georisk trials fail gate |
| room | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | best trial too small |
| kitchen | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | PSNR positive but not robust enough |
| bonsai | Phase-J fallback | false | +0.000000000 | +0.000000000 | +0.000000000 | train-val negative |
| flowers | georisk4/s1 | true | +0.005418777 | +0.000470877 | -0.000586182 | same strong positive as risk-tail |
| counter | georisk4/s1 | true | +0.000055313 | +0.000000417 | -0.000001699 | same all-metric positive as risk-tail |
| **mean** | - | 2/7 | **+0.000782013** | **+0.000067328** | **-0.000083983** | positive but still dominated by flowers |

This is an audit/policy improvement rather than a new performance milestone.
It confirms that geometry-aware ranking alone does not solve the hard scenes.

### Alpha Refit and Stump Relaxed Checks

Evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_alpha_refit_v1_pilot_20260514_summary/summary_3scene.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_zero_candidate_relaxed_v1_selector_20260514_summary/summary.md`

Alpha refit is wired end to end, but the first `counter/garden/bicycle` pilot
does not improve over uniform risk-tail. Relaxed stump discovery finds two
candidates, but the selector rejects all trials. Both are kept as measured
negative results.

## Qualitative Evidence

The clearest current Phase-S qualitative example is now `flowers`, followed by
the local `counter` fix and a tiny `treehill` safe edit. New automatically
selected panels are stored at:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/qualitative_panel_manifest.json`

![flowers risk4 panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/flowers_risk4_s1_00019_phasej_vs_risktail_panel.png)

![counter risk4 panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_riskpilot_20260513_qualitative/counter_risk4_s1_00002_phasej_vs_risktail_panel.png)

GeoRisk/CVaR panels with local crops and error-change maps:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/qualitative_summary.md`

![flowers georisk panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/flowers_georisk4_s1_00019_georisk_cvar_panel.png)

![garden rejected panel](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_georisk_cvar_v1_20260514_qualitative/garden_georisk8_s0p5_00006_georisk_cvar_panel.png)

Counter coupled qualitative summary:

`outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_qualitative_summary.md`

![counter view 00002](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_00002_phasej_top1_coupled_x120diff.png)

![counter view 00026](../../outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_pilot_20260513_qualitative/counter_00026_phasej_top1_coupled_x120diff.png)

| view | top1 dPSNR | coupled/risk4 dPSNR | coupled-vs-Phase-J MAE | max abs |
|---|---:|---:|---:|---:|
| 00002 | -0.000003491 | +0.000518573 | 0.000310 | 14 |
| 00026 | -0.000577692 | +0.000369517 | 0.001723 | 20 |
| 00000 | +0.000002077 | +0.000289366 | 0.001078 | 10 |

The qualitative improvement is local and subtle. The x120 difference panel is
needed for human inspection. This is useful for a method-development slide, but
not yet a strong final-paper visual claim.

## Honest Weaknesses

- Phase-J is strong against clean MeshSplatting, but Phase-S adds only sparse
  extra gains right now.
- Full-frame visual improvements are hard to see without amplified difference
  maps.
- `stump` now has two relaxed candidates, but none are accepted.
- `garden` remains a warning case: risk selection reduces the report-only PSNR
  regression magnitude but does not remove the underlying train/test mismatch.
- `bicycle` has only seven current strict candidates; multi-face repairs fail train-val
  or remain too small.
- GeoRisk/CVaR adds geometry adjacency and train-val CVaR diagnostics, but the
  result shows the remaining bottleneck is carrier capacity/evidence quality
  rather than only selector scoring.

## Next Required Evidence

The full8 closure is now complete for the fixed risk-tail trial set. The next
work is not more reporting; it is a stronger representation operator or a
better train-only risk predictor that can broaden acceptance beyond
`flowers/counter/treehill` without reopening the `garden` false-positive.
