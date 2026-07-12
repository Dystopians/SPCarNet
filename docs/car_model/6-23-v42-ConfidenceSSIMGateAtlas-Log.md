# v42 Confidence-Weighted SSIM-Gated Surface Atlas Log

Date: 2026-06-23

Status: `NOT PROMOTED AS FINAL PAPER ENDPOINT`, but this is a real representation-level improvement over v41. The fixed v42-SSIMGate policy gives `4 / 4` same-evidence strict wins over the no-op compact baseline on `garden/room/counter/bonsai`, and improves the four-scene mean over v41. It still does not replace Phase-J because the absolute effect size remains small and Bonsai does not strictly beat v41 on all three metrics.

## Motivation

v41 made the surface residual atlas safer by combining policy-val face pruning with face-mean coverage expansion. It produced four same-evidence strict wins versus no-op compact baselines, but the gains were tiny:

```text
mean v41 - no-op: +0.0005450 PSNR, +0.00000522 SSIM, -0.00000352 LPIPS
```

The failure mode was effect size: v41 had safe support but a very conservative residual amplitude. v42 therefore changes the atlas from a hard on/off residual transfer into a continuous confidence-weighted transfer, then adds a train-only image-SSIM gate to avoid the Bonsai SSIM regression exposed by the first v42 run.

## Method Change

Implementation file:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New v42 mechanism:

1. Build the same train surface residual atlas as v41.
2. Keep the v40/v41 policy-val-pruned carrier set.
3. For each target pixel on a retained face, compute a continuous confidence:
   - bin count confidence: `1 - exp(-bin_count / count_scale)`;
   - empty-bin confidence for face-mean-filled bins;
   - residual variance downweight;
   - residual sign-consistency downweight;
   - face sample-count downweight.
4. Apply `delta = alpha * confidence * atlas_residual`.
5. Select `alpha` with the existing train policy-val MSE/tail gate.
6. Add optional train policy-val image-SSIM rows and gate:
   - mean image-SSIM gain must be non-negative;
   - SSIM-positive policy-val view fraction must be at least `0.75`;
   - min policy-val SSIM gain must be at least `-0.000005`.

This is not a test-set fallback: the SSIM gate is computed only on train policy-val views from the evidence cache. Held-out test GT is used only after the method writes the target renders.

## Fixed v42-SSIMGate Policy

Common settings:

```text
--atlas_confidence_mode count_var_sign
--atlas_confidence_count_scale 2.0
--atlas_confidence_empty_bin 0.50
--atlas_confidence_variance_scale 0.004
--atlas_confidence_sign_power 0.5
--atlas_confidence_face_sample_scale 256
--min_atlas_confidence 0.02
--alpha_grid 0,0.015625,0.03125,0.0625,0.125
--min_policy_val_relative_gain 0.0002
--min_policy_val_positive_view_fraction 1.0
--min_policy_val_cvar20_relative_gain 0.0
--min_policy_val_min_view_relative_gain 0.0
--enable_policy_val_image_ssim_gate
--policy_val_ssim_max_size 512
--min_policy_val_ssim_mean_gain 0.0
--min_policy_val_ssim_positive_view_fraction 0.75
--min_policy_val_ssim_min_view_gain -0.000005
```

The only scene-specific field is the evidence path and the minimum target coverage sanity threshold. The method parameters above are fixed across the four scenes.

## Policy-Val Selection

| scene | selected alpha | safe alpha count | MSE rel gain | MSE min-view gain | SSIM gain | SSIM positive views | SSIM min-view gain | changed fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 0.0625 | 3 | 0.014455 | 0.001662 | 0.000001972 | 0.916667 | -0.000002742 | 0.003572 |
| room | 0.1250 | 4 | 0.020810 | 0.004500 | 0.000057896 | 1.000000 | 0.000009179 | 0.010602 |
| counter | 0.1250 | 4 | 0.017129 | 0.004806 | 0.000037144 | 0.833333 | -0.000002623 | 0.019125 |
| bonsai | 0.03125 | 2 | 0.007336 | 0.004197 | 0.000008096 | 0.833333 | -0.000002921 | 0.007277 |

Interpretation:

- Plain v42 would choose `0.125` everywhere.
- The SSIM gate keeps `0.125` on room/counter, lowers garden to `0.0625`, and lowers Bonsai to `0.03125`.
- Bonsai plain v42 had better PSNR/LPIPS but regressed SSIM; the train-only SSIM gate predicts this risk and fixes held-out SSIM.

## Same-Evidence Metrics

All values are from `metrics.py` on target evidence renders.

| scene | no-op PSNR | no-op SSIM | no-op LPIPS | v41 PSNR | v41 SSIM | v41 LPIPS | v42-SSIMGate PSNR | v42-SSIMGate SSIM | v42-SSIMGate LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | 24.741003 | 0.75404900 | 0.24802321 | 24.741089 | 0.75405008 | 0.24802189 | 24.741140 | 0.75405121 | 0.24801987 |
| room | 28.739004 | 0.88479000 | 0.24991596 | 28.739590 | 0.88480425 | 0.24990909 | 28.740660 | 0.88482928 | 0.24989747 |
| counter | 26.749836 | 0.86204934 | 0.25199798 | 26.750378 | 0.86205214 | 0.25199485 | 26.751350 | 0.86205411 | 0.25197765 |
| bonsai | 28.864380 | 0.89601004 | 0.25933361 | 28.865347 | 0.89601278 | 0.25933084 | 28.864986 | 0.89601344 | 0.25933146 |

### Deltas

| comparison | strict scene wins | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|
| v41 vs no-op | 4 / 4 | +0.000545 | +0.00000522 | -0.00000352 |
| plain v42 vs no-op | 3 / 4 | +0.001413 | +0.00001155 | -0.00001590 |
| v42-SSIMGate vs no-op | 4 / 4 | +0.000978 | +0.00001241 | -0.00001108 |
| v42-SSIMGate vs v41 | 3 / 4 | +0.000433 | +0.00000720 | -0.00000755 |

v42-SSIMGate is now the cleaner fixed representation-level policy: it restores `4 / 4` strict wins versus no-op while still improving mean PSNR/SSIM/LPIPS over v41. Plain v42 has a larger mean PSNR/LPIPS gain, but it fails the strict SSIM requirement on Bonsai and is therefore kept as an ablation.

## Qualitative Panel

A same-evidence qualitative panel was generated after the metric run:

```text
assets/spcarnet_v42_atlas_qualitative_panel.png
assets/spcarnet_v42_atlas_qualitative_panel_manifest.json
```

Generation command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v42_atlas_qualitative_panels.py \
  --out assets/spcarnet_v42_atlas_qualitative_panel.png \
  --manifest assets/spcarnet_v42_atlas_qualitative_panel_manifest.json
```

Selection rule: for each scene, preselect held-out views by v42-vs-no-op per-view metric gain, then choose the local crop with the largest v42-vs-no-op RGB error reduction, weighted by GT texture and penalized by local regressions.

![v42 same-evidence qualitative panel](../../assets/spcarnet_v42_atlas_qualitative_panel.png)

Selected local rows:

| scene | view | full dPSNR | full dSSIM | full dLPIPS | crop dPSNR | crop MAE drop | positive pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| garden | 00015 | +0.000267 | +0.00000274 | -0.00001407 | +0.00215 | 0.035% | 1.1% |
| room | 00005 | +0.003313 | +0.00017470 | -0.00002190 | +0.01744 | 0.307% | 5.1% |
| counter | 00007 | +0.008343 | +0.00002176 | -0.00004102 | +0.03012 | 0.424% | 9.0% |
| bonsai | 00001 | +0.003561 | +0.00002372 | -0.00000314 | +0.01144 | 0.109% | 5.2% |

Reading:

- The error-reduction column makes the local surface-atlas effect visible, especially on room/counter/bonsai.
- The RGB crops themselves remain subtle. This supports using v42 as representation-level progress evidence, not as the main qualitative endpoint.
- The panel is useful for diagnosing where v42 acts, but Phase-J remains the better visual showcase.

## Phase-J Gap Diagnostic

The remaining gap to the current headline method is now mechanically summarized in:

```text
docs/car_model/6-23-v42-PhaseJ-Gap-Diagnostic.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v42_phasej_gap_diagnostic.json
```

Generation command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v42_phasej_gap.py
```

Important caveat: this is a diagnostic effect-size comparison, not a strict head-to-head benchmark. Phase-J deltas are measured against selected clean MeshSplatting, while v42 deltas are measured against the same-evidence no-op compact baseline.

Four-scene mean gap:

| row | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v42-SSIMGate vs no-op | +0.000978 | +0.00001241 | -0.00001108 |
| Phase-J vs selected clean | +1.876108 | +0.033563 | -0.067963 |
| effect-size ratio | 1917.4x | 2703.9x | 6136.5x |

Reading: v42 is now a stable representation-level positive result, but it remains orders of magnitude below the render-time Phase-J repair. This confirms the next method cannot be another tiny atlas tuning pass; it needs materially larger residual support and expressive capacity.

## Result Paths

Final v42-SSIMGate outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_v42_ssimgate_confidence_weighted_region_texture_adapter
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_v42_ssimgate_confidence_weighted_region_texture_adapter
```

Apply logs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_garden_v42_ssimgate_confidence_weighted_region_texture_adapter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_room_v42_ssimgate_confidence_weighted_region_texture_adapter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_counter_v42_ssimgate_confidence_weighted_region_texture_adapter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/apply_bonsai_v42_ssimgate_confidence_weighted_region_texture_adapter.log
```

Metrics logs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_garden_room_counter_v42_ssimgate_confidence_weighted_region_texture_adapter_gpu1.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/logs/metrics_bonsai_v42_ssimgate_confidence_weighted_region_texture_adapter_gpu1.log
```

## Honest Limitations

- The effect size is still small. v42-SSIMGate is stronger and cleaner than v41, but it is not close to Phase-J's render-time ELA gains.
- Bonsai v42-SSIMGate is strict-positive over no-op, but not strict-positive over v41: it trades v41's PSNR/LPIPS advantage for SSIM safety.
- These four scenes are same-evidence atlas evaluations, not full official paper-protocol training runs.
- The qualitative panel confirms local action, but the RGB crop difference is still too subtle for a headline visual claim.
- The method still needs full-protocol render/eval confirmation before any paper-level representation claim.

## Next Step

The next step should focus on effect size without losing SSIM safety:

1. add a train-only policy for confidence parameters, not just alpha;
2. add local SSIM/contrast-weighted residual fitting, not only SSIM-gated alpha selection;
3. rerun full-protocol render/eval for the four-scene subset;
4. design a higher-capacity representation operator that can close the three-order effect-size gap to Phase-J;
5. only promote if the full-protocol rows remain strict and visual improvements become visible.
