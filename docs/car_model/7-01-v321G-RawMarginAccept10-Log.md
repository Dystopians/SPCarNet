# v321G Raw-Margin Accept10 Reliability Milestone

Date: 2026-07-01

## Short Answer

The reflection became useful only after it was converted into an explicit
protocol audit. The useful lesson was:

> Do not let a new calibrated/risk branch replace a proven incumbent unless the
> source-only evidence has enough support and does not violate fixed-scene
> safety.

v321G is therefore a reliability-policy upgrade, not a new per-scene parameter
search. It preserves v319c on every non-room scene and improves room.

Status remains **not paper-complete** because the improvement is still
concentrated in one scene and the visual gains are subtle in full-frame views.

## Implemented Method Change

File:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Main changes:

- added calibrated lower-bound diagnostics for source reliability;
- added raw-incumbent mode so calibrated LCB can supplement but not overwrite a
  successful raw incumbent decision;
- fixed the auto-margin search so raw-incumbent mode selects the margin from
  raw source predictions rather than from the calibrated override path;
- required the fixed-scene risk model to be source-safe before it can override;
- added a fixed-scene source SSIM guard to prevent stump-like collapses;
- restored a 10% source-reliability accept support floor for the final v321G
  policy, preventing low-support margin overfit on bonsai.

The key code-level correction after v321E/F was:

```text
source_reliability auto-margin selection = raw-only
final target decision = raw incumbent first, calibrated LCB only as optional fallback
source_reliability_min_accept_fraction = 0.10
```

## Negative Evidence That Drove the Fix

v320 q50/s05 showed that simply replacing raw reliability with calibrated LCB
was unsafe:

- mean PSNR dropped from v319c `+0.269725` to `+0.268286`;
- safe scenes dropped from `9/9` to `8/9`;
- stump failed because a low-support learned/risk decision overrode fixed.

v321E fixed stump and improved room, but had a small bonsai regression:

- bonsai view `00005` changed from v319c `hybrid` to `learned`;
- scene PSNR dropped by `-0.000299`;
- cause: source reliability selected a high objective margin with only `9.09%`
  source accept support.

v321G fixes this by enforcing a `0.10` source accept floor.

## Full9 Apply Metrics

Source:

```text
docs/car_model/results/v321g_full9_apply_metrics_vs_prior_summary.json
outputs/carnet/spcarnet_v321g_rawmargin_accept10_full9_20260701
```

| method | mean PSNR gain | mean SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v320q50s05 | +0.268286 | +0.003706 | +0.014003 | +0.039547 | 8 | 8/9 |
| v321E | +0.270871 | +0.003725 | +0.014301 | +0.039726 | 8 | 9/9 |
| v321G | +0.271248 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |

v321G relative to v319c:

- mean PSNR gain: `+0.001523`;
- mean SSIM gain: `+0.00000664`;
- mean min PSNR: tie;
- mean CVaR10 PSNR: tie;
- negative views: tie;
- safe scenes: tie, `9/9`.

Per-scene v321G vs v319c:

| scene | PSNR delta | SSIM delta | safe |
|---|---:|---:|---:|
| bicycle | +0.000000 | +0.000000 | true |
| bonsai | +0.000000 | +0.000000 | true |
| counter | +0.000000 | +0.000000 | true |
| flowers | +0.000000 | +0.000000 | true |
| garden | +0.000000 | +0.000000 | true |
| kitchen | +0.000000 | +0.000000 | true |
| room | +0.013706 | +0.0000597 | true |
| stump | +0.000000 | +0.000000 | true |
| treehill | +0.000000 | +0.000000 | true |

## Clean-MeshSplatting Frontier Metrics

Source:

```text
docs/car_model/results/v321g_frontier_lpips_qualitative_summary.json
docs/car_model/results/v321g_frontier_lpips_qualitative_summary.md
docs/car_model/results/v321g_frontier_panels/
outputs/carnet/spcarnet_v321g_frontier_comparison_full9_20260701
```

| method | scenes | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v319c | 9 | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v321G | 9 | 27.586900 | 0.028173 | 0.087736 | 0.057660 |

v321G is better than v319c on all four frontier metrics:

- PSNR: `+0.003257`;
- MAE: `-0.00000719`;
- LPIPS: `-0.0000102`;
- DISTS: `-0.0000184`.

Qualitative panels:

```text
docs/car_model/results/v321g_frontier_panels/bonsai/00001_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/bonsai/00035_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/flowers/00010_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/flowers/00014_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/garden/00006_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/garden/00017_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/room/00004_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/room/00009_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/treehill/00001_frontier_panel.png
docs/car_model/results/v321g_frontier_panels/treehill/00013_frontier_panel.png
```

## Commands And Evidence Paths

Representative full9 apply command pattern:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v321g_rawmargin_accept10_full9_20260701/<scene> \
  --output_variant source_heldout_auto \
  --enable_per_view_knn_policy --per_view_knn_k 3 --per_view_knn_min_score_delta_vs_scene 0.0005 \
  --enable_per_view_risk_model_policy --per_view_risk_model_require_source_safe \
  --per_view_risk_model_require_predicted_scene_axis_nonregression \
  --enable_source_reliability_policy --source_reliability_min_accept_fraction 0.10 \
  --source_reliability_enable_calibrated_lcb --source_reliability_calibrated_lcb_mode raw_incumbent \
  --source_reliability_calibration_quantile 0.5 --source_reliability_calibration_scale 0.5 \
  --compute_ssim --enable_wandb
```

Frontier command:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/build_support_transport_frontier_comparison.py \
  --method clean26000=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method v319c=outputs/carnet/spcarnet_v319c_incumbent_reliability_full9_20260701 \
  --method v321g=outputs/carnet/spcarnet_v321g_rawmargin_accept10_full9_20260701 \
  --output_dir outputs/carnet/spcarnet_v321g_frontier_comparison_full9_20260701 \
  --scenes bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill \
  --panel_scenes room,bonsai,treehill,garden,flowers \
  --max_panels_per_scene 2 --lpips_max_side 512 --panel_max_side 640 \
  --crop_size 256 --device cuda --enable_wandb
```

Failed command lesson:

- `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>` is
  not a valid `base_model_path` for apply; it lacks
  `train/ours_26000_phasef_extra_compact_base/renders`.
- Correct apply input root is
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model`.

## Honest Weaknesses

- v321G is the best current engineering incumbent, but the improvement over
  v319c is concentrated in `room`.
- Tail metrics and negative-view counts are preserved, not improved.
- Full-frame visual differences remain subtle; the best qualitative story still
  needs crops/error maps plus geometry-complexity evidence.
- This is not yet a 100% top-conference closed loop. It is a stronger,
  cleaner, more reliable milestone.

## Verdict

The answer to "did reflection work?" is:

```text
Yes, but only after it became a concrete no-regression audit.
```

The answer to "is the project finished?" is:

```text
Final status: NOT COMPLETE.
```
