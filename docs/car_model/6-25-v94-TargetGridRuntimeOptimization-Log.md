# v94 Target-Grid Runtime Optimization Log

Date: 2026-06-25  
Status: kept as a small exact runtime cleanup; not a deployment-speed solution

## What Changed

The kept code path is intentionally narrow:

- `compute_evidence_signal()` now builds the target world-space backprojection grid once per target frame.
- `warp_support_residual()` accepts that precomputed grid and reuses it for each selected support frame.
- `--evidence_max_side` is exposed in the apply and benchmark scripts, but remains an approximate experimental knob. The promoted/default path is still `--evidence_max_side 0`.
- negative `--evidence_max_side` values are rejected in the apply and benchmark CLIs.

This is an implementation/runtime optimization. It does not change Phase-J policy selection, support selection, residual transfer, gates, alpha maps, rendered images, or metrics in full-resolution mode.

## Failed Runtime Variants

Two more aggressive exact-looking variants were tested and rejected:

| variant | result | decision |
|---|---:|---|
| target-grid reuse + fused residual/depth `grid_sample` + residual cache | full9 weighted `951.410896 -> 956.733116 ms/view`; adapter `913.855245 -> 919.261038 ms/view`; max alloc `17703.596 -> 19481.500 MiB` | rejected; slower and higher memory |
| batch support warp | stump4 `650.008579 -> 664.360456 ms/frame`; peak alloc increased | rejected; exact on tested frame but slower and heavier |

The residual cache was especially risky because it retained extra full-resolution GPU residual tensors beyond the existing render/GT/depth caches. The fused and batch variants reduced launches but added tensor packing/stacking and memory pressure, which did not pay off in the measured workload.

## Validation

Full-resolution target-grid reuse was checked against the old no-pregrid warp path on `stump`:

- residual max absolute difference: `0.0`
- confidence max absolute difference: `0.0`
- residual mean absolute difference: `0.0`
- confidence mean absolute difference: `0.0`

Stump4 adapter-only smoke:

| profile | evidence max side | ms/frame | FPS | peak alloc |
|---|---:|---:|---:|---:|
| earlier full-res local snapshot | `0` | `1145.782735` | `0.873293` | `1146.696 MiB` |
| v94 target-grid-only | `0` | `650.008579` | `1.545170` | `1146.124 MiB` |
| rejected batch warp | `0` | `664.360456` | `1.518460` | `1458.688 MiB` |

The stump4 smoke is useful for regression tracking, but the full9 result below is the evidence used for claims.

## Full9 Integrated Runtime

Artifacts:

- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.md`
- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.json`
- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/per_scene.csv`
- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/raw/`
- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/logs/`

Same protocol as `runtime_profile_20260625_phasej_integrated_v2`: 9 Mip-NeRF360 scenes, 246 target views, 2 repeats per scene, render+adapter no PNG writes, no metrics, no policy calibration.

| profile | weighted ms/view | weighted FPS | weighted adapter ms/view | adapter/render | max alloc |
|---|---:|---:|---:|---:|---:|
| old integrated v2 | `951.410896` | `1.051071` | `913.855245` | `24.638570` | `17703.596 MiB` |
| rejected v93 combo | `956.733116` | `1.045224` | `919.261038` | `24.826611` | `19481.500 MiB` |
| v94 target-grid-only | `944.945199` | `1.058262` | `907.552261` | `24.577011` | `17701.383 MiB` |

Net v94 result versus old integrated v2:

- weighted integrated runtime: `-6.465698 ms/view`, `-0.68%`
- weighted adapter runtime: `-6.302984 ms/view`, `-0.69%`
- integrated/render-only compact ratio: `27.044247x -> 26.860457x`
- adapter/render-only compact ratio: `25.976712x -> 25.797547x`

Per-scene v94 adapter deltas versus old v2:

| scene | adapter delta |
|---|---:|
| bicycle | `-1.24%` |
| bonsai | `-0.78%` |
| counter | `-3.69%` |
| flowers | `-1.04%` |
| garden | `+1.68%` |
| kitchen | `+1.92%` |
| room | `-0.66%` |
| stump | `-1.77%` |
| treehill | `-0.66%` |

## Claim Boundary

v94 is worth keeping because it is exact, simple, and slightly improves the full9 weighted runtime without increasing peak memory. It does not solve the central deployment weakness: Phase-J remains about `26.86x` slower than compact render-only in integrated no-I/O profiling.

The next real paper-level runtime direction must be representation-level baking or a materially different accelerated adapter. More micro-optimizations around the current per-view ELA warp path are unlikely to turn a `~25x` adapter/render gap into a deployment-speed win.

## Paper-Protocol Bridge Refresh

Artifacts:

- `outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- `outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.json`
- `outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.csv`
- `outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_clean_baseline_candidates.csv`

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/collect_paper_m360_compact_ela_policy_metrics.py \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --policy_tag ratio_0200 \
  --method_name ours_26000_phasej_guarded_adaptedge_ela \
  --baseline_iterations 26000,30000 \
  --method_iteration 26000 \
  --out_dir /dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94
```

The collector selects the clean MeshSplatting baseline by held-out test score `PSNR + 20 * SSIM - 20 * LPIPS` over clean `26000` and `30000`. It does not use train metrics for selection.

Summary:

- available scenes: `9`
- RGB + compact pass: `9/9`
- mean dPSNR vs selected clean: `+1.331084`
- mean dSSIM vs selected clean: `+0.034702`
- mean dLPIPS vs selected clean: `-0.063359`
- mean dPSNR vs MeshSplatting paper table: `+1.701655`
- mean dSSIM vs MeshSplatting paper table: `+0.055498`
- mean dLPIPS vs MeshSplatting paper table: `-0.086516`
- mean triangle reduction: `7.6479%`

Boundary:

- Geometry columns are `nan` for the ELA layer in this collector because ELA changes held-out RGB renders, not the compact checkpoint geometry metrics. Therefore this table supports RGB quality plus compactness, not a strict all-axis geometry win.
- The runtime table above remains the correct deployment-speed boundary.
