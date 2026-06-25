# Adapter Runtime Closure Log

Date: 2026-06-25
Status: updated after full9 integrated Phase-J runtime profiling

## Current State

Runtime evidence now has three promoted pieces:

- render-only MeshSplatting/compact profiling;
- isolated Phase-J adapter postprocess profiling;
- integrated renderer-forward + Phase-J `adapt_frame` profiling in one process.

The honest conclusion is unchanged but sharper: SPCarNet/Phase-J has strong quality, memory, checkpoint-size, and triangle-count evidence, but it is not a speedup method in the current render-time adapter form.

Disk state during profiling:

- `/data`: `28T` size, `27T` used, about `441M` available, `100%` use;
- `/dev/shm`: `252G` size, `162G` used, `91G` available, `65%` use.

Temporary profiler outputs should continue to stage in `/dev/shm`; only small JSON/CSV/MD/log summaries should be copied under `outputs/`.

## Render-Only Full9

Artifacts:

```text
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/per_scene.csv
```

This table excludes PNG writing, `metrics.py`, LPIPS, disk I/O, and Phase-J adapter postprocessing. It supports the memory/size claim, not speed: compact checkpoints reduce peak CUDA allocation, checkpoint bytes, and triangles in all `9 / 9` scenes, but render-only FPS is lower in all `9 / 9` scenes.

## Isolated Adapter Postprocess

Artifacts:

```text
scripts/car_model/benchmark_ela_postprocess_runtime.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/per_scene.csv
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/raw/
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/logs/
```

Scope: `adapt_frame` only; no renderer, PNG writes, `metrics.py`, LPIPS, or policy calibration.

| item | value |
|---|---:|
| scenes | `9` |
| target views | `246` |
| repeats per scene | `2` |
| weighted adapter ms/view | `1061.298183` |
| weighted adapter FPS | `0.942242` |
| max CUDA peak allocated | `4437.766 MiB` |
| approx render + adapter ms/view | `1095.390307` |
| approx render + adapter FPS | `0.912917` |
| adapter/render time ratio | `31.130304x` |

## Integrated Renderer + Adapter

Artifacts:

```text
scripts/car_model/benchmark_phasej_integrated_runtime.py
scripts/car_model/summarize_phasej_runtime_profiles.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/per_scene.csv
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/raw/
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/md/
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/logs/
```

Scope: canonical renderer forward plus Phase-J `adapt_frame` in one process; no PNG writes, `metrics.py`, LPIPS, or policy calibration. The v2 runner enforces Scene/evidence frame name alignment and reuses the support `FrameLoader` cache per repeat.

| item | value |
|---|---:|
| scenes | `9` |
| target views | `246` |
| repeats per scene | `2` |
| weighted integrated ms/view | `951.410896` |
| weighted integrated FPS | `1.051071` |
| weighted render ms/view | `37.090434` |
| weighted adapter ms/view | `913.855245` |
| weighted adapter/render ratio | `24.638570x` |
| max CUDA peak allocated | `17703.596 MiB` |
| integrated/render-only compact ms ratio | `27.044247x` |

This closes the exact no-I/O render+adapter runtime table. It also confirms that the adapter, not triangle rendering, dominates wall-clock time.

## Rate/Frontier and Audits

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/evidence_manifest_delta_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/runtime_adapter_gap_audit.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/v90_v91_process_result_audit.md
```

## Remaining Runtime Boundary

Runtime is now closed for the no-PNG/no-metrics profiling protocol. It is still not closed for deployment-speed claims because:

- the integrated runner excludes PNG writing, `metrics.py`, LPIPS, and downstream I/O;
- the strongest Phase-J RGB gain is still a render-time adapter rather than a checkpoint-baked representation;
- no promoted checkpoint-baked candidate has a full9 runtime and quality table.

The next credible speed-related milestone is either a checkpoint-baked repair endpoint that preserves Phase-J quality, or a faster adapter implementation that reduces per-view warping/support aggregation cost by an order of magnitude.
