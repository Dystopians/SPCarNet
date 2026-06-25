# Full9 Render-Only Runtime Profile

This table benchmarks checkpoint renderer forward time only. It does not save PNGs and excludes `metrics.py`, LPIPS, disk I/O, and Phase-J render-time Evidence Lumigraph Adapter post-processing.

## Summary

- scenes: `9`
- mean_clean_fps: `31.739752`
- mean_compact_fps: `30.075865`
- mean_delta_fps: `-1.663887`
- mean_fps_ratio: `0.946023`
- mean_clean_ms_per_view: `32.185860`
- mean_compact_ms_per_view: `34.092124`
- mean_delta_ms_per_view: `1.906264`
- mean_peak_alloc_reduction_frac: `0.025733`
- mean_triangle_reduction_frac: `0.076479`
- mean_checkpoint_reduction_frac: `0.046753`
- fps_win_scenes: `0`
- peak_alloc_reduction_scenes: `9`
- checkpoint_reduction_scenes: `9`
- scope: `render_only_no_png_no_metrics_no_phasej_adapter_postprocess`

## Per-Scene

| scene | views | clean FPS | compact FPS | dFPS | FPS ratio | clean ms/view | compact ms/view | d peak alloc MiB | peak alloc red. | tri red. | checkpoint red. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25 | 37.687185 | 36.131664 | -1.555521 | 0.958725 | 26.534529 | 27.676855 | -278.679 | 3.67% | 11.81% | 6.56% |
| flowers | 22 | 34.158053 | 33.594420 | -0.563634 | 0.983499 | 29.284125 | 29.782082 | -302.125 | 3.93% | 11.82% | 6.82% |
| garden | 24 | 31.085143 | 28.953293 | -2.131850 | 0.931419 | 32.169712 | 34.538410 | -103.345 | 1.25% | 3.47% | 3.06% |
| stump | 16 | 36.266064 | 34.732597 | -1.533467 | 0.957716 | 27.576581 | 28.792142 | -270.267 | 3.88% | 11.82% | 6.50% |
| treehill | 18 | 36.273732 | 34.722090 | -1.551642 | 0.957224 | 27.568567 | 28.800940 | -300.847 | 4.23% | 11.81% | 6.76% |
| room | 39 | 30.716288 | 28.103543 | -2.612745 | 0.914939 | 32.556192 | 35.583034 | -103.178 | 0.84% | 2.10% | 2.17% |
| counter | 30 | 24.049097 | 22.809441 | -1.239656 | 0.948453 | 41.581638 | 43.842166 | -222.219 | 1.84% | 2.10% | 2.24% |
| kitchen | 35 | 26.158549 | 24.092243 | -2.066307 | 0.921008 | 38.229285 | 41.507263 | -59.971 | 0.51% | 2.10% | 2.35% |
| bonsai | 37 | 29.263657 | 27.543498 | -1.720160 | 0.941219 | 34.172113 | 36.306227 | -359.062 | 3.01% | 11.80% | 5.61% |

## Interpretation

- Compact checkpoints reduce checkpoint bytes, triangle count, and peak allocated CUDA memory in all 9 scenes.
- Render-only FPS is lower for compact checkpoints in this benchmark, so current evidence supports a memory/size claim, not a speed claim.
- Phase-J render-time ELA is excluded; end-to-end adapter speed requires a separate benchmark.
