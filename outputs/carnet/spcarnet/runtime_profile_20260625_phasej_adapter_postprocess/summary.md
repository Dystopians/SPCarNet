# Phase-J ELA Postprocess Runtime Profile

This table measures `adapt_frame` post-processing only. It excludes renderer runtime, PNG writes, image metrics, LPIPS, and policy calibration.

## Summary

- scenes: `9`
- target_frames: `246`
- repeats_per_scene: `[2]`
- mean_scene_ms_per_frame: `1042.8269783143578`
- weighted_ms_per_frame: `1061.2981826619707`
- weighted_frames_per_sec: `0.9422422617287243`
- max_cuda_peak_allocated_mib: `4437.76611328125`
- mean_cuda_peak_allocated_mib: `3575.8132595486113`
- render_only_compact_mean_ms_per_view: `34.092124427487626`
- approx_compact_render_plus_adapter_ms_per_view: `1095.3903070894582`
- approx_compact_render_plus_adapter_fps: `0.9129166047279366`
- approx_adapter_over_render_ms_ratio: `31.13030356671679`
- scope: `phasej_adapter_adapt_frame_no_png_no_metrics_no_renderer_no_calibration`

## Per-Scene

| scene | views | ms/frame | fps | CUDA peak MiB | compact render ms | approx render+adapter ms | approx fps |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25 | 1240.978 | 0.821 | 2865.5 | 27.677 | 1268.655 | 0.788 |
| flowers | 22 | 962.341 | 1.041 | 2866.1 | 29.782 | 992.123 | 1.008 |
| garden | 24 | 982.705 | 1.021 | 2978.8 | 34.538 | 1017.243 | 0.983 |
| stump | 16 | 896.589 | 1.118 | 2865.3 | 28.792 | 925.382 | 1.081 |
| treehill | 18 | 861.887 | 1.172 | 2885.1 | 28.801 | 890.688 | 1.123 |
| room | 39 | 1129.499 | 0.886 | 4415.3 | 35.583 | 1165.082 | 0.858 |
| counter | 30 | 1261.125 | 0.793 | 4431.9 | 43.842 | 1304.967 | 0.766 |
| kitchen | 35 | 1153.460 | 0.878 | 4436.5 | 41.507 | 1194.968 | 0.837 |
| bonsai | 37 | 896.859 | 1.121 | 4437.8 | 36.306 | 933.165 | 1.072 |

## Interpretation

- Phase-J adapter post-processing dominates runtime: the weighted mean adapter time is about `1061.298` ms/view.
- The additive render+adapter estimate is about `1095.390` ms/view, or `0.913` FPS, using the prior render-only compact checkpoint benchmark.
- This is not a speedup claim. It closes the missing postprocess runtime evidence and shows the current render-time adapter is expensive.
- CUDA peak allocation here is for adapter postprocess only and is not directly additive with renderer peak allocation.
