# v93 Adapter Kernel Optimization Log

Date: 2026-06-25  
Status: superseded by v94; useful diagnostic history, but not the current kept runtime path

## Scope

This log documents the current Evidence Lumigraph Adapter runtime optimization milestone. It is a runtime/kernel milestone for `adapt_frame` postprocessing, not a new promoted representation or quality endpoint.

Benchmark scope for the local `/dev/shm` timing snapshots:

- runner: `scripts/car_model/benchmark_ela_postprocess_runtime.py`
- scope: `adapt_frame_no_png_no_metrics_no_renderer_no_calibration`
- scene snapshot: `stump`, compact Phase-F model, Phase-J ELA report
- target split: `test`
- target views: `4` of `16`
- repeats: `2`
- support source: `all_train`
- support frames: `109`
- ELA mode: `residual`
- support count `k`: `4`
- device: `cuda:0`

These are smoke-scale runtime measurements. They are useful for kernel direction and regression tracking, but they are not a full9 deployment-speed claim.

## Exact Optimization Milestone

The current exact path keeps `--evidence_max_side 0`, so support evidence is computed at full resolution. The intended contract is output preservation: the ELA policy, support selection, residual transfer semantics, gates, and alpha calibration are unchanged.

Implemented exact runtime optimizations:

- target-world-grid reuse: the target camera/depth backprojection grid is built once per target frame and reused across support warps instead of being rebuilt for every support frame;
- fused residual/depth `grid_sample`: support residual channels and support depth are packed and sampled with one interpolation grid, replacing separate residual and depth sampling work;
- support residual cache: rendered support, GT support, depth, and clipped residual tensors are reused through the `FrameLoader` cache during repeated support access.

The full-resolution exact smoke result is consistent with the output-preserving contract. Compared with the earlier full-resolution local snapshot, covered fraction is unchanged at `0.892700985074`, mean confidence matches to float-level noise, and the checksum differs by only `1.1920928955078125e-07`. This is runtime evidence, not a substitute for a formal image-quality equivalence table.

## Approximate Fast-Evidence Path

The optional fast path is controlled by:

```text
--evidence_max_side N
```

When `N > 0`, ELA evidence is computed at a capped image side and then upsampled before the normal gates and adaptation step. This is approximate. It can change coverage, confidence, checksums, and final images. It must not be described as output-preserving, and it needs PSNR/SSIM/LPIPS plus visual validation before any quality or deployment claim.

In the local side-400 snapshot, coverage and checksum changed versus full-resolution exact mode:

- covered fraction: `0.892700985074` full-res exact -> `0.924305483699` side-400
- mean confidence: `1.452960044146` full-res exact -> `1.379620641470` side-400
- checksum: `9.382644116879` full-res exact -> `9.215704500675` side-400

## Local Timing Evidence

Source files checked in `/dev/shm`:

| snapshot | status | mtime local | evidence_max_side | ms / target frame mean | target FPS mean | CPU wall mean | CUDA peak alloc max | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `/dev/shm/spcarnet_fastela_stump4_fullres.json` | present | `2026-06-25 00:48:42 -0700` | `0` | `1145.782735` | `0.873293` | `4.583131 s` | `1146.696 MiB` | earlier full-resolution local snapshot |
| `/dev/shm/spcarnet_exactopt_stump4_fullres.json` | present | `2026-06-25 00:50:41 -0700` | `0` | `673.573465` | `1.490694` | `2.694294 s` | `1510.529 MiB` | exact optimized full-resolution path |
| `/dev/shm/spcarnet_fastela_stump4_side400.json` | present | `2026-06-25 00:49:11 -0700` | `400` | `807.418974` | `1.242329` | `3.229676 s` | `1060.537 MiB` | approximate capped-evidence path; quality pending |
| `/dev/shm/spcarnet_exactopt_stump4_side400.json` | present after follow-up | `2026-06-25 00:53:00 -0700` | `400` | `689.398` | `1.472` | `2.758 s` | `1421.2 MiB` | approximate capped-evidence path after exact changes; not faster than exact full-res |

Derived from the present files:

- exact optimized full-res vs earlier full-res snapshot: `1145.782735 -> 673.573465 ms/frame`, or `1.701x` faster with `41.21%` lower mean ms/frame;
- side-400 vs earlier full-res snapshot: `1145.782735 -> 807.418974 ms/frame`, or `1.419x` faster with `29.53%` lower mean ms/frame;
- side-400 vs current exact full-res snapshot: `807.418974` is `19.87%` slower than `673.573465` in this smoke sample, so side-400 is not currently demonstrated as a faster replacement for the exact optimized path;
- the exact side-400 follow-up did not beat the exact full-resolution path, so `--evidence_max_side 400` is not promoted as a default runtime setting.

## Current Boundary

The exact optimization is safe to describe as an output-preserving runtime implementation improvement, subject to normal floating-point tolerance and broader regression checks.

The `--evidence_max_side` path is different: it is a speed/quality tradeoff knob. It should remain experimental until quality is validated on held-out images and preferably full9, because it changes the evidence resolution before the adapter's gates see the signal.

Next evidence moved to v94:

- `docs/car_model/6-25-v94-TargetGridRuntimeOptimization-Log.md` records the final kept target-grid-only path.
- `outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.md` is the current full9 integrated runtime evidence.
