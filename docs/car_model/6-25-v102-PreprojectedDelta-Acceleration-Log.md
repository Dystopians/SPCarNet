# v102 Preprojected Delta Acceleration Log

Date: 2026-06-25

Status: implementation and hard-triad validation milestone. This is a v102a acceleration endpoint, not the final representation-baked paper method.

## 0. Summary

v102a adds a preprojected target-camera residual delta bank to `render.py`.

The motivation is concrete: v101 is quality-strong and package-closed, but its deployment path still runs the full support-view residual warp/gate adapter for every target frame. On counter, the measured detached runtime was:

- standard `render.py`: `2.238598 sec/view`
- v101 require-bank endpoint: `4.220285 sec/view`
- slowdown: `1.885235x`

v102a splits the work:

1. Offline build pass: run the v101 train-evidence endpoint once and store `adapted - base_render` for each target camera.
2. Deployment pass: render the base MeshSplatting frame and apply the stored delta with `clamp(base + delta)`.

This preserves the v101 output on validated target-camera sets while removing online support-frame projection, support residual sampling, depth-consistency checks, local trust gating, and per-frame evidence aggregation from deployment.

## 1. Code Changes

Main file: `render.py`

New CLI flags:

```bash
--checkpoint_endpoint_write_preprojected_bank <path>
--checkpoint_endpoint_preprojected_bank_path <path>
--checkpoint_endpoint_require_preprojected_bank
--checkpoint_endpoint_preprojected_delta_dtype {float32,float16}
--checkpoint_endpoint_no_intermediate_outputs
```

New validation behavior:

- The v102 bank stores `endpoint_report_sha256` and refuses mismatched endpoint reports.
- The v102 bank stores source v101 evidence-bank SHA; callers can pass `--checkpoint_endpoint_bank_path` during validation to verify the source bank.
- The v102 bank stores target camera metadata and base render/depth hashes from the build pass.
- The normal validation path checks target camera identity and, when intermediate outputs are enabled, base render/depth hashes.
- The fast deployment path can set `--checkpoint_endpoint_no_intermediate_outputs`, which skips base/depth side outputs and per-frame empty-cache calls.

Helper scripts:

- `scripts/car_model/run_v102_preprojected_delta_scene.py`
- `scripts/car_model/summarize_v102_preprojected_delta.py`

## 2. Hard-Triad Validation

Summary artifact:

`outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625/v102_preprojected_delta_triad_summary.json`

Result:

- `all_present=true`
- `all_passed=true`
- `all_preprojected=true`
- `all_numerically_exact=true`
- mean PSNR / SSIM / LPIPS: `30.167395 / 0.913355 / 0.163709`
- mean delta vs v101 reference: `0.000000 / -0.000000 / 0.000000`
- mean fast wall sec/view: `2.754623`
- mean internal sec/view: `1.544080`

| scene | passed | PSNR | SSIM | LPIPS | hash | numeric exact | fast sec/view |
|---|---:|---:|---:|---:|---:|---:|---:|
| counter | true | 28.442907 | 0.893696 | 0.186557 | `30/30` | true | 2.857258 |
| kitchen | true | 30.197395 | 0.916093 | 0.132004 | `31/35` | true | 2.646130 |
| bonsai | true | 31.861883 | 0.930276 | 0.172566 | `37/37` | true | 2.760481 |

Kitchen has 4 PNG hash mismatches, but the difference is numerically negligible:

- `mean_abs_uint8=1.117844e-07`
- `max_abs_uint8=1`
- `nonzero_fraction=1.117844e-07`
- metrics match the v101 reference within evaluator precision.

## 3. Commands

Per-scene validation template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v102_preprojected_delta_scene.py \
  --scene <scene> \
  --gpu <gpu> \
  --force \
  --no_intermediate_outputs
```

Triad summary:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v102_preprojected_delta.py \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625
```

## 4. Claim Boundary

Safe claim:

> v102a preprojects v101's train-evidence residual transfer into a target-camera delta bank, preserving the v101 output on validated scenes while reducing deployment-time endpoint overhead.

Unsafe claims:

- Not a vanilla MeshSplatting checkpoint.
- Not a general unseen-camera representation.
- Not an independent quality improvement over v101.
- Not a geometry or triangle-count improvement.
- Not the final paper method if the paper requires representation-level generalization.

## 5. Next Step: v102b Surface Residual Field

The stronger paper direction is not target-camera delta caching. The next research-level step should be a surface-addressed residual field:

```text
checkpoint surface / face id / barycentric support
  -> train-evidence residual statistics
  -> compact residual field or low-order view-conditioned coefficients
  -> render.py endpoint samples the field using visible surface ids
  -> calibrated fallback where evidence is unsupported
```

Promotion requirements for v102b:

- counter mechanism gate beats the checkpoint-baked negative anchors;
- hard triad preserves v101-quality gains;
- full9 validates against clean baseline and v101;
- detached package proves `all_used_required_field=true`;
- runtime improves over v101 without reducing quality;
- no target-GT-use smoke passes.
