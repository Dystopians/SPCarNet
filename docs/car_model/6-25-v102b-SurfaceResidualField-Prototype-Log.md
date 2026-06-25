# v102b Surface Residual Field Prototype Log

Date: 2026-06-25

Status: counter-only prototype evidence. This is not a promoted paper endpoint and does not claim full9 completion.

## 0. Review Verdict

v102b is a useful representation diagnostic, but the current result is negative to weakly positive.

The implementation proves that `render.py` can consume a checkpoint-attached, surface-addressed residual field, and that a field can be built from v102 preprojected deltas plus renderer triangle ids. The counter result improves PSNR and LPIPS slightly over the clean counter reference, but regresses SSIM and loses most of the v101/v102a gain. Therefore the simple "one averaged residual vector per triangle" hypothesis is not strong enough for the paper endpoint.

The honest story is:

```text
v102b falsifies the simplest static surface residual field.
It supports moving to a view-conditioned, face-local residual field.
It should not be written as the final representation-level method.
```

## 1. What Was Implemented

### render.py endpoint support

`render.py` now has a surface-field endpoint path:

```bash
--checkpoint_endpoint_surface_field_path <path>
--checkpoint_endpoint_require_surface_field
```

The field path is mutually exclusive with the v102 preprojected target-camera delta bank path. With `--checkpoint_endpoint_require_surface_field`, the renderer should fail closed if the field is absent. The endpoint reports the support source as a `v102_surface_residual_field:<path>` source and applies the residual through rendered triangle ids.

### Surface field builder

Builder:

```text
scripts/car_model/build_v102_surface_residual_field.py
```

Mechanism:

```text
v102 preprojected target-camera delta bank
  + target render rend_ids
  -> per-pixel triangle id assignment
  -> per-triangle mean RGB residual
  -> v102_surface_residual_field.pt
```

Current strict counter field:

```text
/dev/shm/peilincai_spcarnet_v102_surface_residual_field_20260625/counter/v102_surface_residual_field.pt
```

Manifest:

```text
/dev/shm/peilincai_spcarnet_v102_surface_residual_field_20260625/counter/v102_surface_residual_field.manifest.json
```

Observed manifest facts:

| item | value |
|---|---:|
| scene | `counter` |
| source frames | `30` |
| triangle count | `9,644,247` |
| valid triangles | `2,716,441` |
| valid triangle fraction | `0.281664` |
| residual dtype | `float16` |
| min count | `1` |
| camera validation | `strict_target_camera_match` |
| build elapsed | `20.841 sec` |

Important limitation: this prototype distills a v102 preprojected target-camera delta bank into surface storage. It stores no target GT, but it is still not a train-only field that has proven unseen-camera generalization.

## 2. Command Templates

Build or refresh the v102a preprojected bank for a scene:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v102_preprojected_delta_scene.py \
  --scene counter \
  --gpu <gpu> \
  --force \
  --no_intermediate_outputs
```

Build the v102b surface residual field:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v102_surface_residual_field.py \
  --model_path <counter-recovery-model> \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/counter/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v102_surface_residual_field_20260625/counter/v102_surface_residual_field.pt \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --split test \
  --min_count 1 \
  --residual_dtype float16
```

Render with the surface field:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m <counter-recovery-model> \
  --iteration 26000 \
  --skip_train \
  --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --checkpoint_endpoint_output_method ours_26000_v102b_surface_residual_field \
  --checkpoint_endpoint_surface_field_path /dev/shm/peilincai_spcarnet_v102_surface_residual_field_20260625/counter/v102_surface_residual_field.pt \
  --checkpoint_endpoint_require_surface_field \
  --checkpoint_endpoint_no_intermediate_outputs \
  --quiet
```

Evaluate the rendered method:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  -m <counter-recovery-model> \
  --split test \
  --methods ours_26000_v102b_surface_residual_field \
  --merge_model_results
```

Zero-field ablation slot:

```bash
# Field currently available:
# /dev/shm/peilincai_spcarnet_v102_surface_residual_field_20260625/counter/v102_surface_zero_field.pt
#
# Exact zero-field metrics after completion:
# PSNR 26.726955414 / SSIM 0.860741198 / LPIPS 0.251994997
```

## 3. Counter Results

Known counter metrics:

| method | PSNR | SSIM | LPIPS | role |
|---|---:|---:|---:|---|
| clean counter reference | `26.751773834` | `0.862055242` | `0.252003312` | base MeshSplatting reference |
| v102b surface residual field | `27.058162689` | `0.860652804` | `0.249831960` | current prototype |
| v101/v102a counter reference | `28.442907333` | `0.893695712` | `0.186556786` | render.py endpoint / preprojected-delta ceiling |
| zero surface field | `26.726955414` | `0.860741198` | `0.251994997` | ablation through same surface endpoint with all-zero residual |

Delta table:

| comparison | dPSNR | dSSIM | dLPIPS | interpretation |
|---|---:|---:|---:|---|
| v102b minus clean | `+0.306388855` | `-0.001402438` | `-0.002171352` | weak mixed gain: PSNR/LPIPS improve, SSIM regresses |
| v102b minus v101/v102a | `-1.384744644` | `-0.033042908` | `+0.063275174` | large loss versus the validated endpoint ceiling |
| v102b minus zero-field | `+0.331207275` | `-0.000088394` | `-0.002163038` | residual field gives real PSNR/LPIPS lift over the same endpoint path, but still does not improve SSIM |

The metric source is the current counter evaluator output merged into:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json
```

## 4. Comparison To Clean And v101/v102a

Against clean, v102b is not a clean win. PSNR rises by about `0.306 dB` and LPIPS improves by about `0.00217`, but SSIM drops. That is at best weak positive evidence for a static surface residual field.

Against v101/v102a, v102b is clearly worse. v101/v102a reaches `28.442907333 / 0.893695712 / 0.186556786` on counter, while v102b reaches only `27.058162689 / 0.860652804 / 0.249831960`. The gap is too large to describe v102b as preserving the Phase-J/v101 benefit.

This matters for the paper story because v101/v102a already show that the train-evidence residual endpoint can reproduce strong counter quality when residuals remain target-camera or support-conditioned. v102b shows that collapsing those residuals into one static mean RGB vector per triangle throws away most of the signal.

## 5. Claim Boundary

Safe claims:

- `render.py` has a prototype surface-field endpoint with a fail-closed require flag.
- A counter surface residual field can be built from v102 preprojected deltas and renderer triangle ids.
- The strict counter field covers `2,716,441 / 9,644,247` triangles at `min_count=1`.
- On counter, v102b gives a small PSNR/LPIPS improvement over clean but regresses SSIM.
- v102b is diagnostic evidence that a purely static per-triangle residual average is insufficient.

Unsafe claims:

- Do not claim v102b is full9 validated.
- Do not claim hard-triad validation is complete.
- Do not claim v102b is a final paper endpoint.
- Do not claim it beats or preserves v101/v102a quality.
- Do not claim this is a vanilla MeshSplatting checkpoint.
- Do not claim unseen-camera representation generalization from the current field, because the current field is distilled from a target-camera preprojected bank.
- Do not claim target-GT non-use for the complete v102b pipeline until a field-specific smoke/report is documented, even though the field manifest states that it stores no target GT.

## 6. Why This Is Negative Or Weak-Positive Evidence

The prototype answers the wrong hypothesis too simply:

```text
Hypothesis tested:
  A single RGB residual vector per triangle is enough to approximate the v101/v102a residual endpoint.

Counter evidence:
  It recovers only a small mixed gain over clean and loses most of the v101/v102a improvement.
```

Likely reasons:

- View dependence is removed. The v101/v102a residual behavior depends on target view, support evidence, depth consistency, and local trust. Averaging into one triangle residual cannot represent specular, occlusion, grazing-angle, or visibility-conditioned changes.
- Face-local variation is removed. A triangle may contain spatially varying residuals across barycentric locations. One mean color per triangle smears high-frequency or boundary corrections.
- Coverage is sparse. Only about `28.17%` of triangles receive nonzero measured support in the current strict counter field. Unsupported triangles default to zero, so the field cannot globally preserve the endpoint behavior.
- The field is built from target-camera deltas, not from a train/policy-val residual basis evaluated on unseen held-out cameras. That makes it a useful diagnostic, but not a representation-level generalization result.
- The SSIM regression versus clean is especially important. A paper endpoint cannot be promoted from PSNR/LPIPS-only gains when SSIM moves backward and the stronger v101/v102a reference is far ahead.

The zero-field ablation is now available. It approximately reproduces the base/clean behavior through the same surface endpoint, which verifies that v102b's small PSNR/LPIPS gain is caused by the residual field rather than an obviously broken evaluator path. However, v102b is only marginally above zero and still regresses SSIM, so the conclusion remains weak-positive at best.

## 7. Required Next Direction

The next method should move to a view-conditioned and face-local residual field, not another static per-triangle mean:

```text
surface face id
  + barycentric / local face coordinate
  + view direction or camera bin
  + train/policy-val residual statistics
  -> compact residual coefficients
  -> render.py samples field at visible surface locations
  -> calibrated fallback where evidence is unsupported
```

Minimum next implementation requirements:

1. Split discipline: build the field from train/policy-val evidence only; held-out target views are evaluation only.
2. Face-local support: store at least barycentric bins, small texture grids, or learned low-rank coefficients per supported face/region instead of one mean residual per triangle.
3. View conditioning: include direction/camera-bin coefficients or a low-order basis so the same face can emit different residuals under different views.
4. Fallback calibration: keep no-op or clean fallback for unsupported or high-risk regions.
5. Ablations: zero-field, no-view-conditioning, no-face-local bins, train-only versus train+policy-val, and require-field fail-closed.
6. Gates: do not expand beyond counter unless the method improves all three RGB axes versus clean and closes a meaningful part of the v101/v102a gap.
7. Validation ladder: counter first, then hard triad, then full9 only after the same fixed policy passes earlier gates.
8. Packaging: detached package must report `used_required_field=true`; field-specific target-GT non-use smoke must be saved.

## 8. Paper Story Synthesis

v102b should appear, if at all, as an ablation or negative prototype:

> A static surface-addressed residual average is easy to attach to the MeshSplatting surface, but it discards the view-conditioned and face-local structure that made the train-evidence endpoint effective.

The forward-looking paper story should be:

> The representation-level endpoint needs a surface field, but not a scalar per-triangle color delta. It needs a view-conditioned, face-local residual field with evidence-aware fallback.

Current paper-safe endpoint remains the earlier verified Phase-J/v101/v102a deployment story. v102b is the evidence that the next representation attempt must be richer.
