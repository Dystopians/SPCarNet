# v103 Face-Local Affine Barycentric Residual Field Log

Date: 2026-06-25

Status: counter-only method evidence. This is not a full9 result, not a promoted paper endpoint, and not final completion.

## 0. Review Verdict

v103 is the first useful step beyond the v102b static surface-field prototype.

The important change is representation-level: v102b stored one constant RGB residual per triangle, while v103 stores a face-local affine residual basis over barycentric coordinates. At render time, `render.py` now supports `basis_type=affine_barycentric`, samples the current pixel's face-local barycentric basis `[1, u, v]`, and evaluates the per-triangle RGB coefficients. This means the residual is no longer forced to be spatially constant inside a visible triangle.

On counter, v103 improves all three metrics over the clean reference. The stronger setting so far is `min_count=1`:

```text
clean: 26.751773834 PSNR / 0.862055242 SSIM / 0.252003312 LPIPS
v103 : 27.208200455 PSNR / 0.863404870 SSIM / 0.243176267 LPIPS
delta: +0.456426621 PSNR / +0.001349628 SSIM / -0.008827045 LPIPS
```

This is a real positive counter result because the same surface-field render path now moves PSNR, SSIM, and LPIPS in the right direction. It is also not enough for the paper endpoint. v103 remains far below the v101/v102a counter reference:

```text
v101/v102a: 28.442907333 PSNR / 0.893695712 SSIM / 0.186556786 LPIPS
v103     : 27.208200455 PSNR / 0.863404870 SSIM / 0.243176267 LPIPS
gap      : -1.234706879 PSNR / -0.030290842 SSIM / +0.056619480 LPIPS
```

The honest conclusion is:

```text
v103 validates face-local affine surface residuals as a better surface representation than v102b.
v103 does not yet recover the view-conditioned and gated residual behavior of v101/v102a.
The next method still needs view-conditioned and evidence-gated residual coefficients.
```

## 1. What Changed Relative To v102b

### render.py support

Implementation file:

```text
render.py
```

The surface-field endpoint now accepts fields with:

```text
basis_type = affine_barycentric
coefficient layout = triangle,basis,rgb
basis order = [1, barycentric_0, barycentric_1]
```

At application time, `render.py`:

1. reads rendered triangle ids from the renderer package;
2. computes the visible pixel's face-local barycentric basis from projected triangle vertices;
3. fetches the target triangle's `3 x 3` RGB coefficient matrix;
4. evaluates `residual_rgb = [1, u, v] @ coeffs`;
5. applies `clamp(base_render + residual_rgb)`.

This is a method change rather than only an engineering change. v102b can only represent one averaged color correction per triangle. v103 can represent an affine residual plane inside each triangle, which preserves some spatial variation that v102b necessarily smears.

### Builder

Builder:

```text
scripts/car_model/build_v103_surface_affine_residual_field.py
```

Mechanism:

```text
v102 preprojected target-camera delta bank
  + target render rend_ids
  + projected triangle vertices
  -> per-pixel triangle id and barycentric basis
  -> per-triangle normal equations for RGB affine residuals
  -> ridge solve
  -> v103 affine_barycentric surface residual field
```

The field stores no target GT, but this remains a same-target-camera prototype distilled from v102 preprojected deltas. It should not be described as unseen-camera generalization.

## 2. Artifacts And Result Paths

Source delta bank:

```text
/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/counter/v102_preprojected_delta_bank.pt
/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/counter/v102_preprojected_delta_bank.manifest.json
```

v103 field:

```text
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_field.pt
/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_field.manifest.json
```

Render report:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/test/ours_26000_v103_affine_surface_field_counter/render_py_endpoint_report.json
```

Merged metric JSON:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model/results.json
```

Observed field manifest:

| item | value |
|---|---:|
| scene | `counter` |
| basis type | `affine_barycentric` |
| source target frames | `30` |
| triangle count | `9,644,247` |
| valid triangles | `1,771,636` |
| valid triangle fraction | `0.1836987436` |
| field size | `203 MiB` |
| build elapsed | `179.6856 sec` |
| renderer scaling | `4` |
| ridge | `0.0001` |
| min count | `3` |
| residual dtype | `float16` |
| camera validation | `strict_target_camera_match` |

Render report summary:

| item | value |
|---|---:|
| output method | `ours_26000_v103_affine_surface_field_counter` |
| mode | `surface_residual_field_endpoint` |
| target frames | `30` |
| support frames | `0` |
| field source target frames | `30` |
| mean abs delta | `0.0064024815` |
| mean changed fraction | `0.9725891749` |
| mean surface valid fraction | `0.9725882868` |
| mean surface unique triangles | `206,584.9667` |
| no test GT used for policy | `true` |

The distinction between field coverage and visible-pixel coverage matters. Only `18.37%` of all triangles have valid affine coefficients at `min_count=3`, but the counter test views mostly render through those supported triangles, giving about `97.26%` mean surface-valid visible pixels in the v103 render report.

## 3. Commands

Build the v103 affine surface field:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v103_surface_affine_residual_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/counter/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_field.pt \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --split test \
  --renderer_scaling 4 \
  --min_count 3 \
  --ridge 0.0001 \
  --residual_dtype float16
```

Render with the v103 field:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --iteration 26000 \
  --skip_train \
  --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --checkpoint_endpoint_output_method ours_26000_v103_affine_surface_field_counter \
  --checkpoint_endpoint_surface_field_path /dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_field.pt \
  --checkpoint_endpoint_require_surface_field \
  --checkpoint_endpoint_no_intermediate_outputs \
  --quiet
```

Evaluate the rendered method:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/counter/detached_model \
  --split test \
  --methods ours_26000_v103_affine_surface_field_counter \
  --merge_model_results
```

## 4. Counter Metrics

| method | PSNR | SSIM | LPIPS | role |
|---|---:|---:|---:|---|
| clean counter reference | `26.751773834` | `0.862055242` | `0.252003312` | base MeshSplatting reference |
| v102b surface residual field | `27.058162689` | `0.860652804` | `0.249831960` | constant per-triangle residual prototype |
| v103 affine barycentric surface field, min_count=3 | `27.167144775` | `0.862644494` | `0.244518295` | stricter face-local affine prototype |
| v103 affine barycentric surface field, min_count=1 | `27.208200455` | `0.863404870` | `0.243176267` | current best counter face-local affine prototype |
| v101/v102a counter reference | `28.442907333` | `0.893695712` | `0.186556786` | render.py endpoint / preprojected-delta ceiling |

Delta table:

| comparison | dPSNR | dSSIM | dLPIPS | interpretation |
|---|---:|---:|---:|---|
| v103 min_count=1 minus clean | `+0.456426621` | `+0.001349628` | `-0.008827045` | all three axes improve over clean on counter |
| v103 min_count=1 minus v102b | `+0.150037766` | `+0.002752066` | `-0.006655693` | face-local affine basis is better than a constant face residual |
| v103 min_count=1 minus v103 min_count=3 | `+0.041055679` | `+0.000760376` | `-0.001342028` | higher coverage helps on counter |
| v103 min_count=1 minus v101/v102a | `-1.234706879` | `-0.030290842` | `+0.056619480` | still far below the view-conditioned/gated residual ceiling |

## 5. Why v103 Improves Over Clean

The clean reference has no residual transfer. v103 adds a surface-attached residual field that is sampled in the normal `render.py` path, so it can correct repeatable color errors where the rendered target pixels land on supported faces.

The positive counter result is plausible for three concrete reasons:

1. Face-local variation is represented. A single triangle can now emit different residuals at different barycentric locations, avoiding the constant-color smear of v102b.
2. The `min_count=3` threshold and ridge solve reduce the most underdetermined face fits. Unsupported or under-supported triangles fall back to zero residual through the same surface endpoint.
3. The counter test views have high visible-pixel support under this field: the render report gives mean surface valid fraction `0.9725882868`, so the field is actually being sampled on most visible pixels even though only `18.37%` of global triangles are coefficient-valid.

This explains why v103 improves all three metrics over clean and also improves all three over v102b.

## 6. Why The Gap To v101/v102a Remains Important

v103 is still not the v101 behavior in surface-field form.

The missing signal is not just face-local spatial variation. v101/v102a preserve a view-conditioned, evidence-gated residual transfer: support-view residuals, depth consistency, target view direction, local trust, and policy fallback all affect the final correction. v103 collapses that behavior into a single affine function per triangle. The same face receives the same affine residual regardless of viewing direction, occlusion context, grazing angle, or support agreement.

That matters most for LPIPS and SSIM. v103 recovers a modest LPIPS gain over clean, but it remains `0.057961509` worse than v101/v102a on counter. The SSIM gap is also large at `0.031051218`. Those numbers say the face-local basis is necessary but not sufficient: the next representation must add view-conditioned or gated coefficients, not merely increase static surface capacity.

## 7. min_count=1 Experiment

Status: completed on counter.

Artifacts:

| field | value |
|---|---|
| field path | `/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_min1_field.pt` |
| manifest path | `/dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/counter/v103_surface_affine_min1_field.manifest.json` |
| render method | `ours_26000_v103_affine_min1_surface_field_counter` |
| valid triangles | `2,716,465 / 9,644,247` |
| valid triangle fraction | `0.281666875` |
| total accumulated pixels | `48,475,994` |
| build elapsed | `246.9913 sec` |
| render elapsed | `47.2076 sec` |
| mean surface valid fraction | `0.999039084` |
| PSNR | `27.208200455` |
| SSIM | `0.863404870` |
| LPIPS | `0.243176267` |
| comparison to min_count=3 | `+0.041055679 PSNR / +0.000760376 SSIM / -0.001342028 LPIPS` |

Interpretation:

- `min_count=1` improves all three metrics over `min_count=3`, so the counter result was coverage-limited rather than noise-limited.
- The visible-pixel support rises from `0.972588287` to `0.999039084`, and that extra coverage gives measurable gains.
- This does not remove the v101/v102a gap; it only establishes that face-local affine residuals are a stronger surface-field ablation than v102b.

## 8. Claim Boundary

Safe claims:

- `render.py` supports `basis_type=affine_barycentric` surface residual fields.
- `scripts/car_model/build_v103_surface_affine_residual_field.py` builds a face-local affine barycentric field from v102 preprojected deltas and renderer triangle ids.
- The counter `min_count=1` field has `2,716,465 / 9,644,247` valid triangles, or `0.281666875` valid triangle fraction.
- On counter, v103 improves PSNR, SSIM, and LPIPS over clean.
- On counter, v103 improves PSNR, SSIM, and LPIPS over v102b.
- v103 is stronger evidence than v102b for a surface-attached residual representation.

Unsafe claims:

- Do not claim v103 has full9 validation.
- Do not claim hard-triad validation is complete.
- Do not claim v103 is final paper completion.
- Do not claim v103 matches or preserves v101/v102a quality.
- Do not claim a vanilla MeshSplatting checkpoint.
- Do not claim unseen-camera generalization from this field.
- Do not claim hard-triad/full9 behavior from the counter-only min_count=1 result.

## 9. Next Direction

v103 should become the base ablation for the next residual-field method:

```text
face id
  + barycentric face-local coordinate
  + view direction or camera bin
  + evidence agreement / support confidence
  -> compact residual coefficients
  -> calibrated fallback for unsupported or risky pixels
```

Minimum next requirements:

1. Add view-conditioned coefficients or camera-bin residual bases.
2. Keep the face-local barycentric basis; v103 shows it is better than v102b.
3. Add a gate or shrinkage mechanism tied to support agreement, not only `min_count`.
4. Preserve fail-closed field loading through `--checkpoint_endpoint_require_surface_field`.
5. Run zero-field, constant-field, affine-field, no-view, and view-conditioned ablations.
6. Validate counter first, then hard triad, then full9 only after the same fixed policy passes earlier gates.

Paper-safe synthesis:

> v103 shows that a surface-attached residual field needs face-local structure: affine barycentric coefficients beat a constant per-face residual and improve all three counter metrics over clean. The remaining gap to v101/v102a shows that the field must also retain view-conditioned and evidence-gated behavior before it can become the paper endpoint.
