# SPCarNet v104c Paper-Loop Technical Report

Date: 2026-06-25

Scope: mentor/PPT technical synthesis for the current SPCarNet paper loop. This report summarizes existing v101-v104c documents and the completed v104c full9 fixed-policy artifacts.

## 0. One-Page Takeaway

Clean MeshSplatting is the baseline:

```text
train MeshSplatting checkpoint -> direct held-out render -> metrics
```

Current SPCarNet has two related but different lines:

```text
strong deployed endpoint line:
MeshSplatting surface + train-derived evidence bank + guarded residual endpoint
  -> v101/v102a quality ceiling

representation-field line:
distill endpoint residual behavior into a surface-addressed field
  -> v103/v104a/v104c current research direction
```

The honest story is:

- v101/v102a is still the strongest quality endpoint. On full9, v101 bank-backed render.py endpoint gives mean `26.481309 / 0.783675 / 0.224305` and improves selected clean by `+1.329627 PSNR / +0.034657 SSIM / -0.063316 LPIPS`.
- v104c is the current best surface-field variant. Its fixed policy is now complete on full9: `9/9` scenes pass, and every scene improves over the local selected clean MeshSplatting `ours_26000` baseline on PSNR, SSIM, and LPIPS.
- Full9 v104c mean is `25.829099 / 0.760727 / 0.268548`, improving clean by `+0.677417 PSNR / +0.011709 SSIM / -0.019073 LPIPS`.
- v104c does not yet close the endpoint-to-field gap. Full9 mean remains `-0.652211 PSNR / -0.022949 SSIM / +0.044243 LPIPS` behind the v101/v102a endpoint reference; hard-triad mean is still `-1.307599 / -0.027896 / +0.055355`.

PPT headline should not be "v104c beats the endpoint." The stronger and defensible headline is:

> SPCarNet's evidence endpoint strongly improves MeshSplatting; v104c is the latest fixed-policy step toward baking that endpoint behavior into a compact surface-addressed representation.

## 1. Clean MeshSplatting vs SPCarNet/v104c

Clean MeshSplatting renders exactly what the trained checkpoint stores. It has no explicit post-training notion of which surfaces are trustworthy, which local residuals repeat across views, or when a correction should fall back to zero.

SPCarNet adds an evidence layer on top of that trained surface. The surface is used as an address space: train/support views provide residual, depth, visibility, and local agreement evidence; the method transfers only the residual signal that passes policy and support checks.

v104c is not the whole SPCarNet system. It is the current representation-field attempt to absorb endpoint residual behavior into a persistent field:

```text
visible triangle id + barycentric coordinate + view direction
  -> low-order RGB residual
  -> clamp(base MeshSplatting render + residual)
```

Compared with clean MeshSplatting, v104c can correct repeatable local color/detail errors on supported surfaces. Compared with v101/v102a, v104c is much cheaper conceptually and more representation-like, but it compresses a guarded per-pixel endpoint into one low-order function per triangle. That compression is the remaining quality gap.

## 2. Module Roles

### v101 Evidence Bank

Role: close the deployment/artifact gap for the strong Phase-J/ELA endpoint.

What it does:

- Adds a `render.py` endpoint hook that loads a checkpoint-attached endpoint report.
- Stores train-derived residual, depth, camera, and hash evidence in `v101_evidence_bank.pt`.
- Supports `--checkpoint_endpoint_require_bank`, so bank validation fails closed instead of silently falling back to train folders.
- Validates detached packages: full9 detached packages reproduce the bankfp16 endpoint outputs exactly with `all_used_required_bank=true` and `all_hash_exact=true`.

Claim boundary:

- Strong full9 quality and packaging evidence.
- Not a vanilla MeshSplatting checkpoint.
- Not faster than standard render; counter runtime audit measured `4.220285 sec/view` vs `2.238598 sec/view`.

### v102 Preprojected Delta

Role: accelerate the v101 endpoint on validated target-camera sets.

What it does:

- Offline: runs v101 once and stores `adapted - base_render` per target camera.
- Online: renders base MeshSplatting and applies the stored delta.
- Removes online support-frame projection/gating from deployment.

Hard-triad validation:

- Mean `30.167395 / 0.913355 / 0.163709`.
- Numerically exact against v101 reference on the validated hard triad.

Claim boundary:

- Useful endpoint acceleration and quality ceiling.
- Not unseen-camera generalization.
- Not a representation-baked method.

### v103/v104 Surface Residual Field

Role: start baking residual behavior into a surface-addressed field.

v103 stores a face-local affine barycentric residual:

```text
[1, u, v] -> RGB residual
```

v104a adds view direction:

```text
[1, u, v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

This is the important research transition. v103 proves that a surface field can beat clean on the hard triad. v104a proves that view direction is useful. But both still trail the endpoint ceiling, because they collapse support agreement, occlusion context, local trust, and policy fallback into one small per-triangle model.

### v104c Shrink View-Affine Fallback

Role: stabilize v104a without throwing away its view-conditioned signal.

v104b hard fallback diagnosed the problem: many triangles are under-supported or ill-conditioned for a six-feature RGB fit. Hard fallback was safer but over-conservative.

v104c uses the same render-time payload as v104a, but changes the builder:

- center and scale view-direction features during fitting;
- fold coefficients back into the raw render-time basis;
- compute a fixed algebraic confidence score from rank, view support, and condition diagnostics;
- shrink view-affine coefficients toward the v103 affine fallback instead of hard-discarding them.

Counter diagnostic:

- v104b hard fallback dropped `2,103,953` counter triangles.
- v104c shrink mode used all `2,716,449` observed counter triangles with `shrink_alpha_mean=0.566197`.

## 3. Current Quantitative Results

### Hard-Triad Mean

Hard triad: `counter`, `kitchen`, `bonsai`.

| method | PSNR | SSIM | LPIPS | interpretation |
|---|---:|---:|---:|---|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 | direct baseline |
| v103 affine min_count=1 | 28.384418 | 0.879855 | 0.226611 | first positive surface field |
| v104a raw view-affine | 28.823045 | 0.884927 | 0.219492 | view direction helps |
| v104c shrink view-affine | 28.859798 | 0.885459 | 0.219064 | current best surface field |
| v101/v102a endpoint ceiling | 30.167397 | 0.913355 | 0.163709 | strong endpoint/delta-bank ceiling |

### Hard-Triad Mean Deltas

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c minus clean | +1.037945 | +0.007156 | -0.017830 |
| v104c minus v103 | +0.475380 | +0.005604 | -0.007547 |
| v104c minus v104a | +0.036753 | +0.000532 | -0.000427 |
| v104c minus v101/v102a | -1.307599 | -0.027896 | +0.055355 |

### Per-Scene v104c Hard-Triad Metrics

| scene | PSNR | SSIM | LPIPS | dPSNR vs v104a | dSSIM vs v104a | dLPIPS vs v104a |
|---|---:|---:|---:|---:|---:|---:|
| counter | 27.498068 | 0.867420 | 0.238986 | +0.005690 | +0.000076 | -0.000017 |
| kitchen | 28.770449 | 0.881590 | 0.188021 | +0.005157 | +0.000062 | -0.000076 |
| bonsai | 30.310877 | 0.907367 | 0.230186 | +0.099411 | +0.001457 | -0.001189 |

Interpretation for PPT:

- The v104c gain over v104a is small but consistent.
- The v104c gain over clean is meaningful on the hard triad.
- The endpoint gap is still large and should be shown, not hidden.

### Full9 Fixed-Policy Results

Full9 summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.csv
```

Aggregation status: `present_scenes=9`, `ok_scenes=9`, `all_present=True`, `all_ok=True`.

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting (`ours_26000`) | 25.151682 | 0.749018 | 0.287621 |
| v104c shrink view-affine field | 25.829099 | 0.760727 | 0.268548 |
| v101/v102a endpoint/reference | 26.481310 | 0.783675 | 0.224305 |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c minus clean | +0.677417 | +0.011709 | -0.019073 |
| v104c minus endpoint/reference | -0.652211 | -0.022949 | +0.044243 |

| scene | clean PSNR | v104c PSNR | endpoint PSNR | dPSNR clean | dSSIM clean | dLPIPS clean |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.301613 | 23.717649 | 24.021442 | +0.416037 | +0.015104 | -0.018574 |
| bonsai | 28.895233 | 30.310877 | 31.861889 | +1.415644 | +0.010966 | -0.029307 |
| counter | 26.751774 | 27.498068 | 28.442907 | +0.746294 | +0.005364 | -0.013017 |
| flowers | 19.682257 | 20.075844 | 20.300581 | +0.393587 | +0.019255 | -0.020090 |
| garden | 25.029211 | 25.788094 | 26.310476 | +0.758883 | +0.019228 | -0.026730 |
| kitchen | 27.818552 | 28.770449 | 30.197395 | +0.951897 | +0.005138 | -0.011165 |
| room | 28.747276 | 29.597836 | 30.305668 | +0.850559 | +0.006994 | -0.019239 |
| stump | 25.205042 | 25.459311 | 25.595201 | +0.254269 | +0.009434 | -0.011791 |
| treehill | 20.934181 | 21.243763 | 21.296227 | +0.309582 | +0.013896 | -0.021746 |

Fairness boundary:

- The clean row is the local selected clean MeshSplatting `ours_26000` baseline from `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/*/results.json`, not a copied paper table and not train metrics.
- v104c and clean use the same local scene set and held-out test split.
- v104c uses a target-camera delta/reference distillation path through v102 and a surface-field endpoint. It is not a train-only unseen-camera generalization result and not a vanilla MeshSplatting checkpoint.

## 4. Qualitative Results and Visualization Plan

The qualitative story should be careful. Full-frame differences are often subtle. The strongest visual presentation is crop-level plus error-map evidence, tied to exact manifests.

Recommended PPT layout:

1. Show clean MeshSplatting / SPCarNet / GT side by side.
2. Add absolute-error crops or error heatmaps, not only RGB crops.
3. Caption each crop with scene, view, crop source, and metric delta.
4. Separate v101 qualitative evidence from v104c qualitative evidence.

Existing strong v101 visual asset:

```text
assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png
assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json
```

How to use it:

- Use it as evidence for the strong endpoint line.
- Say crops were selected by held-out LPIPS improvement and local absolute-error reduction.
- Do not imply that every full-frame render has obvious visual improvement.

v104c visualization can now be selected from the completed full9 per-scene renders:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/renders/*.png
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/gt/*.png
```

Safe wording:

> On full9 metrics, v104c improves the fixed-policy surface-field line over clean MeshSplatting. Visual differences should be shown as crop/error-map evidence, because full-frame RGB differences may be subtle.

## 5. Ablations and Claim Boundary

### What Currently Wins

- v101/v102a wins as the strong quality endpoint.
- v104c wins within the surface-field ladder:
  - full9 `9/9` complete against the selected clean baseline;
  - better than clean on every full9 scene and every reported RGB metric;
  - better than clean on hard-triad mean;
  - better than v103;
  - better than v104a;
  - fixed policy across hard-triad scenes.

### What Is Still Weak

- v104c is full9 validated against the selected local clean baseline, but still below v101/v102a on full9 and hard-triad means.
- v103/v104 fields are distilled from v102 target-camera deltas; this is not yet a train-only unseen-camera field claim.
- The current field is large and build-heavy. Hard-triad field manifests are hundreds of MiB and build times are minutes per scene.
- The surface-field endpoint still depends on `render.py` support for triangle ids and field sampling; it is not a plain MeshSplatting checkpoint.

### Ablation Ladder To Show

| ablation | purpose | current status |
|---|---|---|
| clean MeshSplatting | baseline | available |
| v102b constant face residual | shows static per-face RGB is too weak | counter/prototype evidence |
| v103 affine barycentric | shows face-local variation helps | hard-triad positive |
| v104a raw view-affine | shows view direction helps | hard-triad positive |
| v104b hard fallback | shows hard safety gate is over-conservative | counter diagnostic |
| v104c shrink fallback | current best field policy | full9 `9/9` positive vs selected clean |
| v101/v102a endpoint ceiling | upper reference for current endpoint quality | hard-triad and v101 full9 available |

### Safe Claims

- SPCarNet's evidence endpoint improves the selected clean MeshSplatting baseline under the local full9 protocol.
- v101 packages the endpoint through `render.py` with forceable train-derived evidence banks and detached-package validation.
- v102a preprojected deltas reproduce the v101 endpoint on validated target-camera sets while reducing online adapter overhead.
- v104c is a fixed-policy representation-field improvement over v103/v104a on hard triad.
- v104c full9 is complete against the selected local clean `ours_26000` baseline and improves all 9 scenes on PSNR, SSIM, and LPIPS.

### Unsafe Claims

- v104c replaces v101/v102a.
- v104c reaches the endpoint/reference ceiling.
- v104c is an unseen-camera train-only residual field.
- The qualitative improvement is obvious in every full-frame render.
- The endpoint or field is a vanilla MeshSplatting checkpoint without special render logic.
- SPCarNet currently has a speed claim over clean MeshSplatting.

### Next Real Research Step

The next scientific change should target the structural compression loss:

```text
current v104c:
one low-order triangle-local function

needed:
compact residual mixture / evidence-conditioned coefficients / calibrated blend,
then distill that behavior into a surface field with train-only or policy-val evidence
```

Concrete next options:

- two-component per-triangle residual mixture with shrinkage weights;
- policy-val calibrated blend between v104c field output and v102a delta-bank output, then distill back to a field;
- train/policy-val-only field construction to move away from same-target-camera delta distillation;
- fail-closed field package checks: `used_required_field=true`, missing-field failure, zero-field ablation, target-GT non-use smoke.

## 6. Command and Path Appendix

### v101 Full9 Bank-Backed Endpoint

Summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.json
```

Command:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v101_renderpy_endpoint_full9.py \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625 \
  --gpus 1,2,3,5 --max_parallel 4 \
  --method_name ours_26000_v101_bankfp16_renderpy_endpoint_full9_fixed \
  --build_banks --require_bank \
  --bank_root /dev/shm/peilincai_spcarnet_v101_bankfp16_full9_fixed_20260625 \
  --bank_residual_dtype float16 --bank_depth_dtype float16
```

### v102 Hard-Triad Preprojected Delta

Summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625/v102_preprojected_delta_triad_summary.json
```

Per-scene command template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v102_preprojected_delta_scene.py \
  --scene <scene> \
  --gpu <gpu> \
  --force \
  --no_intermediate_outputs
```

### v103 Surface Affine Field

Builder:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v103_surface_affine_residual_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/<scene>/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v103_surface_affine_field_20260625/<scene>/v103_surface_affine_min1_field.pt \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --split test \
  --renderer_scaling 4 \
  --min_count 1 \
  --ridge 0.0001 \
  --residual_dtype float16
```

### v104c Shrink View-Affine Field

Hard-triad summary:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_hardtriad_20260625/v104c_hardtriad_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_hardtriad_20260625/v104c_hardtriad_summary.json
```

Full9 working directory:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/
```

Single-scene fixed-policy runner:

```bash
SCENE=<scene>
GPU=<gpu>

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v104c_shrink_view_affine_scene.py \
  --scene ${SCENE} \
  --gpu ${GPU} \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --field_root /dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625 \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625 \
  --v102_report_root outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --ridge 1e-3 \
  --residual_clip 0.08 \
  --view_std_floor 1e-4 \
  --rank_rtol 1e-7 \
  --condition_max 1e8 \
  --chunk_pixels 262144 \
  --build_v102_if_missing
```

Full9 aggregation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v104c_shrink_view_affine_full9.py \
  --root outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625 \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625
```

Build command template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 /usr/bin/time -f 'v104c <scene> build wall %e sec' \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v104b_centered_view_affine_residual_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625/<scene>/v102_preprojected_delta_bank.pt \
  --output_field /dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/<scene>/v104c_shrink_view_affine_min1_minviews1_field.pt \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --min_count 1 \
  --min_views 1 \
  --ridge 0.001 \
  --residual_clip 0.08 \
  --view_std_floor 1e-4 \
  --rank_rtol 1e-7 \
  --condition_max 1e8 \
  --fallback_mode shrink \
  --chunk_pixels 262144
```

Render/eval pattern:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model \
  --iteration 26000 \
  --skip_train \
  --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --checkpoint_endpoint_output_method ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene> \
  --checkpoint_endpoint_surface_field_path /dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625/<scene>/v104c_shrink_view_affine_min1_minviews1_field.pt \
  --checkpoint_endpoint_require_surface_field \
  --checkpoint_endpoint_no_intermediate_outputs \
  --quiet

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/evaluate_render_split_metrics.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model \
  --split test \
  --methods ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene> \
  --merge_model_results
```

### Source Documents Read

```text
docs/car_model/6-25-v101-RenderPyEndpoint-EvidenceBank-Log.md
docs/car_model/6-25-v101-Subagent-Review-And-PaperStory.md
docs/car_model/6-25-v102-PreprojectedDelta-Acceleration-Log.md
docs/car_model/6-25-v103-FaceLocalAffineResidualField-Log.md
docs/car_model/6-25-v104a-Subagent-Story-And-Weakness-Review.md
docs/car_model/6-25-v104c-ShrinkViewAffine-HardTriad-Log.md
docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_hardtriad_20260625/v104c_hardtriad_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.md
```
