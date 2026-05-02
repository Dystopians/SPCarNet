# MeshPrior Parking Render Metric Comparison

Date: 2026-05-01

## Baseline Naming

The current `parking_phone_tiny/baseline_200iter` run should be treated as an engineering baseline, not the final paper baseline.

Recommended naming:

- **Engineering baseline**: current repository, no MeshPrior proposal application, short 200-iteration parking run.
- **Paper baseline**: original Mesh Splatting / clean upstream method on the same parking data, same training budget, same evaluation scripts.
- **MeshPrior variant**: current repository plus object-prior region mining, proposal gates, copied-patch cleanup, recovery model, and later render-gated training/evaluation.

This distinction matters because the current engineering baseline already includes repository modifications and is too short to support final paper claims.

## Scope

This report compares render metrics for:

- engineering baseline: `outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model`
- recovery cleanup model: `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup`

Both use:

- `render.py`
- `metrics.py`
- test split only
- iteration `200`
- `54` test views

## Commands

Recovery render:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup --images images --eval --iteration 200 --skip_train --quiet
```

Recovery metrics:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup
```

Engineering baseline render:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py -s outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --images images --eval --iteration 200 --skip_train --quiet
```

Engineering baseline metrics:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py -m outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model
```

## Results

| metric | engineering baseline | recovery cleanup | delta |
| --- | ---: | ---: | ---: |
| SSIM | 0.2898596525 | 0.2898600996 | +0.0000004470 |
| PSNR | 10.9499864578 | 10.9499950409 | +0.0000085831 |
| LPIPS | 0.6441746354 | 0.6441848874 | +0.0000102520 |

Geometry proxy comparison from the recovery report:

| metric | engineering baseline | recovery cleanup | delta |
| --- | ---: | ---: | ---: |
| depth AbsRel | 0.3241713746 | 0.3241717166 | +0.0000003420 |
| normal mean angle | 51.6879735355 | 51.6880043094 | +0.0000307739 |

## Interpretation

The copied checkpoint cleanup is render-stable and geometry-stable at this short-run scale. The deltas are effectively neutral.

This is not evidence of a final improvement. It is evidence that the local patch extraction, copied cleanup, checkpoint compaction, recovery model layout, geometry evaluation, and render metric pipeline do not collapse on the parking scene.

## Gate

Stage gate: SOFT PASS.

Next requirements before a paper-level claim:

1. Run the true paper baseline: clean/original Mesh Splatting on the same parking data and budget.
2. Run a longer recovery or MeshPrior-guided variant with render gates enabled.
3. Compare across geometry, render quality, triangle count, speed, and rollback/failure rates.
